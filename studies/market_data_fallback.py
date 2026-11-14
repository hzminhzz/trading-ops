"""Read-only, provenance-aware Hyperliquid market-bar recovery.

Priority at each missing minute: original ML4T -> Vike connected-MCP
receipt -> verified legacy Vike cache -> local normalized Reservoir 1s
archive.  Never treat a trade-less minute as a zero return or an executable
quote.  Reservoir cannot be queried unauthenticated: requester-pays S3
requires a valid AWS session, and raw selected-wallet Reservoir fills alone
lack a market price column.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

MINUTE_MS = 60_000
LEGACY_VIKE = Path("/home/quant/dev/copytrade-pm/data/private/backtest_20261112/bars_vike")
MCP_LOCAL = Path("data/market_recovery_vike")
MCP_SHARED = Path("/home/quant/.local/share/copytrade/research/vike-market-1m-20260701-20261101")
RESERVOIR_LOCAL = Path("/home/quant/.local/share/copytrade/research/reservoir-market-1s")
SOURCES = (
    "ML4T_LOCAL_1M",
    "VIKE_CONNECTED_MCP_1M",
    "VIKE_PREVIOUS_LOCAL_CACHE_1M",
    "HYDROMANCER_RESERVOIR_NORMALIZED_1S",
)


def load_market_recovery(
    symbol: str,
    *,
    start_ms: int,
    end_ms: int,
    existing: Iterable[float],
    mcp_roots: tuple[Path, ...] = (MCP_SHARED, MCP_LOCAL),
    legacy_root: Path = LEGACY_VIKE,
    reservoir_root: Path = RESERVOIR_LOCAL,
) -> tuple[list[tuple[int, float, float]], dict]:
    """Only missing prices are filled; validate and retain original ML4T precedence.

    Returns sparse recovered (minute timestamp, close, volume) rows and an
    explicit source audit.  Vike MCP files are originally received from the
    connected application and written to a Git-ignored private location.
    """
    if not symbol or not symbol.replace("_", "").isalnum():
        raise ValueError("INVALID_MARKET_SYMBOL")
    expected = (end_ms - start_ms) // MINUTE_MS
    if end_ms <= start_ms or expected <= 0:
        raise ValueError("INVALID_RECOVERY_WINDOW")
    local = list(existing)
    if len(local) != expected:
        raise ValueError("RECOVERY_WINDOW_SHAPE")
    recovered: dict[int, tuple[float, float]] = {}
    counts: Counter[str] = Counter()
    first_last: dict[str, list[int]] = {}
    receipts: dict[str, list[dict]] = {}
    seen_vike_files: dict[str, str] = {}

    def insert(ts: int, close: float, volume: float, source: str) -> None:
        if not isinstance(ts, int) or ts % MINUTE_MS or ts < start_ms or ts >= end_ms:
            counts["out_of_window"] += 1
            return
        if not math.isfinite(close) or close <= 0:
            raise ValueError("INVALID_RECOVERY_CANDLE_PRICE")
        if not (math.isnan(volume) or (math.isfinite(volume) and volume >= 0)):
            raise ValueError("INVALID_RECOVERY_CANDLE_VOLUME")
        index = (ts - start_ms) // MINUTE_MS
        if not math.isnan(local[index]):
            counts[source + "_OVERLAP_LOCAL"] += 1
            return
        if ts in recovered:
            previous, _ = recovered[ts]
            if abs(previous - close) > max(1e-12, abs(previous) * 0.0002):
                counts[source + "_CONFLICT_LOWER_PRIORITY"] += 1
            return
        recovered[ts] = (close, volume)
        counts[source + "_FILLED"] += 1
        times = first_last.setdefault(source, [ts, ts])
        times[0] = min(times[0], ts)
        times[1] = max(times[1], ts)

    for root in mcp_roots:
        if not root.is_dir():
            continue
        for f in sorted(root.glob(symbol + "_*.json")):
            content = f.read_bytes()
            file_hash = hashlib.sha256(content).hexdigest()
            if f.name in seen_vike_files:
                if seen_vike_files[f.name] != file_hash:
                    raise ValueError("CONFLICTING_PRIVATE_VIKE_RECEIPT:" + f.name)
                continue
            seen_vike_files[f.name] = file_hash
            obj = json.loads(content)
            if obj.get("symbol") != symbol or obj.get("method") != "vike_connected_mcp":
                raise ValueError("UNTRUSTED_VIKE_RECOVERY_RECEIPT")
            if obj.get("source") != "1m fills rollup":
                raise ValueError("UNEXPECTED_VIKE_ROLLUP_SOURCE")
            rows = obj.get("rows")
            if not isinstance(rows, list):
                raise TypeError("INVALID_VIKE_RECOVERY_ROWS")
            receipts.setdefault("vike_connected_mcp", []).append(
                {
                    "file": f.name,
                    "sha256": file_hash,
                    "rows": len(rows),
                }
            )
            last = -1
            for item in rows:
                if not isinstance(item, list) or len(item) != 4:
                    raise TypeError("INVALID_VIKE_RECOVERY_ROW")
                ts, open_price, close, volume = item
                if not isinstance(ts, int) or ts <= last:
                    raise ValueError("NONMONOTONIC_VIKE_RECOVERY")
                last = ts
                if float(open_price) <= 0:
                    raise ValueError("INVALID_VIKE_RECOVERY_OPEN")
                insert(ts, float(close), float(volume), "VIKE_CONNECTED_MCP_1M")

    previous = legacy_root / (symbol + ".parquet")
    if previous.is_file():
        previous_hash = hashlib.sha256(previous.read_bytes()).hexdigest()
        df = pl.read_parquet(previous, columns=["ts_ms", "close"])
        if df["ts_ms"].n_unique() != df.height:
            raise ValueError("DUPLICATE_LEGACY_VIKE_CANDLES")
        for ts, close in df.iter_rows():
            insert(int(ts), float(close), float("nan"), "VIKE_PREVIOUS_LOCAL_CACHE_1M")
        counts["legacy_cache_file_present"] = 1
        receipts["legacy_cache"] = [
            {"file": previous.name, "sha256": previous_hash, "rows": df.height}
        ]

    # Explicitly normalized 1-second Reservoir candles: a *complete market*
    # trade tape must first be audited and aggregated to 1m. Selected-wallet
    # Reservoir fills cannot be used here: the existing Parquet has no price.
    res = reservoir_root / (symbol + "_1m.parquet")
    if res.is_file():
        df = pl.read_parquet(res, columns=["ts_ms", "close", "volume"])
        for ts, close, volume in df.iter_rows():
            insert(int(ts), float(close), float(volume), "HYDROMANCER_RESERVOIR_NORMALIZED_1S")
        counts["reservoir_normalized_file_present"] = 1

    audit = {
        "sources_in_order": list(SOURCES),
        "recovered_rows_by_source": dict(sorted(counts.items())),
        "source_first_last_utc": {
            k: [
                datetime.fromtimestamp(v[0] / 1000, UTC).isoformat(),
                datetime.fromtimestamp(v[1] / 1000, UTC).isoformat(),
            ]
            for k, v in sorted(first_last.items())
        },
        "connected_vike_receipts": receipts.get("vike_connected_mcp", []),
        "connected_vike_file_count": len(receipts.get("vike_connected_mcp", [])),
        "legacy_vike_cache_receipt": receipts.get("legacy_cache", []),
        "reservoir_fallback_checked": True,
        "reservoir_local_price_cache_available": res.is_file(),
        "reservoir_s3_authenticated": False,  # no AWS data access is inferred
        "missing_minute_not_assumed_flat": True,
    }
    return [(ts, price, vol) for ts, (price, vol) in sorted(recovered.items())], audit
