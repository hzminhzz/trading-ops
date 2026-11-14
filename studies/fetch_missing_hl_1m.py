"""Backfill *only missing* official Hyperliquid 1m OHLCV into private trading-ops.

The portfolio replay remains fully local and offline. Never writes source PM store,
registry, Guardian, or other runtime. No broker/Bybit credentials.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import polars as pl

COINS = ("GMX", "AZTEC", "SEI")
START = 1794441600000  # 2026-11-12T00:00Z
END = 1794614400000  # 2026-11-14T00:00Z exclusive
ROOT = Path("data/private/hyperliquid_candles_1m_20261112_13")
URL = "https://api.hyperliquid.xyz/info"


def fetch(coin: str) -> tuple[pl.DataFrame, dict]:
    body = json.dumps(
        {
            "type": "candleSnapshot",
            "req": {"coin": coin, "interval": "1m", "startTime": START, "endTime": END},
        },
        separators=(",", ":"),
    ).encode()
    req = Request(
        URL,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "trading-ops-research/1.0"},
    )
    with urlopen(req, timeout=25) as response:
        raw = response.read()
    if len(raw) > 4_000_000:
        raise ValueError("OHLCV_RESPONSE_OVERSIZED")
    values = json.loads(raw)
    if not isinstance(values, list):
        raise TypeError("OHLCV_RESPONSE_NOT_CANDLES")
    rows = []
    for obj in values:
        ts = obj["t"]
        if not isinstance(ts, int) or not START <= ts < END:
            continue
        if obj["s"] != coin or obj["i"] != "1m" or obj["T"] != ts + 59_999:
            raise ValueError("HL_CANDLE_IDENTITY_MISMATCH")
        row = {
            "timestamp": datetime.fromtimestamp(ts / 1000, UTC),
            "symbol": coin,
            "open": float(obj["o"]),
            "high": float(obj["h"]),
            "low": float(obj["l"]),
            "close": float(obj["c"]),
            "volume": float(obj["v"]),
            "trade_count": int(obj.get("n", 0)),
        }
        if (
            not 0 < row["low"] <= row["open"] <= row["high"]
            or not row["low"] <= row["close"] <= row["high"]
        ):
            raise ValueError("CANDLE_PRICE_ORDER_INVALID")
        if row["volume"] < 0 or row["trade_count"] < 0:
            raise ValueError("CANDLE_VOLUME_INVALID")
        rows.append(row)
    if len(rows) != 2880 or len({row["timestamp"] for row in rows}) != 2880:
        raise ValueError(f"CANDLE_GAP_OR_DUPLICATE:{coin}:{len(rows)}")
    rows.sort(key=lambda row: row["timestamp"])
    for n, row in enumerate(rows):
        if int(row["timestamp"].timestamp() * 1000) != START + n * 60_000:
            raise ValueError("CANDLE_TIME_GAP")
    frame = pl.DataFrame(
        rows,
        schema={
            "timestamp": pl.Datetime("us", "UTC"),
            "symbol": pl.String,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "trade_count": pl.Int64,
        },
    )
    info = {
        "coin": coin,
        "source": "official_hyperliquid_info_candleSnapshot",
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "first_ms": START,
        "last_ms": END - 60_000,
        "rows": len(rows),
    }
    return frame, info


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    prior_path = ROOT / "manifest.json"
    prior = json.loads(prior_path.read_text()) if prior_path.is_file() else {}
    if prior and prior.get("provider") != "official_hyperliquid_candleSnapshot":
        raise ValueError("EXISTING_MANIFEST_UNTRUSTED")
    manifest = {
        "classification": "PRIVATE_RESEARCH_PRICE_DATA_NOT_BROKER",
        "provider": "official_hyperliquid_candleSnapshot",
        "coin": {},
    }
    for coin in COINS:
        file = ROOT / f"{coin}.parquet"
        if file.exists():
            existing = pl.read_parquet(file)
            if existing.height != 2880:
                raise ValueError("EXISTING_LOCAL_CANDLE_NOT_COMPLETE")
            checksum = hashlib.sha256(file.read_bytes()).hexdigest()
            original = prior.get("coin", {}).get(coin)
            if not original or original.get("parquet_sha256") != checksum:
                raise ValueError("EXISTING_LOCAL_CANDLE_PROVENANCE_UNVERIFIED")
            manifest["coin"][coin] = dict(original)
            continue
        frame, info = fetch(coin)
        frame.write_parquet(file)
        info["parquet_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
        manifest["coin"][coin] = info
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
