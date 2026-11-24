"""Synthetic fact stores for the Slice-4 research tests.

Fills arrive the way production does: as a Reservoir-shaped Parquet archive that
the real migration engine ingests. Equity has no archive adapter, so it is
written through ``FactIngestor`` (the capture path). Nothing here implements a
second store.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any

import polars as pl
from copytrade_pm.facts import FactIngestor, FactStore
from copytrade_pm.facts.migrate import Migration, new_receipt
from copytrade_pm.facts.sources import build_catalogue

GRID_MS = 300_000
DAY_MS = 86_400_000
START = int(datetime(2026, 10, 1, tzinfo=UTC).timestamp() * 1000)
WALLET_A = "0x" + "a" * 40
WALLET_B = "0x" + "b" * 40


def _fill(
    wallet: str,
    coin: str,
    index: int,
    *,
    at_ms: int,
    side: str,
    size: float,
    price: float,
) -> dict:
    return {
        "wallet": wallet,
        "exchange_ts": datetime.fromtimestamp(at_ms / 1000, UTC),
        "coin": coin,
        "dex": "hyperliquid",
        "side": side,
        "size": size,
        "price": price,
        "start_position": 0.0,
        "trade_id": index + 1,
    }


def wallet_round_trips(
    wallet: str,
    coin: str,
    *,
    start_ms: int,
    bars: int,
    base_price: float,
    size: float = 0.5,
) -> list[dict]:
    """A rising tape with alternating buy/sell rounds.

    Buys are slightly larger than sells, so the wallet carries a small long and a
    positive 5m markout — the fixture must not accidentally fail the markout gate
    the Stage-1 audit is supposed to measure.
    """
    rows = []
    for index in range(bars):
        at_ms = start_ms + index * GRID_MS
        buy = index % 6 < 3
        price = base_price * (1.0 + 0.0006 * index + 0.0004 * ((index % 6) - 2))
        rows.append(
            _fill(
                wallet,
                coin,
                index,
                at_ms=at_ms,
                side="buy" if buy else "sell",
                size=size if buy else size * 0.8,
                price=price,
            )
        )
    return rows


def build_archive(
    root: Path,
    rows: list[dict],
    *,
    directory: str = "reservoir-full-history-synthetic",
) -> Path:
    archive = root / directory
    by_day: dict[str, list[dict]] = {}
    for row in rows:
        by_day.setdefault(row["exchange_ts"].date().isoformat(), []).append(row)
    for day, day_rows in sorted(by_day.items()):
        path = archive / "hyperliquid" / "fills" / f"day={day}" / "data.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(day_rows).with_columns(
            pl.col("exchange_ts").cast(pl.Datetime("us", "UTC")),
            pl.col("trade_id").cast(pl.UInt64),
        ).write_parquet(path)
    return archive


def equity_rows(wallet: str, *, start_ms: int, equity_usd: Decimal, days: int) -> list[dict]:
    return [
        {
            "wallet": wallet,
            "account_scope": "perp",
            "abstraction_mode": "unified",
            "equity_usd": equity_usd,
            "withdrawable_usd": equity_usd,
            "margin_used_usd": Decimal(0),
            "observed_at": datetime.fromtimestamp((start_ms + index * DAY_MS) / 1000, UTC),
            "event_ts_ms": start_ms + index * DAY_MS,
            "first_known_at_ms": start_ms + index * DAY_MS + 1000,
            "arrival_seq": index,
            "source_id": "synthetic-capture",
            "source_rank": 1,
            "batch_sha16": "",
        }
        for index in range(days)
    ]


def bars_1m_from_fills(
    fills: list[dict],
    *,
    venue: str = "hyperliquid",
    quote_volume_usd: Decimal = Decimal(5000000),
    first_known_at_offset_ms: int | None = 60_000,
) -> list[dict]:
    """Build canonical 1m venue bars covering every ``(coin, minute)`` traded in ``fills``."""
    by_key: dict[tuple[str, int], float] = {}
    for row in fills:
        at_ms = int(row["exchange_ts"].timestamp() * 1000)
        minute_ms = (at_ms // 60_000) * 60_000
        by_key[(str(row["coin"]).strip().upper(), minute_ms)] = float(row["price"])
    rows: list[dict] = []
    for index, ((coin, minute_ms), price) in enumerate(sorted(by_key.items())):
        px = Decimal(str(round(price, 8)))
        volume = (quote_volume_usd / px).quantize(Decimal("0.00000001")) if px > 0 else Decimal(0)
        known_at = (
            None if first_known_at_offset_ms is None else minute_ms + first_known_at_offset_ms
        )
        rows.append(
            {
                "venue": venue,
                "coin": coin,
                "open": px,
                "high": px,
                "low": px,
                "close": px,
                "volume": volume,
                "quote_volume_usd": quote_volume_usd,
                "bar_ts": datetime.fromtimestamp(minute_ms / 1000, UTC),
                "event_ts_ms": minute_ms,
                "first_known_at_ms": known_at,
                "arrival_seq": index,
                "source_id": "synthetic-bars",
                "source_rank": 1,
                "batch_sha16": "",
            }
        )
    return rows


def build_store(
    root: Path,
    *,
    fills: list[dict],
    equity: dict[str, tuple[Decimal, int]],
    bars_1m_rows: list[dict] | None = None,
    start_ms: int = START,
) -> FactStore:
    archive = build_archive(root, fills)
    store_root = root / "facts"
    sources, absent = build_catalogue(reservoir={"full-history": archive})
    store = FactStore(store_root, create=True)
    Migration(store=store, sources=sources, kinds=("fills",)).run(
        new_receipt(store_root=store_root, sources=sources, not_supplied=absent, kinds=("fills",))
    )
    ingestor = FactIngestor(store)
    for wallet, (value, days) in equity.items():
        ingestor.ingest_rows(
            "equity",
            equity_rows(wallet, start_ms=start_ms, equity_usd=value, days=days),
            source_id="synthetic-capture",
            source_rank=1,
            first_known_at_ms=None,
        )
    bars = bars_1m_from_fills(fills) if bars_1m_rows is None else bars_1m_rows
    if bars:
        ingestor.ingest_rows(
            "bars_1m",
            bars,
            source_id="synthetic-bars",
            source_rank=1,
            first_known_at_ms=None,
        )
    return store


def day_strings(start_ms: int, days: int) -> list[str]:
    start = datetime.fromtimestamp(start_ms / 1000, UTC)
    return [(start + timedelta(days=index)).date().isoformat() for index in range(days)]


def resolve_prop_production_root() -> Path | None:
    env_root = os.environ.get("PROP_PRODUCTION_ROOT")
    candidates = (
        [Path(env_root)]
        if env_root
        else [
            Path("/home/quant/dev/prop-production"),
            Path("/home/quant/.devspace/worktrees/prop-production-phase-d"),
        ]
    )
    for root in candidates:
        if (root / "src" / "prop_production" / "guardian.py").is_file():
            return root
    return None


def cwsa_sample(
    *,
    evidence_root: Path,
    scope: Any,
    limits: Any,
    wallet: str,
    coin: str,
    mid: Decimal,
    boundary: int,
    qty: str,
) -> Any:
    from copytrade_pm.pm_checkpoint_source import collect_checkpoint

    def native(request: dict[str, Any]):
        if request["type"] == "perpDexs":
            body: Any = [None, {"name": "xyz"}]
        else:
            body = {
                "time": boundary,
                "assetPositions": []
                if request.get("dex") == "xyz" or qty == "0"
                else [
                    {
                        "position": {
                            "coin": coin,
                            "szi": qty,
                            "positionValue": str(abs(Decimal(qty)) * mid),
                        }
                    }
                ],
            }
        raw = json.dumps(body).encode()
        return body, raw, sha256(raw).hexdigest()

    return collect_checkpoint(
        scope,
        wallet,
        checkpoint_ms=boundary,
        limits=limits,
        evidence_root=evidence_root,
        info=native,
        now_ms=lambda: boundary + 100,
    )
