"""Behavior and query-plan tests for lazy Polars study ingestion and screening."""

from __future__ import annotations

import importlib
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

STUDIES_DIR = Path(__file__).resolve().parents[1] / "studies"
sys.path.insert(0, str(STUDIES_DIR))

cr = importlib.import_module("canonical_replay")


def _write_fill_parquet(path: Path, rows: list[dict]) -> Path:
    df = pl.DataFrame(rows).with_columns(
        pl.col("exchange_ts").cast(pl.Datetime("us", "UTC")),
        pl.col("trade_id").cast(pl.UInt64),
    )
    df.write_parquet(path)
    return path


def test_same_ms_fills_preserve_arrival_order_not_trade_id(tmp_path: Path) -> None:
    import study_data as sd

    start_dt = datetime(2026, 5, 1, 0, 0, tzinfo=UTC)
    split_dt = datetime(2026, 5, 2, 0, 0, tzinfo=UTC)
    end_dt = datetime(2026, 5, 3, 0, 0, tzinfo=UTC)
    burst_ts = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)

    # Non-monotonic trade_id within same-ms burst:
    # Arrival 0: 0 -> +10 (tid=999), Arrival 1: +10 -> 0 (tid=100).
    # Sorting by tid would put +10->0 first and 0->+10 second (phantom +10 inventory!).
    f1 = _write_fill_parquet(
        tmp_path / "day1.parquet",
        [
            {
                "wallet": "0xaaa",
                "exchange_ts": datetime(2026, 5, 1, 0, 5, tzinfo=UTC),
                "coin": "BTC",
                "dex": "hyperliquid",
                "side": "buy",
                "size": 1.0,
                "start_position": 0.0,
                "trade_id": 10,
            },
            {
                "wallet": "0xaaa",
                "exchange_ts": datetime(2026, 5, 1, 0, 6, tzinfo=UTC),
                "coin": "BTC",
                "dex": "hyperliquid",
                "side": "sell",
                "size": 1.0,
                "start_position": 1.0,
                "trade_id": 11,
            },
            {
                "wallet": "0xaaa",
                "exchange_ts": burst_ts,
                "coin": "BTC",
                "dex": "hyperliquid",
                "side": "buy",
                "size": 10.0,
                "start_position": 0.0,
                "trade_id": 999,
            },
            {
                "wallet": "0xaaa",
                "exchange_ts": burst_ts,
                "coin": "BTC",
                "dex": "hyperliquid",
                "side": "sell",
                "size": 10.0,
                "start_position": 10.0,
                "trade_id": 100,
            },
        ],
    )
    f2 = _write_fill_parquet(
        tmp_path / "day2.parquet",
        [
            {
                "wallet": "0xaaa",
                "exchange_ts": datetime(2026, 5, 2, 23, 30, tzinfo=UTC),
                "coin": "BTC",
                "dex": "hyperliquid",
                "side": "buy",
                "size": 2.0,
                "start_position": 0.0,
                "trade_id": 500,
            },
            {
                "wallet": "0xaaa",
                "exchange_ts": datetime(2026, 5, 2, 23, 31, tzinfo=UTC),
                "coin": "BTC",
                "dex": "hyperliquid",
                "side": "sell",
                "size": 2.0,
                "start_position": 2.0,
                "trade_id": 501,
            },
        ],
    )

    rec = cr.WalletRecord(wallet_id=1, address="0xaaa", alias="W1", configured=True, excluded=False)
    res = sd.ingest_fills([rec], [f1, f2], start=start_dt, end=end_dt, split=split_dt)
    btc = res.grouped["W1"]["BTC"]

    assert btc["tid"].to_list() == ["10", "11", "999", "100", "500", "501"]
    # End-to-end invariant: reconstructed position after burst is flat 0.0, never phantom +10.0.
    import numpy as np

    grid = np.array(
        [
            int(datetime(2026, 5, 1, 12, 0, tzinfo=UTC).timestamp() * 1000),
            int(datetime(2026, 5, 1, 12, 5, tzinfo=UTC).timestamp() * 1000),
        ],
        dtype=np.int64,
    )
    recon = cr.reconstruct_positions_on_grid(btc, grid, policy="full")
    assert recon.pos.tolist() == pytest.approx([0.0, 0.0])


def test_same_ms_continuity_tie_break_heals_raw_seq_inversion_without_breaking_valid_seq(
    tmp_path: Path,
) -> None:
    import study_data as sd

    start_dt = datetime(2026, 5, 1, 0, 0, tzinfo=UTC)
    split_dt = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
    end_dt = datetime(2026, 5, 2, 0, 0, tzinfo=UTC)
    burst_ts = datetime(2026, 5, 1, 6, 0, 0, tzinfo=UTC)

    # Prior fill leaves position at +60.01.
    # At burst_ts, raw arrival seq has a chain break:
    #   row 1 (tid=200): 57.01 -> 54.01
    #   row 2 (tid=100): 60.01 -> 57.01
    # Continuity tie-break restores 60.01 -> 57.01 -> 54.01 so final position is 54.01, not 57.01!
    # Also include a clean cycle 0 -> +5 -> 0 -> +5 on ETH to ensure valid seq order is untouched.
    f1 = _write_fill_parquet(
        tmp_path / "fills.parquet",
        [
            {
                "wallet": "0xaaa",
                "exchange_ts": datetime(2026, 5, 1, 0, 5, tzinfo=UTC),
                "coin": "HYPE",
                "dex": "hyperliquid",
                "side": "buy",
                "size": 60.01,
                "start_position": 0.0,
                "trade_id": 1,
            },
            {
                "wallet": "0xaaa",
                "exchange_ts": burst_ts,
                "coin": "HYPE",
                "dex": "hyperliquid",
                "side": "sell",
                "size": 3.0,
                "start_position": 57.01,
                "trade_id": 200,
            },
            {
                "wallet": "0xaaa",
                "exchange_ts": burst_ts,
                "coin": "HYPE",
                "dex": "hyperliquid",
                "side": "sell",
                "size": 3.0,
                "start_position": 60.01,
                "trade_id": 100,
            },
            {
                "wallet": "0xaaa",
                "exchange_ts": burst_ts,
                "coin": "ETH",
                "dex": "hyperliquid",
                "side": "buy",
                "size": 5.0,
                "start_position": 0.0,
                "trade_id": 900,
            },
            {
                "wallet": "0xaaa",
                "exchange_ts": burst_ts,
                "coin": "ETH",
                "dex": "hyperliquid",
                "side": "sell",
                "size": 5.0,
                "start_position": 5.0,
                "trade_id": 800,
            },
            {
                "wallet": "0xaaa",
                "exchange_ts": datetime(2026, 5, 1, 23, 30, tzinfo=UTC),
                "coin": "HYPE",
                "dex": "hyperliquid",
                "side": "sell",
                "size": 54.01,
                "start_position": 54.01,
                "trade_id": 300,
            },
        ],
    )

    rec = cr.WalletRecord(wallet_id=1, address="0xaaa", alias="W1", configured=True, excluded=False)
    res = sd.ingest_fills([rec], [f1], start=start_dt, end=end_dt, split=split_dt)
    hype = res.grouped["W1"]["HYPE"]
    assert hype["tid"].to_list() == ["1", "100", "200", "300"]
    assert res.grouped["W1"]["ETH"]["tid"].to_list() == ["900", "800"]


def test_utc_filtering_retains_final_berlin_hour_and_excludes_out_of_window(
    tmp_path: Path,
) -> None:
    import study_data as sd

    start_dt = datetime(2026, 5, 1, 0, 0, tzinfo=UTC)
    split_dt = datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
    end_dt = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)

    # Includes fills right before start, inside start hour, in [23:00Z, 24:00Z) on Oct 31, and at exact end (excluded).
    f1 = _write_fill_parquet(
        tmp_path / "boundary.parquet",
        [
            {
                "wallet": "0xAAA",
                "exchange_ts": datetime(2026, 4, 30, 23, 59, tzinfo=UTC),
                "coin": "BTC",
                "dex": "hyperliquid",
                "side": "buy",
                "size": 1.0,
                "start_position": 0.0,
                "trade_id": 1,
            },
            {
                "wallet": "0xAAA",
                "exchange_ts": datetime(2026, 5, 1, 0, 5, tzinfo=UTC),
                "coin": "BTC",
                "dex": "hyperliquid",
                "side": "buy",
                "size": 1.0,
                "start_position": 0.0,
                "trade_id": 2,
            },
            {
                "wallet": "0xAAA",
                "exchange_ts": datetime(2026, 10, 31, 23, 55, tzinfo=UTC),
                "coin": "BTC",
                "dex": "hyperliquid",
                "side": "sell",
                "size": 1.0,
                "start_position": 1.0,
                "trade_id": 3,
            },
            {
                "wallet": "0xAAA",
                "exchange_ts": datetime(2026, 11, 1, 0, 0, tzinfo=UTC),
                "coin": "BTC",
                "dex": "hyperliquid",
                "side": "buy",
                "size": 1.0,
                "start_position": 0.0,
                "trade_id": 4,
            },
        ],
    )
    rec = cr.WalletRecord(wallet_id=1, address="0xaaa", alias="W1", configured=True, excluded=False)
    res = sd.ingest_fills([rec], [f1], start=start_dt, end=end_dt, split=split_dt)
    assert res.grouped["W1"]["BTC"]["tid"].to_list() == ["2", "3"]
    assert res.min_ts_ms == int(datetime(2026, 5, 1, 0, 5, tzinfo=UTC).timestamp() * 1000)
    assert res.max_ts_ms == int(datetime(2026, 10, 31, 23, 55, tzinfo=UTC).timestamp() * 1000)


def test_utc_coverage_fails_loudly_on_timezone_shortfall_and_naive_datetime(
    tmp_path: Path,
) -> None:
    import study_data as sd

    start_dt = datetime(2026, 5, 1, 0, 0, tzinfo=UTC)
    split_dt = datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
    end_dt = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)

    # Last fill stops at 22:55Z on Oct 31 (> 1h short of Nov 1 00:00Z, matching Europe/Berlin cutoff).
    f_short = _write_fill_parquet(
        tmp_path / "short.parquet",
        [
            {
                "wallet": "0xaaa",
                "exchange_ts": datetime(2026, 5, 1, 0, 5, tzinfo=UTC),
                "coin": "BTC",
                "dex": "hyperliquid",
                "side": "buy",
                "size": 1.0,
                "start_position": 0.0,
                "trade_id": 1,
            },
            {
                "wallet": "0xaaa",
                "exchange_ts": datetime(2026, 10, 31, 22, 55, tzinfo=UTC),
                "coin": "BTC",
                "dex": "hyperliquid",
                "side": "sell",
                "size": 1.0,
                "start_position": 1.0,
                "trade_id": 2,
            },
        ],
    )
    rec = cr.WalletRecord(wallet_id=1, address="0xaaa", alias="W1", configured=True, excluded=False)

    with pytest.raises(ValueError, match="UTC_COVERAGE_END_SHORTFALL"):
        sd.ingest_fills([rec], [f_short], start=start_dt, end=end_dt, split=split_dt)

    with pytest.raises(ValueError, match="NAIVE_DATETIME_FORBIDDEN"):
        sd.build_fill_lazy_pipeline(
            [rec],
            [f_short],
            start=datetime(2026, 5, 1),
            end=end_dt,
            split=split_dt,
        )


def test_discover_fill_archive_dedups_overlap_and_flags_missing_days(tmp_path: Path) -> None:
    import study_data as sd

    full_dir = tmp_path / "reservoir-full-history-20261115" / "hyperliquid" / "fills"
    mon_dir = tmp_path / "reservoir-monitored-plus-six-hyperliquid-20261116" / "fills"

    for day, root in [
        ("2026-05-01", full_dir),
        ("2026-05-02", full_dir),
        ("2026-05-02", mon_dir),  # duplicate day should prefer full_dir
        ("2026-05-04", mon_dir),  # 2026-05-03 missing
    ]:
        p = root / f"day={day}" / "data.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        _write_fill_parquet(
            p,
            [
                {
                    "wallet": "0xaaa",
                    "exchange_ts": datetime.fromisoformat(f"{day}T00:05:00+00:00"),
                    "coin": "BTC",
                    "dex": "hyperliquid",
                    "side": "buy",
                    "size": 1.0,
                    "start_position": 0.0,
                    "trade_id": 1,
                }
            ],
        )

    paths, manifest = sd.discover_fill_archive(
        datetime(2026, 5, 1, 0, 0, tzinfo=UTC),
        datetime(2026, 5, 5, 0, 0, tzinfo=UTC),
        archive_root=tmp_path,
    )
    assert len(paths) == 3
    assert "reservoir-full-history-20261115" in paths[1]
    assert manifest["duplicate_date_sources"] == ["2026-05-02"]
    assert manifest["missing_dates"] == ["2026-05-03"]
    assert len(manifest["archive_sha256"]) == 64


def test_cse_explain_plan_verifies_cache_projection_and_predicate_pushdown_without_duckdb(
    tmp_path: Path,
) -> None:
    # Verify study_data does not import duckdb anywhere in its AST.
    import ast

    import study_data as sd

    tree = ast.parse((STUDIES_DIR / "study_data.py").read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "duckdb" not in imported

    start_dt = datetime(2026, 5, 1, 0, 0, tzinfo=UTC)
    split_dt = datetime(2026, 5, 2, 0, 0, tzinfo=UTC)
    end_dt = datetime(2026, 5, 3, 0, 0, tzinfo=UTC)

    # Write mixed extra columns (day + direction + is_liquidation) and non-perp coins (@1, xyz:BTC).
    p1 = tmp_path / "f1.parquet"
    pl.DataFrame(
        [
            {
                "wallet": "0xAAA",
                "exchange_ts": datetime(2026, 5, 1, 0, 10, tzinfo=UTC),
                "coin": "BTC",
                "dex": "hyperliquid",
                "side": "buy",
                "size": 1.0,
                "start_position": 0.0,
                "trade_id": 1,
                "direction": "Open Long",
                "is_liquidation": False,
                "day": datetime(2026, 5, 1, tzinfo=UTC).date(),
            },
            {
                "wallet": "0xAAA",
                "exchange_ts": datetime(2026, 5, 1, 0, 20, tzinfo=UTC),
                "coin": "@107",
                "dex": "hyperliquid",
                "side": "buy",
                "size": 5.0,
                "start_position": 0.0,
                "trade_id": 2,
                "direction": "Spot Dust",
                "is_liquidation": False,
                "day": datetime(2026, 5, 1, tzinfo=UTC).date(),
            },
        ]
    ).with_columns(
        pl.col("exchange_ts").cast(pl.Datetime("us", "UTC")),
        pl.col("trade_id").cast(pl.UInt64),
    ).write_parquet(p1)

    p2 = tmp_path / "f2.parquet"
    pl.DataFrame(
        [
            {
                "wallet": "0xaaa",
                "exchange_ts": datetime(2026, 5, 2, 23, 30, tzinfo=UTC),
                "coin": "BTC",
                "dex": "hyperliquid",
                "side": "sell",
                "size": 1.0,
                "start_position": 1.0,
                "trade_id": 3,
                "direction": "Close Long",
                "is_liquidation": False,
            },
            {
                "wallet": "0xaaa",
                "exchange_ts": datetime(2026, 5, 2, 23, 35, tzinfo=UTC),
                "coin": "xyz:CL",
                "dex": "hyperliquid",
                "side": "buy",
                "size": 2.0,
                "start_position": 0.0,
                "trade_id": 4,
                "direction": "Open Long",
                "is_liquidation": False,
            },
        ]
    ).with_columns(
        pl.col("exchange_ts").cast(pl.Datetime("us", "UTC")),
        pl.col("trade_id").cast(pl.UInt64),
    ).write_parquet(p2)

    rec = cr.WalletRecord(wallet_id=1, address="0xaaa", alias="W1", configured=True, excluded=False)
    pipeline = sd.build_fill_lazy_pipeline([rec], [p1, p2], start=start_dt, end=end_dt, split=split_dt)
    plan = pipeline.explain_combined()

    # CSE shares one scan via CACHE across stats_lf and fills_lf, with projection & predicate pushdowns.
    assert "CACHE[" in plan
    assert plan.count("Parquet SCAN") == 1
    assert "PROJECT 8/11 COLUMNS" in plan
    assert "SELECTION:" in plan
    assert "ROW_INDEX: seq" in plan

    res_stream = sd.ingest_fills([rec], [p1, p2], start=start_dt, end=end_dt, split=split_dt, streaming=True)
    res_auto = sd.ingest_fills([rec], [p1, p2], start=start_dt, end=end_dt, split=split_dt, streaming=False)

    assert sorted(res_stream.grouped["W1"]) == ["BTC"]
    assert res_stream.grouped["W1"]["BTC"].equals(res_auto.grouped["W1"]["BTC"])
    assert res_stream.stats_df.equals(res_auto.stats_df)
    stats_row = res_stream.stats_df.to_dicts()[0]
    assert stats_row["train_fills"] == 1
    assert stats_row["oos_fills"] == 1
    assert stats_row["total_fills"] == 2


def test_lazy_prices_mixed_schema_ms_first_and_us_first_with_pushdown(tmp_path: Path) -> None:
    import numpy as np
    import study_data as sd

    store = tmp_path / "store"
    start_dt = datetime(2026, 5, 1, 0, 0, tzinfo=UTC)
    end_dt = datetime(2026, 5, 3, 0, 0, tzinfo=UTC)
    grid = np.arange(
        int(start_dt.timestamp() * 1000),
        int(end_dt.timestamp() * 1000),
        300_000,
        dtype=np.int64,
    )

    # Coin MS_FIRST: day=01 has ms + 5 cols, day=02 has us + 9 cols
    d1 = (
        store
        / "crypto_futures_ohlcv_1m_HYPERLIQUID_MSFIRST_PERP"
        / "year=2026"
        / "month=05"
        / "day=01"
        / "data.parquet"
    )
    d1.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "timestamp": [
                datetime(2026, 4, 30, 23, 59, tzinfo=UTC),  # out-of-window before start
                datetime(2026, 5, 1, 0, 1, tzinfo=UTC),
                datetime(2026, 5, 1, 0, 4, tzinfo=UTC),
            ],
            "symbol": ["MSFIRST", "MSFIRST", "MSFIRST"],
            "open": [999.0, 101.0, 102.0],
            "close": [999.0, 101.5, 102.5],
            "volume": [10.0, 20.0, 30.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC"))).write_parquet(d1)

    d2 = (
        store
        / "crypto_futures_ohlcv_1m_HYPERLIQUID_MSFIRST_PERP"
        / "year=2026"
        / "month=05"
        / "day=02"
        / "data.parquet"
    )
    d2.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "timestamp": [
                datetime(2026, 5, 2, 0, 1, tzinfo=UTC),
                datetime(2026, 5, 3, 0, 0, tzinfo=UTC),  # out-of-window at end
            ],
            "symbol": ["MSFIRST", "MSFIRST"],
            "open": [201.0, 999.0],
            "high": [205.0, 999.0],
            "low": [200.0, 999.0],
            "close": [202.0, 999.0],
            "volume": [40.0, 50.0],
            "trade_count": pl.Series([7, 8], dtype=pl.UInt32),
            "quote_volume": [8000.0, 9000.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC"))).write_parquet(d2)

    # Coin US_FIRST: day=01 has us + 9 cols, day=02 has ms + 5 cols
    u1 = (
        store
        / "crypto_futures_ohlcv_1m_HYPERLIQUID_USFIRST_PERP"
        / "year=2026"
        / "month=05"
        / "day=01"
        / "data.parquet"
    )
    u1.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "timestamp": [datetime(2026, 5, 1, 0, 1, tzinfo=UTC)],
            "symbol": ["USFIRST"],
            "open": [50.0],
            "high": [51.0],
            "low": [49.0],
            "close": [50.5],
            "volume": [10.0],
            "trade_count": pl.Series([3], dtype=pl.UInt32),
            "quote_volume": [500.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC"))).write_parquet(u1)

    u2 = (
        store
        / "crypto_futures_ohlcv_1m_HYPERLIQUID_USFIRST_PERP"
        / "year=2026"
        / "month=05"
        / "day=02"
        / "data.parquet"
    )
    u2.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "timestamp": [datetime(2026, 5, 2, 0, 1, tzinfo=UTC)],
            "symbol": ["USFIRST"],
            "open": [60.0],
            "close": [61.0],
            "volume": [15.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC"))).write_parquet(u2)

    prices = sd.LazyPrices(grid, store_root=store, vike_root=tmp_path / "no_vike")
    plan_ms = prices.explain("MSFIRST")
    assert "Parquet SCAN" in plan_ms
    assert "PROJECT" in plan_ms
    assert "SELECTION:" in plan_ms

    trade_ms, mark_ms = prices.get("MSFIRST")
    assert trade_ms[0] == pytest.approx(101.0)
    assert mark_ms[0] == pytest.approx(102.5)
    assert trade_ms[288] == pytest.approx(201.0)
    assert mark_ms[288] == pytest.approx(202.0)
    # No out-of-window bar leakage
    assert np.isnan(trade_ms[-1])
    assert prices.stats["MSFIRST"]["status"] == "STORE_ONLY"

    trade_us, mark_us = prices.get("USFIRST")
    assert trade_us[0] == pytest.approx(50.0)
    assert trade_us[288] == pytest.approx(60.0)
    assert mark_us[0] == pytest.approx(50.5)


def test_lazy_prices_vike_fallback_precedence_and_stale_180m_boundary(tmp_path: Path) -> None:
    import numpy as np
    import study_data as sd

    store = tmp_path / "store"
    vike = tmp_path / "vike"
    vike.mkdir(parents=True, exist_ok=True)

    start_dt = datetime(2026, 5, 1, 0, 0, tzinfo=UTC)
    end_dt = datetime(2026, 5, 1, 6, 0, tzinfo=UTC)
    grid = np.arange(
        int(start_dt.timestamp() * 1000),
        int(end_dt.timestamp() * 1000),
        300_000,
        dtype=np.int64,
    )

    # Store provides 00:01Z (must beat conflicting Vike 00:01Z) and 00:04Z.
    d1 = (
        store
        / "crypto_futures_ohlcv_1m_HYPERLIQUID_ETH_PERP"
        / "year=2026"
        / "month=05"
        / "day=01"
        / "data.parquet"
    )
    d1.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "timestamp": [datetime(2026, 5, 1, 0, 1, tzinfo=UTC)],
            "open": [100.0],
            "close": [100.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC"))).write_parquet(d1)

    # Vike provides conflicting 00:01Z (should be ignored) and missing 00:06Z (should fill gap).
    # Last bar is 00:06Z:
    # - At grid=03:00Z (T+4m = 03:04Z): age = 178m <= 180m -> valid!
    # - At grid=03:05Z (T+4m = 03:09Z): age = 183m > 180m -> stale NaN!
    pl.DataFrame(
        {
            "ts_ms": [
                int(datetime(2026, 5, 1, 0, 1, tzinfo=UTC).timestamp() * 1000),
                int(datetime(2026, 5, 1, 0, 6, tzinfo=UTC).timestamp() * 1000),
            ],
            "open": [999.0, 106.0],
            "close": [999.0, 106.5],
        }
    ).write_parquet(vike / "ETH.parquet")

    prices = sd.LazyPrices(grid, store_root=store, vike_root=vike, split_ms=int(grid[12]))
    trade, mark = prices.get("ETH")

    assert trade[0] == pytest.approx(100.0)  # Store overrides Vike at 00:01Z
    assert trade[1] == pytest.approx(106.0)  # Vike fills missing 00:06Z (T+1m for 00:05Z)
    assert mark[0] == pytest.approx(100.0)
    assert mark[1] == pytest.approx(106.5)
    # 03:00Z is index 36 (T+4m = 03:04Z, 178m since 00:06Z -> valid)
    assert mark[36] == pytest.approx(106.5)
    # 03:05Z is index 37 (T+4m = 03:09Z, 183m since 00:06Z -> stale NaN)
    assert np.isnan(mark[37])
    assert prices.stats["ETH"]["status"] == "STORE_PLUS_VIKE"

    tr_miss, mk_miss = prices.get("MISSING")
    assert np.all(np.isnan(tr_miss)) and np.all(np.isnan(mk_miss))
    assert prices.stats["MISSING"]["status"] == "NO_LOCAL_OR_VIKE_PRICE"


def test_historical_train_eligibility_separated_from_live_deployability_with_sqlite_and_parquet(
    tmp_path: Path,
) -> None:
    import sqlite3

    import numpy as np
    import study_data as sd

    db_path = tmp_path / "registry.sqlite"
    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE wallets (wallet_id INTEGER PRIMARY KEY, address TEXT);
        CREATE TABLE monitoring (wallet_id INTEGER PRIMARY KEY, configured INTEGER);
        CREATE TABLE exclusions (wallet_id INTEGER PRIMARY KEY, reason TEXT);
        CREATE TABLE wallet_aliases (wallet_id INTEGER, alias TEXT, known_at TEXT);
        INSERT INTO wallets VALUES
            (1, '0x111'),
            (2, '0x222'),
            (3, '0x333'),
            (4, '0x444');
        INSERT INTO monitoring VALUES (1, 1), (2, 1), (3, 1), (4, 1);
        INSERT INTO wallet_aliases VALUES
            (1, 'BOTH_OK', '2026-04-01'),
            (2, 'HIST_OK_WITHDRAWN_DELISTED', '2026-04-01'),
            (3, 'HIST_LOW_FILLS_HIGH_AV', '2026-04-01'),
            (4, 'HIST_NO_PRICE', '2026-04-01');
        """
    )
    con.commit()
    con.close()

    start_dt = datetime(2026, 5, 1, 0, 0, tzinfo=UTC)
    split_dt = datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
    end_dt = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    split_ms = int(split_dt.timestamp() * 1000)

    # Synthetic fills covering [start, end)
    rows: list[dict] = []
    tid = 1
    for i in range(16):
        ts = datetime(2026, 8, 20, i, 0, tzinfo=UTC)
        rows.append(
            {
                "wallet": "0x111",
                "exchange_ts": ts,
                "coin": "BTC",
                "dex": "hyperliquid",
                "side": "buy" if i % 2 == 0 else "sell",
                "size": 1.0,
                "start_position": float(i % 2),
                "trade_id": tid,
            }
        )
        tid += 1
        rows.append(
            {
                "wallet": "0x222",
                "exchange_ts": ts,
                "coin": "OLDCOIN",  # historically priced, but NOT on present-day Bybit
                "dex": "hyperliquid",
                "side": "buy" if i % 2 == 0 else "sell",
                "size": 1.0,
                "start_position": float(i % 2),
                "trade_id": tid,
            }
        )
        tid += 1
        rows.append(
            {
                "wallet": "0x444",
                "exchange_ts": ts,
                "coin": "NOPRICE",  # listed on Bybit today, but missing historical 1m prices
                "dex": "hyperliquid",
                "side": "buy" if i % 2 == 0 else "sell",
                "size": 1.0,
                "start_position": float(i % 2),
                "trade_id": tid,
            }
        )
        tid += 1

    # Boundary fills + OOS-only unmapped coin on BOTH_OK (must NOT leak into Train screen)
    rows.extend(
        [
            {
                "wallet": "0x111",
                "exchange_ts": datetime(2026, 5, 1, 0, 5, tzinfo=UTC),
                "coin": "BTC",
                "dex": "hyperliquid",
                "side": "buy",
                "size": 1.0,
                "start_position": 0.0,
                "trade_id": tid,
            },
            {
                "wallet": "0x111",
                "exchange_ts": datetime(2026, 10, 31, 23, 55, tzinfo=UTC),
                "coin": "OOS_ONLY_BAD",
                "dex": "hyperliquid",
                "side": "buy",
                "size": 1.0,
                "start_position": 0.0,
                "trade_id": tid + 1,
            },
            {
                "wallet": "0x333",
                "exchange_ts": datetime(2026, 8, 25, 0, 0, tzinfo=UTC),
                "coin": "BTC",
                "dex": "hyperliquid",
                "side": "buy",
                "size": 1.0,
                "start_position": 0.0,
                "trade_id": tid + 2,
            },
        ]
    )
    fills_path = _write_fill_parquet(tmp_path / "fills.parquet", rows)

    # Synthetic 1m OHLCV for BTC and OLDCOIN (not NOPRICE)
    store = tmp_path / "store"
    for coin in ("BTC", "OLDCOIN"):
        p = (
            store
            / f"crypto_futures_ohlcv_1m_HYPERLIQUID_{coin}_PERP"
            / "year=2026"
            / "month=08"
            / "day=20"
            / "data.parquet"
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "timestamp": [datetime(2026, 8, 20, 0, 1, tzinfo=UTC)],
                "open": [100.0],
                "close": [100.0],
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC"))).write_parquet(p)

    records, _ = cr.load_registry_universe(db_path)
    ingestion = sd.ingest_fills(records, [fills_path], start=start_dt, end=end_dt, split=split_dt)

    av_map = {
        "0x111": (5000.0, None, None),
        "0x222": (250.0, None, None),  # withdrawn (< $1,000)
        "0x333": (25_000.0, None, None),
        "0x444": (2500.0, None, None),
    }
    screen_rows = sd.screen_wallets(
        records,
        ingestion,
        split=split_dt,
        bybit_usdt_perps={"BTC", "NOPRICE"},
        account_value_fetcher=lambda rec: av_map[rec.address],
    )
    by_alias = {r["alias"]: r for r in screen_rows}

    # BOTH_OK passes both historical and live, and excludes OOS_ONLY_BAD from Train coins.
    assert by_alias["BOTH_OK"]["historical_eligible"] is True
    assert by_alias["BOTH_OK"]["historical_reasons"] == []
    assert by_alias["BOTH_OK"]["live_deployable"] is True
    assert by_alias["BOTH_OK"]["live_reasons"] == []
    assert by_alias["BOTH_OK"]["historical_coins"] == ["BTC"]
    assert by_alias["BOTH_OK"]["mapped_coins"] == ["BTC"]

    # HIST_OK_WITHDRAWN_DELISTED remains historically eligible despite AV < $1000 & Bybit delisting!
    assert by_alias["HIST_OK_WITHDRAWN_DELISTED"]["historical_eligible"] is True
    assert by_alias["HIST_OK_WITHDRAWN_DELISTED"]["historical_reasons"] == []
    assert by_alias["HIST_OK_WITHDRAWN_DELISTED"]["live_deployable"] is False
    assert set(by_alias["HIST_OK_WITHDRAWN_DELISTED"]["live_reasons"]) == {
        "REJECT_UNMAPPED_MATERIAL",
        "WITHDRAWN_LT_1000",
    }

    # HIST_LOW_FILLS_HIGH_AV fails historical Train check, while live operational check passes.
    assert by_alias["HIST_LOW_FILLS_HIGH_AV"]["historical_eligible"] is False
    assert by_alias["HIST_LOW_FILLS_HIGH_AV"]["historical_reasons"] == ["TRAIN_FILLS_LT_15"]
    assert by_alias["HIST_LOW_FILLS_HIGH_AV"]["live_deployable"] is True
    assert by_alias["HIST_LOW_FILLS_HIGH_AV"]["live_reasons"] == []

    # Now apply historical price coverage gate via LazyPrices
    grid = np.arange(
        int(start_dt.timestamp() * 1000),
        int(end_dt.timestamp() * 1000),
        300_000,
        dtype=np.int64,
    )
    prices = sd.LazyPrices(grid, store_root=store, vike_root=None, split_ms=split_ms)

    hist_chosen, updated_rows = sd.eligible_fills(
        ingestion.grouped,
        screen_rows,
        prices,
        split=split_dt,
        scope="historical",
    )
    updated_by_alias = {r["alias"]: r for r in updated_rows}
    # HIST_OK_WITHDRAWN_DELISTED stays in historical replay; HIST_NO_PRICE fails historical price check.
    assert sorted(hist_chosen) == ["BOTH_OK", "HIST_OK_WITHDRAWN_DELISTED"]
    assert "OLDCOIN" in hist_chosen["HIST_OK_WITHDRAWN_DELISTED"]
    assert updated_by_alias["HIST_NO_PRICE"]["historical_eligible"] is False
    assert "PRICE_MISSING_MATERIAL" in updated_by_alias["HIST_NO_PRICE"]["historical_reasons"]
    # Live deployability of HIST_NO_PRICE wasn't corrupted by historical price check; HIST_OK_WITHDRAWN_DELISTED still not live deployable.
    assert updated_by_alias["HIST_OK_WITHDRAWN_DELISTED"]["live_deployable"] is False

    # Live scope restricts to BOTH historical_eligible AND live_deployable.
    live_chosen, _ = sd.eligible_fills(
        ingestion.grouped,
        updated_rows,
        prices,
        split=split_dt,
        scope="live",
    )
    assert list(live_chosen) == ["BOTH_OK"]


def test_large_same_ms_burst_greedy_repair_and_edge_screen_guards(tmp_path: Path) -> None:
    import study_data as sd

    start_dt = datetime(2026, 5, 1, 0, 0, tzinfo=UTC)
    split_dt = datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
    end_dt = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    burst_ts = datetime(2026, 5, 1, 0, 5, tzinfo=UTC)

    # 8-fill same-ms burst (exercises > 6 greedy continuity path) reversed in raw arrival order.
    burst_rows = [
        {
            "wallet": "0xaaa",
            "exchange_ts": burst_ts,
            "coin": "BTC",
            "dex": "hyperliquid",
            "side": "buy",
            "size": 1.0,
            "start_position": float(7 - idx),
            "trade_id": 1000 + idx,
        }
        for idx in range(8)
    ]
    # Add 8 more later Train fills (>45d before Sep 1 -> inactive!) plus an end-of-window fill.
    extra_rows = [
        {
            "wallet": "0xaaa",
            "exchange_ts": datetime(2026, 5, 2, idx, 0, tzinfo=UTC),
            "coin": "BTC",
            "dex": "hyperliquid",
            "side": "buy",
            "size": 1.0,
            "start_position": float(8 + idx),
            "trade_id": 2000 + idx,
        }
        for idx in range(8)
    ]
    oos_row = {
        "wallet": "0xaaa",
        "exchange_ts": datetime(2026, 10, 31, 23, 30, tzinfo=UTC),
        "coin": "BTC",
        "dex": "hyperliquid",
        "side": "sell",
        "size": 1.0,
        "start_position": 16.0,
        "trade_id": 3000,
    }
    p = _write_fill_parquet(tmp_path / "burst8.parquet", burst_rows + extra_rows + [oos_row])
    rec = cr.WalletRecord(wallet_id=1, address="0xaaa", alias="W_INACTIVE", configured=True, excluded=False)

    ing = sd.ingest_fills([rec], [p], start=start_dt, end=end_dt, split=split_dt)
    btc = ing.grouped["W_INACTIVE"]["BTC"]
    assert btc["pre"][:8].to_list() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    def failing_fetcher(_rec: cr.WalletRecord) -> tuple[float | None, dict | None, str | None]:
        raise OSError("timeout")

    rows = sd.screen_wallets(
        [rec],
        ing,
        split=split_dt,
        bybit_usdt_perps={"BTC"},
        account_value_fetcher=failing_fetcher,
    )
    row = rows[0]
    assert row["historical_eligible"] is False
    assert row["historical_reasons"] == ["TRAIN_INACTIVE_45D"]
    assert row["live_deployable"] is False
    assert row["live_reasons"] == ["ACCOUNT_VALUE_UNPROVEN"]

    # Direct UTC coverage assertions
    start_ms = sd.to_utc_ms(start_dt)
    end_ms = sd.to_utc_ms(end_dt)
    with pytest.raises(ValueError, match="UTC_COVERAGE_EMPTY"):
        sd.assert_utc_coverage(None, None, start_ms, end_ms)
    with pytest.raises(ValueError, match="UTC_COVERAGE_OUT_OF_WINDOW"):
        sd.assert_utc_coverage(start_ms - 1, end_ms - 1, start_ms, end_ms)
    with pytest.raises(ValueError, match="UTC_COVERAGE_START_SHORTFALL"):
        sd.assert_utc_coverage(start_ms + 3_700_000, end_ms - 1000, start_ms, end_ms)
