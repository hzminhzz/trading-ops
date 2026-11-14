"""Provider-order market recovery and isolated follower gap exits."""

from __future__ import annotations

import importlib
import json
import math
import sys
from array import array
from datetime import UTC, datetime, timedelta
from decimal import Decimal as D
from pathlib import Path

import polars as pl
import pytest

STUDIES = Path(__file__).resolve().parents[1] / "studies"
sys.path.insert(0, str(STUDIES))
fallback = importlib.import_module("market_data_fallback")
historical = importlib.import_module("historical_5m_wallet_risk")


def test_mlp_vike_cached_vike_reservoir_priorities(tmp_path: Path):
    start = 1_700_000_040_000
    assert start % 60_000 == 0
    vike = tmp_path / "mcp"
    old = tmp_path / "vike_legacy"
    reservoir = tmp_path / "reservoir"
    for directory in (vike, old, reservoir):
        directory.mkdir()
    (vike / "BTC_000.json").write_text(
        json.dumps(
            {
                "symbol": "BTC",
                "method": "vike_connected_mcp",
                "source": "1m fills rollup",
                "rows": [
                    [start + 60_000, 101, 102, 10],
                    [start + 120_000, 103, 104, 11],
                ],
            }
        )
    )
    pl.DataFrame(
        {"ts_ms": [start + 120_000, start + 180_000], "close": [900.0, 105.0]}
    ).write_parquet(old / "BTC.parquet")
    pl.DataFrame(
        {
            "ts_ms": [start + 180_000, start + 240_000],
            "close": [900.0, 106.0],
            "volume": [13.0, 14.0],
        }
    ).write_parquet(reservoir / "BTC_1m.parquet")
    rec, audit = fallback.load_market_recovery(
        "BTC",
        start_ms=start,
        end_ms=start + 300_000,
        existing=[100.0, math.nan, math.nan, math.nan, math.nan],
        mcp_roots=(vike,),
        legacy_root=old,
        reservoir_root=reservoir,
    )
    assert [(t, p) for t, p, _ in rec] == [
        (start + 60_000, 102.0),
        (start + 120_000, 104.0),
        (start + 180_000, 105.0),
        (start + 240_000, 106.0),
    ]
    assert audit["recovered_rows_by_source"]["VIKE_CONNECTED_MCP_1M_FILLED"] == 2
    assert audit["recovered_rows_by_source"]["VIKE_PREVIOUS_LOCAL_CACHE_1M_FILLED"] == 1
    assert audit["recovered_rows_by_source"]["HYDROMANCER_RESERVOIR_NORMALIZED_1S_FILLED"] == 1
    assert audit["reservoir_fallback_checked"] is True
    assert audit["reservoir_s3_authenticated"] is False
    assert audit["connected_vike_receipts"][0]["sha256"]
    assert audit["legacy_vike_cache_receipt"][0]["sha256"]


def test_vike_invalid_candle_does_not_silently_fill(tmp_path: Path):
    start = 1_700_000_040_000
    root = tmp_path / "mcp"
    root.mkdir()
    (root / "BTC_000.json").write_text(
        json.dumps(
            {
                "symbol": "BTC",
                "method": "vike_connected_mcp",
                "source": "1m fills rollup",
                "rows": [[start, 10, -99, 1]],
            }
        )
    )
    with pytest.raises(ValueError, match="INVALID_RECOVERY_CANDLE_PRICE"):
        fallback.load_market_recovery(
            "BTC",
            start_ms=start,
            end_ms=start + 60_000,
            existing=[math.nan],
            mcp_roots=(root,),
            legacy_root=tmp_path,
            reservoir_root=tmp_path,
        )


def test_sparse_trades_within_staleness_window_not_treated_as_outage():
    start = datetime(2026, 7, 1, tzinfo=UTC)
    bars = historical.LocalBars(start, start + timedelta(minutes=45))
    bars.prices = {"BTC": array("d", [math.nan] * 45)}
    bars.prices["BTC"][0] = 100.0
    for minute in (15, 30):
        assert bars.mark(
            "BTC",
            historical.ms(start) + minute * 60_000,
            tolerance=historical.MARK_STALE_TOLERANCE_MINUTES,
        ) == D(100)


def test_unavailable_single_pair_closes_its_follower_only_then_continues():
    start = datetime(2026, 7, 1, tzinfo=UTC)
    end = start + timedelta(minutes=100)
    bars = historical.LocalBars(start, end)
    btc = array("d", [100.0] * 100)
    for i in range(11, 100):
        btc[i] = math.nan
    bars.prices = {"BTC": btc, "ETH": array("d", [200.0] * 100)}
    bars.volumes = {
        "BTC": array("d", [2.0] * 100),
        "ETH": array("d", [2.0] * 100),
    }
    wallet = historical.WalletSource(
        alias="TEST",
        address="",
        anchor_ms=0,
        anchor={},
        initial={},
        fills=(
            historical.Fill(
                t=historical.ms(start) + 60_000, symbol="BTC", delta=D(1), start_position=D(0)
            ),
            historical.Fill(
                t=historical.ms(start) + 60_000, symbol="ETH", delta=D(1), start_position=D(0)
            ),
            historical.Fill(
                t=historical.ms(start) + 61 * 60_000, symbol="BTC", delta=D(1), start_position=D(1)
            ),
            historical.Fill(
                t=historical.ms(start) + 61 * 60_000, symbol="ETH", delta=D(1), start_position=D(1)
            ),
        ),
        snapshots=(),
        symbols=("BTC", "ETH"),
        validation={},
    )
    result = historical.replay_unit(
        wallet, bars, start=start, train_end=start + timedelta(minutes=40), end=end
    )
    assert result["complete_window_minutes"] is True
    assert result["market_data_staleness_minutes"] == 30
    assert result["counters"]["marked_minutes"] == 100
    assert len(result["hypothetical_forced_exits"]) == 1
    exit_ = result["hypothetical_forced_exits"][0]
    assert exit_["symbol"] == "BTC"
    assert exit_["executed"] is False
    assert exit_["adverse_stress_bps"] == "50"
    assert exit_["last_price_age_min"] >= 31
    assert result["disabled_follower_symbols_after_data_gap"] == ["BTC"]
    assert result["counters"]["modeled_unit_orders"] >= 3
    assert result["train_unit_max_drawdown_usd"]
    assert result["hypothetical_exits_are_not_executable_fills"] is True


def test_published_private_vike_price_cache_is_immutable(tmp_path: Path):
    publisher = importlib.import_module("publish_vike_market_recovery")
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir(mode=0o700)
    receipt = source / "BTC_000.json"
    first = {
        "symbol": "BTC",
        "source": "1m fills rollup",
        "method": "vike_connected_mcp",
        "rows": [[60_000, 100, 101, 10]],
    }
    receipt.write_text(json.dumps(first))
    first_report = publisher.publish(source=source, destination=destination)
    assert first_report["files"] == 1
    assert first_report["copied"] == 1
    assert first_report["rows_total"] == 1
    assert (destination / "BTC_000.json").stat().st_mode & 0o077 == 0
    assert destination.stat().st_mode & 0o077 == 0
    second_report = publisher.publish(source=source, destination=destination)
    assert second_report["reused"] == 1
    assert second_report["fingerprint"] == first_report["fingerprint"]
    first["rows"][0][2] = 105
    receipt.write_text(json.dumps(first))
    with pytest.raises(ValueError, match="IMMUTABLE_VIKE_RECEIPT_HASH_CONFLICT"):
        publisher.publish(source=source, destination=destination)
