"""Hyperliquid-only CFT research must not depend on the destination venue."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies"))
research = importlib.import_module("cft_20261116_exhaustive")
cr = importlib.import_module("canonical_replay")


def fills(timestamps: list[int]) -> pl.DataFrame:
    return pl.DataFrame({"ts": timestamps, "sz": [1.0] * len(timestamps)})


def test_hyperliquid_train_screen_ignores_bybit_and_oos_symbols() -> None:
    source = {
        "HYPE": fills([research.SPLIT - 300_000] * 20),
        "FUTURE": fills([research.SPLIT + 300_000] * 50),
    }
    rec = cr.WalletRecord(
        wallet_id=8, address="0xabc", alias="SAMPLE",
        configured=True, excluded=False,
    )
    result = research.summarize_wallet(rec, source)
    assert result["symbol_status"] == "HYPERLIQUID_TRAIN_AVAILABLE"
    assert result["mapped_coins"] == ["HYPE"]
    assert result["unmapped_coins"] == []
    assert result["train_fills"] == 20
    assert result["oos_fills"] == 50
    assert result["reasons"] == []


def test_hyperliquid_train_ignores_current_source_account_status() -> None:
    results = [
        {"alias": "A", "net_usd": 400, "maxdd_usd": 50, "operational_eligible": False},
        {"alias": "B", "net_usd": 200, "maxdd_usd": 50, "operational_eligible": True},
        {"alias": "C", "net_usd": -20, "maxdd_usd": 60, "operational_eligible": True},
    ]
    assert [r["alias"] for r in research.train_passable(
        results, venue_mode="hyperliquid"
    )] == ["A", "B"]
    assert [r["alias"] for r in research.train_passable(
        results, venue_mode="legacy-bybit"
    )] == ["B"]
    with pytest.raises(ValueError, match="UNKNOWN_BACKTEST_VENUE"):
        research.train_passable(results, venue_mode="UNKNOWN")


def test_new_oos_symbols_only_enter_after_frozen_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(research, "VENUE_MODE", "hyperliquid")
    early = fills([research.SPLIT - 300_000] * 20)
    future = fills([research.SPLIT + 300_000] * 20)
    books = {"A": {"HYPE": early, "NEWCOIN": future}}
    train = {"A": {"HYPE": early}}
    selections = [{"aliases": ["A"], "train_opt_k": [0.1]}]
    oos = research.oos_source_books(selections, books, train)
    assert sorted(oos["A"]) == ["HYPE", "NEWCOIN"]
    assert list(train["A"]) == ["HYPE"]
    monkeypatch.setattr(research, "VENUE_MODE", "legacy-bybit")
    assert research.oos_source_books(selections, books, train) is train


def test_hyperliquid_screen_makes_no_bybit_or_current_value_http_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("CROSS_VENUE_API_CALLED")

    rec = cr.WalletRecord(
        wallet_id=8, address="0xabc", alias="SAMPLE",
        configured=True, excluded=False,
    )
    monkeypatch.setattr(research, "VENUE_MODE", "hyperliquid")
    monkeypatch.setattr(research, "bybit_instruments", fail)
    monkeypatch.setattr(research, "request_wallet_record", fail)
    monkeypatch.setattr(research, "paths_inventory", lambda: (
        ["dummy.parquet"],
        {"archive_sha256": "abc", "files": [], "missing_dates": []},
    ))
    monkeypatch.setattr(
        research.cr, "load_registry_universe",
        lambda *_args, **_kwargs: ([rec], "sha"),
    )
    monkeypatch.setattr(
        research, "load_fills",
        lambda *_args, **_kwargs: {"SAMPLE": {
            "HYPE": fills([research.SPLIT - 300_000] * 20)
        }},
    )
    saved = []
    monkeypatch.setattr(research, "dump", lambda *args: saved.append(args))
    inv, _, records = research.prepare_screen()
    assert records == [rec]
    assert inv["backtest_venue"] == "hyperliquid"
    assert inv["bybit"] is None
    assert inv["market_source"] == "HYPERLIQUID_1M_OHLCV"
    assert inv["screen_survivors"] == ["SAMPLE"]
    assert inv["screen"][0]["account_value_status"] == "NOT_FETCHED"
    assert len(saved) == 1
