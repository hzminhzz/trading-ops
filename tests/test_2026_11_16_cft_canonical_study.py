"""Leakage and sizing gates for the May-Nov CFT canonical research orchestrator."""
from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

SCRIPT = Path(__file__).resolve().parents[1] / "studies/2026-11-16__cft__10k__canonical-full-universe.py"
SPEC = importlib.util.spec_from_file_location("cft_20261116_canonical", SCRIPT)
assert SPEC and SPEC.loader
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def fills(rows: list[tuple[int, int, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema=["ts", "sign", "sz", "pre"],
        orient="row",
    )


def test_exact_train_oos_windows() -> None:
    assert datetime.fromtimestamp(study.START / 1000, UTC) == datetime(2026, 5, 1, tzinfo=UTC)
    assert datetime.fromtimestamp(study.SPLIT / 1000, UTC) == datetime(2026, 9, 1, tzinfo=UTC)
    assert datetime.fromtimestamp(study.END / 1000, UTC) == datetime(2026, 11, 1, tzinfo=UTC)
    assert study.TRAIN_GRID.size == 123 * 288
    assert study.OOS_GRID.size == 61 * 288


def test_selection_symbol_screen_excludes_oos_new_symbol() -> None:
    rec = study.cr.WalletRecord(1, "0x" + "0" * 40, "TEST", True, False)
    a = fills([(study.START + 60000, 1, 1.0, 0.0)])
    b = fills([(study.SPLIT + 60000 + i, 1, 1.0, 0.0) for i in range(10)])
    result = study.summarize_wallet(rec, {"BTC": a, "BAD": b}, {"BTC"})
    assert result["symbol_status"] == "EXACT_MAPPED"
    assert result["mapped_coins"] == ["BTC"]
    assert result["train_fills"] == 1
    assert result["oos_fills"] == 10


def test_missing_price_materiality_uses_train_denominator() -> None:
    class Prices:
        def get(self, coin: str) -> tuple[np.ndarray, np.ndarray]:
            arr = np.ones(study.GRID.size)
            if coin == "ETH":
                arr[: study.CUT] = np.nan
            return arr.copy(), arr

    a = fills([(study.START + 60000, 1, 1.0, 0.0)])
    b = fills([(study.START + 60000, 1, 1.0, 0.0)] + [
        (study.SPLIT + i * 60000, 1, 1.0, 0.0) for i in range(200)
    ])
    screen = {
        "alias": "TEST",
        "mapped_coins": ["BTC", "ETH"],
        "train_fills": 2,
        "total_fills": 202,
        "reasons": [],
    }
    chosen, rows = study.eligible_fills({"TEST": {"BTC": a, "ETH": b}}, [screen], Prices())
    assert chosen == {}
    assert rows[0]["price_missing_train_fills"] == 1
    assert rows[0]["price_missing_train_frac"] == 0.5
    assert rows[0]["reasons"] == ["PRICE_MISSING_MATERIAL"]
