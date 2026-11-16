"""Coin-level venue launch cannot create prelaunch fills or adopt stale exposure."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies"))
cr = importlib.import_module("canonical_replay")


def _fills(rows: list[tuple[int, float, int, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema=["ts", "pre", "sign", "sz"],
        orient="row",
    ).with_columns(pl.col("ts").cast(pl.Int64))


def test_new_only_waits_for_fresh_reopen_after_bybit_launch() -> None:
    grid = 1_800_000_000_000 + np.arange(18, dtype=np.int64) * 300_000
    rows = [
        (int(grid[1]) + 1000, 0.0, 1, 5.0),   # Lead opened pre-listing
        (int(grid[8]) + 1000, 5.0, 1, 3.0),   # Lead adds post-listing: still stale
        (int(grid[11]) + 1000, 8.0, -1, 8.0), # Flat
        (int(grid[13]) + 1000, 0.0, 1, 2.0),  # Genuine new position
    ]
    fills = {"W": {"COIN": _fills(rows)}}
    px = np.full(grid.size, 100.0)
    out, _ = cr.build_unit_replay_by_alias(
        fills, {"COIN": (px, px)}, ["W"], grid,
        policy="new",
        activation_ms=int(grid[0]),
        symbol_activation_ms={"COIN": int(grid[6])},
    )
    pos = out["W"]["COIN"]["pos"].to_numpy()
    assert np.all(pos[:14] == 0.0)
    assert np.all(pos[14:] == 2.0)
    assert np.count_nonzero(out["W"]["COIN"]["delta"].to_numpy()) == 1


def test_full_target_only_starts_when_symbol_historically_available() -> None:
    grid = 1_800_000_000_000 + np.arange(12, dtype=np.int64) * 300_000
    fills = {"W": {"COIN": _fills([(int(grid[1]) + 1000, 0.0, 1, 5.0)])}}
    px = np.full(grid.size, 100.0)
    out, _ = cr.build_unit_replay_by_alias(
        fills, {"COIN": (px, px)}, ["W"], grid,
        policy="full",
        symbol_activation_ms={"COIN": int(grid[6])},
    )
    pos = out["W"]["COIN"]["pos"].to_numpy()
    assert np.all(pos[:6] == 0)
    assert np.all(pos[6:] == 5)


def test_default_without_listing_bound_unchanged() -> None:
    grid = 1_800_000_000_000 + np.arange(12, dtype=np.int64) * 300_000
    fills = {"W": {"COIN": _fills([(int(grid[1]) + 1000, 0.0, 1, 5.0)])}}
    px = np.full(grid.size, 100.0)
    a, _ = cr.build_unit_replay_by_alias(fills, {"COIN": (px, px)}, ["W"], grid)
    b, _ = cr.build_unit_replay_by_alias(
        fills, {"COIN": (px, px)}, ["W"], grid, symbol_activation_ms={}
    )
    for field in ("pos", "delta", "gross", "cost", "net"):
        np.testing.assert_array_equal(
            a["W"]["COIN"][field].to_numpy(),
            b["W"]["COIN"][field].to_numpy(),
        )
