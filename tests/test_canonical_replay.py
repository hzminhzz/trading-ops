"""Financial invariant tests for canonical research replay and Guardian rules."""

from __future__ import annotations

import importlib
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

STUDIES_DIR = Path(__file__).resolve().parents[1] / "studies"
sys.path.insert(0, str(STUDIES_DIR))

cr = importlib.import_module("canonical_replay")
pc = importlib.import_module("pod_constructor")


def _grid_ms(n_bars: int, start: datetime = datetime(2026, 5, 1, 0, 0, tzinfo=UTC)) -> np.ndarray:
    start_ms = int(start.timestamp() * 1000)
    return start_ms + np.arange(n_bars, dtype=np.int64) * 300_000


def _fills_df(rows: list[tuple[int, float, int, float]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=["ts", "pre", "sign", "sz"], orient="row").with_columns(
        pl.col("ts").cast(pl.Int64),
        pl.col("pre").cast(pl.Float64),
        pl.col("sign").cast(pl.Int64),
        pl.col("sz").cast(pl.Float64),
    )


def test_full_policy_never_backdates_pre_position_without_anchor() -> None:
    grid = _grid_ms(10)
    later_ts = int(grid[5]) + 1_000
    fills = _fills_df([(later_ts, 10.0, 1, 2.0)])  # pre[0]=10 observed ONLY at bar 5

    unanchored = cr.reconstruct_positions_on_grid(fills, grid, policy="full")
    assert unanchored.unanchored_pre_inventory is True
    assert unanchored.pos[:6].tolist() == [0.0] * 6
    assert unanchored.pos[6:].tolist() == [12.0] * 4

    anchored = cr.reconstruct_positions_on_grid(
        fills, grid, policy="full", initial_anchor_pos=10.0
    )
    assert anchored.unanchored_pre_inventory is False
    assert anchored.pos[:6].tolist() == [10.0] * 6
    assert anchored.pos[6:].tolist() == [12.0] * 4


def test_new_policy_excludes_pre_activation_and_allows_flip_or_reopen() -> None:
    grid = _grid_ms(12)
    act = int(grid[4])
    fills = _fills_df(
        [
            (int(grid[1]) + 1_000, 0.0, 1, 5.0),  # before activation -> +5
            (int(grid[5]) + 1_000, 5.0, 1, 2.0),  # add while active pre-act -> still excluded
            (int(grid[7]) + 1_000, 7.0, -1, 7.0),  # closes back to 0
            (int(grid[9]) + 1_000, 0.0, -1, 4.0),  # new flat-to-short entry
        ]
    )
    res = cr.reconstruct_positions_on_grid(fills, grid, policy="new", activation_ms=act)
    assert res.pos[:10].tolist() == [0.0] * 10
    assert res.pos[10:].tolist() == [-4.0, -4.0]

    flip_fills = _fills_df([(int(grid[5]) + 1_000, 5.0, -1, 8.0)])  # +5 -> -3 flip
    flip_res = cr.reconstruct_positions_on_grid(flip_fills, grid, policy="new", activation_ms=act)
    assert flip_res.pos[:6].tolist() == [0.0] * 6
    assert flip_res.pos[6:].tolist() == [-3.0] * 6


def test_guardian_terminates_on_pass_static_and_daily_breach() -> None:
    n = 40
    grid = _grid_ms(n)
    t_open = np.full((1, n), 100.0)
    m_close = np.full((1, n), 100.0)
    targets = np.zeros((1, n))
    targets[0, 1:] = 10.0
    m_close[0, 2:] = 185.0
    t_open[0, 3:] = 185.0
    # After passing on bar 2 (+850 gross minus fee > 800), huge dump happens on bar 5.
    m_close[0, 5:] = 10.0
    t_open[0, 6:] = 10.0

    spec = cr.GuardianSpec()
    res, _ = cr.simulate_guardian_challenge(["BTC"], targets, t_open, m_close, grid, spec)
    assert res.verdict == "PASS"
    assert res.terminal_step == 2
    assert res.terminal_net_usd >= 800.0
    assert res.static_floor_dd_usd == pytest.approx(-0.65, abs=1e-9)
    assert res.sweeps == 0

    # Daily breach case (instant -550 drop < -500 limit, sweeps also triggered; fails immediately)
    t_open_d = np.full((1, n), 100.0)
    m_close_d = np.full((1, n), 100.0)
    m_close_d[0, 2:] = 45.0
    t_open_d[0, 3:] = 45.0
    res_d, _ = cr.simulate_guardian_challenge(["BTC"], targets, t_open_d, m_close_d, grid, spec)
    assert res_d.verdict == "FAIL_DAILY"
    assert res_d.terminal_step == 2


def test_guardian_sweep_blocks_stale_pre_inventory_after_reset() -> None:
    # Span across Day 1 reset (bar 1) into Day 2 reset (bar 289)
    n = 310
    grid = _grid_ms(n)
    t_open = np.full((1, n), 100.0)
    m_close = np.full((1, n), 100.0)
    targets = np.zeros((1, n))
    targets[0, 2:295] = 10.0  # leader stays long +10 across Day 2 00:05 reset (bar 289)
    targets[0, 298:] = 4.0  # flat at 295..297, then opens a genuinely fresh long at 298

    # Day 1 bar 4 dips to 58 (-420 <= -400 Guardian sweep, stays above -500 daily limit)
    m_close[0, 4:289] = 58.0
    t_open[0, 5:289] = 58.0
    m_close[0, 289:] = 58.0
    t_open[0, 289:] = 58.0
    # After 00:05 reset at 289, move price wildly while stale target persists (289..294)
    m_close[0, 290:296] = 40.0
    t_open[0, 291:296] = 40.0

    spec_new = cr.GuardianSpec(reenter_after_sweep="new_only")
    res_new, _ = cr.simulate_guardian_challenge(["BTC"], targets, t_open, m_close, grid, spec_new)
    assert res_new.sweeps == 1
    # Stays flat through 290..297 (does NOT lose money on 40.0 move before flat+reopen at 298)
    assert np.all(res_new.step_pnl[5:298] == 0.0)
    assert res_new.step_pnl[298] < 0.0  # Pays entry cost only when fresh trade opens at bar 298

    spec_imm = cr.GuardianSpec(reenter_after_sweep="immediate_target")
    res_imm, _ = cr.simulate_guardian_challenge(["BTC"], targets, t_open, m_close, grid, spec_imm)
    assert res_imm.step_pnl[289] < 0.0  # Immediately re-entered stale +10 at reset bar 289


def test_peak_to_trough_dd_distinguished_from_static_floor_dd() -> None:
    n = 30
    grid = _grid_ms(n)
    t_open = np.full((1, n), 100.0)
    m_close = np.full((1, n), 100.0)
    targets = np.zeros((1, n))
    targets[0, 2:] = 10.0
    # Rises +400 first (eq ~10,400), then falls to 9,200 (-1,200 peak-to-trough, -800 static)
    m_close[0, 4:10] = 140.0
    t_open[0, 5:10] = 140.0
    m_close[0, 10:] = 20.0
    t_open[0, 11:] = 20.0

    # Relax daily/sweep caps so only peak vs static floor distinction is tested
    spec = cr.GuardianSpec(
        profit_target=800.0,
        max_static_dd=1000.0,
        daily_loss_limit=2000.0,
        guardian_sweep_loss=2000.0,
    )
    res, _ = cr.simulate_guardian_challenge(["BTC"], targets, t_open, m_close, grid, spec)
    assert res.verdict == "TIMEOUT"
    assert res.peak_to_trough_dd_usd < -1000.0
    assert res.static_floor_dd_usd > -1000.0
    assert res.min_equity_usd > 9000.0


def test_mc_daily_blocks_not_corrupted_by_early_pass() -> None:
    # 20 UTC days: Day 2 hits +800 PASS, Days 3..20 stay non-zero active for MC resampling
    n_days = 20
    n = n_days * 288
    grid = _grid_ms(n)
    t_open = np.full((1, n), 100.0)
    m_close = np.full((1, n), 100.0)
    targets = np.zeros((1, n))
    targets[0, 2:] = 10.0

    prices = 100.0 + np.linspace(0.0, 200.0, n)
    t_open[0, :] = prices
    m_close[0, :] = prices

    spec = cr.GuardianSpec()
    res, blocks = cr.simulate_guardian_challenge(["BTC"], targets, t_open, m_close, grid, spec)
    assert res.verdict == "PASS"
    assert res.terminal_step is not None and 2000 < res.terminal_step < 2500
    assert blocks[1].size >= 18
    assert float(np.sum(np.abs(blocks[1][10:]))) > 0.0

    mc_out = pc.run_stationary_block_mc_on_daily_blocks(
        blocks,
        "Blended",
        account=10_000.0,
        profit_target=800.0,
        max_static_dd=1_000.0,
        daily_loss_limit=500.0,
        n_paths=200,
        horizon_days=30,
        mean_block_days=3.0,
        rng=np.random.default_rng(42),
    )
    assert mc_out["p_pass"] == 100.0


def test_netting_before_sizing_and_pod_constructor_roundtrip() -> None:
    n = 288 * 10
    grid = _grid_ms(n)
    ts_utc = pl.from_epoch(pl.Series("timestamp", grid, dtype=pl.Int64), time_unit="ms")
    ts_utc = ts_utc.dt.replace_time_zone("UTC")

    # Wallet A and B take opposite positions; exact netting cancels 100% of exposure & cost.
    fills_a = _fills_df([(int(grid[10]) + 1000, 0.0, 1, 5.0), (int(grid[200]) + 1000, 5.0, -1, 5.0)])
    fills_b = _fills_df([(int(grid[10]) + 1000, 0.0, -1, 5.0), (int(grid[200]) + 1000, -5.0, 1, 5.0)])
    px = np.full(n, 100.0)
    unit_frames, _ = cr.build_unit_replay_by_alias(
        {"WA": {"BTC": fills_a}, "WB": {"BTC": fills_b}},
        {"BTC": (px, px)},
        ["WA", "WB"],
        grid,
        policy="new",
    )
    tensor = pc.build_tensor_from_unit_replays(unit_frames, grid.tolist(), pod_equity_usd=10_000.0)
    assert tensor.aliases == ("WA", "WB")

    pos_a = cr.reconstruct_positions_on_grid(fills_a, grid, policy="new").pos
    pos_b = cr.reconstruct_positions_on_grid(fills_b, grid, policy="new").pos
    net_targets = cr.net_member_targets(np.stack([pos_a[None, :], pos_b[None, :]]), [0.5, 0.5])
    assert np.all(net_targets == 0.0)

    res, _ = cr.simulate_guardian_challenge(
        ["BTC"], net_targets, px[None, :], px[None, :], grid, cr.GuardianSpec()
    )
    assert res.full_window_net_usd == 0.0


def test_bybit_screen_separates_negligible_tail_from_material_unmapped() -> None:
    bybit = {"BTC", "ETH", "SOL", "1000PEPE"}
    negligible = {
        "BTC": pl.DataFrame({"px": [100.0] * 199, "sz": [1.0] * 199}),
        "kPEPE": pl.DataFrame({"px": [0.01], "sz": [10.0]}),
    }
    res_ok = cr.assess_bybit_symbol_screen("W_SMALL_TAIL", negligible, bybit)
    assert res_ok.status == "MAPPED_WITH_EXCLUDED_TAIL"
    assert res_ok.scale_alias_candidates == {"kPEPE": "1000PEPE"}
    assert res_ok.mapped_coins == ("BTC",)

    material = {
        "BTC": pl.DataFrame({"px": [100.0] * 5, "sz": [1.0] * 5}),
        "PUMP": pl.DataFrame({"px": [10.0] * 5, "sz": [1.0] * 5}),
    }
    res_bad = cr.assess_bybit_symbol_screen("W_MAT", material, bybit)
    assert res_bad.status == "REJECT_UNMAPPED_MATERIAL"


def test_tier1_replay_reconciles_with_pm03_account() -> None:
    n = 30
    grid = _grid_ms(n)
    rng = np.random.default_rng(7)
    # 2 symbols: open, scale, flip, close
    targets = np.zeros((2, n), dtype=np.float64)
    targets[0, 2:10] = 0.5
    targets[0, 10:18] = -0.3  # side flip
    targets[0, 18:25] = 0.0
    targets[1, 4:20] = 2.0

    t_open = np.vstack(
        [
            10_000.0 * (1.0 + np.cumsum(rng.normal(0.0, 0.002, n))),
            500.0 * (1.0 + np.cumsum(rng.normal(0.0, 0.003, n))),
        ]
    )
    m_close = t_open * (1.0 + rng.normal(0.0, 0.001, (2, n)))

    spec = cr.GuardianSpec(
        initial_equity=10_000.0,
        profit_target=50_000.0,
        max_static_dd=50_000.0,
        daily_loss_limit=50_000.0,
        guardian_sweep_loss=50_000.0,
    )
    tier1, _ = cr.simulate_guardian_challenge(["BTC", "ETH"], targets, t_open, m_close, grid, spec)
    tier2 = cr.reconcile_portfolio_with_pm03(["BTC", "ETH"], grid, targets, t_open, m_close)

    assert tier2["account_verified"] is True
    assert tier2["quantities"] == {"BTC": 0.0, "ETH": 0.0}
    # Second-order slippage*fee term is ~2e-4 * 4.5e-4 = 9e-8 of notional (< $0.05 here)
    assert tier1.full_window_net_usd == pytest.approx(tier2["net_pnl"], abs=0.05)


def test_pair_local_gap_exit_without_halting_other_pairs() -> None:
    n = 12
    grid = _grid_ms(n)
    targets = np.zeros((2, n))
    targets[:, 1:] = 1.0
    t_open = np.full((2, n), 100.0)
    m_close = np.full((2, n), 100.0)
    # Coin 0 ("GAP") loses mark feed at bar 5..11 while open; Coin 1 ("OK") rises smoothly
    m_close[0, 5:] = np.nan
    t_open[0, 5:] = np.nan
    m_close[1, 6:] = 120.0
    t_open[1, 7:] = 120.0

    spec = cr.GuardianSpec(
        profit_target=50_000.0,
        max_static_dd=50_000.0,
        daily_loss_limit=50_000.0,
        guardian_sweep_loss=50_000.0,
    )
    res, _ = cr.simulate_guardian_challenge(["GAP", "OK"], targets, t_open, m_close, grid, spec)
    assert res.forced_gap_exits == 1
    assert res.unpriced_open_steps == 1
    assert res.full_window_net_usd > 15.0  # Coin 1 still captured +20 move


def test_load_registry_universe_queries_all_configured_non_excluded(tmp_path: Path) -> None:
    db_path = tmp_path / "registry.sqlite"
    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE wallets (wallet_id INTEGER PRIMARY KEY, address TEXT);
        CREATE TABLE monitoring (wallet_id INTEGER PRIMARY KEY, configured INTEGER);
        CREATE TABLE exclusions (wallet_id INTEGER PRIMARY KEY, reason TEXT);
        CREATE TABLE wallet_aliases (wallet_id INTEGER, alias TEXT, known_at TEXT);
        INSERT INTO wallets VALUES (1, '0xAAA'), (2, '0xBBB'), (3, '0xCCC'), (4, '0xDDD');
        INSERT INTO monitoring VALUES (1, 1), (2, 1), (3, 0), (4, 1);
        INSERT INTO exclusions VALUES (2, 'WITHDRAWN');
        INSERT INTO wallet_aliases VALUES
            (1, 'W015', '2026-10-01'),
            (2, 'EX_WALLET', '2026-10-01'),
            (3, 'UNCFG', '2026-10-01');
        """
    )
    con.commit()
    con.close()

    records, sha1 = cr.load_registry_universe(db_path)
    assert [r.wallet_id for r in records] == [1, 4]
    assert [r.label for r in records] == ["W015", "W#4"]
    assert records[0].address == "0xaaa"

    records_with_ref, sha2 = cr.load_registry_universe(db_path, include_aliases=("UNCFG",))
    assert [r.wallet_id for r in records_with_ref] == [1, 3, 4]
    assert sha1 != sha2
