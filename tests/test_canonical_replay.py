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


def test_guardian_min_trade_notional_usd_skips_small_deltas_but_allows_full_close() -> None:
    n = 8
    grid = _grid_ms(n)
    t_open = np.full((1, n), 100.0)
    m_close = np.full((1, n), 100.0)

    # Bar 1: $10 open (< $15) -> skipped when min_trade_notional_usd=15
    # Bar 2: $20 open (>= $15) -> executed
    # Bar 3: $25 scale (+$5 delta < $15) -> skipped, stays at $20
    # Bar 4: target stays 0.25 (still +$5 delta < $15 for deadband), price rises to 110 -> PnL on 0.20 = +$2.00, vs +$2.50 without deadband
    # Bar 5: full close to 0.0 -> bypasses deadband and closes cleanly
    targets = np.array([[0.0, 0.10, 0.20, 0.25, 0.25, 0.0, 0.0, 0.0]], dtype=np.float64)
    m_close[0, 4:] = 110.0
    t_open[0, 5:] = 110.0

    spec_deadband = cr.GuardianSpec(
        profit_target=50_000.0,
        max_static_dd=50_000.0,
        daily_loss_limit=50_000.0,
        guardian_sweep_loss=50_000.0,
        cost_bps=0.0,
        min_trade_notional_usd=15.0,
    )
    res_db, _ = cr.simulate_guardian_challenge(["BTC"], targets, t_open, m_close, grid, spec_deadband)

    # Bar 1 skipped (0.0); Bar 2 fills 0.20 ($20); Bar 3 skipped (stays 0.20);
    # Bar 4 marks to 110 (+$2.00); Bar 5 closes 0.20 at 110 (no further PnL).
    assert res_db.step_pnl[1] == 0.0
    assert res_db.step_pnl[3] == 0.0
    assert res_db.full_window_net_usd == pytest.approx(2.0, abs=1e-9)

    # Compare with default min_trade_notional_usd=0.0 (all deltas execute, including Bar 3 0.25 -> +$2.50)
    spec_default = cr.GuardianSpec(
        profit_target=50_000.0,
        max_static_dd=50_000.0,
        daily_loss_limit=50_000.0,
        guardian_sweep_loss=50_000.0,
        cost_bps=0.0,
    )
    res_raw, _ = cr.simulate_guardian_challenge(["BTC"], targets, t_open, m_close, grid, spec_default)
    assert res_raw.full_window_net_usd == pytest.approx(2.5, abs=1e-9)

    # Verify full close bypasses deadband even when closing position < $15 notional
    small_close_targets = np.array([[0.0, 0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    t_drop = np.full((1, n), 100.0)
    m_drop = np.full((1, n), 100.0)
    m_drop[0, 2:] = 50.0
    t_drop[0, 2:] = 50.0  # closing 0.20 at $50 = $10 notional (< $15), must still close!
    m_drop[0, 4:] = 10.0
    res_close, _ = cr.simulate_guardian_challenge(["BTC"], small_close_targets, t_drop, m_drop, grid, spec_deadband)
    assert res_close.step_pnl[4] == 0.0
    assert res_close.full_window_net_usd == pytest.approx(-10.0, abs=1e-9)


def test_simulate_guardian_reuses_first_pass_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    n = 5 * 288
    grid = _grid_ms(n)
    t_open = np.full((1, n), 100.0)
    m_close = np.full((1, n), 100.0)
    targets = np.zeros((1, n))
    targets[0, 5:50] = 1.0
    m_close[0, 20:50] = 110.0
    t_open[0, 21:50] = 110.0
    m_close[0, 50:] = 110.0
    t_open[0, 50:] = 110.0

    calls: list[bool] = []
    real_exec = cr._execute_guardian_pass

    def counting_exec(*args, **kwargs):
        calls.append(bool(kwargs.get("stop_on_terminal")))
        return real_exec(*args, **kwargs)

    monkeypatch.setattr(cr, "_execute_guardian_pass", counting_exec)
    spec_timeout = cr.GuardianSpec(profit_target=800.0)
    res_to, blocks_to = cr.simulate_guardian_challenge(["BTC"], targets, t_open, m_close, grid, spec_timeout)

    assert res_to.verdict == "TIMEOUT"
    assert res_to.terminal_step is None
    assert calls == [True]

    # Compare against an explicit continuous pass to ensure 100% metric & MC block parity.
    cont_raw = real_exec(["BTC"], targets, t_open, m_close, grid, spec_timeout, stop_on_terminal=False)
    assert res_to.full_window_net_usd == cont_raw["full_window_net_usd"]
    assert res_to.terminal_net_usd == cont_raw["terminal_net_usd"]
    np.testing.assert_allclose(res_to.step_pnl, cont_raw["step_pnl"])
    np.testing.assert_allclose(res_to.equity_curve, cont_raw["eq_curve"])
    expected_blocks = cr.extract_mc_daily_blocks(
        cont_raw["step_pnl"],
        cont_raw["swept_flags"],
        grid,
        spec_timeout,
        balance_curve=cont_raw["balance_curve"],
    )
    for got_blk, exp_blk in zip(blocks_to, expected_blocks, strict=True):
        np.testing.assert_array_equal(got_blk, exp_blk)

    # Confirm PASS still triggers the second continuous pass for uncorrupted MC daily blocks.
    calls.clear()
    spec_pass = cr.GuardianSpec(profit_target=5.0)
    res_pass, _ = cr.simulate_guardian_challenge(["BTC"], targets, t_open, m_close, grid, spec_pass)
    assert res_pass.verdict == "PASS"
    assert calls == [True, False]


def test_guardian_sweep_obeys_exit_first_precedence() -> None:
    n = 10
    grid = _grid_ms(n)
    spec = cr.GuardianSpec(
        profit_target=50_000.0,
        max_static_dd=50_000.0,
        daily_loss_limit=50_000.0,
        guardian_sweep_loss=400.0,
        cost_bps=10.0,
    )

    # 1. Carried BTC +10 triggers sweep on bar 3 (100 -> 58 = -420); concurrent BTC scale-up (+20)
    #    and new ETH entry (+10 at 100 -> 95) on bar 3 must NOT be admitted & liquidated.
    t_open = np.full((2, n), 100.0)
    m_close = np.full((2, n), 100.0)
    targets = np.zeros((2, n))
    targets[0, 1:3] = 10.0
    targets[0, 3:] = 20.0
    targets[1, 3:] = 10.0
    m_close[0, 3:] = 58.0
    t_open[0, 4:] = 58.0
    m_close[1, 3:] = 95.0
    t_open[1, 4:] = 95.0

    res, _ = cr.simulate_guardian_challenge(["BTC", "ETH"], targets, t_open, m_close, grid, spec)
    assert res.sweeps == 1
    assert res.step_pnl[1] == pytest.approx(-1.0, abs=1e-9)
    assert res.step_pnl[3] == pytest.approx(-420.58, abs=1e-9)
    assert res.full_window_net_usd == pytest.approx(-421.58, abs=1e-9)

    # 2. Open-time breach on bar 3 (t_open drops to 58 immediately) also blocks concurrent ETH open.
    t_open_gap = np.full((2, n), 100.0)
    m_close_gap = np.full((2, n), 100.0)
    t_open_gap[0, 3:] = 58.0
    m_close_gap[0, 3:] = 58.0
    m_close_gap[1, 3:] = 95.0
    t_open_gap[1, 4:] = 95.0
    res_gap, _ = cr.simulate_guardian_challenge(
        ["BTC", "ETH"], targets, t_open_gap, m_close_gap, grid, spec
    )
    assert res_gap.sweeps == 1
    assert res_gap.step_pnl[3] == pytest.approx(-420.58, abs=1e-9)
    assert res_gap.full_window_net_usd == pytest.approx(-421.58, abs=1e-9)

    # 3. Exit/reduction executes first at bar 3 open (+10 -> +5) and keeps loss above sweep limit.
    targets_red = np.zeros((1, n))
    targets_red[0, 1:3] = 10.0
    targets_red[0, 3:] = 5.0
    t_open_red = np.full((1, n), 100.0)
    m_close_red = np.full((1, n), 100.0)
    m_close_red[0, 3:] = 58.0
    t_open_red[0, 4:] = 58.0
    res_red, _ = cr.simulate_guardian_challenge(
        ["BTC"], targets_red, t_open_red, m_close_red, grid, spec
    )
    assert res_red.sweeps == 0
    assert res_red.step_pnl[3] == pytest.approx(-210.50, abs=1e-9)


def test_guardian_missing_trade_open_forced_reset_applies_gap_haircut() -> None:
    n = 8
    grid = _grid_ms(n)
    # Two coins open +10 at 100 on bar 1.
    # Bar 3: BTC crashes at open (t_open=50, m_close=50 -> -500 triggers open sweep),
    # while ETH has missing trade_open (NaN) and optimistic m_close=120.
    # Forced reset on ETH must NOT silently fill at T+4 close (120) with +200 gain;
    # it must apply conservative gap_exit_haircut_bps (50 bps) + DEFAULT_FEE_BPS (4.5 bps).
    t_open = np.full((2, n), 100.0)
    m_close = np.full((2, n), 100.0)
    targets = np.zeros((2, n))
    targets[:, 1:] = 10.0

    t_open[0, 3:] = 50.0
    m_close[0, 3:] = 50.0
    t_open[1, 3] = np.nan
    m_close[1, 3:] = 120.0
    t_open[1, 4:] = 120.0

    spec = cr.GuardianSpec(
        profit_target=50_000.0,
        max_static_dd=50_000.0,
        daily_loss_limit=50_000.0,
        guardian_sweep_loss=400.0,
        cost_bps=10.0,
    )
    res, _ = cr.simulate_guardian_challenge(["BTC", "ETH"], targets, t_open, m_close, grid, spec)

    assert res.sweeps == 1
    assert res.forced_gap_exits == 1
    assert res.unpriced_open_steps == 1
    # Bar 1 entry: -1.00 - 1.00 = -2.00
    # Bar 3 BTC flatten at open 50: -500.0 - 10 * 50 * 0.001 = -500.50
    # Bar 3 ETH forced gap flatten (ref_px=min(100,120)=100 -> exit_px=99.5):
    #   10*(99.5 - 100) - 10*99.5*0.00045 = -5.44775
    expected_bar3 = -500.50 - 5.44775
    assert res.step_pnl[3] == pytest.approx(expected_bar3, abs=1e-9)
    assert res.full_window_net_usd == pytest.approx(round(-2.0 + expected_bar3, 2), abs=1e-9)

    # Short-side check: ETH is short -10 with optimistic m_close=80 while t_open=NaN -> still penalized.
    t_short = np.full((2, n), 100.0)
    m_short = np.full((2, n), 100.0)
    tgt_short = np.zeros((2, n))
    tgt_short[0, 1:] = 10.0
    tgt_short[1, 1:] = -10.0
    t_short[0, 3:] = 50.0
    m_short[0, 3:] = 50.0
    t_short[1, 3] = np.nan
    m_short[1, 3:] = 80.0
    t_short[1, 4:] = 80.0
    res_s, _ = cr.simulate_guardian_challenge(["BTC", "ETH"], tgt_short, t_short, m_short, grid, spec)
    assert res_s.forced_gap_exits == 1
    assert res_s.unpriced_open_steps == 1
    # ETH short (ref_px=max(100,80)=100 -> exit_px=100.5): -10*(100.5 - 100) - 10*100.5*0.00045 = -5.45225
    assert res_s.step_pnl[3] == pytest.approx(-500.50 - 5.45225, abs=1e-9)


def test_guardian_lot_step_quantization_and_30usd_deadband_parity() -> None:
    n = 8
    grid = _grid_ms(n)
    t_open = np.full((1, n), 100.0)
    m_close = np.full((1, n), 100.0)
    # Bar 1: 0.15 -> step 0.1 < min_amount 0.2 -> skipped
    # Bar 2: 0.38 -> step 0.3 >= min_amount 0.2 -> fills 0.3
    # Bar 3: 0.45 -> delta 0.15 -> step 0.1 < min_amount 0.2 -> skipped (stays 0.3)
    # Bar 4: price rises 100 -> 110 -> +3.00 PnL (vs +4.50 unquantized)
    # Bar 5: full close to 0.0 -> closes cleanly
    targets = np.array([[0.0, 0.15, 0.38, 0.45, 0.45, 0.0, 0.0, 0.0]], dtype=np.float64)
    m_close[0, 4:] = 110.0
    t_open[0, 5:] = 110.0

    spec_q = cr.GuardianSpec(
        profit_target=50_000.0,
        max_static_dd=50_000.0,
        daily_loss_limit=50_000.0,
        guardian_sweep_loss=50_000.0,
        cost_bps=0.0,
        amount_step=0.1,
        min_amount=0.2,
    )
    res_q, _ = cr.simulate_guardian_challenge(["BTC"], targets, t_open, m_close, grid, spec_q)
    assert res_q.step_pnl[1] == 0.0
    assert res_q.step_pnl[4] == pytest.approx(3.0, abs=1e-9)
    assert res_q.full_window_net_usd == pytest.approx(3.0, abs=1e-9)

    # Per-coin quantization + $30 notional deadband parity
    t_pair = np.vstack([np.full(n, 1000.0), np.full(n, 100.0)])
    m_pair = np.vstack([np.full(n, 1000.0), np.full(n, 100.0)])
    pair_targets = np.zeros((2, n))
    # Bar 1: BTC 0.029 -> step 0.02 ($20 < $30 deadband -> skipped); ETH 0.25 -> step 0.2 ($20 < $30 -> skipped)
    pair_targets[0, 1] = 0.029
    pair_targets[1, 1] = 0.25
    # Bar 2: BTC 0.039 -> step 0.03 ($30 >= $30 -> fills 0.03); ETH 0.35 -> step 0.3 ($30 >= $30 -> fills 0.3)
    pair_targets[0, 2:5] = 0.039
    pair_targets[1, 2:5] = 0.35
    # Bar 3: ETH drops to 50 (0.3 * 50 = $15 < $30), BTC stays at 1000 -> PnL = -15.0
    m_pair[1, 3:] = 50.0
    t_pair[1, 4:] = 50.0
    # Bar 5: full close to 0.0 -> bypasses $30 deadband even though ETH is only $15!
    # Bar 6: ETH drops to 10 -> must NOT suffer further loss because bar 5 closed it.
    m_pair[1, 6:] = 10.0

    spec_pair = cr.GuardianSpec(
        profit_target=50_000.0,
        max_static_dd=50_000.0,
        daily_loss_limit=50_000.0,
        guardian_sweep_loss=50_000.0,
        cost_bps=0.0,
        min_trade_notional_usd=30.0,
        amount_step_by_coin={"BTC": 0.01, "ETH": 0.1},
        min_amount_by_coin={"BTC": 0.01, "ETH": 0.1},
    )
    res_pair, _ = cr.simulate_guardian_challenge(
        ["BTC", "ETH"], pair_targets, t_pair, m_pair, grid, spec_pair
    )
    assert res_pair.step_pnl[1] == 0.0
    assert res_pair.step_pnl[3] == pytest.approx(-15.0, abs=1e-9)
    assert res_pair.step_pnl[6] == 0.0
    assert res_pair.full_window_net_usd == pytest.approx(-15.0, abs=1e-9)

    # Verify halt flatten bypasses $30 deadband even when held position is worth < $30.
    halt_targets = np.zeros((2, n))
    halt_targets[0, 1:] = 0.5  # $500 BTC
    halt_targets[1, 1:] = 0.35  # 0.3 ETH ($30)
    t_halt = np.vstack([np.full(n, 1000.0), np.full(n, 100.0)])
    m_halt = np.vstack([np.full(n, 1000.0), np.full(n, 100.0)])
    # Bar 2: ETH drops to 50 ($15 < $30 deadband), BTC drops to 100 (-$450 triggers sweep).
    m_halt[0, 2:] = 100.0
    t_halt[0, 3:] = 100.0
    m_halt[1, 2:] = 50.0
    t_halt[1, 3:] = 50.0
    m_halt[1, 4:] = 10.0  # If ETH wasn't closed by halt flatten, bar 4 would lose more!
    spec_halt = cr.GuardianSpec(
        profit_target=50_000.0,
        max_static_dd=50_000.0,
        daily_loss_limit=50_000.0,
        guardian_sweep_loss=400.0,
        cost_bps=0.0,
        min_trade_notional_usd=30.0,
        amount_step_by_coin={"BTC": 0.01, "ETH": 0.1},
        min_amount_by_coin={"BTC": 0.01, "ETH": 0.1},
    )
    res_halt, _ = cr.simulate_guardian_challenge(
        ["BTC", "ETH"], halt_targets, t_halt, m_halt, grid, spec_halt
    )
    assert res_halt.sweeps == 1
    assert res_halt.step_pnl[4] == 0.0
    assert res_halt.full_window_net_usd == pytest.approx(-465.0, abs=1e-9)


def test_guardian_tracks_peak_gross_leverage_and_enforces_optional_cap() -> None:
    n = 10
    grid = _grid_ms(n)
    t_open = np.full((2, n), 100.0)
    m_close = np.full((2, n), 100.0)
    targets = np.zeros((2, n))
    targets[0, 1:4] = 150.0  # $15,000
    targets[1, 1:4] = 100.0  # $10,000 -> total $25,000 = 2.5x on $10k eq
    targets[0, 4:7] = 50.0  # partial reduce to $5,000
    targets[1, 4:7] = 0.0  # full close ETH

    spec_uncapped = cr.GuardianSpec(
        profit_target=50_000.0,
        max_static_dd=50_000.0,
        daily_loss_limit=50_000.0,
        guardian_sweep_loss=50_000.0,
        cost_bps=0.0,
    )
    res_un, _ = cr.simulate_guardian_challenge(
        ["BTC", "ETH"], targets, t_open, m_close, grid, spec_uncapped
    )
    assert res_un.peak_gross_leverage == pytest.approx(2.5, abs=1e-9)
    assert res_un.full_window_peak_gross_leverage == pytest.approx(2.5, abs=1e-9)

    # With max_gross_leverage=1.5 and non-zero costs + lot steps, entry scales down to <= 1.5x.
    spec_capped = cr.GuardianSpec(
        profit_target=50_000.0,
        max_static_dd=50_000.0,
        daily_loss_limit=50_000.0,
        guardian_sweep_loss=50_000.0,
        cost_bps=10.0,
        amount_step=1.0,
        max_gross_leverage=1.5,
    )
    res_cap, _ = cr.simulate_guardian_challenge(
        ["BTC", "ETH"], targets, t_open, m_close, grid, spec_capped
    )
    assert 1.4 <= res_cap.peak_gross_leverage <= 1.5
    assert 1.4 <= res_cap.full_window_peak_gross_leverage <= 1.5

    # Terminal PASS on bar 2 at 1.0x followed by 2.5x on bar 5 separates terminal vs full-window leverage.
    t_split = np.full((2, n), 100.0)
    m_split = np.full((2, n), 100.0)
    tgt_split = np.zeros((2, n))
    tgt_split[0, 1:] = 100.0  # 1.0x ($10k)
    m_split[0, 2:] = 110.0  # +1000 PASS on bar 2 (eq = 11,000, lev = 11,000 / 11,000 = 1.0)
    t_split[0, 3:] = 110.0
    tgt_split[1, 5:] = 220.0  # 2.0x additional on ETH after PASS
    spec_pass = cr.GuardianSpec(
        profit_target=800.0,
        max_static_dd=50_000.0,
        daily_loss_limit=50_000.0,
        guardian_sweep_loss=50_000.0,
        cost_bps=0.0,
    )
    res_pass, _ = cr.simulate_guardian_challenge(
        ["BTC", "ETH"], tgt_split, t_split, m_split, grid, spec_pass
    )
    assert res_pass.verdict == "PASS"
    assert res_pass.terminal_step == 2
    assert res_pass.peak_gross_leverage == pytest.approx(1.0, abs=1e-9)
    assert res_pass.full_window_peak_gross_leverage == pytest.approx(3.0, abs=1e-9)


def test_guardian_cost_multipliers_and_perpetual_funding_modeling() -> None:
    n = 12
    grid = _grid_ms(n)
    t_open = np.full((1, n), 100.0)
    m_close = np.full((1, n), 100.0)
    targets = np.zeros((1, n))
    targets[0, 1:11] = 100.0  # $10,000 long for 10 bars (bars 1..10), flat at bar 11

    # 1. Cost multipliers 1x / 2x / 3x scale round-trip transaction costs linearly.
    # Round-trip notional = $20,000 at 10 bps = $20 at 1x, $40 at 2x, $60 at 3x.
    nets = []
    for mult in (1.0, 2.0, 3.0):
        spec_c = cr.GuardianSpec(
            profit_target=50_000.0,
            max_static_dd=50_000.0,
            daily_loss_limit=50_000.0,
            guardian_sweep_loss=50_000.0,
            cost_bps=10.0,
            cost_multiplier=mult,
        )
        res_c, _ = cr.simulate_guardian_challenge(["BTC"], targets, t_open, m_close, grid, spec_c)
        nets.append(res_c.full_window_net_usd)
    assert nets == pytest.approx([-20.0, -40.0, -60.0], abs=1e-9)

    # 2. Positive funding_bps_per_8h: longs pay, shorts receive (9.6 bps/8h = 0.1 bps/5m bar = $0.10/bar on $10k).
    spec_f1 = cr.GuardianSpec(
        profit_target=50_000.0,
        max_static_dd=50_000.0,
        daily_loss_limit=50_000.0,
        guardian_sweep_loss=50_000.0,
        cost_bps=0.0,
        funding_bps_per_8h=9.6,
        funding_multiplier=1.0,
    )
    res_long, _ = cr.simulate_guardian_challenge(["BTC"], targets, t_open, m_close, grid, spec_f1)
    assert res_long.total_funding_usd == pytest.approx(-1.0, abs=1e-9)
    assert res_long.full_window_net_usd == pytest.approx(-1.0, abs=1e-9)

    res_short, _ = cr.simulate_guardian_challenge(["BTC"], -targets, t_open, m_close, grid, spec_f1)
    assert res_short.total_funding_usd == pytest.approx(1.0, abs=1e-9)
    assert res_short.full_window_net_usd == pytest.approx(1.0, abs=1e-9)

    spec_f2 = cr.GuardianSpec(
        profit_target=50_000.0,
        max_static_dd=50_000.0,
        daily_loss_limit=50_000.0,
        guardian_sweep_loss=50_000.0,
        cost_bps=0.0,
        funding_bps_per_8h=9.6,
        funding_multiplier=2.0,
    )
    res_f2, _ = cr.simulate_guardian_challenge(["BTC"], targets, t_open, m_close, grid, spec_f2)
    assert res_f2.total_funding_usd == pytest.approx(-2.0, abs=1e-9)

    # 3. Per-bar funding_rates array supports coin-specific rates and terminal vs full-window separation.
    t2 = np.full((2, n), 100.0)
    m2 = np.full((2, n), 100.0)
    m2[0, 2:] = 110.0  # BTC rises +10 on 100 units = +1000 PASS at bar 2
    t2[0, 3:] = 110.0
    tgt2 = np.zeros((2, n))
    tgt2[0, 1:6] = 100.0
    tgt2[1, 1:6] = -50.0  # ETH short $5k
    rates = np.zeros((2, n))
    rates[0, :] = 0.0001  # 1 bps per bar paid by BTC long
    rates[1, :] = 0.0002  # 2 bps per bar received by ETH short

    spec_p = cr.GuardianSpec(
        profit_target=800.0,
        max_static_dd=50_000.0,
        daily_loss_limit=50_000.0,
        guardian_sweep_loss=50_000.0,
        cost_bps=0.0,
    )
    res_p, _ = cr.simulate_guardian_challenge(
        ["BTC", "ETH"], tgt2, t2, m2, grid, spec_p, funding_rates=rates
    )
    assert res_p.verdict == "PASS"
    assert res_p.terminal_step == 2
    # Bars 1..2: BTC pays 100*100*1e-4 + 100*110*1e-4 = 1.0 + 1.1 = 2.1; ETH receives 2*(50*100*2e-4) = 2.0 -> -0.10
    assert res_p.total_funding_usd == pytest.approx(-0.1, abs=1e-9)
    # Bars 1..5: BTC pays 1.0 + 4*1.1 = 5.4; ETH receives 5*1.0 = 5.0 -> -0.40
    assert res_p.full_window_funding_usd == pytest.approx(-0.4, abs=1e-9)

    with pytest.raises(ValueError, match="INVALID_FUNDING_RATES_SHAPE"):
        cr.simulate_guardian_challenge(
            ["BTC", "ETH"], tgt2, t2, m2, grid, spec_p, funding_rates=np.zeros((3, n))
        )


def test_guardian_executed_deltas_reflect_sweep_halt_deadband_and_leverage_cap() -> None:
    n = 295  # crosses UTC 00:05 daily reset at bar 289
    grid = _grid_ms(n)
    t_open = np.full((1, n), 100.0)
    m_close = np.full((1, n), 100.0)
    targets = np.zeros((1, n))

    # Bar 1: open 10.0 ($1,000); bar 2: tiny +0.10 ($10 < $25 deadband) -> skipped!
    targets[0, 1] = 10.0
    targets[0, 2] = 10.1
    # Bar 3: price drops $50 on 10.0 -> -$500 <= -$400 sweep -> flatten -10.0 & halt until reset
    m_close[0, 3:] = 50.0
    t_open[0, 4:] = 50.0
    targets[0, 3:292] = 10.0
    # Bar 292 (after reset): flat -> re-enter new position 40.0 ($2,000 on ~$500 eq -> capped by 1.5x leverage)
    targets[0, 292] = 0.0
    targets[0, 293] = 40.0

    spec = cr.GuardianSpec(
        initial_equity=1_000.0,
        profit_target=50_000.0,
        max_static_dd=50_000.0,
        daily_loss_limit=50_000.0,
        guardian_sweep_loss=400.0,
        cost_bps=0.0,
        min_trade_notional_usd=25.0,
        max_gross_leverage=1.5,
        reenter_after_sweep="new_only",
    )
    res, _ = cr.simulate_guardian_challenge(["BTC"], targets, t_open, m_close, grid, spec)

    assert res.executed_deltas is not None
    assert res.executed_deltas.shape == (1, n)
    assert res.executed_deltas[0, 1] == pytest.approx(10.0, abs=1e-9)
    assert res.executed_deltas[0, 2] == pytest.approx(0.0, abs=1e-9)  # deadband skipped
    assert res.executed_deltas[0, 3] == pytest.approx(-10.0, abs=1e-9)  # sweep flattened
    assert np.allclose(res.executed_deltas[0, 4:293], 0.0, atol=1e-12)  # blocked + new_only stale
    # Re-entry at bar 293 capped at <= 1.5x * $500 / $50 = 15.0 (strictly < raw 40.0)
    assert 0.0 < res.executed_deltas[0, 293] <= 15.0 + 1e-9
    assert res.executed_deltas[0, 293] < 40.0


def test_guardian_spec_validates_daily_loss_basis_and_rates() -> None:
    with pytest.raises(ValueError, match="INVALID_DAILY_LOSS_BASIS"):
        cr.GuardianSpec(daily_loss_basis="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="INVALID_DAILY_LOSS_RATE"):
        cr.GuardianSpec(daily_loss_rate=0.0)
    with pytest.raises(ValueError, match="INVALID_DAILY_LOSS_RATE"):
        cr.GuardianSpec(daily_loss_rate=1.0)
    with pytest.raises(ValueError, match="INVALID_GUARDIAN_SWEEP_RATE"):
        cr.GuardianSpec(guardian_sweep_rate=-0.1)


def test_guardian_balance_reset_vs_equity_reset_overnight_counterexample() -> None:
    # Q11 overnight counterexample (zero costs):
    # Day 1: Position 1 loses $100 and closes before 00:05 UTC reset -> balance = $9,900.
    #        Position 2 opens +10 at $100 and marks to $80 before Day 2 00:05 reset -> equity = $9,700.
    # Day 2: Position 2 drops to $49.90 (equity = $9,399; -$301 from reset equity, -$501 from $9,900 balance).
    #        Also test $50.40 (equity = $9,404 -> breaches 5% of $9,900 = $9,405, not fixed $500 = $9,400).
    n = 310  # bar 289 is Day 2 00:05 UTC reset
    grid = _grid_ms(n)
    t_open = np.full((2, n), 100.0)
    m_close = np.full((2, n), 100.0)
    targets = np.zeros((2, n))

    # Coin 0 (Position 1): long 10 bars 10..19, drops to 90 at bar 15 and closes at bar 20 -> -$100 realized
    targets[0, 10:20] = 10.0
    m_close[0, 15:] = 90.0
    t_open[0, 16:] = 90.0

    # Coin 1 (Position 2): long 10 from bar 100 onward, marks to 80 before Day 2 reset (bar 288)
    targets[1, 100:] = 10.0
    m_close[1, 200:295] = 80.0
    t_open[1, 201:296] = 80.0
    # Drop after reset at bar 295 -> equity falls to $9,399 (10 * 49.90 + 9,900 - 1,000 = 9,399)
    m_close[1, 295:] = 49.90
    t_open[1, 296:] = 49.90

    # 1. Fixed $500 balance-reset (with high sweep cap so FAIL_DAILY is isolated cleanly)
    spec_bal_fixed = cr.GuardianSpec(
        cost_bps=0.0,
        daily_loss_limit=500.0,
        guardian_sweep_loss=10_000.0,
        daily_loss_basis="balance",
    )
    res_bal, _ = cr.simulate_guardian_challenge(["C0", "C1"], targets, t_open, m_close, grid, spec_bal_fixed)
    assert res_bal.verdict == "FAIL_DAILY"
    assert res_bal.terminal_step == 295
    assert res_bal.worst_daily_usd == pytest.approx(-501.0, abs=1e-9)

    # 2. Default Guardian sweep ($400) + balance-reset catches sweep & breach accurately
    spec_bal_default = cr.GuardianSpec(cost_bps=0.0)
    res_bal_sw, _ = cr.simulate_guardian_challenge(
        ["C0", "C1"], targets, t_open, m_close, grid, spec_bal_default
    )
    assert res_bal_sw.verdict == "FAIL_DAILY"
    assert res_bal_sw.sweeps == 1

    # 3. Equity-reset fails to detect the -$501 cumulative daily loss from $9,900 balance
    spec_eq_legacy = cr.GuardianSpec(
        cost_bps=0.0,
        daily_loss_limit=500.0,
        guardian_sweep_loss=400.0,
        daily_loss_basis="equity",
    )
    res_eq, _ = cr.simulate_guardian_challenge(
        ["C0", "C1"], targets, t_open, m_close, grid, spec_eq_legacy
    )
    assert res_eq.verdict == "TIMEOUT"
    assert res_eq.sweeps == 0
    assert res_eq.worst_daily_usd == pytest.approx(-301.0, abs=1e-9)

    # 4. 5% reset-balance rate limit ($9,900 * 0.95 = $9,405): equity = $9,404 breaches rate limit
    m_rate = m_close.copy()
    t_rate = t_open.copy()
    m_rate[1, 295:] = 50.40  # equity = $9,404 (between $9,400 fixed and $9,405 5% floor)
    t_rate[1, 296:] = 50.40
    res_rate_hit, _ = cr.simulate_guardian_challenge(
        ["C0", "C1"],
        targets,
        t_rate,
        m_rate,
        grid,
        cr.GuardianSpec(
            cost_bps=0.0,
            daily_loss_rate=0.05,
            guardian_sweep_loss=10_000.0,
            daily_loss_basis="balance",
        ),
    )
    assert res_rate_hit.verdict == "FAIL_DAILY"
    assert res_rate_hit.terminal_step == 295

    res_fixed_miss, _ = cr.simulate_guardian_challenge(
        ["C0", "C1"],
        targets,
        t_rate,
        m_rate,
        grid,
        spec_bal_fixed,
    )
    assert res_fixed_miss.verdict == "TIMEOUT"

    # 5. Instant reset-boundary breach: realized +$300 (balance=$10,300) while holding -$550 loser (eq=$9,750)
    #    Before Day 2 reset, daily loss vs $10,000 is -$250 (< $500). At 00:05 reset, cashday=$10,300 -> -$550 breach!
    m_inst = np.full((2, n), 100.0)
    t_inst = np.full((2, n), 100.0)
    tgt_inst = np.zeros((2, n))
    tgt_inst[0, 10:20] = 10.0
    m_inst[0, 15:] = 130.0
    t_inst[0, 16:] = 130.0  # +$300 realized -> balance = $10,300
    tgt_inst[1, 100:] = 10.0
    m_inst[1, 200:] = 45.0
    t_inst[1, 201:] = 45.0  # -$550 open uPnL -> eq = $9,750 (-$250 vs Day 1 $10k)
    res_inst, _ = cr.simulate_guardian_challenge(
        ["C0", "C1"], tgt_inst, t_inst, m_inst, grid, spec_bal_fixed
    )
    assert res_inst.verdict == "FAIL_DAILY"
    assert res_inst.terminal_step == 289
    assert res_inst.worst_daily_usd == pytest.approx(-550.0, abs=1e-9)

