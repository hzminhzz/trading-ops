"""Phase 1 regression tests: PIT universe, episode ROPN, tilt, netted balance-reset & Tiers 1-4."""

from __future__ import annotations

import importlib
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDIES_DIR = ROOT / "studies"
if str(STUDIES_DIR) not in sys.path:
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


def _create_synthetic_registry(db_path: Path, *, schema_version: int = 8) -> dict[str, str]:
    addrs = {
        "W1": "0x1111000000000000000000000000000000000001",
        "W2": "0x1111000000000000000000000000000000000002",
        "W3": "0x1111000000000000000000000000000000000003",
        "W4": "0x1111000000000000000000000000000000000004",
    }
    con = sqlite3.connect(db_path)
    con.executescript(
        f"""
        PRAGMA foreign_keys = ON;
        CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
        INSERT INTO schema_migrations VALUES ({schema_version}, '2026-11-04T00:00:00+00:00');
        CREATE TABLE wallets(
            wallet_id INTEGER PRIMARY KEY,
            venue TEXT NOT NULL,
            network TEXT NOT NULL,
            address TEXT NOT NULL UNIQUE
        );
        CREATE TABLE cohorts(cohort_id TEXT PRIMARY KEY, version INTEGER NOT NULL);
        CREATE TABLE cohort_memberships(
            cohort_id TEXT NOT NULL,
            wallet_id INTEGER NOT NULL REFERENCES wallets(wallet_id),
            known_at TEXT NOT NULL
        );
        CREATE TABLE wallet_aliases(
            wallet_id INTEGER NOT NULL REFERENCES wallets(wallet_id),
            alias TEXT NOT NULL,
            known_at TEXT NOT NULL
        );
        CREATE TABLE exclusions(
            wallet_id INTEGER NOT NULL REFERENCES wallets(wallet_id),
            reason TEXT NOT NULL,
            known_at TEXT NOT NULL
        );
        CREATE TABLE monitoring(
            wallet_id INTEGER PRIMARY KEY REFERENCES wallets(wallet_id),
            configured INTEGER NOT NULL,
            known_at TEXT NOT NULL
        );
        CREATE TABLE portfolio_versions(portfolio_id TEXT PRIMARY KEY);
        CREATE TABLE portfolio_memberships(portfolio_id TEXT, wallet_id INTEGER);
        CREATE TABLE portfolio_seals(portfolio_id TEXT PRIMARY KEY);
        CREATE TABLE research_selections(selection_id TEXT PRIMARY KEY);
        CREATE TABLE evidence_artifacts(artifact_id TEXT PRIMARY KEY);
        CREATE TABLE snapshot_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);

        INSERT INTO cohorts VALUES ('monitored_plus_six', 1);
        INSERT INTO wallets VALUES
            (1, 'hyperliquid', 'mainnet', '{addrs["W1"]}'),
            (2, 'hyperliquid', 'mainnet', '{addrs["W2"]}'),
            (3, 'hyperliquid', 'mainnet', '{addrs["W3"]}'),
            (4, 'hyperliquid', 'mainnet', '{addrs["W4"]}');
        INSERT INTO cohort_memberships VALUES
            ('monitored_plus_six', 1, '2026-11-04T02:49:59+00:00'),
            ('monitored_plus_six', 2, '2026-11-04T09:00:00Z'),
            ('monitored_plus_six', 3, '2026-11-05T02:00:00+00:00');
        INSERT INTO wallet_aliases VALUES
            (1, 'W1', '2026-11-04T02:49:59Z'),
            (2, 'W2', '2026-11-04T09:00:00+00:00'),
            (3, 'W3', '2026-11-05T02:00:00Z'),
            (4, 'W4', '2026-11-05T02:00:00Z');
        INSERT INTO monitoring VALUES
            (1, 1, '2026-11-04T02:49:59+00:00'),
            (2, 1, '2026-11-04T09:00:00Z'),
            (3, 1, '2026-11-05T02:00:00Z'),
            (4, 0, '2026-11-05T02:00:00Z');
        INSERT INTO exclusions VALUES
            (1, 'UNDERWATER_BAGHOLDER', '2026-11-04T12:00:00Z'),
            (2, 'DECAY_SCAN_20261116', '2026-11-16T01:00:00+00:00');
        """
    )
    con.commit()
    con.close()
    return addrs


def test_pit_universe_masks_future_exclusions_and_fails_closed_pre_nov4(tmp_path: Path) -> None:
    bad_db = tmp_path / "bad_schema.sqlite"
    _create_synthetic_registry(bad_db, schema_version=7)
    with pytest.raises(ValueError, match="REGISTRY_SCHEMA_VERSION_MISMATCH"):
        cr.load_universe_asof(bad_db, "2026-11-10T00:00:00Z", mode="PIT_STRICT_ASOF")

    db_path = tmp_path / "registry.sqlite"
    addrs = _create_synthetic_registry(db_path, schema_version=8)

    with pytest.raises(ValueError, match="EMPTY_PIT_UNIVERSE_BEFORE_RECEIPT_EPOCH"):
        cr.load_universe_asof(db_path, "2026-10-01T00:00:00+00:00", mode="PIT_STRICT_ASOF")

    retro = cr.load_universe_asof(
        db_path,
        "2026-10-01T00:00:00Z",
        mode="RETROSPECTIVE_ASOF_MASKED",
        pre_asof_active_addresses=[addrs["W1"], addrs["W2"]],
    )
    assert [r.alias for r in retro.records] == ["W1", "W2"]
    assert retro.retrospective_evidence is not None
    assert retro.retrospective_evidence["kind"] == "RETROSPECTIVE_ESTIMATE"
    assert retro.retrospective_evidence["prospective_source_book"] is False
    assert retro.retrospective_evidence["trading_authorized"] is False

    pit_nov10 = cr.load_universe_asof(db_path, "2026-11-10T00:00:00+00:00", mode="PIT_STRICT_ASOF")
    assert [r.alias for r in pit_nov10.records] == ["W2", "W3"]
    assert pit_nov10.retrospective_evidence is None

    pit_nov17 = cr.load_universe_asof(db_path, "2026-11-17T00:00:00Z", mode="PIT_STRICT_ASOF")
    assert [r.alias for r in pit_nov17.records] == ["W3"]
    assert pit_nov17.registry_inventory_sha256 == pit_nov10.registry_inventory_sha256


def test_unanchored_full_policy_blocks_phantom_buy_on_trim_and_anchored_left_censoring() -> None:
    n = 20
    grid = _grid_ms(n)
    fills = _fills_df(
        [
            (int(grid[3]) + 1_000, 10.0, -1, 2.0),
            (int(grid[8]) + 1_000, 8.0, 1, 3.0),
            (int(grid[12]) + 1_000, 11.0, -1, 11.0),
        ]
    )
    recon = cr.reconstruct_positions_on_grid(fills, grid, policy="full")
    assert recon.unanchored_pre_inventory is True
    assert recon.pos[:9].tolist() == [0.0] * 9
    assert recon.pos[9:13].tolist() == [11.0] * 4
    assert recon.pos[13:].tolist() == [0.0] * 7

    px = np.full(n, 100.0)
    frames, _ = cr.build_unit_replay_by_alias(
        {"ANCH": {"BTC": fills}},
        {"BTC": (px, px)},
        ["ANCH"],
        grid,
        policy="full",
        initial_anchors={"ANCH": {"BTC": 10.0}},
    )
    df = frames["ANCH"]["BTC"]
    assert df["pos"][0] == pytest.approx(10.0)
    assert df["delta"][0] == pytest.approx(0.0)
    assert df["cost"][0] == pytest.approx(0.0)

    tensor = pc.build_tensor_from_unit_replays(frames, grid, pod_equity_usd=10_000.0)
    eps = pc.reconstruct_execution_episodes(tensor)[0]
    assert eps["ep_left_censored"].size >= 1
    assert bool(eps["ep_left_censored"][0]) is True
    assert bool(eps["ep_uncensored"][0]) is False

    t_open = np.full(n, 100.0)
    m_close = np.full(n, 100.0)
    targets = np.full(n, 10.0)
    targets[0] = 0.0
    m_close[5] = np.nan
    t_open[5] = np.nan
    m_close[6:] = 75.0
    t_open[6:] = 75.0
    pos_out, _, gross_out, _, stats_out = cr.replay_coin_targets_vectorized(targets, t_open, m_close)
    assert stats_out["forced_gap_exits"] == 1.0
    assert np.all(pos_out[6:] == 0.0)
    assert float(np.sum(gross_out[6:])) == 0.0

    res_g, _ = cr.simulate_guardian_challenge(
        ["BTC"],
        targets[None, :],
        t_open[None, :],
        m_close[None, :],
        grid,
        cr.GuardianSpec(guardian_sweep_loss=50_000.0),
    )
    assert res_g.forced_gap_exits == 1
    assert res_g.full_window_net_usd < 0.0


def test_sparse_grid_reset_and_intrabar_open_breach() -> None:
    day0 = datetime(2026, 5, 1, 0, 0, tzinfo=UTC)
    day1 = datetime(2026, 5, 2, 0, 0, tzinfo=UTC)
    times = [day0 + timedelta(minutes=30 * i) for i in range(48)] + [
        day1 + timedelta(minutes=30 * i) for i in range(1, 20)
    ]
    grid = np.array([int(t.timestamp() * 1000) for t in times], dtype=np.int64)
    n = grid.size
    t_open = np.full((2, n), 100.0)
    m_close = np.full((2, n), 100.0)
    targets = np.zeros((2, n))
    targets[0, 2:10] = 10.0
    m_close[0, 6:] = 130.0
    t_open[0, 7:] = 130.0  # Day 0 closed +$300 win -> balance=$10,300
    targets[1, 20:] = 10.0
    m_close[1, 35:] = 45.0
    t_open[1, 36:] = 45.0  # -$550 open uPnL -> eq=$9,750 (-$250 Day 0, -$550 on Day 1 reset!)

    res_sparse, _ = cr.simulate_guardian_challenge(
        ["C0", "C1"],
        targets,
        t_open,
        m_close,
        grid,
        cr.GuardianSpec(cost_bps=0.0, daily_loss_basis="balance"),
    )
    assert res_sparse.verdict == "FAIL_DAILY"
    assert res_sparse.terminal_step == 48

    n_u = 12
    grid_u = _grid_ms(n_u)
    t_u = np.full((1, n_u), 100.0)
    m_u = np.full((1, n_u), 100.0)
    tgt_u = np.zeros((1, n_u))
    tgt_u[0, 1:] = 10.0
    t_u[0, 3] = 44.0  # eq_open = $9,440 -> -$560 daily breach at open!
    m_u[0, 3] = 98.0  # closes at -$20
    res_intra, _ = cr.simulate_guardian_challenge(
        ["BTC"],
        tgt_u,
        t_u,
        m_u,
        grid_u,
        cr.GuardianSpec(cost_bps=0.0, daily_loss_limit=500.0, guardian_sweep_loss=50_000.0),
    )
    assert res_intra.verdict == "FAIL_DAILY"
    assert res_intra.terminal_step == 3
    assert res_intra.worst_daily_usd <= -560.0


def test_mc_daily_blocks_and_stage2_sizer_respect_balance_reset_and_opposing_seat_crossing() -> None:
    n_days = 2
    n = n_days * 288
    grid = _grid_ms(n)
    m_close = np.full((2, n), 100.0)
    t_open = np.full((2, n), 100.0)
    targets = np.zeros((2, n))
    targets[0, 5:15] = 10.0
    m_close[0, 10:] = 90.0
    t_open[0, 11:] = 90.0  # -$100 realized -> balance=$9,900
    targets[1, 100:] = 10.0
    m_close[1, 200:295] = 80.0
    t_open[1, 201:296] = 80.0  # -$200 carry at 00:05 UTC (eq=$9,700)
    m_close[1, 295:] = 49.9
    t_open[1, 296:] = 49.9  # -$501 vs $9,900 balance

    spec_bal = cr.GuardianSpec(cost_bps=0.0, guardian_sweep_loss=10_000.0, daily_loss_basis="balance")
    res_q11, blocks_q11 = cr.simulate_guardian_challenge(["C0", "C1"], targets, t_open, m_close, grid, spec_bal)
    assert res_q11.verdict == "FAIL_DAILY"
    assert len(blocks_q11) == 7
    assert float(blocks_q11[6][1]) == pytest.approx(-501.0, abs=1e-9)

    mc_q11 = pc.run_stationary_block_mc_on_daily_blocks(
        blocks_q11,
        "Blended",
        n_paths=300,
        horizon_days=5,
        mean_block_days=2.0,
        rng=np.random.default_rng(7),
    )
    assert mc_q11["p_fail_daily"] > 0.0

    n3 = 3 * 288
    grid3 = _grid_ms(n3)
    px3 = np.full(n3, 100.0)
    px3[20:] = 95.0
    px3[289:] = 95.0  # Flat MTM on Day 2 (if opposing cross netted at 95, Day 2 loss = 0!)
    fills_a = _fills_df([(int(grid3[5]) + 1_000, 0.0, 1, 1.0)])
    fills_b = _fills_df([(int(grid3[25]) + 1_000, 0.0, -1, 1.0)])

    frames, _ = cr.build_unit_replay_by_alias(
        {"SEAT_A": {"BTC": fills_a}, "SEAT_B": {"BTC": fills_b}},
        {"BTC": (px3, px3)},
        ["SEAT_A", "SEAT_B"],
        grid3,
        policy="new",
        activation_ms=int(grid3[0]),
        cost_bps=0.0,
    )
    tensor3 = pc.build_tensor_from_unit_replays(frames, grid3, pod_equity_usd=10_000.0)
    cfg3 = pc.PodConstructorConfig(cost_bps=0.0, daily_loss_mode="reset_balance")
    slices3 = pc._daily_reset_slices(tensor3.timestamps_ms, cfg3.bars_per_day, cfg3.reset_bar_offset)

    for w_vec in (np.array([1.0, 0.0]), np.array([0.5, 0.5])):
        solo_n, t_ov, d_ov = pc._build_sparse_overlap_cache(
            tensor3.delta_notional_frac, tensor3.net_ret_frac
        )
        unrl = pc._compute_netted_unrealized_frac_at_resets(
            w_vec, tensor3, (0, 1), slices3, 0.0
        )
        _, _, _, _, _, mdaily_frac, _ = pc._evaluate_weights_unit_scale(
            w_vec, solo_n, t_ov, d_ov, 0.0, slices3, cfg3, unrl_at_resets=unrl
        )
        pos_a = cr.reconstruct_positions_on_grid(fills_a, grid3, policy="new", activation_ms=int(grid3[0])).pos
        pos_b = cr.reconstruct_positions_on_grid(fills_b, grid3, policy="new", activation_ms=int(grid3[0])).pos
        net_tgt = cr.net_member_targets(np.stack([pos_a[None, :], pos_b[None, :]]), w_vec)
        res_guard, _ = cr.simulate_guardian_challenge(
            ["BTC"],
            net_tgt,
            px3[None, :],
            px3[None, :],
            grid3,
            cr.GuardianSpec(cost_bps=0.0, guardian_sweep_loss=50_000.0, daily_loss_limit=50_000.0),
        )
        assert mdaily_frac * 10_000.0 == pytest.approx(-res_guard.worst_daily_usd, abs=1e-9)

    n_f = 2 * 288
    ts_f = _grid_ms(n_f)
    zero = np.zeros(n_f)
    sym_n = np.zeros((2, 2, n_f))
    pos_a_f = np.zeros(n_f)
    dn_a_f = np.zeros(n_f)
    pos_a_f[5:] = 0.5
    dn_a_f[5] = 0.5
    sym_n[0, 0, 20] = -0.05  # Seat A open uPnL -$500 at bar 20
    pos_b_f = np.zeros(n_f)
    dn_b_f = np.zeros(n_f)
    pos_b_f[25:] = -0.5
    dn_b_f[25] = -0.5  # Net cross at bar 25 closes BTC to 0, realizing -$500!
    ten_syn = pc.UnitCopy5mTensor(
        aliases=("A", "B"),
        symbols=("BTC", "ETH"),
        timestamps_ms=ts_f,
        notional_frac=np.array([[pos_a_f, zero], [pos_b_f, zero]], dtype=np.float64),
        delta_notional_frac=np.array([[dn_a_f, zero], [dn_b_f, zero]], dtype=np.float64),
        gross_ret_frac=np.sum(sym_n, axis=1),
        net_ret_frac=np.sum(sym_n, axis=1),
        sym_net_ret_frac=sym_n,
    )
    sl_f = pc._daily_reset_slices(ts_f, 288, 1)
    unrl_syn = pc._compute_netted_unrealized_frac_at_resets(
        np.array([0.5, 0.5]), ten_syn, (0, 1), sl_f, 0.0
    )
    assert unrl_syn[1] == pytest.approx(0.0, abs=1e-12)


def test_right_censored_revenge_bag_caught_by_causal_tilt_and_eb_ropn_shrinkage() -> None:
    n_days = 10
    n_bars = n_days * 288
    ts = _grid_ms(n_bars)
    n_t = 3
    notional = np.zeros((n_t, 1, n_bars), dtype=np.float64)
    d_notional = np.zeros((n_t, 1, n_bars), dtype=np.float64)
    sym_net = np.zeros((n_t, 1, n_bars), dtype=np.float64)

    for e in range(20):
        sb = 10 + e * 60
        eb = sb + 20
        notional[0, 0, sb:eb] = 0.25
        d_notional[0, 0, sb] = 0.25
        d_notional[0, 0, eb] = -0.25
        sym_net[0, 0, sb + 1 : eb] = 0.005 / 19.0
    notional[0, 0, 1500:1510] = 0.0005
    d_notional[0, 0, 1500] = 0.0005
    d_notional[0, 0, 1510] = -0.0005
    sym_net[0, 0, 1505] = 0.02

    for (sb, eb, sz, pnl) in [(10, 30, 0.25, 0.04), (50, 70, 0.25, -0.01)]:
        notional[1, 0, sb:eb] = sz
        d_notional[1, 0, sb] = sz
        d_notional[1, 0, eb] = -sz
        sym_net[1, 0, sb + 1] = pnl
    notional[1, 0, 150:] = 1.0
    d_notional[1, 0, 150] = 1.0
    sym_net[1, 0, 160] = -0.03

    for (sb, eb, pnl) in [(10, 40, 0.02), (80, 110, 0.03), (200, 230, 0.025)]:
        notional[2, 0, sb:eb] = 0.25
        d_notional[2, 0, sb] = 0.25
        d_notional[2, 0, eb] = -0.25
        sym_net[2, 0, sb + 1] = pnl

    net = np.sum(sym_net, axis=1)
    tensor = pc.UnitCopy5mTensor(
        aliases=("MANY_CLEAN", "FEW_REVENGE", "FEW_HIGH"),
        symbols=("BTC",),
        timestamps_ms=ts,
        notional_frac=notional,
        delta_notional_frac=d_notional,
        gross_ret_frac=net.copy(),
        net_ret_frac=net,
        sym_net_ret_frac=sym_net,
    )
    cfg = pc.PodConstructorConfig(min_episodes=2, min_active_bar_frac=0.01, max_tilt_multiplier=1.5)
    passed_ids, metrics, _baselines = pc.stage1_audit_traders(tensor, cfg)

    m_rev = metrics[1]
    assert m_rev.tilt_status == "FAIL"
    assert m_rev.tilt_multiplier == pytest.approx(4.0, abs=1e-6)
    assert not m_rev.passed_behavioral_gate
    assert "martingale_tilt" in m_rev.reject_reason
    assert 1 not in passed_ids

    assert metrics[0].raw_ropn_mean == pytest.approx(0.02, abs=1e-4)
    assert metrics[0].win_hold_p95_bars is not None
    assert metrics[2].win_hold_p95_bars is None
    assert abs(metrics[2].eb_ropn_mean - metrics[2].raw_ropn_mean) >= abs(
        metrics[0].eb_ropn_mean - metrics[0].raw_ropn_mean
    )

    assert pc.classify_holding_style_sleeve(np.array([50, 100, 150, 200]), active_frac=0.1) == "intraday_tactical"
    assert pc.classify_holding_style_sleeve(np.array([500, 700, 900, 1100]), active_frac=0.1) == "multi_day_swing"
    assert pc.classify_holding_style_sleeve(np.array([2500, 3000, 3500, 4000]), active_frac=0.1) == "positional"
    assert pc.classify_holding_style_sleeve(np.array([50, 500, 2500, 2500, 50, 500]), active_frac=0.1) == "mixed"
    assert pc.classify_holding_style_sleeve(np.array([50, 100]), active_frac=0.1) == "unlabeled"

    unqual_base = pc.TraderEpisodeBaseline(
        trader_id=9,
        alias="UNQUAL",
        mean_ep_return=0.0,
        std_ep_return=0.01,
        median_ln_peak_notional=0.0,
        mad_ln_peak_notional=0.25,
        mean_win_ln_duration=2.0,
        std_win_ln_duration=0.5,
        symbol_notional_dist=np.array([1.0]),
        mae_p95=0.02,
        qualified=False,
    )
    st = pc.RosterCUSUMState(alias="UNQUAL")
    pc.update_roster_episode_cusum(
        st,
        unqual_base,
        ep_return=0.05,
        ep_peak_notional_frac=0.1,
        ep_duration_bars=10.0,
        ep_mae=0.0,
        ep_symbol_notional=np.array([1.0]),
    )
    assert st.status == "KILL_CONTAIN_0X"
    assert "UNQUALIFIED_EPISODE_BASELINE" in st.reasons

    _, comps = pc.compute_crowding_distance_matrix([0, 2], tensor, cfg)
    assert bool(comps["pair_overlap_unknown"][0, 1]) is True
    assert int(comps["concurrent_active_days"][0, 1]) < 5


def test_size_pod_dual_barrier_enforces_joint_symbol_1_5x_and_gross_3_0x_including_drawdown() -> None:
    n_days = 10
    n_bars = n_days * 288
    ts = _grid_ms(n_bars)
    notional = np.zeros((3, 2, n_bars), dtype=np.float64)
    d_notional = np.zeros((3, 2, n_bars), dtype=np.float64)
    sym_net = np.zeros((3, 2, n_bars), dtype=np.float64)

    for ti in range(3):
        for b_start in range(10, n_bars - 100, 120):
            notional[ti, 0, b_start : b_start + 80] = 2.0
            d_notional[ti, 0, b_start] = 2.0
            d_notional[ti, 0, b_start + 80] = -2.0
            sym_net[ti, 0, b_start + 1 : b_start + 79] = 0.004 / 78.0
        notional[ti, 1, 50:90] = 1.5
        d_notional[ti, 1, 50] = 1.5
        d_notional[ti, 1, 90] = -1.5
        sym_net[ti, 1, 52] = -0.01  # Early unit drawdown so cum_r < 0 while exposed at bar 52..89
        sym_net[ti, 1, 60] = 0.015

    net = np.sum(sym_net, axis=1)
    tensor = pc.UnitCopy5mTensor(
        aliases=("S0", "S1", "S2"),
        symbols=("BTC", "ETH"),
        timestamps_ms=ts,
        notional_frac=notional,
        delta_notional_frac=d_notional,
        gross_ret_frac=net.copy(),
        net_ret_frac=net,
        sym_net_ret_frac=sym_net,
    )
    cfg_t3 = pc.PodConstructorConfig.for_sizing_tier(3, base=pc.PodConstructorConfig(pod_size=3, min_episodes=4))
    _, metrics, _ = pc.stage1_audit_traders(tensor, cfg_t3)
    sized = pc.size_pod_dual_barrier((0, 1, 2), tensor, metrics, cfg_t3)

    sym_path = np.tensordot(sized.multipliers, notional, axes=([0], [0]))  # (2, T)
    r_net = sized.multipliers @ net
    cum_r = np.cumsum(r_net)
    eq_mult = np.minimum(1.0, 1.0 + cum_r)
    assert np.all(np.max(np.abs(sym_path), axis=0) <= 1.5 * eq_mult + 1e-9)
    assert np.all(np.sum(np.abs(sym_path), axis=0) <= 3.0 * eq_mult + 1e-9)
    assert sized.peak_symbol_exposure_frac <= 1.5 + 1e-9
    assert sized.peak_gross_exposure_frac <= 3.0 + 1e-9

    cfg_t1 = pc.PodConstructorConfig.for_sizing_tier(1, base=pc.PodConstructorConfig(pod_size=3, min_episodes=4))
    sized_t1 = pc.size_pod_dual_barrier((0, 1, 2), tensor, metrics, cfg_t1)
    assert sized_t1.binding_constraint == "Tier1_Standalone_Cap"

    mod_notional = notional * 0.15  # 0.30 peak BTC notional
    mod_sym_net = sym_net * 0.5  # higher relative DD so standalone DD=0.005+
    mod_sym_net[:, 1, 52] = -0.04  # unit standalone DD = 0.04 -> Tier1 k = (0.02/3)/0.04 = 0.1667
    mod_net = np.sum(mod_sym_net, axis=1)
    ten_mod = pc.UnitCopy5mTensor(
        aliases=("S0", "S1", "S2"),
        symbols=("BTC", "ETH"),
        timestamps_ms=ts,
        notional_frac=mod_notional,
        delta_notional_frac=d_notional * 0.15,
        gross_ret_frac=mod_net.copy(),
        net_ret_frac=mod_net,
        sym_net_ret_frac=mod_sym_net,
    )
    _, metrics_mod, _ = pc.stage1_audit_traders(ten_mod, cfg_t1)
    s1_mod = pc.size_pod_dual_barrier((0, 1, 2), ten_mod, metrics_mod, cfg_t1)
    s3_mod = pc.size_pod_dual_barrier((0, 1, 2), ten_mod, metrics_mod, cfg_t3)
    assert s1_mod.binding_constraint == "Tier1_Standalone_Cap"
    assert float(np.sum(s1_mod.multipliers)) < float(np.sum(s3_mod.multipliers))


def _synthetic_wf_tensor(n_days: int = 24) -> pc.UnitCopy5mTensor:
    bpd = 288
    n_bars = n_days * bpd
    ts = _grid_ms(n_bars)
    rng = np.random.default_rng(123)
    n_t, n_s = 6, 3
    notional = np.zeros((n_t, n_s, n_bars), dtype=np.float32)
    d_notional = np.zeros((n_t, n_s, n_bars), dtype=np.float32)
    sym_net = np.zeros((n_t, n_s, n_bars), dtype=np.float64)

    for ti in range(n_t):
        si = ti % n_s
        for ep_idx in range(n_days * 2):
            sb = 10 + ep_idx * 140 + (ti * 7) % 15
            eb = sb + 30
            if eb + 2 >= n_bars:
                continue
            notional[ti, si, sb:eb] = 0.5
            d_notional[ti, si, sb] = 0.5
            d_notional[ti, si, eb] = -0.5
            if ep_idx % 4 == 1:
                sym_net[ti, si, sb + 1 : eb] = -0.001 / 29.0
            else:
                r = rng.normal(0.0002, 0.00005, size=eb - sb - 1)
                sym_net[ti, si, sb + 1 : eb] = r

    net = np.sum(sym_net, axis=1)
    return pc.UnitCopy5mTensor(
        aliases=tuple(f"T{i}" for i in range(n_t)),
        symbols=("BTC", "ETH", "SOL"),
        timestamps_ms=ts,
        notional_frac=notional,
        delta_notional_frac=d_notional,
        gross_ret_frac=net.copy(),
        net_ret_frac=net,
        sym_net_ret_frac=sym_net,
        chop_mask=np.arange(n_bars) >= (n_bars // 2),
    )


def test_full_pipeline_cpcv_and_rolling_walk_forward_zero_test_leakage() -> None:
    tensor = _synthetic_wf_tensor(n_days=24)
    cfg = pc.PodConstructorConfig(
        pod_size=3,
        top_per_cluster=2,
        max_candidate_pods=8,
        min_episodes=4,
        min_active_bar_frac=0.05,
        cpcv_n_blocks=4,
        cpcv_embargo_days=1,
    )

    pbo, logits, recs = pc.compute_full_pipeline_cpcv_pbo(tensor, cfg)
    assert 0.0 <= pbo <= 1.0
    assert logits.size >= 1
    assert any(r["status"] == "OK" for r in recs)

    folds_base = pc.run_rolling_walk_forward(
        tensor, train_days=12, eval_days=6, tiers=(1, 3), base_config=cfg
    )
    assert len(folds_base) == 4  # 2 rolling folds x 2 tiers

    base_18d = tensor.slice_time(0, 18 * 288)
    folds_18d_base = pc.run_rolling_walk_forward(
        base_18d, train_days=12, eval_days=6, tiers=(1, 3), base_config=cfg
    )
    assert len(folds_18d_base) == 2

    pert_net = base_18d.net_ret_frac.copy()
    pert_sym = base_18d.sym_net_ret_frac.copy()
    assert pert_sym is not None
    cut0 = 12 * 288
    pert_sym[:, :, cut0 + 5] -= 0.50
    pert_net[:, cut0 + 5] = np.sum(pert_sym[:, :, cut0 + 5], axis=1)
    tensor_pert = pc.UnitCopy5mTensor(
        aliases=base_18d.aliases,
        symbols=base_18d.symbols,
        timestamps_ms=base_18d.timestamps_ms,
        notional_frac=base_18d.notional_frac.copy(),
        delta_notional_frac=base_18d.delta_notional_frac.copy(),
        gross_ret_frac=pert_net.copy(),
        net_ret_frac=pert_net,
        sym_net_ret_frac=pert_sym,
        chop_mask=base_18d.chop_mask,
    )
    folds_pert = pc.run_rolling_walk_forward(
        tensor_pert, train_days=12, eval_days=6, tiers=(1, 3), base_config=cfg
    )
    for idx in (0, 1):
        assert folds_pert[idx].aliases == folds_18d_base[idx].aliases
        assert folds_pert[idx].pod_indices == folds_18d_base[idx].pod_indices
        assert np.array_equal(folds_pert[idx].multipliers, folds_18d_base[idx].multipliers)
        assert folds_pert[idx].train_net_ret_frac == folds_18d_base[idx].train_net_ret_frac
        assert folds_pert[idx].oos_status != folds_18d_base[idx].oos_status
        assert folds_pert[idx].oos_status in ("FAIL_STATIC", "FAIL_DAILY")
