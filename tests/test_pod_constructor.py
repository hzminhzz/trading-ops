"""Unit tests for studies/pod_constructor.py multi-manager pod abstractions."""

from __future__ import annotations

import importlib
import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies"))
pc = importlib.import_module("pod_constructor")


def _make_timestamps(n_days: int, bars_per_day: int = 288) -> np.ndarray:
    start_ms = int(datetime(2026, 7, 1, tzinfo=UTC).timestamp() * 1000)
    return start_ms + np.arange(n_days * bars_per_day, dtype=np.int64) * 300_000


def test_episode_reconstruction_includes_closing_bar_pnl_and_exit_cost() -> None:
    ts = _make_timestamps(1)[:20]
    notional = np.zeros((1, 1, 20), dtype=np.float64)
    d_notional = np.zeros((1, 1, 20), dtype=np.float64)
    gross = np.zeros((1, 20), dtype=np.float64)
    net = np.zeros((1, 20), dtype=np.float64)

    # Open on bar 2 (+1.0 notional), hold bars 2..4, close back to 0 on bar 5.
    notional[0, 0, 2:5] = 1.0
    d_notional[0, 0, 2] = 1.0
    d_notional[0, 0, 5] = -1.0
    net[0, 2] = -0.00065  # entry cost
    net[0, 3] = 0.01000
    net[0, 4] = 0.00500
    net[0, 5] = -0.00200 - 0.00065  # closing gap + exit cost occurring at close bar 5

    tensor = pc.UnitCopy5mTensor(
        aliases=("W1",),
        symbols=("BTC",),
        timestamps_ms=ts,
        notional_frac=notional,
        delta_notional_frac=d_notional,
        gross_ret_frac=gross,
        net_ret_frac=net,
    )
    eps = pc.reconstruct_execution_episodes(tensor)[0]
    assert eps["ep_pnl"].size == 1
    expected_ep_pnl = float(np.sum(net[0, 2:6]))
    assert eps["ep_pnl"][0] == pytest.approx(expected_ep_pnl, abs=1e-12)
    assert eps["ep_duration_bars"][0] == 3.0


def test_continuous_multi_coin_episodes_split_per_symbol_and_flips() -> None:
    ts = _make_timestamps(1)[:30]
    notional = np.zeros((1, 2, 30), dtype=np.float64)
    d_notional = np.zeros((1, 2, 30), dtype=np.float64)
    sym_net = np.zeros((1, 2, 30), dtype=np.float64)

    # Coin 0 held continuously across entire slice (1 long + 1 sign-flip short = 2 episodes)
    notional[0, 0, 1:15] = 0.5
    notional[0, 0, 15:29] = -0.5
    d_notional[0, 0, 1] = 0.5
    d_notional[0, 0, 15] = -1.0
    d_notional[0, 0, 29] = 0.5
    sym_net[0, 0, 2:29] = 0.001

    # Coin 1 trades 3 short bursts while Coin 0 never goes flat
    for b in (3, 10, 20):
        notional[0, 1, b : b + 3] = 0.3
        d_notional[0, 1, b] = 0.3
        d_notional[0, 1, b + 3] = -0.3
        sym_net[0, 1, b + 1 : b + 3] = 0.002

    net = np.sum(sym_net, axis=1)
    tensor = pc.UnitCopy5mTensor(
        aliases=("MULTI",),
        symbols=("BTC", "ETH"),
        timestamps_ms=ts,
        notional_frac=notional,
        delta_notional_frac=d_notional,
        gross_ret_frac=net.copy(),
        net_ret_frac=net,
        sym_net_ret_frac=sym_net,
    )
    eps = pc.reconstruct_execution_episodes(tensor)[0]
    assert eps["ep_pnl"].size == 5  # 2 BTC (long -> short flip) + 3 ETH episodes


def test_vectorized_episode_reconstruction_edge_cases_and_bounded_memory() -> None:
    import tracemalloc

    n_bars = 20
    ts = _make_timestamps(1)[:n_bars]
    notional = np.zeros((1, 2, n_bars), dtype=np.float64)
    d_notional = np.zeros((1, 2, n_bars), dtype=np.float64)
    sym_net = np.zeros((1, 2, n_bars), dtype=np.float64)

    # Coin 0 (BTC):
    # Ep1 (long): bars 0..2, close flat at bar 3 with exit cost/gap
    notional[0, 0, 0:3] = 1.0
    d_notional[0, 0, 0] = 1.0
    d_notional[0, 0, 3] = -1.0
    sym_net[0, 0, 0:4] = [-0.001, 0.004, -0.002, -0.002]  # cum: [-0.001, 0.003, 0.001, -0.001] -> sum=-0.001, mae=0.004

    # Ep2 (long): bars 4..5, direct flip to short at bar 6 (so bar 6 pnl belongs to Ep3!)
    notional[0, 0, 4:6] = 2.0
    d_notional[0, 0, 4] = 2.0
    sym_net[0, 0, 4:6] = [-0.003, 0.005]  # sum=0.002, mae=0.003

    # Ep3 (short): bars 6..8, close flat at bar 9 with closing pnl
    notional[0, 0, 6:9] = -1.5
    d_notional[0, 0, 6] = -3.5
    d_notional[0, 0, 9] = 1.5
    sym_net[0, 0, 6:10] = [0.001, -0.004, 0.002, 0.001]  # cum: [0.001, -0.003, -0.001, 0.0] -> sum=0.0, mae=0.004

    # Ep4 (1-bar long): bar 12 only, close at bar 13
    notional[0, 0, 12:13] = 0.5
    d_notional[0, 0, 12] = 0.5
    d_notional[0, 0, 13] = -0.5
    sym_net[0, 0, 12:14] = [0.002, -0.001]  # sum=0.001, mae=0.001

    # Ep5 (short): bars 17..19 open at end of slice
    notional[0, 0, 17:20] = -1.0
    d_notional[0, 0, 17] = -1.0
    sym_net[0, 0, 17:20] = [-0.002, -0.001, 0.004]  # sum=0.001, mae=0.003

    # Coin 1 (ETH): overlaps BTC across bars 1..6, closes at bar 7 + tiny sub-eps noise ignored
    notional[0, 1, 1:7] = 0.8
    d_notional[0, 1, 1] = 0.8
    d_notional[0, 1, 7] = -0.8
    sym_net[0, 1, 1:8] = [0.001, 0.002, -0.001, 0.001, -0.002, 0.003, -0.001]  # sum=0.003, mae=0.002
    notional[0, 1, 10:12] = 1e-6  # below eps=1e-4 -> ignored

    net = np.sum(sym_net, axis=1)
    tensor = pc.UnitCopy5mTensor(
        aliases=("EDGE",),
        symbols=("BTC", "ETH"),
        timestamps_ms=ts,
        notional_frac=notional,
        delta_notional_frac=d_notional,
        gross_ret_frac=net.copy(),
        net_ret_frac=net,
        sym_net_ret_frac=sym_net,
    )
    eps = pc.reconstruct_execution_episodes(tensor)[0]
    # Sorted by close bar e_idx: BTC Ep1(3), BTC Ep2(6), ETH(7), BTC Ep3(9), BTC Ep4(13), BTC Ep5(20)
    assert eps["ep_pnl"] == pytest.approx([-0.001, 0.002, 0.003, 0.0, 0.001, 0.001], abs=1e-12)
    assert eps["ep_peak_notional"] == pytest.approx([1.0, 2.0, 0.8, 1.5, 0.5, 1.0], abs=1e-12)
    assert eps["ep_duration_bars"] == pytest.approx([3.0, 2.0, 6.0, 3.0, 1.0, 3.0], abs=1e-12)
    assert eps["ep_mae"] == pytest.approx([0.004, 0.003, 0.002, 0.004, 0.001, 0.003], abs=1e-12)
    expected_sym_notional = np.array(
        [
            [3.0, 0.0],
            [4.0, 0.0],
            [0.0, 4.8],
            [4.5, 0.0],
            [0.5, 0.0],
            [3.0, 0.0],
        ]
    )
    assert eps["ep_symbol_notional"] == pytest.approx(expected_sym_notional, abs=1e-12)
    assert eps["ep_start_bar"].tolist() == [0, 4, 1, 6, 12, 17]
    assert eps["ep_end_bar"].tolist() == [3, 6, 7, 9, 13, 20]
    assert eps["ep_pnl_end_bar"].tolist() == [4, 6, 8, 10, 14, 20]
    assert eps["ep_left_censored"].tolist() == [False, False, False, False, False, False]
    assert eps["ep_right_censored"].tolist() == [False, False, False, False, False, True]
    assert eps["ep_uncensored"].tolist() == [True, True, True, True, True, False]

    eps_unc = pc.reconstruct_execution_episodes(tensor, uncensored_only=True)[0]
    assert eps_unc["ep_pnl"].size == 5
    assert eps_unc["ep_pnl"] == pytest.approx([-0.001, 0.002, 0.003, 0.0, 0.001], abs=1e-12)

    # Bounded memory check: stage1_audit_traders must not allocate full dense (N, S, T) copies
    n_t, n_s, n_b = 40, 80, 2000  # dense float32 = 25.6 MB, dense float64 = 51.2 MB
    big_notional = np.zeros((n_t, n_s, n_b), dtype=np.float32)
    big_delta = np.zeros((n_t, n_s, n_b), dtype=np.float32)
    big_net = np.zeros((n_t, n_b), dtype=np.float64)
    for ti in range(n_t):
        s_i = ti % n_s
        big_notional[ti, s_i, 10:100] = 0.2
        big_delta[ti, s_i, 10] = 0.2
        big_delta[ti, s_i, 100] = -0.2
        big_net[ti, 11:100] = 0.0002
    big_tensor = pc.UnitCopy5mTensor(
        aliases=tuple(f"W{i}" for i in range(n_t)),
        symbols=tuple(f"S{s}" for s in range(n_s)),
        timestamps_ms=_make_timestamps(7)[:n_b],
        notional_frac=big_notional,
        delta_notional_frac=big_delta,
        gross_ret_frac=big_net.copy(),
        net_ret_frac=big_net,
    )
    tracemalloc.start()
    try:
        pc.stage1_audit_traders(big_tensor, pc.PodConstructorConfig(min_episodes=1))
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Peak temporary allocation must be far below a single dense (N, S, T) float32 array (25.6 MB)
    assert peak_bytes < 10 * 1024 * 1024


def test_crowding_distance_matrix_parity_and_bounded_memory() -> None:
    import tracemalloc

    bpd = 24
    n_days = 10
    n_bars = n_days * bpd
    ts = _make_timestamps(1)[:n_bars]
    cfg = pc.PodConstructorConfig(bars_per_day=bpd, tail_window_bars=12)

    notional = np.zeros((4, 3, n_bars), dtype=np.float64)
    d_notional = np.zeros((4, 3, n_bars), dtype=np.float64)
    ret = np.zeros((4, n_bars), dtype=np.float64)

    rng = np.random.default_rng(77)
    # T0 & T1 overlap on BTC (same sign days 0..4, opposite sign days 5..9) + T1 trades ETH
    notional[0, 0, : 5 * bpd] = 0.4
    notional[0, 0, 5 * bpd :] = 0.4
    notional[1, 0, : 5 * bpd] = 0.5
    notional[1, 0, 5 * bpd :] = -0.5
    notional[1, 1, 2 * bpd : 6 * bpd] = 0.2
    ret[0] = rng.normal(0.0002, 0.002, size=n_bars)
    ret[1, : 5 * bpd] = 0.8 * ret[0, : 5 * bpd] + rng.normal(0.0, 0.0005, size=5 * bpd)
    ret[1, 5 * bpd :] = -0.7 * ret[0, 5 * bpd :] + rng.normal(0.0, 0.0005, size=5 * bpd)

    # T2 active only on days 0..2 with zero-variance constant positive return
    notional[2, 2, : 3 * bpd] = 0.3
    ret[2, : 3 * bpd] = 0.0001
    # T3 active only on day 9 (so u_days(T2, T3) = 4 < 5)
    notional[3, 2, 9 * bpd :] = -0.3
    ret[3, 9 * bpd :] = -0.0004

    tensor = pc.UnitCopy5mTensor(
        aliases=("T0", "T1", "T2", "T3"),
        symbols=("BTC", "ETH", "SOL"),
        timestamps_ms=ts,
        notional_frac=notional,
        delta_notional_frac=d_notional,
        gross_ret_frac=ret.copy(),
        net_ret_frac=ret,
    )
    dist, comps = pc.compute_crowding_distance_matrix([0, 1, 2, 3], tensor, cfg)

    # Independent scalar reference for all pairs (a, b)
    active = np.sum(np.abs(notional), axis=1) > cfg.flat_notional_eps_frac
    starts = np.arange(0, n_bars - 12 + 1, 3)
    win_ret = np.column_stack([np.sum(ret[:, s : s + 12], axis=1) for s in starts])
    win_act = np.column_stack([np.any(active[:, s : s + 12], axis=1) for s in starts])
    q05 = [
        float(np.quantile(win_ret[i, win_act[i]], cfg.tail_quantile))
        if np.sum(win_act[i]) >= 5
        else -0.01
        for i in range(4)
    ]
    dd_ex = []
    for i in range(4):
        u, _, _ = pc.compute_underwater_and_cdar(np.cumsum(ret[i]), alpha=cfg.cdar_alpha)
        dd_ex.append(np.maximum(0.0, u - float(np.quantile(u, cfg.dd_tail_quantile))))
    day_r = ret.reshape(4, n_days, bpd).sum(axis=2)
    day_a = active.reshape(4, n_days, bpd).any(axis=2)

    w1, w2, w3, w4 = cfg.crowding_weights
    for a in range(4):
        assert dist[a, a] == 0.0
        for b in range(a + 1, 4):
            uw = win_act[a] | win_act[b]
            da, db = win_ret[a, uw] <= q05[a], win_ret[b, uw] <= q05[b]
            emp = 0.5 * (
                (float(np.mean(da[db])) if np.any(db) else 0.0)
                + (float(np.mean(db[da])) if np.any(da) else 0.0)
            )
            ib = active[a] & active[b]
            ni = int(np.sum(ib))
            ud = day_a[a] | day_a[b]
            if np.sum(ud) >= 5:
                dar, dbr = day_r[a, ud], day_r[b, ud]
                na, nb = dar < 0.0, dbr < 0.0
                pri = 0.5 * (
                    (float(np.mean(na[nb])) if np.any(nb) else 0.0)
                    + (float(np.mean(nb[na])) if np.any(na) else 0.0)
                )
                corr = (
                    float(np.corrcoef(dar, dbr)[0, 1])
                    if np.std(dar) > 1e-12 and np.std(dbr) > 1e-12
                    else 0.0
                )
            else:
                pri, corr = 0.0, 0.0
            w_s = ni / (ni + cfg.overlap_shrinkage_kappa)
            lam = float(np.clip(w_s * emp + (1.0 - w_s) * pri, 0.0, 1.0))
            na_d, nb_d = float(np.linalg.norm(dd_ex[a])), float(np.linalg.norm(dd_ex[b]))
            om = (
                float(np.clip(np.dot(dd_ex[a], dd_ex[b]) / (na_d * nb_d), 0.0, 1.0))
                if na_d > 1e-12 and nb_d > 1e-12
                else 0.0
            )
            if ni > 0:
                pa, pb = notional[a, :, ib], notional[b, :, ib]
                dots = np.sum(pa * pb, axis=1)
                norms = np.linalg.norm(pa, axis=1) * np.linalg.norm(pb, axis=1) + 1e-12
                psi = float(np.clip(np.mean(dots / norms), -1.0, 1.0))
            else:
                psi = 0.0
            reg = float(np.clip(0.5 * (1.0 + corr), 0.0, 1.0))
            expected_d = float(np.clip(1.0 - (w1 * lam + w2 * om + w3 * 0.5 * (1.0 + psi) + w4 * reg), 0.05, 1.0))
            assert comps["lambda_tail"][a, b] == pytest.approx(lam, abs=1e-12)
            assert comps["omega_dd"][a, b] == pytest.approx(om, abs=1e-12)
            assert comps["psi_pos"][a, b] == pytest.approx(psi, abs=1e-12)
            assert comps["regime_corr"][a, b] == pytest.approx(reg, abs=1e-12)
            assert dist[a, b] == pytest.approx(expected_d, abs=1e-12)

    # Bounded memory check: no dense (C, S, T) copy in compute_crowding_distance_matrix
    n_t, n_s, n_b = 40, 80, 2016  # 7 days * 288
    big_notional = np.zeros((n_t, n_s, n_b), dtype=np.float32)
    big_delta = np.zeros((n_t, n_s, n_b), dtype=np.float32)
    big_net = np.zeros((n_t, n_b), dtype=np.float64)
    for ti in range(n_t):
        s_i = ti % n_s
        big_notional[ti, s_i, 10:500] = 0.2
        big_net[ti, 11:500] = 0.0001
    big_tensor = pc.UnitCopy5mTensor(
        aliases=tuple(f"W{i}" for i in range(n_t)),
        symbols=tuple(f"S{s}" for s in range(n_s)),
        timestamps_ms=_make_timestamps(7),
        notional_frac=big_notional,
        delta_notional_frac=big_delta,
        gross_ret_frac=big_net.copy(),
        net_ret_frac=big_net,
    )
    tracemalloc.start()
    try:
        pc.compute_crowding_distance_matrix(list(range(n_t)), big_tensor)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak_bytes < 15 * 1024 * 1024


def test_sparse_overlap_netting_exactness() -> None:
    n_days, bpd = 4, 288
    n_bars = n_days * bpd
    ts = _make_timestamps(n_days, bpd)
    notional = np.zeros((2, 2, n_bars), dtype=np.float64)
    d_notional = np.zeros((2, 2, n_bars), dtype=np.float64)
    gross = np.zeros((2, n_bars), dtype=np.float64)

    # Trader 0 longs BTC (+1.0) at bar 10, closes at bar 30
    d_notional[0, 0, 10] = 1.0
    d_notional[0, 0, 30] = -1.0
    notional[0, 0, 10:30] = 1.0
    gross[0, 11:30] = 0.001

    # Trader 1 shorts BTC (-0.6) at exact same bar 10 and closes at bar 30 -> 60% internal netting
    d_notional[1, 0, 10] = -0.6
    d_notional[1, 0, 30] = 0.6
    notional[1, 0, 10:30] = -0.6
    gross[1, 11:30] = -0.0006

    # Also solo non-overlapping ETH trade on Trader 1 at bar 50
    d_notional[1, 1, 50] = 0.5
    d_notional[1, 1, 60] = -0.5
    notional[1, 1, 50:60] = 0.5
    gross[1, 51:60] = 0.002

    cost_rate = 6.5e-4
    solo_net = gross - np.sum(np.abs(d_notional), axis=1) * cost_rate

    weights = np.array([0.7, 0.3], dtype=np.float64)
    cfg = pc.PodConstructorConfig()
    solo_cache, t_ov, d_ov = pc._build_sparse_overlap_cache(d_notional, solo_net)
    day_slices = pc._daily_reset_slices(ts, bpd, cfg.reset_bar_offset)

    ret_fast, _, _, _, _, _, sav_fast = pc._evaluate_weights_unit_scale(
        weights, solo_cache, t_ov, d_ov, cost_rate, day_slices, cfg
    )

    # Exact brute-force netted 5m PnL over full (S, T) tensor
    dense_gross = weights @ gross
    dense_delta = np.einsum("m,mst->st", weights, d_notional)
    dense_cost = np.sum(np.abs(dense_delta), axis=0) * cost_rate
    expected_ret = float(np.sum(dense_gross - dense_cost))

    unnetted_cost = float(np.sum(np.abs(d_notional) * weights[:, None, None]) * cost_rate)
    assert ret_fast == pytest.approx(expected_ret, abs=1e-12)
    assert sav_fast == pytest.approx(unnetted_cost - float(np.sum(dense_cost)), abs=1e-12)


def test_build_tensor_from_unit_replays_roundtrip_polars() -> None:
    start = datetime(2026, 7, 1, tzinfo=UTC)
    mins = [start + timedelta(minutes=m) for m in range(15)]
    ts_5m = [int((start + timedelta(minutes=5 * b)).timestamp() * 1000) for b in range(3)]

    pos_vals = [0.0] * 5 + [2.0] * 5 + [0.0] * 5
    delta_vals = [0.0] * 15
    delta_vals[5] = 2.0
    delta_vals[10] = -2.0
    open_vals = [100.0] * 15
    gross_vals = [0.0] * 5 + [10.0, 5.0, 0.0, 5.0, 10.0] + [0.0] * 5
    cost_vals = [0.0] * 15
    cost_vals[5] = 0.13
    cost_vals[10] = 0.13
    net_vals = [g - c for g, c in zip(gross_vals, cost_vals)]

    df = pl.DataFrame(
        {
            "timestamp": mins,
            "open": open_vals,
            "close": open_vals,
            "pos": pos_vals,
            "delta": delta_vals,
            "gross": gross_vals,
            "cost": cost_vals,
            "net": net_vals,
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC")))

    tensor = pc.build_tensor_from_unit_replays(
        {"ACT08": {"BTC": df}},
        ts_5m,
        pod_equity_usd=10_000.0,
    )
    assert tensor.aliases == ("ACT08",)
    assert tensor.symbols == ("BTC",)
    assert tensor.notional_frac[0, 0, 1] == pytest.approx(200.0 / 10_000.0)
    assert tensor.delta_notional_frac[0, 0, 1] == pytest.approx(200.0 / 10_000.0)
    assert tensor.delta_notional_frac[0, 0, 2] == pytest.approx(-200.0 / 10_000.0)
    assert float(np.sum(tensor.net_ret_frac[0])) == pytest.approx(sum(net_vals) / 10_000.0)


def test_build_tensor_compact_defaults_fast_path_and_from_arrays_parity() -> None:
    start = datetime(2026, 7, 1, tzinfo=UTC)
    n_bars = 12
    ts_5m = np.array(
        [int((start + timedelta(minutes=5 * b)).timestamp() * 1000) for b in range(n_bars)],
        dtype=np.int64,
    )
    ts_utc = pl.from_epoch(pl.Series("timestamp", ts_5m, dtype=pl.Int64), time_unit="ms").dt.replace_time_zone("UTC")

    open_btc = np.full(n_bars, 100.0)
    pos_btc = np.zeros(n_bars)
    pos_btc[2:6] = 1.5
    pos_btc[6:9] = -1.0
    delta_btc = np.zeros(n_bars)
    delta_btc[2] = 1.5
    delta_btc[6] = -2.5
    delta_btc[9] = 1.0
    gross_btc = np.zeros(n_bars)
    gross_btc[3:9] = 5.0
    cost_btc = np.abs(delta_btc) * open_btc * 6.5e-4
    net_btc = gross_btc - cost_btc

    open_eth = np.full(n_bars, 50.0)
    pos_eth = np.zeros(n_bars)
    pos_eth[1:4] = 4.0
    delta_eth = np.zeros(n_bars)
    delta_eth[1] = 4.0
    delta_eth[4] = -4.0
    gross_eth = np.zeros(n_bars)
    gross_eth[2:4] = -3.0
    cost_eth = np.abs(delta_eth) * open_eth * 6.5e-4
    net_eth = gross_eth - cost_eth

    df_btc = pl.DataFrame(
        {
            "timestamp": ts_utc,
            "open": open_btc,
            "close": open_btc,
            "pos": pos_btc,
            "delta": delta_btc,
            "gross": gross_btc,
            "cost": cost_btc,
            "net": net_btc,
        }
    )
    df_eth = pl.DataFrame(
        {
            "timestamp": ts_utc,
            "open": open_eth,
            "close": open_eth,
            "pos": pos_eth,
            "delta": delta_eth,
            "gross": gross_eth,
            "cost": cost_eth,
            "net": net_eth,
        }
    )

    compact = pc.build_tensor_from_unit_replays(
        {"W1": {"BTC": df_btc}, "W2": {"ETH": df_eth}},
        ts_5m,
        pod_equity_usd=10_000.0,
    )
    assert compact.notional_frac.dtype == np.float32
    assert compact.delta_notional_frac.dtype == np.float32
    assert compact.gross_ret_frac.dtype == np.float64
    assert compact.net_ret_frac.dtype == np.float64
    assert compact.sym_net_ret_frac is None

    full64 = pc.build_tensor_from_unit_replays(
        {"W1": {"BTC": df_btc}, "W2": {"ETH": df_eth}},
        ts_5m,
        pod_equity_usd=10_000.0,
        tensor_dtype=np.float64,
        include_sym_net_ret=True,
    )
    assert full64.notional_frac.dtype == np.float64
    assert full64.sym_net_ret_frac is not None
    assert full64.sym_net_ret_frac.dtype == np.float64

    from_arrays = pc.build_tensor_from_arrays(
        {
            "W1": {
                "BTC": {
                    "open": open_btc,
                    "pos": pos_btc,
                    "delta": delta_btc,
                    "gross": gross_btc,
                    "net": net_btc,
                }
            },
            "W2": {
                "ETH": {
                    "notional": pos_eth * open_eth,
                    "delta_notional": delta_eth * open_eth,
                    "gross": gross_eth,
                    "cost": cost_eth,
                }
            },
        },
        ts_5m,
        pod_equity_usd=10_000.0,
    )
    assert from_arrays.aliases == compact.aliases
    assert from_arrays.symbols == compact.symbols
    assert from_arrays.notional_frac.dtype == np.float32
    assert np.array_equal(from_arrays.notional_frac, compact.notional_frac)
    assert np.array_equal(from_arrays.delta_notional_frac, compact.delta_notional_frac)
    assert np.allclose(from_arrays.gross_ret_frac, compact.gross_ret_frac, atol=1e-12)
    assert np.allclose(from_arrays.net_ret_frac, compact.net_ret_frac, atol=1e-12)
    assert np.allclose(compact.notional_frac, full64.notional_frac.astype(np.float32), atol=1e-7)

    # Unsorted 1m DataFrames fall back cleanly to Polars 5m aggregation + searchsorted alignment
    df_1m_unsorted = (
        pl.DataFrame(
            {
                "timestamp": [
                    start + timedelta(minutes=11),
                    start + timedelta(minutes=10),
                    start + timedelta(minutes=12),
                ],
                "open": [100.0, 100.0, 100.0],
                "close": [100.0, 100.0, 100.0],
                "pos": [2.0, 2.0, 2.0],
                "delta": [0.0, 2.0, 0.0],
                "gross": [3.0, 1.0, 4.0],
                "cost": [0.0, 0.13, 0.0],
                "net": [3.0, 0.87, 4.0],
            }
        )
        .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
    )
    fallback_t = pc.build_tensor_from_unit_replays({"U1": {"BTC": df_1m_unsorted}}, ts_5m)
    assert fallback_t.notional_frac[0, 0, 2] == pytest.approx(200.0 / 10_000.0)
    assert fallback_t.delta_notional_frac[0, 0, 2] == pytest.approx(200.0 / 10_000.0)
    assert fallback_t.net_ret_frac[0, 2] == pytest.approx(7.87 / 10_000.0)

    with pytest.raises(ValueError, match="expected"):
        pc.build_tensor_from_arrays(
            {"W1": {"BTC": {"notional": np.zeros(3), "delta_notional": np.zeros(3), "gross": np.zeros(3), "net": np.zeros(3)}}},
            ts_5m,
        )


def _synthetic_multi_regime_tensor(n_days: int = 20) -> pc.UnitCopy5mTensor:
    rng = np.random.default_rng(1234)
    bpd = 288
    n_bars = n_days * bpd
    ts = _make_timestamps(n_days, bpd)
    chop_mask = np.arange(n_bars) >= (n_bars // 2)

    # 6 traders: 2 trend clones (BTC), 2 alt momentum, 1 defensive chop alpha, 1 tilter
    n_traders, n_syms = 6, 3
    notional = np.zeros((n_traders, n_syms, n_bars), dtype=np.float64)
    d_notional = np.zeros((n_traders, n_syms, n_bars), dtype=np.float64)
    gross = np.zeros((n_traders, n_bars), dtype=np.float64)
    cost_rate = 6.5e-4

    # Common trend + chop crash shock events
    mkt = rng.normal(0.0, 0.001, size=(n_syms, n_bars))
    for crash_t in range(n_bars // 2 + 100, n_bars - 100, 400):
        mkt[:, crash_t : crash_t + 12] -= 0.004

    for i in range(n_traders):
        t = 12
        last_loss = False
        while t < n_bars - 40:
            gap = int(rng.integers(24, 96))
            t += gap
            if t >= n_bars - 36:
                break
            dur = int(rng.integers(6, 20))
            end_t = min(n_bars - 1, t + dur)
            size = 0.25 if (i != 5 or not last_loss) else 0.70

            if i in (0, 1):  # Trend clones long BTC in first half, whipsawed in crashes
                sym = 0
                drift = 0.0012 if not chop_mask[t] else -0.0001
                bar_r = drift + mkt[sym, t + 1 : end_t]
            elif i in (2, 3):  # SOL/ETH asymmetric traders
                sym = 1 if i == 2 else 2
                drift = 0.0007
                bar_r = drift + 0.5 * mkt[sym, t + 1 : end_t]
            elif i == 4:  # Chop / short anchor: flat early, profitable during crashes
                sym = 0
                notional[i, sym, t:end_t] = -size
                d_notional[i, sym, t] = -size
                d_notional[i, sym, end_t] = size
                gross[i, t + 1 : end_t] = size * (-mkt[sym, t + 1 : end_t] + 0.0008)
                t = end_t
                continue
            else:  # Trader 5: tilter
                sym = 1
                bar_r = rng.normal(0.0001, 0.0015, size=(end_t - t - 1))

            notional[i, sym, t:end_t] = size
            d_notional[i, sym, t] = size
            d_notional[i, sym, end_t] = -size
            gross[i, t + 1 : end_t] = size * bar_r
            ep_net = float(np.sum(gross[i, t + 1 : end_t]) - 2.0 * size * cost_rate)
            last_loss = ep_net < 0.0
            t = end_t

    net = gross - np.sum(np.abs(d_notional), axis=1) * cost_rate
    return pc.UnitCopy5mTensor(
        aliases=("TREND_A", "TREND_B", "ETH_SW", "SOL_SW", "CHOP_ANCHOR", "TILTER"),
        symbols=("BTC", "ETH", "SOL"),
        timestamps_ms=ts,
        notional_frac=notional,
        delta_notional_frac=d_notional,
        gross_ret_frac=gross,
        net_ret_frac=net,
        chop_mask=chop_mask,
    )


def test_discover_diverse_pods_screens_tilter_clusters_and_respects_barriers() -> None:
    tensor = _synthetic_multi_regime_tensor(n_days=24)
    cfg = pc.PodConstructorConfig(pod_size=3, min_episodes=10, max_tilt_multiplier=1.40)
    pods, metrics, baselines, diag = pc.discover_diverse_pods(tensor, cfg)

    by_alias = {m.alias: m for m in metrics}
    assert not by_alias["TILTER"].passed_behavioral_gate
    assert "martingale_tilt" in by_alias["TILTER"].reject_reason
    assert by_alias["CHOP_ANCHOR"].passed_behavioral_gate

    top_pod = pods[0]
    assert len(top_pod.aliases) == 3
    assert baselines[top_pod.pod_indices[0]].mae_p95 > 0.0
    assert top_pod.cdar_95 <= cfg.cdar_buffer_frac * cfg.max_dd_frac + 1e-9
    assert top_pod.daily_cvar_99 <= cfg.daily_cvar_buffer_frac * cfg.daily_loss_frac + 1e-9
    assert top_pod.peak_to_trough_dd <= cfg.hard_dd_buffer_frac * cfg.max_dd_frac + 1e-9
    assert top_pod.static_floor_dd <= top_pod.peak_to_trough_dd + 1e-12
    assert top_pod.max_daily_loss <= cfg.hard_daily_buffer_frac * cfg.daily_loss_frac + 1e-9
    assert diag["distance_matrix"].shape[0] >= 4


def test_pre_clustering_min_baseline_order_filtering() -> None:
    base_tensor = _synthetic_multi_regime_tensor(n_days=20)
    # Build a 5-trader tensor from the 4 healthy traders (0..3) + 1 profitable micro-notional trader
    n_bars = base_tensor.timestamps_ms.size
    notional = np.zeros((5, 3, n_bars), dtype=np.float64)
    d_notional = np.zeros((5, 3, n_bars), dtype=np.float64)
    gross = np.zeros((5, n_bars), dtype=np.float64)
    net = np.zeros((5, n_bars), dtype=np.float64)

    notional[:4] = base_tensor.notional_frac[:4]
    d_notional[:4] = base_tensor.delta_notional_frac[:4]
    gross[:4] = base_tensor.gross_ret_frac[:4]
    net[:4] = base_tensor.net_ret_frac[:4]

    # Trader 4 ("MICRO"): 12 clean winning episodes with small drawdown, but $10 unit notional (q10_frac=0.001)
    # At baseline_port_dd_frac=0.02, pod_size=3 -> alloc=0.00667; with unit_dd=0.01, base_k=0.667 -> q10_order=$6.67 < $15
    for e_idx in range(12):
        s_b = 50 + e_idx * 400
        e_b = s_b + 12
        notional[4, 0, s_b:e_b] = 0.001
        d_notional[4, 0, s_b] = 0.001
        d_notional[4, 0, e_b] = -0.001
        gross[4, s_b + 1 : e_b] = 0.001 if e_idx != 3 else -0.0009
        net[4, s_b + 1 : e_b] = gross[4, s_b + 1 : e_b]
    # Give MICRO a controlled 1% standalone max DD (0.01) so base_k = (0.02/3)/0.01 = 0.667
    net[4, 65] = -0.010

    tensor = pc.UnitCopy5mTensor(
        aliases=("TREND_A", "TREND_B", "ETH_SW", "SOL_SW", "MICRO"),
        symbols=base_tensor.symbols,
        timestamps_ms=base_tensor.timestamps_ms,
        notional_frac=notional,
        delta_notional_frac=d_notional,
        gross_ret_frac=gross,
        net_ret_frac=net,
        chop_mask=base_tensor.chop_mask,
    )

    # Default (min_baseline_order_usd=0.0) keeps MICRO in clustering
    cfg_off = pc.PodConstructorConfig(pod_size=3, min_episodes=8, min_active_bar_frac=0.01)
    _, metrics_off, _, diag_off = pc.discover_diverse_pods(tensor, cfg_off)
    assert next(m for m in metrics_off if m.alias == "MICRO").passed_behavioral_gate
    assert diag_off["distance_matrix"].shape == (5, 5)

    # Enabled (min_baseline_order_usd=15.0) rejects MICRO before UPGMA clustering
    cfg_on = pc.PodConstructorConfig(
        pod_size=3,
        min_episodes=8,
        min_active_bar_frac=0.01,
        baseline_port_dd_frac=0.02,
        min_baseline_order_usd=15.0,
    )
    pods_on, metrics_on, _, diag_on = pc.discover_diverse_pods(tensor, cfg_on)
    micro_m = next(m for m in metrics_on if m.alias == "MICRO")
    assert not micro_m.passed_behavioral_gate
    assert "below_min_baseline_order" in micro_m.reject_reason
    assert diag_on["distance_matrix"].shape == (4, 4)
    assert all("MICRO" not in p.aliases for p in pods_on)


def test_leakfree_cpcv_pbo_and_stationary_block_bootstrap() -> None:
    tensor = _synthetic_multi_regime_tensor(n_days=18)
    cfg = pc.PodConstructorConfig(pod_size=3, min_episodes=8, cpcv_n_blocks=4)
    pods, _, _, _ = pc.discover_diverse_pods(tensor, cfg)

    cand_indices = [p.pod_indices for p in pods[:4]]
    pbo, logits = pc.compute_leakfree_cpcv_pbo(cand_indices, tensor, cfg)
    assert 0.0 <= pbo <= 1.0
    assert logits.size >= 1

    # Stationary block sampler should preserve consecutive days on average (gap != pure iid)
    rng = np.random.default_rng(99)
    idx_mat = pc.sample_stationary_block_indices(
        n_days=30, n_paths=500, horizon_days=20, mean_block_days=7.0, rng=rng
    )
    consec_frac = float(np.mean(((idx_mat[:, :-1] + 1) % 30) == idx_mat[:, 1:]))
    assert consec_frac > 0.70  # theoretical E[stay] = 1 - 1/7 = 85.7%

    # Drop-in Guardian MC interface check
    is_chop = np.array([False] * 15 + [True] * 15)
    day_end = np.array([45.0] * 15 + [-10.0] * 15)
    day_max = np.array([80.0] * 15 + [25.0] * 15)
    day_min = np.array([-30.0] * 15 + [-70.0] * 15)
    day_swept = np.zeros(30, dtype=np.int32)
    day_mf = np.ones(30, dtype=bool)
    mc = pc.run_stationary_block_mc_on_daily_blocks(
        (is_chop, day_end, day_max, day_min, day_swept, day_mf),
        "75%-Chop",
        n_paths=500,
        horizon_days=90,
        mean_block_days=5.0,
        rng=rng,
    )
    assert 0.0 <= mc["p_pass"] <= 100.0
    assert abs(mc["p_pass"] + mc["p_fail_static"] + mc["p_fail_daily"] + mc["p_timeout"] - 100.0) < 0.1


def test_leakfree_cpcv_pbo_trader_sliced_equivalence_and_bounded_memory() -> None:
    import tracemalloc

    # Embed 6 active multi-regime traders inside a larger 40-trader x 40-symbol universe
    base = _synthetic_multi_regime_tensor(n_days=12)
    n_t, n_s, n_b = 40, 40, base.timestamps_ms.size
    full_notional = np.zeros((n_t, n_s, n_b), dtype=np.float32)
    full_delta = np.zeros((n_t, n_s, n_b), dtype=np.float32)
    full_gross = np.zeros((n_t, n_b), dtype=np.float64)
    full_net = np.zeros((n_t, n_b), dtype=np.float64)

    # Map base traders 0..4 into sparse non-contiguous slots [2, 11, 23, 35, 38] and non-contiguous coins [3, 17, 29]
    t_slots = [2, 11, 23, 35, 38]
    s_slots = [3, 17, 29]
    for bi, ti in enumerate(t_slots):
        for bs, si in enumerate(s_slots):
            full_notional[ti, si] = base.notional_frac[bi, bs]
            full_delta[ti, si] = base.delta_notional_frac[bi, bs]
        full_gross[ti] = base.gross_ret_frac[bi]
        full_net[ti] = base.net_ret_frac[bi]

    # Add dummy distracting activity on unused traders/symbols
    for extra_t in (0, 5, 19, 30):
        full_notional[extra_t, extra_t, 50:200] = 0.4
        full_delta[extra_t, extra_t, 50] = 0.4
        full_delta[extra_t, extra_t, 200] = -0.4
        full_net[extra_t, 51:200] = 0.0005
        full_gross[extra_t, 51:200] = 0.0005

    full_tensor = pc.UnitCopy5mTensor(
        aliases=tuple(f"W{i}" for i in range(n_t)),
        symbols=tuple(f"C{s}" for s in range(n_s)),
        timestamps_ms=base.timestamps_ms,
        notional_frac=full_notional,
        delta_notional_frac=full_delta,
        gross_ret_frac=full_gross,
        net_ret_frac=full_net,
        chop_mask=base.chop_mask,
    )

    cand_pods = [
        (t_slots[0], t_slots[2], t_slots[3]),
        (t_slots[1], t_slots[2], t_slots[4]),
        (t_slots[0], t_slots[3], t_slots[4]),
        (t_slots[1], t_slots[3], t_slots[4]),
    ]
    cfg = pc.PodConstructorConfig(pod_size=3, min_episodes=4, cpcv_n_blocks=4)

    # Manually sliced universe containing strictly the 5 used traders & 3 used symbols
    sliced_tensor = pc.UnitCopy5mTensor(
        aliases=tuple(f"W{i}" for i in t_slots),
        symbols=tuple(f"C{s}" for s in s_slots),
        timestamps_ms=base.timestamps_ms,
        notional_frac=full_notional[np.ix_(t_slots, s_slots)],
        delta_notional_frac=full_delta[np.ix_(t_slots, s_slots)],
        gross_ret_frac=full_gross[t_slots],
        net_ret_frac=full_net[t_slots],
        chop_mask=base.chop_mask,
    )
    remap = {orig: idx for idx, orig in enumerate(t_slots)}
    remapped_pods = [tuple(remap[t] for t in p) for p in cand_pods]

    pbo_ref, logits_ref = pc.compute_leakfree_cpcv_pbo(remapped_pods, sliced_tensor, cfg)

    tracemalloc.start()
    try:
        pbo_full, logits_full = pc.compute_leakfree_cpcv_pbo(cand_pods, full_tensor, cfg)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert pbo_full == pytest.approx(pbo_ref, abs=1e-12)
    assert np.allclose(logits_full, logits_ref, atol=1e-12, rtol=0.0)
    # Must not allocate full (40, 40, T/2) tr/te copies (~44 MB) across CPCV splits
    assert peak_bytes < 10 * 1024 * 1024


def test_slice_tensor_for_traders_preserves_chop_mask() -> None:
    base = _synthetic_multi_regime_tensor(n_days=8)
    assert base.chop_mask is not None

    sub_tensor, remapped_pods = pc._slice_tensor_for_traders(base, [(0, 2, 4)])
    assert remapped_pods == [(0, 1, 2)]
    assert sub_tensor.chop_mask is not None
    assert np.array_equal(sub_tensor.chop_mask, base.chop_mask)

    _, metrics, _ = pc.stage1_audit_traders(sub_tensor, pc.PodConstructorConfig(min_episodes=2))
    assert any(m.chop_net_over_dd != m.net_over_max_dd for m in metrics)



def test_roster_episode_cusum_detects_decay_and_tilt() -> None:
    baseline = pc.TraderEpisodeBaseline(
        trader_id=0,
        alias="ACT08",
        mean_ep_return=0.005,
        std_ep_return=0.010,
        median_ln_peak_notional=math.log(0.25),
        mad_ln_peak_notional=0.15,
        mean_win_ln_duration=2.0,
        std_win_ln_duration=0.4,
        symbol_notional_dist=np.array([0.8, 0.15, 0.05]),
        mae_p95=0.025,
    )
    state = pc.RosterCUSUMState(alias="ACT08")
    # Normal winning episode keeps state OK
    pc.update_roster_episode_cusum(
        state,
        baseline,
        ep_return=0.008,
        ep_peak_notional_frac=0.25,
        ep_duration_bars=8.0,
        ep_mae=0.005,
        ep_symbol_notional=np.array([0.8, 0.15, 0.05]),
    )
    assert state.status == "OK"

    # Persistent heavy losses + size tilt + alt drift trigger KILL_CONTAIN_0X
    for _ in range(3):
        pc.update_roster_episode_cusum(
            state,
            baseline,
            ep_return=-0.030,
            ep_peak_notional_frac=0.90,
            ep_duration_bars=60.0,
            ep_mae=0.030,
            ep_symbol_notional=np.array([0.05, 0.10, 0.85]),
        )
    assert state.status == "KILL_CONTAIN_0X"
    assert len(state.reasons) >= 1


def test_censored_and_overlapping_episodes_excluded_and_causal_tilt() -> None:
    n_bars = 80
    ts = _make_timestamps(1)[:n_bars]
    notional = np.zeros((1, 3, n_bars), dtype=np.float64)
    d_notional = np.zeros((1, 3, n_bars), dtype=np.float64)
    sym_net = np.zeros((1, 3, n_bars), dtype=np.float64)

    # BTC (sym 0):
    # - Left-censored carry at bar 0..3 (delta[0]=0 -> pre0 != 0), closed at bar 4: MUST be censored!
    notional[0, 0, 0:4] = 1.0
    d_notional[0, 0, 4] = -1.0
    sym_net[0, 0, 1:5] = -0.002

    # - Ep A (uncensored): bars 5..29, closes at bar 30 (known=31) with LOSS (-0.01)
    notional[0, 0, 5:30] = 0.4
    d_notional[0, 0, 5] = 0.4
    d_notional[0, 0, 30] = -0.4
    sym_net[0, 0, 6:31] = -0.0004  # 25 * -0.0004 = -0.010

    # - Ep D (uncensored): opens at bar 35 (AFTER A's loss known at 31, BEFORE B closes at 41!), size 1.5, WIN
    notional[0, 0, 35:45] = 1.5
    d_notional[0, 0, 35] = 1.5
    d_notional[0, 0, 45] = -1.5
    sym_net[0, 0, 36:46] = 0.002

    # ETH (sym 1):
    # - Ep B (overlapping, uncensored): opens at bar 10 (BEFORE A closes!), size 2.0, closes WIN at 40 (known=41)
    #   Crucial: Ep B must NOT be treated as opened after A's loss despite closing after A!
    notional[0, 1, 10:40] = 2.0
    d_notional[0, 1, 10] = 2.0
    d_notional[0, 1, 40] = -2.0
    sym_net[0, 1, 11:41] = 0.001

    # SOL (sym 2):
    # - Ep C (uncensored): opens at bar 50 (after B & D wins), size 0.5, closes WIN at bar 60
    notional[0, 2, 50:60] = 0.5
    d_notional[0, 2, 50] = 0.5
    d_notional[0, 2, 60] = -0.5
    sym_net[0, 2, 51:61] = 0.001

    # - Right-censored open at slice end: bars 65..79 (size 0.35), still open at cutoff!
    notional[0, 2, 65:80] = 0.35
    d_notional[0, 2, 65] = 0.35
    sym_net[0, 2, 66:80] = 0.0005

    net = np.sum(sym_net, axis=1)
    tensor = pc.UnitCopy5mTensor(
        aliases=("OVERLAP",),
        symbols=("BTC", "ETH", "SOL"),
        timestamps_ms=ts,
        notional_frac=notional,
        delta_notional_frac=d_notional,
        gross_ret_frac=net.copy(),
        net_ret_frac=net,
        sym_net_ret_frac=sym_net,
    )

    eps_all = pc.reconstruct_execution_episodes(tensor)[0]
    assert eps_all["ep_pnl"].size == 6
    assert int(np.sum(eps_all["ep_left_censored"])) == 1
    assert int(np.sum(eps_all["ep_right_censored"])) == 1
    assert int(np.sum(eps_all["ep_uncensored"])) == 4

    cfg = pc.PodConstructorConfig(min_episodes=4, max_tilt_multiplier=2.0, min_active_bar_frac=0.1)
    passed_ids, metrics, _ = pc.stage1_audit_traders(tensor, cfg)
    m = metrics[0]
    assert m.num_episodes == 4
    assert m.censored_episodes == 2
    assert m.left_censored_episodes == 1
    assert m.right_censored_episodes == 1
    assert m.open_exposure_at_cutoff == pytest.approx(0.35, abs=1e-6)
    # Only Ep D (1.5) opened after A's loss; only Ep C (0.5) opened after a win -> 1.5 / 0.5 = 3.0!
    assert m.tilt_status == "FAIL"
    assert m.tilt_multiplier == pytest.approx(3.0, abs=1e-6)
    assert passed_ids == []
    assert "martingale_tilt" in m.reject_reason


def test_unknown_tilt_never_fabricates_pass_or_one() -> None:
    n_bars = 120
    ts = _make_timestamps(1)[:n_bars]

    # Trader 0 ("ALL_LOSS"): 4 closed losing episodes, 0 wins (q10-evidence counterexample)
    # Trader 1 ("ALL_WIN"): 4 closed winning episodes, 0 losses
    notional = np.zeros((2, 1, n_bars), dtype=np.float64)
    d_notional = np.zeros((2, 1, n_bars), dtype=np.float64)
    sym_net = np.zeros((2, 1, n_bars), dtype=np.float64)

    for ep in range(4):
        s_b = 5 + ep * 20
        e_b = s_b + 10
        notional[0, 0, s_b:e_b] = 0.5
        d_notional[0, 0, s_b] = 0.5
        d_notional[0, 0, e_b] = -0.5
        sym_net[0, 0, s_b + 1 : e_b + 1] = -0.001

        notional[1, 0, s_b:e_b] = 0.5
        d_notional[1, 0, s_b] = 0.5
        d_notional[1, 0, e_b] = -0.5
        sym_net[1, 0, s_b + 1 : e_b + 1] = 0.001

    net = np.sum(sym_net, axis=1)
    tensor = pc.UnitCopy5mTensor(
        aliases=("ALL_LOSS", "ALL_WIN"),
        symbols=("BTC",),
        timestamps_ms=ts,
        notional_frac=notional,
        delta_notional_frac=d_notional,
        gross_ret_frac=net.copy(),
        net_ret_frac=net,
        sym_net_ret_frac=sym_net,
    )

    cfg_strict = pc.PodConstructorConfig(min_episodes=4, min_active_bar_frac=0.1, reject_unknown_tilt=True)
    passed_strict, m_strict, _ = pc.stage1_audit_traders(tensor, cfg_strict)
    assert passed_strict == []
    assert m_strict[0].tilt_multiplier is None
    assert m_strict[0].tilt_status == "UNKNOWN"
    assert not m_strict[0].passed_behavioral_gate
    assert "tilt_unknown" in m_strict[0].reject_reason

    assert m_strict[1].tilt_multiplier is None
    assert m_strict[1].tilt_status == "UNKNOWN"
    assert not m_strict[1].passed_behavioral_gate
    assert "tilt_unknown" in m_strict[1].reject_reason

    # Ablation toggle: even when reject_unknown_tilt=False, UNKNOWN is never rewritten to 1.0
    cfg_ablate = pc.PodConstructorConfig(min_episodes=4, min_active_bar_frac=0.1, reject_unknown_tilt=False)
    passed_ablate, m_ablate, _ = pc.stage1_audit_traders(tensor, cfg_ablate)
    assert passed_ablate == [1]
    assert m_ablate[1].tilt_multiplier is None
    assert m_ablate[1].tilt_status == "UNKNOWN"


def test_distinct_peak_to_trough_and_static_floor_drawdowns() -> None:
    # Q11 counterexample: 10000 -> 10800 (+0.08) -> 10100 (-0.07)
    # Peak-to-trough DD = 0.07 ($700); static-floor DD below initial equity = 0.0 ($0).
    cum = np.array([0.0, 0.08, 0.01], dtype=np.float64)
    _, peak_dd, static_dd, _ = pc.compute_drawdown_metrics(cum, alpha=0.95)
    assert peak_dd == pytest.approx(0.07, abs=1e-12)
    assert static_dd == pytest.approx(0.0, abs=1e-12)

    n_bars = 288 * 2
    ts = _make_timestamps(2)
    notional = np.zeros((3, 1, n_bars), dtype=np.float64)
    d_notional = np.zeros((3, 1, n_bars), dtype=np.float64)
    sym_net = np.zeros((3, 1, n_bars), dtype=np.float64)

    for ti in range(3):
        # Ep1 (bar 10..30): big gain +0.08
        notional[ti, 0, 10:30] = 0.5
        d_notional[ti, 0, 10] = 0.5
        d_notional[ti, 0, 30] = -0.5
        sym_net[ti, 0, 20] = 0.08
        # Ep2 (bar 60..80): loss -0.07 -> cumulative stays +0.01 above initial 0.0!
        notional[ti, 0, 60:80] = 0.5
        d_notional[ti, 0, 60] = 0.5
        d_notional[ti, 0, 80] = -0.5
        sym_net[ti, 0, 70] = -0.07
        # Ep3..4 (bar 120..200): small wins
        notional[ti, 0, 120:140] = 0.5
        d_notional[ti, 0, 120] = 0.5
        d_notional[ti, 0, 140] = -0.5
        sym_net[ti, 0, 130] = 0.02
        notional[ti, 0, 160:180] = 0.5
        d_notional[ti, 0, 160] = 0.5
        d_notional[ti, 0, 180] = -0.5
        sym_net[ti, 0, 170] = 0.02

    net = np.sum(sym_net, axis=1)
    tensor = pc.UnitCopy5mTensor(
        aliases=("A", "B", "C"),
        symbols=("BTC",),
        timestamps_ms=ts,
        notional_frac=notional,
        delta_notional_frac=d_notional,
        gross_ret_frac=net.copy(),
        net_ret_frac=net,
        sym_net_ret_frac=sym_net,
    )
    cfg = pc.PodConstructorConfig(pod_size=3, min_episodes=4, min_active_bar_frac=0.05)
    _, metrics, _ = pc.stage1_audit_traders(tensor, cfg)
    for m in metrics:
        assert m.standalone_peak_to_trough_dd == pytest.approx(0.07, abs=1e-12)
        assert m.standalone_static_floor_dd == pytest.approx(0.0, abs=1e-12)

    sized = pc.size_pod_dual_barrier((0, 1, 2), tensor, metrics, cfg)
    assert sized.peak_to_trough_dd > 0.0
    assert sized.static_floor_dd == pytest.approx(0.0, abs=1e-12)
    assert sized.static_floor_dd < sized.peak_to_trough_dd


def test_infeasible_risk_share_bounds_raise_instead_of_relaxing() -> None:
    tensor = _synthetic_multi_regime_tensor(n_days=16)
    cfg_audit = pc.PodConstructorConfig(min_episodes=4, reject_unknown_tilt=False)
    _, metrics, _ = pc.stage1_audit_traders(tensor, cfg_audit)

    # 2 seats with max_risk_share=0.45 => 0.90 < 1.0 (previously silently relaxed!)
    with pytest.raises(ValueError, match="INFEASIBLE_RISK_SHARE_BOUNDS"):
        pc.size_pod_dual_barrier(
            (0, 2),
            tensor,
            metrics,
            pc.PodConstructorConfig(pod_size=2, min_risk_share=0.15, max_risk_share=0.45),
        )

    # 3 seats with min_risk_share=0.40 => 1.20 > 1.0
    with pytest.raises(ValueError, match="INFEASIBLE_RISK_SHARE_BOUNDS"):
        pc.size_pod_dual_barrier(
            (0, 2, 3),
            tensor,
            metrics,
            pc.PodConstructorConfig(pod_size=3, min_risk_share=0.40, max_risk_share=0.60),
        )

    # min_risk_share > max_risk_share
    with pytest.raises(ValueError, match="INFEASIBLE_RISK_SHARE_BOUNDS"):
        pc.size_pod_dual_barrier(
            (0, 2, 3),
            tensor,
            metrics,
            pc.PodConstructorConfig(pod_size=3, min_risk_share=0.50, max_risk_share=0.30),
        )


def test_migrated_study_callers_handle_unknown_tilt_and_distinct_dd() -> None:
    import importlib.util

    studies_dir = Path(__file__).resolve().parents[1] / "studies"
    base = _synthetic_multi_regime_tensor(n_days=24)
    n_t, n_s, n_b = base.notional_frac.shape
    notional = np.zeros((n_t + 1, n_s, n_b), dtype=np.float64)
    d_notional = np.zeros((n_t + 1, n_s, n_b), dtype=np.float64)
    gross = np.zeros((n_t + 1, n_b), dtype=np.float64)
    net = np.zeros((n_t + 1, n_b), dtype=np.float64)
    notional[:n_t] = base.notional_frac
    d_notional[:n_t] = base.delta_notional_frac
    gross[:n_t] = base.gross_ret_frac
    net[:n_t] = base.net_ret_frac
    for ep in range(10):
        sb = 20 + ep * 400
        eb = sb + 20
        notional[n_t, 0, sb:eb] = 0.2
        d_notional[n_t, 0, sb] = 0.2
        d_notional[n_t, 0, eb] = -0.2
        gross[n_t, sb + 1 : eb] = 0.001
        net[n_t, sb + 1 : eb] = 0.001
    tensor = pc.UnitCopy5mTensor(
        aliases=(*base.aliases, "ALL_WIN_UNKNOWN"),
        symbols=base.symbols,
        timestamps_ms=base.timestamps_ms,
        notional_frac=notional,
        delta_notional_frac=d_notional,
        gross_ret_frac=gross,
        net_ret_frac=net,
        chop_mask=base.chop_mask,
    )
    cfg = pc.PodConstructorConfig(pod_size=3, min_episodes=8, min_active_bar_frac=0.005)
    passed_ids, metrics, _ = pc.stage1_audit_traders(tensor, cfg)
    dist_mat, comps = pc.compute_crowding_distance_matrix(passed_ids, tensor, cfg)

    v1_path = studies_dir / "2026-11-16__cft__10k__canonical-full-universe.py"
    v1_spec = importlib.util.spec_from_file_location("cft_v1_caller", v1_path)
    assert v1_spec and v1_spec.loader
    v1_mod = importlib.util.module_from_spec(v1_spec)
    v1_spec.loader.exec_module(v1_mod)
    v1_cands, v1_audit = v1_mod.train_pods(tensor)
    assert len(v1_cands) >= 1
    assert "train_static_floor_dd_usd" in v1_cands[0]
    unk_rej = next(r for r in v1_audit["3"]["rejected"] if r["alias"] == "ALL_WIN_UNKNOWN")
    assert unk_rej["tilt"] is None
    assert unk_rej["tilt_status"] == "UNKNOWN"

    q3_path = studies_dir / "2026-11-17__cft__10k__sleeve-vs-upgma.py"
    q3_spec = importlib.util.spec_from_file_location("cft_q3_caller", q3_path)
    assert q3_spec and q3_spec.loader
    q3_mod = importlib.util.module_from_spec(q3_spec)
    q3_spec.loader.exec_module(q3_mod)
    a_rows, _ = q3_mod.upgma_candidates(tensor, metrics, passed_ids, dist_mat, cfg)
    b_rows, _ = q3_mod.sleeve_candidates(tensor, metrics, passed_ids, dist_mat, comps["psi_pos"], cfg)
    c_rows, _ = q3_mod.baseline_candidates(tensor, metrics, passed_ids, cfg)
    assert a_rows and c_rows
    assert "train_static_floor_dd_usd" in a_rows[0]
    assert isinstance(b_rows, list)
