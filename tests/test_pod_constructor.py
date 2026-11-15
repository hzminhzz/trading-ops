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

    ret_fast, _, _, _, _, sav_fast = pc._evaluate_weights_unit_scale(
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
    assert top_pod.max_static_dd <= cfg.hard_dd_buffer_frac * cfg.max_dd_frac + 1e-9
    assert top_pod.max_daily_loss <= cfg.hard_daily_buffer_frac * cfg.daily_loss_frac + 1e-9
    assert diag["distance_matrix"].shape[0] >= 4


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
