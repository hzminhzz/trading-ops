from __future__ import annotations

import importlib
import math
import sys
from datetime import UTC, datetime
from decimal import Decimal as D
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
prp = importlib.import_module("portfolio_risk_policy")
sd = importlib.import_module("study_data")


def _grid_ms(
    n_bars: int,
    start: datetime = datetime(2026, 7, 1, 0, 0, tzinfo=UTC),
) -> np.ndarray:
    start_ms = round(start.timestamp() * 1000.0)
    return start_ms + np.arange(n_bars, dtype=np.int64) * 300_000


def test_cwsa_one_shot_never_tops_up_on_unchanged_checkpoint() -> None:
    assert cr.resolve_scale_alias("kPEPE") == ("1000PEPE", 1.0)
    assert cr.resolve_scale_alias("BTC") == ("BTC", 1.0)

    n_bars = 14
    grid = _grid_ms(n_bars)
    src_pos = np.array(
        [[5.0, 5.0, 0.0, 10.0, 10.0, 10.0, 8.0, 2.0, 12.0, 12.0, 12.0, 12.0, 12.0, 14.0]],
        dtype=np.float64,
    )
    prices = np.full((1, n_bars), 100.0, dtype=np.float64)
    trade_open = prices.copy()
    mark_close = prices.copy()
    trade_open[0, 9] = np.nan
    mark_close[0, 9] = np.nan

    vol_1m = np.full((1, n_bars), 8_000.0, dtype=np.float64)
    liq_cfg = cr.CausalLiquidityConfig(
        l2_lookback_bars=3,
        l2_min_valid_bars=3,
        max_participation_1m=0.05,
    )
    cwsa_cfg = cr.CWSAReplayConfig(
        half_spread_bps=0.0,
        fee_bps=4.5,
        slippage_bps=2.0,
    )

    res = cr.simulate_cwsa_one_shot_replay(
        coins=("BTC",),
        source_positions=src_pos,
        source_marks=prices,
        trade_open=trade_open,
        mark_close=mark_close,
        grid_ms=grid,
        k_scale=1.0,
        vol_1m_usd=vol_1m,
        cwsa_cfg=cwsa_cfg,
        liquidity_cfg=liq_cfg,
    )

    assert res.checkpoint_classes[0, 0] == cr.CheckpointClass.BASELINE.value
    assert res.executed_positions[0, 0] == pytest.approx(0.0)
    assert res.executed_positions[0, 1] == pytest.approx(0.0)
    assert res.executed_positions[0, 2] == pytest.approx(0.0)

    assert res.checkpoint_classes[0, 3] == cr.CheckpointClass.CHANGED.value
    assert res.dispositions[0, 3] == cr.OrderDisposition.CLIPPED.value
    assert res.executed_positions[0, 3] == pytest.approx(4.0)

    assert res.checkpoint_classes[0, 4] == cr.CheckpointClass.UNCHANGED.value
    assert res.checkpoint_classes[0, 5] == cr.CheckpointClass.UNCHANGED.value
    assert res.executed_positions[0, 4] == pytest.approx(4.0)
    assert res.executed_positions[0, 5] == pytest.approx(4.0)
    assert res.executed_deltas[0, 4] == pytest.approx(0.0)
    assert res.executed_deltas[0, 5] == pytest.approx(0.0)

    assert res.checkpoint_classes[0, 6] == cr.CheckpointClass.CHANGED.value
    assert res.executed_positions[0, 6] == pytest.approx(4.0)

    assert res.checkpoint_classes[0, 7] == cr.CheckpointClass.CHANGED.value
    assert res.executed_positions[0, 7] == pytest.approx(2.0)

    assert res.checkpoint_classes[0, 8] == cr.CheckpointClass.CHANGED.value
    assert res.executed_positions[0, 8] == pytest.approx(6.0)

    assert res.checkpoint_classes[0, 9] == cr.CheckpointClass.FENCED.value
    assert res.dispositions[0, 9] == cr.OrderDisposition.QUARANTINED.value
    assert res.executed_positions[0, 9] == pytest.approx(6.0)

    assert res.dispositions[0, 10] == cr.OrderDisposition.QUARANTINED.value
    assert res.executed_positions[0, 10] == pytest.approx(6.0)
    assert res.executed_positions[0, 11] == pytest.approx(6.0)
    assert res.executed_positions[0, 12] == pytest.approx(6.0)

    assert res.checkpoint_classes[0, 13] == cr.CheckpointClass.CHANGED.value
    assert res.executed_positions[0, 13] == pytest.approx(10.0)


def test_q23_reversal_falls_back_to_reduce_only_when_frozen_seat_holds_opposing_residual() -> None:
    intents = (
        cr.SeatMarketIntent(
            seat_id="SEAT_FROZEN",
            held_qty=2.0,
            raw_desired_qty=2.0,
            seat_blocks_cross_zero=True,
            initial_disposition=cr.OrderDisposition.STATIC_HELD,
        ),
        cr.SeatMarketIntent(
            seat_id="SEAT_ACTIVE",
            held_qty=3.0,
            raw_desired_qty=-4.0,
            seat_blocks_cross_zero=False,
            initial_disposition=cr.OrderDisposition.NOOP,
        ),
    )

    alloc = cr.allocate_l3_market_step(
        0,
        intents,
        exec_price_buy=100.0,
        exec_price_sell=100.0,
        mark_price=100.0,
        l2_cap_usd=50_000.0,
        l2_prior_vol_valid=True,
        actual_bar_vol_1m_usd=100_000.0,
        cross_zero_permitted=True,
        protective_exit_haircut_bps=15.0,
    )

    assert alloc.projection_mode == cr.MarketProjectionMode.REDUCE_ONLY_FALLBACK
    assert alloc.seat_projected_qty["SEAT_FROZEN"] == pytest.approx(2.0)
    assert alloc.seat_projected_qty["SEAT_ACTIVE"] == pytest.approx(0.0)
    assert alloc.seat_post_qty["SEAT_FROZEN"] == pytest.approx(2.0)
    assert alloc.seat_post_qty["SEAT_ACTIVE"] == pytest.approx(0.0)
    assert alloc.seat_dispositions["SEAT_ACTIVE"] == cr.OrderDisposition.MISSED
    assert alloc.seat_dispositions["SEAT_FROZEN"] == cr.OrderDisposition.STATIC_HELD
    assert alloc.external_delta_qty == pytest.approx(-3.0)
    assert alloc.missed_notional_usd == pytest.approx(400.0)

    unlocked_intents = (
        cr.SeatMarketIntent(
            seat_id="SEAT_ACTIVE",
            held_qty=3.0,
            raw_desired_qty=-4.0,
            seat_blocks_cross_zero=False,
        ),
    )
    direct_flip = cr.allocate_l3_market_step(
        0,
        unlocked_intents,
        exec_price_buy=100.0,
        exec_price_sell=100.0,
        mark_price=100.0,
        l2_cap_usd=50_000.0,
        l2_prior_vol_valid=True,
        actual_bar_vol_1m_usd=100_000.0,
        cross_zero_permitted=True,
        protective_exit_haircut_bps=15.0,
    )
    assert direct_flip.projection_mode == cr.MarketProjectionMode.CROSS_ZERO_SINGLE_ORDER
    assert direct_flip.seat_post_qty["SEAT_ACTIVE"] == pytest.approx(-4.0)
    assert direct_flip.external_delta_qty == pytest.approx(-7.0)
    assert direct_flip.seat_dispositions["SEAT_ACTIVE"] == cr.OrderDisposition.EXECUTED

    disallowed_flip = cr.allocate_l3_market_step(
        0,
        unlocked_intents,
        exec_price_buy=100.0,
        exec_price_sell=100.0,
        mark_price=100.0,
        l2_cap_usd=50_000.0,
        l2_prior_vol_valid=True,
        actual_bar_vol_1m_usd=100_000.0,
        cross_zero_permitted=False,
        protective_exit_haircut_bps=15.0,
    )
    assert disallowed_flip.projection_mode == cr.MarketProjectionMode.REDUCE_ONLY_FALLBACK
    assert disallowed_flip.seat_post_qty["SEAT_ACTIVE"] == pytest.approx(0.0)
    assert disallowed_flip.seat_dispositions["SEAT_ACTIVE"] == cr.OrderDisposition.MISSED


def test_causal_liquidity_uses_strictly_prior_volume_and_prioritizes_exits(
    tmp_path: Path,
) -> None:
    grid = _grid_ms(8)
    store_root = tmp_path / "store"
    vike_root = tmp_path / "vike"
    btc_part = (
        store_root
        / "crypto_futures_ohlcv_1m_HYPERLIQUID_BTC_PERP"
        / "year=2026"
        / "month=07"
        / "day=01"
    )
    btc_part.mkdir(parents=True)
    vike_root.mkdir(parents=True)

    btc_ts = [
        datetime.fromtimestamp(float(grid[i] + 60_000) / 1000.0, tz=UTC)
        for i in (0, 1, 2, 3, 4, 5)
    ]
    pl.DataFrame(
        {
            "timestamp": pl.Series("timestamp", btc_ts).cast(pl.Datetime("us", "UTC")),
            "open": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "volume": [50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "quote_volume": [5_000.0, None, 7_000.0, 8_000.0, 9_000.0, 10_000.0],
        }
    ).write_parquet(btc_part / "data.parquet")

    pl.DataFrame(
        {
            "ts_ms": [int(grid[0] + 60_000), int(grid[1] + 60_000)],
            "open": [50.0, 50.0],
            "close": [50.0, 50.0],
            "volume": [200.0, 300.0],
        }
    ).write_parquet(vike_root / "ETH.parquet")

    prices = sd.LazyPrices(grid, store_root=store_root, vike_root=vike_root, streaming=False)
    vcache: dict[tuple[str, int, int, int], np.ndarray] = {}
    vols = sd.load_1m_volumes_on_grid(("BTC", "ETH"), grid, prices, vol_cache=vcache)

    assert vols.shape == (2, 8)
    assert vols[0, 0] == pytest.approx(5_000.0)
    assert vols[0, 1] == pytest.approx(6_000.0)
    assert np.isnan(vols[0, 6])
    assert np.isnan(vols[0, 7])
    assert vols[1, 0] == pytest.approx(10_000.0)
    assert vols[1, 1] == pytest.approx(15_000.0)
    assert np.isnan(vols[1, 2])
    assert ("BTC", int(grid[0]), int(grid[-1]), 8) in vcache

    l1_grid = _grid_ms(12)
    l1_vol = np.vstack(
        [
            np.full(12, 400_000.0, dtype=np.float64),
            np.array([400_000.0, np.nan, 400_000.0, 400_000.0] * 3, dtype=np.float64),
        ]
    )
    l1_res = cr.screen_l1_allowed_markets(
        ("BTC", "ILLIQ"),
        l1_vol,
        l1_grid,
        bars_per_day=4,
        cfg=cr.CausalLiquidityConfig(
            l1_window_days=2,
            l1_min_median_daily_usd=1_000_000.0,
            l1_min_1m_completeness=0.95,
        ),
    )
    assert l1_res.allowed_coins == ("BTC",)
    assert not bool(l1_res.rolling_allowed_mask[0, 0])
    assert bool(l1_res.rolling_allowed_mask[0, 8])
    assert not bool(l1_res.rolling_allowed_mask[1, 8])

    src_pos = np.array([[0.0, 0.0, 0.0, 10.0, 10.0, 10.0]], dtype=np.float64)
    px = np.full((1, 6), 100.0, dtype=np.float64)
    vol_a = np.array([[6_000.0, 6_000.0, 6_000.0, 6_000.0, 6_000.0, 6_000.0]], dtype=np.float64)
    vol_b = np.array(
        [[6_000.0, 6_000.0, 6_000.0, 9_000_000_000.0, 9_000_000_000.0, 6_000.0]],
        dtype=np.float64,
    )
    liq_cfg = cr.CausalLiquidityConfig(
        l2_lookback_bars=3,
        l2_min_valid_bars=3,
        max_participation_1m=0.05,
        protective_exit_haircut_bps=15.0,
    )
    res_a = cr.simulate_cwsa_one_shot_replay(
        coins=("BTC",),
        source_positions=src_pos,
        trade_open=px,
        mark_close=px,
        grid_ms=_grid_ms(6),
        k_scale=1.0,
        vol_1m_usd=vol_a,
        liquidity_cfg=liq_cfg,
    )
    res_b = cr.simulate_cwsa_one_shot_replay(
        coins=("BTC",),
        source_positions=src_pos,
        trade_open=px,
        mark_close=px,
        grid_ms=_grid_ms(6),
        k_scale=1.0,
        vol_1m_usd=vol_b,
        liquidity_cfg=liq_cfg,
    )
    assert res_a.participation_audit.lookahead_bars_used == 0
    assert res_a.l2_cap_usd[0, 3] == pytest.approx(300.0)
    assert res_b.l2_cap_usd[0, 3] == pytest.approx(300.0)
    assert res_a.executed_positions[0, 3] == pytest.approx(3.0)
    assert res_b.executed_positions[0, 3] == pytest.approx(3.0)
    assert res_b.l2_cap_usd[0, 5] > res_a.l2_cap_usd[0, 5]

    missing_vol_alloc = cr.allocate_l3_market_step(
        0,
        (
            cr.SeatMarketIntent(seat_id="SEAT_EXIT", held_qty=-5.0, raw_desired_qty=0.0),
            cr.SeatMarketIntent(seat_id="SEAT_OPEN", held_qty=0.0, raw_desired_qty=4.0),
        ),
        exec_price_buy=100.0,
        exec_price_sell=100.0,
        mark_price=100.0,
        l2_cap_usd=0.0,
        l2_prior_vol_valid=False,
        actual_bar_vol_1m_usd=np.nan,
        cross_zero_permitted=True,
        protective_exit_haircut_bps=15.0,
    )
    assert missing_vol_alloc.seat_post_qty["SEAT_EXIT"] == pytest.approx(0.0)
    assert missing_vol_alloc.seat_dispositions["SEAT_EXIT"] == cr.OrderDisposition.EXECUTED
    assert missing_vol_alloc.seat_post_qty["SEAT_OPEN"] == pytest.approx(0.0)
    assert missing_vol_alloc.seat_dispositions["SEAT_OPEN"] == cr.OrderDisposition.MISSED
    assert missing_vol_alloc.extra_haircut_bps == pytest.approx(15.0)
    assert np.isnan(missing_vol_alloc.realized_participation_1m)

    prorata_alloc = cr.allocate_l3_market_step(
        0,
        (
            cr.SeatMarketIntent(seat_id="SEAT_EXIT", held_qty=-2.0, raw_desired_qty=0.0),
            cr.SeatMarketIntent(seat_id="SEAT_A", held_qty=0.0, raw_desired_qty=6.0),
            cr.SeatMarketIntent(seat_id="SEAT_B", held_qty=0.0, raw_desired_qty=2.0),
        ),
        exec_price_buy=100.0,
        exec_price_sell=100.0,
        mark_price=100.0,
        l2_cap_usd=600.0,
        l2_prior_vol_valid=True,
        actual_bar_vol_1m_usd=12_000.0,
        cross_zero_permitted=True,
        protective_exit_haircut_bps=15.0,
    )
    assert prorata_alloc.seat_post_qty["SEAT_EXIT"] == pytest.approx(0.0)
    assert prorata_alloc.seat_post_qty["SEAT_A"] == pytest.approx(3.0)
    assert prorata_alloc.seat_post_qty["SEAT_B"] == pytest.approx(1.0)
    assert prorata_alloc.realized_participation_1m == pytest.approx(600.0 / 12_000.0)


def test_seat_flatten_renets_and_enforces_account_symbol_cap_against_dehedging() -> None:
    gate = prp.apply_renetting_safety_gate(
        {
            "SEAT_LONG": np.array([180.0, 200.0], dtype=np.float64),
            "SEAT_HEDGE": np.array([0.0, 0.0], dtype=np.float64),
        },
        np.array([100.0, 100.0], dtype=np.float64),
        account_equity_usd=10_000.0,
    )
    assert gate.symbol_cap_usd == pytest.approx(15_000.0)
    assert gate.gross_cap_usd == pytest.approx(30_000.0)
    assert 0 in gate.symbol_clamped_coins
    assert 1 in gate.symbol_clamped_coins
    assert gate.net_desired_qty[0] * 100.0 == pytest.approx(15_000.0)
    assert np.sum(np.abs(gate.net_desired_qty) * 100.0) == pytest.approx(30_000.0)

    n_bars = 6
    grid = _grid_ms(n_bars)
    px = np.full((1, n_bars), 100.0, dtype=np.float64)
    vol = np.full((1, n_bars), 10_000_000.0, dtype=np.float64)

    pos_long = np.array([[0.0, 0.0, 0.0, 180.0, 180.0, 180.0]], dtype=np.float64)
    pos_hedge = np.array([[0.0, 0.0, 0.0, -100.0, -100.0, -100.0]], dtype=np.float64)
    av_hedge = np.array([5000.0, 5000.0, 5000.0, 5000.0, 750.0, 750.0], dtype=np.float64)
    av_ages = np.zeros(n_bars, dtype=np.float64)

    specs = (
        prp.SeatRiskSpec(seat_id="SEAT_LONG", tier=1, k_scale=1.0, tier_seat_count_m=2),
        prp.SeatRiskSpec(seat_id="SEAT_HEDGE", tier=1, k_scale=1.0, tier_seat_count_m=2),
    )
    streams = {
        "SEAT_LONG": prp.SeatTelemetryStream(seat_id="SEAT_LONG", source_positions=pos_long),
        "SEAT_HEDGE": prp.SeatTelemetryStream(
            seat_id="SEAT_HEDGE",
            source_positions=pos_hedge,
            source_av_usd=av_hedge,
            source_av_ages_s=av_ages,
        ),
    }

    res = prp.run_portfolio_policy_replay(
        seat_specs=specs,
        seat_streams=streams,
        coins=("BTC",),
        trade_open=px,
        mark_close=px,
        grid_ms=grid,
        vol_1m_usd=vol,
        policy_cfg=prp.PortfolioPolicyConfig(
            initial_balance_usd=10_000.0,
            cwsa=cr.CWSAReplayConfig(fee_bps=0.0, slippage_bps=0.0),
            liquidity=cr.CausalLiquidityConfig(l2_lookback_bars=3, l2_min_valid_bars=3),
        ),
    )

    assert res.net_positions[0, 3] == pytest.approx(80.0)
    assert res.seat_final_states["SEAT_HEDGE"].state == prp.SeatState.RETIRED
    assert res.seat_positions["SEAT_HEDGE"][0, 4] == pytest.approx(0.0)
    assert res.net_positions[0, 4] * 100.0 <= 15_000.0 + 1e-6
    assert res.net_positions[0, 4] == pytest.approx(150.0)
    assert res.renetting_clamped_bars >= 1
    assert res.max_pm03_conservation_err < 1e-10


def test_retirement_latch_is_irreversible_even_if_subsequent_shadow_episode_wins() -> None:
    with pytest.raises(ValueError, match="INVALID_SEAT_RISK_UNIT"):
        prp.SeatRiskSpec(seat_id="SEAT_BAD_R", tier=3, k_scale=0.05, s_i=0.0, d_tier=300.0)
    with pytest.raises(ValueError, match="INVALID_SEAT_RISK_UNIT"):
        prp.SeatRiskSpec(seat_id="SEAT_BAD_K", tier=1, k_scale=-0.01, tier_seat_count_m=3)

    baseline = pc.TraderEpisodeBaseline(
        trader_id=1,
        alias="SEAT_DECAY",
        mean_ep_return=0.02,
        std_ep_return=0.01,
        median_ln_peak_notional=math.log(0.10),
        mad_ln_peak_notional=0.20,
        mean_win_ln_duration=2.0,
        std_win_ln_duration=0.5,
        symbol_notional_dist=np.array([0.5, 0.5], dtype=np.float64),
        mae_p95=0.03,
        qualified=True,
    )
    spec = prp.SeatRiskSpec(
        seat_id="SEAT_DECAY",
        tier=2,
        k_scale=0.05,
        s_i=0.30,
        d_tier=300.0,
        win_hold_p95_bars=None,
        mae_p95_r=0.03,
        episode_baseline=baseline,
    )
    assert spec.activation_r_usd == pytest.approx(90.0)
    st = prp.SeatContainmentState(spec=spec)

    grid = _grid_ms(320)
    for idx in range(305):
        prp.step_fast_seat_clock(
            st,
            bar_ts_ms=int(grid[idx]),
            cum_pnl_usd=-20.0,
            open_upnl_usd=-20.0,
            has_open_pos=True,
            source_av_usd=500.0,
            source_av_age_s=900.0,
        )
    assert st.source_av_status == prp.TriggerEvalStatus.UNKNOWN
    assert st.underwater_status == prp.TriggerEvalStatus.UNKNOWN
    assert st.state == prp.SeatState.ACTIVE

    prp.step_slow_episode_clock(
        st,
        arm=prp.LifecycleArm.RESEARCH_LEADER_REDUCE_FREEZE,
        ep_return=-0.04,
        ep_mae=0.02,
        ep_mae_r=0.02,
        ep_symbol_notional=np.array([0.5, 0.5], dtype=np.float64),
        bar_ts_ms=int(grid[306]),
    )
    assert st.state == prp.SeatState.RETIRED
    assert st.roster_cusum is not None
    assert st.roster_cusum.status == "KILL_CONTAIN_0X"

    for step_i in range(4):
        prp.step_slow_episode_clock(
            st,
            arm=prp.LifecycleArm.RESEARCH_LEADER_REDUCE_FREEZE,
            ep_return=0.12,
            ep_mae=0.001,
            ep_mae_r=0.001,
            ep_symbol_notional=np.array([0.5, 0.5], dtype=np.float64),
            bar_ts_ms=int(grid[307 + step_i]),
        )
    assert st.cusum_pos == pytest.approx(0.0)
    assert st.roster_cusum.status == "KILL_CONTAIN_0X"
    assert st.roster_cusum.terminal_kill_latched is True
    assert st.state == prp.SeatState.RETIRED
    assert st.current_k_scale == pytest.approx(0.0)

    st.transition_to(prp.SeatState.RETIRED, "DUPLICATE_RETIRE_SIGNAL")
    assert st.state == prp.SeatState.RETIRED

    with pytest.raises(ValueError, match="ILLEGAL_RETIREMENT_RESURRECTION"):
        st.transition_to(prp.SeatState.ACTIVE, "ATTEMPT_UNFREEZE")
    with pytest.raises(ValueError, match="ILLEGAL_RETIREMENT_RESURRECTION"):
        st.transition_to(prp.SeatState.FREEZE_NEW_RISK, "ATTEMPT_DOWNGRADE")


def test_four_containment_arms_separate_guardian_static_pause_from_research_leader_reduce() -> None:
    n_bars = 12
    grid = _grid_ms(n_bars)
    vol = np.full((2, n_bars), 5_000_000.0, dtype=np.float64)

    trade_open = np.array(
        [
            [100.0, 100.0, 100.0, 100.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
            [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 52.0, 52.0, 52.0, 52.0],
        ],
        dtype=np.float64,
    )
    mark_close = np.array(
        [
            [100.0, 100.0, 100.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
            [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 52.0, 52.0, 52.0, 54.0, 54.0],
        ],
        dtype=np.float64,
    )
    src_pos = np.array(
        [
            [0.0, 0.0, 0.0, 100.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.0, 10.0, 0.0, 5.0, 8.0, 0.0],
        ],
        dtype=np.float64,
    )

    spec = prp.SeatRiskSpec(
        seat_id="SEAT_A",
        tier=1,
        k_scale=1.0,
        tier_seat_count_m=2,
        initial_balance_usd=10_000.0,
    )
    assert spec.activation_r_usd == pytest.approx(100.0)

    by_arm: dict[prp.LifecycleArm, prp.PortfolioPolicyResult] = {}
    for arm in prp.LifecycleArm:
        by_arm[arm] = prp.run_portfolio_policy_replay(
            seat_specs=(spec,),
            seat_streams={"SEAT_A": src_pos},
            coins=("BTC", "ETH"),
            trade_open=trade_open,
            mark_close=mark_close,
            grid_ms=grid,
            vol_1m_usd=vol,
            policy_cfg=prp.PortfolioPolicyConfig(
                initial_balance_usd=10_000.0,
                lifecycle_arm=arm,
                cwsa=cr.CWSAReplayConfig(fee_bps=0.0, slippage_bps=0.0),
                liquidity=cr.CausalLiquidityConfig(l2_lookback_bars=3, l2_min_valid_bars=3),
            ),
        )
        assert by_arm[arm].max_pm03_conservation_err < 1e-10

    ep_only = by_arm[prp.LifecycleArm.EPISODE_CUSUM_ONLY]
    static_pause = by_arm[prp.LifecycleArm.GUARDIAN_MANDATE_PAUSE_STATIC_HOLD]
    leader_red = by_arm[prp.LifecycleArm.RESEARCH_LEADER_REDUCE_FREEZE]
    next_flat = by_arm[prp.LifecycleArm.NEXT_FLAT_RELEASE]

    assert ep_only.seat_state_history["SEAT_A"][3] == prp.SeatState.ACTIVE.value
    assert ep_only.seat_positions["SEAT_A"][1, 7] == pytest.approx(10.0)

    assert static_pause.seat_state_history["SEAT_A"][3] == prp.SeatState.FREEZE_NEW_RISK.value
    assert static_pause.seat_positions["SEAT_A"][0, 4] == pytest.approx(100.0)
    assert static_pause.seat_positions["SEAT_A"][0, 5] == pytest.approx(100.0)
    assert static_pause.seat_state_history["SEAT_A"][-1] == prp.SeatState.FREEZE_NEW_RISK.value

    assert leader_red.seat_state_history["SEAT_A"][3] == prp.SeatState.FREEZE_NEW_RISK.value
    assert leader_red.seat_positions["SEAT_A"][0, 4] == pytest.approx(50.0)
    assert leader_red.seat_positions["SEAT_A"][0, 5] == pytest.approx(0.0)
    assert leader_red.seat_state_history["SEAT_A"][6] == prp.SeatState.FREEZE_NEW_RISK.value
    assert leader_red.seat_positions["SEAT_A"][1, 7] == pytest.approx(0.0)
    assert leader_red.seat_state_history["SEAT_A"][8] == prp.SeatState.ACTIVE.value
    assert leader_red.seat_positions["SEAT_A"][1, 9] == pytest.approx(5.0)

    assert next_flat.seat_state_history["SEAT_A"][6] == prp.SeatState.ACTIVE.value
    assert next_flat.seat_positions["SEAT_A"][1, 7] == pytest.approx(10.0)


def test_multi_pod_attribution_conserves_cash_fees_and_matches_pm03_account() -> None:
    start_dt = datetime(2026, 7, 1, 23, 50, tzinfo=UTC)
    n_bars = 10
    grid = _grid_ms(n_bars, start=start_dt)
    vol = np.full((2, n_bars), 2_000_000.0, dtype=np.float64)

    trade_open = np.array(
        [
            [100.0, 100.0, 100.0, 100.0, 102.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            [0.010, 0.010, 0.010, 0.010, 0.010, 0.011, 0.011, 0.012, 0.012, 0.012],
        ],
        dtype=np.float64,
    )
    mark_close = np.array(
        [
            [100.0, 100.0, 100.0, 102.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
            [0.010, 0.010, 0.010, 0.010, 0.011, 0.011, 0.012, 0.012, 0.012, 0.013],
        ],
        dtype=np.float64,
    )
    funding = np.zeros((2, n_bars), dtype=np.float64)
    funding[0, 5] = 0.0001

    pos_a = np.array(
        [
            [0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0, 12.0, 12.0],
            [0.0, 0.0, 0.0, 0.0, 1000.0, 1000.0, 1000.0, 1000.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    pos_b = np.array(
        [
            [0.0, 0.0, 0.0, 4.0, -6.0, -6.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    av_b = np.array(
        [5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 800.0, 800.0, 800.0, 800.0, 800.0],
        dtype=np.float64,
    )
    av_ages = np.zeros(n_bars, dtype=np.float64)

    pos_bench_causal = np.array(
        [
            [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 25.0, 0.0, 8.0, 8.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 500.0, 500.0, 500.0],
        ],
        dtype=np.float64,
    )
    pos_bench_future = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 50.0, 50.0, 50.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    specs = (
        prp.SeatRiskSpec(
            seat_id="SEAT_A",
            tier=1,
            k_scale=0.50,
            max_k_scale=0.75,
            tier_seat_count_m=2,
        ),
        prp.SeatRiskSpec(
            seat_id="SEAT_B",
            tier=1,
            k_scale=0.50,
            tier_seat_count_m=2,
        ),
    )
    streams = {
        "SEAT_A": prp.SeatTelemetryStream(seat_id="SEAT_A", source_positions=pos_a),
        "SEAT_B": prp.SeatTelemetryStream(
            seat_id="SEAT_B",
            source_positions=pos_b,
            source_av_usd=av_b,
            source_av_ages_s=av_ages,
        ),
        "SEAT_BENCH_CAUSAL": prp.SeatTelemetryStream(
            seat_id="SEAT_BENCH_CAUSAL",
            source_positions=pos_bench_causal,
        ),
        "SEAT_BENCH_FUTURE": prp.SeatTelemetryStream(
            seat_id="SEAT_BENCH_FUTURE",
            source_positions=pos_bench_future,
        ),
    }
    bench_snaps = (
        prp.PITBenchSnapshot(
            asof_ts_ms=int(grid[2]),
            candidates=(
                prp.PITBenchCandidate(
                    spec=prp.SeatRiskSpec(
                        seat_id="SEAT_BENCH_CAUSAL",
                        tier=1,
                        k_scale=0.50,
                        tier_seat_count_m=2,
                    ),
                    score=1.25,
                ),
            ),
        ),
        prp.PITBenchSnapshot(
            asof_ts_ms=int(grid[-1]) + 900_000,
            candidates=(
                prp.PITBenchCandidate(
                    spec=prp.SeatRiskSpec(
                        seat_id="SEAT_BENCH_FUTURE",
                        tier=1,
                        k_scale=0.50,
                        tier_seat_count_m=2,
                    ),
                    score=99.0,
                ),
            ),
        ),
    )

    base_cfg = prp.PortfolioPolicyConfig(
        initial_balance_usd=10_000.0,
        lifecycle_arm=prp.LifecycleArm.RESEARCH_LEADER_REDUCE_FREEZE,
        cwsa=cr.CWSAReplayConfig(half_spread_bps=1.0, fee_bps=4.5, slippage_bps=2.0),
        liquidity=cr.CausalLiquidityConfig(l2_lookback_bars=3, l2_min_valid_bars=3),
    )

    cross_test_ledger = prp.ConsolidatedPodLedger(
        ("BTC",),
        ("SEAT_BUY", "SEAT_SELL"),
        int(grid[0]),
        initial_balance_usd=10_000.0,
        cwsa_cfg=cr.CWSAReplayConfig(half_spread_bps=2.0, fee_bps=4.5, slippage_bps=2.0),
    )
    cross_alloc = cr.allocate_l3_market_step(
        0,
        (
            cr.SeatMarketIntent(seat_id="SEAT_BUY", held_qty=0.0, raw_desired_qty=5.0),
            cr.SeatMarketIntent(seat_id="SEAT_SELL", held_qty=0.0, raw_desired_qty=-5.0),
        ),
        exec_price_buy=100.02,
        exec_price_sell=99.98,
        mark_price=100.0,
        l2_cap_usd=0.0,
        l2_prior_vol_valid=False,
        actual_bar_vol_1m_usd=1_000.0,
    )
    assert cross_alloc.internal_crosses == (("SEAT_BUY", "SEAT_SELL", 5.0),)
    assert cross_alloc.external_delta_qty == pytest.approx(0.0)
    bar_err, _ = cross_test_ledger.step_bar(
        int(grid[0]),
        allocations={0: cross_alloc},
        raw_intended_by_seat={
            "SEAT_BUY": np.array([5.0], dtype=np.float64),
            "SEAT_SELL": np.array([-5.0], dtype=np.float64),
        },
        decision_mid_prices=np.array([100.0], dtype=np.float64),
        close_mark_prices=np.array([105.0], dtype=np.float64),
    )
    assert bar_err < 1e-12
    assert cross_test_ledger.account.costs == D(0)
    assert cross_test_ledger.seat_cum_pnl_usd("SEAT_BUY") == pytest.approx(25.0)
    assert cross_test_ledger.seat_cum_pnl_usd("SEAT_SELL") == pytest.approx(-25.0)
    assert cross_test_ledger.equity_usd == pytest.approx(10_000.0)

    cmp = prp.compare_seat_replacement_modes(
        seat_specs=specs,
        seat_streams=streams,
        bench_snapshots=bench_snaps,
        coins=("BTC", "kPEPE"),
        trade_open=trade_open,
        mark_close=mark_close,
        grid_ms=grid,
        vol_1m_usd=vol,
        funding_rates=funding,
        base_policy_cfg=base_cfg,
    )

    for mode, m_res in cmp.results_by_mode.items():
        assert m_res.max_pm03_conservation_err < 1e-10, f"conservation drift in {mode}"
        acct = m_res.ledger.account
        acct.verify()
        econ = m_res.economics
        assert isinstance(econ, dict)
        assert abs(D(econ["reconciliation_rounding_residual"])) <= D("0.000000000001")
        assert sum(acct.pm_costs.values(), D(0)) == acct.costs
        assert sum(acct.pm_funding.values(), D(0)) == acct.funding
        assert abs(
            sum(acct.pm_equity_pnl().values(), D(0)) - (acct.equity - acct.rulebook.initial_balance)
        ) <= D("0.000000000001")

    unused_res = cmp.results_by_mode[prp.ReplacementMode.LEAVE_RISK_UNUSED]
    resize_res = cmp.results_by_mode[prp.ReplacementMode.RESIZE_HEALTHY_SEATS]
    bench_res = cmp.results_by_mode[prp.ReplacementMode.PROMOTE_PIT_TOP_BENCH_NEW_POS_ONLY]

    assert unused_res.promoted_bench_seats == ()
    assert len(unused_res.resized_seat_events) == 0

    assert len(resize_res.resized_seat_events) == 1
    assert resize_res.seat_positions["SEAT_A"][0, 6] == pytest.approx(
        unused_res.seat_positions["SEAT_A"][0, 6]
    )
    assert resize_res.seat_positions["SEAT_A"][0, 8] > unused_res.seat_positions["SEAT_A"][0, 8]

    assert bench_res.promoted_bench_seats == ("SEAT_BENCH_CAUSAL",)
    assert "SEAT_BENCH_FUTURE" not in bench_res.promoted_bench_seats
    assert bench_res.seat_positions["SEAT_BENCH_CAUSAL"][0, 6] == pytest.approx(0.0)
    assert bench_res.seat_positions["SEAT_BENCH_CAUSAL"][0, 7] == pytest.approx(0.0)
    assert bench_res.seat_positions["SEAT_BENCH_CAUSAL"][0, 8] > 0.0
    assert bench_res.seat_positions["SEAT_BENCH_CAUSAL"][1, 7] > 0.0


def test_interrogate_edge_regressions() -> None:
    g_gain = prp.CFTGuardianTracker(initial_balance_usd=10_000.0, daily_sweep_usd=400.0)
    assert not g_gain.step(1_782_864_000_000, balance_usd=10_000.0, equity_usd=10_450.0)
    g_loss = prp.CFTGuardianTracker(initial_balance_usd=10_000.0, daily_sweep_usd=400.0)
    assert g_loss.step(1_782_864_000_000, balance_usd=10_000.0, equity_usd=9_550.0)

    t0 = round(datetime(2026, 7, 1, 23, 55, tzinfo=UTC).timestamp() * 1000.0)
    grid_r = _grid_ms(3, start=datetime(2026, 7, 1, 23, 55, tzinfo=UTC))
    spec_r = prp.SeatRiskSpec(seat_id="S1", tier=1, k_scale=1.0, tier_seat_count_m=1)
    res_r = prp.run_portfolio_policy_replay(
        seat_specs=(spec_r,),
        seat_streams={"S1": np.array([[0.0, 10.0, 0.0]], dtype=np.float64)},
        coins=("BTC",),
        trade_open=np.array([[100.0, 100.0, 50.0]], dtype=np.float64),
        mark_close=np.array([[100.0, 100.0, 50.0]], dtype=np.float64),
        grid_ms=grid_r,
        vol_1m_usd=np.full((1, 3), 1e7, dtype=np.float64),
        policy_cfg=prp.PortfolioPolicyConfig(
            cwsa=cr.CWSAReplayConfig(fee_bps=0.0, slippage_bps=0.0),
            liquidity=cr.CausalLiquidityConfig(l2_lookback_bars=1, l2_min_valid_bars=1),
        ),
    )
    assert res_r.ledger.guardian.daily_reset_balance_usd == pytest.approx(10_000.0)
    assert res_r.guardian_breached is True

    base = pc.TraderEpisodeBaseline(
        trader_id=1,
        alias="S1",
        mean_ep_return=0.0005,
        std_ep_return=0.0005,
        median_ln_peak_notional=math.log(0.02),
        mad_ln_peak_notional=0.2,
        mean_win_ln_duration=2.0,
        std_win_ln_duration=0.5,
        symbol_notional_dist=np.array([1.0], dtype=np.float64),
        mae_p95=0.001,
        qualified=True,
    )
    spec_b = prp.SeatRiskSpec(
        seat_id="S1", tier=1, k_scale=1.0, tier_seat_count_m=1, mae_p95_r=0.001, episode_baseline=base
    )
    res_b = prp.run_portfolio_policy_replay(
        seat_specs=(spec_b,),
        seat_streams={"S1": np.array([[0.0, 2.0, 2.0, 0.0, 0.0]], dtype=np.float64)},
        coins=("BTC",),
        trade_open=np.array([[100.0, 100.0, 100.0, 99.0, 99.0]], dtype=np.float64),
        mark_close=np.array([[100.0, 100.0, 98.0, 99.0, 99.0]], dtype=np.float64),
        grid_ms=_grid_ms(5),
        vol_1m_usd=np.full((1, 5), 1e7, dtype=np.float64),
        policy_cfg=prp.PortfolioPolicyConfig(
            liquidity=cr.CausalLiquidityConfig(l2_lookback_bars=1, l2_min_valid_bars=1)
        ),
    )
    assert res_b.seat_final_states["S1"].state == prp.SeatState.ACTIVE

    a_exit = cr.allocate_l3_market_step(
        0,
        (
            cr.SeatMarketIntent("SEAT_A_ENTRY", 0.0, 4.0),
            cr.SeatMarketIntent("SEAT_B_EXIT", -5.0, 0.0),
            cr.SeatMarketIntent("SEAT_C_EXIT", 3.0, 0.0),
        ),
        exec_price_buy=100.0,
        exec_price_sell=100.0,
        mark_price=100.0,
        l2_cap_usd=0.0,
        l2_prior_vol_valid=False,
        actual_bar_vol_1m_usd=np.nan,
    )
    assert a_exit.seat_post_qty["SEAT_A_ENTRY"] == pytest.approx(0.0)
    assert a_exit.external_delta_qty == pytest.approx(2.0)

    a_small = cr.allocate_l3_market_step(
        0,
        (cr.SeatMarketIntent("S0", 5e-7, -5e-7),),
        exec_price_buy=100.0,
        exec_price_sell=100.0,
        mark_price=100.0,
        l2_cap_usd=0.0,
        l2_prior_vol_valid=False,
        actual_bar_vol_1m_usd=np.nan,
        cross_zero_permitted=False,
    )
    assert a_small.projection_mode == cr.MarketProjectionMode.REDUCE_ONLY_FALLBACK
    assert a_small.seat_post_qty["S0"] == pytest.approx(0.0)

    px = np.full((1, 6), 100.0, dtype=np.float64)
    res_p = prp.run_portfolio_policy_replay(
        seat_specs=(prp.SeatRiskSpec(seat_id="MAIN", tier=1, k_scale=1.0, tier_seat_count_m=1),),
        seat_streams={
            "MAIN": prp.SeatTelemetryStream(
                seat_id="MAIN",
                source_positions=np.array([[0.0, 10.0, 10.0, 10.0, 10.0, 10.0]]),
                source_av_usd=np.array([5000.0, 5000.0, 500.0, 5000.0, 5000.0, 5000.0]),
                source_av_ages_s=np.zeros(6),
            ),
            "BENCH": prp.SeatTelemetryStream(
                seat_id="BENCH",
                source_positions=np.array([[0.0, 0.0, 0.0, 5.0, 5.0, 5.0]]),
            ),
        },
        bench_snapshots=(
            prp.PITBenchSnapshot(
                asof_ts_ms=int(t0 * 0),
                candidates=(
                    prp.PITBenchCandidate(
                        spec=prp.SeatRiskSpec(seat_id="BENCH", tier=1, k_scale=1.0, tier_seat_count_m=1),
                        score=1.0,
                    ),
                ),
            ),
        ),
        coins=("BTC",),
        trade_open=px,
        mark_close=px,
        grid_ms=_grid_ms(6),
        vol_1m_usd=np.full((1, 6), 1e8, dtype=np.float64),
        policy_cfg=prp.PortfolioPolicyConfig(
            replacement_mode=prp.ReplacementMode.PROMOTE_PIT_TOP_BENCH_NEW_POS_ONLY,
            cwsa=cr.CWSAReplayConfig(fee_bps=0.0, slippage_bps=0.0),
            liquidity=cr.CausalLiquidityConfig(l2_lookback_bars=1, l2_min_valid_bars=1),
        ),
    )
    assert res_p.promoted_bench_seats == ("BENCH",)
    assert res_p.seat_positions["BENCH"][0, 3] == pytest.approx(5.0)
