from __future__ import annotations

import importlib
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

STUDIES_DIR = Path(__file__).resolve().parents[1] / "studies"
if str(STUDIES_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIES_DIR))

cr = importlib.import_module("canonical_replay")
prp = importlib.import_module("portfolio_risk_policy")


def _grid_ms(n_bars: int) -> np.ndarray:
    t0 = int(datetime(2026, 9, 1, 0, 0, tzinfo=UTC).timestamp() * 1000)
    return t0 + np.arange(n_bars, dtype=np.int64) * 300_000


def test_cwsa_replay_defaults_to_uncapped_execution() -> None:
    """Default CWSA replay skips L1-L3 capping even when order exceeds 5% of 1m volume."""
    src_pos = np.array([[0.0, 0.0, 0.0, 10.0, 10.0, 10.0]], dtype=np.float64)
    px = np.full((1, 6), 100.0, dtype=np.float64)
    # 5% of $6,000 is $300 (3.0 qty at $100), so 10.0 qty ($1,000) is 16.67% > 5%.
    vol = np.full((1, 6), 6_000.0, dtype=np.float64)
    liq_cfg = cr.CausalLiquidityConfig(
        l2_lookback_bars=3,
        l2_min_valid_bars=3,
        max_participation_1m=0.05,
    )
    assert liq_cfg.apply_l3_liquidity_cap is False

    res_uncapped = cr.simulate_cwsa_one_shot_replay(
        coins=("BTC",),
        source_positions=src_pos,
        trade_open=px,
        mark_close=px,
        grid_ms=_grid_ms(6),
        k_scale=1.0,
        vol_1m_usd=vol,
        liquidity_cfg=liq_cfg,
    )
    assert res_uncapped.executed_positions[0, 3] == pytest.approx(10.0)
    assert res_uncapped.participation_audit.clipped_order_count == 0
    assert res_uncapped.participation_audit.missed_order_count == 0
    assert res_uncapped.participation_audit.bars_gt_5pct_count == 1
    assert res_uncapped.participation_audit.max_participation_1m == pytest.approx(1000.0 / 6000.0)

    # Explicit apply_l3_liquidity_cap=True preserves L1-L3 capping at 5% ($300 -> 3.0 qty).
    res_capped = cr.simulate_cwsa_one_shot_replay(
        coins=("BTC",),
        source_positions=src_pos,
        trade_open=px,
        mark_close=px,
        grid_ms=_grid_ms(6),
        k_scale=1.0,
        vol_1m_usd=vol,
        liquidity_cfg=liq_cfg,
        apply_l3_liquidity_cap=True,
    )
    assert res_capped.executed_positions[0, 3] == pytest.approx(3.0)
    assert res_capped.l2_cap_usd[0, 3] == pytest.approx(300.0)
    assert res_capped.participation_audit.clipped_order_count == 1


def test_vectorized_cwsa_mode_respects_uncapped_default_and_explicit_cap() -> None:
    """replay_coin_targets_vectorized(mode='cwsa_one_shot_v3') is uncapped by default."""
    targets = np.array([0.0, 0.0, 0.0, 10.0, 10.0], dtype=np.float64)
    px = np.full(5, 100.0, dtype=np.float64)
    vol = np.full(5, 6_000.0, dtype=np.float64)
    liq_cfg = cr.CausalLiquidityConfig(l2_lookback_bars=3, l2_min_valid_bars=3)

    pos_default, _, _, _, info_default = cr.replay_coin_targets_vectorized(
        targets,
        px,
        px,
        mode="cwsa_one_shot_v3",
        vol_1m_usd=vol,
        liquidity_cfg=liq_cfg,
    )
    assert pos_default[3] == pytest.approx(10.0)
    assert info_default["deferred_notional_usd"] == pytest.approx(0.0)

    pos_capped, _, _, _, info_capped = cr.replay_coin_targets_vectorized(
        targets,
        px,
        px,
        mode="cwsa_one_shot_v3",
        vol_1m_usd=vol,
        liquidity_cfg=liq_cfg,
        apply_l3_liquidity_cap=True,
    )
    assert pos_capped[3] == pytest.approx(3.0)
    assert info_capped["deferred_notional_usd"] == pytest.approx(700.0)


def test_portfolio_policy_replay_defaults_uncapped_and_preserves_opt_in_cap() -> None:
    """PortfolioPolicyConfig defaults to uncapped execution and preserves apply_l3_liquidity_cap=True."""
    px = np.full((1, 5), 100.0, dtype=np.float64)
    vol = np.full((1, 5), 6_000.0, dtype=np.float64)
    spec = prp.SeatRiskSpec(seat_id="S1", tier=1, k_scale=1.0, tier_seat_count_m=1)
    streams = {"S1": np.array([[0.0, 0.0, 0.0, 10.0, 10.0]], dtype=np.float64)}
    grid = _grid_ms(5)

    default_cfg = prp.PortfolioPolicyConfig(
        cwsa=cr.CWSAReplayConfig(fee_bps=0.0, slippage_bps=0.0),
        liquidity=cr.CausalLiquidityConfig(l2_lookback_bars=3, l2_min_valid_bars=3),
    )
    assert default_cfg.apply_l3_liquidity_cap is False

    res_uncapped = prp.run_portfolio_policy_replay(
        seat_specs=(spec,),
        seat_streams=streams,
        coins=("BTC",),
        trade_open=px,
        mark_close=px,
        grid_ms=grid,
        vol_1m_usd=vol,
        policy_cfg=default_cfg,
    )
    assert res_uncapped.net_positions[0, 3] == pytest.approx(10.0)
    assert res_uncapped.participation_audit.clipped_order_count == 0
    assert res_uncapped.participation_audit.bars_gt_5pct_count == 1
    assert res_uncapped.max_pm03_conservation_err < 1e-10

    capped_cfg = prp.PortfolioPolicyConfig(
        cwsa=cr.CWSAReplayConfig(fee_bps=0.0, slippage_bps=0.0),
        liquidity=cr.CausalLiquidityConfig(l2_lookback_bars=3, l2_min_valid_bars=3),
        apply_l3_liquidity_cap=True,
    )
    res_capped = prp.run_portfolio_policy_replay(
        seat_specs=(spec,),
        seat_streams=streams,
        coins=("BTC",),
        trade_open=px,
        mark_close=px,
        grid_ms=grid,
        vol_1m_usd=vol,
        policy_cfg=capped_cfg,
    )
    assert res_capped.net_positions[0, 3] == pytest.approx(3.0)
    assert res_capped.participation_audit.clipped_order_count == 1
    assert res_capped.max_pm03_conservation_err < 1e-10
