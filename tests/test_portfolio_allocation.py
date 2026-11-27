"""Tests for the decoupled portfolio-construction allocation seam."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from copytrade_kernel import KnownAtTs
from copytrade_pm.roster import GateFailure
from research_fixtures import (
    DAY_MS,
    START,
    WALLET_A,
    WALLET_B,
    bars_1m_from_fills,
    build_store,
    wallet_round_trips,
)

from trading_ops.research import (
    AllocationPolicy,
    ScoreConfig,
    allocate_fixed_k_portfolio,
    score_universe,
)

BARS_3D = 3 * 288


def test_decouple_sizing_gates_keeps_wallets_eligible_and_allocates_across_policies(
    tmp_path: Path,
) -> None:
    fills = wallet_round_trips(
        WALLET_A, "BTC", start_ms=START, bars=BARS_3D, base_price=20000.0, size=0.5
    )
    fills += wallet_round_trips(
        WALLET_B, "ETH", start_ms=START, bars=BARS_3D, base_price=2000.0, size=2.0
    )
    thin_bars = bars_1m_from_fills(fills, quote_volume_usd=Decimal(50000))
    store = build_store(
        tmp_path,
        fills=fills,
        equity={WALLET_A: (Decimal(12000), 4), WALLET_B: (Decimal(9000), 4)},
        bars_1m_rows=thin_bars,
    )
    try:
        coupled = score_universe(
            store,
            as_of=KnownAtTs(START + 3 * DAY_MS),
            lookback_days=3,
            config=ScoreConfig(
                min_baseline_order_usd=Decimal(1),
                max_euler_cdar_share=Decimal("0.10"),
                max_p95_participation_1m=Decimal("0.005"),
                max_max_participation_1m=Decimal("0.01"),
                decouple_sizing_gates=False,
            ),
            allow_undated=True,
        )
        decoupled = score_universe(
            store,
            as_of=KnownAtTs(START + 3 * DAY_MS),
            lookback_days=3,
            config=ScoreConfig(
                min_baseline_order_usd=Decimal(1),
                max_euler_cdar_share=Decimal("0.10"),
                max_p95_participation_1m=Decimal("0.005"),
                max_max_participation_1m=Decimal("0.01"),
                decouple_sizing_gates=True,
            ),
            allow_undated=True,
        )
    finally:
        store.close()

    # Coupled scoring rejects wallets under arbitrary initial sizing thresholds.
    assert any(not card.eligible for card in coupled.cards)
    # Decoupled scoring retains intrinsic eligibility without rejecting on participation/CDaR.
    assert decoupled.basis["sizing_gates"] == "DECOUPLED_PORTFOLIO_SEAM"
    assert len(decoupled.eligible) == 2
    for card in decoupled.cards:
        assert GateFailure.PARTICIPATION_LIMIT not in card.failures
        assert GateFailure.MARGINAL_CDAR_LIMIT not in card.failures

    # Without re-scanning FactStore, two AllocationPolicy budgets yield distinct k and constraints.
    conservative = allocate_fixed_k_portfolio(
        decoupled,
        policy=AllocationPolicy(
            risk_unit_usd=Decimal(100),
            max_fixed_k=Decimal("0.5"),
            max_euler_cdar_share=Decimal("0.90"),
            max_p95_participation=Decimal("0.50"),
            max_max_participation=Decimal("0.50"),
        ),
    )
    aggressive = allocate_fixed_k_portfolio(
        decoupled,
        policy=AllocationPolicy(
            risk_unit_usd=Decimal(1000),
            max_fixed_k=Decimal("5.0"),
            max_euler_cdar_share=Decimal("0.90"),
            max_p95_participation=Decimal("0.05"),
            max_max_participation=Decimal("0.10"),
        ),
    )

    cons_by_w = {row.wallet: row for row in conservative.allocations}
    agg_by_w = {row.wallet: row for row in aggressive.allocations}
    assert agg_by_w[WALLET_A].k > cons_by_w[WALLET_A].k
    assert agg_by_w[WALLET_A].p95_participation > cons_by_w[WALLET_A].p95_participation
    assert conservative.eligible_wallets == (WALLET_A, WALLET_B)
    assert GateFailure.PARTICIPATION_LIMIT in agg_by_w[WALLET_A].failures

    bench = conservative.to_bench_candidates(decoupled)
    assert len(bench) == 2
    assert {item.wallet: item.suggested_k for item in bench} == conservative.k_by_wallet


def test_allocate_fixed_k_portfolio_fails_closed_without_unscaled_participation(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    import pytest

    fills = wallet_round_trips(
        WALLET_A, "BTC", start_ms=START, bars=BARS_3D, base_price=20000.0, size=0.5
    )
    store = build_store(
        tmp_path,
        fills=fills,
        equity={WALLET_A: (Decimal(12000), 4)},
        bars_1m_rows=bars_1m_from_fills(fills, quote_volume_usd=Decimal(50000)),
    )
    try:
        card = score_universe(
            store,
            as_of=KnownAtTs(START + 3 * DAY_MS),
            lookback_days=3,
            config=ScoreConfig(min_baseline_order_usd=Decimal(1), decouple_sizing_gates=True),
            allow_undated=True,
        )
    finally:
        store.close()

    stripped = replace(card, _unscaled_participation_by_wallet={})
    with pytest.raises(ValueError, match=f"SCORECARD_MISSING_PARTICIPATION:{WALLET_A}"):
        allocate_fixed_k_portfolio(stripped)
