"""Historical Jul-Oct fixed-k 50%-of-allocated-risk regression checks."""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal as D
from pathlib import Path

import pytest

STUDIES = Path(__file__).resolve().parents[1] / "studies"
SPEC = importlib.util.spec_from_file_location(
    "historical_5m_wallet_risk", STUDIES / "historical_5m_wallet_risk.py"
)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_requested_window_and_train_only_cutoff():
    assert MOD.START == datetime(2026, 7, 1, tzinfo=UTC)
    assert MOD.TRAIN_END == datetime(2026, 10, 1, tzinfo=UTC)
    assert MOD.END == datetime(2026, 11, 1, tzinfo=UTC)
    assert MOD.END > MOD.TRAIN_END > MOD.START


def test_each_wallet_risk_allocation_and_50pct_target():
    allocation = MOD.risk_allocation(static_loss=D(500), n=3)
    assert allocation == D(500) / 3
    target = MOD.wallet_drawdown_target(allocation)
    assert abs(target - D(250) / 3) < D("0.000000000000000000000001")
    assert target < D(300) / 3


@pytest.mark.parametrize(
    ("unit_dd", "expected"),
    [(D(1000), D("0.083333333333")), (D(10), D("8.333333333333")), (D(1), D(10))],
)
def test_unit_drawdown_formula_without_arbitrary_small_k_grid(unit_dd, expected):
    allocation = D(500) / 3
    k = MOD.k_for_wallet_drawdown(unit_dd, allocation)
    assert k == expected
    assert k <= 10
    if expected < 10:
        assert unit_dd * k <= allocation / 2
        assert allocation / 2 - unit_dd * k < D("0.000001")


def test_no_drawdown_or_missing_data_do_not_generate_infinite_k():
    assert MOD.k_for_wallet_drawdown(None, D(100)) is None
    assert MOD.k_for_wallet_drawdown(D(0), D(100)) is None
    with pytest.raises(ValueError, match="NEGATIVE_MAX_DRAWDOWN"):
        MOD.k_for_wallet_drawdown(D(-1), D(100))
    with pytest.raises(ValueError, match="INVALID_RISK_BUDGET"):
        MOD.k_for_wallet_drawdown(D(10), D(0))


def test_portfolio_aggregate_checks_equity_and_daily_floor():
    start = datetime(2026, 7, 1, tzinfo=UTC)
    train = start + timedelta(minutes=2)
    end = start + timedelta(minutes=4)
    members = ("A", "B", "C")
    curves = {
        "A": [0.0, -100.0, 50.0, -200.0],
        "B": [0.0, 0.0, 0.0, 0.0],
        "C": [0.0, 0.0, 0.0, 0.0],
    }
    k = {"A": D("0.5"), "B": D("0.5"), "C": D("0.5")}
    result = MOD.aggregate_sized_portfolio(
        members, k, curves, start=start, train_end=train, end=end
    )
    assert result["status"] == "COMPLETE_RETROSPECTIVE_SCENARIO"
    assert result["train_net_usd"] == -50
    assert result["october_chronological_net_usd"] == -50
    assert result["max_peak_drawdown_usd"] == 125
    assert result["minimum_equity_usd"] == 9900
    assert result["first_equity_floor_breach_utc"] is None
    assert result["not_independent_portfolio_selection_oos"] is True


def test_incomplete_common_holdout_is_not_labeled_full():
    start = datetime(2026, 7, 1, tzinfo=UTC)
    train = start + timedelta(minutes=2)
    end = start + timedelta(minutes=5)
    members = ("A", "B", "C")
    k = dict.fromkeys(members, D(1))
    curves = {"A": [0, 5, 10], "B": [0, 5, 10, 20, 30], "C": [0, 5, 10, 20]}
    result = MOD.aggregate_sized_portfolio(
        members, k, curves, start=start, train_end=train, end=end
    )
    assert result["status"] == "PARTIAL_RETROSPECTIVE_SCENARIO"
    assert result["common_marked_minutes"] == 3
    assert result["complete_requested_window"] is False


def test_order_size_is_source_notional_times_k_not_leader_fills():
    from copytrade_pm.pm_exposure_sizing import target_from_fixed_k

    key = "hyperliquid/mainnet:perp:default/BTC"
    intent = target_from_fixed_k(
        source_quantities={key: D(3)},
        k=D("0.04"),
        source_marks={key: D(100)},
        follower_marks={key: D(95)},
        follower_equity_usd=D(10000),
        max_gross_notional_usd=D(2000),
        max_market_notional_usd=D(2000),
        pod_loss_budget_usd=D(166),
    )
    assert D(intent["target_notionals_usd"][key]) == D(12)
    assert D(intent["target_quantities"][key]) == D(12) / 95
    assert intent["order_route"] == "NONE"
    assert intent["trading_authorized"] is False
