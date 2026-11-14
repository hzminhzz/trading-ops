"""Risk-up / Guardian HALT proxy sensitivity regression tests.

The overlay is intentionally NOT production Guardian, which cannot
automatically clear a protected HALT or auto-arm after a timer.
"""

from __future__ import annotations

import importlib
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies"))
sens = importlib.import_module("guardian_k_sensitivity")


def test_k_multiplier_sweep_is_predeclared_not_optimized_from_october():
    assert sens.MULTIPLIERS == (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)
    assert sens.native.START == datetime(2026, 7, 1, tzinfo=UTC)
    assert sens.native.TRAIN_END == datetime(2026, 10, 1, tzinfo=UTC)
    assert sens.native.END == datetime(2026, 11, 1, tzinfo=UTC)


def test_source_unit_curve_requires_all_members_and_gross():
    ps, gross = sens.derive_base_portfolio_curve(
        ("A", "B"),
        {"A": [0, 10, -5], "B": [0, 20, 20]},
        {"A": [100, 101, 99], "B": [200, 201, 202]},
        {"A": 0.1, "B": 0.2},
    )
    assert ps == pytest.approx([0, 5, 3.5])
    assert gross == pytest.approx([50, 50.3, 50.3])
    with pytest.raises(ValueError, match="INCOMPLETE_UNIT_EQUITY_OR_GROSS_CURVE"):
        sens.derive_base_portfolio_curve(("A",), {"A": [0, 1]}, {"A": [10]}, {"A": 0.2})


def test_daily_floor_is_day_start_not_trailing_peak_and_report_both():
    start = datetime(2026, 7, 1, tzinfo=UTC)
    train_end = start + timedelta(minutes=2)
    # +$500 first, then -$320 peak-to-trough, but still above day_start -$300.
    path = [10000.0, 10500.0, 10180.0]
    metrics = sens.assess_path(path, start=start, train_end=train_end)
    assert metrics["first_daily_contract_breach_utc"] is None
    assert metrics["max_peak_drawdown_usd"] == 320
    assert metrics["worst_utc_intraday_peak_loss_usd"] == 320
    assert metrics["worst_utc_day_start_loss_usd"] == 0
    assert metrics["first_static_contract_breach_utc"] is None


def test_no_guardian_base_multiplier_is_exact_linear_sensitivity():
    start = datetime(2026, 7, 1, tzinfo=UTC)
    res = sens.compare_once(
        [0, -40, 10, -150],
        [1000] * 4,
        multiplier=2.0,
        start=start,
        train_end=start + timedelta(minutes=2),
    )
    unguarded = res["without_guardian"]
    assert unguarded["full_net_usd"] == -300.0
    assert unguarded["max_peak_drawdown_usd"] == 320.0
    assert unguarded["first_daily_contract_breach_utc"] is not None


def test_soft_guardian_flattens_one_minute_later_and_24h_pause_then_reenters():
    start = datetime(2026, 7, 1, tzinfo=UTC)
    # day 1 shock, day 2 positive raw trend. HALT must not credit overnight PnL.
    pnls = [0.0, -260.0] + [-260.0] * 1500
    for i in range(1400, len(pnls)):
        pnls[i] = 300.0 + float(i - 1400)
    gross = [1000.0] * len(pnls)
    result = sens.hypothetical_guardian_overlay(
        pnls,
        gross,
        multiplier=1.0,
        start=start,
        train_end=start + timedelta(days=1),
    )
    assert result["guardian_trigger_count"] == 1
    assert result["hypothetical_flatten_count"] == 1
    assert result["hypothetical_flatten_examples"][0]["executed"] is False
    assert result["hypothetical_auto_resume_count"] == 1
    assert result["halted_minutes"] >= 1440
    assert result["permanent_contract_breach"] is False
    assert result["modeled_extra_flatten_cost_usd"] == 6.0
    assert result["modeled_extra_reentry_cost_usd"] == 0.8
    assert (
        result["full_net_usd"]
        < sens.assess_path(
            [10_000 + x for x in pnls],
            start=start,
            train_end=start + timedelta(days=1),
        )["full_net_usd"]
    )


def test_current_protected_guardian_latched_variant_never_auto_resumes():
    start = datetime(2026, 7, 1, tzinfo=UTC)
    config = sens.HypotheticalGuardian(resume_policy="MANUAL_ONLY_LATCH")
    pnls = [0.0, -260.0] + [500.0] * 2000
    result = sens.hypothetical_guardian_overlay(
        pnls,
        [1000.0] * len(pnls),
        multiplier=1,
        config=config,
        start=start,
        train_end=start + timedelta(minutes=100),
    )
    assert result["guardian_trigger_count"] == 1
    assert result["hypothetical_auto_resume_count"] == 0
    assert result["halted_minutes"] >= 1440
    assert result["resume_policy_not_current_guardian"] is True


def test_gap_breaches_daily_contract_before_flatten_permanently_fails():
    start = datetime(2026, 7, 1, tzinfo=UTC)
    pnls = [0.0, -340.0] + [-340.0] * 1450
    result = sens.hypothetical_guardian_overlay(
        pnls,
        [1000.0] * len(pnls),
        multiplier=1,
        start=start,
        train_end=start + timedelta(days=1),
    )
    assert result["first_daily_contract_breach_utc"] is not None
    assert result["permanent_contract_breach"] is True
    assert result["hypothetical_flatten_count"] == 1
    assert result["hypothetical_auto_resume_count"] == 0


def test_invalid_risk_settings_rejected():
    with pytest.raises(ValueError, match="INVALID_INTERNAL_DAILY_BUFFER"):
        sens.HypotheticalGuardian(internal_daily_stop_usd=300)
    with pytest.raises(ValueError, match="UNSAFE_HALT_OR_FLATTEN_DELAY"):
        sens.HypotheticalGuardian(flatten_delay_minutes=0)
    with pytest.raises(ValueError, match="INVALID_MODELED_EXIT_FRICTION"):
        sens.HypotheticalGuardian(adverse_close_bps=0)
