"""Current CFT wallet admission is a separate, fail-closed post-backtest gate."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies"))
audit = importlib.import_module("cft_current_admission")


def state(equity: float, notional: float, unrealized: float, *, count: int = 1) -> dict:
    return {
        "marginSummary": {"accountValue": str(equity), "totalNtlPos": str(notional)},
        "withdrawable": "0",
        "assetPositions": [
            {"position": {"unrealizedPnl": str(unrealized / count)}}
            for _ in range(count)
        ],
    }


def test_live_open_loss_rejects_wallet_that_had_good_historical_oos() -> None:
    a = audit.assess_wallet(
        "W#1", state(72415, 404827, -84022), acquired_at="2026-11-16T17:00:00Z",
    )
    assert a["status"] == "CURRENT_RESEARCH_NO_GO"
    assert "OPEN_LOSS_TO_EQUITY_EXCESSIVE" in a["checks"]
    assert a["notional_to_perp_equity"] > 5
    assert "HIGH_NOTIONAL_TO_PERP_EQUITY" in a["warnings"]
    assert a["is_delayed_follower_pnl"] is False


def test_small_perp_margin_account_not_accepted_as_large_combined_total() -> None:
    a = audit.assess_wallet(
        "W#94", state(189, 1628, -140), acquired_at="2026-11-16T17:00:00Z",
    )
    assert a["status"] == "CURRENT_RESEARCH_NO_GO"
    assert "PERP_EQUITY_BELOW_MINIMUM" in a["checks"]
    assert "OPEN_LOSS_TO_EQUITY_EXCESSIVE" in a["checks"]


def test_zero_equity_rejected_without_dividing_by_zero() -> None:
    a = audit.assess_wallet(
        "W#280", state(0, 0, 0, count=0), acquired_at="2026-11-16T17:00:00Z",
    )
    assert a["status"] == "CURRENT_RESEARCH_NO_GO"
    assert "PERP_EQUITY_BELOW_MINIMUM" in a["checks"]
    assert a["notional_to_perp_equity"] is None
    assert "NO_CURRENT_OPEN_PERP_POSITIONS" in a["warnings"]


def test_recent_closed_pnl_does_not_erase_open_loss() -> None:
    now = "2026-11-16T17:00:00Z"
    a = audit.assess_wallet(
        "W#94", state(189, 1628, -140),
        recent_fills=[{"closedPnl": "63.40", "time": 1794833411450}],
        acquired_at=now,
    )
    assert a["status"] == "CURRENT_RESEARCH_NO_GO"
    assert a["closed_pnl_30d_usd_excl_funding"] == 63.4
    assert a["fills_30d_returned"] == 1


def test_unknown_live_account_fails_closed() -> None:
    a = audit.assess_wallet("X", None, acquired_at="2026-11-16T17:00:00Z")
    assert a["status"] == "NO_GO"
    assert a["checks"] == ["SOURCE_ACCOUNT_UNAVAILABLE"]
    assert "perp_equity_usd" not in a


def test_invalid_nan_and_incomplete_sources_fail_closed() -> None:
    for raw in (state(float("nan"), 10, 0), {"marginSummary": {}}, ["bad"]):
        a = audit.assess_wallet("X", raw, acquired_at="2026-11-16T17:00:00Z")
        assert a["status"] == "NO_GO"


def test_profitable_but_high_leverage_demands_review_not_automatic_veto() -> None:
    a = audit.assess_wallet(
        "Whale", state(1_741_337, 12_523_549, 3_836_470),
        recent_fills=[{"closedPnl": "100", "time": 1794820000000}],
        acquired_at="2026-11-16T17:00:00Z",
    )
    assert a["status"] == "CURRENT_RESEARCH_ADMISSION_PASS"
    assert "HIGH_NOTIONAL_TO_PERP_EQUITY" in a["warnings"]


def test_any_bad_member_disqualifies_recommendation_even_when_oos_won() -> None:
    frozen = {
        "frozen_portfolios": [
            {"aliases": ["W#1", "W#94"], "train_opt_k": [0.008, 3.96]},
            {"aliases": ["Whale"], "train_opt_k": [0.1]},
            {"aliases": ["MISSING"], "train_opt_k": [0.05]},
        ]
    }
    by_alias = {
        name: audit.assess_wallet(
            name, snap,
            recent_fills=[{"closedPnl": "0", "time": 1794820000000}],
            acquired_at="2026-11-16T17:00:00Z",
        )
        for name, snap in {
            "W#1": state(72415, 404827, -84022),
            "W#94": state(189, 1628, -140),
            "Whale": state(1_741_337, 12_523_549, 3_836_470),
        }.items()
    }
    result = audit.evaluate_portfolios(
        frozen, by_alias, audit.ResearchAdmissionPolicy(), when="2026-11-16T17:00:00Z"
    )
    assert [x["current_admission"] for x in result["portfolios"]] == [
        "NO_GO", "PASS_SNAPSHOT_ONLY", "NO_GO"
    ]
    assert "W#1" in result["portfolios"][0]["member_blockers"]
    assert "MISSING" in result["portfolios"][2]["member_blockers"]
    assert result["current_admission_does_not_rewrite_oos"] is True
    assert result["live_execution_authorized"] is False


def test_recent_activity_failure_is_a_no_go_not_a_positive_pass() -> None:
    base = state(20_000, 2_000, 100)
    a = audit.assess_wallet(
        "W", base, acquired_at="2026-11-16T17:00:00Z",
        recent_fills=None,
    )
    assert a["status"] == "CURRENT_RESEARCH_NO_GO"
    assert "RECENT_FILL_EVIDENCE_UNAVAILABLE" in a["checks"]
    b = audit.assess_wallet(
        "W", base, acquired_at="2026-11-16T17:00:00Z",
        recent_fills=[],
    )
    assert b["status"] == "CURRENT_RESEARCH_NO_GO"
    assert "NO_RECENT_FILLS_30D" in b["checks"]


def test_mismatched_frozen_k_fails_closed() -> None:
    with pytest.raises(ValueError, match="FROZEN_K_MEMBERSHIP_MISMATCH"):
        audit.evaluate_portfolios(
            {"frozen_portfolios": [{"aliases": ["X"], "train_opt_k": []}]},
            {}, audit.ResearchAdmissionPolicy(), when="2026-11-16T17:00:00Z"
        )
