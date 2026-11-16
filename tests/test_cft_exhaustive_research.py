"""Regression tests for exhaustive CFT screening and half-risk sizing."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies"))
module = importlib.import_module("cft_20261116_exhaustive")
cap_primary_multipliers = module.cap_primary_multipliers
classify_wallet_eligibility = module.classify_wallet_eligibility


def test_current_withdrawal_does_not_reject_historical_train() -> None:
    result = classify_wallet_eligibility({"reasons": ["WITHDRAWN_LT_1000"]})
    assert result["historical_eligible"] is True
    assert result["operational_eligible"] is False
    assert result["historical_reasons"] == []
    assert result["operational_reasons"] == ["WITHDRAWN_LT_1000"]


def test_historical_missing_fills_is_not_waived_by_operational_gate() -> None:
    result = classify_wallet_eligibility(
        {"reasons": ["TRAIN_FILLS_LT_15", "ACCOUNT_VALUE_UNPROVEN"]}
    )
    assert result["historical_eligible"] is False
    assert result["operational_eligible"] is False
    assert result["historical_reasons"] == ["TRAIN_FILLS_LT_15"]


def test_no_silent_multiplier_above_half_risk() -> None:
    assert cap_primary_multipliers([1.0, 3.0, 0.5], [2.0, 0.4, 0.5]) == [
        1.0, 0.4, 0.5
    ]
    with pytest.raises(ValueError, match="INVALID_SIZING_PARAMETER"):
        cap_primary_multipliers([float("nan")], [1.0])
    with pytest.raises(ValueError, match="INVALID_SIZING_PARAMETER"):
        cap_primary_multipliers([-1.0], [1.0])
    with pytest.raises(ValueError, match="SIZING_LENGTH_MISMATCH"):
        cap_primary_multipliers([1.0], [1.0, 2.0])
