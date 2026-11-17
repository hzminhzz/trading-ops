"""Cross-repository companion pin, ancestry, and ChallengeRulebookSpec tests."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDIES_DIR = ROOT / "studies"
if str(STUDIES_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIES_DIR))

cr = importlib.import_module("canonical_replay")

PROP_PRODUCTION_ROOT = ROOT.parent / "prop-production"
COPYTRADE_PM_ROOT = ROOT.parent / "copytrade-pm"


def test_cross_repo_pin_ancestry_and_contracts_when_siblings_present() -> None:
    if not PROP_PRODUCTION_ROOT.exists() or not COPYTRADE_PM_ROOT.exists():
        pytest.skip("Sibling prop-production or copytrade-pm checkout not present")

    status = cr.check_cross_repo_pin_status(
        ROOT,
        prop_production_root=PROP_PRODUCTION_ROOT,
        copytrade_pm_root=COPYTRADE_PM_ROOT,
    )
    assert status["ancestry_valid"] is True
    assert status["status"] in ("SYNCED", "PRODUCTION_PIN_PROMOTION_PENDING")
    contracts = status["contracts"]
    assert contracts["default_sizing_policy"] == "FIXED_K_NATIVE_QUANTITY"
    assert contracts["trading_authorized"] is False
    assert contracts["execution_mode"] == "SHADOW"


def test_guardian_spec_from_challenge_spec_matches_canonical_prop_firms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not PROP_PRODUCTION_ROOT.exists():
        pytest.skip("Sibling prop-production checkout not present")

    monkeypatch.syspath_prepend(str(PROP_PRODUCTION_ROOT / "src"))
    prop_firms = importlib.import_module("prop_production.prop_firms")

    cft_spec = prop_firms.get_challenge_spec("crypto-fund-trader:two_phase:phase_1")
    gs = cr.GuardianSpec.from_challenge_spec(cft_spec, scale=2.0)
    assert gs.initial_equity == pytest.approx(10_000.0)
    assert gs.profit_target == pytest.approx(800.0)
    assert gs.daily_loss_limit == pytest.approx(500.0)
    assert gs.max_static_dd == pytest.approx(1_000.0)
    assert gs.guardian_sweep_loss == pytest.approx(400.0)
    assert gs.daily_loss_basis == "balance"
    assert gs.daily_loss_rate == pytest.approx(0.05)
    assert gs.guardian_sweep_rate == pytest.approx(0.04)
    assert gs.reset_utc_hour == 0
    assert gs.reset_utc_minute == 5

    hyro_spec = prop_firms.get_challenge_spec("hyrotrader:two_step:phase_1")
    hyro_gs = cr.GuardianSpec.from_challenge_spec(hyro_spec, scale=2.0)
    assert hyro_gs.daily_loss_basis == "equity"
    assert hyro_gs.daily_loss_rate is None

    vanta_eod = prop_firms.get_challenge_spec("vanta-trading:grow:funded")
    with pytest.raises(ValueError, match="UNSUPPORTED_TRAILING_BASIS"):
        cr.GuardianSpec.from_challenge_spec(vanta_eod)
