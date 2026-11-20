"""Single-source rulebook contract: research reads the canonical kernel catalog only."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from copytrade_kernel.rulebooks import RulebookCatalog

ROOT = Path(__file__).resolve().parents[1]
STUDIES_DIR = ROOT / "studies"
if str(STUDIES_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIES_DIR))

import pod_constructor as pc


def test_no_local_rulebook_table_or_drift_script() -> None:
    """The duplicated research rulebook data file and its drift checker stay deleted."""
    assert not (ROOT / "prop_firms_rules.json").exists()
    assert not (ROOT / "scripts" / "check_rulebook_drift.py").exists()


def test_contract_terms_come_from_kernel_catalog() -> None:
    key = "crypto-fund-trader:two_phase:phase_1"
    assert key in RulebookCatalog.list_keys("crypto-fund-trader")

    contract = pc.load_prop_contract_spec("crypto-fund-trader", account_usd=10_000.0)
    phase_1 = RulebookCatalog.get(key)
    scale = 10_000.0 / float(phase_1.initial)

    assert contract.eval_phases[0].name == phase_1.phase
    assert contract.eval_phases[0].profit_target_usd == pytest.approx(float(phase_1.target) * scale)
    assert contract.eval_phases[0].max_static_dd_usd == pytest.approx(
        float(phase_1.trailing_loss) * scale
    )
    assert contract.eval_phases[0].daily_loss_limit_usd == pytest.approx(
        float(phase_1.daily_loss) * scale
    )
    # Fees are never derived from a local table; an unpriced contract stays unpriced.
    assert contract.challenge_fee_usd is None


def test_program_selection_reads_catalog_programs() -> None:
    propr = pc.load_prop_contract_spec("propr", account_usd=10_000.0)
    assert propr.program_name == "Classic 1-Step"
    assert len(propr.eval_phases) == 1

    classic_2step = pc.load_prop_contract_spec(
        "propr", account_usd=10_000.0, program_filter="Classic 2-Step"
    )
    assert classic_2step.program_name == "Classic 2-Step"
    assert [p.name for p in classic_2step.eval_phases] == ["phase_1", "phase_2"]


def test_unknown_firm_and_unknown_program_fail_closed() -> None:
    with pytest.raises(FileNotFoundError, match="PROP_FIRM_NOT_FOUND:not-a-firm"):
        pc.load_prop_contract_spec("not-a-firm")
    with pytest.raises(ValueError, match="NO_MATCHING_PROP_PHASES:propr:not-a-program"):
        pc.load_prop_contract_spec("propr", program_filter="not-a-program")


def test_fee_is_only_ever_caller_supplied() -> None:
    priced = pc.load_prop_contract_spec(
        "crypto-fund-trader", account_usd=10_000.0, challenge_fee_usd=120.0
    )
    assert priced.challenge_fee_usd == pytest.approx(120.0)
    assert pc.load_prop_contract_spec("crypto-fund-trader").challenge_fee_usd is None
