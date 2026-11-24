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


def test_fee_schedule_keys_exist_in_rulebook_catalog() -> None:
    """Fee schedule owns commercial pricing only and every key resolves in RulebookCatalog."""
    from trading_ops.research import DEFAULT_FEE_SCHEDULE_PATH, load_fee_schedule

    schedule = load_fee_schedule(DEFAULT_FEE_SCHEDULE_PATH)
    challenges = schedule["challenges"]
    assert challenges

    catalog_prefixes = {":".join(k.split(":")[:2]) for k in RulebookCatalog.list_keys()}
    forbidden_rule_keys = {
        "target",
        "daily_loss",
        "trailing_loss",
        "daily_basis",
        "trailing_basis",
        "reset_time_hhmm",
        "profitSplitPct",
        "payout_split_pct",
    }

    for challenge_key, entry in challenges.items():
        assert challenge_key in catalog_prefixes, f"Unknown challenge key {challenge_key!r}"
        assert isinstance(entry, dict)
        assert entry.get("verified_at"), f"Missing verified_at for {challenge_key}"
        assert entry.get("source_urls"), f"Missing source_urls for {challenge_key}"
        assert isinstance(entry.get("tiers"), dict) and entry["tiers"], (
            f"Missing tiers for {challenge_key}"
        )
        assert not (forbidden_rule_keys & set(entry.keys())), (
            f"Risk rule keys leaked into fee schedule for {challenge_key}"
        )


def test_fee_schedule_lookup_exact_tiers_only(tmp_path: Path) -> None:
    """Tier fee lookup never interpolates, returns None on non-finite, and keeps overrides."""
    import json

    from trading_ops.research import lookup_tier_fee

    propr_10k = lookup_tier_fee("propr:classic_1step", 10_000)
    assert propr_10k is not None
    assert propr_10k.fee_usd == pytest.approx(110.0)
    assert propr_10k.refundable_on_pass is False

    propr_5k = lookup_tier_fee("propr:classic_1step", 5_000)
    assert propr_5k is not None
    assert propr_5k.fee_usd == pytest.approx(60.0)

    hyro_10k = lookup_tier_fee("hyrotrader:two_step", 10_000)
    assert hyro_10k is not None
    assert hyro_10k.fee_usd == pytest.approx(119.0)
    assert hyro_10k.refundable_on_pass is True

    assert lookup_tier_fee("crypto-fund-trader:two_phase", 10_000) is None
    assert lookup_tier_fee("propr:classic_1step", 7_500) is None
    assert lookup_tier_fee("propr:classic_1step", float("nan")) is None
    assert lookup_tier_fee("propr:classic_1step", float("inf")) is None

    custom_schedule = tmp_path / "fees.json"
    custom_schedule.write_text(
        json.dumps(
            {
                "version": 2,
                "challenges": {
                    "propr:classic_1step": {
                        "verified_at": "2026-11-24",
                        "source_urls": ["https://example.invalid/base"],
                        "notes": "Base note",
                        "tiers": {
                            "10000": {
                                "fee": 110,
                                "source_urls": ["https://example.invalid/tier-10k"],
                                "notes": "Tier 10k override",
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    override_quote = lookup_tier_fee("propr:classic_1step", 10_000, path=custom_schedule)
    assert override_quote is not None
    assert override_quote.fee_usd == pytest.approx(110.0)
    assert override_quote.source_urls == ("https://example.invalid/tier-10k",)
    assert override_quote.notes == "Tier 10k override"
