"""Native CWSA-only portfolio-screen behavioral safeguards."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cwsa_native_prop10k_audit", ROOT / "studies" / "cwsa_native_prop10k_audit.py"
)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def sample(kind: str, markets: tuple[str, ...] = ()):
    return SimpleNamespace(
        kind=kind,
        positions=tuple(SimpleNamespace(instrument=market, quantity=1) for market in markets),
    )


def test_default_native_market_maps_to_public_bybit_base_without_becoming_demo_proof():
    key = "hyperliquid/mainnet:perp:default/BTC"
    assert MOD.classify_market(key, {"BTC"}) == "PUBLIC_BYBIT_BASE_LISTED_DEMO_UNVERIFIED"
    assert MOD.classify_market(key, {"ETH"}) == "NO_CURRENT_BYBIT_LINEAR_PERP"
    assert MOD.classify_market(key, None) == "BYBIT_LISTING_UNVERIFIED"


def test_identically_named_builder_dex_market_does_not_map_as_bybit():
    assert (
        MOD.classify_market("hyperliquid/mainnet:perp:xyz/BTC", {"BTC"})
        == "HIP3_SEPARATE_MARKET_NOT_BYBIT_EQUIVALENT"
    )
    assert MOD.tradable_book({"hyperliquid/mainnet:perp:xyz/BTC"}, {"BTC"}) is False


def test_full_native_book_must_be_mappable_not_only_changed_market():
    book = {
        "hyperliquid/mainnet:perp:default/BTC",
        "hyperliquid/mainnet:perp:xyz/NVDA",
    }
    assert MOD.tradable_book(book, {"BTC", "NVDA"}) is False
    assert MOD.tradable_book({"hyperliquid/mainnet:perp:default/BTC"}, {"BTC"}) is True


def test_all_portfolio_members_require_continuous_source_proof_for_changed_wave():
    wave = {
        "A": sample("CHANGED", ("hyperliquid/mainnet:perp:default/BTC",)),
        "B": sample("UNCHANGED", ("hyperliquid/mainnet:perp:default/BTC",)),
        "C": sample("UNCHANGED"),
    }
    result = MOD.summarize_wave(wave, ("A", "B", "C"), {"BTC"})
    assert result["complete"]
    assert result["changed_wave"]
    assert result["candidate_full_book_public_bybit_compatible"]
    assert not MOD.summarize_wave(wave, ("A", "B", "MISSING"), {"BTC"})["changed_wave"]


def test_unavailable_and_recovery_books_never_fabricate_new_risk():
    wave = {
        "A": sample("CHANGED", ("hyperliquid/mainnet:perp:default/BTC",)),
        "B": sample("UNAVAILABLE"),
        "C": sample("UNCHANGED"),
    }
    status = MOD.summarize_wave(wave, ("A", "B", "C"), {"BTC"})
    assert not status["complete"]
    assert not status["changed_wave"]
    wave["B"] = sample("RECOVERY_BASELINE", ("hyperliquid/mainnet:perp:default/BTC",))
    status = MOD.summarize_wave(wave, ("A", "B", "C"), {"BTC"})
    assert status["complete"]
    assert not status["changed_wave"]


def test_proposal_k_is_unknown_without_native_follower_economics():
    def audit(members):
        return {
            "members": list(members),
            "proposed_k_by_wallet": {alias: None for alias in members},
            "all_member_changed_waves": 0,
            "full_book_public_bybit_compatible_changed_waves": 0,
            "unmatched_native_markets": [],
        }

    fake = {
        "challenge": {"initial_usd": 10_000},
        "portfolios": {
            "regime.macro/TriadA": audit(("ACT04", "ACT08", "ACT09")),
        },
        "unsealed_research_rosters": {
            "research.crypto/Balanced": audit(("ACT04", "ACT08", "W031")),
            "research.crypto/RegimeAlt": audit(("ACT08", "ACT09", "W031")),
        },
    }
    proposal = MOD.propose_three(fake)
    assert len(proposal["candidates"]) == 3
    assert all(
        set(candidate["member_k"].values()) == {None} for candidate in proposal["candidates"]
    )
    assert [c["source_roster_seal"] for c in proposal["candidates"]] == [
        "EXISTING_SEALED_RESEARCH",
        "UNSEALED_NEW_RESEARCH_COMBINATION",
        "UNSEALED_NEW_RESEARCH_COMBINATION",
    ]
    assert proposal["trading_authorized"] is False


def test_native_fixed_k_quantity_contract_uses_source_notional_and_follower_price():
    from decimal import Decimal

    from copytrade_pm.pm_exposure_sizing import target_from_fixed_k

    key = "hyperliquid/mainnet:perp:default/BTC"
    intent = target_from_fixed_k(
        source_quantities={key: Decimal(2)},
        k=Decimal("0.04"),
        source_marks={key: Decimal(100)},
        follower_marks={key: Decimal(95)},
        follower_equity_usd=Decimal(10000),
        max_gross_notional_usd=Decimal(1000),
        max_market_notional_usd=Decimal(1000),
        pod_loss_budget_usd=Decimal(50),
    )
    assert intent["sizing_policy"] == "FIXED_K_NATIVE_QUANTITY"
    assert intent["target_notionals_usd"][key] == "8.00"
    assert abs(Decimal(intent["target_quantities"][key]) - Decimal(8) / Decimal(95)) < Decimal(
        "1e-25"
    )
    assert intent["risk_supervisor_required"] is True
    assert intent["order_route"] == "NONE"
    assert intent["trading_authorized"] is False
