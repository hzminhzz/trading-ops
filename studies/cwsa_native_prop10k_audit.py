"""Read-only native CWSA checkpoint and Bybit eligibility audit.

Uses the existing copytrade-pm CWSA/PM03 owners; intentionally no new backtest
engine, broker simulation, retrospective leader-fill reconstruction or k fitting.

uv run --frozen python studies/cwsa_native_prop10k_audit.py
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from collections import Counter, defaultdict
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from copytrade_pm.pm_checkpoint_journal import CheckpointOutcome, verify_candidate
from copytrade_pm.pm_checkpoint_source import GRID_MS
from copytrade_pm.pm_cwsa_chapter2_gate import _paper_facts, _source_facts
from copytrade_pm.pm_cwsa_six_paper import ALL_14_WALLETS, PORTFOLIO_CONFIGS, load_registry_seals
from copytrade_pm.pm_sqlite_readonly import connect_readonly

# Frozen historical registry archived in Phase D; read-only retrospective evidence.
REGISTRY = Path(
    "/home/quant/.local/share/copytrade/registry/archive/legacy_wallet_registry_20261119.sqlite"
)
SOURCE = Path(
    "/home/quant/.local/share/copytrade/prospective/"
    "cwsa-5m-source-chapter2-20261112/paper-events.sqlite3"
)
PAPER = Path("/home/quant/.local/share/copytrade/prospective/cwsa-5m-paper-chapter2-20261112")
REPORT = Path("reports/cwsa-native-prop10k-20261114.json")
PROPOSAL = Path("proposals/cwsa-native-prop10k-three-20261114.json")
USER_AGENT = "OpenAI File Downloader, XaiImageApiFetch/1.0"
VALID_BOOK = frozenset({"BASELINE", "CHANGED", "UNCHANGED", "RECOVERY_BASELINE"})
CONTINUOUS = frozenset({"CHANGED", "UNCHANGED"})
TERMINAL_UNTRADEABLE = "BYBIT_INSTRUMENT_NOT_VERIFIED"
# New *unsealed* research combinations drawn exclusively from genuinely
# monitored, individually registry-resolved CWSA source wallets.
RESEARCH_ROSTERS = {
    "research.crypto/Balanced": ("ACT04", "ACT08", "W031"),
    "research.crypto/RegimeAlt": ("ACT08", "ACT09", "W031"),
}


def stamp(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1000, tz=UTC).isoformat()


def classify_market(key: str, bybit_bases: set[str] | None) -> str:
    """Scope identity is never discarded when mapping to a different venue."""
    try:
        namespace, symbol = key.rsplit("/", 1)
    except ValueError:
        return "UNMAPPABLE_NATIVE_KEY"
    if not namespace.startswith("hyperliquid/mainnet:perp:"):
        return "UNMAPPABLE_NATIVE_KEY"
    dex = namespace.removeprefix("hyperliquid/mainnet:perp:")
    if dex != "default":
        return "HIP3_SEPARATE_MARKET_NOT_BYBIT_EQUIVALENT"
    if bybit_bases is None:
        return "BYBIT_LISTING_UNVERIFIED"
    if symbol not in bybit_bases:
        return "NO_CURRENT_BYBIT_LINEAR_PERP"
    return "PUBLIC_BYBIT_BASE_LISTED_DEMO_UNVERIFIED"


def tradable_book(keys: set[str], bybit_bases: set[str] | None) -> bool:
    """Require all nonflat selected-scope positions to have *public* venue matches."""
    return bybit_bases is not None and all(
        classify_market(key, bybit_bases) == "PUBLIC_BYBIT_BASE_LISTED_DEMO_UNVERIFIED"
        for key in keys
    )


def public_bybit_bases() -> tuple[set[str] | None, dict]:
    """Official public instrument metadata, not demo account orderability."""
    bases = set()
    cursor: str | None = None
    pages = 0
    try:
        while True:
            params = {"category": "linear", "limit": "1000"}
            if cursor:
                params["cursor"] = cursor
            req = Request(
                "https://api.bybit.com/v5/market/instruments-info?" + urlencode(params),
                headers={"User-Agent": USER_AGENT},
            )
            with urlopen(req, timeout=12) as resp:
                payload = json.load(resp)
            if payload.get("retCode") != 0:
                raise ValueError("BYBIT_PUBLIC_METADATA_REJECTED")
            pages += 1
            for inst in payload["result"]["list"]:
                if (
                    inst.get("status") == "Trading"
                    and inst.get("contractType") == "LinearPerpetual"
                    and inst.get("settleCoin") in ("USDT", "USDC")
                ):
                    bases.add(inst["baseCoin"])
            cursor = payload["result"].get("nextPageCursor") or None
            if cursor is None:
                break
            if pages > 10:
                raise ValueError("BYBIT_PUBLIC_METADATA_PAGINATION_BOUND")
    except (OSError, HTTPError, URLError, TimeoutError, ValueError, KeyError) as exc:
        return None, {"status": "UNVERIFIED", "reason": type(exc).__name__}
    return bases, {
        "status": "PUBLIC_METADATA_ONLY",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "pages": pages,
        "distinct_base_coins": len(bases),
        "demo_available": "UNKNOWN",
    }


def registry_membership(db) -> dict[str, dict]:
    evidence = {}
    for alias in ALL_14_WALLETS:
        rows = db.execute(
            "SELECT DISTINCT w.wallet_id FROM wallet_aliases a "
            "JOIN wallets w ON w.wallet_id=a.wallet_id "
            "WHERE a.alias=? AND a.namespace='research_v2'",
            (alias,),
        ).fetchall()
        if len(rows) != 1:
            evidence[alias] = {"registered": False, "current_excluded": None}
            continue
        wallet_id = rows[0][0]
        owner = db.execute(
            "SELECT COUNT(*) FROM exclusions WHERE wallet_id=?", (wallet_id,)
        ).fetchone()[0]
        prop = db.execute(
            "SELECT COUNT(*) FROM propfirm_exclusions WHERE wallet_id=?", (wallet_id,)
        ).fetchone()[0]
        evidence[alias] = {
            "registered": True,
            "current_excluded": bool(owner + prop),
            "owner_exclusion_count": owner,
            "prop_exclusion_count": prop,
        }
    return evidence


def summarize_wave(
    at_checkpoint: dict[str, CheckpointOutcome], members: tuple[str, ...], bybit: set[str] | None
) -> dict[str, bool]:
    """Use persisted CWSA outcome kinds; no leader-fill reconstruction."""
    outcomes = [at_checkpoint.get(a) for a in members]
    complete = all(o is not None and o.kind in VALID_BOOK for o in outcomes)
    continuous = all(o is not None and o.kind in CONTINUOUS for o in outcomes)
    candidate = continuous and any(o.kind == "CHANGED" for o in outcomes if o is not None)
    books = {
        p.instrument
        for o in outcomes
        if o is not None and o.kind in VALID_BOOK
        for p in o.positions
        if p.quantity
    }
    return {
        "complete": complete,
        "continuous": continuous,
        "changed_wave": candidate,
        "candidate_full_book_public_bybit_compatible": candidate and tradable_book(books, bybit),
        "any_nondefault_market": any(":default/" not in p for p in books),
    }


def build(source_snapshot: Path) -> dict:
    # Both native owner functions replay/verify their own original SQLite/journal chains.
    # Source backup holds one consistent transaction, avoiding live multi-wallet write races.
    source_owner = _source_facts(source_snapshot)
    paper_owner = _paper_facts(PAPER)
    seals = load_registry_seals(REGISTRY)
    bybit, bybit_meta = public_bybit_bases()
    member_status: dict[str, dict]
    with closing(connect_readonly(REGISTRY)) as reg:
        member_status = registry_membership(reg)

    verified_originals = 0
    wallets: dict[str, dict] = {
        member: {
            "kinds": Counter(),
            "markets": set(),
            "first_checkpoint_ms": None,
            "last_checkpoint_ms": None,
            "published_delays_ms": [],
            "bybit_compatible_changed": 0,
        }
        for member in ALL_14_WALLETS
    }
    waves: dict[int, dict[str, CheckpointOutcome]] = defaultdict(dict)
    # Native checkpoint outcome's typed contract; original hash-chain was validated
    # above by _source_facts. These are *source* effects, never broker fills.
    with closing(connect_readonly(source_snapshot)) as db:
        db.execute("BEGIN")
        rows = db.execute("SELECT outcome FROM cwsa_outcomes ORDER BY checkpoint_ms")
        for (raw,) in rows:
            outcome = CheckpointOutcome.model_validate_json(raw)
            if outcome.source_id not in wallets or outcome.checkpoint_ms % GRID_MS:
                raise ValueError("NONCANONICAL_CWSA_SCOPE_OR_CHECKPOINT")
            wallet = wallets[outcome.source_id]
            wallet["kinds"][outcome.kind] += 1
            # All native source transitions that could establish or change an
            # economic position must replay to original Hyperliquid REST bytes.
            if outcome.kind in {"BASELINE", "RECOVERY_BASELINE", "CHANGED"}:
                if outcome.sample is None or not verify_candidate(outcome.sample, SOURCE.parent):
                    raise ValueError("NATIVE_SOURCE_ORIGINAL_WIRE_PROOF_INVALID")
                verified_originals += 1
            if outcome.kind == "CHANGED" and tradable_book(
                {p.instrument for p in outcome.positions if p.quantity}, bybit
            ):
                wallet["bybit_compatible_changed"] += 1
            cp = outcome.checkpoint_ms
            wallet["first_checkpoint_ms"] = (
                cp
                if wallet["first_checkpoint_ms"] is None
                else min(cp, wallet["first_checkpoint_ms"])
            )
            wallet["last_checkpoint_ms"] = (
                cp
                if wallet["last_checkpoint_ms"] is None
                else max(cp, wallet["last_checkpoint_ms"])
            )
            if outcome.kind in VALID_BOOK:
                if outcome.sample is None or outcome.sample.kind != "COMPLETE":
                    raise ValueError("NATIVE_COMPLETE_BOOK_PROOF_MISSING")
                if outcome.published_ms < outcome.sample.received_ms:
                    raise ValueError("NATIVE_PUBLICATION_BEFORE_RECEIPT")
                wallet["published_delays_ms"].append(outcome.published_ms - cp)
                wallet["markets"].update(p.instrument for p in outcome.positions if p.quantity)
            elif outcome.kind in {"UNAVAILABLE", "QUARANTINE"} and outcome.signal_id:
                raise ValueError("NATIVE_GAP_MUST_NOT_EMIT_SIGNAL")
            if outcome.kind == "CHANGED" and not outcome.signal_id:
                raise ValueError("NATIVE_CHANGE_WITHOUT_SIGNAL")
            if outcome.kind != "CHANGED" and outcome.signal_id is not None:
                raise ValueError("NATIVE_NONCHANGE_WITH_SIGNAL")
            if outcome.source_id in waves[cp]:
                raise ValueError("DUPLICATE_NATIVE_SCOPE_AT_CHECKPOINT")
            waves[cp][outcome.source_id] = outcome
        db.execute("ROLLBACK")

    if set(wallets) != set(source_owner["latest_kinds"]):
        raise ValueError("SOURCE_OWNER_WALLET_CENSUS_MISMATCH")
    # Live selected-wallet writes can straddle the final in-flight UTC wave.
    # Preserve that checkpoint as incomplete, not as a synthetic flat book.
    global_incomplete = sum(set(wave) != set(wallets) for wave in waves.values())

    portfolio_audits = {}
    for name, cfg in PORTFOLIO_CONFIGS.items():
        members = tuple(cfg["members"])
        seal = seals[name]
        if set(members) != set(seal["members"]):
            raise ValueError("REGISTRY_SEAL_MISMATCH")
        market_union = set().union(*(wallets[m]["markets"] for m in members))
        incompatibilities = sorted(
            {
                key: classify_market(key, bybit)
                for key in market_union
                if classify_market(key, bybit) != "PUBLIC_BYBIT_BASE_LISTED_DEMO_UNVERIFIED"
            }.items()
        )
        checks = [summarize_wave(v, members, bybit) for v in waves.values()]
        changes_per_member = {m: wallets[m]["kinds"]["CHANGED"] for m in members}
        excluded = sorted(m for m in members if member_status[m]["current_excluded"] is not False)
        paper = paper_owner[name]
        if paper["signed_mandates"] or paper["modeled_effects"]:
            # No conditional promotion: inspect signed finance owner's own evidence before
            # reporting PnL, even if events have appeared since this audit began.
            raise ValueError("PAPER_ECONOMICS_REQUIRE_NATIVE_FINANCIAL_REPLAY")
        portfolio_audits[name] = {
            "members": list(members),
            "selection_known_at": seal["selection_known_at"],
            "registry_seal": "VERIFIED",
            "excluded_or_unresolved_members": excluded,
            "native_checkpoints": len(waves),
            "complete_source_waves": sum(x["complete"] for x in checks),
            "continuous_source_waves": sum(x["continuous"] for x in checks),
            "all_member_changed_waves": sum(x["changed_wave"] for x in checks),
            "full_book_public_bybit_compatible_changed_waves": sum(
                x["candidate_full_book_public_bybit_compatible"] for x in checks
            ),
            "incomplete_or_recovery_waves": len(waves) - sum(x["continuous"] for x in checks),
            "member_change_counts": changes_per_member,
            "observed_instrument_count": len(market_union),
            "unmatched_native_markets": [
                {"market": key, "reason": reason} for key, reason in incompatibilities
            ],
            "nondefault_native_market_count": sum(":default/" not in key for key in market_union),
            "paper_signed_mandates": paper["signed_mandates"],
            "paper_modeled_effects": paper["modeled_effects"],
            "financial_pnl_usd": None,
            "financial_drawdown_usd": None,
            "bybit_fills": None,
            "proposed_k_by_wallet": {m: None for m in members},
            "sizing_policy": "FIXED_K_NATIVE_QUANTITY",
            "admission": "BLOCKED_EVIDENCE",
        }
    research_audits = {}
    for name, members in RESEARCH_ROSTERS.items():
        all_markets = set().union(*(wallets[m]["markets"] for m in members))
        checks = [summarize_wave(wave, members, bybit) for wave in waves.values()]
        research_audits[name] = {
            "members": list(members),
            "registry_members_verified": all(member_status[m]["registered"] for m in members),
            "currently_excluded_members": [
                m for m in members if member_status[m]["current_excluded"] is not False
            ],
            "portfolio_seal": "NOT_CREATED_UNSEALED_RESEARCH_ONLY",
            "selection_known_before_sample": False,
            "native_checkpoints": len(waves),
            "complete_source_waves": sum(x["complete"] for x in checks),
            "all_member_changed_waves": sum(x["changed_wave"] for x in checks),
            "full_book_public_bybit_compatible_changed_waves": sum(
                x["candidate_full_book_public_bybit_compatible"] for x in checks
            ),
            "unmatched_native_markets": [
                {"market": key, "reason": classify_market(key, bybit)}
                for key in sorted(all_markets)
                if classify_market(key, bybit) != "PUBLIC_BYBIT_BASE_LISTED_DEMO_UNVERIFIED"
            ],
            "proposed_k_by_wallet": {m: None for m in members},
            "financial_pnl_usd": None,
            "financial_drawdown_usd": None,
            "admission": "BLOCKED_EVIDENCE_UNSEALED_SOURCE_ONLY",
        }

    return {
        "schema": "native_cwsa_prop10k_source_eligibility_v1",
        "classification": "RESEARCH_PROPOSAL",
        "economic_authority": "NO_FINANCIAL_REPLAY_AVAILABLE",
        "source_authority": "NATIVE_CWSA_ORIGINAL_REST_FIVE_MINUTE_POSITION_CHECKPOINTS",
        "native_owners_invoked": [
            "copytrade_pm.pm_cwsa_chapter2_gate._source_facts",
            "copytrade_pm.pm_cwsa_chapter2_gate._paper_facts",
            "copytrade_pm.pm_cwsa_six_paper.load_registry_seals",
            "copytrade_pm.pm_checkpoint_journal.CheckpointOutcome",
        ],
        "sample_start_utc": stamp(min(waves)),
        "sample_end_utc": stamp(max(waves)),
        "native_checkpoint_cadence_ms": GRID_MS,
        "native_wallets": len(wallets),
        "global_checkpoint_waves": len(waves),
        "global_incomplete_waves": global_incomplete,
        "global_complete_waves": len(waves) - global_incomplete,
        "native_source_head_sha256": source_owner["head_sha256"],
        "source_read_isolation": "TRANSACTIONALLY_CONSISTENT_SQLITE_BACKUP_READ_ONLY_INPUT",
        "native_policy_epochs": sorted(set(source_owner["policy_epochs_by_wallet"].values())),
        "total_native_changes": sum(v["kinds"]["CHANGED"] for v in wallets.values()),
        "native_original_proven_checkpoint_count": verified_originals,
        "total_unavailable": sum(v["kinds"]["UNAVAILABLE"] for v in wallets.values()),
        "total_recovery": sum(v["kinds"]["RECOVERY_BASELINE"] for v in wallets.values()),
        "registry_status": member_status,
        "wallet_source_counts": {
            a: {
                "changed": v["kinds"]["CHANGED"],
                "public_bybit_compatible_changed": v["bybit_compatible_changed"],
                "unavailable": v["kinds"]["UNAVAILABLE"],
                "recovery_baseline": v["kinds"]["RECOVERY_BASELINE"],
                "observed_native_instruments": len(v["markets"]),
                "first_checkpoint_utc": stamp(v["first_checkpoint_ms"]),
                "last_checkpoint_utc": stamp(v["last_checkpoint_ms"]),
                "max_checkpoint_to_publication_ms": max(v["published_delays_ms"], default=None),
            }
            for a, v in sorted(wallets.items())
        },
        "bybit_public_metadata": bybit_meta,
        "challenge": {
            "initial_usd": 10000,
            "daily_loss_limit_usd": 300,
            "static_total_floor_usd": 9500,
            "daily_reset_definition": "OWNER_CONTRACT_NOT_INDEPENDENTLY_VERIFIED",
        },
        "sizing": {
            "policy": "FIXED_K_NATIVE_QUANTITY",
            "proposed_k": "NULL_UNTIL_CAUSAL_TRAIN_OOS_AND_BYBIT_EXECUTION",
            "native_signed_k_available": False,
            "native_position_book_verified_for_current_source": True,
        },
        "limitations": [
            "Only approximately 54 hours of authentic CWSA checkpoints; not a long train/OOS sample",
            "Historical Reservoir/Vike snapshots do not carry actual first-known native CWSA clocks",
            "Native signed Copy Mandates, follower L2 quotes and PM03/PM04 finance effects absent",
            "Public Bybit linear instruments do not prove account-specific Demo orderability or historic listings",
            "Source change counts and full-book symbol matching are NOT PnL, risk or alpha",
            "All members with incomplete books or unavailable intervals stay blocked; no inferred zero positions",
            "No full-sample drawdown-derived k and no fake fixed-k scaling of historical leader PnL",
        ],
        "portfolios": portfolio_audits,
        "unsealed_research_rosters": research_audits,
        "trading_authorized": False,
        "order_route": "NONE",
        "scientifically_qualified": False,
    }


def propose_three(report: dict) -> dict:
    """One existing sealed research roster plus two unsealed CWSA source variants."""
    names = (
        "regime.macro/TriadA",
        "research.crypto/Balanced",
        "research.crypto/RegimeAlt",
    )
    archetypes = (
        "SEALED_CRYPTO_BASELINE",
        "CRYPTO_BALANCED_NATIVE_SOURCE_VARIANT",
        "REGIME_DIFFERENTIATED_CRYPTO_NATIVE_SOURCE_VARIANT",
    )
    return {
        "schema": "native_cwsa_prop10k_three_research_proposals_v1",
        "classification": "RESEARCH_PROPOSAL",
        "trading_authorized": False,
        "order_route": "NONE",
        "account": report["challenge"],
        "result": "BLOCKED_EVIDENCE_NO_EXECUTABLE_K_OR_NATIVE_PNL",
        "research_selection_postdates_evaluation_window": True,
        "cross_candidate_independence": "NOT_INDEPENDENT_OVERLAPPING_WALLETS",
        "shared_wallets": ["ACT04", "ACT08", "ACT09", "W031"],
        "candidates": [
            {
                "portfolio_id": name,
                "archetype": archetype,
                "members": (
                    report["portfolios"][name]
                    if name in report["portfolios"]
                    else report["unsealed_research_rosters"][name]
                )["members"],
                "source_roster_seal": (
                    "EXISTING_SEALED_RESEARCH"
                    if name in report["portfolios"]
                    else "UNSEALED_NEW_RESEARCH_COMBINATION"
                ),
                "member_k": (
                    report["portfolios"][name]
                    if name in report["portfolios"]
                    else report["unsealed_research_rosters"][name]
                )["proposed_k_by_wallet"],
                "observed_source_signal_waves": (
                    report["portfolios"][name]
                    if name in report["portfolios"]
                    else report["unsealed_research_rosters"][name]
                )["all_member_changed_waves"],
                "bybit_publicly_compatible_source_signal_waves": (
                    report["portfolios"][name]
                    if name in report["portfolios"]
                    else report["unsealed_research_rosters"][name]
                )["full_book_public_bybit_compatible_changed_waves"],
                "unmatched_native_markets": (
                    report["portfolios"][name]
                    if name in report["portfolios"]
                    else report["unsealed_research_rosters"][name]
                )["unmatched_native_markets"],
                "native_pnl": None,
                "native_daily_drawdown": None,
                "native_static_drawdown": None,
                "qualification": "BLOCKED_EVIDENCE",
            }
            for name, archetype in zip(names, archetypes, strict=True)
        ],
        "next_mandatory_evidence": [
            "Freeze selected Bybit-supported complete-account native scope and record exclusions",
            "Acquire sufficiently long native checkpoint/first-known training and untouched OOS windows",
            "Calibrate each k on training only; leave k null until it is defendable",
            "Supply independent Bybit bid/ask, lot, fee, funding, margin and fill observations",
            "Native PM03/PM04 signed paper replay at $10,000 with daily $300 and static $9,500 floors",
        ],
    }


def main() -> None:
    # SQLite online backup reads the live producer but writes only an ephemeral
    # database outside the repo/production tree. It is deleted on exit.
    with tempfile.TemporaryDirectory(prefix="trading-ops-cwsa-readonly-") as tmp:
        snapshot = Path(tmp) / "source.sqlite3"
        with (
            closing(connect_readonly(SOURCE)) as live,
            closing(sqlite3.connect(snapshot)) as frozen,
        ):
            live.backup(frozen, pages=8192)
        report = build(snapshot)
    proposals = propose_three(report)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    PROPOSAL.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    PROPOSAL.write_text(json.dumps(proposals, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "source": {
                    "start": report["sample_start_utc"],
                    "end": report["sample_end_utc"],
                    "wallets": report["native_wallets"],
                    "checkpoints": report["global_checkpoint_waves"],
                    "changes": report["total_native_changes"],
                    "unavailable": report["total_unavailable"],
                },
                "bybit": report["bybit_public_metadata"],
                "portfolios": {
                    name: {
                        "members": v["members"],
                        "changed_waves": v["all_member_changed_waves"],
                        "public_bybit_compatible_changed_waves": v[
                            "full_book_public_bybit_compatible_changed_waves"
                        ],
                        "unmatched_markets": len(v["unmatched_native_markets"]),
                        "mandates": v["paper_signed_mandates"],
                        "finance_effects": v["paper_modeled_effects"],
                        "k": v["proposed_k_by_wallet"],
                        "admission": v["admission"],
                    }
                    for name, v in report["portfolios"].items()
                },
                "research_rosters": {
                    k: {
                        "members": v["members"],
                        "changed_waves": v["all_member_changed_waves"],
                        "bybit_compatible_changed_waves": v[
                            "full_book_public_bybit_compatible_changed_waves"
                        ],
                        "unmatched_markets": len(v["unmatched_native_markets"]),
                    }
                    for k, v in report["unsealed_research_rosters"].items()
                },
                "proposals": [c["portfolio_id"] for c in proposals["candidates"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
