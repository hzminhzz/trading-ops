"""Evidence audit and prospective holdout freeze for CFT portfolio research.

Read-only against SQLite/source/broker. Public Bybit REST is sufficient for
instrument metadata; account API credentials must never be loaded here.

The retrospective September-October scores remain EX_POST and cannot be
upgraded to untouched OOS by new downloads or a later research commit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pit_guardrails import (
    HISTORICAL_OOS_START,
    as_ms,
    assert_prospective_holdout,
    bybit_asof_status,
    parse_bybit_linear_instrument,
    registry_asof,
    registry_first_receipt,
    sha256_json,
    source_clock_census,
    train_only_selection,
)

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "reports/2026-11-16__cft__10k__exhaustive-v3"
OUTPUT = ROOT / "reports/2026-11-16__cft__10k__pit-evidence-v4"
REGISTRY = Path.home() / ".local/share/copytrade/registry/wallet_registry.sqlite"
SOURCE = Path.home() / (
    ".local/share/copytrade/prospective/"
    "cwsa-5m-source-chapter2-20261112/paper-events.sqlite3"
)
USER_AGENT = "OpenAI File Downloader, XaiImageApiFetch/1.0"
BYBIT = "https://api.bybit.com/v5/market"
HOUR_MS = 3_600_000


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_json(endpoint: str, args: dict[str, str]) -> dict[str, Any]:
    url = f"{BYBIT}/{endpoint}?{urllib.parse.urlencode(args)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = resp.read(2_000_000)
    obj = json.loads(body)
    if obj.get("retCode") != 0:
        raise ValueError(f"BYBIT_API_{endpoint.replace('-', '_')}_{obj.get('retCode')}")
    return obj


def fetch_bybit_catalog() -> tuple[dict[str, Any], dict[str, Any]]:
    """Bounded public instrument metadata, including closed markets if exposed."""
    summary: dict[str, Any] = {"status_pages": {}, "collected_utc": datetime.now(UTC).isoformat()}
    listings: dict[str, Any] = {}
    for status in ("Trading", "Closed"):
        cursor = ""
        pages = 0
        while pages < 12:
            params = {"category": "linear", "status": status, "limit": "1000"}
            if cursor:
                params["cursor"] = cursor
            result = _get_json("instruments-info", params)["result"]
            pages += 1
            for item in result.get("list") or []:
                instrument = parse_bybit_linear_instrument(item)
                if instrument is None:
                    continue
                # Do not trust user-facing baseCoin as a scale conversion.
                listings[instrument.symbol] = instrument
            cursor = result.get("nextPageCursor") or ""
            if not cursor:
                break
        if cursor:
            raise RuntimeError(f"BYBIT_PAGINATION_INCOMPLETE_{status}")
        summary["status_pages"][status] = pages
    serial = {
        symbol: {
            "symbol": obj.symbol, "status_now": obj.status_now,
            "launch_ms": obj.launch_ms, "delivery_ms": obj.delivery_ms,
            "min_order_qty_now": obj.min_order_qty_now,
            "qty_step_now": obj.qty_step_now,
            "min_notional_now": obj.min_notional_now,
        } for symbol, obj in sorted(listings.items())
    }
    summary["public_metadata_sha256"] = sha256_json(serial)
    summary["linear_usdt_perpetuals"] = len(listings)
    summary["historical_rulebook_proof"] = "UNPROVEN_LIVE_INSTRUMENT_SNAPSHOT_ONLY"
    return listings, summary


def fetch_completed_bybit_hour(symbol: str, asof_ms: int) -> int | None:
    """Read completed *past* Bybit 1h klines, never candles after the cutoff."""
    end = asof_ms - 1
    start = asof_ms - 25 * HOUR_MS
    obj = _get_json("kline", {
        "category": "linear", "symbol": symbol, "interval": "60",
        "start": str(start), "end": str(end), "limit": "40",
    })
    times = []
    for bar in obj.get("result", {}).get("list", []):
        time_ms = int(bar[0])
        if time_ms + HOUR_MS <= asof_ms:
            times.append(time_ms)
    return max(times) if times else None


def historical_symbol_evidence(
    symbols: set[str], instruments: dict[str, Any], cutoff_ms: int,
    *, query_klines: bool,
) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}
    proof: dict[str, int | None] = {}
    errors: dict[str, str] = {}
    check = [sym for sym in sorted(symbols) if sym in instruments]
    if query_klines:
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = {pool.submit(fetch_completed_bybit_hour, sym, cutoff_ms): sym for sym in check}
            for future in as_completed(jobs):
                sym = jobs[future]
                try:
                    proof[sym] = future.result()
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    errors[sym] = type(exc).__name__
    for sym in sorted(symbols):
        item = instruments.get(sym)
        kline = proof.get(sym)
        data[sym] = {
            "at_cutoff_status": bybit_asof_status(
                item, cutoff_ms, latest_completed_kline_open_ms=kline
            ),
            "last_completed_kline_open_ms": kline,
            "kline_request_error": errors.get(sym),
            "launch_ms": item.launch_ms if item else None,
            "delivery_ms": item.delivery_ms if item else None,
            "current_minimum_not_historical": (
                {"min_qty": item.min_order_qty_now,
                 "qty_step": item.qty_step_now,
                 "min_notional": item.min_notional_now}
                if item else None
            ),
        }
    return data


def _future_boundary(now: datetime) -> datetime:
    # The next UTC day at 00:05, never the current day or a historical boundary.
    following = (now + timedelta(days=1)).date()
    return datetime(following.year, following.month, following.day, 0, 5, tzinfo=UTC)


def _make_forward_freeze(
    prior_freeze: dict[str, Any], prior_freeze_path: Path,
    roster: list[Any], roster_sha: str, current: datetime,
    bybit_catalog: dict[str, Any],
) -> dict[str, Any]:
    start = _future_boundary(current)
    end = start + timedelta(days=30)
    assert_prospective_holdout(frozen_at=current, start=start, end=end, now=current)
    found = {r.alias: r for r in roster}
    candidates = []
    for old in prior_freeze["frozen_portfolios"]:
        members = old["aliases"]
        if len(members) != len(old["train_opt_k"]) or any(m not in found for m in members):
            raise ValueError("PROSPECTIVE_MEMBER_NOT_IN_ASOF_REGISTRY")
        if any(k <= 0 for k in old["train_opt_k"]):
            raise ValueError("PROSPECTIVE_NONPOSITIVE_K")
        candidates.append({
            "members": [
                {"alias": alias, "identity_sha256": found[alias].identity_sha256,
                 "registry_id": found[alias].wallet_id,
                 "known_at": found[alias].monitoring_known_at,
                 "alias_known_at": found[alias].alias_known_at,
                 "frozen_k": float(k)}
                for alias, k in zip(members, old["train_opt_k"])
            ],
            "selection_qualification": "HISTORICAL_EX_POST_FROZEN_FOR_PROSPECTIVE_ONLY",
        })
    doc = {
        "schema": "cft_prospective_holdout_freeze_v1",
        "status": "FROZEN_HYPOTHESIS_NOT_SOURCE_ENABLED",
        "created_at": current.isoformat(),
        "train_data_cutoff_exclusive": current.isoformat(),
        "holdout_start": start.isoformat(),
        "holdout_end_exclusive": end.isoformat(),
        "holdout_evaluation_allowed_after": end.isoformat(),
        "holding_period_days": 30,
        "asof_registry_sha256": roster_sha,
        "asof_registry_count": len(roster),
        "source_freeze_sha256": _sha_file(prior_freeze_path),
        "selection_history": "Previously inspected May-Oct 2026 returns; FORWARD_HOLDOUT_ONLY",
        "bybit_catalog_sha256": bybit_catalog.get("public_metadata_sha256"),
        "instrument_history_status": "CURRENT_SNAPSHOT_NOT_HISTORICAL_RULEBOOK",
        "source_clock_requirement": "ORIGINAL_RECEIVED_AT_AND_PUBLISHED_AT_REQUIRED",
        "source_observation_status": "NOT_ENABLED_FOR_NEW_CANDIDATES",
        "execution_status": "NO_ORDERS_READ_ONLY",
        "policy": "NEW_POSITION_ONLY_AFTER_FROZEN_START",
        "cost_scenario_bps": [6.5, 10, 15],
        "portfolios": candidates,
        "scientifically_qualified": False,
        "economically_qualified": False,
        "trading_authorized": False,
    }
    return {**doc, "freeze_payload_sha256": sha256_json(doc)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-forward", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Do not fetch current Bybit metadata or klines")
    args = parser.parse_args()

    # Historical as-of check runs BEFORE opening any historical OOS results.
    cutoff = datetime.fromisoformat(HISTORICAL_OOS_START)
    old_roster, old_sha = registry_asof(REGISTRY, cutoff)
    registry_dates = registry_first_receipt(REGISTRY)
    source = source_clock_census(SOURCE, cutoff)
    old_freeze_path = PRIOR.with_name(PRIOR.name + "__freeze.json")
    old_freeze = _read_json(old_freeze_path)
    full_train_path = PRIOR.with_name(PRIOR.name + "__train_progress.json")
    progress = _read_json(full_train_path)
    if progress["roster_sha256"] != old_freeze["roster_sha256"]:
        raise RuntimeError("PRIOR_TRAIN_FREEZE_ROSTER_DRIFT")
    train_rows = list(progress["wallets"].values())
    new_train_selection = train_only_selection(train_rows)
    historical_only = [
        row["alias"] for row in train_rows
        if not row.get("operational_eligible", False)
        and row.get("net_usd", 0) > 0 and row.get("maxdd_usd", 0) > 0
    ]

    now = datetime.now(UTC)
    current_roster, current_sha = registry_asof(REGISTRY, now)
    instruments: dict[str, Any] = {}
    bybit_summary: dict[str, Any] = {"status": "OFFLINE_NOT_RECONSTRUCTED"}
    if not args.offline:
        instruments, bybit_summary = fetch_bybit_catalog()

    # Only the old frozen portfolios, not an OOS-driven revised selection.
    prior_inventory = _read_json(PRIOR.with_name(PRIOR.name + "__inventory.json"))
    symbols_by_alias = {
        row["alias"]: set(row.get("mapped_coins", []))
        for row in prior_inventory["screen"]
    }
    test_aliases = {a for p in old_freeze["frozen_portfolios"] for a in p["aliases"]}
    symbols = {f"{coin}USDT" for alias in test_aliases
               for coin in symbols_by_alias.get(alias, ()) if ":" not in coin}
    market_evidence = historical_symbol_evidence(
        symbols, instruments, as_ms(cutoff), query_klines=not args.offline
    )
    counts = Counter(r["at_cutoff_status"] for r in market_evidence.values())
    audit = {
        "schema": "cft_pit_evidence_audit_v1",
        "generated_utc": now.isoformat(),
        "historical_asof_cutoff": cutoff.isoformat(),
        "historical_registry_roster_count": len(old_roster),
        "historical_registry_roster_sha256": old_sha,
        "historical_registry_status": (
            "AVAILABLE" if old_roster else "BLOCKED_NO_ASOF_COHORT"
        ),
        "registry_receipt_census": registry_dates,
        "historical_source_clocks": source,
        "historical_original_first_known_status": (
            "NO_PRE_CUTOFF_CWSA_RECEIPTS"
            if not source["pre_cutoff_receipt_proven"] else
            "RECEIPT_CLOCKS_EXIST_BUT_WALLET_SCOPE_UNVERIFIED"
        ),
        "historical_reconstructed_trade_data": "RETROSPECTIVE_RESERVOIR_NOT_RECEIPT_PROOF",
        "historical_follower_pit_certified": False,
        "historical_oos_independent": False,
        "previous_oos_exposure": "SEPTEMBER_OCTOBER_WAS_ALREADY_OPENED",
        "train_wallets_financially_replayed": len(train_rows),
        "historically_profitable_operationally_ineligible": len(historical_only),
        "historical_train_only_top30_no_current_value_gate": new_train_selection,
        "bybit_public": bybit_summary,
        "bybit_symbol_reconstruction_counts": dict(counts),
        "bybit_symbol_evidence": market_evidence,
        "historical_bybit_lot_rules": "UNPROVEN_PRESENT_DAY_VALUES_NOT_HISTORICAL",
        "prior_freeze_sha256": _sha_file(old_freeze_path),
        "prospective_asof_registry_count": len(current_roster),
        "prospective_asof_registry_sha256": current_sha,
        "qualification": "BLOCKED_HISTORICAL_PIT_AND_INDEPENDENT_OOS",
        "live_broker_credentials_accessed": False,
        "broker_mutations": False,
    }
    dest = OUTPUT.with_name(OUTPUT.name + "__audit.json")
    if dest.exists():
        raise FileExistsError("VERSIONED_AUDIT_ALREADY_EXISTS_DO_NOT_OVERWRITE")
    dest.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    frozen = None
    if args.freeze_forward:
        frozen = _make_forward_freeze(
            old_freeze, old_freeze_path, current_roster, current_sha, now,
            bybit_summary,
        )
        freeze_dest = OUTPUT.with_name(OUTPUT.name + "__forward_freeze.json")
        if freeze_dest.exists():
            raise FileExistsError("FORWARD_FREEZE_EXISTS_DO_NOT_OVERWRITE")
        freeze_dest.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    report = [
        "---", "firm: crypto-fund-trader", "account_usd: 10000",
        "status: BLOCKED_HISTORICAL_PIT", "top_candidate: NOT_SCIENTIFICALLY_QUALIFIED",
        "script: studies/cft_pit_evidence_audit.py", "---", "",
        "# Point-in-time evidence correction (not a backtest rerank)", "",
        f"Historical Sep 1 as-of registry: **{len(old_roster)} provably admitted wallets** from available native metadata.",
        "This is NOT proof that no wallets existed; the recorded metadata arrived later.",
        f"Earliest registry monitoring known-at: `{registry_dates['monitoring']['earliest_known_at']}`.",
        f"Historical as-of CWSA original receipt rows: **{source['outcomes_received_by_cutoff']}** (source journal begins later).",
        "Reservoir exchange-time fills are not first-known receipts.",
        "", "## Retrospective versus contemporary", "",
        f"- 128-wallet train-only ledger: {len(train_rows)} complete; historically profitable wallets that fail today's value gate: {len(historical_only)}.",
        "- The Train-only ranking is supplied in audit JSON. It ignores current account value and September-October OOS outcomes.",
        "- September-October 2026 was already examined and cannot be relabeled untouched OOS.",
        "- Current Bybit instrument availability is never silently promoted to historical listing or historical lot-rule proof.",
        f"- Historical Bybit reconstruction (available instruments): `{dict(counts)}`.",
        "- Positive historical kline evidence only confirms past exchange trading, not historical rulebook/continuous uptime.",
        "", "## Status", "",
        "- HISTORICAL_PIT: BLOCKED (no pre-September known-at roster).",
        "- HISTORICAL_SOURCE_FIRST_KNOWN: BLOCKED (Reservoir backfill has no original receipt timestamps).",
        "- HISTORICAL_BYBIT_TRADABILITY: RECONSTRUCTED where launch bounds and completed candles exist; otherwise UNKNOWN.",
        "- HISTORICAL_LOT_RULES: UNKNOWN.",
        "- SEPTEMBER_OCTOBER_INDEPENDENT_OOS: IMPOSSIBLE_TO_RECOVER.",
        "- SCIENTIFICALLY_QUALIFIED=false; ECONOMICALLY_QUALIFIED=false; TRADING_AUTHORIZED=false.",
    ]
    if frozen:
        report += [
            "", "## New prospective untouched holdout", "",
            f"- Freeze: `{frozen['created_at']}`.",
            f"- Start: `{frozen['holdout_start']}` (strictly after freeze).",
            f"- Mature: `{frozen['holdout_end_exclusive']}`.",
            f"- Payload digest: `{frozen['freeze_payload_sha256']}`.",
            "- Portfolio definitions and k are frozen without reading future holdout observations.",
            "- Original first-known source collection for these new wallets remains unverified; no watcher/service or orders were enabled.",
        ]
    report += ["", "No changes to Guardian, broker keys, tracking registry, services or orders.", ""]
    report_path = OUTPUT.with_name(OUTPUT.name + ".md")
    if report_path.exists():
        raise FileExistsError("VERSIONED_REPORT_ALREADY_EXISTS_DO_NOT_OVERWRITE")
    report_path.write_text("\n".join(report))
    print(json.dumps({
        "historical_roster_count": len(old_roster),
        "pre_cutoff_source_receipts": source["outcomes_received_by_cutoff"],
        "train_replay_count": len(train_rows),
        "train_profitable_failed_current_gate": len(historical_only),
        "bybit_status_counts": dict(counts),
        "freeze_forward": bool(frozen),
        "forward_start": frozen["holdout_start"] if frozen else None,
        "report": str(report_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
