"""Read-only *current* source-wallet admission for CFT research candidates.

Historical follower PnL is never rewritten by this live operational gate.
No credentials, broker client, source registry writer, or execution API is used.
The conservative limits below are research proposals, not Guardian controls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canonical_replay as cr

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = Path.home() / ".local/share/copytrade/registry/wallet_registry.sqlite"
FROZEN = ROOT / "reports/2026-11-16__cft__10k__hyperliquid-only-v5__freeze.json"
REPORT = ROOT / "reports/2026-11-17__cft__10k__current-risk-admission-v1.json"
URL = "https://api.hyperliquid.xyz/info"
USER_AGENT = "CFT-research-current-risk-admission/1.0"


@dataclass(frozen=True)
class ResearchAdmissionPolicy:
    min_perp_account_equity_usd: float = 1000.0
    max_negative_unrealized_pnl_fraction_equity: float = 0.25
    leverage_review_multiple: float = 5.0
    recent_activity_lookback_days: int = 30


def _finite(source: Any, field: str) -> float:
    try:
        value = float(source)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"INVALID_{field}") from exc
    if not math.isfinite(value):
        raise ValueError(f"NONFINITE_{field}")
    return value


def assess_wallet(
    alias: str,
    snapshot: dict[str, Any] | None,
    policy: ResearchAdmissionPolicy = ResearchAdmissionPolicy(),
    *,
    recent_fills: list[dict[str, Any]] | None = None,
    acquired_at: str,
) -> dict[str, Any]:
    """Fail closed on missing accounts and excessive open losses.

    Margin summary is *perpetual margin* equity, not HyperX combined
    total account value; unrealized PnL is a stress indicator, never
    the delayed follower's actual PnL.
    """
    base: dict[str, Any] = {
        "alias": alias, "observed_at": acquired_at,
        "source": "OFFICIAL_HYPERLIQUID_PUBLIC_INFO",
        "account_scope": "PERP_MARGIN_NOT_COMBINED_SPOT",
        "checks": [],
        "warnings": [],
        "status": "NO_GO",
    }
    if not isinstance(snapshot, dict):
        return {**base, "checks": ["SOURCE_ACCOUNT_UNAVAILABLE"]}
    try:
        margin = snapshot["marginSummary"]
        if not isinstance(margin, dict):
            raise TypeError("INVALID_MARGIN_SUMMARY")
        equity = _finite(margin["accountValue"], "ACCOUNT_VALUE")
        notional = _finite(margin["totalNtlPos"], "NOTIONAL")
        positions = snapshot["assetPositions"]
        if not isinstance(positions, list):
            raise TypeError("INVALID_POSITIONS")
        open_pnl = sum(
            _finite(x["position"]["unrealizedPnl"], "POSITION_PNL")
            for x in positions
        )
        withdrawable = _finite(snapshot["withdrawable"], "WITHDRAWABLE")
    except (ValueError, KeyError, TypeError) as exc:
        return {**base, "checks": [f"INVALID_ACCOUNT_EVIDENCE:{type(exc).__name__}"]}
    leverage = notional / equity if equity > 0 else None
    loss_fraction = max(0.0, -open_pnl) / equity if equity > 0 else None
    checks: list[str] = []
    warnings: list[str] = []
    if equity < policy.min_perp_account_equity_usd:
        checks.append("PERP_EQUITY_BELOW_MINIMUM")
    if loss_fraction is not None and loss_fraction > policy.max_negative_unrealized_pnl_fraction_equity:
        checks.append("OPEN_LOSS_TO_EQUITY_EXCESSIVE")
    if leverage is not None and leverage > policy.leverage_review_multiple:
        warnings.append("HIGH_NOTIONAL_TO_PERP_EQUITY")
    if not positions:
        warnings.append("NO_CURRENT_OPEN_PERP_POSITIONS")
    if withdrawable <= 0 and notional > 0:
        warnings.append("ZERO_WITHDRAWABLE_WHILE_EXPOSED")
    if recent_fills is None:
        checks.append("RECENT_FILL_EVIDENCE_UNAVAILABLE")
        count = None
        realized = None
        last_trade = None
    else:
        count = len(recent_fills)
        realized = sum(
            _finite(fill.get("closedPnl", "0"), "CLOSED_PNL")
            for fill in recent_fills
        )
        last_trade = max(
            (int(fill["time"]) for fill in recent_fills), default=None
        )
        if not recent_fills:
            checks.append("NO_RECENT_FILLS_30D")
        if len(recent_fills) >= 2000:
            warnings.append("ACTIVITY_COUNT_AT_API_PAGE_LIMIT")
    return {
        **base,
        "status": "CURRENT_RESEARCH_ADMISSION_PASS" if not checks else "CURRENT_RESEARCH_NO_GO",
        "checks": checks,
        "warnings": warnings,
        "perp_equity_usd": round(equity, 2),
        "open_notional_usd": round(notional, 2),
        "unrealized_pnl_usd": round(open_pnl, 2),
        "withdrawable_usd": round(withdrawable, 2),
        "position_count": len(positions),
        "notional_to_perp_equity": round(leverage, 3) if leverage is not None else None,
        "open_loss_over_perp_equity": round(loss_fraction, 4) if loss_fraction is not None else None,
        "fills_30d_returned": count,
        "closed_pnl_30d_usd_excl_funding": round(realized, 2) if realized is not None else None,
        "latest_fill_at_ms": last_trade,
        "is_delayed_follower_pnl": False,
    }


def _post(payload: dict[str, Any]) -> Any:
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.load(response)


def _fetch(
    address: str, now: datetime, policy: ResearchAdmissionPolicy
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None, list[str]]:
    errs: list[str] = []
    state = None
    recent = None
    try:
        result = _post({"type": "clearinghouseState", "user": address})
        if isinstance(result, dict):
            state = result
    except (ValueError, OSError, TypeError, KeyError) as exc:
        errs.append(f"CLEARINGHOUSE:{type(exc).__name__}")
    try:
        start_ms = int((now - timedelta(days=policy.recent_activity_lookback_days)).timestamp() * 1000)
        fills = _post({
            "type": "userFillsByTime",
            "user": address,
            "startTime": start_ms,
            "aggregateByTime": True,
        })
        if isinstance(fills, list) and all(isinstance(f, dict) for f in fills):
            recent = fills
        else:
            errs.append("FILL_RESPONSE_SCHEMA")
    except (ValueError, OSError, TypeError, KeyError) as exc:
        errs.append(f"USER_FILLS:{type(exc).__name__}")
    return state, recent, errs


def evaluate_portfolios(
    frozen: dict[str, Any],
    snapshots: dict[str, dict[str, Any]],
    policy: ResearchAdmissionPolicy,
    *,
    when: str,
    errors: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    results = []
    for index, portfolio in enumerate(frozen["frozen_portfolios"], 1):
        members = portfolio["aliases"]
        if len(members) != len(portfolio["train_opt_k"]):
            raise ValueError("FROZEN_K_MEMBERSHIP_MISMATCH")
        missing = [alias for alias in members if alias not in snapshots]
        blockers = {alias: snapshots[alias]["checks"]
                    for alias in members if alias in snapshots
                    and snapshots[alias]["status"] != "CURRENT_RESEARCH_ADMISSION_PASS"}
        for alias in missing:
            blockers[alias] = ["SOURCE_ACCOUNT_UNAVAILABLE"]
        results.append({
            "train_rank": index,
            "wallet_aliases": members,
            "current_admission": "NO_GO" if blockers else "PASS_SNAPSHOT_ONLY",
            "member_blockers": blockers,
        })
    return {
        "schema": "cft_current_admission_v1",
        "obtained_at": when,
        "research_policy": asdict(policy),
        "frozen_source": "CFT_HYPERLIQUID_ONLY_V5",
        "selection_period": "MAY_AUG_TRAIN_2026",
        "historical_evaluation": "SEPT_OCT_PREVIOUSLY_OPENED_RETROSPECTIVE",
        "current_admission_does_not_rewrite_oos": True,
        "source_account_scope": "HYPERLIQUID_PERP_MARGIN",
        "snapshots": snapshots,
        "data_errors": errors or {},
        "portfolios": results,
        "live_execution_authorized": False,
        "risk_policy_is_deployment_authority": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args()
    frozen = json.loads(FROZEN.read_text())
    if frozen.get("backtest_venue") != "hyperliquid":
        raise ValueError("FROZEN_VENUE_MISMATCH")
    accounts, _ = cr.load_registry_universe(REGISTRY)
    index = {rec.label: rec for rec in accounts}
    members = {a for p in frozen["frozen_portfolios"] for a in p["aliases"]}
    policy = ResearchAdmissionPolicy()
    now = datetime.now(UTC)
    date = now.isoformat()
    metrics: dict[str, dict[str, Any]] = {}
    errors: dict[str, list[str]] = {}
    for alias in sorted(members):
        if alias not in index:
            metrics[alias] = assess_wallet(alias, None, policy, acquired_at=date)
            errors[alias] = ["CURRENT_ALIAS_NOT_FOUND"]
            continue
        state, fills, err = _fetch(index[alias].address, now, policy)
        metrics[alias] = assess_wallet(
            alias, state, policy, recent_fills=fills, acquired_at=date
        )
        if err:
            errors[alias] = err
    result = evaluate_portfolios(
        frozen, metrics, policy, when=date, errors=errors
    )
    result["roster_alias_census_sha256"] = hashlib.sha256(
        "|".join(sorted(index)).encode()
    ).hexdigest()
    if args.output.exists():
        raise FileExistsError("VERSIONED_RESEARCH_REPORT_ALREADY_EXISTS")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "observed_at": date,
        "members": {a: {"status": x["status"], "checks": x["checks"]}
                    for a, x in metrics.items()},
        "portfolio_admissions": [
            {"rank": x["train_rank"], "status": x["current_admission"]}
            for x in result["portfolios"]
        ],
        "path": str(args.output),
    }))


if __name__ == "__main__":
    main()
