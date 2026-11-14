"""Drawdown-budget k multiplier / hypothetical 24h Guardian sensitivity.

Uses existing native-PM03 *unit-k* follower account paths and source quantities,
not a new trading engine. A 24h resume after HALT is an explicitly
COUNTERFACTUAL policy: current prop-production Guardian is protected-latched,
requires signed remediation and does not auto-arm. The halt overlay assumes
an immediate one-minute risk observation, next-minute stressed market close,
no exposure for 24h, then a fresh 5m-position re-entry with gross-notional
friction. It is NOT a byte-exact replay of PM03/PM04 fills or actual Guardian.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import historical_5m_wallet_risk as native

OUTPUT = Path("reports/guardian-k-sweep-20260701-20261101.json")
MULTIPLIERS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)
STRESS_MULTIPLIERS = (1.5, 1.75, 2.0, 2.5, 3.0)
START_EQUITY = 10_000.0
HARD_DAILY_LOSS = 300.0
HARD_STATIC_FLOOR = 9_500.0
MINUTES_PER_DAY = 1_440
SAMPLE_LIMIT = 12


@dataclass(frozen=True)
class HypotheticalGuardian:
    internal_daily_stop_usd: float = 250.0
    internal_static_floor_usd: float = 9_600.0
    halt_minutes: int = 1_440
    flatten_delay_minutes: int = 1
    adverse_close_bps: float = 60.0
    reentry_bps: float = 8.0
    resume_policy: Literal["AUTO_24H_HYPOTHETICAL", "MANUAL_ONLY_LATCH"] = "AUTO_24H_HYPOTHETICAL"

    def __post_init__(self) -> None:
        if not (0 < self.internal_daily_stop_usd < HARD_DAILY_LOSS):
            raise ValueError("INVALID_INTERNAL_DAILY_BUFFER")
        if not (HARD_STATIC_FLOOR < self.internal_static_floor_usd < START_EQUITY):
            raise ValueError("INVALID_INTERNAL_STATIC_BUFFER")
        if self.halt_minutes < MINUTES_PER_DAY or self.flatten_delay_minutes < 1:
            raise ValueError("UNSAFE_HALT_OR_FLATTEN_DELAY")
        if self.adverse_close_bps <= 0 or self.reentry_bps < 0:
            raise ValueError("INVALID_MODELED_EXIT_FRICTION")


def derive_base_portfolio_curve(
    members: tuple[str, ...],
    wallet_curves: dict[str, list[float]],
    wallet_gross: dict[str, list[float]],
    frozen_ks: dict[str, float],
) -> tuple[list[float], list[float]]:
    """Linear native-PM03 unit-k reference paths; do not call netted execution."""
    if not members or any(
        alias not in wallet_curves or alias not in wallet_gross for alias in members
    ):
        raise ValueError("MISSING_NATIVE_WALLET_PATH")
    n = len(wallet_curves[members[0]])
    if n < 1 or any(len(wallet_curves[m]) != n or len(wallet_gross[m]) != n for m in members):
        raise ValueError("INCOMPLETE_UNIT_EQUITY_OR_GROSS_CURVE")
    if any(not math.isfinite(frozen_ks[m]) or frozen_ks[m] <= 0 for m in members):
        raise ValueError("INVALID_FROZEN_TRAIN_K")
    pnl, gross = [], []
    for i in range(n):
        pnl.append(sum(frozen_ks[m] * wallet_curves[m][i] for m in members))
        gross.append(sum(frozen_ks[m] * wallet_gross[m][i] for m in members))
    return pnl, gross


def assess_path(
    values: list[float],
    *,
    start: datetime = native.START,
    train_end: datetime = native.TRAIN_END,
) -> dict:
    """Measure contractual static loss from inception, not trailing drawdown."""
    if not values:
        raise ValueError("NO_EQUITY_PATH")
    highest = START_EQUITY
    lowest = START_EQUITY
    max_dd = 0.0
    max_day_start_loss = 0.0
    max_day_peak_loss = 0.0
    day_start = START_EQUITY
    day_peak = START_EQUITY
    hard_daily_breach = None
    hard_static_breach = None
    target_at = None
    target_idx = None
    october_start = (native.ms(train_end) - native.ms(start)) // native.MINUTE
    train_pnl = values[october_start - 1] - START_EQUITY if october_start else None
    for i, eq in enumerate(values):
        if not math.isfinite(eq):
            raise ValueError("NONFINITE_EQUITY")
        if i and i % MINUTES_PER_DAY == 0:
            day_start = values[i - 1]
            day_peak = day_start
        day_peak = max(day_peak, eq)
        highest = max(highest, eq)
        lowest = min(lowest, eq)
        max_dd = max(max_dd, highest - eq)
        max_day_start_loss = max(max_day_start_loss, day_start - eq)
        max_day_peak_loss = max(max_day_peak_loss, day_peak - eq)
        if hard_daily_breach is None and eq <= day_start - HARD_DAILY_LOSS:
            hard_daily_breach = native.date(native.ms(start) + i * native.MINUTE)
        if hard_static_breach is None and eq <= HARD_STATIC_FLOOR:
            hard_static_breach = native.date(native.ms(start) + i * native.MINUTE)
        if target_idx is None and eq >= START_EQUITY + 1000:
            target_at = native.date(native.ms(start) + i * native.MINUTE)
            target_idx = i
    return {
        "full_net_usd": round(values[-1] - START_EQUITY, 2),
        "train_net_usd": round(train_pnl, 2) if train_pnl is not None else None,
        "october_net_usd": round(values[-1] - START_EQUITY - train_pnl, 2)
        if train_pnl is not None
        else None,
        "max_peak_drawdown_usd": round(max_dd, 2),
        "minimum_equity_usd": round(lowest, 2),
        "worst_utc_day_start_loss_usd": round(max_day_start_loss, 2),
        "worst_utc_intraday_peak_loss_usd": round(max_day_peak_loss, 2),
        "first_daily_contract_breach_utc": hard_daily_breach,
        "first_static_contract_breach_utc": hard_static_breach,
        "first_10pct_profit_target_utc": target_at,
        "profit_target_would_end_challenge": target_idx is not None,
    }


def hypothetical_guardian_overlay(
    base_pnl: list[float],
    base_gross: list[float],
    *,
    multiplier: float,
    config: HypotheticalGuardian = HypotheticalGuardian(),
    start: datetime = native.START,
    train_end: datetime = native.TRAIN_END,
) -> dict:
    """Time-ordered *counterfactual* flatten/pause/re-enter stress proxy.

    All marks and unit paths come from historical PM03. The proxy reconstructs
    hypothetical equity only; skipping historical orders during a halt and
    re-entering their resulting 5m target cannot reproduce real PM03 basis,
    actual capacity, Guardian signed state, or exact fill timing.
    """
    if len(base_pnl) != len(base_gross) or not base_pnl:
        raise ValueError("UNIT_PNL_AND_GROSS_COVERAGE_MISMATCH")
    if multiplier <= 0 or not math.isfinite(multiplier):
        raise ValueError("INVALID_K_MULTIPLIER")
    phase: Literal["ACTIVE", "EXIT_PENDING", "HALTED", "TERMINAL"] = "ACTIVE"
    equity = START_EQUITY
    day_start = START_EQUITY
    day_peak = START_EQUITY
    exit_due = -1
    eligible_resume_at = -1
    permanent_failure = False
    guardian_triggers = []
    exits = []
    resume_count = 0
    skipped_minutes = 0
    exit_cost_usd = 0.0
    reentry_cost_usd = 0.0
    proxy_equity: list[float] = []

    for i, raw in enumerate(base_pnl):
        prev = base_pnl[i - 1] if i else 0.0
        gross = base_gross[i] * multiplier
        if gross < 0 or not math.isfinite(gross) or not math.isfinite(raw):
            raise ValueError("NONFINITE_MARK_OR_NEGATIVE_GROSS")
        if i and i % MINUTES_PER_DAY == 0:
            day_start = equity
            day_peak = equity

        # Next-minute *modeled* flatten, not instantaneous boundary perfect fill.
        if phase in ("ACTIVE", "EXIT_PENDING"):
            equity += (raw - prev) * multiplier

        if phase == "EXIT_PENDING" and i >= exit_due:
            cost = gross * config.adverse_close_bps / 10_000
            equity -= cost
            exit_cost_usd += cost
            exits.append(
                {
                    "minute": i,
                    "at_utc": native.date(native.ms(start) + i * native.MINUTE),
                    "modeled_gross_exited_usd": round(gross, 2),
                    "assumed_total_exit_cost_usd": round(cost, 2),
                    "executed": False,
                }
            )
            phase = "HALTED"
            eligible_resume_at = i + config.halt_minutes

        if phase == "HALTED":
            skipped_minutes += 1
            if (
                i >= eligible_resume_at
                and i % 5 == 0
                and not permanent_failure
                and config.resume_policy == "AUTO_24H_HYPOTHETICAL"
            ):
                reentry = gross * config.reentry_bps / 10_000
                equity -= reentry
                reentry_cost_usd += reentry
                resume_count += 1
                phase = "ACTIVE"

        day_peak = max(day_peak, equity)
        hard_daily_breach = equity <= day_start - HARD_DAILY_LOSS
        hard_static_breach = equity <= HARD_STATIC_FLOOR
        if hard_daily_breach or hard_static_breach:
            permanent_failure = True

        if phase == "ACTIVE" and (
            equity <= day_start - config.internal_daily_stop_usd
            or equity <= day_peak - config.internal_daily_stop_usd
            or equity <= config.internal_static_floor_usd
        ):
            phase = "EXIT_PENDING"
            exit_due = i + config.flatten_delay_minutes
            guardian_triggers.append(
                {
                    "observed_at_utc": native.date(native.ms(start) + i * native.MINUTE),
                    "trigger_equity_usd": round(equity, 2),
                    "day_start_headroom_usd": round(equity - (day_start - HARD_DAILY_LOSS), 2),
                    "static_headroom_usd": round(equity - HARD_STATIC_FLOOR, 2),
                    "flatten_due_at_utc": native.date(native.ms(start) + exit_due * native.MINUTE),
                }
            )

        # Actual protected Guardian never automatically re-arms; this branch
        # is a counterfactual behavior, not an implementation of that journal.
        if phase == "HALTED" and permanent_failure:
            phase = "TERMINAL"
        # During HALT/TERMINAL, the raw path is ignored (positions are flat).
        # Resume at a five-minute checkpoint rebases to the current leader book.
        proxy_equity.append(equity)

    stats = assess_path(proxy_equity, start=start, train_end=train_end)
    return {
        **stats,
        "classification": "GUARDIAN_24H_HALT_HYPOTHETICAL_OVERLAY_NOT_NATIVE_EXECUTION",
        "multiplier": multiplier,
        "guardian_proxy_policy": asdict(config),
        "guardian_trigger_count": len(guardian_triggers),
        "guardian_trigger_examples": guardian_triggers[:SAMPLE_LIMIT],
        "hypothetical_flatten_count": len(exits),
        "hypothetical_flatten_examples": exits[:SAMPLE_LIMIT],
        "hypothetical_auto_resume_count": resume_count,
        "halted_minutes": skipped_minutes,
        "modeled_extra_flatten_cost_usd": round(exit_cost_usd, 2),
        "modeled_extra_reentry_cost_usd": round(reentry_cost_usd, 2),
        "permanent_contract_breach": permanent_failure,
        "resume_policy_not_current_guardian": True,
        "does_not_reconstruct_actual_broker_or_netting": True,
        "funding_and_margin_unknown": True,
    }


def compare_once(
    base_pnl: list[float],
    base_gross: list[float],
    *,
    multiplier: float,
    config: HypotheticalGuardian = HypotheticalGuardian(),
    start: datetime = native.START,
    train_end: datetime = native.TRAIN_END,
) -> dict:
    passive_equity = [START_EQUITY + multiplier * v for v in base_pnl]
    return {
        "k_multiplier": multiplier,
        "without_guardian": assess_path(passive_equity, start=start, train_end=train_end),
        "with_hypothetical_24h_guardian": hypothetical_guardian_overlay(
            base_pnl,
            base_gross,
            multiplier=multiplier,
            config=config,
            start=start,
            train_end=train_end,
        ),
    }


def run_sweep(*, multipliers: tuple[float, ...] = MULTIPLIERS) -> dict:
    identities, _ = native.registry_wallets()
    sources = {
        name: native.load_source(name, address, native.START, native.END)
        for name, address in identities.items()
    }
    bars = native.LocalBars(native.START, native.END)
    for symbol in sorted({s for source in sources.values() for s in source.symbols}):
        bars.add(symbol)
    curves = {}
    gross_curves = {}
    wallets = {}
    for name, source in sources.items():
        unit = native.replay_unit(source, bars, capture_gross=True)
        if not unit["complete_window_minutes"] or not unit["complete_train_minutes"]:
            raise ValueError("INCOMPLETE_HISTORICAL_REFERENCE_SERIES:" + name)
        curves[name] = unit.pop("_unit_curve_usd")
        gross_curves[name] = unit.pop("_unit_gross_usd")
        k = native.k_for_wallet_drawdown(
            native.d(unit["train_unit_max_drawdown_usd"]), native.risk_allocation()
        )
        if k is None:
            raise ValueError("UNAVAILABLE_CAUSAL_TRAIN_K")
        wallets[name] = {
            "train_frozen_k": str(k),
            "train_unit_max_drawdown": unit["train_unit_max_drawdown_usd"],
            "max_gross_notional_usd": round(max(gross_curves[name]) * float(k), 2),
            "source_first_known_valid": False,
            "source_snapshot_disagreements": source.validation["sampled_snapshot_disagreements"],
        }
    stress_policies = {
        "slow_5m_120bps_exit": HypotheticalGuardian(
            internal_daily_stop_usd=220,
            flatten_delay_minutes=5,
            adverse_close_bps=120,
            reentry_bps=16,
        ),
        "slow_10m_200bps_exit": HypotheticalGuardian(
            internal_daily_stop_usd=200,
            flatten_delay_minutes=10,
            adverse_close_bps=200,
            reentry_bps=20,
        ),
        "manual_latched_no_auto_recovery_proxy": HypotheticalGuardian(
            resume_policy="MANUAL_ONLY_LATCH"
        ),
    }
    outcomes = {}
    for label, members in native.ROSTERS.items():
        baseline_pnl, baseline_gross = derive_base_portfolio_curve(
            members, curves, gross_curves, {m: float(wallets[m]["train_frozen_k"]) for m in members}
        )
        outcomes[label] = {
            "members": list(members),
            "base_gross_max_usd": round(max(baseline_gross), 2),
            "predeclared_multiplier_scenarios": [
                compare_once(baseline_pnl, baseline_gross, multiplier=m) for m in multipliers
            ],
            "stress_and_current_latched_guardian_proxies": {
                title: [
                    hypothetical_guardian_overlay(
                        baseline_pnl, baseline_gross, multiplier=m, config=config
                    )
                    for m in STRESS_MULTIPLIERS
                ]
                for title, config in stress_policies.items()
            },
        }
    return {
        "schema": "guardian_24h_fixed_k_multiplier_retrospective_sensitivity_v1",
        "classification": "RESEARCH_ONLY_NO_TRADE_RETROSPECTIVE_SCENARIO",
        "simulated_policy_differs_from_current_guardian": True,
        "trading_authorized": False,
        "source_first_known_valid": False,
        "window_start_utc": native.START.isoformat(),
        "window_end_exclusive_utc": native.END.isoformat(),
        "train_end_exclusive_utc": native.TRAIN_END.isoformat(),
        "default_guardian_soft_daily_stop_usd": 250,
        "default_guardian_soft_static_floor_usd": 9600,
        "hard_daily_contract_loss_usd": 300,
        "hard_static_contract_floor_usd": 9500,
        "hypothetical_flatten_latency_minutes": 1,
        "minimum_pause_minutes": 1440,
        "stress_multiplier_values": STRESS_MULTIPLIERS,
        "current_protected_guardian_requires_signed_recovery": True,
        "wallets": wallets,
        "portfolios": outcomes,
        "market_coverage_note": "Local 1m + connected Vike + cached Vike. Missing source market checkpoint intents remain omitted; funding and broker TCA unverified.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    report = run_sweep()
    path = Path(args.output)
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "research_status": report["classification"],
                "wallet_ks": {name: v["train_frozen_k"] for name, v in report["wallets"].items()},
                "scenarios": {
                    name: [
                        {
                            "k_x": row["k_multiplier"],
                            "no_guardian_pnl": row["without_guardian"]["full_net_usd"],
                            "no_guardian_peak_dd": row["without_guardian"]["max_peak_drawdown_usd"],
                            "no_guardian_worst_day": row["without_guardian"][
                                "worst_utc_day_start_loss_usd"
                            ],
                            "guardian_proxy_pnl": row["with_hypothetical_24h_guardian"][
                                "full_net_usd"
                            ],
                            "guardian_proxy_halts": row["with_hypothetical_24h_guardian"][
                                "guardian_trigger_count"
                            ],
                            "guardian_proxy_breach": row["with_hypothetical_24h_guardian"][
                                "permanent_contract_breach"
                            ],
                        }
                        for row in data["predeclared_multiplier_scenarios"]
                    ]
                    for name, data in report["portfolios"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
