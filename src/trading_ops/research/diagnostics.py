"""Canonical ledger-backed backtest diagnostics shared across research studies."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import numpy as np
from copytrade_pm.facts import FactStore

from .backtest import BacktestReport, BacktestWindow
from .scoring import GRID_MS

__all__ = ["summarize_backtest_diagnostics"]


def summarize_backtest_diagnostics(
    report: BacktestReport,
    *,
    facts: FactStore,
    window: BacktestWindow,
    initial_equity: Decimal = Decimal(10_000),
    chop_start_day: str = "2026-10-01",
    allow_undated: bool = True,
    is_altcoin: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Summarize drawdowns, exact execution-bar participation, and EV blocks."""
    # Modeled execution occurs at fill.execution_ms (checkpoint_ms + GRID_MS), which at a
    # midnight-ending window falls on window.end_ms's UTC date.
    bars_end_day = datetime.fromtimestamp(window.end_ms / 1000, UTC).date().isoformat()
    bars_df = facts.scan(
        "bars_1m",
        start_day=window.start_day,
        end_day=bars_end_day,
        allow_unknown_known_at=allow_undated,
    ).collect()
    bar_lookup: dict[tuple[str, int], tuple[float, float]] = {}
    if bars_df.height > 0:
        for row in bars_df.select(["coin", "event_ts_ms", "close", "quote_volume_usd"]).iter_rows():
            bar_lookup[(str(row[0]).upper(), int(row[1]))] = (float(row[2]), float(row[3]))

    initial = float(initial_equity)
    ledger_by_ms = {row.checkpoint_ms: row for row in report.ledger}
    participations: list[float] = []
    unknown_vol_trades = 0
    core4_notional_usd = 0.0
    alt_short_notional_usd = 0.0
    alt_long_notional_usd = 0.0
    total_exec_notional_usd = 0.0

    for fill in report.fills:
        coin = fill.market.upper()
        ntl_usd = abs(float(fill.net_delta_qty)) * float(fill.executable_price)
        total_exec_notional_usd += ntl_usd
        if is_altcoin is not None:
            if is_altcoin(coin):
                if fill.net_delta_qty < 0:
                    alt_short_notional_usd += ntl_usd
                else:
                    alt_long_notional_usd += ntl_usd
            else:
                core4_notional_usd += ntl_usd

        # Missing-evidence invariant: query exact execution-deadline bar (never substitute adjacent bars).
        px_vol = bar_lookup.get((coin, fill.execution_ms))
        if px_vol is not None and px_vol[1] > 0.0:
            participations.append(ntl_usd / px_vol[1])
        else:
            unknown_vol_trades += 1

    day_events: dict[str, list[tuple[float, float]]] = {}
    peak_eq = initial
    max_p2t_dd = 0.0
    min_eq_overall = initial
    eq_now = initial
    cash_now = initial

    bar = window.start_ms
    while bar < window.end_ms:
        day_str = datetime.fromtimestamp(bar / 1000, UTC).date().isoformat()
        row = ledger_by_ms.get(bar)
        if row is not None:
            eq_now = float(row.equity)
            cash_now = float(row.cash)
        peak_eq = max(peak_eq, eq_now)
        max_p2t_dd = max(max_p2t_dd, peak_eq - eq_now)
        min_eq_overall = min(min_eq_overall, eq_now)
        day_events.setdefault(day_str, []).append((eq_now, cash_now))
        bar += GRID_MS

    static_floor_dd = max(0.0, initial - min_eq_overall)

    sorted_days = sorted(day_events.keys())
    n_days = len(sorted_days)
    is_chop = np.zeros(n_days, dtype=bool)
    day_end = np.zeros(n_days, dtype=np.float64)
    day_max = np.zeros(n_days, dtype=np.float64)
    day_min_eq = np.zeros(n_days, dtype=np.float64)
    day_swept = np.zeros(n_days, dtype=np.float64)
    day_max_first = np.ones(n_days, dtype=bool)
    day_min_bal = np.zeros(n_days, dtype=np.float64)

    prev_close_eq = initial
    for i, day in enumerate(sorted_days):
        pts = day_events[day]
        eqs = [p[0] - prev_close_eq for p in pts]
        bals = [p[1] - prev_close_eq for p in pts]
        is_chop[i] = day >= chop_start_day
        day_end[i] = eqs[-1]
        day_max[i] = max(0.0, max(eqs))
        day_min_eq[i] = min(0.0, min(eqs))
        day_min_bal[i] = min(0.0, min(bals))
        idx_max = int(np.argmax(eqs))
        idx_min = int(np.argmin(eqs))
        day_max_first[i] = idx_max <= idx_min
        prev_close_eq = pts[-1][0]

    blocks = (is_chop, day_end, day_max, day_min_eq, day_swept, day_max_first, day_min_bal)

    total_trades = len(report.fills)
    coverage = round(len(participations) / total_trades, 6) if total_trades > 0 else 1.0
    capacity_evidence_complete = unknown_vol_trades == 0
    if participations:
        arr = np.array(participations, dtype=np.float64)
        part_summary: dict[str, Any] = {
            "n_trades": len(participations),
            "total_trades": total_trades,
            "unknown_vol_trades": unknown_vol_trades,
            "unknown_px_trades": 0,
            "observed_volume_coverage": coverage,
            "capacity_evidence_complete": capacity_evidence_complete,
            "execution_time_convention": "CHECKPOINT_DEADLINE_MS",
            "median_1m": round(float(np.median(arr)), 6),
            "p95_1m": round(float(np.quantile(arr, 0.95)), 6),
            "max_1m": round(float(np.max(arr)), 6),
            "bars_above_1pct": int(np.sum(arr > 0.01)),
            "bars_above_5pct": int(np.sum(arr > 0.05)),
        }
    elif unknown_vol_trades > 0:
        # Fail closed: unknown capacity is None rather than 0.0 so missing evidence cannot pass gates.
        part_summary = {
            "n_trades": 0,
            "total_trades": total_trades,
            "unknown_vol_trades": unknown_vol_trades,
            "unknown_px_trades": 0,
            "observed_volume_coverage": coverage,
            "capacity_evidence_complete": False,
            "execution_time_convention": "CHECKPOINT_DEADLINE_MS",
            "median_1m": None,
            "p95_1m": None,
            "max_1m": None,
            "bars_above_1pct": 0,
            "bars_above_5pct": 0,
        }
    else:
        part_summary = {
            "n_trades": 0,
            "total_trades": 0,
            "unknown_vol_trades": 0,
            "unknown_px_trades": 0,
            "observed_volume_coverage": 1.0,
            "capacity_evidence_complete": True,
            "execution_time_convention": "CHECKPOINT_DEADLINE_MS",
            "median_1m": 0.0,
            "p95_1m": 0.0,
            "max_1m": 0.0,
            "bars_above_1pct": 0,
            "bars_above_5pct": 0,
        }

    result: dict[str, Any] = {
        "peak_to_trough_dd_usd": round(max_p2t_dd, 2),
        "static_floor_dd_usd": round(static_floor_dd, 2),
        "participation": part_summary,
        "blocks": blocks,
        "challenge_outcome": report.challenge_outcome,
        "challenge_target_reached": report.challenge_target_reached,
        "survival_no_breach": report.challenge_outcome != "BREACHED",
        "portfolio_constraints_valid": report.portfolio_constraints_valid,
    }
    if is_altcoin is not None:
        result["intent_notional_mix"] = {
            "total_usd": round(total_exec_notional_usd, 2),
            "core4_share": round(core4_notional_usd / total_exec_notional_usd, 4)
            if total_exec_notional_usd > 0
            else 0.0,
            "alt_short_share": round(alt_short_notional_usd / total_exec_notional_usd, 4)
            if total_exec_notional_usd > 0
            else 0.0,
            "alt_long_share": round(alt_long_notional_usd / total_exec_notional_usd, 4)
            if total_exec_notional_usd > 0
            else 0.0,
        }
    return result
