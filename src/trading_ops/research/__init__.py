"""Research operations over ``copytrade_pm`` state and ``copytrade_kernel``.

Public surface (two deep operations plus the golden identity)::

    from trading_ops.research import ScoreConfig, run_kernel_backtest, score_universe

    scorecard = score_universe(facts, as_of=as_of, lookback_days=120, config=ScoreConfig())
    scorecard.record(roster)            # RosterStore.record_scorecard, idempotent

    report = run_kernel_backtest(
        facts,
        window=BacktestWindow.from_utc(start, end),
        seats=seats,
        rulebook=RulebookCatalog.get(...).to_rulebook_fields(...) -> Rulebook,
    )

* :mod:`trading_ops.research.scoring` — vectorized Stage-1 ``float64`` audit.
* :mod:`trading_ops.research.backtest` — Stage-2 daily reconcile + 5m kernel walk.
* :mod:`trading_ops.research.golden` — the cross-engine ``decision_sha``.
* :mod:`trading_ops.research.fee_schedule` — commercial per-tier challenge fee lookup.
"""

from __future__ import annotations

from .backtest import (
    DAY_MS,
    BacktestConfig,
    BacktestReport,
    BacktestWindow,
    MarketSpec,
    run_kernel_backtest,
)
from .fee_schedule import (
    DEFAULT_FEE_SCHEDULE_PATH,
    FeeQuote,
    load_fee_schedule,
    lookup_tier_fee,
)
from .golden import (
    DECISION_SHA_SCHEMA,
    GoldenStep,
    decision_payload,
    decision_sha,
    ending_state_rows,
    golden_step,
    net_intent_rows,
    state_payload,
)
from .scoring import (
    BARS_PER_DAY,
    GRID_MS,
    ScoreAuditRow,
    ScoreConfig,
    ScoreExclusion,
    UniverseScorecard,
    bench_from_scorecard,
    score_universe,
)

__all__ = [
    "BARS_PER_DAY",
    "DAY_MS",
    "DECISION_SHA_SCHEMA",
    "DEFAULT_FEE_SCHEDULE_PATH",
    "GRID_MS",
    "BacktestConfig",
    "BacktestReport",
    "BacktestWindow",
    "FeeQuote",
    "GoldenStep",
    "MarketSpec",
    "ScoreAuditRow",
    "ScoreConfig",
    "ScoreExclusion",
    "UniverseScorecard",
    "bench_from_scorecard",
    "decision_payload",
    "decision_sha",
    "ending_state_rows",
    "golden_step",
    "load_fee_schedule",
    "lookup_tier_fee",
    "net_intent_rows",
    "run_kernel_backtest",
    "score_universe",
    "state_payload",
]
