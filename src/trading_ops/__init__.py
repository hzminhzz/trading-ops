"""Vectorized research layer over ``copytrade_pm`` state and ``copytrade_kernel`` decisions.

Two deep operations, both read-only:

* :func:`trading_ops.research.score_universe` — Polars/``float64`` Stage-1 wallet
  audit that writes :class:`copytrade_pm.roster.WalletScorecard` rows into
  ``RosterStore``.
* :func:`trading_ops.research.run_kernel_backtest` — Stage-2 walk-forward that
  reconciles the roster on a daily cadence and steps the exact kernel
  ``step_checkpoint`` every five minutes.

This package owns no financial math: every decision is
``copytrade_kernel.step_checkpoint``, and every roster transition is
``copytrade_pm.roster.reconcile_roster``.
"""

__all__ = ["research"]
