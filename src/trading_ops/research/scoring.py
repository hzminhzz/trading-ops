"""Vectorized Stage-1 universe scoring (Polars / ``float64``) into the roster.

One deep operation: :func:`score_universe`. It reads a point-in-time
``copytrade_pm.facts.FactStore`` slice, computes the Stage-1 audit for every
wallet with observable facts, and returns
:class:`copytrade_pm.roster.WalletScorecard` values that
``RosterStore.record_scorecard`` accepts unchanged.

Discipline encoded here:

* **PIT only.** The slice is filtered by ``first_known_at_ms <= as_of``; a fill
  the store cannot date is invisible, never assumed earlier.
  A retrospective archive has no recorded capture clock at all.  Research over
  such a store must say so explicitly with ``allow_undated=True``; the resulting
  scorecard basis records ``pit = RETROSPECTIVE_UNDATED`` and the rows remain
  excluded from any strict PIT read.
* **Missing evidence is missing, not zero.** A wallet with no equity row, no
  dated order, or no derivable markout is *excluded* with a typed reason and
  reported in :attr:`UniverseScorecard.excluded`. It never receives a
  fabricated scorecard whose numbers would enter admission.
* **``float64`` only inside Stage-1.** Every value crosses into exact
  :class:`~decimal.Decimal` once, at the ``WalletScorecard`` boundary.
* **The roster owns thresholds, this module owns measurements.** A gate is
  emitted only for a measured quantity compared with a ``ScoreConfig`` value.

The default Stage-1 pipeline scores each wallet on its 5m bar-aligned
:class:`FollowerExecutionStream` (amending ADR 0004 §1 per ADR 0005): strips
non-``MarketSpec`` symbols, nets fills into ``GRID_MS`` rebalances with exact
coin inventory, evaluates ``suggested_k``-scaled participation against PIT
``bars_1m`` venue volume (auditing any tolerated gaps below
``max_missing_volume_frac`` in ``UniverseScorecard.notes`` and excluding above
it via ``VENUE_VOLUME_UNKNOWN``), and scores follower trajectory edge alongside
Euler marginal CDaR. Upstream ``WalletScorecard`` / ``ScoreAuditRow`` boundary
field names (``p95_participation_1m``, ``markout_5m_bps``, ``markout_fills``)
are preserved for ``RosterStore`` compatibility while ``UniverseScorecard.basis``
and ``ScoreConfig.data`` pin the active measurement mode.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal
from typing import Any

import numpy as np
import polars as pl
from copytrade_kernel import KnownAtTs
from copytrade_pm.facts import FactSlice, FactStore, ProvenanceTier
from copytrade_pm.roster import GateFailure, WalletScorecard, digest

__all__ = [
    "BARS_PER_DAY",
    "GRID_MS",
    "FollowerExecutionStream",
    "ScoreAuditRow",
    "ScoreConfig",
    "ScoreExclusion",
    "UniverseScorecard",
    "score_universe",
]

GRID_MS = 300_000
BARS_PER_DAY = 288
_MICRO = Decimal("0.000001")
_NON_MARKET_SYMBOL_RE = r"[/:\s]"
PARTICIPATION_BASIS = "FACT_STORE_BARS_5M_FOLLOWER_K_SCALED"
LEGACY_PARTICIPATION_BASIS = "FACT_STORE_BARS_1M_VENUE_VOLUME"
EXPECTANCY_BASIS_BY_MODE: dict[str, str] = {
    "trajectory": "FOLLOWER_BAR_TRAJECTORY_EDGE",
    "bar_markout": "FOLLOWER_BAR_5M_MARKOUT",
    "raw_markout": "FACT_STORE_BARS_1M_CLOSE_5M",
}


@dataclass(frozen=True, slots=True)
class FollowerExecutionStream:
    """Bar-aligned copy-follower execution stream on the 5m checkpoint grid."""

    panel: pl.DataFrame
    rebalances: pl.DataFrame

    def filter_wallets(self, wallets: Sequence[str] | pl.DataFrame) -> FollowerExecutionStream:
        """Restrict both bar panel and rebalance stream to ``wallets``."""
        if isinstance(wallets, pl.DataFrame):
            selected = wallets.select("wallet")
            return FollowerExecutionStream(
                panel=self.panel.join(selected, on="wallet", how="inner"),
                rebalances=self.rebalances.join(selected, on="wallet", how="inner"),
            )
        allowed = list(wallets)
        return FollowerExecutionStream(
            panel=self.panel.filter(pl.col("wallet").is_in(allowed)),
            rebalances=self.rebalances.filter(pl.col("wallet").is_in(allowed)),
        )


@dataclass(frozen=True, slots=True)
class ScoreConfig:
    """Stage-1 research thresholds; each one is a documented research choice."""

    min_account_value_usd: Decimal = Decimal(1000)
    min_baseline_order_usd: Decimal = Decimal(15)
    max_euler_cdar_share: Decimal = Decimal("0.45")
    max_p95_participation_1m: Decimal = Decimal("0.01")
    max_max_participation_1m: Decimal = Decimal("0.05")
    active_days_window_days: int = 14
    pod_equity_usd: Decimal = Decimal(10000)
    risk_unit_usd: Decimal = Decimal(200)
    cluster_count: int = 3
    cdar_alpha: float = 0.95
    max_fixed_k: Decimal = Decimal(10)
    min_markout_fills: int = 20
    exclude_non_market_symbols: bool = True
    net_orders_by_bar: bool = True
    participation_scale_by_k: bool = True
    max_missing_volume_frac: Decimal = Decimal("0.02")
    expectancy_mode: str = "trajectory"
    decouple_sizing_gates: bool = False

    def __post_init__(self) -> None:
        for name in (
            "min_account_value_usd",
            "min_baseline_order_usd",
            "pod_equity_usd",
            "risk_unit_usd",
            "max_fixed_k",
        ):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
                raise ValueError(f"INVALID_SCORE_CONFIG:{name}")
        for name in (
            "max_euler_cdar_share",
            "max_p95_participation_1m",
            "max_max_participation_1m",
        ):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not Decimal(0) < value <= Decimal(1):
                raise ValueError(f"INVALID_SCORE_CONFIG:{name}")
        for flag in (
            "exclude_non_market_symbols",
            "net_orders_by_bar",
            "participation_scale_by_k",
            "decouple_sizing_gates",
        ):
            if type(getattr(self, flag)) is not bool:
                raise ValueError(f"INVALID_SCORE_CONFIG:{flag}")
        frac = self.max_missing_volume_frac
        if not isinstance(frac, Decimal) or not frac.is_finite() or not Decimal(0) <= frac <= Decimal(1):
            raise ValueError("INVALID_SCORE_CONFIG:max_missing_volume_frac")
        if self.expectancy_mode not in EXPECTANCY_BASIS_BY_MODE:
            raise ValueError("INVALID_SCORE_CONFIG:expectancy_mode")
        if not 0.5 <= self.cdar_alpha < 1:
            raise ValueError("INVALID_SCORE_CONFIG:cdar_alpha")
        if self.active_days_window_days < 1 or self.cluster_count < 1:
            raise ValueError("INVALID_SCORE_CONFIG:window")
        object.__setattr__(self, "active_days_window_days", min(self.active_days_window_days, 14))

    @property
    def data(self) -> dict[str, Any]:
        return {
            "min_account_value_usd": str(self.min_account_value_usd),
            "min_baseline_order_usd": str(self.min_baseline_order_usd),
            "max_euler_cdar_share": str(self.max_euler_cdar_share),
            "max_p95_participation_1m": str(self.max_p95_participation_1m),
            "max_max_participation_1m": str(self.max_max_participation_1m),
            "active_days_window_days": self.active_days_window_days,
            "pod_equity_usd": str(self.pod_equity_usd),
            "risk_unit_usd": str(self.risk_unit_usd),
            "cluster_count": self.cluster_count,
            "cdar_alpha": self.cdar_alpha,
            "max_fixed_k": str(self.max_fixed_k),
            "min_markout_fills": self.min_markout_fills,
            "exclude_non_market_symbols": self.exclude_non_market_symbols,
            "net_orders_by_bar": self.net_orders_by_bar,
            "participation_scale_by_k": self.participation_scale_by_k,
            "max_missing_volume_frac": str(self.max_missing_volume_frac),
            "expectancy_mode": self.expectancy_mode,
            "decouple_sizing_gates": self.decouple_sizing_gates,
        }


@dataclass(frozen=True, slots=True)
class ScoreExclusion:
    """A wallet the audit refused to score, with the missing evidence named."""

    wallet: str
    reason: str
    observed: int = 0

    @property
    def data(self) -> dict[str, Any]:
        return {"wallet": self.wallet, "reason": self.reason, "observed": self.observed}


@dataclass(frozen=True, slots=True)
class ScoreAuditRow:
    """Measured Stage-1 quantities, kept next to (never inside) the scorecard."""

    wallet: str
    q10_order_usd: Decimal
    markout_5m_bps: Decimal
    markout_fills: int
    p95_participation_1m: Decimal
    max_participation_1m: Decimal
    cdar_share: Decimal
    wallet_max_dd_usd: Decimal
    suggested_k: Decimal
    active_days_14d: int

    @property
    def data(self) -> dict[str, Any]:
        return {
            "wallet": self.wallet,
            "q10_order_usd": str(self.q10_order_usd),
            "markout_5m_bps": str(self.markout_5m_bps),
            "markout_fills": self.markout_fills,
            "p95_participation_1m": str(self.p95_participation_1m),
            "max_participation_1m": str(self.max_participation_1m),
            "cdar_share": str(self.cdar_share),
            "wallet_max_dd_usd": str(self.wallet_max_dd_usd),
            "suggested_k": str(self.suggested_k),
            "active_days_14d": self.active_days_14d,
        }


def _basis_for(config: ScoreConfig, *, undated: bool) -> dict[str, str]:
    participation = (
        PARTICIPATION_BASIS
        if config.net_orders_by_bar and config.participation_scale_by_k
        else LEGACY_PARTICIPATION_BASIS
    )
    basis = {
        "participation": participation,
        "cdar_share": "EULER_MARGINAL_CDAR_COHORT",
        "markout": EXPECTANCY_BASIS_BY_MODE[config.expectancy_mode],
        "pit": "RETROSPECTIVE_UNDATED" if undated else "STRICT_KNOWN_AT",
    }
    if config.decouple_sizing_gates:
        basis["sizing_gates"] = "DECOUPLED_PORTFOLIO_SEAM"
    return basis


@dataclass(frozen=True, slots=True)
class UniverseScorecard:
    """One Stage-1 audit: roster-ready cards plus the evidence behind them."""

    as_of: KnownAtTs
    lookback_days: int
    config: ScoreConfig
    cards: tuple[WalletScorecard, ...]
    audit: tuple[ScoreAuditRow, ...]
    excluded: tuple[ScoreExclusion, ...]
    tiers: tuple[str, ...]
    slice_days: int
    slice_rows: int
    source_sha: str
    notes: tuple[str, ...] = ()
    basis: dict[str, str] = field(default_factory=dict)
    undated: bool = False
    _returns_by_wallet: dict[str, tuple[float, ...]] = field(
        default_factory=dict, repr=False, compare=False
    )
    _unscaled_participation_by_wallet: dict[str, tuple[float, float]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def wallet_returns(self, wallet: str) -> tuple[float, ...]:
        """Return bar-aligned unit-scale wallet return series, or fail closed."""
        if wallet not in self._returns_by_wallet:
            raise ValueError(f"SCORECARD_MISSING_RETURNS:{wallet}")
        return self._returns_by_wallet[wallet]

    def wallet_unscaled_participation(self, wallet: str) -> tuple[float, float]:
        """Return unscaled ``(p95, max)`` venue participation shares, or fail closed."""
        if wallet not in self._unscaled_participation_by_wallet:
            raise ValueError(f"SCORECARD_MISSING_PARTICIPATION:{wallet}")
        return self._unscaled_participation_by_wallet[wallet]

    def __post_init__(self) -> None:
        if not self.basis:
            object.__setattr__(self, "basis", _basis_for(self.config, undated=self.undated))

    @property
    def by_wallet(self) -> dict[str, WalletScorecard]:
        return {card.wallet: card for card in self.cards}

    @property
    def eligible(self) -> tuple[WalletScorecard, ...]:
        return tuple(card for card in self.cards if card.eligible)

    def data(self) -> dict[str, Any]:
        return {
            "schema": "universe_scorecard_v1",
            "as_of_ms": self.as_of.value,
            "lookback_days": self.lookback_days,
            "config": self.config.data,
            "source_sha": self.source_sha,
            "undated": self.undated,
            "tiers": list(self.tiers),
            "slice_days": self.slice_days,
            "slice_rows": self.slice_rows,
            "basis": dict(sorted(self.basis.items())),
            "notes": list(self.notes),
            "cards": [card.data for card in self.cards],
            "excluded": [row.data for row in self.excluded],
            "audit": [row.data for row in self.audit],
        }

    def record(self, store: Any, *, recorded_ms: int | None = None) -> tuple[str, ...]:
        """Write every card through ``RosterStore.record_scorecard`` (idempotent)."""
        shas = []
        for card in self.cards:
            shas.append(store.record_scorecard(card, recorded_ms=recorded_ms))
        return tuple(shas)


# --------------------------------------------------------------------------- #
# helpers


def _dec(value: float | None, places: str = "0.000000001") -> Decimal:
    """Exact Decimal from a float metric, quantised once at the boundary."""
    if value is None or not math.isfinite(value):
        raise ValueError("NONFINITE_STAGE1_METRIC")
    return Decimal(repr(float(value))).quantize(Decimal(places), rounding=ROUND_DOWN)


def _utc_day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).date().isoformat()


def _signed_side(column: str) -> pl.Expr:
    """+1 buy / -1 sell, or null for an unrecognised venue verb (fail closed)."""
    normalised = pl.col(column).cast(pl.String).str.to_uppercase()
    return (
        pl.when(normalised.is_in(["B", "BUY", "LONG", "BID"]))
        .then(pl.lit(1.0))
        .when(normalised.is_in(["A", "S", "SELL", "SHORT", "ASK"]))
        .then(pl.lit(-1.0))
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def _build_follower_stream(orders: pl.DataFrame, *, pod_equity: float) -> FollowerExecutionStream:
    """Build the 5m bar-netted follower execution panel and nonzero rebalances."""
    ordered = orders.sort(["wallet", "coin_norm", "event_ts_ms", "arrival_seq"])
    bar_panel = (
        ordered.with_columns(
            (pl.col("side_sign") * pl.col("abs_usd") / pl.col("price")).alias("signed_qty"),
        )
        .group_by(["wallet", "coin_norm", "bar_ms"])
        .agg(
            pl.col("coin").last().alias("coin"),
            pl.col("day").last().alias("day"),
            pl.col("signed_qty").sum().alias("delta_qty"),
            pl.col("signed_usd").sum().alias("delta_usd"),
            pl.col("price").last().alias("bar_price"),
            pl.len().alias("fills_in_bar"),
        )
        .sort(["wallet", "coin_norm", "bar_ms"])
        .with_columns(
            pl.col("delta_qty").cum_sum().over(["wallet", "coin_norm"]).alias("position_qty"),
        )
        .with_columns(
            pl.col("position_qty")
            .shift(1)
            .over(["wallet", "coin_norm"])
            .fill_null(0.0)
            .alias("prior_qty"),
            pl.col("bar_price").shift(1).over(["wallet", "coin_norm"]).alias("_prior_price"),
        )
        .with_columns(
            (pl.col("prior_qty") * (pl.col("bar_price") - pl.col("_prior_price")))
            .fill_null(0.0)
            .fill_nan(0.0)
            .alias("pnl_usd"),
            pl.col("delta_usd").abs().alias("abs_delta_usd"),
            (pl.col("position_qty") * pl.col("bar_price")).alias("position_usd"),
            pl.when(pl.col("delta_usd") > 0.0)
            .then(pl.lit(1.0))
            .when(pl.col("delta_usd") < 0.0)
            .then(pl.lit(-1.0))
            .otherwise(pl.lit(0.0))
            .alias("side_sign"),
        )
        .with_columns(
            (pl.col("position_usd") / pod_equity).alias("notional_frac"),
            (pl.col("pnl_usd") / pod_equity).alias("net_ret_frac"),
        )
        .drop("_prior_price")
    )
    rebalances = bar_panel.filter(pl.col("abs_delta_usd") > 0.0).with_columns(
        pl.col("bar_ms").alias("event_ts_ms"),
        pl.col("bar_price").alias("price"),
        pl.col("abs_delta_usd").alias("abs_usd"),
    )
    return FollowerExecutionStream(panel=bar_panel, rebalances=rebalances)


def euler_marginal_cdar_shares(
    returns: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Exact Euler additive marginal CDaR shares over weighted cohort returns."""
    n = returns.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if returns.ndim != 2 or returns.shape[1] == 0:
        return np.zeros(n, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError("INVALID_EULER_WEIGHTS")
    scaled = returns * w[:, None]
    equity = np.cumsum(scaled, axis=1)
    equity_pad = np.concatenate([np.zeros((n, 1), dtype=np.float64), equity], axis=1)
    port_pad = equity_pad.sum(axis=0)
    t_len = returns.shape[1]
    peak_idx = np.empty(t_len, dtype=np.int64)
    best_idx = 0
    best_val = 0.0
    for t in range(t_len):
        val = float(port_pad[t + 1])
        if val >= best_val:
            best_val = val
            best_idx = t + 1
        peak_idx[t] = best_idx
    drawdown = port_pad[peak_idx] - port_pad[1:]
    if not np.any(drawdown > 0.0):
        return np.zeros(n, dtype=np.float64)
    cut = float(np.quantile(drawdown, alpha))
    tail = np.flatnonzero(drawdown >= cut)
    if tail.size == 0:
        tail = np.array([int(np.argmax(drawdown))], dtype=np.int64)
    port_cdar = float(drawdown[tail].mean())
    if port_cdar <= 0.0:
        return np.zeros(n, dtype=np.float64)
    contrib = (equity_pad[:, peak_idx[tail]] - equity[:, tail]).mean(axis=1)
    return contrib / port_cdar


def peak_drawdown(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    curve = np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(values)])
    peak = np.maximum.accumulate(curve)
    return float(np.max(peak - curve))


def sizing_gate_failures(
    raw_p95: float,
    raw_max: float,
    raw_cdar_share: float,
    *,
    max_p95: Decimal,
    max_max: Decimal,
    max_cdar: Decimal,
) -> tuple[GateFailure, ...]:
    """Evaluate participation and Euler marginal CDaR limits consistently."""
    failures: list[GateFailure] = []
    if raw_p95 > float(max_p95) or raw_max > float(max_max):
        failures.append(GateFailure.PARTICIPATION_LIMIT)
    if Decimal(repr(max(raw_cdar_share, 0.0))) > max_cdar:
        failures.append(GateFailure.MARGINAL_CDAR_LIMIT)
    return tuple(failures)


def _upgma_labels(dist: np.ndarray, clusters: int) -> list[int]:
    """Deterministic UPGMA over a symmetric distance matrix (``float64``)."""
    n = dist.shape[0]
    if n == 0:
        return []
    groups: list[list[int]] = [[index] for index in range(n)]
    work = dist.astype(np.float64, copy=True)
    np.fill_diagonal(work, np.inf)
    while len(groups) > clusters and len(groups) > 1:
        flat = int(np.argmin(work))
        i, j = divmod(flat, work.shape[1])
        if i > j:
            i, j = j, i
        merged = groups[i] + groups[j]
        size_i, size_j = len(groups[i]), len(groups[j])
        for k in range(len(groups)):
            if k in (i, j):
                continue
            value = (work[i, k] * size_i + work[j, k] * size_j) / (size_i + size_j)
            work[i, k] = work[k, i] = value
        work[i, j] = work[j, i] = np.inf
        work[j, :] = np.inf
        work[:, j] = np.inf
        groups[i] = merged
        groups.pop(j)
        work = np.delete(np.delete(work, j, axis=0), j, axis=1)
    labels = [0] * n
    for label, group in enumerate(groups):
        for index in group:
            labels[index] = label
    return labels


def _cluster_labels(matrix: np.ndarray, clusters: int) -> list[str]:
    """Correlation-distance UPGMA labels; ``upgma_0..upgma_{n-1}``."""
    n = matrix.shape[0]
    if n == 0:
        return []
    if n == 1:
        return ["upgma_0"]
    centred = matrix - matrix.mean(axis=1, keepdims=True)
    scale = np.sqrt((centred**2).sum(axis=1, keepdims=True))
    scale[scale == 0.0] = 1.0
    unit = centred / scale
    corr = np.clip(unit @ unit.T, -1.0, 1.0)
    dist = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - corr)))
    np.fill_diagonal(dist, 0.0)
    return [f"upgma_{label}" for label in _upgma_labels(dist, max(1, min(clusters, n)))]


def _suggested_k(drawdown_usd: float, config: ScoreConfig) -> tuple[Decimal, bool]:
    """k such that the modeled wallet drawdown uses half of one seat's risk unit."""
    if drawdown_usd <= 0.0:
        return config.max_fixed_k, True
    raw = config.risk_unit_usd / (Decimal(2) * _dec(drawdown_usd))
    if raw > config.max_fixed_k:
        return config.max_fixed_k, False
    if raw <= 0:
        raise ValueError("INVALID_SUGGESTED_K")
    return raw.quantize(_MICRO, rounding=ROUND_DOWN), False


def _slice(facts: FactStore, kind: str, **kwargs: Any) -> FactSlice:
    return facts.scan(kind, **kwargs)


def _q10_order_stats(
    stream: FollowerExecutionStream, raw_orders: pl.DataFrame, *, net_by_bar: bool
) -> pl.DataFrame:
    """Compute per-wallet 10th-percentile order size on follower rebalances or raw fills."""
    if not net_by_bar:
        return raw_orders.group_by("wallet").agg(
            pl.col("abs_usd").quantile(0.10, interpolation="nearest").alias("q10_order_usd"),
            pl.len().alias("orders"),
        )
    all_wallets = raw_orders.group_by("wallet").agg(pl.len().alias("raw_fills"))
    rebalance_q10 = stream.rebalances.group_by("wallet").agg(
        pl.col("abs_usd").quantile(0.10, interpolation="nearest").alias("q10_order_usd"),
        pl.len().alias("rebalances"),
    )
    return all_wallets.join(rebalance_q10, on="wallet", how="left").select(
        pl.col("wallet"),
        pl.col("q10_order_usd").fill_null(0.0).alias("q10_order_usd"),
        pl.coalesce(pl.col("rebalances"), pl.col("raw_fills")).cast(pl.Int64).alias("orders"),
    )


def _venue_volume_panel(
    orders: pl.DataFrame,
    bars: pl.DataFrame,
    *,
    bucket_ms: int,
    max_missing_frac: float,
) -> tuple[pl.DataFrame, list[ScoreExclusion], list[str]]:
    """Join demand to PIT venue quote volume; audit gaps and exclude above cap."""
    demand = (
        orders.with_columns(
            (pl.col("event_ts_ms") // bucket_ms * bucket_ms).alias("venue_bucket_ms"),
        )
        .group_by(["wallet", "coin_norm", "venue_bucket_ms"])
        .agg(pl.col("abs_usd").sum().alias("wallet_usd"))
    )
    if bars.height > 0:
        venue = (
            bars.with_columns(
                pl.col("coin").str.strip_chars().str.to_uppercase().alias("coin_norm"),
                (pl.col("event_ts_ms").cast(pl.Int64) // bucket_ms * bucket_ms).alias(
                    "venue_bucket_ms"
                ),
                pl.col("quote_volume_usd").cast(pl.Float64).alias("quote_volume_usd"),
            )
            .filter(pl.col("quote_volume_usd").is_finite() & (pl.col("quote_volume_usd") > 0.0))
            .group_by(["coin_norm", "venue_bucket_ms"])
            .agg(pl.col("quote_volume_usd").sum().alias("quote_volume_usd"))
        )
        joined = demand.join(venue, on=["coin_norm", "venue_bucket_ms"], how="left")
    else:
        joined = demand.with_columns(pl.lit(None, dtype=pl.Float64).alias("quote_volume_usd"))

    is_missing = (
        pl.col("quote_volume_usd").is_null()
        | ~pl.col("quote_volume_usd").is_finite()
        | (pl.col("quote_volume_usd") <= 0.0)
    )
    coverage = joined.group_by("wallet").agg(
        pl.len().alias("total"),
        is_missing.cast(pl.Int64).sum().alias("missing"),
    )
    exclusions: list[ScoreExclusion] = []
    volume_notes: list[str] = []
    for row in coverage.iter_rows(named=True):
        missing = int(row["missing"])
        total = int(row["total"])
        if missing <= 0:
            continue
        if (total - missing) == 0 or (float(missing) / float(total)) > max_missing_frac:
            exclusions.append(
                ScoreExclusion(wallet=row["wallet"], reason="VENUE_VOLUME_UNKNOWN", observed=missing)
            )
        else:
            volume_notes.append(f"TOLERATED_MISSING_VOLUME_BARS:{row['wallet']}:{missing}/{total}")
    excluded_wallets = {row.wallet for row in exclusions}
    valid = joined.filter(~pl.col("wallet").is_in(list(excluded_wallets)) & ~is_missing)
    return valid, exclusions, volume_notes


def _participation_shares(
    valid_volume: pl.DataFrame, *, scale_by_wallet: dict[str, float]
) -> dict[str, tuple[float, float]]:
    """Per-wallet follower-scaled p95/max share of PIT venue quote volume."""
    if valid_volume.height == 0:
        return {}
    scale_df = pl.DataFrame(
        {
            "wallet": list(scale_by_wallet.keys()),
            "k_scale": [float(v) for v in scale_by_wallet.values()],
        },
        schema={"wallet": pl.String, "k_scale": pl.Float64},
    )
    agg = (
        valid_volume.join(scale_df, on="wallet", how="left")
        .with_columns(
            (
                (pl.col("wallet_usd") / pl.col("quote_volume_usd"))
                * pl.col("k_scale").fill_null(1.0)
            ).alias("follower_share")
        )
        .group_by("wallet")
        .agg(
            pl.col("follower_share").quantile(0.95, interpolation="nearest").alias("p95_share"),
            pl.col("follower_share").max().alias("max_share"),
        )
    )
    return {
        row["wallet"]: (float(row["p95_share"]), float(row["max_share"]))
        for row in agg.iter_rows(named=True)
    }


def _trajectory_expectancy(panel: pl.DataFrame, bars: pl.DataFrame) -> pl.DataFrame:
    """Follower bar-trajectory PnL per dollar of rebalanced notional (bps)."""
    empty_schema = {"wallet": pl.String, "expectancy_bps": pl.Float64, "observations": pl.UInt32}
    if panel.height == 0:
        return pl.DataFrame(schema=empty_schema)
    tail_pnl = (
        panel.sort(["wallet", "coin_norm", "bar_ms"])
        .group_by(["wallet", "coin_norm"])
        .agg(
            pl.col("bar_ms").last().alias("last_bar_ms"),
            pl.col("position_qty").last().alias("last_qty"),
            pl.col("bar_price").last().alias("last_price"),
        )
    )
    if bars.height > 0:
        latest_closes = (
            bars.with_columns(
                pl.col("coin").str.strip_chars().str.to_uppercase().alias("coin_norm"),
                pl.col("event_ts_ms").cast(pl.Int64).alias("close_ts_ms"),
                pl.col("close").cast(pl.Float64).alias("latest_close"),
            )
            .filter(pl.col("latest_close").is_finite() & (pl.col("latest_close") > 0.0))
            .sort(["coin_norm", "close_ts_ms", "venue"])
            .group_by("coin_norm")
            .agg(
                pl.col("close_ts_ms").last(),
                pl.col("latest_close").last(),
            )
        )
        tail_pnl = (
            tail_pnl.join(latest_closes, on="coin_norm", how="left")
            .with_columns(
                pl.when(
                    pl.col("latest_close").is_not_null()
                    & (pl.col("close_ts_ms") > pl.col("last_bar_ms"))
                    & (pl.col("last_qty").abs() > 1e-12)
                )
                .then(pl.col("last_qty") * (pl.col("latest_close") - pl.col("last_price")))
                .otherwise(0.0)
                .alias("tail_pnl_usd")
            )
            .group_by("wallet")
            .agg(pl.col("tail_pnl_usd").sum().alias("tail_pnl_usd"))
        )
    else:
        tail_pnl = tail_pnl.group_by("wallet").agg(pl.lit(0.0).alias("tail_pnl_usd"))

    base = panel.group_by("wallet").agg(
        pl.col("pnl_usd").sum().alias("realized_mtm_pnl_usd"),
        pl.col("abs_delta_usd").sum().alias("turnover_usd"),
        (pl.col("prior_qty").abs() > 1e-12).cast(pl.UInt32).sum().alias("observations"),
    )
    return (
        base.join(tail_pnl, on="wallet", how="left")
        .filter(pl.col("turnover_usd") > 0.0)
        .with_columns(
            (
                (pl.col("realized_mtm_pnl_usd") + pl.col("tail_pnl_usd").fill_null(0.0))
                / pl.col("turnover_usd")
                * 10_000.0
            ).alias("expectancy_bps")
        )
        .select("wallet", "expectancy_bps", "observations")
    )


def _markout(orders: pl.DataFrame, bars: pl.DataFrame) -> pl.DataFrame:
    """5m markout in bps against PIT ``bars_1m`` close prices."""
    empty_schema = {"wallet": pl.String, "expectancy_bps": pl.Float64, "observations": pl.UInt32}
    if bars.height == 0 or orders.height == 0:
        return pl.DataFrame(schema=empty_schema)
    closes = (
        bars.with_columns(
            pl.col("coin").str.strip_chars().str.to_uppercase().alias("coin_norm"),
            pl.col("event_ts_ms").cast(pl.Int64),
            pl.col("close").cast(pl.Float64).alias("future_price"),
        )
        .filter(pl.col("future_price").is_finite() & (pl.col("future_price") > 0))
        .sort(["coin_norm", "event_ts_ms", "venue"])
        .group_by(["coin_norm", "event_ts_ms"])
        .agg(pl.col("future_price").last())
        .sort(["coin_norm", "event_ts_ms"])
    )
    if closes.height == 0:
        return pl.DataFrame(schema=empty_schema)
    with_horizon = orders.with_columns(
        (pl.col("event_ts_ms") + GRID_MS).alias("target_ts_ms")
    ).sort(["coin_norm", "target_ts_ms"])
    joined = with_horizon.join_asof(
        closes,
        left_on="target_ts_ms",
        right_on="event_ts_ms",
        by="coin_norm",
        strategy="forward",
        check_sortedness=False,
    ).with_columns(
        (
            pl.col("side_sign") * (pl.col("future_price") / pl.col("price") - 1.0) * 10_000.0
        ).alias("expectancy_bps")
    )
    return (
        joined.drop_nulls("expectancy_bps")
        .group_by("wallet")
        .agg(
            pl.col("expectancy_bps").mean().alias("expectancy_bps"),
            pl.len().cast(pl.UInt32).alias("observations"),
        )
    )


_EXPECTANCY_EVALUATORS: dict[
    str, Callable[[FollowerExecutionStream, pl.DataFrame, pl.DataFrame], pl.DataFrame]
] = {
    "trajectory": lambda stream, _raw, bars: _trajectory_expectancy(stream.panel, bars),
    "bar_markout": lambda stream, _raw, bars: _markout(stream.rebalances, bars),
    "raw_markout": lambda _stream, raw_orders, bars: _markout(raw_orders, bars),
}


def _score_expectancy(
    stream: FollowerExecutionStream, raw_orders: pl.DataFrame, bars: pl.DataFrame, *, mode: str
) -> pl.DataFrame:
    return _EXPECTANCY_EVALUATORS[mode](stream, raw_orders, bars)


# --------------------------------------------------------------------------- #
# public operation


def score_universe(
    facts: FactStore,
    *,
    as_of: KnownAtTs,
    lookback_days: int,
    config: ScoreConfig | None = None,
    tiers: Iterable[ProvenanceTier] | None = None,
    wallets: Sequence[str] | None = None,
    allow_undated: bool = False,
) -> UniverseScorecard:
    """Vectorized Stage-1 audit of every wallet with point-in-time evidence.

    Returns roster-ready :class:`WalletScorecard` values; record them with
    :meth:`UniverseScorecard.record` (``RosterStore.record_scorecard``).

    ``allow_undated=True`` admits archive rows the store never dated (the
    Reservoir backfill).  That is a retrospective research mode, recorded in the
    scorecard basis, and never a point-in-time claim.
    """
    if not isinstance(facts, FactStore):
        raise TypeError("FACT_STORE_REQUIRED")
    if not isinstance(as_of, KnownAtTs):
        raise TypeError("INVALID_CLOCK")
    if type(lookback_days) is not int or lookback_days < 1:
        raise ValueError("INVALID_LOOKBACK")
    if type(allow_undated) is not bool:
        raise ValueError("INVALID_UNDATED_FLAG")
    settings = config or ScoreConfig()
    if not isinstance(settings, ScoreConfig):
        raise TypeError("INVALID_SCORE_CONFIG")
    observed_tiers = tuple(
        tier.value
        for tier in (
            tuple(tiers)
            if tiers is not None
            else (ProvenanceTier.OBSERVED_FETCH, ProvenanceTier.RETROSPECTIVE)
        )
    )
    start_ms = as_of.value - lookback_days * 86_400_000
    start_day, end_day = _utc_day(start_ms), _utc_day(as_of.value)

    fills = facts.scan(
        "fills",
        wallets=wallets,
        start_day=start_day,
        end_day=end_day,
        as_of_ms=as_of.value,
        tiers=tiers,
        allow_unknown_known_at=allow_undated,
    )
    equity = facts.scan(
        "equity",
        wallets=wallets,
        start_day=start_day,
        end_day=end_day,
        as_of_ms=as_of.value,
        tiers=tiers,
        allow_unknown_known_at=allow_undated,
    )
    raw = fills.collect().filter(
        (pl.col("event_ts_ms") >= start_ms) & (pl.col("event_ts_ms") < as_of.value)
    )
    equity_frame = equity.collect().filter(pl.col("event_ts_ms") <= as_of.value)

    def _empty_universe(
        slice_rows: int, empty_excluded: tuple[ScoreExclusion, ...], empty_notes: list[str]
    ) -> UniverseScorecard:
        return UniverseScorecard(
            as_of=as_of,
            lookback_days=lookback_days,
            config=settings,
            cards=(),
            audit=(),
            excluded=empty_excluded,
            tiers=observed_tiers,
            slice_days=len(fills.days),
            slice_rows=slice_rows,
            undated=allow_undated,
            source_sha=digest(
                {
                    "as_of_ms": as_of.value,
                    "lookback_days": lookback_days,
                    "config": settings.data,
                    "slice": fills.as_dict(),
                    "rows": slice_rows,
                }
            ),
            notes=tuple(sorted(set(empty_notes))),
        )

    if raw.height == 0:
        return _empty_universe(0, (), ["NO_FILL_ROWS"])

    orders = (
        raw.with_columns(
            _signed_side("side").alias("side_sign"),
            pl.coalesce(
                pl.col("size_usd").cast(pl.Float64),
                (pl.col("size").cast(pl.Float64) * pl.col("price").cast(pl.Float64)),
            ).alias("abs_usd"),
            pl.col("price").cast(pl.Float64).alias("price"),
            pl.col("event_ts_ms").cast(pl.Int64),
            (pl.col("event_ts_ms") // GRID_MS * GRID_MS).alias("bar_ms"),
            (pl.col("event_ts_ms") // 86_400_000).alias("day"),
        )
        .filter(
            pl.col("side_sign").is_not_null()
            & pl.col("price").is_finite()
            & (pl.col("price") > 0)
            & (pl.col("abs_usd") > 0)
        )
        .with_columns(
            (pl.col("side_sign") * pl.col("abs_usd")).alias("signed_usd"),
            pl.col("coin").str.strip_chars().str.to_uppercase().alias("coin_norm"),
        )
    )

    dropped = raw.height - orders.height
    notes: list[str] = []
    if dropped:
        notes.append(f"UNSCORED_FILLS_UNKNOWN_SIDE_OR_PRICE:{dropped}")
    excluded: list[ScoreExclusion] = []

    if settings.exclude_non_market_symbols and orders.height > 0:
        non_market_mask = pl.col("coin_norm").str.contains(_NON_MARKET_SYMBOL_RE) | (
            pl.col("coin_norm") == ""
        )
        non_market = orders.filter(non_market_mask)
        if non_market.height > 0:
            notes.append(f"UNSCORED_NON_MARKET_SYMBOLS:{non_market.height}")
            orders = orders.filter(~non_market_mask)
            tradable_wallets = orders.select("wallet").unique()
            only_non_market = (
                non_market.join(tradable_wallets, on="wallet", how="anti")
                .group_by("wallet")
                .agg(pl.len().alias("observed"))
            )
            for row in only_non_market.iter_rows(named=True):
                excluded.append(
                    ScoreExclusion(
                        wallet=row["wallet"],
                        reason="NO_TRADABLE_MARKET_FILLS",
                        observed=int(row["observed"]),
                    )
                )

    if orders.height == 0:
        return _empty_universe(
            raw.height,
            tuple(sorted(excluded, key=lambda row: (row.wallet, row.reason))),
            notes + ["NO_SCORABLE_WALLET"],
        )

    orders = orders.sort(["wallet", "coin_norm", "event_ts_ms", "arrival_seq"])
    pod_equity = float(settings.pod_equity_usd)
    stream = _build_follower_stream(orders, pod_equity=pod_equity)

    q10 = _q10_order_stats(stream, orders, net_by_bar=settings.net_orders_by_bar)
    eligible_q10 = q10.filter(pl.col("q10_order_usd") >= float(settings.min_baseline_order_usd))
    for row in q10.filter(
        pl.col("q10_order_usd") < float(settings.min_baseline_order_usd)
    ).iter_rows(named=True):
        excluded.append(
            ScoreExclusion(
                wallet=row["wallet"],
                reason="BASELINE_ORDER_BELOW_FLOOR",
                observed=int(row["orders"]),
            )
        )
    eligible_q10_wallets = eligible_q10.select("wallet")
    orders = orders.join(eligible_q10_wallets, on="wallet", how="inner")
    stream = stream.filter_wallets(eligible_q10_wallets)

    account_value: dict[str, Decimal] = {}
    if equity_frame.height:
        latest = (
            equity_frame.sort("event_ts_ms")
            .group_by("wallet")
            .agg(pl.col("equity_usd").cast(pl.Float64).last().alias("equity_usd"))
        )
        for row in latest.iter_rows(named=True):
            if row["equity_usd"] is not None and math.isfinite(row["equity_usd"]):
                account_value[row["wallet"]] = _dec(row["equity_usd"], "0.01")

    scored: list[str] = []
    for wallet in sorted(orders["wallet"].unique().to_list()):
        if wallet not in account_value:
            excluded.append(ScoreExclusion(wallet=wallet, reason="ACCOUNT_VALUE_UNKNOWN"))
            continue
        scored.append(wallet)
    if not scored:
        return _empty_universe(
            raw.height,
            tuple(sorted(excluded, key=lambda row: (row.wallet, row.reason))),
            notes + ["NO_SCORABLE_WALLET"],
        )

    orders = orders.filter(pl.col("wallet").is_in(scored))
    stream = stream.filter_wallets(scored)
    traded_coins = sorted(orders["coin_norm"].unique().to_list())
    bars_slice = facts.scan(
        "bars_1m",
        coins=traded_coins,
        start_day=start_day,
        end_day=end_day,
        as_of_ms=as_of.value,
        tiers=tiers,
        allow_unknown_known_at=allow_undated,
    )
    bars_frame = bars_slice.collect()

    participation_source = stream.rebalances if settings.net_orders_by_bar else orders
    bucket_ms = GRID_MS if settings.net_orders_by_bar else 60_000
    valid_volume, volume_exclusions, volume_notes = _venue_volume_panel(
        participation_source,
        bars_frame,
        bucket_ms=bucket_ms,
        max_missing_frac=float(settings.max_missing_volume_frac),
    )
    if volume_notes:
        notes.extend(volume_notes)
    if volume_exclusions:
        excluded.extend(volume_exclusions)
        excluded_volume_wallets = {row.wallet for row in volume_exclusions}
        scored = [wallet for wallet in scored if wallet not in excluded_volume_wallets]
        orders = orders.filter(pl.col("wallet").is_in(scored))
        stream = stream.filter_wallets(scored)
    if not scored:
        return _empty_universe(
            raw.height,
            tuple(sorted(excluded, key=lambda row: (row.wallet, row.reason))),
            notes + ["NO_SCORABLE_WALLET"],
        )

    panel = stream.panel
    bars = panel.select("bar_ms").unique().sort("bar_ms")["bar_ms"].to_numpy()
    width = len(bars)
    index = {bar: position for position, bar in enumerate(bars)}
    returns = np.zeros((len(scored), width), dtype=np.float64)
    for row_index, wallet in enumerate(scored):
        rows = panel.filter(pl.col("wallet") == wallet)
        grouped = rows.group_by("bar_ms").agg(
            pl.col("net_ret_frac").sum(),
            pl.col("delta_usd").sum().alias("delta_usd"),
        )
        positions = [index[bar] for bar in grouped["bar_ms"].to_list()]
        returns[row_index, positions] = grouped["net_ret_frac"].to_numpy()

    drawdowns = [peak_drawdown(returns[row_index]) * pod_equity for row_index in range(len(scored))]
    k_results = [_suggested_k(drawdown, settings) for drawdown in drawdowns]
    scale_by_wallet = {
        wallet: (float(k_results[row_index][0]) if settings.participation_scale_by_k else 1.0)
        for row_index, wallet in enumerate(scored)
    }
    participation_rows = _participation_shares(valid_volume, scale_by_wallet=scale_by_wallet)
    unscaled_participation_rows = (
        participation_rows
        if not settings.participation_scale_by_k
        else _participation_shares(
            valid_volume, scale_by_wallet={wallet: 1.0 for wallet in scored}
        )
    )

    markout = _score_expectancy(stream, orders, bars_frame, mode=settings.expectancy_mode)
    window_start = as_of.value - settings.active_days_window_days * 86_400_000
    active_source = stream.rebalances if settings.net_orders_by_bar else orders
    active = (
        active_source.filter(pl.col("event_ts_ms") >= window_start)
        .group_by("wallet")
        .agg(pl.col("day").n_unique().alias("active_days"))
    )

    weights = np.array([float(k) for k, _ in k_results], dtype=np.float64)
    cdar_shares = euler_marginal_cdar_shares(returns, weights, settings.cdar_alpha)
    labels = _cluster_labels(returns, settings.cluster_count)

    markout_rows = {
        row["wallet"]: (float(row["expectancy_bps"]), int(row["observations"]))
        for row in markout.iter_rows(named=True)
    }
    active_rows = {row["wallet"]: int(row["active_days"]) for row in active.iter_rows(named=True)}
    q10_rows = {row["wallet"]: _dec(row["q10_order_usd"]) for row in q10.iter_rows(named=True)}

    cards: list[WalletScorecard] = []
    audit: list[ScoreAuditRow] = []
    for row_index, wallet in enumerate(scored):
        raw_cdar_share = float(cdar_shares[row_index])
        clamped_cdar_share = min(max(raw_cdar_share, 0.0), 1.0)
        raw_p95, raw_worst = participation_rows[wallet]
        markout_bps, markout_fills = markout_rows.get(wallet, (0.0, 0))
        active_days = min(active_rows.get(wallet, 0), settings.active_days_window_days)
        drawdown = drawdowns[row_index]
        suggested_k, no_dd = k_results[row_index]
        failures: list[GateFailure] = []
        value = account_value[wallet]
        if value < settings.min_account_value_usd:
            failures.append(GateFailure.ACCOUNT_VALUE_BELOW_FLOOR)
        if active_days <= 0:
            failures.append(GateFailure.NO_ACTIVE_FILLS)
        if markout_fills == 0:
            failures.append(GateFailure.MARKOUT_NOT_COPYABLE)
        else:
            if markout_bps < 0:
                failures.append(GateFailure.NEGATIVE_5M_MARKOUT)
            if markout_fills < settings.min_markout_fills:
                failures.append(GateFailure.MARKOUT_NOT_COPYABLE)
        if not settings.decouple_sizing_gates:
            failures.extend(
                sizing_gate_failures(
                    raw_p95,
                    raw_worst,
                    raw_cdar_share,
                    max_p95=settings.max_p95_participation_1m,
                    max_max=settings.max_max_participation_1m,
                    max_cdar=settings.max_euler_cdar_share,
                )
            )
        if no_dd:
            notes.append(f"NO_DRAWDOWN_EVIDENCE:{wallet}")
        cards.append(
            WalletScorecard(
                wallet=wallet,
                as_of=as_of,
                account_value_usd=value,
                active_days_14d=active_days,
                markout_5m_bps=_dec(markout_bps),
                p95_participation_1m=_dec(min(max(raw_p95, 0.0), 1.0), "0.000000000001"),
                euler_cdar_share=_dec(clamped_cdar_share, "0.000000000001"),
                suggested_k=suggested_k,
                cluster_id=labels[row_index],
                copyable_drift=(
                    GateFailure.NEGATIVE_5M_MARKOUT not in failures
                    and GateFailure.MARKOUT_NOT_COPYABLE not in failures
                ),
                failures=tuple(failures),
                source_sha=digest(
                    {
                        "wallet": wallet,
                        "as_of_ms": as_of.value,
                        "lookback_days": lookback_days,
                        "rows": int(np.count_nonzero(returns[row_index])),
                        "config": settings.data,
                    }
                ),
            )
        )
        audit.append(
            ScoreAuditRow(
                wallet=wallet,
                q10_order_usd=q10_rows.get(wallet, Decimal(0)),
                markout_5m_bps=_dec(markout_bps),
                markout_fills=markout_fills,
                p95_participation_1m=_dec(raw_p95, "0.000000000001"),
                max_participation_1m=_dec(raw_worst, "0.000000000001"),
                cdar_share=_dec(raw_cdar_share, "0.000000000001"),
                wallet_max_dd_usd=_dec(drawdown, "0.01"),
                suggested_k=suggested_k,
                active_days_14d=active_days,
            )
        )

    source_sha = digest(
        {
            "as_of_ms": as_of.value,
            "lookback_days": lookback_days,
            "config": settings.data,
            "tiers": list(observed_tiers),
            "slice": fills.as_dict(),
            "rows": raw.height,
            "cards": [card.sha for card in cards],
            "excluded": [row.data for row in excluded],
        }
    )
    returns_by_wallet = {
        wallet: tuple(float(v) for v in returns[row_index])
        for row_index, wallet in enumerate(scored)
    }
    return UniverseScorecard(
        as_of=as_of,
        lookback_days=lookback_days,
        config=settings,
        cards=tuple(sorted(cards, key=lambda card: card.wallet)),
        audit=tuple(audit),
        excluded=tuple(sorted(excluded, key=lambda row: (row.wallet, row.reason))),
        tiers=observed_tiers,
        slice_days=len(fills.days),
        slice_rows=raw.height,
        undated=allow_undated,
        source_sha=source_sha,
        notes=tuple(sorted(set(notes))),
        _returns_by_wallet=returns_by_wallet,
        _unscaled_participation_by_wallet=unscaled_participation_rows,
    )


def bench_from_scorecard(
    scorecard: UniverseScorecard,
    *,
    cluster_distance: Decimal | None = None,
) -> tuple[Any, ...]:
    """Convert eligible cards into ``BenchCandidate`` rows with bench-conditioned Euler CDaR."""
    from copytrade_pm.roster import BenchCandidate

    if not isinstance(scorecard, UniverseScorecard):
        raise TypeError("UNIVERSE_SCORECARD_REQUIRED")
    distance = Decimal(1) if cluster_distance is None else Decimal(cluster_distance)
    if distance < 0:
        raise ValueError("INVALID_CLUSTER_DISTANCE")
    eligible = [card for card in scorecard.cards if card.eligible]
    if not eligible:
        return ()
    if scorecard._returns_by_wallet and all(
        card.wallet in scorecard._returns_by_wallet for card in eligible
    ):
        bench_returns = np.array(
            [scorecard._returns_by_wallet[card.wallet] for card in eligible],
            dtype=np.float64,
        )
        bench_weights = np.array([float(card.suggested_k) for card in eligible], dtype=np.float64)
        bench_shares = euler_marginal_cdar_shares(
            bench_returns, bench_weights, scorecard.config.cdar_alpha
        )
        marginal_shares = [
            _dec(min(max(float(bench_shares[idx]), 0.0), 1.0), "0.000000000001")
            for idx in range(len(eligible))
        ]
    else:
        marginal_shares = [card.euler_cdar_share for card in eligible]
    return tuple(
        BenchCandidate(
            wallet=card.wallet,
            scorecard_sha=card.sha,
            suggested_k=card.suggested_k,
            marginal_cdar_share=marginal_shares[idx],
            cluster_id=card.cluster_id,
            cluster_distance=distance,
        )
        for idx, card in enumerate(eligible)
    )
