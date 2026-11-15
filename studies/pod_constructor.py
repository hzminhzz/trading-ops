"""Multi-manager pod construction, dual-barrier sizing, and survival validation.

Designed natively around ``FIXED_K_NATIVE_QUANTITY`` (no leader-NAV dependency),
``Policy A`` 5-minute follower unit-copy USD streams, canonical ``00:05 UTC`` daily
reset barriers, and Politis-Romano stationary block bootstrap survival testing.

Provides four reusable building blocks for ``trading-ops`` and ``portfolio-research``:
  1. Stage 1 Behavioral Audit & Episode Reconstruction (execution-aligned 5m MTM).
  2. Stage 1.5a Sparse-Aware Crowding Distance Tensor (active-union 4h tail
     co-distress, underwater co-drawdown, signed netting cosine, regime balance).
  3. Stage 1.5b/2 Orthogonal Style Clustering (UPGMA average linkage) & Joint
     Dual-Barrier Simplex Optimizer (CDaR_alpha + Daily CVaR_alpha + exact sparse
     5m netted turnover, bounded risk shares, and Bybit min-lot diagnostics).
  4. Leak-Free CPCV/PBO, Regime-Weighted Stationary Block Bootstrap on exact
     Guardian-simulated daily blocks, and Episode-Time CUSUM Roster Monitoring
     for ``wallet-research`` decay/tilt detection.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Constants & Pure-NumPy/Stdlib Normal Helpers (zero SciPy dependency)
# ---------------------------------------------------------------------------

_SQRT2 = math.sqrt(2.0)
_EULER_GAMMA = 0.5772156649015329


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function via math.erf."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam rational approximation, |err| < 1.15e-9)."""
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1), got {p}")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q) + c[5]
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q) + 1.0
        return num / den
    if p > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q) + c[5]
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q) + 1.0
        return -(num / den)
    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    return num / den


def rankdata_average(values: np.ndarray) -> np.ndarray:
    """1-based ranks with average tie handling, matching scipy.stats.rankdata."""
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    sorted_vals = arr[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def sample_skew_kurt(x: np.ndarray) -> tuple[float, float]:
    """Unbiased sample skewness and Pearson kurtosis (normal = 3.0)."""
    n = x.size
    if n < 4:
        return 0.0, 3.0
    mean = float(np.mean(x))
    diff = x - mean
    m2 = float(np.mean(diff * diff))
    if m2 <= 1e-24:
        return 0.0, 3.0
    s = math.sqrt(m2)
    m3 = float(np.mean(diff**3))
    m4 = float(np.mean(diff**4))
    g1 = m3 / (s**3)
    skew_unbiased = (math.sqrt(n * (n - 1)) / (n - 2)) * g1
    g2 = m4 / (m2 * m2) - 3.0
    excess_unbiased = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6.0)
    return float(skew_unbiased), float(excess_unbiased + 3.0)


def compute_underwater_and_cdar(
    cum_returns: np.ndarray,
    alpha: float = 0.95,
) -> tuple[np.ndarray, float, float]:
    """Return underwater curve u_t >= 0, MaxDD, and Conditional Drawdown-at-Risk."""
    running_peak = np.maximum(0.0, np.maximum.accumulate(cum_returns))
    underwater = running_peak - cum_returns
    max_dd = float(np.max(underwater)) if underwater.size > 0 else 0.0
    q_alpha = float(np.quantile(underwater, alpha)) if underwater.size > 0 else 0.0
    tail_mask = underwater >= q_alpha
    cdar = float(np.mean(underwater[tail_mask])) if np.any(tail_mask) else q_alpha
    return underwater, max_dd, cdar


def compute_jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute bounded Jensen-Shannon divergence in [0, 1] (base 2)."""
    eps = 1e-12
    p_norm = np.asarray(p, dtype=np.float64) + eps
    q_norm = np.asarray(q, dtype=np.float64) + eps
    p_norm /= np.sum(p_norm)
    q_norm /= np.sum(q_norm)
    m = 0.5 * (p_norm + q_norm)
    kl_pm = float(np.sum(p_norm * np.log2(p_norm / m)))
    kl_qm = float(np.sum(q_norm * np.log2(q_norm / m)))
    return float(np.clip(0.5 * (kl_pm + kl_qm), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Configuration & Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PodConstructorConfig:
    """Parameters governing behavioral screening, clustering, and dual-barrier sizing."""

    bars_per_day: int = 288
    days_per_year: int = 365
    reset_bar_offset: int = 1  # 00:05 UTC = bar index 1 within UTC day on 5m grid
    flat_notional_eps_frac: float = 1e-4  # Fraction of pod_equity_usd below which flat
    cost_bps: float = 6.5  # 4.5 bps taker fee + 2.0 bps slippage

    # Account & Rulebook specs (fractions of pod_equity_usd)
    pod_equity_usd: float = 10_000.0
    profit_target_frac: float = 0.08  # +8% ($800 on $10k CFT)
    max_dd_frac: float = 0.10  # 10% static DD ($1,000)
    daily_loss_frac: float = 0.05  # 5% daily loss ($500)
    daily_loss_mode: Literal["reset_balance", "intraday_trailing"] = "reset_balance"
    min_lot_usd: float = 10.0

    # Stage 1 Behavioral Screen
    min_episodes: int = 4
    max_tilt_multiplier: float = 1.50
    min_active_bar_frac: float = 0.01

    # Stage 1.5a Crowding Distance Tensor
    tail_window_bars: int = 48  # 4h window (48 x 5m)
    tail_quantile: float = 0.05
    overlap_shrinkage_kappa: float = 500.0
    dd_tail_quantile: float = 0.80
    crowding_weights: tuple[float, float, float, float] = (0.40, 0.30, 0.15, 0.15)
    # (Lower-tail Lambda, Co-DD Omega, Netting Cosine Psi, Regime-Correlation Penalty)

    # Stage 1.5b/2 Clustering & Simplex Optimizer
    pod_size: int = 4
    top_per_cluster: int = 3
    max_candidate_pods: int = 81
    cdar_alpha: float = 0.95
    cdar_buffer_frac: float = 0.60
    hard_dd_buffer_frac: float = 0.75
    daily_cvar_alpha: float = 0.99
    daily_cvar_buffer_frac: float = 0.60
    hard_daily_buffer_frac: float = 0.75
    min_risk_share: float = 0.15
    max_risk_share: float = 0.45
    max_k_mult: float = 5.0  # Safety ceiling on per-wallet multiplier under FIXED_K_NATIVE_QUANTITY

    # Stage 4 Validation
    cpcv_n_blocks: int = 6
    cpcv_embargo_days: int = 1
    bootstrap_mean_block_days: float = 7.0
    bootstrap_horizon_days: int = 90
    bootstrap_n_paths: int = 5_000

    # Stage 5 Roster CUSUM Monitor (for wallet-research Stage 1)
    cusum_slack_k: float = 0.50
    cusum_warn_h: float = 2.50
    cusum_kill_h: float = 4.00
    mae_kill_multiple: float = 2.50


@dataclass
class UnitCopy5mTensor:
    """Canonical FIXED_K_NATIVE_QUANTITY 5m dataset normalized by fixed pod_equity_usd.

    Every value represents unit-copy (k = 1) follower USD exposure or PnL divided by
    ``pod_equity_usd``, avoiding any dependency on unverified source-wallet NAV.
    """

    aliases: tuple[str, ...]
    symbols: tuple[str, ...]
    timestamps_ms: np.ndarray  # (T,) UTC 5m checkpoint timestamps
    notional_frac: np.ndarray  # (N, S, T) signed USD notional / pod_equity_usd at k=1
    delta_notional_frac: np.ndarray  # (N, S, T) traded signed USD notional / pod_equity_usd at k=1
    gross_ret_frac: np.ndarray  # (N, T) unit-copy gross USD PnL / pod_equity_usd
    net_ret_frac: np.ndarray  # (N, T) unit-copy solo net USD PnL / pod_equity_usd
    sym_net_ret_frac: np.ndarray | None = None  # (N, S, T) optional per-symbol solo net PnL
    chop_mask: np.ndarray | None = None  # (T,) optional regime mask (True = chop/drawdown window)
    pod_equity_usd: float = 10_000.0

    def __post_init__(self) -> None:
        n, s, t = self.notional_frac.shape
        if len(self.aliases) != n or len(self.symbols) != s:
            raise ValueError("ALIAS_OR_SYMBOL_SHAPE_MISMATCH")
        if self.timestamps_ms.shape != (t,):
            raise ValueError("TIMESTAMP_SHAPE_MISMATCH")
        if self.delta_notional_frac.shape != (n, s, t):
            raise ValueError("DELTA_NOTIONAL_SHAPE_MISMATCH")
        if self.gross_ret_frac.shape != (n, t) or self.net_ret_frac.shape != (n, t):
            raise ValueError("RETURN_SHAPE_MISMATCH")
        if self.sym_net_ret_frac is not None and self.sym_net_ret_frac.shape != (n, s, t):
            raise ValueError("SYM_NET_RET_SHAPE_MISMATCH")
        if self.chop_mask is not None and self.chop_mask.shape != (t,):
            raise ValueError("CHOP_MASK_SHAPE_MISMATCH")

    def slice_time(self, start_bar: int, end_bar: int) -> UnitCopy5mTensor:
        """Return a contiguous time slice [start_bar, end_bar)."""
        return UnitCopy5mTensor(
            aliases=self.aliases,
            symbols=self.symbols,
            timestamps_ms=self.timestamps_ms[start_bar:end_bar],
            notional_frac=self.notional_frac[:, :, start_bar:end_bar],
            delta_notional_frac=self.delta_notional_frac[:, :, start_bar:end_bar],
            gross_ret_frac=self.gross_ret_frac[:, start_bar:end_bar],
            net_ret_frac=self.net_ret_frac[:, start_bar:end_bar],
            sym_net_ret_frac=(
                self.sym_net_ret_frac[:, :, start_bar:end_bar]
                if self.sym_net_ret_frac is not None
                else None
            ),
            chop_mask=self.chop_mask[start_bar:end_bar] if self.chop_mask is not None else None,
            pod_equity_usd=self.pod_equity_usd,
        )


@dataclass(frozen=True)
class TraderAuditMetrics:
    """Behavioral, skill, and drawdown profile for a solo wallet at unit k = 1."""

    trader_id: int
    alias: str
    active_bars: int
    active_frac: float
    num_episodes: int
    raw_sharpe_ann: float
    eb_shrunk_sharpe_ann: float
    deflated_sharpe_prob: float
    net_over_max_dd: float
    chop_net_over_dd: float
    standalone_cdar95: float
    standalone_max_dd: float
    tilt_multiplier: float
    active_pos_q10_frac: float
    passed_behavioral_gate: bool
    reject_reason: str = ""


@dataclass(frozen=True)
class TraderEpisodeBaseline:
    """In-sample episode baseline parameters for Episode-Time CUSUM monitoring."""

    trader_id: int
    alias: str
    mean_ep_return: float
    std_ep_return: float
    median_ln_peak_notional: float
    mad_ln_peak_notional: float
    mean_win_ln_duration: float
    std_win_ln_duration: float
    symbol_notional_dist: np.ndarray
    mae_p95: float


@dataclass
class SizedPodCandidate:
    """Ranked candidate pod with joint dual-barrier calibrated fixed multipliers k_i."""

    aliases: tuple[str, ...]
    pod_indices: tuple[int, ...]
    multipliers: np.ndarray
    objective_net_return: float
    cdar_95: float
    max_static_dd: float
    daily_cvar_99: float
    max_daily_loss: float
    net_over_max_dd: float
    netting_savings_frac: float
    binding_constraint: str
    feasible: bool
    min_lot_warnings: tuple[str, ...] = ()


@dataclass
class RosterCUSUMState:
    """Sequential Episode-Time CUSUM monitor state for wallet-research Stage 1."""

    alias: str
    s_decay: float = 0.0
    s_tilt: float = 0.0
    episodes_observed: int = 0
    recent_symbol_notionals: list[np.ndarray] = field(default_factory=list)
    status: Literal["OK", "WARN_THROTTLE_0.5X", "KILL_CONTAIN_0X"] = "OK"
    reasons: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Tensor Builder Adapter from Policy A Replay Output
# ---------------------------------------------------------------------------


def build_tensor_from_unit_replays(
    unit_pos_by_alias: dict[str, dict[str, object]],
    timestamps_ms: Sequence[int],
    *,
    pod_equity_usd: float = 10_000.0,
    chop_split_ms: int | None = None,
) -> UnitCopy5mTensor:
    """Aggregate Policy A 1m replay DataFrames into an execution-aligned 5m tensor.

    Accepts ``pos_by_coin`` dicts returned by ``replay_portfolio_5m({alias: 1.0}, ...)``
    where each Polars DataFrame holds ``timestamp, open, close, pos, delta, gross, cost, net``.
    Since Policy A updates ``pos`` strictly at ``minute % 5 == 0`` opens, 5m aggregation
    preserves exact gross PnL, open-minute turnover costs, and open notional linearity.
    """
    import polars as pl

    aliases = tuple(sorted(unit_pos_by_alias.keys()))
    symbol_set: set[str] = set()
    for by_coin in unit_pos_by_alias.values():
        symbol_set.update(by_coin.keys())
    symbols = tuple(sorted(symbol_set))

    ts_arr = np.asarray(timestamps_ms, dtype=np.int64)
    n_traders = len(aliases)
    n_syms = len(symbols)
    n_bars = ts_arr.size

    notional_frac = np.zeros((n_traders, n_syms, n_bars), dtype=np.float64)
    delta_notional_frac = np.zeros((n_traders, n_syms, n_bars), dtype=np.float64)
    gross_ret_frac = np.zeros((n_traders, n_bars), dtype=np.float64)
    net_ret_frac = np.zeros((n_traders, n_bars), dtype=np.float64)
    sym_net_ret_frac = np.zeros((n_traders, n_syms, n_bars), dtype=np.float64)

    ts_ms_col = pl.DataFrame({"ts_ms": ts_arr})
    sym_idx = {s: idx for idx, s in enumerate(symbols)}

    for ti, alias in enumerate(aliases):
        for coin, df_obj in unit_pos_by_alias[alias].items():
            si = sym_idx[coin]
            df = df_obj  # type: ignore[assignment]
            b5 = (
                df.sort("timestamp")
                .with_columns(
                    (
                        (pl.col("timestamp").dt.epoch("ms") // 300_000) * 300_000
                    ).alias("ts_ms")
                )
                .group_by("ts_ms", maintain_order=True)
                .agg(
                    (pl.col("pos").first() * pl.col("open").first()).alias("notional"),
                    (pl.col("delta").first() * pl.col("open").first()).alias("d_notional"),
                    pl.col("gross").sum().alias("gross"),
                    pl.col("net").sum().alias("net"),
                )
            )
            aligned = ts_ms_col.join(b5, on="ts_ms", how="left").fill_null(0.0)
            notional_frac[ti, si] = aligned["notional"].to_numpy() / pod_equity_usd
            delta_notional_frac[ti, si] = aligned["d_notional"].to_numpy() / pod_equity_usd
            coin_net = aligned["net"].to_numpy() / pod_equity_usd
            sym_net_ret_frac[ti, si] = coin_net
            gross_ret_frac[ti] += aligned["gross"].to_numpy() / pod_equity_usd
            net_ret_frac[ti] += coin_net

    chop_mask = (ts_arr >= chop_split_ms) if chop_split_ms is not None else None
    return UnitCopy5mTensor(
        aliases=aliases,
        symbols=symbols,
        timestamps_ms=ts_arr,
        notional_frac=notional_frac,
        delta_notional_frac=delta_notional_frac,
        gross_ret_frac=gross_ret_frac,
        net_ret_frac=net_ret_frac,
        sym_net_ret_frac=sym_net_ret_frac,
        chop_mask=chop_mask,
        pod_equity_usd=pod_equity_usd,
    )


# ---------------------------------------------------------------------------
# Stage 1: Execution-Aligned Episode Reconstruction & Behavioral Gate
# ---------------------------------------------------------------------------


def reconstruct_execution_episodes(
    tensor: UnitCopy5mTensor,
    eps: float = 1e-4,
) -> dict[int, dict[str, np.ndarray]]:
    """Reconstruct trade episodes aligned with Policy A 5m execution.

    When multi-coin traders keep staggered positions open continuously (never hitting
    total account flat), account-level grouping collapses 70+ days into 1 run.
    Grouping per ``(trader, symbol)`` on non-zero intervals AND sign reversals, while
    folding closing-bar exit PnL/cost into the ending episode, captures every real
    per-coin position lifecycle in chronological close order.
    """
    n_traders, n_syms, n_bars = tensor.notional_frac.shape
    abs_notional = np.abs(tensor.notional_frac)
    gross_notional = np.sum(abs_notional, axis=1)  # (N, T)
    episodes_by_trader: dict[int, dict[str, np.ndarray]] = {}

    for i in range(n_traders):
        raw_episodes: list[tuple[int, float, float, float, float, np.ndarray]] = []
        if tensor.sym_net_ret_frac is not None:
            for s in range(n_syms):
                pos_s = tensor.notional_frac[i, s]
                abs_s = abs_notional[i, s]
                d_s = np.abs(tensor.delta_notional_frac[i, s])
                ret_s = tensor.sym_net_ret_frac[i, s]
                held_s = abs_s > eps
                if not np.any(held_s):
                    continue
                signs = np.where(held_s, np.where(pos_s >= 0.0, 1, -1), 0)
                t = 0
                while t < n_bars:
                    if signs[t] == 0:
                        t += 1
                        continue
                    cur_sign = signs[t]
                    s_idx = t
                    while t < n_bars and signs[t] == cur_sign:
                        t += 1
                    e_idx = t
                    # Include flat closing bar's exit slippage & mark gap
                    pnl_end = e_idx
                    if e_idx < n_bars and signs[e_idx] == 0 and d_s[e_idx] > 1e-7:
                        pnl_end = e_idx + 1
                    r_slice = ret_s[s_idx:pnl_end]
                    cum_r = np.cumsum(r_slice)
                    peaks = np.maximum(0.0, np.maximum.accumulate(cum_r))
                    sym_vec = np.zeros(n_syms, dtype=np.float64)
                    sym_vec[s] = float(np.sum(abs_s[s_idx:e_idx]))
                    raw_episodes.append(
                        (
                            e_idx,
                            float(np.sum(r_slice)),
                            float(np.max(abs_s[s_idx:e_idx])),
                            float(e_idx - s_idx),
                            float(np.max(peaks - cum_r)) if cum_r.size > 0 else 0.0,
                            sym_vec,
                        )
                    )
        else:
            # Fallback when sym_net_ret_frac is absent: slice per symbol using proportional
            # notional share of account net_ret_frac so continuous multi-coin traders still
            # yield accurate per-symbol episode counts unless single-symbol/array fallback fits.
            held_any = gross_notional[i] > eps
            padded_any = np.concatenate(([False], held_any, [False]))
            n_acct_runs = int(np.sum(padded_any[1:] & ~padded_any[:-1]))
            if n_acct_runs >= 5 or n_syms == 1:
                abs_delta = np.sum(np.abs(tensor.delta_notional_frac[i]), axis=0)
                starts = np.flatnonzero(padded_any[1:] & ~padded_any[:-1])
                ends = np.flatnonzero(~padded_any[1:] & padded_any[:-1])
                for s_idx, e_idx in zip(starts, ends):
                    pnl_end = e_idx + 1 if (e_idx < n_bars and abs_delta[e_idx] > 1e-7) else e_idx
                    r_slice = tensor.net_ret_frac[i, s_idx:pnl_end]
                    cum_r = np.cumsum(r_slice)
                    peaks = np.maximum(0.0, np.maximum.accumulate(cum_r))
                    raw_episodes.append(
                        (
                            int(e_idx),
                            float(np.sum(r_slice)),
                            float(np.max(gross_notional[i, s_idx:e_idx])),
                            float(e_idx - s_idx),
                            float(np.max(peaks - cum_r)) if cum_r.size > 0 else 0.0,
                            np.sum(abs_notional[i, :, s_idx:e_idx], axis=1),
                        )
                    )
            else:
                denom = np.maximum(gross_notional[i], 1e-12)
                for s in range(n_syms):
                    pos_s = tensor.notional_frac[i, s]
                    abs_s = abs_notional[i, s]
                    d_s = np.abs(tensor.delta_notional_frac[i, s])
                    held_s = abs_s > eps
                    if not np.any(held_s):
                        continue
                    signs = np.where(held_s, np.where(pos_s >= 0.0, 1, -1), 0)
                    share = abs_s / denom
                    ret_est = tensor.net_ret_frac[i] * share
                    t = 0
                    while t < n_bars:
                        if signs[t] == 0:
                            t += 1
                            continue
                        cur_sign = signs[t]
                        s_idx = t
                        while t < n_bars and signs[t] == cur_sign:
                            t += 1
                        e_idx = t
                        pnl_end = e_idx + 1 if (e_idx < n_bars and signs[e_idx] == 0 and d_s[e_idx] > 1e-7) else e_idx
                        r_slice = ret_est[s_idx:pnl_end]
                        cum_r = np.cumsum(r_slice)
                        peaks = np.maximum(0.0, np.maximum.accumulate(cum_r))
                        sym_vec = np.zeros(n_syms, dtype=np.float64)
                        sym_vec[s] = float(np.sum(abs_s[s_idx:e_idx]))
                        raw_episodes.append(
                            (
                                e_idx,
                                float(np.sum(r_slice)),
                                float(np.max(abs_s[s_idx:e_idx])),
                                float(e_idx - s_idx),
                                float(np.max(peaks - cum_r)) if cum_r.size > 0 else 0.0,
                                sym_vec,
                            )
                        )

        raw_episodes.sort(key=lambda item: item[0])
        n_ep = len(raw_episodes)
        episodes_by_trader[i] = {
            "ep_pnl": np.array([r[1] for r in raw_episodes], dtype=np.float64),
            "ep_peak_notional": np.array([r[2] for r in raw_episodes], dtype=np.float64),
            "ep_duration_bars": np.array([r[3] for r in raw_episodes], dtype=np.float64),
            "ep_mae": np.array([r[4] for r in raw_episodes], dtype=np.float64),
            "ep_symbol_notional": (
                np.vstack([r[5] for r in raw_episodes])
                if n_ep > 0
                else np.zeros((0, n_syms), dtype=np.float64)
            ),
        }
    return episodes_by_trader


def stage1_audit_traders(
    tensor: UnitCopy5mTensor,
    config: PodConstructorConfig = PodConstructorConfig(),
) -> tuple[list[int], list[TraderAuditMetrics], dict[int, TraderEpisodeBaseline]]:
    """Enrich candidates with behavioral tilt, EB-Sharpe, DSR (diagnostic), and baselines.

    Per ML4T-METHODOLOGY-LESSONS.md, DSR is recorded strictly as a diagnostic and never
    used as a blanket rejection cutoff on discretionary wallets.
    """
    n_traders, n_syms, n_bars = tensor.notional_frac.shape
    abs_notional = np.abs(tensor.notional_frac)
    gross_notional = np.sum(abs_notional, axis=1)
    active_mask = gross_notional > config.flat_notional_eps_frac

    episodes_map = reconstruct_execution_episodes(tensor)
    bars_per_year = float(config.bars_per_day * config.days_per_year)

    raw_sr_bar = np.zeros(n_traders, dtype=np.float64)
    raw_sr_ann = np.zeros(n_traders, dtype=np.float64)
    sr_var_ann = np.zeros(n_traders, dtype=np.float64)
    skew_vec = np.zeros(n_traders, dtype=np.float64)
    kurt_vec = np.zeros(n_traders, dtype=np.float64)
    tilt_vec = np.zeros(n_traders, dtype=np.float64)
    cdar95_vec = np.zeros(n_traders, dtype=np.float64)
    maxdd_vec = np.zeros(n_traders, dtype=np.float64)
    net_dd_vec = np.zeros(n_traders, dtype=np.float64)
    chop_net_dd_vec = np.zeros(n_traders, dtype=np.float64)
    q10_vec = np.zeros(n_traders, dtype=np.float64)
    baselines: dict[int, TraderEpisodeBaseline] = {}

    for i in range(n_traders):
        act_idx = np.flatnonzero(active_mask[i])
        t_act = act_idx.size
        act_frac = t_act / float(max(1, n_bars))

        if t_act >= 20:
            r_act = tensor.net_ret_frac[i, act_idx]
            mu_act = float(np.mean(r_act))
            sd_act = float(np.std(r_act, ddof=1)) + 1e-12
            sr_b = mu_act / sd_act
            ann_fac = math.sqrt(bars_per_year * act_frac)
            sr_a = sr_b * ann_fac
            g3, g4 = sample_skew_kurt(r_act)
            var_b = max(
                1e-8,
                (1.0 - g3 * sr_b + ((g4 - 1.0) / 4.0) * (sr_b * sr_b)) / max(1.0, t_act - 1.0),
            )
            var_a = var_b * (bars_per_year * act_frac)
            nz_pos = abs_notional[i][abs_notional[i] > config.flat_notional_eps_frac]
            q10 = float(np.quantile(nz_pos, 0.10)) if nz_pos.size > 0 else 0.0
        else:
            sr_b, sr_a, var_a, g3, g4, q10 = 0.0, 0.0, 1.0, 0.0, 3.0, 0.0

        raw_sr_bar[i] = sr_b
        raw_sr_ann[i] = sr_a
        sr_var_ann[i] = var_a
        skew_vec[i] = g3
        kurt_vec[i] = g4
        q10_vec[i] = q10

        cum_r = np.cumsum(tensor.net_ret_frac[i])
        _, mdd, cdar95 = compute_underwater_and_cdar(cum_r, alpha=config.cdar_alpha)
        maxdd_vec[i] = max(mdd, 1e-6)
        cdar95_vec[i] = max(cdar95, 1e-6)
        tot_ret = float(cum_r[-1]) if cum_r.size > 0 else 0.0
        net_dd_vec[i] = tot_ret / max(mdd, 1e-6)

        if tensor.chop_mask is not None and np.any(tensor.chop_mask):
            chop_r = tensor.net_ret_frac[i, tensor.chop_mask]
            chop_cum = np.cumsum(chop_r)
            _, chop_mdd, _ = compute_underwater_and_cdar(chop_cum, alpha=config.cdar_alpha)
            chop_net = float(chop_cum[-1]) if chop_cum.size > 0 else 0.0
            chop_net_dd_vec[i] = chop_net / max(chop_mdd, 1e-6)
        else:
            chop_net_dd_vec[i] = net_dd_vec[i]

        ep = episodes_map[i]
        ep_pnl = ep["ep_pnl"]
        ep_peak = ep["ep_peak_notional"]
        ep_dur = ep["ep_duration_bars"]
        ep_mae = ep["ep_mae"]
        ep_sym = ep["ep_symbol_notional"]
        n_ep = ep_pnl.size

        if n_ep >= 4:
            prev_loss = ep_pnl[:-1] < 0.0
            prev_win = ep_pnl[:-1] >= 0.0
            next_peak = ep_peak[1:]
            med_loss = (
                float(np.median(next_peak[prev_loss]))
                if np.any(prev_loss)
                else float(np.median(ep_peak))
            )
            med_win = (
                float(np.median(next_peak[prev_win]))
                if np.any(prev_win)
                else float(np.median(ep_peak))
            )
            tilt_vec[i] = med_loss / max(med_win, 1e-8)
        else:
            tilt_vec[i] = 1.0

        if n_ep >= 2:
            ln_peak = np.log(np.maximum(ep_peak, 1e-8))
            med_ln = float(np.median(ln_peak))
            mad_ln = float(np.median(np.abs(ln_peak - med_ln))) + 1e-4
            win_mask = ep_pnl >= 0.0
            ln_dur = np.log(np.maximum(ep_dur, 1.0))
            w_dur = ln_dur[win_mask] if np.any(win_mask) else ln_dur
            sym_tot = np.sum(ep_sym, axis=0) + 1e-8
            baselines[i] = TraderEpisodeBaseline(
                trader_id=i,
                alias=tensor.aliases[i],
                mean_ep_return=float(np.mean(ep_pnl)),
                std_ep_return=float(np.std(ep_pnl, ddof=1)) + 1e-5,
                median_ln_peak_notional=med_ln,
                mad_ln_peak_notional=mad_ln,
                mean_win_ln_duration=float(np.mean(w_dur)),
                std_win_ln_duration=float(np.std(w_dur, ddof=1)) + 0.15 if w_dur.size > 1 else 0.50,
                symbol_notional_dist=sym_tot / np.sum(sym_tot),
                mae_p95=float(np.quantile(ep_mae, 0.95)) + 1e-5,
            )
        else:
            baselines[i] = TraderEpisodeBaseline(
                trader_id=i,
                alias=tensor.aliases[i],
                mean_ep_return=0.0,
                std_ep_return=0.01,
                median_ln_peak_notional=0.0,
                mad_ln_peak_notional=0.25,
                mean_win_ln_duration=2.0,
                std_win_ln_duration=0.50,
                symbol_notional_dist=np.full(n_syms, 1.0 / max(1, n_syms)),
                mae_p95=0.02,
            )

    # Empirical Bayes (James-Stein) shrinkage across active-scaled annual Sharpes
    grand_mean_sr = float(np.mean(raw_sr_ann)) if n_traders > 0 else 0.0
    avg_var = float(np.mean(sr_var_ann)) if n_traders > 0 else 1.0
    scatter = float(np.sum((raw_sr_ann - grand_mean_sr) ** 2)) + 1e-12
    shrink = max(0.0, 1.0 - (max(1, n_traders - 3) * avg_var) / scatter)
    eb_sharpes = grand_mean_sr + shrink * (raw_sr_ann - grand_mean_sr)

    # Diagnostic Deflated Sharpe Ratio on annualized Sharpe scale
    std_sr_ann = float(np.std(raw_sr_ann, ddof=1)) + 1e-8 if n_traders > 1 else 1.0
    n_eff = max(2, n_traders)
    sr0_ann = std_sr_ann * (
        (1.0 - _EULER_GAMMA) * norm_ppf(1.0 - 1.0 / n_eff)
        + _EULER_GAMMA * norm_ppf(1.0 - 1.0 / (n_eff * math.e))
    )

    metrics: list[TraderAuditMetrics] = []
    passed_ids: list[int] = []
    for i in range(n_traders):
        t_act = int(np.sum(active_mask[i]))
        act_frac = t_act / float(max(1, n_bars))
        n_ep = episodes_map[i]["ep_pnl"].size
        se_ann = math.sqrt(max(1e-8, sr_var_ann[i]))
        dsr_prob = norm_cdf((raw_sr_ann[i] - sr0_ann) / se_ann)

        reasons: list[str] = []
        if act_frac < config.min_active_bar_frac:
            reasons.append(f"low_activity({act_frac:.3f}<{config.min_active_bar_frac})")
        if n_ep < config.min_episodes:
            reasons.append(f"few_episodes({n_ep}<{config.min_episodes})")
        if tilt_vec[i] > config.max_tilt_multiplier:
            reasons.append(f"martingale_tilt({tilt_vec[i]:.2f}>{config.max_tilt_multiplier})")
        if net_dd_vec[i] <= 0.0 and chop_net_dd_vec[i] <= 0.0:
            reasons.append("non_positive_net_return")

        passed = len(reasons) == 0
        if passed:
            passed_ids.append(i)
        metrics.append(
            TraderAuditMetrics(
                trader_id=i,
                alias=tensor.aliases[i],
                active_bars=t_act,
                active_frac=act_frac,
                num_episodes=n_ep,
                raw_sharpe_ann=float(raw_sr_ann[i]),
                eb_shrunk_sharpe_ann=float(eb_sharpes[i]),
                deflated_sharpe_prob=float(dsr_prob),
                net_over_max_dd=float(net_dd_vec[i]),
                chop_net_over_dd=float(chop_net_dd_vec[i]),
                standalone_cdar95=float(cdar95_vec[i]),
                standalone_max_dd=float(maxdd_vec[i]),
                tilt_multiplier=float(tilt_vec[i]),
                active_pos_q10_frac=float(q10_vec[i]),
                passed_behavioral_gate=passed,
                reject_reason=", ".join(reasons),
            )
        )

    return passed_ids, metrics, baselines


# ---------------------------------------------------------------------------
# Stage 1.5a: Sparse-Aware Crowding Distance Tensor
# ---------------------------------------------------------------------------


def compute_crowding_distance_matrix(
    candidate_ids: Sequence[int],
    tensor: UnitCopy5mTensor,
    config: PodConstructorConfig = PodConstructorConfig(),
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute pairwise dissimilarity D in [0.05, 1.0] across candidate wallets.

    Combines four sparse-aware components:
      - lambda_tail: 4h active-union lower-tail co-distress, shrunk toward daily active co-loss
      - omega_dd:    top-20% underwater drawdown concordance
      - psi_pos:     active-intersection signed notional cosine (crowding vs netting)
      - regime_corr: active daily-return concordance weighted across market regimes
    """
    cand = list(candidate_ids)
    n_cand = len(cand)
    sub_pos = tensor.notional_frac[cand]  # (C, S, T)
    sub_ret = tensor.net_ret_frac[cand]  # (C, T)
    n_bars = sub_ret.shape[1]

    gross = np.sum(np.abs(sub_pos), axis=1)
    active_mask = gross > config.flat_notional_eps_frac

    w_len = min(config.tail_window_bars, max(1, n_bars))
    stride = max(1, w_len // 4)
    starts = np.arange(0, max(1, n_bars - w_len + 1), stride)
    n_win = starts.size

    win_ret = np.zeros((n_cand, n_win), dtype=np.float64)
    win_act = np.zeros((n_cand, n_win), dtype=bool)
    for wi, sb in enumerate(starts):
        eb = min(n_bars, sb + w_len)
        win_ret[:, wi] = np.sum(sub_ret[:, sb:eb], axis=1)
        win_act[:, wi] = np.any(active_mask[:, sb:eb], axis=1)

    q05 = np.zeros(n_cand, dtype=np.float64)
    for idx in range(n_cand):
        act_w = win_ret[idx, win_act[idx]]
        q05[idx] = float(np.quantile(act_w, config.tail_quantile)) if act_w.size >= 5 else -0.01

    dd_excess = np.zeros((n_cand, n_bars), dtype=np.float64)
    for idx in range(n_cand):
        u_curve, _, _ = compute_underwater_and_cdar(np.cumsum(sub_ret[idx]), alpha=config.cdar_alpha)
        q80 = float(np.quantile(u_curve, config.dd_tail_quantile)) if u_curve.size > 0 else 0.0
        dd_excess[idx] = np.maximum(0.0, u_curve - q80)

    # Daily active return matrix for shrinking low-overlap pairs and regime checks
    bpd = config.bars_per_day
    n_days = max(1, n_bars // bpd)
    usable = n_days * bpd
    day_ret = np.sum(sub_ret[:, :usable].reshape(n_cand, n_days, bpd), axis=2)
    day_act = np.any(active_mask[:, :usable].reshape(n_cand, n_days, bpd), axis=2)

    lambda_mat = np.eye(n_cand, dtype=np.float64)
    omega_mat = np.eye(n_cand, dtype=np.float64)
    psi_mat = np.eye(n_cand, dtype=np.float64)
    regime_mat = np.eye(n_cand, dtype=np.float64)
    dist_mat = np.zeros((n_cand, n_cand), dtype=np.float64)

    w1, w2, w3, w4 = config.crowding_weights

    for a in range(n_cand):
        for b in range(a + 1, n_cand):
            # A: Active-Union lower-tail co-distress
            u_win = win_act[a] | win_act[b]
            if np.any(u_win):
                ra, rb = win_ret[a, u_win], win_ret[b, u_win]
                da, db = ra <= q05[a], rb <= q05[b]
                p_ab = float(np.mean(da[db])) if np.any(db) else 0.0
                p_ba = float(np.mean(db[da])) if np.any(da) else 0.0
                emp_lambda = 0.5 * (p_ab + p_ba)
            else:
                emp_lambda = 0.0

            inter_bars = active_mask[a] & active_mask[b]
            n_inter = int(np.sum(inter_bars))
            u_days = day_act[a] | day_act[b]
            if np.sum(u_days) >= 5:
                da_r, db_r = day_ret[a, u_days], day_ret[b, u_days]
                neg_a = da_r < 0.0
                neg_b = db_r < 0.0
                prior_lambda = (
                    0.5
                    * (
                        (float(np.mean(neg_a[neg_b])) if np.any(neg_b) else 0.0)
                        + (float(np.mean(neg_b[neg_a])) if np.any(neg_a) else 0.0)
                    )
                )
                corr_day = float(np.corrcoef(da_r, db_r)[0, 1]) if np.std(da_r) > 1e-12 and np.std(db_r) > 1e-12 else 0.0
            else:
                prior_lambda = 0.0
                corr_day = 0.0

            w_shr = n_inter / (n_inter + config.overlap_shrinkage_kappa)
            tilde_lambda = float(np.clip(w_shr * emp_lambda + (1.0 - w_shr) * prior_lambda, 0.0, 1.0))

            # B: Underwater Co-Drawdown
            na, nb = float(np.linalg.norm(dd_excess[a])), float(np.linalg.norm(dd_excess[b]))
            omega_dd = float(np.clip(np.dot(dd_excess[a], dd_excess[b]) / (na * nb), 0.0, 1.0)) if na > 1e-12 and nb > 1e-12 else 0.0

            # C: Signed Notional Netting Cosine on active intersection
            if n_inter > 0:
                pa = sub_pos[a, :, inter_bars]
                pb = sub_pos[b, :, inter_bars]
                if pa.shape[0] != n_inter:
                    pa, pb = pa.T, pb.T
                dots = np.sum(pa * pb, axis=1)
                norms = np.linalg.norm(pa, axis=1) * np.linalg.norm(pb, axis=1) + 1e-12
                psi_pos = float(np.clip(np.mean(dots / norms), -1.0, 1.0))
            else:
                psi_pos = 0.0

            reg_pen = float(np.clip(0.5 * (1.0 + corr_day), 0.0, 1.0))

            lambda_mat[a, b] = lambda_mat[b, a] = tilde_lambda
            omega_mat[a, b] = omega_mat[b, a] = omega_dd
            psi_mat[a, b] = psi_mat[b, a] = psi_pos
            regime_mat[a, b] = regime_mat[b, a] = reg_pen

            crowding = (
                w1 * tilde_lambda
                + w2 * omega_dd
                + w3 * (0.5 * (1.0 + psi_pos))
                + w4 * reg_pen
            )
            d_ab = float(np.clip(1.0 - crowding, 0.05, 1.0))
            dist_mat[a, b] = dist_mat[b, a] = d_ab

    return dist_mat, {
        "lambda_tail": lambda_mat,
        "omega_dd": omega_mat,
        "psi_pos": psi_mat,
        "regime_corr": regime_mat,
    }


def cluster_candidates_upgma(dist_mat: np.ndarray, n_clusters: int) -> np.ndarray:
    """Pure-NumPy UPGMA (average-linkage) clustering on a precomputed distance matrix."""
    n = dist_mat.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    k_target = max(1, min(n_clusters, n))
    if k_target == n:
        return np.arange(n, dtype=np.int32)

    clusters: list[list[int]] = [[i] for i in range(n)]
    while len(clusters) > k_target:
        best_i, best_j = 0, 1
        best_d = float("inf")
        m = len(clusters)
        for i in range(m):
            for j in range(i + 1, m):
                d_ij = float(np.mean(dist_mat[np.ix_(clusters[i], clusters[j])]))
                if d_ij < best_d:
                    best_d = d_ij
                    best_i, best_j = i, j
        merged = clusters[best_i] + clusters[best_j]
        clusters = [c for idx, c in enumerate(clusters) if idx not in (best_i, best_j)]
        clusters.append(merged)

    labels = np.empty(n, dtype=np.int32)
    for c_idx, members in enumerate(clusters):
        for m_idx in members:
            labels[m_idx] = c_idx
    return labels


# ---------------------------------------------------------------------------
# Stage 1.5b/2: Sparse-Netted Dual-Barrier Simplex Sizer
# ---------------------------------------------------------------------------


def _build_sparse_overlap_cache(
    pod_delta: np.ndarray,
    pod_net: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separate exact solo net returns from multi-member netting overlap bars.

    Since delta_P is zero on >99% of (symbol, bar) cells and only cells where >=2
    members trade the same symbol on the same bar can reduce aggregate turnover cost,
    this gives 100% exact 5m netted PnL across all S symbols in O(nnz_overlap) time.
    """
    nz = np.abs(pod_delta) > 1e-12  # (M, S, T)
    active_traders_per_cell = np.sum(nz, axis=0)  # (S, T)
    s_ov, t_ov = np.where(active_traders_per_cell >= 2)
    if s_ov.size == 0:
        return pod_net, np.zeros(0, dtype=np.int32), np.zeros((0, pod_delta.shape[0]))

    delta_ov = pod_delta[:, s_ov, t_ov].T  # (E_ov, M)
    return pod_net, t_ov.astype(np.int32), delta_ov


def _daily_reset_slices(timestamps_ms: np.ndarray, bars_per_day: int, reset_bar_offset: int) -> list[tuple[int, int]]:
    """Compute exact daily slices anchored to 00:05 UTC (or configured reset offset)."""
    n_bars = timestamps_ms.size
    if n_bars == 0:
        return []
    day_starts = np.where(((timestamps_ms // 300_000) % bars_per_day) == reset_bar_offset)[0]
    if day_starts.size == 0 or day_starts[0] > 0:
        starts = [0] + day_starts.tolist()
    else:
        starts = day_starts.tolist()
    slices: list[tuple[int, int]] = []
    for idx, sb in enumerate(starts):
        eb = starts[idx + 1] if idx + 1 < len(starts) else n_bars
        if eb > sb:
            slices.append((int(sb), int(eb)))
    return slices


def _evaluate_weights_unit_scale(
    weights: np.ndarray,
    solo_net: np.ndarray,
    t_ov: np.ndarray,
    delta_ov: np.ndarray,
    cost_rate: float,
    day_slices: Sequence[tuple[int, int]],
    config: PodConstructorConfig,
) -> tuple[float, float, float, float, float, float]:
    """Evaluate exact 5m netted unit-scale stats for normalized weight vector w."""
    r_net = weights @ solo_net  # (T,)
    savings_tot = 0.0
    if t_ov.size > 0:
        netted_cost = np.abs(delta_ov @ weights) * cost_rate
        solo_weighted = (np.abs(delta_ov) @ weights) * cost_rate
        diff = solo_weighted - netted_cost
        savings_tot = float(np.sum(diff))
        np.add.at(r_net, t_ov, diff)

    cum_r = np.cumsum(r_net)
    _, max_dd, cdar_95 = compute_underwater_and_cdar(cum_r, alpha=config.cdar_alpha)
    tot_ret = float(cum_r[-1]) if cum_r.size > 0 else 0.0

    daily_losses = np.empty(len(day_slices), dtype=np.float64)
    for di, (sb, eb) in enumerate(day_slices):
        base = cum_r[sb - 1] if sb > 0 else 0.0
        intraday = cum_r[sb:eb] - base
        if config.daily_loss_mode == "intraday_trailing":
            peaks = np.maximum(0.0, np.maximum.accumulate(intraday))
            daily_losses[di] = float(np.max(peaks - intraday))
        else:
            daily_losses[di] = float(max(0.0, -float(np.min(intraday))))

    max_daily = float(np.max(daily_losses)) if daily_losses.size > 0 else 0.0
    q99 = float(np.quantile(daily_losses, config.daily_cvar_alpha)) if daily_losses.size > 0 else 0.0
    tail = daily_losses >= q99
    daily_cvar = float(np.mean(daily_losses[tail])) if np.any(tail) else q99
    return tot_ret, max_dd, cdar_95, daily_cvar, max_daily, savings_tot


def size_pod_dual_barrier(
    pod_indices: Sequence[int],
    tensor: UnitCopy5mTensor,
    metrics_list: Sequence[TraderAuditMetrics],
    config: PodConstructorConfig = PodConstructorConfig(),
) -> SizedPodCandidate:
    """Solve joint Pod k_i on exact 5m netted streams using homogeneity & barrier caps.

    Because pathwise CDaR, Daily CVaR, MaxDD, and exact 5m netted returns are all
    positively homogeneous of degree 1 in overall pod scale lambda > 0, optimizing
    bounded risk-share weights over the low-dimensional simplex (M <= 4) with closed-form
    scale projection finds the global constrained optimum cleanly with zero LP coarsening.
    """
    pod = tuple(pod_indices)
    m_size = len(pod)
    cost_rate = config.cost_bps * 1e-4

    solo_net, t_ov, delta_ov = _build_sparse_overlap_cache(
        tensor.delta_notional_frac[list(pod)],
        tensor.net_ret_frac[list(pod)],
    )
    day_slices = _daily_reset_slices(tensor.timestamps_ms, config.bars_per_day, config.reset_bar_offset)

    # Risk normalization scale per member (inverse CDaR_95 keeps shares comparable)
    risk_unit = np.array(
        [max(metrics_list[idx].standalone_cdar95, 1e-4) for idx in pod],
        dtype=np.float64,
    )
    inv_risk = 1.0 / risk_unit
    inv_risk /= np.sum(inv_risk)

    cdar_lim = config.cdar_buffer_frac * config.max_dd_frac
    hard_dd_lim = config.hard_dd_buffer_frac * config.max_dd_frac
    dcvar_lim = config.daily_cvar_buffer_frac * config.daily_loss_frac
    hard_daily_lim = config.hard_daily_buffer_frac * config.daily_loss_frac

    def score_shares(shares: np.ndarray) -> tuple[float, np.ndarray, str, tuple[float, ...]]:
        # Map risk shares s_i into unit direction w_i proportional to s_i / risk_i
        w = shares / risk_unit
        w_sum = float(np.sum(w))
        if w_sum <= 1e-12:
            return -1e9, np.zeros(m_size), "INVALID", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        w /= w_sum

        ret_u, mdd_u, cdar_u, dcvar_u, mdaily_u, sav_u = _evaluate_weights_unit_scale(
            w, solo_net, t_ov, delta_ov, cost_rate, day_slices, config
        )
        # Closed-form maximum feasible scale lambda* under all 4 risk caps + max_k_mult
        caps: list[tuple[str, float]] = [
            ("CDaR_95", cdar_lim / max(cdar_u, 1e-9)),
            ("DailyCVaR_99", dcvar_lim / max(dcvar_u, 1e-9)),
            ("Hard_MaxDD", hard_dd_lim / max(mdd_u, 1e-9)),
            ("Hard_Daily", hard_daily_lim / max(mdaily_u, 1e-9)),
            ("Max_K_Cap", config.max_k_mult / max(float(np.max(w)), 1e-9)),
        ]
        bind_name, lam = min(caps, key=lambda item: item[1])
        k_vec = lam * w
        scaled_ret = lam * ret_u
        return (
            scaled_ret,
            k_vec,
            bind_name,
            (scaled_ret, lam * cdar_u, lam * mdd_u, lam * dcvar_u, lam * mdaily_u, lam * sav_u),
        )

    # Generate deterministic grid over bounded risk-share simplex [min_s, max_s]
    s_lo, s_hi = config.min_risk_share, config.max_risk_share
    if s_lo * m_size > 1.0 + 1e-9 or s_hi * m_size < 1.0 - 1e-9:
        s_lo = max(0.05, 1.0 / (2.0 * m_size))
        s_hi = min(0.95, 1.5 / m_size)

    candidates_s: list[np.ndarray] = [np.full(m_size, 1.0 / m_size)]
    step = 0.05
    grid_vals = np.round(np.arange(s_lo, s_hi + 1e-9, step), 4)
    for combo in product(grid_vals, repeat=max(0, m_size - 1)):
        rem = round(1.0 - sum(combo), 6)
        if s_lo - 1e-6 <= rem <= s_hi + 1e-6:
            candidates_s.append(np.array([*combo, max(0.0, rem)], dtype=np.float64))

    best_obj = -float("inf")
    best_s = np.full(m_size, 1.0 / m_size)
    best_res: tuple[float, np.ndarray, str, tuple[float, ...]] | None = None

    for s_vec in candidates_s:
        res = score_shares(s_vec)
        if res[0] > best_obj:
            best_obj = res[0]
            best_s = s_vec
            best_res = res

    # Local coordinate refinement around best simplex grid point
    for delta in (0.025, 0.01):
        improved = True
        while improved:
            improved = False
            for a in range(m_size):
                for b in range(m_size):
                    if a == b:
                        continue
                    trial = best_s.copy()
                    trial[a] += delta
                    trial[b] -= delta
                    if trial[a] <= s_hi + 1e-9 and trial[b] >= s_lo - 1e-9:
                        res = score_shares(trial)
                        if res[0] > best_obj + 1e-9:
                            best_obj = res[0]
                            best_s = trial
                            best_res = res
                            improved = True

    assert best_res is not None
    _, k_opt, bind_name, (net_ret, cdar, mdd, dcvar, mdaily, savings) = best_res
    warnings_list: list[str] = []
    for k_val, idx in zip(k_opt, pod):
        m_info = metrics_list[idx]
        min_pos_usd = k_val * m_info.active_pos_q10_frac * tensor.pod_equity_usd
        if m_info.active_pos_q10_frac > 0 and min_pos_usd < config.min_lot_usd:
            warnings_list.append(
                f"{m_info.alias}: q10_order=${min_pos_usd:.2f}<${config.min_lot_usd:.0f}"
            )

    feasible = (net_ret > 0.0) and (mdd <= config.max_dd_frac) and (mdaily <= config.daily_loss_frac)
    return SizedPodCandidate(
        aliases=tuple(tensor.aliases[idx] for idx in pod),
        pod_indices=pod,
        multipliers=k_opt,
        objective_net_return=net_ret,
        cdar_95=cdar,
        max_static_dd=mdd,
        daily_cvar_99=dcvar,
        max_daily_loss=mdaily,
        net_over_max_dd=net_ret / max(mdd, 1e-6),
        netting_savings_frac=savings,
        binding_constraint=bind_name,
        feasible=feasible,
        min_lot_warnings=tuple(warnings_list),
    )


def discover_diverse_pods(
    tensor: UnitCopy5mTensor,
    config: PodConstructorConfig = PodConstructorConfig(),
) -> tuple[list[SizedPodCandidate], list[TraderAuditMetrics], dict[int, TraderEpisodeBaseline], dict[str, np.ndarray]]:
    """Run Stage 1 -> 1.5a -> 1.5b -> Stage 2 end-to-end on the training slice."""
    passed_ids, metrics_list, baselines = stage1_audit_traders(tensor, config)
    if len(passed_ids) < config.pod_size:
        raise ValueError(
            f"Only {len(passed_ids)} wallets passed behavioral audit; need >= {config.pod_size}."
        )

    dist_mat, comps = compute_crowding_distance_matrix(passed_ids, tensor, config)
    n_clusters = min(config.pod_size, len(passed_ids))
    labels = cluster_candidates_upgma(dist_mat, n_clusters)

    clusters: dict[int, list[int]] = {}
    for pos_idx, c_label in enumerate(labels):
        tid = passed_ids[pos_idx]
        clusters.setdefault(int(c_label), []).append(tid)

    cluster_picks: list[list[int]] = []
    for c_id in sorted(clusters.keys()):
        members = clusters[c_id]
        # Preserve both all-window Net/DD leaders AND chop-resilient cluster members
        by_overall = sorted(members, key=lambda t: metrics_list[t].net_over_max_dd, reverse=True)
        by_chop = sorted(members, key=lambda t: metrics_list[t].chop_net_over_dd, reverse=True)
        picked: list[int] = []
        for lst in (by_overall, by_chop):
            for tid in lst:
                if tid not in picked and len(picked) < config.top_per_cluster:
                    picked.append(tid)
        cluster_picks.append(picked)

    rank_lookup = {
        tid: r_idx for lst in cluster_picks for r_idx, tid in enumerate(lst)
    }
    raw_combos = {
        tuple(sorted(p)) for p in product(*cluster_picks) if len(set(p)) == len(cluster_picks)
    }
    candidate_pods = sorted(
        raw_combos,
        key=lambda p: (sum(rank_lookup[t] for t in p), p),
    )[: config.max_candidate_pods]

    sized = [
        size_pod_dual_barrier(pod, tensor, metrics_list, config) for pod in candidate_pods
    ]
    sized.sort(
        key=lambda s: (s.feasible, s.net_over_max_dd, s.objective_net_return),
        reverse=True,
    )
    return sized, metrics_list, baselines, {"distance_matrix": dist_mat, **comps}


# ---------------------------------------------------------------------------
# Stage 4: Leak-Free CPCV/PBO & Guardian Stationary Block Bootstrap
# ---------------------------------------------------------------------------


def compute_leakfree_cpcv_pbo(
    candidate_pods: Sequence[tuple[int, ...]],
    tensor: UnitCopy5mTensor,
    config: PodConstructorConfig = PodConstructorConfig(),
) -> tuple[float, np.ndarray]:
    """Compute leak-free CPCV Probability of Backtest Overfitting (PBO).

    Unlike naive in-sample replay, weights k_i are re-solved strictly inside each
    training block subset (with boundary episode purging / embargo) before scoring OOS.
    """
    n_pods = len(candidate_pods)
    if n_pods < 2:
        return 0.0, np.array([1.0], dtype=np.float64)

    n_bars = tensor.timestamps_ms.size
    bpd = config.bars_per_day
    n_days = max(2, n_bars // bpd)
    n_blocks = min(config.cpcv_n_blocks, n_days // 2)
    if n_blocks % 2 != 0:
        n_blocks -= 1
    n_blocks = max(2, n_blocks)
    half_b = n_blocks // 2
    block_days = n_days // n_blocks
    embargo = config.cpcv_embargo_days

    logits: list[float] = []
    for train_blocks in combinations(range(n_blocks), half_b):
        train_set = set(train_blocks)
        tr_bars: list[int] = []
        te_bars: list[int] = []

        for b in range(n_blocks):
            sd = b * block_days
            ed = (b + 1) * block_days
            if b in train_set:
                if (b + 1 < n_blocks) and ((b + 1) not in train_set) and (ed - sd > embargo):
                    ed -= embargo
                if (b - 1 >= 0) and ((b - 1) not in train_set) and (ed - sd > embargo):
                    sd += embargo
                tr_bars.extend(range(sd * bpd, ed * bpd))
            else:
                te_bars.extend(range(sd * bpd, ed * bpd))

        if len(tr_bars) < bpd * 2 or len(te_bars) < bpd * 2:
            continue

        tr_idx = np.asarray(tr_bars, dtype=np.int32)
        te_idx = np.asarray(te_bars, dtype=np.int32)

        tr_tensor = UnitCopy5mTensor(
            aliases=tensor.aliases,
            symbols=tensor.symbols,
            timestamps_ms=tensor.timestamps_ms[tr_idx],
            notional_frac=tensor.notional_frac[:, :, tr_idx],
            delta_notional_frac=tensor.delta_notional_frac[:, :, tr_idx],
            gross_ret_frac=tensor.gross_ret_frac[:, tr_idx],
            net_ret_frac=tensor.net_ret_frac[:, tr_idx],
            sym_net_ret_frac=(
                tensor.sym_net_ret_frac[:, :, tr_idx]
                if tensor.sym_net_ret_frac is not None
                else None
            ),
            pod_equity_usd=tensor.pod_equity_usd,
        )
        te_tensor = UnitCopy5mTensor(
            aliases=tensor.aliases,
            symbols=tensor.symbols,
            timestamps_ms=tensor.timestamps_ms[te_idx],
            notional_frac=tensor.notional_frac[:, :, te_idx],
            delta_notional_frac=tensor.delta_notional_frac[:, :, te_idx],
            gross_ret_frac=tensor.gross_ret_frac[:, te_idx],
            net_ret_frac=tensor.net_ret_frac[:, te_idx],
            sym_net_ret_frac=(
                tensor.sym_net_ret_frac[:, :, te_idx]
                if tensor.sym_net_ret_frac is not None
                else None
            ),
            pod_equity_usd=tensor.pod_equity_usd,
        )

        _, tr_metrics, _ = stage1_audit_traders(tr_tensor, config)
        te_slices = _daily_reset_slices(te_tensor.timestamps_ms, bpd, config.reset_bar_offset)
        cost_rate = config.cost_bps * 1e-4

        is_scores = np.zeros(n_pods, dtype=np.float64)
        oos_scores = np.zeros(n_pods, dtype=np.float64)
        for p_idx, pod in enumerate(candidate_pods):
            sized_tr = size_pod_dual_barrier(pod, tr_tensor, tr_metrics, config)
            is_scores[p_idx] = sized_tr.net_over_max_dd

            solo_te, t_ov_te, delta_ov_te = _build_sparse_overlap_cache(
                te_tensor.delta_notional_frac[list(pod)],
                te_tensor.net_ret_frac[list(pod)],
            )
            ret_te, mdd_te, _, _, _, _ = _evaluate_weights_unit_scale(
                sized_tr.multipliers,
                solo_te,
                t_ov_te,
                delta_ov_te,
                cost_rate,
                te_slices,
                config,
            )
            oos_scores[p_idx] = ret_te / max(mdd_te, 1e-6)

        best_is = int(np.argmax(is_scores))
        ranks = rankdata_average(oos_scores)
        omega = float(np.clip(ranks[best_is] / float(n_pods + 1), 1e-4, 1.0 - 1e-4))
        logits.append(math.log(omega / (1.0 - omega)))

    logits_arr = np.asarray(logits, dtype=np.float64) if logits else np.array([0.0])
    pbo = float(np.mean(logits_arr <= 0.0))
    return pbo, logits_arr


def sample_stationary_block_indices(
    n_days: int,
    n_paths: int,
    horizon_days: int,
    mean_block_days: float,
    day_jump_probs: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Politis-Romano (1994) stationary block index generator for UTC daily blocks.

    Supports both uniform resampling (``day_jump_probs=None``) and regime-weighted
    block jumps (e.g. ``75%-Chop``), while stepping sequentially within each random
    geometric block so multi-day swing chains and volatility clustering survive intact.
    """
    if rng is None:
        rng = np.random.default_rng()
    p_vec = (
        np.full(n_days, 1.0 / n_days, dtype=np.float64)
        if day_jump_probs is None
        else np.asarray(day_jump_probs, dtype=np.float64)
    )
    p_vec = p_vec / np.sum(p_vec)
    p_geom = min(1.0, 1.0 / max(1.0, mean_block_days))

    out = np.empty((n_paths, horizon_days), dtype=np.int32)
    out[:, 0] = rng.choice(n_days, size=n_paths, p=p_vec)
    jumps = rng.random((n_paths, horizon_days)) < p_geom
    jump_targets = rng.choice(n_days, size=(n_paths, horizon_days), p=p_vec)
    for t in range(1, horizon_days):
        out[:, t] = np.where(jumps[:, t], jump_targets[:, t], (out[:, t - 1] + 1) % n_days)
    return out


def run_stationary_block_mc_on_daily_blocks(
    blocks: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    regime: Literal["Blended", "75%-Chop"],
    *,
    account: float = 10_000.0,
    profit_target: float = 800.0,
    max_static_dd: float = 1_000.0,
    daily_loss_limit: float = 500.0,
    n_paths: int = 5_000,
    horizon_days: int = 90,
    mean_block_days: float = 7.0,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Run Politis-Romano stationary block bootstrap on exact Guardian-simulated daily blocks.

    Drop-in upgrade for ``run_monte_carlo`` in ``2026-11-15__cft__10k__portfolio-candidates.py``:
    accepts the exact 6-tuple ``(is_chop, day_end, day_max, day_min, day_swept, day_max_first)``.
    """
    is_chop, day_end, day_max, day_min, day_swept, day_max_first = blocks
    n_days = day_end.size
    if regime == "Blended":
        probs = np.full(n_days, 1.0 / n_days, dtype=np.float64)
    elif regime == "75%-Chop":
        chop_n = int(np.sum(is_chop))
        trend_n = n_days - chop_n
        if chop_n == 0 or trend_n == 0:
            probs = np.full(n_days, 1.0 / n_days, dtype=np.float64)
        else:
            probs = np.where(is_chop, 0.75 / chop_n, 0.25 / trend_n)
    else:
        raise ValueError(f"Unsupported regime: {regime}")

    sample_idx = sample_stationary_block_indices(
        n_days=n_days,
        n_paths=n_paths,
        horizon_days=horizon_days,
        mean_block_days=mean_block_days,
        day_jump_probs=probs,
        rng=rng,
    )

    equity = np.full(n_paths, account, dtype=np.float64)
    status = np.zeros(n_paths, dtype=np.int8)  # 0=active, 1=PASS, 2=FAIL_STATIC, 3=FAIL_DAILY
    pass_day = np.full(n_paths, -1, dtype=np.int32)
    sweeps = np.zeros(n_paths, dtype=np.int32)

    for step in range(horizon_days):
        active = status == 0
        if not np.any(active):
            break
        idx = sample_idx[active, step]
        eq_a = equity[active]
        d_end, d_max, d_min = day_end[idx], day_max[idx], day_min[idx]
        d_swp, d_mf = day_swept[idx], day_max_first[idx]

        hit_pass = (eq_a + d_max) >= (account + profit_target)
        hit_static = (eq_a + d_min) <= (account - max_static_dd)
        hit_daily = d_min <= -daily_loss_limit

        new_status = np.zeros(eq_a.size, dtype=np.int8)
        both_pf = hit_pass & (hit_static | hit_daily)
        only_p = hit_pass & ~both_pf
        only_f = ~hit_pass & (hit_static | hit_daily)

        fail_type = np.where(
            hit_daily & (~hit_static | ((account - max_static_dd) < (eq_a - daily_loss_limit))),
            3,
            2,
        )
        new_status[only_p] = 1
        new_status[only_f] = fail_type[only_f]
        new_status[both_pf] = np.where(d_mf[both_pf], 1, fail_type[both_pf])

        act_idx = np.flatnonzero(active)
        passed_now = act_idx[new_status == 1]
        status[passed_now] = 1
        pass_day[passed_now] = step + 1
        status[act_idx[new_status == 2]] = 2
        status[act_idx[new_status == 3]] = 3

        still_mask = new_status == 0
        still_idx = act_idx[still_mask]
        sweeps[still_idx] += d_swp[still_mask]
        equity[still_idx] += d_end[still_mask]

    return {
        "p_pass": round(float(np.mean(status == 1)) * 100.0, 2),
        "p_fail_static": round(float(np.mean(status == 2)) * 100.0, 2),
        "p_fail_daily": round(float(np.mean(status == 3)) * 100.0, 2),
        "p_timeout": round(float(np.mean(status == 0)) * 100.0, 2),
        "med_days_pass": round(float(np.median(pass_day[status == 1])), 1)
        if np.any(status == 1)
        else -1.0,
        "avg_sweeps": round(float(np.mean(sweeps)), 2),
    }


# ---------------------------------------------------------------------------
# Stage 5 Advisory Monitor: Episode-Time CUSUM for wallet-research Stage 1
# ---------------------------------------------------------------------------


def update_roster_episode_cusum(
    state: RosterCUSUMState,
    baseline: TraderEpisodeBaseline,
    *,
    ep_return: float,
    ep_peak_notional_frac: float,
    ep_duration_bars: float,
    ep_mae: float,
    ep_symbol_notional: np.ndarray,
    config: PodConstructorConfig = PodConstructorConfig(),
) -> RosterCUSUMState:
    """Update Page's dual-channel CUSUM (edge decay + behavioral tilt) for roster audit.

    Emits non-executing governance status (``OK`` / ``WARN_THROTTLE_0.5X`` /
    ``KILL_CONTAIN_0X``) to feed the manual ``Contain -> Replace -> Version`` workflow.
    """
    state.episodes_observed += 1
    z_pnl = (ep_return - baseline.mean_ep_return) / max(baseline.std_ep_return, 1e-5)

    ln_peak = math.log(max(ep_peak_notional_frac, 1e-8))
    z_size = (ln_peak - baseline.median_ln_peak_notional) / (
        baseline.mad_ln_peak_notional * 1.4826
    )

    if ep_return < 0.0:
        z_hold = (
            math.log(max(ep_duration_bars, 1.0)) - baseline.mean_win_ln_duration
        ) / max(baseline.std_win_ln_duration, 0.10)
    else:
        z_hold = 0.0

    state.recent_symbol_notionals.append(np.asarray(ep_symbol_notional, dtype=np.float64))
    if len(state.recent_symbol_notionals) > 10:
        state.recent_symbol_notionals.pop(0)
    roll_sym = np.sum(np.vstack(state.recent_symbol_notionals), axis=0)
    z_jsd = compute_jensen_shannon_divergence(baseline.symbol_notional_dist, roll_sym) / 0.15

    b_score = 0.50 * max(0.0, z_size) + 0.30 * max(0.0, z_hold) + 0.20 * max(0.0, z_jsd)
    state.s_decay = max(0.0, state.s_decay - z_pnl - config.cusum_slack_k)
    state.s_tilt = max(0.0, state.s_tilt + b_score - config.cusum_slack_k)

    mae_breach = ep_mae > (config.mae_kill_multiple * baseline.mae_p95)
    reasons: list[str] = []
    if state.s_decay > config.cusum_kill_h:
        reasons.append(f"CUSUM_decay({state.s_decay:.2f}>{config.cusum_kill_h})")
    if state.s_tilt > config.cusum_kill_h:
        reasons.append(f"CUSUM_tilt({state.s_tilt:.2f}>{config.cusum_kill_h})")
    if mae_breach:
        reasons.append(f"MAE_breach({ep_mae:.4f}>{config.mae_kill_multiple * baseline.mae_p95:.4f})")

    if reasons:
        state.status = "KILL_CONTAIN_0X"
        state.reasons = tuple(reasons)
        return state

    warn_reasons: list[str] = []
    if state.s_decay > config.cusum_warn_h:
        warn_reasons.append(f"CUSUM_decay_warn({state.s_decay:.2f}>{config.cusum_warn_h})")
    if state.s_tilt > config.cusum_warn_h:
        warn_reasons.append(f"CUSUM_tilt_warn({state.s_tilt:.2f}>{config.cusum_warn_h})")

    if warn_reasons:
        state.status = "WARN_THROTTLE_0.5X"
        state.reasons = tuple(warn_reasons)
    else:
        state.status = "OK"
        state.reasons = ()
    return state
