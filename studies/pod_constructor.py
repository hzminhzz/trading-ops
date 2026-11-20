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
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import combinations, product
from typing import Any, Literal

import numpy as np
from copytrade_kernel.rulebooks import ChallengeRulebookSpec, RulebookCatalog

StyleSleeve = Literal[
    "intraday_tactical",
    "multi_day_swing",
    "positional",
    "mixed",
    "unlabeled",
]

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


def compute_drawdown_metrics(
    cum_returns: np.ndarray,
    alpha: float = 0.95,
) -> tuple[np.ndarray, float, float, float]:
    """Return underwater curve u_t >= 0, peak-to-trough DD, initial-floor static DD, and CDaR."""
    running_peak = np.maximum(0.0, np.maximum.accumulate(cum_returns))
    underwater = running_peak - cum_returns
    peak_to_trough_dd = float(np.max(underwater)) if underwater.size > 0 else 0.0
    static_floor_dd = float(max(0.0, -float(np.min(cum_returns)))) if cum_returns.size > 0 else 0.0
    q_alpha = float(np.quantile(underwater, alpha)) if underwater.size > 0 else 0.0
    tail_mask = underwater >= q_alpha
    cdar = float(np.mean(underwater[tail_mask])) if np.any(tail_mask) else q_alpha
    return underwater, peak_to_trough_dd, static_floor_dd, cdar


def compute_underwater_and_cdar(
    cum_returns: np.ndarray,
    alpha: float = 0.95,
) -> tuple[np.ndarray, float, float]:
    """Return underwater curve u_t >= 0, peak-to-trough MaxDD, and Conditional Drawdown-at-Risk."""
    underwater, peak_to_trough_dd, _static_dd, cdar = compute_drawdown_metrics(
        cum_returns, alpha=alpha
    )
    return underwater, peak_to_trough_dd, cdar


def compute_segmented_drawdown_metrics(
    r_net: np.ndarray,
    segments: Sequence[tuple[int, int]],
    alpha: float = 0.95,
) -> tuple[np.ndarray, float, float, float]:
    """Compute drawdown metrics separately per contiguous block to avoid cross-gap splicing."""
    valid_segs = [(int(s), int(e)) for s, e in segments if e > s]
    if not valid_segs:
        empty = np.zeros(0, dtype=np.float64)
        return empty, 0.0, 0.0, 0.0
    u_parts: list[np.ndarray] = []
    peak_dd = 0.0
    static_dd = 0.0
    for s, e in valid_segs:
        cum_seg = np.cumsum(r_net[s:e])
        u_seg, pdd_seg, sdd_seg, _ = compute_drawdown_metrics(cum_seg, alpha=alpha)
        u_parts.append(u_seg)
        peak_dd = max(peak_dd, pdd_seg)
        static_dd = max(static_dd, sdd_seg)
    underwater = np.concatenate(u_parts) if len(u_parts) > 1 else u_parts[0]
    q_alpha = float(np.quantile(underwater, alpha)) if underwater.size > 0 else 0.0
    tail_mask = underwater >= q_alpha
    cdar = float(np.mean(underwater[tail_mask])) if np.any(tail_mask) else q_alpha
    return underwater, peak_dd, static_dd, cdar


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
    min_tilt_comparisons: int = 1
    reject_unknown_tilt: bool = True
    reject_hidden_bagholder: bool = True
    max_bagholder_open_loss_ratio: float = 0.50
    min_active_bar_frac: float = 0.01
    baseline_port_dd_frac: float = 0.02
    min_baseline_order_usd: float = 0.0  # >0 filters q10_order at baseline k before clustering

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
    risk_share_mode: Literal["standalone_proxy", "marginal_euler"] = "standalone_proxy"
    max_marginal_cdar_share: float | None = None
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
    latch_kill_state: bool = True

    min_episode_notional_usd: float = 10.0
    min_overlap_days: int = 5
    sizing_tier: Literal[1, 2, 3, 4] | None = None
    tier_peak_dd_cap_frac: float | None = None
    tier_daily_loss_cap_frac: float | None = None
    max_symbol_exposure_frac: float | None = None
    max_gross_exposure_frac: float | None = None

    @classmethod
    def for_sizing_tier(
        cls,
        tier: Literal[1, 2, 3, 4],
        *,
        base: PodConstructorConfig | None = None,
    ) -> PodConstructorConfig:
        """Return pre-registered sizing configuration for Tiers 1–4."""
        b = base if base is not None else cls()
        max_share = min(b.max_risk_share, 0.45) if b.pod_size >= 3 else b.max_risk_share
        if tier == 1:
            return replace(
                b,
                sizing_tier=1,
                tier_peak_dd_cap_frac=None,
                tier_daily_loss_cap_frac=None,
                max_symbol_exposure_frac=None,
                max_gross_exposure_frac=None,
            )
        if tier == 2:
            return replace(
                b,
                sizing_tier=2,
                tier_peak_dd_cap_frac=0.03,
                tier_daily_loss_cap_frac=0.30 * b.daily_loss_frac,
                max_risk_share=max_share,
                max_symbol_exposure_frac=1.5,
                max_gross_exposure_frac=3.0,
            )
        if tier == 3:
            return replace(
                b,
                sizing_tier=3,
                tier_peak_dd_cap_frac=0.05,
                tier_daily_loss_cap_frac=0.50 * b.daily_loss_frac,
                max_risk_share=max_share,
                max_symbol_exposure_frac=1.5,
                max_gross_exposure_frac=3.0,
            )
        if tier == 4:
            return replace(
                b,
                sizing_tier=4,
                tier_peak_dd_cap_frac=0.065,
                tier_daily_loss_cap_frac=0.65 * b.daily_loss_frac,
                max_risk_share=max_share,
                max_symbol_exposure_frac=1.5,
                max_gross_exposure_frac=3.0,
            )
        raise ValueError(f"INVALID_SIZING_TIER:{tier}")


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
    exec_open_px: np.ndarray | None = None  # (S, T) aligned open prices
    mark_close_px: np.ndarray | None = None  # (S, T) aligned close mark prices
    carried_in_mask: np.ndarray | None = None  # (N, S) pre-window carry-in inventory flags

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
        if self.exec_open_px is not None and self.exec_open_px.shape != (s, t):
            raise ValueError("EXEC_OPEN_PX_SHAPE_MISMATCH")
        if self.mark_close_px is not None and self.mark_close_px.shape != (s, t):
            raise ValueError("MARK_CLOSE_PX_SHAPE_MISMATCH")
        if self.carried_in_mask is not None and self.carried_in_mask.shape != (n, s):
            raise ValueError("CARRIED_IN_MASK_SHAPE_MISMATCH")

    def slice_time(self, start_bar: int, end_bar: int) -> UnitCopy5mTensor:
        """Return a contiguous time slice [start_bar, end_bar)."""
        if start_bar > 0:
            prev_n = self.notional_frac[:, :, start_bar - 1]
            cur_n = (
                self.notional_frac[:, :, start_bar]
                if end_bar > start_bar
                else np.zeros_like(prev_n)
            )
            carried = (np.abs(prev_n) > 1e-4) & ((prev_n * cur_n) > 0.0)
        else:
            carried = self.carried_in_mask
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
            exec_open_px=(
                self.exec_open_px[:, start_bar:end_bar]
                if self.exec_open_px is not None
                else None
            ),
            mark_close_px=(
                self.mark_close_px[:, start_bar:end_bar]
                if self.mark_close_px is not None
                else None
            ),
            carried_in_mask=carried,
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
    standalone_peak_to_trough_dd: float
    standalone_static_floor_dd: float
    tilt_multiplier: float | None
    tilt_status: Literal["OK", "FAIL", "UNKNOWN"]
    active_pos_q10_frac: float
    passed_behavioral_gate: bool
    reject_reason: str = ""
    censored_episodes: int = 0
    left_censored_episodes: int = 0
    right_censored_episodes: int = 0
    open_exposure_at_cutoff: float = 0.0
    raw_ropn_mean: float = 0.0
    eb_ropn_mean: float = 0.0
    eb_ropn_post_sd: float = 0.0
    right_censored_pnl_frac: float = 0.0
    style_sleeve: StyleSleeve = "unlabeled"
    win_hold_p95_bars: float | None = None
    within_ep_scaling_median: float = 1.0

    @property
    def mean_ep_ropn(self) -> float:
        """Backward-compatible alias for ``raw_ropn_mean``."""
        return self.raw_ropn_mean

    @property
    def eb_shrunk_ropn(self) -> float:
        """Backward-compatible alias for ``eb_ropn_mean``."""
        return self.eb_ropn_mean

    @property
    def eb_ropn_se(self) -> float:
        """Backward-compatible alias for ``eb_ropn_post_sd``."""
        return self.eb_ropn_post_sd


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
    qualified: bool = True


@dataclass
class SizedPodCandidate:
    """Ranked candidate pod with joint dual-barrier calibrated fixed multipliers k_i."""

    aliases: tuple[str, ...]
    pod_indices: tuple[int, ...]
    multipliers: np.ndarray
    objective_net_return: float
    cdar_95: float
    peak_to_trough_dd: float
    static_floor_dd: float
    daily_cvar_99: float
    max_daily_loss: float
    net_over_max_dd: float
    netting_savings_frac: float
    binding_constraint: str
    feasible: bool
    min_lot_warnings: tuple[str, ...] = ()
    peak_symbol_exposure_frac: float = 0.0
    peak_gross_exposure_frac: float = 0.0
    marginal_cdar_shares: tuple[float, ...] = ()
    marginal_daily_cvar_shares: tuple[float, ...] = ()


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
    terminal_kill_latched: bool = False
    latched_at_episode: int | None = None
    latched_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize CUSUM state for persistent storage across process restarts."""
        return {
            "alias": self.alias,
            "s_decay": float(self.s_decay),
            "s_tilt": float(self.s_tilt),
            "episodes_observed": int(self.episodes_observed),
            "recent_symbol_notionals": [arr.tolist() for arr in self.recent_symbol_notionals],
            "status": self.status,
            "reasons": list(self.reasons),
            "terminal_kill_latched": bool(self.terminal_kill_latched),
            "latched_at_episode": self.latched_at_episode,
            "latched_reasons": list(self.latched_reasons),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RosterCUSUMState:
        """Restore persisted CUSUM state including any terminal kill latch."""
        raw_recent = data.get("recent_symbol_notionals", [])
        recent = (
            [np.asarray(x, dtype=np.float64) for x in raw_recent]
            if isinstance(raw_recent, Sequence) and not isinstance(raw_recent, (str, bytes))
            else []
        )
        raw_reasons = data.get("reasons", [])
        reasons = (
            tuple(str(x) for x in raw_reasons)
            if isinstance(raw_reasons, Sequence) and not isinstance(raw_reasons, (str, bytes))
            else ()
        )
        raw_latched = data.get("latched_reasons", [])
        latched_reasons = (
            tuple(str(x) for x in raw_latched)
            if isinstance(raw_latched, Sequence) and not isinstance(raw_latched, (str, bytes))
            else ()
        )
        lat_ep = data.get("latched_at_episode")
        return cls(
            alias=str(data["alias"]),
            s_decay=float(data.get("s_decay", 0.0)),  # type: ignore[arg-type]
            s_tilt=float(data.get("s_tilt", 0.0)),  # type: ignore[arg-type]
            episodes_observed=int(data.get("episodes_observed", 0)),  # type: ignore[arg-type]
            recent_symbol_notionals=recent,
            status=str(data.get("status", "OK")),  # type: ignore[arg-type]
            reasons=reasons,
            terminal_kill_latched=bool(data.get("terminal_kill_latched", False)),
            latched_at_episode=int(lat_ep) if isinstance(lat_ep, (int, float)) else None,
            latched_reasons=latched_reasons,
        )

    def reset_for_new_mandate_generation(self) -> None:
        """Explicitly clear latch and accumulators only when a new approved mandate generation starts."""
        self.s_decay = 0.0
        self.s_tilt = 0.0
        self.status = "OK"
        self.reasons = ()
        self.terminal_kill_latched = False
        self.latched_at_episode = None
        self.latched_reasons = ()


# ---------------------------------------------------------------------------
# Tensor Builder Adapter from Policy A Replay Output
# ---------------------------------------------------------------------------


def _init_unit_tensor_buffers(
    unit_map: Mapping[str, Mapping[str, object]],
    timestamps_ms: Sequence[int] | np.ndarray,
    *,
    pod_equity_usd: float,
    tensor_dtype: np.dtype | type,
    include_sym_net_ret: bool,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
]:
    if pod_equity_usd <= 0.0:
        raise ValueError(f"pod_equity_usd must be positive, got {pod_equity_usd}")
    dtype = np.dtype(tensor_dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"tensor_dtype must be float32 or float64, got {dtype}")

    aliases = tuple(sorted(unit_map.keys()))
    symbol_set: set[str] = set()
    for by_coin in unit_map.values():
        symbol_set.update(by_coin.keys())
    symbols = tuple(sorted(symbol_set))

    ts_arr = np.asarray(timestamps_ms, dtype=np.int64)
    n_traders, n_syms, n_bars = len(aliases), len(symbols), ts_arr.size

    notional_frac = np.zeros((n_traders, n_syms, n_bars), dtype=dtype)
    delta_notional_frac = np.zeros((n_traders, n_syms, n_bars), dtype=dtype)
    gross_ret_frac = np.zeros((n_traders, n_bars), dtype=np.float64)
    net_ret_frac = np.zeros((n_traders, n_bars), dtype=np.float64)
    sym_net_ret_frac = (
        np.zeros((n_traders, n_syms, n_bars), dtype=dtype) if include_sym_net_ret else None
    )
    return (
        aliases,
        symbols,
        ts_arr,
        notional_frac,
        delta_notional_frac,
        gross_ret_frac,
        net_ret_frac,
        sym_net_ret_frac,
    )


def _write_aligned_coin_arrays(
    ti: int,
    si: int,
    bar_idx: np.ndarray | slice,
    notional_usd: np.ndarray,
    delta_notional_usd: np.ndarray,
    gross_usd: np.ndarray,
    net_usd: np.ndarray,
    *,
    inv_equity: float,
    notional_frac: np.ndarray,
    delta_notional_frac: np.ndarray,
    gross_ret_frac: np.ndarray,
    net_ret_frac: np.ndarray,
    sym_net_ret_frac: np.ndarray | None,
) -> None:
    # Skip writing all-zero slices so untouched OS zero pages consume no physical RSS.
    if notional_usd.any():
        notional_frac[ti, si, bar_idx] = notional_usd * inv_equity
    if delta_notional_usd.any():
        delta_notional_frac[ti, si, bar_idx] = delta_notional_usd * inv_equity
    if gross_usd.any():
        gross_ret_frac[ti, bar_idx] += gross_usd * inv_equity
    if net_usd.any():
        coin_net = net_usd * inv_equity
        net_ret_frac[ti, bar_idx] += coin_net
        if sym_net_ret_frac is not None:
            sym_net_ret_frac[ti, si, bar_idx] = coin_net


def build_tensor_from_unit_replays(
    unit_pos_by_alias: dict[str, dict[str, object]],
    timestamps_ms: Sequence[int] | np.ndarray,
    *,
    pod_equity_usd: float = 10_000.0,
    chop_split_ms: int | None = None,
    tensor_dtype: np.dtype | type = np.float32,
    include_sym_net_ret: bool = True,
) -> UnitCopy5mTensor:
    """Aggregate Policy A replay DataFrames into an execution-aligned 5m tensor.

    Accepts ``pos_by_coin`` dicts returned by ``replay_portfolio_5m({alias: 1.0}, ...)``
    or ``canonical_replay.build_unit_replay_by_alias`` where each Polars DataFrame holds
    ``timestamp, open, close, pos, delta, gross, cost, net``. Uses a direct NumPy fast-path
    when rows are already aligned to ``timestamps_ms`` and falls back to Polars 5m
    aggregation + ``searchsorted`` alignment otherwise.
    """
    import polars as pl

    (
        aliases,
        symbols,
        ts_arr,
        notional_frac,
        delta_notional_frac,
        gross_ret_frac,
        net_ret_frac,
        sym_net_ret_frac,
    ) = _init_unit_tensor_buffers(
        unit_pos_by_alias,
        timestamps_ms,
        pod_equity_usd=pod_equity_usd,
        tensor_dtype=tensor_dtype,
        include_sym_net_ret=include_sym_net_ret,
    )
    n_bars = ts_arr.size
    n_syms = len(symbols)
    inv_eq = 1.0 / pod_equity_usd
    sym_idx = {s: idx for idx, s in enumerate(symbols)}
    ts_strict_inc = n_bars > 0 and bool(np.all(ts_arr[1:] > ts_arr[:-1]))
    ts_ms_col: pl.DataFrame | None = None
    open_buf = np.zeros((n_syms, n_bars), dtype=np.float64) if (n_syms > 0 and n_bars > 0) else None
    close_buf = np.zeros((n_syms, n_bars), dtype=np.float64) if (n_syms > 0 and n_bars > 0) else None
    sym_px_seen = np.zeros(n_syms, dtype=bool)

    for ti, alias in enumerate(aliases):
        for coin, df_obj in unit_pos_by_alias[alias].items():
            si = sym_idx[coin]
            df = df_obj  # type: ignore[assignment]
            if df.height == 0 or n_bars == 0:
                continue

            has_close = "close" in df.columns
            # Fast path: DataFrame already aligned 1-to-1 with timestamps_ms and non-null
            if (
                df.height == n_bars
                and df["open"].null_count() == 0
                and df["pos"].null_count() == 0
                and df["delta"].null_count() == 0
                and df["gross"].null_count() == 0
                and df["net"].null_count() == 0
            ):
                ts_col = df["timestamp"]
                df_ts_ms = (
                    ts_col.to_numpy()
                    if ts_col.dtype.is_integer()
                    else ts_col.dt.epoch("ms").to_numpy()
                )
                if np.array_equal(df_ts_ms, ts_arr):
                    open_arr = df["open"].to_numpy().astype(np.float64, copy=False)
                    pos_arr = df["pos"].to_numpy().astype(np.float64, copy=False)
                    delta_arr = df["delta"].to_numpy().astype(np.float64, copy=False)
                    gross_arr = df["gross"].to_numpy().astype(np.float64, copy=False)
                    net_arr = df["net"].to_numpy().astype(np.float64, copy=False)
                    if (
                        has_close
                        and open_buf is not None
                        and close_buf is not None
                        and df["close"].null_count() == 0
                        and not sym_px_seen[si]
                    ):
                        open_buf[si] = open_arr
                        close_buf[si] = df["close"].to_numpy().astype(np.float64, copy=False)
                        sym_px_seen[si] = True
                    _write_aligned_coin_arrays(
                        ti,
                        si,
                        slice(None),
                        pos_arr * open_arr,
                        delta_arr * open_arr,
                        gross_arr,
                        net_arr,
                        inv_equity=inv_eq,
                        notional_frac=notional_frac,
                        delta_notional_frac=delta_notional_frac,
                        gross_ret_frac=gross_ret_frac,
                        net_ret_frac=net_ret_frac,
                        sym_net_ret_frac=sym_net_ret_frac,
                    )
                    continue

            df_sorted = df if df["timestamp"].is_sorted() else df.sort("timestamp")
            ts_expr = (
                pl.col("timestamp")
                if df_sorted["timestamp"].dtype.is_integer()
                else pl.col("timestamp").dt.epoch("ms")
            )
            agg_exprs = [
                (pl.col("pos").first() * pl.col("open").first()).alias("notional"),
                (pl.col("delta").first() * pl.col("open").first()).alias("d_notional"),
                pl.col("gross").sum().alias("gross"),
                pl.col("net").sum().alias("net"),
            ]
            if has_close:
                agg_exprs.extend(
                    [
                        pl.col("open").first().alias("open_px"),
                        pl.col("close").last().alias("close_px"),
                    ]
                )
            b5 = (
                df_sorted.with_columns(((ts_expr // 300_000) * 300_000).alias("ts_ms"))
                .group_by("ts_ms", maintain_order=True)
                .agg(agg_exprs)
                .fill_null(0.0)
            )
            if ts_strict_inc:
                b5_ts = b5["ts_ms"].to_numpy()
                ix = np.searchsorted(ts_arr, b5_ts)
                valid = (ix < n_bars) & (ts_arr[np.minimum(ix, n_bars - 1)] == b5_ts)
                if not np.any(valid):
                    continue
                v_ix = ix[valid]
                notional_arr = b5["notional"].to_numpy().astype(np.float64, copy=False)[valid]
                d_notional_arr = b5["d_notional"].to_numpy().astype(np.float64, copy=False)[valid]
                gross_arr = b5["gross"].to_numpy().astype(np.float64, copy=False)[valid]
                net_arr = b5["net"].to_numpy().astype(np.float64, copy=False)[valid]
                if has_close and open_buf is not None and close_buf is not None and not sym_px_seen[si]:
                    open_buf[si, v_ix] = b5["open_px"].to_numpy().astype(np.float64, copy=False)[valid]
                    close_buf[si, v_ix] = b5["close_px"].to_numpy().astype(np.float64, copy=False)[valid]
                    if v_ix.size == n_bars:
                        sym_px_seen[si] = True
                bar_idx: np.ndarray | slice = v_ix
            else:
                if ts_ms_col is None:
                    ts_ms_col = pl.DataFrame({"ts_ms": ts_arr})
                aligned = ts_ms_col.join(b5, on="ts_ms", how="left").fill_null(0.0)
                notional_arr = aligned["notional"].to_numpy().astype(np.float64, copy=False)
                d_notional_arr = aligned["d_notional"].to_numpy().astype(np.float64, copy=False)
                gross_arr = aligned["gross"].to_numpy().astype(np.float64, copy=False)
                net_arr = aligned["net"].to_numpy().astype(np.float64, copy=False)
                bar_idx = slice(None)

            _write_aligned_coin_arrays(
                ti,
                si,
                bar_idx,
                notional_arr,
                d_notional_arr,
                gross_arr,
                net_arr,
                inv_equity=inv_eq,
                notional_frac=notional_frac,
                delta_notional_frac=delta_notional_frac,
                gross_ret_frac=gross_ret_frac,
                net_ret_frac=net_ret_frac,
                sym_net_ret_frac=sym_net_ret_frac,
            )

    has_full_px = (
        open_buf is not None
        and close_buf is not None
        and n_syms > 0
        and bool(np.all(sym_px_seen))
    )
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
        exec_open_px=open_buf if has_full_px else None,
        mark_close_px=close_buf if has_full_px else None,
    )


def build_tensor_from_arrays(
    unit_arrays_by_alias: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    timestamps_ms: Sequence[int] | np.ndarray,
    *,
    pod_equity_usd: float = 10_000.0,
    chop_split_ms: int | None = None,
    tensor_dtype: np.dtype | type = np.float32,
    include_sym_net_ret: bool = True,
) -> UnitCopy5mTensor:
    """Build a ``UnitCopy5mTensor`` directly from pre-aligned 1D NumPy arrays.

    Each ``unit_arrays_by_alias[alias][coin]`` mapping must contain arrays of shape ``(T,)``:
      - Notional inputs: either ``("notional", "delta_notional")`` or ``("open", "pos", "delta")``
      - Return inputs: ``"gross"`` and either ``"net"`` or ``"cost"``
    """
    (
        aliases,
        symbols,
        ts_arr,
        notional_frac,
        delta_notional_frac,
        gross_ret_frac,
        net_ret_frac,
        sym_net_ret_frac,
    ) = _init_unit_tensor_buffers(
        unit_arrays_by_alias,
        timestamps_ms,
        pod_equity_usd=pod_equity_usd,
        tensor_dtype=tensor_dtype,
        include_sym_net_ret=include_sym_net_ret,
    )
    n_bars = ts_arr.size
    n_syms = len(symbols)
    inv_eq = 1.0 / pod_equity_usd
    sym_idx = {s: idx for idx, s in enumerate(symbols)}
    open_buf = np.zeros((n_syms, n_bars), dtype=np.float64) if (n_syms > 0 and n_bars > 0) else None
    close_buf = np.zeros((n_syms, n_bars), dtype=np.float64) if (n_syms > 0 and n_bars > 0) else None
    sym_px_seen = np.zeros(n_syms, dtype=bool)

    def _vec(arrs: Mapping[str, np.ndarray], key: str, alias: str, coin: str) -> np.ndarray:
        arr = np.asarray(arrs[key], dtype=np.float64)
        if arr.shape != (n_bars,):
            raise ValueError(
                f"Array '{key}' for ({alias}, {coin}) has shape {arr.shape}, expected ({n_bars},)"
            )
        return arr

    for ti, alias in enumerate(aliases):
        for coin, arrs in unit_arrays_by_alias[alias].items():
            si = sym_idx[coin]
            if "notional" in arrs and "delta_notional" in arrs:
                notional_usd = _vec(arrs, "notional", alias, coin)
                delta_notional_usd = _vec(arrs, "delta_notional", alias, coin)
            elif "open" in arrs and "pos" in arrs and "delta" in arrs:
                open_arr = _vec(arrs, "open", alias, coin)
                notional_usd = _vec(arrs, "pos", alias, coin) * open_arr
                delta_notional_usd = _vec(arrs, "delta", alias, coin) * open_arr
            else:
                raise ValueError(
                    f"Missing notional keys for ({alias}, {coin}); provide "
                    "('notional', 'delta_notional') or ('open', 'pos', 'delta')"
                )

            if (
                "open" in arrs
                and "close" in arrs
                and open_buf is not None
                and close_buf is not None
                and not sym_px_seen[si]
            ):
                open_buf[si] = _vec(arrs, "open", alias, coin)
                close_buf[si] = _vec(arrs, "close", alias, coin)
                sym_px_seen[si] = True

            if "gross" not in arrs:
                raise ValueError(f"Missing 'gross' array for ({alias}, {coin})")
            gross_usd = _vec(arrs, "gross", alias, coin)
            if "net" in arrs:
                net_usd = _vec(arrs, "net", alias, coin)
            elif "cost" in arrs:
                net_usd = gross_usd - _vec(arrs, "cost", alias, coin)
            else:
                raise ValueError(f"Missing 'net' or 'cost' array for ({alias}, {coin})")

            _write_aligned_coin_arrays(
                ti,
                si,
                slice(None),
                notional_usd,
                delta_notional_usd,
                gross_usd,
                net_usd,
                inv_equity=inv_eq,
                notional_frac=notional_frac,
                delta_notional_frac=delta_notional_frac,
                gross_ret_frac=gross_ret_frac,
                net_ret_frac=net_ret_frac,
                sym_net_ret_frac=sym_net_ret_frac,
            )

    has_full_px = (
        open_buf is not None
        and close_buf is not None
        and n_syms > 0
        and bool(np.all(sym_px_seen))
    )
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
        exec_open_px=open_buf if has_full_px else None,
        mark_close_px=close_buf if has_full_px else None,
    )


# ---------------------------------------------------------------------------
# Stage 1: Execution-Aligned Episode Reconstruction & Behavioral Gate
# ---------------------------------------------------------------------------


def _compute_slice_maes(
    ret_f64: np.ndarray,
    starts: np.ndarray,
    pnl_ends: np.ndarray,
) -> np.ndarray:
    """Compute per-episode maximum adverse excursion across variable-length slices."""
    n_runs = starts.size
    ep_mae = np.empty(n_runs, dtype=np.float64)
    for r_i in range(n_runs):
        r_slice = ret_f64[starts[r_i] : pnl_ends[r_i]]
        if r_slice.size > 0:
            cum_r = np.cumsum(r_slice)
            peaks = np.maximum(0.0, np.maximum.accumulate(cum_r))
            ep_mae[r_i] = float(np.max(peaks - cum_r))
        else:
            ep_mae[r_i] = 0.0
    return ep_mae


def _extract_symbol_episodes_vectorized(
    pos_s: np.ndarray,
    delta_s: np.ndarray,
    ret_s: np.ndarray,
    eps: float,
    *,
    carried_in: bool = False,
) -> (
    tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
    | None
):
    """Vectorized per-symbol run extraction splitting on flat gaps and sign reversals."""
    n_bars = pos_s.size
    if n_bars == 0:
        return None
    abs_s = np.abs(pos_s).astype(np.float64, copy=False)
    held_s = abs_s > eps
    if not np.any(held_s):
        return None

    signs = np.where(held_s, np.where(pos_s >= 0.0, 1, -1), 0)
    changes = np.flatnonzero(signs[1:] != signs[:-1]) + 1
    run_starts = np.concatenate((np.array([0], dtype=np.intp), changes))
    run_ends = np.concatenate((changes, np.array([n_bars], dtype=np.intp)))
    nz_mask = signs[run_starts] != 0
    starts = run_starts[nz_mask]
    ends = run_ends[nz_mask]
    n_runs = starts.size
    if n_runs == 0:
        return None

    clipped_ends = np.minimum(ends, n_bars - 1)
    include_close = (
        (ends < n_bars)
        & (signs[clipped_ends] == 0)
        & (np.abs(delta_s[clipped_ends]).astype(np.float64, copy=False) > 1e-7)
    )
    pnl_ends = np.where(include_close, ends + 1, ends)

    # Interleave [start_k, end_k] on 1-element padded buffers for exact reduceat without overflow.
    idx_hold = np.empty(2 * n_runs, dtype=np.intp)
    idx_hold[0::2] = starts
    idx_hold[1::2] = ends
    abs_pad = np.empty(n_bars + 1, dtype=np.float64)
    abs_pad[:-1] = abs_s
    abs_pad[-1] = 0.0
    ep_peak = np.maximum.reduceat(abs_pad, idx_hold)[0::2]
    ep_sym_tot = np.add.reduceat(abs_pad, idx_hold)[0::2]
    ep_entry = abs_s[starts].astype(np.float64, copy=True)

    ret_f64 = np.asarray(ret_s, dtype=np.float64)
    idx_pnl = np.empty(2 * n_runs, dtype=np.intp)
    idx_pnl[0::2] = starts
    idx_pnl[1::2] = pnl_ends
    ret_pad = np.empty(n_bars + 1, dtype=np.float64)
    ret_pad[:-1] = ret_f64
    ret_pad[-1] = 0.0
    ep_pnl = np.add.reduceat(ret_pad, idx_pnl)[0::2]

    ep_dur = (ends - starts).astype(np.float64)
    ep_mae = _compute_slice_maes(ret_f64, starts, pnl_ends)

    left_censored = np.zeros(n_runs, dtype=bool)
    if starts[0] == 0:
        pre0 = float(pos_s[0]) - float(delta_s[0])
        if carried_in or (abs(pre0) > eps and pre0 * float(pos_s[0]) > 0.0):
            left_censored[0] = True
    right_censored = ends >= n_bars

    return (
        starts.astype(np.int64),
        ends.astype(np.int64),
        pnl_ends.astype(np.int64),
        left_censored,
        right_censored,
        ep_pnl,
        ep_peak,
        ep_entry,
        ep_dur,
        ep_mae,
        ep_sym_tot,
    )


def _empty_episodes_dict(n_syms: int) -> dict[str, np.ndarray]:
    return {
        "ep_pnl": np.zeros(0, dtype=np.float64),
        "ep_peak_notional": np.zeros(0, dtype=np.float64),
        "ep_entry_notional": np.zeros(0, dtype=np.float64),
        "ep_duration_bars": np.zeros(0, dtype=np.float64),
        "ep_mae": np.zeros(0, dtype=np.float64),
        "ep_symbol_notional": np.zeros((0, n_syms), dtype=np.float64),
        "ep_start_bar": np.zeros(0, dtype=np.int64),
        "ep_end_bar": np.zeros(0, dtype=np.int64),
        "ep_pnl_end_bar": np.zeros(0, dtype=np.int64),
        "ep_left_censored": np.zeros(0, dtype=bool),
        "ep_right_censored": np.zeros(0, dtype=bool),
        "ep_uncensored": np.zeros(0, dtype=bool),
    }


def filter_uncensored_episodes(episodes: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return a copy/slice containing only uncensored closed episodes."""
    mask = episodes["ep_uncensored"]
    return {key: arr[mask] for key, arr in episodes.items()}


def _collect_symbol_episode_batches(
    batches: list[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ],
    n_syms: int,
) -> dict[str, np.ndarray]:
    if not batches:
        return _empty_episodes_dict(n_syms)
    all_starts = np.concatenate([b[1] for b in batches])
    all_ends = np.concatenate([b[2] for b in batches])
    all_pnl_ends = np.concatenate([b[3] for b in batches])
    all_left = np.concatenate([b[4] for b in batches])
    all_right = np.concatenate([b[5] for b in batches])
    all_pnl = np.concatenate([b[6] for b in batches])
    all_peak = np.concatenate([b[7] for b in batches])
    all_entry = np.concatenate([b[8] for b in batches])
    all_dur = np.concatenate([b[9] for b in batches])
    all_mae = np.concatenate([b[10] for b in batches])
    all_sym_tot = np.concatenate([b[11] for b in batches])
    all_sym_idx = np.concatenate(
        [np.full(b[1].size, b[0], dtype=np.intp) for b in batches]
    )
    order = np.argsort(all_ends, kind="stable")
    n_ep = order.size
    sym_mat = np.zeros((n_ep, n_syms), dtype=np.float64)
    sym_mat[np.arange(n_ep), all_sym_idx[order]] = all_sym_tot[order]
    left_ord = all_left[order]
    right_ord = all_right[order]
    return {
        "ep_pnl": all_pnl[order],
        "ep_peak_notional": all_peak[order],
        "ep_entry_notional": all_entry[order],
        "ep_duration_bars": all_dur[order],
        "ep_mae": all_mae[order],
        "ep_symbol_notional": sym_mat,
        "ep_start_bar": all_starts[order],
        "ep_end_bar": all_ends[order],
        "ep_pnl_end_bar": all_pnl_ends[order],
        "ep_left_censored": left_ord,
        "ep_right_censored": right_ord,
        "ep_uncensored": (~left_ord) & (~right_ord),
    }


def reconstruct_execution_episodes(
    tensor: UnitCopy5mTensor,
    eps: float = 1e-4,
    *,
    uncensored_only: bool = False,
) -> dict[int, dict[str, np.ndarray]]:
    """Reconstruct trade episodes aligned with Policy A 5m execution.

    When multi-coin traders keep staggered positions open continuously (never hitting
    total account flat), account-level grouping collapses 70+ days into 1 run.
    Grouping per ``(trader, symbol)`` on non-zero intervals AND sign reversals, while
    folding closing-bar exit PnL/cost into the ending episode, captures every real
    per-coin position lifecycle in chronological close order alongside censoring flags.
    """
    n_traders, n_syms, n_bars = tensor.notional_frac.shape
    episodes_by_trader: dict[int, dict[str, np.ndarray]] = {}

    for i in range(n_traders):
        if tensor.sym_net_ret_frac is not None:
            batches = []
            for s in range(n_syms):
                pos_s = tensor.notional_frac[i, s]
                if not pos_s.any():
                    continue
                carried_s = (
                    bool(tensor.carried_in_mask[i, s])
                    if tensor.carried_in_mask is not None
                    else False
                )
                extracted = _extract_symbol_episodes_vectorized(
                    pos_s,
                    tensor.delta_notional_frac[i, s],
                    tensor.sym_net_ret_frac[i, s],
                    eps,
                    carried_in=carried_s,
                )
                if extracted is not None:
                    batches.append((s, *extracted))
            ep_dict = _collect_symbol_episode_batches(batches, n_syms)
            episodes_by_trader[i] = (
                filter_uncensored_episodes(ep_dict) if uncensored_only else ep_dict
            )
            continue

        # Fallback when sym_net_ret_frac is absent: compute trader gross notional per active coin.
        gross_i = np.zeros(n_bars, dtype=np.float64)
        active_syms_i: list[int] = []
        for s in range(n_syms):
            pos_s = tensor.notional_frac[i, s]
            if pos_s.any():
                active_syms_i.append(s)
                gross_i += np.abs(pos_s).astype(np.float64, copy=False)

        if not active_syms_i:
            episodes_by_trader[i] = _empty_episodes_dict(n_syms)
            continue

        held_any = gross_i > eps
        padded_any = np.concatenate(([False], held_any, [False]))
        starts = np.flatnonzero(padded_any[1:] & ~padded_any[:-1])
        ends = np.flatnonzero(~padded_any[1:] & padded_any[:-1])
        n_acct_runs = starts.size

        if n_acct_runs >= 5 or n_syms == 1:
            if n_acct_runs == 0:
                episodes_by_trader[i] = _empty_episodes_dict(n_syms)
                continue
            abs_delta = np.zeros(n_bars, dtype=np.float64)
            has_pre_bar0 = False
            for s in active_syms_i:
                d_s = tensor.delta_notional_frac[i, s]
                if d_s.any():
                    abs_delta += np.abs(d_s).astype(np.float64, copy=False)
                if tensor.carried_in_mask is not None and bool(tensor.carried_in_mask[i, s]):
                    has_pre_bar0 = True
                p0 = float(tensor.notional_frac[i, s, 0])
                if abs(p0) > eps:
                    pre0 = p0 - float(d_s[0])
                    if abs(pre0) > eps and pre0 * p0 > 0.0:
                        has_pre_bar0 = True

            clipped_ends = np.minimum(ends, n_bars - 1)
            include_close = (ends < n_bars) & (abs_delta[clipped_ends] > 1e-7)
            pnl_ends = np.where(include_close, ends + 1, ends)

            idx_hold = np.empty(2 * n_acct_runs, dtype=np.intp)
            idx_hold[0::2] = starts
            idx_hold[1::2] = ends
            gross_pad = np.empty(n_bars + 1, dtype=np.float64)
            gross_pad[:-1] = gross_i
            gross_pad[-1] = 0.0
            ep_peak = np.maximum.reduceat(gross_pad, idx_hold)[0::2]
            ep_entry = gross_i[starts].astype(np.float64, copy=True)

            ret_i = np.asarray(tensor.net_ret_frac[i], dtype=np.float64)
            idx_pnl = np.empty(2 * n_acct_runs, dtype=np.intp)
            idx_pnl[0::2] = starts
            idx_pnl[1::2] = pnl_ends
            ret_pad = np.empty(n_bars + 1, dtype=np.float64)
            ret_pad[:-1] = ret_i
            ret_pad[-1] = 0.0
            ep_pnl = np.add.reduceat(ret_pad, idx_pnl)[0::2]
            ep_dur = (ends - starts).astype(np.float64)
            ep_mae = _compute_slice_maes(ret_i, starts, pnl_ends)

            sym_mat = np.zeros((n_acct_runs, n_syms), dtype=np.float64)
            abs_pad = np.empty(n_bars + 1, dtype=np.float64)
            abs_pad[-1] = 0.0
            for s in active_syms_i:
                abs_pad[:-1] = np.abs(tensor.notional_frac[i, s]).astype(np.float64, copy=False)
                sym_mat[:, s] = np.add.reduceat(abs_pad, idx_hold)[0::2]

            left_c = np.zeros(n_acct_runs, dtype=bool)
            if starts[0] == 0 and has_pre_bar0:
                left_c[0] = True
            right_c = ends >= n_bars
            ep_dict = {
                "ep_pnl": ep_pnl,
                "ep_peak_notional": ep_peak,
                "ep_entry_notional": ep_entry,
                "ep_duration_bars": ep_dur,
                "ep_mae": ep_mae,
                "ep_symbol_notional": sym_mat,
                "ep_start_bar": starts.astype(np.int64),
                "ep_end_bar": ends.astype(np.int64),
                "ep_pnl_end_bar": pnl_ends.astype(np.int64),
                "ep_left_censored": left_c,
                "ep_right_censored": right_c,
                "ep_uncensored": (~left_c) & (~right_c),
            }
            episodes_by_trader[i] = (
                filter_uncensored_episodes(ep_dict) if uncensored_only else ep_dict
            )
        else:
            share_base = np.asarray(tensor.net_ret_frac[i], dtype=np.float64) / np.maximum(
                gross_i, 1e-12
            )
            batches = []
            for s in active_syms_i:
                pos_s = tensor.notional_frac[i, s]
                ret_est = share_base * np.abs(pos_s).astype(np.float64, copy=False)
                carried_s = (
                    bool(tensor.carried_in_mask[i, s])
                    if tensor.carried_in_mask is not None
                    else False
                )
                extracted = _extract_symbol_episodes_vectorized(
                    pos_s,
                    tensor.delta_notional_frac[i, s],
                    ret_est,
                    eps,
                    carried_in=carried_s,
                )
                if extracted is not None:
                    batches.append((s, *extracted))
            ep_dict = _collect_symbol_episode_batches(batches, n_syms)
            episodes_by_trader[i] = (
                filter_uncensored_episodes(ep_dict) if uncensored_only else ep_dict
            )
    return episodes_by_trader


def _compute_trader_activity_q10(
    tensor: UnitCopy5mTensor,
    trader_idx: int,
    eps: float,
) -> tuple[np.ndarray, float]:
    """Return active bar indices and 10th-percentile non-zero symbol notional fraction."""
    n_syms = len(tensor.symbols)
    n_bars = tensor.timestamps_ms.size
    gross_i = np.zeros(n_bars, dtype=np.float64)
    nz_parts: list[np.ndarray] = []
    for s in range(n_syms):
        pos_s = tensor.notional_frac[trader_idx, s]
        if pos_s.any():
            abs_s = np.abs(pos_s).astype(np.float64, copy=False)
            gross_i += abs_s
            nz_s = abs_s[abs_s > eps]
            if nz_s.size > 0:
                nz_parts.append(nz_s)
    act_idx = np.flatnonzero(gross_i > eps)
    if act_idx.size >= 20 and nz_parts:
        q10 = float(np.quantile(np.concatenate(nz_parts), 0.10))
    else:
        q10 = 0.0
    return act_idx, q10


def _compute_causal_post_loss_tilt(
    ep_start: np.ndarray,
    ep_end: np.ndarray,
    ep_known: np.ndarray,
    ep_pnl: np.ndarray,
    ep_peak: np.ndarray,
    *,
    max_tilt_multiplier: float,
    min_comparisons: int = 1,
) -> tuple[float | None, Literal["OK", "FAIL", "UNKNOWN"]]:
    """Compare peak notional of entries opened strictly after a known closed loss vs win."""
    n_ep = ep_pnl.size
    if n_ep < 2:
        return None, "UNKNOWN"

    order_start = np.argsort(ep_start, kind="mergesort")
    order_known = np.lexsort((ep_end, ep_known))

    after_loss: list[float] = []
    after_win: list[float] = []
    ptr = 0
    last_known_pnl: float | None = None

    for s_idx in order_start:
        s_bar = int(ep_start[s_idx])
        while ptr < n_ep and int(ep_known[order_known[ptr]]) <= s_bar:
            k_idx = int(order_known[ptr])
            pnl_val = float(ep_pnl[k_idx])
            if abs(pnl_val) > 0.0:
                last_known_pnl = pnl_val
            ptr += 1
        if last_known_pnl is None:
            continue
        if last_known_pnl < 0.0:
            after_loss.append(float(ep_peak[s_idx]))
        elif last_known_pnl > 0.0:
            after_win.append(float(ep_peak[s_idx]))

    if len(after_loss) < min_comparisons or len(after_win) < min_comparisons:
        return None, "UNKNOWN"

    med_loss = float(np.median(after_loss))
    med_win = float(np.median(after_win))
    tilt = med_loss / max(med_win, 1e-8)
    status: Literal["OK", "FAIL"] = "FAIL" if tilt > max_tilt_multiplier else "OK"
    return tilt, status


def compute_eb_shrunk_ropn(
    episodes_map: Mapping[int, Mapping[str, np.ndarray]],
    n_traders: int,
    *,
    pod_equity_usd: float = 10_000.0,
    min_episode_notional_usd: float = 10.0,
    flat_eps_frac: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Empirical-Bayes normal-normal hierarchical shrunk ROPN per trader.

    Uses strictly uncensored episodes meeting the minimum peak-notional dust floor
    so dust-denominated noise never collapses shrinkage factors.
    """
    raw_means = np.zeros(n_traders, dtype=np.float64)
    eb_means = np.zeros(n_traders, dtype=np.float64)
    post_sds = np.zeros(n_traders, dtype=np.float64)
    min_ep_frac = max(float(flat_eps_frac), float(min_episode_notional_usd) / max(pod_equity_usd, 1.0))

    raw_vars = np.full(n_traders, np.nan, dtype=np.float64)
    counts = np.zeros(n_traders, dtype=np.int64)
    for i in range(n_traders):
        ep_all = episodes_map.get(i)
        if ep_all is None:
            continue
        ep = filter_uncensored_episodes(ep_all)
        peaks = ep["ep_peak_notional"]
        keep = (peaks * pod_equity_usd >= min_episode_notional_usd) & (peaks >= min_ep_frac)
        if not np.any(keep):
            continue
        ropn = ep["ep_pnl"][keep] / np.maximum(peaks[keep], min_ep_frac)
        counts[i] = int(ropn.size)
        raw_means[i] = float(np.mean(ropn))
        if ropn.size >= 2:
            raw_vars[i] = float(np.var(ropn, ddof=1))

    multi_idx = np.flatnonzero(counts >= 2)
    if multi_idx.size == 0:
        return raw_means, raw_means.copy(), post_sds

    med_var = float(np.median(raw_vars[multi_idx]))
    within_v = np.maximum(raw_vars[multi_idx], 0.25 * med_var + 1e-8) / counts[multi_idx]
    if multi_idx.size == 1:
        idx0 = int(multi_idx[0])
        eb_means[idx0] = raw_means[idx0]
        post_sds[idx0] = math.sqrt(float(within_v[0]))
        return raw_means, eb_means, post_sds

    w_fe = 1.0 / np.maximum(within_v, 1e-12)
    w_sum = float(np.sum(w_fe))
    mu0 = float(np.sum(w_fe * raw_means[multi_idx]) / w_sum)
    q_stat = float(np.sum(w_fe * ((raw_means[multi_idx] - mu0) ** 2)))
    c_denom = w_sum - float(np.sum(w_fe * w_fe) / w_sum)
    tau2 = max(0.0, (q_stat - float(multi_idx.size - 1)) / max(c_denom, 1e-12))

    var_floor = max(med_var, 1e-8)
    for i in range(n_traders):
        n_i = int(counts[i])
        if n_i <= 0:
            eb_means[i] = mu0
            post_sds[i] = math.sqrt(tau2)
            continue
        var_i = float(raw_vars[i]) if n_i >= 2 and np.isfinite(raw_vars[i]) else var_floor
        v_i = max(var_i, 0.25 * med_var + 1e-8) / float(n_i)
        denom = tau2 + v_i
        b_i = (tau2 / denom) if denom > 0.0 else 0.0
        eb_means[i] = (1.0 - b_i) * mu0 + b_i * raw_means[i]
        post_sds[i] = math.sqrt((tau2 * v_i) / denom) if denom > 0.0 else math.sqrt(v_i)
    return raw_means, eb_means, post_sds


def classify_holding_style_sleeve(
    uncensored_durations_bars: np.ndarray,
    *,
    active_frac: float,
    min_episodes: int = 4,
    min_active_frac: float = 0.02,
) -> StyleSleeve:
    """Classify deterministic holding-time style sleeve over uncensored completed episodes."""
    d = np.asarray(uncensored_durations_bars, dtype=np.float64)
    if active_frac < min_active_frac or d.size < min_episodes:
        return "unlabeled"
    s_intra = float(np.mean(d <= 288.0))
    s_swing = float(np.mean((d > 288.0) & (d <= 2016.0)))
    s_pos = float(np.mean(d > 2016.0))
    if max(s_intra, s_swing, s_pos) < 0.50:
        return "mixed"
    med = float(np.median(d))
    if med <= 288.0:
        return "intraday_tactical"
    if med <= 2016.0:
        return "multi_day_swing"
    return "positional"


def _check_right_censored_revenge_tilt(
    ep_all: Mapping[str, np.ndarray],
    unc_ep: Mapping[str, np.ndarray],
    *,
    max_tilt_multiplier: float,
    existing_tilt: float | None,
    existing_status: Literal["OK", "FAIL", "UNKNOWN"],
) -> tuple[float | None, Literal["OK", "FAIL", "UNKNOWN"]]:
    """Detect oversized right-censored open positions launched strictly after a closed loss."""
    rc_mask = ep_all["ep_right_censored"] & (~ep_all["ep_left_censored"]) & (ep_all["ep_start_bar"] > 0)
    if not np.any(rc_mask):
        return existing_tilt, existing_status

    u_pnl = unc_ep["ep_pnl"]
    u_known = unc_ep["ep_pnl_end_bar"]
    u_peak = unc_ep["ep_peak_notional"]
    if u_pnl.size == 0:
        return existing_tilt, existing_status

    order_rc = np.flatnonzero(rc_mask)
    best_tilt = existing_tilt
    status = existing_status
    for idx in order_rc:
        s_bar = int(ep_all["ep_start_bar"][idx])
        prior_mask = (u_known <= s_bar) & (np.abs(u_pnl) > 0.0)
        if not np.any(prior_mask):
            continue
        prior_idx = np.flatnonzero(prior_mask)
        latest_k = int(prior_idx[int(np.argmax(u_known[prior_idx]))])
        if float(u_pnl[latest_k]) >= 0.0:
            continue
        win_prior = prior_idx[u_pnl[prior_idx] > 0.0]
        if win_prior.size > 0:
            ref_size = float(np.median(u_peak[win_prior]))
        else:
            all_wins = u_peak[u_pnl > 0.0]
            ref_size = float(np.median(all_wins)) if all_wins.size > 0 else float(np.median(u_peak[prior_idx]))
        rc_size = max(float(ep_all["ep_entry_notional"][idx]), float(ep_all["ep_peak_notional"][idx]))
        ratio = rc_size / max(ref_size, 1e-8)
        if ratio > max_tilt_multiplier:
            best_tilt = ratio if best_tilt is None else max(best_tilt, ratio)
            status = "FAIL"
    return best_tilt, status


def _hidden_bagholder_reason(
    right_censored_pnl: float,
    tot_ret: float,
    closed_gains: float,
    config: PodConstructorConfig,
) -> str | None:
    """Return rejection reason when right-censored open losses violate bagholder guard."""
    if not config.reject_hidden_bagholder or right_censored_pnl >= -1e-9:
        return None
    if (tot_ret <= 0.0) or (
        abs(right_censored_pnl) > config.max_bagholder_open_loss_ratio * max(closed_gains, 1e-8)
    ):
        return (
            f"right_censored_bagholder(open_pnl={right_censored_pnl:.4f},"
            f"closed_gains={closed_gains:.4f})"
        )
    return None


def stage1_audit_traders(
    tensor: UnitCopy5mTensor,
    config: PodConstructorConfig = PodConstructorConfig(),
) -> tuple[list[int], list[TraderAuditMetrics], dict[int, TraderEpisodeBaseline]]:
    """Enrich candidates with uncensored episode underwriting, causal tilt, and baselines.

    Per ML4T-METHODOLOGY-LESSONS.md, DSR is recorded strictly as a diagnostic and never
    used as a blanket rejection cutoff on discretionary wallets.
    """
    n_traders, n_syms, n_bars = tensor.notional_frac.shape
    episodes_map = reconstruct_execution_episodes(tensor)
    bars_per_year = float(config.bars_per_day * config.days_per_year)
    raw_ropn_vec, eb_ropn_vec, eb_ropn_sd_vec = compute_eb_shrunk_ropn(
        episodes_map,
        n_traders,
        pod_equity_usd=tensor.pod_equity_usd,
        min_episode_notional_usd=config.min_episode_notional_usd,
        flat_eps_frac=config.flat_notional_eps_frac,
    )

    active_bars_vec = np.zeros(n_traders, dtype=np.int64)
    raw_sr_bar = np.zeros(n_traders, dtype=np.float64)
    raw_sr_ann = np.zeros(n_traders, dtype=np.float64)
    sr_var_ann = np.zeros(n_traders, dtype=np.float64)
    skew_vec = np.zeros(n_traders, dtype=np.float64)
    kurt_vec = np.zeros(n_traders, dtype=np.float64)
    tilt_vals: list[float | None] = [None] * n_traders
    tilt_statuses: list[Literal["OK", "FAIL", "UNKNOWN"]] = ["UNKNOWN"] * n_traders
    uncensored_counts = np.zeros(n_traders, dtype=np.int64)
    censored_counts = np.zeros(n_traders, dtype=np.int64)
    left_censored_counts = np.zeros(n_traders, dtype=np.int64)
    right_censored_counts = np.zeros(n_traders, dtype=np.int64)
    right_censored_pnls = np.zeros(n_traders, dtype=np.float64)
    closed_gains_vec = np.zeros(n_traders, dtype=np.float64)
    tot_ret_vec = np.zeros(n_traders, dtype=np.float64)
    open_exposures = np.zeros(n_traders, dtype=np.float64)
    style_sleeves: list[StyleSleeve] = ["unlabeled"] * n_traders
    win_hold_p95_list: list[float | None] = [None] * n_traders
    scaling_med_vec = np.ones(n_traders, dtype=np.float64)
    cdar95_vec = np.zeros(n_traders, dtype=np.float64)
    peak_dd_vec = np.zeros(n_traders, dtype=np.float64)
    static_dd_vec = np.zeros(n_traders, dtype=np.float64)
    net_dd_vec = np.zeros(n_traders, dtype=np.float64)
    chop_net_dd_vec = np.zeros(n_traders, dtype=np.float64)
    q10_vec = np.zeros(n_traders, dtype=np.float64)
    baselines: dict[int, TraderEpisodeBaseline] = {}

    for i in range(n_traders):
        act_idx, q10 = _compute_trader_activity_q10(tensor, i, config.flat_notional_eps_frac)
        t_act = act_idx.size
        active_bars_vec[i] = t_act
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
        else:
            sr_b, sr_a, var_a, g3, g4 = 0.0, 0.0, 1.0, 0.0, 3.0

        raw_sr_bar[i] = sr_b
        raw_sr_ann[i] = sr_a
        sr_var_ann[i] = var_a
        skew_vec[i] = g3
        kurt_vec[i] = g4
        q10_vec[i] = q10

        cum_r = np.cumsum(tensor.net_ret_frac[i])
        _, peak_dd, static_dd, cdar95 = compute_drawdown_metrics(cum_r, alpha=config.cdar_alpha)
        peak_dd_vec[i] = max(peak_dd, 1e-6)
        static_dd_vec[i] = static_dd
        cdar95_vec[i] = max(cdar95, 1e-6)
        tot_ret = float(cum_r[-1]) if cum_r.size > 0 else 0.0
        tot_ret_vec[i] = tot_ret
        net_dd_vec[i] = tot_ret / max(peak_dd, 1e-6)

        if tensor.chop_mask is not None and np.any(tensor.chop_mask):
            chop_r = tensor.net_ret_frac[i, tensor.chop_mask]
            chop_cum = np.cumsum(chop_r)
            _, chop_mdd, _ = compute_underwater_and_cdar(chop_cum, alpha=config.cdar_alpha)
            chop_net = float(chop_cum[-1]) if chop_cum.size > 0 else 0.0
            chop_net_dd_vec[i] = chop_net / max(chop_mdd, 1e-6)
        else:
            chop_net_dd_vec[i] = net_dd_vec[i]

        ep_all = episodes_map[i]
        left_c_cnt = int(np.sum(ep_all["ep_left_censored"]))
        right_c_cnt = int(np.sum(ep_all["ep_right_censored"]))
        censored_cnt = int(np.sum(~ep_all["ep_uncensored"]))
        left_censored_counts[i] = left_c_cnt
        right_censored_counts[i] = right_c_cnt
        censored_counts[i] = censored_cnt
        right_censored_pnls[i] = float(np.sum(ep_all["ep_pnl"][ep_all["ep_right_censored"]]))
        if n_bars > 0:
            open_exposures[i] = float(
                sum(abs(float(tensor.notional_frac[i, s, -1])) for s in range(n_syms))
            )

        ep = filter_uncensored_episodes(ep_all)
        ep_pnl = ep["ep_pnl"]
        ep_peak = ep["ep_peak_notional"]
        ep_entry = ep["ep_entry_notional"]
        ep_dur = ep["ep_duration_bars"]
        ep_mae = ep["ep_mae"]
        ep_sym = ep["ep_symbol_notional"]
        n_ep = ep_pnl.size
        uncensored_counts[i] = n_ep
        closed_gains_vec[i] = float(np.sum(np.maximum(0.0, ep_pnl))) if n_ep > 0 else 0.0

        style_sleeves[i] = classify_holding_style_sleeve(
            ep_dur,
            active_frac=act_frac,
            min_episodes=max(4, min(config.min_episodes, 4)),
        )
        if n_ep > 0:
            scaling_med_vec[i] = float(
                np.median(ep_peak / np.maximum(ep_entry, config.flat_notional_eps_frac))
            )
        win_durs = ep_dur[ep_pnl > 0.0]
        if win_durs.size >= 20:
            win_hold_p95_list[i] = float(np.quantile(win_durs, 0.95))

        tilt_val, tilt_stat = _compute_causal_post_loss_tilt(
            ep["ep_start_bar"],
            ep["ep_end_bar"],
            ep["ep_pnl_end_bar"],
            ep_pnl,
            ep_peak,
            max_tilt_multiplier=config.max_tilt_multiplier,
            min_comparisons=config.min_tilt_comparisons,
        )
        tilt_val, tilt_stat = _check_right_censored_revenge_tilt(
            ep_all,
            ep,
            max_tilt_multiplier=config.max_tilt_multiplier,
            existing_tilt=tilt_val,
            existing_status=tilt_stat,
        )
        tilt_vals[i] = tilt_val
        tilt_statuses[i] = tilt_stat

        if n_ep >= 2:
            ln_peak = np.log(np.maximum(ep_peak, 1e-8))
            med_ln = float(np.median(ln_peak))
            mad_ln = float(np.median(np.abs(ln_peak - med_ln))) + 1e-4
            win_mask = ep_pnl > 0.0
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
                qualified=True,
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
                qualified=False,
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
        t_act = int(active_bars_vec[i])
        act_frac = t_act / float(max(1, n_bars))
        n_ep = int(uncensored_counts[i])
        se_ann = math.sqrt(max(1e-8, sr_var_ann[i]))
        dsr_prob = norm_cdf((raw_sr_ann[i] - sr0_ann) / se_ann)

        reasons: list[str] = []
        if act_frac < config.min_active_bar_frac:
            reasons.append(f"low_activity({act_frac:.3f}<{config.min_active_bar_frac})")
        if n_ep < config.min_episodes:
            reasons.append(f"few_episodes({n_ep}<{config.min_episodes})")
        t_val = tilt_vals[i]
        t_stat = tilt_statuses[i]
        if t_stat == "FAIL" and t_val is not None:
            reasons.append(f"martingale_tilt({t_val:.2f}>{config.max_tilt_multiplier})")
        elif t_stat == "UNKNOWN" and config.reject_unknown_tilt:
            reasons.append("tilt_unknown")
        if net_dd_vec[i] <= 0.0 and chop_net_dd_vec[i] <= 0.0:
            reasons.append("non_positive_net_return")
        bag_reason = _hidden_bagholder_reason(
            float(right_censored_pnls[i]), float(tot_ret_vec[i]), float(closed_gains_vec[i]), config
        )
        if bag_reason is not None:
            reasons.append(bag_reason)
        if config.min_baseline_order_usd > 0.0:
            if config.baseline_port_dd_frac <= 0.0:
                raise ValueError("baseline_port_dd_frac must be positive when min_baseline_order_usd > 0")
            base_k = (config.baseline_port_dd_frac / max(1, config.pod_size)) / max(
                float(peak_dd_vec[i]), 1e-6
            )
            eff_base_k = min(base_k, config.max_k_mult)
            q10_order_usd = eff_base_k * float(q10_vec[i]) * tensor.pod_equity_usd
            if q10_order_usd < config.min_baseline_order_usd:
                reasons.append(
                    f"below_min_baseline_order(${q10_order_usd:.2f}<${config.min_baseline_order_usd:.0f})"
                )

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
                standalone_peak_to_trough_dd=float(peak_dd_vec[i]),
                standalone_static_floor_dd=float(static_dd_vec[i]),
                tilt_multiplier=t_val,
                tilt_status=t_stat,
                active_pos_q10_frac=float(q10_vec[i]),
                passed_behavioral_gate=passed,
                reject_reason=", ".join(reasons),
                censored_episodes=int(censored_counts[i]),
                left_censored_episodes=int(left_censored_counts[i]),
                right_censored_episodes=int(right_censored_counts[i]),
                open_exposure_at_cutoff=float(open_exposures[i]),
                raw_ropn_mean=float(raw_ropn_vec[i]),
                eb_ropn_mean=float(eb_ropn_vec[i]),
                eb_ropn_post_sd=float(eb_ropn_sd_vec[i]),
                right_censored_pnl_frac=float(right_censored_pnls[i]),
                style_sleeve=style_sleeves[i],
                win_hold_p95_bars=win_hold_p95_list[i],
                within_ep_scaling_median=float(scaling_med_vec[i]),
            )
        )

    return passed_ids, metrics, baselines


# ---------------------------------------------------------------------------
# Stage 1.5a: Sparse-Aware Crowding Distance Tensor
# ---------------------------------------------------------------------------


def _pairwise_union_cooccurrence(
    events: np.ndarray,
    active: np.ndarray,
    *,
    min_union: int = 1,
) -> np.ndarray:
    """Compute symmetric pairwise P(E_a | E_b) / 2 + P(E_b | E_a) / 2 over A | B."""
    c_n = events.shape[0]
    e_f = events.astype(np.float64, copy=False)
    m_f = active.astype(np.float64, copy=False)
    em = e_f * m_f
    e_inact = e_f * (1.0 - m_f)

    cnt_m = np.sum(m_f, axis=1)
    n_u = cnt_m[:, None] + cnt_m[None, :] - (m_f @ m_f.T)
    cnt_eb_on_u = (m_f @ e_f.T) + np.sum(em, axis=1)[None, :] - (m_f @ em.T)
    cnt_ea_eb_on_u = (e_f @ e_f.T) - (e_inact @ e_inact.T)

    valid = (n_u >= min_union) & (cnt_eb_on_u > 0.5)
    p_ab = np.divide(cnt_ea_eb_on_u, cnt_eb_on_u, out=np.zeros((c_n, c_n)), where=valid)
    return 0.5 * (p_ab + p_ab.T)


def _pairwise_union_corr(
    values: np.ndarray,
    active: np.ndarray,
    *,
    min_union: int = 5,
) -> np.ndarray:
    """Compute pairwise Pearson correlation restricted to active union A | B."""
    c_n = values.shape[0]
    x = np.asarray(values, dtype=np.float64)
    m_f = active.astype(np.float64, copy=False)
    xm = x * m_f
    x2 = x * x
    x2m = x2 * m_f
    x_inact = x * (1.0 - m_f)

    cnt_m = np.sum(m_f, axis=1)
    n_u = cnt_m[:, None] + cnt_m[None, :] - (m_f @ m_f.T)
    sum_a = np.sum(xm, axis=1)[:, None] + (x @ m_f.T) - (xm @ m_f.T)
    sum_b = sum_a.T
    ss_a = np.sum(x2m, axis=1)[:, None] + (x2 @ m_f.T) - (x2m @ m_f.T)
    ss_b = ss_a.T
    prod_ab = (x @ x.T) - (x_inact @ x_inact.T)

    u_valid = n_u >= min_union
    inv_n = np.where(u_valid, 1.0 / np.maximum(n_u, 1.0), 0.0)
    mean_a = sum_a * inv_n
    mean_b = sum_b * inv_n
    var_a = np.maximum(0.0, ss_a * inv_n - mean_a * mean_a)
    var_b = np.maximum(0.0, ss_b * inv_n - mean_b * mean_b)
    std_a = np.sqrt(var_a)
    std_b = np.sqrt(var_b)

    corr_valid = u_valid & (std_a > 1e-12) & (std_b > 1e-12)
    cov_ab = prod_ab * inv_n - mean_a * mean_b
    corr = np.divide(cov_ab, std_a * std_b, out=np.zeros((c_n, c_n)), where=corr_valid)
    corr = np.clip(corr, -1.0, 1.0)

    # Disambiguate rare nearly-constant rows where float64 sum-of-squares cancellation matters.
    tol_a = var_a <= 1e-12 * np.maximum(1.0, ss_a * inv_n)
    tol_b = var_b <= 1e-12 * np.maximum(1.0, ss_b * inv_n)
    suspect = u_valid & (tol_a | tol_b)
    for a in range(c_n):
        for b in range(a + 1, c_n):
            if suspect[a, b]:
                u_d = active[a] | active[b]
                da_r, db_r = x[a, u_d], x[b, u_d]
                val = (
                    float(np.corrcoef(da_r, db_r)[0, 1])
                    if np.std(da_r) > 1e-12 and np.std(db_r) > 1e-12
                    else 0.0
                )
                corr[a, b] = corr[b, a] = val
    return corr


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
    n_syms = len(tensor.symbols)
    sub_ret = np.asarray(tensor.net_ret_frac[cand], dtype=np.float64)  # (C, T)
    n_bars = sub_ret.shape[1]

    active_mask = np.zeros((n_cand, n_bars), dtype=bool)
    pos_norm = np.zeros((n_cand, n_bars), dtype=np.float64)
    active_syms_by_cand: list[tuple[int, ...]] = []
    for idx, tid in enumerate(cand):
        gross_i = np.zeros(n_bars, dtype=np.float64)
        norm_sq_i = np.zeros(n_bars, dtype=np.float64)
        nz_syms: list[int] = []
        for s in range(n_syms):
            pos_s = tensor.notional_frac[tid, s]
            if pos_s.any():
                nz_syms.append(s)
                p_f64 = pos_s.astype(np.float64, copy=False)
                gross_i += np.abs(p_f64)
                norm_sq_i += p_f64 * p_f64
        active_syms_by_cand.append(tuple(nz_syms))
        if nz_syms:
            active_mask[idx] = gross_i > config.flat_notional_eps_frac
            pos_norm[idx] = np.sqrt(norm_sq_i)

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

    # A: Vectorized Active-Union lower-tail co-distress + daily prior + overlap counts
    emp_lambda = _pairwise_union_cooccurrence(win_ret <= q05[:, None], win_act, min_union=1)
    prior_lambda = _pairwise_union_cooccurrence(day_ret < 0.0, day_act, min_union=5)
    act_f = active_mask.astype(np.float64)
    n_inter_mat = act_f @ act_f.T
    w_shr = n_inter_mat / (n_inter_mat + config.overlap_shrinkage_kappa)
    lambda_mat = np.clip(w_shr * emp_lambda + (1.0 - w_shr) * prior_lambda, 0.0, 1.0)
    np.fill_diagonal(lambda_mat, 1.0)

    # B: Vectorized Underwater Co-Drawdown
    dd_norms = np.linalg.norm(dd_excess, axis=1)
    valid_dd = dd_norms > 1e-12
    dd_unit = np.zeros_like(dd_excess)
    dd_unit[valid_dd] = dd_excess[valid_dd] / dd_norms[valid_dd, None]
    omega_mat = np.clip(dd_unit @ dd_unit.T, 0.0, 1.0)
    np.fill_diagonal(omega_mat, 1.0)

    # D: Vectorized active-union daily correlation regime penalty
    corr_day = _pairwise_union_corr(day_ret, day_act, min_union=5)
    regime_mat = np.clip(0.5 * (1.0 + corr_day), 0.0, 1.0)
    np.fill_diagonal(regime_mat, 1.0)

    # C: Signed Notional Netting Cosine on active intersection (only common active coins)
    psi_mat = np.eye(n_cand, dtype=np.float64)
    active_sets = [set(syms) for syms in active_syms_by_cand]
    for a in range(n_cand):
        tid_a = cand[a]
        set_a = active_sets[a]
        if not set_a:
            continue
        for b in range(a + 1, n_cand):
            if n_inter_mat[a, b] <= 0.5:
                continue
            common_syms = sorted(set_a & active_sets[b])
            if not common_syms:
                continue
            tid_b = cand[b]
            inter_bars = active_mask[a] & active_mask[b]
            dots = np.zeros(int(np.sum(inter_bars)), dtype=np.float64)
            for s in common_syms:
                dots += (
                    tensor.notional_frac[tid_a, s, inter_bars].astype(np.float64, copy=False)
                    * tensor.notional_frac[tid_b, s, inter_bars].astype(np.float64, copy=False)
                )
            norms = pos_norm[a, inter_bars] * pos_norm[b, inter_bars] + 1e-12
            psi_pos = float(np.clip(np.mean(dots / norms), -1.0, 1.0))
            psi_mat[a, b] = psi_mat[b, a] = psi_pos

    w1, w2, w3, w4 = config.crowding_weights
    crowding = (
        w1 * lambda_mat
        + w2 * omega_mat
        + w3 * (0.5 * (1.0 + psi_mat))
        + w4 * regime_mat
    )
    dist_mat = np.clip(1.0 - crowding, 0.05, 1.0)
    np.fill_diagonal(dist_mat, 0.0)

    day_act_i = day_act.astype(np.int32)
    concurrent_days = day_act_i @ day_act_i.T
    pair_overlap_unknown = (concurrent_days < int(config.min_overlap_days)) & (
        ~np.eye(n_cand, dtype=bool)
    )

    return dist_mat, {
        "lambda_tail": lambda_mat,
        "omega_dd": omega_mat,
        "psi_pos": psi_mat,
        "regime_corr": regime_mat,
        "concurrent_active_days": concurrent_days,
        "pair_overlap_unknown": pair_overlap_unknown,
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


def _contiguous_index_segments(bar_idx: np.ndarray) -> list[tuple[int, int]]:
    """Return relative contiguous ``[s, e)`` ranges inside a time-sliced tensor."""
    n = int(bar_idx.size)
    if n == 0:
        return []
    breaks = np.flatnonzero(np.diff(bar_idx) != 1) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), breaks))
    ends = np.concatenate((breaks, np.array([n], dtype=np.int64)))
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e > s]


def _daily_reset_slices(
    timestamps_ms: np.ndarray,
    bars_per_day: int,
    reset_bar_offset: int,
    *,
    segments: Sequence[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Compute exact daily slices anchored to 00:05 UTC without spanning purged block gaps."""
    n_bars = timestamps_ms.size
    if n_bars == 0:
        return []
    segs = list(segments) if segments is not None else [(0, n_bars)]
    slices: list[tuple[int, int]] = []
    for seg_s, seg_e in segs:
        if seg_e <= seg_s:
            continue
        seg_ts = timestamps_ms[seg_s:seg_e]
        day_starts = np.where(((seg_ts // 300_000) % bars_per_day) == reset_bar_offset)[0]
        if day_starts.size == 0 or day_starts[0] > 0:
            starts = [seg_s] + [seg_s + int(x) for x in day_starts.tolist()]
        else:
            starts = [seg_s + int(x) for x in day_starts.tolist()]
        for idx, sb in enumerate(starts):
            eb = starts[idx + 1] if idx + 1 < len(starts) else seg_e
            if eb > sb:
                slices.append((int(sb), int(eb)))
    return slices


def _apply_fill_to_basis_scalar(
    cur_pos: float,
    new_pos: float,
    cur_entry: float,
    px: float,
    *,
    eps: float = 1e-12,
) -> float:
    """Update weighted-average entry price across a netted position change."""
    delta = new_pos - cur_pos
    if abs(delta) <= eps:
        return 0.0 if abs(new_pos) <= eps else cur_entry
    if abs(new_pos) <= eps:
        return 0.0
    if abs(cur_pos) <= eps or cur_pos * new_pos < 0.0:
        return px
    if cur_pos * delta > 0.0:
        return (abs(cur_pos) * cur_entry + abs(delta) * px) / abs(new_pos)
    return cur_entry


def _build_reset_seg_context(
    n_bars: int,
    day_slices: Sequence[tuple[int, int]],
    segments: Sequence[tuple[int, int]] | None,
) -> tuple[list[tuple[int, int]], np.ndarray, set[int]]:
    """Build daily reset targets, per-bar segment IDs, and segment start offsets."""
    if segments is not None and len(segments) > 1:
        bar_seg_id = np.full(n_bars, -1, dtype=np.int32)
        seg_starts: set[int] = set()
        for seg_idx, (s, e) in enumerate(segments):
            s_i, e_i = int(s), int(e)
            if e_i > s_i:
                bar_seg_id[s_i:e_i] = seg_idx
                seg_starts.add(s_i)
    else:
        bar_seg_id = np.zeros(n_bars, dtype=np.int32)
        seg_starts = {int(s) for s, e in segments if e > s} if segments is not None else {0}
    reset_targets = [
        (di, int(sb - 1))
        for di, (sb, _eb) in enumerate(day_slices)
        if sb > 0 and sb not in seg_starts
    ]
    return reset_targets, bar_seg_id, seg_starts


def _compute_eff_open_and_q_members(
    notional_mat: np.ndarray,
    open_s: np.ndarray,
    close_s: np.ndarray,
    pod_eq: float,
    seg_starts: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute effective execution prices and member quantities with gap forward-fill."""
    eff_open = np.where(
        np.isfinite(open_s) & (open_s > 0.0),
        open_s,
        np.where(np.isfinite(close_s) & (close_s > 0.0), close_s, np.nan),
    )
    valid_px = np.isfinite(eff_open) & (eff_open > 0.0)
    inv_open = np.where(valid_px, pod_eq / np.maximum(eff_open, 1e-12), 0.0)
    q_mat = notional_mat * inv_open[None, :]
    missing_bars = np.flatnonzero(~valid_px)
    if missing_bars.size > 0:
        nz_mask = np.abs(notional_mat) > 1e-12
        for b in missing_bars:
            b_i = int(b)
            if b_i == 0 or b_i in seg_starts:
                q_mat[:, b_i] = 0.0
            else:
                q_mat[:, b_i] = np.where(nz_mask[:, b_i], q_mat[:, b_i - 1], 0.0)
    return eff_open, q_mat


def _accumulate_symbol_unrl_at_resets(
    unrl: np.ndarray,
    nz_bars: np.ndarray,
    q_nz: np.ndarray,
    px_nz: np.ndarray,
    nz_seg_id: np.ndarray,
    close_s: np.ndarray,
    bar_seg_id: np.ndarray,
    reset_targets: Sequence[tuple[int, int]],
    pod_equity_usd: float,
    *,
    _reset_arrays: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """Trace netted cost basis across segment-isolated bars and add reset unrealized PnL."""
    if nz_bars.size == 0 or not reset_targets:
        return
    if _reset_arrays is None:
        di_arr = np.fromiter(
            (di for di, _ in reset_targets), dtype=np.int64, count=len(reset_targets)
        )
        r_arr = np.fromiter(
            (r for _, r in reset_targets), dtype=np.int64, count=len(reset_targets)
        )
    else:
        di_arr, r_arr = _reset_arrays

    r_seg = bar_seg_id[r_arr]
    k_arr = np.searchsorted(nz_bars, r_arr, side="right") - 1
    valid = (r_seg >= 0) & (k_arr >= 0)
    if not np.any(valid):
        return
    di_v = di_arr[valid]
    r_v = r_arr[valid]
    k_v = k_arr[valid]
    r_seg_v = r_seg[valid]
    q_v = q_nz[k_v]
    cl_v = close_s[r_v]
    active = (nz_seg_id[k_v] == r_seg_v) & (np.abs(q_v) > 1e-12) & np.isfinite(cl_v)
    if not np.any(active):
        return
    di_a = di_v[active]
    k_a = k_v[active]
    q_a = q_v[active]
    cl_a = cl_v[active]

    max_k = int(k_a.max()) + 1
    q_sub = q_nz[:max_k]
    px_sub = px_nz[:max_k]
    seg_sub = nz_seg_id[:max_k]
    seg_start = np.empty(max_k, dtype=bool)
    seg_start[0] = True
    if max_k > 1:
        seg_start[1:] = seg_sub[1:] != seg_sub[:-1]
    prev_q = np.empty(max_k, dtype=np.float64)
    prev_q[0] = 0.0
    if max_k > 1:
        prev_q[1:] = np.where(seg_start[1:], 0.0, q_sub[:-1])
    delta = q_sub - prev_q
    abs_q = np.abs(q_sub)
    abs_prev = np.abs(prev_q)
    abs_delta = np.abs(delta)
    is_flat = abs_q <= 1e-12
    changed = abs_delta > 1e-12
    is_hard_open = changed & (~is_flat) & ((abs_prev <= 1e-12) | (prev_q * q_sub < 0.0))
    is_reset = seg_start | is_flat | is_hard_open
    last_reset = np.maximum.accumulate(np.where(is_reset, np.arange(max_k, dtype=np.int64), 0))
    is_inc = changed & (~is_flat) & (~is_hard_open) & (prev_q * delta > 0.0)

    ep_starts = last_reset[k_a]
    uniq_starts = np.unique(ep_starts)
    entry_a = np.empty(k_a.size, dtype=np.float64)
    for st in uniq_starts:
        st_i = int(st)
        m_ep = ep_starts == st_i
        ks_ep = k_a[m_ep]
        end_i = int(ks_ep.max())
        init_e = 0.0 if is_flat[st_i] else (float(px_sub[st_i]) if is_hard_open[st_i] else 0.0)
        if end_i == st_i:
            entry_a[m_ep] = init_e
            continue
        inc_rel = np.flatnonzero(is_inc[st_i + 1 : end_i + 1])
        if inc_rel.size == 0:
            entry_a[m_ep] = init_e
            continue
        inc_idx = inc_rel + (st_i + 1)
        alpha = abs_prev[inc_idx] / abs_q[inc_idx]
        beta = (abs_delta[inc_idx] / abs_q[inc_idx]) * px_sub[inc_idx]
        log_p = np.cumsum(np.log(alpha))
        if float(log_p[-1]) >= -500.0:
            p_arr = np.exp(log_p)
            e_inc = p_arr * (init_e + np.cumsum(beta / p_arr))
        else:
            ce = init_e
            e_inc = np.empty(inc_idx.size, dtype=np.float64)
            for i_inc in range(inc_idx.size):
                ce = float(alpha[i_inc]) * ce + float(beta[i_inc])
                e_inc[i_inc] = ce
        pos_in_inc = np.searchsorted(inc_idx, ks_ep, side="right") - 1
        entry_a[m_ep] = np.where(pos_in_inc >= 0, e_inc[np.maximum(pos_in_inc, 0)], init_e)

    np.add.at(unrl, di_a, q_a * (cl_a - entry_a) / pod_equity_usd)


def _compute_netted_unrealized_frac_at_resets(
    weights: np.ndarray,
    tensor: UnitCopy5mTensor,
    pod_indices: Sequence[int],
    day_slices: Sequence[tuple[int, int]],
    cost_rate: float,
    *,
    active_syms: Sequence[int] | None = None,
    segments: Sequence[tuple[int, int]] | None = None,
    eps: float = 1e-9,
) -> np.ndarray:
    """Compute exact account-netted open unrealized PnL fraction entering each daily reset."""
    n_days = len(day_slices)
    unrl = np.zeros(n_days, dtype=np.float64)
    n_bars = tensor.timestamps_ms.size
    if n_days == 0 or n_bars == 0:
        return unrl

    reset_targets, bar_seg_id, seg_starts = _build_reset_seg_context(
        n_bars, day_slices, segments
    )
    if not reset_targets:
        return unrl

    pod_list = list(pod_indices)
    if active_syms is None:
        syms = [
            s
            for s in range(len(tensor.symbols))
            if tensor.notional_frac[pod_list, s, :].any()
            or tensor.delta_notional_frac[pod_list, s, :].any()
        ]
    else:
        syms = list(active_syms)
    if not syms:
        return unrl

    # Case 1: Exact price arrays available -> replicate Guardian's netted quantity & cost basis
    if tensor.exec_open_px is not None and tensor.mark_close_px is not None:
        pod_eq = tensor.pod_equity_usd
        seg_reset_full = np.empty(n_bars, dtype=bool)
        seg_reset_full[0] = True
        if n_bars > 1:
            seg_reset_full[1:] = bar_seg_id[1:] != bar_seg_id[:-1]
        for s in syms:
            open_s = np.asarray(tensor.exec_open_px[s], dtype=np.float64)
            close_s = np.asarray(tensor.mark_close_px[s], dtype=np.float64)
            pod_n_s = tensor.notional_frac[pod_list, s, :].astype(np.float64, copy=False)
            eff_open, q_members_full = _compute_eff_open_and_q_members(
                pod_n_s, open_s, close_s, pod_eq, seg_starts
            )
            q_s = weights @ q_members_full
            prev_q_s = np.where(seg_reset_full, 0.0, np.roll(q_s, 1))
            nz_bars = np.flatnonzero(np.abs(q_s - prev_q_s) > 1e-12)
            if nz_bars.size == 0:
                continue
            eff_nz = eff_open[nz_bars]
            px_nz = np.where(np.isfinite(eff_nz) & (eff_nz > 0.0), eff_nz, 1.0)
            _accumulate_symbol_unrl_at_resets(
                unrl,
                nz_bars,
                q_s[nz_bars],
                px_nz,
                bar_seg_id[nz_bars],
                close_s,
                bar_seg_id,
                reset_targets,
                pod_eq,
            )
        return unrl

    # Case 2: Synthetic fractional tensor -> trace sparse symbol netted unrealized recurrence
    idx_grid = np.ix_(pod_list, syms)
    n_mat = np.tensordot(
        weights, tensor.notional_frac[idx_grid].astype(np.float64, copy=False), axes=([0], [0])
    )
    dn_mat = np.tensordot(
        weights,
        tensor.delta_notional_frac[idx_grid].astype(np.float64, copy=False),
        axes=([0], [0]),
    )
    if tensor.sym_net_ret_frac is not None:
        pod_sym_gross = (
            tensor.sym_net_ret_frac[idx_grid].astype(np.float64, copy=False)
            + np.abs(tensor.delta_notional_frac[idx_grid].astype(np.float64, copy=False)) * cost_rate
        )
        g_mat = np.tensordot(weights, pod_sym_gross, axes=([0], [0]))
    elif len(syms) == 1:
        g_mat = (weights @ tensor.gross_ret_frac[pod_list])[None, :]
    else:
        g_tot = weights @ tensor.gross_ret_frac[pod_list]
        exp_wt = np.maximum(np.abs(n_mat), np.abs(n_mat - dn_mat))
        denom = np.sum(exp_wt, axis=0)
        g_mat = np.where(
            denom[None, :] > 1e-12,
            g_tot[None, :] * (exp_wt / np.maximum(denom[None, :], 1e-12)),
            0.0,
        )

    for a in range(len(syms)):
        n_a = n_mat[a]
        dn_a = dn_mat[a]
        g_a = g_mat[a]
        held_a = np.abs(n_a) > eps
        prev_held = np.concatenate(([False], held_a[:-1]))
        sign_flip = held_a & prev_held & ((n_a * np.concatenate(([0.0], n_a[:-1]))) < -1e-12)
        chg = np.flatnonzero((np.abs(dn_a) > eps) | (held_a != prev_held) | sign_flip)
        if chg.size == 0 and not np.any(held_a):
            continue
        csum = np.cumsum(g_a)
        u_after = np.empty(chg.size, dtype=np.float64)
        u_last = 0.0
        b_last = -1
        for k, b in enumerate(chg):
            b_i = int(b)
            prev_n = float(n_a[b_i - 1]) if b_i > 0 else 0.0
            base_c = float(csum[b_last]) if b_last >= 0 else 0.0
            u_before = 0.0 if abs(prev_n) <= eps else (u_last + (float(csum[b_i - 1]) - base_c))
            cur_n = float(n_a[b_i])
            g_b = float(g_a[b_i])
            if abs(cur_n) <= eps:
                u_last = 0.0
            elif abs(prev_n) <= eps:
                u_last = g_b
            elif prev_n * cur_n < -1e-12:
                u_last = 0.0
            elif abs(cur_n) < abs(prev_n) - eps:
                u_last = (u_before + g_b) * (abs(cur_n) / abs(prev_n))
            else:
                u_last = u_before + g_b
            b_last = b_i
            u_after[k] = u_last

        for di, r in reset_targets:
            if abs(float(n_a[r])) <= eps:
                continue
            k = int(np.searchsorted(chg, r, side="right")) - 1
            b_k = int(chg[k]) if k >= 0 else -1
            u_k = float(u_after[k]) if k >= 0 else 0.0
            base_k = float(csum[b_k]) if b_k >= 0 else 0.0
            unrl[di] += u_k + (float(csum[r]) - base_k)
    return unrl


def _evaluate_weights_with_cum(
    weights: np.ndarray,
    solo_net: np.ndarray,
    t_ov: np.ndarray,
    delta_ov: np.ndarray,
    cost_rate: float,
    day_slices: Sequence[tuple[int, int]],
    config: PodConstructorConfig,
    *,
    segments: Sequence[tuple[int, int]] | None = None,
    unrl_at_resets: np.ndarray | None = None,
) -> tuple[float, float, float, float, float, float, float, np.ndarray]:
    """Internal unit-scale evaluation returning ``cum_r`` alongside risk/return summary."""
    r_net = weights @ solo_net  # (T,)
    savings_tot = 0.0
    if t_ov.size > 0:
        netted_cost = np.abs(delta_ov @ weights) * cost_rate
        solo_weighted = (np.abs(delta_ov) @ weights) * cost_rate
        diff = solo_weighted - netted_cost
        savings_tot = float(np.sum(diff))
        np.add.at(r_net, t_ov, diff)

    cum_r = np.cumsum(r_net)
    if segments is not None and len(segments) > 1:
        _, peak_to_trough_dd, static_floor_dd, cdar_95 = compute_segmented_drawdown_metrics(
            r_net, segments, alpha=config.cdar_alpha
        )
        tot_ret = float(np.sum([np.sum(r_net[s:e]) for s, e in segments if e > s]))
    else:
        _, peak_to_trough_dd, static_floor_dd, cdar_95 = compute_drawdown_metrics(
            cum_r, alpha=config.cdar_alpha
        )
        tot_ret = float(cum_r[-1]) if cum_r.size > 0 else 0.0

    daily_losses = np.empty(len(day_slices), dtype=np.float64)
    for di, (sb, eb) in enumerate(day_slices):
        base = cum_r[sb - 1] if sb > 0 else 0.0
        intraday = cum_r[sb:eb] - base
        if config.daily_loss_mode == "intraday_trailing":
            peaks = np.maximum(0.0, np.maximum.accumulate(intraday))
            daily_losses[di] = float(np.max(peaks - intraday))
        else:
            unrl_carry = float(unrl_at_resets[di]) if unrl_at_resets is not None else 0.0
            daily_losses[di] = float(max(0.0, -unrl_carry, -float(np.min(unrl_carry + intraday))))

    max_daily = float(np.max(daily_losses)) if daily_losses.size > 0 else 0.0
    q99 = float(np.quantile(daily_losses, config.daily_cvar_alpha)) if daily_losses.size > 0 else 0.0
    tail = daily_losses >= q99
    daily_cvar = float(np.mean(daily_losses[tail])) if np.any(tail) else q99
    return tot_ret, peak_to_trough_dd, static_floor_dd, cdar_95, daily_cvar, max_daily, savings_tot, cum_r


def _evaluate_weights_unit_scale(
    weights: np.ndarray,
    solo_net: np.ndarray,
    t_ov: np.ndarray,
    delta_ov: np.ndarray,
    cost_rate: float,
    day_slices: Sequence[tuple[int, int]],
    config: PodConstructorConfig,
    *,
    segments: Sequence[tuple[int, int]] | None = None,
    unrl_at_resets: np.ndarray | None = None,
) -> tuple[float, float, float, float, float, float, float]:
    """Evaluate exact 5m netted unit-scale stats for normalized weight vector w."""
    res = _evaluate_weights_with_cum(
        weights,
        solo_net,
        t_ov,
        delta_ov,
        cost_rate,
        day_slices,
        config,
        segments=segments,
        unrl_at_resets=unrl_at_resets,
    )
    return res[:7]


def compute_euler_marginal_risk_shares(
    weights: np.ndarray,
    solo_net: np.ndarray,
    t_ov: np.ndarray,
    delta_ov: np.ndarray,
    cost_rate: float,
    day_slices: Sequence[tuple[int, int]],
    config: PodConstructorConfig,
    *,
    segments: Sequence[tuple[int, int]] | None = None,
    unrl_at_resets: np.ndarray | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Compute exact Euler additive member contributions to portfolio CDaR and DailyCVaR.

    Attributes overlap cost-netting savings proportionally to traded member turnover at each
    overlap cell so ``sum_m r_attr[m, t] == r_net[t]`` everywhere, and decomposes tail
    underwater (single- or multi-segment) and daily losses identically to 1.0.
    """
    m_size, n_bars = solo_net.shape
    w = np.asarray(weights, dtype=np.float64)
    w_sum = float(np.sum(w))
    w_norm = (w / w_sum) if w_sum > 1e-12 else np.full(m_size, 1.0 / max(1, m_size), dtype=np.float64)

    r_attr = w[:, None] * solo_net
    if t_ov.size > 0:
        scaled = np.abs(delta_ov * w[None, :])
        denom = np.sum(scaled, axis=1)
        netted_cost = np.abs(delta_ov @ w) * cost_rate
        diff = (denom * cost_rate) - netted_cost
        valid_e = denom > 1e-18
        if np.any(valid_e):
            share_e = np.divide(
                scaled, denom[:, None], out=np.zeros_like(scaled), where=valid_e[:, None]
            )
            contrib_e = share_e * diff[:, None]
            for m_idx in range(m_size):
                np.add.at(r_attr[m_idx], t_ov, contrib_e[:, m_idx])

    r_net = np.sum(r_attr, axis=0)
    cum_r = np.cumsum(r_net)
    cum_m = np.cumsum(r_attr, axis=1)

    valid_segs = (
        [(int(s), int(e)) for s, e in segments if e > s]
        if (segments is not None and len(segments) > 1)
        else ([(0, n_bars)] if n_bars > 0 else [])
    )
    if not valid_segs:
        cdar_shares = tuple(1.0 / m_size for _ in range(m_size))
    else:
        uw_parts: list[np.ndarray] = []
        uw_contrib_parts: list[np.ndarray] = []
        for s_seg, e_seg in valid_segs:
            cum_seg = np.cumsum(r_net[s_seg:e_seg])
            cum_m_seg = np.cumsum(r_attr[:, s_seg:e_seg], axis=1)
            running_peak = np.maximum(0.0, np.maximum.accumulate(cum_seg))
            uw_parts.append(running_peak - cum_seg)
            rp_pad = np.concatenate(([0.0], running_peak))
            up_bars = np.flatnonzero(rp_pad[1:] > rp_pad[:-1])
            if up_bars.size == 0:
                uw_contrib_parts.append(-cum_m_seg)
            else:
                local_idx = np.arange(e_seg - s_seg, dtype=np.intp)
                pos = np.searchsorted(up_bars, local_idx, side="right") - 1
                has_peak = pos >= 0
                peak_bars = up_bars[np.maximum(pos, 0)]
                peak_contrib = np.where(has_peak[None, :], cum_m_seg[:, peak_bars], 0.0)
                uw_contrib_parts.append(peak_contrib - cum_m_seg)

        underwater = np.concatenate(uw_parts) if len(uw_parts) > 1 else uw_parts[0]
        uw_contrib = (
            np.concatenate(uw_contrib_parts, axis=1)
            if len(uw_contrib_parts) > 1
            else uw_contrib_parts[0]
        )
        q_alpha = float(np.quantile(underwater, config.cdar_alpha)) if underwater.size > 0 else 0.0
        tail_idx = np.flatnonzero(underwater >= q_alpha)
        cdar_val = float(np.mean(underwater[tail_idx])) if tail_idx.size > 0 else q_alpha
        if cdar_val <= 1e-12 or tail_idx.size == 0:
            cdar_shares = tuple(1.0 / m_size for _ in range(m_size))
        else:
            comp_cdar = np.mean(uw_contrib[:, tail_idx], axis=1)
            cdar_shares = tuple(float(x / cdar_val) for x in comp_cdar)

    n_days = len(day_slices)
    if n_days == 0:
        dcvar_shares = tuple(1.0 / m_size for _ in range(m_size))
    else:
        daily_losses = np.empty(n_days, dtype=np.float64)
        daily_comp = np.zeros((m_size, n_days), dtype=np.float64)
        for di, (sb, eb) in enumerate(day_slices):
            base_r = cum_r[sb - 1] if sb > 0 else 0.0
            base_m = cum_m[:, sb - 1] if sb > 0 else np.zeros(m_size, dtype=np.float64)
            intraday = cum_r[sb:eb] - base_r
            if config.daily_loss_mode == "intraday_trailing":
                peaks = np.maximum(0.0, np.maximum.accumulate(intraday))
                dd_in = peaks - intraday
                rel_t = int(np.argmax(dd_in))
                loss_d = float(dd_in[rel_t])
                if peaks[rel_t] <= 0.0:
                    peak_m = base_m
                else:
                    rel_p = int(np.argmax(intraday[: rel_t + 1]))
                    peak_m = cum_m[:, sb + rel_p]
                trough_m = cum_m[:, sb + rel_t]
                if loss_d > 0.0:
                    daily_comp[:, di] = peak_m - trough_m
            else:
                unrl_carry = float(unrl_at_resets[di]) if unrl_at_resets is not None else 0.0
                rel_t = int(np.argmin(intraday))
                min_intra = float(intraday[rel_t])
                loss_d = float(max(0.0, -unrl_carry, -(unrl_carry + min_intra)))
                if loss_d > 0.0:
                    if min_intra >= 0.0:
                        daily_comp[:, di] = loss_d * w_norm
                    else:
                        daily_comp[:, di] = (base_m - cum_m[:, sb + rel_t]) - unrl_carry * w_norm
            daily_losses[di] = loss_d

        q99 = float(np.quantile(daily_losses, config.daily_cvar_alpha))
        tail_days = daily_losses >= q99
        dcvar_val = float(np.mean(daily_losses[tail_days])) if np.any(tail_days) else q99
        if dcvar_val <= 1e-12 or not np.any(tail_days):
            dcvar_shares = tuple(1.0 / m_size for _ in range(m_size))
        else:
            comp_dc = np.mean(daily_comp[:, tail_days], axis=1)
            dcvar_shares = tuple(float(x / dcvar_val) for x in comp_dc)

    return cdar_shares, dcvar_shares


def _build_unrl_reset_cache(
    pod_notional: np.ndarray,
    active_syms: Sequence[int],
    exec_open_px: np.ndarray,
    mark_close_px: np.ndarray,
    day_slices: Sequence[tuple[int, int]],
    segments: Sequence[tuple[int, int]] | None,
    pod_equity_usd: float,
) -> tuple[
    list[tuple[int, int]],
    list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
]:
    """Precompute per-symbol member quantity snapshots on candidate change bars."""
    n_bars = exec_open_px.shape[1] if exec_open_px.ndim == 2 else 0
    reset_targets, bar_seg_id, seg_starts = _build_reset_seg_context(
        n_bars, day_slices, segments
    )
    sym_cache: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    if not reset_targets or not active_syms:
        return reset_targets, sym_cache

    seg_reset_m = np.empty(n_bars, dtype=bool)
    seg_reset_m[0] = True
    if n_bars > 1:
        seg_reset_m[1:] = bar_seg_id[1:] != bar_seg_id[:-1]

    for a_idx, s in enumerate(active_syms):
        open_s = np.asarray(exec_open_px[s], dtype=np.float64)
        close_s = np.asarray(mark_close_px[s], dtype=np.float64)
        eff_open, q_members_full = _compute_eff_open_and_q_members(
            pod_notional[:, a_idx, :], open_s, close_s, pod_equity_usd, seg_starts
        )
        prev_q_m = np.where(seg_reset_m[None, :], 0.0, np.roll(q_members_full, 1, axis=1))
        cand_bars = np.flatnonzero(np.any(np.abs(q_members_full - prev_q_m) > 1e-12, axis=0))
        if cand_bars.size > 0:
            eff_cand = eff_open[cand_bars]
            px_cand = np.where(np.isfinite(eff_cand) & (eff_cand > 0.0), eff_cand, 1.0)
            sym_cache.append(
                (
                    q_members_full[:, cand_bars].copy(),
                    px_cand,
                    close_s,
                    cand_bars,
                    bar_seg_id[cand_bars],
                    bar_seg_id,
                )
            )
    return reset_targets, sym_cache


def _eval_unrl_reset_cache(
    weights: np.ndarray,
    n_days: int,
    reset_targets: Sequence[tuple[int, int]],
    sym_cache: Sequence[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
    pod_equity_usd: float,
) -> np.ndarray:
    """Evaluate netted unrealized PnL fraction at daily resets using sparse candidate bars."""
    unrl = np.zeros(n_days, dtype=np.float64)
    if not reset_targets or not sym_cache:
        return unrl

    di_arr = np.fromiter((di for di, _ in reset_targets), dtype=np.int64, count=len(reset_targets))
    r_arr = np.fromiter((r for _, r in reset_targets), dtype=np.int64, count=len(reset_targets))
    reset_arrays = (di_arr, r_arr)

    for q_members_cand, px_cand, close_s, cand_bars, cand_seg_id, bar_seg_id in sym_cache:
        q_cand = weights @ q_members_cand
        prev_q_cand = np.empty_like(q_cand)
        prev_q_cand[0] = 0.0
        if q_cand.size > 1:
            prev_q_cand[1:] = np.where(cand_seg_id[1:] != cand_seg_id[:-1], 0.0, q_cand[:-1])
        keep = np.abs(q_cand - prev_q_cand) > 1e-12
        if not np.any(keep):
            continue
        _accumulate_symbol_unrl_at_resets(
            unrl,
            cand_bars[keep],
            q_cand[keep],
            px_cand[keep],
            cand_seg_id[keep],
            close_s,
            bar_seg_id,
            reset_targets,
            pod_equity_usd,
            _reset_arrays=reset_arrays,
        )
    return unrl


def size_pod_dual_barrier(
    pod_indices: Sequence[int],
    tensor: UnitCopy5mTensor,
    metrics_list: Sequence[TraderAuditMetrics],
    config: PodConstructorConfig = PodConstructorConfig(),
    *,
    segments: Sequence[tuple[int, int]] | None = None,
) -> SizedPodCandidate:
    """Solve joint Pod k_i on exact 5m netted streams using homogeneity & barrier caps.

    Because pathwise CDaR, Daily CVaR, peak-to-trough DD, static-floor DD, and exact 5m
    netted returns are all positively homogeneous of degree 1 in overall pod scale
    lambda > 0, optimizing bounded risk-share weights over the low-dimensional simplex
    with closed-form scale projection (including joint 1.5x symbol & 3.0x gross
    current-equity caps) finds the global constrained optimum without post-hoc clipping.
    """
    pod = tuple(pod_indices)
    m_size = len(pod)
    s_lo = float(config.min_risk_share)
    s_hi = float(config.max_risk_share)
    if (
        m_size <= 0
        or s_lo < 0.0
        or s_hi > 1.0
        or s_lo > s_hi
        or s_lo * m_size > 1.0 + 1e-9
        or s_hi * m_size < 1.0 - 1e-9
    ):
        raise ValueError(
            f"INFEASIBLE_RISK_SHARE_BOUNDS: pod_size={m_size}, "
            f"min_risk_share={s_lo}, max_risk_share={s_hi}"
        )

    cost_rate = config.cost_bps * 1e-4
    pod_list = list(pod)
    solo_net, t_ov, delta_ov = _build_sparse_overlap_cache(
        tensor.delta_notional_frac[pod_list],
        tensor.net_ret_frac[pod_list],
    )
    day_slices = _daily_reset_slices(
        tensor.timestamps_ms, config.bars_per_day, config.reset_bar_offset, segments=segments
    )
    active_syms = [
        s
        for s in range(len(tensor.symbols))
        if tensor.notional_frac[pod_list, s, :].any()
        or tensor.delta_notional_frac[pod_list, s, :].any()
    ]
    pod_notional = (
        tensor.notional_frac[np.ix_(pod_list, active_syms)].astype(np.float64, copy=False)
        if active_syms
        else np.zeros((m_size, 0, tensor.timestamps_ms.size), dtype=np.float64)
    )

    risk_unit = np.array(
        [max(metrics_list[idx].standalone_cdar95, 1e-4) for idx in pod],
        dtype=np.float64,
    )

    cdar_lim = config.cdar_buffer_frac * config.max_dd_frac
    hard_dd_lim = (
        float(config.tier_peak_dd_cap_frac)
        if config.tier_peak_dd_cap_frac is not None
        else (config.hard_dd_buffer_frac * config.max_dd_frac)
    )
    dcvar_lim = config.daily_cvar_buffer_frac * config.daily_loss_frac
    hard_daily_lim = (
        float(config.tier_daily_loss_cap_frac)
        if config.tier_daily_loss_cap_frac is not None
        else (config.hard_daily_buffer_frac * config.daily_loss_frac)
    )
    tier1_base_k = (
        np.array(
            [
                min(
                    (config.baseline_port_dd_frac / max(1, m_size))
                    / max(metrics_list[idx].standalone_peak_to_trough_dd, 1e-6),
                    config.max_k_mult,
                )
                for idx in pod
            ],
            dtype=np.float64,
        )
        if config.sizing_tier == 1
        else None
    )
    enforce_marginal = (config.risk_share_mode == "marginal_euler") or (
        config.max_marginal_cdar_share is not None
    )
    marginal_cap = (
        float(config.max_marginal_cdar_share)
        if config.max_marginal_cdar_share is not None
        else s_hi
    )

    open_px = tensor.exec_open_px
    close_px = tensor.mark_close_px
    if config.daily_loss_mode == "reset_balance" and open_px is not None and close_px is not None:
        reset_targets, unrl_sym_cache = _build_unrl_reset_cache(
            pod_notional,
            active_syms,
            open_px,
            close_px,
            day_slices,
            segments,
            tensor.pod_equity_usd,
        )
    else:
        reset_targets, unrl_sym_cache = [], []

    def _marginal_ok(m_shares: tuple[float, ...]) -> bool:
        if max(m_shares) > marginal_cap + 1e-9:
            return False
        return not (config.risk_share_mode == "marginal_euler" and min(m_shares) < s_lo - 1e-9)

    def score_shares(
        shares: np.ndarray,
    ) -> tuple[
        bool,
        float,
        np.ndarray,
        str,
        tuple[float, ...],
        tuple[float, ...],
        tuple[float, ...],
        np.ndarray,
        np.ndarray | None,
    ]:
        w = shares / risk_unit
        w_sum = float(np.sum(w))
        if w_sum <= 1e-12:
            return False, -1e9, np.zeros(m_size), "INVALID", (0.0,) * 9, (), (), w, None
        w /= w_sum

        if config.daily_loss_mode != "reset_balance":
            unrl_resets = None
        elif open_px is not None and close_px is not None:
            unrl_resets = _eval_unrl_reset_cache(
                w, len(day_slices), reset_targets, unrl_sym_cache, tensor.pod_equity_usd
            )
        else:
            unrl_resets = _compute_netted_unrealized_frac_at_resets(
                w,
                tensor,
                pod,
                day_slices,
                cost_rate,
                active_syms=active_syms,
                segments=segments,
                eps=config.flat_notional_eps_frac,
            )

        m_cdar_s: tuple[float, ...] = ()
        m_dcvar_s: tuple[float, ...] = ()
        if enforce_marginal:
            m_cdar_s, m_dcvar_s = compute_euler_marginal_risk_shares(
                w,
                solo_net,
                t_ov,
                delta_ov,
                cost_rate,
                day_slices,
                config,
                segments=segments,
                unrl_at_resets=unrl_resets,
            )
            if not _marginal_ok(m_cdar_s):
                return (
                    False,
                    -1e9,
                    np.zeros(m_size),
                    "MARGINAL_BOUND_EXCEEDED",
                    (0.0,) * 9,
                    m_cdar_s,
                    m_dcvar_s,
                    w,
                    unrl_resets,
                )

        ret_u, peak_u, static_u, cdar_u, dcvar_u, mdaily_u, sav_u, cum_r_u = _evaluate_weights_with_cum(
            w,
            solo_net,
            t_ov,
            delta_ov,
            cost_rate,
            day_slices,
            config,
            segments=segments,
            unrl_at_resets=unrl_resets,
        )

        # Closed-form maximum feasible scale lambda* under all risk + current-equity exposure caps
        caps: list[tuple[str, float]] = [
            ("CDaR_95", cdar_lim / max(cdar_u, 1e-9)),
            ("DailyCVaR_99", dcvar_lim / max(dcvar_u, 1e-9)),
            ("Hard_MaxDD", hard_dd_lim / max(peak_u, 1e-9)),
            ("Hard_Daily", hard_daily_lim / max(mdaily_u, 1e-9)),
            ("Max_K_Cap", config.max_k_mult / max(float(np.max(w)), 1e-9)),
        ]
        if pod_notional.shape[1] > 0:
            sym_u = np.tensordot(w, pod_notional, axes=([0], [0]))  # (S_act, T)
            abs_sym_u = np.abs(sym_u)
            max_sym_bar_u = np.max(abs_sym_bar := abs_sym_u, axis=0)
            gross_bar_u = np.sum(abs_sym_bar, axis=0)
            peak_sym_u = float(np.max(max_sym_bar_u))
            peak_gross_u = float(np.max(gross_bar_u))
        else:
            t_len = cum_r_u.size
            max_sym_bar_u = np.zeros(t_len, dtype=np.float64)
            gross_bar_u = np.zeros(t_len, dtype=np.float64)
            peak_sym_u = 0.0
            peak_gross_u = 0.0

        neg_cum_u = np.minimum(0.0, cum_r_u)
        if config.max_symbol_exposure_frac is not None:
            c_sym = float(config.max_symbol_exposure_frac)
            denom_sym = (
                float(np.max(max_sym_bar_u - c_sym * neg_cum_u)) if max_sym_bar_u.size > 0 else 0.0
            )
            caps.append(("Symbol_1.5x_Cap", c_sym / max(denom_sym, 1e-9)))
        if config.max_gross_exposure_frac is not None:
            c_gr = float(config.max_gross_exposure_frac)
            denom_gr = (
                float(np.max(gross_bar_u - c_gr * neg_cum_u)) if gross_bar_u.size > 0 else 0.0
            )
            caps.append(("Gross_3.0x_Cap", c_gr / max(denom_gr, 1e-9)))
        if tier1_base_k is not None:
            caps.append(("Tier1_Standalone_Cap", float(np.min(tier1_base_k / np.maximum(w, 1e-12)))))

        bind_name, lam = min(caps, key=lambda item: item[1])
        k_vec = lam * w
        scaled_ret = lam * ret_u
        return (
            True,
            scaled_ret,
            k_vec,
            bind_name,
            (
                scaled_ret,
                lam * cdar_u,
                lam * peak_u,
                lam * static_u,
                lam * dcvar_u,
                lam * mdaily_u,
                lam * sav_u,
                lam * peak_sym_u,
                lam * peak_gross_u,
            ),
            m_cdar_s,
            m_dcvar_s,
            w,
            unrl_resets,
        )

    # Generate deterministic grid over feasible bounded risk-share simplex [min_s, max_s]
    candidates_s: list[np.ndarray] = [np.full(m_size, 1.0 / m_size)]
    step = 0.05
    grid_vals = np.round(np.arange(s_lo, s_hi + 1e-9, step), 4)
    for combo in product(grid_vals, repeat=max(0, m_size - 1)):
        rem = round(1.0 - sum(combo), 6)
        if s_lo - 1e-6 <= rem <= s_hi + 1e-6:
            candidates_s.append(np.array([*combo, max(0.0, rem)], dtype=np.float64))

    best_obj = -float("inf")
    best_s = np.full(m_size, 1.0 / m_size)
    best_res: (
        tuple[
            bool,
            float,
            np.ndarray,
            str,
            tuple[float, ...],
            tuple[float, ...],
            tuple[float, ...],
            np.ndarray,
            np.ndarray | None,
        ]
        | None
    ) = None

    for s_vec in candidates_s:
        res = score_shares(s_vec)
        if res[0] and (best_res is None or res[1] > best_obj):
            best_obj = res[1]
            best_s = s_vec
            best_res = res

    if best_res is None:
        raise ValueError(
            f"INFEASIBLE_MARGINAL_RISK_SHARE_BOUNDS: pod={pod}, "
            f"mode={config.risk_share_mode}, max_marginal_cdar_share={marginal_cap}"
        )

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
                        if res[0] and res[1] > best_obj + 1e-9:
                            best_obj = res[1]
                            best_s = trial
                            best_res = res
                            improved = True

    assert best_res is not None
    (
        _,
        _,
        k_opt,
        bind_name,
        (
            net_ret,
            cdar,
            peak_dd,
            static_dd,
            dcvar,
            mdaily,
            savings,
            peak_sym_exp,
            peak_gross_exp,
        ),
        m_cdar_s,
        m_dcvar_s,
        best_w,
        best_unrl,
    ) = best_res
    if not m_cdar_s:
        m_cdar_s, m_dcvar_s = compute_euler_marginal_risk_shares(
            best_w,
            solo_net,
            t_ov,
            delta_ov,
            cost_rate,
            day_slices,
            config,
            segments=segments,
            unrl_at_resets=best_unrl,
        )

    warnings_list: list[str] = []
    for k_val, idx in zip(k_opt, pod):
        m_info = metrics_list[idx]
        min_pos_usd = k_val * m_info.active_pos_q10_frac * tensor.pod_equity_usd
        if m_info.active_pos_q10_frac > 0 and min_pos_usd < config.min_lot_usd:
            warnings_list.append(
                f"{m_info.alias}: q10_order=${min_pos_usd:.2f}<${config.min_lot_usd:.0f}"
            )

    feasible = (
        (net_ret > 0.0)
        and (peak_dd <= config.max_dd_frac)
        and (static_dd <= config.max_dd_frac)
        and (mdaily <= config.daily_loss_frac)
        and (
            config.max_symbol_exposure_frac is None
            or peak_sym_exp <= float(config.max_symbol_exposure_frac) + 1e-6
        )
        and (
            config.max_gross_exposure_frac is None
            or peak_gross_exp <= float(config.max_gross_exposure_frac) + 1e-6
        )
    )
    return SizedPodCandidate(
        aliases=tuple(tensor.aliases[idx] for idx in pod),
        pod_indices=pod,
        multipliers=k_opt,
        objective_net_return=net_ret,
        cdar_95=cdar,
        peak_to_trough_dd=peak_dd,
        static_floor_dd=static_dd,
        daily_cvar_99=dcvar,
        max_daily_loss=mdaily,
        net_over_max_dd=net_ret / max(peak_dd, 1e-6),
        netting_savings_frac=savings,
        binding_constraint=bind_name,
        feasible=feasible,
        min_lot_warnings=tuple(warnings_list),
        peak_symbol_exposure_frac=peak_sym_exp,
        peak_gross_exposure_frac=peak_gross_exp,
        marginal_cdar_shares=m_cdar_s,
        marginal_daily_cvar_shares=m_dcvar_s,
    )


def _size_pod_if_feasible(
    pod: tuple[int, ...],
    tensor: UnitCopy5mTensor,
    metrics_list: Sequence[TraderAuditMetrics],
    config: PodConstructorConfig,
    *,
    segments: Sequence[tuple[int, int]] | None = None,
) -> tuple[SizedPodCandidate | None, ValueError | None]:
    try:
        return size_pod_dual_barrier(pod, tensor, metrics_list, config, segments=segments), None
    except ValueError as exc:
        if str(exc).startswith("INFEASIBLE_MARGINAL_RISK_SHARE_BOUNDS"):
            return None, exc
        raise


def discover_diverse_pods(
    tensor: UnitCopy5mTensor,
    config: PodConstructorConfig = PodConstructorConfig(),
    *,
    segments: Sequence[tuple[int, int]] | None = None,
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

    sized: list[SizedPodCandidate] = []
    last_infeasible: ValueError | None = None
    for pod in candidate_pods:
        cand, err = _size_pod_if_feasible(pod, tensor, metrics_list, config, segments=segments)
        if cand is not None:
            sized.append(cand)
        elif err is not None:
            last_infeasible = err
    if not sized:
        if last_infeasible is not None:
            raise last_infeasible
        raise ValueError("NO_FEASIBLE_CANDIDATE_PODS")
    sized.sort(
        key=lambda s: (s.feasible, s.net_over_max_dd, s.objective_net_return),
        reverse=True,
    )
    return sized, metrics_list, baselines, {"distance_matrix": dist_mat, **comps}


def _cpcv_sizing_metrics(
    tensor: UnitCopy5mTensor,
    config: PodConstructorConfig,
) -> list[TraderAuditMetrics]:
    """Compute exact per-trader risk and activity metrics required by size_pod_dual_barrier."""
    n_traders = len(tensor.aliases)
    n_bars = tensor.timestamps_ms.size
    metrics: list[TraderAuditMetrics] = []
    for i in range(n_traders):
        act_idx, q10 = _compute_trader_activity_q10(tensor, i, config.flat_notional_eps_frac)
        t_act = act_idx.size
        cum_r = np.cumsum(tensor.net_ret_frac[i])
        _, peak_dd, static_dd, cdar95 = compute_drawdown_metrics(cum_r, alpha=config.cdar_alpha)
        max_dd = max(peak_dd, 1e-6)
        tot_ret = float(cum_r[-1]) if cum_r.size > 0 else 0.0
        metrics.append(
            TraderAuditMetrics(
                trader_id=i,
                alias=tensor.aliases[i],
                active_bars=t_act,
                active_frac=t_act / float(max(1, n_bars)),
                num_episodes=0,
                raw_sharpe_ann=0.0,
                eb_shrunk_sharpe_ann=0.0,
                deflated_sharpe_prob=0.5,
                net_over_max_dd=tot_ret / max_dd,
                chop_net_over_dd=tot_ret / max_dd,
                standalone_cdar95=max(cdar95, 1e-6),
                standalone_peak_to_trough_dd=max_dd,
                standalone_static_floor_dd=static_dd,
                tilt_multiplier=None,
                tilt_status="UNKNOWN",
                active_pos_q10_frac=q10,
                passed_behavioral_gate=True,
            )
        )
    return metrics


def _slice_tensor_for_traders(
    tensor: UnitCopy5mTensor,
    candidate_pods: Sequence[tuple[int, ...]],
) -> tuple[UnitCopy5mTensor, list[tuple[int, ...]]]:
    """Slice tensor strictly to unique traders in candidate_pods and their active symbols."""
    unique_traders = sorted({tid for pod in candidate_pods for tid in pod})
    remap = {tid: new_idx for new_idx, tid in enumerate(unique_traders)}
    remapped_pods = [tuple(remap[tid] for tid in pod) for pod in candidate_pods]

    n_syms_full = len(tensor.symbols)
    used_sym_mask = np.zeros(n_syms_full, dtype=bool)
    for tid in unique_traders:
        for s in range(n_syms_full):
            if not used_sym_mask[s] and (
                tensor.notional_frac[tid, s].any() or tensor.delta_notional_frac[tid, s].any()
            ):
                used_sym_mask[s] = True
    active_syms = np.flatnonzero(used_sym_mask)
    if active_syms.size == 0 and n_syms_full > 0:
        active_syms = np.array([0], dtype=np.intp)

    sub_tensor = UnitCopy5mTensor(
        aliases=tuple(tensor.aliases[tid] for tid in unique_traders),
        symbols=tuple(tensor.symbols[int(s)] for s in active_syms),
        timestamps_ms=tensor.timestamps_ms,
        notional_frac=tensor.notional_frac[np.ix_(unique_traders, active_syms)],
        delta_notional_frac=tensor.delta_notional_frac[np.ix_(unique_traders, active_syms)],
        gross_ret_frac=tensor.gross_ret_frac[unique_traders],
        net_ret_frac=tensor.net_ret_frac[unique_traders],
        sym_net_ret_frac=(
            tensor.sym_net_ret_frac[np.ix_(unique_traders, active_syms)]
            if tensor.sym_net_ret_frac is not None
            else None
        ),
        chop_mask=tensor.chop_mask,
        pod_equity_usd=tensor.pod_equity_usd,
        exec_open_px=tensor.exec_open_px[active_syms] if tensor.exec_open_px is not None else None,
        mark_close_px=(
            tensor.mark_close_px[active_syms] if tensor.mark_close_px is not None else None
        ),
        carried_in_mask=(
            tensor.carried_in_mask[np.ix_(unique_traders, active_syms)]
            if tensor.carried_in_mask is not None
            else None
        ),
    )
    return sub_tensor, remapped_pods


def _compute_active_2d_mask(tensor: UnitCopy5mTensor) -> np.ndarray:
    """Return an (N, S) bool mask where at least one 3D tensor slab is non-zero."""
    mask = np.any(tensor.notional_frac, axis=2) | np.any(tensor.delta_notional_frac, axis=2)
    if tensor.sym_net_ret_frac is not None:
        mask |= np.any(tensor.sym_net_ret_frac, axis=2)
    return mask


def _normalize_bar_idx(bar_idx: np.ndarray, n_total: int) -> np.ndarray:
    """Validate 1D integer bar indices and normalize negative offsets."""
    raw = np.asarray(bar_idx)
    if raw.dtype == np.bool_ or raw.ndim != 1:
        raise ValueError("BAR_IDX_MUST_BE_1D_INT")
    idx = np.asarray(raw, dtype=np.intp).copy()
    neg = idx < 0
    if np.any(neg):
        idx[neg] += n_total
    if idx.size > 0 and np.any((idx < 0) | (idx >= n_total)):
        raise IndexError("BAR_IDX_OUT_OF_BOUNDS")
    return idx


def _slice_3d_sparse_bars(
    arr: np.ndarray,
    bar_idx: np.ndarray,
    active_mask_2d: np.ndarray,
) -> np.ndarray:
    """Slice a 3D (N, S, T) tensor along axis 2, keeping inactive zero slabs demand-zero."""
    idx = _normalize_bar_idx(bar_idx, arr.shape[2])
    out = np.zeros((arr.shape[0], arr.shape[1], idx.size), dtype=arr.dtype)
    if idx.size == 0 or not np.any(active_mask_2d):
        return out

    breaks = np.flatnonzero(np.diff(idx) != 1) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [idx.size]))
    runs = [
        (int(idx[ds]), int(idx[de - 1]) + 1, int(ds), int(de))
        for ds, de in zip(starts, ends, strict=True)
    ]
    active_pairs = np.argwhere(active_mask_2d)
    if len(runs) <= 16:
        for ti_arr, si_arr in active_pairs:
            ti, si = int(ti_arr), int(si_arr)
            for src_s, src_e, dst_s, dst_e in runs:
                out[ti, si, dst_s:dst_e] = arr[ti, si, src_s:src_e]
    else:
        for ti_arr, si_arr in active_pairs:
            ti, si = int(ti_arr), int(si_arr)
            out[ti, si] = arr[ti, si, idx]
    return out


def _slice_tensor_bars(
    tensor: UnitCopy5mTensor,
    bar_idx: np.ndarray,
    *,
    active_mask_2d: np.ndarray | None = None,
    include_sym_net_ret: bool = True,
) -> UnitCopy5mTensor:
    """Slice a tensor along the time axis using an integer bar-index array."""
    idx = _normalize_bar_idx(bar_idx, tensor.timestamps_ms.size)
    mask2d = active_mask_2d if active_mask_2d is not None else _compute_active_2d_mask(tensor)
    if idx.size > 0 and int(idx[0]) > 0:
        b0 = int(idx[0])
        prev_n = tensor.notional_frac[:, :, b0 - 1]
        cur_n = tensor.notional_frac[:, :, b0]
        carried: np.ndarray | None = (np.abs(prev_n) > 1e-4) & ((prev_n * cur_n) > 0.0)
    else:
        carried = tensor.carried_in_mask.copy() if tensor.carried_in_mask is not None else None
    return UnitCopy5mTensor(
        aliases=tensor.aliases,
        symbols=tensor.symbols,
        timestamps_ms=tensor.timestamps_ms[idx],
        notional_frac=_slice_3d_sparse_bars(tensor.notional_frac, idx, mask2d),
        delta_notional_frac=_slice_3d_sparse_bars(tensor.delta_notional_frac, idx, mask2d),
        gross_ret_frac=tensor.gross_ret_frac[:, idx],
        net_ret_frac=tensor.net_ret_frac[:, idx],
        sym_net_ret_frac=(
            _slice_3d_sparse_bars(tensor.sym_net_ret_frac, idx, mask2d)
            if include_sym_net_ret and tensor.sym_net_ret_frac is not None
            else None
        ),
        chop_mask=tensor.chop_mask[idx] if tensor.chop_mask is not None else None,
        pod_equity_usd=tensor.pod_equity_usd,
        exec_open_px=tensor.exec_open_px[:, idx] if tensor.exec_open_px is not None else None,
        mark_close_px=(
            tensor.mark_close_px[:, idx] if tensor.mark_close_px is not None else None
        ),
        carried_in_mask=carried,
    )


def compute_leakfree_cpcv_pbo(
    candidate_pods: Sequence[tuple[int, ...]],
    tensor: UnitCopy5mTensor,
    config: PodConstructorConfig = PodConstructorConfig(),
) -> tuple[float, np.ndarray]:
    """Compute leak-free CPCV Probability of Backtest Overfitting (PBO).

    Unlike naive in-sample replay, weights k_i are re-solved strictly inside each
    training block subset (with segmented non-spliced drawdowns and boundary purging)
    before scoring OOS. Slices the input tensor strictly to unique traders appearing in
    ``candidate_pods`` (and symbols traded by them) before iterating splits.
    """
    n_pods = len(candidate_pods)
    if n_pods < 2:
        return 0.0, np.array([1.0], dtype=np.float64)

    if not any(candidate_pods):
        return 0.0, np.array([1.0], dtype=np.float64)

    sub_tensor, remapped_pods = _slice_tensor_for_traders(tensor, candidate_pods)
    active_mask_2d = _compute_active_2d_mask(sub_tensor)
    te_need_sym_net = sub_tensor.exec_open_px is None or sub_tensor.mark_close_px is None
    n_bars = sub_tensor.timestamps_ms.size
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
        tr_segments = _contiguous_index_segments(tr_idx)
        te_segments = _contiguous_index_segments(te_idx)

        tr_tensor = _slice_tensor_bars(sub_tensor, tr_idx, active_mask_2d=active_mask_2d)
        te_tensor = _slice_tensor_bars(
            sub_tensor,
            te_idx,
            active_mask_2d=active_mask_2d,
            include_sym_net_ret=te_need_sym_net,
        )
        try:
            tr_metrics = _cpcv_sizing_metrics(tr_tensor, config)
            te_slices = _daily_reset_slices(
                te_tensor.timestamps_ms, bpd, config.reset_bar_offset, segments=te_segments
            )
            cost_rate = config.cost_bps * 1e-4

            is_scores: list[float] = []
            oos_scores: list[float] = []
            for pod in remapped_pods:
                sized_tr, _ = _size_pod_if_feasible(
                    pod, tr_tensor, tr_metrics, config, segments=tr_segments
                )
                if sized_tr is None:
                    continue
                is_scores.append(sized_tr.net_over_max_dd)

                solo_te, t_ov_te, delta_ov_te = _build_sparse_overlap_cache(
                    te_tensor.delta_notional_frac[list(pod)],
                    te_tensor.net_ret_frac[list(pod)],
                )
                unrl_te = (
                    _compute_netted_unrealized_frac_at_resets(
                        sized_tr.multipliers,
                        te_tensor,
                        pod,
                        te_slices,
                        cost_rate,
                        segments=te_segments,
                        eps=config.flat_notional_eps_frac,
                    )
                    if config.daily_loss_mode == "reset_balance"
                    else None
                )
                ret_te, peak_dd_te, _, _, _, _, _ = _evaluate_weights_unit_scale(
                    sized_tr.multipliers,
                    solo_te,
                    t_ov_te,
                    delta_ov_te,
                    cost_rate,
                    te_slices,
                    config,
                    segments=te_segments,
                    unrl_at_resets=unrl_te,
                )
                oos_scores.append(ret_te / max(peak_dd_te, 1e-6))

            n_feas = len(is_scores)
            if n_feas < 2:
                continue
            best_is = int(np.argmax(np.asarray(is_scores, dtype=np.float64)))
            ranks = rankdata_average(np.asarray(oos_scores, dtype=np.float64))
            omega = float(np.clip(ranks[best_is] / float(n_feas + 1), 1e-4, 1.0 - 1e-4))
            logits.append(math.log(omega / (1.0 - omega)))
        finally:
            del tr_tensor, te_tensor

    logits_arr = np.asarray(logits, dtype=np.float64) if logits else np.array([0.0])
    pbo = float(np.mean(logits_arr <= 0.0))
    return pbo, logits_arr


def _stage1_audit_segmented(
    tensor: UnitCopy5mTensor,
    segments: Sequence[tuple[int, int]],
    config: PodConstructorConfig,
) -> tuple[list[int], list[TraderAuditMetrics]]:
    """Run Stage 1 screening on non-adjacent CPCV segments without cross-gap stitching."""
    n_traders, n_syms, n_bars = tensor.notional_frac.shape
    if len(segments) <= 1:
        passed_ids, metrics, _ = stage1_audit_traders(tensor, config)
        return passed_ids, metrics

    # Censor episodes inside each contiguous segment so open positions never span purged gaps.
    seg_slices = [tensor.slice_time(s, e) for s, e in segments if e > s]
    ep_maps = [reconstruct_execution_episodes(slc) for slc in seg_slices]
    merged_eps: dict[int, dict[str, np.ndarray]] = {}
    for i in range(n_traders):
        parts = [em[i] for em in ep_maps if em[i]["ep_pnl"].size > 0]
        if not parts:
            merged_eps[i] = _empty_episodes_dict(n_syms)
            continue
        merged: dict[str, np.ndarray] = {}
        bar_offset = 0
        offset_parts: list[dict[str, np.ndarray]] = []
        for slc, em in zip(seg_slices, ep_maps):
            ep_i = em[i]
            if ep_i["ep_pnl"].size > 0:
                copied = dict(ep_i)
                copied["ep_start_bar"] = ep_i["ep_start_bar"] + bar_offset
                copied["ep_end_bar"] = ep_i["ep_end_bar"] + bar_offset
                copied["ep_pnl_end_bar"] = ep_i["ep_pnl_end_bar"] + bar_offset
                offset_parts.append(copied)
            bar_offset += slc.timestamps_ms.size
        for k in offset_parts[0]:
            merged[k] = np.concatenate([p[k] for p in offset_parts], axis=0)
        merged_eps[i] = merged

    raw_ropn_vec, eb_ropn_vec, eb_ropn_sd_vec = compute_eb_shrunk_ropn(
        merged_eps,
        n_traders,
        pod_equity_usd=tensor.pod_equity_usd,
        min_episode_notional_usd=config.min_episode_notional_usd,
        flat_eps_frac=config.flat_notional_eps_frac,
    )

    metrics: list[TraderAuditMetrics] = []
    passed_ids: list[int] = []
    bars_per_year = float(config.bars_per_day * config.days_per_year)
    for i in range(n_traders):
        act_idx, q10 = _compute_trader_activity_q10(tensor, i, config.flat_notional_eps_frac)
        t_act = act_idx.size
        act_frac = t_act / float(max(1, n_bars))
        if t_act >= 20:
            r_act = tensor.net_ret_frac[i, act_idx]
            sr_b = float(np.mean(r_act)) / (float(np.std(r_act, ddof=1)) + 1e-12)
            sr_a = sr_b * math.sqrt(bars_per_year * act_frac)
        else:
            sr_b, sr_a = 0.0, 0.0

        _, peak_dd, static_dd, cdar95 = compute_segmented_drawdown_metrics(
            tensor.net_ret_frac[i], segments, alpha=config.cdar_alpha
        )
        tot_ret = float(np.sum([np.sum(tensor.net_ret_frac[i, s:e]) for s, e in segments if e > s]))
        net_over_dd = tot_ret / max(peak_dd, 1e-6)
        chop_net_over_dd = net_over_dd
        if tensor.chop_mask is not None and np.any(tensor.chop_mask):
            chop_segs = [(s, e) for s, e in segments if np.any(tensor.chop_mask[s:e])]
            if chop_segs:
                chop_r = np.where(tensor.chop_mask, tensor.net_ret_frac[i], 0.0)
                _, c_mdd, _, _ = compute_segmented_drawdown_metrics(
                    chop_r, chop_segs, alpha=config.cdar_alpha
                )
                chop_net_over_dd = float(np.sum(chop_r)) / max(c_mdd, 1e-6)

        ep_all = merged_eps[i]
        right_c_pnl = float(np.sum(ep_all["ep_pnl"][ep_all["ep_right_censored"]]))
        ep = filter_uncensored_episodes(ep_all)
        n_ep = int(ep["ep_pnl"].size)
        closed_g = float(np.sum(np.maximum(0.0, ep["ep_pnl"]))) if n_ep > 0 else 0.0
        tilt_val, tilt_stat = _compute_causal_post_loss_tilt(
            ep["ep_start_bar"],
            ep["ep_end_bar"],
            ep["ep_pnl_end_bar"],
            ep["ep_pnl"],
            ep["ep_peak_notional"],
            max_tilt_multiplier=config.max_tilt_multiplier,
            min_comparisons=config.min_tilt_comparisons,
        )
        tilt_val, tilt_stat = _check_right_censored_revenge_tilt(
            ep_all,
            ep,
            max_tilt_multiplier=config.max_tilt_multiplier,
            existing_tilt=tilt_val,
            existing_status=tilt_stat,
        )

        reasons: list[str] = []
        if act_frac < config.min_active_bar_frac:
            reasons.append(f"low_activity({act_frac:.3f}<{config.min_active_bar_frac})")
        if n_ep < config.min_episodes:
            reasons.append(f"few_episodes({n_ep}<{config.min_episodes})")
        if tilt_stat == "FAIL" and tilt_val is not None:
            reasons.append(f"martingale_tilt({tilt_val:.2f}>{config.max_tilt_multiplier})")
        elif tilt_stat == "UNKNOWN" and config.reject_unknown_tilt:
            reasons.append("tilt_unknown")
        if net_over_dd <= 0.0 and chop_net_over_dd <= 0.0:
            reasons.append("non_positive_net_return")
        bag_reason = _hidden_bagholder_reason(right_c_pnl, tot_ret, closed_g, config)
        if bag_reason is not None:
            reasons.append(bag_reason)
        if config.min_baseline_order_usd > 0.0 and config.baseline_port_dd_frac > 0.0:
            base_k = (config.baseline_port_dd_frac / max(1, config.pod_size)) / max(peak_dd, 1e-6)
            eff_base_k = min(base_k, config.max_k_mult)
            if eff_base_k * q10 * tensor.pod_equity_usd < config.min_baseline_order_usd:
                reasons.append("below_min_baseline_order")

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
                raw_sharpe_ann=sr_a,
                eb_shrunk_sharpe_ann=sr_a,
                deflated_sharpe_prob=0.5,
                net_over_max_dd=net_over_dd,
                chop_net_over_dd=chop_net_over_dd,
                standalone_cdar95=max(cdar95, 1e-6),
                standalone_peak_to_trough_dd=max(peak_dd, 1e-6),
                standalone_static_floor_dd=static_dd,
                tilt_multiplier=tilt_val,
                tilt_status=tilt_stat,
                active_pos_q10_frac=q10,
                passed_behavioral_gate=passed,
                reject_reason=", ".join(reasons),
                censored_episodes=int(np.sum(~ep_all["ep_uncensored"])),
                left_censored_episodes=int(np.sum(ep_all["ep_left_censored"])),
                right_censored_episodes=int(np.sum(ep_all["ep_right_censored"])),
                open_exposure_at_cutoff=0.0,
                raw_ropn_mean=float(raw_ropn_vec[i]),
                eb_ropn_mean=float(eb_ropn_vec[i]),
                eb_ropn_post_sd=float(eb_ropn_sd_vec[i]),
                right_censored_pnl_frac=right_c_pnl,
            )
        )
    return passed_ids, metrics


def compute_full_pipeline_cpcv_pbo(
    tensor: UnitCopy5mTensor,
    config: PodConstructorConfig = PodConstructorConfig(),
) -> tuple[float, np.ndarray, list[dict[str, Any]]]:
    """Run full Stage 1 -> UPGMA -> Stage 2 discovery strictly inside each CPCV training fold."""
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
    cost_rate = config.cost_bps * 1e-4

    active_mask_2d = _compute_active_2d_mask(tensor)
    te_need_sym_net = tensor.exec_open_px is None or tensor.mark_close_px is None
    logits: list[float] = []
    split_records: list[dict[str, Any]] = []
    for split_idx, train_blocks in enumerate(combinations(range(n_blocks), half_b)):
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
        tr_segments = _contiguous_index_segments(tr_idx)
        te_segments = _contiguous_index_segments(te_idx)

        tr_tensor = _slice_tensor_bars(tensor, tr_idx, active_mask_2d=active_mask_2d)
        te_tensor = _slice_tensor_bars(
            tensor,
            te_idx,
            active_mask_2d=active_mask_2d,
            include_sym_net_ret=te_need_sym_net,
        )
        try:
            passed_ids, tr_metrics = _stage1_audit_segmented(tr_tensor, tr_segments, config)
            if len(passed_ids) < config.pod_size:
                split_records.append(
                    {
                        "split_idx": split_idx,
                        "train_blocks": train_blocks,
                        "status": "INSUFFICIENT_TRAIN_ELIGIBLE",
                        "passed_count": len(passed_ids),
                    }
                )
                continue

            dist_mat, _ = compute_crowding_distance_matrix(passed_ids, tr_tensor, config)
            n_clusters = min(config.pod_size, len(passed_ids))
            labels = cluster_candidates_upgma(dist_mat, n_clusters)
            clusters: dict[int, list[int]] = {}
            for pos_idx, c_label in enumerate(labels):
                clusters.setdefault(int(c_label), []).append(passed_ids[pos_idx])

            cluster_picks: list[list[int]] = []
            for c_id in sorted(clusters):
                members = clusters[c_id]
                by_overall = sorted(
                    members, key=lambda t: tr_metrics[t].net_over_max_dd, reverse=True
                )
                by_chop = sorted(
                    members, key=lambda t: tr_metrics[t].chop_net_over_dd, reverse=True
                )
                picked: list[int] = []
                for lst in (by_overall, by_chop):
                    for tid in lst:
                        if tid not in picked and len(picked) < config.top_per_cluster:
                            picked.append(tid)
                cluster_picks.append(picked)

            rank_lookup = {tid: r_idx for lst in cluster_picks for r_idx, tid in enumerate(lst)}
            raw_combos = {
                tuple(sorted(p)) for p in product(*cluster_picks) if len(set(p)) == len(cluster_picks)
            }
            cand_pods = sorted(raw_combos, key=lambda p: (sum(rank_lookup[t] for t in p), p))[
                : config.max_candidate_pods
            ]
            if len(cand_pods) < 2:
                split_records.append(
                    {
                        "split_idx": split_idx,
                        "train_blocks": train_blocks,
                        "status": "FEWER_THAN_TWO_CANDIDATES",
                        "passed_count": len(passed_ids),
                    }
                )
                continue

            te_slices = _daily_reset_slices(
                te_tensor.timestamps_ms, bpd, config.reset_bar_offset, segments=te_segments
            )
            is_scores_fp: list[float] = []
            oos_scores_fp: list[float] = []
            sized_list: list[SizedPodCandidate] = []
            for pod in cand_pods:
                sized_tr, _ = _size_pod_if_feasible(
                    pod, tr_tensor, tr_metrics, config, segments=tr_segments
                )
                if sized_tr is None:
                    continue
                sized_list.append(sized_tr)
                is_scores_fp.append(sized_tr.net_over_max_dd)

                solo_te, t_ov_te, delta_ov_te = _build_sparse_overlap_cache(
                    te_tensor.delta_notional_frac[list(pod)],
                    te_tensor.net_ret_frac[list(pod)],
                )
                unrl_te = (
                    _compute_netted_unrealized_frac_at_resets(
                        sized_tr.multipliers,
                        te_tensor,
                        pod,
                        te_slices,
                        cost_rate,
                        segments=te_segments,
                        eps=config.flat_notional_eps_frac,
                    )
                    if config.daily_loss_mode == "reset_balance"
                    else None
                )
                ret_te, peak_dd_te, _, _, _, _, _ = _evaluate_weights_unit_scale(
                    sized_tr.multipliers,
                    solo_te,
                    t_ov_te,
                    delta_ov_te,
                    cost_rate,
                    te_slices,
                    config,
                    segments=te_segments,
                    unrl_at_resets=unrl_te,
                )
                oos_scores_fp.append(ret_te / max(peak_dd_te, 1e-6))

            n_feas_fp = len(sized_list)
            if n_feas_fp < 2:
                split_records.append(
                    {
                        "split_idx": split_idx,
                        "train_blocks": train_blocks,
                        "status": "FEWER_THAN_TWO_FEASIBLE_CANDIDATES",
                        "passed_count": len(passed_ids),
                    }
                )
                continue

            best_is = int(np.argmax(np.asarray(is_scores_fp, dtype=np.float64)))
            ranks = rankdata_average(np.asarray(oos_scores_fp, dtype=np.float64))
            omega = float(np.clip(ranks[best_is] / float(n_feas_fp + 1), 1e-4, 1.0 - 1e-4))
            logit = math.log(omega / (1.0 - omega))
            logits.append(logit)
            split_records.append(
                {
                    "split_idx": split_idx,
                    "train_blocks": train_blocks,
                    "status": "OK",
                    "selected_aliases": sized_list[best_is].aliases,
                    "selected_multipliers": [float(x) for x in sized_list[best_is].multipliers],
                    "is_score": float(is_scores_fp[best_is]),
                    "oos_score": float(oos_scores_fp[best_is]),
                    "logit": logit,
                }
            )
        finally:
            del tr_tensor, te_tensor

    logits_arr = np.asarray(logits, dtype=np.float64) if logits else np.array([0.0])
    pbo = float(np.mean(logits_arr <= 0.0))
    return pbo, logits_arr, split_records


@dataclass(frozen=True)
class WalkForwardFoldResult:
    """Single non-overlapping OOS evaluation fold frozen strictly from its prior training window."""

    fold_idx: int
    tier: Literal[1, 2, 3, 4]
    train_start_ms: int
    train_end_ms: int
    eval_start_ms: int
    eval_end_ms: int
    aliases: tuple[str, ...]
    pod_indices: tuple[int, ...]
    multipliers: np.ndarray
    train_feasible: bool
    train_net_ret_frac: float
    train_peak_to_trough_dd: float
    train_static_floor_dd: float
    train_max_daily_loss: float
    oos_net_ret_frac: float
    oos_peak_to_trough_dd: float
    oos_static_floor_dd: float
    oos_max_daily_loss: float
    oos_peak_symbol_exposure_frac: float
    oos_peak_gross_exposure_frac: float
    oos_status: Literal["PASS", "FAIL_STATIC", "FAIL_DAILY", "RIGHT_CENSORED_OPEN"]
    binding_constraint: str


def _first_barrier_oos_status(
    cum_r: np.ndarray,
    day_slices: Sequence[tuple[int, int]],
    unrl_at_resets: np.ndarray | None,
    config: PodConstructorConfig,
) -> Literal["PASS", "FAIL_STATIC", "FAIL_DAILY", "RIGHT_CENSORED_OPEN"]:
    """Determine chronological first-barrier challenge verdict over an OOS return path."""
    if cum_r.size == 0:
        return "RIGHT_CENSORED_OPEN"
    for di, (sb, eb) in enumerate(day_slices):
        base = cum_r[sb - 1] if sb > 0 else 0.0
        unrl_carry = float(unrl_at_resets[di]) if unrl_at_resets is not None else 0.0
        if config.daily_loss_mode == "reset_balance" and unrl_carry <= -config.daily_loss_frac:
            return "FAIL_DAILY"
        day_peak = 0.0
        for t in range(sb, eb):
            c_t = float(cum_r[t])
            if c_t <= -config.max_dd_frac:
                return "FAIL_STATIC"
            intraday = c_t - base
            if config.daily_loss_mode == "intraday_trailing":
                day_peak = max(day_peak, intraday)
                if (day_peak - intraday) >= config.daily_loss_frac:
                    return "FAIL_DAILY"
            else:
                if (unrl_carry + intraday) <= -config.daily_loss_frac:
                    return "FAIL_DAILY"
            if c_t >= config.profit_target_frac:
                return "PASS"
    return "RIGHT_CENSORED_OPEN"


def run_rolling_walk_forward(
    tensor: UnitCopy5mTensor,
    *,
    train_days: Literal[60, 90] | int = 60,
    eval_days: int = 30,
    tiers: Sequence[Literal[1, 2, 3, 4]] = (1, 2, 3, 4),
    base_config: PodConstructorConfig = PodConstructorConfig(),
) -> list[WalkForwardFoldResult]:
    """Evaluate Tiers 1–4 on leak-free rolling training -> non-overlapping OOS folds."""
    bpd = base_config.bars_per_day
    train_bars = int(train_days) * bpd
    eval_bars = int(eval_days) * bpd
    n_bars = tensor.timestamps_ms.size
    if train_bars <= 0 or eval_bars <= 0:
        raise ValueError("WALK_FORWARD_WINDOWS_MUST_BE_POSITIVE")

    results: list[WalkForwardFoldResult] = []
    fold_idx = 0
    cut_bar = train_bars
    cost_rate = base_config.cost_bps * 1e-4

    while cut_bar + eval_bars <= n_bars:
        train_t = tensor.slice_time(cut_bar - train_bars, cut_bar)
        eval_t = tensor.slice_time(cut_bar, cut_bar + eval_bars)
        eval_slices = _daily_reset_slices(
            eval_t.timestamps_ms, bpd, base_config.reset_bar_offset
        )

        for tier in tiers:
            cfg_t = PodConstructorConfig.for_sizing_tier(tier, base=base_config)
            sized_list, _, _, _ = discover_diverse_pods(train_t, cfg_t)
            top = sized_list[0]
            pod = top.pod_indices
            pod_list = list(pod)

            solo_oos, t_ov_oos, delta_ov_oos = _build_sparse_overlap_cache(
                eval_t.delta_notional_frac[pod_list],
                eval_t.net_ret_frac[pod_list],
            )
            unrl_oos = (
                _compute_netted_unrealized_frac_at_resets(
                    top.multipliers,
                    eval_t,
                    pod,
                    eval_slices,
                    cost_rate,
                    eps=cfg_t.flat_notional_eps_frac,
                )
                if cfg_t.daily_loss_mode == "reset_balance"
                else None
            )
            (
                oos_ret,
                oos_pdd,
                oos_sdd,
                _cdar,
                _dcvar,
                oos_mdaily,
                _sav,
                cum_r_oos,
            ) = _evaluate_weights_with_cum(
                top.multipliers,
                solo_oos,
                t_ov_oos,
                delta_ov_oos,
                cost_rate,
                eval_slices,
                cfg_t,
                unrl_at_resets=unrl_oos,
            )

            eval_active_syms = [
                s
                for s in range(len(eval_t.symbols))
                if eval_t.notional_frac[pod_list, s, :].any()
            ]
            if eval_active_syms:
                sym_oos = np.tensordot(
                    top.multipliers,
                    eval_t.notional_frac[np.ix_(pod_list, eval_active_syms)].astype(
                        np.float64, copy=False
                    ),
                    axes=([0], [0]),
                )
                abs_sym_oos = np.abs(sym_oos)
                oos_peak_sym = float(np.max(abs_sym_oos))
                oos_peak_gross = float(np.max(np.sum(abs_sym_oos, axis=0)))
            else:
                oos_peak_sym = 0.0
                oos_peak_gross = 0.0

            oos_status = _first_barrier_oos_status(cum_r_oos, eval_slices, unrl_oos, cfg_t)
            results.append(
                WalkForwardFoldResult(
                    fold_idx=fold_idx,
                    tier=tier,
                    train_start_ms=int(train_t.timestamps_ms[0]),
                    train_end_ms=int(train_t.timestamps_ms[-1]),
                    eval_start_ms=int(eval_t.timestamps_ms[0]),
                    eval_end_ms=int(eval_t.timestamps_ms[-1]),
                    aliases=top.aliases,
                    pod_indices=pod,
                    multipliers=top.multipliers,
                    train_feasible=top.feasible,
                    train_net_ret_frac=top.objective_net_return,
                    train_peak_to_trough_dd=top.peak_to_trough_dd,
                    train_static_floor_dd=top.static_floor_dd,
                    train_max_daily_loss=top.max_daily_loss,
                    oos_net_ret_frac=oos_ret,
                    oos_peak_to_trough_dd=oos_pdd,
                    oos_static_floor_dd=oos_sdd,
                    oos_max_daily_loss=oos_mdaily,
                    oos_peak_symbol_exposure_frac=oos_peak_sym,
                    oos_peak_gross_exposure_frac=oos_peak_gross,
                    oos_status=oos_status,
                    binding_constraint=top.binding_constraint,
                )
            )
        fold_idx += 1
        cut_bar += eval_bars
    return results


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
    blocks: tuple[np.ndarray, ...],
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

    Accepts either legacy 6-tuples ``(is_chop, day_end, day_max, day_min, day_swept,
    day_max_first)`` or Phase-1 balance-reset 7-tuples ``(..., day_min_vs_bal)``.
    """
    if len(blocks) == 7:
        is_chop, day_end, day_max, day_min, day_swept, day_max_first, day_min_vs_bal = blocks
    elif len(blocks) == 6:
        is_chop, day_end, day_max, day_min, day_swept, day_max_first = blocks
        day_min_vs_bal = day_min
    else:
        raise ValueError(f"UNEXPECTED_MC_BLOCKS_LEN:{len(blocks)}")
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
        d_min_bal = day_min_vs_bal[idx]
        d_swp, d_mf = day_swept[idx], day_max_first[idx]

        hit_pass = (eq_a + d_max) >= (account + profit_target)
        hit_static = (eq_a + d_min) <= (account - max_static_dd)
        hit_daily = d_min_bal <= -daily_loss_limit

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


@dataclass(frozen=True)
class PropPhaseSpec:
    """Single evaluation or funded phase specification for multi-phase contract EV."""

    name: str
    profit_target_usd: float | None
    max_static_dd_usd: float
    daily_loss_limit_usd: float
    daily_loss_basis: Literal["balance", "equity"] = "balance"
    min_trading_days: int = 0
    consistency_max_single_day_pct: float | None = None
    horizon_days: int = 90


@dataclass(frozen=True)
class PropContractSpec:
    """Complete multi-phase prop-firm contract state machine for Net Payout EV pricing."""

    firm_slug: str
    program_name: str
    account_usd: float
    eval_phases: tuple[PropPhaseSpec, ...]
    funded_phase: PropPhaseSpec
    payout_split_pct: float = 0.80
    challenge_fee_usd: float | None = None
    fee_refunded_on_first_payout: bool = False
    funded_cycle_days: int = 14
    funded_target_cap_usd: float | None = None


def _extract_leading_number(raw: object) -> float | None:
    """Return the leading numeric token of a published rule string (no invented defaults)."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if not isinstance(raw, str):
        return None
    text = re.sub(r"(?<=\d),(?=\d)", "", raw.strip())
    token = ""
    for ch in text:
        if ch.isdigit() or ch == ".":
            token += ch
        elif token:
            break
    if not token or token == ".":
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _catalog_int(raw: object) -> int:
    """Read a catalog integer that may be missing or explicitly ``UNKNOWN``/``AMBIGUOUS``."""
    if isinstance(raw, bool) or raw is None:
        return 0
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.strip().isdigit():
        return int(raw.strip())
    return 0


def _catalog_fraction(raw: object) -> float | None:
    """Convert a published percentage or split ratio into a 0-1 fraction."""
    value = _extract_leading_number(raw)
    if value is None or value <= 0.0:
        return None
    return value / 100.0 if value > 1.0 else value


def _daily_loss_basis(raw: object) -> Literal["balance", "equity"]:
    return "equity" if str(raw or "").strip().upper().startswith("EQUITY") else "balance"


def _is_eval_phase(spec: ChallengeRulebookSpec) -> bool:
    return spec.phase.startswith("phase_")


def load_prop_contract_spec(
    firm_slug: str,
    *,
    account_usd: float = 10_000.0,
    challenge_fee_usd: float | None = None,
    min_eval_trading_days: int | None = None,
    funded_cycle_days: int = 30,
    program_filter: str | None = None,
) -> PropContractSpec:
    """Build a multi-phase contract from the canonical ``copytrade_kernel`` rulebook catalog.

    The kernel catalog is the single source of truth for published challenge terms; this
    function only rescales those terms to ``account_usd``. It never invents a fee, profit
    target, drawdown or payout split: an unrouted or unpriced contract keeps
    ``challenge_fee_usd=None`` until a caller supplies a verified fee.
    """
    firm_specs = RulebookCatalog.list_specs(firm_slug=firm_slug)
    if not firm_specs:
        raise FileNotFoundError(f"PROP_FIRM_NOT_FOUND:{firm_slug}")

    programs: dict[str, list[ChallengeRulebookSpec]] = {}
    for firm_spec in firm_specs:
        programs.setdefault(firm_spec.program, []).append(firm_spec)

    hint = program_filter.strip().lower() if program_filter is not None else None
    if hint is not None:
        selected = next((name for name in programs if hint in name.lower()), None)
        if selected is None:
            raise ValueError(f"NO_MATCHING_PROP_PHASES:{firm_slug}:{program_filter}")
    else:
        selected = next(
            (name for name, items in programs.items() if any(_is_eval_phase(s) for s in items)),
            next(iter(programs)),
        )

    program_specs = programs[selected]
    eval_raw = sorted(
        (s for s in program_specs if _is_eval_phase(s)),
        key=lambda s: int(s.phase.rsplit("_", 1)[-1]) if s.phase.rsplit("_", 1)[-1].isdigit() else 0,
    )
    if not eval_raw:
        raise ValueError(f"NO_MATCHING_PROP_PHASES:{firm_slug}:{program_filter}")

    funded_raw = next((s for s in program_specs if s.phase == "funded"), None)
    if funded_raw is None:
        funded_raw = next((s for s in firm_specs if s.phase == "funded"), None)

    base = _extract_leading_number(eval_raw[0].initial)
    if base is None or base <= 0.0:
        raise ValueError(f"PROP_CONTRACT_INVALID_BASE:{firm_slug}:{selected}")
    scale = float(account_usd) / base

    def _phase_amounts(spec: ChallengeRulebookSpec) -> tuple[float, float]:
        drawdown = _extract_leading_number(spec.trailing_loss)
        daily = _extract_leading_number(spec.daily_loss)
        if drawdown is None or daily is None:
            raise ValueError(f"PROP_CONTRACT_INCOMPLETE:{spec.key}")
        return drawdown * scale, daily * scale

    eval_specs: list[PropPhaseSpec] = []
    for spec in eval_raw:
        target = _extract_leading_number(spec.target)
        if target is None:
            raise ValueError(f"PROP_CONTRACT_INCOMPLETE:{spec.key}")
        drawdown_usd, daily_usd = _phase_amounts(spec)
        eval_specs.append(
            PropPhaseSpec(
                name=spec.phase,
                profit_target_usd=target * scale,
                max_static_dd_usd=drawdown_usd,
                daily_loss_limit_usd=daily_usd,
                daily_loss_basis=_daily_loss_basis(spec.daily_basis),
                min_trading_days=(
                    int(min_eval_trading_days)
                    if min_eval_trading_days is not None
                    else _catalog_int(spec.min_trading_days)
                ),
                consistency_max_single_day_pct=_catalog_fraction(
                    spec.consistency_max_single_day_pct
                ),
                horizon_days=90,
            )
        )

    if funded_raw is not None:
        drawdown_usd, daily_usd = _phase_amounts(funded_raw)
        funded_mtd = _catalog_int(funded_raw.min_trading_days)
        funded_days = max(int(funded_cycle_days), funded_mtd)
        funded_spec = PropPhaseSpec(
            name="funded",
            profit_target_usd=None,
            max_static_dd_usd=drawdown_usd,
            daily_loss_limit_usd=daily_usd,
            daily_loss_basis=_daily_loss_basis(funded_raw.daily_basis),
            min_trading_days=funded_mtd,
            consistency_max_single_day_pct=_catalog_fraction(
                funded_raw.consistency_max_single_day_pct
            ),
            horizon_days=funded_days,
        )
        payout_split = _catalog_fraction(funded_raw.payout_split_pct)
    else:
        ref = eval_specs[-1]
        funded_days = int(funded_cycle_days)
        funded_spec = PropPhaseSpec(
            name="funded",
            profit_target_usd=None,
            max_static_dd_usd=ref.max_static_dd_usd,
            daily_loss_limit_usd=ref.daily_loss_limit_usd,
            daily_loss_basis=ref.daily_loss_basis,
            min_trading_days=0,
            consistency_max_single_day_pct=None,
            horizon_days=funded_days,
        )
        payout_split = None

    return PropContractSpec(
        firm_slug=firm_slug,
        program_name=program_filter if program_filter is not None else selected,
        account_usd=float(account_usd),
        eval_phases=tuple(eval_specs),
        funded_phase=funded_spec,
        payout_split_pct=payout_split if payout_split is not None else 0.80,
        challenge_fee_usd=challenge_fee_usd,
        funded_cycle_days=funded_days,
        fee_refunded_on_first_payout=False,
        funded_target_cap_usd=None,
    )


def _wilson_ci95(p_hat: float, n_eff: float) -> tuple[float, float]:
    z = 1.959963984540054
    z2 = z * z
    denom = 1.0 + z2 / max(n_eff, 1.0)
    center = (p_hat + z2 / (2.0 * max(n_eff, 1.0))) / denom
    half = (
        z
        * math.sqrt(
            max(0.0, (p_hat * (1.0 - p_hat) + z2 / (4.0 * max(n_eff, 1.0))) / max(n_eff, 1.0))
        )
    ) / denom
    return max(0.0, center - half), min(1.0, center + half)


def evaluate_multiphase_contract_ev(
    blocks: tuple[np.ndarray, ...],
    contract: PropContractSpec,
    regime: Literal["Blended", "75%-Chop"] = "Blended",
    *,
    n_paths: int = 5_000,
    mean_block_days: float = 7.0,
    rng: np.random.Generator | None = None,
) -> dict[str, object]:
    """Simulate Fee -> Eval Phase(s) -> Funded Payout state machine with N_eff 95% CIs."""
    if len(blocks) == 7:
        is_chop, day_end, day_max, day_min_eq, _day_swept, day_max_first, day_min_bal = blocks
    elif len(blocks) == 6:
        is_chop, day_end, day_max, day_min_eq, _day_swept, day_max_first = blocks
        day_min_bal = day_min_eq
    else:
        raise ValueError(f"UNEXPECTED_MC_BLOCKS_LEN:{len(blocks)}")

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

    total_horizon = sum(p.horizon_days for p in contract.eval_phases) + contract.funded_cycle_days
    sample_idx = sample_stationary_block_indices(
        n_days=n_days,
        n_paths=n_paths,
        horizon_days=max(1, total_horizon),
        mean_block_days=mean_block_days,
        day_jump_probs=probs,
        rng=rng,
    )

    cursor = np.zeros(n_paths, dtype=np.int32)
    alive = np.ones(n_paths, dtype=bool)
    phase_pass_rates: dict[str, float] = {}

    for ph in contract.eval_phases:
        gain = np.zeros(n_paths, dtype=np.float64)
        credited_gain = np.zeros(n_paths, dtype=np.float64)
        active_days = np.zeros(n_paths, dtype=np.int32)
        ph_alive = alive.copy()
        passed = np.zeros(n_paths, dtype=bool)
        daily_arr = day_min_bal if ph.daily_loss_basis == "balance" else day_min_eq
        c_frac = (
            (
                ph.consistency_max_single_day_pct / 100.0
                if ph.consistency_max_single_day_pct > 1.0
                else ph.consistency_max_single_day_pct
            )
            if ph.consistency_max_single_day_pct is not None
            and ph.consistency_max_single_day_pct > 0.0
            else None
        )
        base_target = float(ph.profit_target_usd or 0.0)

        for _ in range(ph.horizon_days):
            act_idx = np.flatnonzero(ph_alive)
            if act_idx.size == 0:
                break
            d_i = sample_idx[act_idx, cursor[act_idx]]
            cursor[act_idx] += 1

            d_end = day_end[d_i]
            d_max = day_max[d_i]
            d_min = day_min_eq[d_i]
            d_dly = daily_arr[d_i]
            d_mf = day_max_first[d_i]

            is_act_day = (
                (np.abs(d_end) > 1e-9) | (np.abs(d_max) > 1e-9) | (np.abs(d_min) > 1e-9)
            )
            active_days[act_idx] += is_act_day.astype(np.int32)

            if c_frac is not None and base_target > 0.0:
                day_credit = np.where(d_end > 0.0, np.minimum(d_end, c_frac * base_target), d_end)
                hit_pass = ((credited_gain[act_idx] + day_credit) >= base_target) & (
                    active_days[act_idx] >= ph.min_trading_days
                )
            else:
                day_credit = d_end
                hit_pass = ((gain[act_idx] + d_max) >= base_target) & (
                    active_days[act_idx] >= ph.min_trading_days
                )

            hit_fail = ((gain[act_idx] + d_min) <= -ph.max_static_dd_usd) | (
                d_dly <= -ph.daily_loss_limit_usd
            )
            both_pf = hit_pass & hit_fail
            win_mask = (hit_pass & ~hit_fail) | (both_pf & d_mf)
            fail_mask = hit_fail & ~win_mask

            passed[act_idx[win_mask]] = True
            ph_alive[act_idx[win_mask | fail_mask]] = False
            cont = ~(win_mask | fail_mask)
            gain[act_idx[cont]] += d_end[cont]
            credited_gain[act_idx[cont]] += day_credit[cont]

        alive = passed
        phase_pass_rates[ph.name] = round(float(np.mean(alive)) * 100.0, 2)

    funded_gain = np.zeros(n_paths, dtype=np.float64)
    funded_days = np.zeros(n_paths, dtype=np.int32)
    f_alive = alive.copy()
    f_ph = contract.funded_phase
    f_daily_arr = day_min_bal if f_ph.daily_loss_basis == "balance" else day_min_eq
    for _ in range(contract.funded_cycle_days):
        act_idx = np.flatnonzero(f_alive)
        if act_idx.size == 0:
            break
        d_i = sample_idx[act_idx, cursor[act_idx]]
        cursor[act_idx] += 1

        d_end = day_end[d_i]
        d_max = day_max[d_i]
        d_min = day_min_eq[d_i]
        d_dly = f_daily_arr[d_i]
        is_act_day = (np.abs(d_end) > 1e-9) | (np.abs(d_max) > 1e-9) | (np.abs(d_min) > 1e-9)
        funded_days[act_idx] += is_act_day.astype(np.int32)

        hit_fail = ((funded_gain[act_idx] + d_min) <= -f_ph.max_static_dd_usd) | (
            d_dly <= -f_ph.daily_loss_limit_usd
        )
        f_alive[act_idx[hit_fail]] = False
        surv = ~hit_fail
        funded_gain[act_idx[surv]] += d_end[surv]

    if contract.funded_target_cap_usd is not None:
        capped_gain = np.minimum(np.maximum(0.0, funded_gain), contract.funded_target_cap_usd)
    else:
        capped_gain = np.maximum(0.0, funded_gain)

    paid_mask = f_alive & (capped_gain > 0.0) & (funded_days >= f_ph.min_trading_days)
    gross_no_refund = np.where(paid_mask, capped_gain * contract.payout_split_pct, 0.0)
    refund_bonus = (
        float(contract.challenge_fee_usd)
        if (contract.fee_refunded_on_first_payout and contract.challenge_fee_usd is not None)
        else 0.0
    )
    gross_payouts = np.where(paid_mask, gross_no_refund + refund_bonus, 0.0)

    p_eval_all = float(np.mean(alive))
    p_payout = float(np.mean(paid_mask))
    n_eff_blocks = max(2.0, n_days / max(1.0, mean_block_days))
    ci_lo, ci_hi = _wilson_ci95(p_payout, n_eff_blocks)

    exp_gross_no_refund = float(np.mean(gross_no_refund))
    exp_gross = float(np.mean(gross_payouts))
    cond_payout = float(np.mean(gross_payouts[paid_mask])) if np.any(paid_mask) else 0.0
    gross_ci_lo = ci_lo * cond_payout
    gross_ci_hi = ci_hi * cond_payout
    med_days_total = float(np.median(cursor[paid_mask])) if np.any(paid_mask) else -1.0

    breakeven_fee = (
        exp_gross_no_refund / max(1.0 - p_payout, 1e-9)
        if contract.fee_refunded_on_first_payout
        else exp_gross
    )

    if contract.challenge_fee_usd is not None:
        fee = float(contract.challenge_fee_usd)
        net_ev = exp_gross - fee
        net_ci_lo = gross_ci_lo - fee
        net_ci_hi = gross_ci_hi - fee
        ev_per_day = net_ev / med_days_total if med_days_total > 0.0 else None
        ev_status = "VERIFIED_FEE_NET_EV"
    else:
        net_ev = None
        net_ci_lo = None
        net_ci_hi = None
        ev_per_day = None
        ev_status = "BREAK_EVEN_ONLY_FEE_UNVERIFIED"

    return {
        "firm_slug": contract.firm_slug,
        "account_usd": contract.account_usd,
        "regime": regime,
        "n_paths": int(n_paths),
        "n_source_days": int(n_days),
        "n_eff_blocks": round(n_eff_blocks, 2),
        "phase_pass_rates_pct": phase_pass_rates,
        "p_pass_all_eval_pct": round(p_eval_all * 100.0, 2),
        "p_funded_payout_pct": round(p_payout * 100.0, 2),
        "p_funded_payout_ci95_pct": [round(ci_lo * 100.0, 2), round(ci_hi * 100.0, 2)],
        "expected_gross_payout_usd": round(exp_gross, 2),
        "breakeven_fee_usd": round(breakeven_fee, 2),
        "challenge_fee_usd": contract.challenge_fee_usd,
        "net_ev_usd": round(net_ev, 2) if net_ev is not None else None,
        "net_ev_ci95_usd": (
            [round(net_ci_lo, 2), round(net_ci_hi, 2)]
            if net_ci_lo is not None and net_ci_hi is not None
            else None
        ),
        "med_days_to_payout": round(med_days_total, 1),
        "ev_per_day_usd": round(ev_per_day, 2) if ev_per_day is not None else None,
        "ev_status": ev_status,
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
    ``KILL_CONTAIN_0X``) with an explicit terminal latch across restarts until a new
    mandate generation resets the monitor.
    """
    if not baseline.qualified:
        state.status = "KILL_CONTAIN_0X"
        state.reasons = ("UNQUALIFIED_EPISODE_BASELINE",)
        if config.latch_kill_state and not state.terminal_kill_latched:
            state.terminal_kill_latched = True
            state.latched_at_episode = state.episodes_observed
            state.latched_reasons = state.reasons
        return state
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
        if config.latch_kill_state and not state.terminal_kill_latched:
            state.terminal_kill_latched = True
            state.latched_at_episode = state.episodes_observed
            state.latched_reasons = tuple(reasons)
        return state

    if config.latch_kill_state and state.terminal_kill_latched:
        state.status = "KILL_CONTAIN_0X"
        state.reasons = state.latched_reasons or ("LATCHED_KILL_RETIREMENT",)
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
