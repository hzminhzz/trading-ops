"""Multi-seat dual-clock containment, de-hedging re-netting, PM03 attribution, and replacement."""

from __future__ import annotations

import math
import sys
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal as D
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from copytrade_pm.pm03_account import ZERO, Account, Instrument, Rulebook
from copytrade_pm.pm03_attribution import IntendedVirtualBooks, explain_economics

try:
    import canonical_replay as cr
    import pod_constructor as pc
except ModuleNotFoundError:
    _STUDIES_DIR = str(Path(__file__).resolve().parent)
    if _STUDIES_DIR not in sys.path:
        sys.path.insert(0, _STUDIES_DIR)
    import canonical_replay as cr
    import pod_constructor as pc

QUANT_12 = D("0.000000000001")
TIER_DEFAULT_DD_CEILING_USD: dict[int, float] = {
    2: 300.0,
    3: 500.0,
    4: 650.0,
}


class SeatState(StrEnum):
    """Per-seat containment state with irreversible terminal ``RETIRED`` latch."""

    ACTIVE = "ACTIVE"
    FREEZE_NEW_RISK = "FREEZE_NEW_RISK"
    RETIRED = "RETIRED"


class TriggerEvalStatus(StrEnum):
    """Explicit tri-state evaluation status so missing telemetry stays ``UNKNOWN``."""

    OK = "OK"
    TRIGGERED = "TRIGGERED"
    UNKNOWN = "UNKNOWN"


class LifecycleArm(StrEnum):
    """Four pre-registered Phase 2 containment and release policy arms."""

    EPISODE_CUSUM_ONLY = "EPISODE_CUSUM_ONLY"
    GUARDIAN_MANDATE_PAUSE_STATIC_HOLD = "GUARDIAN_MANDATE_PAUSE_STATIC_HOLD"
    RESEARCH_LEADER_REDUCE_FREEZE = "RESEARCH_LEADER_REDUCE_FREEZE"
    NEXT_FLAT_RELEASE = "NEXT_FLAT_RELEASE"


class ReplacementMode(StrEnum):
    """Post-retirement seat risk replacement policies."""

    LEAVE_RISK_UNUSED = "LEAVE_RISK_UNUSED"
    RESIZE_HEALTHY_SEATS = "RESIZE_HEALTHY_SEATS"
    PROMOTE_PIT_TOP_BENCH_NEW_POS_ONLY = "PROMOTE_PIT_TOP_BENCH_NEW_POS_ONLY"


def _q12(value: float | D) -> D:
    dec = value if isinstance(value, D) else D(f"{float(value):.12f}")
    out = dec.quantize(QUANT_12)
    return ZERO if out == ZERO else out


def _qpx(value: float | D) -> D:
    if isinstance(value, D):
        dec = value
    else:
        f_val = float(value)
        if not math.isfinite(f_val) or f_val <= 0.0:
            raise ValueError("NONPOSITIVE_DECIMAL_PRICE")
        dec = D(str(f_val))
    if not dec.is_finite() or dec <= ZERO:
        raise ValueError("NONPOSITIVE_DECIMAL_PRICE")
    if dec >= QUANT_12:
        return dec.quantize(QUANT_12)
    return dec


@dataclass(frozen=True)
class SeatRiskSpec:
    """Fixed activation risk unit ``R_i`` and monitoring thresholds for one pod seat."""

    seat_id: str
    tier: Literal[1, 2, 3, 4] = 1
    k_scale: float = 0.0
    initial_balance_usd: float = 10_000.0
    tier_seat_count_m: int = 3
    s_i: float | None = None
    d_tier: float | None = None
    win_hold_p95_bars: float | None = None
    mae_p95_r: float | None = None
    episode_baseline: pc.TraderEpisodeBaseline | None = None
    cusum_drift_k: float = 0.50
    cusum_freeze_h: float = 2.50
    cusum_retire_h: float = 4.00
    max_k_scale: float | None = None
    activation_r_usd: float | None = None

    def __post_init__(self) -> None:
        if not self.seat_id or not isinstance(self.seat_id, str):
            raise ValueError("INVALID_SEAT_ID")
        if not math.isfinite(self.k_scale) or self.k_scale < 0.0:
            raise ValueError("INVALID_SEAT_RISK_UNIT")
        if not math.isfinite(self.initial_balance_usd) or self.initial_balance_usd <= 0.0:
            raise ValueError("INVALID_SEAT_RISK_UNIT")

        if self.activation_r_usd is not None:
            r_val = float(self.activation_r_usd)
        elif self.tier == 1:
            if self.tier_seat_count_m <= 0:
                raise ValueError("INVALID_SEAT_RISK_UNIT")
            r_val = (0.02 * float(self.initial_balance_usd)) / float(self.tier_seat_count_m)
        elif self.tier in (2, 3, 4):
            if self.s_i is None or not math.isfinite(self.s_i):
                raise ValueError("INVALID_SEAT_RISK_UNIT")
            eff_d = (
                float(self.d_tier)
                if self.d_tier is not None
                else TIER_DEFAULT_DD_CEILING_USD[int(self.tier)]
                * (float(self.initial_balance_usd) / 10_000.0)
            )
            if not math.isfinite(eff_d) or eff_d <= 0.0:
                raise ValueError("INVALID_SEAT_RISK_UNIT")
            r_val = float(self.s_i) * eff_d
        else:
            raise ValueError("INVALID_SEAT_RISK_UNIT")

        if not math.isfinite(r_val) or r_val <= 0.0:
            raise ValueError("INVALID_SEAT_RISK_UNIT")
        object.__setattr__(self, "activation_r_usd", r_val)

    @property
    def r_unit_usd(self) -> float:
        assert self.activation_r_usd is not None
        return float(self.activation_r_usd)


@dataclass
class SeatContainmentState:
    """Dual-clock containment state for a single seat across 5m bars and shadow episodes."""

    spec: SeatRiskSpec
    state: SeatState = SeatState.ACTIVE
    current_k_scale: float = field(init=False)
    pending_k_scale: float | None = None
    hwm_pnl_usd: float = 0.0
    cum_pnl_usd: float = 0.0
    open_upnl_usd: float = 0.0
    cft_day_idx: int | None = None
    cft_day_start_pnl_usd: float = 0.0
    pod_loss_used_usd: float = 0.0
    underwater_bars: int = 0
    underwater_status: TriggerEvalStatus = TriggerEvalStatus.OK
    source_av_status: TriggerEvalStatus = TriggerEvalStatus.UNKNOWN
    data_fence_active: bool = False
    data_recovery_clean_bars: int = 0
    cusum_pos: float = 0.0
    roster_cusum: pc.RosterCUSUMState | None = None
    shadow_clean_episodes_since_freeze: int = 0
    had_risk_freeze: bool = False
    last_freeze_bar: int = -1
    flat_since_bar: int = 0
    freeze_reasons: set[str] = field(default_factory=set)
    retire_reason: str | None = None
    transition_log: list[tuple[int, str, str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.current_k_scale = float(self.spec.k_scale)
        if self.spec.episode_baseline is not None and self.roster_cusum is None:
            self.roster_cusum = pc.RosterCUSUMState(alias=self.spec.seat_id)

    def transition_to(
        self,
        target: SeatState,
        reason: str,
        *,
        bar_ts_ms: int = 0,
        bar_idx: int = -1,
    ) -> None:
        """Apply state transition enforcing the irreversible ``RETIRED`` terminal latch."""
        if self.state == SeatState.RETIRED:
            if target == SeatState.RETIRED:
                return
            raise ValueError("ILLEGAL_RETIREMENT_RESURRECTION")
        if target == self.state:
            return
        prev = self.state
        self.state = target
        self.transition_log.append((int(bar_ts_ms), prev.value, target.value, str(reason)))
        if target == SeatState.FREEZE_NEW_RISK:
            self.shadow_clean_episodes_since_freeze = 0
            self.last_freeze_bar = int(bar_idx)
        elif target == SeatState.RETIRED:
            self.retire_reason = str(reason)
            self.current_k_scale = 0.0
            self.pending_k_scale = None
        elif target == SeatState.ACTIVE:
            self.freeze_reasons.clear()
            self.shadow_clean_episodes_since_freeze = 0
            self.had_risk_freeze = False


@dataclass(frozen=True)
class ClosedShadowEpisode:
    """Completed unit-copy (``k=1``) shadow episode observed on one ``(seat, coin)`` stream."""

    seat_id: str
    coin_idx: int
    open_bar: int
    close_bar: int
    duration_bars: int
    ep_net_pnl_usd: float
    peak_notional_usd: float
    ep_ropn: float
    ep_mae_usd: float
    ep_mae_ropn: float
    symbol_notional_vec: np.ndarray
    ep_equity_ret: float = 0.0
    ep_equity_mae: float = 0.0
    is_clean: bool = True


class ShadowUnitEpisodeTracker:
    """Causal ``k=1`` shadow episode extractor running independently of follower freezes."""

    def __init__(
        self,
        seat_id: str,
        n_coins: int,
        *,
        cost_bps: float = cr.DEFAULT_COST_BPS,
        pod_equity_usd: float = 10_000.0,
        eps: float = 1e-12,
    ) -> None:
        self.seat_id = seat_id
        self.n_coins = int(n_coins)
        self.cost_rate = float(cost_bps) * 1e-4
        self.pod_equity_usd = float(pod_equity_usd)
        self.eps = float(eps)

        self._pos = np.zeros(self.n_coins, dtype=np.float64)
        self._prev_mark = np.full(self.n_coins, np.nan, dtype=np.float64)
        self._open_bar = np.full(self.n_coins, -1, dtype=np.int64)
        self._dur = np.zeros(self.n_coins, dtype=np.int64)
        self._cum_pnl = np.zeros(self.n_coins, dtype=np.float64)
        self._min_pnl = np.zeros(self.n_coins, dtype=np.float64)
        self._peak_notional = np.zeros(self.n_coins, dtype=np.float64)
        self._initialized = np.zeros(self.n_coins, dtype=bool)
        self._anchor_src = np.zeros(self.n_coins, dtype=np.float64)
        self._ep_clean = np.ones(self.n_coins, dtype=bool)

    def step_bar(
        self,
        bar_idx: int,
        source_positions: np.ndarray,
        trade_open: np.ndarray,
        mark_close: np.ndarray,
        *,
        source_ages_ms: np.ndarray | None = None,
        mark_ages_ms: np.ndarray | None = None,
        source_stale_ms: int = 600_000,
        mark_stale_ms: int = 300_000,
    ) -> list[ClosedShadowEpisode]:
        """Advance one 5m bar across all coins and emit newly closed shadow episodes."""
        closed: list[ClosedShadowEpisode] = []
        eq_denom = max(self.pod_equity_usd, 1.0)

        for ci in range(self.n_coins):
            src_q = float(source_positions[ci])
            to_px = float(trade_open[ci])
            mc_px = float(mark_close[ci])
            s_age = int(source_ages_ms[ci]) if source_ages_ms is not None else 0
            m_age = int(mark_ages_ms[ci]) if mark_ages_ms is not None else 0
            bar_telem_ok = (
                math.isfinite(src_q)
                and 0 <= s_age <= source_stale_ms
                and math.isfinite(mc_px)
                and mc_px > 0.0
                and 0 <= m_age <= mark_stale_ms
            )

            if not math.isfinite(src_q):
                if abs(self._pos[ci]) > self.eps:
                    self._ep_clean[ci] = False
                continue

            if not self._initialized[ci]:
                self._initialized[ci] = True
                self._anchor_src[ci] = src_q
                if math.isfinite(mc_px) and mc_px > 0.0:
                    self._prev_mark[ci] = mc_px
                continue

            if abs(self._anchor_src[ci]) > self.eps:
                if abs(src_q) <= self.eps or cr._opposes(src_q, float(self._anchor_src[ci]), self.eps):
                    self._anchor_src[ci] = 0.0
                else:
                    if math.isfinite(mc_px) and mc_px > 0.0:
                        self._prev_mark[ci] = mc_px
                    continue

            if not (math.isfinite(mc_px) and mc_px > 0.0):
                if abs(self._pos[ci]) > self.eps:
                    self._ep_clean[ci] = False
                continue

            if not bar_telem_ok and abs(self._pos[ci]) > self.eps:
                self._ep_clean[ci] = False

            has_trade_open = math.isfinite(to_px) and to_px > 0.0
            exec_px = to_px if has_trade_open else mc_px
            old_q = float(self._pos[ci])
            prev_m = float(self._prev_mark[ci])

            if abs(old_q) > self.eps:
                if math.isfinite(prev_m):
                    self._cum_pnl[ci] += old_q * (exec_px - prev_m)
                closing_or_flipping = (abs(src_q) <= self.eps) or cr._opposes(
                    old_q, src_q, self.eps
                )
                if closing_or_flipping:
                    exit_cost = abs(old_q) * exec_px * self.cost_rate
                    ep_net = float(self._cum_pnl[ci] - exit_cost)
                    ep_min = min(float(self._min_pnl[ci]), ep_net)
                    mae_usd = max(0.0, -ep_min)
                    peak_not = max(float(self._peak_notional[ci]), abs(old_q) * exec_px)
                    sym_vec = np.zeros(self.n_coins, dtype=np.float64)
                    sym_vec[ci] = peak_not
                    denom = max(peak_not, 1e-8)
                    clean_ep = bool(self._ep_clean[ci] and bar_telem_ok and has_trade_open)
                    closed.append(
                        ClosedShadowEpisode(
                            seat_id=self.seat_id,
                            coin_idx=ci,
                            open_bar=int(self._open_bar[ci]),
                            close_bar=int(bar_idx),
                            duration_bars=max(1, int(self._dur[ci])),
                            ep_net_pnl_usd=ep_net,
                            peak_notional_usd=peak_not,
                            ep_ropn=ep_net / denom,
                            ep_mae_usd=mae_usd,
                            ep_mae_ropn=mae_usd / denom,
                            symbol_notional_vec=sym_vec,
                            ep_equity_ret=ep_net / eq_denom,
                            ep_equity_mae=mae_usd / eq_denom,
                            is_clean=clean_ep,
                        )
                    )
                    self._pos[ci] = 0.0
                    self._open_bar[ci] = -1
                    self._dur[ci] = 0
                    self._cum_pnl[ci] = 0.0
                    self._min_pnl[ci] = 0.0
                    self._peak_notional[ci] = 0.0
                    self._ep_clean[ci] = True
                    old_q = 0.0

            if abs(src_q) > self.eps:
                d_q = src_q - old_q
                if abs(d_q) > self.eps and not has_trade_open:
                    self._ep_clean[ci] = False
                trade_cost = abs(d_q) * exec_px * self.cost_rate if abs(d_q) > self.eps else 0.0
                if abs(old_q) <= self.eps:
                    self._open_bar[ci] = int(bar_idx)
                    self._dur[ci] = 1
                    self._cum_pnl[ci] = src_q * (mc_px - exec_px) - trade_cost
                    self._min_pnl[ci] = min(0.0, float(self._cum_pnl[ci]))
                    self._peak_notional[ci] = max(abs(src_q) * exec_px, abs(src_q) * mc_px)
                    self._ep_clean[ci] = bool(bar_telem_ok and has_trade_open)
                else:
                    self._dur[ci] += 1
                    self._cum_pnl[ci] += src_q * (mc_px - exec_px) - trade_cost
                    self._min_pnl[ci] = min(float(self._min_pnl[ci]), float(self._cum_pnl[ci]))
                    self._peak_notional[ci] = max(
                        float(self._peak_notional[ci]),
                        abs(src_q) * exec_px,
                        abs(src_q) * mc_px,
                    )
                self._pos[ci] = src_q
            self._prev_mark[ci] = mc_px

        return closed


def step_fast_seat_clock(
    seat_state: SeatContainmentState,
    *,
    bar_ts_ms: int,
    cum_pnl_usd: float,
    open_upnl_usd: float,
    has_open_pos: bool,
    arm: LifecycleArm = LifecycleArm.RESEARCH_LEADER_REDUCE_FREEZE,
    source_av_usd: float | None = None,
    source_av_age_s: float | None = None,
    source_age_s: float = 0.0,
    mark_age_s: float = 0.0,
    follower_is_flat: bool | None = None,
    leader_is_flat: bool = False,
    max_coin_underwater_bars: int | None = None,
    bar_idx: int = -1,
) -> SeatState:
    """Evaluate 5-minute fast containment rules, CFT 00:05Z pod loss, and flat release."""
    cft_day = cr._reset_day_index(int(bar_ts_ms), 0, 5)
    if seat_state.cft_day_idx != cft_day:
        seat_state.cft_day_idx = cft_day
        seat_state.cft_day_start_pnl_usd = float(seat_state.cum_pnl_usd)
        seat_state.pod_loss_used_usd = 0.0

    seat_state.cum_pnl_usd = float(cum_pnl_usd)
    seat_state.open_upnl_usd = float(open_upnl_usd)
    seat_state.hwm_pnl_usd = max(seat_state.hwm_pnl_usd, float(cum_pnl_usd))
    intraday_loss = max(0.0, seat_state.cft_day_start_pnl_usd - seat_state.cum_pnl_usd)
    seat_state.pod_loss_used_usd = max(seat_state.pod_loss_used_usd, intraday_loss)

    eff_flat = (not has_open_pos) if follower_is_flat is None else bool(follower_is_flat)
    if not eff_flat:
        seat_state.flat_since_bar = -1
    elif seat_state.flat_since_bar < 0:
        seat_state.flat_since_bar = max(0, int(bar_idx))

    if (
        source_av_usd is None
        or source_av_age_s is None
        or not math.isfinite(source_av_usd)
        or not math.isfinite(source_av_age_s)
        or source_av_age_s < 0.0
        or source_av_age_s > 600.0
    ):
        seat_state.source_av_status = TriggerEvalStatus.UNKNOWN
    elif source_av_usd < 1000.0:
        seat_state.source_av_status = TriggerEvalStatus.TRIGGERED
        seat_state.transition_to(
            SeatState.RETIRED,
            "SOURCE_AV_BELOW_1000",
            bar_ts_ms=bar_ts_ms,
            bar_idx=bar_idx,
        )
    else:
        seat_state.source_av_status = TriggerEvalStatus.OK

    r_unit = seat_state.spec.r_unit_usd
    if seat_state.hwm_pnl_usd - seat_state.cum_pnl_usd >= 1.5 * r_unit - 1e-12:
        seat_state.transition_to(
            SeatState.RETIRED,
            "SEAT_DRAWDOWN_1_5R",
            bar_ts_ms=bar_ts_ms,
            bar_idx=bar_idx,
        )

    if seat_state.state == SeatState.RETIRED:
        return SeatState.RETIRED

    stale_telemetry = (
        not math.isfinite(source_age_s)
        or not math.isfinite(mark_age_s)
        or source_age_s < 0.0
        or mark_age_s < 0.0
        or source_age_s > 600.0
        or mark_age_s > 300.0
    )
    if stale_telemetry:
        seat_state.data_fence_active = True
        seat_state.data_recovery_clean_bars = 0
        seat_state.freeze_reasons.add("DATA_HEALTH_FENCE")
        if seat_state.state == SeatState.ACTIVE:
            seat_state.transition_to(
                SeatState.FREEZE_NEW_RISK,
                "DATA_HEALTH_FENCE",
                bar_ts_ms=bar_ts_ms,
                bar_idx=bar_idx,
            )
    elif seat_state.data_fence_active:
        seat_state.data_recovery_clean_bars += 1
        if seat_state.data_recovery_clean_bars >= 2:
            seat_state.data_fence_active = False
            seat_state.freeze_reasons.discard("DATA_HEALTH_FENCE")

    next_frozen_bar = int(bar_idx) + 1 if bar_idx >= 0 else -1

    if arm != LifecycleArm.EPISODE_CUSUM_ONLY:
        if -seat_state.open_upnl_usd >= 1.0 * r_unit - 1e-12:
            seat_state.freeze_reasons.add("OPEN_UPNL_1R")
            seat_state.had_risk_freeze = True
            seat_state.shadow_clean_episodes_since_freeze = 0
            seat_state.last_freeze_bar = next_frozen_bar
            if seat_state.state == SeatState.ACTIVE:
                seat_state.transition_to(
                    SeatState.FREEZE_NEW_RISK,
                    "OPEN_UPNL_1R",
                    bar_ts_ms=bar_ts_ms,
                    bar_idx=next_frozen_bar,
                )
        else:
            seat_state.freeze_reasons.discard("OPEN_UPNL_1R")

        if max_coin_underwater_bars is not None:
            seat_state.underwater_bars = max(0, int(max_coin_underwater_bars))
            if seat_state.underwater_bars == 0:
                seat_state.freeze_reasons.discard("UNDERWATER_HOLD_TIMER")
        elif has_open_pos and seat_state.open_upnl_usd < -1e-12:
            seat_state.underwater_bars += 1
        else:
            seat_state.underwater_bars = 0
            seat_state.freeze_reasons.discard("UNDERWATER_HOLD_TIMER")

        wh_bars = seat_state.spec.win_hold_p95_bars
        if wh_bars is None or not math.isfinite(wh_bars) or wh_bars <= 0.0:
            seat_state.underwater_status = TriggerEvalStatus.UNKNOWN
        else:
            limit_bars = max(288, math.ceil(2.0 * float(wh_bars)))
            if seat_state.underwater_bars >= limit_bars:
                seat_state.underwater_status = TriggerEvalStatus.TRIGGERED
                seat_state.freeze_reasons.add("UNDERWATER_HOLD_TIMER")
                seat_state.had_risk_freeze = True
                seat_state.shadow_clean_episodes_since_freeze = 0
                seat_state.last_freeze_bar = next_frozen_bar
                if seat_state.state == SeatState.ACTIVE:
                    seat_state.transition_to(
                        SeatState.FREEZE_NEW_RISK,
                        "UNDERWATER_HOLD_TIMER",
                        bar_ts_ms=bar_ts_ms,
                        bar_idx=next_frozen_bar,
                    )
            else:
                seat_state.underwater_status = TriggerEvalStatus.OK

    if seat_state.state == SeatState.FREEZE_NEW_RISK:
        cusum_ok = seat_state.cusum_pos <= seat_state.spec.cusum_freeze_h
        if not seat_state.had_risk_freeze and not seat_state.freeze_reasons and cusum_ok:
            seat_state.transition_to(
                SeatState.ACTIVE,
                "DATA_FENCE_RECOVERED",
                bar_ts_ms=bar_ts_ms,
                bar_idx=bar_idx,
            )
            return seat_state.state
        if arm == LifecycleArm.GUARDIAN_MANDATE_PAUSE_STATIC_HOLD:
            return seat_state.state
        if (
            arm == LifecycleArm.NEXT_FLAT_RELEASE
            and eff_flat
            and leader_is_flat
            and not seat_state.freeze_reasons
            and cusum_ok
        ):
            seat_state.transition_to(
                SeatState.ACTIVE,
                "NEXT_FLAT_RELEASED",
                bar_ts_ms=bar_ts_ms,
                bar_idx=bar_idx,
            )
        elif (
            arm == LifecycleArm.RESEARCH_LEADER_REDUCE_FREEZE
            and eff_flat
            and seat_state.shadow_clean_episodes_since_freeze >= 1
            and not seat_state.freeze_reasons
            and cusum_ok
        ):
            seat_state.transition_to(
                SeatState.ACTIVE,
                "SHADOW_EPISODE_RELEASED",
                bar_ts_ms=bar_ts_ms,
                bar_idx=bar_idx,
            )

    return seat_state.state


def step_slow_episode_clock(
    seat_state: SeatContainmentState,
    *,
    arm: LifecycleArm = LifecycleArm.RESEARCH_LEADER_REDUCE_FREEZE,
    ep_return_z: float = 0.0,
    ep_mae_r: float = 0.0,
    ep_return: float | None = None,
    ep_peak_notional_frac: float = 0.10,
    ep_duration_bars: float = 12.0,
    ep_mae: float | None = None,
    ep_symbol_notional: np.ndarray | None = None,
    opened_post_freeze: bool = True,
    episode_is_clean: bool = True,
    follower_is_flat: bool = True,
    leader_is_flat: bool = True,
    bar_ts_ms: int = 0,
    bar_idx: int = -1,
) -> SeatState:
    """Evaluate closed unit-copy shadow episode CUSUM/MAE rules and release gates."""
    spec = seat_state.spec
    baseline_kill = False

    if spec.episode_baseline is not None:
        base = spec.episode_baseline
        if seat_state.roster_cusum is None:
            seat_state.roster_cusum = pc.RosterCUSUMState(alias=spec.seat_id)
        sym_vec = (
            np.asarray(ep_symbol_notional, dtype=np.float64)
            if ep_symbol_notional is not None
            else np.asarray(base.symbol_notional_dist, dtype=np.float64).copy()
        )
        ret_val = (
            float(ep_return)
            if ep_return is not None
            else float(base.mean_ep_return) + float(ep_return_z) * max(base.std_ep_return, 1e-5)
        )
        mae_val = float(ep_mae) if ep_mae is not None else float(ep_mae_r)
        pc_cfg = pc.PodConstructorConfig(
            cusum_slack_k=spec.cusum_drift_k,
            cusum_warn_h=spec.cusum_freeze_h,
            cusum_kill_h=spec.cusum_retire_h,
        )
        pc.update_roster_episode_cusum(
            seat_state.roster_cusum,
            base,
            ep_return=ret_val,
            ep_peak_notional_frac=ep_peak_notional_frac,
            ep_duration_bars=ep_duration_bars,
            ep_mae=mae_val,
            ep_symbol_notional=sym_vec,
            config=pc_cfg,
        )
        seat_state.cusum_pos = max(
            float(seat_state.roster_cusum.s_decay),
            float(seat_state.roster_cusum.s_tilt),
        )
        baseline_kill = (seat_state.roster_cusum.status == "KILL_CONTAIN_0X") or (
            not base.qualified
        )
    else:
        seat_state.cusum_pos = max(
            0.0,
            seat_state.cusum_pos + (-float(ep_return_z) - float(spec.cusum_drift_k)),
        )

    if seat_state.state == SeatState.RETIRED:
        return SeatState.RETIRED

    mae_kill = (
        spec.episode_baseline is None
        and spec.mae_p95_r is not None
        and float(ep_mae_r) > 2.5 * float(spec.mae_p95_r)
    ) or (
        spec.episode_baseline is not None
        and spec.mae_p95_r is not None
        and float(ep_mae if ep_mae is not None else ep_mae_r) > 2.5 * float(spec.mae_p95_r)
    )
    cusum_kill = seat_state.cusum_pos > spec.cusum_retire_h
    if cusum_kill or mae_kill or baseline_kill:
        seat_state.transition_to(
            SeatState.RETIRED,
            "EPISODE_CUSUM_OR_MAE_KILL",
            bar_ts_ms=bar_ts_ms,
            bar_idx=bar_idx,
        )
        return seat_state.state

    next_frozen_bar = int(bar_idx) + 1 if bar_idx >= 0 else -1

    if seat_state.cusum_pos > spec.cusum_freeze_h:
        seat_state.freeze_reasons.add("EPISODE_CUSUM_WARN")
        seat_state.had_risk_freeze = True
        seat_state.shadow_clean_episodes_since_freeze = 0
        seat_state.last_freeze_bar = next_frozen_bar
        if seat_state.state == SeatState.ACTIVE:
            seat_state.transition_to(
                SeatState.FREEZE_NEW_RISK,
                "EPISODE_CUSUM_WARN",
                bar_ts_ms=bar_ts_ms,
                bar_idx=next_frozen_bar,
            )
        return seat_state.state

    seat_state.freeze_reasons.discard("EPISODE_CUSUM_WARN")
    if seat_state.state != SeatState.FREEZE_NEW_RISK:
        return seat_state.state

    if opened_post_freeze and episode_is_clean and follower_is_flat:
        seat_state.shadow_clean_episodes_since_freeze += 1

    if arm == LifecycleArm.GUARDIAN_MANDATE_PAUSE_STATIC_HOLD:
        return seat_state.state
    if arm == LifecycleArm.EPISODE_CUSUM_ONLY and not seat_state.freeze_reasons:
        seat_state.transition_to(
            SeatState.ACTIVE,
            "EPISODE_CUSUM_RECOVERED",
            bar_ts_ms=bar_ts_ms,
            bar_idx=bar_idx,
        )
    elif (
        arm == LifecycleArm.RESEARCH_LEADER_REDUCE_FREEZE
        and follower_is_flat
        and seat_state.shadow_clean_episodes_since_freeze >= 1
        and not seat_state.freeze_reasons
    ):
        seat_state.transition_to(
            SeatState.ACTIVE,
            "SHADOW_EPISODE_RELEASED",
            bar_ts_ms=bar_ts_ms,
            bar_idx=bar_idx,
        )
    elif (
        arm == LifecycleArm.NEXT_FLAT_RELEASE
        and follower_is_flat
        and leader_is_flat
        and not seat_state.freeze_reasons
    ):
        seat_state.transition_to(
            SeatState.ACTIVE,
            "NEXT_FLAT_RELEASED",
            bar_ts_ms=bar_ts_ms,
            bar_idx=bar_idx,
        )

    return seat_state.state


@dataclass(frozen=True)
class RenettingGateConfig:
    """Pre-dispatch aggregate symbol and gross exposure ceiling parameters."""

    fixed_sym_cap_usd: float = 15_000.0
    sym_equity_mult: float = 1.5
    fixed_gross_cap_usd: float = 30_000.0
    gross_equity_mult: float = 3.0
    eps: float = 1e-12


@dataclass(frozen=True)
class RenettingGateResult:
    """Output of pre-dispatch de-hedging re-netting safety gate."""

    adjusted_seat_desired: dict[str, np.ndarray]
    net_desired_qty: np.ndarray
    symbol_cap_usd: float
    gross_cap_usd: float
    symbol_clamped_coins: tuple[int, ...]
    gross_clamped: bool
    trimmed_notional_usd: float


def _is_seat_locked_on_coin(
    sid: str,
    ci: int,
    locked_seat_mask: Mapping[str, np.ndarray] | Collection[str] | None,
) -> bool:
    if locked_seat_mask is None:
        return False
    if isinstance(locked_seat_mask, Mapping):
        if sid not in locked_seat_mask:
            return False
        return bool(locked_seat_mask[sid][ci])
    return sid in locked_seat_mask


def _trim_dominant_two_tier(
    adjusted: dict[str, np.ndarray],
    ci: int,
    dom_sign: float,
    excess_qty: float,
    price: float,
    locked_seat_mask: Mapping[str, np.ndarray] | Collection[str] | None,
    allow_locked_trim: bool,
    eps: float,
) -> float:
    unlocked_sids = [
        sid
        for sid, vec in adjusted.items()
        if cr._same_sign(float(vec[ci]), dom_sign, eps)
        and not _is_seat_locked_on_coin(sid, ci, locked_seat_mask)
    ]
    locked_sids = [
        sid
        for sid, vec in adjusted.items()
        if cr._same_sign(float(vec[ci]), dom_sign, eps)
        and _is_seat_locked_on_coin(sid, ci, locked_seat_mask)
    ]

    trimmed_usd = 0.0
    unlocked_sum = sum(abs(float(adjusted[sid][ci])) for sid in unlocked_sids)
    rem_excess = excess_qty

    if rem_excess > eps and unlocked_sum > eps:
        trim_u = min(rem_excess, unlocked_sum)
        scale_u = max(0.0, (unlocked_sum - trim_u) / unlocked_sum)
        for sid in unlocked_sids:
            val = float(adjusted[sid][ci])
            new_val = val * scale_u
            trimmed_usd += abs(val - new_val) * price
            adjusted[sid][ci] = new_val
        rem_excess -= trim_u

    if rem_excess > eps and allow_locked_trim and locked_sids:
        locked_sum = sum(abs(float(adjusted[sid][ci])) for sid in locked_sids)
        if locked_sum > eps:
            trim_l = min(rem_excess, locked_sum)
            scale_l = max(0.0, (locked_sum - trim_l) / locked_sum)
            for sid in locked_sids:
                val = float(adjusted[sid][ci])
                new_val = val * scale_l
                trimmed_usd += abs(val - new_val) * price
                adjusted[sid][ci] = new_val

    return trimmed_usd


def apply_renetting_safety_gate(
    seat_desired_qty: Mapping[str, np.ndarray],
    mark_prices: np.ndarray,
    *,
    account_equity_usd: float,
    held_by_seat: Mapping[str, np.ndarray] | None = None,
    locked_seat_mask: Mapping[str, np.ndarray] | Collection[str] | None = None,
    dehedging_active: np.ndarray | bool = True,
    cfg: RenettingGateConfig = RenettingGateConfig(),
) -> RenettingGateResult:
    """Clamp post-containment dominant-side seat targets to symbol and gross caps."""
    marks = np.asarray(mark_prices, dtype=np.float64)
    n_coins = int(marks.size)
    adjusted = {
        sid: np.asarray(vec, dtype=np.float64).copy()
        for sid, vec in sorted(seat_desired_qty.items())
    }
    eq_pos = max(0.0, float(account_equity_usd))
    sym_cap = min(float(cfg.fixed_sym_cap_usd), float(cfg.sym_equity_mult) * eq_pos)
    gross_cap = min(float(cfg.fixed_gross_cap_usd), float(cfg.gross_equity_mult) * eq_pos)

    if not adjusted or n_coins == 0:
        return RenettingGateResult(
            adjusted_seat_desired=adjusted,
            net_desired_qty=np.zeros(n_coins, dtype=np.float64),
            symbol_cap_usd=sym_cap,
            gross_cap_usd=gross_cap,
            symbol_clamped_coins=(),
            gross_clamped=False,
            trimmed_notional_usd=0.0,
        )

    dehedged_coin = np.zeros(n_coins, dtype=bool)
    for ci in range(n_coins):
        if isinstance(dehedging_active, np.ndarray):
            dehedged_coin[ci] = bool(dehedging_active[ci])
        else:
            dehedged_coin[ci] = bool(dehedging_active)
        if held_by_seat is not None:
            net_now = sum(float(vec[ci]) for vec in adjusted.values())
            if abs(net_now) <= cfg.eps:
                dehedged_coin[ci] = False
            else:
                d_sgn = 1.0 if net_now > 0.0 else -1.0
                old_off = sum(
                    abs(float(held_by_seat[sid][ci]))
                    for sid in adjusted
                    if sid in held_by_seat
                    and cr._opposes(float(held_by_seat[sid][ci]), d_sgn, cfg.eps)
                )
                new_off = sum(
                    abs(float(adjusted[sid][ci]))
                    for sid in adjusted
                    if cr._opposes(float(adjusted[sid][ci]), d_sgn, cfg.eps)
                )
                dehedged_coin[ci] = new_off < old_off - cfg.eps

    sym_clamped: list[int] = []
    trimmed_usd = 0.0

    for ci in range(n_coins):
        m_c = float(marks[ci])
        if not (math.isfinite(m_c) and m_c > 0.0):
            continue
        net_c = sum(float(vec[ci]) for vec in adjusted.values())
        net_usd = abs(net_c) * m_c
        if net_usd <= sym_cap + 1e-9 or abs(net_c) <= cfg.eps:
            continue
        dom_sign = 1.0 if net_c > 0.0 else -1.0
        excess_qty = abs(net_c) - (sym_cap / m_c)
        cut_usd = _trim_dominant_two_tier(
            adjusted,
            ci,
            dom_sign,
            excess_qty,
            m_c,
            locked_seat_mask,
            bool(dehedged_coin[ci]),
            cfg.eps,
        )
        if cut_usd > 0.0:
            trimmed_usd += cut_usd
            sym_clamped.append(ci)

    net_vec = np.zeros(n_coins, dtype=np.float64)
    gross_usd = 0.0
    for ci in range(n_coins):
        net_vec[ci] = sum(float(vec[ci]) for vec in adjusted.values())
        m_c = float(marks[ci])
        if math.isfinite(m_c) and m_c > 0.0:
            gross_usd += abs(net_vec[ci]) * m_c

    gross_clamped = False
    if gross_usd > gross_cap + 1e-9 and gross_usd > cfg.eps:
        g_scale = max(0.0, min(1.0, gross_cap / gross_usd))
        any_gross_cut = False
        for ci in range(n_coins):
            m_c = float(marks[ci])
            if not (math.isfinite(m_c) and m_c > 0.0):
                continue
            net_c = float(net_vec[ci])
            if abs(net_c) <= cfg.eps:
                continue
            dom_sign = 1.0 if net_c > 0.0 else -1.0
            excess_qty = abs(net_c) * (1.0 - g_scale)
            cut_usd = _trim_dominant_two_tier(
                adjusted,
                ci,
                dom_sign,
                excess_qty,
                m_c,
                locked_seat_mask,
                bool(dehedged_coin[ci]),
                cfg.eps,
            )
            if cut_usd > 0.0:
                trimmed_usd += cut_usd
                any_gross_cut = True
            net_vec[ci] = sum(float(vec[ci]) for vec in adjusted.values())
        gross_clamped = any_gross_cut

    return RenettingGateResult(
        adjusted_seat_desired=adjusted,
        net_desired_qty=net_vec,
        symbol_cap_usd=sym_cap,
        gross_cap_usd=gross_cap,
        symbol_clamped_coins=tuple(sym_clamped),
        gross_clamped=gross_clamped,
        trimmed_notional_usd=trimmed_usd,
    )


@dataclass
class CFTGuardianTracker:
    """CFT 00:05 UTC realized-balance reset, sweep, daily-loss, and static-floor tracker."""

    initial_balance_usd: float = 10_000.0
    reset_hour_utc: int = 0
    reset_minute_utc: int = 5
    daily_sweep_usd: float | None = 400.0
    daily_sweep_rate: float | None = 0.04
    daily_loss_usd: float = 500.0
    daily_loss_rate: float | None = 0.05
    max_static_dd_usd: float = 1_000.0
    profit_target_usd: float | None = 800.0

    current_day_idx: int | None = None
    daily_reset_balance_usd: float = field(init=False)
    swept_today: bool = False
    sweep_count: int = 0
    breached: bool = False
    breach_reason: str | None = None
    target_passed: bool = False

    def __post_init__(self) -> None:
        self.daily_reset_balance_usd = float(self.initial_balance_usd)

    def roll_day(self, bar_ts_ms: int, pre_bar_balance_usd: float) -> bool:
        """Roll the 00:05 UTC session anchor using pre-bar realized balance."""
        day_idx = cr._reset_day_index(int(bar_ts_ms), self.reset_hour_utc, self.reset_minute_utc)
        if self.current_day_idx != day_idx:
            self.current_day_idx = day_idx
            self.daily_reset_balance_usd = float(pre_bar_balance_usd)
            self.swept_today = False
            return True
        return False

    def evaluate_equity(self, equity_usd: float) -> bool:
        """Evaluate static/daily breaches, profit target, and protective loss sweep."""
        static_floor = float(self.initial_balance_usd) - float(self.max_static_dd_usd)
        daily_allow = (
            self.daily_reset_balance_usd * float(self.daily_loss_rate)
            if self.daily_loss_rate is not None
            else float(self.daily_loss_usd)
        )
        daily_floor = self.daily_reset_balance_usd - daily_allow

        if not self.breached and not self.target_passed:
            if float(equity_usd) <= static_floor + 1e-12:
                self.breached = True
                self.breach_reason = "STATIC_FLOOR_BREACH"
            elif float(equity_usd) <= daily_floor + 1e-12:
                self.breached = True
                self.breach_reason = "DAILY_LOSS_BREACH"
            elif self.profit_target_usd is not None and float(equity_usd) >= (
                float(self.initial_balance_usd) + float(self.profit_target_usd) - 1e-12
            ):
                self.target_passed = True

        if self.breached or self.target_passed or self.swept_today:
            return False

        if self.daily_sweep_rate is not None:
            sweep_allow: float | None = self.daily_reset_balance_usd * float(self.daily_sweep_rate)
        else:
            sweep_allow = self.daily_sweep_usd

        if sweep_allow is not None and (self.daily_reset_balance_usd - float(equity_usd)) >= (
            float(sweep_allow) - 1e-12
        ):
            self.swept_today = True
            self.sweep_count += 1
            return True
        return False

    def step(self, bar_ts_ms: int, *, balance_usd: float, equity_usd: float) -> bool:
        """Roll 00:05 UTC session if needed and evaluate barriers at ``bar_ts_ms``."""
        self.roll_day(bar_ts_ms, balance_usd)
        return self.evaluate_equity(equity_usd)


class ConsolidatedPodLedger:
    """Exact Decimal PM03 Account + IntendedVirtualBooks + CFT 00:05Z Guardian wrapper."""

    def __init__(
        self,
        coins: Sequence[str],
        seat_ids: Sequence[str],
        start_ts_ms: int,
        *,
        initial_balance_usd: float = 10_000.0,
        cwsa_cfg: cr.CWSAReplayConfig | None = None,
        guardian: CFTGuardianTracker | None = None,
    ) -> None:
        self.coins = tuple(str(c) for c in coins)
        self.cwsa_cfg = cwsa_cfg if cwsa_cfg is not None else cr.CWSAReplayConfig()
        inst_list: list[Instrument] = []
        seen_syms: set[str] = set()
        for raw_coin in self.coins:
            sym, _ = cr.resolve_scale_alias(raw_coin, self.cwsa_cfg.scale_aliases)
            if sym in seen_syms:
                raise ValueError(f"DUPLICATE_FOLLOWER_INSTRUMENT:{sym}")
            seen_syms.add(sym)
            inst_list.append(Instrument.parse(f"hyperliquid/mainnet:perp:default/{sym}"))
        self.instruments = tuple(inst_list)

        start_iso = datetime.fromtimestamp(float(start_ts_ms) / 1000.0, tz=UTC).isoformat()
        self.rulebook = Rulebook(
            program="CFT_CONSOLIDATED_POD_LEDGER",
            stage="phase_1",
            source="TRADING_OPS_PHASE2_POLICY",
            observed_at=start_iso,
            version="phase2-v1",
            timezone="UTC",
            initial_balance=_q12(initial_balance_usd),
            profit_target=None,
            daily_loss=None,
            static_loss=None,
            trailing_loss=None,
            permitted=self.instruments,
            confirmed=True,
            account_terms_confirmed=False,
        )
        self.account = Account.start(self.rulebook, start_iso)
        self.intent = IntendedVirtualBooks()
        self.execution_price_drag: D = ZERO
        self.max_conservation_err: float = 0.0
        self.internal_cross_count: int = 0
        self.external_fill_count: int = 0

        self._last_clean_mark: dict[Instrument, D] = {}
        self._seat_entries: dict[str, dict[Instrument, float]] = {
            str(sid): {} for sid in seat_ids
        }
        for sid in seat_ids:
            s_str = str(sid)
            self.account.pm_quantities.setdefault(s_str, {})
            self.account.pm_cash.setdefault(s_str, ZERO)
            self.account.pm_costs.setdefault(s_str, ZERO)
            self.account.pm_funding.setdefault(s_str, ZERO)
            self.intent.qty.setdefault(s_str, {})
            self.intent.cash.setdefault(s_str, ZERO)

        self.guardian = (
            guardian
            if guardian is not None
            else CFTGuardianTracker(initial_balance_usd=initial_balance_usd)
        )

    def ensure_seat(self, seat_id: str) -> None:
        """Register a newly admitted bench seat in PM03 virtual books with zero initial balances."""
        sid = str(seat_id)
        self._seat_entries.setdefault(sid, {})
        self.account.pm_quantities.setdefault(sid, {})
        self.account.pm_cash.setdefault(sid, ZERO)
        self.account.pm_costs.setdefault(sid, ZERO)
        self.account.pm_funding.setdefault(sid, ZERO)
        self.intent.qty.setdefault(sid, {})
        self.intent.cash.setdefault(sid, ZERO)

    @property
    def equity_usd(self) -> float:
        return float(self.account.equity)

    @property
    def balance_usd(self) -> float:
        return float(self.account.balance)

    def preview_equity_usd(self, candidate_prices: np.ndarray) -> float:
        """Preview consolidated mark-to-market equity at ``candidate_prices`` without mutation."""
        total = self.account.cash
        for ci, inst in enumerate(self.instruments):
            q_d = self.account.quantities.get(inst, ZERO)
            if q_d == ZERO:
                continue
            px_f = float(candidate_prices[ci])
            if math.isfinite(px_f) and px_f > 0.0:
                mk_d = _qpx(px_f)
            elif inst in self._last_clean_mark:
                mk_d = self._last_clean_mark[inst]
            elif inst in self.account.marks:
                mk_d = self.account.marks[inst]
            else:
                continue
            total += q_d * mk_d
        return float(total)

    def seat_held_qty(self, seat_id: str, coin_idx: int) -> float:
        inst = self.instruments[coin_idx]
        return float(self.account.pm_quantities.get(seat_id, {}).get(inst, ZERO))

    def seat_has_open_pos(self, seat_id: str, *, eps: float = 1e-12) -> bool:
        book = self.account.pm_quantities.get(seat_id, {})
        return any(abs(float(q)) > eps for q in book.values())

    def seat_cum_pnl_usd(self, seat_id: str) -> float:
        return float(self.account.pm_equity_pnl().get(seat_id, ZERO))

    def seat_coin_open_upnl_usd(
        self, seat_id: str, coin_idx: int, *, eps: float = 1e-12
    ) -> float:
        inst = self.instruments[coin_idx]
        q_f = float(self.account.pm_quantities.get(seat_id, {}).get(inst, ZERO))
        if abs(q_f) <= eps or inst not in self.account.marks:
            return 0.0
        m_f = float(self.account.marks[inst])
        e_f = self._seat_entries.get(seat_id, {}).get(inst, m_f)
        return q_f * (m_f - e_f)

    def seat_open_upnl_usd(self, seat_id: str, *, eps: float = 1e-12) -> float:
        return sum(
            self.seat_coin_open_upnl_usd(seat_id, ci, eps=eps)
            for ci in range(len(self.instruments))
        )

    def _record_seat_basis_delta(
        self,
        seat_id: str,
        inst: Instrument,
        old_qty: float,
        new_qty: float,
        fill_px: float,
    ) -> None:
        entries = self._seat_entries.setdefault(seat_id, {})
        cur_e = entries.get(inst, fill_px)
        _realized, next_e = cr._apply_fill_to_basis(old_qty, new_qty, cur_e, fill_px)
        if abs(new_qty) <= 1e-12:
            entries.pop(inst, None)
        else:
            entries[inst] = next_e

    def step_bar(
        self,
        bar_ts_ms: int,
        *,
        allocations: Mapping[int, cr.MarketStepAllocation],
        raw_intended_by_seat: Mapping[str, np.ndarray],
        decision_mid_prices: np.ndarray,
        close_mark_prices: np.ndarray,
        funding_rates_bar: np.ndarray | None = None,
    ) -> tuple[float, bool]:
        """Settle internal crosses, external fills, funding, marks, and verify conservation."""
        self.guardian.roll_day(bar_ts_ms, self.balance_usd)

        decision_iso = datetime.fromtimestamp(
            float(bar_ts_ms + 60_000) / 1000.0, tz=UTC
        ).isoformat()
        close_iso = datetime.fromtimestamp(float(bar_ts_ms + 240_000) / 1000.0, tz=UTC).isoformat()

        for sid in raw_intended_by_seat:
            self.ensure_seat(sid)

        decision_marks: dict[Instrument, D] = {}
        close_marks: dict[Instrument, D] = {}
        fresh_mark_insts: set[Instrument] = set()

        for ci, inst in enumerate(self.instruments):
            mid_f = float(decision_mid_prices[ci])
            mc_f = float(close_mark_prices[ci])
            has_mid = math.isfinite(mid_f) and mid_f > 0.0
            has_close = math.isfinite(mc_f) and mc_f > 0.0
            if has_mid or has_close:
                fresh_mark_insts.add(inst)

            if has_mid:
                d_mk = _qpx(mid_f)
                decision_marks[inst] = d_mk
                self._last_clean_mark[inst] = d_mk
            elif has_close:
                d_mk = _qpx(mc_f)
                decision_marks[inst] = d_mk
                self._last_clean_mark[inst] = d_mk
            elif inst in self._last_clean_mark:
                decision_marks[inst] = self._last_clean_mark[inst]

            if has_close:
                c_mk = _qpx(mc_f)
                close_marks[inst] = c_mk
                self._last_clean_mark[inst] = c_mk
            elif inst in decision_marks:
                close_marks[inst] = decision_marks[inst]

        if decision_marks:
            self.account.mark(decision_marks, decision_iso)

        pre_swept = self.guardian.evaluate_equity(self.equity_usd)

        intent_req_d: dict[str, dict[Instrument, D]] = {}
        for sid, raw_vec in sorted(raw_intended_by_seat.items()):
            book_d: dict[Instrument, D] = {}
            for ci, inst in enumerate(self.instruments):
                val = float(raw_vec[ci])
                if inst in decision_marks:
                    q_d = _q12(val)
                    if q_d != ZERO:
                        book_d[inst] = q_d
                else:
                    prev_i = self.intent.qty.get(sid, {}).get(inst, ZERO)
                    if prev_i != ZERO:
                        book_d[inst] = prev_i
            intent_req_d[sid] = book_d

        if decision_marks or not any(intent_req_d.values()):
            self.intent.mark_and_target(intent_req_d, decision_marks)

        for ci in sorted(allocations):
            inst = self.instruments[ci]
            if inst not in decision_marks:
                continue
            alloc = allocations[ci]
            mid_d = decision_marks[inst]

            pms = sorted(set(alloc.seat_post_qty) | set(self.account.pm_quantities))
            old_d = {
                pm: self.account.pm_quantities.get(pm, {}).get(inst, ZERO)
                for pm in pms
            }
            post_d = {
                pm: _q12(alloc.seat_post_qty.get(pm, float(old_d[pm])))
                for pm in pms
            }
            deltas_d = {pm: post_d[pm] - old_d[pm] for pm in pms}
            buys_d = {pm: q for pm, q in deltas_d.items() if q > ZERO}
            sells_d = {pm: -q for pm, q in deltas_d.items() if q < ZERO}

            cur_seat_f = {pm: float(old_d[pm]) for pm in pms}
            for buyer in sorted(buys_d):
                for seller in sorted(sells_d):
                    crossed = min(buys_d[buyer], sells_d[seller])
                    if crossed <= ZERO:
                        continue
                    self.account.internal_cross(inst, buyer, seller, crossed, mid_d)
                    self.internal_cross_count += 1
                    buys_d[buyer] -= crossed
                    sells_d[seller] -= crossed
                    c_f = float(crossed)
                    m_f = float(mid_d)
                    self._record_seat_basis_delta(
                        buyer, inst, cur_seat_f[buyer], cur_seat_f[buyer] + c_f, m_f
                    )
                    cur_seat_f[buyer] += c_f
                    self._record_seat_basis_delta(
                        seller, inst, cur_seat_f[seller], cur_seat_f[seller] - c_f, m_f
                    )
                    cur_seat_f[seller] -= c_f

            pending_d = {pm: q for pm, q in sorted(buys_d.items()) if q > ZERO}
            if not pending_d:
                pending_d = {pm: -q for pm, q in sorted(sells_d.items()) if q > ZERO}

            ext_delta_d = sum(pending_d.values(), ZERO)
            if ext_delta_d != ZERO:
                sign_d = D(1) if ext_delta_d > ZERO else D(-1)
                adverse_bps = (
                    D(str(self.cwsa_cfg.half_spread_bps))
                    + D(str(self.cwsa_cfg.slippage_bps))
                    + D(str(alloc.extra_haircut_bps))
                )
                fill_px_d = (mid_d * (D(1) + sign_d * adverse_bps / D(10000))).quantize(QUANT_12)
                fee_d = (
                    abs(ext_delta_d * fill_px_d) * D(str(self.cwsa_cfg.fee_bps)) / D(10000)
                ).quantize(QUANT_12)
                drag_d = max((fill_px_d - mid_d) * ext_delta_d, ZERO)
                self.account.simulate_fill(
                    inst,
                    ext_delta_d,
                    fill_px_d,
                    fee_d,
                    accepted=True,
                    allocated=pending_d,
                )
                self.external_fill_count += 1
                self.execution_price_drag += drag_d
                f_px_f = float(fill_px_d)
                for pm, q_alloc in pending_d.items():
                    nxt_f = cur_seat_f[pm] + float(q_alloc)
                    self._record_seat_basis_delta(pm, inst, cur_seat_f[pm], nxt_f, f_px_f)
                    cur_seat_f[pm] = nxt_f

        if funding_rates_bar is not None:
            for ci, inst in enumerate(self.instruments):
                r_f = float(funding_rates_bar[ci])
                if (
                    not math.isfinite(r_f)
                    or abs(r_f) <= 1e-15
                    or inst not in fresh_mark_insts
                    or inst not in decision_marks
                ):
                    continue
                mk_d = decision_marks[inst]
                r_d = D(str(r_f))
                fund_alloc: dict[str, D] = {}
                for pm in sorted(self.account.pm_quantities):
                    q_pm = self.account.pm_quantities[pm].get(inst, ZERO)
                    if q_pm != ZERO:
                        fund_alloc[pm] = (-q_pm * mk_d * r_d).quantize(QUANT_12)
                if fund_alloc:
                    tot_fund = sum(fund_alloc.values(), ZERO)
                    self.account.post_funding(tot_fund, fund_alloc)

        if close_marks:
            self.account.mark(close_marks, close_iso)
            self.intent.mark_and_target(intent_req_d, close_marks)

        self.account.verify()
        econ = explain_economics(self.account, self.intent, self.execution_price_drag)

        eq_err = abs(
            self.account.equity
            - (self.rulebook.initial_balance + sum(self.account.pm_equity_pnl().values(), ZERO))
        )
        cash_err = abs(
            (self.account.cash - self.rulebook.initial_balance)
            - sum(self.account.pm_cash.values(), ZERO)
        )
        cost_err = abs(self.account.costs - sum(self.account.pm_costs.values(), ZERO))
        fund_err = abs(self.account.funding - sum(self.account.pm_funding.values(), ZERO))
        econ_err = abs(D(econ["reconciliation_rounding_residual"]))
        bar_err = float(max(eq_err, cash_err, cost_err, fund_err, econ_err))
        self.max_conservation_err = max(self.max_conservation_err, bar_err)

        post_swept = self.guardian.evaluate_equity(self.equity_usd)
        return bar_err, (pre_swept or post_swept)

    def explain(self) -> dict[str, Any]:
        """Return exact 4-layer PM03 economic attribution dictionary."""
        return explain_economics(self.account, self.intent, self.execution_price_drag)


@dataclass(frozen=True)
class SeatTelemetryStream:
    """Causal per-seat leader positions and optional account-value series aligned to ``grid_ms``."""

    seat_id: str
    source_positions: np.ndarray
    source_marks: np.ndarray | None = None
    source_ages_ms: np.ndarray | None = None
    source_av_usd: np.ndarray | None = None
    source_av_ages_s: np.ndarray | None = None


@dataclass(frozen=True)
class PITBenchCandidate:
    """Point-in-time bench replacement candidate with pre-sized ``SeatRiskSpec``."""

    spec: SeatRiskSpec
    score: float


@dataclass(frozen=True)
class PITBenchSnapshot:
    """Causal PIT bench ranking snapshot known at ``asof_ts_ms``."""

    asof_ts_ms: int
    candidates: tuple[PITBenchCandidate, ...]
    mode: Literal["PIT_STRICT_ASOF", "RETROSPECTIVE_ASOF_MASKED"] = "PIT_STRICT_ASOF"


@dataclass(frozen=True)
class PortfolioPolicyConfig:
    """Configuration governing multi-seat containment, liquidity, re-netting, and Guardian."""

    initial_balance_usd: float = 10_000.0
    lifecycle_arm: LifecycleArm = LifecycleArm.RESEARCH_LEADER_REDUCE_FREEZE
    replacement_mode: ReplacementMode = ReplacementMode.LEAVE_RISK_UNUSED
    cwsa: cr.CWSAReplayConfig = field(default_factory=cr.CWSAReplayConfig)
    liquidity: cr.CausalLiquidityConfig = field(default_factory=cr.CausalLiquidityConfig)
    renetting: RenettingGateConfig = field(default_factory=RenettingGateConfig)
    daily_sweep_usd: float | None = 400.0
    daily_sweep_rate: float | None = 0.04
    daily_loss_usd: float = 500.0
    daily_loss_rate: float | None = 0.05
    max_static_dd_usd: float = 1_000.0
    profit_target_usd: float | None = 800.0
    stop_on_guardian_breach: bool = True


@dataclass(frozen=True)
class PortfolioPolicyResult:
    """Full trajectory, attribution, conservation, and audit result of multi-seat replay."""

    config: PortfolioPolicyConfig
    coins: tuple[str, ...]
    grid_ms: np.ndarray
    equity_curve: np.ndarray
    balance_curve: np.ndarray
    net_positions: np.ndarray
    seat_positions: dict[str, np.ndarray]
    seat_state_history: dict[str, tuple[str, ...]]
    seat_final_states: dict[str, SeatContainmentState]
    ledger: ConsolidatedPodLedger
    economics: dict[str, Any]
    max_pm03_conservation_err: float
    participation_audit: cr.RealizedParticipationAudit
    renetting_clamped_bars: int
    promoted_bench_seats: tuple[str, ...]
    resized_seat_events: tuple[tuple[int, str, float, float], ...]
    guardian_breached: bool
    guardian_breach_reason: str | None
    guardian_target_passed: bool
    guardian_sweeps: int


@dataclass(frozen=True)
class SeatReplacementComparison:
    """Matched comparison across the 3 seat-replacement policies."""

    results_by_mode: dict[ReplacementMode, PortfolioPolicyResult]
    summary_by_mode: dict[str, dict[str, Any]]


def _normalize_seat_streams(
    seat_streams: Mapping[str, SeatTelemetryStream | np.ndarray] | Sequence[SeatTelemetryStream],
    n_coins: int,
    n_bars: int,
) -> dict[str, SeatTelemetryStream]:
    out: dict[str, SeatTelemetryStream] = {}
    if isinstance(seat_streams, Mapping):
        for sid, item in seat_streams.items():
            if isinstance(item, SeatTelemetryStream):
                arr = np.asarray(item.source_positions, dtype=np.float64)
                if arr.shape != (n_coins, n_bars):
                    raise ValueError(f"INVALID_STREAM_SHAPE:{sid}:{arr.shape}")
                out[str(sid)] = item
            else:
                arr = np.asarray(item, dtype=np.float64)
                if arr.shape != (n_coins, n_bars):
                    raise ValueError(f"INVALID_STREAM_SHAPE:{sid}:{arr.shape}")
                out[str(sid)] = SeatTelemetryStream(seat_id=str(sid), source_positions=arr)
    else:
        for item in seat_streams:
            arr = np.asarray(item.source_positions, dtype=np.float64)
            if arr.shape != (n_coins, n_bars):
                raise ValueError(f"INVALID_STREAM_SHAPE:{item.seat_id}:{arr.shape}")
            out[str(item.seat_id)] = item
    return out


def _latch_stale_open_legs(
    active_sids: Sequence[str],
    streams: Mapping[str, SeatTelemetryStream],
    pair_states: Mapping[str, Sequence[cr.PairCWSAState]],
    stale_open_gate: dict[str, np.ndarray],
    stale_open_snap: dict[str, np.ndarray],
    bar_idx: int,
    n_coins: int,
    eps: float,
) -> None:
    for sid in active_sids:
        stm = streams[sid]
        for ci in range(n_coins):
            src_cur = float(stm.source_positions[ci, bar_idx])
            if math.isfinite(src_cur):
                pair_states[sid][ci].prev_source_qty = src_cur
                if abs(src_cur) > eps:
                    stale_open_gate[sid][ci] = True
                    stale_open_snap[sid][ci] = src_cur


def run_portfolio_policy_replay(
    *,
    seat_specs: Sequence[SeatRiskSpec],
    seat_streams: Mapping[str, SeatTelemetryStream | np.ndarray] | Sequence[SeatTelemetryStream],
    coins: Sequence[str],
    trade_open: np.ndarray,
    mark_close: np.ndarray,
    grid_ms: np.ndarray,
    vol_1m_usd: np.ndarray | None = None,
    mark_ages_ms: np.ndarray | None = None,
    allowed_markets_mask: np.ndarray | None = None,
    funding_rates: np.ndarray | None = None,
    bench_snapshots: Sequence[PITBenchSnapshot] = (),
    policy_cfg: PortfolioPolicyConfig = PortfolioPolicyConfig(),
) -> PortfolioPolicyResult:
    """Run multi-seat CWSA, dual-clock containment, re-netting, L3 liquidity, and PM03 ledger."""
    coin_tuple = tuple(str(c) for c in coins)
    n_coins = len(coin_tuple)
    grid_arr = np.asarray(grid_ms, dtype=np.int64)
    n_bars = int(grid_arr.size)
    if n_bars == 0:
        raise ValueError("EMPTY_POLICY_GRID")

    t_open = np.asarray(trade_open, dtype=np.float64)
    m_close = np.asarray(mark_close, dtype=np.float64)
    if t_open.shape != (n_coins, n_bars) or m_close.shape != (n_coins, n_bars):
        raise ValueError("PRICE_SHAPE_MISMATCH")

    streams = _normalize_seat_streams(seat_streams, n_coins, n_bars)
    vol_mat = (
        np.asarray(vol_1m_usd, dtype=np.float64)
        if vol_1m_usd is not None
        else np.full((n_coins, n_bars), np.nan, dtype=np.float64)
    )
    mk_ages = (
        np.asarray(mark_ages_ms, dtype=np.int64)
        if mark_ages_ms is not None
        else np.zeros((n_coins, n_bars), dtype=np.int64)
    )
    if allowed_markets_mask is not None:
        allow_arr = np.asarray(allowed_markets_mask, dtype=bool)
        allow_2d = (
            np.broadcast_to(allow_arr[:, None], (n_coins, n_bars))
            if allow_arr.ndim == 1
            else allow_arr
        )
    else:
        allow_2d = np.ones((n_coins, n_bars), dtype=bool)

    fund_mat = np.asarray(funding_rates, dtype=np.float64) if funding_rates is not None else None
    cwsa_cfg = policy_cfg.cwsa
    liq_cfg = policy_cfg.liquidity
    eps = cwsa_cfg.qty_eps
    spread_frac = cwsa_cfg.half_spread_bps * 1e-4
    qty_mults = np.array(
        [cr.resolve_scale_alias(c, cwsa_cfg.scale_aliases)[1] for c in coin_tuple],
        dtype=np.float64,
    )

    admitted_states: dict[str, SeatContainmentState] = {}
    pair_states: dict[str, list[cr.PairCWSAState]] = {}
    seat_coin_k: dict[str, np.ndarray] = {}
    seat_coin_underwater: dict[str, np.ndarray] = {}
    stale_open_gate: dict[str, np.ndarray] = {}
    stale_open_snap: dict[str, np.ndarray] = {}
    shadow_trackers: dict[str, ShadowUnitEpisodeTracker] = {}
    ever_admitted: set[str] = set()

    for spec in seat_specs:
        if spec.seat_id not in streams:
            raise ValueError(f"MISSING_SEAT_STREAM:{spec.seat_id}")
        sid = spec.seat_id
        admitted_states[sid] = SeatContainmentState(spec=spec)
        pair_states[sid] = [cr.PairCWSAState() for _ in range(n_coins)]
        seat_coin_k[sid] = np.full(n_coins, float(spec.k_scale), dtype=np.float64)
        seat_coin_underwater[sid] = np.zeros(n_coins, dtype=np.int64)
        stale_open_gate[sid] = np.zeros(n_coins, dtype=bool)
        stale_open_snap[sid] = np.zeros(n_coins, dtype=np.float64)
        shadow_trackers[sid] = ShadowUnitEpisodeTracker(
            sid,
            n_coins,
            cost_bps=cwsa_cfg.fee_bps + cwsa_cfg.slippage_bps,
            pod_equity_usd=policy_cfg.initial_balance_usd,
            eps=eps,
        )
        ever_admitted.add(sid)

    initial_target_r_sum = sum(s.spec.r_unit_usd for s in admitted_states.values())
    handled_retired_seats: set[str] = set()
    promoted_bench: list[str] = []
    resized_events: list[tuple[int, str, float, float]] = []

    guardian = CFTGuardianTracker(
        initial_balance_usd=policy_cfg.initial_balance_usd,
        daily_sweep_usd=policy_cfg.daily_sweep_usd,
        daily_sweep_rate=policy_cfg.daily_sweep_rate,
        daily_loss_usd=policy_cfg.daily_loss_usd,
        daily_loss_rate=policy_cfg.daily_loss_rate,
        max_static_dd_usd=policy_cfg.max_static_dd_usd,
        profit_target_usd=policy_cfg.profit_target_usd,
    )
    ledger = ConsolidatedPodLedger(
        coin_tuple,
        tuple(admitted_states),
        int(grid_arr[0]),
        initial_balance_usd=policy_cfg.initial_balance_usd,
        cwsa_cfg=cwsa_cfg,
        guardian=guardian,
    )

    eq_curve = np.zeros(n_bars, dtype=np.float64)
    bal_curve = np.zeros(n_bars, dtype=np.float64)
    net_pos_hist = np.zeros((n_coins, n_bars), dtype=np.float64)
    seat_pos_hist: dict[str, np.ndarray] = {
        sid: np.zeros((n_coins, n_bars), dtype=np.float64) for sid in admitted_states
    }
    state_hist: dict[str, list[str]] = {sid: [] for sid in admitted_states}

    renetting_clamped_bars = 0
    trade_ratios: list[float] = []
    trade_count = 0
    missing_vol_trades = 0
    total_ext_notional = 0.0
    total_missed_notional = 0.0
    clipped_cnt = 0
    missed_cnt = 0
    haircut_cnt = 0

    sorted_bench = sorted(bench_snapshots, key=lambda s: s.asof_ts_ms)

    for j in range(n_bars):
        bar_ts_ms = int(grid_arr[j])
        prior_start = max(0, j - liq_cfg.l2_lookback_bars)
        bar_caps, bar_vol_valid = cr.estimate_causal_bar_capacity(
            vol_mat[:, prior_start:j], liq_cfg
        )

        active_sids = sorted(admitted_states)
        coin_executable = np.ones(n_coins, dtype=bool)
        for ci in range(n_coins):
            to_px = float(t_open[ci, j])
            mc_px = float(m_close[ci, j])
            mk_age_ms = int(mk_ages[ci, j])
            if not (
                math.isfinite(to_px)
                and to_px > 0.0
                and math.isfinite(mc_px)
                and mc_px > 0.0
                and 0 <= mk_age_ms <= cwsa_cfg.mark_stale_ms
            ):
                coin_executable[ci] = False

        guardian.roll_day(bar_ts_ms, ledger.balance_usd)
        decision_px_bar = np.where(coin_executable, t_open[:, j], np.nan)
        eq_open_preview = ledger.preview_equity_usd(decision_px_bar)
        if guardian.evaluate_equity(eq_open_preview):
            _latch_stale_open_legs(
                active_sids,
                streams,
                pair_states,
                stale_open_gate,
                stale_open_snap,
                j,
                n_coins,
                eps,
            )

        seat_src_age_s: dict[str, float] = {}
        seat_mk_age_s: dict[str, float] = {}

        for sid in active_sids:
            st_seat = admitted_states[sid]
            stm = streams[sid]

            relevant_src_ages: list[float] = []
            relevant_mk_ages: list[float] = []
            for ci in range(n_coins):
                if not allow_2d[ci, j]:
                    continue
                held_c = abs(ledger.seat_held_qty(sid, ci)) > eps
                src_c = float(stm.source_positions[ci, j])
                src_active = (not math.isfinite(src_c)) or abs(src_c) > eps or (
                    abs(pair_states[sid][ci].prev_source_qty) > eps
                )
                if held_c or src_active:
                    if stm.source_ages_ms is not None:
                        s_ms = int(stm.source_ages_ms[ci, j])
                        relevant_src_ages.append(-1.0 if s_ms < 0 else float(s_ms) / 1000.0)
                    m_ms = int(mk_ages[ci, j])
                    relevant_mk_ages.append(-1.0 if m_ms < 0 else float(m_ms) / 1000.0)

            src_age_s = (
                -1.0
                if any(a < 0.0 for a in relevant_src_ages)
                else (max(relevant_src_ages) if relevant_src_ages else 0.0)
            )
            mk_age_s = (
                -1.0
                if any(a < 0.0 for a in relevant_mk_ages)
                else (max(relevant_mk_ages) if relevant_mk_ages else 0.0)
            )
            seat_src_age_s[sid] = src_age_s
            seat_mk_age_s[sid] = mk_age_s

            if stm.source_av_usd is not None and stm.source_av_ages_s is not None:
                av_val = float(stm.source_av_usd[j])
                av_age = float(stm.source_av_ages_s[j])
                if (
                    math.isfinite(av_val)
                    and math.isfinite(av_age)
                    and 0.0 <= av_age <= 600.0
                    and av_val < 1000.0
                ):
                    st_seat.source_av_status = TriggerEvalStatus.TRIGGERED
                    st_seat.transition_to(
                        SeatState.RETIRED,
                        "SOURCE_AV_BELOW_1000",
                        bar_ts_ms=bar_ts_ms,
                        bar_idx=j,
                    )

            if st_seat.state != SeatState.RETIRED and (
                src_age_s < 0.0 or mk_age_s < 0.0 or src_age_s > 600.0 or mk_age_s > 300.0
            ):
                st_seat.data_fence_active = True
                st_seat.data_recovery_clean_bars = 0
                st_seat.freeze_reasons.add("DATA_HEALTH_FENCE")
                if st_seat.state == SeatState.ACTIVE:
                    st_seat.transition_to(
                        SeatState.FREEZE_NEW_RISK,
                        "DATA_HEALTH_FENCE",
                        bar_ts_ms=bar_ts_ms,
                        bar_idx=j,
                    )

        guardian_flatten = policy_cfg.stop_on_guardian_breach and guardian.breached
        sweep_lockout = guardian.swept_today
        raw_intended_by_seat: dict[str, np.ndarray] = {
            sid: np.zeros(n_coins, dtype=np.float64) for sid in active_sids
        }
        post_contain_desired: dict[str, np.ndarray] = {
            sid: np.zeros(n_coins, dtype=np.float64) for sid in active_sids
        }
        held_by_seat_bar: dict[str, np.ndarray] = {
            sid: np.zeros(n_coins, dtype=np.float64) for sid in active_sids
        }
        static_held_mask: dict[str, np.ndarray] = {
            sid: np.zeros(n_coins, dtype=bool) for sid in active_sids
        }
        seat_blocks_cz: dict[tuple[str, int], bool] = {}
        seat_init_disp: dict[tuple[str, int], cr.OrderDisposition] = {}

        for ci in range(n_coins):
            to_px = float(t_open[ci, j])
            mc_px = float(m_close[ci, j])
            mk_age_ms = int(mk_ages[ci, j])
            if coin_executable[ci]:
                q_buy = to_px * (1.0 + spread_frac)
                q_sell = to_px * max(1e-9, 1.0 - spread_frac)
            else:
                q_buy = float("nan")
                q_sell = float("nan")

            for sid in active_sids:
                st_seat = admitted_states[sid]
                p_st = pair_states[sid][ci]
                p_st.held_follower_qty = ledger.seat_held_qty(sid, ci)
                held_q = p_st.held_follower_qty
                held_by_seat_bar[sid][ci] = held_q

                stm = streams[sid]
                src_q = float(stm.source_positions[ci, j])
                src_mk = (
                    float(stm.source_marks[ci, j])
                    if stm.source_marks is not None
                    else (to_px if math.isfinite(to_px) else mc_px)
                )
                src_age_ms = (
                    int(stm.source_ages_ms[ci, j]) if stm.source_ages_ms is not None else 0
                )

                if st_seat.pending_k_scale is not None and abs(held_q) <= eps:
                    seat_coin_k[sid][ci] = float(st_seat.pending_k_scale)

                if stale_open_gate[sid][ci]:
                    snap_s = float(stale_open_snap[sid][ci])
                    if abs(src_q) <= eps or cr._opposes(src_q, snap_s, eps):
                        stale_open_gate[sid][ci] = False
                        stale_open_snap[sid][ci] = 0.0

                eff_k = float(seat_coin_k[sid][ci])
                _cls, cwsa_want, pair_exec = cr.classify_and_propose_cwsa_pair(
                    p_st,
                    source_qty=src_q,
                    source_mark=src_mk,
                    mark_price=mc_px,
                    exec_quote_buy=q_buy,
                    exec_quote_sell=q_sell,
                    k_scale=eff_k,
                    source_age_ms=src_age_ms,
                    mark_age_ms=mk_age_ms,
                    qty_multiplier=float(qty_mults[ci]),
                    initial_follower_qty=held_q,
                    cfg=cwsa_cfg,
                )

                if stale_open_gate[sid][ci]:
                    cwsa_want = 0.0

                raw_intended_by_seat[sid][ci] = cwsa_want

                if not allow_2d[ci, j]:
                    if cr._same_sign(held_q, cwsa_want, eps):
                        cwsa_want = math.copysign(min(abs(held_q), abs(cwsa_want)), held_q)
                    else:
                        cwsa_want = 0.0

                if coin_executable[ci] and (
                    guardian_flatten or sweep_lockout or st_seat.state == SeatState.RETIRED
                ):
                    post_contain_desired[sid][ci] = 0.0
                    seat_blocks_cz[(sid, ci)] = False
                    seat_init_disp[(sid, ci)] = cr.OrderDisposition.NOOP
                    continue

                if not pair_exec:
                    post_contain_desired[sid][ci] = held_q
                    seat_blocks_cz[(sid, ci)] = abs(held_q) > eps
                    seat_init_disp[(sid, ci)] = cr.OrderDisposition.QUARANTINED
                    static_held_mask[sid][ci] = abs(held_q) > eps
                    continue

                if st_seat.state == SeatState.FREEZE_NEW_RISK:
                    if policy_cfg.lifecycle_arm == LifecycleArm.GUARDIAN_MANDATE_PAUSE_STATIC_HOLD:
                        post_contain_desired[sid][ci] = held_q
                        seat_blocks_cz[(sid, ci)] = abs(held_q) > eps
                        static_held_mask[sid][ci] = abs(held_q) > eps
                        seat_init_disp[(sid, ci)] = (
                            cr.OrderDisposition.STATIC_HELD
                            if abs(held_q) > eps
                            else cr.OrderDisposition.NOOP
                        )
                    else:
                        if cr._same_sign(held_q, cwsa_want, eps):
                            des = math.copysign(min(abs(held_q), abs(cwsa_want)), held_q)
                        else:
                            des = 0.0
                        post_contain_desired[sid][ci] = des
                        seat_blocks_cz[(sid, ci)] = abs(held_q) > eps
                        seat_init_disp[(sid, ci)] = cr.OrderDisposition.NOOP
                else:
                    post_contain_desired[sid][ci] = cwsa_want
                    seat_blocks_cz[(sid, ci)] = False
                    seat_init_disp[(sid, ci)] = cr.OrderDisposition.NOOP

        gate_res = apply_renetting_safety_gate(
            post_contain_desired,
            decision_px_bar,
            account_equity_usd=eq_open_preview,
            held_by_seat=held_by_seat_bar,
            locked_seat_mask=static_held_mask,
            cfg=policy_cfg.renetting,
        )
        if gate_res.symbol_clamped_coins or gate_res.gross_clamped:
            renetting_clamped_bars += 1

        bar_allocations: dict[int, cr.MarketStepAllocation] = {}
        for ci in range(n_coins):
            if not coin_executable[ci]:
                continue
            to_px = float(t_open[ci, j])
            mc_px = float(m_close[ci, j])
            q_buy = to_px * (1.0 + spread_frac)
            q_sell = to_px * max(1e-9, 1.0 - spread_frac)

            intents_ci = [
                cr.SeatMarketIntent(
                    seat_id=sid,
                    held_qty=ledger.seat_held_qty(sid, ci),
                    raw_desired_qty=float(gate_res.adjusted_seat_desired[sid][ci]),
                    seat_blocks_cross_zero=seat_blocks_cz[(sid, ci)],
                    initial_disposition=seat_init_disp[(sid, ci)],
                )
                for sid in active_sids
            ]
            alloc = cr.allocate_l3_market_step(
                ci,
                intents_ci,
                exec_price_buy=q_buy,
                exec_price_sell=q_sell,
                mark_price=mc_px,
                l2_cap_usd=float(bar_caps[ci]),
                l2_prior_vol_valid=bool(bar_vol_valid[ci]),
                actual_bar_vol_1m_usd=float(vol_mat[ci, j]),
                cross_zero_permitted=cwsa_cfg.cross_zero_permitted,
                protective_exit_haircut_bps=liq_cfg.protective_exit_haircut_bps,
                eps=eps,
            )
            bar_allocations[ci] = alloc
            total_missed_notional += alloc.missed_notional_usd
            if alloc.extra_haircut_bps > 0.0:
                haircut_cnt += 1
            for disp in alloc.seat_dispositions.values():
                if disp == cr.OrderDisposition.CLIPPED:
                    clipped_cnt += 1
                elif disp == cr.OrderDisposition.MISSED:
                    missed_cnt += 1
            if abs(alloc.external_delta_qty) > eps:
                trade_count += 1
                total_ext_notional += alloc.external_notional_usd
                if math.isfinite(alloc.realized_participation_1m):
                    trade_ratios.append(alloc.realized_participation_1m)
                else:
                    missing_vol_trades += 1

        fund_bar = fund_mat[:, j] if fund_mat is not None else None
        close_px_bar = np.where(
            (mk_ages[:, j] >= 0) & (mk_ages[:, j] <= cwsa_cfg.mark_stale_ms),
            m_close[:, j],
            np.nan,
        )
        _bar_err, bar_swept = ledger.step_bar(
            bar_ts_ms,
            allocations=bar_allocations,
            raw_intended_by_seat=raw_intended_by_seat,
            decision_mid_prices=decision_px_bar,
            close_mark_prices=close_px_bar,
            funding_rates_bar=fund_bar,
        )
        if bar_swept or guardian.swept_today:
            _latch_stale_open_legs(
                active_sids,
                streams,
                pair_states,
                stale_open_gate,
                stale_open_snap,
                j,
                n_coins,
                eps,
            )

        for ci, inst in enumerate(ledger.instruments):
            net_pos_hist[ci, j] = float(ledger.account.quantities.get(inst, ZERO))
            for sid in active_sids:
                held_now = ledger.seat_held_qty(sid, ci)
                pair_states[sid][ci].held_follower_qty = held_now
                seat_pos_hist[sid][ci, j] = held_now

        eq_curve[j] = ledger.equity_usd
        bal_curve[j] = ledger.balance_usd

        for sid in active_sids:
            st_seat = admitted_states[sid]
            prev_state = st_seat.state
            stm = streams[sid]
            av_val = float(stm.source_av_usd[j]) if stm.source_av_usd is not None else None
            av_age = float(stm.source_av_ages_s[j]) if stm.source_av_ages_s is not None else None
            follower_flat = not ledger.seat_has_open_pos(sid, eps=eps)
            leader_flat = bool(np.all(np.abs(stm.source_positions[:, j]) <= eps))

            for ci in range(n_coins):
                c_held = abs(ledger.seat_held_qty(sid, ci)) > eps
                c_upnl = ledger.seat_coin_open_upnl_usd(sid, ci, eps=eps)
                if c_held and c_upnl < -1e-12:
                    seat_coin_underwater[sid][ci] += 1
                else:
                    seat_coin_underwater[sid][ci] = 0
            max_uw_bars = int(np.max(seat_coin_underwater[sid])) if n_coins > 0 else 0

            step_fast_seat_clock(
                st_seat,
                bar_ts_ms=bar_ts_ms,
                cum_pnl_usd=ledger.seat_cum_pnl_usd(sid),
                open_upnl_usd=ledger.seat_open_upnl_usd(sid, eps=eps),
                has_open_pos=not follower_flat,
                arm=policy_cfg.lifecycle_arm,
                source_av_usd=av_val,
                source_av_age_s=av_age,
                source_age_s=seat_src_age_s[sid],
                mark_age_s=seat_mk_age_s[sid],
                follower_is_flat=follower_flat,
                leader_is_flat=leader_flat,
                max_coin_underwater_bars=max_uw_bars,
                bar_idx=j,
            )

            s_ages_bar = stm.source_ages_ms[:, j] if stm.source_ages_ms is not None else None
            closed_eps = shadow_trackers[sid].step_bar(
                j,
                stm.source_positions[:, j],
                t_open[:, j],
                m_close[:, j],
                source_ages_ms=s_ages_bar,
                mark_ages_ms=mk_ages[:, j],
                source_stale_ms=cwsa_cfg.source_stale_ms,
                mark_stale_ms=cwsa_cfg.mark_stale_ms,
            )
            use_eq_units = st_seat.spec.episode_baseline is not None
            for ep in closed_eps:
                z_proxy = ep.ep_ropn / 0.05
                ep_ret_val = ep.ep_equity_ret if use_eq_units else ep.ep_ropn
                ep_mae_val = ep.ep_equity_mae if use_eq_units else ep.ep_mae_ropn
                qualifies_post_freeze = (
                    follower_flat
                    and st_seat.flat_since_bar >= 0
                    and ep.open_bar >= max(st_seat.last_freeze_bar, st_seat.flat_since_bar)
                )
                step_slow_episode_clock(
                    st_seat,
                    arm=policy_cfg.lifecycle_arm,
                    ep_return_z=z_proxy,
                    ep_mae_r=ep_mae_val,
                    ep_return=ep_ret_val,
                    ep_peak_notional_frac=ep.peak_notional_usd / max(
                        policy_cfg.initial_balance_usd, 1.0
                    ),
                    ep_duration_bars=float(ep.duration_bars),
                    ep_mae=ep_mae_val,
                    ep_symbol_notional=ep.symbol_notional_vec,
                    opened_post_freeze=qualifies_post_freeze,
                    episode_is_clean=ep.is_clean,
                    follower_is_flat=follower_flat,
                    leader_is_flat=leader_flat,
                    bar_ts_ms=bar_ts_ms,
                    bar_idx=j,
                )

            if prev_state == SeatState.FREEZE_NEW_RISK and st_seat.state == SeatState.ACTIVE:
                for ci in range(n_coins):
                    if abs(pair_states[sid][ci].held_follower_qty) <= eps:
                        src_cur = float(stm.source_positions[ci, j])
                        if math.isfinite(src_cur):
                            pair_states[sid][ci].prev_source_qty = src_cur
                            if abs(src_cur) > eps:
                                stale_open_gate[sid][ci] = True
                                stale_open_snap[sid][ci] = src_cur

            state_hist[sid].append(st_seat.state.value)

        newly_retired_flat = [
            sid
            for sid in active_sids
            if admitted_states[sid].state == SeatState.RETIRED
            and sid not in handled_retired_seats
            and not ledger.seat_has_open_pos(sid, eps=eps)
        ]
        if newly_retired_flat:
            for r_sid in newly_retired_flat:
                handled_retired_seats.add(r_sid)

            if policy_cfg.replacement_mode == ReplacementMode.RESIZE_HEALTHY_SEATS:
                healthy = [
                    sid
                    for sid in active_sids
                    if admitted_states[sid].state == SeatState.ACTIVE
                    and not admitted_states[sid].freeze_reasons
                    and admitted_states[sid].cusum_pos <= 1.0
                ]
                healthy_r = sum(admitted_states[h].spec.r_unit_usd for h in healthy)
                if healthy and healthy_r > eps:
                    scale_r = initial_target_r_sum / healthy_r
                    for h_sid in healthy:
                        h_st = admitted_states[h_sid]
                        old_k = h_st.current_k_scale
                        cap_k = (
                            float(h_st.spec.max_k_scale)
                            if h_st.spec.max_k_scale is not None
                            else 1.5 * float(h_st.spec.k_scale)
                        )
                        new_k = min(cap_k, float(h_st.spec.k_scale) * scale_r)
                        if new_k > old_k + eps:
                            h_st.current_k_scale = new_k
                            h_st.pending_k_scale = new_k
                            resized_events.append((bar_ts_ms, h_sid, old_k, new_k))

            elif policy_cfg.replacement_mode == ReplacementMode.PROMOTE_PIT_TOP_BENCH_NEW_POS_ONLY:
                causal_snaps = [s for s in sorted_bench if s.asof_ts_ms <= bar_ts_ms]
                if causal_snaps:
                    latest = causal_snaps[-1]
                    ranked = sorted(latest.candidates, key=lambda c: c.score, reverse=True)
                    for _ in newly_retired_flat:
                        chosen: PITBenchCandidate | None = None
                        for cand in ranked:
                            c_id = cand.spec.seat_id
                            if c_id not in ever_admitted and c_id in streams:
                                chosen = cand
                                break
                        if chosen is None:
                            break
                        b_sid = chosen.spec.seat_id
                        ever_admitted.add(b_sid)
                        promoted_bench.append(b_sid)
                        b_state = SeatContainmentState(spec=chosen.spec, flat_since_bar=j)
                        admitted_states[b_sid] = b_state
                        b_stm = streams[b_sid]
                        b_src_now = b_stm.source_positions[:, j]
                        b_pairs: list[cr.PairCWSAState] = []
                        for ci in range(n_coins):
                            sq = float(b_src_now[ci])
                            s_age = (
                                int(b_stm.source_ages_ms[ci, j])
                                if b_stm.source_ages_ms is not None
                                else 0
                            )
                            if (
                                math.isfinite(sq)
                                and 0 <= s_age <= cwsa_cfg.source_stale_ms
                                and coin_executable[ci]
                            ):
                                b_pairs.append(
                                    cr.PairCWSAState(
                                        initialized=True,
                                        prev_source_qty=sq,
                                        held_follower_qty=0.0,
                                    )
                                )
                            else:
                                b_pairs.append(cr.PairCWSAState())
                        pair_states[b_sid] = b_pairs
                        seat_coin_k[b_sid] = np.full(
                            n_coins, float(chosen.spec.k_scale), dtype=np.float64
                        )
                        seat_coin_underwater[b_sid] = np.zeros(n_coins, dtype=np.int64)
                        stale_open_gate[b_sid] = np.isfinite(b_src_now) & (np.abs(b_src_now) > eps)
                        stale_open_snap[b_sid] = np.where(np.isfinite(b_src_now), b_src_now, 0.0)
                        b_tracker = ShadowUnitEpisodeTracker(
                            b_sid,
                            n_coins,
                            cost_bps=cwsa_cfg.fee_bps + cwsa_cfg.slippage_bps,
                            pod_equity_usd=policy_cfg.initial_balance_usd,
                            eps=eps,
                        )
                        b_s_ages = (
                            b_stm.source_ages_ms[:, j]
                            if b_stm.source_ages_ms is not None
                            else None
                        )
                        b_tracker.step_bar(
                            j,
                            b_src_now,
                            t_open[:, j],
                            m_close[:, j],
                            source_ages_ms=b_s_ages,
                            mark_ages_ms=mk_ages[:, j],
                            source_stale_ms=cwsa_cfg.source_stale_ms,
                            mark_stale_ms=cwsa_cfg.mark_stale_ms,
                        )
                        shadow_trackers[b_sid] = b_tracker
                        ledger.ensure_seat(b_sid)
                        seat_pos_hist[b_sid] = np.zeros((n_coins, n_bars), dtype=np.float64)
                        state_hist[b_sid] = ["UNADMITTED"] * (j + 1)

    ratios_np = np.asarray(trade_ratios, dtype=np.float64)
    audit = cr.RealizedParticipationAudit(
        lookahead_bars_used=0,
        trade_count=trade_count,
        volume_covered_trades=int(ratios_np.size),
        unpriced_or_missing_vol_trades=missing_vol_trades,
        total_external_notional_usd=total_ext_notional,
        total_missed_notional_usd=total_missed_notional,
        max_participation_1m=float(np.max(ratios_np)) if ratios_np.size else 0.0,
        p95_participation_1m=float(np.quantile(ratios_np, 0.95)) if ratios_np.size else 0.0,
        median_participation_1m=float(np.median(ratios_np)) if ratios_np.size else 0.0,
        bars_gt_5pct_count=int(np.count_nonzero(ratios_np > 0.05)) if ratios_np.size else 0,
        clipped_order_count=clipped_cnt,
        missed_order_count=missed_cnt,
        protective_haircut_count=haircut_cnt,
    )

    return PortfolioPolicyResult(
        config=policy_cfg,
        coins=coin_tuple,
        grid_ms=grid_arr,
        equity_curve=eq_curve,
        balance_curve=bal_curve,
        net_positions=net_pos_hist,
        seat_positions=seat_pos_hist,
        seat_state_history={sid: tuple(hist) for sid, hist in state_hist.items()},
        seat_final_states=admitted_states,
        ledger=ledger,
        economics=ledger.explain(),
        max_pm03_conservation_err=ledger.max_conservation_err,
        participation_audit=audit,
        renetting_clamped_bars=renetting_clamped_bars,
        promoted_bench_seats=tuple(promoted_bench),
        resized_seat_events=tuple(resized_events),
        guardian_breached=guardian.breached,
        guardian_breach_reason=guardian.breach_reason,
        guardian_target_passed=guardian.target_passed,
        guardian_sweeps=guardian.sweep_count,
    )


def compare_seat_replacement_modes(
    *,
    seat_specs: Sequence[SeatRiskSpec],
    seat_streams: Mapping[str, SeatTelemetryStream | np.ndarray] | Sequence[SeatTelemetryStream],
    bench_snapshots: Sequence[PITBenchSnapshot] = (),
    coins: Sequence[str],
    trade_open: np.ndarray,
    mark_close: np.ndarray,
    grid_ms: np.ndarray,
    vol_1m_usd: np.ndarray | None = None,
    mark_ages_ms: np.ndarray | None = None,
    allowed_markets_mask: np.ndarray | None = None,
    funding_rates: np.ndarray | None = None,
    base_policy_cfg: PortfolioPolicyConfig = PortfolioPolicyConfig(),
    modes: Collection[ReplacementMode] = (
        ReplacementMode.LEAVE_RISK_UNUSED,
        ReplacementMode.RESIZE_HEALTHY_SEATS,
        ReplacementMode.PROMOTE_PIT_TOP_BENCH_NEW_POS_ONLY,
    ),
) -> SeatReplacementComparison:
    """Run matched counterfactual replays across the 3 seat-replacement policies."""
    results: dict[ReplacementMode, PortfolioPolicyResult] = {}
    summary: dict[str, dict[str, Any]] = {}

    for mode in modes:
        cfg_m = replace(base_policy_cfg, replacement_mode=mode)
        res = run_portfolio_policy_replay(
            seat_specs=seat_specs,
            seat_streams=seat_streams,
            coins=coins,
            trade_open=trade_open,
            mark_close=mark_close,
            grid_ms=grid_ms,
            vol_1m_usd=vol_1m_usd,
            mark_ages_ms=mark_ages_ms,
            allowed_markets_mask=allowed_markets_mask,
            funding_rates=funding_rates,
            bench_snapshots=bench_snapshots,
            policy_cfg=cfg_m,
        )
        results[mode] = res
        net_pnl = float(res.equity_curve[-1] - cfg_m.initial_balance_usd)
        min_eq = float(np.min(res.equity_curve))
        static_dd = max(0.0, cfg_m.initial_balance_usd - min_eq)
        summary[mode.value] = {
            "final_equity_usd": float(res.equity_curve[-1]),
            "net_pnl_usd": net_pnl,
            "static_floor_dd_usd": static_dd,
            "max_pm03_conservation_err": res.max_pm03_conservation_err,
            "promoted_bench_seats": res.promoted_bench_seats,
            "resized_seat_events": len(res.resized_seat_events),
            "guardian_breached": res.guardian_breached,
            "guardian_target_passed": res.guardian_target_passed,
        }

    return SeatReplacementComparison(results_by_mode=results, summary_by_mode=summary)
