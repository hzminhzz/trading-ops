"""Canonical research replay, universe discovery, Guardian rules, and PM03 reconciliation.

Provides shared, import-safe building blocks for ``trading-ops`` studies so individual
reports reuse one audited implementation rather than duplicating ad-hoc simulators:

1. ``load_registry_universe`` — dynamic SQLite screening over all configured wallets.
2. ``assess_bybit_symbol_screen`` — materiality-aware symbol mapping and scale alias flags.
3. ``reconstruct_positions_on_grid`` — causal ``new`` vs anchored ``full`` replay with zero
   pre-first-fill backward lookahead.
4. ``replay_unit_coins_to_5m`` — execution-aligned 5m frames directly consumable by
   ``pod_constructor.build_tensor_from_unit_replays``, with pair-local gap termination.
5. ``simulate_guardian_challenge`` — causal Challenge Lifecycle simulator distinguishing
   ``static_floor_dd_usd`` from ``peak_to_trough_dd_usd`` and separating terminal challenge
   verdicts from continuous empirical daily MC blocks.
6. ``reconcile_portfolio_with_pm03`` — Tier-2 ``copytrade_pm.pm03_account`` Decimal check.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal as D
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from copytrade_pm.pm03_account import Account, Instrument, Rulebook
from copytrade_pm.pm03_execution import CostModel, MarketObservation, execute_targets

PolicyType = Literal["new", "full"]
DEFAULT_COST_BPS = 6.5
DEFAULT_FEE_BPS = 4.5
DEFAULT_SLIPPAGE_BPS = 2.0
GAP_EXIT_HAIRCUT_BPS = 50.0


@dataclass(frozen=True)
class WalletRecord:
    """Read-only wallet metadata from ``wallet_registry.sqlite``."""

    wallet_id: int
    address: str
    alias: str | None
    configured: bool
    excluded: bool

    @property
    def label(self) -> str:
        """Sanitized identifier safe for public reports and JSON outputs."""
        return self.alias if self.alias else f"W#{self.wallet_id}"


@dataclass(frozen=True)
class SymbolScreenResult:
    """Materiality-aware venue symbol screen for one candidate wallet."""

    label: str
    total_fills: int
    mapped_fills: int
    unmapped_coins: tuple[str, ...]
    mapped_coins: tuple[str, ...]
    unmapped_fill_frac: float
    unmapped_notional_frac: float | None
    scale_alias_candidates: dict[str, str]
    status: Literal[
        "EXACT_MAPPED",
        "MAPPED_WITH_EXCLUDED_TAIL",
        "REJECT_UNMAPPED_MATERIAL",
        "REJECT_NO_FILLS",
    ]


@dataclass(frozen=True)
class ReconstructedGrid:
    """Causal 5-minute signed position vector with provenance flags."""

    timestamps_ms: np.ndarray
    pos: np.ndarray
    unanchored_pre_inventory: bool
    first_fill_ms: int | None
    first_observed_pre_pos: float


@dataclass(frozen=True)
class GuardianSpec:
    """Prop-firm challenge and internal Guardian risk specification."""

    initial_equity: float = 10_000.0
    profit_target: float = 800.0
    max_static_dd: float = 1_000.0
    daily_loss_limit: float = 500.0
    guardian_sweep_loss: float = 400.0
    cost_bps: float = DEFAULT_COST_BPS
    cost_multiplier: float = 1.0
    funding_bps_per_8h: float = 0.0
    funding_multiplier: float = 1.0
    reset_utc_hour: int = 0
    reset_utc_minute: int = 5
    daily_loss_basis: Literal["balance", "equity"] = "balance"
    daily_loss_rate: float | None = None
    guardian_sweep_rate: float | None = None
    reenter_after_sweep: Literal["new_only", "immediate_target"] = "new_only"
    min_trade_notional_usd: float = 0.0
    amount_step: float = 0.0
    min_amount: float = 0.0
    amount_step_by_coin: Mapping[str, float] | None = None
    min_amount_by_coin: Mapping[str, float] | None = None
    max_gross_leverage: float | None = None

    def __post_init__(self) -> None:
        if self.daily_loss_basis not in ("balance", "equity"):
            raise ValueError(f"INVALID_DAILY_LOSS_BASIS:{self.daily_loss_basis}")
        if self.daily_loss_rate is not None and not (0.0 < float(self.daily_loss_rate) < 1.0):
            raise ValueError(f"INVALID_DAILY_LOSS_RATE:{self.daily_loss_rate}")
        if self.guardian_sweep_rate is not None and not (
            0.0 < float(self.guardian_sweep_rate) < 1.0
        ):
            raise ValueError(f"INVALID_GUARDIAN_SWEEP_RATE:{self.guardian_sweep_rate}")


@dataclass
class ChallengeRunResult:
    """Separated terminal challenge metrics and continuous empirical MC path."""

    verdict: Literal["PASS", "FAIL_STATIC", "FAIL_DAILY", "TIMEOUT"]
    terminal_step: int | None
    terminal_ts_ms: int | None
    terminal_net_usd: float
    min_equity_usd: float
    static_floor_dd_usd: float
    peak_to_trough_dd_usd: float
    worst_daily_usd: float
    sweeps: int
    forced_gap_exits: int
    unpriced_open_steps: int
    deferred_notional_usd: float
    full_window_net_usd: float
    full_window_peak_to_trough_dd_usd: float
    full_window_static_floor_dd_usd: float
    full_window_sweeps: int
    step_pnl: np.ndarray
    equity_curve: np.ndarray
    peak_gross_leverage: float = 0.0
    full_window_peak_gross_leverage: float = 0.0
    total_funding_usd: float = 0.0
    full_window_funding_usd: float = 0.0
    executed_deltas: np.ndarray | None = None


def roster_digest(records: Sequence[WalletRecord]) -> str:
    """Compute a deterministic SHA-256 hash of roster composition without raw addresses."""
    canonical = ";".join(
        f"{r.wallet_id}:{r.label}:{int(r.configured)}:{int(r.excluded)}"
        for r in sorted(records, key=lambda rec: rec.wallet_id)
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def is_hl_perp_coin(coin: str) -> bool:
    """Return True for main-DEX perpetual symbols (exclude spot ``@/#`` and HIP-3 ``dex:``)."""
    return bool(coin) and not coin.startswith(("@", "#")) and "/" not in coin and ":" not in coin


def load_registry_universe(
    registry_db: Path | str,
    *,
    include_excluded: bool = False,
    include_aliases: Sequence[str] = (),
) -> tuple[list[WalletRecord], str]:
    """Query all configured wallets from native ``wallet_registry.sqlite`` read-only."""
    db_path = Path(registry_db).expanduser()
    if not db_path.exists():
        raise FileNotFoundError(f"Registry DB not found: {db_path}")

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        integrity = con.execute("PRAGMA quick_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"REGISTRY_INTEGRITY_FAILED:{integrity}")

        rows = con.execute(
            """
            SELECT w.wallet_id,
                   lower(w.address) AS address,
                   COALESCE(m.configured, 0) AS configured,
                   EXISTS(SELECT 1 FROM exclusions e WHERE e.wallet_id = w.wallet_id) AS excluded,
                   (
                       SELECT wa.alias
                       FROM wallet_aliases wa
                       WHERE wa.wallet_id = w.wallet_id
                       ORDER BY wa.known_at DESC, rowid DESC
                       LIMIT 1
                   ) AS alias
            FROM wallets w
            LEFT JOIN monitoring m USING (wallet_id)
            ORDER BY w.wallet_id
            """
        ).fetchall()
    finally:
        con.close()

    wanted_aliases = set(include_aliases)
    records: list[WalletRecord] = []
    for wallet_id, address, configured, excluded, alias in rows:
        is_cfg = bool(configured)
        is_excl = bool(excluded)
        if is_cfg and (include_excluded or not is_excl):
            keep = True
        elif alias is not None and alias in wanted_aliases:
            keep = include_excluded or not is_excl
        else:
            keep = False
        if keep:
            records.append(
                WalletRecord(
                    wallet_id=int(wallet_id),
                    address=str(address),
                    alias=str(alias) if alias is not None else None,
                    configured=is_cfg,
                    excluded=is_excl,
                )
            )
    return records, roster_digest(records)


def assess_bybit_symbol_screen(
    label: str,
    fills_by_coin: dict[str, pl.DataFrame],
    bybit_usdt_perps: set[str],
    *,
    max_unmapped_fill_frac: float = 0.05,
    max_unmapped_notional_frac: float = 0.05,
) -> SymbolScreenResult:
    """Screen a wallet's Hyperliquid perps against Bybit linear USDT perps by materiality."""
    total_fills = 0
    mapped_fills = 0
    total_notional = 0.0
    unmapped_notional = 0.0
    has_notional = True

    mapped_coins: list[str] = []
    unmapped_coins: list[str] = []
    scale_candidates: dict[str, str] = {}

    for coin in sorted(fills_by_coin):
        if not is_hl_perp_coin(coin):
            continue
        df = fills_by_coin[coin]
        cnt = int(df.height)
        if cnt == 0:
            continue
        total_fills += cnt
        coin_notional = 0.0
        if "px" in df.columns and "sz" in df.columns:
            coin_notional = float((df["px"].abs() * df["sz"].abs()).sum())
            total_notional += coin_notional
        elif "notional" in df.columns:
            coin_notional = float(df["notional"].abs().sum())
            total_notional += coin_notional
        else:
            has_notional = False

        if coin in bybit_usdt_perps:
            mapped_fills += cnt
            mapped_coins.append(coin)
        else:
            unmapped_coins.append(coin)
            unmapped_notional += coin_notional
            if coin.startswith("k"):
                candidate = f"1000{coin[1:]}"
                if candidate in bybit_usdt_perps:
                    scale_candidates[coin] = candidate

    if total_fills == 0:
        return SymbolScreenResult(
            label=label,
            total_fills=0,
            mapped_fills=0,
            unmapped_coins=(),
            mapped_coins=(),
            unmapped_fill_frac=0.0,
            unmapped_notional_frac=None,
            scale_alias_candidates={},
            status="REJECT_NO_FILLS",
        )

    fill_frac = (total_fills - mapped_fills) / total_fills
    notional_frac = (unmapped_notional / total_notional) if has_notional and total_notional > 0 else None

    if not unmapped_coins:
        status: Literal[
            "EXACT_MAPPED",
            "MAPPED_WITH_EXCLUDED_TAIL",
            "REJECT_UNMAPPED_MATERIAL",
            "REJECT_NO_FILLS",
        ] = "EXACT_MAPPED"
    elif fill_frac <= max_unmapped_fill_frac and (
        notional_frac is None or notional_frac <= max_unmapped_notional_frac
    ):
        status = "MAPPED_WITH_EXCLUDED_TAIL"
    else:
        status = "REJECT_UNMAPPED_MATERIAL"

    return SymbolScreenResult(
        label=label,
        total_fills=total_fills,
        mapped_fills=mapped_fills,
        unmapped_coins=tuple(unmapped_coins),
        mapped_coins=tuple(mapped_coins),
        unmapped_fill_frac=fill_frac,
        unmapped_notional_frac=notional_frac,
        scale_alias_candidates=scale_candidates,
        status=status,
    )


def reconstruct_positions_on_grid(
    fills: pl.DataFrame,
    grid_ms: np.ndarray,
    *,
    policy: PolicyType = "new",
    activation_ms: int | None = None,
    initial_anchor_pos: float | None = None,
    eps: float = 1e-9,
) -> ReconstructedGrid:
    """Reconstruct causal 5m positions without backward lookahead from future ``pre[0]``.

    Expects ``fills`` sorted by ``ts`` with columns ``ts`` (epoch ms), ``pre`` (start position
    prior to fill), ``sign`` (+1/-1), and ``sz`` (positive fill size).
    """
    n_grid = int(grid_ms.size)
    if fills.is_empty():
        base = float(initial_anchor_pos) if (policy == "full" and initial_anchor_pos) else 0.0
        return ReconstructedGrid(
            timestamps_ms=grid_ms,
            pos=np.full(n_grid, base, dtype=np.float64),
            unanchored_pre_inventory=False,
            first_fill_ms=None,
            first_observed_pre_pos=0.0,
        )

    times = fills["ts"].to_numpy().astype(np.int64)
    pre = fills["pre"].to_numpy().astype(np.float64)
    signs = fills["sign"].to_numpy().astype(np.float64)
    sz = fills["sz"].to_numpy().astype(np.float64)
    post = pre + signs * sz

    first_fill_ms = int(times[0])
    first_pre = float(pre[0])
    unanchored = initial_anchor_pos is None and abs(first_pre) > eps

    # Strict causal lookup: fill at timestamp t is only reflected at checkpoints > t.
    ix = np.searchsorted(times, grid_ms, side="left") - 1

    if policy == "full":
        pre_first_fill = float(initial_anchor_pos) if initial_anchor_pos is not None else 0.0
        pos = np.where(ix >= 0, post[np.clip(ix, 0, len(times) - 1)], pre_first_fill)
        return ReconstructedGrid(
            timestamps_ms=grid_ms,
            pos=pos.astype(np.float64),
            unanchored_pre_inventory=unanchored,
            first_fill_ms=first_fill_ms,
            first_observed_pre_pos=first_pre,
        )

    act_ms = int(activation_ms) if activation_ms is not None else int(grid_ms[0])
    n_fills = len(times)
    eligible = np.zeros(n_fills, dtype=bool)
    active = False
    prev = first_pre

    for i in range(n_fills):
        after = float(post[i])
        if times[i] < act_ms:
            prev = after
            continue
        if (abs(prev) < eps and abs(after) >= eps) or (prev * after < -1e-12):
            active = True
        if abs(after) < eps:
            active = False
        eligible[i] = active
        prev = after

    valid_ix = ix >= 0
    clipped = np.clip(ix, 0, n_fills - 1)
    pos = np.where(valid_ix & eligible[clipped], post[clipped], 0.0)
    return ReconstructedGrid(
        timestamps_ms=grid_ms,
        pos=pos.astype(np.float64),
        unanchored_pre_inventory=unanchored,
        first_fill_ms=first_fill_ms,
        first_observed_pre_pos=first_pre,
    )


def replay_coin_targets_vectorized(
    targets: np.ndarray,
    trade_open: np.ndarray,
    mark_close: np.ndarray,
    *,
    cost_bps: float = DEFAULT_COST_BPS,
    gap_exit_haircut_bps: float = GAP_EXIT_HAIRCUT_BPS,
    fee_bps: float = DEFAULT_FEE_BPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Simulate causal per-bar fills, fees, and pair-local stale-mark exits on a 1-coin target."""
    n = int(targets.size)
    cost_rate = cost_bps * 1e-4
    haircut_rate = gap_exit_haircut_bps * 1e-4
    fee_rate = fee_bps * 1e-4

    pos = np.zeros(n, dtype=np.float64)
    delta = np.zeros(n, dtype=np.float64)
    gross = np.zeros(n, dtype=np.float64)
    cost = np.zeros(n, dtype=np.float64)

    cur_pos = 0.0
    prev_mark = np.nan
    forced_exits = 0
    unpriced_open_steps = 0
    deferred_notional_usd = 0.0

    for j in range(n):
        m_close = float(mark_close[j])
        t_open = float(trade_open[j])
        want = float(targets[j])

        # Pair-local conservative termination if holding an open leg into a data outage.
        if not np.isfinite(m_close):
            if abs(cur_pos) > 1e-12:
                unpriced_open_steps += 1
                if np.isfinite(prev_mark) and prev_mark > 0:
                    direction = 1.0 if cur_pos > 0 else -1.0
                    exit_px = prev_mark * (1.0 - direction * haircut_rate)
                    gross[j] = cur_pos * (exit_px - prev_mark)
                    cost[j] = abs(cur_pos) * exit_px * fee_rate
                    delta[j] = -cur_pos
                    cur_pos = 0.0
                    forced_exits += 1
            pos[j] = cur_pos
            continue

        desired = want
        if abs(want - cur_pos) > 1e-9 and not np.isfinite(t_open):
            desired = cur_pos
            deferred_notional_usd += abs(want - cur_pos) * m_close

        exec_px = t_open if np.isfinite(t_open) else m_close
        d_qty = desired - cur_pos
        bar_gross = 0.0
        if np.isfinite(prev_mark):
            bar_gross += cur_pos * (exec_px - prev_mark)
        bar_gross += desired * (m_close - exec_px)
        bar_cost = abs(d_qty) * exec_px * cost_rate

        pos[j] = desired
        delta[j] = d_qty
        gross[j] = bar_gross
        cost[j] = bar_cost
        cur_pos = desired
        prev_mark = m_close

    return (
        pos,
        delta,
        gross,
        cost,
        {
            "forced_gap_exits": float(forced_exits),
            "unpriced_open_steps": float(unpriced_open_steps),
            "deferred_notional_usd": float(deferred_notional_usd),
        },
    )


def build_unit_replay_by_alias(
    grouped_fills: dict[str, dict[str, pl.DataFrame]],
    prices_by_coin: dict[str, tuple[np.ndarray, np.ndarray]],
    aliases: Sequence[str],
    grid_ms: np.ndarray,
    *,
    policy: PolicyType = "new",
    activation_ms: int | None = None,
    initial_anchors: dict[str, dict[str, float]] | None = None,
    cost_bps: float = DEFAULT_COST_BPS,
) -> tuple[dict[str, dict[str, pl.DataFrame]], dict[str, dict[str, int]]]:
    """Build Policy-A 5m DataFrames ready for ``pod_constructor.build_tensor_from_unit_replays``.

    Each DataFrame carries ``["timestamp", "open", "close", "pos", "delta", "gross", "cost", "net"]``
    aligned identically across ``grid_ms``.
    """
    ts_col = pl.from_epoch(pl.Series("timestamp", grid_ms, dtype=pl.Int64), time_unit="ms")
    ts_utc = ts_col.dt.replace_time_zone("UTC")
    unit_frames: dict[str, dict[str, pl.DataFrame]] = {}
    diagnostics: dict[str, dict[str, int]] = {}

    for alias in aliases:
        by_coin = grouped_fills.get(alias, {})
        wallet_frames: dict[str, pl.DataFrame] = {}
        diag = {"unanchored_coins": 0, "forced_gap_exits": 0, "unpriced_open_steps": 0}
        anchors = (initial_anchors or {}).get(alias, {})

        for coin, coin_fills in by_coin.items():
            if coin not in prices_by_coin:
                continue
            trade_open, mark_close = prices_by_coin[coin]
            recon = reconstruct_positions_on_grid(
                coin_fills,
                grid_ms,
                policy=policy,
                activation_ms=activation_ms,
                initial_anchor_pos=anchors.get(coin),
            )
            if recon.unanchored_pre_inventory:
                diag["unanchored_coins"] += 1

            pos, delta, gross, cost, stats = replay_coin_targets_vectorized(
                recon.pos, trade_open, mark_close, cost_bps=cost_bps
            )
            diag["forced_gap_exits"] += int(stats["forced_gap_exits"])
            diag["unpriced_open_steps"] += int(stats["unpriced_open_steps"])

            # pod_constructor treats open=close fallback marks seamlessly for notional weighting.
            eff_open = np.where(np.isfinite(trade_open), trade_open, mark_close)
            eff_open = np.where(np.isfinite(eff_open), eff_open, 0.0)
            eff_close = np.where(np.isfinite(mark_close), mark_close, eff_open)
            net = gross - cost

            wallet_frames[coin] = pl.DataFrame(
                {
                    "timestamp": ts_utc,
                    "open": eff_open,
                    "close": eff_close,
                    "pos": pos,
                    "delta": delta,
                    "gross": gross,
                    "cost": cost,
                    "net": net,
                }
            )
        unit_frames[alias] = wallet_frames
        diagnostics[alias] = diag
    return unit_frames, diagnostics


def _is_reset_bar(ts_ms: int, reset_hour: int, reset_minute: int) -> bool:
    sec_in_day = (ts_ms // 1000) % 86_400
    target_sec = (reset_hour * 3600) + (reset_minute * 60)
    return target_sec <= sec_in_day < target_sec + 300


def _update_reentry_eligibility(
    desired: np.ndarray,
    sweep_snapshot: np.ndarray,
    coin_eligible: np.ndarray,
    *,
    reenter_mode: Literal["new_only", "immediate_target"],
    eps: float = 1e-9,
) -> np.ndarray:
    """Enforce strict new-position admission after a Guardian daily sweep."""
    if reenter_mode == "immediate_target":
        return desired.copy()

    out = np.zeros_like(desired)
    for c in range(desired.size):
        target = float(desired[c])
        if coin_eligible[c]:
            out[c] = target
            continue
        snap = float(sweep_snapshot[c])
        # Only re-admit after going flat (< eps) or cleanly flipping sign past zero.
        if abs(target) < eps or abs(snap) < eps or (snap * target < -1e-12):
            coin_eligible[c] = True
            out[c] = target
            sweep_snapshot[c] = 0.0
        else:
            out[c] = 0.0
    return out


def _split_exit_first_phase1(cur: float, desired: float, *, eps: float = 1e-12) -> float:
    """Return the exit/reduction-only intermediate position allowed before sweep checks."""
    if abs(desired) <= eps or abs(cur) <= eps or (cur * desired < -eps):
        return 0.0
    if abs(desired) < abs(cur):
        return desired
    return cur


def _conservative_gap_exit_price(
    cur: float,
    prev_m: float,
    m_c: float,
    haircut_rate: float,
) -> tuple[float, float]:
    """Return ``(ref_mark, exit_px)`` for a conservative forced-flatten when open trade price is missing."""
    direction = 1.0 if cur > 0 else -1.0
    if np.isfinite(prev_m) and prev_m > 0:
        ref_px = min(prev_m, m_c) if cur > 0 else max(prev_m, m_c)
        return prev_m, ref_px * (1.0 - direction * haircut_rate)
    return m_c, m_c * (1.0 - direction * haircut_rate)


def _conservative_gap_exit_pnl(
    cur: float,
    prev_m: float,
    m_c: float,
    haircut_rate: float,
    fee_rate: float,
) -> float:
    """Compute conservative forced-flatten PnL when ``trade_open`` is unavailable."""
    base_mark, exit_px = _conservative_gap_exit_price(cur, prev_m, m_c, haircut_rate)
    return cur * (exit_px - base_mark) - abs(cur) * exit_px * fee_rate


def _apply_fill_to_basis(
    cur_pos: float,
    new_pos: float,
    cur_entry: float,
    px: float,
    *,
    eps: float = 1e-12,
) -> tuple[float, float]:
    """Update weighted-average entry price and return closed realized PnL."""
    delta = new_pos - cur_pos
    if abs(delta) <= eps:
        return 0.0, (0.0 if abs(new_pos) <= eps else cur_entry)
    realized = 0.0
    if abs(cur_pos) > eps and cur_pos * delta < 0.0:
        close_qty = min(abs(cur_pos), abs(delta))
        direction = 1.0 if cur_pos > 0.0 else -1.0
        realized += close_qty * (px - cur_entry) * direction
    if abs(new_pos) <= eps:
        return realized, 0.0
    if abs(cur_pos) <= eps or cur_pos * new_pos < 0.0:
        return realized, px
    if cur_pos * delta > 0.0:
        return realized, (abs(cur_pos) * cur_entry + abs(delta) * px) / abs(new_pos)
    return realized, cur_entry


def _effective_allowance(cashday: float, fixed_limit: float, rate: float | None) -> float:
    """Compute dollar daily/sweep allowance from fixed amount or reset-basis percentage."""
    if rate is not None:
        return max(0.0, cashday * float(rate))
    return float(fixed_limit)


def _quantize_order_target(
    cur: float,
    want: float,
    exec_px: float,
    step: float,
    min_amt: float,
    min_notional: float,
    *,
    eps: float = 1e-12,
) -> float:
    """Apply Bybit lot-step rounding, min-amount, and notional deadband (bypassing full closes)."""
    if abs(want) <= eps:
        return 0.0
    raw_delta = want - cur
    if abs(raw_delta) <= eps:
        return cur
    if step > 0.0:
        sign = -1.0 if raw_delta < 0.0 else 1.0
        quotient = abs(raw_delta) / step
        units = math.floor(quotient + max(1e-12, quotient * 1e-12))
        qty_step = round(sign * units * step, 12)
    else:
        qty_step = raw_delta
    if min_amt > 0.0 and abs(qty_step) + 1e-12 < min_amt:
        return cur
    if min_notional > 0.0 and abs(qty_step) * exec_px + 1e-9 < min_notional:
        return cur
    desired = round(cur + qty_step, 12) if step > 0.0 else (cur + qty_step)
    return 0.0 if abs(desired) <= eps else desired


def _calc_gross_leverage(gross_notional: float, equity: float) -> float:
    if gross_notional <= 1e-12:
        return 0.0
    if equity <= 1e-9:
        return float("inf")
    return gross_notional / equity


def _resolve_funding_matrix(
    funding_rates: np.ndarray | None,
    timestamps_ms: np.ndarray,
    nc: int,
    n: int,
    spec: GuardianSpec,
) -> np.ndarray | None:
    """Resolve optional per-bar signed funding rate matrix ``(nc, n)`` scaled by multiplier."""
    if funding_rates is not None:
        fr = np.asarray(funding_rates, dtype=np.float64)
        if fr.ndim == 1 and fr.shape[0] == n:
            return np.broadcast_to(fr[None, :], (nc, n)) * spec.funding_multiplier
        if fr.ndim == 2 and fr.shape in ((1, n), (nc, n)):
            return np.broadcast_to(fr, (nc, n)) * spec.funding_multiplier
        raise ValueError(f"INVALID_FUNDING_RATES_SHAPE:{fr.shape}")
    if spec.funding_bps_per_8h == 0.0 or spec.funding_multiplier == 0.0:
        return None
    default_dt = int(timestamps_ms[1] - timestamps_ms[0]) if n > 1 else 300_000
    dt_ms = np.empty(n, dtype=np.float64)
    dt_ms[0] = float(default_dt)
    if n > 1:
        dt_ms[1:] = np.diff(timestamps_ms).astype(np.float64)
    per_bar = spec.funding_bps_per_8h * 1e-4 * (dt_ms / 28_800_000.0) * spec.funding_multiplier
    return np.broadcast_to(per_bar[None, :], (nc, n))


def _cap_phase2_leverage(
    pos: np.ndarray,
    prev_mark: np.ndarray,
    exec_px_arr: np.ndarray,
    p1_arr: np.ndarray,
    desired_arr: np.ndarray,
    priced: np.ndarray,
    step_arr: np.ndarray,
    min_amt_arr: np.ndarray,
    eq: float,
    outage_pnl: float,
    cost_rate: float,
    spec: GuardianSpec,
) -> None:
    """Scale Phase-2 entries/increases proportionally so open leverage stays within ``max_gross_leverage``."""
    if spec.max_gross_leverage is None:
        return
    nc = desired_arr.size
    eq_open_p1 = eq + outage_pnl
    used_notional = 0.0
    add_notional = 0.0
    for ci in range(nc):
        if not priced[ci]:
            continue
        cur = float(pos[ci])
        prev_m = float(prev_mark[ci])
        exec_px = exec_px_arr[ci]
        p1_c = p1_arr[ci]
        if np.isfinite(prev_m):
            eq_open_p1 += cur * (exec_px - prev_m)
        eq_open_p1 -= abs(p1_c - cur) * exec_px * cost_rate
        used_notional += abs(p1_c) * exec_px
        add_notional += abs(desired_arr[ci] - p1_c) * exec_px
    if add_notional <= 1e-12:
        return
    cap = max(0.0, float(spec.max_gross_leverage))
    max_allowed = cap * max(0.0, eq_open_p1)
    if used_notional + add_notional * (1.0 + cap * cost_rate) <= max_allowed + 1e-12:
        return
    avail = max(0.0, max_allowed - used_notional)
    scale = min(1.0, avail / (add_notional * (1.0 + cap * cost_rate)))
    for ci in range(nc):
        if not priced[ci]:
            continue
        p1_c = p1_arr[ci]
        delta2 = desired_arr[ci] - p1_c
        if abs(delta2) <= 1e-12:
            continue
        if scale <= 1e-12:
            desired_arr[ci] = p1_c
        else:
            desired_arr[ci] = _quantize_order_target(
                p1_c,
                p1_c + delta2 * scale,
                exec_px_arr[ci],
                float(step_arr[ci]),
                float(min_amt_arr[ci]),
                spec.min_trade_notional_usd,
            )


def _record_sweep_halt(
    pos: np.ndarray,
    sweep_snapshot: np.ndarray,
    raw_want: np.ndarray,
    coin_eligible: np.ndarray,
    swept_flags: np.ndarray,
    step_idx: int,
    reenter_mode: str,
    sweeps: int,
) -> tuple[int, bool]:
    """Flatten residual positions, snapshot targets, and lock re-entry after a sweep breach."""
    pos[:] = 0.0
    sweep_snapshot[:] = raw_want
    if reenter_mode == "new_only":
        coin_eligible[:] = False
    swept_flags[step_idx] = True
    return sweeps + 1, True


def _execute_guardian_pass(
    coins: Sequence[str],
    net_targets: np.ndarray,
    trade_open: np.ndarray,
    mark_close: np.ndarray,
    timestamps_ms: np.ndarray,
    spec: GuardianSpec,
    *,
    stop_on_terminal: bool,
    funding_rates: np.ndarray | None = None,
) -> dict:
    """Shared causal Guardian execution kernel used for both terminal and MC views."""
    nc, n = net_targets.shape
    cost_rate = spec.cost_bps * spec.cost_multiplier * 1e-4
    haircut_rate = GAP_EXIT_HAIRCUT_BPS * 1e-4
    fee_rate = DEFAULT_FEE_BPS * spec.cost_multiplier * 1e-4
    funding_mat = _resolve_funding_matrix(funding_rates, timestamps_ms, nc, n, spec)
    step_arr = np.array(
        [
            float(spec.amount_step_by_coin.get(c, spec.amount_step))
            if spec.amount_step_by_coin is not None
            else float(spec.amount_step)
            for c in coins
        ],
        dtype=np.float64,
    )
    min_amt_arr = np.array(
        [
            float(spec.min_amount_by_coin.get(c, spec.min_amount))
            if spec.min_amount_by_coin is not None
            else float(spec.min_amount)
            for c in coins
        ],
        dtype=np.float64,
    )

    pos = np.zeros(nc, dtype=np.float64)
    entry_px = np.zeros(nc, dtype=np.float64)
    prev_mark = np.full(nc, np.nan, dtype=np.float64)
    sweep_snapshot = np.zeros(nc, dtype=np.float64)
    coin_eligible = np.ones(nc, dtype=bool)

    eq = spec.initial_equity
    balance = spec.initial_equity
    cashday = spec.initial_equity
    eff_daily_limit = _effective_allowance(cashday, spec.daily_loss_limit, spec.daily_loss_rate)
    eff_sweep_loss = _effective_allowance(cashday, spec.guardian_sweep_loss, spec.guardian_sweep_rate)
    peak = spec.initial_equity
    min_eq = spec.initial_equity
    peak_to_trough_dd = 0.0
    worst_daily = 0.0
    sweeps = 0
    blocked_until_reset = False
    forced_gap_exits = 0
    unpriced_open_steps = 0
    deferred_notional_usd = 0.0
    peak_gross_leverage = 0.0
    total_funding_usd = 0.0

    verdict: Literal["PASS", "FAIL_STATIC", "FAIL_DAILY", "TIMEOUT"] = "TIMEOUT"
    terminal_step: int | None = None
    terminal_ts_ms: int | None = None
    terminal_net = 0.0

    step_pnl = np.zeros(n, dtype=np.float64)
    eq_curve = np.zeros(n, dtype=np.float64)
    swept_flags = np.zeros(n, dtype=bool)
    executed_deltas = np.zeros((nc, n), dtype=np.float64)

    for j in range(n):
        prev_pos = pos.copy()
        ts_ms = int(timestamps_ms[j])
        if j > 0 and _is_reset_bar(ts_ms, spec.reset_utc_hour, spec.reset_utc_minute):
            cashday = balance if spec.daily_loss_basis == "balance" else eq
            eff_daily_limit = _effective_allowance(
                cashday, spec.daily_loss_limit, spec.daily_loss_rate
            )
            eff_sweep_loss = _effective_allowance(
                cashday, spec.guardian_sweep_loss, spec.guardian_sweep_rate
            )
            blocked_until_reset = False
            worst_daily = min(worst_daily, eq - cashday)
            if terminal_step is None and (eq - cashday) <= -eff_daily_limit:
                verdict = "FAIL_DAILY"
                terminal_step = j
                terminal_ts_ms = ts_ms
                terminal_net = eq - spec.initial_equity
                if stop_on_terminal:
                    eq_curve[j:] = eq
                    break

        p = trade_open[:, j]
        m = mark_close[:, j]
        raw_want = net_targets[:, j]

        if not blocked_until_reset:
            admitted = _update_reentry_eligibility(
                raw_want,
                sweep_snapshot,
                coin_eligible,
                reenter_mode=spec.reenter_after_sweep,
            )
        else:
            admitted = np.zeros(nc, dtype=np.float64)

        priced = np.zeros(nc, dtype=bool)
        exec_px_arr = np.zeros(nc, dtype=np.float64)
        desired_arr = np.zeros(nc, dtype=np.float64)
        p1_arr = np.zeros(nc, dtype=np.float64)
        outage_pnl = 0.0
        open_mtm = 0.0

        for ci in range(nc):
            m_c = float(m[ci])
            t_c = float(p[ci])
            prev_m = float(prev_mark[ci])
            cur = float(pos[ci])

            if not np.isfinite(m_c):
                if abs(cur) > 1e-12:
                    unpriced_open_steps += 1
                    if np.isfinite(prev_m) and prev_m > 0:
                        direction = 1.0 if cur > 0 else -1.0
                        exit_px = prev_m * (1.0 - direction * haircut_rate)
                        outage_pnl += cur * (exit_px - prev_m) - abs(cur) * exit_px * fee_rate
                        realized, _ = _apply_fill_to_basis(cur, 0.0, float(entry_px[ci]), exit_px)
                        balance += realized - abs(cur) * exit_px * fee_rate
                        entry_px[ci] = 0.0
                        pos[ci] = 0.0
                        forced_gap_exits += 1
                continue

            priced[ci] = True
            exec_px = t_c if np.isfinite(t_c) else m_c
            exec_px_arr[ci] = exec_px
            if np.isfinite(t_c) and np.isfinite(prev_m) and abs(cur) > 1e-12:
                open_mtm += cur * (t_c - prev_m)

            want_c = float(admitted[ci])
            if blocked_until_reset:
                desired_c = 0.0
            elif abs(want_c - cur) > 1e-9 and not np.isfinite(t_c):
                desired_c = cur
                deferred_notional_usd += abs(want_c - cur) * m_c
            else:
                desired_c = _quantize_order_target(
                    cur,
                    want_c,
                    exec_px,
                    float(step_arr[ci]),
                    float(min_amt_arr[ci]),
                    spec.min_trade_notional_usd,
                )

            desired_arr[ci] = desired_c
            p1_arr[ci] = 0.0 if blocked_until_reset else _split_exit_first_phase1(cur, desired_c)

        bar_pnl = outage_pnl
        if not blocked_until_reset and (eq + outage_pnl + open_mtm - cashday) <= -eff_sweep_loss:
            # Open-time sweep breach: flatten carried positions at open without admitting entries.
            gross_open = 0.0
            for ci in range(nc):
                if not priced[ci]:
                    continue
                cur = float(pos[ci])
                prev_m = float(prev_mark[ci])
                m_c = float(m[ci])
                t_c = float(p[ci])
                if abs(cur) > 1e-12:
                    gross_open += abs(cur) * (t_c if np.isfinite(t_c) else m_c)
                    if np.isfinite(t_c):
                        if np.isfinite(prev_m):
                            bar_pnl += cur * (t_c - prev_m)
                        bar_pnl -= abs(cur) * t_c * cost_rate
                        realized, _ = _apply_fill_to_basis(cur, 0.0, float(entry_px[ci]), t_c)
                        balance += realized - abs(cur) * t_c * cost_rate
                    else:
                        _, exit_px = _conservative_gap_exit_price(cur, prev_m, m_c, haircut_rate)
                        bar_pnl += _conservative_gap_exit_pnl(
                            cur, prev_m, m_c, haircut_rate, fee_rate
                        )
                        realized, _ = _apply_fill_to_basis(cur, 0.0, float(entry_px[ci]), exit_px)
                        balance += realized - abs(cur) * exit_px * fee_rate
                        unpriced_open_steps += 1
                        forced_gap_exits += 1
                entry_px[ci] = 0.0
                pos[ci] = 0.0
                prev_mark[ci] = m_c
            peak_gross_leverage = max(
                peak_gross_leverage, _calc_gross_leverage(gross_open, eq + outage_pnl + open_mtm)
            )
            eq += bar_pnl
            sweeps, blocked_until_reset = _record_sweep_halt(
                pos,
                sweep_snapshot,
                raw_want,
                coin_eligible,
                swept_flags,
                j,
                spec.reenter_after_sweep,
                sweeps,
            )
        else:
            p1_pnl = outage_pnl
            p1_funding = 0.0
            if not blocked_until_reset:
                for ci in range(nc):
                    if not priced[ci]:
                        continue
                    cur = float(pos[ci])
                    prev_m = float(prev_mark[ci])
                    exec_px = exec_px_arr[ci]
                    m_c = float(m[ci])
                    p1_c = p1_arr[ci]
                    if np.isfinite(prev_m):
                        p1_pnl += cur * (exec_px - prev_m)
                    p1_pnl += p1_c * (m_c - exec_px) - abs(p1_c - cur) * exec_px * cost_rate
                    if funding_mat is not None and abs(p1_c) > 1e-12:
                        f_c = -p1_c * m_c * float(funding_mat[ci, j])
                        p1_pnl += f_c
                        p1_funding += f_c

            if not blocked_until_reset and (eq + p1_pnl - cashday) <= -eff_sweep_loss:
                # Exit-first sweep breach: exits/reductions settled, new opens/increases blocked.
                flatten_cost = 0.0
                gross_p1 = 0.0
                for ci in range(nc):
                    if priced[ci]:
                        cur = float(pos[ci])
                        exec_px = exec_px_arr[ci]
                        m_c = float(m[ci])
                        p1_c = p1_arr[ci]
                        realized1, new_entry1 = _apply_fill_to_basis(
                            cur, p1_c, float(entry_px[ci]), exec_px
                        )
                        balance += realized1 - abs(p1_c - cur) * exec_px * cost_rate
                        if funding_mat is not None and abs(p1_c) > 1e-12:
                            balance += -p1_c * m_c * float(funding_mat[ci, j])
                        realized2, _ = _apply_fill_to_basis(p1_c, 0.0, new_entry1, m_c)
                        balance += realized2 - abs(p1_c) * m_c * cost_rate
                        entry_px[ci] = 0.0
                        gross_p1 += abs(p1_c) * m_c
                        flatten_cost += abs(p1_c) * m_c * cost_rate
                        pos[ci] = 0.0
                        prev_mark[ci] = m_c
                peak_gross_leverage = max(
                    peak_gross_leverage, _calc_gross_leverage(gross_p1, eq + p1_pnl)
                )
                bar_pnl = p1_pnl - flatten_cost
                total_funding_usd += p1_funding
                eq += bar_pnl
                sweeps, blocked_until_reset = _record_sweep_halt(
                    pos,
                    sweep_snapshot,
                    raw_want,
                    coin_eligible,
                    swept_flags,
                    j,
                    spec.reenter_after_sweep,
                    sweeps,
                )
            else:
                if not blocked_until_reset:
                    _cap_phase2_leverage(
                        pos,
                        prev_mark,
                        exec_px_arr,
                        p1_arr,
                        desired_arr,
                        priced,
                        step_arr,
                        min_amt_arr,
                        eq,
                        outage_pnl,
                        cost_rate,
                        spec,
                    )

                bar_pnl = outage_pnl
                gross_close = 0.0
                bar_funding = 0.0
                for ci in range(nc):
                    if not priced[ci]:
                        continue
                    cur = float(pos[ci])
                    prev_m = float(prev_mark[ci])
                    exec_px = exec_px_arr[ci]
                    m_c = float(m[ci])
                    p1_c = p1_arr[ci]
                    desired_c = desired_arr[ci]
                    if blocked_until_reset and abs(cur) > 1e-12 and not np.isfinite(float(p[ci])):
                        _, exit_px = _conservative_gap_exit_price(cur, prev_m, m_c, haircut_rate)
                        bar_pnl += _conservative_gap_exit_pnl(
                            cur, prev_m, m_c, haircut_rate, fee_rate
                        )
                        realized, _ = _apply_fill_to_basis(cur, 0.0, float(entry_px[ci]), exit_px)
                        balance += realized - abs(cur) * exit_px * fee_rate
                        entry_px[ci] = 0.0
                        unpriced_open_steps += 1
                        forced_gap_exits += 1
                        pos[ci] = 0.0
                        prev_mark[ci] = m_c
                        continue
                    if np.isfinite(prev_m):
                        bar_pnl += cur * (exec_px - prev_m)
                    trade_cost = (abs(p1_c - cur) + abs(desired_c - p1_c)) * exec_px * cost_rate
                    bar_pnl += desired_c * (m_c - exec_px) - trade_cost
                    realized, new_entry = _apply_fill_to_basis(
                        cur, desired_c, float(entry_px[ci]), exec_px
                    )
                    balance += realized - trade_cost
                    if funding_mat is not None and abs(desired_c) > 1e-12:
                        f_c = -desired_c * m_c * float(funding_mat[ci, j])
                        bar_pnl += f_c
                        bar_funding += f_c
                        balance += f_c
                    entry_px[ci] = new_entry
                    pos[ci] = desired_c
                    prev_mark[ci] = m_c
                    gross_close += abs(desired_c) * m_c

                total_funding_usd += bar_funding
                eq += bar_pnl
                peak_gross_leverage = max(
                    peak_gross_leverage, _calc_gross_leverage(gross_close, eq)
                )
                if not blocked_until_reset and (eq - cashday) <= -eff_sweep_loss:
                    valid_open = priced & (np.abs(pos) > 1e-12)
                    flatten_cost = float(np.sum(np.abs(pos[valid_open]) * m[valid_open])) * cost_rate
                    for ci in np.flatnonzero(valid_open):
                        m_c = float(m[ci])
                        cur_open = float(pos[ci])
                        realized, _ = _apply_fill_to_basis(
                            cur_open, 0.0, float(entry_px[ci]), m_c
                        )
                        balance += realized - abs(cur_open) * m_c * cost_rate
                        entry_px[ci] = 0.0
                    bar_pnl -= flatten_cost
                    eq -= flatten_cost
                    sweeps, blocked_until_reset = _record_sweep_halt(
                        pos,
                        sweep_snapshot,
                        raw_want,
                        coin_eligible,
                        swept_flags,
                        j,
                        spec.reenter_after_sweep,
                        sweeps,
                    )

        executed_deltas[:, j] = pos - prev_pos
        step_pnl[j] = bar_pnl
        eq_curve[j] = eq
        peak = max(peak, eq)
        min_eq = min(min_eq, eq)
        peak_to_trough_dd = min(peak_to_trough_dd, eq - peak)
        worst_daily = min(worst_daily, eq - cashday)

        hit_pass = (eq - spec.initial_equity) >= spec.profit_target
        hit_static = eq <= (spec.initial_equity - spec.max_static_dd)
        hit_daily = (eq - cashday) <= -eff_daily_limit

        if terminal_step is None and (hit_pass or hit_static or hit_daily):
            if hit_static:
                verdict = "FAIL_STATIC"
            elif hit_daily:
                verdict = "FAIL_DAILY"
            else:
                verdict = "PASS"
            terminal_step = j
            terminal_ts_ms = ts_ms
            terminal_net = eq - spec.initial_equity
            if stop_on_terminal:
                if j + 1 < n:
                    eq_curve[j + 1 :] = eq
                break

    if terminal_step is None:
        terminal_net = eq - spec.initial_equity

    return {
        "verdict": verdict,
        "terminal_step": terminal_step,
        "terminal_ts_ms": terminal_ts_ms,
        "terminal_net_usd": round(terminal_net, 2),
        "min_equity_usd": round(min_eq, 2),
        "static_floor_dd_usd": round(min(0.0, min_eq - spec.initial_equity), 2),
        "peak_to_trough_dd_usd": round(peak_to_trough_dd, 2),
        "worst_daily_usd": round(worst_daily, 2),
        "sweeps": sweeps,
        "forced_gap_exits": forced_gap_exits,
        "unpriced_open_steps": unpriced_open_steps,
        "deferred_notional_usd": round(deferred_notional_usd, 2),
        "full_window_net_usd": round(eq - spec.initial_equity, 2),
        "peak_gross_leverage": peak_gross_leverage,
        "total_funding_usd": round(total_funding_usd, 4),
        "step_pnl": step_pnl,
        "eq_curve": eq_curve,
        "swept_flags": swept_flags,
        "executed_deltas": executed_deltas,
    }


def extract_mc_daily_blocks(
    step_pnl: np.ndarray,
    swept_flags: np.ndarray,
    timestamps_ms: np.ndarray,
    spec: GuardianSpec,
    *,
    chop_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Partition continuous 5m PnL into reset-aligned daily blocks for ``pod_constructor`` MC."""
    n = int(timestamps_ms.size)
    reset_indices = [
        j
        for j in range(n)
        if _is_reset_bar(int(timestamps_ms[j]), spec.reset_utc_hour, spec.reset_utc_minute)
    ]
    if not reset_indices:
        raise ValueError("NO_DAILY_RESET_BOUNDARIES_FOR_MC")

    starts = reset_indices
    if starts[0] > 0:
        # Align cleanly to UTC reset boundary rather than partial pre-reset sliver.
        pass

    day_chop: list[bool] = []
    day_end: list[float] = []
    day_max: list[float] = []
    day_min: list[float] = []
    day_swp: list[int] = []
    day_mf: list[bool] = []

    step_bars = 288
    for idx_pos, s_idx in enumerate(starts):
        next_start = starts[idx_pos + 1] if idx_pos + 1 < len(starts) else (s_idx + step_bars)
        e_idx = min(n, next_start)
        if (e_idx - s_idx) < (step_bars - 12):
            continue
        seg = step_pnl[s_idx:e_idx]
        cum = np.cumsum(seg)
        day_end.append(float(cum[-1]))
        day_max.append(float(np.max(cum)))
        day_min.append(float(np.min(cum)))
        day_swp.append(int(np.any(swept_flags[s_idx:e_idx])))
        day_mf.append(bool(np.argmax(cum) < np.argmin(cum)))
        if chop_mask is not None:
            day_chop.append(bool(np.mean(chop_mask[s_idx:e_idx]) >= 0.5))
        else:
            day_chop.append(False)

    return (
        np.asarray(day_chop, dtype=bool),
        np.asarray(day_end, dtype=np.float64),
        np.asarray(day_max, dtype=np.float64),
        np.asarray(day_min, dtype=np.float64),
        np.asarray(day_swp, dtype=np.int32),
        np.asarray(day_mf, dtype=bool),
    )


def simulate_guardian_challenge(
    coins: Sequence[str],
    net_targets: np.ndarray,
    trade_open: np.ndarray,
    mark_close: np.ndarray,
    timestamps_ms: np.ndarray,
    spec: GuardianSpec = GuardianSpec(),
    *,
    chop_mask: np.ndarray | None = None,
    funding_rates: np.ndarray | None = None,
) -> tuple[ChallengeRunResult, tuple]:
    """Run causal Guardian simulation returning both frozen terminal verdict and MC blocks."""
    term = _execute_guardian_pass(
        coins,
        net_targets,
        trade_open,
        mark_close,
        timestamps_ms,
        spec,
        stop_on_terminal=True,
        funding_rates=funding_rates,
    )
    if term["terminal_step"] is None:
        cont = term
    else:
        cont = _execute_guardian_pass(
            coins,
            net_targets,
            trade_open,
            mark_close,
            timestamps_ms,
            spec,
            stop_on_terminal=False,
            funding_rates=funding_rates,
        )
    blocks = extract_mc_daily_blocks(
        cont["step_pnl"],
        cont["swept_flags"],
        timestamps_ms,
        spec,
        chop_mask=chop_mask,
    )
    result = ChallengeRunResult(
        verdict=term["verdict"],
        terminal_step=term["terminal_step"],
        terminal_ts_ms=term["terminal_ts_ms"],
        terminal_net_usd=term["terminal_net_usd"],
        min_equity_usd=term["min_equity_usd"],
        static_floor_dd_usd=term["static_floor_dd_usd"],
        peak_to_trough_dd_usd=term["peak_to_trough_dd_usd"],
        worst_daily_usd=term["worst_daily_usd"],
        sweeps=term["sweeps"],
        forced_gap_exits=term["forced_gap_exits"],
        unpriced_open_steps=term["unpriced_open_steps"],
        deferred_notional_usd=term["deferred_notional_usd"],
        full_window_net_usd=cont["full_window_net_usd"],
        full_window_peak_to_trough_dd_usd=cont["peak_to_trough_dd_usd"],
        full_window_static_floor_dd_usd=cont["static_floor_dd_usd"],
        full_window_sweeps=cont["sweeps"],
        step_pnl=cont["step_pnl"],
        equity_curve=cont["eq_curve"],
        peak_gross_leverage=term["peak_gross_leverage"],
        full_window_peak_gross_leverage=cont["peak_gross_leverage"],
        total_funding_usd=term["total_funding_usd"],
        full_window_funding_usd=cont["total_funding_usd"],
        executed_deltas=cont["executed_deltas"],
    )
    return result, blocks


def net_member_targets(member_pos: np.ndarray, k_weights: Sequence[float]) -> np.ndarray:
    """Net member target positions across coins (M, C, T) weighted by ``k_weights``."""
    w = np.asarray(k_weights, dtype=np.float64)[:, None, None]
    return np.sum(member_pos * w, axis=0)


def reconcile_portfolio_with_pm03(
    coins: Sequence[str],
    timestamps_ms: np.ndarray,
    net_targets: np.ndarray,
    trade_open: np.ndarray,
    mark_close: np.ndarray,
    *,
    initial_equity: float = 10_000.0,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    pm_name: str = "pod",
) -> dict:
    """Validate Tier-1 netted portfolio execution against canonical Decimal ``PM03 Account``."""
    instruments = tuple(Instrument.parse(f"hyperliquid/mainnet:perp:default/{c}") for c in coins)
    start_iso = datetime.fromtimestamp(int(timestamps_ms[0]) / 1000.0, tz=UTC).isoformat()
    rb = Rulebook(
        program="CANONICAL_REPLAY_RECONCILIATION",
        stage="research_tier2",
        source="LOCAL_1M_OHLCV_PROXY",
        observed_at=start_iso,
        version="reconcile-v1",
        timezone="UTC",
        initial_balance=D(str(initial_equity)),
        profit_target=None,
        daily_loss=None,
        static_loss=None,
        trailing_loss=None,
        permitted=instruments,
        confirmed=True,
        account_terms_confirmed=False,
    )
    account = Account.start(rb, start_iso)
    costs = CostModel(
        fee_bps=D(str(fee_bps)),
        full_spread_bps=D(0),
        slippage_bps=D(str(slippage_bps)),
        impact_bps_at_full_participation=D(0),
        max_participation=D(1),
        max_age_seconds=300,
        latency_seconds=0,
    )

    fill_count = 0
    for j in range(int(timestamps_ms.size)):
        ts_ms = int(timestamps_ms[j])
        decision_iso = datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC).isoformat()
        close_iso = datetime.fromtimestamp((ts_ms + 240_000) / 1000.0, tz=UTC).isoformat()

        observations: dict[Instrument, MarketObservation] = {}
        close_marks: dict[Instrument, D] = {}
        desired_book: dict[Instrument, D] = {}

        for ci, inst in enumerate(instruments):
            m_c = float(mark_close[ci, j])
            t_c = float(trade_open[ci, j])
            if not np.isfinite(m_c):
                continue
            exec_px = t_c if np.isfinite(t_c) else m_c
            desired_qty = D(str(round(float(net_targets[ci, j]), 12)))
            cur_qty = account.quantities.get(inst, D(0))
            if abs(float(desired_qty - cur_qty)) > 1e-9 and not np.isfinite(t_c):
                desired_qty = cur_qty
            desired_book[inst] = desired_qty
            observations[inst] = MarketObservation(
                instrument=inst,
                event_at=decision_iso,
                first_known_at=decision_iso,
                mark=D(str(exec_px)),
                volume_base=D(1_000_000_000),
                quality="OHLCV_1M_PROXY",
                supported=True,
            )
            close_marks[inst] = D(str(m_c))

        report = execute_targets(
            account,
            {pm_name: desired_book},
            observations,
            costs,
            decision_at=decision_iso,
            hypothetical_fill_at=decision_iso,
            permitted=True,
        )
        fill_count += len(report["fills"])
        account.mark(close_marks, close_iso)

    account.verify()
    return {
        "final_equity": float(account.equity),
        "net_pnl": float(account.equity - D(str(initial_equity))),
        "realized_pnl": float(account.realized_pnl),
        "total_costs": float(account.costs),
        "fill_count": fill_count,
        "quantities": {inst.symbol: float(qty) for inst, qty in account.quantities.items()},
        "halted": account.breached,
        "account_verified": True,
    }
