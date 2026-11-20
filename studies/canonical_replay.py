"""Research tape reconstruction: wallet records, symbol screens and unit replays.

This module is a *research extractor*. It reconstructs what a leader wallet did on
a 5m grid and normalizes it for ``pod_constructor``. It decides nothing:

* no Guardian lifecycle, floor, sweep or re-entry simulation,
* no mandate/risk clamping, no netting, no PM03 accounting reconciliation,
* no wallet-registry universe discovery (that registry was archived in Phase D).

Decisions are ``copytrade_kernel.step_checkpoint`` and seat membership is
``copytrade_pm.roster.reconcile_roster``, both driven by
``trading_ops.research`` (see ``docs/adr/0004-slice4-kernel-research-path.md``).

Public surface:

1. ``WalletRecord`` — the wallet label/identity row the studies pass around.
2. ``assess_bybit_symbol_screen`` — materiality-aware symbol mapping and alias flags.
3. ``reconstruct_positions_on_grid`` — causal ``new`` vs anchored ``full`` replay with
   zero pre-first-fill backward lookahead.
4. ``replay_coin_targets_vectorized`` / ``build_unit_replay_by_alias`` —
   execution-aligned 5m frames with pair-local gap termination.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import numpy as np
import polars as pl

PolicyType = Literal["new", "full"]
DEFAULT_COST_BPS = 6.5
DEFAULT_FEE_BPS = 4.5
DEFAULT_SLIPPAGE_BPS = 2.0
GAP_EXIT_HAIRCUT_BPS = 50.0


@dataclass(frozen=True)
class WalletRecord:
    """Read-only wallet metadata from frozen ``legacy_wallet_registry_20261119.sqlite``.

    Live research uses ``trading_ops.research`` over ``FactStore`` and ``RosterStore``.
    """

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


def _parse_reset_hhmm(value: str | None) -> tuple[int, int]:
    """Parse canonical ``HH:MM`` UTC reset schedule, defaulting UNKNOWN to 00:00."""
    if value is None:
        return 0, 0
    raw = str(value).strip()
    if not raw or raw.upper() == "UNKNOWN":
        return 0, 0
    if ":" not in raw:
        raise ValueError(f"INVALID_RESET_TIME:{value}")
    hh_s, mm_s = raw.split(":", 1)
    if not (hh_s.isdigit() and mm_s.isdigit() and len(hh_s) == 2 and len(mm_s) == 2):
        raise ValueError(f"INVALID_RESET_TIME:{value}")
    hh, mm = int(hh_s), int(mm_s)
    if not (0 <= hh < 24 and 0 <= mm < 60):
        raise ValueError(f"INVALID_RESET_TIME:{value}")
    return hh, mm


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


def _parse_utc_us(value: str | datetime) -> int:
    """Parse an ISO-8601 UTC timestamp or tz-aware datetime into integer UTC microseconds."""
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("UTC_OFFSET_REQUIRED")
        dt = value.astimezone(UTC)
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.strip())
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("UTC_OFFSET_REQUIRED")
        dt = parsed.astimezone(UTC)
    else:
        raise TypeError(f"Unsupported timestamp type: {type(value).__name__}")
    return round(dt.timestamp() * 1_000_000.0)


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
    notional_frac = (
        (unmapped_notional / total_notional) if has_notional and total_notional > 0 else None
    )

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
        if not unanchored:
            pre_first_fill = float(initial_anchor_pos) if initial_anchor_pos is not None else 0.0
            pos = np.where(ix >= 0, post[np.clip(ix, 0, len(times) - 1)], pre_first_fill)
        else:
            eff_post = np.zeros(len(times), dtype=np.float64)
            synced = False
            prev = first_pre
            for i in range(len(times)):
                after = float(post[i])
                if not synced:
                    is_same_side_inc = (prev * after > 1e-12) and (abs(after) > abs(prev) + eps)
                    is_fresh_open = (abs(prev) <= eps) and (abs(after) > eps)
                    is_flip = (prev * after) < -1e-12
                    if is_same_side_inc or is_fresh_open or is_flip:
                        synced = True
                eff_post[i] = after if synced else 0.0
                prev = after
            pos = np.where(ix >= 0, eff_post[np.clip(ix, 0, len(times) - 1)], 0.0)
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


def _apply_pair_outage_gap_exit(
    cur_pos: float,
    want: float,
    prev_mark: float,
    haircut_rate: float,
    fee_rate: float,
) -> tuple[float, float, float, float, int, int, float | None]:
    if abs(cur_pos) <= 1e-12:
        return cur_pos, 0.0, 0.0, 0.0, 0, 0, None
    snap_val = want if abs(want) > 1e-9 else cur_pos
    if np.isfinite(prev_mark) and prev_mark > 0:
        direction = 1.0 if cur_pos > 0 else -1.0
        exit_px = prev_mark * (1.0 - direction * haircut_rate)
        bar_gross = cur_pos * (exit_px - prev_mark)
        bar_cost = abs(cur_pos) * exit_px * fee_rate
        return 0.0, -cur_pos, bar_gross, bar_cost, 1, 1, snap_val
    return cur_pos, 0.0, 0.0, 0.0, 0, 1, None


def replay_coin_targets_vectorized(
    targets: np.ndarray,
    trade_open: np.ndarray,
    mark_close: np.ndarray,
    *,
    cost_bps: float = DEFAULT_COST_BPS,
    gap_exit_haircut_bps: float = GAP_EXIT_HAIRCUT_BPS,
    fee_bps: float = DEFAULT_FEE_BPS,
    initial_pos: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Simulate causal per-bar fills, fees, and pair-local stale-mark exits on a 1-coin target.

    Target chasing only: the CWSA one-shot policy is a kernel decision and lives in
    ``copytrade_kernel.step_checkpoint`` / ``trading_ops.research.run_kernel_backtest``.
    """
    n = int(targets.size)
    cost_rate = cost_bps * 1e-4
    haircut_rate = gap_exit_haircut_bps * 1e-4
    fee_rate = fee_bps * 1e-4

    pos = np.zeros(n, dtype=np.float64)
    delta = np.zeros(n, dtype=np.float64)
    gross = np.zeros(n, dtype=np.float64)
    cost = np.zeros(n, dtype=np.float64)

    cur_pos = float(initial_pos)
    prev_mark = np.nan
    outage_fenced_snap: float | None = None
    forced_exits = 0
    unpriced_open_steps = 0
    deferred_notional_usd = 0.0

    for j in range(n):
        m_close = float(mark_close[j])
        t_open = float(trade_open[j])
        want = float(targets[j])

        if not np.isfinite(m_close):
            cur_pos, d_gap, g_gap, c_gap, f_inc, u_inc, snap_new = _apply_pair_outage_gap_exit(
                cur_pos, want, prev_mark, haircut_rate, fee_rate
            )
            delta[j] = d_gap
            gross[j] = g_gap
            cost[j] = c_gap
            forced_exits += f_inc
            unpriced_open_steps += u_inc
            if snap_new is not None:
                outage_fenced_snap = snap_new
            pos[j] = cur_pos
            continue

        if outage_fenced_snap is not None:
            if abs(want) <= 1e-9 or (want * outage_fenced_snap < -1e-12):
                outage_fenced_snap = None
            else:
                want = 0.0

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

            anchor_val = anchors.get(coin) if policy == "full" else None
            pos, delta, gross, cost, stats = replay_coin_targets_vectorized(
                recon.pos,
                trade_open,
                mark_close,
                cost_bps=cost_bps,
                initial_pos=float(anchor_val) if anchor_val is not None else 0.0,
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
