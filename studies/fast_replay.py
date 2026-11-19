"""Vectorised drop-in replacements for the two Python loops in canonical replay.

`canonical_replay.reconstruct_positions_on_grid` walks every fill in Python and
`replay_coin_targets_vectorized` walks every 5m bar in Python. At 197 days x 261
wallets x ~20 coins that is ~150M scalar iterations, which is why a full-window
replay takes 14 minutes and never leaves one core busy.

Both are pure state machines over numpy arrays, so both can be expressed as
shifted array arithmetic:

  eligibility  start = flat-to-open or flip, stop = back to flat,
               active = cumsum(start) > cumsum(stop), gated by last start >= activation
  bar replay   gross[j] = pos[j-1]*(open[j]-close[j-1]) + pos[j]*(close[j]-open[j])
               cost[j]  = |pos[j]-pos[j-1]| * open[j] * cost_rate

A non-finite mark or open only forces the canonical slow path when a position is
carried into the unpriced bar, because gap handling then carries state
(`outage_fenced_snap`) across bars. Flat in, flat out is provably a no-op and
stays vectorised.

`python fast_replay.py` verifies exact equality against canonical on live data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

STUDIES_DIR = Path(__file__).resolve().parent
if str(STUDIES_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIES_DIR))

EPS = 1e-9


def eligible_new_policy(
    ts: np.ndarray, pre: np.ndarray, sign: np.ndarray, sz: np.ndarray, act_ms: int
) -> np.ndarray:
    """Per-fill eligibility for policy="new", matching canonical_replay exactly."""
    ts = np.asarray(ts, dtype=np.int64)
    pre = np.asarray(pre, dtype=np.float64)
    after = pre + np.asarray(sign, dtype=np.float64) * np.asarray(sz, dtype=np.float64)
    prev = np.empty_like(after)
    prev[0] = pre[0]
    prev[1:] = after[:-1]

    start = ((np.abs(prev) < EPS) & (np.abs(after) >= EPS)) | (prev * after < -1e-12)
    stop = np.abs(after) < EPS

    # Canonical `active` is a latch, not a counter: repeated flat-to-flat fills
    # must not accumulate against it, so track the most recent start/stop event.
    # Canonical applies the start test first and the stop test second, so a fill
    # that both flips and lands back at flat (residual < EPS) ends inactive.
    n = after.size
    is_event = start | stop
    last_event = np.maximum.accumulate(np.where(is_event, np.arange(n), -1))
    active = np.zeros(n, dtype=bool)
    has_event = last_event >= 0
    active[has_event] = start[last_event[has_event]] & ~stop[last_event[has_event]]

    last_start = np.maximum.accumulate(np.where(start, ts, np.iinfo(np.int64).min))
    return active & (last_start >= int(act_ms))


def positions_on_grid(
    fills_ts: np.ndarray,
    pre: np.ndarray,
    sign: np.ndarray,
    sz: np.ndarray,
    grid_ms: np.ndarray,
    act_ms: int,
) -> np.ndarray:
    """Causal 5m target positions under policy="new", vectorised."""
    if len(fills_ts) == 0:
        return np.zeros(grid_ms.size, dtype=np.float64)
    ts = np.asarray(fills_ts, dtype=np.int64)
    after = np.asarray(pre, np.float64) + np.asarray(sign, np.float64) * np.asarray(sz, np.float64)
    ok = eligible_new_policy(ts, pre, sign, sz, act_ms)
    ix = np.searchsorted(ts, grid_ms, side="left") - 1
    valid = ix >= 0
    clipped = np.clip(ix, 0, len(ts) - 1)
    return np.where(valid & ok[clipped], after[clipped], 0.0)


def replay_fast(
    targets: np.ndarray,
    trade_open: np.ndarray,
    mark_close: np.ndarray,
    *,
    cost_bps: float,
    fee_bps: float,
    gap_exit_haircut_bps: float = 50.0,
) -> np.ndarray | None:
    """Vectorised legacy bar replay with event-compressed gap handling."""
    t_arr = np.asarray(targets, dtype=np.float64)
    op = np.asarray(trade_open, dtype=np.float64)
    cl = np.asarray(mark_close, dtype=np.float64)
    n = t_arr.size
    if n == 0:
        return np.empty(0, dtype=np.float64)

    cl_ok = np.isfinite(cl)
    op_ok = np.isfinite(op)
    cl_gap = ~cl_ok
    op_gap = ~op_ok & cl_ok
    has_cl_gap = bool(cl_gap.any())
    has_op_gap = bool(op_gap.any())

    cl_eff = _ffill(cl) if has_cl_gap else cl
    prev_c = np.empty_like(cl_eff)
    prev_c[0] = np.nan
    prev_c[1:] = cl_eff[:-1]
    prev_t = np.empty_like(t_arr)
    prev_t[0] = 0.0
    prev_t[1:] = t_arr[:-1]

    need_cl_state = has_cl_gap and not (
        np.all(t_arr[cl_gap] == 0.0) and np.all(prev_t[cl_gap] == 0.0)
    )
    need_op_state = has_op_gap and not np.all(t_arr[op_gap] == prev_t[op_gap])

    if not need_cl_state and not need_op_state:
        op_eff = np.where(op_ok, op, cl_eff)
        carry = np.where(np.isfinite(prev_c), prev_t * (op_eff - np.nan_to_num(prev_c)), 0.0)
        gross = carry + t_arr * (cl_eff - op_eff)
        cost = np.abs(t_arr - prev_t) * op_eff * (cost_bps * 1e-4)
        net = gross - cost
        if has_cl_gap:
            net[cl_gap] = 0.0
        return net

    t_change = np.empty(n, dtype=bool)
    t_change[0] = t_arr[0] != 0.0
    t_change[1:] = t_arr[1:] != t_arr[:-1]
    after_gap = np.empty(n, dtype=bool)
    after_gap[0] = False
    after_gap[1:] = cl_gap[:-1] | op_gap[:-1]
    ev_idx = np.flatnonzero(t_change | cl_gap | op_gap | after_gap)

    pos = t_arr.copy()
    cur_pos = 0.0
    snap: float | None = None
    prev_ev = 0
    for idx_pos in range(ev_idx.size):
        j = int(ev_idx[idx_pos])
        if j > prev_ev:
            pos[prev_ev:j] = cur_pos
        if cl_gap[j]:
            if abs(cur_pos) > 1e-12:
                pm = float(prev_c[j])
                if np.isfinite(pm) and pm > 0.0:
                    w = float(t_arr[j])
                    snap = w if abs(w) > 1e-9 else cur_pos
                    cur_pos = 0.0
            pos[j] = cur_pos
            prev_ev = j + 1
            continue
        w = float(t_arr[j])
        if snap is not None:
            if abs(w) <= 1e-9 or (w * snap < -1e-12):
                snap = None
            else:
                w = 0.0
        desired = cur_pos if (abs(w - cur_pos) > 1e-9 and not op_ok[j]) else w
        pos[j] = desired
        cur_pos = desired
        prev_ev = j + 1
    if prev_ev < n:
        pos[prev_ev:n] = cur_pos

    if isinstance(targets, np.ndarray) and targets.flags.writeable and pos is not targets:
        targets[:] = pos
    prev_pos = np.empty_like(pos)
    prev_pos[0] = 0.0
    prev_pos[1:] = pos[:-1]

    op_eff = np.where(op_ok, op, cl_eff)
    carry = np.where(np.isfinite(prev_c), prev_pos * (op_eff - np.nan_to_num(prev_c)), 0.0)
    gross = carry + pos * (cl_eff - op_eff)
    cost = np.abs(pos - prev_pos) * op_eff * (cost_bps * 1e-4)
    net = gross - cost

    if has_cl_gap:
        net[cl_gap] = 0.0
        exit_mask = cl_gap & (np.abs(prev_pos) > 1e-12) & np.isfinite(prev_c) & (prev_c > 0.0)
        if exit_mask.any():
            cp = prev_pos[exit_mask]
            pm = prev_c[exit_mask]
            direction = np.where(cp > 0.0, 1.0, -1.0)
            exit_px = pm * (1.0 - direction * (gap_exit_haircut_bps * 1e-4))
            g_gap = cp * (exit_px - pm)
            c_gap = np.abs(cp) * exit_px * (fee_bps * 1e-4)
            net[exit_mask] = g_gap - c_gap
    return net


def _ffill(a: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs; leading NaNs stay NaN."""
    idx = np.maximum.accumulate(np.where(np.isfinite(a), np.arange(a.size), -1))
    out = np.full(a.size, np.nan)
    ok = idx >= 0
    out[ok] = a[idx[ok]]
    return out


def verify(
    sample: int = 40,
    start_iso: str = "2026-05-01T00:00:00Z",
    end_iso: str = "2026-11-14T00:00:00Z",
    seed: int = 7,
    full_eligibility: bool = False,
) -> int:
    """Compare against canonical on real (alias, coin) chains; nonzero exit on mismatch.

    ``full_eligibility`` sweeps every chain (2.29M fills over the reference window)
    instead of the sampled replay check, reproducing the eligibility claim in
    LESSONS.md rather than only asserting it.
    """
    import random
    from datetime import UTC, datetime

    import canonical_replay as cr
    import study_data as sd

    start = sd.to_utc_ms(datetime.fromisoformat(start_iso).astimezone(UTC))
    end = sd.to_utc_ms(datetime.fromisoformat(end_iso).astimezone(UTC))
    paths, _ = sd.discover_fill_archive(start, end)
    records, _ = cr.load_registry_universe(
        Path.home() / ".local/share/copytrade/registry/wallet_registry.sqlite",
        include_excluded=True,
    )
    ing = sd.ingest_fills(records, paths, start=start, end=end, split=start)
    grid = start + np.arange((end - start) // 300_000, dtype=np.int64) * 300_000
    prices = sd.LazyPrices(
        grid, store_root=sd.DEFAULT_STORE_ROOT, vike_root=sd.DEFAULT_VIKE_ROOT, split_ms=start
    )

    chains = [
        (alias, coin, df)
        for alias, by in ing.grouped.items()
        for coin, df in by.items()
    ]
    random.Random(seed).shuffle(chains)

    def canon_eligible(ts: np.ndarray, pre: np.ndarray, sign: np.ndarray, sz: np.ndarray) -> np.ndarray:
        """Canonical scalar loop, copied from reconstruct_positions_on_grid(policy='new')."""
        after = pre + sign * sz
        n = len(ts)
        el = np.zeros(n, dtype=bool)
        active = False
        prev = float(pre[0])
        for i in range(n):
            a = float(after[i])
            if ts[i] < start:
                prev = a
                continue
            if (abs(prev) < EPS and abs(a) >= EPS) or (prev * a < -1e-12):
                active = True
            if abs(a) < EPS:
                active = False
            el[i] = active
            prev = a
        return el

    if full_eligibility:
        bad = 0
        fills = 0
        for _alias, _coin, df in chains:
            ts = df["ts"].to_numpy().astype(np.int64)
            pre = df["pre"].to_numpy().astype(np.float64)
            sign = df["sign"].to_numpy().astype(np.float64)
            sz = df["sz"].to_numpy().astype(np.float64)
            fills += ts.size
            if not np.array_equal(
                canon_eligible(ts, pre, sign, sz),
                eligible_new_policy(ts, pre, sign, sz, start),
            ):
                bad += 1
        print(f"eligibility sweep: chains={len(chains)} fills={fills} mismatched={bad}")
        return 0 if bad == 0 else 1

    checked = 0
    worst_pos = 0.0
    worst_net = 0.0
    gap_fallbacks = 0
    nan_bad = 0
    for alias, coin, df in chains:
        if checked >= sample:
            break
        try:
            t_open, m_close = prices.get(coin)
        except (KeyError, ValueError, OSError):
            continue
        recon = cr.reconstruct_positions_on_grid(df, grid, policy="new", activation_ms=start)
        mine = positions_on_grid(
            df["ts"].to_numpy(), df["pre"].to_numpy(), df["sign"].to_numpy(),
            df["sz"].to_numpy(), grid, start,
        )
        worst_pos = max(worst_pos, float(np.max(np.abs(recon.pos - mine))))

        frames, _ = cr.build_unit_replay_by_alias(
            {alias: {coin: df}}, {coin: (t_open, m_close)}, [alias], grid,
            policy="new", activation_ms=start,
        )
        ref_net = frames[alias][coin]["net"].to_numpy() if frames.get(alias) else None
        if ref_net is None:
            continue
        fast = replay_fast(mine, t_open, m_close,
                           cost_bps=cr.DEFAULT_COST_BPS, fee_bps=cr.DEFAULT_FEE_BPS)
        if fast is None:
            gap_fallbacks += 1
            checked += 1
            continue
        diff = np.abs(ref_net - fast)
        if not np.isfinite(diff).all():
            # NaN/inf would be swallowed by max(); treat it as a hard mismatch
            nan_bad += 1
            worst_net = float("inf")
        else:
            worst_net = max(worst_net, float(diff.max()) if diff.size else 0.0)
        checked += 1

    print(f"chains checked={checked} gap_fallbacks={gap_fallbacks} nan_bad={nan_bad}")
    print(f"max |pos diff| = {worst_pos:.3e}   max |net diff| = {worst_net:.3e} USD")
    ok = worst_pos == 0.0 and worst_net == 0.0 and nan_bad == 0 and checked > 0
    if gap_fallbacks:
        print(f"note: {gap_fallbacks} chains fell back to canonical and are not covered by this check")
    print("EQUAL" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(verify(full_eligibility="--full" in sys.argv))
