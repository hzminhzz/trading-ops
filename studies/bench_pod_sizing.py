"""Fixed-workload benchmark and correctness evaluator for pod discovery and sizing."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

STUDIES_DIR = Path(__file__).resolve().parent
if str(STUDIES_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIES_DIR))

import canonical_replay as cr
import fast_replay as fr
import pod_constructor as pc
import study_data as sd

START_ISO = "2026-05-01T00:00:00Z"
END_ISO = "2026-09-01T00:00:00Z"
GRID_MS = 300_000
MIN_FILLS = 150
MIN_COIN_COVERAGE = 0.99
POD_EQUITY_USD = 10_000.0

SCALAR_FIELDS: tuple[str, ...] = (
    "objective_net_return",
    "cdar_95",
    "peak_to_trough_dd",
    "static_floor_dd",
    "daily_cvar_99",
    "max_daily_loss",
    "net_over_max_dd",
    "netting_savings_frac",
)
LIST_FIELDS: tuple[str, ...] = ("multipliers", "marginal_cdar_shares")


def setup(wallets: int) -> tuple[pc.UnitCopy5mTensor, pc.PodConstructorConfig]:
    """Build the fixed-window tensor and Tier-2 Euler sizing config for N wallets."""
    start = sd.to_utc_ms(datetime.fromisoformat(START_ISO))
    end = sd.to_utc_ms(datetime.fromisoformat(END_ISO))
    grid = start + np.arange((end - start) // GRID_MS, dtype=np.int64) * GRID_MS

    records, _ = cr.load_registry_universe(
        Path.home() / ".local/share/copytrade/registry/wallet_registry.sqlite",
        include_excluded=True,
    )
    paths, _ = sd.discover_fill_archive(start, end)
    ing = sd.ingest_fills(records, paths, start=start, end=end, split=end)
    subset = sorted(
        a for a, by in ing.grouped.items() if sum(df.height for df in by.values()) >= MIN_FILLS
    )[:wallets]

    prices = sd.LazyPrices(
        grid,
        store_root=sd.DEFAULT_STORE_ROOT,
        vike_root=sd.DEFAULT_VIKE_ROOT,
        split_ms=end,
    )
    coins = sorted({c for a in subset for c in ing.grouped[a]})
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for coin in coins:
        try:
            cand = prices.get(coin)
        except (KeyError, ValueError, OSError):
            continue
        if float(prices.stats.get(coin, {}).get("mark_valid_frac", 0.0)) >= MIN_COIN_COVERAGE:
            cache[coin] = cand

    cost_rate = cr.DEFAULT_COST_BPS * 1e-4
    compact: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for alias in subset:
        by_coin: dict[str, dict[str, np.ndarray]] = {}
        for coin, df in ing.grouped[alias].items():
            px = cache.get(coin)
            if px is None:
                continue
            op, cl = px
            pos = fr.positions_on_grid(
                df["ts"].to_numpy(),
                df["pre"].to_numpy(),
                df["sign"].to_numpy(),
                df["sz"].to_numpy(),
                grid,
                start,
            )
            net = fr.replay_fast(
                pos,
                op,
                cl,
                cost_bps=cr.DEFAULT_COST_BPS,
                fee_bps=cr.DEFAULT_FEE_BPS,
            )
            if net is None:
                raise RuntimeError(f"replay_fast returned None for ({alias}, {coin})")
            prev_pos = np.empty_like(pos)
            prev_pos[0] = 0.0
            prev_pos[1:] = pos[:-1]
            delta = pos - prev_pos
            op_eff = np.where(np.isfinite(op), op, cl)
            op_eff = np.where(np.isfinite(op_eff), op_eff, 0.0)
            cl_eff = np.where(np.isfinite(cl), cl, op_eff)
            cost = np.abs(delta) * op_eff * cost_rate
            gross = net + cost
            by_coin[coin] = {
                "open": op_eff,
                "close": cl_eff,
                "pos": pos,
                "delta": delta,
                "gross": gross,
                "net": net,
            }
        if by_coin:
            compact[alias] = by_coin

    tensor = pc.build_tensor_from_arrays(
        compact,
        grid,
        pod_equity_usd=POD_EQUITY_USD,
        chop_split_ms=int(grid[int(0.75 * grid.size)]),
    )
    if tensor.symbols:
        op_mat = np.vstack([cache[c][0] for c in tensor.symbols])
        cl_mat = np.vstack([cache[c][1] for c in tensor.symbols])
    else:
        op_mat = np.empty((0, grid.size), dtype=np.float64)
        cl_mat = np.empty((0, grid.size), dtype=np.float64)
    tensor_px = replace(tensor, exec_open_px=op_mat, mark_close_px=cl_mat)

    cfg = pc.PodConstructorConfig.for_sizing_tier(
        2,
        base=pc.PodConstructorConfig(
            pod_size=3,
            risk_share_mode="marginal_euler",
            max_marginal_cdar_share=0.45,
            reject_hidden_bagholder=True,
            reject_unknown_tilt=True,
        ),
    )
    return tensor_px, cfg


def serialize_pods(pods: Sequence[pc.SizedPodCandidate]) -> list[dict[str, object]]:
    """Serialize ranked SizedPodCandidate items into deterministic JSON-safe dicts."""
    return [
        {
            "pod_indices": [int(x) for x in p.pod_indices],
            "feasible": bool(p.feasible),
            "binding_constraint": str(p.binding_constraint),
            "multipliers": [float(x) for x in p.multipliers],
            "objective_net_return": float(p.objective_net_return),
            "cdar_95": float(p.cdar_95),
            "peak_to_trough_dd": float(p.peak_to_trough_dd),
            "static_floor_dd": float(p.static_floor_dd),
            "daily_cvar_99": float(p.daily_cvar_99),
            "max_daily_loss": float(p.max_daily_loss),
            "net_over_max_dd": float(p.net_over_max_dd),
            "netting_savings_frac": float(p.netting_savings_frac),
            "marginal_cdar_shares": [float(x) for x in p.marginal_cdar_shares],
        }
        for p in pods
    ]


def _pod_key(item: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(x) for x in item["pod_indices"])


def _check_float_pair(gv: float, wv: float, tol: float) -> tuple[int, float]:
    if not (math.isfinite(gv) and math.isfinite(wv)):
        return 1, float("inf")
    diff = abs(gv - wv)
    return (1 if diff > tol else 0), diff


def compare_pods(
    got: list[dict[str, Any]],
    want: list[dict[str, Any]],
    tol: float,
) -> tuple[int, float]:
    """Compare serialized pod outputs in both directions within numeric tolerance."""
    if not want:
        raise SystemExit("reference is empty: the equality check cannot be vacuous")

    got_map = {_pod_key(item): item for item in got}
    want_map = {_pod_key(item): item for item in want}

    mismatches = sum(1 for k in want_map if k not in got_map)
    mismatches += sum(1 for k in got_map if k not in want_map)
    max_diff = 0.0

    for key, w in want_map.items():
        g = got_map.get(key)
        if g is None:
            continue
        if bool(g["feasible"]) != bool(w["feasible"]):
            mismatches += 1
        if str(g["binding_constraint"]) != str(w["binding_constraint"]):
            mismatches += 1

        for field in SCALAR_FIELDS:
            mm, diff = _check_float_pair(float(g[field]), float(w[field]), tol)
            mismatches += mm
            max_diff = max(max_diff, diff)

        for field in LIST_FIELDS:
            gl = [float(x) for x in g[field]]
            wl = [float(x) for x in w[field]]
            if len(gl) != len(wl):
                mismatches += 1
                max_diff = float("inf")
                continue
            for gv, wv in zip(gl, wl, strict=True):
                mm, diff = _check_float_pair(gv, wv, tol)
                mismatches += mm
                max_diff = max(max_diff, diff)

    return mismatches, max_diff


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wallets", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--budget", type=float, default=None, help="median seconds budget")
    ap.add_argument("--save-ref", type=Path, default=None)
    ap.add_argument("--ref", type=Path, default=None)
    ap.add_argument("--tol", type=float, default=1e-12)
    args = ap.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    want_ref: list[dict[str, Any]] | None = None
    if args.ref is not None:
        want_ref = json.loads(args.ref.read_text(encoding="utf-8"))
        if not want_ref:
            raise SystemExit("reference is empty: the equality check cannot be vacuous")

    tensor_px, cfg = setup(args.wallets)
    print(
        f"workload: {len(tensor_px.aliases)} wallets, "
        f"{tensor_px.timestamps_ms.size} bars, {len(tensor_px.symbols)} coins"
    )

    stats: list[float] = []
    last_serialized: list[dict[str, Any]] = []
    mismatches = 0
    worst_diff = 0.0
    ref_saved = False

    for i in range(args.warmup + args.repeats):
        t0 = time.perf_counter()
        pods, _, _, _ = pc.discover_diverse_pods(tensor_px, cfg)
        dt = time.perf_counter() - t0
        serialized = serialize_pods(pods)
        warm = i < args.warmup
        tag = "warmup" if warm else "run"
        print(f"  [{tag}] {dt:.3f}s  pods={len(serialized)}")

        if warm:
            continue

        stats.append(dt)
        last_serialized = serialized
        if args.save_ref is not None and not ref_saved:
            args.save_ref.parent.mkdir(parents=True, exist_ok=True)
            args.save_ref.write_text(
                json.dumps(serialized, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            ref_saved = True

        if want_ref is not None:
            mm, md = compare_pods(serialized, want_ref, args.tol)
            mismatches += mm
            worst_diff = max(worst_diff, md)

    med = statistics.median(stats)
    spread = max(stats) - min(stats)
    print(f"median={med:.3f}s spread={spread:.3f}s runs={len(stats)} pods={len(last_serialized)}")

    if want_ref is not None:
        print(f"ref mismatches={mismatches} max_diff={worst_diff:.3e} tol={args.tol:.3e}")

    if args.budget is not None:
        ok = med <= args.budget
        print(f"budget={args.budget:.3f}s -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            return 1

    if want_ref is not None and (mismatches > 0 or worst_diff > args.tol):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
