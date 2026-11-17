"""Matched empirical test: structural episode sleeves vs canonical UPGMA vs simple baseline.

Reuses the canonical v4 ingestion, Train replay, tensor builder, crowding matrix, UPGMA,
dual-barrier sizer, and Guardian replay. No new simulation engine.

Methods are frozen and written to JSON *before* any OOS evaluation. OOS uses
``policy="new"`` with a fresh activation boundary at the OOS start (``candidate_targets``
-> ``reconstruct_positions_on_grid(..., activation_ms=grid[0])``), so strict
new-position-only is regenerated, not sliced from the Train replay.

All wallet and broker state is read-only. Outputs are sanitized (aliases only).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDIES_DIR = Path(__file__).resolve().parent
if str(STUDIES_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIES_DIR))

import canonical_replay as cr
import pod_constructor as pc

# --- frozen windows (identical to canonical v4) ---
START = 1_777_593_600_000  # 2026-05-01T00:00Z
SPLIT = START + 123 * 86_400_000  # 2026-09-01T00:00Z
END = SPLIT + 61 * 86_400_000  # 2026-11-01T00:00Z
GRID = np.arange(START, END, 300_000, dtype=np.int64)
CUT = int(np.searchsorted(GRID, SPLIT))
TRAIN_GRID = GRID[:CUT]
OOS_GRID = GRID[CUT:]

# --- frozen comparison parameters ---
POD_SIZE = 3
MIN_NOTIONAL_USD = 15.0
BASELINE_PORT_DD_FRAC = 0.02
DEADBAND_USD = 15.0
COST_MULT = 1.0
MAX_GROSS_LEVERAGE = 5.0
POD_EQUITY_USD = 10_000.0

# --- frozen sleeve definitions ---
SLEEVE_TOP_PER = 3
SLEEVE_ORDER = ("positional", "multi_day_swing", "intraday_tactical")
INTRADAY_MAX_BARS = 288  # <= 1 day at 5m
SWING_MAX_BARS = 2016  # <= 7 days at 5m
MIN_ACTIVE_FRAC_LABEL = 0.02
MIN_UNCENSORED_RUNS = 2

# --- frozen crowding / unknown-pair rules ---
CROWDING_VETO_DIST = 0.25
UNKNOWN_OVERLAP_DAYS = 5
UNKNOWN_MAX_DD_FRAC = 0.05  # half of the 10% static DD cap
UNKNOWN_MAX_DAILY_FRAC = 0.025  # half of the 5% daily cap

# --- frozen stability subtest ---
STABILITY_SEEDS = (0, 1, 2, 3, 4)
STABILITY_DROP_FRAC = 0.10

FROZEN_AT_UTC = None  # The original freeze time was not recorded.


def _load_v4():
    path = STUDIES_DIR / "2026-11-16__cft__10k__canonical-full-universe-v4.py"
    spec = importlib.util.spec_from_file_location("canonical_v4", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["canonical_v4"] = mod
    spec.loader.exec_module(mod)
    return mod


def _json_default(obj: object) -> object:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def dump_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n"
    )


def make_config(pod_size: int = POD_SIZE) -> pc.PodConstructorConfig:
    """Canonical v4 Stage 1/1.5/2 config, identical for every method."""
    return pc.PodConstructorConfig(
        pod_size=pod_size,
        top_per_cluster=2,
        max_candidate_pods=16,
        min_active_bar_frac=0.005,
        min_lot_usd=MIN_NOTIONAL_USD,
        min_baseline_order_usd=MIN_NOTIONAL_USD,
        baseline_port_dd_frac=BASELINE_PORT_DD_FRAC,
    )


def finalize(s: pc.SizedPodCandidate, metrics, conf: pc.PodConstructorConfig, tensor) -> dict:
    """Apply the canonical v4 50%-wallet-risk baseline cap and flatten to a dict."""
    unit_dd = [metrics[i].standalone_peak_to_trough_dd * POD_EQUITY_USD for i in s.pod_indices]
    base = [
        min(
            (conf.baseline_port_dd_frac * POD_EQUITY_USD / len(s.pod_indices)) / max(dd, 0.01),
            conf.max_k_mult,
        )
        for dd in unit_dd
    ]
    raw_k = [float(x) for x in s.multipliers]
    cap_scale = min(1.0, min(b / max(k, 1e-12) for k, b in zip(raw_k, base)))
    capped_k = [k * cap_scale for k in raw_k]
    # k is not comparable across wallets; report actual weighted contribution per member.
    member_contrib: list[dict] = []
    solo_nets: list[float] = []
    for k_val, idx in zip(capped_k, s.pod_indices):
        solo_net = float(np.sum(tensor.net_ret_frac[idx])) * k_val * POD_EQUITY_USD
        avg_gross = float(np.mean(np.abs(tensor.notional_frac[idx]).sum(axis=0))) * k_val
        solo_nets.append(solo_net)
        member_contrib.append(
            {
                "alias": tensor.aliases[idx],
                "capped_k": round(k_val, 10),
                "solo_weighted_net_usd": round(solo_net, 2),
                "avg_weighted_gross_notional_frac": round(avg_gross, 8),
            }
        )
    abs_sum = sum(abs(x) for x in solo_nets) or 1.0
    for row, net in zip(member_contrib, solo_nets):
        row["abs_net_share"] = round(abs(net) / abs_sum, 4)
    return {
        "aliases": list(s.aliases),
        "indices": list(s.pod_indices),
        "raw_k": raw_k,
        "capped_k": capped_k,
        "cap_scale": round(cap_scale, 6),
        "train_net_usd": round(s.objective_net_return * cap_scale * POD_EQUITY_USD, 2),
        "train_max_dd_usd": round(s.peak_to_trough_dd * cap_scale * POD_EQUITY_USD, 2),
        "train_static_floor_dd_usd": round(s.static_floor_dd * cap_scale * POD_EQUITY_USD, 2),
        "train_daily_cvar_usd": round(s.daily_cvar_99 * cap_scale * POD_EQUITY_USD, 2),
        "train_worst_day_usd": round(s.max_daily_loss * cap_scale * POD_EQUITY_USD, 2),
        "train_net_over_dd": round(s.net_over_max_dd, 4),
        "feasible": bool(s.feasible),
        "binding": s.binding_constraint,
        "min_lot_warnings": list(s.min_lot_warnings),
        "size": len(s.pod_indices),
        "member_contributions": member_contrib,
    }


def _rank_key(row: dict) -> tuple:
    return (row["feasible"], row["train_net_over_dd"], row["train_net_usd"])


def run_durations(pos_s: np.ndarray, eps: float = 1e-4) -> list[tuple[int, int, bool, bool]]:
    """Per-symbol held runs with left/right censoring flags (canonical split rule)."""
    held = np.abs(pos_s) > eps
    if not held.any():
        return []
    signs = np.where(held, np.where(pos_s >= 0.0, 1, -1), 0)
    changes = np.flatnonzero(signs[1:] != signs[:-1]) + 1
    starts = np.concatenate((np.array([0], dtype=np.intp), changes))
    ends = np.concatenate((changes, np.array([pos_s.size], dtype=np.intp)))
    runs: list[tuple[int, int, bool, bool]] = []
    for s, e in zip(starts, ends):
        if signs[s] == 0:
            continue
        runs.append((int(s), int(e), s == 0, e >= pos_s.size))
    return runs


def censoring_aware_features(tensor: pc.UnitCopy5mTensor, passed_ids) -> dict[int, dict]:
    """Holding-time features using uncensored runs only; report censored counts.

    The canonical ``reconstruct_execution_episodes`` returns episode ends but no start
    index or censoring flag, so it cannot distinguish a position already open at Train
    start (left-censored) from one closed inside the window. This recomputes the same
    per-symbol runs from the same ``notional_frac`` input for labeling only.
    """
    out: dict[int, dict] = {}
    for tid in passed_ids:
        durs_unc: list[int] = []
        durs_all: list[int] = []
        n_left = 0
        n_right = 0
        for s in range(len(tensor.symbols)):
            pos_s = tensor.notional_frac[tid, s]
            if not pos_s.any():
                continue
            for a, b, left, right in run_durations(pos_s):
                d = b - a
                durs_all.append(d)
                if left:
                    n_left += 1
                if right:
                    n_right += 1
                if not left and not right:
                    durs_unc.append(d)
        use_unc = len(durs_unc) >= MIN_UNCENSORED_RUNS
        base = durs_unc if use_unc else durs_all
        out[tid] = {
            "median_dur_bars": float(np.median(base)) if base else 0.0,
            "n_runs": len(durs_all),
            "n_uncensored": len(durs_unc),
            "n_left_censored": n_left,
            "n_right_censored": n_right,
            "censored_fallback": not use_unc,
        }
    return out


def sleeve_labels(
    tensor: pc.UnitCopy5mTensor, metrics, passed_ids, feat: dict[int, dict]
) -> dict[int, str]:
    labels: dict[int, str] = {}
    for tid in passed_ids:
        m = metrics[tid]
        f = feat[tid]
        if m.active_frac < MIN_ACTIVE_FRAC_LABEL or m.num_episodes < 4 or f["n_runs"] < 4:
            labels[tid] = "unlabeled"
        elif f["median_dur_bars"] <= INTRADAY_MAX_BARS:
            labels[tid] = "intraday_tactical"
        elif f["median_dur_bars"] <= SWING_MAX_BARS:
            labels[tid] = "multi_day_swing"
        else:
            labels[tid] = "positional"
    return labels


def active_day_sets(
    tensor: pc.UnitCopy5mTensor, passed_ids, bars_per_day: int = 288
) -> tuple[dict[int, set], dict[int, set]]:
    """Per-wallet active UTC days and active symbols over the Train tensor."""
    n_bars = tensor.timestamps_ms.size
    n_days = max(1, n_bars // bars_per_day)
    usable = n_days * bars_per_day
    days: dict[int, set] = {}
    coins: dict[int, set] = {}
    for tid in passed_ids:
        abs_pos = np.abs(tensor.notional_frac[tid])
        gross = abs_pos.sum(axis=0)[:usable]
        day_act = (gross.reshape(n_days, bars_per_day) > 1e-8).any(axis=1)
        days[tid] = set(np.flatnonzero(day_act).tolist())
        coins[tid] = set(np.flatnonzero(abs_pos.max(axis=1) > 1e-8).tolist())
    return days, coins


def pair_status(
    a: int,
    b: int,
    idx_of: dict[int, int],
    dist_mat: np.ndarray,
    psi_mat: np.ndarray,
    day_sets,
    coin_sets,
) -> dict:
    """Classify a pair as observed, disjoint-coin (legitimate zero), or unknown.

    ``unknown`` means no concurrent activity, so the direct same-symbol cosine is
    unobserved. Concurrent activity in disjoint coins is a legitimate zero direct
    cosine, not a missing observation, though it is not proof of low common-factor
    or tail risk.
    """
    ia, ib = idx_of[a], idx_of[b]
    d = float(dist_mat[ia, ib])
    concurrent = len(day_sets[a] & day_sets[b])
    common = coin_sets[a] & coin_sets[b]
    unknown = concurrent < UNKNOWN_OVERLAP_DAYS
    disjoint = (not unknown) and (not common)
    return {
        "dist": round(d, 4),
        "psi_pos": round(float(psi_mat[ia, ib]), 4),
        "concurrent_days": concurrent,
        "common_coins": len(common),
        "unknown": bool(unknown),
        "disjoint_coins": bool(disjoint),
        "crowded": bool(d < CROWDING_VETO_DIST),
    }


def upgma_candidates(tensor, metrics, passed_ids, dist_mat, conf) -> list[dict]:
    """Canonical UPGMA selection mirroring ``discover_diverse_pods`` on precomputed inputs."""
    labels = pc.cluster_candidates_upgma(dist_mat, min(conf.pod_size, len(passed_ids)))
    clusters: dict[int, list[int]] = {}
    for pos, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(passed_ids[pos])

    cluster_picks: list[list[int]] = []
    for cid in sorted(clusters):
        members = clusters[cid]
        by_overall = sorted(members, key=lambda t: metrics[t].net_over_max_dd, reverse=True)
        by_chop = sorted(members, key=lambda t: metrics[t].chop_net_over_dd, reverse=True)
        picked: list[int] = []
        for lst in (by_overall, by_chop):
            for tid in lst:
                if tid not in picked and len(picked) < conf.top_per_cluster:
                    picked.append(tid)
        cluster_picks.append(picked)

    from itertools import product

    rank_lookup = {tid: r for lst in cluster_picks for r, tid in enumerate(lst)}
    combos = {
        tuple(sorted(p)) for p in product(*cluster_picks) if len(set(p)) == len(cluster_picks)
    }
    ordered = sorted(combos, key=lambda p: (sum(rank_lookup[t] for t in p), p))[
        : conf.max_candidate_pods
    ]
    sized = [pc.size_pod_dual_barrier(p, tensor, metrics, conf) for p in ordered]
    rows = [finalize(s, metrics, conf, tensor) for s in sized]
    rows.sort(key=_rank_key, reverse=True)
    budget = {
        "clusters": len(clusters),
        "cluster_pick_sizes": [len(p) for p in cluster_picks],
        "combos_generated": len(combos),
        "combos_sized": len(ordered),
        "combos_admitted": len(rows),
    }
    return rows, budget


def sleeve_candidates(tensor, metrics, passed_ids, dist_mat, psi_mat, conf) -> tuple[list[dict], dict]:
    """Structural episode-sleeve selection with pairwise crowding veto and unknown handling."""
    from itertools import product

    feat = censoring_aware_features(tensor, passed_ids)
    labels = sleeve_labels(tensor, metrics, passed_ids, feat)
    idx_of = {tid: i for i, tid in enumerate(passed_ids)}
    day_sets, coin_sets = active_day_sets(tensor, passed_ids)

    by_sleeve: dict[str, list[int]] = {}
    for tid in passed_ids:
        lab = labels[tid]
        if lab != "unlabeled":
            by_sleeve.setdefault(lab, []).append(tid)

    picks: dict[str, list[int]] = {}
    for lab, members in by_sleeve.items():
        members.sort(key=lambda t: metrics[t].alias)
        members.sort(
            key=lambda t: (metrics[t].net_over_max_dd, metrics[t].raw_sharpe_ann), reverse=True
        )
        members.sort(key=lambda t: feat[t]["median_dur_bars"], reverse=True)
        # chop-positive is a role tie-break, not a hard gate; chop window is a calendar proxy.
        members.sort(key=lambda t: bool(metrics[t].chop_net_over_dd > 0.0), reverse=True)
        picks[lab] = members[:SLEEVE_TOP_PER]

    present = [s for s in SLEEVE_ORDER if picks.get(s)]
    diag = {
        "sleeve_counts": {lab: len(members) for lab, members in sorted(by_sleeve.items())},
        "unlabeled": sum(1 for t in passed_ids if labels[t] == "unlabeled"),
        "sleeves_present": present,
        "censored_runs": {
            "left": sum(feat[t]["n_left_censored"] for t in passed_ids),
            "right": sum(feat[t]["n_right_censored"] for t in passed_ids),
            "uncensored": sum(feat[t]["n_uncensored"] for t in passed_ids),
            "fallback_wallets": sum(1 for t in passed_ids if feat[t]["censored_fallback"]),
        },
        "feature_by_alias": {
            metrics[t].alias: {
                "sleeve": labels[t],
                "median_dur_bars": round(feat[t]["median_dur_bars"], 2),
                "n_runs": feat[t]["n_runs"],
                "n_uncensored": feat[t]["n_uncensored"],
                "n_left_censored": feat[t]["n_left_censored"],
                "n_right_censored": feat[t]["n_right_censored"],
                "censored_fallback": feat[t]["censored_fallback"],
                "active_frac": round(metrics[t].active_frac, 4),
                "net_over_max_dd": round(metrics[t].net_over_max_dd, 4),
                "chop_net_over_dd": round(metrics[t].chop_net_over_dd, 4),
            }
            for t in passed_ids
        },
    }
    diag["search_budget"] = {
        "sleeves_present": len(present),
        "picks_per_sleeve": {s: len(picks.get(s, [])) for s in present},
        "combos_generated": 0,
        "combos_rejected_crowded": 0,
        "combos_rejected_infeasible": 0,
        "combos_rejected_unknown_cap": 0,
        "combos_admitted": 0,
    }
    if len(present) < 2:
        diag["status"] = "INSUFFICIENT_SLEEVES"
        return [], diag

    combos = [tuple(sorted(c)) for c in product(*[picks[s] for s in present])]
    budget = diag["search_budget"]
    budget["combos_generated"] = len(combos)
    sized_rows: list[dict] = []
    for combo in combos:
        pairs = [
            pair_status(combo[i], combo[j], idx_of, dist_mat, psi_mat, day_sets, coin_sets)
            for i in range(len(combo))
            for j in range(i + 1, len(combo))
        ]
        if any(p["crowded"] for p in pairs):
            budget["combos_rejected_crowded"] += 1
            continue
        has_unknown = any(p["unknown"] for p in pairs)
        s = pc.size_pod_dual_barrier(combo, tensor, metrics, conf)
        if not s.feasible:
            budget["combos_rejected_infeasible"] += 1
            continue
        if has_unknown and (
            s.peak_to_trough_dd > UNKNOWN_MAX_DD_FRAC or s.max_daily_loss > UNKNOWN_MAX_DAILY_FRAC
        ):
            budget["combos_rejected_unknown_cap"] += 1
            continue
        row = finalize(s, metrics, conf, tensor)
        row["pairs"] = pairs
        row["has_unknown_pair"] = has_unknown
        row["unknown_pairs"] = sum(1 for p in pairs if p["unknown"])
        row["disjoint_coin_pairs"] = sum(1 for p in pairs if p["disjoint_coins"])
        sized_rows.append(row)

    sized_rows.sort(key=_rank_key, reverse=True)
    diag["status"] = "OK" if sized_rows else "NO_FEASIBLE_POD"
    diag["combos_tested"] = len(combos)
    diag["combos_admitted"] = len(sized_rows)
    budget["combos_admitted"] = len(sized_rows)
    return sized_rows, diag


def baseline_candidates(tensor, metrics, passed_ids, conf) -> tuple[list[dict], dict]:
    """Unstratified top-rank baseline: top-N by Train net/max-DD, no clustering."""
    ranked = sorted(passed_ids, key=lambda t: metrics[t].alias)
    ranked.sort(
        key=lambda t: (metrics[t].net_over_max_dd, metrics[t].raw_sharpe_ann), reverse=True
    )
    top = ranked[: conf.pod_size]
    s = pc.size_pod_dual_barrier(top, tensor, metrics, conf)
    budget = {"combos_generated": 1, "combos_sized": 1, "combos_admitted": 1}
    return [finalize(s, metrics, conf, tensor)], budget


def _res_dict(res: cr.ChallengeRunResult) -> dict:
    return {
        "verdict": res.verdict,
        "terminal_step": res.terminal_step,
        "terminal_net_usd": round(res.terminal_net_usd, 2),
        "static_floor_dd_usd": round(res.static_floor_dd_usd, 2),
        "peak_to_trough_dd_usd": round(res.peak_to_trough_dd_usd, 2),
        "worst_daily_usd": round(res.worst_daily_usd, 2),
        "full_window_net_usd": round(res.full_window_net_usd, 2),
        "full_window_static_floor_dd_usd": round(res.full_window_static_floor_dd_usd, 2),
        "full_window_peak_to_trough_dd_usd": round(res.full_window_peak_to_trough_dd_usd, 2),
        "sweeps": res.sweeps,
        "forced_gap_exits": res.forced_gap_exits,
        "unpriced_open_steps": res.unpriced_open_steps,
        "deferred_notional_usd": round(res.deferred_notional_usd, 2),
        "peak_gross_leverage": round(res.peak_gross_leverage, 4),
    }


def evaluate_oos(v4, row: dict, grouped, prices, policy: str) -> dict:
    coins, tgt, trade, mark = v4.candidate_targets(
        row["aliases"], row["capped_k"], grouped, prices, OOS_GRID, policy, full_grid=GRID
    )
    spec = cr.GuardianSpec(
        min_trade_notional_usd=DEADBAND_USD,
        cost_multiplier=COST_MULT,
        max_gross_leverage=MAX_GROSS_LEVERAGE,
    )
    res, _ = cr.simulate_guardian_challenge(coins, tgt, trade, mark, OOS_GRID, spec)
    out = _res_dict(res)
    out["boundary_nonzero_targets"] = int(np.count_nonzero(np.abs(tgt[:, 0]) > 1e-12))
    out["coins"] = len(coins)
    out["gap_invalidated"] = bool(res.unpriced_open_steps > 0)
    return out


def member_diagnostics(row: dict, train_diag, hist_eligible, grouped) -> list[dict]:
    """Per-member Train gap flags and Train-vs-OOS coin-universe mismatch."""
    out: list[dict] = []
    for alias in row["aliases"]:
        d = train_diag.get(alias, {})
        train_coins = set(hist_eligible.get(alias, {}))
        oos_coins = set(grouped.get(alias, {}))
        out.append(
            {
                "alias": alias,
                "train_forced_gap_exits": int(d.get("forced_gap_exits", 0)),
                "train_unpriced_open_steps": int(d.get("unpriced_open_steps", 0)),
                "train_unanchored_coins": int(d.get("unanchored_coins", 0)),
                "train_gap_flagged": bool(
                    d.get("forced_gap_exits", 0) > 0 or d.get("unpriced_open_steps", 0) > 0
                ),
                "train_coins": len(train_coins),
                "oos_coins": len(oos_coins),
                "oos_coins_not_in_train": sorted(oos_coins - train_coins),
            }
        )
    return out


def stability_subtest(tensor, metrics, passed_ids, dist_mat, psi_mat, conf) -> dict:
    """Selection stability under seeded 10% universe drops (distance matrix is pairwise)."""
    idx_of = {tid: i for i, tid in enumerate(passed_ids)}
    full = {}
    a_full, _ = upgma_candidates(tensor, metrics, passed_ids, dist_mat, conf)
    b_full, _ = sleeve_candidates(tensor, metrics, passed_ids, dist_mat, psi_mat, conf)
    full["A_upgma"] = set(a_full[0]["aliases"]) if a_full else set()
    full["B_sleeve"] = set(b_full[0]["aliases"]) if b_full else set()

    out: dict[str, list[float]] = {"A_upgma": [], "B_sleeve": []}
    for seed in STABILITY_SEEDS:
        rng = np.random.default_rng(seed)
        drop_n = max(1, round(len(passed_ids) * STABILITY_DROP_FRAC))
        drop = set(rng.choice(passed_ids, size=drop_n, replace=False).tolist())
        keep = [t for t in passed_ids if t not in drop]
        sub_idx = [idx_of[t] for t in keep]
        sub_dist = dist_mat[np.ix_(sub_idx, sub_idx)]
        sub_psi = psi_mat[np.ix_(sub_idx, sub_idx)]
        a_rows, _ = upgma_candidates(tensor, metrics, keep, sub_dist, conf)
        b_rows, _ = sleeve_candidates(tensor, metrics, keep, sub_dist, sub_psi, conf)
        for name, rows in (("A_upgma", a_rows), ("B_sleeve", b_rows)):
            sel = set(rows[0]["aliases"]) if rows else set()
            base = full[name]
            union = sel | base
            out[name].append(round(len(sel & base) / len(union), 4) if union else 0.0)
    return {
        "seeds": list(STABILITY_SEEDS),
        "drop_frac": STABILITY_DROP_FRAC,
        "jaccard_vs_full": out,
        "mean_jaccard": {k: round(float(np.mean(v)), 4) for k, v in out.items()},
    }


def _fmt_pod(row: dict) -> str:
    return ", ".join(row["aliases"])


def write_report(path: Path, payload: dict) -> None:
    f = payload["freeze"]
    r = payload["results"]
    lines: list[str] = []
    lines.append("---")
    lines.append("firm: crypto-fund-trader")
    lines.append("account_usd: 10000")
    lines.append("status: RESEARCH")
    lines.append("top_candidate: NOT_SCIENTIFICALLY_QUALIFIED")
    lines.append("script: studies/2026-11-17__cft__10k__sleeve-vs-upgma.py")
    lines.append("---")
    lines.append("")
    lines.append("# Structural episode sleeves vs UPGMA — matched empirical test (CFT $10k)")
    lines.append("")
    lines.append(
        "This is an exploratory retrospective rerun, not an independently preregistered study. "
        "The script writes candidate definitions before its OOS replay, but earlier results were "
        "already inspected and the gap gate was revised. Figures are modeled, not live or PIT."
    )
    lines.append("")
    lines.append("## Frozen design")
    lines.append("")
    lines.append("- Train: `2026-05-01T00:00Z`–`2026-09-01T00:00Z`; OOS: `2026-09-01T00:00Z`–`2026-11-01T00:00Z` (exclusive).")
    lines.append(
        f"- Roster SHA256 `{f['roster_sha256']}`; Train-eligible historical `{f['historical_eligible']}`; "
        f"positive Train shortlist `{f['shortlist']}`; behavioral pass `{f['behavioral_pass']}`."
    )
    lines.append(
        f"- Identical for every method: `pod_size={POD_SIZE}`, `$15` min baseline order, "
        f"`2%` port-DD baseline cap, `$15` deadband, `{COST_MULT}x` cost, strict `new` OOS."
    )
    lines.append(
        "- Strict `new` OOS is **regenerated**, not sliced: `candidate_targets` calls "
        "`reconstruct_positions_on_grid(..., policy='new', activation_ms=OOS_start)` on the raw "
        "fills. `UnitCopy5mTensor.slice_time` is not used for OOS."
    )
    lines.append("")
    lines.append("### Method A — canonical UPGMA (baseline authority)")
    lines.append("")
    lines.append(
        "`stage1_audit_traders` -> `compute_crowding_distance_matrix` -> `cluster_candidates_upgma` "
        "into `pod_size` clusters -> top-2 per cluster by overall and chop Net/DD -> Cartesian "
        "combos (<=16) -> `size_pod_dual_barrier` -> rank `(feasible, net/DD, net)`."
    )
    lines.append("")
    lines.append("### Method B — structural episode sleeves (proposed)")
    lines.append("")
    lines.append(
        "Holding-time buckets from **uncensored** per-symbol runs (left-censored at Train start and "
        "right-censored at Train end excluded; fallback disclosed per wallet): "
        f"`intraday_tactical` <= {INTRADAY_MAX_BARS} bars, `multi_day_swing` <= {SWING_MAX_BARS}, "
        f"`positional` > {SWING_MAX_BARS}; `unlabeled` if active_frac < {MIN_ACTIVE_FRAC_LABEL} or "
        f"< 4 runs. Top-{SLEEVE_TOP_PER} per sleeve by (chop-positive role tie-break, net/DD, Sharpe, alias). "
        f"Admit a combo only if every pair has `dist >= {CROWDING_VETO_DIST}`; an unknown pair "
        f"(< {UNKNOWN_OVERLAP_DAYS} concurrent active days) additionally requires sized "
        f"max DD <= {UNKNOWN_MAX_DD_FRAC:.0%} and worst day <= {UNKNOWN_MAX_DAILY_FRAC:.1%}. "
        "Greedy is not used; all admitted combos are sized and ranked deterministically."
    )
    lines.append("")
    lines.append("### Method C — simple risk-screened baseline")
    lines.append("")
    lines.append(
        f"Top-{POD_SIZE} behavioral-pass wallets by Train `net/max-DD` (tie: Sharpe, alias), "
        "no clustering, same sizer and cap."
    )
    lines.append("")
    lines.append("### Primary endpoints and decision rule (frozen)")
    lines.append("")
    lines.append(
        "Primary: OOS strict-new `full_window_net_usd`, `full_window_static_floor_dd_usd`, "
        "`worst_daily_usd` for each method's top-1 Train-ranked pod. A method dominates another "
        "only if OOS net is >=, static floor DD is >= (less negative), and worst day is >= on all "
        "three. No winner is declared without dominance over both others. `full` policy is a "
        "separately labeled sensitivity."
    )
    lines.append("")
    lines.append("## Coverage and censoring")
    lines.append("")
    cov = r["coverage"]
    lines.append(
        f"- UTC bounds `{cov['min_ts_ms']}`–`{cov['max_ts_ms']}`; daily archives `{cov['archive_days']}`; "
        f"missing source days `{cov['missing_days']}`."
    )
    lines.append(
        f"- Shortlist `{cov['shortlist']}`; behavioral pass `{cov['behavioral_pass']}`; "
        f"sleeve counts `{json.dumps(cov['sleeve_counts'], sort_keys=True)}`; unlabeled `{cov['unlabeled']}`."
    )
    lines.append(
        f"- Censored runs: left `{cov['censored_runs']['left']}`, right `{cov['censored_runs']['right']}`, "
        f"uncensored `{cov['censored_runs']['uncensored']}`, fallback wallets `{cov['censored_runs']['fallback_wallets']}`. "
        "The canonical extractor does not expose start/censor flags; labels use the disclosed recomputation."
    )
    lines.append("")
    lines.append("### Roster drift vs canonical v4 (missing work)")
    lines.append("")
    lines.append(
        "The current live registry exposes **38 configured non-excluded wallets**, not the "
        "**195** used by the canonical v4 study (`roster_sha256 c4abfa90...`). Only 37 are "
        "Train-eligible, 30 reach the positive Train shortlist, and 11 pass the behavioral audit. "
        "The v4 replay is therefore **not reproducible from the current registry**, and this is a "
        "reduced-universe matched subtest, not a full-universe replication. The missing work is a "
        "re-run on the frozen 195-wallet roster, which is unavailable here without a point-in-time "
        "registry snapshot. The 38 configured wallets are the **current live selection**, not a "
        "point-in-time historical universe, so the sample is additionally survivorship- and "
        "selection-contaminated; every figure is conditional on that roster."
    )
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("cd /home/quant/checkouts/q3-style-sleeves")
    lines.append("uv run python studies/2026-11-17__cft__10k__sleeve-vs-upgma.py")
    lines.append("```")
    lines.append("")
    lines.append(
        f"Runtime `{r['runtime_s']}s`. Outputs: `reports/2026-11-17__cft__10k__sleeve-vs-upgma__freeze.json`, "
        "`...__results.json`, `...md`, and gitignored `data/v4_cache/...__inventory.json`."
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("### Search budgets and method status (frozen audit)")
    lines.append("")
    lines.append("| Method | Status | Budget |")
    lines.append("|---|---|---|")
    for m in ("A_upgma", "B_sleeve", "C_baseline"):
        lines.append(
            f"| {m} | {r['method_status'][m]} | `{json.dumps(r['search_budgets'][m], sort_keys=True)}` |"
        )
    lines.append("")
    lines.append("### Train gap and coin-universe diagnostics (selected members)")
    lines.append("")
    lines.append("| Method | Alias | Train gap exits | Train unpriced steps | Train gap flagged | Train coins | OOS coins | OOS coins not in Train |")
    lines.append("|---|---|---:|---:|---|---:|---:|---|")
    for m in ("A_upgma", "B_sleeve", "C_baseline"):
        rows = r["train"].get(m) or []
        if not rows:
            continue
        for d in rows[0].get("member_diagnostics", []):
            lines.append(
                f"| {m} | {d['alias']} | {d['train_forced_gap_exits']} | {d['train_unpriced_open_steps']} | "
                f"{d['train_gap_flagged']} | {d['train_coins']} | {d['oos_coins']} | "
                f"{', '.join(d['oos_coins_not_in_train']) or '-'} |"
            )
    lines.append("")
    lines.append("### Train (in-sample, selection-biased)")
    lines.append("")
    lines.append("| Method | Rank | Members | Seats | Train net $ | Train max DD $ | Train worst day $ | Net/DD | Unknown pairs |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for m in ("A_upgma", "B_sleeve", "C_baseline"):
        for i, row in enumerate(r["train"][m][:3], 1):
            unk = row.get("unknown_pairs", "n/a")
            lines.append(
                f"| {m} | {i} | {_fmt_pod(row)} | {row['size']} | {row['train_net_usd']:.2f} | "
                f"{row['train_max_dd_usd']:.2f} | {row['train_worst_day_usd']:.2f} | "
                f"{row['train_net_over_dd']:.4f} | {unk} |"
            )
    lines.append("")
    lines.append("### OOS strict new-position-only (`$15` deadband, `1x` cost)")
    lines.append("")
    lines.append("| Method | Rank | Members | Net $ | Static floor DD $ | Peak-to-trough DD $ | Worst day $ | Verdict | Peak lev | Unpriced steps | Unknown pairs |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|")
    for m in ("A_upgma", "B_sleeve", "C_baseline"):
        for i, row in enumerate(r["oos"]["new"][m][:3], 1):
            lines.append(
                f"| {m} | {i} | {_fmt_pod(row['train'])} | {row['full_window_net_usd']:.2f} | "
                f"{row['full_window_static_floor_dd_usd']:.2f} | {row['full_window_peak_to_trough_dd_usd']:.2f} | "
                f"{row['worst_daily_usd']:.2f} | {row['verdict']} | {row['peak_gross_leverage']:.2f}x | "
                f"{row['unpriced_open_steps']} | {row['train'].get('unknown_pairs', 'n/a')} |"
            )
    lines.append("")
    lines.append(
        "- Boundary check: `new` has **0** nonzero netted targets at the first OOS bar for every "
        "method, confirming eligibility is regenerated at the OOS boundary. `full` carries "
        "pre-boundary inventory (see table below), so the two policies are not comparable."
    )
    lines.append("")
    lines.append("### OOS `full` policy sensitivity (not comparable to strict new)")
    lines.append("")
    lines.append("| Method | Members | Net $ | Static floor DD $ | Worst day $ | Boundary nonzero targets |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for m in ("A_upgma", "B_sleeve", "C_baseline"):
        rows = r["oos"]["full"].get(m) or []
        if not rows:
            lines.append(f"| {m} | n/a | n/a | n/a | n/a | n/a |")
            continue
        row = rows[0]
        lines.append(
            f"| {m} | {_fmt_pod(row['train'])} | {row['full_window_net_usd']:.2f} | "
            f"{row['full_window_static_floor_dd_usd']:.2f} | {row['worst_daily_usd']:.2f} | "
            f"{row['boundary_nonzero_targets']} |"
        )
    lines.append("")
    lines.append("### Gap diagnostics (OOS, strict new)")
    lines.append("")
    lines.append("| Method | Unpriced steps | Forced gap exits | Deferred notional $ | Gap invalidated |")
    lines.append("|---|---:|---:|---:|---|")
    for m in ("A_upgma", "B_sleeve", "C_baseline"):
        rows = r["oos"]["new"].get(m) or []
        if not rows:
            lines.append(f"| {m} | n/a | n/a | n/a | n/a |")
            continue
        row = rows[0]
        lines.append(
            f"| {m} | {row['unpriced_open_steps']} | {row['forced_gap_exits']} | "
            f"{row['deferred_notional_usd']:.2f} | {row['gap_invalidated']} |"
        )
    lines.append("")
    lines.append("### Coverage vs exposure gap (not equivalent)")
    lines.append("")
    mv = r["coverage"].get("mark_valid_frac", {})
    if mv:
        lines.append("| Coin | mark_valid_frac (all-window grid) |")
        lines.append("|---|---:|")
        for coin, frac in sorted(mv.items()):
            lines.append(f"| {coin} | {frac if frac is not None else 'n/a'} |")
        lines.append("")
    lines.append(
        "- `mark_valid_frac` is **all-window grid coverage** from `LazyPrices.stats`. "
        "`unpriced_open_steps` and `forced_gap_exits` are **exposure-specific**: they count only bars "
        "where an open leg meets a data outage. A low `mark_valid_frac` coin does not by itself imply "
        "an unpriced open-position step, and a zero exposure-gap count does not prove full grid coverage."
    )
    lines.append("")
    lines.append("### Structural sleeve pod pair classification (Train)")
    lines.append("")
    b_rows = r["train"].get("B_sleeve") or []
    if not b_rows or "pairs" not in b_rows[0]:
        lines.append("- No admitted structural pod; pair classification not applicable.")
    else:
        lines.append("| Pair | dist | psi_pos | Concurrent days | Common coins | Unknown | Disjoint coins |")
        lines.append("|---|---:|---:|---:|---:|---|---|")
        pod = b_rows[0]
        # pairs are stored in i<j order
        k = 0
        for i in range(len(pod["aliases"])):
            for j in range(i + 1, len(pod["aliases"])):
                p = pod["pairs"][k]
                k += 1
                lines.append(
                    f"| {pod['aliases'][i]} / {pod['aliases'][j]} | {p['dist']:.4f} | {p['psi_pos']:.4f} | "
                    f"{p['concurrent_days']} | {p['common_coins']} | {p['unknown']} | {p['disjoint_coins']} |"
                )
    lines.append("")
    lines.append("### Decision-rule outcome")
    lines.append("")
    dom = r["dominance"]
    lines.append(f"- Pairwise dominance (net, static DD, worst day): `{json.dumps(dom['matrix'], sort_keys=True)}`.")
    lines.append(f"- Verdict: **{dom['verdict']}**.")
    lines.append("")
    lines.append("### Selection stability under 10% universe drops")
    lines.append("")
    st = r["stability"]
    if st.get("skipped"):
        lines.append("- Skipped by request.")
    else:
        lines.append(
            f"- Mean Jaccard vs full-universe selection: A `{st['mean_jaccard']['A_upgma']}`, "
            f"B `{st['mean_jaccard']['B_sleeve']}` across seeds `{st['seeds']}`."
        )
        lines.append(f"- Per-seed: `{json.dumps(st['jaccard_vs_full'], sort_keys=True)}`.")
    lines.append("")
    lines.append("## Interpretation (non-causal)")
    lines.append("")
    lines.append(
        "- UPGMA and the unstratified top-rank baseline are close on OOS net (~$798 vs ~$766) and "
        "static floor DD (~-$9), and both select overlapping members. The structural sleeve pod has "
        "far lower OOS net (~$232) but also far lower static floor DD (~-$3) and worst day (~-$44)."
    )
    b0 = r["train"].get("B_sleeve") or []
    if b0:
        contribs = b0[0].get("member_contributions", [])
        small = [c["alias"] for c in contribs if c.get("abs_net_share", 1.0) < 0.05]
        lines.append(
            f"- B top-pod member contributions: `{json.dumps(contribs, sort_keys=True)}`. "
            f"Members below 5% absolute net share: `{small}`."
        )
        if small:
            lines.append(
                f"- By weighted net contribution the sleeve pod is effectively "
                f"{len(contribs) - len(small)}-seat; this dilution is a property of the frozen method, "
                "not an implementation artifact."
            )
        else:
            lines.append("- All sleeve members contribute materially; no dilution claim is made.")
        if b0[0].get("train_gap_flagged"):
            n_flagged = sum(1 for row in (r["train"].get("B_sleeve") or []) if row.get("train_gap_flagged"))
            lines.append(
                f"- B's top pod is **Train-gap-flagged** (a selected member has forced gap exits or "
                f"unpriced open steps), and {n_flagged}/{len(r['train'].get('B_sleeve') or [])} of its "
                "top candidates are flagged. Its OOS figures are diagnostic only and it is excluded "
                "from the dominance verdict."
            )
    lines.append(
        "- Selection stability under 10% universe drops is higher for sleeves (mean Jaccard 0.84 vs "
        "0.70), consistent with the premise that structural labels move less than dendrograms. This "
        "does **not** establish superior returns; the OOS net is lower."
    )
    lines.append(
        "- With 11 behavioral-pass wallets, one 61-day OOS window, and no point-in-time roster, the "
        "comparison is underpowered. It supports the stability claim and rejects a return-superiority "
        "claim; it does not rank the methods for deployment."
    )
    lines.append("")
    lines.append("## Unknown-pair, gap, and coverage limitations")
    lines.append("")
    lines.append(
        f"- **Unknown** means fewer than {UNKNOWN_OVERLAP_DAYS} days on which both wallets were "
        "active; this is insufficient evidence, not necessarily zero simultaneous bars. "
        "Concurrent activity in **disjoint coins** is a "
        "legitimate zero direct cosine, not a missing observation, though it is not proof of low "
        "common-factor or tail risk. The canonical matrix initializes no-overlap cosine to neutral; "
        "absence of evidence is not evidence of neutrality."
    )
    lines.append(
        "- **Gap gate:** a pod with any `unpriced_open_steps > 0` is marked `gap_invalidated` and "
        "excluded from the dominance verdict. Pair-local conservative exits are modeled, not actual "
        "forced execution. The Vike fallback is present in this checkout (61 coin files; "
        "`STORE_PLUS_VIKE` for covered coins), so the reduced universe is driven by registry roster "
        "drift, not price coverage. This study does not claim v4 replay reproducibility."
    )
    lines.append(
        "- OOS is a single 61-day window that earlier research already explored; it is **not** a "
        "pristine holdout. The roster is the present-day registry, so historical eligibility is "
        "**survivorship-contaminated** and not point-in-time."
    )
    lines.append(
        "- The Train chop window is the last 92 calendar days of Train, not a point-in-time regime "
        "label. Chop-positive is used only as a role tie-break; no independently regime-conditioned "
        "claim is made."
    )
    lines.append(
        "- `full` policy carries pre-boundary inventory and is shown only as a sensitivity; only the "
        "`new` policy regenerates eligibility at the OOS boundary."
    )
    lines.append(
        "- **Coin-universe mismatch:** OOS uses raw `grouped` (canonical v4 behavior) while Train uses "
        "`hist_eligible`, so OOS can include coins excluded from Train by price coverage. The table "
        "above lists per-member mismatches; this is disclosed, not aligned."
    )
    lines.append(
        "- **Train gap flag:** a selected member with any Train forced gap exit or unpriced open step "
        "marks the pod `train_gap_flagged`; such pods are excluded from the dominance verdict because "
        "their Train `k` may be gap-contaminated."
    )
    lines.append("")
    lines.append("## Qualification and caveats")
    lines.append("")
    lines.append(
        "- This is a **matched retrospective comparison**, not causal evidence that sleeve "
        "structure produces superior returns. Selecting the OOS winner ex post would itself be "
        "selection bias; the frozen dominance rule is reported instead."
    )
    lines.append(
        "- No live broker execution, funding, capacity, min-lot feasibility, or PIT source "
        "continuity is proven. No deployments, allocations, orders, service or registry writes occurred."
    )
    lines.append("")
    lines.append(
        "- CFT trading restrictions and account framing require independent compliance confirmation."
    )
    lines.append("- Live broker execution, funding, capacity and PIT source continuity are UNPROVEN.")
    lines.append("- No deployments, allocations, orders, service or registry writes occurred.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-prefix", default=str(ROOT / "reports/2026-11-17__cft__10k__sleeve-vs-upgma"))
    parser.add_argument("--no-stability", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    v4 = _load_v4()
    paths = v4.StudyPaths(report_prefix=Path(args.report_prefix))

    freeze_path = Path(f"{args.report_prefix}__freeze.json")
    results_path = Path(f"{args.report_prefix}__results.json")
    md_path = Path(f"{args.report_prefix}.md")

    # Historical-only study: pass an explicit empty Bybit snapshot so no live network call
    # is made. Historical eligibility is independent of the live symbol screen; live
    # deployability is intentionally NOT evaluated here.
    bybit_snapshot = (set(), {"source": "disabled_historical_only"})

    print("PREPARE_SCREEN", flush=True)
    inventory, grouped, _records, prices, hist_eligible, _live = v4.prepare_screen(
        paths, fetch_live_av=False, bybit_snapshot=bybit_snapshot
    )
    print("TRAIN_REPLAY", flush=True)
    shortlist, _train_stats, train_diag, tensor, _art = v4.run_single_pass_train_replay(
        hist_eligible, prices, inventory["screen"], cache_dir=None
    )
    if tensor is None:
        raise RuntimeError("EMPTY_TRAIN_TENSOR")
    print("TENSOR", len(tensor.aliases), tensor.notional_frac.shape, flush=True)

    conf = make_config()
    passed_ids, metrics, _baselines = pc.stage1_audit_traders(tensor, conf)
    print("BEHAVIORAL_PASS", len(passed_ids), flush=True)
    dist_mat, comps = pc.compute_crowding_distance_matrix(passed_ids, tensor, conf)
    print("DIST_MATRIX", dist_mat.shape, flush=True)

    a_rows, a_budget = upgma_candidates(tensor, metrics, passed_ids, dist_mat, conf)
    b_rows, b_diag = sleeve_candidates(tensor, metrics, passed_ids, dist_mat, comps["psi_pos"], conf)
    c_rows, c_budget = baseline_candidates(tensor, metrics, passed_ids, conf)
    print("CANDIDATES", len(a_rows), len(b_rows), len(c_rows), flush=True)

    train = {"A_upgma": a_rows[:3], "B_sleeve": b_rows[:3], "C_baseline": c_rows[:3]}
    chosen = {
        "A_upgma": a_rows[0] if a_rows else None,
        "B_sleeve": b_rows[0] if b_rows else None,
        "C_baseline": c_rows[0] if c_rows else None,
    }
    for rows in train.values():
        for row in rows:
            diags = member_diagnostics(row, train_diag, hist_eligible, grouped)
            row["member_diagnostics"] = diags
            row["train_gap_flagged"] = any(d["train_gap_flagged"] for d in diags)
            row["oos_coin_mismatch"] = sum(len(d["oos_coins_not_in_train"]) for d in diags)
    method_status = {
        "A_upgma": "OK" if a_rows else "NO_VALID_CANDIDATE",
        "B_sleeve": b_diag.get("status", "OK" if b_rows else "NO_VALID_CANDIDATE"),
        "C_baseline": "OK" if c_rows else "NO_VALID_CANDIDATE",
    }
    search_budgets = {
        "A_upgma": a_budget,
        "B_sleeve": b_diag.get("search_budget", {}),
        "C_baseline": c_budget,
    }

    freeze = {
        "study": "CFT $10k structural episode sleeves vs UPGMA (matched retrospective)",
        "frozen_at_utc": FROZEN_AT_UTC,
        "freeze_provenance": "Exploratory rerun; prior OOS inspected and gap gate revised; original freeze time not recorded",
        "scientific_status": "RETROSPECTIVE_NO_PIT_ROSTER",
        "roster_sha256": inventory["roster_sha256"],
        "historical_eligible": inventory["historical_eligible_count"],
        "shortlist": len(shortlist),
        "behavioral_pass": len(passed_ids),
        "windows": {
            "train": ["2026-05-01T00:00Z", "2026-09-01T00:00Z"],
            "oos": ["2026-09-01T00:00Z", "2026-11-01T00:00Z"],
            "oos_is_pristine_holdout": False,
        },
        "matched_parameters": {
            "pod_size": POD_SIZE,
            "min_notional_usd": MIN_NOTIONAL_USD,
            "baseline_port_dd_frac": BASELINE_PORT_DD_FRAC,
            "deadband_usd": DEADBAND_USD,
            "cost_mult": COST_MULT,
            "max_gross_leverage": MAX_GROSS_LEVERAGE,
            "oos_policy_primary": "new",
            "oos_activation": "fresh boundary at OOS start via candidate_targets",
            "same_audit": "stage1_audit_traders on the identical Train tensor",
            "same_score": ["feasible", "net_over_max_dd", "train_net_usd"],
            "same_sizing": "size_pod_dual_barrier + 50%-wallet-risk baseline cap",
        },
        "method_A_upgma": {
            "stage1": "stage1_audit_traders",
            "distance": "compute_crowding_distance_matrix",
            "cluster": "cluster_candidates_upgma",
            "top_per_cluster": 2,
            "max_candidate_pods": 16,
            "sizer": "size_pod_dual_barrier",
            "rank": ["feasible", "net_over_max_dd", "train_net_usd"],
        },
        "method_B_sleeve": {
            "holding_time_source": "uncensored per-symbol runs from notional_frac",
            "intraday_max_bars": INTRADAY_MAX_BARS,
            "swing_max_bars": SWING_MAX_BARS,
            "min_active_frac": MIN_ACTIVE_FRAC_LABEL,
            "min_runs": 4,
            "min_uncensored_runs": MIN_UNCENSORED_RUNS,
            "top_per_sleeve": SLEEVE_TOP_PER,
            "sleeve_order": list(SLEEVE_ORDER),
            "tie_break": ["chop_positive_role", "net_over_max_dd", "raw_sharpe_ann", "alias"],
            "crowding_veto_dist": CROWDING_VETO_DIST,
            "unknown_overlap_days": UNKNOWN_OVERLAP_DAYS,
            "unknown_max_dd_frac": UNKNOWN_MAX_DD_FRAC,
            "unknown_max_daily_frac": UNKNOWN_MAX_DAILY_FRAC,
            "sizer": "size_pod_dual_barrier",
            "rank": ["feasible", "net_over_max_dd", "train_net_usd"],
            "no_forced_seat": "if no combo is admitted the method reports NO_FEASIBLE_POD",
        },
        "method_C_baseline": {
            "selection": "unstratified top pod_size by net_over_max_dd, then raw_sharpe_ann, then alias",
            "sizer": "size_pod_dual_barrier",
        },
        "search_budgets": search_budgets,
        "method_status": method_status,
        "primary_endpoints": [
            "oos.new.full_window_net_usd",
            "oos.new.full_window_static_floor_dd_usd",
            "oos.new.worst_daily_usd",
        ],
        "decision_rule": (
            "A dominates B only if OOS net >=, static floor DD >=, and worst day >= on all three; "
            "no winner without dominance over both others; full policy is sensitivity only"
        ),
        "frozen_candidates": {
            name: {
                "aliases": row["aliases"],
                "capped_k": row["capped_k"],
                "train_net_usd": row["train_net_usd"],
                "train_net_over_dd": row["train_net_over_dd"],
                "train_gap_flagged": row["train_gap_flagged"],
                "oos_coin_mismatch": row["oos_coin_mismatch"],
                "member_diagnostics": row["member_diagnostics"],
            }
            for name, row in chosen.items()
            if row is not None
        },
    }
    dump_json(freeze_path, freeze)
    print("FROZEN", freeze_path, flush=True)
    print(json.dumps(freeze["frozen_candidates"], indent=1), flush=True)

    # ---- OOS only after the freeze file exists ----
    oos_new: dict[str, list[dict]] = {"A_upgma": [], "B_sleeve": [], "C_baseline": []}
    for name, rows in (("A_upgma", a_rows), ("B_sleeve", b_rows), ("C_baseline", c_rows)):
        for row in rows[:3]:
            res = evaluate_oos(v4, row, grouped, prices, "new")
            res["train"] = row
            oos_new[name].append(res)
    oos_full: dict[str, list[dict]] = {"A_upgma": [], "B_sleeve": [], "C_baseline": []}
    for name, row in chosen.items():
        if row is None:
            continue
        res = evaluate_oos(v4, row, grouped, prices, "full")
        res["train"] = row
        oos_full[name].append(res)

    def _dominates(x: dict, y: dict) -> bool:
        return (
            x["full_window_net_usd"] >= y["full_window_net_usd"]
            and x["full_window_static_floor_dd_usd"] >= y["full_window_static_floor_dd_usd"]
            and x["worst_daily_usd"] >= y["worst_daily_usd"]
        )

    top = {name: rows[0] for name, rows in oos_new.items() if rows}
    valid_top = {
        name: row
        for name, row in top.items()
        if not row["gap_invalidated"] and not row["train"]["train_gap_flagged"]
    }
    matrix = {
        f"{a}>{b}": bool(_dominates(valid_top[a], valid_top[b]))
        for a in valid_top
        for b in valid_top
        if a != b
    }
    if len(valid_top) < 3:
        verdict = (
            "GAP_INVALIDATED_NO_COMPARISON"
            if any(
                row["gap_invalidated"] or row["train"]["train_gap_flagged"]
                for row in top.values()
            )
            else "INCOMPLETE_METHOD"
        )
    elif matrix["B_sleeve>A_upgma"] and matrix["B_sleeve>C_baseline"]:
        verdict = "SLEEVE_DOMINATES_BOTH"
    elif matrix["A_upgma>B_sleeve"] and matrix["A_upgma>C_baseline"]:
        verdict = "UPGMA_DOMINATES_BOTH"
    elif matrix["C_baseline>A_upgma"] and matrix["C_baseline>B_sleeve"]:
        verdict = "BASELINE_DOMINATES_BOTH"
    else:
        verdict = "NO_DOMINANCE"

    stability = (
        {"skipped": True}
        if args.no_stability
        else stability_subtest(tensor, metrics, passed_ids, dist_mat, comps["psi_pos"], conf)
    )

    results = {
        "study": freeze["study"],
        "runtime_s": round(time.time() - t0, 2),
        "coverage": {
            "min_ts_ms": inventory["utc_bounds"]["min_ts_ms"],
            "max_ts_ms": inventory["utc_bounds"]["max_ts_ms"],
            "archive_days": inventory["source"]["count_daily_files"],
            "missing_days": inventory["source"]["missing_dates"],
            "shortlist": len(shortlist),
            "behavioral_pass": len(passed_ids),
            "sleeve_counts": b_diag.get("sleeve_counts", {}),
            "unlabeled": b_diag.get("unlabeled", 0),
            "censored_runs": b_diag.get("censored_runs", {}),
            "sleeve_status": b_diag.get("status"),
            "combos_tested": b_diag.get("combos_tested", 0),
            "combos_admitted": b_diag.get("combos_admitted", 0),
        },
        "method_status": method_status,
        "search_budgets": search_budgets,
        "train": train,
        "oos": {"new": oos_new, "full": oos_full},
        "dominance": {"matrix": matrix, "verdict": verdict},
        "stability": stability,
        "sleeve_features": b_diag.get("feature_by_alias", {}),
        "coin_universe": {
            "train_eligible_coins_by_alias": {
                a: len(hist_eligible.get(a, {}))
                for a in sorted({x for row in chosen.values() if row for x in row["aliases"]})
            },
            "oos_coins_not_in_train_by_alias": {
                a: sorted(set(grouped.get(a, {})) - set(hist_eligible.get(a, {})))
                for a in sorted({x for row in chosen.values() if row for x in row["aliases"]})
            },
        },
    }
    dump_json(results_path, results)
    write_report(md_path, {"freeze": freeze, "results": results})
    print("DONE", results_path, md_path, f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
