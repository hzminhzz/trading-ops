"""Fixed-workload evaluator for the 5m wallet replay.

Contract
--------
pass  : every wallet's per-day net/notional rows are bit-identical to `--mode canonical`
        AND median replay seconds <= budget
fail  : any row mismatch, or median above budget

Workload is fixed and deterministic (first N aliases after sorting, full window),
so modes are directly comparable. Warm-up runs are excluded from the statistics.

  python studies/bench_replay.py --mode canonical --wallets 60 --repeats 3
  python studies/bench_replay.py --mode fast      --wallets 60 --repeats 3
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

STUDIES_DIR = Path(__file__).resolve().parent
if str(STUDIES_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIES_DIR))

import canonical_replay as cr
import fast_replay as fr
import study_data as sd

START_ISO = "2026-05-01T00:00:00Z"
END_ISO = "2026-11-14T00:00:00Z"
GRID_MS = 300_000
BARS_PER_DAY = 86_400_000 // GRID_MS
MIN_COIN_COVERAGE = 0.99
COST_BPS = cr.DEFAULT_COST_BPS
FEE_BPS = cr.DEFAULT_FEE_BPS


def setup() -> tuple[dict, np.ndarray, np.ndarray, dict, list[str]]:
    start = sd.to_utc_ms(datetime.fromisoformat(START_ISO))
    end = sd.to_utc_ms(datetime.fromisoformat(END_ISO))
    paths, _ = sd.discover_fill_archive(start, end)
    records, _ = cr.load_registry_universe(
        Path.home() / ".local/share/copytrade/registry/wallet_registry.sqlite",
        include_excluded=True,
    )
    ing = sd.ingest_fills(records, paths, start=start, end=end, split=start)
    grid = start + np.arange((end - start) // GRID_MS, dtype=np.int64) * GRID_MS
    prices = sd.LazyPrices(
        grid, store_root=sd.DEFAULT_STORE_ROOT, vike_root=sd.DEFAULT_VIKE_ROOT, split_ms=start
    )
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for coin in sorted({c for by in ing.grouped.values() for c in by}):
        try:
            cand = prices.get(coin)
        except (KeyError, ValueError, OSError):
            continue
        if float(prices.stats.get(coin, {}).get("mark_valid_frac", 0.0)) >= MIN_COIN_COVERAGE:
            cache[coin] = cand
    aliases = sorted(ing.grouped)
    return ing.grouped, grid, start, cache, aliases


def run(
    grouped: dict, grid: np.ndarray, start: int, cache: dict, aliases: list[str], mode: str
) -> tuple[list[tuple[str, int, float, float]], dict]:
    n_bars = grid.size
    day_idx = ((grid - start) // 86_400_000).astype(np.int64)
    n_days = int(day_idx.max()) + 1
    rows: list[tuple[str, int, float, float]] = []
    t_pos = t_rep = t_fallback = 0.0
    fallback_coins = 0
    t0 = time.perf_counter()
    for alias in aliases:
        by_coin = grouped.get(alias) or {}
        net = np.zeros(n_bars)
        notional = np.zeros(n_bars)
        for coin, df in by_coin.items():
            px = cache.get(coin)
            if px is None:
                continue
            if mode == "canonical":
                frames, _ = cr.build_unit_replay_by_alias(
                    {alias: {coin: df}}, {coin: px}, [alias], grid,
                    policy="new", activation_ms=start,
                )
                wf = (frames.get(alias) or {}).get(coin)
                if wf is None:
                    continue
                net += wf["net"].to_numpy()
                notional += np.abs(wf["pos"].to_numpy()) * wf["close"].to_numpy()
                continue
            a = time.perf_counter()
            pos = fr.positions_on_grid(
                df["ts"].to_numpy(), df["pre"].to_numpy(),
                df["sign"].to_numpy(), df["sz"].to_numpy(), grid, start,
            )
            b = time.perf_counter()
            fast = fr.replay_fast(pos, px[0], px[1], cost_bps=COST_BPS, fee_bps=FEE_BPS)
            c = time.perf_counter()
            t_pos += b - a
            t_rep += c - b
            if fast is None:
                d = time.perf_counter()
                frames, _ = cr.build_unit_replay_by_alias(
                    {alias: {coin: df}}, {coin: px}, [alias], grid,
                    policy="new", activation_ms=start,
                )
                t_fallback += time.perf_counter() - d
                fallback_coins += 1
                wf = (frames.get(alias) or {}).get(coin)
                if wf is None:
                    continue
                net += wf["net"].to_numpy()
                notional += np.abs(wf["pos"].to_numpy()) * wf["close"].to_numpy()
            else:
                net += fast
                notional += np.abs(pos) * np.nan_to_num(px[1])
        net_d = np.bincount(day_idx, weights=net, minlength=n_days)
        ntl_d = np.bincount(day_idx, weights=notional, minlength=n_days)
        rows.extend(
            (alias, int(d), float(net_d[d]), float(ntl_d[d] / BARS_PER_DAY))
            # mirror the study: a day that booked PnL at zero notional still counts
            for d in np.flatnonzero((ntl_d != 0) | (net_d != 0))
        )
    return rows, {
        "seconds": time.perf_counter() - t0,
        "positions_s": t_pos,
        "replay_s": t_rep,
        "fallback_s": t_fallback,
        "fallback_coins": fallback_coins,
    }


def _run_stages() -> int:
    import importlib.util
    import subprocess
    import tempfile

    t0 = time.perf_counter()
    subprocess.run(
        [sys.executable, "-c", "import numpy, polars, canonical_replay, fast_replay, study_data"],
        cwd=str(STUDIES_DIR),
        check=True,
    )
    t_boot = time.perf_counter() - t0

    study_path = STUDIES_DIR / "2026-11-18__governance__nov-decay-persistence.py"
    study = None
    if study_path.is_file():
        spec = importlib.util.spec_from_file_location("nov_decay", study_path)
        assert spec and spec.loader
        study = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(study)

    registry_db = (
        study.REGISTRY_DB
        if study is not None
        else Path.home() / ".local/share/copytrade/registry/wallet_registry.sqlite"
    )
    eval_start_iso = study.EVAL_START_ISO if study is not None else "2026-09-01T00:00:00Z"
    min_fills = study.MIN_FILLS if study is not None else 150

    t1 = time.perf_counter()
    start = sd.to_utc_ms(datetime.fromisoformat(START_ISO))
    end = sd.to_utc_ms(datetime.fromisoformat(END_ISO))
    paths, _ = sd.discover_fill_archive(start, end)
    t2 = time.perf_counter()
    records, _ = cr.load_registry_universe(registry_db, include_excluded=True)
    t3 = time.perf_counter()
    ing = sd.ingest_fills(records, paths, start=start, end=end, split=start)
    t4 = time.perf_counter()
    eval_cut = sd.to_utc_ms(datetime.fromisoformat(eval_start_iso))
    counts = {a: sum(int((f["ts"].to_numpy() < eval_cut).sum()) for f in by.values()) for a, by in ing.grouped.items()}
    study_aliases = sorted(a for a, n in counts.items() if n >= min_fills)
    t5 = time.perf_counter()

    grid = start + np.arange((end - start) // GRID_MS, dtype=np.int64) * GRID_MS
    prices = sd.LazyPrices(grid, store_root=sd.DEFAULT_STORE_ROOT, vike_root=sd.DEFAULT_VIKE_ROOT, split_ms=start)
    coins = sorted({c for by in ing.grouped.values() for c in by})
    cache, per_coin_ms, excluded = {}, [], 0
    for coin in coins:
        tc0 = time.perf_counter()
        try:
            cand = prices.get(coin)
        except (KeyError, ValueError, OSError):
            per_coin_ms.append((time.perf_counter() - tc0) * 1000.0)
            continue
        if float(prices.stats.get(coin, {}).get("mark_valid_frac", 0.0)) < MIN_COIN_COVERAGE:
            excluded += 1
        else:
            cache[coin] = cand
        per_coin_ms.append((time.perf_counter() - tc0) * 1000.0)
    t6 = time.perf_counter()
    for coin in coins:
        prices.get(coin)
    t7 = time.perf_counter()

    rows, prof = run(ing.grouped, grid, start, cache, study_aliases, "fast")
    t_agg = prof["seconds"] - prof["positions_s"] - prof["replay_s"] - prof["fallback_s"]

    import polars as pl
    daily = pl.DataFrame(rows, schema=["alias", "day", "net", "ntl"], orient="row")
    with tempfile.TemporaryDirectory() as td:
        pq = Path(td) / "daily.parquet"
        tw0 = time.perf_counter()
        daily.write_parquet(pq)
        t_pq_w = time.perf_counter() - tw0
        tr0 = time.perf_counter()
        _ = pl.read_parquet(pq)
        t_pq_r = time.perf_counter() - tr0
    if study is not None:
        te0 = time.perf_counter()
        s_tbl = study.risk_units(study.weekly_table(daily))
        te1 = time.perf_counter()
        _ = study.persistence(s_tbl)
        te2 = time.perf_counter()
        eval_wk = (study.utc_dt(eval_start_iso) - study.utc_dt(study.ANCHOR_ISO)).days // 7
        eval_wks = list(range(eval_wk, int(s_tbl["week"].max()) + 1))
        sealed = study.select_top(s_tbl, eval_wk, eval_wk)
        _ = [study.simulate(s_tbl, p, sealed, eval_wks) for p in ("STATIC_ROSTER", "WEEKLY_TOP3", "MONTHLY_TOP3", "BOTTOM_DECILE_21D")]
        te3 = time.perf_counter()
        stage_e = f"(e) rows={daily.height} weekly+risk={te1-te0:.4f}s persist={te2-te1:.4f}s simulate={te3-te2:.4f}s total={te3-te0:.4f}s"
    else:
        stage_e = f"(e) rows={daily.height} (study script absent; downstream skipped)"

    print(f"(f) startup+imports={t_boot:.3f}s pq_write={t_pq_w:.4f}s pq_read={t_pq_r:.4f}s")
    print(f"(a) discover={t2-t1:.3f}s registry={t3-t2:.3f}s ingest={t4-t3:.3f}s admit={t5-t4:.3f}s total={t5-t1:.3f}s ({len(study_aliases)}/{len(counts)} wallets)")
    print(f"(b) prices={t6-t5:.3f}s (warm_cache={t7-t6:.5f}s) coins={len(coins)} usable={len(cache)} excl={excluded} ms/coin: med={statistics.median(per_coin_ms):.1f} p95={float(np.percentile(per_coin_ms,95)):.1f} max={max(per_coin_ms):.1f}")
    print(f"(c+d) wallets={len(study_aliases)} total={prof['seconds']:.3f}s pos={prof['positions_s']:.3f}s replay={prof['replay_s']:.3f}s fallback={prof['fallback_s']:.3f}s ({prof['fallback_coins']} coins) daily_agg={t_agg:.3f}s")
    print(stage_e)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["canonical", "fast"], default="fast")
    ap.add_argument("--wallets", type=int, default=60)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--budget", type=float, default=None, help="median seconds budget")
    ap.add_argument("--stages", action="store_true", help="measure full-study stage timings")
    args = ap.parse_args()
    if args.stages:
        return _run_stages()

    grouped, grid, start, cache, aliases = setup()
    subset = aliases[: args.wallets]
    print(f"workload: {len(subset)} wallets, {grid.size} bars, {len(cache)} coins, mode={args.mode}")

    ref = None
    if args.mode == "fast":
        ref, _ = run(grouped, grid, start, cache, subset[:10], "canonical")

    stats = []
    mismatch = 0
    for i in range(args.warmup + args.repeats):
        rows, prof = run(grouped, grid, start, cache, subset, args.mode)
        warm = i < args.warmup
        if not warm:
            stats.append(prof["seconds"])
        tag = "warmup" if warm else "run"
        print(
            f"  [{tag}] {prof['seconds']:.2f}s  pos={prof['positions_s']:.2f}s "
            f"replay={prof['replay_s']:.2f}s fallback={prof['fallback_s']:.2f}s "
            f"({prof['fallback_coins']} coins)"
        )
        if ref is not None and not warm:
            if not ref:
                raise SystemExit("reference is empty: the equality check cannot be vacuous")
            ref_aliases = {a for a, *_ in ref}
            got = {(a, d): (n, q) for a, d, n, q in rows if a in ref_aliases}
            want = {(a, d): (n, q) for a, d, n, q in ref}
            # count both directions: a row fast invents is as wrong as one it drops
            bad = [k for k in want if k not in got or got[k] != want[k]]
            extra = [k for k in got if k not in want]
            mismatch += len(bad) + len(extra)

    med = statistics.median(stats)
    spread = max(stats) - min(stats)
    print(f"median={med:.2f}s spread={spread:.2f}s runs={len(stats)}")
    if ref is not None:
        print(f"row mismatches vs canonical (10-wallet reference): {mismatch}")
    if args.budget is not None:
        ok = med <= args.budget
        print(f"budget={args.budget:.2f}s -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            return 1
    return 1 if (ref is not None and mismatch) else 0


if __name__ == "__main__":
    raise SystemExit(main())
