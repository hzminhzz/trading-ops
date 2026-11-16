"""Reproducible May-Nov 2026 CFT $10k study orchestrator (v4): Polars-only & Parquet-cached.

All wallet and broker state is read-only. Intermediate Parquet artifacts are persisted
strictly under gitignored ``data/v4_cache/``; sanitized reports live under ``reports/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
STUDIES_DIR = Path(__file__).resolve().parent
if str(STUDIES_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIES_DIR))

import canonical_replay as cr
import pod_constructor as pc
import study_data as sd

START = 1_777_593_600_000  # 2026-05-01T00:00Z
SPLIT = START + 123 * 86_400_000  # 2026-09-01T00:00Z
END = SPLIT + 61 * 86_400_000  # 2026-11-01T00:00Z
GRID = np.arange(START, END, 300_000, dtype=np.int64)
CUT = int(np.searchsorted(GRID, SPLIT))
TRAIN_GRID = GRID[:CUT]
OOS_GRID = GRID[CUT:]
TRIAD = ("ACT04", "ACT08", "ACT09")
TRIAD_K = (0.00045, 0.00356, 0.00099)
MIN_NOTIONAL_USD = 15.0
ALT_DEADBAND_USD = 30.0
BASELINE_PORT_DD_FRAC = 0.02
POD_SIZES = (3, 4)
DEADBANDS_USD = (15.0, 30.0)
COST_MULTIPLIERS = (1.0, 2.0, 3.0)
POLICIES = ("new", "full")
MAX_GROSS_LEVERAGE = 5.0

_VOL_SCAN_SCHEMA = pl.Schema(
    {
        "timestamp": pl.Datetime("us", "UTC"),
        "open": pl.Float64,
        "volume": pl.Float64,
        "quote_volume": pl.Float64,
    }
)
_FUNDING_SCAN_SCHEMA = pl.Schema(
    {
        "timestamp": pl.Datetime("us", "UTC"),
        "funding_rate": pl.Float64,
        "funding_interval_hours": pl.Int64,
    }
)


@dataclass(frozen=True)
class StudyPaths:
    """Resolved paths for read-only inputs, gitignored Parquet cache, and sanitized reports."""

    registry: Path = Path.home() / ".local/share/copytrade/registry/wallet_registry.sqlite"
    archive: Path = Path.home() / ".local/share/copytrade/research"
    store: Path = sd.DEFAULT_STORE_ROOT
    vike: Path | None = sd.DEFAULT_VIKE_ROOT
    cache_dir: Path = ROOT / "data/v4_cache"
    report_prefix: Path = ROOT / "reports/2026-11-16__cft__10k__canonical-full-universe-v4"

    @property
    def inventory_path(self) -> Path:
        """Gitignored path for raw file-manifest and wallet inventory cache."""
        return self.cache_dir / f"{self.report_prefix.name}__inventory.json"


def _rel_artifact_path(path: Path) -> str:
    """Return repo-relative path when inside ``ROOT`` so committed JSON avoids host prefixes."""
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _json_default(obj: object) -> object:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def dump_json(path: Path, data: dict) -> None:
    """Write formatted deterministic JSON with NumPy scalar conversion."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n"
    )


def prepare_screen(
    paths: StudyPaths = StudyPaths(),
    *,
    fetch_live_av: bool = True,
    bybit_snapshot: tuple[set[str] | Mapping[str, tuple[float, float]], dict] | None = None,
    account_value_fetcher: Callable[
        [cr.WalletRecord], tuple[float | None, dict | None, str | None]
    ]
    | None = None,
) -> tuple[
    dict,
    dict[str, dict[str, pl.DataFrame]],
    list[cr.WalletRecord],
    sd.LazyPrices,
    dict[str, dict[str, pl.DataFrame]],
    dict[str, dict[str, pl.DataFrame]],
]:
    """Ingest fills lazily in UTC, load 1m prices, and separate historical vs live screens."""
    records, roster_sha = cr.load_registry_universe(paths.registry, include_aliases=TRIAD)
    archive_paths, inv_manifest = sd.discover_fill_archive(START, END, archive_root=paths.archive)
    if inv_manifest["missing_dates"]:
        raise RuntimeError(f"MISSING_SOURCE_DAYS:{inv_manifest['missing_dates'][:10]}")

    ingestion = sd.ingest_fills(records, archive_paths, start=START, end=END, split=SPLIT)
    if bybit_snapshot is None:
        bybit_specs, bybit_meta = sd.fetch_bybit_usdt_perp_specs()
        bybit_coins = set(bybit_specs)
    elif isinstance(bybit_snapshot[0], Mapping):
        bybit_specs = {
            str(c): (float(v[0]), float(v[1])) for c, v in bybit_snapshot[0].items()
        }
        bybit_meta = bybit_snapshot[1]
        bybit_coins = set(bybit_specs)
    else:
        bybit_coins = set(bybit_snapshot[0])
        bybit_meta = bybit_snapshot[1]
        bybit_specs = {}
    screen_rows = sd.screen_wallets(
        records,
        ingestion,
        split=SPLIT,
        bybit_usdt_perps=bybit_coins,
        account_value_fetcher=account_value_fetcher,
        fetch_live_av=fetch_live_av and (account_value_fetcher is None),
    )

    prices = sd.LazyPrices(
        GRID,
        store_root=paths.store,
        vike_root=paths.vike,
        start=START,
        end=END,
        split_ms=SPLIT,
    )
    hist_eligible, rows_hist = sd.eligible_fills(
        ingestion.grouped,
        screen_rows,
        prices,
        split=SPLIT,
        scope="historical",
    )
    live_eligible, updated_rows = sd.eligible_fills(
        ingestion.grouped,
        rows_hist,
        prices,
        split=SPLIT,
        scope="live",
    )

    inventory = {
        "study": "May-Nov 2026 CFT $10k canonical v4",
        "status": "RETROSPECTIVE_SCENARIO",
        "window_start": "2026-05-01T00:00Z",
        "train_end": "2026-09-01T00:00Z",
        "window_end": "2026-11-01T00:00Z",
        "utc_bounds": {"min_ts_ms": ingestion.min_ts_ms, "max_ts_ms": ingestion.max_ts_ms},
        "roster_sha256": roster_sha,
        "roster_count": len(records),
        "source": inv_manifest,
        "bybit": bybit_meta,
        "bybit_lot_specs": {
            c: [float(s[0]), float(s[1])] for c, s in sorted(bybit_specs.items())
        },
        "cost_bps_each_execution": cr.DEFAULT_COST_BPS,
        "screen": updated_rows,
        "screen_survivors_historical": sorted(hist_eligible),
        "screen_survivors_live": sorted(live_eligible),
        "historical_eligible_count": len(hist_eligible),
        "live_deployable_count": len(live_eligible),
        "price_source_stats": prices.stats,
        "symbol_screen_basis": "TRAIN_ONLY_FILLS_BEFORE_2026_09_01",
        "pit_roster_provenance": "UNPROVEN_PRESENT_DAY_REGISTRY",
        "source_cwsa_checkpoint_provenance": "UNPROVEN_HISTORICAL_FILLS_ONLY",
    }
    dump_json(paths.inventory_path, inventory)
    return inventory, ingestion.grouped, records, prices, hist_eligible, live_eligible


def _export_train_parquet_artifacts(
    cache_dir: Path,
    shortlist: list[str],
    retained_frames: dict[str, dict[str, pl.DataFrame]],
    retained_equity: dict[str, tuple[np.ndarray, np.ndarray]],
    train_stats: list[dict],
    train_grid: np.ndarray,
) -> dict[str, str]:
    """Persist Train unit streams, solo 5m equity curves, and summary stats to Parquet."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    streams_path = cache_dir / "train_unit_streams.parquet"
    equity_path = cache_dir / "train_unit_equity.parquet"
    summary_path = cache_dir / "train_solo_summary.parquet"

    stream_parts: list[pl.DataFrame] = []
    active_filter = (
        (pl.col("pos").abs() > 1e-12)
        | (pl.col("delta").abs() > 1e-12)
        | (pl.col("gross") != 0.0)
        | (pl.col("cost") != 0.0)
        | (pl.col("net") != 0.0)
    )
    for alias in shortlist:
        for coin, df in sorted(retained_frames.get(alias, {}).items()):
            nz = df.filter(active_filter)
            if nz.is_empty():
                continue
            stream_parts.append(
                nz.with_columns(
                    pl.lit(alias).alias("alias"),
                    pl.lit(coin).alias("coin"),
                ).select(["alias", "coin", "timestamp", "open", "close", "pos", "delta", "gross", "cost", "net"])
            )

    if stream_parts:
        pl.concat(stream_parts).write_parquet(streams_path, compression="zstd")
    else:
        pl.DataFrame(
            schema={
                "alias": pl.String,
                "coin": pl.String,
                "timestamp": pl.Datetime("ms", "UTC"),
                "open": pl.Float64,
                "close": pl.Float64,
                "pos": pl.Float64,
                "delta": pl.Float64,
                "gross": pl.Float64,
                "cost": pl.Float64,
                "net": pl.Float64,
            }
        ).write_parquet(streams_path, compression="zstd")

    eq_parts: list[pl.DataFrame] = []
    ts_series = pl.Series("ts_ms", train_grid, dtype=pl.Int64)
    for alias in shortlist:
        if alias not in retained_equity:
            continue
        net_arr, eq_arr = retained_equity[alias]
        eq_parts.append(
            pl.DataFrame(
                {
                    "alias": [alias] * train_grid.size,
                    "ts_ms": ts_series,
                    "net_usd": net_arr,
                    "equity_usd": eq_arr,
                }
            )
        )
    if eq_parts:
        pl.concat(eq_parts).write_parquet(equity_path, compression="zstd")
    else:
        pl.DataFrame(
            schema={"alias": pl.String, "ts_ms": pl.Int64, "net_usd": pl.Float64, "equity_usd": pl.Float64}
        ).write_parquet(equity_path, compression="zstd")

    summary_rows = [
        {
            "alias": r["alias"],
            "train_fills": int(r["train_fills"]),
            "available_coins": int(r["available_coins"]),
            "net_usd": float(r["net_usd"]),
            "maxdd_usd": float(r["maxdd_usd"]),
            "net_over_dd": float(r["net_over_dd"]),
            "last_train_month_net": float(r["last_train_month_net"]),
            "live_deployable": bool(r["live_deployable"]),
        }
        for r in train_stats
    ]
    pl.DataFrame(summary_rows).write_parquet(summary_path, compression="zstd")
    return {
        "train_unit_streams_parquet": _rel_artifact_path(streams_path),
        "train_unit_equity_parquet": _rel_artifact_path(equity_path),
        "train_solo_summary_parquet": _rel_artifact_path(summary_path),
    }


def run_single_pass_train_replay(
    eligible_hist: dict[str, dict[str, pl.DataFrame]],
    prices: sd.LazyPrices,
    screen_rows: list[dict],
    *,
    cache_dir: Path | None = None,
    train_grid: np.ndarray = TRAIN_GRID,
    start_ms: int = START,
) -> tuple[list[str], list[dict], dict[str, dict[str, int]], pc.UnitCopy5mTensor | None, dict[str, str]]:
    """Replay every Train-eligible wallet exactly once, export Parquet caches, and build Train tensor."""
    n_train = int(train_grid.size)
    row_by_alias = {r["alias"]: r for r in screen_rows}
    all_coins = sorted({c for by in eligible_hist.values() for c in by})
    train_prices = {c: tuple(x[:n_train] for x in prices.get(c)) for c in all_coins}

    train_stats: list[dict] = []
    diagnostics: dict[str, dict[str, int]] = {}
    retained_frames: dict[str, dict[str, pl.DataFrame]] = {}
    retained_equity: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    last_month_offset = min(92 * 288, max(0, n_train - 1))

    for i, (alias, coins) in enumerate(sorted(eligible_hist.items())):
        if i % 25 == 0:
            print("TRAIN_REPLAY", i, "of", len(eligible_hist), flush=True)
        frames, diag = cr.build_unit_replay_by_alias(
            {alias: coins},
            {c: train_prices[c] for c in coins if c in train_prices},
            [alias],
            train_grid,
            policy="new",
            activation_ms=start_ms,
        )
        wallet_frames = frames[alias]
        diagnostics[alias] = diag[alias]
        if not wallet_frames:
            continue

        totals = np.zeros(n_train, dtype=np.float64)
        for f in wallet_frames.values():
            totals += f["net"].to_numpy()
        eq = np.cumsum(totals)
        peak = np.maximum.accumulate(np.maximum(eq, 0.0))
        dd = float(np.max(peak - eq)) if eq.size else 0.0
        net = float(totals.sum())
        last_month = float(totals[last_month_offset:].sum()) if n_train > 92 * 288 else net

        row_meta = row_by_alias.get(alias, {})
        info = {
            "alias": alias,
            "net_usd": round(net, 2),
            "maxdd_usd": round(dd, 2),
            "net_over_dd": round(net / max(dd, 0.01), 4),
            "last_train_month_net": round(last_month, 2),
            "gap": diag[alias],
            "available_coins": len(wallet_frames),
            "train_fills": int(row_meta.get("train_fills", 0)),
            "live_deployable": bool(row_meta.get("live_deployable", False)),
        }
        train_stats.append(info)

        if net > 0.0 and dd > 0.0:
            retained_frames[alias] = wallet_frames
            retained_equity[alias] = (totals, eq)
        else:
            del wallet_frames, frames

    positive = [r for r in train_stats if r["net_usd"] > 0.0 and r["maxdd_usd"] > 0.0]
    positive.sort(key=lambda x: (x["net_over_dd"], x["net_usd"]), reverse=True)
    shortlist = [r["alias"] for r in positive]

    artifacts: dict[str, str] = {}
    if cache_dir is not None:
        artifacts = _export_train_parquet_artifacts(
            cache_dir, shortlist, retained_frames, retained_equity, train_stats, train_grid
        )

    tensor: pc.UnitCopy5mTensor | None = None
    if retained_frames:
        chop_ms = int(train_grid[last_month_offset]) if n_train > 92 * 288 else None
        tensor = pc.build_tensor_from_unit_replays(
            retained_frames,
            train_grid,
            pod_equity_usd=10_000.0,
            chop_split_ms=chop_ms,
        )
    retained_frames.clear()
    retained_equity.clear()
    return shortlist, train_stats, diagnostics, tensor, artifacts


def _select_diverse_candidates(selected: list[dict]) -> list[dict]:
    """Sort and deduplicate up to 5 diverse candidate pods without fabricating feasibility."""
    ordered = sorted(
        selected,
        key=lambda x: (
            x["feasible"],
            len(x["min_lot_warnings"]) == 0,
            x["train_net_dd"],
            x["train_net_usd"],
        ),
        reverse=True,
    )
    candidates: list[dict] = []
    seen: list[set[str]] = []
    for row in ordered:
        aset = set(row["aliases"])
        if any(len(aset & s) >= min(len(aset), len(s)) - 1 for s in seen):
            continue
        seen.append(aset)
        candidates.append(row)
        if len(candidates) >= 5:
            break
    if len(candidates) < 3:
        for row in ordered:
            if row not in candidates:
                candidates.append(row)
            if len(candidates) >= 3:
                break
    return candidates


def _run_pod_discovery_pass(
    tensor: pc.UnitCopy5mTensor,
    row_by_alias: dict[str, dict],
    *,
    pod_sizes: Sequence[int],
    min_baseline_order_usd: float,
    min_lot_usd: float,
    baseline_port_dd_frac: float,
) -> tuple[list[dict], dict]:
    """Run Stage 1/1.5/2 discovery across pod_sizes with 2%-port-DD baseline k cap."""
    selected: list[dict] = []
    auditing: dict = {}
    strict_filter = min_baseline_order_usd > 0.0

    for size in pod_sizes:
        conf = pc.PodConstructorConfig(
            pod_size=size,
            top_per_cluster=2,
            max_candidate_pods=16,
            min_active_bar_frac=0.005,
            min_lot_usd=min_lot_usd,
            min_baseline_order_usd=min_baseline_order_usd,
            baseline_port_dd_frac=baseline_port_dd_frac,
        )
        try:
            sized, metrics, _, components = pc.discover_diverse_pods(tensor, conf)
        except ValueError as exc:
            _, metrics_fail, _ = pc.stage1_audit_traders(tensor, conf)
            auditing[str(size)] = {
                "status": "INSUFFICIENT",
                "error": str(exc),
                "passed": [x.alias for x in metrics_fail if x.passed_behavioral_gate],
                "rejected": [
                    {
                        "alias": x.alias,
                        "reason": x.reject_reason,
                        "tilt": round(x.tilt_multiplier, 3),
                    }
                    for x in metrics_fail
                    if not x.passed_behavioral_gate
                ],
                "sized": 0,
            }
            continue

        auditing[str(size)] = {
            "status": "OK",
            "passed": [x.alias for x in metrics if x.passed_behavioral_gate],
            "rejected": [
                {"alias": x.alias, "reason": x.reject_reason, "tilt": round(x.tilt_multiplier, 3)}
                for x in metrics
                if not x.passed_behavioral_gate
            ],
            "distance_shape": list(components["distance_matrix"].shape),
            "sized": len(sized),
        }

        for s in sized:
            unit_dd = [metrics[i].standalone_max_dd * tensor.pod_equity_usd for i in s.pod_indices]
            base = [
                min((baseline_port_dd_frac * tensor.pod_equity_usd / size) / max(dd, 0.01), conf.max_k_mult)
                for dd in unit_dd
            ]
            raw_opt_k = [float(x) for x in s.multipliers]
            cap_scale = min(1.0, min(b / max(k, 1e-12) for k, b in zip(raw_opt_k, base)))
            capped_k = [k * cap_scale for k in raw_opt_k]
            warnings_capped: list[str] = []
            for k_val, idx in zip(capped_k, s.pod_indices):
                m_info = metrics[idx]
                min_pos_usd = k_val * m_info.active_pos_q10_frac * tensor.pod_equity_usd
                if m_info.active_pos_q10_frac > 0 and min_pos_usd < min_lot_usd:
                    warnings_capped.append(
                        f"{m_info.alias}: q10_order=${min_pos_usd:.2f}<${min_lot_usd:.0f}"
                    )
            non_deployable = {
                a: row_by_alias[a].get("live_reasons", [])
                for a in s.aliases
                if a in row_by_alias and not row_by_alias[a].get("live_deployable", False)
            }
            selected.append(
                {
                    "aliases": list(s.aliases),
                    "train_opt_k": capped_k,
                    "train_uncapped_opt_k": raw_opt_k,
                    "cap_scale": round(cap_scale, 6),
                    "train_half_risk_k": [float(x) for x in base],
                    "train_unit_dd_usd": [float(x) for x in unit_dd],
                    "feasible": s.feasible,
                    "strict_min_notional_passed": strict_filter,
                    "live_deployable": len(non_deployable) == 0 and all(a in row_by_alias for a in s.aliases),
                    "live_non_deployable_members": non_deployable,
                    "train_net_usd": round(s.objective_net_return * cap_scale * tensor.pod_equity_usd, 2),
                    "train_peak_dd_usd": round(s.max_static_dd * cap_scale * tensor.pod_equity_usd, 2),
                    "train_daily_cvar_usd": round(s.daily_cvar_99 * cap_scale * tensor.pod_equity_usd, 2),
                    "train_worst_day_usd": round(s.max_daily_loss * cap_scale * tensor.pod_equity_usd, 2),
                    "train_net_dd": round(s.net_over_max_dd, 4),
                    "netting_savings_usd": round(
                        s.netting_savings_frac * cap_scale * tensor.pod_equity_usd, 2
                    ),
                    "binding_constraint": (
                        s.binding_constraint
                        if cap_scale >= 0.999999
                        else f"BaselineCap({s.binding_constraint})"
                    ),
                    "min_lot_warnings": warnings_capped,
                    "size": size,
                    "indices": list(s.pod_indices),
                }
            )
    return _select_diverse_candidates(selected), auditing


def train_pods(
    tensor: pc.UnitCopy5mTensor,
    screen_rows: list[dict] | None = None,
    *,
    pod_sizes: Sequence[int] = POD_SIZES,
    min_notional_usd: float = MIN_NOTIONAL_USD,
    baseline_port_dd_frac: float = BASELINE_PORT_DD_FRAC,
) -> tuple[list[dict], dict]:
    """Discover pods with strict >=$15 pre-clustering filter, 2% DD cap, and leak-free CPCV/PBO."""
    row_by_alias = {r["alias"]: r for r in (screen_rows or [])}
    candidates, auditing = _run_pod_discovery_pass(
        tensor,
        row_by_alias,
        pod_sizes=pod_sizes,
        min_baseline_order_usd=min_notional_usd,
        min_lot_usd=min_notional_usd,
        baseline_port_dd_frac=baseline_port_dd_frac,
    )

    if candidates:
        auditing["strict_min_notional_status"] = "OK"
        cpcv_conf = pc.PodConstructorConfig(
            pod_size=int(candidates[0]["size"]),
            top_per_cluster=2,
            max_candidate_pods=16,
            min_active_bar_frac=0.005,
            min_lot_usd=min_notional_usd,
            min_baseline_order_usd=min_notional_usd,
            baseline_port_dd_frac=baseline_port_dd_frac,
        )
        if len(candidates) >= 2:
            pbo, logits = pc.compute_leakfree_cpcv_pbo(
                [tuple(r["indices"]) for r in candidates], tensor, cpcv_conf
            )
            auditing["cpcv"] = {
                "pbo": round(pbo, 4),
                "n_splits": int(logits.size),
                "logits": [round(float(x), 4) for x in logits],
            }
        else:
            auditing["cpcv"] = {"pbo": None, "n_splits": 0, "logits": []}
    else:
        auditing["strict_min_notional_status"] = "INSUFFICIENT_FOR_POD_SIZES"
        auditing["cpcv"] = {"pbo": None, "n_splits": 0, "logits": []}

    return candidates, auditing


def candidate_targets(
    aliases: Sequence[str],
    k: Sequence[float],
    grouped: dict[str, dict[str, pl.DataFrame]],
    prices: sd.LazyPrices,
    grid: np.ndarray,
    policy: cr.PolicyType,
    *,
    full_grid: np.ndarray = GRID,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct causal netted member targets and aligned price matrices over ``grid``."""
    coins = sorted({c for a in aliases for c in grouped.get(a, {})})
    prices_by = {c: prices.get(c) for c in coins}
    start = int(np.searchsorted(full_grid, grid[0]))
    stop = start + int(grid.size)
    member_pos = np.zeros((len(aliases), len(coins), grid.size), dtype=np.float64)
    for ai, alias in enumerate(aliases):
        by_coin = grouped.get(alias, {})
        for ci, coin in enumerate(coins):
            f = by_coin.get(coin)
            if f is None:
                continue
            recon = cr.reconstruct_positions_on_grid(
                f, grid, policy=policy, activation_ms=int(grid[0])
            )
            member_pos[ai, ci] = recon.pos
    net = cr.net_member_targets(member_pos, k)
    if not coins:
        empty = np.zeros((0, grid.size), dtype=np.float64)
        return coins, net, empty, empty.copy()
    trade = np.vstack([prices_by[c][0][start:stop] for c in coins])
    mark = np.vstack([prices_by[c][1][start:stop] for c in coins])
    return coins, net, trade, mark


def _epoch_ms_to_utc_datetime(timestamps_ms: np.ndarray) -> pl.DataFrame:
    """Convert millisecond epoch array into a single-column UTC ``Datetime('us')`` DataFrame."""
    return pl.DataFrame(
        {
            "timestamp": pl.from_epoch(
                pl.Series("timestamp", timestamps_ms, dtype=pl.Int64), time_unit="ms"
            )
            .dt.replace_time_zone("UTC")
            .cast(pl.Datetime("us", "UTC"))
        }
    )


def load_1m_volumes_on_grid(
    coins: Sequence[str],
    grid_ms: np.ndarray,
    prices: sd.LazyPrices,
    vol_cache: dict[tuple[str, int, int], np.ndarray],
) -> np.ndarray:
    """Load 1m USD dollar volumes at execution bars ``grid_ms + 60_000`` for candidate coins."""
    if not coins:
        return np.zeros((0, grid_ms.size), dtype=np.float64)

    g_key = (int(grid_ms[0]), int(grid_ms.size))
    missing = [c for c in coins if (c, *g_key) not in vol_cache]
    if missing:
        trade_grid_df = _epoch_ms_to_utc_datetime(grid_ms + 60_000)
        start_dt = sd.to_utc_datetime(int(grid_ms[0]))
        end_dt = sd.to_utc_datetime(int(grid_ms[-1]) + 300_000)
        start_lit = pl.lit(start_dt, dtype=pl.Datetime("us", "UTC"))
        end_lit = pl.lit(end_dt, dtype=pl.Datetime("us", "UTC"))

        for coin in missing:
            key = (coin, *g_key)
            paths = prices.coin_paths(coin)
            if not paths:
                vol_cache[key] = np.full(grid_ms.size, np.nan, dtype=np.float64)
                continue
            lf = (
                pl.scan_parquet(
                    paths,
                    schema=_VOL_SCAN_SCHEMA,
                    hive_partitioning=False,
                    missing_columns="insert",
                    extra_columns="ignore",
                    cast_options=sd.OHLCV_CAST_OPTIONS,
                )
                .filter((pl.col("timestamp") >= start_lit) & (pl.col("timestamp") < end_lit))
                .with_columns(
                    pl.coalesce(
                        pl.col("quote_volume"),
                        pl.col("volume") * pl.col("open"),
                    ).alias("vol_usd")
                )
                .select(["timestamp", "vol_usd"])
                .unique(subset="timestamp", keep="first")
            )
            bars = lf.collect(engine=prices.engine)
            joined = (
                trade_grid_df.join(bars, on="timestamp", how="left")["vol_usd"]
                .fill_null(float("nan"))
                .to_numpy()
                .astype(np.float64, copy=True)
            )
            vol_cache[key] = joined

    return np.vstack([vol_cache[(c, *g_key)] for c in coins])


def load_usdt_funding_rates_on_grid(
    coins: Sequence[str],
    grid_ms: np.ndarray,
    store_root: Path | str,
    fund_cache: dict[tuple[str, int, int], np.ndarray],
) -> tuple[np.ndarray, int]:
    """Load local USDT perp funding rates prorated to 5m bars via backward asof join."""
    if not coins:
        return np.zeros((0, grid_ms.size), dtype=np.float64), 0

    root = Path(store_root).expanduser()
    g_key = (int(grid_ms[0]), int(grid_ms.size))
    missing = [c for c in coins if (c, *g_key) not in fund_cache]
    if missing:
        grid_df = _epoch_ms_to_utc_datetime(grid_ms)
        start_lit = pl.lit(sd.to_utc_datetime(int(grid_ms[0]) - 12 * 3_600_000), dtype=pl.Datetime("us", "UTC"))
        end_lit = pl.lit(sd.to_utc_datetime(int(grid_ms[-1]) + 300_000), dtype=pl.Datetime("us", "UTC"))

        for coin in missing:
            key = (coin, *g_key)
            f_dir = root / f"crypto_futures_funding_{coin}USDT_PERP"
            f_paths = sorted(str(p) for p in f_dir.glob("year=*/month=*/data.parquet")) if f_dir.is_dir() else []
            if not f_paths:
                fund_cache[key] = np.zeros(grid_ms.size, dtype=np.float64)
                continue
            lf = (
                pl.scan_parquet(
                    f_paths,
                    schema=_FUNDING_SCAN_SCHEMA,
                    hive_partitioning=False,
                    missing_columns="insert",
                    extra_columns="ignore",
                    cast_options=sd.OHLCV_CAST_OPTIONS,
                )
                .filter((pl.col("timestamp") >= start_lit) & (pl.col("timestamp") < end_lit))
                .with_columns(
                    (
                        pl.col("funding_rate")
                        / (pl.coalesce(pl.col("funding_interval_hours"), pl.lit(8)).cast(pl.Float64) * 12.0)
                    ).alias("rate_5m")
                )
                .select(["timestamp", "rate_5m"])
                .unique(subset="timestamp", keep="first")
                .sort("timestamp")
            )
            fdf = lf.collect()
            if fdf.is_empty():
                fund_cache[key] = np.zeros(grid_ms.size, dtype=np.float64)
                continue
            joined = (
                grid_df.join_asof(
                    fdf,
                    on="timestamp",
                    strategy="backward",
                    tolerance=timedelta(hours=12),
                )["rate_5m"]
                .fill_null(0.0)
                .to_numpy()
                .astype(np.float64, copy=True)
            )
            fund_cache[key] = joined

    covered = sum(1 for c in coins if np.any(fund_cache[(c, *g_key)] != 0.0))
    return np.vstack([fund_cache[(c, *g_key)] for c in coins]), covered


def compute_1m_participation_diagnostics(
    coins: Sequence[str],
    run_result: cr.ChallengeRunResult | np.ndarray,
    trade_open: np.ndarray,
    mark_close: np.ndarray,
    vol_usd_mat: np.ndarray,
    grid_ms: np.ndarray,
) -> dict:
    """Compute Guardian-executed trade notionals vs 1m bar USD volume participation diagnostics."""
    if isinstance(run_result, cr.ChallengeRunResult):
        if run_result.executed_deltas is None:
            raise ValueError("MISSING_EXECUTED_DELTAS")
        executed_deltas = run_result.executed_deltas
    else:
        executed_deltas = run_result

    _nc, n = executed_deltas.shape
    ratios: list[float] = []
    trade_count = 0
    no_vol_trades = 0
    total_notional = 0.0
    max_part = -1.0
    max_coin: str | None = None
    max_ts: int | None = None

    for ci, coin in enumerate(coins):
        delta_c = executed_deltas[ci]
        to_c = trade_open[ci]
        mc_c = mark_close[ci]
        vol_c = vol_usd_mat[ci]
        for j in range(n):
            exec_delta = abs(float(delta_c[j]))
            if exec_delta <= 1e-12:
                continue
            t_c = float(to_c[j])
            m_c = float(mc_c[j])
            exec_px = t_c if np.isfinite(t_c) else m_c
            if not np.isfinite(exec_px):
                continue
            notional = exec_delta * exec_px
            trade_count += 1
            total_notional += notional
            b_vol = float(vol_c[j])
            if np.isfinite(b_vol) and b_vol > 0.0:
                pct = (notional / b_vol) * 100.0
                ratios.append(pct)
                if pct > max_part:
                    max_part = pct
                    max_coin = coin
                    max_ts = int(grid_ms[j])
            else:
                no_vol_trades += 1

    cov_trades = len(ratios)
    arr = np.asarray(ratios, dtype=np.float64)
    return {
        "trade_count": trade_count,
        "volume_covered_trades": cov_trades,
        "trades_without_1m_volume": no_vol_trades,
        "volume_coverage_frac": round(cov_trades / trade_count, 4) if trade_count else 0.0,
        "total_traded_notional_usd": round(total_notional, 2),
        "median_participation_pct": round(float(np.median(arr)), 4) if cov_trades else None,
        "p95_participation_pct": round(float(np.quantile(arr, 0.95)), 4) if cov_trades else None,
        "p99_participation_pct": round(float(np.quantile(arr, 0.99)), 4) if cov_trades else None,
        "max_participation_pct": round(max_part, 4) if cov_trades else None,
        "max_participation_coin": max_coin,
        "max_participation_ts_ms": max_ts,
        "trades_gt_1pct_vol": int(np.count_nonzero(arr > 1.0)) if cov_trades else 0,
        "trades_gt_10pct_vol": int(np.count_nonzero(arr > 10.0)) if cov_trades else 0,
    }


def result_dict(x: cr.ChallengeRunResult) -> dict:
    """Serialize ChallengeRunResult metrics into sanitized JSON-ready fields."""
    return {
        "verdict": x.verdict,
        "terminal_ts_ms": x.terminal_ts_ms,
        "terminal_net_usd": x.terminal_net_usd,
        "worst_daily_usd": x.worst_daily_usd,
        "static_floor_dd_usd": x.static_floor_dd_usd,
        "peak_to_trough_dd_usd": x.peak_to_trough_dd_usd,
        "sweeps": x.sweeps,
        "gap_exits": x.forced_gap_exits,
        "unpriced_open_steps": x.unpriced_open_steps,
        "deferred_notional_usd": x.deferred_notional_usd,
        "peak_gross_leverage": round(x.peak_gross_leverage, 4),
        "total_funding_usd": x.total_funding_usd,
        "full_window_net_usd": x.full_window_net_usd,
        "full_window_peak_to_trough_dd_usd": x.full_window_peak_to_trough_dd_usd,
        "full_window_static_floor_dd_usd": x.full_window_static_floor_dd_usd,
        "full_window_sweeps": x.full_window_sweeps,
        "full_window_peak_gross_leverage": round(x.full_window_peak_gross_leverage, 4),
        "full_window_funding_usd": x.full_window_funding_usd,
    }


def _append_oos_cache_rows(
    candidate_idx: int,
    policy: str,
    deadband_usd: float,
    cost_mult: float,
    grid_ms: np.ndarray,
    obj: cr.ChallengeRunResult,
    blocks: tuple,
    equity_sink: list[pl.DataFrame] | None,
    block_sink: list[pl.DataFrame] | None,
) -> None:
    if equity_sink is not None:
        equity_sink.append(
            pl.DataFrame(
                {
                    "candidate_idx": [candidate_idx] * grid_ms.size,
                    "policy": [policy] * grid_ms.size,
                    "deadband_usd": [deadband_usd] * grid_ms.size,
                    "cost_mult": [cost_mult] * grid_ms.size,
                    "ts_ms": pl.Series("ts_ms", grid_ms, dtype=pl.Int64),
                    "step_pnl_usd": obj.step_pnl,
                    "equity_usd": obj.equity_curve,
                }
            )
        )
    if block_sink is not None:
        is_chop, d_end, d_max, d_min, d_swp, d_mf = blocks
        n_d = int(d_end.size)
        if n_d > 0:
            block_sink.append(
                pl.DataFrame(
                    {
                        "candidate_idx": [candidate_idx] * n_d,
                        "policy": [policy] * n_d,
                        "deadband_usd": [deadband_usd] * n_d,
                        "cost_mult": [cost_mult] * n_d,
                        "day_idx": np.arange(n_d, dtype=np.int32),
                        "is_chop": is_chop,
                        "day_end_usd": d_end,
                        "day_max_usd": d_max,
                        "day_min_usd": d_min,
                        "day_swept": d_swp,
                        "day_max_first": d_mf,
                    }
                )
            )


def evaluate_one(
    row: dict,
    grouped: dict[str, dict[str, pl.DataFrame]],
    prices: sd.LazyPrices,
    grid: np.ndarray,
    *,
    full_grid: np.ndarray = GRID,
    store_root: Path | str = sd.DEFAULT_STORE_ROOT,
    vol_cache: dict[tuple[str, int, int], np.ndarray] | None = None,
    fund_cache: dict[tuple[str, int, int], np.ndarray] | None = None,
    montecarlo: bool = True,
    pm03: bool = False,
    equity_sink: list[pl.DataFrame] | None = None,
    block_sink: list[pl.DataFrame] | None = None,
    deadbands: Sequence[float] = DEADBANDS_USD,
    cost_multipliers: Sequence[float] = COST_MULTIPLIERS,
    mc_n_paths: int = 5000,
    include_volume_participation: bool = True,
    include_funding_stress: bool = True,
    amount_step_by_coin: Mapping[str, float] | None = None,
    min_amount_by_coin: Mapping[str, float] | None = None,
    max_gross_leverage: float | None = MAX_GROSS_LEVERAGE,
) -> dict:
    """Evaluate a candidate pod across ``new``/``full`` policies, deadbands, and cost multipliers."""
    aliases = row["aliases"]
    k = row["train_opt_k"]
    cand_idx = int(row.get("index", 0))
    v_cache = vol_cache if vol_cache is not None else {}
    f_cache = fund_cache if fund_cache is not None else {}
    mask = grid >= int(datetime(2026, 10, 1, tzinfo=UTC).timestamp() * 1000)
    out: dict = {}

    def _guardian_spec(db_usd: float, *, cost_mult: float = 1.0, funding_bps: float = 0.0) -> cr.GuardianSpec:
        return cr.GuardianSpec(
            min_trade_notional_usd=float(db_usd),
            cost_multiplier=float(cost_mult),
            funding_bps_per_8h=float(funding_bps),
            amount_step_by_coin=amount_step_by_coin,
            min_amount_by_coin=min_amount_by_coin,
            max_gross_leverage=max_gross_leverage,
        )

    for policy in POLICIES:
        coins, tgt, trade, mark = candidate_targets(
            aliases, k, grouped, prices, grid, policy, full_grid=full_grid
        )
        vol_mat = (
            load_1m_volumes_on_grid(coins, grid, prices, v_cache)
            if include_volume_participation
            else None
        )
        fund_mat, fund_covered = (
            load_usdt_funding_rates_on_grid(coins, grid, store_root, f_cache)
            if include_funding_stress
            else (None, 0)
        )
        db_map: dict[str, dict] = {}

        for db in deadbands:
            db_key = str(int(db))
            cm_map: dict[str, dict] = {}
            base_blocks = None
            base_obj = None

            for mult in cost_multipliers:
                spec = _guardian_spec(db, cost_mult=mult)
                obj, blocks = cr.simulate_guardian_challenge(
                    coins, tgt, trade, mark, grid, spec, chop_mask=mask
                )
                cm_key = f"{int(mult)}x"
                cm_map[cm_key] = result_dict(obj)
                if abs(mult - 1.0) < 1e-9:
                    base_obj, base_blocks = obj, blocks
                    _append_oos_cache_rows(
                        cand_idx, policy, float(db), float(mult), grid, obj, blocks, equity_sink, block_sink
                    )
                elif equity_sink is not None:
                    _append_oos_cache_rows(
                        cand_idx, policy, float(db), float(mult), grid, obj, blocks, equity_sink, None
                    )

            db_entry: dict = {"cost_multipliers": cm_map}
            if include_volume_participation and vol_mat is not None and base_obj is not None:
                db_entry["volume_participation"] = compute_1m_participation_diagnostics(
                    coins, base_obj, trade, mark, vol_mat, grid
                )

            if include_funding_stress and fund_mat is not None and abs(db - 15.0) < 1e-9:
                obj_hist, _ = cr.simulate_guardian_challenge(
                    coins,
                    tgt,
                    trade,
                    mark,
                    grid,
                    _guardian_spec(db),
                    chop_mask=mask,
                    funding_rates=fund_mat,
                )
                obj_1bp, _ = cr.simulate_guardian_challenge(
                    coins,
                    tgt,
                    trade,
                    mark,
                    grid,
                    _guardian_spec(db, funding_bps=1.0),
                    chop_mask=mask,
                )
                db_entry["funding_stress"] = {
                    "historical_usdt_proxy": {
                        **result_dict(obj_hist),
                        "funding_covered_coins": fund_covered,
                        "funding_total_coins": len(coins),
                    },
                    "constant_1bp_8h": result_dict(obj_1bp),
                }

            if montecarlo and base_blocks is not None and base_obj is not None:
                mc_map: dict[str, dict] = {}
                for k_mult in (1.0, 1.5, 2.0):
                    if abs(k_mult - 1.0) < 1e-9:
                        mblocks = base_blocks
                    else:
                        _, mblocks = cr.simulate_guardian_challenge(
                            coins,
                            tgt * k_mult,
                            trade,
                            mark,
                            grid,
                            _guardian_spec(db),
                            chop_mask=mask,
                        )
                    for regime in ("Blended", "75%-Chop"):
                        seed = 20261116 + int(k_mult * 100) + int(db) + (0 if regime == "Blended" else 1)
                        mc_map[f"{k_mult:.1f}x/{regime}"] = pc.run_stationary_block_mc_on_daily_blocks(
                            mblocks,
                            regime,
                            n_paths=mc_n_paths,
                            horizon_days=90,
                            mean_block_days=6.0,
                            rng=np.random.default_rng(seed),
                        )
                db_entry["mc"] = mc_map

            db_map[db_key] = db_entry

        pol_entry: dict = {"deadbands": db_map}
        if pm03 and policy == "new":
            n7 = min(288 * 7, int(grid.size))
            try:
                pm = cr.reconcile_portfolio_with_pm03(
                    coins[:], grid[:n7], tgt[:, :n7], trade[:, :n7], mark[:, :n7]
                )
                check, _ = cr.simulate_guardian_challenge(
                    coins,
                    tgt[:, :n7],
                    trade[:, :n7],
                    mark[:, :n7],
                    grid[:n7],
                    cr.GuardianSpec(
                        profit_target=1e10,
                        max_static_dd=1e10,
                        daily_loss_limit=1e10,
                        guardian_sweep_loss=1e10,
                        min_trade_notional_usd=0.0,
                    ),
                )
                pm["tier1_7d_net"] = check.full_window_net_usd
                pm["drift_usd"] = round(pm["net_pnl"] - check.full_window_net_usd, 2)
                pm["window_utc"] = "first 7 OOS days; $0 deadband vs PM03 Decimal parity"
                pol_entry["pm03"] = pm
            except (ValueError, RuntimeError, TypeError) as exc:
                pol_entry["pm03"] = {"status": "ERROR", "detail": type(exc).__name__}

        out[policy] = pol_entry
    return out


def export_oos_parquet_artifacts(
    cache_dir: Path,
    equity_records: Sequence[pl.DataFrame],
    block_records: Sequence[pl.DataFrame],
) -> dict[str, str]:
    """Export OOS 5m equity curves and MC daily blocks into compressed Parquet artifacts."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    eq_path = cache_dir / "oos_equity_curves.parquet"
    blk_path = cache_dir / "oos_daily_blocks.parquet"

    if equity_records:
        pl.concat(list(equity_records)).write_parquet(eq_path, compression="zstd")
    else:
        pl.DataFrame(
            schema={
                "candidate_idx": pl.Int64,
                "policy": pl.String,
                "deadband_usd": pl.Float64,
                "cost_mult": pl.Float64,
                "ts_ms": pl.Int64,
                "step_pnl_usd": pl.Float64,
                "equity_usd": pl.Float64,
            }
        ).write_parquet(eq_path, compression="zstd")

    if block_records:
        pl.concat(list(block_records)).write_parquet(blk_path, compression="zstd")
    else:
        pl.DataFrame(
            schema={
                "candidate_idx": pl.Int64,
                "policy": pl.String,
                "deadband_usd": pl.Float64,
                "cost_mult": pl.Float64,
                "day_idx": pl.Int32,
                "is_chop": pl.Boolean,
                "day_end_usd": pl.Float64,
                "day_max_usd": pl.Float64,
                "day_min_usd": pl.Float64,
                "day_swept": pl.Int32,
                "day_max_first": pl.Boolean,
            }
        ).write_parquet(blk_path, compression="zstd")

    return {
        "oos_equity_curves_parquet": _rel_artifact_path(eq_path),
        "oos_daily_blocks_parquet": _rel_artifact_path(blk_path),
    }


def fetch_hyperx_context(
    shortlist: Sequence[str],
    records: Sequence[cr.WalletRecord],
) -> dict[str, dict]:
    """Fetch 30d HyperX discovery context strictly post-shortlist (never used for selection)."""
    rec_by_label = {r.label: r for r in records}

    def _one(alias: str) -> tuple[str, dict]:
        rec = rec_by_label.get(alias)
        if rec is None:
            return alias, {"status": "UNPROVEN", "error": "MISSING_RECORD"}
        try:
            req = urllib.request.Request(
                f"https://data-api.hyperx.trade/wallet_metrics/{rec.address}?period=30d",
                headers={"User-Agent": sd.USER_AGENT},
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                metrics = json.load(response).get("metrics", {})
            return alias, {
                "pnl_30d": round(float(metrics.get("total_realized_pnl", 0)), 2),
                "trades_30d": int(metrics.get("total_complete_trade_count", 0)),
                "win_rate_30d": round(float(metrics.get("win_rate", 0)), 4),
            }
        except (OSError, KeyError, TypeError, ValueError) as exc:
            return alias, {"status": "UNPROVEN", "error": type(exc).__name__}

    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for alias, val in pool.map(_one, shortlist):
            out[alias] = val
    return out


def conflict(reference: tuple, candidate: tuple) -> dict:
    """Count opposite-direction overlap intervals and coin-hours against Triad A."""
    rc, rq, _, _ = reference
    cc, cq, _, _ = candidate
    common = set(rc) & set(cc)
    count = 0
    hours = 0.0
    coins: list[str] = []
    for coin in sorted(common):
        opp = rq[rc.index(coin)] * cq[cc.index(coin)] < -1e-12
        n_opp = int(np.count_nonzero(opp))
        if n_opp:
            count += int(np.sum(np.diff(opp.astype(np.int8), prepend=0) == 1))
            hours += 5.0 * n_opp / 60.0
            coins.append(coin)
    return {"events": count, "coin_hours": round(hours, 2), "symbols": coins}


def _slice_tensor_for_aliases(
    tensor: pc.UnitCopy5mTensor,
    aliases: Sequence[str],
) -> pc.UnitCopy5mTensor | None:
    wanted = [i for i, a in enumerate(tensor.aliases) if a in set(aliases)]
    if not wanted:
        return None
    sub, _ = pc._slice_tensor_for_traders(tensor, [tuple(wanted)])
    return sub


def render_markdown_report(
    *,
    inventory: dict,
    freeze: dict,
    output: dict,
    script_rel: str = "studies/2026-11-16__cft__10k__canonical-full-universe-v4.py",
) -> str:
    """Render sanitized Markdown summary report covering all v4 metrics and caveats."""
    cpcv_info = freeze.get("train_audit", {}).get("cpcv", {})
    pbo_val = cpcv_info.get("pbo")
    results = output.get("results", [])
    ref_stats = output.get("triada", {})

    lines: list[str] = [
        "---",
        "firm: crypto-fund-trader",
        "account_usd: 10000",
        "status: RESEARCH",
        "top_candidate: NOT_SCIENTIFICALLY_QUALIFIED",
        f"script: {script_rel}",
        "---",
        "",
        "# CFT canonical 5m portfolio research (v4) — May–November 2026",
        "",
        (
            f"Full roster: {inventory['roster_count']}; SHA256 `{inventory['roster_sha256']}`. "
            f"Train `2026-05-01T00:00Z`–`2026-09-01T00:00Z`; OOS `2026-09-01T00:00Z`–`2026-11-01T00:00Z` (exclusive). "
            f"Runtime: **{output['elapsed_s']:.2f}s**."
        ),
        "",
        (
            f"Separated eligibility screens — **historical (`scope=historical`)**: "
            f"{inventory['historical_eligible_count']} / {inventory['roster_count']} Train-eligible; "
            f"**live deployable (`scope=live`, Bybit + HL AV >= $1,000)**: "
            f"{inventory['live_deployable_count']} / {inventory['roster_count']}."
        ),
        "",
        "All figures are retrospective modeled follower net returns, not live Bybit fills or PIT portfolio selection.",
        "",
    ]

    if results:
        lines += [
            "## Candidate summary (primary `$15` deadband, `1x` cost)",
            "",
            "| Train rank | Members | Live deployable | Frozen k | Train new net/static DD | OOS new net/static DD | OOS full net/static DD | OOS worst daily | Peak gross lev | OOS sweeps | 90d MC 1x blended pass | Triad A conflict coin-h |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in results:
            tr15 = r["train"]["new"]["deadbands"]["15"]["cost_multipliers"]["1x"]
            oos_new_15 = r["oos"]["new"]["deadbands"]["15"]["cost_multipliers"]["1x"]
            oos_full_15 = r["oos"]["full"]["deadbands"]["15"]["cost_multipliers"]["1x"]
            mc1x = r["oos"]["new"]["deadbands"]["15"]["mc"]["1.0x/Blended"]["p_pass"]
            live_str = "YES" if r["optimization"].get("live_deployable") else "NO"
            lines.append(
                "| {idx} | {members} | {live} | {ks} | {tn:+.0f} / {td:.0f} | {on:+.0f} / {od:.0f} | "
                "{fn:+.0f} / {fd:.0f} | {wd:.0f} | {lev:.2f}x | {sw} | {mc:.2f}% | {conf:.1f} |".format(
                    idx=r["index"],
                    members=", ".join(r["aliases"]),
                    live=live_str,
                    ks=", ".join(f"{a}={v:.6g}" for a, v in zip(r["aliases"], r["k"])),
                    tn=tr15["full_window_net_usd"],
                    td=tr15["full_window_static_floor_dd_usd"],
                    on=oos_new_15["full_window_net_usd"],
                    od=oos_new_15["full_window_static_floor_dd_usd"],
                    fn=oos_full_15["full_window_net_usd"],
                    fd=oos_full_15["full_window_static_floor_dd_usd"],
                    wd=oos_new_15["worst_daily_usd"],
                    lev=oos_new_15["full_window_peak_gross_leverage"],
                    sw=oos_new_15["sweeps"],
                    mc=mc1x,
                    conf=r.get("triada_conflicts", {}).get("coin_hours", 0.0),
                )
            )

        lines += [
            "",
            "## Deadband (`$15` vs `$30`) & Cost Multiplier (`1x` / `2x` / `3x`) Stress + 1m Participation",
            "",
            "| Pod | Policy | Deadband | 1x net / static DD | 2x net / static DD | 3x net / static DD | Peak gross lev | 1m vol cov | 1m participation (med / p95 / max) | Trades >1% vol |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in results:
            for pol in POLICIES:
                for db in ("15", "30"):
                    db_node = r["oos"][pol]["deadbands"][db]
                    cm = db_node["cost_multipliers"]
                    vp = db_node["volume_participation"]
                    if vp["max_participation_pct"] is not None:
                        coin_tag = vp.get("max_participation_coin") or "?"
                        part_str = (
                            f"{vp['median_participation_pct']:.3f}% / "
                            f"{vp['p95_participation_pct']:.2f}% / "
                            f"{vp['max_participation_pct']:.2f}% ({coin_tag})"
                        )
                    else:
                        part_str = "N/A"
                    lines.append(
                        "| {idx} | {pol} | ${db} | {n1:+.0f} / {d1:.0f} | {n2:+.0f} / {d2:.0f} | "
                        "{n3:+.0f} / {d3:.0f} | {lev:.2f}x | {cov:.1f}% | {mp} | {gt1} |".format(
                            idx=r["index"],
                            pol=pol,
                            db=db,
                            n1=cm["1x"]["full_window_net_usd"],
                            d1=cm["1x"]["full_window_static_floor_dd_usd"],
                            n2=cm["2x"]["full_window_net_usd"],
                            d2=cm["2x"]["full_window_static_floor_dd_usd"],
                            n3=cm["3x"]["full_window_net_usd"],
                            d3=cm["3x"]["full_window_static_floor_dd_usd"],
                            lev=cm["1x"]["full_window_peak_gross_leverage"],
                            cov=vp["volume_coverage_frac"] * 100.0,
                            mp=part_str,
                            gt1=vp["trades_gt_1pct_vol"],
                        )
                    )

        lines += [
            "",
            "## Perpetual Funding Sensitivity (`$15` deadband, `1x` cost)",
            "",
            "| Pod | Policy | USDT proxy funding ($) | Proxy funding-adj net ($) | Covered coins | +1 bp/8h carry ($) | +1 bp/8h adj net ($) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for r in results:
            for pol in POLICIES:
                fs = r["oos"][pol]["deadbands"]["15"]["funding_stress"]
                hp = fs["historical_usdt_proxy"]
                c1 = fs["constant_1bp_8h"]
                lines.append(
                    f"| {r['index']} | {pol} | {hp['full_window_funding_usd']:+.2f} | "
                    f"{hp['full_window_net_usd']:+.2f} | {hp['funding_covered_coins']}/{hp['funding_total_coins']} | "
                    f"{c1['full_window_funding_usd']:+.2f} | {c1['full_window_net_usd']:+.2f} |"
                )
    else:
        lines += [
            "## Candidate summary",
            "",
            (
                "**INSUFFICIENT STRICT CANDIDATES:** Fewer than 3 Train-positive wallets passed the strict "
                "`min_baseline_order_usd=$15` pre-clustering filter across pod sizes `(3, 4)`; no strict candidates were fabricated."
            ),
        ]

    lines += [
        "",
        "## Frozen sizing & pre-clustering `$15` min-notional audit",
        (
            "The optimizer fits exclusively on Train (`2026-05-01..2026-09-01`). The 50%-wallet-risk baseline equals "
            "`2%` portfolio drawdown (`$200` on `$10,000`, split evenly per wallet); optimized `k` is capped at `<= 1.0x` "
            "of each wallet's baseline, and wallets with baseline `q10_order < $15` are filtered prior to UPGMA clustering."
        ),
    ]
    train_audit = freeze.get("train_audit", {})
    for sz in POD_SIZES:
        info = train_audit.get(str(sz), {})
        passed_n = len(info.get("passed", []))
        rej = info.get("rejected", [])
        below15 = sum(1 for x in rej if "below_min_baseline_order" in x.get("reason", ""))
        lines.append(
            f"- **Pod size {sz}**: status=`{info.get('status', 'N/A')}`, passed={passed_n}, "
            f"rejected={len(rej)} (below `$15` baseline filter={below15})."
        )
    for i, s in enumerate(freeze.get("frozen_portfolios", []), 1):
        kcmp = ", ".join(
            f"{alias} baseline={base:.7g}, optimized={opt:.7g} ({opt / max(base, 1e-12):.2f}x)"
            for alias, base, opt in zip(s["aliases"], s["train_half_risk_k"], s["train_opt_k"])
        )
        non_dep = s.get("live_non_deployable_members", {})
        live_note = "all live-deployable" if s.get("live_deployable") else f"live-excluded={non_dep}"
        lines.append(
            f"- **Pod {i}**: {kcmp}; binding={s['binding_constraint']}; feasible={s['feasible']}; "
            f"{live_note}; min-lot warnings: {', '.join(s['min_lot_warnings']) or 'none'}."
        )

    live_audit = freeze.get("live_scoped_audit", {})
    if live_audit:
        lines.append(
            f"- **Live-deployable subset (`scope=live`) pod audit**: "
            f"live shortlist={live_audit.get('live_shortlist_count', 0)}, "
            f"live-only candidates={len(live_audit.get('candidates', []))}, "
            f"live-only CPCV PBO={live_audit.get('cpcv', {}).get('pbo')}."
        )

    if results:
        lines += [
            "",
            "## Monte Carlo sensitivity (`new` policy, `$15` and `$30` deadbands)",
            "Stationary 6-day blocks; 5,000 resamples (or configured paths), 90-day horizon.",
            "",
            "| Pod | Deadband | Multiplier | Blended pass / static fail / daily fail | 75%-chop pass / static fail / daily fail | Median pass days (blended) | Avg sweeps (blended) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for r in results:
            for db in ("15", "30"):
                mc_map = r["oos"]["new"]["deadbands"][db]["mc"]
                for mult in ("1.0x", "1.5x", "2.0x"):
                    b = mc_map[f"{mult}/Blended"]
                    c = mc_map[f"{mult}/75%-Chop"]
                    lines.append(
                        f"| {r['index']} | ${db} | {mult} | "
                        f"{b['p_pass']:.2f}% / {b['p_fail_static']:.2f}% / {b['p_fail_daily']:.2f}% | "
                        f"{c['p_pass']:.2f}% / {c['p_fail_static']:.2f}% / {c['p_fail_daily']:.2f}% | "
                        f"{b['med_days_pass']} | {b['avg_sweeps']:.2f} |"
                    )

    lines += [
        "",
        "## Model, pipeline, and coverage",
        f"- {len(inventory['source']['files'])} complete daily Reservoir archives; `{inventory['source']['archive_sha256']}` content digest.",
        (
            f"- UTC Polars lazy scan + CSE bounds: `min_ts_ms={inventory['utc_bounds']['min_ts_ms']}`, "
            f"`max_ts_ms={inventory['utc_bounds']['max_ts_ms']}` (strict UTC, zero DuckDB)."
        ),
        (
            f"- Screening separation: {inventory['roster_count']} roster wallets -> "
            f"{inventory['historical_eligible_count']} `scope=historical` eligible vs "
            f"{inventory['live_deployable_count']} `scope=live` deployable; "
            f"{output['shortlist_count']} positive Train survivors shortlisted "
            f"({output.get('live_shortlist_count', 0)} live-deployable)."
        ),
        f"- Leak-free CPCV PBO across frozen candidates: **{pbo_val}** ({cpcv_info.get('n_splits', 0)} purged/embargoed splits).",
        "- Single-pass Train replay reused directly into `build_tensor_from_unit_replays` and persisted to gitignored `data/v4_cache/*.parquet`.",
        "",
        "## Guardian, PM03, and Triad A conflicts",
    ]
    for r in results:
        oos_new_15 = r["oos"]["new"]["deadbands"]["15"]["cost_multipliers"]["1x"]
        pm = r["oos"]["new"].get("pm03", {})
        conf = r.get("triada_conflicts", {})
        lines.append(
            f"- **Candidate {r['index']}**: verdict=`{oos_new_15['verdict']}`, peak-DD={oos_new_15['peak_to_trough_dd_usd']}, "
            f"worst daily={oos_new_15['worst_daily_usd']}, sweeps={oos_new_15['sweeps']}, peak lev={oos_new_15['full_window_peak_gross_leverage']:.2f}x; "
            f"PM03 7d drift={pm.get('drift_usd', 'UNPROVEN')}; Triad A reverse intervals={conf.get('events', 'UNPROVEN')} "
            f"({conf.get('coin_hours', 0.0)} coin-h)."
        )
    if ref_stats:
        rn15 = ref_stats["new"]["deadbands"]["15"]["cost_multipliers"]["1x"]["full_window_net_usd"]
        rf15 = ref_stats["full"]["deadbands"]["15"]["cost_multipliers"]["1x"]["full_window_net_usd"]
        lines.append(
            f"- **Triad A reference (`$15` deadband, `1x` cost)**: `new`={rn15:+.2f}, `full`={rf15:+.2f}; static `k` source unverified live."
        )

    lines += [
        "",
        "## Qualification and caveats",
        "- Qualification gate requires `PBO < 0.50`, positive OOS net under `$15`/`$30` deadbands and `2x` cost stress, `live_deployable=True`, and prospective CWSA paper validation in `copytrade-pm`.",
        "- Companion JSON and gitignored Parquet caches store full screening flags, CPCV logits, daily blocks, and equity curves.",
        "",
        "### Three-line caveat footer",
        "- CFT trading restrictions and account framing require independent compliance confirmation.",
        "- Live broker execution, funding, capacity and PIT source continuity are UNPROVEN.",
        "- No deployments, allocations, orders, service or registry writes occurred.",
        "",
    ]
    return "\n".join(lines)


def run_study(
    *,
    paths: StudyPaths = StudyPaths(),
    through: str = "all",
    fetch_live_av: bool = True,
    bybit_snapshot: tuple[set[str] | Mapping[str, tuple[float, float]], dict] | None = None,
    account_value_fetcher: Callable[
        [cr.WalletRecord], tuple[float | None, dict | None, str | None]
    ]
    | None = None,
    fetch_hyperx: bool = True,
    mc_n_paths: int = 5000,
) -> dict:
    """Execute the May-Nov 2026 CFT $10k v4 study pipeline through ``through`` stage."""
    t_start = time.monotonic()
    inv_path = paths.inventory_path
    freeze_path = paths.report_prefix.with_name(paths.report_prefix.name + "__freeze.json")
    results_path = paths.report_prefix.with_name(paths.report_prefix.name + "__results.json")
    md_path = paths.report_prefix.with_name(paths.report_prefix.name + ".md")

    if through == "evaluate":
        inventory = json.loads(inv_path.read_text())
        freeze = json.loads(freeze_path.read_text())
        records, roster_sha = cr.load_registry_universe(paths.registry, include_aliases=TRIAD)
        if roster_sha != inventory["roster_sha256"] or roster_sha != freeze["roster_sha256"]:
            raise RuntimeError("FROZEN_ROSTER_SHA_CHANGED")
        if freeze["cutoff_ms"] != SPLIT or freeze["archive_sha256"] != inventory["source"]["archive_sha256"]:
            raise RuntimeError("FROZEN_ARCHIVE_OR_TRAIN_SPLIT_CHANGED")
        archive_paths = [f["path"] for f in inventory["source"]["files"]]
        ingestion = sd.ingest_fills(records, archive_paths, start=START, end=END, split=SPLIT)
        grouped = ingestion.grouped
        prices = sd.LazyPrices(
            GRID, store_root=paths.store, vike_root=paths.vike, start=START, end=END, split_ms=SPLIT
        )
        shortlist = freeze["shortlist"]
        selections = freeze["frozen_portfolios"]
        rows_by_alias = {r["alias"]: r for r in inventory["screen"]}
        eligible = {
            a: {c: grouped[a][c] for c in rows_by_alias[a]["historical_coins"] if c in grouped.get(a, {})}
            for a in shortlist
        }
    else:
        inventory, grouped, records, prices, hist_eligible, _live_eligible = prepare_screen(
            paths,
            fetch_live_av=fetch_live_av,
            bybit_snapshot=bybit_snapshot,
            account_value_fetcher=account_value_fetcher,
        )
        if through == "screen":
            return inventory

        shortlist, train_stats, diags, tensor, train_pq_artifacts = run_single_pass_train_replay(
            hist_eligible,
            prices,
            inventory["screen"],
            cache_dir=paths.cache_dir,
            train_grid=TRAIN_GRID,
            start_ms=START,
        )
        if fetch_hyperx and shortlist:
            hx = fetch_hyperx_context(shortlist, records)
            for row in inventory["screen"]:
                if row["alias"] in hx:
                    row["hyperx"] = hx[row["alias"]]
        inventory["train_stats"] = train_stats
        inventory["shortlist"] = shortlist
        inventory["train_parquet_artifacts"] = train_pq_artifacts
        inventory["selection_uses_only_train_returns"] = True
        dump_json(inv_path, inventory)

        if tensor is None or len(shortlist) < 3:
            selections: list[dict] = []
            audit: dict = {
                "strict_min_notional_status": "INSUFFICIENT_TRAIN_POSITIVE_WALLETS",
                "cpcv": {"pbo": None, "n_splits": 0, "logits": []},
            }
            live_scoped_audit: dict = {}
        else:
            selections, audit = train_pods(tensor, screen_rows=inventory["screen"])
            row_by_alias = {r["alias"]: r for r in inventory["screen"]}
            live_shortlist = [a for a in shortlist if row_by_alias.get(a, {}).get("live_deployable")]
            live_scoped_audit = {"live_shortlist_count": len(live_shortlist), "candidates": [], "cpcv": {"pbo": None}}
            if len(live_shortlist) >= 3:
                live_tensor = _slice_tensor_for_aliases(tensor, live_shortlist)
                if live_tensor is not None:
                    live_cands, live_audit = train_pods(live_tensor, screen_rows=inventory["screen"])
                    live_scoped_audit = {
                        "live_shortlist_count": len(live_shortlist),
                        "candidates": live_cands,
                        "cpcv": live_audit.get("cpcv", {}),
                        "strict_min_notional_status": live_audit.get("strict_min_notional_status"),
                    }
            del tensor

        freeze = {
            "freeze_time_utc": datetime.now(UTC).isoformat(),
            "cutoff_ms": SPLIT,
            "roster_sha256": inventory["roster_sha256"],
            "archive_sha256": inventory["source"]["archive_sha256"],
            "bybit": inventory["bybit"],
            "shortlist": shortlist,
            "live_shortlist": [
                a for a in shortlist if next((r for r in inventory["screen"] if r["alias"] == a), {}).get("live_deployable")
            ],
            "train_audit": audit,
            "live_scoped_audit": live_scoped_audit,
            "train_diagnostics": diags,
            "frozen_portfolios": selections,
            "parquet_artifacts": train_pq_artifacts,
            "cost_bps": cr.DEFAULT_COST_BPS,
            "scientific_status": "RETROSPECTIVE_NO_PIT_ROSTER",
        }
        dump_json(freeze_path, freeze)
        if through == "train":
            return freeze
        eligible = hist_eligible

    reference = None
    if all(a in grouped for a in TRIAD):
        reference = candidate_targets(
            list(TRIAD), list(TRIAD_K), grouped, prices, OOS_GRID, "full", full_grid=GRID
        )

    lot_specs_raw = inventory.get("bybit_lot_specs", {})
    amount_step_by_coin = {c: float(v[0]) for c, v in lot_specs_raw.items()} or None
    min_amount_by_coin = {c: float(v[1]) for c, v in lot_specs_raw.items()} or None

    vol_cache: dict[tuple[str, int, int], np.ndarray] = {}
    fund_cache: dict[tuple[str, int, int], np.ndarray] = {}
    equity_records: list[pl.DataFrame] = []
    block_records: list[pl.DataFrame] = []
    results: list[dict] = []

    for idx, row in enumerate(selections, 1):
        print("OOS_CANDIDATE", idx, row["aliases"], flush=True)
        row_with_idx = {**row, "index": idx}
        train_result = evaluate_one(
            row_with_idx,
            eligible,
            prices,
            TRAIN_GRID,
            full_grid=GRID,
            store_root=paths.store,
            vol_cache=vol_cache,
            fund_cache=fund_cache,
            montecarlo=False,
            pm03=False,
            deadbands=(MIN_NOTIONAL_USD,),
            cost_multipliers=(1.0,),
            include_volume_participation=False,
            include_funding_stress=False,
            amount_step_by_coin=amount_step_by_coin,
            min_amount_by_coin=min_amount_by_coin,
            max_gross_leverage=MAX_GROSS_LEVERAGE,
        )
        oos_result = evaluate_one(
            row_with_idx,
            eligible,
            prices,
            OOS_GRID,
            full_grid=GRID,
            store_root=paths.store,
            vol_cache=vol_cache,
            fund_cache=fund_cache,
            montecarlo=True,
            pm03=(idx <= 3),
            equity_sink=equity_records,
            block_sink=block_records,
            mc_n_paths=mc_n_paths,
            amount_step_by_coin=amount_step_by_coin,
            min_amount_by_coin=min_amount_by_coin,
            max_gross_leverage=MAX_GROSS_LEVERAGE,
        )
        ctx = candidate_targets(
            row["aliases"], row["train_opt_k"], eligible, prices, OOS_GRID, "new", full_grid=GRID
        )
        entry = {
            "index": idx,
            "aliases": row["aliases"],
            "k": row["train_opt_k"],
            "baseline_k": row["train_half_risk_k"],
            "train": train_result,
            "oos": oos_result,
            "optimization": {
                k: v for k, v in row.items() if k not in ("train_opt_k", "train_half_risk_k", "indices")
            },
        }
        if reference is not None:
            entry["triada_conflicts"] = conflict(reference, ctx)
        results.append(entry)

    ref_stats: dict = {}
    if reference is not None:
        triad_row = {"index": 0, "aliases": list(TRIAD), "train_opt_k": list(TRIAD_K)}
        ref_stats = evaluate_one(
            triad_row,
            grouped,
            prices,
            OOS_GRID,
            full_grid=GRID,
            store_root=paths.store,
            vol_cache=vol_cache,
            fund_cache=fund_cache,
            montecarlo=True,
            pm03=False,
            mc_n_paths=mc_n_paths,
            amount_step_by_coin=amount_step_by_coin,
            min_amount_by_coin=min_amount_by_coin,
            max_gross_leverage=MAX_GROSS_LEVERAGE,
        )
        ref_stats["k_source"] = "prop-production/bybit_executor.py static defaults, live runtime unverified"

    oos_pq_artifacts = export_oos_parquet_artifacts(paths.cache_dir, equity_records, block_records)
    elapsed_s = round(time.monotonic() - t_start, 2)

    output = {
        "status": "RETROSPECTIVE_SCENARIO",
        "calibration_frozen_before_oos": True,
        "roster_sha256": inventory["roster_sha256"],
        "source_archive_sha256": inventory["source"]["archive_sha256"],
        "train_window": ["2026-05-01", "2026-09-01"],
        "oos_window": ["2026-09-01", "2026-11-01"],
        "screened_wallets": len(inventory["screen"]),
        "historical_eligible_count": inventory["historical_eligible_count"],
        "live_deployable_count": inventory["live_deployable_count"],
        "shortlist_count": len(shortlist),
        "live_shortlist_count": len(freeze.get("live_shortlist", [])),
        "frozen_candidates": len(selections),
        "cpcv": freeze.get("train_audit", {}).get("cpcv", {}),
        "parquet_artifacts": {**freeze.get("parquet_artifacts", {}), **oos_pq_artifacts},
        "results": results,
        "triada": ref_stats,
        "limitations": {
            "historic_known_at_roster": "UNPROVEN",
            "follower_bybit_fills": "UNPROVEN",
            "market_impact_funding": "UNPROVEN",
            "CWSA_historical_5m_snapshots": "UNPROVEN",
            "historical_bybit_symbol_eligibility": "UNPROVEN",
            "hyperx_discovery_only": True,
            "pm03_subset": "first 7 OOS days; $0 deadband vs PM03 Decimal parity",
        },
        "elapsed_s": elapsed_s,
    }
    dump_json(results_path, output)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown_report(inventory=inventory, freeze=freeze, output=output))
    print("SAVED_RESULT", str(results_path), "ELAPSED_S", elapsed_s, flush=True)
    return output


def main() -> None:
    """CLI entrypoint for reproducible May-Nov 2026 CFT v4 research orchestrator."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--through", choices=("screen", "train", "all", "evaluate"), default="all")
    args = parser.parse_args()
    run_study(through=args.through)


if __name__ == "__main__":
    main()
