"""Lazy Polars ingestion, UTC coverage validation, OHLCV adapter, and PIT/live screening."""

from __future__ import annotations

import glob
import hashlib
import itertools
import json
import os
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Callable, MutableMapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import canonical_replay as cr
import numpy as np
import polars as pl

DEFAULT_ARCHIVE_ROOT = Path.home() / ".local/share/copytrade/research"
DEFAULT_STORE_ROOT = Path("/home/quant/dev/quant/ml4t-data/.data/store")
DEFAULT_VIKE_ROOT = Path("/home/quant/dev/copytrade-pm/data/private/backtest_20261112/bars_vike")
DEFAULT_MIN_TRAIN_FILLS = 15
DEFAULT_MAX_TRAIN_INACTIVITY_DAYS = 45
DEFAULT_MIN_ACCOUNT_VALUE_USD = 1000.0
DEFAULT_STALE_MARK_MINUTES = 180
DEFAULT_UTC_BOUNDARY_TOLERANCE_MS = 3_600_000


def sha256_file(path: Path | str) -> str:
    """Compute SHA-256 hex digest of a file using fixed-size chunked reads."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_fill_archive(
    start: int | datetime,
    end: int | datetime,
    *,
    archive_root: Path | str = DEFAULT_ARCHIVE_ROOT,
) -> tuple[list[str], dict]:
    """Discover deduplicated daily Reservoir Parquet files covering UTC ``[start, end)``."""
    start_dt = to_utc_datetime(start)
    end_dt = to_utc_datetime(end)
    if end_dt <= start_dt:
        raise ValueError("INVALID_UTC_WINDOW")

    root = Path(archive_root).expanduser()
    start_day = start_dt.date().isoformat()
    end_day = (end_dt - timedelta(milliseconds=1)).date().isoformat()

    full_paths = sorted(glob.glob(str(root / "reservoir-full-history-*/hyperliquid/fills/day=*/data.parquet")))
    mon_paths = sorted(
        glob.glob(str(root / "reservoir-monitored-plus-six-hyperliquid-*/fills/day=*/data.parquet"))
    )

    by_day: dict[str, str] = {}
    duplicates: list[str] = []
    for path in full_paths + mon_paths:
        day = path.split("day=")[-1][:10]
        if not (start_day <= day <= end_day):
            continue
        if day in by_day:
            duplicates.append(day)
            continue
        by_day[day] = path

    paths = [by_day[day] for day in sorted(by_day)]
    required_days: list[str] = []
    cur = start_dt.date()
    final_date = (end_dt - timedelta(milliseconds=1)).date()
    while cur <= final_date:
        required_days.append(cur.isoformat())
        cur += timedelta(days=1)
    missing = sorted(set(required_days) - set(by_day))

    files_meta = [
        {
            "day": p.split("day=")[-1][:10],
            "path": p,
            "bytes": os.stat(p).st_size,
            "sha256": sha256_file(p),
        }
        for p in paths
    ]
    archive_sha256 = hashlib.sha256("".join(x["sha256"] for x in files_meta).encode()).hexdigest()
    manifest = {
        "count_daily_files": len(paths),
        "duplicate_date_sources": sorted(duplicates),
        "missing_dates": missing,
        "files": files_meta,
        "archive_sha256": archive_sha256,
    }
    return paths, manifest


def to_utc_datetime(ts: int | datetime) -> datetime:
    """Normalize an epoch-millisecond integer or tz-aware datetime to UTC."""
    if isinstance(ts, datetime):
        if ts.tzinfo is None or ts.utcoffset() is None:
            raise ValueError("NAIVE_DATETIME_FORBIDDEN")
        return ts.astimezone(UTC)
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        return datetime.fromtimestamp(float(ts) / 1000.0, tz=UTC)
    raise TypeError(f"Unsupported timestamp type: {type(ts).__name__}")


def to_utc_ms(ts: int | datetime) -> int:
    """Convert an epoch-millisecond integer or tz-aware datetime to UTC epoch ms."""
    if isinstance(ts, datetime):
        dt = to_utc_datetime(ts)
        return round(dt.timestamp() * 1000.0)
    if isinstance(ts, int) and not isinstance(ts, bool):
        return ts
    raise TypeError(f"Unsupported timestamp type: {type(ts).__name__}")


@dataclass(frozen=True)
class FillLazyPipeline:
    """Shared Polars lazy execution graph over fill Parquet archives (CSE)."""

    base_lf: pl.LazyFrame
    stats_lf: pl.LazyFrame
    fills_lf: pl.LazyFrame
    start_ms: int
    split_ms: int
    end_ms: int

    def explain_combined(self) -> str:
        """Return the combined CSE query plan across stats and grouped fill extraction."""
        return pl.explain_all([self.stats_lf, self.fills_lf])

    def explain_fills(self) -> str:
        """Return the optimized query plan for ordered fill extraction."""
        return self.fills_lf.explain()

    def explain_stats(self) -> str:
        """Return the optimized query plan for per-(wallet, coin) screening stats."""
        return self.stats_lf.explain()


@dataclass(frozen=True)
class FillIngestionResult:
    """Materialized fill tables, screening stats, and UTC window bounds."""

    grouped: dict[str, dict[str, pl.DataFrame]]
    stats_df: pl.DataFrame
    min_ts_ms: int
    max_ts_ms: int
    explain_plan: str


def build_fill_lazy_pipeline(
    records: Sequence[cr.WalletRecord],
    paths: Sequence[Path | str],
    *,
    start: int | datetime,
    end: int | datetime,
    split: int | datetime,
) -> FillLazyPipeline:
    """Construct a single Polars lazy scan serving both stats and ordered fills via CSE."""
    path_strs = [str(p) for p in paths]
    if not path_strs:
        raise ValueError("EMPTY_FILL_ARCHIVE_PATHS")
    addresses = sorted({r.address.lower() for r in records if r.address})
    if not addresses:
        raise ValueError("EMPTY_WALLET_UNIVERSE")

    start_dt = to_utc_datetime(start)
    end_dt = to_utc_datetime(end)
    split_dt = to_utc_datetime(split)
    start_ms = to_utc_ms(start_dt)
    end_ms = to_utc_ms(end_dt)
    split_ms = to_utc_ms(split_dt)
    if end_ms <= start_ms:
        raise ValueError("INVALID_UTC_WINDOW")

    perp_expr = (
        pl.col("coin").is_not_null()
        & (pl.col("coin").str.len_bytes() > 0)
        & (~pl.col("coin").str.starts_with("@"))
        & (~pl.col("coin").str.starts_with("#"))
        & (~pl.col("coin").str.contains("/", literal=True))
        & (~pl.col("coin").str.contains(":", literal=True))
    )

    raw_lf = pl.scan_parquet(
        path_strs,
        hive_partitioning=False,
        extra_columns="ignore",
        row_index_name="seq",
        cast_options=pl.ScanCastOptions(datetime_cast="millisecond-upcast"),
    )

    base_lf = raw_lf.filter(
        (pl.col("dex") == "hyperliquid")
        & (pl.col("exchange_ts") >= pl.lit(start_dt, dtype=pl.Datetime("us", "UTC")))
        & (pl.col("exchange_ts") < pl.lit(end_dt, dtype=pl.Datetime("us", "UTC")))
        & pl.col("wallet").str.to_lowercase().is_in(addresses)
        & perp_expr
    ).select(
        [
            pl.col("wallet").str.to_lowercase().alias("wallet"),
            pl.col("coin"),
            pl.col("exchange_ts").dt.epoch("ms").alias("ts"),
            pl.when(pl.col("side") == "buy")
            .then(1)
            .otherwise(-1)
            .cast(pl.Int8)
            .alias("sign"),
            pl.col("size").cast(pl.Float64).alias("sz"),
            pl.col("start_position").cast(pl.Float64).alias("pre"),
            pl.col("trade_id").cast(pl.String).alias("tid"),
            pl.col("seq"),
        ]
    )

    stats_lf = base_lf.group_by(["wallet", "coin"]).agg(
        [
            pl.len().cast(pl.Int64).alias("total_fills"),
            (pl.col("ts") < split_ms).sum().cast(pl.Int64).alias("train_fills"),
            (pl.col("ts") >= split_ms).sum().cast(pl.Int64).alias("oos_fills"),
            pl.col("ts").min().alias("first_fill_ms"),
            pl.col("ts").filter(pl.col("ts") < split_ms).max().alias("last_train_fill_ms"),
            pl.col("ts").max().alias("last_fill_ms"),
        ]
    )

    fills_lf = base_lf.sort(["wallet", "coin", "ts", "seq"])

    return FillLazyPipeline(
        base_lf=base_lf,
        stats_lf=stats_lf,
        fills_lf=fills_lf,
        start_ms=start_ms,
        split_ms=split_ms,
        end_ms=end_ms,
    )


def assert_utc_coverage(
    min_ts_ms: int | None,
    max_ts_ms: int | None,
    start: int | datetime,
    end: int | datetime,
    *,
    max_start_lag_ms: int | None = DEFAULT_UTC_BOUNDARY_TOLERANCE_MS,
    max_end_lag_ms: int | None = DEFAULT_UTC_BOUNDARY_TOLERANCE_MS,
) -> None:
    """Fail loudly if loaded fill timestamps fall outside or short of the requested UTC window."""
    if min_ts_ms is None or max_ts_ms is None:
        raise ValueError("UTC_COVERAGE_EMPTY: no fills matched the requested window")
    start_ms = to_utc_ms(start)
    end_ms = to_utc_ms(end)
    if min_ts_ms < start_ms or max_ts_ms >= end_ms:
        raise ValueError(
            f"UTC_COVERAGE_OUT_OF_WINDOW: [{min_ts_ms}, {max_ts_ms}] outside [{start_ms}, {end_ms})"
        )
    if max_start_lag_ms is not None and (min_ts_ms - start_ms) > max_start_lag_ms:
        raise ValueError(
            f"UTC_COVERAGE_START_SHORTFALL: first fill {min_ts_ms} lags start {start_ms} "
            f"by {min_ts_ms - start_ms}ms (> {max_start_lag_ms}ms)"
        )
    if max_end_lag_ms is not None and (end_ms - max_ts_ms) > max_end_lag_ms:
        raise ValueError(
            f"UTC_COVERAGE_END_SHORTFALL: last fill {max_ts_ms} falls short of end {end_ms} "
            f"by {end_ms - max_ts_ms}ms (> {max_end_lag_ms}ms)"
        )


def _count_burst_breaks(
    seq_indices: list[int],
    pre: np.ndarray,
    post: np.ndarray,
    prev_post: float | None,
    next_pre: float | None,
    eps: float,
) -> int:
    breaks = 0
    if prev_post is not None and abs(float(pre[seq_indices[0]]) - prev_post) > eps:
        breaks += 1
    for a, b in itertools.pairwise(seq_indices):
        if abs(float(pre[b]) - float(post[a])) > eps:
            breaks += 1
    if next_pre is not None and abs(next_pre - float(post[seq_indices[-1]])) > eps:
        breaks += 1
    return breaks


def _greedy_same_ms_order(
    orig: list[int],
    start: int,
    pre: np.ndarray,
    post: np.ndarray,
    eps: float,
) -> list[int]:
    used = {start}
    path = [start]
    while len(path) < len(orig):
        cur_post = float(post[path[-1]])
        nxt = next(
            (idx for idx in orig if idx not in used and abs(float(pre[idx]) - cur_post) <= eps),
            None,
        )
        if nxt is None:
            nxt = next(idx for idx in orig if idx not in used)
        used.add(nxt)
        path.append(nxt)
    return path


def _solve_same_ms_burst(
    lo: int,
    hi: int,
    pre: np.ndarray,
    post: np.ndarray,
    prev_post: float | None,
    next_pre: float | None,
    eps: float,
) -> list[int]:
    orig = list(range(lo, hi))
    orig_breaks = _count_burst_breaks(orig, pre, post, prev_post, next_pre, eps)
    if orig_breaks == 0:
        return orig

    in_deg = [
        sum(1 for a in orig if a != b and abs(float(post[a]) - float(pre[b])) <= eps)
        for b in orig
    ]
    starts = sorted(
        orig,
        key=lambda idx: (
            0 if (prev_post is not None and abs(float(pre[idx]) - prev_post) <= eps) else 1,
            0 if in_deg[idx - lo] == 0 else 1,
            idx,
        ),
    )

    best_order = orig
    best_breaks = orig_breaks

    if (hi - lo) > 6:
        for s in starts[:12]:
            candidate = _greedy_same_ms_order(orig, s, pre, post, eps)
            brk = _count_burst_breaks(candidate, pre, post, prev_post, next_pre, eps)
            if brk < best_breaks:
                best_breaks = brk
                best_order = candidate
                if best_breaks == 0:
                    break
        return best_order

    used = [False] * (hi - lo)
    path: list[int] = []

    def dfs() -> None:
        nonlocal best_order, best_breaks
        if len(path) == hi - lo:
            brk = _count_burst_breaks(path, pre, post, prev_post, next_pre, eps)
            if brk < best_breaks:
                best_breaks = brk
                best_order = path.copy()
            return
        cur_post = float(post[path[-1]])
        matches = [
            idx for idx in orig if not used[idx - lo] and abs(float(pre[idx]) - cur_post) <= eps
        ]
        candidates = matches if matches else [idx for idx in orig if not used[idx - lo]]
        for nxt in candidates:
            used[nxt - lo] = True
            path.append(nxt)
            dfs()
            path.pop()
            used[nxt - lo] = False
            if best_breaks == 0:
                return

    for s in starts:
        used[s - lo] = True
        path.append(s)
        dfs()
        path.pop()
        used[s - lo] = False
        if best_breaks == 0:
            break

    return best_order


def repair_same_ms_fill_order(part: pl.DataFrame, *, eps: float = 1e-6) -> pl.DataFrame:
    """Repair rare raw-sequence inversions inside same-ms bursts using start_position continuity.

    Valid same-ms chains already matching raw Parquet arrival order are untouched.
    """
    n = part.height
    if n < 2:
        return part
    ts = part["ts"].to_numpy()
    pre = part["pre"].to_numpy().astype(np.float64)
    sign = part["sign"].to_numpy().astype(np.float64)
    sz = part["sz"].to_numpy().astype(np.float64)
    post = pre + sign * sz

    same_ts = ts[1:] == ts[:-1]
    bad_link = np.abs(pre[1:] - post[:-1]) > eps
    has_same_ms_break = bool(np.any(same_ts & bad_link))
    has_entry_break = bool(np.any(same_ts[1:] & bad_link[:-1])) if n > 2 else False
    if not (has_same_ms_break or has_entry_break):
        return part

    order = list(range(n))
    i = 0
    while i < n:
        j = i + 1
        while j < n and ts[j] == ts[i]:
            j += 1
        if j - i > 1 and (
            bool(np.any(bad_link[i : j - 1])) or (i > 0 and bool(bad_link[i - 1]))
        ):
            prev_post = float(post[i - 1]) if i > 0 else None
            next_pre = float(pre[j]) if j < n else None
            order[i:j] = _solve_same_ms_burst(i, j, pre, post, prev_post, next_pre, eps)
        i = j

    if order == list(range(n)):
        return part
    return part[order]


def ingest_fills(
    records: Sequence[cr.WalletRecord],
    paths: Sequence[Path | str],
    *,
    start: int | datetime,
    end: int | datetime,
    split: int | datetime,
    streaming: bool = True,
    max_start_lag_ms: int | None = DEFAULT_UTC_BOUNDARY_TOLERANCE_MS,
    max_end_lag_ms: int | None = DEFAULT_UTC_BOUNDARY_TOLERANCE_MS,
) -> FillIngestionResult:
    """Execute the shared CSE fill pipeline and return ordered per-wallet/coin DataFrames."""
    pipeline = build_fill_lazy_pipeline(
        records,
        paths,
        start=start,
        end=end,
        split=split,
    )
    explain_plan = pipeline.explain_combined()
    engine = "streaming" if streaming else "auto"
    stats_df, fills_df = pl.collect_all([pipeline.stats_lf, pipeline.fills_lf], engine=engine)

    if fills_df.is_empty():
        raise ValueError("UTC_COVERAGE_EMPTY: no fills loaded from archive")

    min_ts_ms = int(stats_df["first_fill_ms"].min())  # type: ignore[arg-type]
    max_ts_ms = int(stats_df["last_fill_ms"].max())  # type: ignore[arg-type]
    assert_utc_coverage(
        min_ts_ms,
        max_ts_ms,
        pipeline.start_ms,
        pipeline.end_ms,
        max_start_lag_ms=max_start_lag_ms,
        max_end_lag_ms=max_end_lag_ms,
    )

    label_by_addr = {r.address.lower(): r.label for r in records}
    grouped: dict[str, dict[str, pl.DataFrame]] = defaultdict(dict)
    partitions = fills_df.partition_by(["wallet", "coin"], as_dict=True, maintain_order=True)
    for (address, coin), part in partitions.items():
        label = label_by_addr.get(str(address))
        if label is None:
            continue
        grouped[label][str(coin)] = repair_same_ms_fill_order(
            part.select("ts", "sign", "sz", "pre", "tid")
        )

    return FillIngestionResult(
        grouped=dict(grouped),
        stats_df=stats_df,
        min_ts_ms=min_ts_ms,
        max_ts_ms=max_ts_ms,
        explain_plan=explain_plan,
    )


OHLCV_SCAN_SCHEMA = pl.Schema(
    {
        "timestamp": pl.Datetime("us", "UTC"),
        "open": pl.Float64,
        "close": pl.Float64,
    }
)
OHLCV_CAST_OPTIONS = pl.ScanCastOptions(
    datetime_cast="millisecond-upcast",
    float_cast="upcast",
)


def _extract_partition_day(path: Path) -> str | None:
    parts = path.parts
    year = month = day = None
    for part in parts:
        if part.startswith("year="):
            year = part.removeprefix("year=")
        elif part.startswith("month="):
            month = part.removeprefix("month=")
        elif part.startswith("day="):
            day = part.removeprefix("day=")
    if year and month and day:
        return f"{year}-{month}-{day}"
    return None


_UNSET = object()


class LazyPrices:
    """Lazy Polars 1m OHLCV adapter with schema-drift tolerance and Vike fallback."""

    def __init__(
        self,
        grid_ms: np.ndarray,
        *,
        store_root: Path | str = DEFAULT_STORE_ROOT,
        vike_root: Path | str | None = DEFAULT_VIKE_ROOT,
        start: int | datetime | None = None,
        end: int | datetime | None = None,
        step_ms: int = 300_000,
        stale_limit_minutes: int = DEFAULT_STALE_MARK_MINUTES,
        split_ms: int | None = None,
        streaming: bool = True,
    ) -> None:
        if grid_ms.size == 0:
            raise ValueError("EMPTY_PRICE_GRID")
        self.grid = np.asarray(grid_ms, dtype=np.int64)
        self.store_root = Path(store_root).expanduser()
        self.vike_root = Path(vike_root).expanduser() if vike_root is not None else None
        start_ms = to_utc_ms(start) if start is not None else int(self.grid[0])
        end_ms = to_utc_ms(end) if end is not None else int(self.grid[-1]) + step_ms
        if end_ms <= start_ms:
            raise ValueError("INVALID_UTC_WINDOW")
        self.start_dt = to_utc_datetime(start_ms)
        self.end_dt = to_utc_datetime(end_ms)
        self.start_day = self.start_dt.date().isoformat()
        self.end_day = (self.end_dt - timedelta(milliseconds=1)).date().isoformat()
        self.stale_limit_minutes = int(stale_limit_minutes)
        self.cut = int(np.searchsorted(self.grid, split_ms)) if split_ms is not None else None
        self.engine: str = "streaming" if streaming else "auto"

        self._trade_grid_df = pl.DataFrame(
            {
                "timestamp": pl.from_epoch(
                    pl.Series("timestamp", self.grid + 60_000, dtype=pl.Int64), time_unit="ms"
                )
                .dt.replace_time_zone("UTC")
                .cast(pl.Datetime("us", "UTC"))
            }
        )
        self._mark_grid_df = pl.DataFrame(
            {
                "timestamp": pl.from_epoch(
                    pl.Series("timestamp", self.grid + 240_000, dtype=pl.Int64), time_unit="ms"
                )
                .dt.replace_time_zone("UTC")
                .cast(pl.Datetime("us", "UTC"))
            }
        )

        self.cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.stats: dict[str, dict] = {}

    def coin_paths(
        self,
        coin: str,
        *,
        start_day: str | None = None,
        end_day: str | None = None,
    ) -> list[str]:
        """Return sorted daily OHLCV Parquet paths within the configured UTC date range."""
        coin_dir = self.store_root / f"crypto_futures_ohlcv_1m_HYPERLIQUID_{coin}_PERP"
        if not coin_dir.is_dir():
            return []
        eff_start = self.start_day if start_day is None else start_day
        eff_end = self.end_day if end_day is None else end_day
        candidates = sorted(coin_dir.glob("year=*/month=*/day=*/data.parquet"))
        selected: list[str] = []
        for path in candidates:
            day = _extract_partition_day(path)
            if day is None or (eff_start <= day <= eff_end):
                selected.append(str(path))
        return selected

    def _vike_path(self, coin: str) -> Path | None:
        if self.vike_root is None:
            return None
        path = self.vike_root / f"{coin}.parquet"
        return path if path.is_file() else None

    def build_scan_plan(
        self,
        coin: str,
        *,
        paths: list[str] | None = None,
        vike_path: Path | None | object = _UNSET,
    ) -> pl.LazyFrame | None:
        """Build the lazy Parquet scan graph with UTC predicate and projection pushdown."""
        resolved_paths = self.coin_paths(coin) if paths is None else paths
        resolved_vike: Path | None = (
            self._vike_path(coin) if vike_path is _UNSET else vike_path  # type: ignore[assignment]
        )
        if not resolved_paths and resolved_vike is None:
            return None

        start_lit = pl.lit(self.start_dt, dtype=pl.Datetime("us", "UTC"))
        end_lit = pl.lit(self.end_dt, dtype=pl.Datetime("us", "UTC"))

        parts: list[pl.LazyFrame] = []
        if resolved_paths:
            store_lf = (
                pl.scan_parquet(
                    resolved_paths,
                    schema=OHLCV_SCAN_SCHEMA,
                    hive_partitioning=False,
                    extra_columns="ignore",
                    cast_options=OHLCV_CAST_OPTIONS,
                )
                .filter((pl.col("timestamp") >= start_lit) & (pl.col("timestamp") < end_lit))
                .select(["timestamp", "open", "close"])
            )
            parts.append(store_lf)

        if resolved_vike is not None:
            vike_lf = (
                pl.scan_parquet(str(resolved_vike), hive_partitioning=False)
                .select(
                    [
                        pl.from_epoch("ts_ms", time_unit="ms")
                        .dt.replace_time_zone("UTC")
                        .cast(pl.Datetime("us", "UTC"))
                        .alias("timestamp"),
                        pl.col("open").cast(pl.Float64),
                        pl.col("close").cast(pl.Float64),
                    ]
                )
                .filter((pl.col("timestamp") >= start_lit) & (pl.col("timestamp") < end_lit))
            )
            parts.append(vike_lf)

        if len(parts) == 1:
            return parts[0].unique(subset="timestamp", keep="first").sort("timestamp")

        tagged = [
            parts[0].with_columns(pl.lit(0, dtype=pl.UInt8).alias("_src")),
            parts[1].with_columns(pl.lit(1, dtype=pl.UInt8).alias("_src")),
        ]
        return (
            pl.concat(tagged)
            .sort(["timestamp", "_src"])
            .unique(subset="timestamp", keep="first", maintain_order=True)
            .drop("_src")
        )

    def explain(self, coin: str) -> str:
        """Return the optimized Polars query plan for a coin's 1m OHLCV scan."""
        plan = self.build_scan_plan(coin)
        if plan is None:
            return "EMPTY_PRICE_PLAN"
        return plan.explain()

    def get(self, coin: str) -> tuple[np.ndarray, np.ndarray]:
        """Return causal ``(trade_open, mark_close)`` NumPy arrays aligned to ``self.grid``."""
        if coin in self.cache:
            return self.cache[coin]

        paths = self.coin_paths(coin)
        vike_path = self._vike_path(coin)
        if not paths and vike_path is None:
            empty = np.full(self.grid.size, np.nan, dtype=np.float64)
            self.cache[coin] = (empty, empty.copy())
            self.stats[coin] = {
                "status": "NO_LOCAL_OR_VIKE_PRICE",
                "mark_valid_frac": 0.0,
                "trade_valid_frac": 0.0,
                "store_days": 0,
            }
            return self.cache[coin]

        plan = self.build_scan_plan(coin, paths=paths, vike_path=vike_path)
        assert plan is not None
        bars = plan.collect(engine=self.engine)
        trade = (
            self._trade_grid_df.join(bars.select("timestamp", "open"), on="timestamp", how="left")[
                "open"
            ]
            .fill_null(float("nan"))
            .to_numpy()
            .astype(np.float64, copy=True)
        )
        mark = (
            self._mark_grid_df.join_asof(
                bars.select("timestamp", "close"),
                on="timestamp",
                strategy="backward",
                tolerance=timedelta(minutes=self.stale_limit_minutes),
            )["close"]
            .fill_null(float("nan"))
            .to_numpy()
            .astype(np.float64, copy=True)
        )

        self.cache[coin] = (trade, mark)
        if paths and vike_path is not None:
            status = "STORE_PLUS_VIKE"
        elif paths:
            status = "STORE_ONLY"
        else:
            status = "VIKE_ONLY"

        stat = {
            "status": status,
            "mark_valid_frac": round(float(np.mean(np.isfinite(mark))), 5),
            "trade_valid_frac": round(float(np.mean(np.isfinite(trade))), 5),
            "store_days": len(paths),
        }
        if self.cut is not None:
            stat["train_mark_valid_frac"] = round(float(np.mean(np.isfinite(mark[: self.cut]))), 5)
            stat["oos_mark_valid_frac"] = round(float(np.mean(np.isfinite(mark[self.cut :]))), 5)
        self.stats[coin] = stat
        return trade, mark


OHLCV_VOL_SCAN_SCHEMA = pl.Schema(
    {
        "timestamp": pl.Datetime("us", "UTC"),
        "open": pl.Float64,
        "volume": pl.Float64,
        "quote_volume": pl.Float64,
    }
)


def _scan_vike_volume_lf(
    vike_path: Path,
    start_lit: pl.Expr,
    end_lit: pl.Expr,
) -> pl.LazyFrame | None:
    raw_lf = pl.scan_parquet(str(vike_path), hive_partitioning=False)
    schema = raw_lf.collect_schema()
    cols = set(schema.names())
    vol_exprs: list[pl.Expr] = []
    if "quote_volume" in cols:
        vol_exprs.append(pl.col("quote_volume").cast(pl.Float64).fill_nan(None))
    if "volume" in cols and "open" in cols:
        vol_exprs.append(
            (pl.col("volume").cast(pl.Float64) * pl.col("open").cast(pl.Float64)).fill_nan(None)
        )
    if not vol_exprs:
        return None
    if "ts_ms" in cols:
        ts_expr = (
            pl.from_epoch("ts_ms", time_unit="ms")
            .dt.replace_time_zone("UTC")
            .cast(pl.Datetime("us", "UTC"))
            .alias("timestamp")
        )
    elif "timestamp" in cols:
        ts_dtype = schema["timestamp"]
        base_ts = pl.col("timestamp")
        if isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is None:
            base_ts = base_ts.dt.replace_time_zone("UTC")
        ts_expr = base_ts.cast(pl.Datetime("us", "UTC")).alias("timestamp")
    else:
        return None
    vol_expr = pl.coalesce(*vol_exprs) if len(vol_exprs) > 1 else vol_exprs[0]
    return (
        raw_lf.select([ts_expr, vol_expr.alias("vol_usd")])
        .filter(
            (pl.col("timestamp") >= start_lit)
            & (pl.col("timestamp") < end_lit)
            & pl.col("vol_usd").is_not_null()
            & pl.col("vol_usd").is_not_nan()
        )
    )


def load_1m_volumes_on_grid(
    coins: Sequence[str],
    grid_ms: np.ndarray,
    prices: LazyPrices,
    vol_cache: MutableMapping[tuple[str, int, int, int], np.ndarray] | None = None,
) -> np.ndarray:
    """Load 1m USD dollar volumes aligned to execution bars ``grid_ms + 60_000``."""
    grid_arr = np.asarray(grid_ms, dtype=np.int64)
    if not coins:
        return np.zeros((0, int(grid_arr.size)), dtype=np.float64)
    if grid_arr.size == 0:
        return np.zeros((len(coins), 0), dtype=np.float64)

    cache: MutableMapping[tuple[str, int, int, int], np.ndarray] = (
        vol_cache if vol_cache is not None else {}
    )
    g_key = (int(grid_arr[0]), int(grid_arr[-1]), int(grid_arr.size))
    missing = [c for c in coins if (c, *g_key) not in cache]
    if missing:
        trade_grid_df = pl.DataFrame(
            {
                "timestamp": pl.from_epoch(
                    pl.Series("timestamp", grid_arr + 60_000, dtype=pl.Int64),
                    time_unit="ms",
                )
                .dt.replace_time_zone("UTC")
                .cast(pl.Datetime("us", "UTC"))
            }
        )
        start_dt = min(prices.start_dt, to_utc_datetime(int(grid_arr[0])))
        end_dt = max(prices.end_dt, to_utc_datetime(int(grid_arr[-1]) + 300_000))
        exp_start_day = start_dt.date().isoformat()
        exp_end_day = (end_dt - timedelta(milliseconds=1)).date().isoformat()
        start_lit = pl.lit(start_dt, dtype=pl.Datetime("us", "UTC"))
        end_lit = pl.lit(end_dt, dtype=pl.Datetime("us", "UTC"))

        for coin in missing:
            key = (coin, *g_key)
            paths = prices.coin_paths(coin, start_day=exp_start_day, end_day=exp_end_day)
            vike_path = prices._vike_path(coin)
            parts: list[pl.LazyFrame] = []
            if paths:
                store_lf = (
                    pl.scan_parquet(
                        paths,
                        schema=OHLCV_VOL_SCAN_SCHEMA,
                        hive_partitioning=False,
                        missing_columns="insert",
                        extra_columns="ignore",
                        cast_options=OHLCV_CAST_OPTIONS,
                    )
                    .filter((pl.col("timestamp") >= start_lit) & (pl.col("timestamp") < end_lit))
                    .with_columns(
                        pl.coalesce(
                            pl.col("quote_volume").fill_nan(None),
                            (pl.col("volume") * pl.col("open")).fill_nan(None),
                        ).alias("vol_usd")
                    )
                    .select(["timestamp", "vol_usd"])
                    .filter(pl.col("vol_usd").is_not_null() & pl.col("vol_usd").is_not_nan())
                    .with_columns(pl.lit(0, dtype=pl.UInt8).alias("_src"))
                )
                parts.append(store_lf)
            if vike_path is not None:
                vike_lf = _scan_vike_volume_lf(vike_path, start_lit, end_lit)
                if vike_lf is not None:
                    parts.append(vike_lf.with_columns(pl.lit(1, dtype=pl.UInt8).alias("_src")))
            if not parts:
                cache[key] = np.full(grid_arr.size, np.nan, dtype=np.float64)
                continue
            combined = (
                parts[0].drop("_src").unique(subset="timestamp", keep="first")
                if len(parts) == 1
                else (
                    pl.concat(parts)
                    .sort(["timestamp", "_src"])
                    .unique(subset="timestamp", keep="first", maintain_order=True)
                    .drop("_src")
                )
            )
            bars = combined.collect(engine=prices.engine)
            joined = (
                trade_grid_df.join(bars, on="timestamp", how="left")["vol_usd"]
                .fill_null(float("nan"))
                .to_numpy()
                .astype(np.float64, copy=True)
            )
            cache[key] = joined

    return np.vstack([cache[(c, *g_key)] for c in coins])


USER_AGENT = "trading-ops-research/1.0"


def fetch_bybit_usdt_perp_specs(*, timeout: int = 18) -> tuple[dict[str, tuple[float, float]], dict]:
    """Fetch current Bybit linear USDT perpetual ``{baseCoin: (qtyStep, minOrderQty)}`` specs."""
    specs: dict[str, tuple[float, float]] = {}
    cursor = ""
    pages = 0
    for _ in range(12):
        uri = (
            "https://api.bybit.com/v5/market/instruments-info"
            "?category=linear&status=Trading&limit=1000"
        )
        if cursor:
            uri += "&cursor=" + urllib.parse.quote(cursor)
        req = urllib.request.Request(uri, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            obj = json.load(resp)
        if int(obj.get("retCode", -1)) != 0:
            raise RuntimeError("BYBIT_INSTRUMENTS_ERROR")
        data = obj["result"]
        pages += 1
        for row in data["list"]:
            if row.get("quoteCoin") == "USDT" and row.get("contractType") == "LinearPerpetual":
                lot_filter = row.get("lotSizeFilter") or {}
                qty_step = float(lot_filter.get("qtyStep", 0.0) or 0.0)
                min_qty = float(lot_filter.get("minOrderQty", 0.0) or 0.0)
                specs[row["baseCoin"]] = (qty_step, min_qty)
        cursor = data.get("nextPageCursor") or ""
        if not cursor:
            break
    if not specs or cursor:
        raise RuntimeError("INCOMPLETE_BYBIT_PAGINATION")
    return specs, {
        "pages": pages,
        "base_coins": len(specs),
        "lookup_utc": datetime.now(UTC).isoformat(),
    }


def fetch_bybit_usdt_perps(*, timeout: int = 18) -> tuple[set[str], dict]:
    """Fetch current Bybit linear USDT perpetual base-coin listing snapshot."""
    specs, meta = fetch_bybit_usdt_perp_specs(timeout=timeout)
    return set(specs), meta


def fetch_hyperliquid_account_value(
    rec: cr.WalletRecord, *, timeout: int = 8
) -> tuple[float | None, dict | None, str | None]:
    """Fetch present-day Hyperliquid clearinghouse account value for operational checks."""
    try:
        req = urllib.request.Request(
            "https://api.hyperliquid.xyz/info",
            data=json.dumps({"type": "clearinghouseState", "user": rec.address}).encode(),
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            state = json.load(response)
        account_value = float(state["marginSummary"]["accountValue"])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return None, None, type(exc).__name__
    return account_value, None, None


def _stats_from_fills(fills_by_coin: dict[str, pl.DataFrame], split_ms: int) -> dict:
    all_t = (
        np.concatenate([x["ts"].to_numpy() for x in fills_by_coin.values()])
        if fills_by_coin
        else np.array([], dtype=np.int64)
    )
    train_mask = all_t < split_ms
    train_count = int(np.count_nonzero(train_mask))
    oos_count = int(all_t.size - train_count)
    return {
        "train_fills": train_count,
        "oos_fills": oos_count,
        "total_fills": int(all_t.size),
        "first_fill_ms": int(all_t.min()) if all_t.size else None,
        "last_train_fill_ms": int(all_t[train_mask].max()) if train_count else None,
        "last_fill_ms": int(all_t.max()) if all_t.size else None,
    }


def _withdrawn_tag(min_account_value_usd: float) -> str:
    return f"WITHDRAWN_LT_{int(min_account_value_usd)}"


def _stats_from_cse_frame(wallet_stats: pl.DataFrame) -> dict:
    if wallet_stats.is_empty():
        return {
            "train_fills": 0,
            "oos_fills": 0,
            "total_fills": 0,
            "first_fill_ms": None,
            "last_train_fill_ms": None,
            "last_fill_ms": None,
        }
    train_fills = int(wallet_stats["train_fills"].sum())
    oos_fills = int(wallet_stats["oos_fills"].sum())
    total_fills = int(wallet_stats["total_fills"].sum())
    first_raw = wallet_stats["first_fill_ms"].min()
    last_train_raw = wallet_stats["last_train_fill_ms"].max()
    last_raw = wallet_stats["last_fill_ms"].max()
    return {
        "train_fills": train_fills,
        "oos_fills": oos_fills,
        "total_fills": total_fills,
        "first_fill_ms": int(first_raw) if first_raw is not None else None,
        "last_train_fill_ms": int(last_train_raw) if last_train_raw is not None else None,
        "last_fill_ms": int(last_raw) if last_raw is not None else None,
    }


def summarize_wallet(
    rec: cr.WalletRecord,
    fills_by_coin: dict[str, pl.DataFrame],
    *,
    split: int | datetime,
    bybit_usdt_perps: set[str] | None = None,
    min_train_fills: int = DEFAULT_MIN_TRAIN_FILLS,
    max_train_inactivity_days: int = DEFAULT_MAX_TRAIN_INACTIVITY_DAYS,
    account_value_usd: float | None = None,
    account_value_status: str = "NOT_FETCHED",
    min_account_value_usd: float = DEFAULT_MIN_ACCOUNT_VALUE_USD,
    wallet_stats: pl.DataFrame | None = None,
) -> dict:
    """Summarize historical Train-only eligibility separately from live operational deployability."""
    split_ms = to_utc_ms(split)

    # Strictly pre-split Hyperliquid perp fills only; never inspect OOS-only symbols.
    train_coins = {
        coin: filtered
        for coin, df in fills_by_coin.items()
        if cr.is_hl_perp_coin(coin) and (filtered := df.filter(pl.col("ts") < split_ms)).height > 0
    }
    historical_coins = sorted(train_coins)

    counts = (
        _stats_from_cse_frame(wallet_stats)
        if wallet_stats is not None
        else _stats_from_fills(fills_by_coin, split_ms)
    )
    train_count = counts["train_fills"]
    last_train_ms = counts["last_train_fill_ms"]

    historical_reasons: list[str] = []
    if train_count == 0:
        historical_reasons.append("REJECT_NO_FILLS")
    if train_count < min_train_fills:
        historical_reasons.append(f"TRAIN_FILLS_LT_{min_train_fills}")
    if (
        last_train_ms is not None
        and last_train_ms < split_ms - max_train_inactivity_days * 86_400_000
    ):
        historical_reasons.append(f"TRAIN_INACTIVE_{max_train_inactivity_days}D")

    live_reasons: list[str] = []
    if bybit_usdt_perps is not None:
        screen = cr.assess_bybit_symbol_screen(rec.label, train_coins, bybit_usdt_perps)
        symbol_status = screen.status
        mapped_coins = list(screen.mapped_coins)
        unmapped_coins = list(screen.unmapped_coins)
        unmapped_fill_frac: float | None = round(screen.unmapped_fill_frac, 5)
        unmapped_notional_frac = screen.unmapped_notional_frac
        scale_alias_candidates = screen.scale_alias_candidates
        if screen.status in ("REJECT_UNMAPPED_MATERIAL", "REJECT_NO_FILLS"):
            live_reasons.append(screen.status)
    else:
        symbol_status = "NOT_CHECKED"
        mapped_coins = []
        unmapped_coins = []
        unmapped_fill_frac = None
        unmapped_notional_frac = None
        scale_alias_candidates = {}
        live_reasons.append("BYBIT_SNAPSHOT_NOT_CHECKED")

    if account_value_status == "NOT_FETCHED":
        live_reasons.append("ACCOUNT_VALUE_NOT_FETCHED")
    elif account_value_usd is None:
        live_reasons.append("ACCOUNT_VALUE_UNPROVEN")
    elif account_value_usd < min_account_value_usd:
        live_reasons.append(_withdrawn_tag(min_account_value_usd))

    return {
        "alias": rec.label,
        "configured": rec.configured,
        "excluded": rec.excluded,
        "train_fills": counts["train_fills"],
        "oos_fills": counts["oos_fills"],
        "total_fills": counts["total_fills"],
        "first_fill_ms": counts["first_fill_ms"],
        "last_train_fill_ms": counts["last_train_fill_ms"],
        "last_fill_ms": counts["last_fill_ms"],
        "historical_coins": historical_coins,
        "historical_reasons": historical_reasons,
        "historical_eligible": len(historical_reasons) == 0,
        "symbol_status": symbol_status,
        "mapped_coins": mapped_coins,
        "unmapped_coins": unmapped_coins,
        "unmapped_fill_frac": unmapped_fill_frac,
        "unmapped_notional_frac": unmapped_notional_frac,
        "scale_alias_candidates": scale_alias_candidates,
        "account_value_usd": round(account_value_usd, 2) if account_value_usd is not None else None,
        "account_value_status": account_value_status,
        "live_reasons": live_reasons,
        "live_deployable": len(live_reasons) == 0,
    }


def apply_live_account_values(
    rows: list[dict],
    av_by_alias: dict[str, tuple[float | None, dict | None, str | None]],
    *,
    min_account_value_usd: float = DEFAULT_MIN_ACCOUNT_VALUE_USD,
) -> list[dict]:
    """Attach present-day HL account-value checks strictly to ``live_reasons`` / ``live_deployable``."""
    av_tags = {
        "ACCOUNT_VALUE_NOT_FETCHED",
        "ACCOUNT_VALUE_UNPROVEN",
        _withdrawn_tag(min_account_value_usd),
    }
    updated: list[dict] = []
    for row in rows:
        item = dict(row)
        item["historical_reasons"] = list(row["historical_reasons"])
        live_reasons = [r for r in row["live_reasons"] if r not in av_tags]
        if item["alias"] in av_by_alias:
            av, _meta, err = av_by_alias[item["alias"]]
            item["account_value_usd"] = round(av, 2) if av is not None else None
            item["account_value_status"] = "VERIFIED" if av is not None else f"UNPROVEN:{err}"
            if av is None:
                live_reasons.append("ACCOUNT_VALUE_UNPROVEN")
            elif av < min_account_value_usd:
                live_reasons.append(_withdrawn_tag(min_account_value_usd))
        else:
            if item["account_value_status"] == "NOT_FETCHED":
                live_reasons.append("ACCOUNT_VALUE_NOT_FETCHED")
        item["live_reasons"] = live_reasons
        item["live_deployable"] = len(live_reasons) == 0
        updated.append(item)
    return updated


def screen_wallets(
    records: Sequence[cr.WalletRecord],
    ingestion: FillIngestionResult,
    *,
    split: int | datetime,
    bybit_usdt_perps: set[str] | None = None,
    account_value_fetcher: Callable[
        [cr.WalletRecord], tuple[float | None, dict | None, str | None]
    ]
    | None = None,
    fetch_live_av: bool = False,
    min_train_fills: int = DEFAULT_MIN_TRAIN_FILLS,
    max_train_inactivity_days: int = DEFAULT_MAX_TRAIN_INACTIVITY_DAYS,
    min_account_value_usd: float = DEFAULT_MIN_ACCOUNT_VALUE_USD,
    max_workers: int = 14,
) -> list[dict]:
    """Screen all universe wallets using CSE stats and separated historical/live gates."""
    stats_by_addr = ingestion.stats_df.partition_by("wallet", as_dict=True, maintain_order=False)
    empty_stats = ingestion.stats_df.head(0)

    rows = [
        summarize_wallet(
            rec,
            ingestion.grouped.get(rec.label, {}),
            split=split,
            bybit_usdt_perps=bybit_usdt_perps,
            min_train_fills=min_train_fills,
            max_train_inactivity_days=max_train_inactivity_days,
            min_account_value_usd=min_account_value_usd,
            wallet_stats=stats_by_addr.get((rec.address.lower(),), empty_stats),
        )
        for rec in records
    ]

    fetcher = account_value_fetcher
    if fetcher is None and fetch_live_av:
        fetcher = fetch_hyperliquid_account_value

    if fetcher is not None:
        av_by_alias: dict[str, tuple[float | None, dict | None, str | None]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(fetcher, rec): rec.label for rec in records}
            for fut in as_completed(futures):
                label = futures[fut]
                try:
                    av_by_alias[label] = fut.result()
                except (OSError, KeyError, TypeError, ValueError) as exc:
                    av_by_alias[label] = (None, None, type(exc).__name__)
        rows = apply_live_account_values(
            rows, av_by_alias, min_account_value_usd=min_account_value_usd
        )

    return rows


def eligible_fills(
    grouped: dict[str, dict[str, pl.DataFrame]],
    rows: list[dict],
    prices: LazyPrices,
    *,
    split: int | datetime,
    scope: Literal["historical", "live"] = "historical",
    max_missing_price_frac: float = 0.05,
) -> tuple[dict[str, dict[str, pl.DataFrame]], list[dict]]:
    """Filter wallet fills by Train 1m price coverage under ``historical`` or ``live`` scope."""
    split_ms = to_utc_ms(split)
    cut = int(np.searchsorted(prices.grid, split_ms))
    chosen: dict[str, dict[str, pl.DataFrame]] = {}
    updated: list[dict] = []

    for row in rows:
        item = dict(row)
        item["historical_reasons"] = list(row["historical_reasons"])
        item["live_reasons"] = list(row["live_reasons"])
        updated.append(item)

        if scope == "historical":
            is_active = item["historical_eligible"]
            candidate_coins = item["historical_coins"]
            prefix, reasons_key, flag_key = "", "historical_reasons", "historical_eligible"
        elif scope == "live":
            is_active = item["historical_eligible"] and item["live_deployable"]
            candidate_coins = item["mapped_coins"]
            prefix, reasons_key, flag_key = "live_", "live_reasons", "live_deployable"
        else:
            raise ValueError(f"UNSUPPORTED_SCREEN_SCOPE:{scope}")

        if not is_active:
            continue

        alias = item["alias"]
        wallet_fills = grouped.get(alias, {})
        usable: dict[str, pl.DataFrame] = {}
        no_price = 0
        for coin in candidate_coins:
            if coin not in wallet_fills:
                continue
            _tr, mk = prices.get(coin)
            if np.count_nonzero(np.isfinite(mk[:cut])) == 0:
                no_price += int((wallet_fills[coin]["ts"] < split_ms).sum())
                continue
            usable[coin] = wallet_fills[coin]

        missing_frac = round(no_price / max(1, item["train_fills"]), 5)
        item[f"{prefix}price_missing_train_fills"] = no_price
        item[f"{prefix}price_missing_train_frac"] = missing_frac
        if missing_frac > max_missing_price_frac or not usable:
            if "PRICE_MISSING_MATERIAL" not in item[reasons_key]:
                item[reasons_key].append("PRICE_MISSING_MATERIAL")
            item[flag_key] = False
        else:
            chosen[alias] = usable

    return chosen, updated
