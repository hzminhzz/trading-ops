#!/usr/bin/env python3
"""Altcoin-Short Hyperliquid Discovery & $10k Prop-Challenge Kernel Study.

Evaluates Gate-2 shortlisted altcoin-shorting Hyperliquid wallets on Google Colab:
- Reservoir S3 backfill (2026-05-01..2026-11-18) merged into isolated FactStore.
- Stage-1 PIT verification (`allow_undated=False` -> 0 cards) and retrospective Stage-1
  `score_universe` across `train_may_aug`, `oos_sep_nov`, and `full_may_nov_diag`
  under both strict `ScoreConfig()` and diagnostic `k`-scaled participation.
- Altcoin-short expertise forensics from FactStore `fills` + `bars_1m` (altcoin share,
  short-side share, altcoin-short PnL vs long PnL, 5m markout, capacity).
- Stage-2 strict OOS (`2026-09-01..2026-11-19`) `run_kernel_backtest` across `1x` and `2x`
  cost stress on `propr:classic_1step`, `vanta-trading:classic_1step_3pct`, and
  `hyrotrader:two_step` ($10,000 scale=2).

Provenance: RETROSPECTIVE_RESEARCH_ONLY (paper eligibility only, never TRADING_AUTHORIZED).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import re
import shutil
import sys
import tarfile
import tempfile
import threading
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from copytrade_kernel import KnownAtTs, Rulebook, RulebookCatalog
from copytrade_pm.facts import FactIngestor, FactStore
from copytrade_pm.facts import manifest as mf
from copytrade_pm.facts.migrate import INPUT_SEQ, dedupe_day
from copytrade_pm.facts.schemas import KNOWN_AT_BASIS_UNRECORDED, column_names
from copytrade_pm.facts.sources import RESERVOIR_LIQUID12_DISCOVERY, adapt_reservoir_fills
from copytrade_pm.roster import RosterStore, SeatSpec

ROOT = Path(__file__).resolve().parents[1]
for _extra_path in (ROOT / "src", ROOT / "studies"):
    if str(_extra_path) not in sys.path:
        sys.path.insert(0, str(_extra_path))

import pod_constructor as pc

from trading_ops.research import (
    DEFAULT_FEE_SCHEDULE_PATH,
    BacktestConfig,
    BacktestReport,
    BacktestWindow,
    MarketSpec,
    ScoreConfig,
    bench_from_scorecard,
    lookup_tier_fee,
    run_kernel_backtest,
    score_universe,
)
from trading_ops.research.scoring import _euler_marginal_cdar_shares

HEX_ADDRESS_RE = re.compile(r"(?i)0x[0-9a-f]{4,}")
STUDY_STEM = "2026-11-25__multi__10k__altcoin-short-discovery-study"
ACCOUNT_USD = 10_000.0
CORE_4_MAJORS = frozenset({"BTC", "ETH", "SOL", "HYPE"})
NON_ALT_COMMODITIES = frozenset({"PAXG", "XAUT"})

DEFAULT_EQUITY_ANCHOR_DAYS = (
    "2026-05-01",
    "2026-06-01",
    "2026-07-01",
    "2026-08-01",
    "2026-09-01",
    "2026-10-01",
    "2026-11-01",
    "2026-11-18",
)
TRAIN_AS_OF = datetime(2026, 9, 1, tzinfo=UTC)
TRAIN_LOOKBACK_DAYS = 123
OOS_AS_OF = datetime(2026, 11, 19, tzinfo=UTC)
OOS_LOOKBACK_DAYS = 79
FULL_AS_OF = datetime(2026, 11, 19, tzinfo=UTC)
FULL_LOOKBACK_DAYS = 202
OOS_START = datetime(2026, 9, 1, tzinfo=UTC)
OOS_END = datetime(2026, 11, 19, tzinfo=UTC)

FIRMS: tuple[tuple[str, str, str, str | None], ...] = (
    ("propr", "propr:classic_1step:phase_1", "propr:classic_1step", "Classic 1-Step"),
    (
        "vanta-trading",
        "vanta-trading:classic_1step_3pct:phase_1",
        "vanta-trading:classic_1step_3pct",
        None,
    ),
    ("hyrotrader", "hyrotrader:two_step:phase_1", "hyrotrader:two_step", None),
)

PERMISSIVE_SCORE_CONFIG = ScoreConfig(
    min_baseline_order_usd=Decimal("0.01"),
    max_p95_participation_1m=Decimal(1),
    max_max_participation_1m=Decimal(1),
    max_euler_cdar_share=Decimal(1),
    min_markout_fills=1,
)

RELAXED_PARTICIPATION_SCORE_CONFIG = ScoreConfig(
    max_p95_participation_1m=Decimal(1),
    max_max_participation_1m=Decimal(1),
)


def _utc_ms(dt: datetime) -> int:
    return int(dt.astimezone(UTC).timestamp() * 1000)


def _is_altcoin(coin: str) -> bool:
    c = coin.strip().upper()
    return bool(c) and c not in CORE_4_MAJORS and c not in NON_ALT_COMMODITIES


def _assert_sanitized_text(label: str, text: str) -> None:
    match = HEX_ADDRESS_RE.search(text)
    if match is not None:
        raise RuntimeError(f"HEX_ADDRESS_LEAK:{label}:{match.group(0)[:10]}")


def _sanitize_text(text: str, address_to_alias: Mapping[str, str]) -> str:
    if not text:
        return text

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0).lower()
        return address_to_alias.get(token, match.group(0))

    cleaned = HEX_ADDRESS_RE.sub(_replace, text)
    _assert_sanitized_text("sanitized_text", cleaned)
    return cleaned


def _fmt(value: Decimal | float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def resolve_max_workers(max_workers: int, cpu_count: int | None = None) -> int:
    if max_workers < 0:
        raise ValueError("INVALID_MAX_WORKERS")
    if max_workers == 0:
        detected = cpu_count if cpu_count is not None else (os.cpu_count() or 2)
        return max(1, detected - 1)
    return max_workers


def load_private_inputs(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    alias_to_address = {
        str(alias): str(addr).strip().lower() for alias, addr in raw["alias_to_address"].items()
    }
    address_to_alias = {
        str(addr).strip().lower(): str(alias) for addr, alias in raw["address_to_alias"].items()
    }
    addresses = tuple(sorted({str(addr).strip().lower() for addr in raw["addresses"]}))
    return {
        "addresses": addresses,
        "alias_to_address": alias_to_address,
        "address_to_alias": address_to_alias,
        "equity_snapshots": list(raw.get("equity_snapshots", [])),
        "gate2_summary": dict(raw.get("gate2_summary", {})),
    }


def prepare_reservoir_cache(
    *,
    reservoir_tar: Path | None,
    aws_creds_json: Path | None,
    staging_root: Path,
    addresses: Sequence[str],
    start_day: str = "2026-05-01",
    end_day: str = "2026-11-18",
) -> dict[str, Any]:
    """Unpack cached Reservoir fills and fetch any missing UTC days on Colab."""
    staging_root.mkdir(parents=True, exist_ok=True)
    if reservoir_tar is not None and reservoir_tar.is_file():
        with tarfile.open(reservoir_tar, "r:*") as tf:
            tf.extractall(staging_root, filter="data")

    all_dir = staging_root / "by_dex" / "hyperliquid" / "fills" / "perp" / "all"
    all_dir.mkdir(parents=True, exist_ok=True)

    cur = date.fromisoformat(start_day)
    end_d = date.fromisoformat(end_day)
    days: list[str] = []
    while cur <= end_d:
        days.append(cur.isoformat())
        cur += timedelta(days=1)

    cached_days = [
        d for d in days if (all_dir / f"date={d}" / "fills.parquet").is_file()
    ]
    missing_days = [d for d in days if d not in set(cached_days)]
    fetched_days = 0
    missing_s3 = 0

    if missing_days and aws_creds_json is not None and aws_creds_json.is_file():
        try:
            creds = json.loads(aws_creds_json.read_text(encoding="utf-8"))
        finally:
            aws_creds_json.unlink(missing_ok=True)

        import duckdb

        tls = threading.local()
        addr_list = sorted({a.strip().lower() for a in addresses})

        def _get_con():
            con = getattr(tls, "con", None)
            if con is None:
                con = duckdb.connect()
                con.execute("INSTALL httpfs; LOAD httpfs;")
                con.execute("SET enable_object_cache=false;")
                con.execute("SET threads=1;")
                con.execute("SET s3_region='ap-northeast-1';")
                con.execute("SET s3_requester_pays=true;")
                if creds.get("AWS_SESSION_TOKEN"):
                    con.execute(
                        "CREATE SECRET aws_res (TYPE S3, KEY_ID ?, SECRET ?, SESSION_TOKEN ?)",
                        [
                            creds["AWS_ACCESS_KEY_ID"],
                            creds["AWS_SECRET_ACCESS_KEY"],
                            creds["AWS_SESSION_TOKEN"],
                        ],
                    )
                else:
                    con.execute(
                        "CREATE SECRET aws_res (TYPE S3, KEY_ID ?, SECRET ?)",
                        [creds["AWS_ACCESS_KEY_ID"], creds["AWS_SECRET_ACCESS_KEY"]],
                    )
                con.execute("CREATE TEMP TABLE target_addrs (address VARCHAR);")
                con.executemany("INSERT INTO target_addrs VALUES (?)", [(a,) for a in addr_list])
                tls.con = con
            return con

        def _fetch_one(day_str: str) -> str:
            day_dir = all_dir / f"date={day_str}"
            day_dir.mkdir(parents=True, exist_ok=True)
            out_file = day_dir / "fills.parquet"
            tmp_file = day_dir / "fills.parquet.tmp"
            s3_url = (
                "s3://hydromancer-reservoir/by_dex/hyperliquid/fills/perp/all/"
                f"date={day_str}/fills.parquet"
            )
            sql = f"""
                COPY (
                    SELECT
                        lower(f.address) AS wallet,
                        epoch_ms(f.timestamp) AS exchange_ts,
                        f.coin,
                        f.dex,
                        f.side,
                        f.size,
                        f.start_position,
                        CAST(f.trade_id AS VARCHAR) AS trade_id,
                        f.direction,
                        f.is_liquidation,
                        f.price,
                        f.fee
                    FROM read_parquet('{s3_url}') f
                    JOIN target_addrs u ON lower(f.address) = u.address
                ) TO '{tmp_file.as_posix()}' (FORMAT PARQUET, COMPRESSION 'zstd')
            """
            con = _get_con()
            for attempt in range(3):
                try:
                    con.execute(sql)
                    os.replace(tmp_file, out_file)
                    return "FETCHED"
                except Exception as exc:
                    tmp_file.unlink(missing_ok=True)
                    msg = str(exc)
                    if "404" in msg or "NoSuchKey" in msg:
                        return "MISSING_S3"
                    if attempt == 2:
                        raise
                    time.sleep(2.0 * (attempt + 1))
            return "MISSING_S3"

        print(
            f"[reservoir] fetching {len(missing_days)} missing S3 days on Colab (4 workers)...",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = {pool.submit(_fetch_one, d): d for d in missing_days}
            for idx, fut in enumerate(as_completed(futs), 1):
                st = fut.result()
                if st == "FETCHED":
                    fetched_days += 1
                else:
                    missing_s3 += 1
                if idx % 25 == 0 or idx == len(missing_days):
                    print(
                        f"[reservoir] {idx}/{len(missing_days)} fetched={fetched_days} missing={missing_s3}",
                        flush=True,
                    )
    elif aws_creds_json is not None and aws_creds_json.is_file():
        aws_creds_json.unlink(missing_ok=True)

    return {
        "days_expected": len(days),
        "days_pre_cached": len(cached_days),
        "days_fetched_on_colab": fetched_days,
        "days_missing_s3": missing_s3,
        "all_dir": str(all_dir),
    }


def _resolve_batch_path(store: FactStore, batch: mf.DayBatch) -> Path:
    recorded = Path(batch.path)
    if recorded.is_file():
        return recorded
    fallback = store.day_dir(batch.kind, batch.day) / f"{batch.batch_sha16}.parquet"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"DAY_BATCH_MISSING:{batch.kind}:{batch.day}:{batch.path}")


def _record_day_batch(
    dst: FactStore,
    *,
    kind: str,
    day: str,
    frame: pl.DataFrame,
    write: Any,
    rel_prefix: str,
) -> None:
    con = dst.connect()
    frame_cols = frame.select(list(column_names(kind)))
    by_source = (
        frame_cols.with_columns(
            pl.when(pl.col("first_known_at_ms").is_not_null())
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("_obs")
        )
        .group_by(["source_id", "_obs"])
        .agg(pl.len().alias("rows"), pl.col("first_known_at_ms").min().alias("min_known"))
        .sort(["source_id", "_obs"])
    )
    inputs: list[mf.InputRecord] = []
    source_totals: dict[str, int] = {}
    for row in by_source.iter_rows(named=True):
        source_id = str(row["source_id"])
        rows_count = int(row["rows"])
        tier_tag = "observed" if int(row["_obs"]) == 1 else "retrospective"
        min_known = None if row["min_known"] is None else int(row["min_known"])
        input_sha = hashlib.sha256(
            f"{kind}|{day}|{source_id}|{tier_tag}|{write.file_sha256}|{rows_count}".encode()
        ).hexdigest()
        inputs.append(
            mf.InputRecord(
                kind=kind,
                day=day,
                source_id=source_id,
                rel_path=f"{rel_prefix}/{kind}/{day}/{tier_tag}",
                input_sha256=input_sha,
                rows_read=rows_count,
                rows_kept=rows_count,
                rows_duplicate_within=0,
                rows_duplicate_cross=0,
                rows_conflicting_values=0,
                rows_rejected=0,
                rows_null_key=0,
                first_known_at_ms=min_known,
            )
        )
        source_totals[source_id] = source_totals.get(source_id, 0) + rows_count

    fingerprint = hashlib.sha256(
        json.dumps(
            sorted((it.source_id, it.rel_path, it.input_sha256) for it in inputs),
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    day_sources = [
        {
            "source_id": sid,
            "files_read": 1,
            "rows_read": cnt,
            "rows_kept": cnt,
            "rows_duplicate_within": 0,
            "rows_duplicate_cross": 0,
            "rows_conflicting_values": 0,
            "rows_rejected": 0,
            "rows_null_key": 0,
            "rows_quantized_to_zero": 0,
        }
        for sid, cnt in sorted(source_totals.items())
    ]
    known_values = [
        int(val) for val in frame_cols["first_known_at_ms"].to_list() if val is not None
    ]
    mf.record_day(
        con,
        batch=mf.DayBatch(
            kind=kind,
            day=day,
            batch_sha16=write.batch_sha16,
            path=str(write.path),
            rows=write.rows,
            file_sha256=write.file_sha256,
            inputs_fingerprint=fingerprint,
        ),
        source_ids=sorted(source_totals),
        bytes_count=write.bytes_count,
        inputs=inputs,
        day_sources=day_sources,
        first_known_min_ms=min(known_values) if known_values else None,
        first_known_max_ms=max(known_values) if known_values else None,
        closed_at=datetime.now(tz=UTC).isoformat(),
    )


def _build_equity_anchors(
    equity_df: pl.DataFrame, anchor_days: Sequence[str]
) -> list[dict[str, Any]]:
    if equity_df.height == 0 or not anchor_days:
        return []
    ordered = equity_df.sort(["wallet", "event_ts_ms"])
    existing_by_wallet: dict[str, set[str]] = {}
    rows_by_wallet: dict[str, list[dict[str, Any]]] = {}
    for row in ordered.iter_rows(named=True):
        wallet = str(row["wallet"]).lower()
        row_day = datetime.fromtimestamp(int(row["event_ts_ms"]) / 1000, UTC).date().isoformat()
        existing_by_wallet.setdefault(wallet, set()).add(row_day)
        rows_by_wallet.setdefault(wallet, []).append(row)

    anchors: list[dict[str, Any]] = []
    for wallet, wallet_rows in sorted(rows_by_wallet.items()):
        seen_days = existing_by_wallet[wallet]
        for day_str in anchor_days:
            if day_str in seen_days:
                continue
            anchor_dt = datetime.fromisoformat(f"{day_str}T00:00:00+00:00")
            anchor_ms = int(anchor_dt.timestamp() * 1000)
            prior = [row for row in wallet_rows if int(row["event_ts_ms"]) <= anchor_ms]
            base = prior[-1] if prior else wallet_rows[0]
            anchors.append(
                {
                    "wallet": wallet,
                    "account_scope": base["account_scope"],
                    "abstraction_mode": base["abstraction_mode"],
                    "equity_usd": base["equity_usd"],
                    "withdrawable_usd": base["withdrawable_usd"],
                    "margin_used_usd": base["margin_used_usd"],
                    "observed_at": anchor_dt,
                    "event_ts_ms": anchor_ms,
                    "first_known_at_ms": None,
                    "known_at_basis": KNOWN_AT_BASIS_UNRECORDED,
                    "arrival_seq": 0,
                    "source_id": base["source_id"],
                    "source_rank": int(base["source_rank"]),
                    "batch_sha16": "",
                }
            )
    return anchors


def build_isolated_fact_store(
    src_store_path: Path,
    dst_store_path: Path,
    *,
    wallets: Sequence[str],
    reservoir_all_dir: Path | None = None,
    equity_snapshots: Sequence[Mapping[str, Any]] = (),
    anchor_days: Sequence[str] = DEFAULT_EQUITY_ANCHOR_DAYS,
    start_day: str = "2026-05-01",
    end_day: str = "2026-11-18",
) -> tuple[str, ...]:
    """Create an isolated FactStore containing target wallets, merged Reservoir fills, equity, and bars_1m."""
    target_wallets = sorted({str(w).strip().lower() for w in wallets if str(w).strip()})
    if not target_wallets:
        raise ValueError("EMPTY_WALLETS")
    if dst_store_path.exists():
        shutil.rmtree(dst_store_path)

    raw_coins: set[str] = set()
    norm_coins: set[str] = set()
    with (
        FactStore(src_store_path, create=False) as src,
        FactStore(dst_store_path, create=True) as dst,
    ):
        src_con = src.connect()
        dst.connect()

        src_batches = {
            batch.day: batch
            for batch in mf.day_batches(src_con, "fills", start_day=start_day, end_day=end_day)
        }
        cur_d = date.fromisoformat(start_day)
        end_d = date.fromisoformat(end_day)
        all_days: list[str] = []
        while cur_d <= end_d:
            all_days.append(cur_d.isoformat())
            cur_d += timedelta(days=1)

        for day in all_days:
            frames_to_merge: list[pl.DataFrame] = []
            batch = src_batches.get(day)
            if batch is not None:
                batch_path = _resolve_batch_path(src, batch)
                src_df = (
                    pl.scan_parquet(batch_path)
                    .filter(pl.col("wallet").is_in(target_wallets))
                    .collect()
                )
                if src_df.height > 0:
                    frames_to_merge.append(
                        src_df.with_columns(pl.lit(0, dtype=pl.UInt32).alias(INPUT_SEQ))
                    )

            if reservoir_all_dir is not None:
                res_path = reservoir_all_dir / f"date={day}" / "fills.parquet"
                if res_path.is_file() and res_path.stat().st_size > 0:
                    raw_res = pl.read_parquet(res_path).filter(
                        pl.col("wallet").str.to_lowercase().is_in(target_wallets)
                    )
                    if raw_res.height > 0:
                        adapted = adapt_reservoir_fills(
                            raw_res,
                            source_id=RESERVOIR_LIQUID12_DISCOVERY,
                            source_rank=4,
                            first_known_at_ms=None,
                            known_at_basis=KNOWN_AT_BASIS_UNRECORDED,
                        )
                        valid_res = adapted.frame.filter(
                            pl.col("wallet").is_not_null() & pl.col("event_ts_ms").is_not_null()
                        )
                        if valid_res.height > 0:
                            frames_to_merge.append(
                                valid_res.with_columns(pl.lit(1, dtype=pl.UInt32).alias(INPUT_SEQ))
                            )

            if not frames_to_merge:
                continue

            if len(frames_to_merge) == 1:
                day_df = frames_to_merge[0].drop(INPUT_SEQ)
            else:
                combined = pl.concat(frames_to_merge, how="vertical_relaxed")
                day_df = dedupe_day(combined, "fills").frame

            if day_df.height == 0:
                continue
            for coin in day_df.select(pl.col("coin").unique()).to_series().to_list():
                if coin is not None and str(coin):
                    c_str = str(coin)
                    raw_coins.add(c_str)
                    norm_coins.add(c_str.strip().upper())
            prepared = dst.prepare_day("fills", day, day_df)
            write = dst.write_day(prepared)
            _record_day_batch(
                dst,
                kind="fills",
                day=day,
                frame=prepared.frame,
                write=write,
                rel_prefix="cohort",
            )

        equity_frames: list[pl.DataFrame] = []
        for batch in mf.day_batches(src_con, "equity"):
            batch_path = _resolve_batch_path(src, batch)
            part = (
                pl.scan_parquet(batch_path)
                .filter(pl.col("wallet").is_in(target_wallets))
                .collect()
            )
            if part.height > 0:
                equity_frames.append(part)

        ingestor = FactIngestor(dst)
        if equity_frames:
            combined_eq = pl.concat(equity_frames, how="vertical_relaxed")
            ingestor.ingest_frame(
                "equity",
                combined_eq,
                source_id="cohort-equity",
                source_rank=1,
                first_known_at_ms=None,
            )
        if equity_snapshots:
            target_set = set(target_wallets)
            snap_rows = []
            for snap in equity_snapshots:
                w = str(snap["wallet"]).strip().lower()
                if w not in target_set:
                    continue
                obs_ms = int(snap["observed_at_ms"])
                snap_rows.append(
                    {
                        "wallet": w,
                        "account_scope": "HL_MAIN_PERP_ONLY",
                        "abstraction_mode": "default",
                        "equity_usd": Decimal(f"{float(snap['account_value']):.12f}"),
                        "withdrawable_usd": Decimal(f"{float(snap['withdrawable']):.12f}"),
                        "margin_used_usd": Decimal(f"{float(snap['margin_used']):.12f}"),
                        "observed_at": datetime.fromtimestamp(obs_ms / 1000.0, tz=UTC),
                        "event_ts_ms": obs_ms,
                        "first_known_at_ms": None,
                        "known_at_basis": KNOWN_AT_BASIS_UNRECORDED,
                        "arrival_seq": 0,
                        "source_id": "official_hyperliquid_info",
                        "source_rank": 10,
                        "batch_sha16": "",
                    }
                )
            if snap_rows:
                ingestor.ingest_rows(
                    "equity",
                    snap_rows,
                    source_id="official_hyperliquid_info",
                    source_rank=10,
                    first_known_at_ms=None,
                )

        all_eq = dst.scan("equity", wallets=target_wallets, allow_unknown_known_at=True).collect()
        if all_eq.height > 0:
            anchor_rows = _build_equity_anchors(all_eq, anchor_days)
            if anchor_rows:
                ingestor.ingest_rows(
                    "equity",
                    anchor_rows,
                    source_id="equity-anchor",
                    source_rank=1,
                    first_known_at_ms=None,
                )

        sorted_norm_coins = sorted(norm_coins)
        if sorted_norm_coins:
            for batch in mf.day_batches(src_con, "bars_1m", start_day=start_day, end_day=end_day):
                batch_path = _resolve_batch_path(src, batch)
                day_df = (
                    pl.scan_parquet(batch_path)
                    .filter(
                        pl.col("coin").str.strip_chars().str.to_uppercase().is_in(sorted_norm_coins)
                    )
                    .collect()
                )
                if day_df.height == 0:
                    continue
                prepared = dst.prepare_day("bars_1m", batch.day, day_df)
                write = dst.write_day(prepared)
                _record_day_batch(
                    dst,
                    kind="bars_1m",
                    day=batch.day,
                    frame=prepared.frame,
                    write=write,
                    rel_prefix="cohort",
                )
    return tuple(sorted(raw_coins))


def compute_altcoin_short_profiles(
    facts: FactStore,
    *,
    wallets: Sequence[str],
    address_to_alias: Mapping[str, str],
    start_ms: int,
    end_ms: int,
) -> dict[str, dict[str, Any]]:
    """Compute per-wallet altcoin share, short-side dominance, and short vs long PnL from FactStore."""
    fills_df = (
        facts.scan("fills", wallets=list(wallets), allow_unknown_known_at=True)
        .collect()
        .filter((pl.col("event_ts_ms") >= start_ms) & (pl.col("event_ts_ms") < end_ms))
        .sort(["wallet", "coin", "event_ts_ms", "arrival_seq"])
    )
    bars_df = (
        facts.scan("bars_1m", allow_unknown_known_at=True)
        .collect()
        .filter((pl.col("event_ts_ms") >= start_ms) & (pl.col("event_ts_ms") < end_ms))
    )

    last_close: dict[str, float] = {}
    bar_close_lookup: dict[tuple[str, int], float] = {}
    if bars_df.height > 0:
        for row in bars_df.sort("event_ts_ms").group_by("coin").agg(pl.col("close").last()).iter_rows():
            last_close[str(row[0]).strip().upper()] = float(row[1])
        for row in bars_df.select(["coin", "event_ts_ms", "close"]).iter_rows():
            bar_close_lookup[(str(row[0]).strip().upper(), int(row[1]))] = float(row[2])

    profiles: dict[str, dict[str, Any]] = {}
    if fills_df.height == 0:
        return profiles

    for wallet_key, w_df in fills_df.group_by("wallet", maintain_order=True):
        addr = str(wallet_key[0] if isinstance(wallet_key, tuple) else wallet_key).lower()
        alias = address_to_alias.get(addr, addr)

        total_ntl = 0.0
        alt_ntl = 0.0
        short_ntl = 0.0
        long_ntl = 0.0
        alt_short_ntl = 0.0
        short_fills = 0
        total_fills = 0
        coin_ntl: dict[str, float] = defaultdict(float)

        short_realized_pnl = 0.0
        long_realized_pnl = 0.0
        alt_short_realized_pnl = 0.0
        total_fees = 0.0

        # Track position and entry basis per coin to split realized PnL by side cleanly
        pos_qty: dict[str, float] = {}
        entry_px: dict[str, float] = {}

        # Also track 5m markout on altcoin-short opening fills
        alt_short_markout_w = 0.0
        alt_short_markout_num = 0.0
        alt_short_markout_fills = 0

        for row in w_df.iter_rows(named=True):
            coin_raw = str(row["coin"] or "").strip()
            coin = coin_raw.upper()
            side_raw = str(row["side"] or "").strip().lower()
            dir_raw = str(row["direction"] or "").strip()
            px = float(row["price"]) if row["price"] is not None else 0.0
            sz = float(row["size"]) if row["size"] is not None else 0.0
            fee = float(row["fee"]) if row["fee"] is not None else 0.0
            ts_ms = int(row["event_ts_ms"])

            if px <= 0.0 or sz <= 0.0:
                continue
            sign = 1.0 if side_raw in ("b", "buy", "bid", "long") else (-1.0 if side_raw in ("a", "s", "sell", "ask", "short") else 0.0)
            if sign == 0.0:
                continue

            ntl = px * sz
            total_fills += 1
            total_ntl += ntl
            total_fees += fee
            coin_ntl[coin] += ntl
            is_alt = _is_altcoin(coin)
            if is_alt:
                alt_ntl += ntl

            # Determine short vs long attribution via direction or running position
            cur = pos_qty.get(coin, 0.0)
            is_short_fill = False
            if dir_raw:
                if "Short" in dir_raw:
                    is_short_fill = True
            else:
                if cur < 0 or (cur == 0 and sign < 0):
                    is_short_fill = True

            if is_short_fill:
                short_fills += 1
                short_ntl += ntl
                if is_alt:
                    alt_short_ntl += ntl
            else:
                long_ntl += ntl

            # 5m forward markout on altcoin-short opening/increasing fills
            if is_alt and sign < 0:
                min_0 = (ts_ms // 60_000) * 60_000
                p5 = bar_close_lookup.get((coin, min_0 + 300_000))
                if p5 is not None and p5 > 0.0:
                    alt_short_markout_fills += 1
                    alt_short_markout_w += ntl
                    alt_short_markout_num += ntl * (-1.0) * (p5 - px) / px

            # Realized PnL tracking
            delta = sign * sz
            if cur == 0.0:
                pos_qty[coin] = delta
                entry_px[coin] = px
            elif (cur > 0 and delta > 0) or (cur < 0 and delta < 0):
                new_abs = abs(cur) + abs(delta)
                entry_px[coin] = (abs(cur) * entry_px[coin] + abs(delta) * px) / new_abs
                pos_qty[coin] = cur + delta
            else:
                closed = min(abs(cur), abs(delta))
                pnl = (px - entry_px[coin]) * closed * (1.0 if cur > 0 else -1.0)
                if cur < 0:
                    short_realized_pnl += pnl
                    if is_alt:
                        alt_short_realized_pnl += pnl
                else:
                    long_realized_pnl += pnl
                rem = cur + delta
                if abs(rem) < 1e-12:
                    pos_qty.pop(coin, None)
                    entry_px.pop(coin, None)
                elif (cur > 0 and rem < 0) or (cur < 0 and rem > 0):
                    pos_qty[coin] = rem
                    entry_px[coin] = px
                else:
                    pos_qty[coin] = rem

        # Add unrealized MTM at window end
        short_unreal = 0.0
        long_unreal = 0.0
        alt_short_unreal = 0.0
        for coin, qty in pos_qty.items():
            mark_px = last_close.get(coin)
            if mark_px is None or mark_px <= 0.0:
                continue
            upnl = (mark_px - entry_px[coin]) * qty
            if qty < 0:
                short_unreal += upnl
                if _is_altcoin(coin):
                    alt_short_unreal += upnl
            else:
                long_unreal += upnl

        top_coins = sorted(coin_ntl.items(), key=lambda kv: kv[1], reverse=True)[:5]
        alt_short_markout_bps = (
            (alt_short_markout_num / alt_short_markout_w) * 1e4
            if alt_short_markout_w > 0
            else None
        )
        profiles[alias] = {
            "alias": alias,
            "total_fills": total_fills,
            "distinct_coins": len(coin_ntl),
            "total_notional_usd": round(total_ntl, 2),
            "altcoin_notional_frac": round(alt_ntl / total_ntl, 4) if total_ntl > 0 else 0.0,
            "short_notional_frac": round(short_ntl / max(1e-9, short_ntl + long_ntl), 4)
            if (short_ntl + long_ntl) > 0
            else 0.0,
            "short_fill_frac": round(short_fills / total_fills, 4) if total_fills > 0 else 0.0,
            "alt_short_notional_frac": round(alt_short_ntl / total_ntl, 4) if total_ntl > 0 else 0.0,
            "short_net_pnl_usd": round(short_realized_pnl + short_unreal, 2),
            "long_net_pnl_usd": round(long_realized_pnl + long_unreal, 2),
            "alt_short_net_pnl_usd": round(alt_short_realized_pnl + alt_short_unreal, 2),
            "total_fees_usd": round(total_fees, 2),
            "alt_short_entry_markout_5m_bps": round(alt_short_markout_bps, 4)
            if alt_short_markout_bps is not None
            else None,
            "alt_short_entry_fills": alt_short_markout_fills,
            "top_coins": [
                {"coin": c, "share": round(v / total_ntl, 4)} for c, v in top_coins
            ]
            if total_ntl > 0
            else [],
        }
    return profiles


def summarize_scorecard_window(
    facts: FactStore,
    *,
    name: str,
    as_of: KnownAtTs,
    lookback_days: int,
    wallets: Sequence[str],
    address_to_alias: Mapping[str, str],
    roster: RosterStore,
    score_config: ScoreConfig | None = None,
) -> tuple[dict[str, Any], Any]:
    scorecard = score_universe(
        facts,
        as_of=as_of,
        lookback_days=lookback_days,
        wallets=list(wallets),
        config=score_config or ScoreConfig(),
        allow_undated=True,
    )
    recorded_shas = scorecard.record(roster)
    bench = bench_from_scorecard(scorecard)
    bench_by_wallet = {row.wallet: row for row in bench}
    audit_by_wallet = {row.wallet: row for row in scorecard.audit}
    cards_by_wallet = scorecard.by_wallet
    excluded_by_wallet = {row.wallet: row for row in scorecard.excluded}

    excluded_counts: dict[str, int] = {}
    failed_counts: dict[str, int] = {}
    wallet_rows: list[dict[str, Any]] = []
    for addr in sorted(wallets):
        alias = address_to_alias[addr]
        card = cards_by_wallet.get(addr)
        audit = audit_by_wallet.get(addr)
        if card is not None and audit is not None:
            failures = [f.value for f in card.failures]
            for f in failures:
                failed_counts[f] = failed_counts.get(f, 0) + 1
            bench_row = bench_by_wallet.get(addr)
            wallet_rows.append(
                {
                    "alias": alias,
                    "eligible": card.eligible,
                    "failures": failures,
                    "excluded_reason": None,
                    "account_value_usd": _fmt(card.account_value_usd, 2),
                    "q10_order_usd": _fmt(audit.q10_order_usd, 2),
                    "markout_5m_bps": _fmt(audit.markout_5m_bps, 4),
                    "markout_fills": audit.markout_fills,
                    "p95_participation_1m": _fmt(audit.p95_participation_1m, 6),
                    "max_participation_1m": _fmt(audit.max_participation_1m, 6),
                    "est_k_scaled_p95_1m": round(
                        float(card.suggested_k) * float(audit.p95_participation_1m), 6
                    ),
                    "est_k_scaled_max_1m": round(
                        float(card.suggested_k) * float(audit.max_participation_1m), 6
                    ),
                    "cdar_share": _fmt(audit.cdar_share, 6),
                    "bench_cdar_share": _fmt(bench_row.marginal_cdar_share, 6)
                    if bench_row
                    else None,
                    "wallet_max_dd_usd": _fmt(audit.wallet_max_dd_usd, 2),
                    "suggested_k": _fmt(card.suggested_k, 6),
                    "active_days_14d": audit.active_days_14d,
                    "cluster_id": card.cluster_id,
                }
            )
        elif addr in excluded_by_wallet:
            reason = excluded_by_wallet[addr].reason
            excluded_counts[reason] = excluded_counts.get(reason, 0) + 1
            wallet_rows.append(
                {
                    "alias": alias,
                    "eligible": False,
                    "failures": [],
                    "excluded_reason": reason,
                    "observed": excluded_by_wallet[addr].observed,
                }
            )
        else:
            excluded_counts["NO_FILLS_IN_WINDOW"] = (
                excluded_counts.get("NO_FILLS_IN_WINDOW", 0) + 1
            )
            wallet_rows.append(
                {
                    "alias": alias,
                    "eligible": False,
                    "failures": [],
                    "excluded_reason": "NO_FILLS_IN_WINDOW",
                    "observed": 0,
                }
            )

    eligible_aliases = [r["alias"] for r in wallet_rows if r["eligible"]]
    summary = {
        "window_name": name,
        "as_of_iso": datetime.fromtimestamp(as_of.value / 1000, tz=UTC).isoformat(),
        "lookback_days": lookback_days,
        "basis": dict(sorted(scorecard.basis.items())),
        "source_sha": scorecard.source_sha,
        "slice_days": scorecard.slice_days,
        "slice_rows": scorecard.slice_rows,
        "total_wallets": len(wallets),
        "scored_count": len(scorecard.cards),
        "eligible_count": len(eligible_aliases),
        "excluded_count": len(wallets) - len(scorecard.cards),
        "scorecards_recorded": len(recorded_shas),
        "excluded_counts": dict(sorted(excluded_counts.items())),
        "failed_gate_counts": dict(sorted(failed_counts.items())),
        "eligible_aliases": eligible_aliases,
        "wallets": wallet_rows,
    }
    return summary, scorecard


def form_train_pods(
    train_scorecard: Any,
    *,
    address_to_alias: Mapping[str, str],
    allowed_wallets: set[str] | None = None,
    pod_prefix: str = "ASPOD",
    max_euler_share: float = 0.45,
) -> list[dict[str, Any]]:
    """Form 3-4 seat low-crowding pods strictly from Train-eligible wallets."""
    eligible_cards = [
        c
        for c in train_scorecard.cards
        if c.eligible and (allowed_wallets is None or c.wallet in allowed_wallets)
    ]
    if len(eligible_cards) < 3:
        return []

    audit_map = {row.wallet: row for row in train_scorecard.audit}
    returns_map = train_scorecard._returns_by_wallet

    elig_addrs = [c.wallet for c in eligible_cards]
    mat = np.array([returns_map[a] for a in elig_addrs], dtype=np.float64)
    centred = mat - mat.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.sum(centred**2, axis=1, keepdims=True))
    norms[norms == 0.0] = 1.0
    unit = centred / norms
    corr = np.clip(unit @ unit.T, -1.0, 1.0)
    idx_by_addr = {a: i for i, a in enumerate(elig_addrs)}

    def _pod_metrics(addrs: list[str]) -> tuple[bool, list[float], float]:
        sub_mat = np.array([returns_map[a] for a in addrs], dtype=np.float64)
        sub_k = np.array(
            [float(next(c.suggested_k for c in eligible_cards if c.wallet == a)) for a in addrs],
            dtype=np.float64,
        )
        shares = _euler_marginal_cdar_shares(sub_mat, sub_k, train_scorecard.config.cdar_alpha)
        shares_f = [round(float(s), 6) for s in shares]
        pairs = []
        for i in range(len(addrs)):
            for j in range(i + 1, len(addrs)):
                pairs.append(float(corr[idx_by_addr[addrs[i]], idx_by_addr[addrs[j]]]))
        avg_corr = float(np.mean(pairs)) if pairs else 0.0
        ok = all(max(0.0, float(s)) <= max_euler_share for s in shares)
        return ok, shares_f, round(avg_corr, 4)

    def _greedy_low_corr(seed_addr: str, size: int) -> list[str]:
        chosen = [seed_addr]
        used_clusters = {next(c.cluster_id for c in eligible_cards if c.wallet == seed_addr)}
        remaining = [a for a in elig_addrs if a != seed_addr]
        while len(chosen) < size and remaining:
            def _key(cand: str) -> tuple[int, float, float]:
                cl = next(c.cluster_id for c in eligible_cards if c.wallet == cand)
                new_cluster = 0 if cl not in used_clusters else 1
                max_c = max(float(corr[idx_by_addr[cand], idx_by_addr[s]]) for s in chosen)
                mk = float(audit_map[cand].markout_5m_bps)
                return (new_cluster, round(max_c, 4), -mk)

            remaining.sort(key=_key)
            picked = None
            for cand in remaining:
                ok, _, _ = _pod_metrics(chosen + [cand])
                if ok or len(chosen) + 1 < size:
                    picked = cand
                    break
            if picked is None:
                picked = remaining[0]
            chosen.append(picked)
            used_clusters.add(next(c.cluster_id for c in eligible_cards if c.wallet == picked))
            remaining.remove(picked)
        return chosen

    by_markout = sorted(
        elig_addrs, key=lambda a: float(audit_map[a].markout_5m_bps), reverse=True
    )
    by_low_dd = sorted(
        elig_addrs, key=lambda a: float(audit_map[a].wallet_max_dd_usd)
    )

    candidate_specs: list[tuple[str, list[str]]] = [
        (f"{pod_prefix}-DIV3", _greedy_low_corr(by_markout[0], 3)),
        (f"{pod_prefix}-LOWDD3", _greedy_low_corr(by_low_dd[0], 3)),
        (f"{pod_prefix}-TOP3", by_markout[:3]),
    ]
    if len(elig_addrs) >= 4:
        candidate_specs.append((f"{pod_prefix}-DIV4", _greedy_low_corr(by_markout[0], 4)))
        candidate_specs.append((f"{pod_prefix}-LOWDD4", _greedy_low_corr(by_low_dd[0], 4)))

    pods: list[dict[str, Any]] = []
    seen_sets: set[tuple[str, ...]] = set()
    for pod_id, addrs in candidate_specs:
        key = tuple(sorted(addrs))
        if key in seen_sets or len(addrs) < 3:
            continue
        ok, shares, avg_corr = _pod_metrics(addrs)
        if not ok:
            continue
        seen_sets.add(key)
        pods.append(
            {
                "pod_id": pod_id,
                "size": len(addrs),
                "aliases": [address_to_alias[a] for a in addrs],
                "addresses": addrs,
                "train_euler_shares": shares,
                "train_avg_pairwise_corr": avg_corr,
                "train_suggested_k": [
                    _fmt(next(c.suggested_k for c in eligible_cards if c.wallet == a), 6)
                    for a in addrs
                ],
                "train_clusters": [
                    next(c.cluster_id for c in eligible_cards if c.wallet == a) for a in addrs
                ],
            }
        )
    return pods[:4]


def _compute_oos_diagnostics(
    report: BacktestReport,
    *,
    pod_facts: FactStore,
    window: BacktestWindow,
    fee_rate: Decimal,
    shortfall_bps: Decimal,
    spread_bps: Decimal,
) -> dict[str, Any]:
    """Reconstruct exact MTM equity curve, DDs, 1m bar participation, and 7-tuple EV blocks."""
    bars_df = pod_facts.scan(
        "bars_1m",
        start_day=window.start_day,
        end_day=window.end_day,
        allow_unknown_known_at=True,
    ).collect()
    bar_lookup: dict[tuple[str, int], tuple[float, float]] = {}
    if bars_df.height > 0:
        for row in bars_df.select(["coin", "event_ts_ms", "close", "quote_volume_usd"]).iter_rows():
            bar_lookup[(str(row[0]).upper(), int(row[1]))] = (float(row[2]), float(row[3]))

    positions: dict[str, Decimal] = {}
    entry_basis: dict[str, Decimal] = {}
    realized = Decimal(0)
    initial = Decimal(10_000)
    half_spread = float(spread_bps) / 20_000.0
    cost_rate = fee_rate + shortfall_bps / Decimal(10_000)

    participations: list[float] = []
    unknown_vol_trades = 0
    day_events: dict[str, list[tuple[float, float]]] = {}
    step_by_ms = {step.checkpoint_ms: step for step in report.decisions}

    peak_eq = float(initial)
    max_p2t_dd = 0.0
    min_eq_overall = float(initial)

    bar = window.start_ms
    while bar < window.end_ms:
        day_str = datetime.fromtimestamp(bar / 1000, UTC).date().isoformat()
        step = step_by_ms.get(bar)
        if step is not None and step.status == "ACCEPTED":
            for intent in step.net_intents:
                coin = str(intent["market"]).upper()
                delta = Decimal(str(intent["net_delta_qty"]))
                if delta == 0:
                    continue
                px_vol = bar_lookup.get((coin, bar))
                if px_vol is None:
                    px_vol = bar_lookup.get((coin, bar - 60_000)) or bar_lookup.get(
                        (coin, bar + 60_000)
                    )
                if px_vol is None or px_vol[0] <= 0.0:
                    unknown_vol_trades += 1
                    continue
                mid_px, qvol = px_vol
                if qvol > 0.0:
                    participations.append(abs(float(delta)) * mid_px / qvol)
                else:
                    unknown_vol_trades += 1

                exec_px = Decimal(
                    f"{mid_px * (1.0 + half_spread if delta > 0 else 1.0 - half_spread):.8f}"
                )
                realized -= abs(delta) * exec_px * cost_rate
                cur = positions.get(coin, Decimal(0))
                new_qty = cur + delta
                if cur != 0 and (cur > 0) != (delta > 0):
                    closed = min(abs(cur), abs(delta))
                    realized += (exec_px - entry_basis[coin]) * closed * (1 if cur > 0 else -1)
                if new_qty == 0:
                    positions.pop(coin, None)
                    entry_basis.pop(coin, None)
                elif cur == 0 or (cur > 0) != (new_qty > 0):
                    positions[coin] = new_qty
                    entry_basis[coin] = exec_px
                else:
                    old_abs = abs(cur)
                    add_abs = abs(delta)
                    entry_basis[coin] = (old_abs * entry_basis[coin] + add_abs * exec_px) / abs(
                        new_qty
                    )
                    positions[coin] = new_qty

        unreal = Decimal(0)
        for coin, qty in positions.items():
            px_vol = bar_lookup.get((coin, bar))
            if px_vol is not None and px_vol[0] > 0.0:
                unreal += (Decimal(repr(px_vol[0])) - entry_basis[coin]) * qty

        cash_now = float(initial + realized)
        eq_now = cash_now + float(unreal)
        peak_eq = max(peak_eq, eq_now)
        max_p2t_dd = max(max_p2t_dd, peak_eq - eq_now)
        min_eq_overall = min(min_eq_overall, eq_now)
        day_events.setdefault(day_str, []).append((eq_now, cash_now))
        bar += 300_000

    static_floor_dd = max(0.0, float(initial) - min_eq_overall)

    sorted_days = sorted(day_events.keys())
    n_days = len(sorted_days)
    is_chop = np.zeros(n_days, dtype=bool)
    day_end = np.zeros(n_days, dtype=np.float64)
    day_max = np.zeros(n_days, dtype=np.float64)
    day_min_eq = np.zeros(n_days, dtype=np.float64)
    day_swept = np.zeros(n_days, dtype=np.float64)
    day_max_first = np.ones(n_days, dtype=bool)
    day_min_bal = np.zeros(n_days, dtype=np.float64)

    prev_close_eq = float(initial)
    for i, d in enumerate(sorted_days):
        pts = day_events[d]
        eqs = [p[0] - prev_close_eq for p in pts]
        bals = [p[1] - prev_close_eq for p in pts]
        is_chop[i] = d >= "2026-10-01"
        day_end[i] = eqs[-1]
        day_max[i] = max(0.0, max(eqs))
        day_min_eq[i] = min(0.0, min(eqs))
        day_min_bal[i] = min(0.0, min(bals))
        idx_max = int(np.argmax(eqs))
        idx_min = int(np.argmin(eqs))
        day_max_first[i] = idx_max <= idx_min
        prev_close_eq = pts[-1][0]

    blocks = (is_chop, day_end, day_max, day_min_eq, day_swept, day_max_first, day_min_bal)

    if participations:
        arr = np.array(participations, dtype=np.float64)
        part_summary = {
            "n_trades": len(participations),
            "unknown_vol_trades": unknown_vol_trades,
            "median_1m": round(float(np.median(arr)), 6),
            "p95_1m": round(float(np.quantile(arr, 0.95)), 6),
            "max_1m": round(float(np.max(arr)), 6),
            "bars_above_1pct": int(np.sum(arr > 0.01)),
            "bars_above_5pct": int(np.sum(arr > 0.05)),
        }
    else:
        part_summary = {
            "n_trades": 0,
            "unknown_vol_trades": unknown_vol_trades,
            "median_1m": 0.0,
            "p95_1m": 0.0,
            "max_1m": 0.0,
            "bars_above_1pct": 0,
            "bars_above_5pct": 0,
        }

    return {
        "peak_to_trough_dd_usd": round(max_p2t_dd, 2),
        "static_floor_dd_usd": round(static_floor_dd, 2),
        "participation": part_summary,
        "blocks": blocks,
    }


def _evaluate_one_task(
    pod: dict[str, Any],
    firm_slug: str,
    catalog_key: str,
    sim_key: str,
    program_filter: str | None,
    cost_label: str,
    cfg_mode: str,
    pod_cache_path: Path,
    raw_coins: tuple[str, ...],
    address_to_alias: dict[str, str],
    k_scaled_participation: bool = False,
) -> dict[str, Any]:
    """Run one (pod, firm, cost_mult, cfg_mode) Stage-2 kernel backtest + EV evaluation."""
    t0 = time.perf_counter()
    pod_id = str(pod["pod_id"])

    if cost_label == "1x":
        fee_rate = Decimal("0.00045")
        shortfall_bps = Decimal(30)
        spread_bps = Decimal(4)
    else:
        fee_rate = Decimal("0.00090")
        shortfall_bps = Decimal(60)
        spread_bps = Decimal(8)

    seats = tuple(SeatSpec(f"seat_{i}", Decimal(200), Decimal("0.45")) for i in range(1, pod["size"] + 1))
    if k_scaled_participation:
        score_cfg = RELAXED_PARTICIPATION_SCORE_CONFIG
    elif cfg_mode == "strict_gate":
        score_cfg = ScoreConfig()
    else:
        score_cfg = PERMISSIVE_SCORE_CONFIG

    with tempfile.TemporaryDirectory(prefix=f"{pod_id.lower()}_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        pod_store_path = tmp_dir / "facts"
        shutil.copytree(pod_cache_path, pod_store_path)
        markets = tuple(MarketSpec(coin, fee_rate=fee_rate) for coin in raw_coins)
        account_id = f"{firm_slug}-10k-{pod_id.lower()}"
        ev_sha = hashlib.sha256(f"{catalog_key}:{pod_id}:{cost_label}".encode()).hexdigest()
        rb_fields = RulebookCatalog.get(catalog_key).to_rulebook_fields(
            account_binding=account_id,
            scale=2,
            evidence_sha=ev_sha,
        )
        rulebook = Rulebook(**rb_fields)
        if Decimal(rulebook.initial) != Decimal(10_000):
            raise RuntimeError(f"UNEXPECTED_INITIAL_EQUITY:{catalog_key}:{rulebook.initial}")

        window = BacktestWindow.from_utc(OOS_START, OOS_END)
        cfg = BacktestConfig(
            account_id=account_id,
            initial_equity=Decimal(10_000),
            lookback_days=TRAIN_LOOKBACK_DAYS,
            shortfall_bps=shortfall_bps,
            quote_spread_bps=spread_bps,
            score_config=score_cfg,
            allow_undated=True,
        )
        roster_path = tmp_dir / "roster.sqlite"
        with FactStore(pod_store_path, create=False) as pod_facts:
            roster = RosterStore(roster_path, create=True)
            try:
                report = run_kernel_backtest(
                    pod_facts,
                    window=window,
                    seats=seats,
                    rulebook=rulebook,
                    markets=markets,
                    config=cfg,
                    roster=roster,
                )
            finally:
                roster.close()

            diag = _compute_oos_diagnostics(
                report,
                pod_facts=pod_facts,
                window=window,
                fee_rate=fee_rate,
                shortfall_bps=shortfall_bps,
                spread_bps=spread_bps,
            )

        quote = lookup_tier_fee(sim_key, ACCOUNT_USD, path=DEFAULT_FEE_SCHEDULE_PATH)
        fee_usd = quote.fee_usd if quote is not None else None
        refundable = quote.refundable_on_pass if quote is not None else False
        contract = pc.load_prop_contract_spec(
            firm_slug,
            account_usd=ACCOUNT_USD,
            challenge_fee_usd=fee_usd,
            program_filter=program_filter,
        )
        contract = replace(contract, fee_refunded_on_first_payout=refundable)
        rng = np.random.default_rng(42)
        ev_blended = pc.evaluate_multiphase_contract_ev(
            diag["blocks"], contract, regime="Blended", n_paths=5000, rng=rng
        )
        ev_chop = pc.evaluate_multiphase_contract_ev(
            diag["blocks"], contract, regime="75%-Chop", n_paths=5000, rng=rng
        )

    net_pnl = float(report.ending_equity - Decimal(10_000))
    decisions = report.decisions
    return {
        "pod_id": pod_id,
        "aliases": list(pod["aliases"]),
        "k_scaled_mode": k_scaled_participation,
        "firm_slug": firm_slug,
        "catalog_key": catalog_key,
        "sim_inputs_key": sim_key,
        "cost_label": cost_label,
        "cfg_mode": cfg_mode,
        "verified_fee_usd": fee_usd,
        "refundable_on_pass": refundable,
        "ending_equity": round(float(report.ending_equity), 2),
        "net_pnl_usd": round(net_pnl, 2),
        "peak_to_trough_dd_usd": diag["peak_to_trough_dd_usd"],
        "static_floor_dd_usd": diag["static_floor_dd_usd"],
        "challenge_pass_rate": float(report.challenge_pass_rate),
        "challenge_reasons": [_sanitize_text(r, address_to_alias) for r in report.challenge_reasons],
        "allocation_mode_share": dict(sorted(report.allocation_mode_share.items())),
        "decision_count": int(report.decision_count),
        "first_decision_sha": decisions[0].decision_sha if decisions else None,
        "last_decision_sha": decisions[-1].decision_sha if decisions else None,
        "participation": diag["participation"],
        "ev_blended": ev_blended,
        "ev_75pct_chop": ev_chop,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


def _run_stage2_for_pods(
    pods: list[dict[str, Any]],
    *,
    cohort_store_path: Path,
    tmp_dir: Path,
    address_to_alias: Mapping[str, str],
    max_workers: int,
    k_scaled_participation: bool,
) -> list[dict[str, Any]]:
    if not pods:
        return []
    pod_caches: dict[str, tuple[Path, tuple[str, ...]]] = {}
    for pod in pods:
        pid = str(pod["pod_id"])
        p_cache = tmp_dir / f"pod_cache_{pid}"
        p_coins = build_isolated_fact_store(
            cohort_store_path, p_cache, wallets=list(pod["addresses"])
        )
        pod_caches[pid] = (p_cache, p_coins)
    print(f"[stage2] pre-built {len(pod_caches)} isolated pod FactStores (k_scaled={k_scaled_participation})", flush=True)

    tasks = []
    for pod in pods:
        for firm_slug, catalog_key, sim_key, program_filter in FIRMS:
            for cost_label in ("1x", "2x"):
                tasks.append(
                    (pod, firm_slug, catalog_key, sim_key, program_filter, cost_label, "strict_gate")
                )

    workers = min(resolve_max_workers(max_workers), len(tasks))
    plain_a2a = {str(k): str(v) for k, v in address_to_alias.items()}
    mp_ctx = multiprocessing.get_context("spawn")
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as pool:
        futs = {
            pool.submit(
                _evaluate_one_task,
                pod,
                firm_slug,
                catalog_key,
                sim_key,
                program_filter,
                cost_label,
                cfg_mode,
                pod_caches[str(pod["pod_id"])][0],
                pod_caches[str(pod["pod_id"])][1],
                plain_a2a,
                k_scaled_participation,
            ): (pod["pod_id"], firm_slug, cost_label, cfg_mode)
            for pod, firm_slug, catalog_key, sim_key, program_filter, cost_label, cfg_mode in tasks
        }
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            print(
                f"  [done] {res['pod_id']} {res['firm_slug']} {res['cost_label']} "
                f"net=${res['net_pnl_usd']} pass={res['challenge_pass_rate']} "
                f"max_part={res['participation']['max_1m']:.4f} "
                f"ev_blend=${res['ev_blended'].get('net_ev_usd')}",
                flush=True,
            )
    results.sort(key=lambda r: (r["pod_id"], r["firm_slug"], r["cost_label"]))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Altcoin-short HL discovery & $10k prop study")
    parser.add_argument(
        "--facts-store",
        type=Path,
        default=Path("/content/facts_slim")
        if Path("/content/facts_slim").is_dir()
        else Path.home() / ".local/share/copytrade/facts",
    )
    parser.add_argument(
        "--inputs-json",
        type=Path,
        default=Path("/content/private_inputs.json")
        if Path("/content/private_inputs.json").is_file()
        else Path.home()
        / ".local/share/copytrade/research/altcoin_shorts_20261125/private_inputs.json",
    )
    parser.add_argument(
        "--reservoir-tar",
        type=Path,
        default=Path("/content/reservoir_cached.tar")
        if Path("/content/reservoir_cached.tar").is_file()
        else None,
    )
    parser.add_argument(
        "--aws-creds-json",
        type=Path,
        default=Path("/content/.aws_reservoir.json")
        if Path("/content/.aws_reservoir.json").is_file()
        else None,
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--study-stem", type=str, default=STUDY_STEM)
    parser.add_argument("--max-workers", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)

    t_start = time.monotonic()
    priv = load_private_inputs(args.inputs_json)
    addresses = priv["addresses"]
    address_to_alias = priv["address_to_alias"]

    print(
        f"[study] loaded {len(addresses)} shortlisted altcoin-short wallets, facts={args.facts_store}",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="altcoin_short_study_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        reservoir_stats = prepare_reservoir_cache(
            reservoir_tar=args.reservoir_tar,
            aws_creds_json=args.aws_creds_json,
            staging_root=tmp_dir / "reservoir_staging",
            addresses=addresses,
        )
        cohort_store_path = tmp_dir / "cohort_facts"
        t_build = time.monotonic()
        raw_coins = build_isolated_fact_store(
            args.facts_store,
            cohort_store_path,
            wallets=addresses,
            reservoir_all_dir=Path(reservoir_stats["all_dir"]),
            equity_snapshots=priv["equity_snapshots"],
        )
        print(
            f"[study] built isolated cohort FactStore ({len(raw_coins)} coins) in "
            f"{time.monotonic() - t_build:.1f}s",
            flush=True,
        )

        with FactStore(cohort_store_path, create=False) as facts:
            fills_cov = facts.coverage("fills", start_day="2026-05-01", end_day="2026-11-18")
            bars_cov = facts.coverage("bars_1m", start_day="2026-05-01", end_day="2026-11-18")
            fills_cohort_df = facts.scan("fills", wallets=list(addresses), allow_unknown_known_at=True).collect()
            bars_coins_present = set(
                facts.scan("bars_1m", allow_unknown_known_at=True).collect()["coin"].unique().to_list()
            ) if bars_cov.rows > 0 else set()
            missing_bars_coins = sorted({c.strip().upper() for c in raw_coins} - bars_coins_present)
            backfill_coverage = {
                "reservoir": {k: v for k, v in reservoir_stats.items() if k != "all_dir"},
                "cohort_fills_rows": int(fills_cohort_df.height),
                "cohort_distinct_wallets": int(fills_cohort_df["wallet"].n_unique()) if fills_cohort_df.height else 0,
                "fills_days_present": len(fills_cov.days_present),
                "bars_1m_days_present": len(bars_cov.days_present),
                "bars_1m_missing_coins": missing_bars_coins,
            }
            print(f"[coverage] {json.dumps(backfill_coverage)}", flush=True)

            # 1. Strict PIT verification (must return 0 cards)
            strict = score_universe(
                facts,
                as_of=KnownAtTs(_utc_ms(FULL_AS_OF)),
                lookback_days=FULL_LOOKBACK_DAYS,
                wallets=list(addresses),
                allow_undated=False,
            )
            if len(strict.cards) != 0 or strict.basis.get("pit") != "STRICT_KNOWN_AT":
                raise RuntimeError(
                    f"STRICT_PIT_CHECK_FAILED:cards={len(strict.cards)}:basis={strict.basis}"
                )

            # 2. Altcoin-short behavior profiles from FactStore fills + bars_1m
            profiles_full = compute_altcoin_short_profiles(
                facts,
                wallets=addresses,
                address_to_alias=address_to_alias,
                start_ms=_utc_ms(datetime(2026, 5, 1, tzinfo=UTC)),
                end_ms=_utc_ms(FULL_AS_OF),
            )
            profiles_oos = compute_altcoin_short_profiles(
                facts,
                wallets=addresses,
                address_to_alias=address_to_alias,
                start_ms=_utc_ms(OOS_START),
                end_ms=_utc_ms(OOS_END),
            )

            # 3. Stage-1 score_universe across Train, OOS, and Full (strict unscaled + diagnostic k-scaled)
            roster = RosterStore(tmp_dir / "stage1_roster.sqlite", create=True)
            try:
                train_summary, train_sc = summarize_scorecard_window(
                    facts,
                    name="train_may_aug",
                    as_of=KnownAtTs(_utc_ms(TRAIN_AS_OF)),
                    lookback_days=TRAIN_LOOKBACK_DAYS,
                    wallets=addresses,
                    address_to_alias=address_to_alias,
                    roster=roster,
                    score_config=ScoreConfig(),
                )
                oos_summary, _ = summarize_scorecard_window(
                    facts,
                    name="oos_sep_nov",
                    as_of=KnownAtTs(_utc_ms(OOS_AS_OF)),
                    lookback_days=OOS_LOOKBACK_DAYS,
                    wallets=addresses,
                    address_to_alias=address_to_alias,
                    roster=roster,
                    score_config=ScoreConfig(),
                )
                full_summary, _ = summarize_scorecard_window(
                    facts,
                    name="full_may_nov_diag",
                    as_of=KnownAtTs(_utc_ms(FULL_AS_OF)),
                    lookback_days=FULL_LOOKBACK_DAYS,
                    wallets=addresses,
                    address_to_alias=address_to_alias,
                    roster=roster,
                    score_config=ScoreConfig(),
                )

                # Diagnostic relaxed-participation windows (paired with est_k_scaled_*_1m filter)
                k_cfg = RELAXED_PARTICIPATION_SCORE_CONFIG
                train_k_summary, train_k_sc = summarize_scorecard_window(
                    facts,
                    name="train_may_aug_k_scaled_diag",
                    as_of=KnownAtTs(_utc_ms(TRAIN_AS_OF)),
                    lookback_days=TRAIN_LOOKBACK_DAYS,
                    wallets=addresses,
                    address_to_alias=address_to_alias,
                    roster=roster,
                    score_config=k_cfg,
                )
                oos_k_summary, _ = summarize_scorecard_window(
                    facts,
                    name="oos_sep_nov_k_scaled_diag",
                    as_of=KnownAtTs(_utc_ms(OOS_AS_OF)),
                    lookback_days=OOS_LOOKBACK_DAYS,
                    wallets=addresses,
                    address_to_alias=address_to_alias,
                    roster=roster,
                    score_config=k_cfg,
                )
                full_k_summary, _ = summarize_scorecard_window(
                    facts,
                    name="full_may_nov_k_scaled_diag",
                    as_of=KnownAtTs(_utc_ms(FULL_AS_OF)),
                    lookback_days=FULL_LOOKBACK_DAYS,
                    wallets=addresses,
                    address_to_alias=address_to_alias,
                    roster=roster,
                    score_config=k_cfg,
                )
            finally:
                roster.close()

        print(
            f"[stage1-strict] train eligible={train_summary['eligible_count']}, "
            f"oos eligible={oos_summary['eligible_count']}, full eligible={full_summary['eligible_count']}",
            flush=True,
        )
        print(
            f"[stage1-kscaled-diag] train eligible={train_k_summary['eligible_count']}, "
            f"oos eligible={oos_k_summary['eligible_count']}, full eligible={full_k_summary['eligible_count']}",
            flush=True,
        )

        # Restrict pod formation to verified altcoin-short specialists (>=50% altcoin & >=50% short notional over full/train)
        verified_alt_short_addrs = {
            priv["alias_to_address"][alias]
            for alias, prof in profiles_full.items()
            if alias in priv["alias_to_address"]
            and prof["altcoin_notional_frac"] >= 0.50
            and prof["short_notional_frac"] >= 0.50
        }

        train_k_audit = {row.wallet: row for row in train_k_sc.audit}
        k_scaled_alt_short_addrs = {
            c.wallet
            for c in train_k_sc.cards
            if c.eligible
            and c.wallet in verified_alt_short_addrs
            and c.wallet in train_k_audit
            and float(c.suggested_k) * float(train_k_audit[c.wallet].p95_participation_1m) <= 0.01
            and float(c.suggested_k) * float(train_k_audit[c.wallet].max_participation_1m) <= 0.05
        }

        strict_pods = form_train_pods(
            train_sc,
            address_to_alias=address_to_alias,
            allowed_wallets=verified_alt_short_addrs,
            pod_prefix="ASPOD",
        )
        diag_k_pods = form_train_pods(
            train_k_sc,
            address_to_alias=address_to_alias,
            allowed_wallets=k_scaled_alt_short_addrs,
            pod_prefix="ASKPOD",
        )
        print(
            f"[pods] strict={[p['pod_id'] for p in strict_pods]} diag_k={[p['pod_id'] for p in diag_k_pods]}",
            flush=True,
        )

        strict_stage2 = _run_stage2_for_pods(
            strict_pods,
            cohort_store_path=cohort_store_path,
            tmp_dir=tmp_dir,
            address_to_alias=address_to_alias,
            max_workers=args.max_workers,
            k_scaled_participation=False,
        )
        diag_k_stage2 = _run_stage2_for_pods(
            diag_k_pods,
            cohort_store_path=cohort_store_path,
            tmp_dir=tmp_dir,
            address_to_alias=address_to_alias,
            max_workers=args.max_workers,
            k_scaled_participation=True,
        )

    passing_strict_2x = [
        r
        for r in strict_stage2
        if r["cost_label"] == "2x"
        and r["net_pnl_usd"] > 0.0
        and r["challenge_pass_rate"] == 1.0
        and r["participation"]["max_1m"] <= 0.05
        and r["participation"]["p95_1m"] <= 0.01
        and (
            r["allocation_mode_share"].get("ACTIVE", 0.0)
            + r["allocation_mode_share"].get("PARTIAL", 0.0)
        )
        > 0.0
        and (r["ev_blended"].get("net_ev_usd") or -9999.0) > 0.0
    ]
    verdict = "PAPER_ELIGIBLE" if passing_strict_2x else "NO_DEPLOYABLE_POD"

    output = {
        "schema": "altcoin_short_discovery_study_v1",
        "provenance": "RETROSPECTIVE_RESEARCH_ONLY",
        "verdict": verdict,
        "elapsed_sec": round(time.monotonic() - t_start, 2),
        "cohort_coins": list(raw_coins),
        "backfill_coverage": backfill_coverage,
        "gate2_summary": {
            k: v for k, v in priv["gate2_summary"].items() if k != "candidates"
        },
        "gate2_candidates": priv["gate2_summary"].get("candidates", []),
        "strict_pit_check": {
            "cards_count": len(strict.cards),
            "basis": dict(sorted(strict.basis.items())),
        },
        "altcoin_short_profiles_full": profiles_full,
        "altcoin_short_profiles_oos": profiles_oos,
        "stage1_strict": {
            "train_may_aug": train_summary,
            "oos_sep_nov": oos_summary,
            "full_may_nov_diag": full_summary,
        },
        "stage1_k_scaled_diag": {
            "train_may_aug": train_k_summary,
            "oos_sep_nov": oos_k_summary,
            "full_may_nov_diag": full_k_summary,
        },
        "strict_train_pods": [{k: v for k, v in p.items() if k != "addresses"} for p in strict_pods],
        "diag_k_train_pods": [{k: v for k, v in p.items() if k != "addresses"} for p in diag_k_pods],
        "strict_stage2_runs": strict_stage2,
        "diag_k_stage2_runs": diag_k_stage2,
        "passing_strict_2x_runs": [
            {
                "pod_id": r["pod_id"],
                "firm_slug": r["firm_slug"],
                "net_pnl_usd": r["net_pnl_usd"],
                "ev_blended_net_usd": r["ev_blended"].get("net_ev_usd"),
                "max_1m_part": r["participation"]["max_1m"],
            }
            for r in passing_strict_2x
        ],
    }

    payload_text = json.dumps(output, indent=2)
    _assert_sanitized_text("study_output_json", payload_text)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.study_stem}__results.json"
    out_path.write_text(payload_text, encoding="utf-8")
    print(f"[study] wrote sanitized results -> {out_path} (verdict={verdict})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
