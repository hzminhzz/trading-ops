#!/usr/bin/env python3
"""Stage-2 Hybrid Pod Backtest: Liquid-Major/L12 Anchors + Altcoin-Short Specialists.

Evaluates 45 deduplicated wallets (31 proven/candidate liquid-major & L12 anchors
plus 14 verified altcoin-short specialists) on Google Colab across May 1 - Nov 18, 2026:
- Reservoir S3 backfill (2026-05-01..2026-11-18) merged into an isolated FactStore.
- Strict PIT verification (`allow_undated=False` -> 0 cards) and retrospective
  Stage-1 `score_universe` on `train_may_aug` (123d), `oos_sep_nov` (79d), and
  `full_may_nov_diag` (202d) under strict `ScoreConfig()` and
  `RELAXED_PARTICIPATION_SCORE_CONFIG` (with k-scaled `<=1%/5%` audit).
- Train-frozen 3-4 seat pod construction comparing:
    * ANCHOR-STRICT3: strict Train-eligible anchors only (no alt-shorts)
    * KANCHOR-DIV3: k-scaled (<=1%/5%) Train-eligible anchors only (benchmark)
    * HYB-KSCALE: k-scaled (<=1%/5%) Train-eligible anchors + k-scaled alt-shorts
    * HYB-RELAXED: k-scaled anchors + RELAXED-eligible alt-shorts (e.g. AS#3276acd5)
    * HYB-UPTREND-DIAG: optional diagnostic pairing anchors + compact uptrend alt-shorts
- Stage-2 strict OOS (`2026-09-01..2026-11-19`) `run_kernel_backtest` across
  `propr:classic_1step:phase_1` ($110), `vanta-trading:classic_1step_3pct:phase_1` ($127),
  and `hyrotrader:two_step:phase_1` ($119 refundable) at 1x and 2x execution costs,
  recording `peak_to_trough_dd_usd`, `static_floor_dd_usd`, `bars_1m` participation,
  `golden.decision_sha`, short/altcoin intent breakdown, and `evaluate_multiphase_contract_ev`
  (Blended & 75%-Chop, n_paths=5000, Wilson 95% CIs at `N_eff`).

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
    summarize_backtest_diagnostics,
)
from trading_ops.research.scoring import euler_marginal_cdar_shares as _euler_marginal_cdar_shares

HEX_ADDRESS_RE = re.compile(r"(?i)0x[0-9a-f]{4,}")
STUDY_STEM = "2026-11-25__multi__10k__hybrid-altcoin-short-pod-study"
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
    return int(dt.timestamp() * 1000)


def _is_altcoin(coin: str) -> bool:
    c = str(coin).strip().upper()
    return bool(c) and c not in CORE_4_MAJORS and c not in NON_ALT_COMMODITIES


def _assert_sanitized_text(label: str, text: str) -> None:
    hits = HEX_ADDRESS_RE.findall(text)
    if hits:
        raise RuntimeError(f"HEX_LEAK_DETECTED:{label}:samples={hits[:3]}")


def _sanitize_text(text: str, address_to_alias: Mapping[str, str]) -> str:
    lowered_map = {str(k).lower(): str(v) for k, v in address_to_alias.items()}

    def _replace(match: re.Match[str]) -> str:
        tok = match.group(0).lower()
        return lowered_map.get(tok, f"W#{hashlib.sha256(tok.encode()).hexdigest()[:8]}")

    out = HEX_ADDRESS_RE.sub(_replace, text)
    _assert_sanitized_text("sanitized_text", out)
    return out


def _fmt(val: Decimal | float | None, digits: int = 4) -> float | None:
    if val is None:
        return None
    return round(float(val), digits)


def resolve_max_workers(requested: int) -> int:
    if requested > 0:
        return requested
    cpus = os.cpu_count() or 2
    return max(1, cpus - 1)


def load_private_inputs(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    addresses = [str(a).strip().lower() for a in raw["addresses"] if str(a).strip()]
    a2a = {str(k).strip().lower(): str(v) for k, v in raw["address_to_alias"].items()}
    al2a = {str(k): str(v).strip().lower() for k, v in raw["alias_to_address"].items()}
    sleeve_by_alias = {str(k): str(v) for k, v in raw.get("sleeve_by_alias", {}).items()}
    return {
        "addresses": addresses,
        "address_to_alias": a2a,
        "alias_to_address": al2a,
        "sleeve_by_alias": sleeve_by_alias,
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
    max_s3_workers: int = 16,
) -> dict[str, Any]:
    """Ensure Reservoir fills parquet days exist locally via tar extraction or S3 scan."""
    all_dir = staging_root / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    if reservoir_tar is not None and reservoir_tar.is_file():
        with tarfile.open(reservoir_tar, "r:*") as tf:
            tf.extractall(staging_root)
        extracted_days = sorted(p.parent.name.replace("date=", "") for p in all_dir.glob("date=*/fills.parquet"))
        return {
            "source": "tar",
            "days_present": len(extracted_days),
            "missing_days": [],
            "total_rows": None,
            "all_dir": str(all_dir),
        }

    if aws_creds_json is None or not aws_creds_json.is_file():
        return {
            "source": "facts_mirror_only",
            "days_present": 0,
            "missing_days": [],
            "total_rows": 0,
            "all_dir": str(all_dir),
        }

    creds = json.loads(aws_creds_json.read_text(encoding="utf-8"))
    key_id = str(creds["AWS_ACCESS_KEY_ID"]).strip()
    secret = str(creds["AWS_SECRET_ACCESS_KEY"]).strip()
    token = str(creds.get("AWS_SESSION_TOKEN", "")).strip()
    if not key_id or not secret:
        raise RuntimeError("EMPTY_AWS_RESERVOIR_CREDS")

    import duckdb

    init_con = duckdb.connect()
    init_con.execute("INSTALL httpfs; LOAD httpfs;")
    init_con.close()

    cur_d = date.fromisoformat(start_day)
    end_d = date.fromisoformat(end_day)
    days: list[str] = []
    while cur_d <= end_d:
        days.append(cur_d.isoformat())
        cur_d += timedelta(days=1)

    addr_sql_list = ", ".join("'" + a.lower() + "'" for a in addresses)
    tls = threading.local()

    def _get_con() -> duckdb.DuckDBPyConnection:
        con = getattr(tls, "con", None)
        if con is not None:
            return con
        con = duckdb.connect()
        con.execute("LOAD httpfs;")
        con.execute("SET threads=1;")
        con.execute("SET enable_object_cache=true;")
        con.execute("SET s3_region='ap-northeast-1';")
        con.execute("SET s3_requester_pays=true;")
        con.execute("SET enable_progress_bar=false;")
        escaped_key = key_id.replace("'", "''")
        escaped_sec = secret.replace("'", "''")
        if token:
            escaped_tok = token.replace("'", "''")
            con.execute(
                f"CREATE SECRET (TYPE s3, KEY_ID '{escaped_key}', SECRET '{escaped_sec}', "
                f"SESSION_TOKEN '{escaped_tok}', REGION 'ap-northeast-1');"
            )
        else:
            con.execute(
                f"CREATE SECRET (TYPE s3, KEY_ID '{escaped_key}', SECRET '{escaped_sec}', "
                f"REGION 'ap-northeast-1');"
            )
        tls.con = con
        return con

    def _fetch_one(day_str: str) -> tuple[str, int, bool]:
        out_day_dir = all_dir / f"date={day_str}"
        out_day_dir.mkdir(parents=True, exist_ok=True)
        out_pq = out_day_dir / "fills.parquet"
        if out_pq.is_file() and out_pq.stat().st_size > 0:
            return day_str, int(pl.read_parquet(out_pq).height), True

        s3_path = f"s3://hydromancer-reservoir/by_dex/hyperliquid/fills/perp/all/date={day_str}/fills.parquet"
        tmp_pq = out_day_dir / "fills.parquet.tmp"
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
          FROM read_parquet('{s3_path}') f
          WHERE lower(f.address) IN ({addr_sql_list})
        ) TO '{tmp_pq}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        con = _get_con()
        try:
            con.execute(sql)
        except Exception as exc:
            tmp_pq.unlink(missing_ok=True)
            msg = str(exc).lower()
            if "no files found" in msg or "404" in msg or "nosuchkey" in msg or "does not exist" in msg:
                return day_str, 0, False
            raise
        n_rows = int(pl.read_parquet(tmp_pq).height)
        if n_rows > 0:
            tmp_pq.replace(out_pq)
        else:
            tmp_pq.unlink(missing_ok=True)
        return day_str, n_rows, True

    missing_days: list[str] = []
    total_rows = 0
    done = 0
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=max_s3_workers) as pool:
        futs = {pool.submit(_fetch_one, d): d for d in days}
        for fut in as_completed(futs):
            d_str, n_rows, ok = fut.result()
            done += 1
            total_rows += n_rows
            if not ok:
                missing_days.append(d_str)
            if done % 50 == 0 or done == len(days):
                print(
                    f"[reservoir-s3] {done}/{len(days)} days scanned "
                    f"(rows={total_rows:,}, missing={len(missing_days)}, elapsed={time.monotonic() - t0:.1f}s)",
                    flush=True,
                )

    return {
        "source": "s3_duckdb_live",
        "days_present": len(days) - len(missing_days),
        "missing_days": sorted(missing_days),
        "total_rows": total_rows,
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
    rel_prefix: str = "hybrid",
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
                rel_prefix="hybrid",
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
                source_id="hybrid-equity",
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

        sorted_norm_coins = sorted(norm_coins | {"BTC", "ETH", "SOL", "HYPE"})
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
                rel_prefix="hybrid",
            )
    return tuple(sorted(raw_coins))


def compute_regime_and_wallet_profiles(
    facts: FactStore,
    *,
    wallets: Sequence[str],
    address_to_alias: Mapping[str, str],
    sleeve_by_alias: Mapping[str, str],
    start_ms: int,
    end_ms: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Compute market regimes from bars_1m and daily MTM PnL / altcoin-short profiles per wallet."""
    bars_lf = facts.scan("bars_1m", allow_unknown_known_at=True).frame.filter(
        (pl.col("event_ts_ms") >= start_ms) & (pl.col("event_ts_ms") < end_ms)
    )
    daily_closes_df = (
        bars_lf.with_columns(
            pl.col("coin").str.strip_chars().str.to_uppercase().alias("coin_up"),
            pl.from_epoch("event_ts_ms", time_unit="ms").dt.strftime("%Y-%m-%d").alias("day"),
        )
        .sort("event_ts_ms")
        .group_by(["coin_up", "day"])
        .agg(pl.col("close").cast(pl.Float64).last().alias("close"))
        .sort(["coin_up", "day"])
        .collect()
    )

    day_close_map: dict[tuple[str, str], float] = {}
    coin_days_seen: dict[str, int] = defaultdict(int)
    for coin_up, day_str, close_px in daily_closes_df.iter_rows():
        if close_px is not None and close_px > 0.0:
            day_close_map[(coin_up, day_str)] = float(close_px)
            coin_days_seen[coin_up] += 1

    all_days = sorted({d for _, d in day_close_map})
    min_alt_days = 150 if len(all_days) >= 160 else max(1, len(all_days) // 2)
    index_alts = sorted(c for c, cnt in coin_days_seen.items() if _is_altcoin(c) and cnt >= min_alt_days)
    alt_day_rets: dict[str, float] = {}
    btc_day_rets: dict[str, float] = {}
    for i in range(1, len(all_days)):
        d_prev, d_cur = all_days[i - 1], all_days[i]
        rets = []
        for c in index_alts:
            p0 = day_close_map.get((c, d_prev))
            p1 = day_close_map.get((c, d_cur))
            if p0 and p1 and p0 > 0:
                rets.append(p1 / p0 - 1.0)
        if rets:
            alt_day_rets[d_cur] = float(np.mean(rets))
        b0, b1 = day_close_map.get(("BTC", d_prev)), day_close_map.get(("BTC", d_cur))
        if b0 and b1 and b0 > 0:
            btc_day_rets[d_cur] = b1 / b0 - 1.0

    months = sorted({d[:7] for d in all_days})
    monthly_regime: dict[str, dict[str, Any]] = {}
    uptrend_months: set[str] = set()
    uptrend_days: set[str] = {d for d, r in alt_day_rets.items() if r > 0.0}
    for m in months:
        m_days = [d for d in all_days if d.startswith(m) and d in alt_day_rets]
        comp_alt = 1.0
        for d in m_days:
            comp_alt *= 1.0 + alt_day_rets[d]
        alt_ret = comp_alt - 1.0
        comp_btc = 1.0
        for d in m_days:
            if d in btc_day_rets:
                comp_btc *= 1.0 + btc_day_rets[d]
        btc_ret = comp_btc - 1.0
        is_up = alt_ret > 0.0
        if is_up:
            uptrend_months.add(m)
        monthly_regime[m] = {
            "alt_index_ret_pct": round(alt_ret * 100.0, 2),
            "btc_ret_pct": round(btc_ret * 100.0, 2),
            "regime": "UPTREND" if is_up else "DOWNTREND",
        }

    fills_df = (
        facts.scan("fills", wallets=list(wallets), allow_unknown_known_at=True)
        .frame.filter((pl.col("event_ts_ms") >= start_ms) & (pl.col("event_ts_ms") < end_ms))
        .with_columns(
            pl.col("coin").str.strip_chars().str.to_uppercase().alias("coin_up"),
            pl.col("price").cast(pl.Float64).alias("px_f"),
            pl.col("size").cast(pl.Float64).alias("sz_f"),
            pl.col("fee").cast(pl.Float64).fill_null(0.0).alias("fee_f"),
            ((pl.col("event_ts_ms") // 60_000) * 60_000 + 300_000).alias("target_5m_ms"),
            pl.from_epoch("event_ts_ms", time_unit="ms").dt.strftime("%Y-%m-%d").alias("day"),
        )
        .sort(["wallet", "event_ts_ms", "arrival_seq"])
        .collect()
    )

    bars_5m = (
        bars_lf.select(
            pl.col("coin").str.strip_chars().str.to_uppercase().alias("coin_up"),
            pl.col("event_ts_ms").cast(pl.Int64).alias("target_5m_ms"),
            pl.col("close").cast(pl.Float64).alias("close_5m"),
        )
        .unique(subset=["coin_up", "target_5m_ms"], keep="last")
        .collect()
    )
    fills_df = fills_df.join(bars_5m, on=["coin_up", "target_5m_ms"], how="left")

    regime_summary = {
        "index_altcoin_count": len(index_alts),
        "days_total": len(all_days),
        "uptrend_days_count": len(uptrend_days),
        "downtrend_days_count": max(0, len(all_days) - len(uptrend_days)),
        "uptrend_months": sorted(uptrend_months),
        "downtrend_months": sorted(set(months) - uptrend_months),
        "monthly_regimes": monthly_regime,
    }

    profiles: dict[str, dict[str, Any]] = {}
    if fills_df.height == 0:
        return regime_summary, profiles

    for wallet_key, w_df in fills_df.group_by("wallet", maintain_order=True):
        addr = str(wallet_key[0] if isinstance(wallet_key, tuple) else wallet_key).lower()
        alias = address_to_alias.get(addr, addr)
        sleeve = sleeve_by_alias.get(alias, "UNKNOWN")

        total_ntl = 0.0
        alt_ntl = 0.0
        short_ntl = 0.0
        long_ntl = 0.0
        alt_short_ntl = 0.0
        short_fills = 0
        total_fills = 0
        total_fees = 0.0
        coin_ntl: dict[str, float] = defaultdict(float)

        m5_stats = {
            "full": [0.0, 0.0, 0],
            "train": [0.0, 0.0, 0],
            "oos": [0.0, 0.0, 0],
        }

        pos_qty: dict[str, float] = {}
        entry_px: dict[str, float] = {}
        last_known_px: dict[str, float] = {}

        cum_short_real = 0.0
        cum_long_real = 0.0
        cum_alt_short_real = 0.0
        cum_fees = 0.0

        prev_short_tot = 0.0
        prev_long_tot = 0.0
        prev_alt_short_tot = 0.0
        prev_fees = 0.0

        daily_pnl: dict[str, tuple[float, float, float, float]] = {}
        fills_by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in w_df.iter_rows(named=True):
            fills_by_day[row["day"]].append(row)

        for day_str in all_days:
            for row in fills_by_day.get(day_str, ()):
                coin = str(row["coin_up"] or "")
                side_raw = str(row["side"] or "").strip().lower()
                dir_raw = str(row["direction"] or "").strip()
                px = float(row["px_f"] or 0.0)
                sz = float(row["sz_f"] or 0.0)
                fee = float(row["fee_f"] or 0.0)
                close_5m = row["close_5m"]

                if px <= 0.0 or sz <= 0.0:
                    continue
                sign = 1.0 if side_raw in ("b", "buy", "bid", "long") else (-1.0 if side_raw in ("a", "s", "sell", "ask", "short") else 0.0)
                if sign == 0.0:
                    continue

                ntl = px * sz
                total_fills += 1
                total_ntl += ntl
                total_fees += fee
                cum_fees += fee
                coin_ntl[coin] += ntl
                last_known_px[coin] = px
                is_alt = _is_altcoin(coin)
                if is_alt:
                    alt_ntl += ntl

                cur = pos_qty.get(coin, 0.0)
                is_short_fill = ("Short" in dir_raw) if dir_raw else (cur < 0 or (cur == 0 and sign < 0))
                if is_short_fill:
                    short_fills += 1
                    short_ntl += ntl
                    if is_alt:
                        alt_short_ntl += ntl
                else:
                    long_ntl += ntl

                if is_alt and sign < 0 and close_5m is not None and close_5m > 0.0:
                    ret_bps_w = ntl * (-1.0) * (float(close_5m) - px) / px
                    bucket = "train" if day_str < "2026-09-01" else "oos"
                    for key in ("full", bucket):
                        m5_stats[key][0] += ret_bps_w
                        m5_stats[key][1] += ntl
                        m5_stats[key][2] += 1

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
                        cum_short_real += pnl
                        if is_alt:
                            cum_alt_short_real += pnl
                    else:
                        cum_long_real += pnl
                    rem = cur + delta
                    if abs(rem) < 1e-12:
                        pos_qty.pop(coin, None)
                        entry_px.pop(coin, None)
                    elif (cur > 0 and rem < 0) or (cur < 0 and rem > 0):
                        pos_qty[coin] = rem
                        entry_px[coin] = px
                    else:
                        pos_qty[coin] = rem

            short_unr = 0.0
            long_unr = 0.0
            alt_short_unr = 0.0
            for coin, qty in pos_qty.items():
                mark_px = day_close_map.get((coin, day_str)) or last_known_px.get(coin)
                if not mark_px or mark_px <= 0.0:
                    continue
                upnl = (mark_px - entry_px[coin]) * qty
                if qty < 0:
                    short_unr += upnl
                    if _is_altcoin(coin):
                        alt_short_unr += upnl
                else:
                    long_unr += upnl

            cur_sh = cum_short_real + short_unr
            cur_lg = cum_long_real + long_unr
            cur_alt_sh = cum_alt_short_real + alt_short_unr

            d_fee = cum_fees - prev_fees
            d_sh = cur_sh - prev_short_tot
            d_lg = cur_lg - prev_long_tot
            d_alt_sh = cur_alt_sh - prev_alt_short_tot
            d_tot = d_sh + d_lg - d_fee

            daily_pnl[day_str] = (d_tot, d_sh, d_lg, d_alt_sh)
            prev_short_tot = cur_sh
            prev_long_tot = cur_lg
            prev_alt_short_tot = cur_alt_sh
            prev_fees = cum_fees

        monthly_tot: dict[str, float] = {m: 0.0 for m in months}
        monthly_alt_sh: dict[str, float] = {m: 0.0 for m in months}
        monthly_fills: dict[str, int] = {m: 0 for m in months}
        for d_str, flist in fills_by_day.items():
            m = d_str[:7]
            if m in monthly_fills:
                monthly_fills[m] += len(flist)

        train_tot = train_sh = train_alt_sh = 0.0
        oos_tot = oos_sh = oos_alt_sh = 0.0
        up_m_tot = up_m_alt_sh = 0.0
        up_d_tot = up_d_alt_sh = 0.0
        dn_m_tot = dn_m_alt_sh = 0.0

        for d_str, (d_tot, d_sh, _d_lg, d_alt_sh) in daily_pnl.items():
            m = d_str[:7]
            monthly_tot[m] = monthly_tot.get(m, 0.0) + d_tot
            monthly_alt_sh[m] = monthly_alt_sh.get(m, 0.0) + d_alt_sh
            if d_str < "2026-09-01":
                train_tot += d_tot
                train_sh += d_sh
                train_alt_sh += d_alt_sh
            else:
                oos_tot += d_tot
                oos_sh += d_sh
                oos_alt_sh += d_alt_sh
            if m in uptrend_months:
                up_m_tot += d_tot
                up_m_alt_sh += d_alt_sh
            else:
                dn_m_tot += d_tot
                dn_m_alt_sh += d_alt_sh
            if d_str in uptrend_days:
                up_d_tot += d_tot
                up_d_alt_sh += d_alt_sh

        active_months = [m for m in months if monthly_fills.get(m, 0) > 0 or abs(monthly_tot.get(m, 0.0)) > 1.0]
        pos_months = sum(1 for m in active_months if monthly_tot.get(m, 0.0) > 0.0)
        pos_alt_sh_months = sum(1 for m in active_months if monthly_alt_sh.get(m, 0.0) > 0.0)
        all_regime_expert = (
            train_tot > 0.0 and oos_tot > 0.0 and train_sh > 0.0 and oos_sh > 0.0 and up_m_tot > 0.0
        )

        top_coins = sorted(coin_ntl.items(), key=lambda kv: kv[1], reverse=True)[:5]
        profiles[alias] = {
            "alias": alias,
            "sleeve": sleeve,
            "total_fills": total_fills,
            "distinct_coins": len(coin_ntl),
            "total_notional_usd": round(total_ntl, 2),
            "altcoin_notional_frac": round(alt_ntl / total_ntl, 4) if total_ntl > 0 else 0.0,
            "short_notional_frac": round(short_ntl / max(1e-9, short_ntl + long_ntl), 4)
            if (short_ntl + long_ntl) > 0
            else 0.0,
            "short_fill_frac": round(short_fills / total_fills, 4) if total_fills > 0 else 0.0,
            "alt_short_notional_frac": round(alt_short_ntl / total_ntl, 4) if total_ntl > 0 else 0.0,
            "net_total_pnl_usd": round(prev_short_tot + prev_long_tot - prev_fees, 2),
            "short_net_pnl_usd": round(prev_short_tot, 2),
            "long_net_pnl_usd": round(prev_long_tot, 2),
            "alt_short_net_pnl_usd": round(prev_alt_short_tot, 2),
            "total_fees_usd": round(total_fees, 2),
            "train_total_pnl_usd": round(train_tot, 2),
            "train_short_pnl_usd": round(train_sh, 2),
            "train_alt_short_pnl_usd": round(train_alt_sh, 2),
            "oos_total_pnl_usd": round(oos_tot, 2),
            "oos_short_pnl_usd": round(oos_sh, 2),
            "oos_alt_short_pnl_usd": round(oos_alt_sh, 2),
            "uptrend_months_total_pnl_usd": round(up_m_tot, 2),
            "uptrend_months_alt_short_pnl_usd": round(up_m_alt_sh, 2),
            "uptrend_days_total_pnl_usd": round(up_d_tot, 2),
            "uptrend_days_alt_short_pnl_usd": round(up_d_alt_sh, 2),
            "downtrend_months_total_pnl_usd": round(dn_m_tot, 2),
            "downtrend_months_alt_short_pnl_usd": round(dn_m_alt_sh, 2),
            "active_months": len(active_months),
            "profitable_months": pos_months,
            "profitable_alt_short_months": pos_alt_sh_months,
            "all_regime_expert": all_regime_expert,
            "monthly_total_pnl_usd": {m: round(monthly_tot[m], 2) for m in months},
            "monthly_alt_short_pnl_usd": {m: round(monthly_alt_sh[m], 2) for m in months},
            "alt_short_entry_markout_5m_bps_full": round((m5_stats["full"][0] / m5_stats["full"][1]) * 1e4, 4)
            if m5_stats["full"][1] > 0
            else None,
            "alt_short_entry_markout_5m_bps_train": round((m5_stats["train"][0] / m5_stats["train"][1]) * 1e4, 4)
            if m5_stats["train"][1] > 0
            else None,
            "alt_short_entry_markout_5m_bps_oos": round((m5_stats["oos"][0] / m5_stats["oos"][1]) * 1e4, 4)
            if m5_stats["oos"][1] > 0
            else None,
            "top_coins": [
                {"coin": c, "share": round(v / total_ntl, 4)} for c, v in top_coins
            ]
            if total_ntl > 0
            else [],
        }

    return regime_summary, profiles


def summarize_scorecard_window(
    facts: FactStore,
    *,
    name: str,
    as_of: KnownAtTs,
    lookback_days: int,
    wallets: Sequence[str],
    address_to_alias: Mapping[str, str],
    roster: RosterStore | None = None,
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
    recorded_shas = scorecard.record(roster) if roster is not None else ()
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
                        min(1.0, float(card.suggested_k)) * float(audit.p95_participation_1m), 6
                    ),
                    "est_k_scaled_max_1m": round(
                        min(1.0, float(card.suggested_k)) * float(audit.max_participation_1m), 6
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

    eligible_aliases = sorted(r["alias"] for r in wallet_rows if r["eligible"])
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


def _score_pod_metrics(
    addrs: list[str],
    *,
    returns_map: Mapping[str, Any],
    eligible_cards: Sequence[Any],
    corr: np.ndarray,
    idx_by_addr: Mapping[str, int],
    cdar_alpha: float,
    max_euler_share: float,
) -> tuple[bool, list[float], float]:
    sub_mat = np.array([returns_map[a] for a in addrs], dtype=np.float64)
    sub_k = np.array(
        [float(next(c.suggested_k for c in eligible_cards if c.wallet == a)) for a in addrs],
        dtype=np.float64,
    )
    shares = _euler_marginal_cdar_shares(sub_mat, sub_k, cdar_alpha)
    shares_f = [round(float(s), 6) for s in shares]
    pairs = [
        float(corr[idx_by_addr[addrs[i]], idx_by_addr[addrs[j]]])
        for i in range(len(addrs))
        for j in range(i + 1, len(addrs))
    ]
    avg_corr = float(np.mean(pairs)) if pairs else 0.0
    ok = all(max(0.0, float(s)) <= max_euler_share for s in shares)
    return ok, shares_f, round(avg_corr, 4)


def _greedy_low_corr_pod(
    seed_addrs: list[str],
    candidate_pool: list[str],
    *,
    size: int,
    returns_map: Mapping[str, Any],
    eligible_cards: Sequence[Any],
    audit_map: Mapping[str, Any],
    corr: np.ndarray,
    idx_by_addr: Mapping[str, int],
    cdar_alpha: float,
    max_euler_share: float,
) -> list[str]:
    chosen = list(seed_addrs)
    used_clusters = {next(c.cluster_id for c in eligible_cards if c.wallet == s) for s in chosen}
    remaining = [a for a in candidate_pool if a not in set(chosen)]
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
            ok, _, _ = _score_pod_metrics(
                chosen + [cand],
                returns_map=returns_map,
                eligible_cards=eligible_cards,
                corr=corr,
                idx_by_addr=idx_by_addr,
                cdar_alpha=cdar_alpha,
                max_euler_share=max_euler_share,
            )
            if ok or len(chosen) + 1 < size:
                picked = cand
                break
        if picked is None:
            picked = remaining[0]
        chosen.append(picked)
        used_clusters.add(next(c.cluster_id for c in eligible_cards if c.wallet == picked))
        remaining.remove(picked)
    return chosen


def _make_pod_record(
    pod_id: str,
    addrs: list[str],
    *,
    sleeve_desc: str,
    address_to_alias: Mapping[str, str],
    sleeve_by_alias: Mapping[str, str],
    eligible_cards: Sequence[Any],
    returns_map: Mapping[str, Any],
    corr: np.ndarray,
    idx_by_addr: Mapping[str, int],
    cdar_alpha: float,
    max_euler_share: float,
) -> dict[str, Any] | None:
    if len(addrs) < 3:
        return None
    ok, shares, avg_corr = _score_pod_metrics(
        addrs,
        returns_map=returns_map,
        eligible_cards=eligible_cards,
        corr=corr,
        idx_by_addr=idx_by_addr,
        cdar_alpha=cdar_alpha,
        max_euler_share=max_euler_share,
    )
    if not ok:
        return None
    aliases = [address_to_alias[a] for a in addrs]
    return {
        "pod_id": pod_id,
        "sleeve_design": sleeve_desc,
        "size": len(addrs),
        "aliases": aliases,
        "seat_sleeves": [sleeve_by_alias.get(al, "UNKNOWN") for al in aliases],
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


def form_hybrid_study_pods(
    *,
    train_strict_sc: Any,
    train_relaxed_sc: Any,
    train_perm_sc: Any,
    address_to_alias: Mapping[str, str],
    sleeve_by_alias: Mapping[str, str],
    anchor_addrs: set[str],
    alt_short_addrs: set[str],
    k_scaled_addrs: set[str],
    compact_uptrend_as_addrs: set[str],
    max_euler_share: float = 0.45,
) -> dict[str, list[dict[str, Any]]]:
    """Form crisp Train-frozen baseline vs hybrid pods across strict, k-scaled, and diagnostic tiers."""

    def _corr_ctx(sc: Any, pool_addrs: set[str]) -> tuple[list[Any], dict[str, Any], Any, dict[str, int], np.ndarray]:
        cards = [c for c in sc.cards if c.wallet in pool_addrs]
        audit_map = {row.wallet: row for row in sc.audit}
        returns_map = sc._returns_by_wallet
        addrs = [c.wallet for c in cards if c.wallet in returns_map]
        cards = [c for c in cards if c.wallet in set(addrs)]
        mat = np.array([returns_map[a] for a in addrs], dtype=np.float64) if addrs else np.zeros((0, 1))
        if len(addrs) > 0:
            centred = mat - mat.mean(axis=1, keepdims=True)
            norms = np.sqrt(np.sum(centred**2, axis=1, keepdims=True))
            norms[norms == 0.0] = 1.0
            unit = centred / norms
            corr = np.clip(unit @ unit.T, -1.0, 1.0)
        else:
            corr = np.zeros((0, 0))
        idx_by_addr = {a: i for i, a in enumerate(addrs)}
        return cards, audit_map, returns_map, idx_by_addr, corr

    # 1. Strict Gate Baseline & Hybrid (ScoreConfig())
    strict_pods: list[dict[str, Any]] = []
    strict_elig = {c.wallet for c in train_strict_sc.cards if c.eligible}
    s_anchor_pool = sorted(strict_elig & anchor_addrs)
    s_as_pool = sorted(strict_elig & alt_short_addrs)
    if len(s_anchor_pool) >= 3:
        cards, audit_map, rmap, idx_map, corr = _corr_ctx(train_strict_sc, strict_elig)
        s_anchor_sorted = sorted(s_anchor_pool, key=lambda a: float(audit_map[a].markout_5m_bps), reverse=True)
        picked = _greedy_low_corr_pod(
            [s_anchor_sorted[0]],
            s_anchor_sorted,
            size=min(4, len(s_anchor_sorted)),
            returns_map=rmap,
            eligible_cards=cards,
            audit_map=audit_map,
            corr=corr,
            idx_by_addr=idx_map,
            cdar_alpha=train_strict_sc.config.cdar_alpha,
            max_euler_share=max_euler_share,
        )
        rec = _make_pod_record(
            "ANCHOR-STRICT",
            picked,
            sleeve_desc="strict_anchors_only",
            address_to_alias=address_to_alias,
            sleeve_by_alias=sleeve_by_alias,
            eligible_cards=cards,
            returns_map=rmap,
            corr=corr,
            idx_by_addr=idx_map,
            cdar_alpha=train_strict_sc.config.cdar_alpha,
            max_euler_share=max_euler_share,
        )
        if rec is not None:
            strict_pods.append(rec)
    if len(s_as_pool) >= 1 and len(s_anchor_pool) >= 2:
        cards, audit_map, rmap, idx_map, corr = _corr_ctx(train_strict_sc, strict_elig)
        picked = _greedy_low_corr_pod(
            [s_as_pool[0]],
            s_anchor_pool,
            size=min(4, 1 + len(s_anchor_pool)),
            returns_map=rmap,
            eligible_cards=cards,
            audit_map=audit_map,
            corr=corr,
            idx_by_addr=idx_map,
            cdar_alpha=train_strict_sc.config.cdar_alpha,
            max_euler_share=max_euler_share,
        )
        rec = _make_pod_record(
            "HYB-STRICT",
            picked,
            sleeve_desc="strict_hybrid",
            address_to_alias=address_to_alias,
            sleeve_by_alias=sleeve_by_alias,
            eligible_cards=cards,
            returns_map=rmap,
            corr=corr,
            idx_by_addr=idx_map,
            cdar_alpha=train_strict_sc.config.cdar_alpha,
            max_euler_share=max_euler_share,
        )
        if rec is not None:
            strict_pods.append(rec)

    # 2. RELAXED_PARTICIPATION_SCORE_CONFIG Benchmark Anchors vs Hybrids
    relaxed_pods: list[dict[str, Any]] = []
    rel_elig = {c.wallet for c in train_relaxed_sc.cards if c.eligible}
    k_anchor_pool = sorted(k_scaled_addrs & anchor_addrs)
    k_as_pool = sorted(k_scaled_addrs & alt_short_addrs)
    rel_as_pool = sorted(rel_elig & alt_short_addrs)

    full_rel_pool = rel_elig
    cards, audit_map, rmap, idx_map, corr = _corr_ctx(train_relaxed_sc, full_rel_pool)
    k_anchor_sorted = sorted(k_anchor_pool, key=lambda a: float(audit_map[a].markout_5m_bps), reverse=True)
    k_as_sorted = sorted(k_as_pool, key=lambda a: float(audit_map[a].markout_5m_bps), reverse=True)
    rel_as_sorted = sorted(rel_as_pool, key=lambda a: float(audit_map[a].markout_5m_bps), reverse=True)

    # 2a. KANCHOR-DIV3: Pure k-scaled anchor benchmark (zero altcoin shorts)
    if len(k_anchor_sorted) >= 3:
        picked_ka = _greedy_low_corr_pod(
            [k_anchor_sorted[0]],
            k_anchor_sorted,
            size=3,
            returns_map=rmap,
            eligible_cards=cards,
            audit_map=audit_map,
            corr=corr,
            idx_by_addr=idx_map,
            cdar_alpha=train_relaxed_sc.config.cdar_alpha,
            max_euler_share=max_euler_share,
        )
        rec = _make_pod_record(
            "KANCHOR-DIV3",
            picked_ka,
            sleeve_desc="kscaled_anchors_benchmark",
            address_to_alias=address_to_alias,
            sleeve_by_alias=sleeve_by_alias,
            eligible_cards=cards,
            returns_map=rmap,
            corr=corr,
            idx_by_addr=idx_map,
            cdar_alpha=train_relaxed_sc.config.cdar_alpha,
            max_euler_share=max_euler_share,
        )
        if rec is not None:
            relaxed_pods.append(rec)

    # 2b. HYB-KSCALE3/4: k-scaled (<=1%/5%) anchors + k-scaled (<=1%/5%) altcoin shorts
    if k_as_sorted and len(k_anchor_sorted) >= 2:
        seeds = k_as_sorted[: min(2, len(k_as_sorted))]
        target_sz = min(4, len(seeds) + 2) if len(seeds) == 2 else 3
        picked_hk = _greedy_low_corr_pod(
            seeds,
            k_anchor_sorted,
            size=target_sz,
            returns_map=rmap,
            eligible_cards=cards,
            audit_map=audit_map,
            corr=corr,
            idx_by_addr=idx_map,
            cdar_alpha=train_relaxed_sc.config.cdar_alpha,
            max_euler_share=max_euler_share,
        )
        rec = _make_pod_record(
            f"HYB-KSCALE{len(picked_hk)}",
            picked_hk,
            sleeve_desc="kscaled_anchors_plus_kscaled_alt_shorts",
            address_to_alias=address_to_alias,
            sleeve_by_alias=sleeve_by_alias,
            eligible_cards=cards,
            returns_map=rmap,
            corr=corr,
            idx_by_addr=idx_map,
            cdar_alpha=train_relaxed_sc.config.cdar_alpha,
            max_euler_share=max_euler_share,
        )
        if rec is not None:
            relaxed_pods.append(rec)

    # 2c. HYB-RELAXED4: k-scaled anchors + ALL RELAXED Train-eligible alt-shorts (e.g. AS#821eae6a + AS#3276acd5)
    if len(rel_as_sorted) >= 2 and len(k_anchor_sorted) >= 2:
        seeds_rel = rel_as_sorted[:2]
        picked_hr = _greedy_low_corr_pod(
            seeds_rel,
            k_anchor_sorted,
            size=4,
            returns_map=rmap,
            eligible_cards=cards,
            audit_map=audit_map,
            corr=corr,
            idx_by_addr=idx_map,
            cdar_alpha=train_relaxed_sc.config.cdar_alpha,
            max_euler_share=max_euler_share,
        )
        seen_sets = {tuple(sorted(p["addresses"])) for p in relaxed_pods}
        if tuple(sorted(picked_hr)) not in seen_sets:
            rec = _make_pod_record(
                "HYB-RELAXED4",
                picked_hr,
                sleeve_desc="kscaled_anchors_plus_all_relaxed_alt_shorts",
                address_to_alias=address_to_alias,
                sleeve_by_alias=sleeve_by_alias,
                eligible_cards=cards,
                returns_map=rmap,
                corr=corr,
                idx_by_addr=idx_map,
                cdar_alpha=train_relaxed_sc.config.cdar_alpha,
                max_euler_share=max_euler_share,
            )
            if rec is not None:
                relaxed_pods.append(rec)

    # 3. Compact Diagnostic Uptrend Hybrid Pod (PERMISSIVE_SCORE_CONFIG, 1 firm only)
    diag_pods: list[dict[str, Any]] = []
    perm_all = {c.wallet for c in train_perm_sc.cards if c.wallet in train_perm_sc._returns_by_wallet}
    d_anchors = sorted(k_anchor_sorted, key=lambda a: float(audit_map[a].markout_5m_bps), reverse=True)
    d_as_cand = sorted(compact_uptrend_as_addrs & perm_all)
    if len(d_anchors) >= 2 and len(d_as_cand) >= 1:
        dcards, daudit, drmap, didx, dcorr = _corr_ctx(train_perm_sc, set(d_anchors) | set(d_as_cand))
        seeds_d = [a for a in d_as_cand if a in didx][:2]
        picked_d = _greedy_low_corr_pod(
            seeds_d,
            [a for a in d_anchors if a in didx],
            size=min(4, len(seeds_d) + 2),
            returns_map=drmap,
            eligible_cards=dcards,
            audit_map=daudit,
            corr=dcorr,
            idx_by_addr=didx,
            cdar_alpha=train_perm_sc.config.cdar_alpha,
            max_euler_share=0.55,
        )
        rec_d = _make_pod_record(
            "HYB-UPTREND-DIAG",
            picked_d,
            sleeve_desc="diagnostic_k_anchors_plus_compact_uptrend_alt_shorts",
            address_to_alias=address_to_alias,
            sleeve_by_alias=sleeve_by_alias,
            eligible_cards=dcards,
            returns_map=drmap,
            corr=dcorr,
            idx_by_addr=didx,
            cdar_alpha=train_perm_sc.config.cdar_alpha,
            max_euler_share=0.55,
        )
        if rec_d is not None:
            diag_pods.append(rec_d)

    return {
        "strict": strict_pods,
        "relaxed_k_scaled": relaxed_pods,
        "permissive_diag": diag_pods,
    }


def _compute_oos_diagnostics(
    report: BacktestReport,
    *,
    pod_facts: FactStore,
    window: BacktestWindow,
    fee_rate: Decimal,
    shortfall_bps: Decimal,
    spread_bps: Decimal,
) -> dict[str, Any]:
    """Compute MTM equity curve, DDs, 1m bar participation, intent breakdown, and EV blocks from ``report.ledger``."""
    del fee_rate, shortfall_bps, spread_bps
    return summarize_backtest_diagnostics(
        report,
        facts=pod_facts,
        window=window,
        is_altcoin=_is_altcoin,
    )


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
    if cfg_mode == "strict_gate":
        score_cfg = ScoreConfig()
    elif cfg_mode == "relaxed_part":
        score_cfg = RELAXED_PARTICIPATION_SCORE_CONFIG
    else:
        score_cfg = PERMISSIVE_SCORE_CONFIG

    with tempfile.TemporaryDirectory(prefix=f"{pod_id.lower()}_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        pod_store_path = tmp_dir / "facts"
        shutil.copytree(pod_cache_path, pod_store_path)
        markets = tuple(MarketSpec(coin, fee_rate=fee_rate) for coin in raw_coins)
        account_id = f"{firm_slug}-10k-{pod_id.lower()}"
        ev_sha = hashlib.sha256(f"{catalog_key}:{pod_id}:{cost_label}:{cfg_mode}".encode()).hexdigest()
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
        "sleeve_design": pod["sleeve_design"],
        "aliases": list(pod["aliases"]),
        "seat_sleeves": list(pod["seat_sleeves"]),
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
        "challenge_target_reached": report.challenge_target_reached,
        "challenge_outcome": report.challenge_outcome,
        "survival_no_breach": report.challenge_outcome != "BREACHED",
        "portfolio_constraints_valid": report.portfolio_constraints_valid,
        "challenge_reasons": [_sanitize_text(r, address_to_alias) for r in report.challenge_reasons],
        "allocation_mode_share": dict(sorted(report.allocation_mode_share.items())),
        "decision_count": int(report.decision_count),
        "first_decision_sha": decisions[0].decision_sha if decisions else None,
        "last_decision_sha": decisions[-1].decision_sha if decisions else None,
        "participation": diag["participation"],
        "intent_notional_mix": diag["intent_notional_mix"],
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
    cfg_mode: str,
    firms: Sequence[tuple[str, str, str, str | None]] = FIRMS,
) -> list[dict[str, Any]]:
    if not pods:
        return []
    pod_caches: dict[str, tuple[Path, tuple[str, ...]]] = {}
    for pod in pods:
        pid = str(pod["pod_id"])
        p_cache = tmp_dir / f"pod_cache_{cfg_mode}_{pid}"
        p_coins = build_isolated_fact_store(
            cohort_store_path, p_cache, wallets=list(pod["addresses"])
        )
        pod_caches[pid] = (p_cache, p_coins)
    print(f"[stage2] pre-built {len(pod_caches)} isolated pod FactStores (mode={cfg_mode})", flush=True)

    tasks = []
    for pod in pods:
        pid = str(pod["pod_id"])
        pod_firms = (FIRMS[0],) if pid.startswith("KANCHOR") else firms
        for firm_slug, catalog_key, sim_key, program_filter in pod_firms:
            for cost_label in ("1x", "2x"):
                tasks.append(
                    (pod, firm_slug, catalog_key, sim_key, program_filter, cost_label, cfg_mode)
                )
    # Schedule heavy multi-coin HYB pods onto workers first so light KANCHOR runs share the idle worker.
    tasks.sort(
        key=lambda t: (
            0 if str(t[0]["pod_id"]) in {"HYB-RELAXED4", "HYB-KSCALE3"} else 1,
            str(t[0]["pod_id"]),
        )
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
                mode,
                pod_caches[str(pod["pod_id"])][0],
                pod_caches[str(pod["pod_id"])][1],
                plain_a2a,
            ): (pod["pod_id"], firm_slug, cost_label, mode)
            for pod, firm_slug, catalog_key, sim_key, program_filter, cost_label, mode in tasks
        }
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            max_part = res["participation"]["max_1m"]
            max_part_str = f"{max_part:.4f}" if max_part is not None else "UNKNOWN"
            print(
                f"  [done] {res['pod_id']} {res['firm_slug']} {res['cost_label']} ({res['cfg_mode']}) "
                f"net=${res['net_pnl_usd']} dd_p2t=${res['peak_to_trough_dd_usd']} "
                f"survived={res['survival_no_breach']} max_part={max_part_str} "
                f"alt_sh={res['intent_notional_mix']['alt_short_share']:.2%} "
                f"ev_blend=${res['ev_blended'].get('net_ev_usd')} "
                f"ci={res['ev_blended'].get('net_ev_ci95_usd')} "
                f"elapsed={res['elapsed_sec']}s",
                flush=True,
            )
    results.sort(key=lambda r: (r["pod_id"], r["firm_slug"], r["cost_label"]))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hybrid Altcoin-Short + Liquid-Anchor $10k Stage-2 Study")
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
        / ".local/share/copytrade/research/altcoin_shorts_20261125/private_inputs_hybrid.json",
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
    parser.add_argument("--run-diag-stage2", action="store_true", default=False)
    parser.add_argument("--pod-filter", type=str, default="")
    args = parser.parse_args(list(argv) if argv is not None else None)

    t_start = time.monotonic()
    priv = load_private_inputs(args.inputs_json)
    addresses = priv["addresses"]
    address_to_alias = priv["address_to_alias"]
    alias_to_address = priv["alias_to_address"]
    sleeve_by_alias = priv["sleeve_by_alias"]

    print(
        f"[study] loaded {len(addresses)} hybrid wallets (anchors + alt-shorts), facts={args.facts_store}",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="hybrid_alt_short_") as tmp_dir_str:
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
            bars_coins_present = (
                set(facts.scan("bars_1m", allow_unknown_known_at=True).collect()["coin"].unique().to_list())
                if bars_cov.rows > 0
                else set()
            )
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

            # 2. Vectorized regime & daily MTM profiles across Train/OOS/Uptrend
            t_prof = time.monotonic()
            regime_summary, profiles_full = compute_regime_and_wallet_profiles(
                facts,
                wallets=addresses,
                address_to_alias=address_to_alias,
                sleeve_by_alias=sleeve_by_alias,
                start_ms=_utc_ms(datetime(2026, 5, 1, tzinfo=UTC)),
                end_ms=_utc_ms(FULL_AS_OF),
            )
            print(
                f"[regime] uptrend_months={regime_summary['uptrend_months']} "
                f"profiles={len(profiles_full)} elapsed={time.monotonic() - t_prof:.1f}s",
                flush=True,
            )

            # 3. Stage-1 score_universe across Train, OOS, and Full
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

                train_k_summary, train_k_sc = summarize_scorecard_window(
                    facts,
                    name="train_may_aug_relaxed_part",
                    as_of=KnownAtTs(_utc_ms(TRAIN_AS_OF)),
                    lookback_days=TRAIN_LOOKBACK_DAYS,
                    wallets=addresses,
                    address_to_alias=address_to_alias,
                    roster=roster,
                    score_config=RELAXED_PARTICIPATION_SCORE_CONFIG,
                )
                oos_k_summary, _ = summarize_scorecard_window(
                    facts,
                    name="oos_sep_nov_relaxed_part",
                    as_of=KnownAtTs(_utc_ms(OOS_AS_OF)),
                    lookback_days=OOS_LOOKBACK_DAYS,
                    wallets=addresses,
                    address_to_alias=address_to_alias,
                    roster=roster,
                    score_config=RELAXED_PARTICIPATION_SCORE_CONFIG,
                )
                full_k_summary, _ = summarize_scorecard_window(
                    facts,
                    name="full_may_nov_relaxed_part",
                    as_of=KnownAtTs(_utc_ms(FULL_AS_OF)),
                    lookback_days=FULL_LOOKBACK_DAYS,
                    wallets=addresses,
                    address_to_alias=address_to_alias,
                    roster=roster,
                    score_config=RELAXED_PARTICIPATION_SCORE_CONFIG,
                )
                _, train_perm_sc = summarize_scorecard_window(
                    facts,
                    name="train_may_aug_permissive_diag",
                    as_of=KnownAtTs(_utc_ms(TRAIN_AS_OF)),
                    lookback_days=TRAIN_LOOKBACK_DAYS,
                    wallets=addresses,
                    address_to_alias=address_to_alias,
                    roster=roster,
                    score_config=PERMISSIVE_SCORE_CONFIG,
                )
            finally:
                roster.close()

        print(
            f"[stage1-strict] train eligible={train_summary['eligible_aliases']} "
            f"oos eligible={oos_summary['eligible_count']}",
            flush=True,
        )
        print(
            f"[stage1-relaxed] train eligible={train_k_summary['eligible_aliases']} "
            f"oos eligible={oos_k_summary['eligible_count']}",
            flush=True,
        )

        # Exclude wallets trading missing-bars coins (e.g. XYZ:* stocks) or >30 coins from Stage-2 pod construction
        missing_coin_set = set(missing_bars_coins)
        bad_stage2_wallets: set[str] = set()
        if fills_cohort_df.height > 0:
            coin_stats = (
                fills_cohort_df.with_columns(
                    pl.col("coin").str.strip_chars().str.to_uppercase().alias("coin_u")
                )
                .group_by("wallet")
                .agg(
                    pl.col("coin_u").n_unique().alias("n_coins"),
                    pl.col("coin_u").is_in(list(missing_coin_set)).any().alias("has_missing")
                    if missing_coin_set
                    else pl.lit(False).alias("has_missing"),
                )
            )
            for row in coin_stats.iter_rows(named=True):
                if bool(row["has_missing"]) or int(row["n_coins"]) > 30:
                    bad_stage2_wallets.add(str(row["wallet"]).lower())
        print(
            f"[stage2-filter] excluded {len(bad_stage2_wallets)} missing-bars/>30-coin wallets from pod selection: "
            f"{sorted(address_to_alias.get(w, w) for w in bad_stage2_wallets)}",
            flush=True,
        )

        anchor_addrs = {
            addr
            for addr, al in address_to_alias.items()
            if sleeve_by_alias.get(al, "").startswith("ANCHOR_") and addr not in bad_stage2_wallets
        }
        alt_short_addrs = {
            addr
            for addr, al in address_to_alias.items()
            if sleeve_by_alias.get(al, "") == "ALT_SHORT" and addr not in bad_stage2_wallets
        }
        train_k_audit = {row.wallet: row for row in train_k_sc.audit}
        k_scaled_addrs = {
            c.wallet
            for c in train_k_sc.cards
            if c.eligible
            and c.wallet in train_k_audit
            and float(c.suggested_k) * float(train_k_audit[c.wallet].p95_participation_1m) <= 0.01
            and float(c.suggested_k) * float(train_k_audit[c.wallet].max_participation_1m) <= 0.05
        }
        compact_uptrend_as_addrs = {
            alias_to_address[al]
            for al in ("AS#0ff64c60", "AS#87ea3330", "AS#9f361941", "AS#6d77ae70")
            if al in alias_to_address
        }

        pod_groups = form_hybrid_study_pods(
            train_strict_sc=train_sc,
            train_relaxed_sc=train_k_sc,
            train_perm_sc=train_perm_sc,
            address_to_alias=address_to_alias,
            sleeve_by_alias=sleeve_by_alias,
            anchor_addrs=anchor_addrs,
            alt_short_addrs=alt_short_addrs,
            k_scaled_addrs=k_scaled_addrs,
            compact_uptrend_as_addrs=compact_uptrend_as_addrs,
        )
        print(
            f"[pods] strict={[p['pod_id'] + ':' + ','.join(p['aliases']) for p in pod_groups['strict']]} | "
            f"relaxed_k={[p['pod_id'] + ':' + ','.join(p['aliases']) for p in pod_groups['relaxed_k_scaled']]} | "
            f"diag={[p['pod_id'] + ':' + ','.join(p['aliases']) for p in pod_groups['permissive_diag']]}",
            flush=True,
        )

        pod_filter_set = {
            tok.strip() for tok in str(args.pod_filter).split(",") if tok.strip()
        }

        def _filter_pods(plist: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if not pod_filter_set:
                return plist
            return [p for p in plist if str(p["pod_id"]) in pod_filter_set]

        strict_stage2 = _run_stage2_for_pods(
            _filter_pods(pod_groups["strict"]),
            cohort_store_path=cohort_store_path,
            tmp_dir=tmp_dir,
            address_to_alias=address_to_alias,
            max_workers=args.max_workers,
            cfg_mode="strict_gate",
        )
        relaxed_stage2 = _run_stage2_for_pods(
            _filter_pods(pod_groups["relaxed_k_scaled"]),
            cohort_store_path=cohort_store_path,
            tmp_dir=tmp_dir,
            address_to_alias=address_to_alias,
            max_workers=args.max_workers,
            cfg_mode="relaxed_part",
        )
        diag_stage2 = (
            _run_stage2_for_pods(
                _filter_pods(pod_groups["permissive_diag"]),
                cohort_store_path=cohort_store_path,
                tmp_dir=tmp_dir,
                address_to_alias=address_to_alias,
                max_workers=args.max_workers,
                cfg_mode="permissive",
                firms=(FIRMS[0],),
            )
            if args.run_diag_stage2
            else []
        )

    def _is_passing_2x(r: dict[str, Any]) -> bool:
        return (
            r["cost_label"] == "2x"
            and r["net_pnl_usd"] > 0.0
            and r["survival_no_breach"]
            and r["portfolio_constraints_valid"]
            and r["participation"]["capacity_evidence_complete"]
            and r["participation"]["n_trades"] > 0
            and r["participation"]["max_1m"] is not None
            and r["participation"]["max_1m"] <= 0.05
            and r["participation"]["p95_1m"] is not None
            and r["participation"]["p95_1m"] <= 0.01
            and (
                r["allocation_mode_share"].get("ACTIVE", 0.0)
                + r["allocation_mode_share"].get("PARTIAL", 0.0)
            )
            > 0.0
            and (r["ev_blended"].get("net_ev_usd") or -9999.0) > 0.0
        )

    passing_strict_2x = [r for r in strict_stage2 if _is_passing_2x(r)]
    passing_relaxed_2x = [r for r in relaxed_stage2 if _is_passing_2x(r)]
    if any(r["pod_id"].startswith("HYB") for r in passing_strict_2x):
        verdict = "PAPER_ELIGIBLE"
    elif any(r["pod_id"].startswith("HYB") for r in passing_relaxed_2x):
        verdict = "PAPER_ELIGIBLE_K_SCALED_ONLY"
    else:
        verdict = "NO_DEPLOYABLE_HYBRID_POD"

    output = {
        "schema": "hybrid_altcoin_short_pod_study_v1",
        "provenance": "RETROSPECTIVE_RESEARCH_ONLY",
        "verdict": verdict,
        "elapsed_sec": round(time.monotonic() - t_start, 2),
        "cohort_coins": list(raw_coins),
        "backfill_coverage": backfill_coverage,
        "regime_summary": regime_summary,
        "gate2_summary": {
            k: v for k, v in priv["gate2_summary"].items() if k != "candidates"
        },
        "cohort_candidates": priv["gate2_summary"].get("candidates", []),
        "strict_pit_check": {
            "cards_count": len(strict.cards),
            "basis": dict(sorted(strict.basis.items())),
        },
        "wallet_profiles": profiles_full,
        "stage1_strict": {
            "train_may_aug": train_summary,
            "oos_sep_nov": oos_summary,
            "full_may_nov_diag": full_summary,
        },
        "stage1_relaxed_part": {
            "train_may_aug": train_k_summary,
            "oos_sep_nov": oos_k_summary,
            "full_may_nov_diag": full_k_summary,
        },
        "pods": {
            tier: [{k: v for k, v in p.items() if k != "addresses"} for p in plist]
            for tier, plist in pod_groups.items()
        },
        "strict_stage2_runs": strict_stage2,
        "relaxed_k_scaled_stage2_runs": relaxed_stage2,
        "permissive_diag_stage2_runs": diag_stage2,
        "passing_strict_2x_runs": passing_strict_2x,
        "passing_relaxed_2x_runs": passing_relaxed_2x,
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
