#!/usr/bin/env python3
"""Phase-D kernel-backed study: May-Nov 2026 disjoint 10-pod evaluation on CFT $10k.

Stage-1 scoring uses ``trading_ops.research.score_universe`` + ``bench_from_scorecard``
and records cards into an isolated temporary ``RosterStore``.
Stage-2 walk-forward uses ``trading_ops.research.run_kernel_backtest`` over isolated
per-pod ``FactStore`` and ``RosterStore`` instances.
All archive reads use ``allow_undated=True`` (``RETROSPECTIVE_UNDATED``) after
explicitly verifying that strict PIT (``allow_undated=False``) returns zero cards.
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
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl
from copytrade_kernel import KnownAtTs, Rulebook, RulebookCatalog
from copytrade_pm.facts import FactIngestor, FactStore
from copytrade_pm.facts import manifest as mf
from copytrade_pm.facts.schemas import KNOWN_AT_BASIS_UNRECORDED, column_names
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

HEX_ADDRESS_RE = re.compile(r"(?i)0x[0-9a-f]{4,}")
STUDY_STEM = "2026-11-21__cft__10k__may-nov-disjoint-10pods"
ACCOUNT_ID = "cft-10k-research"
ACCOUNT_USD = 10_000.0
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
STAGE1_WINDOWS: tuple[tuple[str, datetime, int], ...] = (
    ("train_may_aug", datetime(2026, 9, 1, tzinfo=UTC), 123),
    ("prenov_may_oct", datetime(2026, 11, 1, tzinfo=UTC), 184),
    ("nov_dump", datetime(2026, 11, 19, tzinfo=UTC), 18),
    ("full_may_nov", datetime(2026, 11, 19, tzinfo=UTC), 202),
)
STRICT_SEATS = tuple(SeatSpec(f"seat_{idx}", Decimal(200), Decimal("0.45")) for idx in range(1, 4))
PERMISSIVE_SEATS = tuple(SeatSpec(f"seat_{idx}", Decimal(200), Decimal(1)) for idx in range(1, 4))
PERMISSIVE_SCORE_CONFIG = ScoreConfig(
    min_baseline_order_usd=Decimal("0.01"),
    max_p95_participation_1m=Decimal(1),
    max_max_participation_1m=Decimal(1),
    max_euler_cdar_share=Decimal(1),
)
STAGE2_NOV_WINDOW = (datetime(2026, 11, 1, tzinfo=UTC), datetime(2026, 11, 19, tzinfo=UTC))
STAGE2_OOS_WINDOW = (datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 11, 19, tzinfo=UTC))
STAGE2_TRAIN_WINDOW = (datetime(2026, 5, 1, tzinfo=UTC), datetime(2026, 9, 1, tzinfo=UTC))


def _assert_sanitized_text(label: str, text: str) -> None:
    match = HEX_ADDRESS_RE.search(text)
    if match is not None:
        raise RuntimeError(f"HEX_ADDRESS_LEAK:{label}:{match.group(0)[:10]}")


def _assert_no_hex_leaks(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = HEX_ADDRESS_RE.search(text)
    if match is not None:
        raise RuntimeError(f"HEX_ADDRESS_LEAK:{path}:{match.group(0)[:10]}")


def _sanitize_text(text: str, address_to_alias: Mapping[str, str]) -> str:
    if not text:
        return text

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0).lower()
        return address_to_alias.get(token, match.group(0))

    cleaned = HEX_ADDRESS_RE.sub(_replace, text)
    _assert_sanitized_text("sanitized_text", cleaned)
    return cleaned


def _utc_ms(dt: datetime) -> int:
    return int(dt.astimezone(UTC).timestamp() * 1000)


def _fmt_float(value: Decimal | float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def load_private_inputs(inputs_json: Path) -> dict[str, Any]:
    """Load private alias/address maps and 10-pod seed metadata into memory."""
    raw = json.loads(inputs_json.read_text(encoding="utf-8"))
    alias_to_address = {
        str(alias): str(addr).strip().lower() for alias, addr in raw["alias_to_address"].items()
    }
    address_to_alias = {
        str(addr).strip().lower(): str(alias) for addr, alias in raw["address_to_alias"].items()
    }
    addresses = tuple(sorted({str(addr).strip().lower() for addr in raw["addresses"]}))
    seed = dict(raw["seed"])
    return {
        "addresses": addresses,
        "alias_to_address": alias_to_address,
        "address_to_alias": address_to_alias,
        "seed_sha": str(seed["source_freeze_sha256"]),
        "tiers_46": list(seed.get("tiers_46", [])),
        "pods_10": list(seed.get("pods_10", [])),
    }


def run_strict_pit_check(
    facts: FactStore,
    *,
    wallets: Sequence[str],
    as_of: KnownAtTs,
    lookback_days: int,
    address_to_alias: Mapping[str, str],
) -> dict[str, Any]:
    """Verify strict PIT returns 0 cards on undated retrospective archive rows."""
    strict = score_universe(
        facts,
        as_of=as_of,
        lookback_days=lookback_days,
        wallets=list(wallets),
        allow_undated=False,
    )
    if len(strict.cards) != 0:
        raise RuntimeError(f"STRICT_PIT_UNEXPECTED_CARDS:{len(strict.cards)}")
    if strict.basis.get("pit") != "STRICT_KNOWN_AT":
        raise RuntimeError(f"STRICT_PIT_UNEXPECTED_BASIS:{strict.basis.get('pit')}")
    return {
        "as_of_iso": datetime.fromtimestamp(as_of.value / 1000, tz=UTC).isoformat(),
        "lookback_days": lookback_days,
        "cards_count": len(strict.cards),
        "excluded_count": len(strict.excluded),
        "basis": dict(sorted(strict.basis.items())),
        "notes": [_sanitize_text(note, address_to_alias) for note in strict.notes],
    }


def audit_stage1_window(
    facts: FactStore,
    *,
    name: str,
    as_of: KnownAtTs,
    lookback_days: int,
    wallets: Sequence[str],
    address_to_alias: Mapping[str, str],
    roster: RosterStore | None = None,
    config: ScoreConfig | None = None,
) -> dict[str, Any]:
    """Run canonical Stage-1 scoring on one window and return sanitized wallet rows."""
    scorecard = score_universe(
        facts,
        as_of=as_of,
        lookback_days=lookback_days,
        config=config,
        wallets=list(wallets),
        allow_undated=True,
    )
    recorded_shas: tuple[str, ...] = ()
    if roster is not None:
        recorded_shas = scorecard.record(roster)

    bench = bench_from_scorecard(scorecard)
    bench_by_wallet = {row.wallet: row.marginal_cdar_share for row in bench}
    audit_by_wallet = {row.wallet: row for row in scorecard.audit}
    excluded_by_wallet = {row.wallet: row for row in scorecard.excluded}
    card_by_wallet = scorecard.by_wallet

    excluded_by_reason: dict[str, list[str]] = {}
    failed_by_gate: dict[str, list[str]] = {}
    wallet_rows: list[dict[str, Any]] = []

    ordered_wallets = sorted(
        wallets,
        key=lambda addr: address_to_alias.get(addr.lower(), addr.lower()),
    )
    for addr in ordered_wallets:
        key = addr.lower()
        alias = address_to_alias[key]
        card = card_by_wallet.get(key)
        audit = audit_by_wallet.get(key)
        exclusion = excluded_by_wallet.get(key)

        if card is not None and audit is not None:
            failures = [gate.value for gate in card.failures]
            for gate_name in failures:
                failed_by_gate.setdefault(gate_name, []).append(alias)
            wallet_rows.append(
                {
                    "alias": alias,
                    "eligible": bool(card.eligible),
                    "failures": failures,
                    "excluded_reason": None,
                    "observed": int(audit.markout_fills),
                    "q10_order_usd": _fmt_float(audit.q10_order_usd, 4),
                    "markout_5m_bps": _fmt_float(audit.markout_5m_bps, 4),
                    "markout_fills": int(audit.markout_fills),
                    "p95_participation_1m": _fmt_float(audit.p95_participation_1m, 8),
                    "max_participation_1m": _fmt_float(audit.max_participation_1m, 8),
                    "cdar_share": _fmt_float(audit.cdar_share, 6),
                    "bench_cdar_share": _fmt_float(bench_by_wallet.get(key), 6),
                    "wallet_max_dd_usd": _fmt_float(audit.wallet_max_dd_usd, 2),
                    "suggested_k": _fmt_float(audit.suggested_k, 6),
                    "active_days_14d": int(audit.active_days_14d),
                    "cluster_id": str(card.cluster_id),
                }
            )
        elif exclusion is not None:
            excluded_by_reason.setdefault(exclusion.reason, []).append(alias)
            wallet_rows.append(
                {
                    "alias": alias,
                    "eligible": False,
                    "failures": [],
                    "excluded_reason": exclusion.reason,
                    "observed": int(exclusion.observed),
                    "q10_order_usd": None,
                    "markout_5m_bps": None,
                    "markout_fills": None,
                    "p95_participation_1m": None,
                    "max_participation_1m": None,
                    "cdar_share": None,
                    "bench_cdar_share": None,
                    "wallet_max_dd_usd": None,
                    "suggested_k": None,
                    "active_days_14d": None,
                    "cluster_id": None,
                }
            )
        else:
            excluded_by_reason.setdefault("NO_FILLS_IN_WINDOW", []).append(alias)
            wallet_rows.append(
                {
                    "alias": alias,
                    "eligible": False,
                    "failures": [],
                    "excluded_reason": "NO_FILLS_IN_WINDOW",
                    "observed": 0,
                    "q10_order_usd": None,
                    "markout_5m_bps": None,
                    "markout_fills": None,
                    "p95_participation_1m": None,
                    "max_participation_1m": None,
                    "cdar_share": None,
                    "bench_cdar_share": None,
                    "wallet_max_dd_usd": None,
                    "suggested_k": None,
                    "active_days_14d": None,
                    "cluster_id": None,
                }
            )

    eligible_aliases = [row["alias"] for row in wallet_rows if row["eligible"]]
    bench_aliases = [address_to_alias[row.wallet] for row in bench]
    return {
        "window_name": name,
        "as_of_iso": datetime.fromtimestamp(as_of.value / 1000, tz=UTC).isoformat(),
        "lookback_days": lookback_days,
        "allow_undated": True,
        "basis": dict(sorted(scorecard.basis.items())),
        "source_sha": scorecard.source_sha,
        "slice_days": scorecard.slice_days,
        "slice_rows": scorecard.slice_rows,
        "total_wallets": len(wallet_rows),
        "scored_count": len(scorecard.cards),
        "eligible_count": len(eligible_aliases),
        "excluded_count": len(scorecard.excluded),
        "scorecards_recorded": len(recorded_shas),
        "eligible_aliases": eligible_aliases,
        "bench_aliases": bench_aliases,
        "excluded_by_reason": {k: sorted(v) for k, v in sorted(excluded_by_reason.items())},
        "failed_by_gate": {k: sorted(v) for k, v in sorted(failed_by_gate.items())},
        "notes": [_sanitize_text(note, address_to_alias) for note in scorecard.notes],
        "wallets": wallet_rows,
    }


def run_stage1_audit(
    facts: FactStore,
    *,
    wallets: Sequence[str],
    address_to_alias: Mapping[str, str],
    windows: Sequence[tuple[str, datetime, int]] = STAGE1_WINDOWS,
) -> dict[str, Any]:
    """Run strict PIT verification and all canonical Stage-1 windows."""
    strict_pit = run_strict_pit_check(
        facts,
        wallets=wallets,
        as_of=KnownAtTs(_utc_ms(datetime(2026, 11, 19, tzinfo=UTC))),
        lookback_days=202,
        address_to_alias=address_to_alias,
    )
    window_results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="stage1_roster_") as tmp_dir:
        roster = RosterStore(Path(tmp_dir) / "roster.sqlite", create=True)
        try:
            for name, as_of_dt, lookback_days in windows:
                print(
                    f"[stage1] scoring {name} (as_of={as_of_dt.date().isoformat()}, "
                    f"lookback_days={lookback_days})...",
                    flush=True,
                )
                window_results[name] = audit_stage1_window(
                    facts,
                    name=name,
                    as_of=KnownAtTs(_utc_ms(as_of_dt)),
                    lookback_days=lookback_days,
                    wallets=wallets,
                    address_to_alias=address_to_alias,
                    roster=roster,
                )
        finally:
            roster.close()
    return {
        "strict_pit_check": strict_pit,
        "windows": window_results,
    }


def build_cft_rulebook(*, account_binding: str = ACCOUNT_ID, scale: int = 2, evidence_sha: str) -> Rulebook:
    """Build canonical CFT $10k Phase-1 Rulebook from RulebookCatalog."""
    fields = RulebookCatalog.get("crypto-fund-trader:two_phase:phase_1").to_rulebook_fields(
        account_binding=account_binding,
        scale=scale,
        evidence_sha=evidence_sha,
    )
    return Rulebook(**fields)


def load_multi_firm_catalog(
    sim_inputs_path: Path,
    *,
    account_usd: float = ACCOUNT_USD,
) -> list[dict[str, Any]]:
    """Load canonical PropContractSpecs paired with verified fee schedule tiers."""
    vanta_3pct_program = RulebookCatalog.get("vanta-trading:classic_1step_3pct:phase_1").program
    firm_queries: tuple[tuple[str, str, str, str | None], ...] = (
        (
            "crypto-fund-trader",
            "crypto-fund-trader:two_phase:phase_1",
            "crypto-fund-trader:two_phase",
            None,
        ),
        ("hyrotrader", "hyrotrader:two_step:phase_1", "hyrotrader:two_step", None),
        (
            "vanta-trading",
            "vanta-trading:classic_1step_3pct:phase_1",
            "vanta-trading:classic_1step_3pct",
            vanta_3pct_program,
        ),
        ("propr", "propr:classic_1step:phase_1", "propr:classic_1step", "Classic 1-Step"),
    )

    rows: list[dict[str, Any]] = []
    for firm_slug, catalog_key, sim_key, program_filter in firm_queries:
        quote = lookup_tier_fee(sim_key, account_usd, path=sim_inputs_path)
        fee_usd = quote.fee_usd if quote is not None else None
        refundable_on_pass = quote.refundable_on_pass if quote is not None else False
        payout_frequency = quote.payout_frequency if quote is not None else None
        verified_at = quote.verified_at if quote is not None else None
        sim_notes = quote.notes if quote is not None else None
        ev_status = "SOURCED_SIM_INPUTS" if quote is not None else "BREAK_EVEN_ONLY_FEE_UNVERIFIED"

        spec = pc.load_prop_contract_spec(
            firm_slug,
            account_usd=account_usd,
            challenge_fee_usd=fee_usd,
            program_filter=program_filter,
        )
        rows.append(
            {
                "firm_slug": spec.firm_slug,
                "catalog_key": catalog_key,
                "sim_inputs_key": sim_key,
                "program_name": spec.program_name,
                "account_usd": spec.account_usd,
                "eval_phases": [
                    {
                        "name": phase.name,
                        "profit_target_usd": phase.profit_target_usd,
                        "max_static_dd_usd": phase.max_static_dd_usd,
                        "daily_loss_limit_usd": phase.daily_loss_limit_usd,
                        "daily_loss_basis": phase.daily_loss_basis,
                        "min_trading_days": phase.min_trading_days,
                    }
                    for phase in spec.eval_phases
                ],
                "payout_split_pct": spec.payout_split_pct,
                "challenge_fee_usd": spec.challenge_fee_usd,
                "ev_status": ev_status,
                "refundable_on_pass": refundable_on_pass,
                "payout_frequency": payout_frequency,
                "verified_at": verified_at,
                "sim_notes": sim_notes,
            }
        )
    return rows


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
        .agg(
            pl.len().alias("rows"),
            pl.col("first_known_at_ms").min().alias("min_known"),
        )
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
            sorted((item.source_id, item.rel_path, item.input_sha256) for item in inputs),
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    day_sources = [
        {
            "source_id": source_id,
            "files_read": 1,
            "rows_read": count,
            "rows_kept": count,
            "rows_duplicate_within": 0,
            "rows_duplicate_cross": 0,
            "rows_conflicting_values": 0,
            "rows_rejected": 0,
            "rows_null_key": 0,
            "rows_quantized_to_zero": 0,
        }
        for source_id, count in sorted(source_totals.items())
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
    equity_df: pl.DataFrame,
    anchor_days: Sequence[str],
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


def build_pod_fact_store(
    src_store_path: Path,
    dst_store_path: Path,
    *,
    pod_wallets: Sequence[str],
    anchor_days: Sequence[str] = DEFAULT_EQUITY_ANCHOR_DAYS,
    start_day: str | None = None,
    end_day: str | None = None,
) -> tuple[str, ...]:
    """Create an isolated lightweight FactStore for one pod's wallets and traded coins."""
    target_wallets = sorted({str(w).strip().lower() for w in pod_wallets if str(w).strip()})
    if not target_wallets:
        raise ValueError("EMPTY_POD_WALLETS")
    normalized_anchors = tuple(
        dict.fromkeys(date.fromisoformat(d.strip()).isoformat() for d in anchor_days if d.strip())
    )
    if dst_store_path.exists():
        shutil.rmtree(dst_store_path)

    raw_coins: set[str] = set()
    norm_coins: set[str] = set()
    with FactStore(src_store_path, create=False) as src, FactStore(
        dst_store_path, create=True
    ) as dst:
        src_con = src.connect()
        dst.connect()

        for batch in mf.day_batches(src_con, "fills", start_day=start_day, end_day=end_day):
            batch_path = _resolve_batch_path(src, batch)
            day_df = (
                pl.scan_parquet(batch_path)
                .filter(pl.col("wallet").is_in(target_wallets))
                .collect()
            )
            if day_df.height == 0:
                continue
            for coin in day_df.select(pl.col("coin").unique()).to_series().to_list():
                if coin is not None and str(coin):
                    coin_str = str(coin)
                    raw_coins.add(coin_str)
                    norm_coins.add(coin_str.strip().upper())
            prepared = dst.prepare_day("fills", batch.day, day_df)
            write = dst.write_day(prepared)
            _record_day_batch(
                dst,
                kind="fills",
                day=batch.day,
                frame=prepared.frame,
                write=write,
                rel_prefix="pod",
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

        if equity_frames:
            combined_eq = pl.concat(equity_frames, how="vertical_relaxed")
            ingestor = FactIngestor(dst)
            ingestor.ingest_frame(
                "equity",
                combined_eq,
                source_id="pod-equity",
                source_rank=1,
                first_known_at_ms=None,
            )
            anchor_rows = _build_equity_anchors(combined_eq, normalized_anchors)
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
                    .filter(pl.col("coin").str.strip_chars().str.to_uppercase().is_in(sorted_norm_coins))
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
                    rel_prefix="pod",
                )

    return tuple(sorted(raw_coins))


def summarize_backtest_report(
    report: BacktestReport,
    *,
    pod_id: str,
    config_name: str,
    window_name: str,
    lookback_days: int,
    initial_equity: Decimal,
    address_to_alias: Mapping[str, str],
) -> dict[str, Any]:
    """Convert a ``BacktestReport`` into a sanitized JSON-serializable summary."""
    decisions = report.decisions
    net_pnl = report.ending_equity - initial_equity
    return {
        "pod_id": pod_id,
        "config_name": config_name,
        "window_name": window_name,
        "window_start": datetime.fromtimestamp(report.window.start_ms / 1000, tz=UTC).isoformat(),
        "window_end": datetime.fromtimestamp(report.window.end_ms / 1000, tz=UTC).isoformat(),
        "lookback_days": lookback_days,
        "status": "OK",
        "allocation_mode_share": dict(sorted(report.allocation_mode_share.items())),
        "ending_equity": round(float(report.ending_equity), 2),
        "net_pnl_usd": round(float(net_pnl), 2),
        "challenge_pass_rate": float(report.challenge_pass_rate),
        "challenge_reasons": [
            _sanitize_text(reason, address_to_alias) for reason in report.challenge_reasons
        ],
        "decision_count": int(report.decision_count),
        "first_decision_sha": decisions[0].decision_sha if decisions else None,
        "last_decision_sha": decisions[-1].decision_sha if decisions else None,
        "reconcile_plans_count": len(report.reconcile_plans),
        "scorecards_recorded": int(report.scorecards_recorded),
        "warnings_count": len(report.warnings),
        "warnings_sample": [
            _sanitize_text(warning, address_to_alias) for warning in report.warnings[:10]
        ],
        "source_sha": report.source_sha,
    }


def _empty_backtest_summary(
    *,
    pod_id: str,
    config_name: str,
    window_name: str,
    window: BacktestWindow,
    lookback_days: int,
    initial_equity: Decimal,
) -> dict[str, Any]:
    return {
        "pod_id": pod_id,
        "config_name": config_name,
        "window_name": window_name,
        "window_start": datetime.fromtimestamp(window.start_ms / 1000, tz=UTC).isoformat(),
        "window_end": datetime.fromtimestamp(window.end_ms / 1000, tz=UTC).isoformat(),
        "lookback_days": lookback_days,
        "status": "NO_FACT_ROWS_IN_WINDOW",
        "allocation_mode_share": {},
        "ending_equity": float(initial_equity),
        "net_pnl_usd": 0.0,
        "challenge_pass_rate": 1.0,
        "challenge_reasons": [],
        "decision_count": 0,
        "first_decision_sha": None,
        "last_decision_sha": None,
        "reconcile_plans_count": 0,
        "scorecards_recorded": 0,
        "warnings_count": 0,
        "warnings_sample": [],
        "source_sha": None,
    }


def run_pod_backtest_case(
    facts: FactStore,
    *,
    pod_id: str,
    config_name: str,
    window_name: str,
    window: BacktestWindow,
    lookback_days: int,
    seats: Sequence[SeatSpec],
    score_config: ScoreConfig,
    rulebook: Rulebook,
    markets: Sequence[MarketSpec],
    roster_path: Path,
    address_to_alias: Mapping[str, str],
) -> dict[str, Any]:
    """Run one Stage-2 ``run_kernel_backtest`` config/window with an isolated RosterStore."""
    initial_equity = Decimal(rulebook.initial)
    if not markets:
        return _empty_backtest_summary(
            pod_id=pod_id,
            config_name=config_name,
            window_name=window_name,
            window=window,
            lookback_days=lookback_days,
            initial_equity=initial_equity,
        )

    cfg = BacktestConfig(
        account_id=rulebook.account_binding,
        initial_equity=initial_equity,
        lookback_days=lookback_days,
        score_config=score_config,
        allow_undated=True,
    )
    roster = RosterStore(roster_path, create=True)
    try:
        try:
            report = run_kernel_backtest(
                facts,
                window=window,
                seats=seats,
                rulebook=rulebook,
                markets=markets,
                config=cfg,
                roster=roster,
            )
        except ValueError as exc:
            if str(exc) == "NO_FACT_ROWS_IN_WINDOW":
                return _empty_backtest_summary(
                    pod_id=pod_id,
                    config_name=config_name,
                    window_name=window_name,
                    window=window,
                    lookback_days=lookback_days,
                    initial_equity=initial_equity,
                )
            raise
    finally:
        roster.close()

    return summarize_backtest_report(
        report,
        pod_id=pod_id,
        config_name=config_name,
        window_name=window_name,
        lookback_days=lookback_days,
        initial_equity=initial_equity,
        address_to_alias=address_to_alias,
    )


def evaluate_single_pod(
    pod: Mapping[str, Any],
    *,
    src_store_path: Path,
    alias_to_address: Mapping[str, str],
    address_to_alias: Mapping[str, str],
    rulebook: Rulebook,
    anchor_days: Sequence[str] = DEFAULT_EQUITY_ANCHOR_DAYS,
    oos_window: tuple[datetime, datetime] = STAGE2_OOS_WINDOW,
    nov_window: tuple[datetime, datetime] = STAGE2_NOV_WINDOW,
) -> dict[str, Any]:
    """Build an isolated pod FactStore and evaluate strict_gate + permissive_constituent."""
    t0 = time.perf_counter()
    pod_id = str(pod["pod_id"])
    aliases = [str(alias) for alias in pod["aliases"]]
    pod_wallets = [alias_to_address[alias] for alias in aliases]

    with tempfile.TemporaryDirectory(prefix=f"{pod_id.lower()}_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        pod_store_path = tmp_dir / "facts"
        raw_coins = build_pod_fact_store(
            src_store_path,
            pod_store_path,
            pod_wallets=pod_wallets,
            anchor_days=anchor_days,
        )
        markets = tuple(MarketSpec(coin) for coin in raw_coins)
        runs: list[dict[str, Any]] = []

        with FactStore(pod_store_path, create=False) as pod_facts:
            for cfg_name, seats, score_cfg, win_name, win_pair, lookback in (
                (
                    "strict_gate",
                    STRICT_SEATS,
                    ScoreConfig(),
                    "nov_dump",
                    nov_window,
                    30,
                ),
                (
                    "strict_gate",
                    STRICT_SEATS,
                    ScoreConfig(),
                    "sep_nov_oos",
                    oos_window,
                    123,
                ),
                (
                    "permissive_constituent",
                    PERMISSIVE_SEATS,
                    PERMISSIVE_SCORE_CONFIG,
                    "nov_dump",
                    nov_window,
                    123,
                ),
                (
                    "permissive_constituent",
                    PERMISSIVE_SEATS,
                    PERMISSIVE_SCORE_CONFIG,
                    "sep_nov_oos",
                    oos_window,
                    123,
                ),
            ):
                result = run_pod_backtest_case(
                    pod_facts,
                    pod_id=pod_id,
                    config_name=cfg_name,
                    window_name=win_name,
                    window=BacktestWindow.from_utc(*win_pair),
                    lookback_days=lookback,
                    seats=seats,
                    score_config=score_cfg,
                    rulebook=rulebook,
                    markets=markets,
                    roster_path=tmp_dir / f"roster_{cfg_name}_{win_name}.sqlite",
                    address_to_alias=address_to_alias,
                )
                runs.append(result)

            sep_nov_perm = next(
                r
                for r in runs
                if r["config_name"] == "permissive_constituent" and r["window_name"] == "sep_nov_oos"
            )
            perm_occupied_share = (
                sep_nov_perm["allocation_mode_share"].get("ACTIVE", 0.0)
                + sep_nov_perm["allocation_mode_share"].get("PARTIAL", 0.0)
            )
            run_train = (
                pod_id == "MPOD-06"
                or (perm_occupied_share > 0.0 and sep_nov_perm["challenge_pass_rate"] == 1.0)
            )
            if run_train:
                runs.append(
                    run_pod_backtest_case(
                        pod_facts,
                        pod_id=pod_id,
                        config_name="permissive_constituent",
                        window_name="train_may_aug",
                        window=BacktestWindow.from_utc(*STAGE2_TRAIN_WINDOW),
                        lookback_days=30,
                        seats=PERMISSIVE_SEATS,
                        score_config=PERMISSIVE_SCORE_CONFIG,
                        rulebook=rulebook,
                        markets=markets,
                        roster_path=tmp_dir / "roster_permissive_constituent_train_may_aug.sqlite",
                        address_to_alias=address_to_alias,
                    )
                )

    elapsed = round(time.perf_counter() - t0, 2)
    msg = (
        f"[stage2] {pod_id} ({', '.join(aliases)}) coins={len(raw_coins)} "
        f"runs={len(runs)} elapsed={elapsed:.2f}s"
    )
    print(msg, flush=True)

    return {
        "pod_id": pod_id,
        "aliases": aliases,
        "coins": list(raw_coins),
        "frozen_k": list(pod.get("frozen_k", [])),
        "prenov_refit_k": list(pod.get("prenov_refit_k", [])),
        "prior_fleet_metrics": dict(pod.get("prior_fleet_metrics", {})),
        "elapsed_sec": elapsed,
        "runs": runs,
    }


def _order_pods(pods: Sequence[Mapping[str, Any]], pod_filter: str | None) -> list[dict[str, Any]]:
    selected_ids = (
        {item.strip().upper() for item in pod_filter.split(",") if item.strip()}
        if pod_filter
        else None
    )
    filtered = [
        dict(pod)
        for pod in pods
        if selected_ids is None or str(pod["pod_id"]).upper() in selected_ids
    ]
    if selected_ids is not None and not filtered:
        raise ValueError(f"NO_MATCHING_PODS:{pod_filter}")
    filtered.sort(key=lambda pod: (0 if pod["pod_id"] == "MPOD-06" else 1, str(pod["pod_id"])))
    return filtered


def _evaluate_pod_process(
    pod: dict[str, Any],
    src_store_path: Path,
    alias_to_address: dict[str, str],
    address_to_alias: dict[str, str],
    evidence_sha: str,
    anchor_days: tuple[str, ...],
    oos_window: tuple[datetime, datetime] = STAGE2_OOS_WINDOW,
    nov_window: tuple[datetime, datetime] = STAGE2_NOV_WINDOW,
) -> dict[str, Any]:
    """Child-process Stage-2 pod worker rebuilding Rulebook from picklable args."""
    rulebook = build_cft_rulebook(evidence_sha=evidence_sha)
    return evaluate_single_pod(
        pod,
        src_store_path=src_store_path,
        alias_to_address=alias_to_address,
        address_to_alias=address_to_alias,
        rulebook=rulebook,
        anchor_days=anchor_days,
        oos_window=oos_window,
        nov_window=nov_window,
    )


def resolve_max_workers(max_workers: int, cpu_count: int | None = None) -> int:
    """Resolve worker count; 0 reserves one CPU core for Jupyter responsiveness."""
    if max_workers < 0:
        raise ValueError("INVALID_MAX_WORKERS")
    if max_workers == 0:
        detected = cpu_count if cpu_count is not None else (os.cpu_count() or 2)
        return max(1, detected - 1)
    return max_workers


def run_stage2_pods(
    pods: Sequence[Mapping[str, Any]],
    *,
    src_store_path: Path,
    alias_to_address: Mapping[str, str],
    address_to_alias: Mapping[str, str],
    rulebook: Rulebook,
    max_workers: int = 4,
    anchor_days: Sequence[str] = DEFAULT_EQUITY_ANCHOR_DAYS,
    oos_window: tuple[datetime, datetime] = STAGE2_OOS_WINDOW,
    nov_window: tuple[datetime, datetime] = STAGE2_NOV_WINDOW,
) -> list[dict[str, Any]]:
    """Run isolated Stage-2 kernel backtests across selected pods."""
    resolved_workers = resolve_max_workers(max_workers)
    ordered = list(pods)
    if resolved_workers == 1 or len(ordered) <= 1:
        results = [
            evaluate_single_pod(
                pod,
                src_store_path=src_store_path,
                alias_to_address=alias_to_address,
                address_to_alias=address_to_alias,
                rulebook=rulebook,
                anchor_days=anchor_days,
                oos_window=oos_window,
                nov_window=nov_window,
            )
            for pod in ordered
        ]
    else:
        results = []
        plain_alias_to_addr = {str(k): str(v) for k, v in alias_to_address.items()}
        plain_addr_to_alias = {str(k): str(v) for k, v in address_to_alias.items()}
        plain_anchors = tuple(str(day) for day in anchor_days)
        mp_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(resolved_workers, len(ordered)),
            mp_context=mp_ctx,
        ) as pool:
            if _evaluate_pod_process.__module__ in ("__main__", "__mp_main__"):
                futures = {
                    pool.submit(
                        _evaluate_pod_process,
                        dict(pod),
                        Path(src_store_path),
                        plain_alias_to_addr,
                        plain_addr_to_alias,
                        str(rulebook.evidence_sha),
                        plain_anchors,
                        oos_window,
                        nov_window,
                    ): str(pod["pod_id"])
                    for pod in ordered
                }
            else:
                study_file = str(Path(__file__).resolve())
                futures = {
                    pool.submit(
                        pc._exec_file_callable,
                        study_file,
                        "_evaluate_pod_process",
                        dict(pod),
                        Path(src_store_path),
                        plain_alias_to_addr,
                        plain_addr_to_alias,
                        str(rulebook.evidence_sha),
                        plain_anchors,
                        oos_window,
                        nov_window,
                    ): str(pod["pod_id"])
                    for pod in ordered
                }
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda item: str(item["pod_id"]))
    return results


def select_top_candidate(stage2_pods: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Rank pods deterministically by strict OOS pass, then permissive OOS pass."""
    strict_candidates: list[tuple[float, str, Mapping[str, Any], Mapping[str, Any]]] = []
    perm_candidates: list[tuple[float, str, Mapping[str, Any], Mapping[str, Any]]] = []

    for pod in stage2_pods:
        for run in pod.get("runs", []):
            if run["window_name"] != "sep_nov_oos" or run["status"] != "OK":
                continue
            mode_share = run["allocation_mode_share"]
            occupied_share = float(mode_share.get("ACTIVE", 0.0)) + float(
                mode_share.get("PARTIAL", 0.0)
            )
            if occupied_share <= 0.0 or float(run["challenge_pass_rate"]) < 1.0:
                continue
            entry = (float(run["net_pnl_usd"]), str(pod["pod_id"]), pod, run)
            if run["config_name"] == "strict_gate":
                strict_candidates.append(entry)
            elif run["config_name"] == "permissive_constituent":
                perm_candidates.append(entry)

    if strict_candidates:
        strict_candidates.sort(key=lambda item: (-item[0], item[1]))
        net_pnl, pod_id, pod, run = strict_candidates[0]
        return {
            "label": f"{pod_id}_STRICT_OOS_RETROSPECTIVE",
            "pod_id": pod_id,
            "selection_basis": "strict_gate:sep_nov_oos",
            "aliases": list(pod["aliases"]),
            "frozen_k": list(pod["frozen_k"]),
            "net_pnl_usd": net_pnl,
            "first_decision_sha": run["first_decision_sha"],
            "last_decision_sha": run["last_decision_sha"],
        }
    if perm_candidates:
        perm_candidates.sort(key=lambda item: (-item[0], item[1]))
        net_pnl, pod_id, pod, run = perm_candidates[0]
        return {
            "label": f"{pod_id}_PERMISSIVE_OOS_RETROSPECTIVE",
            "pod_id": pod_id,
            "selection_basis": "permissive_constituent:sep_nov_oos",
            "aliases": list(pod["aliases"]),
            "frozen_k": list(pod["frozen_k"]),
            "net_pnl_usd": net_pnl,
            "first_decision_sha": run["first_decision_sha"],
            "last_decision_sha": run["last_decision_sha"],
        }
    return {
        "label": "NONE_ELIGIBLE_RETROSPECTIVE",
        "pod_id": None,
        "selection_basis": "NONE",
        "aliases": [],
        "frozen_k": [],
        "net_pnl_usd": 0.0,
        "first_decision_sha": None,
        "last_decision_sha": None,
    }


def render_markdown_report(results: Mapping[str, Any]) -> str:
    """Render sanitized Markdown report with YAML frontmatter and three-line caveat footer."""
    top = results["top_candidate"]
    strict_pit = results["stage1"]["strict_pit_check"]
    full_win = results["stage1"]["windows"]["full_may_nov"]

    lines: list[str] = [
        "---",
        "firm: crypto-fund-trader",
        f"account_usd: {int(ACCOUNT_USD)}",
        "status: RETROSPECTIVE_RESEARCH_ONLY",
        f"top_candidate: {top['label']}",
        f"script: studies/{STUDY_STEM}.py",
        "---",
        "",
        "# May-Nov 2026 Disjoint 10-Pod Kernel Research Study (CFT $10k)",
        "",
        "## Provenance and Isolation Disclosure",
        "",
        (
            f"- **Strict PIT check (`allow_undated=False`):** returned `{strict_pit['cards_count']}` "
            f"cards with `basis.pit={strict_pit['basis']['pit']}` over "
            f"`{strict_pit['lookback_days']}` days as of `{strict_pit['as_of_iso']}`."
        ),
        (
            "- **Retrospective archive opt-in (`allow_undated=True`):** all Stage-1 scorecards and "
            "Stage-2 kernel walk-forwards explicitly opt into `allow_undated=True` "
            "(`basis.pit=RETROSPECTIVE_UNDATED`) over the slimmed isolated 46-wallet `FactStore` "
            "(with carried-forward monthly equity anchors under `known_at_basis=unrecorded`)."
        ),
        (
            "- **Isolation guarantee:** no live registry or live roster store was read or mutated; "
            "all Stage-1 and Stage-2 cards were written to temporary `RosterStore` instances only."
        ),
        "",
        "## Stage-1 Canonical Scoring Summary (46 Wallets, Default `ScoreConfig`)",
        "",
        "| Window | As of (UTC) | Lookback (d) | Scored | Eligible | Excluded | Eligible aliases |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for win_name in ("train_may_aug", "prenov_may_oct", "nov_dump", "full_may_nov"):
        win = results["stage1"]["windows"][win_name]
        elig = ", ".join(win["eligible_aliases"]) if win["eligible_aliases"] else "none"
        lines.append(
            f"| `{win_name}` | `{win['as_of_iso']}` | {win['lookback_days']} | "
            f"{win['scored_count']} | {win['eligible_count']} | {win['excluded_count']} | {elig} |"
        )

    lines.extend(
        [
            "",
            "### Full May-Nov (`202d`) Exclusions and Gate Failures",
            "",
            "| Category | Code | Count | Aliases |",
            "|---|---|---:|---|",
        ]
    )
    for reason, aliases in full_win["excluded_by_reason"].items():
        lines.append(
            f"| `EXCLUDED` | `{reason}` | {len(aliases)} | {', '.join(aliases)} |"
        )
    for gate, aliases in full_win["failed_by_gate"].items():
        lines.append(
            f"| `GATE_FAIL` | `{gate}` | {len(aliases)} | {', '.join(aliases)} |"
        )

    lines.extend(
        [
            "",
            "### Full May-Nov (`202d`) Per-Wallet Stage-1 Metrics",
            "",
            (
                "| Alias | Eligible | Excluded / Failures | Observed | q10 ($) | Markout 5m (bps) | "
                "p95 Part | Max Part | CDaR Share | Bench CDaR | Max DD ($) | k | Active 14d | Cluster |"
            ),
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in full_win["wallets"]:
        issue = (
            row["excluded_reason"]
            if row["excluded_reason"] is not None
            else (", ".join(row["failures"]) if row["failures"] else "PASS")
        )
        lines.append(
            f"| `{row['alias']}` | `{row['eligible']}` | `{issue}` | {row['observed']} | "
            f"{row['q10_order_usd']} | {row['markout_5m_bps']} | {row['p95_participation_1m']} | "
            f"{row['max_participation_1m']} | {row['cdar_share']} | {row['bench_cdar_share']} | "
            f"{row['wallet_max_dd_usd']} | {row['suggested_k']} | {row['active_days_14d']} | "
            f"{row['cluster_id']} |"
        )

    lines.extend(
        [
            "",
            "## Stage-2 Kernel Walk-Forward (`run_kernel_backtest`)",
            "",
            (
                "| Pod | Config | Window | Status | Mode Share | Net PnL ($) | End Eq ($) | "
                "Pass Rate | Reasons | Decisions | Plans | Cards | First/Last `decision_sha` |"
            ),
            "|---|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---|",
        ]
    )
    for pod in results["stage2_pods"]:
        for run in pod["runs"]:
            mode_str = ", ".join(
                f"{k}={v:.3f}" for k, v in sorted(run["allocation_mode_share"].items())
            ) or "none"
            reasons = ", ".join(run["challenge_reasons"]) if run["challenge_reasons"] else "none"
            first_sha = (run["first_decision_sha"] or "none")[:12]
            last_sha = (run["last_decision_sha"] or "none")[:12]
            lines.append(
                f"| `{run['pod_id']}` | `{run['config_name']}` | `{run['window_name']}` | "
                f"`{run['status']}` | `{mode_str}` | {run['net_pnl_usd']:.2f} | "
                f"{run['ending_equity']:.2f} | {run['challenge_pass_rate']:.2f} | `{reasons}` | "
                f"{run['decision_count']} | {run['reconcile_plans_count']} | "
                f"{run['scorecards_recorded']} | `{first_sha}..{last_sha}` |"
            )

    lines.extend(
        [
            "",
            "## Multi-Firm Rulebook Catalog and `sim_inputs.json` Fee Schedule ($10k)",
            "",
            (
                "| Firm | Program | Target ($) | Max DD ($) | Daily ($) | Daily Basis | "
                "Split | Fee ($) | Refundable | Fee Status |"
            ),
            "|---|---|---:|---:|---:|---|---:|---:|---|---|",
        ]
    )
    for firm in results["multi_firm_catalog"]:
        phase1 = firm["eval_phases"][0]
        fee_str = f"{firm['challenge_fee_usd']:.2f}" if firm["challenge_fee_usd"] is not None else "n/a"
        lines.append(
            f"| `{firm['firm_slug']}` | `{firm['program_name']}` | "
            f"{phase1['profit_target_usd']:.0f} | {phase1['max_static_dd_usd']:.0f} | "
            f"{phase1['daily_loss_limit_usd']:.0f} | `{phase1['daily_loss_basis']}` | "
            f"{firm['payout_split_pct']:.2f} | {fee_str} | `{firm['refundable_on_pass']}` | "
            f"`{firm['ev_status']}` |"
        )

    lines.extend(
        [
            "",
            "## Qualification and caveats",
            "",
            "### Three-line caveat footer",
            "- CFT trading restrictions and account framing require independent compliance confirmation.",
            "- Live broker execution, funding, capacity, and strict PIT source continuity are UNPROVEN.",
            "- No deployments, allocations, orders, service writes, or live registry/roster mutations occurred.",
            "",
        ]
    )
    return "\n".join(lines)


def build_proposal(results: Mapping[str, Any], *, rulebook: Rulebook) -> dict[str, Any]:
    """Build an approval-gated non-authorizing research proposal."""
    top = results["top_candidate"]
    return {
        "schema": "trading_ops_research_proposal_v1",
        "created_at": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "PROPOSAL_UNAPPROVED_RETROSPECTIVE",
        "approval_required": True,
        "trading_authorized": False,
        "roster_mutated": False,
        "provenance": "RETROSPECTIVE_RESEARCH_ONLY",
        "allow_undated": True,
        "target_account_profile": {
            "firm": "crypto-fund-trader",
            "catalog_key": "crypto-fund-trader:two_phase:phase_1",
            "account_usd": int(rulebook.initial),
            "profit_target_usd": int(rulebook.target),
            "daily_loss_limit_usd": int(rulebook.daily_loss),
            "daily_basis": rulebook.daily_basis,
            "max_static_dd_usd": int(rulebook.trailing_loss),
            "trailing_basis": rulebook.trailing_basis,
            "fee_status": "BREAK_EVEN_ONLY_FEE_UNVERIFIED",
        },
        "candidate": {
            "source_study_script": f"studies/{STUDY_STEM}.py",
            "source_report_md": f"reports/{STUDY_STEM}.md",
            "source_results_json": f"reports/{STUDY_STEM}__results.json",
            "top_candidate_label": top["label"],
            "pod_id": top["pod_id"],
            "selection_basis": top["selection_basis"],
            "aliases": top["aliases"],
            "seed_frozen_k": top["frozen_k"],
            "sep_nov_oos_net_pnl_usd": top["net_pnl_usd"],
            "first_decision_sha": top["first_decision_sha"],
            "last_decision_sha": top["last_decision_sha"],
        },
        "unresolved_gates": [
            {
                "code": "STRICT_PIT_EMPTY",
                "detail": "Retrospective FactStore returned 0 cards with allow_undated=False.",
            },
            {
                "code": "CFT_FEE_UNVERIFIED",
                "detail": "crypto-fund-trader is absent from sim_inputs.json; fee remains unverified.",
            },
            {
                "code": "UPSTREAM_SEAL_AND_CWSA_PENDING",
                "detail": "Prospective CWSA observation and copytrade-pm sealing require explicit approval.",
            },
        ],
    }


def write_sanitized_outputs(
    out_dir: Path,
    *,
    results: Mapping[str, Any],
    report_md: str,
    proposal: Mapping[str, Any],
) -> dict[str, Path]:
    """Write report, results JSON, and proposal JSON, failing closed on any hex address."""
    reports_dir = out_dir / "reports"
    proposals_dir = out_dir / "proposals"
    reports_dir.mkdir(parents=True, exist_ok=True)
    proposals_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / f"{STUDY_STEM}.md"
    results_path = reports_dir / f"{STUDY_STEM}__results.json"
    proposal_path = proposals_dir / f"{STUDY_STEM}.json"

    results_text = json.dumps(results, indent=2) + "\n"
    proposal_text = json.dumps(proposal, indent=2) + "\n"

    _assert_sanitized_text("report_md", report_md)
    _assert_sanitized_text("results_json", results_text)
    _assert_sanitized_text("proposal_json", proposal_text)

    report_path.write_text(report_md, encoding="utf-8")
    results_path.write_text(results_text, encoding="utf-8")
    proposal_path.write_text(proposal_text, encoding="utf-8")

    for emitted in (report_path, results_path, proposal_path):
        _assert_no_hex_leaks(emitted)

    return {
        "report_md": report_path,
        "results_json": results_path,
        "proposal_json": proposal_path,
    }


def run_study(
    *,
    facts_store: Path,
    inputs_json: Path,
    sim_inputs: Path,
    out_dir: Path,
    max_workers: int = 4,
    pods_filter: str | None = None,
) -> dict[str, Any]:
    """Execute the complete Phase-D study pipeline and emit sanitized artifacts."""
    resolved_workers = resolve_max_workers(max_workers)
    inputs = load_private_inputs(inputs_json)
    selected_pods = _order_pods(inputs["pods_10"], pods_filter)
    rulebook = build_cft_rulebook(evidence_sha=inputs["seed_sha"])

    print(
        f"[study] starting {STUDY_STEM}: wallets={len(inputs['addresses'])}, "
        f"pods={len(selected_pods)}, max_workers={resolved_workers}",
        flush=True,
    )
    with FactStore(facts_store, create=False) as facts:
        stage1 = run_stage1_audit(
            facts,
            wallets=inputs["addresses"],
            address_to_alias=inputs["address_to_alias"],
        )

    multi_firm = load_multi_firm_catalog(sim_inputs, account_usd=ACCOUNT_USD)
    stage2_pods = run_stage2_pods(
        selected_pods,
        src_store_path=facts_store,
        alias_to_address=inputs["alias_to_address"],
        address_to_alias=inputs["address_to_alias"],
        rulebook=rulebook,
        max_workers=resolved_workers,
    )
    top_candidate = select_top_candidate(stage2_pods)

    results: dict[str, Any] = {
        "schema": "may_nov_disjoint_10pods_results_v1",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "provenance": "RETROSPECTIVE_RESEARCH_ONLY",
        "allow_undated": True,
        "seed_sha": inputs["seed_sha"],
        "account_id": ACCOUNT_ID,
        "account_usd": ACCOUNT_USD,
        "stage1": stage1,
        "multi_firm_catalog": multi_firm,
        "stage2_pods": stage2_pods,
        "top_candidate": top_candidate,
    }
    report_md = render_markdown_report(results)
    proposal = build_proposal(results, rulebook=rulebook)
    outputs = write_sanitized_outputs(
        out_dir,
        results=results,
        report_md=report_md,
        proposal=proposal,
    )
    return {
        "results": results,
        "outputs": {k: str(v) for k, v in outputs.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts-store", type=Path, default=Path("data/facts_may_nov_46_slim"))
    parser.add_argument("--inputs-json", type=Path, default=Path("data/private_46_inputs.json"))
    parser.add_argument(
        "--sim-inputs",
        type=Path,
        default=DEFAULT_FEE_SCHEDULE_PATH,
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--pods", type=str, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_study(
        facts_store=args.facts_store,
        inputs_json=args.inputs_json,
        sim_inputs=args.sim_inputs,
        out_dir=args.out_dir,
        max_workers=args.max_workers,
        pods_filter=args.pods,
    )
    top = summary["results"]["top_candidate"]
    payload = {
        "status": "OK",
        "top_candidate": top["label"],
        "outputs": summary["outputs"],
    }
    out_text = json.dumps(payload, indent=2)
    _assert_sanitized_text("stdout", out_text)
    print(out_text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
