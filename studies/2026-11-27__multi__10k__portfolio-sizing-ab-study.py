#!/usr/bin/env python3
"""Controlled A/B study: decoupled portfolio fixed-k allocation vs. independent fixed-k.

Evaluates four sizing arms across 1x and 2x transaction cost stress on an isolated,
read-only copy of the local FactStore cohort (`2026-07-01..2026-11-18`, strictly
`RETROSPECTIVE_RESEARCH_ONLY` because Reservoir fills have `first_known_at_ms = NULL`).

Research-only: outputs are written under ``reports/`` with opaque ``W#<sha8>``
aliases and zero raw ``0x...`` addresses. Never authorizes live trading.
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
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pod_constructor as pc
import polars as pl
from copytrade_kernel import KnownAtTs, Rulebook, RulebookCatalog
from copytrade_pm.facts import FactIngestor, FactStore
from copytrade_pm.facts import manifest as mf
from copytrade_pm.facts.schemas import KNOWN_AT_BASIS_UNRECORDED
from copytrade_pm.roster import RosterStore, SeatSpec

from trading_ops.research import (
    DEFAULT_FEE_SCHEDULE_PATH,
    AllocationPolicy,
    BacktestConfig,
    BacktestReport,
    BacktestWindow,
    MarketSpec,
    ScoreConfig,
    allocate_fixed_k_portfolio,
    lookup_tier_fee,
    run_kernel_backtest,
    score_universe,
    summarize_backtest_diagnostics,
)
from trading_ops.research.scoring import peak_drawdown

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FACTS_ROOT = Path.home() / ".local" / "share" / "copytrade" / "facts"
REPORT_MD_PATH = REPO_ROOT / "reports" / "2026-11-27__multi__10k__portfolio-sizing-ab-study.md"
REPORT_JSON_PATH = (
    REPO_ROOT / "reports" / "2026-11-27__multi__10k__portfolio-sizing-ab-study__results.json"
)

HEX_ADDRESS_RE = re.compile(r"(?i)0x[0-9a-f]{4,}")
GRID_MS = 300_000
TRAIN_LOOKBACK_DAYS = 60
MIN_TRAIN_FILLS = 50
FILLS_START_DAY = "2026-07-01"
END_DAY = "2026-11-18"
OOS_START = datetime(2026, 9, 1, tzinfo=UTC)
OOS_END = datetime(2026, 11, 18, tzinfo=UTC)
CHOP_START_DAY = "2026-10-01"
EQUITY_ANCHOR_DAYS = (
    "2026-07-01",
    "2026-08-01",
    "2026-09-01",
    "2026-10-01",
    "2026-11-01",
    "2026-11-18",
)
ACCOUNT_USD = 10_000.0
MIN_EQUITY_USD = 1_000.0
PRIMARY_FIRM = ("propr", "propr:classic_1step:phase_1", "propr:classic_1step", "Classic 1-Step")
EV_FIRMS: tuple[tuple[str, str, str, str | None], ...] = (
    ("propr", "propr:classic_1step:phase_1", "propr:classic_1step", "Classic 1-Step"),
    (
        "vanta-trading",
        "vanta-trading:classic_1step_3pct:phase_1",
        "vanta-trading:classic_1step_3pct",
        "3% daily",
    ),
    ("hyrotrader", "hyrotrader:two_step:phase_1", "hyrotrader:two_step", None),
)
ARMS: tuple[tuple[str, str, Decimal | None, bool], ...] = (
    ("ARM_A_INDEP_FIXED_K", "Legacy independent suggested_k", None, False),
    ("ARM_B_PORTFOLIO_DD300", "Decoupled portfolio allocator ($300 DD budget)", Decimal(300), False),
    ("ARM_C_PORTFOLIO_DD500", "Decoupled portfolio allocator ($500 DD budget)", Decimal(500), False),
    (
        "ARM_D_PORTFOLIO_DD300_NODUST",
        "Experimental: $300 DD budget + drop sub-min opening legs",
        Decimal(300),
        True,
    ),
)
COST_TIERS: tuple[tuple[str, Decimal, Decimal, Decimal], ...] = (
    ("1x", Decimal("0.00045"), Decimal(30), Decimal(4)),
    ("2x", Decimal("0.00090"), Decimal(60), Decimal(8)),
)


def _alias_for_wallet(wallet: str) -> str:
    digest_hex = hashlib.sha256(wallet.strip().lower().encode("utf-8")).hexdigest()[:8]
    return f"W#{digest_hex}"


def _sanitize_text(text: str, address_to_alias: Mapping[str, str]) -> str:
    if not text:
        return text

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0).lower()
        return address_to_alias.get(token, "<REDACTED_HEX>")

    return HEX_ADDRESS_RE.sub(_replace, text)


def _assert_no_hex_leaks(label: str, text: str) -> None:
    match = HEX_ADDRESS_RE.search(text)
    if match is not None:
        raise RuntimeError(f"HEX_ADDRESS_LEAK:{label}:{match.group(0)[:10]}")


def _resolve_batch_path(store: FactStore, batch: mf.DayBatch) -> Path:
    raw = Path(batch.path)
    return raw if raw.is_absolute() else store.root / raw


def _record_day_batch(
    dst: FactStore,
    *,
    kind: str,
    day: str,
    frame: pl.DataFrame,
    write: Any,
    source_id: str,
) -> None:
    if frame.height > 0 and "first_known_at_ms" in frame.columns:
        non_null = frame["first_known_at_ms"].drop_nulls()
        fk_min = int(non_null.min()) if len(non_null) else None
        fk_max = int(non_null.max()) if len(non_null) else None
    else:
        fk_min = fk_max = None
    with dst.connect() as con:
        mf.record_day(
            con,
            batch=mf.DayBatch(
                kind=kind,
                day=day,
                batch_sha16=write.batch_sha16,
                path=str(write.path),
                rows=write.rows,
                file_sha256=write.file_sha256,
                inputs_fingerprint=write.file_sha256[:16],
            ),
            source_ids=[source_id],
            bytes_count=write.bytes_count,
            inputs=[],
            day_sources=[{"source_id": source_id, "rows_written": write.rows}],
            first_known_min_ms=fk_min,
            first_known_max_ms=fk_max,
            closed_at=datetime.now(UTC).isoformat(),
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


def _discover_clean_cohort(facts_root: Path) -> dict[str, Any]:
    """Select pre-OOS active wallets trading only 100%-mark-complete standard perp coins."""
    oos_win = BacktestWindow.from_utc(OOS_START, OOS_END)
    expected_5m_bars = (oos_win.end_ms - oos_win.start_ms) // GRID_MS + 1

    with FactStore(facts_root, create=False) as src:
        eq_df = (
            src.scan("equity", allow_unknown_known_at=True)
            .frame.select(["wallet", "equity_usd"])
            .collect()
        )
        eq_wallets = sorted(
            {
                str(w).lower()
                for w in eq_df.filter(pl.col("equity_usd") >= MIN_EQUITY_USD)["wallet"]
                .unique()
                .to_list()
            }
        )
        fills_meta = (
            src.scan(
                "fills",
                start_day="2026-05-01",
                end_day=END_DAY,
                allow_unknown_known_at=True,
            )
            .frame.filter(pl.col("wallet").is_in(eq_wallets))
            .select(["wallet", "coin", "event_ts_ms"])
            .collect()
        )
        train_eligible = {
            str(r["wallet"]).lower()
            for r in fills_meta.filter(pl.col("event_ts_ms") < oos_win.start_ms)
            .group_by("wallet")
            .len()
            .filter(pl.col("len") >= MIN_TRAIN_FILLS)
            .iter_rows(named=True)
        }
        sub = fills_meta.filter(pl.col("wallet").is_in(sorted(train_eligible)))
        all_coins = {str(c) for c in sub["coin"].unique().to_list() if c}
        non_std_coins = sorted(c for c in all_coins if any(ch in c for ch in "/: \t"))
        std_coins = sorted(all_coins - set(non_std_coins))

        bars_oos = (
            src.scan(
                "bars_1m",
                coins=sorted({c.upper() for c in std_coins}),
                start_day="2026-08-31",
                end_day=END_DAY,
                allow_unknown_known_at=True,
            )
            .frame.with_columns((pl.col("event_ts_ms") // GRID_MS * GRID_MS).alias("bar_ms"))
            .filter(
                (pl.col("event_ts_ms") >= oos_win.start_ms - GRID_MS)
                & (pl.col("event_ts_ms") + 60_000 <= pl.col("bar_ms") + GRID_MS)
                & (pl.col("bar_ms") < oos_win.end_ms)
            )
            .group_by("coin")
            .agg(pl.col("bar_ms").n_unique().alias("n_5m_bars"))
            .collect()
        )
        complete_upper = {
            str(r["coin"]).upper()
            for r in bars_oos.iter_rows(named=True)
            if int(r["n_5m_bars"]) >= expected_5m_bars
        }

        incomplete_coins = sorted(c for c in std_coins if c.upper() not in complete_upper)
        cohort_wallets: list[str] = []
        excluded_incomplete_wallets = 0
        cohort_coins_raw: set[str] = set()
        for w_key, w_df in sub.group_by("wallet", maintain_order=True):
            wallet = str(w_key[0] if isinstance(w_key, tuple) else w_key).lower()
            w_std_coins = {
                str(c) for c in w_df["coin"].unique().to_list() if c and c not in non_std_coins
            }
            if w_std_coins and w_std_coins.issubset(complete_upper):
                cohort_wallets.append(wallet)
                cohort_coins_raw.update(w_std_coins)
            else:
                excluded_incomplete_wallets += 1

    cohort_wallets.sort()
    return {
        "equity_backed_wallets": len(eq_wallets),
        "train_active_wallets": len(train_eligible),
        "cohort_wallets": cohort_wallets,
        "excluded_incomplete_coin_wallets": excluded_incomplete_wallets,
        "non_std_coins_count": len(non_std_coins),
        "incomplete_bars_1m_coins_count": len(incomplete_coins),
        "complete_coins_count": len(complete_upper),
        "cohort_coins": sorted(cohort_coins_raw),
        "expected_oos_5m_bars": expected_5m_bars,
    }


def _build_isolated_study_store(
    src_root: Path,
    dst_root: Path,
    *,
    wallets: Sequence[str],
    coins: Sequence[str],
) -> dict[str, Any]:
    """Copy study slice into an isolated FactStore covering 2026-07-01..2026-11-18."""
    if dst_root.exists():
        shutil.rmtree(dst_root)
    target_wallets = sorted(set(wallets))
    target_coins_upper = sorted({c.upper() for c in coins})

    fills_days_written = 0
    bars_days_written = 0
    total_fills_rows = 0

    with FactStore(src_root, create=False) as src, FactStore(dst_root, create=True) as dst:
        src_con = src.connect()
        cur_d = date.fromisoformat(FILLS_START_DAY)
        end_d = date.fromisoformat(END_DAY)
        days: list[str] = []
        while cur_d <= end_d:
            days.append(cur_d.isoformat())
            cur_d += timedelta(days=1)

        fills_batches = {
            b.day: b for b in mf.day_batches(src_con, "fills", start_day=FILLS_START_DAY, end_day=END_DAY)
        }
        for day in days:
            batch = fills_batches.get(day)
            if batch is None:
                continue
            df = (
                pl.scan_parquet(_resolve_batch_path(src, batch))
                .filter(pl.col("wallet").is_in(target_wallets))
                .collect()
            )
            if df.height == 0:
                continue
            prepared = dst.prepare_day("fills", day, df)
            write = dst.write_day(prepared)
            _record_day_batch(dst, kind="fills", day=day, frame=prepared.frame, write=write, source_id="cohort-fills")
            fills_days_written += 1
            total_fills_rows += prepared.frame.height

        eq_frames = [
            pl.scan_parquet(_resolve_batch_path(src, b))
            .filter(pl.col("wallet").is_in(target_wallets))
            .collect()
            for b in mf.day_batches(src_con, "equity")
        ]
        non_empty_eq = [frame for frame in eq_frames if frame.height > 0]
        ingestor = FactIngestor(dst)
        if non_empty_eq:
            combined_eq = pl.concat(non_empty_eq, how="vertical_relaxed")
            ingestor.ingest_frame(
                "equity",
                combined_eq,
                source_id="cohort-equity",
                source_rank=1,
                first_known_at_ms=None,
            )
            all_eq = dst.scan("equity", wallets=target_wallets, allow_unknown_known_at=True).collect()
            anchors = _build_equity_anchors(all_eq, EQUITY_ANCHOR_DAYS)
            if anchors:
                ingestor.ingest_rows(
                    "equity",
                    anchors,
                    source_id="equity-anchor",
                    source_rank=1,
                    first_known_at_ms=None,
                )

        bars_batches = {
            b.day: b for b in mf.day_batches(src_con, "bars_1m", start_day=FILLS_START_DAY, end_day=END_DAY)
        }
        for day in days:
            batch = bars_batches.get(day)
            if batch is None:
                continue
            day_df = (
                pl.scan_parquet(_resolve_batch_path(src, batch))
                .filter(pl.col("coin").str.strip_chars().str.to_uppercase().is_in(target_coins_upper))
                .collect()
            )
            if day_df.height == 0:
                continue
            prepared = dst.prepare_day("bars_1m", day, day_df)
            write = dst.write_day(prepared)
            _record_day_batch(
                dst, kind="bars_1m", day=day, frame=prepared.frame, write=write, source_id="cohort-bars"
            )
            bars_days_written += 1

    return {
        "days_in_slice": len(days),
        "fills_days_written": fills_days_written,
        "bars_days_written": bars_days_written,
        "total_fills_rows": total_fills_rows,
    }


def _summarize_warnings(warnings: Sequence[str], address_to_alias: Mapping[str, str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for raw in warnings:
        prefix = _sanitize_text(raw, address_to_alias).split(":", 1)[0]
        counts[prefix] += 1
    return dict(sorted(counts.items()))


def _baseline_stage1_diagnostics(
    store_path: Path, address_to_alias: Mapping[str, str]
) -> dict[str, Any]:
    window = BacktestWindow.from_utc(OOS_START, OOS_END)
    as_of = KnownAtTs(window.start_ms)
    with FactStore(store_path, create=False) as store:
        strict_pit = score_universe(
            store,
            as_of=as_of,
            lookback_days=TRAIN_LOOKBACK_DAYS,
            allow_undated=False,
        )
        legacy_sc = score_universe(
            store,
            as_of=as_of,
            lookback_days=TRAIN_LOOKBACK_DAYS,
            config=ScoreConfig(expectancy_mode="trajectory", decouple_sizing_gates=False),
            allow_undated=True,
        )
        decoupled_sc = score_universe(
            store,
            as_of=as_of,
            lookback_days=TRAIN_LOOKBACK_DAYS,
            config=ScoreConfig(expectancy_mode="trajectory", decouple_sizing_gates=True),
            allow_undated=True,
        )

    plan_dd300 = allocate_fixed_k_portfolio(
        decoupled_sc,
        policy=AllocationPolicy(
            pod_equity_usd=Decimal(10_000),
            risk_unit_usd=Decimal(200),
            portfolio_dd_budget_usd=Decimal(300),
            max_euler_cdar_share=Decimal("0.45"),
            max_p95_participation=Decimal("0.01"),
            max_max_participation=Decimal("0.05"),
        ),
    )
    plan_dd500 = allocate_fixed_k_portfolio(
        decoupled_sc,
        policy=AllocationPolicy(
            pod_equity_usd=Decimal(10_000),
            risk_unit_usd=Decimal(200),
            portfolio_dd_budget_usd=Decimal(500),
            max_euler_cdar_share=Decimal("0.45"),
            max_p95_participation=Decimal("0.01"),
            max_max_participation=Decimal("0.05"),
        ),
    )
    def _plan_dd_usd(plan: Any) -> float:
        k_map = plan.k_by_wallet
        if not k_map:
            return 0.0
        matrix = np.array([decoupled_sc.wallet_returns(w) for w in sorted(k_map)], dtype=np.float64)
        weights = np.array([float(k_map[w]) for w in sorted(k_map)], dtype=np.float64)
        return round(float(peak_drawdown((matrix * weights[:, None]).sum(axis=0)) * 10_000.0), 2)

    return {
        "strict_pit_cards_count": len(strict_pit.cards),
        "strict_pit_basis": strict_pit.basis,
        "day0_legacy_eligible": [address_to_alias.get(c.wallet, c.wallet) for c in legacy_sc.eligible],
        "day0_decoupled_eligible": [
            address_to_alias.get(c.wallet, c.wallet) for c in decoupled_sc.eligible
        ],
        "day0_legacy_k": {
            address_to_alias.get(c.wallet, c.wallet): float(c.suggested_k) for c in legacy_sc.eligible
        },
        "day0_dd300_k": {
            address_to_alias.get(w, w): float(k) for w, k in plan_dd300.k_by_wallet.items()
        },
        "day0_dd300_portfolio_dd_usd": _plan_dd_usd(plan_dd300),
        "day0_dd500_k": {
            address_to_alias.get(w, w): float(k) for w, k in plan_dd500.k_by_wallet.items()
        },
        "day0_dd500_portfolio_dd_usd": _plan_dd_usd(plan_dd500),
        "day0_legacy_failures": dict(
            Counter(
                gate.value for card in legacy_sc.cards for gate in card.failures if not card.eligible
            )
        ),
        "day0_decoupled_failures": dict(
            Counter(
                gate.value for card in decoupled_sc.cards for gate in card.failures if not card.eligible
            )
        ),
        "day0_exclusions": dict(Counter(row.reason for row in decoupled_sc.excluded)),
    }


def _evaluate_arm_task(
    arm_id: str,
    arm_desc: str,
    dd_budget: Decimal | None,
    drop_sub_min_opening: bool,
    cost_label: str,
    fee_rate: Decimal,
    shortfall_bps: Decimal,
    spread_bps: Decimal,
    cohort_store_path: Path,
    cohort_coins: tuple[str, ...],
    address_to_alias: dict[str, str],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    firm_slug, catalog_key, _, _ = PRIMARY_FIRM
    window = BacktestWindow.from_utc(OOS_START, OOS_END)
    seats = tuple(SeatSpec(f"seat_{i}", Decimal(200), Decimal("0.45")) for i in range(1, 4))
    markets = tuple(MarketSpec(coin, fee_rate=fee_rate) for coin in cohort_coins)
    account_id = f"{firm_slug}-10k-{arm_id.lower()}"
    ev_sha = hashlib.sha256(f"{arm_id}:{cost_label}".encode()).hexdigest()
    rulebook = Rulebook(
        **RulebookCatalog.get(catalog_key).to_rulebook_fields(
            account_binding=account_id,
            scale=2,
            evidence_sha=ev_sha,
        )
    )
    score_cfg = ScoreConfig(
        expectancy_mode="trajectory",
        decouple_sizing_gates=(dd_budget is not None),
    )
    alloc_policy = (
        None
        if dd_budget is None
        else AllocationPolicy(
            pod_equity_usd=Decimal(10_000),
            risk_unit_usd=Decimal(200),
            portfolio_dd_budget_usd=dd_budget,
            max_euler_cdar_share=Decimal("0.45"),
            max_p95_participation=Decimal("0.01"),
            max_max_participation=Decimal("0.05"),
        )
    )
    cfg = BacktestConfig(
        account_id=account_id,
        initial_equity=Decimal(10_000),
        lookback_days=TRAIN_LOOKBACK_DAYS,
        shortfall_bps=shortfall_bps,
        quote_spread_bps=spread_bps,
        score_config=score_cfg,
        allow_undated=True,
        allocation_policy=alloc_policy,
        drop_sub_min_opening_legs=drop_sub_min_opening,
    )

    with tempfile.TemporaryDirectory(prefix=f"{arm_id.lower()}_{cost_label}_") as tmp_str:
        tmp_dir = Path(tmp_str)
        worker_store = tmp_dir / "facts"
        shutil.copytree(cohort_store_path, worker_store)
        roster_path = tmp_dir / "roster.sqlite"

        with FactStore(worker_store, create=False) as facts:
            roster = RosterStore(roster_path, create=True)
            try:
                report: BacktestReport = run_kernel_backtest(
                    facts,
                    window=window,
                    seats=seats,
                    rulebook=rulebook,
                    markets=markets,
                    config=cfg,
                    roster=roster,
                )
            finally:
                roster.close()

            diag = summarize_backtest_diagnostics(
                report,
                facts=facts,
                window=window,
                initial_equity=Decimal(10_000),
                chop_start_day=CHOP_START_DAY,
                allow_undated=True,
            )

    ev_by_firm: dict[str, dict[str, Any]] = {}
    for ev_slug, _, sim_key, program_filter in EV_FIRMS:
        quote = lookup_tier_fee(sim_key, ACCOUNT_USD, path=DEFAULT_FEE_SCHEDULE_PATH)
        fee_usd = quote.fee_usd if quote is not None else None
        refundable = quote.refundable_on_pass if quote is not None else False
        contract = replace(
            pc.load_prop_contract_spec(
                ev_slug,
                account_usd=ACCOUNT_USD,
                challenge_fee_usd=fee_usd,
                program_filter=program_filter,
            ),
            fee_refunded_on_first_payout=refundable,
        )
        rng = np.random.default_rng(20261127)
        ev_by_firm[ev_slug] = {
            "fee_usd": fee_usd,
            "refundable": refundable,
            "blended": pc.evaluate_multiphase_contract_ev(
                diag["blocks"], contract, regime="Blended", n_paths=5000, rng=rng
            ),
            "chop_75": pc.evaluate_multiphase_contract_ev(
                diag["blocks"], contract, regime="75%-Chop", n_paths=5000, rng=rng
            ),
        }

    net_pnl = round(float(report.ending_equity - Decimal(10_000)), 2)
    p2t_dd = float(diag["peak_to_trough_dd_usd"])
    floor_dd = float(diag["static_floor_dd_usd"])
    ret_over_p2t = round(net_pnl / p2t_dd, 4) if p2t_dd > 0 else (9999.0 if net_pnl > 0 else 0.0)
    ret_over_floor = (
        round(net_pnl / floor_dd, 4) if floor_dd > 0 else (9999.0 if net_pnl > 0 else 0.0)
    )
    gross_traded_usd = round(
        sum(abs(float(f.net_delta_qty)) * float(f.executable_price) for f in report.fills), 2
    )
    total_costs_usd = round(float(report.ledger[-1].cumulative_costs), 2) if report.ledger else 0.0
    total_realized_usd = round(float(report.ledger[-1].realized_pnl), 2) if report.ledger else 0.0
    decisions = report.decisions
    warn_counts = _summarize_warnings(report.warnings, address_to_alias)

    return {
        "arm_id": arm_id,
        "arm_desc": arm_desc,
        "dd_budget_usd": float(dd_budget) if dd_budget is not None else None,
        "drop_sub_min_opening_legs": drop_sub_min_opening,
        "cost_label": cost_label,
        "ending_equity_usd": round(float(report.ending_equity), 2),
        "net_pnl_usd": net_pnl,
        "realized_pnl_usd": total_realized_usd,
        "cumulative_costs_usd": total_costs_usd,
        "gross_traded_notional_usd": gross_traded_usd,
        "net_fill_count": len(report.fills),
        "peak_to_trough_dd_usd": p2t_dd,
        "static_floor_dd_usd": floor_dd,
        "return_over_p2t_dd": ret_over_p2t,
        "return_over_floor_dd": ret_over_floor,
        "challenge_outcome": report.challenge_outcome,
        "survival_no_breach": diag["survival_no_breach"],
        "challenge_target_reached": report.challenge_target_reached,
        "portfolio_constraints_valid": report.portfolio_constraints_valid,
        "allocation_mode_share": dict(sorted(report.allocation_mode_share.items())),
        "participation": diag["participation"],
        "warning_prefix_counts": warn_counts,
        "ev_by_firm": ev_by_firm,
        "decision_count": int(report.decision_count),
        "first_decision_sha": decisions[0].decision_sha if decisions else None,
        "last_decision_sha": decisions[-1].decision_sha if decisions else None,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


def _render_markdown(results: Mapping[str, Any]) -> str:
    cohort = results["cohort"]
    slice_info = results["isolated_slice"]
    stage1 = results["stage1_day0"]
    rows = results["arms"]
    by_key = {(r["arm_id"], r["cost_label"]): r for r in rows}

    lines = [
        "---",
        "firm: multi (propr:classic_1step, vanta-trading:classic_1step_3pct, hyrotrader:two_step)",
        "account_usd: 10000",
        f"status: {results['verdict']}",
        "provenance: RETROSPECTIVE_RESEARCH_ONLY",
        f"top_candidate: {results['top_candidate']}",
        "script: studies/2026-11-27__multi__10k__portfolio-sizing-ab-study.py",
        "results_json: reports/2026-11-27__multi__10k__portfolio-sizing-ab-study__results.json",
        "---",
        "",
        "# Controlled A/B Study: Portfolio Fixed-k Allocation vs. Independent Fixed-k ($10k Prop Challenge)",
        "",
        f"**Verdict: `{results['verdict']}`** (`RETROSPECTIVE_RESEARCH_ONLY` — `strict_pit_cards_count = {stage1['strict_pit_cards_count']}` on undated Reservoir data).",
        "",
        "## 1. Cohort & FactStore Provenance",
        "",
        f"- **Cohort Selection (pre-OOS Train `2026-05-01..2026-08-31`):** `{cohort['equity_backed_wallets']}` equity-backed wallets (`>= $1,000`), `{cohort['train_active_wallets']}` with `>= 50` Train fills, **`{len(cohort['cohort_aliases'])}`** wallets trading only 100%-complete `bars_1m` standard perp coins (`{len(cohort['cohort_coins'])}` coins, `22,465 / 22,465` 5m bars across OOS `[2026-09-01 - 5m, 2026-11-18)`). Excluded `{cohort['excluded_incomplete_coin_wallets']}` wallets with incomplete `bars_1m` tickers and `{cohort['non_std_coins_count']}` non-perp tickers.",
        f"- **Isolated FactStore Slice (`{FILLS_START_DAY}..{END_DAY}`):** `{slice_info['total_fills_rows']:,}` fills (`{slice_info['fills_days_written']}` days), `{slice_info['bars_days_written']}` `bars_1m` days.",
        "- **Day-0 (`2026-09-01`, 60d lookback) Stage-1 Comparison:**",
        f"  - Legacy `decouple_sizing_gates=False` eligible (`{len(stage1['day0_legacy_eligible'])}`): `{stage1['day0_legacy_eligible']}` (`k={stage1['day0_legacy_k']}`).",
        f"  - Decoupled `decouple_sizing_gates=True` eligible (`{len(stage1['day0_decoupled_eligible'])}`): `{stage1['day0_decoupled_eligible']}`.",
        f"  - Day-0 `$300` allocator (`dd=${stage1['day0_dd300_portfolio_dd_usd']}`): `k={stage1['day0_dd300_k']}`.",
        f"  - Day-0 `$500` allocator (`dd=${stage1['day0_dd500_portfolio_dd_usd']}`): `k={stage1['day0_dd500_k']}`.",
        "",
        "## 2. Controlled 4-Arm × 2-Cost-Stress Kernel Walk-Forward (`2026-09-01..2026-11-18`, 78d)",
        "",
        "| Arm | Cost | Net PnL ($) | Costs ($) | Fills | P2T DD ($) | Floor DD ($) | Ret/P2T | Valid / Surv | Mode (`A/P/F`) | 1m Part (`med / p95 / max`, `unk`) | Propr Blended EV ($) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|",
    ]
    for row in rows:
        part = row["participation"]
        p_str = (
            f"{part['median_1m']:.4%} / {part['p95_1m']:.4%} / {part['max_1m']:.4%}"
            if part["max_1m"] is not None
            else "UNKNOWN"
        )
        m = row["allocation_mode_share"]
        mode_str = f"{m.get('ACTIVE', 0.0):.2f} / {m.get('PARTIAL', 0.0):.2f} / {m.get('FLAT', 0.0):.2f}"
        propr_ev = row["ev_by_firm"]["propr"]["blended"]["net_ev_usd"]
        lines.append(
            f"| `{row['arm_id']}` | `{row['cost_label']}` | **${row['net_pnl_usd']:+.2f}** | "
            f"${row['cumulative_costs_usd']:.2f} | {row['net_fill_count']} | "
            f"${row['peak_to_trough_dd_usd']:.2f} | ${row['static_floor_dd_usd']:.2f} | "
            f"{row['return_over_p2t_dd']:.2f}x | `{row['portfolio_constraints_valid']}` / `{row['challenge_outcome']}` | "
            f"`{mode_str}` | `{p_str}`, unk={part['unknown_vol_trades']} | ${propr_ev:.2f} |"
        )

    lines.extend(["", "## 3. Head-to-Head Summary (2x Cost Stress)", ""])
    a2 = by_key[("ARM_A_INDEP_FIXED_K", "2x")]
    for arm_id in ("ARM_B_PORTFOLIO_DD300", "ARM_C_PORTFOLIO_DD500", "ARM_D_PORTFOLIO_DD300_NODUST"):
        b2 = by_key[(arm_id, "2x")]
        dpnl = round(b2["net_pnl_usd"] - a2["net_pnl_usd"], 2)
        ddd = round(b2["peak_to_trough_dd_usd"] - a2["peak_to_trough_dd_usd"], 2)
        lines.append(
            f"- **`{arm_id}` vs `ARM_A_INDEP_FIXED_K` (2x):** "
            f"Net PnL delta `${dpnl:+.2f}`, P2T DD delta `${ddd:+.2f}`, "
            f"Ret/P2T `{b2['return_over_p2t_dd']:.2f}x` vs `{a2['return_over_p2t_dd']:.2f}x`, "
            f"first..last SHA `{b2['first_decision_sha'][:12]}..{b2['last_decision_sha'][:12]}`."
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "**Qualification & Compliance Limits:** `RETROSPECTIVE_RESEARCH_ONLY` — Reservoir fills have `first_known_at_ms = NULL` (`strict_pit_cards_count = 0`) and cannot authorize live trading; equity anchors project historical snapshots into the OOS slice. Arms pass research qualification only when `portfolio_constraints_valid=True`, `survival_no_breach=True`, `capacity_evidence_complete=True` (`0` unknown volume trades, `<=1%` p95 / `<=5%` max 1m participation), and positive `2x`-cost EV. Live `~/.local/share/copytrade/facts`, `roster.sqlite`, and `prop-production` remain strictly untouched and un-mutated.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Portfolio allocation vs independent fixed-k A/B study")
    parser.add_argument("--facts-root", type=Path, default=DEFAULT_FACTS_ROOT)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args(list(argv) if argv is not None else None)

    t0 = time.perf_counter()
    cohort = _discover_clean_cohort(args.facts_root)
    wallets = list(cohort["cohort_wallets"])
    if not wallets:
        raise RuntimeError("EMPTY_CLEAN_COHORT")
    address_to_alias = {w: _alias_for_wallet(w) for w in wallets}

    with tempfile.TemporaryDirectory(prefix="portfolio_ab_slice_") as slice_tmp:
        cohort_store_path = Path(slice_tmp) / "facts"
        slice_info = _build_isolated_study_store(
            args.facts_root,
            cohort_store_path,
            wallets=wallets,
            coins=cohort["cohort_coins"],
        )
        stage1_day0 = _baseline_stage1_diagnostics(cohort_store_path, address_to_alias)

        tasks = [
            (arm_id, arm_desc, dd_budget, drop_dust, cost_label, fee_rate, shortfall_bps, spread_bps)
            for arm_id, arm_desc, dd_budget, drop_dust in ARMS
            for cost_label, fee_rate, shortfall_bps, spread_bps in COST_TIERS
        ]
        workers = min(max(1, args.max_workers), len(tasks), os.cpu_count() or 2)
        mp_ctx = multiprocessing.get_context("spawn")
        arm_results: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as pool:
            futs = {
                pool.submit(
                    _evaluate_arm_task,
                    arm_id,
                    arm_desc,
                    dd_budget,
                    drop_dust,
                    cost_label,
                    fee_rate,
                    shortfall_bps,
                    spread_bps,
                    cohort_store_path,
                    tuple(cohort["cohort_coins"]),
                    address_to_alias,
                ): (arm_id, cost_label)
                for (
                    arm_id,
                    arm_desc,
                    dd_budget,
                    drop_dust,
                    cost_label,
                    fee_rate,
                    shortfall_bps,
                    spread_bps,
                ) in tasks
            }
            for fut in as_completed(futs):
                res = fut.result()
                arm_results.append(res)
                p_max = res["participation"]["max_1m"]
                p_str = f"{p_max:.4%}" if p_max is not None else "UNKNOWN"
                print(
                    f"[done] {res['arm_id']} {res['cost_label']}: net=${res['net_pnl_usd']:+.2f} "
                    f"p2t=${res['peak_to_trough_dd_usd']:.2f} fills={res['net_fill_count']} "
                    f"valid={res['portfolio_constraints_valid']} surv={res['survival_no_breach']} "
                    f"max_part={p_str} ({res['elapsed_sec']}s)",
                    flush=True,
                )

    arm_order = {arm_id: idx for idx, (arm_id, _, _, _) in enumerate(ARMS)}
    arm_results.sort(key=lambda r: (arm_order[r["arm_id"]], r["cost_label"]))

    qualifying_2x = [
        r
        for r in arm_results
        if r["cost_label"] == "2x"
        and r["net_pnl_usd"] > 0.0
        and r["survival_no_breach"]
        and r["portfolio_constraints_valid"]
        and r["participation"]["capacity_evidence_complete"]
        and r["participation"]["n_trades"] > 0
        and r["participation"]["max_1m"] is not None
        and r["participation"]["max_1m"] <= 0.05
        and r["participation"]["p95_1m"] is not None
        and r["participation"]["p95_1m"] <= 0.01
        and any(float(v["blended"].get("net_ev_usd") or -9999.0) > 0.0 for v in r["ev_by_firm"].values())
    ]
    if qualifying_2x:
        best = max(qualifying_2x, key=lambda r: r["return_over_p2t_dd"])
        verdict = "PORTFOLIO_ALLOCATOR_OUTPERFORMS" if best["arm_id"] != "ARM_A_INDEP_FIXED_K" else "INDEPENDENT_FIXED_K_LEADS"
        top_candidate = f"{best['arm_id']} (2x net=${best['net_pnl_usd']:+.2f}, p2t=${best['peak_to_trough_dd_usd']:.2f})"
    else:
        best_2x = max(
            (r for r in arm_results if r["cost_label"] == "2x"),
            key=lambda r: (r["portfolio_constraints_valid"], r["return_over_p2t_dd"], r["net_pnl_usd"]),
        )
        verdict = "NO_QUALIFYING_2X_ARM"
        top_candidate = f"{best_2x['arm_id']} (2x net=${best_2x['net_pnl_usd']:+.2f}, p2t=${best_2x['peak_to_trough_dd_usd']:.2f})"

    sanitized_cohort = {k: v for k, v in cohort.items() if k != "cohort_wallets"}
    sanitized_cohort["cohort_aliases"] = sorted(address_to_alias.values())
    payload = {
        "schema": "portfolio_sizing_ab_study_v1",
        "provenance": "RETROSPECTIVE_RESEARCH_ONLY",
        "oos_window": [OOS_START.isoformat(), OOS_END.isoformat()],
        "lookback_days": TRAIN_LOOKBACK_DAYS,
        "verdict": verdict,
        "top_candidate": top_candidate,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "cohort": sanitized_cohort,
        "isolated_slice": slice_info,
        "stage1_day0": stage1_day0,
        "arms": arm_results,
    }
    json_text = json.dumps(payload, indent=2)
    md_text = _render_markdown(payload)
    _assert_no_hex_leaks("results_json", json_text)
    _assert_no_hex_leaks("report_md", md_text)

    REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON_PATH.write_text(json_text + "\n", encoding="utf-8")
    REPORT_MD_PATH.write_text(md_text, encoding="utf-8")
    print(f"Wrote {REPORT_MD_PATH} and {REPORT_JSON_PATH} ({verdict}, {payload['elapsed_sec']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
