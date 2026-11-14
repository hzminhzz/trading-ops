"""Causal *short-window* native CWSA -> PM03 local-Hyperliquid-1m follower replay.

No historical fills reconstructed into source positions. No Bybit data.
Official checkpoint outcomes and on-disk original REST books are the signal source.
Uses native target_from_fixed_k and native PM03 execute_targets/account.
Only two 2026-11 days overlap available causal CWSA and local 1m bars.
Simulation is RESEARCH_PROPOSAL / SCENARIO_ONLY, never signed mandates.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import tempfile
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal as D
from pathlib import Path

import polars as pl
from copytrade_pm.pm03_account import Account, Instrument, Rulebook
from copytrade_pm.pm03_execution import CostModel, MarketObservation, execute_targets
from copytrade_pm.pm_checkpoint_journal import CheckpointOutcome, verify_candidate
from copytrade_pm.pm_checkpoint_paper import _marks as official_source_marks
from copytrade_pm.pm_cwsa_chapter2_gate import _source_facts
from copytrade_pm.pm_exposure_sizing import target_from_fixed_k
from copytrade_pm.pm_sqlite_readonly import connect_readonly
from copytrade_pm.pm_wire_evidence import read_wire
from cwsa_native_prop10k_audit import PORTFOLIO_CONFIGS, RESEARCH_ROSTERS, SOURCE

LOCAL_1M = Path("/home/quant/dev/quant/ml4t-data/.data/store")
SUPPLEMENTAL_1M = Path("data/private/hyperliquid_candles_1m_20261112_13")
START = datetime(2026, 11, 12, 5, 10, tzinfo=UTC)
SPLIT = datetime(2026, 11, 13, tzinfo=UTC)
END = datetime(2026, 11, 14, tzinfo=UTC)
MINUTE_MS = 60_000
GRID_MS = 300_000
K_GRID = (D(0), D("0.00002"), D("0.00005"), D("0.0001"), D("0.0002"), D("0.0005"), D("0.001"))
START_CAPITAL = D(10000)
TRAIN_DAILY_CAP = D(150)
TRAIN_PEAK_CAP = D(250)
TRAIN_MIN_EXECUTION_SHARE = D("0.70")
GROSS_CAP = D(15000)
MARKET_CAP = D(5000)
POD_LOSS_BUDGET = D(100)
STRESS_RESERVE = D(50)
MIN_ORDER_NOTIONAL = D(10)  # research scenario, not independently pinned venue rules
SOURCE_MARKET_PREFIX = "hyperliquid/mainnet:perp:default/"
REPORT = Path("reports/native-1m-economic-replay-20261114.json")
PROPOSAL = Path("proposals/native-hl-1m-three-20261114.json")
ROSTERS = {
    "regime.macro/TriadA": tuple(PORTFOLIO_CONFIGS["regime.macro/TriadA"]["members"]),
    **RESEARCH_ROSTERS,
}


def mill(t: datetime) -> int:
    return int(t.timestamp() * 1000)


def stamp(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat()


def decimal(v: float | str | D) -> D:
    x = v if isinstance(v, D) else D(str(v))
    if not x.is_finite():
        raise ValueError("NON_FINITE_PRICE_OR_QUANTITY")
    return x


def first_native_decision(published_ms: int) -> int:
    """Single attempt just after verified publication, using PREVIOUS 1m close/volume.

    An execution at next-minute open commonly violates native CWSA source-age
    (45 seconds). The complete *previous* bar is knowable at publication;
    the current minute high/low/close is NOT used in the execution decision.
    """
    return published_ms + 1


def check_bar_row(row: dict) -> None:
    """Invalid bars are missing, not flat or fictitious quotes."""
    prices = [row["open"], row["high"], row["low"], row["close"]]
    if not all(math.isfinite(x) and x > 0 for x in prices):
        raise ValueError("INVALID_LOCAL_OHLCV")
    if (
        not row["low"]
        <= min(row["open"], row["close"])
        <= max(row["open"], row["close"])
        <= row["high"]
    ):
        raise ValueError("LOCAL_OHLCV_CANDLE_CONTRADICTION")
    if not math.isfinite(row["volume"]) or row["volume"] < 0:
        raise ValueError("INVALID_LOCAL_VOLUME")


def load_local_bars(symbols: set[str]) -> tuple[dict[str, dict[int, tuple[D, D, D]]], dict]:
    """Open and close *existing* local Hyperliquid one-minute bars; no external query."""
    out = {}
    proofs = {}
    for symbol in sorted(symbols):
        folder = LOCAL_1M / f"crypto_futures_ohlcv_1m_HYPERLIQUID_{symbol}_PERP"
        parts = [folder / f"year=2026/month=11/day={day:02d}/data.parquet" for day in (12, 13)]
        if all(path.is_file() for path in parts):
            frame = pl.read_parquet(parts).sort("timestamp")
            proof = {
                "status": "CANONICAL_ML4T_LOCAL_HYPERLIQUID_1M",
                "day12_sha256": hashlib.sha256(parts[0].read_bytes()).hexdigest(),
                "day13_sha256": hashlib.sha256(parts[1].read_bytes()).hexdigest(),
            }
        else:
            local = SUPPLEMENTAL_1M / f"{symbol}.parquet"
            manifest_path = SUPPLEMENTAL_1M / "manifest.json"
            if not local.is_file() or not manifest_path.is_file():
                proofs[symbol] = {"status": "MISSING_LOCAL_1M_PARTITION"}
                continue
            manifest = json.loads(manifest_path.read_text())
            rec = manifest.get("coin", {}).get(symbol, {})
            if (
                manifest.get("provider") != "official_hyperliquid_candleSnapshot"
                or rec.get("parquet_sha256") != hashlib.sha256(local.read_bytes()).hexdigest()
                or rec.get("rows") != 2880
            ):
                raise ValueError("SUPPLEMENTAL_LOCAL_1M_PROVENANCE_FAILED")
            frame = pl.read_parquet(local).sort("timestamp")
            proof = {
                "status": "SUPPLEMENTAL_VERIFIED_OFFICIAL_HL_LOCAL_1M",
                "parquet_sha256": rec["parquet_sha256"],
            }
        key: dict[int, tuple[D, D, D]] = {}
        for row in frame.iter_rows(named=True):
            ts = mill(row["timestamp"])
            if ts < mill(START) - MINUTE_MS or ts >= mill(END):
                continue
            check_bar_row(row)
            if ts in key:
                raise ValueError("DUPLICATE_1M_BAR")
            key[ts] = (decimal(row["open"]), decimal(row["close"]), decimal(row["volume"]))
        out[symbol] = key
        proofs[symbol] = {
            **proof,
            "rows": len(key),
            "start_ms": min(key) if key else None,
            "end_ms": max(key) if key else None,
        }
    return out, proofs


@dataclass(frozen=True)
class Wave:
    checkpoint_ms: int
    at_ms: int
    changes: dict[str, CheckpointOutcome]
    complete: bool
    reasons: tuple[str, ...]


def source_waves(db_snapshot: Path) -> tuple[dict[int, dict[str, CheckpointOutcome]], dict]:
    """Use native CWSA source owner, proof-bound original positions and receipt clocks."""
    facts = _source_facts(db_snapshot)
    rows: dict[int, dict[str, CheckpointOutcome]] = defaultdict(dict)
    accepted = Counter()
    with closing(connect_readonly(db_snapshot)) as db:
        db.execute("BEGIN")
        for (raw,) in db.execute("SELECT outcome FROM cwsa_outcomes ORDER BY checkpoint_ms"):
            outcome = CheckpointOutcome.model_validate_json(raw)
            cp = outcome.checkpoint_ms
            if cp < mill(START) or cp >= mill(END):
                continue
            if cp % GRID_MS or outcome.source_id in rows[cp]:
                raise ValueError("CWSA_GRID_OR_DUPLICATE")
            if outcome.kind in ("CHANGED", "BASELINE", "RECOVERY_BASELINE"):
                if not outcome.sample or not verify_candidate(outcome.sample, SOURCE.parent):
                    raise ValueError("CWSA_ORIGINAL_SOURCE_PROOF_INVALID")
                accepted["original_verified"] += 1
            if (
                outcome.kind
                in ("UNAVAILABLE", "QUARANTINE", "RECOVERY_BASELINE", "BASELINE", "UNCHANGED")
                and outcome.signal_id
            ):
                raise ValueError("NONCHANGED_SOURCE_SIGNAL")
            if outcome.kind == "CHANGED" and not outcome.signal_id:
                raise ValueError("MISSING_SOURCE_SIGNAL")
            if outcome.kind in ("BASELINE", "CHANGED", "UNCHANGED", "RECOVERY_BASELINE") and (
                outcome.sample is None or outcome.published_ms < outcome.sample.received_ms
            ):
                raise ValueError("CWSA_SAMPLE_CAUSALITY")
            accepted[outcome.kind] += 1
            rows[cp][outcome.source_id] = outcome
        db.execute("ROLLBACK")
    return dict(rows), {
        "native_head_sha256": facts["head_sha256"],
        "checkpoints": len(rows),
        "proof_counts": dict(accepted),
    }


def get_marks(o: CheckpointOutcome) -> dict[str, D]:
    """Official REST positionValue/quantity source marks via native adapter."""
    if o.sample is None:
        raise ValueError("SOURCE_SAMPLE_MISSING")
    originals = {r.raw_sha: read_wire(SOURCE, r.raw_sha) for r in o.sample.receipts}
    marks = official_source_marks(o, originals)
    if set(marks) != {p.instrument for p in o.positions}:
        raise ValueError("MISSING_NATIVE_SOURCE_MARK")
    return {k: decimal(v) for k, v in marks.items()}


def build_events(
    raw: dict[int, dict[str, CheckpointOutcome]], members: tuple[str, ...]
) -> tuple[list[Wave], Counter]:
    """Replay owner-specific CWSA checkpoint publications in causal arrival order.

    Canonical Copy Mandates consume a wallet independently, not after all
    three wallets have published a checkpoint. Other owners must already
    have verified recent source observations; missing/recovery is not flat.
    """
    counters = Counter()
    events = []
    timeline = sorted(
        (o.published_ms, o.source_id, o)
        for cp, books in raw.items()
        for alias, o in books.items()
        if alias in members
    )
    latest = {}
    for published, owner, o in timeline:
        latest[owner] = o
        if o.kind != "CHANGED":
            if o.kind in ("BASELINE", "RECOVERY_BASELINE", "UNAVAILABLE", "QUARANTINE"):
                counters["nontrade_native_baseline_gap_recovery"] += 1
            continue
        if any(
            member not in latest
            or latest[member].kind not in ("BASELINE", "RECOVERY_BASELINE", "CHANGED", "UNCHANGED")
            or latest[member].checkpoint_ms < o.checkpoint_ms - GRID_MS
            for member in members
        ):
            counters["blocked_other_wallet_not_yet_verified"] += 1
            continue
        decision = first_native_decision(published)
        deadline = min(
            o.checkpoint_ms + GRID_MS,
            min(r.state_ms for r in o.sample.receipts) + o.sample.limits.max_state_age_ms,
        )
        if decision >= deadline:
            counters["expired_native_opportunity"] += 1
            continue
        if not mill(START) <= decision < mill(END):
            counters["outside_local_overlap"] += 1
            continue
        events.append(Wave(o.checkpoint_ms, decision, {owner: o}, True, ()))
        counters["actionable_native_source_waves"] += 1
    return events, counters


def market_symbols(events: list[Wave]) -> set[str]:
    return {
        p.instrument.removeprefix(SOURCE_MARKET_PREFIX)
        for ev in events
        for o in ev.changes.values()
        for p in o.positions
        if p.quantity and p.instrument.startswith(SOURCE_MARKET_PREFIX)
    }


def model_cost(factor: D) -> CostModel:
    return CostModel(
        fee_bps=D("4.5") * factor,
        full_spread_bps=D(2) * factor,
        slippage_bps=D(2) * factor,
        impact_bps_at_full_participation=D(5) * factor,
        max_participation=D("0.1"),
        # A fully closed one-minute OHLCV bar is necessarily older than an
        # actual L2 quote. This 90s limit is research-only, NOT production L2.
        max_age_seconds=90,
        latency_seconds=0,
    )


def rulebook(members: tuple[str, ...], start: datetime) -> Rulebook:
    return Rulebook(
        program="OWNER_10K_STATIC_5PCT_RESEARCH_SCENARIO",
        stage="evaluation",
        source="USER_SPECIFIED_NOT_SIGNED",
        observed_at=start.isoformat(),
        version="research-2026-11-14",
        timezone="UTC",
        initial_balance=START_CAPITAL,
        profit_target=None,
        daily_loss=D(300),
        static_loss=D(500),
        trailing_loss=None,
        permitted=(),
        confirmed=True,
        account_terms_confirmed=False,
        max_gross_leverage=D("1.5"),
    )


def simulate(
    members: tuple[str, ...],
    events: list[Wave],
    bars: dict[str, dict[int, tuple[D, D, D]]],
    *,
    k: D,
    factor: D = D(1),
    min_order_notional: D | None = MIN_ORDER_NOTIONAL,
    stop_ms: int = mill(END),
) -> dict:
    """Native one-shot publication-time intent, prior completed 1m close proxy, PM03 MTM."""
    t0 = mill(START)
    account = Account.start(rulebook(members, START), START.isoformat())
    # Original native instrument keys retained across virtual paper and source.
    permitted = tuple(
        sorted(
            (Instrument.parse(SOURCE_MARKET_PREFIX + s) for s in bars),
            key=lambda x: x.key,
        )
    )
    account.rulebook = Rulebook(**{**account.rulebook.__dict__, "permitted": permitted})
    costs = model_cost(factor)
    by_minute: dict[int, list[Wave]] = defaultdict(list)
    for ev in events:
        if ev.at_ms < stop_ms and k > 0:
            by_minute[(ev.at_ms // MINUTE_MS) * MINUTE_MS].append(ev)
    for bucket in by_minute.values():
        bucket.sort(key=lambda ev: ev.at_ms)
    stats = Counter()
    fills_by_member = Counter()
    pnl_curve = []
    previous_peak = START_CAPITAL
    max_dd = D(0)
    min_equity = START_CAPITAL
    worst_day_loss = D(0)
    max_gross = D(0)
    day_open = START_CAPITAL
    last_date = None
    blocked_equity = False
    current_bad = None
    for ts in range(t0, stop_ms, MINUTE_MS):
        bar_dt = datetime.fromtimestamp(ts / 1000, UTC)
        prev_time = ts - MINUTE_MS
        # Consume only existing local bar data. If a held market cannot be
        # marked, risk is unknown; fail-closed on all future new risk.
        held = {inst for inst, q in account.quantities.items() if q}
        for ev in by_minute.get(ts, []):
            if blocked_equity or account.breached:
                stats["blocked_halted"] += 1
                continue
            sources = {}
            fail = None
            for owner, outcome in ev.changes.items():
                if any(
                    not p.instrument.startswith(SOURCE_MARKET_PREFIX) for p in outcome.positions
                ):
                    fail = "nondefault_market"
                    break
                try:
                    sources[owner] = get_marks(outcome)
                except ValueError, KeyError, OSError:
                    fail = "native_source_mark_missing"
                    break
            if fail:
                stats["blocked_" + fail] += 1
                continue
            instruments = set(held)
            instruments.update(
                k
                for o in ev.changes.values()
                for k in (Instrument.parse(p.instrument) for p in o.positions)
            )
            need = {inst.symbol for inst in instruments}
            if any(sym not in bars or prev_time not in bars[sym] for sym in need):
                stats["blocked_missing_1m_ohlcv"] += 1
                continue
            quotes = {}
            for inst in instruments:
                # Last COMPLETED minute price/volume, available at publication.
                px = bars[inst.symbol][prev_time][1]
                prior_volume = bars[inst.symbol][prev_time][2]
                quotes[inst] = MarketObservation(
                    instrument=inst,
                    event_at=stamp(ts - 1),
                    first_known_at=stamp(ts),
                    mark=px,
                    volume_base=prior_volume,
                    quality="OHLCV_1M_PROXY",
                    supported=True,
                    capacity_basis="TRADED_VOLUME",
                )
            # Native intent; unknown quote/time/k is not a zero book.
            requested = {owner: dict(book) for owner, book in account.pm_quantities.items()}
            failed = False
            for owner, outcome in ev.changes.items():
                native = {p.instrument: p.quantity for p in outcome.positions if p.quantity}
                follower = {
                    inst: bars[Instrument.parse(inst).symbol][prev_time][1] for inst in native
                }
                try:
                    result = target_from_fixed_k(
                        source_quantities=native,
                        k=k,
                        source_marks={name: sources[owner][name] for name in native},
                        follower_marks=follower,
                        follower_equity_usd=account.equity,
                        max_gross_notional_usd=GROSS_CAP,
                        max_market_notional_usd=MARKET_CAP,
                        pod_loss_budget_usd=POD_LOSS_BUDGET,
                    )
                except ValueError, KeyError, TypeError:
                    stats["blocked_fixed_k_risk_or_mark"] += 1
                    failed = True
                    break
                requested[owner] = {
                    Instrument.parse(key): decimal(q)
                    for key, q in result["target_quantities"].items()
                }
            if failed:
                continue
            target = account.external_targets(requested)
            gross = sum((abs(q * quotes[inst].mark) for inst, q in target.items()), D(0))
            if gross > GROSS_CAP or gross > account.equity * D("1.5"):
                stats["blocked_gross_risk"] += 1
                continue
            # No partial rescue or multi-minute splitting. Previous completed
            # minute traded volume is the only causal capacity proxy.
            external = {
                inst: target.get(inst, D(0)) - account.quantities.get(inst, D(0))
                for inst in instruments
            }
            if min_order_notional is not None and any(
                q and abs(q * quotes[inst].mark) < min_order_notional
                for inst, q in external.items()
            ):
                stats["blocked_below_min_order"] += 1
                continue
            if any(
                abs(q) > quotes[inst].volume_base * costs.max_participation
                for inst, q in external.items()
                if q
            ):
                stats["blocked_insufficient_prior_1m_volume"] += 1
                continue
            if not any(q for q in external.values()):
                stats["no_external_change"] += 1
            # Native account risk; confirmed=True is only internal hypothetical
            # Rulebook construction, not a signed or authorized prop contract.
            first_risk = account.evaluate(
                requested,
                {inst: obs.mark for inst, obs in quotes.items()},
                trusted=True,
                market_known=True,
                stressed_loss=gross * D("0.10"),
                reserve=STRESS_RESERVE,
            )
            if not first_risk["allowed"]:
                stats[
                    "blocked_pm03_"
                    + (first_risk["reasons"][0] if first_risk["reasons"] else "unknown")
                ] += 1
                continue
            effect = execute_targets(
                account,
                requested,
                quotes,
                costs,
                decision_at=stamp(ev.at_ms),
                hypothetical_fill_at=stamp(ev.at_ms),
                permitted=True,
            )
            if effect["status"] == "PARTIAL":
                # "Single shot" events may never be silently spread over 24m.
                # Partial means the economics are not the requested policy.
                raise ValueError("NO_PARTIAL_TARGET_EXECUTION_ACCEPTED")
            if effect["status"] in ("BLOCKED", "REJECTED"):
                stats["blocked_pm03_execution"] += 1
                for reason in effect.get("rejections", []):
                    stats["native_rejection_" + reason.split(":")[0]] += 1
                continue
            stats["attempts"] += 1
            stats["fills"] += len(effect["fills"])
            for fill in effect["fills"]:
                if (
                    abs(decimal(fill["quantity"]) * decimal(fill["execution_price"]))
                    < MIN_ORDER_NOTIONAL
                ):
                    stats["sub_10_usd_fills"] += 1
                for owner, amount in fill["allocation"].items():
                    if decimal(amount):
                        fills_by_member[owner] += 1
            stats["internal_crosses"] += len(effect["internal_crosses"])
            stats["executed_notional_usd_x10000"] += round(
                sum(
                    abs(decimal(x["quantity"]) * decimal(x["execution_price"]))
                    for x in effect["fills"]
                )
                * 10000
            )
            held = {inst for inst, q in account.quantities.items() if q}

        # Close mark at 59.999s, never at minute open before known data.
        if held:
            missing = [
                inst.symbol
                for inst in held
                if inst.symbol not in bars or ts not in bars[inst.symbol]
            ]
            if missing:
                stats["missing_mark_minutes"] += 1
                blocked_equity = True
                current_bad = missing[0]
                continue
        # PM03 retains closed instrument keys with zero inventory; provide a
        # harmless last-known mark for those flat keys, never for active risk.
        tracked = set(account.quantities) | {
            inst for book in account.pm_quantities.values() for inst, q in book.items() if q
        }
        prices = {
            inst: (
                bars[inst.symbol][ts][1]
                if inst.symbol in bars and ts in bars[inst.symbol]
                else account.marks[inst]
            )
            for inst in tracked
        }
        account.mark(prices, stamp(ts + MINUTE_MS - 1))
        if last_date != bar_dt.date():
            day_open = account.daily_start_balance
            last_date = bar_dt.date()
        min_equity = min(min_equity, account.equity)
        previous_peak = max(previous_peak, account.equity)
        max_dd = max(max_dd, previous_peak - account.equity)
        worst_day_loss = max(worst_day_loss, day_open - account.equity)
        max_gross = max(max_gross, account.gross())
        pnl_curve.append((ts, account.equity, account.costs, account.realized_pnl))
        if account.breached:
            stats["breach_events"] += 1
            # Account latches permanent HALT; no further new risk accepted.
    if not pnl_curve:
        return {"status": "NO_MARKABLE_MINUTES", "k": str(k)}
    train = [x for x in pnl_curve if x[0] < mill(SPLIT)]
    oos = [x for x in pnl_curve if x[0] >= mill(SPLIT)]
    first_oos_equity = train[-1][1] if train else START_CAPITAL
    oos_high = first_oos_equity
    oos_peak_dd = D(0)
    for _, eq, _, _ in oos:
        oos_high = max(oos_high, eq)
        oos_peak_dd = max(oos_peak_dd, oos_high - eq)
    oos_min = min((x[1] for x in oos), default=None)
    oos_fees = oos[-1][2] - train[-1][2] if oos and train else D(0)
    return {
        "status": "PARTIAL_COVERAGE_UNQUALIFIED" if blocked_equity else "SCENARIO_ONLY",
        "k_per_wallet": str(k),
        "train_net_usd": round(float(train[-1][1] - START_CAPITAL), 4) if train else None,
        "oos_net_usd": round(float(oos[-1][1] - first_oos_equity), 4) if oos else None,
        "oos_minimum_equity_usd": round(float(oos_min), 4) if oos_min is not None else None,
        "oos_peak_drawdown_usd": round(float(oos_peak_dd), 4) if oos else None,
        "oos_fees_usd": round(float(oos_fees), 4) if oos else None,
        "full_net_usd": round(float(account.equity - START_CAPITAL), 4),
        "final_equity_usd": round(float(account.equity), 4),
        "minimum_equity_usd": round(float(min_equity), 4),
        "peak_drawdown_usd": round(float(max_dd), 4),
        "worst_daily_loss_usd": round(float(worst_day_loss), 4),
        "max_gross_notional_usd": round(float(max_gross), 4),
        "total_fees_usd": round(float(account.costs), 4),
        "realized_pnl_usd": round(float(account.realized_pnl), 4),
        "unrealized_pnl_usd": round(float(account.unrealized_pnl), 4),
        "funding_usd": None,
        "trades": stats["fills"],
        "attempts": stats["attempts"],
        "internal_crosses": stats["internal_crosses"],
        "breach": account.breached,
        "floor": {k: str(v) for k, v in account.floor()[1].items()},
        "coverage_gap_symbol": current_bad,
        "stats": dict(stats),
        "assumed_min_order_usd": str(min_order_notional)
        if min_order_notional is not None
        else None,
        "fills_by_member": dict(fills_by_member),
        "sampled_minutes": len(pnl_curve),
        "oos_minutes": len(oos),
    }


def choose_train_k(rows: dict[str, dict], native_source_events: int) -> tuple[D, str]:
    """Train-only risk screen; no OOS feedback or alpha promotion."""
    possible = []
    for raw, metrics in rows.items():
        if raw == "0" or metrics["status"] != "SCENARIO_ONLY" or not metrics["sampled_minutes"]:
            continue
        net = metrics["train_net_usd"]
        if (
            net is None
            or net <= 0
            or metrics["worst_daily_loss_usd"] > float(TRAIN_DAILY_CAP)
            or metrics["peak_drawdown_usd"] > float(TRAIN_PEAK_CAP)
            or metrics["breach"]
        ):
            continue
        # Without this pre-specified gate the "optimal" k is often a size
        # that only fills the tiny profitable minority of source changes.
        if (
            native_source_events == 0
            or D(metrics["attempts"]) / native_source_events < TRAIN_MIN_EXECUTION_SHARE
        ):
            continue
        possible.append((net, decimal(raw)))
    if not possible:
        return D(0), "NO_POSITIVE_COMPLETE_TRAIN_CANDIDATE"
    possible.sort(key=lambda row: (row[0], -row[1]), reverse=True)
    return possible[0][1], "CHOSEN_ON_TRAIN_ONLY_EXPLORATORY"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="trading-ops-native-finance-snapshot-") as tmp:
        snapshot = Path(tmp) / "source.sqlite3"
        with (
            closing(connect_readonly(SOURCE)) as live,
            closing(sqlite3.connect(snapshot)) as frozen,
        ):
            live.backup(frozen, pages=8192)
        waves, source = source_waves(snapshot)
    # No edits or writes to source; read only original-wire sidecars.
    event_map = {}
    all_symbols = set()
    for pid, members in ROSTERS.items():
        events, count = build_events(waves, members)
        event_map[pid] = (events, count)
        all_symbols |= market_symbols(events)
    bars, coverage = load_local_bars(all_symbols)
    outputs = {}
    for pid, (events, count) in event_map.items():
        initial = {
            str(k): simulate(ROSTERS[pid], events, bars, k=k, stop_ms=mill(SPLIT)) for k in K_GRID
        }
        # OOS isn't consulted in choosing k.
        train_events = sum(ev.at_ms < mill(SPLIT) for ev in events)
        frozen_k, reason = choose_train_k(initial, train_events)
        # Entire path starts before split, so changes aren't reset by a
        # fictitious account restart at OOS.
        final = simulate(ROSTERS[pid], events, bars, k=frozen_k)
        # A fixed exploratory k reports economics even if the train gate declines.
        diagnostic = simulate(ROSTERS[pid], events, bars, k=D("0.0001"), min_order_notional=None)
        strict_reference = simulate(ROSTERS[pid], events, bars, k=D("0.0001"))
        larger_fractional = simulate(
            ROSTERS[pid], events, bars, k=D("0.0002"), min_order_notional=None
        )
        cost_stress = {
            str(f): simulate(
                ROSTERS[pid], events, bars, k=D("0.0001"), factor=f, min_order_notional=None
            )
            for f in (D(1), D(2), D(3))
        }
        outputs[pid] = {
            "members": list(ROSTERS[pid]),
            "native_source": dict(count),
            "train_grid": initial,
            "train_native_source_events": train_events,
            "train_min_execution_share": str(TRAIN_MIN_EXECUTION_SHARE),
            "frozen_k_per_wallet": {alias: str(frozen_k) for alias in ROSTERS[pid]},
            "k_selection_reason": reason,
            "train_selected_full_path": final,
            "fixed_k_0_0001_diagnostic": diagnostic,
            "fixed_k_0_0001_with_min_order": strict_reference,
            "fixed_k_0_0002_unrestricted_diagnostic": larger_fractional,
            "fixed_k_0_0001_ignores_minimum_order_size": True,
            "fixed_k_0_0001_cost_stress": cost_stress,
            "selection_known_before_sample": False,
            "scientific_qualification": "NOT_QUALIFIED",
        }
    result = {
        "schema": "native_cwsa_local_1m_pm03_equity_replay_v1",
        "classification": "SCENARIO_ONLY",
        "no_broker_or_order_authorization": True,
        "genuine_cwsa_source": True,
        "source": source,
        "train_start_utc": START.isoformat(),
        "split_utc": SPLIT.isoformat(),
        "end_exclusive_utc": END.isoformat(),
        "price_source": "LOCAL_ML4T_AND_VERIFIED_OFFICIAL_HYPERLIQUID_1M_OHLCV",
        "execution": "ONE_ATTEMPT_AFTER_INDIVIDUAL_NATIVE_PUBLICATION_PREVIOUS_COMPLETED_1M_CLOSE_PROXY",
        "capacity": "PREVIOUS_COMPLETE_1M_TRADED_VOLUME_MAX_10_PERCENT",
        "costs": {
            "fee_bps": "4.5",
            "full_spread_bps": "2",
            "slippage_bps": "2",
            "sqrt_participation_impact_bps_at_full_share": "5",
            "funding": "NOT_AVAILABLE_NOT_MODELED",
        },
        "risk": {
            "daily_loss_usd": 300,
            "static_loss_usd": 500,
            "static_initial_floor_usd": 9500,
            "timezone": "UTC",
            "max_gross_notional_usd": str(GROSS_CAP),
            "max_notional_per_market_usd": str(MARKET_CAP),
            "research_min_order_usd": str(MIN_ORDER_NOTIONAL),
            "pod_loss_budget_usd": str(POD_LOSS_BUDGET),
            "10pct_adverse_stress_plus_reserve_usd": str(STRESS_RESERVE),
            "research_rulebook_only": True,
        },
        "market_coverage": coverage,
        "portfolios": outputs,
        "no_trade_recommendation": True,
        "limitations": [
            "Only November 12-13 local 1m bars overlap original causal CWSA source",
            "Native checkpoint source publication and original marks are verified; follower execution uses previous completed 1m close proxy, not a broker fill",
            "Previous fully completed 1m close is an observable proxy (max quote age 90s), NOT production native 2s L2 bid/ask; latency displacement remains unverified",
            "Synthetic fill occurs after actual CWSA publication and before native max-source-age expiration; no later candle close/open is used for decision or pricing",
            "Funding, orderbook depth, lot precision and actual venue minimum orders are unknown; assumed $10 minimum is an explicit conservative stress scenario",
            "Portfolios were selected after the observed window and the capacity gate was revised after inspecting this sample; OOS is purely chronological exploratory, not a clean independent holdout",
            "Several large k candidates skip most source opportunities on participation grounds; a >=70 percent source-event execution gate applies on training only",
            "Training duration is less than one day; no statistical alpha or prop-challenge survival claim",
            "No signed Copy Mandate or broker-financial authority is created",
        ],
    }
    proposal = {
        "schema": "native_local_hl_1m_three_portfolio_research_proposal_v1",
        "classification": "SCENARIO_ONLY_NOT_EXECUTION_QUALIFIED",
        "trading_authorized": False,
        "order_route": "NONE",
        "daily_loss_usd": 300,
        "initial_equity_usd": 10000,
        "static_equity_floor_usd": 9500,
        "true_train_oos": "NOT_AVAILABLE_RETROSPECTIVE_ROSTER_AND_SHORT_SAMPLE",
        "proposals": [
            {
                "name": pid,
                "members": list(ROSTERS[pid]),
                "policy": "FIXED_K_NATIVE_QUANTITY",
                "admitted_k": v["frozen_k_per_wallet"],
                "reason": v["k_selection_reason"],
                "research_fixed_k_0_0001_net_usd": v["fixed_k_0_0001_diagnostic"]["full_net_usd"],
                "research_fixed_k_0_0001_sub_min_order_fills": v["fixed_k_0_0001_diagnostic"][
                    "stats"
                ].get("sub_10_usd_fills", 0),
                "research_fixed_k_0_0001_total_fills": v["fixed_k_0_0001_diagnostic"]["trades"],
                "strict_min_order_replay_status": v["fixed_k_0_0001_with_min_order"]["status"],
                "admission": "NO_TRADE_UNQUALIFIED",
            }
            for pid, v in outputs.items()
        ],
    }
    REPORT.parent.mkdir(exist_ok=True)
    PROPOSAL.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    PROPOSAL.write_text(json.dumps(proposal, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "source": source,
                "missing_markets": [
                    s for s, r in coverage.items() if r["status"].startswith("MISSING")
                ],
                "portfolios": {
                    pid: {
                        "source": v["native_source"],
                        "selected_k": v["frozen_k_per_wallet"],
                        "selected": v["train_selected_full_path"],
                        "diagnostic": v["fixed_k_0_0001_diagnostic"],
                    }
                    for pid, v in outputs.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
