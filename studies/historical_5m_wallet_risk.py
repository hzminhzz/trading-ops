"""Historical fixed-k risk budget research, without retrospective first-known claims.

Default [2026-07-01, 2026-11-01), with training [Jul1, Oct1)
and chronological October evaluation. Reconstruct the selected parent
Hyperliquid perp position vector at UTC 5-minute checkpoints from the
native read-only wallet_store fill ledger, anchored to a historical complete
position snapshot. This is an EX-POST BOOK ORACLE, not a genuine CWSA receipt.

A baseline unit-k replay uses copytrade-pm target_from_fixed_k,
pm03_account.Account, and pm03_execution.CostModel. Train-only wallet k is
50% of its allocated dollar loss budget divided by its unit max drawdown.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from array import array
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN
from decimal import Decimal as D
from pathlib import Path

import polars as pl
from copytrade_pm.pm03_account import Account, Instrument, Rulebook
from copytrade_pm.pm03_execution import CostModel, MarketObservation
from copytrade_pm.pm_exposure_sizing import target_from_fixed_k
from copytrade_pm.wallet_store import scan

START = datetime(2026, 7, 1, tzinfo=UTC)
TRAIN_END = datetime(2026, 10, 1, tzinfo=UTC)
END = datetime(2026, 11, 1, tzinfo=UTC)
MINUTE = 60_000
GRID = 5 * MINUTE
EQUITY = D(10_000)
STATIC_LOSS = D(500)
DAILY_LOSS = D(300)
RISK_USE = D("0.5")
WALLET_COUNT = 3
SOURCE_PREFIX = "hyperliquid/mainnet:perp:default/"
REGISTRY = Path("/home/quant/.local/share/copytrade/registry/wallet_registry.sqlite")
STORE = Path("/home/quant/dev/copytrade-pm/data/private/wallet_store")
BARS = Path("/home/quant/dev/quant/ml4t-data/.data/store")
REPORT = Path("reports/historical-5m-k-risk-20260701-20261101.json")
PROPOSAL = Path("proposals/historical-5m-k-risk-20260701-20261101.json")
ROSTERS = {
    "regime.macro/TriadA": ("ACT04", "ACT08", "ACT09"),
    "research.crypto/Balanced": ("ACT04", "ACT08", "W031"),
    "research.crypto/RegimeAlt": ("ACT08", "ACT09", "W031"),
}


def ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def date(ms_value: int) -> str:
    return datetime.fromtimestamp(ms_value / 1000, UTC).isoformat()


def d(v: float | str | D) -> D:
    number = v if isinstance(v, D) else D(str(v))
    if not number.is_finite():
        raise ValueError("NONFINITE_FINANCIAL_INPUT")
    return number


def risk_allocation(*, static_loss: D = STATIC_LOSS, n: int = WALLET_COUNT) -> D:
    if static_loss <= 0 or n <= 0:
        raise ValueError("INVALID_RISK_ALLOCATION")
    return static_loss / n


def wallet_drawdown_target(allocation: D, *, fraction: D = RISK_USE) -> D:
    if allocation <= 0 or not 0 < fraction <= 1:
        raise ValueError("INVALID_RISK_FRACTION")
    return allocation * fraction


def k_for_wallet_drawdown(
    unit_max_drawdown: D | None,
    budget: D,
    *,
    fraction: D = RISK_USE,
    ceiling: D = D(10),
) -> D | None:
    """Train-only: no arbitrary fixed k grid or full-sample worst-loss solving."""
    if budget <= 0 or not 0 < fraction <= 1 or ceiling <= 0:
        raise ValueError("INVALID_RISK_BUDGET")
    if unit_max_drawdown is None:
        return None
    if unit_max_drawdown < 0:
        raise ValueError("NEGATIVE_MAX_DRAWDOWN")
    if unit_max_drawdown == 0:
        return None
    raw = wallet_drawdown_target(budget, fraction=fraction) / unit_max_drawdown
    return min(raw, ceiling).quantize(D("0.000000000001"), rounding=ROUND_DOWN)


def open_readonly(path: Path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def registry_wallets() -> tuple[dict[str, str], dict]:
    """Resolve current private SQLite, no hardcoded wallet addresses."""
    with open_readonly(REGISTRY) as db:
        verdict = db.execute("PRAGMA quick_check").fetchone()[0]
        if verdict != "ok":
            raise ValueError("REGISTRY_INTEGRITY_FAILED")
        addresses: dict[str, str] = {}
        facts: dict = {}
        for alias in sorted({m for members in ROSTERS.values() for m in members}):
            rows = db.execute(
                "SELECT DISTINCT w.wallet_id,w.address FROM wallet_aliases a "
                "JOIN wallets w ON a.wallet_id=w.wallet_id WHERE a.alias=?",
                (alias,),
            ).fetchall()
            if len(rows) != 1:
                raise ValueError("REGISTRY_AMBIGUOUS_OR_MISSING_ALIAS:" + alias)
            identity, address = rows[0]
            owners = db.execute(
                "SELECT COUNT(*) FROM exclusions WHERE wallet_id=?", (identity,)
            ).fetchone()[0]
            prop = db.execute(
                "SELECT COUNT(*) FROM propfirm_exclusions WHERE wallet_id=?", (identity,)
            ).fetchone()[0]
            if owners or prop:
                raise ValueError("REGISTRY_EXCLUDED_SOURCE:" + alias)
            addresses[alias] = address.lower()
            facts[alias] = {
                "registered": True,
                "current_exclusions": 0,
                "historical_selection_as_of_start": False,
            }
        members = db.execute(
            "SELECT member_alias FROM portfolio_memberships WHERE "
            "portfolio_id='regime.macro/TriadA' AND "
            "version=(SELECT MAX(version) FROM portfolio_versions "
            "WHERE portfolio_id='regime.macro/TriadA') "
            "ORDER BY member_alias"
        ).fetchall()
        if tuple(x[0] for x in members) != tuple(sorted(ROSTERS["regime.macro/TriadA"])):
            raise ValueError("SEALED_PORTFOLIO_ROSTER_DRIFT")
        return addresses, facts


@dataclass(frozen=True)
class Fill:
    t: int
    symbol: str
    delta: D
    start_position: D | None


@dataclass(frozen=True)
class WalletSource:
    alias: str
    address: str
    anchor_ms: int
    anchor: dict[str, D]
    initial: dict[str, D]
    fills: tuple[Fill, ...]
    snapshots: tuple[tuple[int, dict[str, D]], ...]
    symbols: tuple[str, ...]
    validation: dict


def _signed(direction: str, quantity: float) -> D:
    if direction not in ("long", "short") or not math.isfinite(quantity) or quantity < 0:
        raise ValueError("INVALID_SOURCE_POSITION")
    return (D(1) if direction == "long" else D(-1)) * d(quantity)


def _changed_book(book: dict[str, D], symbol: str, delta: D) -> None:
    next_value = book.get(symbol, D(0)) + delta
    if abs(next_value) < D("0.00000001"):
        book.pop(symbol, None)
    else:
        book[symbol] = next_value


def load_source(alias: str, address: str, start: datetime, end: datetime) -> WalletSource:
    """Anchor retrospective fill books, never call them original first-known CWSA."""
    positions = (
        scan("positions", wallets=[address], start=start, end=end, store_dir=STORE)
        .filter(pl.col("dex") == "HL")
        .select("snapshot_ts", "symbol", "direction", "size_token")
        .sort("snapshot_ts")
        .collect()
    )
    if positions.is_empty():
        raise ValueError("NO_FULL_BOOK_SNAPSHOT:" + alias)
    first_ms = ms(positions["snapshot_ts"][0])
    anchored = positions.filter(pl.col("snapshot_ts") == positions["snapshot_ts"][0])
    anchor: dict[str, D] = {}
    for row in anchored.iter_rows(named=True):
        key = row["symbol"]
        if key in anchor:
            raise ValueError("DUPLICATE_SNAPSHOT_COIN")
        amount = _signed(row["direction"], row["size_token"])
        if amount:
            anchor[key] = amount
    if not anchor:
        raise ValueError("EMPTY_ANCHOR_CANNOT_PROVE_COMPLETENESS")

    fills_df = (
        scan("fills", wallets=[address], start=start, end=end, store_dir=STORE)
        .filter(pl.col("dex") == "HL")
        .select("ts", "symbol", "side", "size_token", "start_position", "trade_id")
        .sort("ts", "symbol", "trade_id")
        .collect()
    )
    seen: set[tuple] = set()
    fill_list: list[Fill] = []
    duplicates = 0
    for row in fills_df.iter_rows(named=True):
        key = (row["ts"], row["symbol"], row["side"], row["size_token"], row["trade_id"])
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        if row["side"] not in ("buy", "sell"):
            raise ValueError("INVALID_HISTORICAL_SIDE")
        symbol = row["symbol"]
        if not math.isfinite(row["size_token"]) or row["size_token"] <= 0:
            raise ValueError("INVALID_HISTORICAL_FILL_SIZE")
        fill_list.append(
            Fill(
                ms(row["ts"]),
                symbol,
                d(row["size_token"]) * (D(1) if row["side"] == "buy" else D(-1)),
                d(row["start_position"]) if row["start_position"] is not None else None,
            )
        )
    fill_list.sort(key=lambda r: (r.t, r.symbol))
    # Backsolve *only* the retrospective initial vector from a later full snapshot.
    # This is a hindsight oracle, never a source receipt known at start.
    initial = dict(anchor)
    backcast_count = 0
    for row in reversed(fill_list):
        if row.t >= first_ms:
            continue
        _changed_book(initial, row.symbol, -row.delta)
        backcast_count += 1

    # Verify at regular, independently archived snapshot cutoffs.
    # Missing fill evidence or drift is a provenance blocker, not a flat book.
    grouped = {}
    for row in positions.iter_rows(named=True):
        at = ms(row["snapshot_ts"])
        snap = grouped.setdefault(at, {})
        if row["symbol"] in snap:
            raise ValueError("DUPLICATE_SNAPSHOT_SYMBOL")
        amount = _signed(row["direction"], row["size_token"])
        if amount:
            snap[row["symbol"]] = amount
    checks = [
        t
        for t in sorted(grouped)
        if t == first_ms
        or (
            t > first_ms
            and datetime.fromtimestamp(t / 1000, UTC).hour == 0
            and datetime.fromtimestamp(t / 1000, UTC).minute <= 10
        )
    ]
    checks.extend([max(grouped)])
    checks = sorted(set(checks))
    state = dict(initial)
    idx = 0
    mismatches = 0
    worst_abs = D(0)
    examples = []
    for t in checks:
        while idx < len(fill_list) and fill_list[idx].t < t:
            _changed_book(state, fill_list[idx].symbol, fill_list[idx].delta)
            idx += 1
        snapshot = grouped[t]
        gap = max(
            (abs(snapshot.get(s, D(0)) - state.get(s, D(0))) for s in set(snapshot) | set(state)),
            default=D(0),
        )
        worst_abs = max(worst_abs, gap)
        if gap > D("0.000001"):
            mismatches += 1
            if len(examples) < 5:
                examples.append({"date": date(t), "max_abs_quantity_gap": str(gap)})
    all_symbols = tuple(sorted(set(anchor) | set(initial) | {r.symbol for r in fill_list}))
    return WalletSource(
        alias,
        address,
        first_ms,
        anchor,
        initial,
        tuple(fill_list),
        tuple(sorted(grouped.items())),
        all_symbols,
        {
            "first_full_snapshot_utc": date(first_ms),
            "latest_snapshot_utc": date(max(grouped)),
            "snapshot_rows": positions.height,
            "snapshot_timestamps": len(grouped),
            "sampled_snapshot_checks": len(checks),
            "sampled_snapshot_disagreements": mismatches,
            "sampled_snapshot_max_abs_qty_error": str(worst_abs),
            "first_fills_backcast_for_start": backcast_count,
            "fills_in_window": len(fill_list),
            "duplicate_fill_rows_removed": duplicates,
            "backcast_from_future_anchor": first_ms > ms(start),
            "anchor_and_fills_consistent": mismatches == 0,
            "historical_first_known_receipts": False,
            "validation_examples": examples,
        },
    )


class LocalBars:
    """Local 1m Hyperliquid OHLCV only. No network, zero/flat interpolation."""

    def __init__(self, start: datetime, end: datetime):
        self.start_ms = ms(start)
        self.end_ms = ms(end)
        self.count = (self.end_ms - self.start_ms) // MINUTE
        self.prices = {}
        self.volumes = {}
        self.coverage = {}

    def add(self, symbol: str) -> None:
        if symbol in self.prices:
            return
        data = array("d", [math.nan]) * self.count
        volume = array("d", [math.nan]) * self.count
        base = BARS / f"crypto_futures_ohlcv_1m_HYPERLIQUID_{symbol}_PERP"
        fs = []
        current = datetime.fromtimestamp(self.start_ms / 1000, UTC).date()
        final = datetime.fromtimestamp(self.end_ms / 1000, UTC).date()
        while current < final:
            path = (
                base
                / f"year={current.year:04d}/month={current.month:02d}/day={current.day:02d}/data.parquet"
            )
            if path.is_file():
                fs.append(path)
            current += timedelta(days=1)
        if fs:
            frame = (
                pl.scan_parquet(
                    [str(f) for f in fs],
                    extra_columns="ignore",
                    missing_columns="insert",
                    cast_options=pl.ScanCastOptions(datetime_cast="microsecond-downcast"),
                )
                .select("timestamp", "close", "volume")
                .collect()
            )
            n = 0
            for time, close, vol in frame.iter_rows():
                index = (ms(time) - self.start_ms) // MINUTE
                if index < 0 or index >= self.count:
                    continue
                if not math.isfinite(close) or close <= 0 or not math.isfinite(vol) or vol < 0:
                    raise ValueError("INVALID_SOURCE_1M_CANDLE:" + symbol)
                if not math.isnan(data[index]):
                    raise ValueError("DUPLICATE_MARKET_1M_CANDLE:" + symbol)
                data[index] = close
                volume[index] = vol
                n += 1
        else:
            n = 0
        self.prices[symbol] = data
        self.volumes[symbol] = volume
        self.coverage[symbol] = {
            "days_present": len(fs),
            "minute_rows": n,
            "expected_window_minutes": self.count,
            "fraction": round(n / self.count, 5),
            "first_available_utc": date(
                self.start_ms
                + next((i for i, x in enumerate(data) if not math.isnan(x)), self.count) * MINUTE
            )
            if n
            else None,
        }

    def mark(self, symbol: str, t: int, *, lag_minutes: int = 0, tolerance: int = 5) -> D | None:
        if symbol not in self.prices:
            return None
        index = (t - self.start_ms) // MINUTE - lag_minutes
        for j in range(index, max(-1, index - tolerance - 1), -1):
            if 0 <= j < self.count and not math.isnan(self.prices[symbol][j]):
                return d(self.prices[symbol][j])
        return None

    def prev_volume(self, symbol: str, t: int) -> D | None:
        if symbol not in self.volumes:
            return None
        i = (t - self.start_ms) // MINUTE - 1
        if i < 0 or i >= self.count or math.isnan(self.volumes[symbol][i]):
            return None
        return d(self.volumes[symbol][i])


def cost_model() -> CostModel:
    return CostModel(
        fee_bps=D("4.5"),
        full_spread_bps=D(2),
        slippage_bps=D(2),
        impact_bps_at_full_participation=D(0),
        max_participation=D(1),
        max_age_seconds=300,
        latency_seconds=0,
    )


def replay_unit(
    wallet: WalletSource,
    bars: LocalBars,
    *,
    start: datetime = START,
    train_end: datetime = TRAIN_END,
    end: datetime = END,
) -> dict:
    """Historical-oracle unit book, PM03 financial ledger; 1m MTM, 5m targets."""
    high_balance = D(1000000000000)
    all_inst = tuple(Instrument.parse(SOURCE_PREFIX + s) for s in wallet.symbols)
    rb = Rulebook(
        program="RETROSPECTIVE_UNIT_K_SLOPE_ONLY",
        stage="scenario",
        source="LOCAL_HL_1M_RETROSPECTIVE_NO_FIRST_KNOWN",
        observed_at=start.isoformat(),
        version="research-v1",
        timezone="UTC",
        initial_balance=high_balance,
        profit_target=None,
        daily_loss=None,
        static_loss=None,
        trailing_loss=None,
        permitted=all_inst,
        confirmed=True,
        account_terms_confirmed=False,
    )
    account = Account.start(rb, start.isoformat())
    book = dict(wallet.initial)
    prev_book = None
    # Ordered native historical position observations override fill-derived
    # estimates at their stated event time. Never claim these were first-known.
    events = sorted(
        [(r.t, 0, r) for r in wallet.fills]
        + [(t, 1, snapshot) for t, snapshot in wallet.snapshots],
        key=lambda row: (row[0], row[1]),
    )
    fillidx = 0
    costs = cost_model()
    counters = Counter()
    peak = D(0)
    dd = D(0)
    train_peak = D(0)
    train_dd = D(0)
    min_equity = D(0)
    train_last = D(0)
    first_gap = None
    marks = {}
    unit_curve = []

    def mark_book(t: int, *, lag: int = 0) -> dict[Instrument, D] | None:
        active = {inst for inst, quantity in account.quantities.items() if quantity}
        present = active | {Instrument.parse(SOURCE_PREFIX + s) for s in book if book[s]}
        result = {}
        for inst in present:
            px = bars.mark(inst.symbol, t, lag_minutes=lag)
            if px is None:
                return None
            result[inst] = px
        return result

    for now in range(ms(start), ms(end), MINUTE):
        # Both fills and archived full books use retrospective event clocks,
        # not genuine historical first-known observations. A full book fences
        # inconsistent inferred inventory; no invented flat from a gap.
        while fillidx < len(events) and events[fillidx][0] < now:
            _, kind, payload = events[fillidx]
            if kind == 0:
                _changed_book(book, payload.symbol, payload.delta)
            else:
                if max(
                    (
                        abs(book.get(s, D(0)) - payload.get(s, D(0)))
                        for s in set(book) | set(payload)
                    ),
                    default=D(0),
                ) > D("0.000001"):
                    counters["retrospective_snapshot_inventory_corrections"] += 1
                book = dict(payload)
            fillidx += 1
        if now % GRID == 0:
            if prev_book is None:
                prev_book = dict(book)
                counters["retrospective_initial_baseline_no_trade"] += 1
            elif book != prev_book:
                prev_book = dict(book)
                prior = mark_book(now, lag=1)
                if prior is None:
                    counters["unknown_target_mark"] += 1
                else:
                    source_qty = {SOURCE_PREFIX + s: amount for s, amount in book.items() if amount}
                    price = {k: prior[Instrument.parse(k)] for k in source_qty}
                    try:
                        proposed = target_from_fixed_k(
                            source_quantities=source_qty,
                            k=D(1),
                            source_marks=price,
                            follower_marks=price,
                            follower_equity_usd=high_balance,
                            max_gross_notional_usd=D(100000000000),
                            max_market_notional_usd=D(100000000000),
                            pod_loss_budget_usd=D(100000000),
                        )
                    except (KeyError, ValueError) as exc:
                        counters["intent_rejected"] += 1
                        if first_gap is None:
                            first_gap = str(exc)
                        continue
                    desired = {
                        Instrument.parse(k): d(v) for k, v in proposed["target_quantities"].items()
                    }
                    all_marked = set(desired) | {i for i, q in account.quantities.items() if q}
                    fill_marks = {}
                    for inst in all_marked:
                        px = bars.mark(inst.symbol, now, lag_minutes=1)
                        if px is None:
                            fill_marks = None
                            break
                        fill_marks[inst] = px
                    if fill_marks is None:
                        counters["unknown_execution_mark"] += 1
                    else:
                        # PM03 marks before any trade and tracks realized basis/cost.
                        current_marks = {
                            **{
                                inst: bars.mark(inst.symbol, now, lag_minutes=1)
                                for inst, q in account.quantities.items()
                                if q
                            },
                            **fill_marks,
                        }
                        current_marks = {k: v for k, v in current_marks.items() if v is not None}
                        # Native PM03 retains flat quantity keys for audit;
                        # they need a cached mark, never a fabricated active mark.
                        current_marks = {**account.marks, **current_marks}
                        account.mark(current_marks, date(now - 1))
                        for inst in sorted(all_marked):
                            delta = desired.get(inst, D(0)) - account.quantities.get(inst, D(0))
                            if not delta:
                                continue
                            quote = MarketObservation(
                                instrument=inst,
                                event_at=date(now - 1),
                                first_known_at=date(now - 1),
                                mark=fill_marks[inst],
                                volume_base=D(1000000000000000),
                                quality="OHLCV_1M_PROXY",
                                supported=True,
                            )
                            px, fee, _ = costs.price_and_fee(delta, quote)
                            account.simulate_fill(
                                inst, delta, px, fee, accepted=True, allocated={wallet.alias: delta}
                            )
                            counters["modeled_unit_orders"] += 1
        # All open sources and actual simulated positions must mark continuously.
        active = {inst for inst, q in account.quantities.items() if q}
        if active:
            for inst in active:
                px = bars.mark(inst.symbol, now, tolerance=5)
                if px is None:
                    first_gap = {"time": date(now), "symbol": inst.symbol}
                    counters["unmarkable_open_position"] += 1
                    break
                marks[inst] = px
            if first_gap:
                break
        # Zero quantity keys retained by PM03 must also have marks for Account.mark.
        present = set(account.quantities)
        account.mark(
            {inst: marks.get(inst, account.marks.get(inst, D(1))) for inst in present},
            date(now + MINUTE - 1),
        )
        equity_change = account.equity - high_balance
        peak = max(peak, equity_change)
        dd = max(dd, peak - equity_change)
        min_equity = min(min_equity, equity_change)
        if now < ms(train_end):
            train_peak = max(train_peak, equity_change)
            train_dd = max(train_dd, train_peak - equity_change)
            train_last = equity_change
        counters["marked_minutes"] += 1
        unit_curve.append(float(equity_change))
    return {
        "wallet": wallet.alias,
        "scenario": "RETROSPECTIVE_EVENT_TIME_NOT_CWSA_FIRST_KNOWN",
        "window_start": start.isoformat(),
        "window_end_exclusive": end.isoformat(),
        "train_end_exclusive": train_end.isoformat(),
        "unit_k": 1,
        "_unit_curve_usd": unit_curve,
        "train_unit_max_drawdown_usd": str(train_dd),
        "unit_full_max_drawdown_usd": str(dd),
        "train_unit_net_usd": str(train_last),
        "full_unit_net_usd": str(account.equity - high_balance),
        "unit_fees_usd": str(account.costs),
        "open_position_mark_gap": first_gap,
        "counters": dict(counters),
        "complete_train_minutes": counters["marked_minutes"]
        >= (ms(train_end) - ms(start)) // MINUTE,
        "complete_window_minutes": counters["marked_minutes"] >= (ms(end) - ms(start)) // MINUTE,
    }


def aggregate_sized_portfolio(
    members: tuple[str, ...],
    k_by_member: dict[str, D | None],
    curves: dict[str, list[float]],
    *,
    start: datetime = START,
    train_end: datetime = TRAIN_END,
    end: datetime = END,
) -> dict:
    """Equal-footing economic sensitivity, NOT an executable netted PM03 fill tape."""
    if any(k_by_member[m] is None for m in members):
        return {"status": "BLOCKED_MISSING_TRAIN_K"}
    if any(m not in curves for m in members):
        return {"status": "BLOCKED_MISSING_UNIT_EQUITY_PATH"}
    count = min(len(curves[m]) for m in members)
    if not count:
        return {"status": "BLOCKED_NO_SHARED_MARKED_MINUTES"}
    if any(k_by_member[m] < 0 for m in members):
        raise ValueError("NEGATIVE_SCENARIO_K")
    train_boundary = (ms(train_end) - ms(start)) // MINUTE
    if count < train_boundary:
        return {"status": "BLOCKED_INCOMPLETE_TRAIN_EQUITY_PATH"}
    initial = float(EQUITY)
    peak = initial
    max_dd = 0.0
    minimum = initial
    worst_daily = 0.0
    daily_baseline = initial
    max_oos_dd = 0.0
    oos_peak = None
    oct_boundary_equity = None
    first_breach = None
    final = initial
    for i in range(count):
        eq = initial + sum(float(k_by_member[m]) * curves[m][i] for m in members)
        if i and i % 1440 == 0:
            daily_baseline = final
        minimum = min(minimum, eq)
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)
        worst_daily = max(worst_daily, daily_baseline - eq)
        if i == train_boundary - 1:
            oct_boundary_equity = eq
        if i >= train_boundary:
            if oos_peak is None:
                oos_peak = eq
            oos_peak = max(oos_peak, eq)
            max_oos_dd = max(max_oos_dd, oos_peak - eq)
        if first_breach is None and (eq <= 9500 or eq <= daily_baseline - 300):
            first_breach = date(ms(start) + i * MINUTE)
        final = eq
    fully_observed = count == (ms(end) - ms(start)) // MINUTE
    return {
        "status": "COMPLETE_RETROSPECTIVE_SCENARIO"
        if fully_observed
        else "PARTIAL_RETROSPECTIVE_SCENARIO",
        "start_utc": start.isoformat(),
        "source_supported_through_utc": date(ms(start) + count * MINUTE),
        "end_requested_exclusive_utc": end.isoformat(),
        "common_marked_minutes": count,
        "complete_requested_window": fully_observed,
        "train_net_usd": round(oct_boundary_equity - initial, 2),
        "october_chronological_net_usd": round(final - oct_boundary_equity, 2),
        "total_net_through_coverage_usd": round(final - initial, 2),
        "minimum_equity_usd": round(minimum, 2),
        "max_peak_drawdown_usd": round(max_dd, 2),
        "october_peak_drawdown_usd": round(max_oos_dd, 2),
        "worst_utc_day_from_prior_day_end_usd": round(worst_daily, 2),
        "first_equity_floor_breach_utc": first_breach,
        "missing_realized_daily_balance": True,
        "funding_unknown": True,
        "lot_and_venue_capacity_unverified": True,
        "does_not_net_cross_wallet_fills": True,
        "not_independent_portfolio_selection_oos": True,
    }


def report(
    *,
    audit_only: bool = False,
    start: datetime = START,
    train_end: datetime = TRAIN_END,
    end: datetime = END,
) -> dict:
    if not (start < train_end < end) or any(
        v.tzinfo is None or v.hour or v.minute or v.second for v in (start, train_end, end)
    ):
        raise ValueError("BACKTEST_WINDOW_MUST_HAVE_CAUSAL_UTC_MIDNIGHT_SPLIT")
    addresses, registry = registry_wallets()
    sources = {a: load_source(a, addr, start, end) for a, addr in addresses.items()}
    market = LocalBars(start, end)
    symbols = {s for source in sources.values() for s in source.symbols}
    for coin in sorted(symbols):
        market.add(coin)
    wallets = {}
    curves = {}
    for alias, source in sources.items():
        unit = (
            None
            if audit_only
            else replay_unit(source, market, start=start, train_end=train_end, end=end)
        )
        if unit is not None:
            curves[alias] = unit.pop("_unit_curve_usd")
        budget = risk_allocation()
        wallet_k = k_for_wallet_drawdown(
            d(unit["train_unit_max_drawdown_usd"])
            if unit is not None and unit["complete_train_minutes"]
            else None,
            budget,
        )
        wallets[alias] = {
            "registry": registry[alias],
            "source": source.validation,
            "train_wallet_risk_allocation_usd": str(budget),
            "risk_target_50pct_usd": str(wallet_drawdown_target(budget)),
            "research_calibrated_k": str(wallet_k) if wallet_k is not None else None,
            "accepted_executable_k": None,
            "calibrated_k": str(wallet_k) if wallet_k is not None else None,
            "calibrated_train_drawdown_usd": (
                str(wallet_k * d(unit["train_unit_max_drawdown_usd"]))
                if wallet_k is not None
                else None
            ),
            "unit": unit,
        }
    scenario_portfolios = {}
    for name, members in ROSTERS.items():
        k_by_member = {
            m: (
                d(wallets[m]["research_calibrated_k"])
                if wallets[m]["research_calibrated_k"] is not None
                else None
            )
            for m in members
        }
        scenario_portfolios[name] = {
            "members": members,
            "k_by_member": {m: wallets[m]["research_calibrated_k"] for m in members},
            "accepted_executable_k": {m: None for m in members},
            "chronological_scenario": aggregate_sized_portfolio(
                members, k_by_member, curves, start=start, train_end=train_end, end=end
            )
            if not audit_only
            else {"status": "AUDIT_ONLY"},
            "qualification": "NOT_SCIENTIFIC_OOS_NO_NATIVE_JUL_OCT_SOURCE_RECEIPTS",
        }
    return {
        "schema": "historical_position_checkpoints_fixed_k_risk_half_budget_v1",
        "classification": "RETROSPECTIVE_SCENARIO_ONLY",
        "trading_authorized": False,
        "order_route": "NONE",
        "source_first_known_valid": False,
        "defaults": {
            "start_utc": start.isoformat(),
            "end_exclusive_utc": end.isoformat(),
            "train_end_exclusive_utc": train_end.isoformat(),
            "per_wallet_risk_allocation_usd": str(risk_allocation()),
            "per_wallet_risk_drawdown_target_usd": str(wallet_drawdown_target(risk_allocation())),
            "sizing_policy": "FIXED_K_NATIVE_QUANTITY",
            "sizing_formula": "k = (0.50 * wallet_allocated_risk_usd) / training_unit_k_1_max_drawdown_usd",
            "portfolio_static_floor_usd": "9500",
            "daily_loss_usd": "300",
        },
        "wallets": wallets,
        "portfolios": scenario_portfolios,
        "price_coverage": market.coverage,
        "audited_wallet_count": len(sources),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--start", default="2026-07-01", help="UTC inclusive YYYY-MM-DD")
    parser.add_argument("--train-end", default="2026-10-01", help="UTC exclusive training cutoff")
    parser.add_argument("--end", default="2026-11-01", help="UTC exclusive YYYY-MM-DD")
    args = parser.parse_args()

    def parse_day(value: str) -> datetime:
        result = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
        if result.strftime("%Y-%m-%d") != value:
            raise ValueError("INVALID_DAY_FORMAT")
        return result

    start, train_end, end = map(parse_day, (args.start, args.train_end, args.end))
    result = report(audit_only=args.audit_only, start=start, train_end=train_end, end=end)
    REPORT.parent.mkdir(exist_ok=True)
    out = (
        REPORT.with_name("historical-5m-k-risk-source-audit-20260701-20261101.json")
        if args.audit_only
        else REPORT
    )
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not args.audit_only:
        PROPOSAL.parent.mkdir(exist_ok=True)
        proposals = {
            "schema": "historical_half_wallet_risk_k_proposal_v1",
            "classification": "RESEARCH_PROPOSAL_SCENARIO_ONLY",
            "trading_authorized": False,
            "order_route": "NONE",
            "start_utc": start.isoformat(),
            "end_exclusive_utc": end.isoformat(),
            "train_end_exclusive_utc": train_end.isoformat(),
            "wallet_static_risk_usd": str(risk_allocation()),
            "wallet_target_maxdd_usd": str(wallet_drawdown_target(risk_allocation())),
            "candidate_portfolios": [
                {
                    "name": name,
                    "members": details["members"],
                    "risk_sized_research_k": details["k_by_member"],
                    "accepted_execution_k": details["accepted_executable_k"],
                    "retrospective_scenario": details["chronological_scenario"],
                    "scientifically_qualified": False,
                }
                for name, details in result["portfolios"].items()
            ],
        }
        PROPOSAL.write_text(json.dumps(proposals, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "defaults": result["defaults"],
                "wallets": {
                    name: {"source": v["source"], "k": v["calibrated_k"], "unit": v["unit"]}
                    for name, v in result["wallets"].items()
                },
                "missing_1m_coverage": {
                    k: v for k, v in result["price_coverage"].items() if v["fraction"] < 0.75
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
