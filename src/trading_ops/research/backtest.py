"""Stage-2 walk-forward: daily roster reconciliation, 5m exact kernel steps.

One deep operation: :func:`run_kernel_backtest`. It owns no financial rule.
Roster transitions come from ``copytrade_pm.roster.reconcile_roster`` on a daily
cadence (never inside the 5m loop), and every five-minute decision is
``copytrade_kernel.step_checkpoint`` — the same function the PM03 paper shell and
Guardian call.

What this layer *does* own, and labels as research:

* **Source legs.** A seated wallet's observed per-coin quantity at the bar close
  and the bar-close trade price become ``SourceLeg`` rows. That is a
  retrospective archive reconstruction, not a prospective CWSA packet, so the
  provenance is ``RetrospectiveArchive`` and nothing here authorizes trading.
* **Fill modelling.** An accepted execution projection's ``targets`` are booked
  as immediate modelled fills at the executable side plus explicit fee and
  shortfall, per owner, so the account basis the next checkpoint proves is the
  sum of the owner books the kernel sized.
* **A daily ``account_step``** on modelled equity, so daily/trailing floors are
  evaluated by the kernel accounting engine, not by a research re-implementation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from typing import Any

import polars as pl
from copytrade_kernel import (
    AccountState,
    AccountValuation,
    CheckpointBatch,
    CheckpointMs,
    DecisionInputs,
    DecisionMs,
    FixedKSizing,
    FrozenObject,
    KernelState,
    KnownAtTs,
    MarketContext,
    NativeInstrument,
    OpeningMissed,
    OwnerBook,
    RetrospectiveArchive,
    RiskPolicy,
    Rulebook,
    SourceLeg,
    SourceOutcome,
    VenueCapability,
    account_step,
    step_checkpoint,
)
from copytrade_pm.facts import FactStore, ProvenanceTier
from copytrade_pm.facts.manifest import day_batches
from copytrade_pm.roster import (
    ReconcilerPolicy,
    RosterStore,
    SeatBook,
    SeatSpec,
    digest,
    reconcile_roster,
)

from .golden import GoldenStep, ending_state_rows, golden_step
from .scoring import GRID_MS, ScoreConfig, bench_from_scorecard, score_universe

__all__ = [
    "DAY_MS",
    "BacktestConfig",
    "BacktestReport",
    "BacktestWindow",
    "MarketSpec",
    "run_kernel_backtest",
]

DAY_MS = 86_400_000
_ZERO = Decimal(0)
_POLICY_EPOCH = "0" * 63 + "1"


@dataclass(frozen=True, slots=True)
class BacktestWindow:
    """Half-open UTC millisecond window ``[start_ms, end_ms)`` on the 5m grid."""

    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if type(self.start_ms) is not int or type(self.end_ms) is not int:
            raise ValueError("INVALID_WINDOW")
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("INVALID_WINDOW")
        if self.start_ms % GRID_MS or self.end_ms % GRID_MS:
            raise ValueError("WINDOW_NOT_ON_5M_GRID")

    @classmethod
    def from_utc(cls, start: datetime, end: datetime) -> BacktestWindow:
        return cls(_ms(start), _ms(end))

    @property
    def start_day(self) -> str:
        return datetime.fromtimestamp(self.start_ms / 1000, UTC).date().isoformat()

    @property
    def end_day(self) -> str:
        return datetime.fromtimestamp((self.end_ms - 1) / 1000, UTC).date().isoformat()


def _ms(value: datetime) -> int:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("UTC_DATETIME_REQUIRED")
    stamp = int(value.astimezone(UTC).timestamp() * 1000)
    if stamp < 0:
        raise ValueError("INVALID_CLOCK")
    return stamp


@dataclass(frozen=True, slots=True)
class MarketSpec:
    """The modelled paper-venue envelope for one coin."""

    symbol: str
    qty_step: Decimal = Decimal("0.00001")
    price_tick: Decimal = Decimal("0.1")
    min_qty: Decimal = Decimal("0.00001")
    min_notional: Decimal = Decimal(10)
    liquidity_notional: Decimal = Decimal(1000000)
    initial_rate: Decimal = Decimal("0.02")
    fee_rate: Decimal = Decimal("0.00045")
    funding_bound: Decimal = Decimal("0.0001")
    liquidation_rate: Decimal = Decimal("0.005")
    maintenance_rate: Decimal = Decimal("0.01")

    def __post_init__(self) -> None:
        if not self.symbol or any(character in self.symbol for character in "/: "):
            raise ValueError("INVALID_MARKET_SYMBOL")
        for name in (
            "qty_step",
            "price_tick",
            "min_qty",
            "min_notional",
            "liquidity_notional",
            "initial_rate",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"INVALID_MARKET_SPEC:{name}")

    @property
    def instrument(self) -> NativeInstrument:
        return NativeInstrument(f"hyperliquid/mainnet:perp:default/{self.symbol}")

    def fields(self, *, now_ms: int) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "instrument_type": "perp",
            "verified": True,
            "source_mode": "SYNTHETIC",
            "valid_until_ms": now_ms + 365 * DAY_MS,
            "qty_step": str(self.qty_step),
            "price_tick": str(self.price_tick),
            "min_qty": str(self.min_qty),
            "min_notional": str(self.min_notional),
            "liquidity_notional": str(self.liquidity_notional),
            "initial_rate": str(self.initial_rate),
            "fee_rate": str(self.fee_rate),
            "funding_bound": str(self.funding_bound),
            "liquidation_rate": str(self.liquidation_rate),
            "tiers": [{"cap": "1000000000", "rate": str(self.maintenance_rate)}],
            "source_instrument": self.instrument.key,
        }


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Research choices for one walk-forward run (never trading authorization)."""

    account_id: str = "cft-10k-research"
    initial_equity: Decimal = Decimal(10000)
    lookback_days: int = 30
    reconcile_cadence_ms: int = DAY_MS
    shortfall_bps: Decimal = Decimal(30)
    scale_steps: int = 4
    made_limit: Decimal = Decimal(1000)
    pod_loss_budget: Decimal = Decimal(400)
    hedge_tolerance: Decimal = Decimal("0.05")
    minimum_headroom: Decimal = Decimal(50)
    max_gross_notional: Decimal = Decimal(20000)
    quote_spread_bps: Decimal = Decimal(4)
    scenario_shock: Decimal = Decimal("-0.10")
    score_config: ScoreConfig = field(default_factory=ScoreConfig)
    allow_undated: bool = False

    def __post_init__(self) -> None:
        if not self.account_id:
            raise ValueError("INVALID_ACCOUNT_ID")
        if self.initial_equity <= 0:
            raise ValueError("INVALID_EQUITY")
        if self.reconcile_cadence_ms != DAY_MS:
            raise ValueError("RECONCILE_CADENCE_MUST_BE_DAILY")
        if not 1 <= self.scale_steps <= 1000:
            raise ValueError("INVALID_SCALE_STEPS")
        if self.shortfall_bps < 30:
            raise ValueError("SHORTFALL_BELOW_30_BPS")
        if self.hedge_tolerance > Decimal("0.1"):
            raise ValueError("HEDGE_TOLERANCE")
        if self.lookback_days < 1:
            raise ValueError("INVALID_LOOKBACK")

    def policy_fields(self, coins: Sequence[str], *, now_ms: int) -> dict[str, Any]:
        return {
            "schema": "risk_policy_v1",
            "source_mode": "SYNTHETIC",
            "scale_steps": self.scale_steps,
            "scenarios": [{coin: str(self.scenario_shock) for coin in coins}],
            "minimum_headroom": str(self.minimum_headroom),
            "made_limit": str(self.made_limit),
            "pod_loss_budget": str(self.pod_loss_budget),
            "hedge_tolerance": str(self.hedge_tolerance),
            "shortfall_bps": str(self.shortfall_bps),
            "margin_multiplier": "1",
            "checkpoint_execution": {
                "quote_max_age_ms": GRID_MS,
                "account_max_age_ms": GRID_MS,
                "max_gross_notional": str(self.max_gross_notional),
            },
        }


@dataclass(frozen=True, slots=True)
class BacktestReport:
    """Everything one run produced, including golden decision identities."""

    window: BacktestWindow
    account_id: str
    decisions: tuple[GoldenStep, ...]
    ending_state: KernelState
    ending_equity: Decimal
    allocation_mode_share: dict[str, float]
    challenge_pass_rate: float
    challenge_reasons: tuple[str, ...]
    coverage: dict[str, Any]
    reconcile_plans: tuple[str, ...]
    scorecards_recorded: int
    warnings: tuple[str, ...]
    source_sha: str

    @property
    def decision_count(self) -> int:
        return len(self.decisions)

    @property
    def ending_positions(self) -> tuple[tuple[str, str, str], ...]:
        return ending_state_rows(self.ending_state)

    def data(self) -> dict[str, Any]:
        return {
            "schema": "kernel_backtest_report_v1",
            "window": [self.window.start_ms, self.window.end_ms],
            "account_id": self.account_id,
            "decision_count": self.decision_count,
            "decisions": [step.data() for step in self.decisions],
            "ending_positions": [list(row) for row in self.ending_positions],
            "ending_equity": str(self.ending_equity),
            "allocation_mode_share": dict(sorted(self.allocation_mode_share.items())),
            "challenge_pass_rate": self.challenge_pass_rate,
            "challenge_reasons": list(self.challenge_reasons),
            "coverage": self.coverage,
            "reconcile_plans": list(self.reconcile_plans),
            "scorecards_recorded": self.scorecards_recorded,
            "warnings": list(self.warnings),
            "source_sha": self.source_sha,
        }


# --------------------------------------------------------------------------- #
# research book: positions, entry basis and realized PnL per owner


@dataclass
class _Book:
    """Modelled paper book; the account leg is always the sum of owner legs."""

    positions: dict[str, Decimal] = field(default_factory=dict)
    basis: dict[str, dict[str, Decimal]] = field(default_factory=dict)
    owner_positions: dict[str, dict[str, Decimal]] = field(default_factory=dict)
    owner_basis: dict[str, dict[str, dict[str, Decimal]]] = field(default_factory=dict)
    owner_realized: dict[str, Decimal] = field(default_factory=dict)
    realized: Decimal = _ZERO

    def owner_book(self, owner: str) -> dict[str, Decimal]:
        return self.owner_positions.setdefault(owner, {})

    def apply(
        self,
        *,
        targets: dict[str, str],
        pod_targets: dict[str, dict[str, str]],
        quotes: dict[str, dict[str, Any]],
        markets: dict[str, MarketSpec],
        config: BacktestConfig,
    ) -> Decimal:
        """Book modelled fills for one accepted projection; return cash cost."""
        cost = _ZERO
        for owner in sorted(set(pod_targets) | set(self.owner_positions)):
            holder = self.owner_book(owner)
            basis = self.owner_basis.setdefault(owner, {})
            book = pod_targets.get(owner, {})
            for market in sorted(set(book) | set(holder)):
                desired = Decimal(book.get(market, "0"))
                current = holder.get(market, _ZERO)
                delta = desired - current
                if not delta:
                    continue
                executable = Decimal(quotes[market]["ask"] if delta > 0 else quotes[market]["bid"])
                spec = markets[market]
                delta = (delta / spec.qty_step).to_integral_value(
                    rounding=ROUND_DOWN
                ) * spec.qty_step
                if not delta:
                    continue
                cost += (
                    abs(delta)
                    * executable
                    * (spec.fee_rate + config.shortfall_bps / Decimal(10_000))
                )
                desired = current + delta
                entry = basis.get(market)
                if entry is not None and current and (current > 0) == (delta > 0):
                    closed = _ZERO
                elif entry is not None:
                    closed = min(abs(current), abs(delta))
                    self.owner_realized[owner] = self.owner_realized.get(owner, _ZERO) + (
                        (executable - entry["entry_price"]) * closed * (1 if current > 0 else -1)
                    )
                else:
                    closed = _ZERO
                if not desired:
                    holder.pop(market, None)
                    basis.pop(market, None)
                    continue
                if entry is None or not current or (current > 0) != (desired > 0):
                    new_entry = executable
                else:
                    remaining = abs(current) - closed
                    new_entry = (
                        remaining * entry["entry_price"] + (abs(delta) - closed) * executable
                    ) / abs(desired)
                holder[market] = desired
                basis[market] = {"quantity": desired, "entry_price": new_entry}
        self._reconcile(targets)
        return cost

    def _reconcile(self, targets: dict[str, str]) -> None:
        """Account positions are the owner sum; the kernel guarantees equality."""
        self.positions = {}
        self.basis = {}
        for owner, holder in sorted(self.owner_positions.items()):
            for market, quantity in sorted(holder.items()):
                self.positions[market] = self.positions.get(market, _ZERO) + quantity
        for market, target in targets.items():
            if self.positions.get(market, _ZERO) != Decimal(target):
                raise ValueError(
                    "KERNEL_OWNER_SUM_MISMATCH:"
                    f"{market}:{self.positions.get(market, _ZERO)}!={target}"
                )
        for market, quantity in self.positions.items():
            if not quantity:
                continue
            legs = [
                (abs(holder[market]), holder_basis[market]["entry_price"])
                for owner, holder in self.owner_positions.items()
                for holder_basis in (self.owner_basis.get(owner, {}),)
                if market in holder and market in holder_basis
            ]
            total = sum((size for size, _ in legs), _ZERO)
            if total != abs(quantity):
                raise ValueError("KERNEL_OWNER_BASIS_MISMATCH")
            entry = sum((size * price for size, price in legs), _ZERO) / total
            self.basis[market] = {"quantity": quantity, "entry_price": entry}

    def unrealized(self, prices: dict[str, float]) -> Decimal:
        total = _ZERO
        for market, quantity in self.positions.items():
            entry = self.basis.get(market)
            price = prices.get(market)
            if entry is None or price is None:
                continue
            total += (Decimal(repr(price)) - entry["entry_price"]) * quantity
        return total

    def owner_pnl(self, owner: str, prices: dict[str, float]) -> Decimal:
        pnl = self.owner_realized.get(owner, _ZERO)
        for market, quantity in self.owner_positions.get(owner, {}).items():
            entry = self.owner_basis.get(owner, {}).get(market)
            price = prices.get(market)
            if entry is None or price is None:
                continue
            pnl += (Decimal(repr(price)) - entry["entry_price"]) * quantity
        return pnl


# --------------------------------------------------------------------------- #


def _panel(
    facts: FactStore,
    window: BacktestWindow,
    *,
    lookback_days: int,
    tiers: Iterable[ProvenanceTier] | None,
    allow_undated: bool,
) -> pl.DataFrame:
    """Per ``(wallet, coin, bar)`` observed source quantity and bar-close price."""
    start_day = (
        datetime.fromtimestamp((window.start_ms - lookback_days * DAY_MS) / 1000, UTC)
        .date()
        .isoformat()
    )
    fills = facts.scan(
        "fills",
        start_day=start_day,
        end_day=window.end_day,
        as_of_ms=window.end_ms,
        tiers=tiers,
        allow_unknown_known_at=allow_undated,
    ).collect()
    if fills.height == 0:
        return fills
    normalised = pl.col("side").cast(pl.String).str.to_uppercase()
    sign = (
        pl.when(normalised.is_in(["B", "BUY", "LONG", "BID"]))
        .then(pl.lit(1.0))
        .when(normalised.is_in(["A", "S", "SELL", "SHORT", "ASK"]))
        .then(pl.lit(-1.0))
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    orders = fills.with_columns(
        sign.alias("side_sign"),
        pl.col("size").cast(pl.Float64).alias("size"),
        pl.col("price").cast(pl.Float64).alias("price"),
        pl.col("event_ts_ms").cast(pl.Int64),
        (pl.col("event_ts_ms") // GRID_MS * GRID_MS).alias("bar_ms"),
    ).filter(pl.col("side_sign").is_not_null() & pl.col("size").is_finite() & (pl.col("size") != 0))
    return (
        orders.sort(["wallet", "coin", "event_ts_ms", "arrival_seq"])
        .with_columns(
            (pl.col("side_sign") * pl.col("size"))
            .cum_sum()
            .over(["wallet", "coin"])
            .alias("quantity")
        )
        .group_by(["wallet", "coin", "bar_ms"])
        .agg(
            pl.col("quantity").last().alias("quantity"),
            pl.col("price").last().alias("price"),
        )
        .sort(["coin", "bar_ms", "wallet"])
    )


def _prices(panel: pl.DataFrame) -> dict[str, dict[int, float]]:
    result: dict[str, dict[int, float]] = {}
    for row in panel.group_by(["coin", "bar_ms"]).agg(pl.col("price").last()).iter_rows(named=True):
        result.setdefault(row["coin"], {})[int(row["bar_ms"])] = float(row["price"])
    return result


def _quantities(panel: pl.DataFrame, bar_ms: int) -> dict[str, dict[str, Decimal]]:
    result: dict[str, dict[str, Decimal]] = {}
    for row in panel.filter(pl.col("bar_ms") == bar_ms).iter_rows(named=True):
        quantity = row["quantity"]
        if quantity is None:
            continue
        value = Decimal(repr(float(quantity)))
        if value:
            result.setdefault(row["wallet"], {})[row["coin"]] = value
    return result


def _fmt_decimal(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == 0:
        return "0"
    return format(normalized, "f")


def _quote(price: float, *, spread_bps: Decimal, known_at_ms: int) -> dict[str, Any]:
    half = float(spread_bps) / 20_000.0
    return {
        "bid": _fmt_decimal(Decimal(f"{price * (1.0 - half):.8f}")),
        "ask": _fmt_decimal(Decimal(f"{price * (1.0 + half):.8f}")),
        "known_at_ms": known_at_ms,
    }


def _kernel_state(
    owners: Sequence[str],
    book: _Book,
    memory: dict[str, dict[NativeInstrument, OpeningMissed]],
) -> KernelState:
    return KernelState(
        owner_books=tuple(
            OwnerBook(
                owner=owner,
                positions=tuple(
                    sorted(
                        (market, quantity)
                        for market, quantity in book.owner_positions.get(owner, {}).items()
                        if quantity
                    )
                ),
                entry_memory=tuple(
                    sorted(memory.get(owner, {}).items(), key=lambda row: row[0].key)
                ),
            )
            for owner in sorted(owners)
        )
    )


def _basis_payload(book: _Book) -> dict[str, Any]:
    return {
        market: {"quantity": str(entry["quantity"]), "entry_price": str(entry["entry_price"])}
        for market, entry in sorted(book.basis.items())
    }


def _day_range(start: str, end: str) -> list[str]:
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    days = []
    while first <= last:
        days.append(first.isoformat())
        first += timedelta(days=1)
    return days


def _rulebook_sha(rulebook: Rulebook) -> str:
    return digest(
        {
            "account_binding": rulebook.account_binding,
            "status": rulebook.status,
            "initial": rulebook.initial,
            "target": rulebook.target,
            "daily_loss": rulebook.daily_loss,
            "trailing_loss": rulebook.trailing_loss,
            "internal_daily_stop": rulebook.internal_daily_stop,
            "daily_basis": rulebook.daily_basis,
            "trailing_basis": rulebook.trailing_basis,
            "evidence_sha": rulebook.evidence_sha,
        }
    )


def _source_quantity(frame: CheckpointBatch[Any], owner: str, key: str) -> Decimal:
    for outcome in frame.outcomes:
        if outcome.owner != owner:
            continue
        for leg in outcome.legs:
            if leg.instrument.key == key:
                return leg.quantity
    raise ValueError("Q15_REJECTION_WITHOUT_LEG")


def run_kernel_backtest(
    facts: FactStore,
    *,
    window: BacktestWindow,
    seats: Sequence[SeatSpec],
    rulebook: Rulebook,
    markets: Sequence[MarketSpec] | None = None,
    config: BacktestConfig | None = None,
    reconciler_policy: ReconcilerPolicy | None = None,
    allowed_tiers: Sequence[ProvenanceTier] | None = None,
    roster: RosterStore | None = None,
) -> BacktestReport:
    """Walk forward: reconcile the roster daily, step the kernel every five minutes."""
    if not isinstance(facts, FactStore):
        raise TypeError("FACT_STORE_REQUIRED")
    if not isinstance(window, BacktestWindow):
        raise TypeError("INVALID_WINDOW")
    if not seats or not all(isinstance(seat, SeatSpec) for seat in seats):
        raise TypeError("INVALID_SEATS")
    if not isinstance(rulebook, Rulebook):
        raise TypeError("RULEBOOK_REQUIRED")
    if roster is not None and not isinstance(roster, RosterStore):
        raise TypeError("INVALID_ROSTER")
    settings = config or BacktestConfig()
    if rulebook.account_binding != settings.account_id:
        raise ValueError("RULEBOOK_ACCOUNT")
    policy = reconciler_policy or ReconcilerPolicy.sticky_pit_bench()
    tiers = tuple(allowed_tiers) if allowed_tiers is not None else None

    panel = _panel(
        facts,
        window,
        lookback_days=settings.lookback_days,
        tiers=tiers,
        allow_undated=settings.allow_undated,
    )
    coins = sorted(panel["coin"].unique().to_list()) if panel.height else []
    if not coins:
        raise ValueError("NO_FACT_ROWS_IN_WINDOW")
    specs = {spec.symbol: spec for spec in (markets or [MarketSpec(coin) for coin in coins])}
    if set(coins) - set(specs):
        raise ValueError("MARKET_SPEC_MISSING")
    prices = _prices(panel)

    book_state = SeatBook.empty(
        settings.account_id,
        tuple(seats),
        as_of=KnownAtTs(window.start_ms),
        policy_epoch=_POLICY_EPOCH,
    )
    book = _Book()
    memory: dict[str, dict[NativeInstrument, OpeningMissed]] = {}
    equity = settings.initial_equity
    decisions: list[GoldenStep] = []
    plans: list[str] = []
    modes: list[str] = []
    reasons: set[str] = set()
    warnings: set[str] = set()
    recorded = 0
    account_state: dict[str, Any] = {}
    rulebook_sha = _rulebook_sha(rulebook)
    # One reconciliation per UTC day: at the window start (the entry point of the
    # walk-forward) and at every following midnight, so a mid-day start still
    # steps every five minute bar in its first segment.
    anchors = [window.start_ms]
    cursor = (window.start_ms // DAY_MS + 1) * DAY_MS
    while cursor < window.end_ms:
        anchors.append(cursor)
        cursor += DAY_MS

    for anchor_index, day_ms in enumerate(anchors):
        day_end = anchors[anchor_index + 1] if anchor_index + 1 < len(anchors) else window.end_ms
        scorecard = score_universe(
            facts,
            as_of=KnownAtTs(day_ms),
            lookback_days=settings.lookback_days,
            config=settings.score_config,
            tiers=tiers,
            allow_undated=settings.allow_undated,
        )
        if roster is not None:
            scorecard.record(roster)
        recorded += len(scorecard.cards)
        plan = reconcile_roster(
            current=book_state,
            policy=policy,
            as_of=KnownAtTs(day_ms),
            scorecard=scorecard.by_wallet,
            bench=bench_from_scorecard(scorecard),
        )
        book_state = plan.next_book
        plans.append(plan.sha)
        modes.append(plan.mode.value)
        warnings.update(f"{warning}:{day_ms}" for warning in plan.warnings)
        day_marks = {coin: prices[coin][day_ms] for coin in coins if day_ms in prices.get(coin, {})}
        anchor_cash = settings.initial_equity + book.realized
        anchor_unrealized = book.unrealized(day_marks)
        equity = anchor_cash + anchor_unrealized
        account_state, step_reasons = account_step(
            account_state,
            AccountValuation(
                account_binding=rulebook.account_binding,
                observed_at_ms=day_ms,
                equity=equity,
                cash=anchor_cash,
                unrealized=anchor_unrealized,
                initial_margin=_ZERO,
                maintenance_margin=_ZERO,
                ready=True,
            ),
            rulebook,
            day_ms,
            rulebook_sha=rulebook_sha,
        )
        reasons.update(step_reasons)
        bar = day_ms
        while bar < day_end:
            leases = [lease for _, lease in book_state.seats if lease is not None]
            has_open_residuals = any(any(holder.values()) for holder in book.owner_positions.values())
            if not leases and not has_open_residuals:
                warnings.add(f"FLAT_NO_OCCUPIED_SEAT:{bar}")
                bar += GRID_MS
                continue
            if not leases and has_open_residuals:
                warnings.add(f"FLAT_WITH_DRAINING_RESIDUALS:{bar}")
            quantities = _quantities(panel, bar)
            quotes: dict[str, dict[str, Any]] = {}
            for coin in coins:
                price = prices.get(coin, {}).get(bar)
                if price is None:
                    continue
                quotes[coin] = _quote(
                    price,
                    spread_bps=settings.quote_spread_bps,
                    known_at_ms=bar + GRID_MS - 1000,
                )
            if not quotes:
                warnings.add(f"NO_QUOTE_AT_BAR:{bar}")
                bar += GRID_MS
                continue
            now_ms = bar + GRID_MS
            outcomes = tuple(
                sorted(
                    (
                        SourceOutcome(
                            owner=lease.owner,
                            sizing=FixedKSizing(lease.k),
                            published_at=KnownAtTs(bar),
                            deadline=DecisionMs(now_ms),
                            legs=tuple(
                                SourceLeg(
                                    instrument=specs[coin].instrument,
                                    market=coin,
                                    quantity=quantity,
                                    mark=Decimal(quotes[coin]["bid"]),
                                )
                                for coin, quantity in sorted(
                                    quantities.get(lease.wallet, {}).items()
                                )
                                if coin in quotes
                            ),
                            provenance=RetrospectiveArchive(
                                archive_sha=f"{bar:016x}".rjust(16, "0")
                            ),
                        )
                        for lease in leases
                    ),
                    key=lambda outcome: outcome.owner,
                )
            )
            frame = CheckpointBatch(checkpoint_ms=CheckpointMs(bar), outcomes=outcomes)
            owners = sorted(
                {lease.owner for lease in leases}
                | {owner for owner, holder in book.owner_positions.items() if any(holder.values())}
            )
            state = _kernel_state(owners, book, memory)
            marks = {
                coin: (float(quote["bid"]) + float(quote["ask"])) / 2.0
                for coin, quote in quotes.items()
            }
            cash_now = settings.initial_equity + book.realized
            unrealized_now = book.unrealized(marks)
            equity_now = cash_now + unrealized_now
            account_state, step_reasons = account_step(
                account_state,
                AccountValuation(
                    account_binding=rulebook.account_binding,
                    observed_at_ms=bar,
                    equity=equity_now,
                    cash=cash_now,
                    unrealized=unrealized_now,
                    initial_margin=_ZERO,
                    maintenance_margin=_ZERO,
                    ready=True,
                ),
                rulebook,
                now_ms,
                rulebook_sha=rulebook_sha,
            )
            reasons.update(step_reasons)
            basis = _basis_payload(book)
            inputs = DecisionInputs(
                account=AccountState(
                    finance=FrozenObject.parse(
                        {
                            "equity": account_state.get("equity", str(equity_now)),
                            "headroom": account_state.get("headroom", "0"),
                            "realized_made": account_state.get("realized_made", "0"),
                            "pod_loss_used": {
                                owner: str(max(_ZERO, -book.owner_pnl(owner, marks)))
                                for owner in owners
                            },
                        }
                    ),
                    positions=FrozenObject.parse(
                        {
                            market: str(quantity)
                            for market, quantity in sorted(book.positions.items())
                            if quantity
                        }
                    ),
                    owner_positions=FrozenObject.parse(
                        {
                            owner: {
                                market: str(quantity)
                                for market, quantity in sorted(
                                    book.owner_positions.get(owner, {}).items()
                                )
                                if quantity
                            }
                            for owner in owners
                        }
                    ),
                    ledger_basis=FrozenObject.parse(basis),
                    observed_basis=FrozenObject.parse(basis),
                ),
                market=MarketContext(
                    quotes=FrozenObject.parse(quotes),
                    markets=FrozenObject.parse(
                        {coin: specs[coin].fields(now_ms=now_ms) for coin in quotes}
                    ),
                    capabilities=tuple(
                        VenueCapability(coin, True, True) for coin in sorted(quotes)
                    ),
                ),
                policy=RiskPolicy(
                    FrozenObject.parse(settings.policy_fields(sorted(quotes), now_ms=now_ms))
                ),
                now_ms=DecisionMs(now_ms),
            )
            step = step_checkpoint(state=state, frame=frame, inputs=inputs)
            execution = step.execution
            if execution is None:
                raise ValueError("KERNEL_INPUTS_REQUIRED")
            status = execution.risk.data().get("status")
            if status == "ACCEPTED" and execution.execution_plan is not None:
                projection = execution.execution_plan.data()
                book.realized -= book.apply(
                    targets=projection["targets"],
                    pod_targets=projection["pod_targets"],
                    quotes=quotes,
                    markets=specs,
                    config=settings,
                )
            elif status != "ACCEPTED":
                warnings.add(f"KERNEL_{status}_AT_{bar}")
            else:
                warnings.add(f"KERNEL_PROJECTION_{execution.projection_error}_AT_{bar}")
            for owner, keys in step.opening_rejections:
                for key in keys:
                    memory.setdefault(owner, {})[NativeInstrument(key)] = OpeningMissed(
                        CheckpointMs(bar), _source_quantity(frame, owner, key)
                    )
            decisions.append(
                golden_step(
                    checkpoint_ms=bar,
                    state=state,
                    step=step,
                    next_state=_kernel_state(owners, book, memory),
                )
            )
            equity = settings.initial_equity + book.realized + book.unrealized(marks)
            bar += GRID_MS

    present = {
        batch.day
        for batch in day_batches(
            facts.connect(), "fills", start_day=window.start_day, end_day=window.end_day
        )
    }
    expected = _day_range(window.start_day, window.end_day)
    missing = tuple(day for day in expected if day not in present)
    breaches = {
        reason
        for reason in reasons
        if reason
        in {
            "CONTRACT_DAILY_FLOOR",
            "INTERNAL_DAILY_STOP",
            "TRAILING_FLOOR",
            "MIDNIGHT_BASELINE_UNKNOWN",
            "RULEBOOK_CHANGED_REQUIRES_REVIEW",
            "RULEBOOK_UNVERIFIED",
        }
    }
    share: dict[str, float] = {}
    for mode in modes:
        share[mode] = share.get(mode, 0.0) + 1.0
    if modes:
        share = {mode: round(count / len(modes), 6) for mode, count in share.items()}
    ending_state = _kernel_state(
        sorted(
            {lease.owner for _, lease in book_state.seats if lease is not None}
            | {owner for owner, holder in book.owner_positions.items() if any(holder.values())}
        ),
        book,
        memory,
    )
    return BacktestReport(
        window=window,
        account_id=settings.account_id,
        decisions=tuple(decisions),
        ending_state=ending_state,
        ending_equity=equity,
        allocation_mode_share=share,
        challenge_pass_rate=0.0 if breaches else 1.0,
        challenge_reasons=tuple(sorted(breaches)),
        coverage={
            "kind": "fills",
            "days_expected": len(expected),
            "days_present": len(present),
            "missing_days": len(missing),
            "missing_days_sample": list(missing[:10]),
        },
        reconcile_plans=tuple(plans),
        scorecards_recorded=recorded,
        warnings=tuple(sorted(warnings)),
        source_sha=digest(
            {
                "window": [window.start_ms, window.end_ms],
                "account_id": settings.account_id,
                "decisions": [step.decision_sha for step in decisions],
                "ending": [list(row) for row in ending_state_rows(ending_state)],
            }
        ),
    )
