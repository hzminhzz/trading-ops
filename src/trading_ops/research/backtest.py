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

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from typing import Any

import numpy as np
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
    NetOrderIntent,
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

from .allocation import AllocationPolicy, allocate_fixed_k_portfolio
from .golden import GoldenStep, ending_state_rows, golden_step
from .scoring import (
    GRID_MS,
    ScoreConfig,
    UniverseScorecard,
    bench_from_scorecard,
    euler_marginal_cdar_shares,
    peak_drawdown,
    score_universe,
    sizing_gate_failures,
)

__all__ = [
    "DAY_MS",
    "BacktestConfig",
    "BacktestReport",
    "BacktestWindow",
    "LedgerBar",
    "LedgerFill",
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
    allocation_policy: AllocationPolicy | None = None
    drop_sub_min_opening_legs: bool = False

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
        if self.allocation_policy is not None and not isinstance(
            self.allocation_policy, AllocationPolicy
        ):
            raise TypeError("INVALID_ALLOCATION_POLICY")
        if type(self.drop_sub_min_opening_legs) is not bool:
            raise ValueError("INVALID_DROP_SUB_MIN_OPENING_LEGS")

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
class LedgerFill:
    """One modelled net broker execution booked by the canonical replay ledger."""

    checkpoint_ms: int
    market: str
    net_delta_qty: Decimal
    executable_price: Decimal
    fee_usd: Decimal
    shortfall_usd: Decimal
    realized_pnl_usd: Decimal
    reduce_only: bool

    @property
    def execution_ms(self) -> int:
        """Timestamp of modeled execution at the checkpoint deadline."""
        return self.checkpoint_ms + GRID_MS

    def data(self) -> dict[str, Any]:
        return {
            "checkpoint_ms": self.checkpoint_ms,
            "execution_ms": self.execution_ms,
            "market": self.market,
            "net_delta_qty": str(self.net_delta_qty),
            "executable_price": str(self.executable_price),
            "fee_usd": str(self.fee_usd),
            "shortfall_usd": str(self.shortfall_usd),
            "realized_pnl_usd": str(self.realized_pnl_usd),
            "reduce_only": self.reduce_only,
        }


@dataclass(frozen=True, slots=True)
class LedgerBar:
    """One 5m bar valuation snapshot from the canonical replay ledger."""

    checkpoint_ms: int
    cash: Decimal
    unrealized: Decimal
    equity: Decimal
    realized_pnl: Decimal
    cumulative_costs: Decimal
    gross_notional: Decimal
    net_fill_count: int

    def data(self) -> dict[str, Any]:
        return {
            "checkpoint_ms": self.checkpoint_ms,
            "cash": str(self.cash),
            "unrealized": str(self.unrealized),
            "equity": str(self.equity),
            "realized_pnl": str(self.realized_pnl),
            "cumulative_costs": str(self.cumulative_costs),
            "gross_notional": str(self.gross_notional),
            "net_fill_count": self.net_fill_count,
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
    ledger: tuple[LedgerBar, ...] = ()
    fills: tuple[LedgerFill, ...] = ()
    challenge_target_reached: bool = False
    challenge_outcome: str = "SURVIVED"
    portfolio_constraints_valid: bool = True

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
            "challenge_target_reached": self.challenge_target_reached,
            "challenge_outcome": self.challenge_outcome,
            "portfolio_constraints_valid": self.portfolio_constraints_valid,
            "challenge_reasons": list(self.challenge_reasons),
            "coverage": self.coverage,
            "reconcile_plans": list(self.reconcile_plans),
            "scorecards_recorded": self.scorecards_recorded,
            "warnings": list(self.warnings),
            "ledger": [bar.data() for bar in self.ledger],
            "fills": [fill.data() for fill in self.fills],
            "source_sha": self.source_sha,
        }


# --------------------------------------------------------------------------- #
# canonical replay ledger: net broker execution + owner attribution sub-ledger


def _apply_position_delta(
    positions: dict[str, Decimal],
    basis: dict[str, dict[str, Decimal]],
    *,
    market: str,
    delta: Decimal,
    executable: Decimal,
) -> Decimal:
    """Apply ``delta`` to ``positions``/``basis`` in place; return gross realized PnL."""
    current = positions.get(market, _ZERO)
    desired = current + delta
    entry = basis.get(market)
    realized = _ZERO
    if entry is not None and current and (current > 0) != (delta > 0):
        closed = min(abs(current), abs(delta))
        realized = (executable - entry["entry_price"]) * closed * (1 if current > 0 else -1)
    if not desired:
        positions.pop(market, None)
        basis.pop(market, None)
        return realized
    if entry is None or not current or (current > 0) != (desired > 0):
        new_entry = executable
    elif (current > 0) == (delta > 0):
        new_entry = (abs(current) * entry["entry_price"] + abs(delta) * executable) / abs(desired)
    else:
        new_entry = entry["entry_price"]
    positions[market] = desired
    basis[market] = {"quantity": desired, "entry_price": new_entry}
    return realized


@dataclass
class _ReplayLedger:
    """Canonical replay ledger: net broker execution plus owner attribution.

    Account cash, costs, positions, and basis are driven strictly by net broker
    executions (``NetOrderIntent`` / ``targets`` deltas), never gross owner legs.
    Owner attribution is maintained separately and asserted to sum to account
    positions on every step without forbidding opposing-owner books.
    """

    positions: dict[str, Decimal] = field(default_factory=dict)
    basis: dict[str, dict[str, Decimal]] = field(default_factory=dict)
    realized_pnl: Decimal = _ZERO
    costs: Decimal = _ZERO
    owner_positions: dict[str, dict[str, Decimal]] = field(default_factory=dict)
    owner_basis: dict[str, dict[str, dict[str, Decimal]]] = field(default_factory=dict)
    owner_realized: dict[str, Decimal] = field(default_factory=dict)
    fills: list[LedgerFill] = field(default_factory=list)

    @property
    def realized(self) -> Decimal:
        """Net account realized PnL (gross trading PnL minus execution costs)."""
        return self.realized_pnl - self.costs

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
        checkpoint_ms: int = 0,
        net_orders: tuple[NetOrderIntent, ...] | None = None,
    ) -> tuple[LedgerFill, ...]:
        """Book one accepted execution projection; return the net broker fills."""
        bar_fills, net_deltas = self._apply_net_executions(
            targets=targets,
            quotes=quotes,
            markets=markets,
            config=config,
            checkpoint_ms=checkpoint_ms,
            net_orders=net_orders,
        )
        self._apply_owner_attribution(
            pod_targets=pod_targets,
            quotes=quotes,
            net_deltas=net_deltas,
        )
        self._reconcile(targets)
        self.fills.extend(bar_fills)
        return bar_fills

    def _apply_net_executions(
        self,
        *,
        targets: dict[str, str],
        quotes: dict[str, dict[str, Any]],
        markets: dict[str, MarketSpec],
        config: BacktestConfig,
        checkpoint_ms: int,
        net_orders: tuple[NetOrderIntent, ...] | None,
    ) -> tuple[tuple[LedgerFill, ...], dict[str, Decimal]]:
        if net_orders is not None:
            intent_deltas = {intent.market: intent for intent in net_orders}
            for market in sorted(set(targets) | set(self.positions) | set(intent_deltas)):
                expected = Decimal(targets.get(market, "0")) - self.positions.get(market, _ZERO)
                observed = intent_deltas[market].net_delta_qty if market in intent_deltas else _ZERO
                if expected != observed:
                    raise ValueError(
                        f"KERNEL_NET_INTENT_MISMATCH:{market}:{observed}!={expected}"
                    )
            execution_items = [
                (intent.market, intent.net_delta_qty, intent.reduce_only)
                for intent in sorted(net_orders, key=lambda row: row.market)
                if intent.net_delta_qty
            ]
        else:
            execution_items = []
            for market in sorted(set(targets) | set(self.positions)):
                current = self.positions.get(market, _ZERO)
                delta = Decimal(targets.get(market, "0")) - current
                if delta:
                    reduce_only = bool(current and current * delta < 0 and abs(delta) <= abs(current))
                    execution_items.append((market, delta, reduce_only))

        bar_fills: list[LedgerFill] = []
        net_deltas: dict[str, Decimal] = {}
        for market, delta, reduce_only in execution_items:
            net_deltas[market] = delta
            executable = Decimal(quotes[market]["ask"] if delta > 0 else quotes[market]["bid"])
            spec = markets[market]
            notional = abs(delta) * executable
            fee_usd = notional * spec.fee_rate
            shortfall_usd = notional * (config.shortfall_bps / Decimal(10_000))
            self.costs += fee_usd + shortfall_usd
            fill_realized = _apply_position_delta(
                self.positions,
                self.basis,
                market=market,
                delta=delta,
                executable=executable,
            )
            self.realized_pnl += fill_realized
            bar_fills.append(
                LedgerFill(
                    checkpoint_ms=checkpoint_ms,
                    market=market,
                    net_delta_qty=delta,
                    executable_price=executable,
                    fee_usd=fee_usd,
                    shortfall_usd=shortfall_usd,
                    realized_pnl_usd=fill_realized,
                    reduce_only=reduce_only,
                )
            )
        return tuple(bar_fills), net_deltas

    def _apply_owner_attribution(
        self,
        *,
        pod_targets: dict[str, dict[str, str]],
        quotes: dict[str, dict[str, Any]],
        net_deltas: dict[str, Decimal],
    ) -> None:
        all_owners = sorted(set(pod_targets) | set(self.owner_positions))
        deltas_by_market: dict[str, dict[str, Decimal]] = {}
        for owner in all_owners:
            holder = self.owner_book(owner)
            self.owner_basis.setdefault(owner, {})
            book = pod_targets.get(owner, {})
            for market in sorted(set(book) | set(holder)):
                delta = Decimal(book.get(market, "0")) - holder.get(market, _ZERO)
                if delta:
                    deltas_by_market.setdefault(market, {})[owner] = delta

        for market, owner_deltas in sorted(deltas_by_market.items()):
            mid = (Decimal(quotes[market]["bid"]) + Decimal(quotes[market]["ask"])) / Decimal(2)
            net_delta = net_deltas.get(market, _ZERO)
            if net_delta == _ZERO:
                same_price = opp_price = mid
            else:
                executable = Decimal(
                    quotes[market]["ask"] if net_delta > 0 else quotes[market]["bid"]
                )
                same_side_gross = sum(
                    (abs(d) for d in owner_deltas.values() if d * net_delta > 0), _ZERO
                )
                net_frac = abs(net_delta) / same_side_gross
                same_price = net_frac * executable + (Decimal(1) - net_frac) * mid
                opp_price = mid
            for owner, delta in sorted(owner_deltas.items()):
                fill_px = (
                    same_price if (net_delta != _ZERO and delta * net_delta > 0) else opp_price
                )
                realized = _apply_position_delta(
                    self.owner_positions[owner],
                    self.owner_basis[owner],
                    market=market,
                    delta=delta,
                    executable=fill_px,
                )
                if realized:
                    self.owner_realized[owner] = self.owner_realized.get(owner, _ZERO) + realized

    def _reconcile(self, targets: dict[str, str]) -> None:
        """Account positions must always equal the sum of owner positions."""
        owner_sums: dict[str, Decimal] = {}
        for holder in self.owner_positions.values():
            for market, quantity in holder.items():
                if quantity:
                    owner_sums[market] = owner_sums.get(market, _ZERO) + quantity
        for market in sorted(set(targets) | set(owner_sums) | set(self.positions)):
            expected = Decimal(targets.get(market, "0"))
            account_qty = self.positions.get(market, _ZERO)
            owner_qty = owner_sums.get(market, _ZERO)
            if account_qty != expected or owner_qty != expected:
                raise ValueError(
                    "KERNEL_OWNER_SUM_MISMATCH:"
                    f"{market}:account={account_qty}:owners={owner_qty}!={expected}"
                )

    def unrealized(self, prices: dict[str, float]) -> Decimal:
        total = _ZERO
        for market, quantity in self.positions.items():
            entry = self.basis.get(market)
            price = prices.get(market)
            if entry is None or price is None:
                continue
            total += (Decimal(repr(price)) - entry["entry_price"]) * quantity
        return total

    def gross_notional(self, prices: dict[str, float]) -> Decimal:
        total = _ZERO
        for market, quantity in self.positions.items():
            price = prices.get(market)
            if price is not None and quantity:
                total += abs(quantity) * Decimal(repr(price))
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
        .then(pl.lit(1))
        .when(normalised.is_in(["A", "S", "SELL", "SHORT", "ASK"]))
        .then(pl.lit(-1))
        .otherwise(pl.lit(None, dtype=pl.Int8))
    )
    orders = (
        fills.with_columns(
            sign.alias("side_sign"),
            pl.col("size").cast(pl.Decimal(38, 12)).alias("size_dec"),
            pl.col("price").cast(pl.Float64).alias("price"),
            pl.col("event_ts_ms").cast(pl.Int64),
            (
                pl.max_horizontal(
                    pl.col("event_ts_ms"),
                    pl.col("first_known_at_ms").fill_null(pl.col("event_ts_ms")) - 1,
                )
                // GRID_MS
                * GRID_MS
            ).alias("bar_ms"),
        )
        .filter(
            pl.col("side_sign").is_not_null()
            & pl.col("size_dec").is_not_null()
            & (pl.col("size_dec") != 0)
            & (pl.col("event_ts_ms") < window.end_ms)
            & (pl.col("bar_ms") < window.end_ms)
        )
        .with_columns(
            pl.when(pl.col("side_sign") > 0)
            .then(pl.col("size_dec"))
            .otherwise(-pl.col("size_dec"))
            .alias("signed_size")
        )
    )
    if orders.height == 0:
        return orders.select(
            ["wallet", "coin", "bar_ms", "price"]
        ).with_columns(pl.lit(Decimal(0)).cast(pl.Decimal(38, 12)).alias("quantity"))
    return _reconstruct_causal_panel(orders)


def _reconstruct_causal_panel(orders: pl.DataFrame) -> pl.DataFrame:
    """Reconstruct causal ``(wallet, coin, bar_ms)`` inventory without lookahead or double counting."""
    has_start_col = "start_position" in orders.columns
    select_cols = [
        "wallet",
        "coin",
        "bar_ms",
        "event_ts_ms",
        "arrival_seq",
        "signed_size",
        "price",
    ]
    if has_start_col:
        select_cols.append("start_position")
    ordered = orders.select(select_cols).sort(
        ["wallet", "coin", "bar_ms", "event_ts_ms", "arrival_seq"]
    )

    grouped: dict[tuple[str, str], list[tuple[Any, ...]]] = {}
    for row in ordered.iter_rows():
        grouped.setdefault((str(row[0]), str(row[1])), []).append(row)

    out_rows: list[tuple[str, str, int, Decimal, float]] = []
    for (wallet, coin), rows in sorted(grouped.items()):
        auth_event_key = (-1, -1)
        auth_pos = _ZERO
        post_auth_deltas: list[tuple[tuple[int, int], Decimal]] = []
        by_bar: dict[int, tuple[Decimal, float]] = {}
        for row in rows:
            bar_ms = int(row[2])
            event_key = (int(row[3]), int(row[4]))
            signed_size = row[5] if isinstance(row[5], Decimal) else Decimal(str(row[5]))
            price = float(row[6])
            raw_start = row[7] if has_start_col else None
            if raw_start is not None:
                if event_key > auth_event_key:
                    auth_event_key = event_key
                    start_dec = (
                        raw_start
                        if isinstance(raw_start, Decimal)
                        else Decimal(str(raw_start))
                    )
                    auth_pos = start_dec + signed_size
                    post_auth_deltas = [
                        (k, d) for k, d in post_auth_deltas if k > auth_event_key
                    ]
            elif event_key > auth_event_key:
                post_auth_deltas.append((event_key, signed_size))
            current_pos = auth_pos + sum((d for _, d in post_auth_deltas), _ZERO)
            by_bar[bar_ms] = (current_pos, price)
        for bar_ms, (qty, price) in sorted(by_bar.items()):
            out_rows.append((wallet, coin, bar_ms, qty, price))

    schema = {
        "wallet": pl.String,
        "coin": pl.String,
        "bar_ms": pl.Int64,
        "quantity": pl.Decimal(38, 12),
        "price": pl.Float64,
    }
    if not out_rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(out_rows, schema=schema, orient="row").sort(["coin", "bar_ms", "wallet"])


def _prices(panel: pl.DataFrame) -> dict[str, dict[int, float]]:
    result: dict[str, dict[int, float]] = {}
    if panel.height == 0:
        return result
    valid = panel.filter(pl.col("price").is_finite() & (pl.col("price") > 0))
    for row in valid.group_by(["coin", "bar_ms"]).agg(pl.col("price").last()).iter_rows(named=True):
        result.setdefault(row["coin"], {})[int(row["bar_ms"])] = float(row["price"])
    return result


def _venue_prices(
    facts: FactStore,
    window: BacktestWindow,
    coins: Sequence[str],
    *,
    tiers: Iterable[ProvenanceTier] | None,
    allow_undated: bool,
) -> dict[str, dict[int, float]]:
    """Use only completed venue candles known by the bucket's decision deadline."""
    start_day = datetime.fromtimestamp((window.start_ms - GRID_MS) / 1000, UTC).date().isoformat()
    scan_coins = sorted(set(coins) | {coin.upper() for coin in coins})
    bars = facts.scan(
        "bars_1m",
        coins=scan_coins,
        start_day=start_day,
        end_day=window.end_day,
        as_of_ms=window.end_ms,
        tiers=tiers,
        allow_unknown_known_at=allow_undated,
    ).collect()
    if bars.height == 0:
        return {}
    bars = (
        bars.with_columns(
            (pl.col("event_ts_ms") // GRID_MS * GRID_MS).alias("bar_ms"),
            pl.col("close").cast(pl.Float64).alias("price"),
        )
        .filter(
            (pl.col("event_ts_ms") >= window.start_ms - GRID_MS)
            & (pl.col("event_ts_ms") + 60_000 <= pl.col("bar_ms") + GRID_MS)
            & (pl.col("bar_ms") < window.end_ms)
            & (
                pl.col("first_known_at_ms").is_null()
                | (pl.col("first_known_at_ms") <= pl.col("bar_ms") + GRID_MS)
            )
            & pl.col("price").is_finite()
            & (pl.col("price") > 0)
        )
        .sort(["coin", "bar_ms", "event_ts_ms", "source_rank", "arrival_seq"])
    )
    by_upper = {coin.upper(): series for coin, series in _prices(bars).items()}
    return {coin: by_upper[coin.upper()] for coin in coins if coin.upper() in by_upper}


def _quantities_by_bar(panel: pl.DataFrame) -> dict[int, dict[str, dict[str, Decimal]]]:
    result: dict[int, dict[str, dict[str, Decimal]]] = {}
    if panel.height == 0:
        return result
    for bar_ms, wallet, coin, quantity in panel.select(
        ["bar_ms", "wallet", "coin", "quantity"]
    ).iter_rows():
        if quantity is None:
            continue
        value = quantity if isinstance(quantity, Decimal) else Decimal(str(quantity))
        # Zero is an explicit inventory clear, distinct from no-trade quiet bars.
        result.setdefault(int(bar_ms), {}).setdefault(wallet, {})[coin] = value.normalize()
    return result


def _update_source_positions(
    positions: dict[str, dict[str, Decimal]],
    updates: dict[str, dict[str, Decimal]],
) -> None:
    """Apply explicit inventory updates; quiet bars carry the previous quantity."""
    for wallet, changes in updates.items():
        book = positions.setdefault(wallet, {})
        for coin, quantity in changes.items():
            if quantity:
                book[coin] = quantity
            else:
                book.pop(coin, None)


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
    book: _ReplayLedger,
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


def _basis_payload(book: _ReplayLedger) -> dict[str, Any]:
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


def _validate_seated_portfolio(
    book_state: SeatBook,
    scorecard: UniverseScorecard,
    policy: AllocationPolicy,
    day_ms: int,
    *,
    draining_anchors: Mapping[str, Decimal] | None = None,
) -> list[str]:
    """Validate seated portfolio (incumbents, promotions, and draining residuals)."""
    leases = [lease for _, lease in book_state.seats if lease is not None]
    active_wallets = {lease.wallet for lease in leases}
    anchors = {
        wallet: k_val
        for wallet, k_val in sorted((draining_anchors or {}).items())
        if wallet not in active_wallets and k_val > _ZERO
    }
    if not leases and not anchors:
        return []
    warnings: list[str] = []
    by_wallet = scorecard.by_wallet
    portfolio_entries: list[tuple[str, Decimal]] = []
    for lease in leases:
        if lease.wallet not in by_wallet:
            warnings.append(f"SEATED_SCORECARD_MISSING:{day_ms}:{lease.wallet}")
            continue
        portfolio_entries.append((lease.wallet, lease.k))
    for wallet, k_val in sorted(anchors.items()):
        if wallet not in by_wallet:
            warnings.append(f"SEATED_SCORECARD_MISSING:{day_ms}:{wallet}")
            continue
        portfolio_entries.append((wallet, k_val))
    if not portfolio_entries:
        return warnings

    matrix = np.array(
        [scorecard.wallet_returns(wallet) for wallet, _ in portfolio_entries],
        dtype=np.float64,
    )
    weights = np.array([float(k_val) for _, k_val in portfolio_entries], dtype=np.float64)
    seated_dd = peak_drawdown((matrix * weights[:, None]).sum(axis=0)) * float(
        policy.pod_equity_usd
    )
    if (
        policy.portfolio_dd_budget_usd is not None
        and seated_dd > float(policy.portfolio_dd_budget_usd) + 1e-9
    ):
        warnings.append(f"SEATED_PORTFOLIO_DD_ABOVE_BUDGET:{day_ms}")
    cdar_shares = euler_marginal_cdar_shares(matrix, weights, policy.cdar_alpha)
    for idx, (wallet, k_val) in enumerate(portfolio_entries):
        raw_p95, raw_max = scorecard.wallet_unscaled_participation(wallet)
        failures = sizing_gate_failures(
            raw_p95 * float(k_val),
            raw_max * float(k_val),
            float(cdar_shares[idx]),
            max_p95=policy.max_p95_participation,
            max_max=policy.max_max_participation,
            max_cdar=policy.max_euler_cdar_share,
        )
        if failures:
            codes = ",".join(gate.value for gate in failures)
            warnings.append(f"SEATED_PORTFOLIO_CONSTRAINT_BREACH:{day_ms}:{wallet}:{codes}")
    return warnings


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
    prices = _venue_prices(
        facts, window, coins, tiers=tiers, allow_undated=settings.allow_undated
    )
    quantities_by_bar = _quantities_by_bar(panel)
    source_positions: dict[str, dict[str, Decimal]] = {}
    for prior_bar in sorted(b for b in quantities_by_bar if b < window.start_ms):
        _update_source_positions(source_positions, quantities_by_bar[prior_bar])

    book_state = SeatBook.empty(
        settings.account_id,
        tuple(seats),
        as_of=KnownAtTs(window.start_ms),
        policy_epoch=_POLICY_EPOCH,
    )
    book = _ReplayLedger()
    memory: dict[str, dict[NativeInstrument, OpeningMissed]] = {}
    equity = settings.initial_equity
    decisions: list[GoldenStep] = []
    ledger_bars: list[LedgerBar] = []
    plans: list[str] = []
    modes: list[str] = []
    reasons: set[str] = set()
    warnings: set[str] = set()
    wallet_by_owner: dict[str, str] = {}
    k_by_owner: dict[str, Decimal] = {}
    portfolio_constraints_valid = True
    if settings.drop_sub_min_opening_legs:
        warnings.add("EXPERIMENTAL_RESEARCH_OPENING_DUST_FILTER")
    recorded = 0
    account_state: dict[str, Any] = {}
    rulebook_sha = _rulebook_sha(rulebook)
    target_equity = settings.initial_equity + Decimal(rulebook.target)
    target_reached = False
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
        draining_anchors: dict[str, Decimal] = {}
        if settings.allocation_policy is not None:
            by_wallet = scorecard.by_wallet
            retained_incumbents = {
                lease.wallet: lease.k
                for _, lease in book_state.seats
                if lease is not None
                and lease.wallet in by_wallet
                and not any(gate in policy.evict_on for gate in by_wallet[lease.wallet].failures)
            }
            active_retained_owners = {
                lease.owner
                for _, lease in book_state.seats
                if lease is not None and lease.wallet in retained_incumbents
            }
            draining_anchors = {
                wallet_by_owner[owner]: k_by_owner[owner]
                for owner, holder in book.owner_positions.items()
                if any(qty != _ZERO for qty in holder.values())
                and owner not in active_retained_owners
                and owner in wallet_by_owner
                and wallet_by_owner[owner] not in retained_incumbents
            }
            bench = allocate_fixed_k_portfolio(
                scorecard,
                policy=settings.allocation_policy,
                incumbent_k={**draining_anchors, **retained_incumbents},
            ).to_bench_candidates(scorecard)
        else:
            bench = bench_from_scorecard(scorecard)
        plan = reconcile_roster(
            current=book_state,
            policy=policy,
            as_of=KnownAtTs(day_ms),
            scorecard=scorecard.by_wallet,
            bench=bench,
        )
        if settings.allocation_policy is not None:
            seated_violations = _validate_seated_portfolio(
                plan.next_book,
                scorecard,
                settings.allocation_policy,
                day_ms,
                draining_anchors=draining_anchors,
            )
            if seated_violations:
                old_wallets = {
                    lease.wallet for _, lease in book_state.seats if lease is not None
                }
                new_wallets = {
                    lease.wallet for _, lease in plan.next_book.seats if lease is not None
                }
                if new_wallets - old_wallets:
                    warnings.add(f"SEATED_PROMOTIONS_BLOCKED:{day_ms}")
                    plan = reconcile_roster(
                        current=book_state,
                        policy=policy,
                        as_of=KnownAtTs(day_ms),
                        scorecard=scorecard.by_wallet,
                        bench=(),
                    )
                    seated_violations = _validate_seated_portfolio(
                        plan.next_book,
                        scorecard,
                        settings.allocation_policy,
                        day_ms,
                        draining_anchors=draining_anchors,
                    )
            if seated_violations:
                portfolio_constraints_valid = False
                warnings.add(f"SEATED_PORTFOLIO_UNQUALIFIED:{day_ms}")
                warnings.update(seated_violations)
        book_state = plan.next_book
        for _, lease in book_state.seats:
            if lease is not None:
                wallet_by_owner[lease.owner] = lease.wallet
                k_by_owner[lease.owner] = lease.k
        plans.append(plan.sha)
        modes.append(plan.mode.value)
        warnings.update(f"{warning}:{day_ms}" for warning in plan.warnings)
        previous_bar = day_ms - GRID_MS
        day_marks = {
            coin: prices[coin][previous_bar]
            for coin in coins
            if previous_bar in prices.get(coin, {})
        }
        held_coins = {
            coin for coin, quantity in book.positions.items() if quantity
        } | {
            coin
            for holder in book.owner_positions.values()
            for coin, quantity in holder.items()
            if quantity
        }
        if held_coins - day_marks.keys():
            raise ValueError(f"MISSING_HELD_MARK_AT_DAILY_ANCHOR:{day_ms}")
        anchor_cash = settings.initial_equity + book.realized
        anchor_unrealized = book.unrealized(day_marks)
        equity = anchor_cash + anchor_unrealized
        if Decimal(rulebook.target) > 0 and equity >= target_equity:
            target_reached = True
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
            _update_source_positions(source_positions, quantities_by_bar.get(bar, {}))
            leases = [lease for _, lease in book_state.seats if lease is not None]
            has_open_residuals = any(any(holder.values()) for holder in book.owner_positions.values())
            if not leases and not has_open_residuals:
                warnings.add(f"FLAT_NO_OCCUPIED_SEAT:{bar}")
                bar += GRID_MS
                continue
            if not leases and has_open_residuals:
                warnings.add(f"FLAT_WITH_DRAINING_RESIDUALS:{bar}")
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
            required_coins = (
                {
                    coin
                    for lease in leases
                    for coin in source_positions.get(lease.wallet, {})
                }
                | {coin for coin, quantity in book.positions.items() if quantity}
                | {
                    coin
                    for holder in book.owner_positions.values()
                    for coin, quantity in holder.items()
                    if quantity
                }
            )
            missing_quotes = required_coins - quotes.keys()
            if missing_quotes:
                raise ValueError(f"MISSING_REQUIRED_MARK:{bar}:{','.join(sorted(missing_quotes))}")
            if not quotes:
                warnings.add(f"NO_QUOTE_AT_BAR:{bar}")
                bar += GRID_MS
                continue
            now_ms = bar + GRID_MS
            outcome_list: list[SourceOutcome[RetrospectiveArchive]] = []
            for lease in leases:
                owner_held = book.owner_positions.get(lease.owner, {})
                legs: list[SourceLeg] = []
                for coin, quantity in sorted(source_positions.get(lease.wallet, {}).items()):
                    if settings.drop_sub_min_opening_legs and owner_held.get(coin, _ZERO) == _ZERO:
                        spec = specs[coin]
                        mark_px = Decimal(quotes[coin]["bid"])
                        exec_px = Decimal(
                            quotes[coin]["ask"] if quantity > 0 else quotes[coin]["bid"]
                        )
                        req_notional = abs(quantity * mark_px * lease.k)
                        step_qty = ((req_notional / exec_px) / spec.qty_step).to_integral_value(
                            rounding=ROUND_DOWN
                        ) * spec.qty_step
                        if (
                            step_qty == _ZERO
                            or step_qty < spec.min_qty
                            or (step_qty * exec_px) < spec.min_notional
                        ):
                            warnings.add(f"DROPPED_SUB_MIN_OPENING_LEG:{bar}:{coin}")
                            continue
                    legs.append(
                        SourceLeg(
                            instrument=specs[coin].instrument,
                            market=coin,
                            quantity=quantity,
                            mark=Decimal(quotes[coin]["bid"]),
                        )
                    )
                outcome_list.append(
                    SourceOutcome(
                        owner=lease.owner,
                        sizing=FixedKSizing(lease.k),
                        published_at=KnownAtTs(bar),
                        deadline=DecisionMs(now_ms),
                        legs=tuple(legs),
                        provenance=RetrospectiveArchive(archive_sha=f"{bar:016x}".rjust(16, "0")),
                    )
                )
            outcomes = tuple(sorted(outcome_list, key=lambda outcome: outcome.owner))
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
            bar_fills: tuple[LedgerFill, ...] = ()
            if status == "ACCEPTED" and execution.execution_plan is not None:
                projection = execution.execution_plan.data()
                bar_fills = book.apply(
                    targets=projection["targets"],
                    pod_targets=projection["pod_targets"],
                    quotes=quotes,
                    markets=specs,
                    config=settings,
                    checkpoint_ms=bar,
                    net_orders=execution.net_orders,
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
            bar_cash = settings.initial_equity + book.realized
            bar_unrealized = book.unrealized(marks)
            equity = bar_cash + bar_unrealized
            if bar_fills or bar + GRID_MS >= day_end:
                account_state, post_reasons = account_step(
                    account_state,
                    AccountValuation(
                        account_binding=rulebook.account_binding,
                        observed_at_ms=bar,
                        equity=equity,
                        cash=bar_cash,
                        unrealized=bar_unrealized,
                        initial_margin=_ZERO,
                        maintenance_margin=_ZERO,
                        ready=True,
                    ),
                    rulebook,
                    now_ms,
                    rulebook_sha=rulebook_sha,
                )
                reasons.update(post_reasons)
            if Decimal(rulebook.target) > 0 and equity >= target_equity:
                target_reached = True
            ledger_bars.append(
                LedgerBar(
                    checkpoint_ms=bar,
                    cash=bar_cash,
                    unrealized=bar_unrealized,
                    equity=equity,
                    realized_pnl=book.realized_pnl,
                    cumulative_costs=book.costs,
                    gross_notional=book.gross_notional(marks),
                    net_fill_count=len(bar_fills),
                )
            )
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
    challenge_target_reached = bool(target_reached and not breaches)
    challenge_outcome = (
        "BREACHED" if breaches else ("PASSED" if challenge_target_reached else "SURVIVED")
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
        ledger=tuple(ledger_bars),
        fills=tuple(book.fills),
        challenge_target_reached=challenge_target_reached,
        challenge_outcome=challenge_outcome,
        portfolio_constraints_valid=portfolio_constraints_valid,
    )
