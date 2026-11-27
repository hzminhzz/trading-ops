"""Stage-2 ``run_kernel_backtest``: daily reconcile, 5m kernel steps."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from copytrade_kernel import KnownAtTs, Rulebook, RulebookCatalog
from copytrade_pm.facts import FactIngestor
from copytrade_pm.roster import RosterStore, SeatSpec
from research_fixtures import (
    DAY_MS,
    START,
    WALLET_A,
    WALLET_B,
    build_store,
    equity_rows,
    wallet_round_trips,
)

from trading_ops.research import (
    GRID_MS,
    BacktestConfig,
    BacktestWindow,
    MarketSpec,
    ScoreConfig,
    run_kernel_backtest,
)

PERMISSIVE = ScoreConfig(
    min_baseline_order_usd=Decimal(1),
    # A research k cap: the synthetic path has almost no drawdown, so the
    # unconstrained fixed-k rule would size the whole book out of the gross cap.
    max_fixed_k=Decimal("0.005"),
    max_euler_cdar_share=Decimal(1),
    max_p95_participation_1m=Decimal(1),
    max_max_participation_1m=Decimal(1),
)
SEATS = (
    SeatSpec("seat_1", Decimal(200), Decimal("0.60")),
    SeatSpec("seat_2", Decimal(200), Decimal("0.60")),
)


def _rulebook() -> Rulebook:
    return Rulebook(
        **RulebookCatalog.get("crypto-fund-trader:two_phase:phase_1").to_rulebook_fields(
            account_binding="cft-10k-research", scale=2, evidence_sha="d" * 64
        )
    )


WINDOW_START = START + 3 * DAY_MS


def _store(tmp_path: Path):
    """Seven days of tape so the walk-forward window has a full lookback behind it."""
    fills = wallet_round_trips(
        WALLET_A, "BTC", start_ms=START, bars=7 * 288, base_price=20000.0, size=0.1
    )
    fills += wallet_round_trips(
        WALLET_B, "ETH", start_ms=START, bars=7 * 288, base_price=2000.0, size=1.0
    )
    return build_store(
        tmp_path,
        fills=fills,
        equity={WALLET_A: (Decimal(12000), 9), WALLET_B: (Decimal(9000), 9)},
    )


def _run(store, **overrides):
    config = BacktestConfig(
        lookback_days=2,
        allow_undated=True,
        score_config=PERMISSIVE,
        **overrides,
    )
    return run_kernel_backtest(
        store,
        window=BacktestWindow(WINDOW_START, WINDOW_START + 2 * DAY_MS),
        seats=SEATS,
        rulebook=_rulebook(),
        markets=(MarketSpec("BTC"), MarketSpec("ETH")),
        config=config,
    )


def test_backtest_steps_every_bar_and_reconciles_once_per_day(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        report = _run(store)
    finally:
        store.close()

    # One reconciliation per UTC day: the entry anchor plus the next midnight.
    assert len(report.reconcile_plans) == 2
    # Every five minute bar in the window produced exactly one kernel decision.
    assert report.decision_count == 2 * DAY_MS // GRID_MS
    assert report.allocation_mode_share == {"ACTIVE": 1.0}
    assert report.coverage["missing_days"] == 0
    assert report.coverage["days_expected"] == 2
    assert report.challenge_pass_rate == 1.0
    assert set(report.challenge_reasons) == set()


def test_decision_identity_is_deterministic_across_runs(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        first = _run(store)
        second = _run(store)
    finally:
        store.close()

    assert [step.decision_sha for step in first.decisions] == [
        step.decision_sha for step in second.decisions
    ]
    assert first.source_sha == second.source_sha
    assert first.ending_positions == second.ending_positions
    assert first.ending_equity == second.ending_equity
    # A decision digest must be a real 64-hex identity, not a placeholder.
    assert all(len(step.decision_sha) == 64 for step in first.decisions)
    assert len({step.decision_sha for step in first.decisions}) > 1


def test_positions_and_basis_follow_the_kernel_projection(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        report = _run(store)
    finally:
        store.close()

    ending = {(owner, market): quantity for owner, market, quantity in report.ending_positions}
    assert ending, "the kernel must have opened at least one seat position"
    assert all(Decimal(quantity) != 0 for quantity in ending.values())
    # Net intents are exact strings, never floats.
    for step in report.decisions:
        for intent in step.net_intents:
            assert isinstance(intent["net_delta_qty"], str)
            assert intent["net_delta_qty"] != "0"
    assert report.ending_equity != Decimal(10000)


def test_backtest_records_stage1_scorecards_into_the_roster(tmp_path: Path) -> None:
    store = _store(tmp_path)
    roster = RosterStore(tmp_path / "roster.sqlite", create=True)
    try:
        report = run_kernel_backtest(
            store,
            window=BacktestWindow(WINDOW_START, WINDOW_START + 2 * DAY_MS),
            seats=SEATS,
            rulebook=_rulebook(),
            markets=(MarketSpec("BTC"), MarketSpec("ETH")),
            config=BacktestConfig(lookback_days=2, allow_undated=True, score_config=PERMISSIVE),
            roster=roster,
        )
        cards = roster.latest_scorecard(as_of=KnownAtTs(WINDOW_START + 2 * DAY_MS))
    finally:
        roster.close()
        store.close()

    assert report.scorecards_recorded == 4  # two daily anchors x two wallets
    assert set(cards) == {WALLET_A, WALLET_B}


def test_reconcile_cadence_is_pinned_to_daily(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="RECONCILE_CADENCE_MUST_BE_DAILY"):
        BacktestConfig(reconcile_cadence_ms=GRID_MS)


def test_mid_backtest_eviction_preserves_frozen_residuals_promotes_bench_and_renets(
    tmp_path: Path,
) -> None:
    fills = wallet_round_trips(
        WALLET_A, "BTC", start_ms=START, bars=7 * 288, base_price=20000.0, size=0.1
    )
    fills += wallet_round_trips(
        WALLET_B, "BTC", start_ms=START, bars=7 * 288, base_price=20000.0, size=0.1
    )
    # WALLET_A starts eligible with $12k, then drops below $1k before Day 2 anchor.
    # WALLET_B stays eligible at $9k on the bench and gets promoted onto seat_1 on Day 2.
    store = build_store(
        tmp_path,
        fills=fills,
        equity={WALLET_A: (Decimal(12000), 3), WALLET_B: (Decimal(9000), 9)},
    )
    ingestor = FactIngestor(store)
    drop_ts = WINDOW_START + DAY_MS - 60_000
    ingestor.ingest_rows(
        "equity",
        [
            {
                "wallet": WALLET_A,
                "account_scope": "perp",
                "abstraction_mode": "unified",
                "equity_usd": Decimal(500),
                "withdrawable_usd": Decimal(500),
                "margin_used_usd": Decimal(0),
                "observed_at": datetime.fromtimestamp(drop_ts / 1000, UTC),
                "event_ts_ms": drop_ts,
                "first_known_at_ms": drop_ts + 1000,
                "arrival_seq": 99,
                "source_id": "synthetic-capture",
                "source_rank": 1,
                "batch_sha16": "",
            }
        ],
        source_id="synthetic-capture",
        source_rank=1,
        first_known_at_ms=None,
    )
    try:
        report = run_kernel_backtest(
            store,
            window=BacktestWindow(WINDOW_START, WINDOW_START + 2 * DAY_MS),
            seats=(SeatSpec("seat_1", Decimal(200), Decimal("1.0")),),
            rulebook=_rulebook(),
            markets=(MarketSpec("BTC"),),
            config=BacktestConfig(
                lookback_days=2,
                allow_undated=True,
                score_config=ScoreConfig(
                    min_baseline_order_usd=Decimal(1),
                    max_fixed_k=Decimal("0.005"),
                    max_euler_cdar_share=Decimal(1),
                    max_p95_participation_1m=Decimal(1),
                    max_max_participation_1m=Decimal(1),
                ),
            ),
        )
    finally:
        store.close()

    assert len(report.reconcile_plans) == 2
    assert report.decision_count == 2 * DAY_MS // GRID_MS
    assert report.allocation_mode_share == {"ACTIVE": 1.0}
    day1_positions = {
        (owner, market): qty
        for owner, market, qty in report.decisions[DAY_MS // GRID_MS - 1].ending_positions
    }
    assert len(day1_positions) == 1
    (g1_owner, g1_market), g1_qty = next(iter(day1_positions.items()))
    assert g1_market == "BTC"
    assert Decimal(g1_qty) > 0

    ending = {(owner, market): qty for owner, market, qty in report.ending_positions}
    # Evicted owner g1's residual is preserved (frozen_pods) alongside the Q14
    # bench-promoted owner g2 on the same market (re-netted by the kernel).
    assert (g1_owner, "BTC") in ending
    assert ending[(g1_owner, "BTC")] == g1_qty
    g2_owners = {owner for owner, market in ending if owner != g1_owner and market == "BTC"}
    assert len(g2_owners) == 1
    g2_owner = next(iter(g2_owners))
    assert Decimal(ending[(g2_owner, "BTC")]) > 0


def test_flat_with_draining_residuals_steps_kernel_and_retains_ending_state(
    tmp_path: Path,
) -> None:
    fills = wallet_round_trips(
        WALLET_A, "BTC", start_ms=START, bars=7 * 288, base_price=20000.0, size=0.1
    )
    store = build_store(
        tmp_path,
        fills=fills,
        equity={WALLET_A: (Decimal(12000), 3)},
    )
    ingestor = FactIngestor(store)
    drop_ts = WINDOW_START + DAY_MS - 60_000
    ingestor.ingest_rows(
        "equity",
        [
            {
                **equity_rows(WALLET_A, start_ms=drop_ts, equity_usd=Decimal(500), days=1)[0],
                "arrival_seq": 99,
            }
        ],
        source_id="synthetic-capture",
        source_rank=1,
        first_known_at_ms=None,
    )
    try:
        report = run_kernel_backtest(
            store,
            window=BacktestWindow(WINDOW_START, WINDOW_START + 2 * DAY_MS),
            seats=(SeatSpec("seat_1", Decimal(200), Decimal("1.0")),),
            rulebook=_rulebook(),
            markets=(MarketSpec("BTC"),),
            config=BacktestConfig(lookback_days=2, allow_undated=True, score_config=PERMISSIVE),
        )
    finally:
        store.close()

    assert report.allocation_mode_share == {"ACTIVE": 0.5, "FLAT": 0.5}
    assert report.decision_count == 2 * DAY_MS // GRID_MS
    assert any(w.startswith("FLAT_WITH_DRAINING_RESIDUALS:") for w in report.warnings)
    assert not any(w.startswith("FLAT_NO_OCCUPIED_SEAT:") for w in report.warnings)
    day1_positions = {
        (owner, market): qty
        for owner, market, qty in report.decisions[DAY_MS // GRID_MS - 1].ending_positions
    }
    assert len(day1_positions) == 1
    (g1_owner, _), g1_qty = next(iter(day1_positions.items()))
    ending = {(owner, market): qty for owner, market, qty in report.ending_positions}
    assert ending == {(g1_owner, "BTC"): g1_qty}


def test_backtest_computes_headroom_and_realized_made_via_account_step(
    tmp_path: Path, monkeypatch
) -> None:
    from trading_ops.research import backtest as research_backtest

    captured_finances: list[dict] = []
    real_step = research_backtest.step_checkpoint

    def spy(*, state, frame, inputs=None):
        if inputs is not None:
            captured_finances.append(inputs.account.finance.data())
        return real_step(state=state, frame=frame, inputs=inputs)

    monkeypatch.setattr(research_backtest, "step_checkpoint", spy)

    fills = wallet_round_trips(
        WALLET_A, "BTC", start_ms=START, bars=4 * 288, base_price=20000.0, size=0.1
    )
    store = build_store(
        tmp_path,
        fills=fills,
        equity={WALLET_A: (Decimal(12000), 6)},
    )
    try:
        run_kernel_backtest(
            store,
            window=BacktestWindow(WINDOW_START, WINDOW_START + 3 * GRID_MS),
            seats=(SeatSpec("seat_1", Decimal(200), Decimal("1.0")),),
            rulebook=_rulebook(),
            markets=(MarketSpec("BTC"),),
            config=BacktestConfig(
                lookback_days=2,
                allow_undated=True,
                score_config=PERMISSIVE,
            ),
        )
    finally:
        store.close()

    assert len(captured_finances) == 3
    # Rulebook has initial=10000, internal_daily_stop=400, daily_loss=500, trailing_loss=1000.
    # risk_floor is 9600, so initial headroom must be 400 (NOT 10000 - 1000 = 9000).
    assert Decimal(captured_finances[0]["headroom"]) == Decimal(400)
    assert Decimal(captured_finances[0]["realized_made"]) == Decimal(0)
    # After the first fill incurs costs, intraday drawdown (realized_made) is positive.
    assert Decimal(captured_finances[1]["realized_made"]) > Decimal(0)
    assert Decimal(captured_finances[1]["headroom"]) < Decimal(400)


def test_closed_position_pnl_reaches_account_cash_after_costs() -> None:
    """A profitable or losing round trip changes account cash, not just owner PnL."""
    from trading_ops.research.backtest import _ReplayLedger

    for exit_price, expected_cash in (("110", "10009.27550"), ("90", "9989.34450")):
        ledger = _ReplayLedger()
        markets = {"BTC": MarketSpec("BTC")}
        config = BacktestConfig()
        for target, price in (("1", "100"), ("0", exit_price)):
            ledger.apply(
                targets={"BTC": target},
                pod_targets={"synthetic": {"BTC": target}},
                quotes={"BTC": {"ask": price, "bid": price}},
                markets=markets,
                config=config,
            )
        assert not ledger.positions
        assert config.initial_equity + ledger.realized == Decimal(expected_cash)


def test_opposing_owners_reconcile_without_basis_mismatch() -> None:
    """Opposing owner positions in the same market must net cleanly without crashing basis."""
    from trading_ops.research.backtest import _ReplayLedger

    ledger = _ReplayLedger()
    markets = {"BTC": MarketSpec("BTC")}
    config = BacktestConfig()
    fills = ledger.apply(
        targets={"BTC": "1"},
        pod_targets={"owner_a": {"BTC": "2"}, "owner_b": {"BTC": "-1"}},
        quotes={"BTC": {"ask": "100", "bid": "100"}},
        markets=markets,
        config=config,
    )
    assert ledger.positions == {"BTC": Decimal(1)}
    assert ledger.basis["BTC"]["entry_price"] == Decimal(100)
    assert len(fills) == 1
    assert fills[0].net_delta_qty == Decimal(1)
    # Fee + shortfall on 1 net BTC ($100 * 34.5 bps = $0.345), NOT 3 gross BTC.
    assert ledger.costs == Decimal("0.34500")


def test_internally_netted_zero_delta_incurs_zero_broker_cost() -> None:
    """Owners offsetting to zero net delta must incur zero broker fees or shortfall."""
    from trading_ops.research.backtest import _ReplayLedger

    ledger = _ReplayLedger()
    markets = {"BTC": MarketSpec("BTC")}
    config = BacktestConfig()
    fills = ledger.apply(
        targets={"BTC": "0"},
        pod_targets={"owner_a": {"BTC": "1"}, "owner_b": {"BTC": "-1"}},
        quotes={"BTC": {"ask": "100", "bid": "100"}},
        markets=markets,
        config=config,
    )
    assert fills == ()
    assert ledger.costs == Decimal(0)
    assert ledger.positions == {}


def test_swing_position_survives_quiet_checkpoint_and_emits_canonical_ledger(
    tmp_path: Path,
) -> None:
    """Quiet checkpoints carry source positions and emit bar-level canonical ledger rows."""
    from dataclasses import replace

    from research_fixtures import _fill, bars_1m_from_fills

    training = wallet_round_trips(
        WALLET_A,
        "BTC",
        start_ms=WINDOW_START - 2 * DAY_MS,
        bars=2 * 288 - 1,
        base_price=10000.0,
        size=0.078125,
    )
    net = sum(
        Decimal(str(row["size"])) * (1 if row["side"] == "buy" else -1) for row in training
    )
    fills = training + [
        _fill(
            WALLET_A,
            "BTC",
            len(training),
            at_ms=WINDOW_START - GRID_MS,
            side="sell" if net > 0 else "buy",
            size=float(abs(net)),
            price=10000.0,
            start_position=float(net),
        ),
        _fill(
            WALLET_A,
            "BTC",
            len(training) + 1,
            at_ms=WINDOW_START,
            side="buy",
            size=3.0,
            price=10000.0,
            start_position=0.0,
        ),
        _fill(
            WALLET_A,
            "BTC",
            len(training) + 2,
            at_ms=WINDOW_START + 2 * GRID_MS,
            side="sell",
            size=3.0,
            price=10100.0,
            start_position=3.0,
        ),
    ]
    quiet_mark = _fill(
        WALLET_A,
        "BTC",
        len(fills) + 1,
        at_ms=WINDOW_START + GRID_MS,
        side="buy",
        size=1.0,
        price=10050.0,
    )
    store = build_store(
        tmp_path,
        fills=fills,
        equity={WALLET_A: (Decimal(12000), 6)},
        bars_1m_rows=bars_1m_from_fills(fills + [quiet_mark]),
    )
    try:
        report = run_kernel_backtest(
            store,
            window=BacktestWindow(WINDOW_START, WINDOW_START + 3 * GRID_MS),
            seats=(SeatSpec("seat_1", Decimal(200), Decimal("1.0")),),
            rulebook=_rulebook(),
            markets=(MarketSpec("BTC"),),
            config=BacktestConfig(
                lookback_days=2,
                allow_undated=True,
                score_config=replace(PERMISSIVE, expectancy_mode="bar_markout"),
                quote_spread_bps=Decimal(0),
            ),
        )
    finally:
        store.close()

    assert report.decision_count == 3
    first, quiet, close = report.decisions
    assert len(first.ending_positions) == 1
    assert quiet.ending_positions == first.ending_positions
    assert not quiet.net_intents
    assert close.ending_positions == ()
    assert report.ending_positions == ()
    assert Decimal(first.ending_positions[0][2]) == Decimal("0.015")
    assert report.ending_equity == Decimal("10000.459825")
    assert len(report.ledger) == 3
    assert report.ledger[-1].equity == report.ending_equity
    assert report.ledger[1].net_fill_count == 0
    assert len(report.fills) == 2
    assert report.fills[0].net_delta_qty == Decimal("0.015")
    assert report.fills[1].net_delta_qty == Decimal("-0.015")
    assert report.ledger[-1].cash == (
        Decimal(10000)
        + sum((f.realized_pnl_usd for f in report.fills), Decimal(0))
        - sum((f.fee_usd + f.shortfall_usd for f in report.fills), Decimal(0))
    )


def test_netted_zero_opposing_owner_residuals_still_require_bar_mark(
    tmp_path: Path,
) -> None:
    """Two owners netting to 0 in account positions must still require a bar mark to drain."""
    from dataclasses import replace

    import pytest
    from research_fixtures import _fill, bars_1m_from_fills

    training_a = wallet_round_trips(
        WALLET_A,
        "BTC",
        start_ms=WINDOW_START - 2 * DAY_MS,
        bars=2 * 288 - 1,
        base_price=10000.0,
        size=0.078125,
    )
    net_a = sum(
        Decimal(str(row["size"])) * (1 if row["side"] == "buy" else -1) for row in training_a
    )
    training_b = wallet_round_trips(
        WALLET_B,
        "BTC",
        start_ms=WINDOW_START - 2 * DAY_MS,
        bars=2 * 288 - 1,
        base_price=10000.0,
        size=0.078125,
    )
    net_b = sum(
        Decimal(str(row["size"])) * (1 if row["side"] == "buy" else -1) for row in training_b
    )
    fills = training_a + training_b + [
        _fill(
            WALLET_A,
            "BTC",
            len(training_a),
            at_ms=WINDOW_START - GRID_MS,
            side="sell" if net_a > 0 else "buy",
            size=float(abs(net_a)),
            price=10000.0,
            start_position=float(net_a),
        ),
        _fill(
            WALLET_B,
            "BTC",
            len(training_b),
            at_ms=WINDOW_START - GRID_MS,
            side="sell" if net_b > 0 else "buy",
            size=float(abs(net_b)),
            price=10000.0,
            start_position=float(net_b),
        ),
        # At WINDOW_START both wallets open opposite +2 / -2 BTC (net 0); at +GRID_MS BTC has no bar mark.
        _fill(WALLET_A, "BTC", 9001, at_ms=WINDOW_START, side="buy", size=2.0, price=10000.0),
        _fill(WALLET_B, "BTC", 9002, at_ms=WINDOW_START, side="sell", size=2.0, price=10000.0),
    ]
    bars = [
        row
        for row in bars_1m_from_fills(fills)
        if row["event_ts_ms"] <= WINDOW_START
    ]
    store = build_store(
        tmp_path,
        fills=fills,
        equity={WALLET_A: (Decimal(12000), 6), WALLET_B: (Decimal(12000), 6)},
        bars_1m_rows=bars,
    )
    try:
        with pytest.raises(ValueError, match=f"MISSING_REQUIRED_MARK:{WINDOW_START + GRID_MS}:BTC"):
            run_kernel_backtest(
                store,
                window=BacktestWindow(WINDOW_START, WINDOW_START + 2 * GRID_MS),
                seats=(
                    SeatSpec("seat_1", Decimal(200), Decimal("1.0")),
                    SeatSpec("seat_2", Decimal(200), Decimal("1.0")),
                ),
                rulebook=_rulebook(),
                markets=(MarketSpec("BTC"),),
                config=BacktestConfig(
                    lookback_days=2,
                    allow_undated=True,
                    score_config=replace(PERMISSIVE, expectancy_mode="bar_markout"),
                    quote_spread_bps=Decimal(0),
                ),
            )
    finally:
        store.close()


def test_late_known_fill_does_not_leak_to_earlier_checkpoint(tmp_path: Path) -> None:
    """A fill received ten minutes late must appear at its known checkpoint, not earlier."""
    from dataclasses import replace

    import polars as pl
    from copytrade_pm.facts import manifest as mf
    from copytrade_pm.facts.schemas import schema_for
    from research_fixtures import _fill, bars_1m_from_fills

    training = wallet_round_trips(
        WALLET_A,
        "BTC",
        start_ms=WINDOW_START - 2 * DAY_MS,
        bars=2 * 288 - 1,
        base_price=10000.0,
        size=0.078125,
    )
    net = sum(
        Decimal(str(row["size"])) * (1 if row["side"] == "buy" else -1) for row in training
    )
    training.append(
        _fill(
            WALLET_A,
            "BTC",
            len(training),
            at_ms=WINDOW_START - GRID_MS,
            side="sell" if net > 0 else "buy",
            size=float(abs(net)),
            price=10000.0,
            start_position=float(net),
        )
    )
    store = build_store(
        tmp_path,
        fills=training,
        equity={WALLET_A: (Decimal(12000), 6)},
        bars_1m_rows=bars_1m_from_fills(
            training
            + [
                _fill(WALLET_A, "BTC", 9001, at_ms=WINDOW_START, side="buy", size=1.0, price=10000.0),
                _fill(
                    WALLET_A,
                    "BTC",
                    9002,
                    at_ms=WINDOW_START + GRID_MS,
                    side="buy",
                    size=1.0,
                    price=10000.0,
                ),
                _fill(
                    WALLET_A,
                    "BTC",
                    9003,
                    at_ms=WINDOW_START + 2 * GRID_MS,
                    side="buy",
                    size=1.0,
                    price=10000.0,
                ),
            ]
        ),
    )
    # Fill occurs at WINDOW_START (T0) but only becomes known 10 minutes later (T0 + 2*GRID_MS + 1000).
    late_known_at = WINDOW_START + 2 * GRID_MS + 1_000
    ts_dt = datetime.fromtimestamp(WINDOW_START / 1000, UTC)
    day = ts_dt.date().isoformat()
    existing = store.read_day("fills", day)
    offered = pl.DataFrame(
        [
            {
                "wallet": WALLET_A,
                "coin": "BTC",
                "dex": "hyperliquid",
                "dex_raw": "hyperliquid",
                "provider": "vike",
                "side": "buy",
                "size": Decimal(2),
                "price": Decimal(10000),
                "fee": Decimal(0),
                "fee_token": "USDC",
                "size_usd": Decimal(20000),
                "start_position": Decimal(0),
                "closed_pnl": Decimal(0),
                "trade_id": "99001",
                "order_id": "99001",
                "direction": "",
                "is_liquidation": False,
                "is_maker": False,
                "crossed": True,
                "exchange_ts": ts_dt,
                "event_ts_ms": WINDOW_START,
                "source_ts": ts_dt.isoformat(),
                "first_known_at_ms": late_known_at,
                "known_at_basis": "SYNTHETIC_LATE",
                "arrival_seq": 0,
                "source_id": "synthetic-late",
                "source_rank": 1,
                "batch_sha16": "",
            }
        ],
        schema=schema_for("fills"),
    )
    combined = offered if existing is None else pl.concat([existing, offered], how="vertical_relaxed")
    prepared = store.prepare_day("fills", day, combined)
    write = store.write_day(prepared)
    with store.connect() as con:
        mf.record_day(
            con,
            batch=mf.DayBatch(
                kind="fills",
                day=day,
                batch_sha16=write.batch_sha16,
                path=str(write.path),
                rows=write.rows,
                file_sha256=write.file_sha256,
                inputs_fingerprint=write.file_sha256[:16],
            ),
            source_ids=["synthetic-late"],
            bytes_count=write.bytes_count,
            inputs=[],
            day_sources=[{"source_id": "synthetic-late", "rows_written": write.rows}],
            first_known_min_ms=late_known_at,
            first_known_max_ms=late_known_at,
            closed_at=datetime.now(UTC).isoformat(),
        )
    try:
        report = run_kernel_backtest(
            store,
            window=BacktestWindow(WINDOW_START, WINDOW_START + 3 * GRID_MS),
            seats=(SeatSpec("seat_1", Decimal(200), Decimal("1.0")),),
            rulebook=_rulebook(),
            markets=(MarketSpec("BTC"),),
            config=BacktestConfig(
                lookback_days=2,
                allow_undated=True,
                score_config=replace(PERMISSIVE, expectancy_mode="bar_markout"),
                quote_spread_bps=Decimal(0),
            ),
        )
    finally:
        store.close()

    t0_step, t1_step, t2_step = report.decisions
    assert t0_step.ending_positions == ()
    assert t1_step.ending_positions == ()
    assert len(t2_step.ending_positions) == 1
    assert Decimal(t2_step.ending_positions[0][2]) == Decimal("0.01")


def test_nonzero_start_position_reconstructs_full_inventory(tmp_path: Path) -> None:
    """An existing start_position of 4 BTC plus a buy of 1 BTC must reconstruct 5 BTC."""
    from research_fixtures import _fill

    from trading_ops.research.backtest import _panel

    fill = {
        **_fill(WALLET_A, "BTC", 0, at_ms=WINDOW_START, side="buy", size=1.0, price=10000.0),
        "start_position": 4.0,
    }
    store = build_store(
        tmp_path,
        fills=[fill],
        equity={WALLET_A: (Decimal(12000), 3)},
    )
    try:
        panel = _panel(
            store,
            BacktestWindow(WINDOW_START, WINDOW_START + GRID_MS),
            lookback_days=1,
            tiers=None,
            allow_undated=True,
        )
    finally:
        store.close()

    assert panel.height == 1
    assert Decimal(str(panel["quantity"][0])) == Decimal(5)


def test_internally_crossed_owners_conserve_account_pnl_with_zero_phantom_spread() -> None:
    """Owner A +1 and Owner B -1 across bid/ask must net to 0 account PnL and 0 sum(owner_pnl)."""
    from trading_ops.research.backtest import _ReplayLedger

    ledger = _ReplayLedger()
    markets = {"BTC": MarketSpec("BTC")}
    config = BacktestConfig()
    fills = ledger.apply(
        targets={"BTC": "0"},
        pod_targets={"owner_a": {"BTC": "1"}, "owner_b": {"BTC": "-1"}},
        quotes={"BTC": {"bid": "99", "ask": "101"}},
        markets=markets,
        config=config,
    )
    assert fills == ()
    assert ledger.costs == Decimal(0)
    marks = {"BTC": 100.0}
    assert ledger.owner_pnl("owner_a", marks) == Decimal(0)
    assert ledger.owner_pnl("owner_b", marks) == Decimal(0)
    assert ledger.realized_pnl + ledger.unrealized(marks) == Decimal(0)


def test_drop_sub_min_opening_legs_executes_valid_market_without_no_trade(tmp_path: Path) -> None:
    """With drop_sub_min_opening_legs=True, $5 BTC opening dust drops so $50 ETH executes."""
    from dataclasses import replace

    from research_fixtures import _fill, bars_1m_from_fills

    training = wallet_round_trips(
        WALLET_A,
        "ETH",
        start_ms=WINDOW_START - 2 * DAY_MS,
        bars=2 * 288 - 1,
        base_price=100.0,
        size=0.625,
    )
    net = sum(
        Decimal(str(row["size"])) * (1 if row["side"] == "buy" else -1) for row in training
    )
    fills = training + [
        _fill(
            WALLET_A,
            "ETH",
            len(training),
            at_ms=WINDOW_START - GRID_MS,
            side="sell" if net > 0 else "buy",
            size=float(abs(net)),
            price=100.0,
            start_position=float(net),
        ),
        # At WINDOW_START with k=0.005: BTC 1 * 1000 * 0.005 = $5 (<$10 min); ETH 100 * 100 * 0.005 = $50.
        _fill(WALLET_A, "BTC", len(training) + 1, at_ms=WINDOW_START, side="buy", size=1.0, price=1000.0),
        _fill(WALLET_A, "ETH", len(training) + 2, at_ms=WINDOW_START, side="buy", size=100.0, price=100.0),
    ]
    store = build_store(
        tmp_path,
        fills=fills,
        equity={WALLET_A: (Decimal(12000), 6)},
        bars_1m_rows=bars_1m_from_fills(fills),
    )
    try:
        report = run_kernel_backtest(
            store,
            window=BacktestWindow(WINDOW_START, WINDOW_START + GRID_MS),
            seats=(SeatSpec("seat_1", Decimal(200), Decimal("1.0")),),
            rulebook=_rulebook(),
            markets=(MarketSpec("BTC"), MarketSpec("ETH")),
            config=BacktestConfig(
                lookback_days=2,
                allow_undated=True,
                score_config=replace(PERMISSIVE, expectancy_mode="bar_markout"),
                quote_spread_bps=Decimal(0),
                drop_sub_min_opening_legs=True,
            ),
        )
    finally:
        store.close()

    assert any(w.startswith(f"DROPPED_SUB_MIN_OPENING_LEG:{WINDOW_START}:BTC") for w in report.warnings)
    assert len(report.fills) == 1
    assert report.fills[0].market == "ETH"
    assert report.fills[0].net_delta_qty == Decimal("0.5")


def test_backtest_with_allocation_policy_uses_portfolio_allocator(tmp_path: Path) -> None:
    """BacktestConfig.allocation_policy wires allocate_fixed_k_portfolio into daily reconcile."""
    from dataclasses import replace

    from trading_ops.research import AllocationPolicy

    fills = wallet_round_trips(
        WALLET_A, "BTC", start_ms=START, bars=5 * 288, base_price=20000.0, size=0.1
    )
    fills += wallet_round_trips(
        WALLET_B, "ETH", start_ms=START, bars=5 * 288, base_price=2000.0, size=1.0
    )
    store = build_store(
        tmp_path,
        fills=fills,
        equity={WALLET_A: (Decimal(12000), 7), WALLET_B: (Decimal(9000), 7)},
    )
    try:
        report = run_kernel_backtest(
            store,
            window=BacktestWindow(WINDOW_START, WINDOW_START + DAY_MS),
            seats=SEATS,
            rulebook=_rulebook(),
            markets=(MarketSpec("BTC"), MarketSpec("ETH")),
            config=BacktestConfig(
                lookback_days=2,
                allow_undated=True,
                score_config=replace(PERMISSIVE, decouple_sizing_gates=True),
                allocation_policy=AllocationPolicy(
                    risk_unit_usd=Decimal(100),
                    max_fixed_k=Decimal("0.002"),
                    max_euler_cdar_share=Decimal("0.80"),
                    max_p95_participation=Decimal(1),
                    max_max_participation=Decimal(1),
                ),
            ),
        )
    finally:
        store.close()

    assert report.allocation_mode_share == {"ACTIVE": 1.0}
    assert report.challenge_outcome in {"SURVIVED", "PASSED"}
    assert report.fills


def test_out_of_order_late_fill_does_not_double_count_inventory(tmp_path: Path) -> None:
    """Late Fill 1 (start=4, +1) arriving after Fill 2 (start=5, +2) must stay 7 BTC, not 8."""
    import polars as pl
    from copytrade_pm.facts import FactStore
    from copytrade_pm.facts import manifest as mf
    from copytrade_pm.facts.schemas import schema_for

    from trading_ops.research.backtest import _panel

    fact_store = FactStore(tmp_path / "facts", create=True)
    try:
        t1 = WINDOW_START
        t2 = WINDOW_START + GRID_MS
        t3 = WINDOW_START + 2 * GRID_MS
        day = datetime.fromtimestamp(t1 / 1000, UTC).date().isoformat()
        rows = [
            {
                "wallet": WALLET_A,
                "coin": "BTC",
                "dex": "hyperliquid",
                "dex_raw": "hyperliquid",
                "provider": "vike",
                "side": "buy",
                "size": Decimal(1),
                "price": Decimal(10000),
                "fee": Decimal(0),
                "fee_token": "USDC",
                "size_usd": Decimal(10000),
                "start_position": Decimal(4),
                "closed_pnl": Decimal(0),
                "trade_id": "101",
                "order_id": "101",
                "direction": "",
                "is_liquidation": False,
                "is_maker": False,
                "crossed": True,
                "exchange_ts": datetime.fromtimestamp(t1 / 1000, UTC),
                "event_ts_ms": t1,
                "source_ts": datetime.fromtimestamp(t1 / 1000, UTC).isoformat(),
                "first_known_at_ms": t3 + 1_000,  # Fill 1 arrives late at t3
                "known_at_basis": "SYNTHETIC_LATE",
                "arrival_seq": 0,
                "source_id": "synthetic",
                "source_rank": 1,
                "batch_sha16": "",
            },
            {
                "wallet": WALLET_A,
                "coin": "BTC",
                "dex": "hyperliquid",
                "dex_raw": "hyperliquid",
                "provider": "vike",
                "side": "buy",
                "size": Decimal(2),
                "price": Decimal(10050),
                "fee": Decimal(0),
                "fee_token": "USDC",
                "size_usd": Decimal(20100),
                "start_position": Decimal(5),
                "closed_pnl": Decimal(0),
                "trade_id": "102",
                "order_id": "102",
                "direction": "",
                "is_liquidation": False,
                "is_maker": False,
                "crossed": True,
                "exchange_ts": datetime.fromtimestamp(t2 / 1000, UTC),
                "event_ts_ms": t2,
                "source_ts": datetime.fromtimestamp(t2 / 1000, UTC).isoformat(),
                "first_known_at_ms": t2 + 1_000,  # Fill 2 arrives normally at t2
                "known_at_basis": "SYNTHETIC_ONTIME",
                "arrival_seq": 1,
                "source_id": "synthetic",
                "source_rank": 1,
                "batch_sha16": "",
            },
        ]
        prepared = fact_store.prepare_day("fills", day, pl.DataFrame(rows, schema=schema_for("fills")))
        write = fact_store.write_day(prepared)
        with fact_store.connect() as con:
            mf.record_day(
                con,
                batch=mf.DayBatch(
                    kind="fills",
                    day=day,
                    batch_sha16=write.batch_sha16,
                    path=str(write.path),
                    rows=write.rows,
                    file_sha256=write.file_sha256,
                    inputs_fingerprint=write.file_sha256[:16],
                ),
                source_ids=["synthetic"],
                bytes_count=write.bytes_count,
                inputs=[],
                day_sources=[{"source_id": "synthetic", "rows_written": write.rows}],
                first_known_min_ms=t2 + 1_000,
                first_known_max_ms=t3 + 1_000,
                closed_at=datetime.now(UTC).isoformat(),
            )
        panel = _panel(
            fact_store,
            BacktestWindow(t1, t3 + GRID_MS),
            lookback_days=1,
            tiers=None,
            allow_undated=False,
        )
    finally:
        fact_store.close()

    by_bar = {int(row["bar_ms"]): Decimal(str(row["quantity"])) for row in panel.iter_rows(named=True)}
    assert by_bar[t2] == Decimal(7)
    assert by_bar[t3] == Decimal(7)


def test_terminal_bar_execution_cost_breach_updates_challenge_outcome(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A breach triggered by execution costs on the final bar must mark BREACHED."""
    from dataclasses import replace

    from research_fixtures import _fill, bars_1m_from_fills

    from trading_ops.research.backtest import _ReplayLedger

    training = wallet_round_trips(
        WALLET_A,
        "BTC",
        start_ms=WINDOW_START - 2 * DAY_MS,
        bars=2 * 288 - 1,
        base_price=10000.0,
        size=0.078125,
    )
    net = sum(
        Decimal(str(row["size"])) * (1 if row["side"] == "buy" else -1) for row in training
    )
    fills = training + [
        _fill(
            WALLET_A,
            "BTC",
            len(training),
            at_ms=WINDOW_START - GRID_MS,
            side="sell" if net > 0 else "buy",
            size=float(abs(net)),
            price=10000.0,
            start_position=float(net),
        ),
        _fill(
            WALLET_A,
            "BTC",
            len(training) + 1,
            at_ms=WINDOW_START,
            side="buy",
            size=1.0,
            price=10000.0,
        ),
    ]
    store = build_store(
        tmp_path,
        fills=fills,
        equity={WALLET_A: (Decimal(12000), 6)},
        bars_1m_rows=bars_1m_from_fills(fills),
    )
    tight_rulebook = Rulebook(
        account_binding="cft-10k-research",
        status="SYNTHETIC",
        initial="10000",
        target="1000",
        daily_loss="500",
        trailing_loss="1000",
        internal_daily_stop="10",
        daily_basis="MAX_BALANCE_EQUITY",
        trailing_basis="INTRADAY_EQUITY",
        timezone="UTC",
        valid_until_ms=10**15,
        evidence_sha="d" * 64,
    )
    orig_apply = _ReplayLedger.apply

    def _costly_apply(self: _ReplayLedger, **kwargs: Any) -> Any:
        bar_fills = orig_apply(self, **kwargs)
        # Simulate severe execution slippage on the terminal fill exceeding internal_daily_stop.
        if bar_fills:
            self.costs += Decimal(15)
        return bar_fills

    monkeypatch.setattr(_ReplayLedger, "apply", _costly_apply)
    try:
        report = run_kernel_backtest(
            store,
            window=BacktestWindow(WINDOW_START, WINDOW_START + GRID_MS),
            seats=(SeatSpec("seat_1", Decimal(200), Decimal("1.0")),),
            rulebook=tight_rulebook,
            markets=(MarketSpec("BTC"),),
            config=BacktestConfig(
                lookback_days=2,
                allow_undated=True,
                score_config=replace(PERMISSIVE, expectancy_mode="bar_markout"),
                quote_spread_bps=Decimal(0),
                scenario_shock=Decimal("-0.01"),
                pod_loss_budget=Decimal(500),
                minimum_headroom=Decimal(1),
            ),
        )
    finally:
        store.close()

    assert len(report.fills) == 1
    assert report.ending_equity < Decimal(9990)
    assert report.challenge_outcome == "BREACHED"
    assert report.challenge_pass_rate == 0.0
    assert "INTERNAL_DAILY_STOP" in report.challenge_reasons


def test_summarize_backtest_diagnostics_does_not_substitute_adjacent_minute_volume(
    tmp_path: Path,
) -> None:
    """Missing bars_1m at execution_ms must fail closed (None rates, unknown_vol_trades=1)."""
    from dataclasses import replace

    from research_fixtures import _fill, bars_1m_from_fills

    from trading_ops.research import summarize_backtest_diagnostics

    training = wallet_round_trips(
        WALLET_A,
        "BTC",
        start_ms=WINDOW_START - 2 * DAY_MS,
        bars=2 * 288 - 1,
        base_price=10000.0,
        size=0.078125,
    )
    net = sum(
        Decimal(str(row["size"])) * (1 if row["side"] == "buy" else -1) for row in training
    )
    fills = training + [
        _fill(
            WALLET_A,
            "BTC",
            len(training),
            at_ms=WINDOW_START - GRID_MS,
            side="sell" if net > 0 else "buy",
            size=float(abs(net)),
            price=10000.0,
            start_position=float(net),
        ),
        _fill(
            WALLET_A,
            "BTC",
            len(training) + 1,
            at_ms=WINDOW_START,
            side="buy",
            size=1.0,
            price=10000.0,
            start_position=0.0,
        ),
    ]
    # Backtest mark sees WINDOW_START + 60_000, while execution deadline WINDOW_START + GRID_MS is missing.
    shifted_bar_fill = _fill(
        WALLET_A, "BTC", 9999, at_ms=WINDOW_START + 60_000, side="buy", size=1.0, price=10000.0
    )
    bars = bars_1m_from_fills(training + [shifted_bar_fill])
    store = build_store(
        tmp_path / "missing_exec_vol",
        fills=fills,
        equity={WALLET_A: (Decimal(12000), 6)},
        bars_1m_rows=bars,
    )
    try:
        win = BacktestWindow(WINDOW_START, WINDOW_START + GRID_MS)
        report = run_kernel_backtest(
            store,
            window=win,
            seats=(SeatSpec("seat_1", Decimal(200), Decimal("1.0")),),
            rulebook=_rulebook(),
            markets=(MarketSpec("BTC"),),
            config=BacktestConfig(
                lookback_days=2,
                allow_undated=True,
                score_config=replace(PERMISSIVE, expectancy_mode="bar_markout"),
                quote_spread_bps=Decimal(0),
            ),
        )
        diag = summarize_backtest_diagnostics(report, facts=store, window=win)
    finally:
        store.close()

    assert len(report.fills) == 1
    assert diag["participation"]["n_trades"] == 0
    assert diag["participation"]["total_trades"] == 1
    assert diag["participation"]["unknown_vol_trades"] == 1
    assert diag["participation"]["capacity_evidence_complete"] is False
    assert diag["participation"]["execution_time_convention"] == "CHECKPOINT_DEADLINE_MS"
    assert diag["participation"]["median_1m"] is None
    assert diag["participation"]["p95_1m"] is None
    assert diag["participation"]["max_1m"] is None

    # Adding the exact execution-deadline bar (checkpoint_ms + GRID_MS) completes capacity evidence.
    exec_bar_fill = _fill(
        WALLET_A,
        "BTC",
        10000,
        at_ms=WINDOW_START + GRID_MS,
        side="buy",
        size=10.0,
        price=10000.0,
    )
    store_complete = build_store(
        tmp_path / "complete_exec_vol",
        fills=fills,
        equity={WALLET_A: (Decimal(12000), 6)},
        bars_1m_rows=bars_1m_from_fills(training + [shifted_bar_fill, exec_bar_fill]),
    )
    try:
        report_ok = run_kernel_backtest(
            store_complete,
            window=win,
            seats=(SeatSpec("seat_1", Decimal(200), Decimal("1.0")),),
            rulebook=_rulebook(),
            markets=(MarketSpec("BTC"),),
            config=BacktestConfig(
                lookback_days=2,
                allow_undated=True,
                score_config=replace(PERMISSIVE, expectancy_mode="bar_markout"),
                quote_spread_bps=Decimal(0),
            ),
        )
        diag_ok = summarize_backtest_diagnostics(report_ok, facts=store_complete, window=win)
    finally:
        store_complete.close()

    assert diag_ok["participation"]["n_trades"] == 1
    assert diag_ok["participation"]["unknown_vol_trades"] == 0
    assert diag_ok["participation"]["capacity_evidence_complete"] is True
    assert diag_ok["participation"]["max_1m"] is not None
    assert diag_ok["survival_no_breach"] is True


def test_causal_panel_is_prefix_invariant_when_future_fills_appended(tmp_path: Path) -> None:
    """Extending history with future fills must never alter earlier reconstructed positions."""
    from research_fixtures import _fill

    from trading_ops.research.backtest import _panel

    t0 = WINDOW_START
    t1 = WINDOW_START + GRID_MS
    t2 = WINDOW_START + 2 * GRID_MS
    prefix_fills = [
        _fill(WALLET_A, "BTC", 1, at_ms=t0, side="buy", size=1.0, price=10000.0, start_position=0.0),
        _fill(WALLET_A, "BTC", 2, at_ms=t1, side="buy", size=2.0, price=10000.0, start_position=0.0),
    ]
    extended_fills = prefix_fills + [
        _fill(WALLET_A, "BTC", 3, at_ms=t2, side="buy", size=1.0, price=10000.0, start_position=2.0),
    ]

    store_prefix = build_store(
        tmp_path / "prefix",
        fills=prefix_fills,
        equity={WALLET_A: (Decimal(12000), 3)},
    )
    store_extended = build_store(
        tmp_path / "extended",
        fills=extended_fills,
        equity={WALLET_A: (Decimal(12000), 3)},
    )
    try:
        panel_prefix = _panel(
            store_prefix,
            BacktestWindow(t0, t2),
            lookback_days=1,
            tiers=None,
            allow_undated=True,
        )
        panel_extended = _panel(
            store_extended,
            BacktestWindow(t0, t2 + GRID_MS),
            lookback_days=1,
            tiers=None,
            allow_undated=True,
        )
    finally:
        store_prefix.close()
        store_extended.close()

    by_bar_prefix = {
        int(row["bar_ms"]): Decimal(str(row["quantity"]))
        for row in panel_prefix.iter_rows(named=True)
    }
    by_bar_extended = {
        int(row["bar_ms"]): Decimal(str(row["quantity"]))
        for row in panel_extended.iter_rows(named=True)
    }
    assert by_bar_prefix == {t0: Decimal(1), t1: Decimal(2)}
    assert by_bar_extended[t0] == by_bar_prefix[t0]
    assert by_bar_extended[t1] == by_bar_prefix[t1]
    assert by_bar_extended[t2] == Decimal(3)


def test_seated_portfolio_constraint_breach_fails_closed_without_liquidating_incumbent(
    tmp_path: Path,
) -> None:
    """When incumbent exceeds DD budget on Day 1, fail closed while preserving incumbent without auto-liquidation."""
    from dataclasses import replace

    from research_fixtures import _fill, bars_1m_from_fills

    from trading_ops.research import AllocationPolicy

    training_a = wallet_round_trips(
        WALLET_A,
        "BTC",
        start_ms=START,
        bars=3 * 288 - 1,
        base_price=20000.0,
        size=0.05,
    )
    net_a = sum(
        Decimal(str(row["size"])) * (1 if row["side"] == "buy" else -1) for row in training_a
    )
    # Day 0 has a large drawdown swing on WALLET_A so on Day 1 its frozen lease.k exceeds budget ($50).
    day0_fills_a = [
        _fill(
            WALLET_A,
            "BTC",
            len(training_a),
            at_ms=WINDOW_START - GRID_MS,
            side="sell" if net_a > 0 else "buy",
            size=float(abs(net_a)),
            price=20000.0,
            start_position=float(net_a),
        ),
        _fill(
            WALLET_A,
            "BTC",
            len(training_a) + 1,
            at_ms=WINDOW_START,
            side="buy",
            size=100.0,
            price=20000.0,
            start_position=0.0,
        ),
        _fill(
            WALLET_A,
            "BTC",
            len(training_a) + 2,
            at_ms=WINDOW_START + GRID_MS,
            side="sell",
            size=50.0,
            price=18000.0,
            start_position=100.0,
        ),
        _fill(
            WALLET_A,
            "BTC",
            len(training_a) + 3,
            at_ms=WINDOW_START + 2 * GRID_MS,
            side="buy",
            size=50.0,
            price=19500.0,
            start_position=50.0,
        ),
    ]
    training_b = wallet_round_trips(
        WALLET_B,
        "ETH",
        start_ms=WINDOW_START,
        bars=289,
        base_price=2000.0,
        size=0.5,
    )
    btc_mark_tape = [
        _fill(
            WALLET_A,
            "BTC",
            20_000 + step,
            at_ms=WINDOW_START + step * GRID_MS,
            side="buy",
            size=0.01,
            price=19500.0,
        )
        for step in range(3, 289)
    ]
    fills = training_a + day0_fills_a + training_b
    store = build_store(
        tmp_path,
        fills=fills,
        equity={WALLET_A: (Decimal(12000), 6), WALLET_B: (Decimal(12000), 6)},
        bars_1m_rows=bars_1m_from_fills(fills + btc_mark_tape),
    )
    try:
        report = run_kernel_backtest(
            store,
            window=BacktestWindow(WINDOW_START, WINDOW_START + DAY_MS + GRID_MS),
            seats=(
                SeatSpec("seat_1", Decimal(200), Decimal("1.0")),
                SeatSpec("seat_2", Decimal(200), Decimal("1.0")),
            ),
            rulebook=_rulebook(),
            markets=(MarketSpec("BTC"), MarketSpec("ETH")),
            config=BacktestConfig(
                lookback_days=2,
                allow_undated=True,
                score_config=replace(
                    PERMISSIVE,
                    expectancy_mode="bar_markout",
                    decouple_sizing_gates=True,
                    max_p95_participation_1m=Decimal(1),
                    max_max_participation_1m=Decimal(1),
                    max_euler_cdar_share=Decimal(1),
                ),
                quote_spread_bps=Decimal(0),
                allocation_policy=AllocationPolicy(
                    risk_unit_usd=Decimal(50),
                    portfolio_dd_budget_usd=Decimal(50),
                    max_fixed_k=Decimal("0.005"),
                    max_euler_cdar_share=Decimal(1),
                    max_p95_participation=Decimal(1),
                    max_max_participation=Decimal(1),
                ),
            ),
        )
    finally:
        store.close()

    assert report.portfolio_constraints_valid is False
    payload = report.data()
    assert payload["portfolio_constraints_valid"] is False
    assert payload["fills"]
    assert all(f["execution_ms"] == f["checkpoint_ms"] + GRID_MS for f in payload["fills"])
    assert any(w.startswith("SEATED_PORTFOLIO_UNQUALIFIED:") for w in report.warnings)
    assert any(w.startswith("SEATED_PORTFOLIO_DD_ABOVE_BUDGET:") for w in report.warnings)
    # Incumbent WALLET_A preserves its position without being auto-liquidated on Day 1.
    assert any(market == "BTC" and Decimal(qty) > 0 for _, market, qty in report.ending_positions)
    # WALLET_B is blocked from promotion under failed portfolio qualification.
    assert not any(market == "ETH" for _, market, _ in report.ending_positions)


def test_mixed_authoritative_and_delayed_null_start_fills_preserve_inventory(
    tmp_path: Path,
) -> None:
    """Delayed unknown-start fills arriving after an unanchored fill must not be dropped."""
    import polars as pl
    from copytrade_pm.facts import FactStore
    from copytrade_pm.facts import manifest as mf
    from copytrade_pm.facts.schemas import schema_for

    from trading_ops.research.backtest import _panel

    fact_store = FactStore(tmp_path / "facts", create=True)
    try:
        t0 = WINDOW_START
        t1 = WINDOW_START + GRID_MS
        t2 = WINDOW_START + 2 * GRID_MS
        t3 = WINDOW_START + 3 * GRID_MS
        day = datetime.fromtimestamp(t0 / 1000, UTC).date().isoformat()
        rows = [
            {
                "wallet": WALLET_A,
                "coin": "BTC",
                "dex": "hyperliquid",
                "dex_raw": "hyperliquid",
                "provider": "vike",
                "side": "buy",
                "size": Decimal(1),
                "price": Decimal(10000),
                "fee": Decimal(0),
                "fee_token": "USDC",
                "size_usd": Decimal(10000),
                "start_position": Decimal(0),
                "closed_pnl": Decimal(0),
                "trade_id": "201",
                "order_id": "201",
                "direction": "",
                "is_liquidation": False,
                "is_maker": False,
                "crossed": True,
                "exchange_ts": datetime.fromtimestamp(t0 / 1000, UTC),
                "event_ts_ms": t0,
                "source_ts": datetime.fromtimestamp(t0 / 1000, UTC).isoformat(),
                "first_known_at_ms": t0 + 1_000,
                "known_at_basis": "SYNTHETIC_AUTH",
                "arrival_seq": 0,
                "source_id": "synthetic",
                "source_rank": 1,
                "batch_sha16": "",
            },
            {
                "wallet": WALLET_A,
                "coin": "BTC",
                "dex": "hyperliquid",
                "dex_raw": "hyperliquid",
                "provider": "vike",
                "side": "buy",
                "size": Decimal(2),
                "price": Decimal(10050),
                "fee": Decimal(0),
                "fee_token": "USDC",
                "size_usd": Decimal(20100),
                "start_position": None,
                "closed_pnl": Decimal(0),
                "trade_id": "202",
                "order_id": "202",
                "direction": "",
                "is_liquidation": False,
                "is_maker": False,
                "crossed": True,
                "exchange_ts": datetime.fromtimestamp(t2 / 1000, UTC),
                "event_ts_ms": t2,
                "source_ts": datetime.fromtimestamp(t2 / 1000, UTC).isoformat(),
                "first_known_at_ms": t2 + 1_000,
                "known_at_basis": "SYNTHETIC_MISSING_START",
                "arrival_seq": 1,
                "source_id": "synthetic",
                "source_rank": 1,
                "batch_sha16": "",
            },
            {
                "wallet": WALLET_A,
                "coin": "BTC",
                "dex": "hyperliquid",
                "dex_raw": "hyperliquid",
                "provider": "vike",
                "side": "buy",
                "size": Decimal(3),
                "price": Decimal(10100),
                "fee": Decimal(0),
                "fee_token": "USDC",
                "size_usd": Decimal(30300),
                "start_position": None,
                "closed_pnl": Decimal(0),
                "trade_id": "203",
                "order_id": "203",
                "direction": "",
                "is_liquidation": False,
                "is_maker": False,
                "crossed": True,
                "exchange_ts": datetime.fromtimestamp(t1 / 1000, UTC),
                "event_ts_ms": t1,
                "source_ts": datetime.fromtimestamp(t1 / 1000, UTC).isoformat(),
                "first_known_at_ms": t3 + 1_000,  # Delayed past t2 fill
                "known_at_basis": "SYNTHETIC_DELAYED_MISSING_START",
                "arrival_seq": 2,
                "source_id": "synthetic",
                "source_rank": 1,
                "batch_sha16": "",
            },
        ]
        prepared = fact_store.prepare_day("fills", day, pl.DataFrame(rows, schema=schema_for("fills")))
        write = fact_store.write_day(prepared)
        with fact_store.connect() as con:
            mf.record_day(
                con,
                batch=mf.DayBatch(
                    kind="fills",
                    day=day,
                    batch_sha16=write.batch_sha16,
                    path=str(write.path),
                    rows=write.rows,
                    file_sha256=write.file_sha256,
                    inputs_fingerprint=write.file_sha256[:16],
                ),
                source_ids=["synthetic"],
                bytes_count=write.bytes_count,
                inputs=[],
                day_sources=[{"source_id": "synthetic", "rows_written": write.rows}],
                first_known_min_ms=t0 + 1_000,
                first_known_max_ms=t3 + 1_000,
                closed_at=datetime.now(UTC).isoformat(),
            )
        panel = _panel(
            fact_store,
            BacktestWindow(t0, t3 + GRID_MS),
            lookback_days=1,
            tiers=None,
            allow_undated=False,
        )
    finally:
        fact_store.close()

    by_bar = {int(row["bar_ms"]): Decimal(str(row["quantity"])) for row in panel.iter_rows(named=True)}
    assert by_bar == {t0: Decimal(1), t2: Decimal(3), t3: Decimal(6)}


def test_backtest_report_serialization_preserves_constraints_and_execution_ms() -> None:
    """Serialized BacktestReport.data() must expose portfolio_constraints_valid and fill execution_ms."""
    from copytrade_kernel import KernelState, OwnerBook

    from trading_ops.research.backtest import BacktestReport, LedgerFill

    fill = LedgerFill(
        checkpoint_ms=WINDOW_START,
        market="BTC",
        net_delta_qty=Decimal("0.25"),
        executable_price=Decimal(10000),
        fee_usd=Decimal("1.25"),
        shortfall_usd=Decimal("7.5"),
        realized_pnl_usd=Decimal(0),
        reduce_only=False,
    )
    base_kwargs: dict[str, Any] = {
        "window": BacktestWindow(WINDOW_START, WINDOW_START + GRID_MS),
        "account_id": "cft-10k-research",
        "decisions": (),
        "ending_state": KernelState(owner_books=(OwnerBook("owner_1", (), ()),)),
        "ending_equity": Decimal(10000),
        "allocation_mode_share": {"ACTIVE": 1.0},
        "challenge_pass_rate": 1.0,
        "challenge_reasons": (),
        "coverage": {"kind": "fills"},
        "reconcile_plans": (),
        "scorecards_recorded": 1,
        "warnings": (),
        "source_sha": "a" * 64,
        "fills": (fill,),
    }
    valid_payload = BacktestReport(**base_kwargs, portfolio_constraints_valid=True).data()
    invalid_payload = BacktestReport(**base_kwargs, portfolio_constraints_valid=False).data()

    assert valid_payload["portfolio_constraints_valid"] is True
    assert invalid_payload["portfolio_constraints_valid"] is False
    assert invalid_payload["fills"][0]["checkpoint_ms"] == WINDOW_START
    assert invalid_payload["fills"][0]["execution_ms"] == WINDOW_START + GRID_MS


def test_reconstruct_causal_panel_handles_null_fill_price(tmp_path: Path) -> None:
    """Real monitored-plus-six Reservoir fills can carry price=None while inventory is valid."""
    import polars as pl
    from copytrade_pm.facts import FactStore
    from copytrade_pm.facts import manifest as mf
    from copytrade_pm.facts.schemas import schema_for

    from trading_ops.research.backtest import _panel

    fact_store = FactStore(tmp_path / "facts", create=True)
    try:
        t0 = WINDOW_START
        day = datetime.fromtimestamp(t0 / 1000, UTC).date().isoformat()
        rows = [
            {
                "wallet": WALLET_A,
                "coin": "ETH",
                "dex": "hyperliquid",
                "dex_raw": "hyperliquid",
                "provider": "vike",
                "side": "buy",
                "size": Decimal(3),
                "price": None,
                "fee": Decimal(0),
                "fee_token": "USDC",
                "size_usd": Decimal(0),
                "start_position": Decimal(2),
                "closed_pnl": Decimal(0),
                "trade_id": "301",
                "order_id": "301",
                "direction": "",
                "is_liquidation": False,
                "is_maker": False,
                "crossed": True,
                "exchange_ts": datetime.fromtimestamp(t0 / 1000, UTC),
                "event_ts_ms": t0,
                "source_ts": datetime.fromtimestamp(t0 / 1000, UTC).isoformat(),
                "first_known_at_ms": t0 + 1_000,
                "known_at_basis": "SYNTHETIC_NULL_PRICE",
                "arrival_seq": 0,
                "source_id": "synthetic",
                "source_rank": 1,
                "batch_sha16": "",
            }
        ]
        prepared = fact_store.prepare_day("fills", day, pl.DataFrame(rows, schema=schema_for("fills")))
        write = fact_store.write_day(prepared)
        with fact_store.connect() as con:
            mf.record_day(
                con,
                batch=mf.DayBatch(
                    kind="fills",
                    day=day,
                    batch_sha16=write.batch_sha16,
                    path=str(write.path),
                    rows=write.rows,
                    file_sha256=write.file_sha256,
                    inputs_fingerprint=write.file_sha256[:16],
                ),
                source_ids=["synthetic"],
                bytes_count=write.bytes_count,
                inputs=[],
                day_sources=[{"source_id": "synthetic", "rows_written": write.rows}],
                first_known_min_ms=t0 + 1_000,
                first_known_max_ms=t0 + 1_000,
                closed_at=datetime.now(UTC).isoformat(),
            )
        panel = _panel(
            fact_store,
            BacktestWindow(t0, t0 + GRID_MS),
            lookback_days=1,
            tiers=None,
            allow_undated=False,
        )
    finally:
        fact_store.close()

    assert panel.height == 1
    assert Decimal(str(panel["quantity"][0])) == Decimal(5)
    assert panel["price"][0] is None
