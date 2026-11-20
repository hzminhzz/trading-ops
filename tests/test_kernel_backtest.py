"""Stage-2 ``run_kernel_backtest``: daily reconcile, 5m kernel steps."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

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
