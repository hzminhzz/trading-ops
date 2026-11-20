"""Stage-1 ``score_universe``: vectorized, PIT, and roster-ready."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from copytrade_kernel import KnownAtTs
from copytrade_pm.roster import GateFailure, RosterStore
from research_fixtures import (
    DAY_MS,
    GRID_MS,
    START,
    WALLET_A,
    WALLET_B,
    bars_1m_from_fills,
    build_store,
    wallet_round_trips,
)

from trading_ops.research import (
    ScoreConfig,
    UniverseScorecard,
    bench_from_scorecard,
    score_universe,
)

WALLET_C = "0x" + "c" * 40

BARS_3D = 3 * 288
PERMISSIVE = ScoreConfig(
    min_baseline_order_usd=Decimal(1),
    max_euler_cdar_share=Decimal(1),
    max_p95_participation_1m=Decimal(1),
    max_max_participation_1m=Decimal(1),
)


def _store(
    tmp_path: Path,
    *,
    equity: dict[str, tuple[Decimal, int]] | None = None,
    bars_1m_rows: list[dict] | None = None,
):
    fills = wallet_round_trips(
        WALLET_A, "BTC", start_ms=START, bars=BARS_3D, base_price=20000.0, size=0.1
    )
    # Same unit notional on both wallets, so neither dominates the CDaR share.
    fills += wallet_round_trips(
        WALLET_B, "ETH", start_ms=START, bars=BARS_3D, base_price=2000.0, size=1.0
    )
    return build_store(
        tmp_path,
        fills=fills,
        equity=equity
        if equity is not None
        else {
            WALLET_A: (Decimal(12000), 4),
            WALLET_B: (Decimal(9000), 4),
        },
        bars_1m_rows=bars_1m_rows,
    )


def test_score_universe_measures_every_wallet_with_evidence(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        card = score_universe(
            store,
            as_of=KnownAtTs(START + 3 * DAY_MS),
            lookback_days=3,
            config=PERMISSIVE,
            allow_undated=True,
        )
    finally:
        store.close()

    assert isinstance(card, UniverseScorecard)
    assert [entry.wallet for entry in card.cards] == [WALLET_A, WALLET_B]
    assert card.excluded == ()
    assert card.basis["pit"] == "RETROSPECTIVE_UNDATED"
    assert card.basis["participation"] == "FACT_STORE_BARS_1M_VENUE_VOLUME"
    assert card.basis["cdar_share"] == "EULER_MARGINAL_CDAR_COHORT"
    assert card.basis["markout"] == "FACT_STORE_BARS_1M_CLOSE_5M"
    assert card.slice_rows == 2 * BARS_3D
    # Real measured quantities, not placeholders.
    for row in card.audit:
        # The final bar has no 5m-later trade price, so markout is measured on 863 rows.
        assert row.markout_fills == BARS_3D - 1
        assert row.markout_5m_bps != Decimal(0)
        assert row.wallet_max_dd_usd > 0
        assert Decimal(0) < row.suggested_k <= Decimal(10)
    shares = [row.cdar_share for row in card.audit]
    # Quantised to 12 dp at the Decimal boundary, so the shares sum to 1 within 1e-9.
    assert abs(sum(shares) - Decimal(1)) < Decimal("1e-9")


def test_strict_pit_never_sees_undated_archive_rows(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        card = score_universe(
            store, as_of=KnownAtTs(START + 3 * DAY_MS), lookback_days=3, config=PERMISSIVE
        )
    finally:
        store.close()

    # The Reservoir backfill carries no capture clock: strict PIT is empty.
    assert card.cards == ()
    assert card.basis["pit"] == "STRICT_KNOWN_AT"
    assert "NO_FILL_ROWS" in card.notes


def test_missing_account_value_excludes_a_wallet_instead_of_zeroing_it(tmp_path: Path) -> None:
    store = _store(tmp_path, equity={WALLET_A: (Decimal(12000), 4)})
    try:
        card = score_universe(
            store,
            as_of=KnownAtTs(START + 3 * DAY_MS),
            lookback_days=3,
            config=PERMISSIVE,
            allow_undated=True,
        )
    finally:
        store.close()

    assert [entry.wallet for entry in card.cards] == [WALLET_A]
    assert [(row.wallet, row.reason) for row in card.excluded] == [
        (WALLET_B, "ACCOUNT_VALUE_UNKNOWN")
    ]


def test_thresholds_emit_typed_gates_from_measured_values(tmp_path: Path) -> None:
    fills = wallet_round_trips(
        WALLET_A, "BTC", start_ms=START, bars=BARS_3D, base_price=20000.0, size=0.1
    )
    fills += wallet_round_trips(
        WALLET_B, "ETH", start_ms=START, bars=BARS_3D, base_price=2000.0, size=1.0
    )
    thin_bars = bars_1m_from_fills(fills, quote_volume_usd=Decimal(10000))
    store = build_store(
        tmp_path,
        fills=fills,
        equity={WALLET_A: (Decimal(500), 4), WALLET_B: (Decimal(9000), 4)},
        bars_1m_rows=thin_bars,
    )
    try:
        card = score_universe(
            store,
            as_of=KnownAtTs(START + 3 * DAY_MS),
            lookback_days=3,
            config=ScoreConfig(
                min_baseline_order_usd=Decimal(1),
                max_euler_cdar_share=Decimal("0.05"),
                max_p95_participation_1m=Decimal("0.10"),
                max_max_participation_1m=Decimal("0.10"),
            ),
            allow_undated=True,
        )
    finally:
        store.close()

    by_wallet = card.by_wallet
    low_value = {gate.value for gate in by_wallet[WALLET_A].failures}
    assert "LOW_ACCOUNT_VALUE_LT_1K" in low_value
    assert "P95_PARTICIPATION_ABOVE_LIMIT" in low_value
    # The larger wallet is still gated by the portfolio concentration cap.
    assert "MARGINAL_CDAR_ABOVE_LIMIT" in {gate.value for gate in by_wallet[WALLET_B].failures}
    assert by_wallet[WALLET_A].eligible is False


def test_dust_orders_are_excluded_before_any_scorecard_exists(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        card = score_universe(
            store,
            as_of=KnownAtTs(START + 3 * DAY_MS),
            lookback_days=3,
            config=ScoreConfig(min_baseline_order_usd=Decimal(100000)),
            allow_undated=True,
        )
    finally:
        store.close()

    assert card.cards == ()
    assert {row.reason for row in card.excluded} == {"BASELINE_ORDER_BELOW_FLOOR"}


def test_scorecard_records_through_the_roster_idempotently(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        card = score_universe(
            store,
            as_of=KnownAtTs(START + 3 * DAY_MS),
            lookback_days=3,
            config=PERMISSIVE,
            allow_undated=True,
        )
    finally:
        store.close()

    roster = RosterStore(tmp_path / "roster.sqlite", create=True)
    try:
        first = card.record(roster)
        second = card.record(roster)
        assert (
            first == second == tuple(card.by_wallet[wallet].sha for wallet in (WALLET_A, WALLET_B))
        )
        latest = roster.latest_scorecard(as_of=KnownAtTs(START + 3 * DAY_MS))
        assert set(latest) == {WALLET_A, WALLET_B}
        assert latest[WALLET_A].sha == card.by_wallet[WALLET_A].sha
    finally:
        roster.close()


def test_unknown_side_rows_are_dropped_not_guessed(tmp_path: Path) -> None:
    fills = wallet_round_trips(
        WALLET_A, "BTC", start_ms=START, bars=288, base_price=20000.0, size=0.1
    )
    for row in fills[:10]:
        row["side"] = "liquidate"
    store = build_store(tmp_path, fills=fills, equity={WALLET_A: (Decimal(12000), 4)})
    try:
        card = score_universe(
            store,
            as_of=KnownAtTs(START + DAY_MS),
            lookback_days=2,
            config=PERMISSIVE,
            allow_undated=True,
        )
    finally:
        store.close()

    assert card.slice_rows == 288
    assert any(note.startswith("UNSCORED_FILLS_UNKNOWN_SIDE_OR_PRICE:10") for note in card.notes)


def test_missing_or_zero_venue_volume_excludes_wallet(tmp_path: Path) -> None:
    fills_a = wallet_round_trips(
        WALLET_A, "BTC", start_ms=START, bars=BARS_3D, base_price=20000.0, size=0.1
    )
    fills_b = wallet_round_trips(
        WALLET_B, "ETH", start_ms=START, bars=BARS_3D, base_price=2000.0, size=1.0
    )
    fills_c = wallet_round_trips(
        WALLET_C, "SOL", start_ms=START, bars=BARS_3D, base_price=200.0, size=10.0
    )
    # BTC has valid bars; ETH is missing bars entirely; SOL has one zero-volume bar.
    bars = bars_1m_from_fills(fills_a)
    sol_bars = bars_1m_from_fills(fills_c)
    sol_bars[5]["quote_volume_usd"] = Decimal(0)
    sol_bars[5]["volume"] = Decimal(0)
    bars.extend(sol_bars)

    store = build_store(
        tmp_path,
        fills=fills_a + fills_b + fills_c,
        equity={
            WALLET_A: (Decimal(12000), 4),
            WALLET_B: (Decimal(9000), 4),
            WALLET_C: (Decimal(9000), 4),
        },
        bars_1m_rows=bars,
    )
    try:
        card = score_universe(
            store,
            as_of=KnownAtTs(START + 3 * DAY_MS),
            lookback_days=3,
            config=PERMISSIVE,
            allow_undated=True,
        )
    finally:
        store.close()

    assert [entry.wallet for entry in card.cards] == [WALLET_A]
    excluded = {row.wallet: (row.reason, row.observed) for row in card.excluded}
    assert excluded[WALLET_B] == ("VENUE_VOLUME_UNKNOWN", BARS_3D)
    assert excluded[WALLET_C] == ("VENUE_VOLUME_UNKNOWN", 1)


def test_strict_pit_hides_future_bars_1m(tmp_path: Path) -> None:
    fills = wallet_round_trips(
        WALLET_A, "BTC", start_ms=START, bars=BARS_3D, base_price=20000.0, size=0.1
    )
    fills += wallet_round_trips(
        WALLET_B, "ETH", start_ms=START, bars=BARS_3D, base_price=2000.0, size=1.0
    )
    as_of_ms = START + 3 * DAY_MS
    bars = bars_1m_from_fills(fills)
    for row in bars:
        if row["coin"] == "ETH":
            row["first_known_at_ms"] = as_of_ms + 60_000

    store = build_store(
        tmp_path,
        fills=fills,
        equity={WALLET_A: (Decimal(12000), 4), WALLET_B: (Decimal(9000), 4)},
        bars_1m_rows=bars,
    )
    try:
        before = score_universe(
            store,
            as_of=KnownAtTs(as_of_ms),
            lookback_days=3,
            config=PERMISSIVE,
            allow_undated=True,
        )
        after = score_universe(
            store,
            as_of=KnownAtTs(as_of_ms + 120_000),
            lookback_days=3,
            config=PERMISSIVE,
            allow_undated=True,
        )
    finally:
        store.close()

    assert [entry.wallet for entry in before.cards] == [WALLET_A]
    assert [(row.wallet, row.reason) for row in before.excluded] == [
        (WALLET_B, "VENUE_VOLUME_UNKNOWN")
    ]
    assert [entry.wallet for entry in after.cards] == [WALLET_A, WALLET_B]
    assert after.excluded == ()


def test_true_venue_participation_breach_gates_before_clamp(tmp_path: Path) -> None:
    fills = wallet_round_trips(
        WALLET_A, "BTC", start_ms=START, bars=BARS_3D, base_price=20000.0, size=0.1
    )
    # Wallet notional is ~$2,000/bar against $1,000 venue quote volume -> raw share > 1.0.
    bars = bars_1m_from_fills(fills, quote_volume_usd=Decimal(1000))
    store = build_store(
        tmp_path,
        fills=fills,
        equity={WALLET_A: (Decimal(12000), 4)},
        bars_1m_rows=bars,
    )
    try:
        card = score_universe(
            store,
            as_of=KnownAtTs(START + 3 * DAY_MS),
            lookback_days=3,
            config=PERMISSIVE,
            allow_undated=True,
        )
    finally:
        store.close()

    wallet_card = card.by_wallet[WALLET_A]
    assert GateFailure.PARTICIPATION_LIMIT in wallet_card.failures
    assert wallet_card.p95_participation_1m == Decimal(1)
    audit_row = next(row for row in card.audit if row.wallet == WALLET_A)
    assert audit_row.p95_participation_1m > Decimal(1)
    assert audit_row.max_participation_1m > Decimal(1)


def test_euler_marginal_cdar_distinguishes_correlated_vs_hedging_and_bench(
    tmp_path: Path,
) -> None:
    # Construct a tape with a clear mid-window drawdown so correlated vs hedging
    # wallets have opposite Euler marginal CDaR contributions.
    bars_count = 60
    prices = [
        20000.0 * (1.0 + 0.002 * i)
        if i < 20
        else 20000.0 * (1.04 - 0.004 * (i - 20))
        if i < 35
        else 20000.0 * (0.98 + 0.003 * (i - 35))
        for i in range(bars_count)
    ]

    def _tape(wallet: str, *, mode: str, size: float) -> list[dict]:
        rows = []
        for idx, px in enumerate(prices):
            at_ms = START + idx * GRID_MS
            qty = size
            if mode == "long":
                side = "buy"
            elif mode == "hedge":
                # Small net-short hedge during the drawdown (bars 18..34), long on rising legs.
                if idx == 18:
                    side, qty = "sell", size * 20.0
                elif 19 <= idx < 35:
                    side = "sell"
                elif idx == 35:
                    side, qty = "buy", size * 25.0
                else:
                    side = "buy"
            else:
                raise ValueError(mode)
            rows.append(
                {
                    "wallet": wallet,
                    "exchange_ts": __import__("datetime").datetime.fromtimestamp(
                        at_ms / 1000, __import__("datetime").UTC
                    ),
                    "coin": "BTC",
                    "dex": "hyperliquid",
                    "side": side,
                    "size": qty,
                    "price": px,
                    "start_position": 0.0,
                    "trade_id": idx + 1,
                }
            )
        return rows

    fills = (
        _tape(WALLET_A, mode="long", size=0.1)
        + _tape(WALLET_B, mode="hedge", size=0.01)
        + _tape(WALLET_C, mode="long", size=0.15)
    )
    store = build_store(
        tmp_path,
        fills=fills,
        equity={
            WALLET_A: (Decimal(12000), 2),
            WALLET_B: (Decimal(12000), 2),
            WALLET_C: (Decimal(500), 2),  # ineligible: below $1k floor
        },
        bars_1m_rows=bars_1m_from_fills(fills),
    )
    try:
        card = score_universe(
            store,
            as_of=KnownAtTs(START + DAY_MS),
            lookback_days=1,
            config=ScoreConfig(
                min_baseline_order_usd=Decimal(1),
                max_euler_cdar_share=Decimal(1),
                max_p95_participation_1m=Decimal(1),
                max_max_participation_1m=Decimal(1),
                max_fixed_k=Decimal("0.05"),
                min_markout_fills=1,
            ),
            allow_undated=True,
        )
    finally:
        store.close()

    audit_by_wallet = {row.wallet: row for row in card.audit}
    assert abs(sum(row.cdar_share for row in card.audit) - Decimal(1)) < Decimal("1e-9")
    assert audit_by_wallet[WALLET_A].cdar_share > Decimal(0)
    assert audit_by_wallet[WALLET_C].cdar_share > Decimal(0)
    assert audit_by_wallet[WALLET_B].cdar_share < Decimal(0)
    assert card.by_wallet[WALLET_B].euler_cdar_share == Decimal(0)

    # Bench conditioning: WALLET_C is ineligible (low account value), so bench_from_scorecard
    # recomputes Euler marginal CDaR over the eligible subset only.
    bench = bench_from_scorecard(card)
    bench_wallets = {item.wallet: item for item in bench}
    assert WALLET_C not in bench_wallets
    assert WALLET_A in bench_wallets
    assert bench_wallets[WALLET_A].marginal_cdar_share > card.by_wallet[WALLET_A].euler_cdar_share
