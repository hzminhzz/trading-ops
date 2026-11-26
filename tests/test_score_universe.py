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
    assert card.basis["participation"] == "FACT_STORE_BARS_5M_FOLLOWER_K_SCALED"
    assert card.basis["cdar_share"] == "EULER_MARGINAL_CDAR_COHORT"
    assert card.basis["markout"] == "FOLLOWER_BAR_TRAJECTORY_EDGE"
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
    # BTC has valid bars; ETH is missing bars entirely; SOL has one zero-volume bar (0.116% <= 2%).
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
        strict_card = score_universe(
            store,
            as_of=KnownAtTs(START + 3 * DAY_MS),
            lookback_days=3,
            config=ScoreConfig(
                min_baseline_order_usd=Decimal(1),
                max_euler_cdar_share=Decimal(1),
                max_p95_participation_1m=Decimal(1),
                max_max_participation_1m=Decimal(1),
                max_missing_volume_frac=Decimal(0),
            ),
            allow_undated=True,
        )
    finally:
        store.close()

    assert [entry.wallet for entry in card.cards] == [WALLET_A, WALLET_C]
    excluded = {row.wallet: (row.reason, row.observed) for row in card.excluded}
    assert excluded[WALLET_B] == ("VENUE_VOLUME_UNKNOWN", BARS_3D)
    assert WALLET_C not in excluded
    assert f"TOLERATED_MISSING_VOLUME_BARS:{WALLET_C}:1/{BARS_3D}" in card.notes

    assert [entry.wallet for entry in strict_card.cards] == [WALLET_A]
    strict_excluded = {row.wallet: (row.reason, row.observed) for row in strict_card.excluded}
    assert strict_excluded[WALLET_B] == ("VENUE_VOLUME_UNKNOWN", BARS_3D)
    assert strict_excluded[WALLET_C] == ("VENUE_VOLUME_UNKNOWN", 1)


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


def _raw_fill(
    wallet: str,
    coin: str,
    *,
    at_ms: int,
    side: str,
    size: float,
    price: float,
    trade_id: int,
) -> dict:
    from datetime import UTC, datetime

    return {
        "wallet": wallet,
        "exchange_ts": datetime.fromtimestamp(at_ms / 1000, UTC),
        "coin": coin,
        "dex": "hyperliquid",
        "side": side,
        "size": size,
        "price": price,
        "start_position": 0.0,
        "trade_id": trade_id,
    }


def test_non_market_symbols_are_stripped_before_scoring(tmp_path: Path) -> None:
    fills_a = wallet_round_trips(
        WALLET_A, "BTC", start_ms=START, bars=BARS_3D, base_price=20000.0, size=0.1
    )
    # Stray HIP-3 stock ticker fills that MarketSpec rejects, plus WALLET_C trading only XYZ:AAPL.
    fills_a.append(
        _raw_fill(WALLET_A, "XYZ:AAPL", at_ms=START + 1000, side="buy", size=1.0, price=200.0, trade_id=9901)
    )
    fills_c = [
        _raw_fill(WALLET_C, "XYZ:AAPL", at_ms=START + i * GRID_MS, side="buy", size=1.0, price=200.0, trade_id=10_000 + i)
        for i in range(5)
    ]
    bars = bars_1m_from_fills([row for row in fills_a if row["coin"] == "BTC"])
    store = build_store(
        tmp_path,
        fills=fills_a + fills_c,
        equity={WALLET_A: (Decimal(12000), 4), WALLET_C: (Decimal(12000), 4)},
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
    assert any(note == "UNSCORED_NON_MARKET_SYMBOLS:6" for note in card.notes)
    excluded = {row.wallet: (row.reason, row.observed) for row in card.excluded}
    assert excluded[WALLET_C] == ("NO_TRADABLE_MARKET_FILLS", 5)


def test_sub15_twap_splits_net_above_floor_while_dust_bots_fail(tmp_path: Path) -> None:
    # WALLET_A splits $50 per 5m bar into 10 x $5 fills (eligible when bar-netted).
    # WALLET_B places 2 x $5 fills per 5m bar = $10 net (still below $15 floor).
    fills: list[dict] = []
    tid = 1
    for bar_idx in range(40):
        bar_ts = START + bar_idx * GRID_MS
        px = 100.0 * (1.0 + 0.001 * bar_idx)
        side = "buy" if bar_idx % 2 == 0 else "sell"
        for sub in range(10):
            fills.append(
                _raw_fill(WALLET_A, "BTC", at_ms=bar_ts + sub * 1000, side=side, size=5.0 / px, price=px, trade_id=tid)
            )
            tid += 1
        for sub in range(2):
            fills.append(
                _raw_fill(WALLET_B, "BTC", at_ms=bar_ts + sub * 1000, side=side, size=5.0 / px, price=px, trade_id=tid)
            )
            tid += 1

    bars = bars_1m_from_fills(fills, quote_volume_usd=Decimal(500_000))
    store = build_store(
        tmp_path,
        fills=fills,
        equity={WALLET_A: (Decimal(12000), 2), WALLET_B: (Decimal(12000), 2)},
        bars_1m_rows=bars,
    )
    try:
        default_card = score_universe(
            store,
            as_of=KnownAtTs(START + DAY_MS),
            lookback_days=2,
            config=ScoreConfig(
                max_euler_cdar_share=Decimal(1),
                max_p95_participation_1m=Decimal(1),
                max_max_participation_1m=Decimal(1),
            ),
            allow_undated=True,
        )
        legacy_card = score_universe(
            store,
            as_of=KnownAtTs(START + DAY_MS),
            lookback_days=2,
            config=ScoreConfig(
                max_euler_cdar_share=Decimal(1),
                max_p95_participation_1m=Decimal(1),
                max_max_participation_1m=Decimal(1),
                net_orders_by_bar=False,
            ),
            allow_undated=True,
        )
    finally:
        store.close()

    assert [entry.wallet for entry in default_card.cards] == [WALLET_A]
    excluded = {row.wallet: row.reason for row in default_card.excluded}
    assert excluded[WALLET_B] == "BASELINE_ORDER_BELOW_FLOOR"
    assert legacy_card.cards == ()
    assert {row.wallet for row in legacy_card.excluded} == {WALLET_A, WALLET_B}


def test_k_scaled_participation_admits_scaled_whale_vs_unscaled_limit(tmp_path: Path) -> None:
    fills = wallet_round_trips(
        WALLET_A, "BTC", start_ms=START, bars=60, base_price=20000.0, size=0.5
    )
    # ~$10,000 per bar vs $150,000 venue quote volume (~6.7% raw, >> 1% limit),
    # while suggested_k ~ 0.10 scales follower participation down to ~0.67% (< 1%).
    bars = bars_1m_from_fills(fills, quote_volume_usd=Decimal(150_000))
    store = build_store(
        tmp_path,
        fills=fills,
        equity={WALLET_A: (Decimal(12000), 2)},
        bars_1m_rows=bars,
    )
    try:
        scaled = score_universe(
            store,
            as_of=KnownAtTs(START + DAY_MS),
            lookback_days=1,
            config=ScoreConfig(max_euler_cdar_share=Decimal(1), max_fixed_k=Decimal("0.10")),
            allow_undated=True,
        )
        unscaled = score_universe(
            store,
            as_of=KnownAtTs(START + DAY_MS),
            lookback_days=1,
            config=ScoreConfig(
                max_euler_cdar_share=Decimal(1),
                max_fixed_k=Decimal("0.10"),
                participation_scale_by_k=False,
            ),
            allow_undated=True,
        )
    finally:
        store.close()

    assert GateFailure.PARTICIPATION_LIMIT not in scaled.by_wallet[WALLET_A].failures
    assert GateFailure.PARTICIPATION_LIMIT in unscaled.by_wallet[WALLET_A].failures


def test_trajectory_expectancy_separates_swing_winners_from_noise_losers(tmp_path: Path) -> None:
    # WALLET_A (swing winner): buys dips that drop -20 bps at +5m then rally +150 bps before close.
    # WALLET_B (churn loser): buys bounces +20 bps at +5m then collapse -150 bps before close.
    fills: list[dict] = []
    bars_rows: list[dict] = []
    tid = 1
    for cycle in range(25):
        base_bar = cycle * 4
        t0 = START + base_bar * GRID_MS
        t1 = START + (base_bar + 1) * GRID_MS
        t2 = START + (base_bar + 2) * GRID_MS
        # WALLET_A on BTC: entry 100.0, +5m close 99.8 (-20 bps), exit at t2 101.5 (+150 bps)
        fills.append(_raw_fill(WALLET_A, "BTC", at_ms=t0, side="buy", size=10.0, price=100.0, trade_id=tid))
        fills.append(_raw_fill(WALLET_A, "BTC", at_ms=t2, side="sell", size=10.0, price=101.5, trade_id=tid + 1))
        # WALLET_B on ETH: entry 100.0, +5m close 100.2 (+20 bps), exit at t2 98.5 (-150 bps)
        fills.append(_raw_fill(WALLET_B, "ETH", at_ms=t0, side="buy", size=10.0, price=100.0, trade_id=tid + 2))
        fills.append(_raw_fill(WALLET_B, "ETH", at_ms=t2, side="sell", size=10.0, price=98.5, trade_id=tid + 3))
        tid += 4

        from datetime import UTC, datetime

        for coin, p0, p1, p2, p3 in (
            ("BTC", 100.0, 99.8, 101.5, 101.5),
            ("ETH", 100.0, 100.2, 98.5, 98.5),
        ):
            for ts_ms, px in ((t0, p0), (t1, p1), (t2, p2), (t2 + GRID_MS, p3)):
                dt = datetime.fromtimestamp(ts_ms / 1000, UTC)
                bars_rows.append(
                    {
                        "coin": coin,
                        "venue": "hyperliquid",
                        "bar_ts": dt,
                        "open": Decimal(str(px)),
                        "high": Decimal(str(px)),
                        "low": Decimal(str(px)),
                        "close": Decimal(str(px)),
                        "volume": Decimal(10000),
                        "quote_volume_usd": Decimal(1_000_000),
                        "event_ts_ms": ts_ms,
                        "first_known_at_ms": ts_ms + 60_000,
                        "arrival_seq": len(bars_rows),
                        "source_id": "synthetic-bars",
                        "source_rank": 1,
                        "batch_sha16": "",
                    }
                )

    store = build_store(
        tmp_path,
        fills=fills,
        equity={WALLET_A: (Decimal(12000), 2), WALLET_B: (Decimal(12000), 2)},
        bars_1m_rows=bars_rows,
    )
    try:
        traj = score_universe(
            store,
            as_of=KnownAtTs(START + DAY_MS),
            lookback_days=2,
            config=PERMISSIVE,
            allow_undated=True,
        )
        markout_only = score_universe(
            store,
            as_of=KnownAtTs(START + DAY_MS),
            lookback_days=2,
            config=ScoreConfig(
                min_baseline_order_usd=Decimal(1),
                max_euler_cdar_share=Decimal(1),
                max_p95_participation_1m=Decimal(1),
                max_max_participation_1m=Decimal(1),
                expectancy_mode="bar_markout",
            ),
            allow_undated=True,
        )
    finally:
        store.close()

    assert GateFailure.NEGATIVE_5M_MARKOUT not in traj.by_wallet[WALLET_A].failures
    assert traj.by_wallet[WALLET_A].markout_5m_bps > Decimal(0)
    assert GateFailure.NEGATIVE_5M_MARKOUT in traj.by_wallet[WALLET_B].failures
    assert traj.by_wallet[WALLET_B].markout_5m_bps < Decimal(0)
    # Inverted on raw +5m entry noise:
    assert GateFailure.NEGATIVE_5M_MARKOUT in markout_only.by_wallet[WALLET_A].failures
    assert GateFailure.NEGATIVE_5M_MARKOUT not in markout_only.by_wallet[WALLET_B].failures
