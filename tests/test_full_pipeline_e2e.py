"""Full-pipeline end-to-end tests: mock wallet universe -> Phase-D kernel -> signed release -> Guardian.

Runs strictly over isolated temporary directories (``tmp_path``) with synthetic
wallets, ``FactStore`` (strict PIT ``allow_undated=False``), ``RosterStore``,
``RulebookCatalog`` + ``lookup_tier_fee``, ``run_kernel_backtest``, offline
Ed25519 ``portfolio_release_v2`` signing, and ``prop-production`` ``Guardian``
+ ``CheckpointSourceReceiver(wire_version=3)`` + ``Journal`` + ``FakeBroker``.

Establishes ``ENGINEERING_VERIFIED`` behaviour only; never grants
``PROSPECTIVE_PAPER_ACTIVE``, ``SCIENTIFICALLY_QUALIFIED``, or ``TRADING_AUTHORIZED``.
"""

from __future__ import annotations

import sqlite3
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from research_fixtures import (
    DAY_MS,
    GRID_MS,
    WALLET_A,
    WALLET_B,
    bars_1m_from_fills,
    cwsa_sample,
    equity_rows,
    resolve_prop_production_root,
    wallet_round_trips,
)

PROD = resolve_prop_production_root()
if PROD is None:
    raise RuntimeError("prop-production sibling repo not found; set PROP_PRODUCTION_ROOT")
for candidate in (PROD / "src", PROD / "tests"):
    if str(candidate) not in sys.path and candidate.is_dir():
        sys.path.append(str(candidate))

from copytrade_kernel import KnownAtTs, Rulebook, RulebookCatalog
from copytrade_kernel.clocks import DecisionMs
from copytrade_pm.facts import FactIngestor, FactStore
from copytrade_pm.facts.migrate import Migration, new_receipt
from copytrade_pm.facts.sources import build_catalogue
from copytrade_pm.pm_checkpoint_journal import CwsaPolicy
from copytrade_pm.pm_checkpoint_source import (
    CheckpointScope,
    SamplingLimits,
    export_checkpoint_packet,
)
from copytrade_pm.pm_forward_journal import PaperJournal
from copytrade_pm.production_manifest import canonical as pm_canonical
from copytrade_pm.roster import (
    AllocationMode,
    GateFailure,
    ReconcilerPolicy,
    ReleaseBinding,
    RosterStore,
    SeatSpec,
    build_release_proposal,
    finalise_payload,
    reconcile_roster,
)

from trading_ops.research import (
    BacktestConfig,
    BacktestWindow,
    MarketSpec,
    ScoreConfig,
    bench_from_scorecard,
    lookup_tier_fee,
    run_kernel_backtest,
    score_universe,
)

WALLET_LOW_AV = "0x" + "c" * 40
WALLET_NO_EQUITY = "0x" + "d" * 40

ACCOUNT_ID = "fixture-account"
BASE_START = int(datetime(2026, 11, 10, tzinfo=UTC).timestamp() * 1000)
T0 = BASE_START + 3 * DAY_MS
T1 = T0 + GRID_MS
T2 = T0 + 2 * GRID_MS
T3 = T0 + 3 * GRID_MS

BTC_BID = Decimal(9999)
BTC_ASK = Decimal(10001)
BTC_MID = Decimal(10000)
ETH_BID = Decimal(999)
ETH_ASK = Decimal(1001)
ETH_MID = Decimal(1000)

DEPTH = Decimal(200)
BTC_LIQUIDITY = DEPTH * BTC_MID
ETH_LIQUIDITY = DEPTH * ETH_MID
INITIAL_RATE = Decimal("0.5")
QTY_STEP = Decimal("0.00001")
RESEARCH_COMMIT = "a" * 40


@dataclass(frozen=True)
class E2EReleaseFixture:
    release: Any
    final_payload: dict[str, Any]
    signed_envelope: dict[str, Any]
    authority_key: Ed25519PrivateKey
    source_key: Ed25519PrivateKey
    cwsa_policy: CwsaPolicy
    cwsa_limits: SamplingLimits
    scopes: dict[str, CheckpointScope]
    g_rules: Any
    g_policy: dict[str, Any]
    g_markets: dict[str, Any]
    owners_by_wallet: dict[str, str]
    k_by_owner: dict[str, Decimal]
    leader_positions: dict[int, dict[str, str]]


@dataclass
class GuardianStack:
    floor: Any
    journal: Any
    broker: Any
    receiver: Any
    guardian: Any


def _pit_fill_row(
    wallet: str,
    coin: str,
    at_ms: int,
    side: str,
    size: float,
    price: float,
    trade_id: int,
) -> dict[str, Any]:
    px = Decimal(str(round(price, 8)))
    sz = Decimal(str(round(size, 8)))
    ts = datetime.fromtimestamp(at_ms / 1000, UTC)
    return {
        "wallet": wallet,
        "coin": coin,
        "side": side,
        "size": sz,
        "price": px,
        "size_usd": (sz * px).quantize(Decimal("0.00000001")),
        "trade_id": str(trade_id),
        "order_id": str(trade_id),
        "exchange_ts": ts,
        "event_ts_ms": at_ms,
        "source_ts": ts.isoformat(),
    }


def _to_pit_fills(raw_fills: list[dict[str, Any]], *, start_trade_id: int = 1) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, row in enumerate(raw_fills):
        at_ms = int(row["exchange_ts"].timestamp() * 1000)
        rows.append(
            _pit_fill_row(
                row["wallet"],
                row["coin"],
                at_ms,
                row["side"],
                float(row["size"]),
                float(row["price"]),
                start_trade_id + offset,
            )
        )
    return rows


def _flat_round_trips_before_t0(
    wallet: str,
    coin: str,
    *,
    start_ms: int,
    base_price: float,
    size: float,
    start_trade_id: int,
) -> list[dict[str, Any]]:
    raw = wallet_round_trips(
        wallet,
        coin,
        start_ms=start_ms,
        bars=2 * 288 - 1,
        base_price=base_price,
        size=size,
    )
    net = sum(Decimal(str(r["size"])) * (1 if r["side"] == "buy" else -1) for r in raw)
    rows = _to_pit_fills(raw, start_trade_id=start_trade_id)
    if net != 0:
        rows.append(
            _pit_fill_row(
                wallet,
                coin,
                T0 - GRID_MS,
                "sell" if net > 0 else "buy",
                float(abs(net)),
                base_price,
                start_trade_id + len(rows),
            )
        )
    return rows


def _build_strict_pit_universe(root: Path) -> FactStore:
    lookback_start = T0 - 2 * DAY_MS
    fills: list[dict[str, Any]] = []

    # WALLET_A: BTC healthy leader (T0=0 -> T1=+2 -> T2=+1 -> T3=0)
    fills.extend(
        _flat_round_trips_before_t0(
            WALLET_A, "BTC", start_ms=lookback_start, base_price=10000.0, size=0.0625, start_trade_id=1
        )
    )
    tid = len(fills) + 1
    fills.extend(
        [
            _pit_fill_row(WALLET_A, "BTC", T1, "buy", 2.0, 10000.0, tid),
            _pit_fill_row(WALLET_A, "BTC", T2, "sell", 1.0, 10000.0, tid + 1),
            _pit_fill_row(WALLET_A, "BTC", T3, "sell", 1.0, 10000.0, tid + 2),
        ]
    )

    # WALLET_B: ETH healthy leader (T0=0 -> T1=+4 -> T2=+2 -> T3=0)
    tid = len(fills) + 1
    fills.extend(
        _flat_round_trips_before_t0(
            WALLET_B, "ETH", start_ms=lookback_start, base_price=1000.0, size=0.625, start_trade_id=tid
        )
    )
    tid = len(fills) + 1
    fills.extend(
        [
            _pit_fill_row(WALLET_B, "ETH", T1, "buy", 4.0, 1000.0, tid),
            _pit_fill_row(WALLET_B, "ETH", T2, "sell", 2.0, 1000.0, tid + 1),
            _pit_fill_row(WALLET_B, "ETH", T3, "sell", 2.0, 1000.0, tid + 2),
        ]
    )

    # Bad candidate 1: account value below $1k floor
    tid = len(fills) + 1
    fills.extend(
        _flat_round_trips_before_t0(
            WALLET_LOW_AV,
            "BTC",
            start_ms=lookback_start,
            base_price=10000.0,
            size=0.0625,
            start_trade_id=tid,
        )
    )

    # Bad candidate 2: omitted from equity table (must be excluded, never zeroed)
    tid = len(fills) + 1
    fills.extend(
        _flat_round_trips_before_t0(
            WALLET_NO_EQUITY,
            "ETH",
            start_ms=lookback_start,
            base_price=2000.0,
            size=0.3125,
            start_trade_id=tid,
        )
    )

    store_root = root / "facts"
    ws_root = root / "wallet_store"
    ws_root.mkdir(parents=True, exist_ok=True)
    by_wallet_day: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in fills:
        day = row["exchange_ts"].date().isoformat()
        by_wallet_day.setdefault((row["wallet"], day), []).append(row)

    con = sqlite3.connect(ws_root / "catalog.sqlite")
    con.execute(
        "CREATE TABLE partitions(kind TEXT, wallet TEXT, day TEXT, path TEXT, rows INTEGER, "
        "bytes INTEGER, sha256 TEXT, fetched_at TEXT, source TEXT, source_cmd TEXT)"
    )
    for (wallet, day), day_rows in sorted(by_wallet_day.items()):
        year, month, dom = day.split("-")
        p_dir = ws_root / "fills" / f"wallet={wallet}" / f"year={year}" / f"month={month}" / f"day={dom}"
        p_dir.mkdir(parents=True, exist_ok=True)
        p_file = p_dir / "data.parquet"
        pl.DataFrame(
            {
                "ts": [r["exchange_ts"] for r in day_rows],
                "wallet": [r["wallet"] for r in day_rows],
                "symbol": [r["coin"] for r in day_rows],
                "dex": ["HL" for _ in day_rows],
                "side": [r["side"] for r in day_rows],
                "size_token": [float(r["size"]) for r in day_rows],
                "price": [float(r["price"]) for r in day_rows],
                "size_usd": [float(r["size_usd"]) for r in day_rows],
                "fee": [0.0 for _ in day_rows],
                "fee_token": ["USDC" for _ in day_rows],
                "is_maker": [False for _ in day_rows],
                "order_id": [r["order_id"] for r in day_rows],
                "trade_id": [int(r["trade_id"]) for r in day_rows],
                "crossed": [True for _ in day_rows],
                "closed_pnl": [0.0 for _ in day_rows],
                "start_position": [0.0 for _ in day_rows],
                "source": ["vike" for _ in day_rows],
                "source_ts": [r["source_ts"] for r in day_rows],
            }
        ).with_columns(
            pl.col("ts").cast(pl.Datetime("us", "UTC")),
            pl.col("trade_id").cast(pl.UInt64),
        ).write_parquet(p_file)
        max_event_ms = max(r["event_ts_ms"] for r in day_rows)
        fetched_iso = datetime.fromtimestamp((max_event_ms + 1_000) / 1000, UTC).isoformat()
        con.execute(
            "INSERT INTO partitions VALUES(?,?,?,?,?,?,?,?,?,?)",
            ("fills", wallet, day, str(p_file), len(day_rows), p_file.stat().st_size, "x", fetched_iso, "vike", ""),
        )
    con.commit()
    con.close()

    sources, absent = build_catalogue(wallet_store=ws_root)
    store = FactStore(store_root, create=True)
    Migration(store=store, sources=sources, kinds=("fills",)).run(
        new_receipt(store_root=store_root, sources=sources, not_supplied=absent, kinds=("fills",))
    )
    ingestor = FactIngestor(store)
    for wallet, av in (
        (WALLET_A, Decimal(12000)),
        (WALLET_B, Decimal(15000)),
        (WALLET_LOW_AV, Decimal(500)),
    ):
        ingestor.ingest_rows(
            "equity",
            equity_rows(wallet, start_ms=lookback_start, equity_usd=av, days=5),
            source_id="synthetic-capture",
            source_rank=1,
            first_known_at_ms=None,
        )
    bars = bars_1m_from_fills(fills, first_known_at_offset_ms=60_000)
    ingestor.ingest_rows("bars_1m", bars, source_id="synthetic-bars", source_rank=1, first_known_at_ms=None)
    return store


def _fact_store_net_position(store: FactStore, wallet: str, coin: str, *, up_to_ms: int) -> str:
    df = (
        store.scan(
            "fills",
            wallets=[wallet],
            coins=[coin],
            as_of_ms=T3 + 2_000,
            allow_unknown_known_at=False,
        )
        .frame.filter(pl.col("event_ts_ms") <= up_to_ms)
        .select(["side", "size"])
        .collect()
    )
    net = sum(
        (Decimal(str(row["size"])) * (1 if row["side"] == "buy" else -1))
        for row in df.iter_rows(named=True)
    )
    if net == 0:
        return "0"
    formatted = format(net.normalize(), "f")
    return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted


def _market_spec(symbol: str, liquidity_notional: Decimal) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "instrument_type": "perp",
        "verified": True,
        "source_mode": "SYNTHETIC",
        "valid_until_ms": 10**15,
        "qty_step": "0.00001",
        "price_tick": "0.00000001",
        "min_qty": "0.00001",
        "min_notional": "10",
        "liquidity_notional": str(liquidity_notional),
        "initial_rate": str(INITIAL_RATE),
        "fee_rate": "0",
        "funding_bound": "0",
        "liquidation_rate": "0",
        "tiers": [{"cap": str(liquidity_notional), "rate": str(INITIAL_RATE)}],
        "source_instrument": f"hyperliquid/mainnet:perp:default/{symbol}",
    }


def _checkpoint_scope(wallet: str) -> CheckpointScope:
    return CheckpointScope(
        source_id=wallet,
        account_sha=sha256(wallet.lower().encode()).hexdigest(),
        generation=sha256(f"gen:{wallet}".encode()).hexdigest(),
        selection_sha=sha256(f"sel:{wallet}".encode()).hexdigest(),
        dexes=("default", "xyz"),
    )


def _guardian_env_fixtures(rulebook_fields: dict[str, Any]) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from prop_production.finance import Rulebook as GuardianRulebook

    fee_quote = lookup_tier_fee("propr:classic_1step", 10_000)
    assert fee_quote is not None and fee_quote.fee_usd == pytest.approx(110.0)
    g_rules = GuardianRulebook(**rulebook_fields)
    g_policy = {
        "schema": "risk_policy_v1",
        "source_mode": "SYNTHETIC",
        "scale_steps": 4,
        "scenarios": [{"BTC": "-0.10", "ETH": "-0.10"}],
        "minimum_headroom": "50",
        "made_limit": "1000",
        "pod_loss_budget": "400",
        "hedge_tolerance": "0.05",
        "shortfall_bps": "30",
        "margin_multiplier": "1",
        "checkpoint_execution": {
            "quote_max_age_ms": 2000,
            "account_max_age_ms": 2000,
            "max_gross_notional": "20000",
        },
    }
    g_markets = {
        "BTC": _market_spec("BTC", BTC_LIQUIDITY),
        "ETH": _market_spec("ETH", ETH_LIQUIDITY),
    }
    return g_rules, g_policy, g_markets


def _stage_and_sign_e2e_release(tmp_path: Path) -> E2EReleaseFixture:
    """Run Stages 0-4 and return a bundled ``E2EReleaseFixture``."""
    from prop_production.copy_mandates import CopyMandate
    from prop_production.evidence import digest as prod_digest
    from prop_production.evidence import sign as prod_sign
    from prop_production.release import ReleaseVerifier

    store = _build_strict_pit_universe(tmp_path)
    roster_db = tmp_path / "roster.sqlite"
    score_cfg = ScoreConfig(
        min_baseline_order_usd=Decimal(1),
        max_fixed_k=Decimal("0.01"),
        max_euler_cdar_share=Decimal("0.70"),
        max_p95_participation_1m=Decimal(1),
        max_max_participation_1m=Decimal(1),
    )
    seats = (
        SeatSpec("seat_1", Decimal(200), Decimal("0.70")),
        SeatSpec("seat_2", Decimal(200), Decimal("0.70")),
    )
    as_of = KnownAtTs(T0 - 1_000)

    cwsa_limits = SamplingLimits(
        max_state_age_ms=4000,
        max_acquisition_ms=3000,
        max_dex_skew_ms=2000,
        publication_grace_ms=5000,
    )
    cwsa_policy = CwsaPolicy(limits=cwsa_limits, measurement_sha="c" * 64)
    k_spec = RulebookCatalog.get("propr:classic_1step:phase_1")
    rulebook_fields = k_spec.to_rulebook_fields(
        account_binding=ACCOUNT_ID, scale=2, evidence_sha="d" * 64
    )

    try:
        # Stage 1: strict PIT Stage-1 audit (allow_undated=False)
        universe = score_universe(
            store, as_of=as_of, lookback_days=2, config=score_cfg, allow_undated=False
        )
        assert universe.basis["pit"] == "STRICT_KNOWN_AT"
        assert {ex.wallet for ex in universe.excluded} == {WALLET_NO_EQUITY}
        cards_by_wallet = {c.wallet: c for c in universe.cards}
        assert GateFailure.ACCOUNT_VALUE_BELOW_FLOOR in cards_by_wallet[WALLET_LOW_AV].failures
        assert cards_by_wallet[WALLET_A].eligible
        assert cards_by_wallet[WALLET_B].eligible

        roster = RosterStore(roster_db, create=True, writable=True)
        try:
            roster.register_account(
                ACCOUNT_ID, seats, policy_epoch=cwsa_policy.epoch, at_ms=as_of.value
            )
            universe.record(roster)
            bench = bench_from_scorecard(universe)
            roster.record_bench(ACCOUNT_ID, bench, as_of=as_of)

            # Stage 2: Roster reconciliation -> SeatBook
            plan = reconcile_roster(
                current=roster.book_at(ACCOUNT_ID, as_of=as_of),
                scorecard=roster.latest_scorecard(as_of=as_of),
                bench=roster.latest_bench(ACCOUNT_ID, as_of=as_of),
                vetoes=roster.active_vetoes(as_of=as_of),
                policy=ReconcilerPolicy.sticky_pit_bench(),
                as_of=as_of,
            )
            assert plan.trading_authorized is False
            receipt = roster.commit_plan(plan, committed_ms=as_of.value)
            book = roster.book_at(ACCOUNT_ID, as_of=as_of)
            assert book is not None
            assert book.mode == AllocationMode.ACTIVE
            assert {lease.wallet for lease in book.leases} == {WALLET_A, WALLET_B}
            inventory_sha = roster.inventory_sha()
        finally:
            roster.close()

        # Stage 3: Stage-2 kernel walk-forward + derive leader checkpoint positions from FactStore
        bt_rulebook = Rulebook(**rulebook_fields)
        bt_report = run_kernel_backtest(
            store,
            window=BacktestWindow(T0, T3 + GRID_MS),
            seats=seats,
            rulebook=bt_rulebook,
            markets=(MarketSpec("BTC"), MarketSpec("ETH")),
            config=BacktestConfig(
                account_id=ACCOUNT_ID,
                lookback_days=2,
                allow_undated=False,
                score_config=score_cfg,
            ),
        )
        assert bt_report.challenge_pass_rate == Decimal(1)
        assert len(bt_report.decisions) >= 2
        assert bt_report.ending_state.owner_books

        leader_positions = {
            ts: {
                WALLET_A: _fact_store_net_position(store, WALLET_A, "BTC", up_to_ms=ts),
                WALLET_B: _fact_store_net_position(store, WALLET_B, "ETH", up_to_ms=ts),
            }
            for ts in (T0, T1, T2, T3)
        }
        assert leader_positions[T0] == {WALLET_A: "0", WALLET_B: "0"}
        assert leader_positions[T1] == {WALLET_A: "2", WALLET_B: "4"}
        assert leader_positions[T2] == {WALLET_A: "1", WALLET_B: "2"}
        assert leader_positions[T3] == {WALLET_A: "0", WALLET_B: "0"}
    finally:
        store.close()

    # Stage 4: Build unsigned ReleaseProposal, sign offline, and verify via ReleaseVerifier
    g_rules, g_policy, g_markets = _guardian_env_fixtures(rulebook_fields)
    risk_policy_sha = prod_digest({"policy": g_policy, "markets": g_markets})
    scope_a = _checkpoint_scope(WALLET_A)
    scope_b = _checkpoint_scope(WALLET_B)
    lease_markets = {
        lease.seat_id: (
            ("hyperliquid/mainnet:perp:default/BTC",)
            if lease.wallet == WALLET_A
            else ("hyperliquid/mainnet:perp:default/ETH",)
        )
        for lease in book.leases
    }
    seal_ts = as_of.value
    binding = ReleaseBinding(
        account_id=ACCOUNT_ID,
        generation=1,
        research_commit=RESEARCH_COMMIT,
        decision_sha=bt_report.decisions[-1].decision_sha,
        selection_known_at_ms=seal_ts,
        source_mode="SYNTHETIC",
        risk_policy_sha=risk_policy_sha,
        rulebook_sha=g_rules.sha,
        checkpoint_policy=cwsa_policy.model_dump(mode="json"),
        checkpoint_scopes={
            WALLET_A: scope_a.model_dump(mode="json"),
            WALLET_B: scope_b.model_dump(mode="json"),
        },
        allowed_markets=lease_markets,
        created_ms=seal_ts,
        expires_at_ms=T3 + DAY_MS,
    )
    proposal = build_release_proposal(
        book,
        binding,
        inventory_sha=inventory_sha,
        plan_sha=receipt.plan_sha,
        created_ms=seal_ts,
    )

    authority_key = Ed25519PrivateKey.generate()
    source_key = Ed25519PrivateKey.generate()
    authority_sha = sha256(authority_key.public_key().public_bytes_raw()).hexdigest()
    final_payload = finalise_payload(proposal.payload, authority_sha)
    signed_envelope = prod_sign(final_payload, authority_key, "approver")

    release = ReleaseVerifier(
        authority_key.public_key(), "approver", ACCOUNT_ID, risk_policy_sha, g_rules.sha
    ).verify(signed_envelope, now_ms=seal_ts)

    mandates = [CopyMandate.parse(m) for m in final_payload["mandates"]]
    owners_by_wallet = {m.data["source_id"]: m.owner for m in mandates}
    k_by_owner = {m.owner: Decimal(m.data["k"]) for m in mandates}

    return E2EReleaseFixture(
        release=release,
        final_payload=final_payload,
        signed_envelope=signed_envelope,
        authority_key=authority_key,
        source_key=source_key,
        cwsa_policy=cwsa_policy,
        cwsa_limits=cwsa_limits,
        scopes={WALLET_A: scope_a, WALLET_B: scope_b},
        g_rules=g_rules,
        g_policy=g_policy,
        g_markets=g_markets,
        owners_by_wallet=owners_by_wallet,
        k_by_owner=k_by_owner,
        leader_positions=leader_positions,
    )


@contextmanager
def _open_guardian_stack(
    g_root: Path,
    fx: E2EReleaseFixture,
    *,
    initialize: bool,
    fill_on_submit: bool = False,
) -> Iterator[GuardianStack]:
    from prop_production.checkpoint_receiver import CheckpointSourceReceiver
    from prop_production.copy_mandates import CopyMandate
    from prop_production.evidence import sign as prod_sign
    from prop_production.guardian import Guardian
    from prop_production.journal import Journal
    from prop_production.judge.broker import FakeBroker
    from prop_production.release import ReleaseFloor

    if initialize:
        g_root.mkdir(mode=0o700, exist_ok=True)
    floor = ReleaseFloor(g_root / "floor.sqlite3", initialize=initialize)
    journal = Journal(g_root / "guardian.sqlite3", ACCOUNT_ID, initialize=initialize)
    broker = FakeBroker(g_root / "broker.sqlite3")
    broker.fill_on_submit = fill_on_submit
    receiver = CheckpointSourceReceiver(
        g_root / "source.sqlite3",
        release=fx.release,
        source_key=fx.source_key.public_key(),
        wire_version=3,
    )
    try:
        if initialize:
            floor.accept(fx.release)
        guardian = Guardian(
            journal,
            fx.release,
            floor,
            fx.source_key.public_key(),
            fx.g_rules,
            fx.g_policy,
            fx.g_markets,
            broker,
            mandate_key=fx.authority_key.public_key(),
        )
        if initialize:
            for m_body in fx.final_payload["mandates"]:
                mandate = CopyMandate.parse(m_body)
                guardian.activate_copy_mandate(
                    prod_sign(mandate.data, fx.authority_key, "copy-mandate-v1"),
                    now_ms=fx.final_payload["created_at_ms"],
                )
        yield GuardianStack(
            floor=floor,
            journal=journal,
            broker=broker,
            receiver=receiver,
            guardian=guardian,
        )
    finally:
        receiver.close()
        journal.close()
        broker.close()
        floor.close()


def _emit_checkpoint(
    src_journal: PaperJournal,
    fx: E2EReleaseFixture,
    wallet: str,
    coin: str,
    mid: Decimal,
    boundary: int,
    qty: str,
    *,
    after_cursor: int,
) -> tuple[bytes, str]:
    scope = fx.scopes[wallet]
    out = src_journal.append_checkpoint_sample(
        fx.cwsa_policy.epoch,
        cwsa_sample(
            evidence_root=src_journal.root,
            scope=scope,
            limits=fx.cwsa_limits,
            wallet=wallet,
            coin=coin,
            mid=mid,
            boundary=boundary,
            qty=qty,
        ),
        boundary + 101,
    )
    pkt = pm_canonical(
        export_checkpoint_packet(
            src_journal.path,
            scope,
            fx.cwsa_policy,
            signer=fx.source_key,
            after_cursor=after_cursor,
            issued_ms=boundary + 102,
        )
    )
    return pkt, out.outcome_id


def _emit_pair(
    src_journal: PaperJournal,
    fx: E2EReleaseFixture,
    boundary: int,
    qty_by_wallet: dict[str, str],
    *,
    after_cursor: int,
) -> tuple[tuple[bytes, bytes], dict[str, str]]:
    pkt_a, out_a = _emit_checkpoint(
        src_journal,
        fx,
        WALLET_A,
        "BTC",
        BTC_MID,
        boundary,
        qty_by_wallet[WALLET_A],
        after_cursor=after_cursor,
    )
    pkt_b, out_b = _emit_checkpoint(
        src_journal,
        fx,
        WALLET_B,
        "ETH",
        ETH_MID,
        boundary,
        qty_by_wallet[WALLET_B],
        after_cursor=after_cursor,
    )
    return (pkt_a, pkt_b), {
        fx.owners_by_wallet[WALLET_A]: out_a,
        fx.owners_by_wallet[WALLET_B]: out_b,
    }


def _run_wave(
    stack: GuardianStack,
    src_journal: PaperJournal,
    fx: E2EReleaseFixture,
    boundary: int,
    *,
    cursor: int,
    btc_px: Decimal,
    eth_px: Decimal,
) -> int:
    (pa, pb), reqs = _emit_pair(
        src_journal,
        fx,
        boundary,
        fx.leader_positions[boundary],
        after_cursor=cursor,
    )
    wave_ms = boundary + 102
    stack.receiver.ingest(pa, now_ms=wave_ms)
    stack.receiver.ingest(pb, now_ms=wave_ms)
    stack.guardian.reconcile(wave_ms)
    quotes = {
        "BTC": {"bid": str(BTC_BID), "ask": str(BTC_ASK), "known_at_ms": wave_ms},
        "ETH": {"bid": str(ETH_BID), "ask": str(ETH_ASK), "known_at_ms": wave_ms},
    }
    decision = stack.guardian.process_checkpoint(
        stack.receiver, reqs, quotes=quotes, now_ms=DecisionMs(wave_ms)
    )
    assert decision["status"] == "REVIEWABLE", decision.get("reasons") or decision
    intent_ids = stack.guardian.reserve_checkpoint(
        decision["decision_id"], receiver=stack.receiver, now_ms=wave_ms + 1
    )
    assert len(intent_ids) == 2
    snap_before = stack.journal.snapshot()
    for idx, iid in enumerate(intent_ids):
        stack.guardian.dispatch(iid, wave_ms + 2 + idx)
        mkt = snap_before["intents"][iid]["market"]
        fill_px = btc_px if mkt == "BTC" else eth_px
        stack.broker.fill(iid, "1", str(fill_px))
    stack.guardian.reconcile(wave_ms + 5)
    return len(intent_ids)


def _scaled_qty(k: Decimal, leader_qty: str, *, mid: Decimal, ask: Decimal) -> Decimal:
    return (k * Decimal(leader_qty) * mid / ask).quantize(QTY_STEP, rounding=ROUND_DOWN)


def test_full_pipeline_mock_wallets_to_signed_release_and_guardian_execution(
    tmp_path: Path,
) -> None:
    """Mock 4-wallet strict-PIT FactStore -> Roster -> Backtest -> Signed Release -> Guardian."""
    fx = _stage_and_sign_e2e_release(tmp_path)
    assert fx.release.payload["schema"] == "portfolio_release_v2"
    assert len(fx.release.payload["mandates"]) == 2

    owner_a = fx.owners_by_wallet[WALLET_A]
    owner_b = fx.owners_by_wallet[WALLET_B]
    exp_t1_btc = _scaled_qty(
        fx.k_by_owner[owner_a], fx.leader_positions[T1][WALLET_A], mid=BTC_MID, ask=BTC_ASK
    )
    exp_t1_eth = _scaled_qty(
        fx.k_by_owner[owner_b], fx.leader_positions[T1][WALLET_B], mid=ETH_MID, ask=ETH_ASK
    )
    exp_t2_btc = _scaled_qty(
        fx.k_by_owner[owner_a], fx.leader_positions[T2][WALLET_A], mid=BTC_MID, ask=BTC_ASK
    )
    exp_t2_eth = _scaled_qty(
        fx.k_by_owner[owner_b], fx.leader_positions[T2][WALLET_B], mid=ETH_MID, ask=ETH_ASK
    )

    src_journal = PaperJournal(tmp_path / "cwsa_source")
    src_journal.register_checkpoint_source(
        fx.scopes[WALLET_A], fx.cwsa_policy, first_checkpoint_ms=T0, at_ms=T0 - 1
    )
    src_journal.register_checkpoint_source(
        fx.scopes[WALLET_B], fx.cwsa_policy, first_checkpoint_ms=T0, at_ms=T0 - 1
    )

    g_root = tmp_path / "guardian"

    # Lifecycle phase 1: T0 baseline + T1 open on initial Guardian stack
    with _open_guardian_stack(g_root, fx, initialize=True, fill_on_submit=False) as stack1:
        (pkt_a0, pkt_b0), _ = _emit_pair(src_journal, fx, T0, fx.leader_positions[T0], after_cursor=0)
        stack1.receiver.ingest(pkt_a0, now_ms=T0 + 102)
        stack1.receiver.ingest(pkt_b0, now_ms=T0 + 102)
        for owner in fx.owners_by_wallet.values():
            stack1.guardian.baseline_copy_mandate(owner, stack1.receiver, now_ms=T0 + 102)

        assert _run_wave(stack1, src_journal, fx, T1, cursor=2, btc_px=BTC_ASK, eth_px=ETH_ASK) == 2
        snap_t1 = stack1.journal.snapshot()
        assert Decimal(snap_t1["pod_positions"][owner_a]["BTC"]) == exp_t1_btc
        assert Decimal(snap_t1["pod_positions"][owner_b]["ETH"]) == exp_t1_eth
        broker_pos_t1 = {p.market: p.quantity for p in stack1.broker.observe(T1 + 108).positions}
        assert broker_pos_t1 == {"BTC": exp_t1_btc, "ETH": exp_t1_eth}

    # Lifecycle phase 2: Full Guardian restart while carrying live T1 positions -> T2 reduce -> T3 close
    with _open_guardian_stack(g_root, fx, initialize=False, fill_on_submit=False) as stack2:
        assert _run_wave(stack2, src_journal, fx, T2, cursor=3, btc_px=BTC_BID, eth_px=ETH_BID) == 2
        snap_t2 = stack2.journal.snapshot()
        assert Decimal(snap_t2["pod_positions"][owner_a]["BTC"]) == exp_t2_btc
        assert Decimal(snap_t2["pod_positions"][owner_b]["ETH"]) == exp_t2_eth
        broker_pos_t2 = {p.market: p.quantity for p in stack2.broker.observe(T2 + 108).positions}
        assert broker_pos_t2 == {"BTC": exp_t2_btc, "ETH": exp_t2_eth}

        assert _run_wave(stack2, src_journal, fx, T3, cursor=4, btc_px=BTC_BID, eth_px=ETH_BID) == 2
        snap_t3 = stack2.journal.snapshot()
        assert snap_t3["halts"] == []
        assert snap_t3["runtime"]["state"] == "DISARMED"
        assert stack2.broker.observe(T3 + 108).positions == ()

        intents = list(snap_t3["intents"].values())
        assert len(intents) == 6
        assert all(i["status"] == "FILLED" for i in intents)
        buys = [i for i in intents if Decimal(i["quantity"]) > 0]
        sells = [i for i in intents if Decimal(i["quantity"]) < 0]
        assert len(buys) == 2 and {i["market"] for i in buys} == {"BTC", "ETH"}
        assert len(sells) == 4
        assert [i for i in sells if i["market"] == "BTC"] and len([i for i in sells if i["market"] == "BTC"]) == 2
        assert [i for i in sells if i["market"] == "ETH"] and len([i for i in sells if i["market"] == "ETH"]) == 2

        open_positions = [
            (owner, mkt, qty)
            for owner, pos_map in snap_t3["pod_positions"].items()
            for mkt, qty in pos_map.items()
            if Decimal(qty) != 0
        ]
        assert open_positions == []


def test_full_pipeline_fail_closed_on_tampered_release_and_halt_persistence(
    tmp_path: Path,
) -> None:
    """Tampered signed release rejects fail-closed; Guardian HALT persists across restart."""
    from prop_production.release import ReleaseVerifier

    fx = _stage_and_sign_e2e_release(tmp_path)

    # 1. Tamper with member k in signed envelope -> ReleaseVerifier must fail closed
    tampered = deepcopy(fx.signed_envelope)
    tampered["payload"]["members"][0]["k"] = "9.99"
    with pytest.raises(ValueError):
        ReleaseVerifier(
            fx.authority_key.public_key(),
            "approver",
            ACCOUNT_ID,
            fx.release.payload["risk_policy_sha"],
            fx.g_rules.sha,
        ).verify(tampered, now_ms=T0 - 10)

    # 2. Trigger HALT via broker reconciliation uncertainty, restart Guardian, verify PERSISTENT_HALT
    src_journal = PaperJournal(tmp_path / "cwsa_source")
    src_journal.register_checkpoint_source(
        fx.scopes[WALLET_A], fx.cwsa_policy, first_checkpoint_ms=T0, at_ms=T0 - 1
    )

    g_root = tmp_path / "guardian_halt"
    with _open_guardian_stack(g_root, fx, initialize=True) as stack1:
        pkt_a0, _ = _emit_checkpoint(
            src_journal, fx, WALLET_A, "BTC", BTC_MID, T0, fx.leader_positions[T0][WALLET_A], after_cursor=0
        )
        stack1.receiver.ingest(pkt_a0, now_ms=T0 + 102)
        stack1.guardian.baseline_copy_mandate(
            fx.owners_by_wallet[WALLET_A], stack1.receiver, now_ms=T0 + 102
        )
        stack1.broker.history_complete = False
        stack1.guardian.reconcile(T0 + 105)
        assert "RECONCILIATION_UNCERTAIN" in stack1.journal.snapshot()["halts"]

    # Reopen persisted Journal/Floor/Broker in fresh Guardian -> must remain HALTED & block open
    with _open_guardian_stack(g_root, fx, initialize=False) as stack2:
        pkt_a1, out_a1 = _emit_checkpoint(
            src_journal, fx, WALLET_A, "BTC", BTC_MID, T1, fx.leader_positions[T1][WALLET_A], after_cursor=2
        )
        stack2.receiver.ingest(pkt_a1, now_ms=T1 + 102)
        quotes = {
            "BTC": {"bid": str(BTC_BID), "ask": str(BTC_ASK), "known_at_ms": T1 + 102},
            "ETH": {"bid": str(ETH_BID), "ask": str(ETH_ASK), "known_at_ms": T1 + 102},
        }
        decision = stack2.guardian.process_checkpoint(
            stack2.receiver,
            {fx.owners_by_wallet[WALLET_A]: out_a1},
            quotes=quotes,
            now_ms=DecisionMs(T1 + 102),
        )
        assert decision["status"] in {"MISSED", "PRE_DISPATCH_BLOCKED"}, decision
        assert "PERSISTENT_HALT" in decision["reasons"], decision
        assert "RECONCILIATION_UNCERTAIN" in stack2.journal.snapshot()["halts"]
        assert stack2.journal.snapshot()["intents"] == {}
        assert stack2.broker.observe(T1 + 105).positions == ()
