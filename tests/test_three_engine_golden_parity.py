"""Cross-engine golden parity: research backtest, PM03 paper shell, Guardian.

One shared multi-checkpoint stream (t0 baseline qty=0, t1 open qty=3 BTC, t2
partial reduce qty=1 BTC) is driven end to end through all three real shells:

1. Research backtest: ``run_kernel_backtest`` on a synthetic ``FactStore``.
2. PM03 paper: ``PaperJournal.consume_checkpoint`` + reopened ``PaperJournal``.
3. Guardian: ``Guardian`` + ``CheckpointSourceReceiver(wire_version=3)`` +
   ``Journal`` + ``FakeBroker`` + reopened ``Journal.snapshot()``.

Every active checkpoint (t1, t2) must agree byte-for-byte on ``decision_sha``
(canonicalized into ``NativeInstrument`` via bijective ``market_map``),
``net_intents``, and persisted ``ending_positions``.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from research_fixtures import (
    DAY_MS,
    GRID_MS,
    WALLET_A,
    build_store,
    cwsa_sample,
    resolve_prop_production_root,
    wallet_round_trips,
)

PROD = resolve_prop_production_root()
if PROD is not None:
    for candidate in (PROD / "src", PROD / "tests"):
        if str(candidate) not in sys.path and candidate.is_dir():
            sys.path.append(str(candidate))

copytrade_kernel = pytest.importorskip("copytrade_kernel")
pytest.importorskip("copytrade_pm.pm_checkpoint_paper")
pytest.importorskip("prop_production.checkpoint_kernel")

from copytrade_kernel import (
    AccountState,
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
    OwnerBook,
    ProspectiveCWSA,
    RiskPolicy,
    SourceLeg,
    SourceOutcome,
    VenueCapability,
    step_checkpoint,
)
from copytrade_kernel.accounting import Rulebook as KernelRulebook
from copytrade_pm.pm03_account import Instrument
from copytrade_pm.pm03_account import digest as pm_digest
from copytrade_pm.pm03_risk import RiskPolicy as PaperRiskPolicy
from copytrade_pm.pm03_rules import propr_rulebook
from copytrade_pm.pm_checkpoint_journal import CwsaPolicy
from copytrade_pm.pm_checkpoint_source import (
    CheckpointScope,
    SamplingLimits,
    export_checkpoint_packet,
)
from copytrade_pm.pm_forward_ccxt_market import parse_ccxt_book, parse_ccxt_market
from copytrade_pm.pm_forward_journal import PaperJournal
from copytrade_pm.pm_forward_paper import PaperManifest
from copytrade_pm.pm_mandate import SignedMandate
from copytrade_pm.production_manifest import canonical as pm_canonical
from copytrade_pm.production_manifest import sign as pm_sign
from copytrade_pm.roster import SeatSpec

from trading_ops.research import (
    BacktestConfig,
    BacktestWindow,
    MarketSpec,
    ScoreConfig,
    decision_payload,
    decision_sha,
    net_intent_rows,
    run_kernel_backtest,
)
from trading_ops.research import backtest as research_backtest

BTC = "hyperliquid/mainnet:perp:default/BTC"
BTC_INST = Instrument("hyperliquid", "mainnet:perp:default", "BTC")
MARKET = "BTC"
MARKET_MAP = {MARKET: BTC}
ACCOUNT_ID = "fixture-account"
PARITY_START = int(datetime(2026, 11, 10, tzinfo=UTC).timestamp() * 1000)
T0 = PARITY_START + 3 * DAY_MS
T1 = T0 + GRID_MS
T2 = T0 + 2 * GRID_MS
BID = Decimal(9999)
ASK = Decimal(10001)
MID = Decimal(10000)
SHORTFALL_BPS = Decimal(30)
FILL_FEE_RATE = SHORTFALL_BPS / Decimal(10000)
INITIAL_RATE = Decimal("0.5")
DEPTH = Decimal(200)
LIQUIDITY = DEPTH * MID

OWNER = "owner-parity"
CHECKPOINT_MS = T1
NOW_MS = T1 + GRID_MS
SIZE = Decimal(3)
MARK = Decimal(9999)
K = "0.025"

MARKETS = {
    MARKET: {
        "symbol": MARKET,
        "instrument_type": "perp",
        "verified": True,
        "source_mode": "SYNTHETIC",
        "valid_until_ms": NOW_MS + 10_000_000,
        "qty_step": "0.00001",
        "price_tick": "0.00000001",
        "min_qty": "0.00001",
        "min_notional": "10",
        "liquidity_notional": str(LIQUIDITY),
        "initial_rate": str(INITIAL_RATE),
        "fee_rate": "0",
        "funding_bound": "0",
        "liquidation_rate": "0",
        "tiers": [{"cap": str(LIQUIDITY), "rate": str(INITIAL_RATE)}],
    }
}
POLICY = {
    "schema": "risk_policy_v1",
    "source_mode": "SYNTHETIC",
    "scale_steps": 4,
    "scenarios": [{MARKET: "-0.10"}],
    "minimum_headroom": "50",
    "made_limit": "1000",
    "pod_loss_budget": "400",
    "hedge_tolerance": "0.05",
    "shortfall_bps": "30",
    "margin_multiplier": "1",
    "checkpoint_execution": {
        "quote_max_age_ms": 5000,
        "account_max_age_ms": 5000,
        "max_gross_notional": "20000",
    },
}
QUOTES = {MARKET: {"bid": str(BID), "ask": str(ASK), "known_at_ms": NOW_MS - 1000}}
FINANCE = {"equity": "10000", "headroom": "400", "realized_made": "0", "pod_loss_used": {}}


def _stamp(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat()


def _fill(at_ms: int, side: str, size: float, price: float, trade_id: int) -> dict[str, Any]:
    return {
        "wallet": WALLET_A,
        "exchange_ts": datetime.fromtimestamp(at_ms / 1000, UTC),
        "coin": MARKET,
        "dex": "hyperliquid",
        "side": side,
        "size": size,
        "price": price,
        "start_position": 0.0,
        "trade_id": trade_id,
    }


def _ccxt_btc() -> dict[str, Any]:
    return {
        "symbol": "BTC/USDC:USDC",
        "id": "0",
        "base": "BTC",
        "quote": "USDC",
        "settle": "USDC",
        "swap": True,
        "spot": False,
        "linear": True,
        "contract": True,
        "type": "swap",
        "active": True,
        "precision": {"amount": 0.00001, "price": 1.0},
        "limits": {"cost": {"min": 10}},
        "info": {
            "name": "BTC",
            "dex": None,
            "hip3": False,
            "baseId": 0,
            "szDecimals": 5,
        },
    }


def _frame() -> CheckpointBatch:
    return CheckpointBatch(
        checkpoint_ms=CheckpointMs(CHECKPOINT_MS),
        outcomes=(
            SourceOutcome(
                owner=OWNER,
                sizing=FixedKSizing(Decimal(K)),
                published_at=KnownAtTs(CHECKPOINT_MS),
                deadline=DecisionMs(NOW_MS),
                legs=(
                    SourceLeg(
                        instrument=NativeInstrument(BTC),
                        market=MARKET,
                        quantity=SIZE,
                        mark=MARK,
                    ),
                ),
                provenance=ProspectiveCWSA(
                    source_cursor=1, chain_sha="a" * 64, packet_sha="b" * 64
                ),
            ),
        ),
    )


def _state() -> KernelState:
    return KernelState(owner_books=(OwnerBook(OWNER, (), ()),))


def _research_inputs() -> DecisionInputs:
    return DecisionInputs(
        account=AccountState(
            finance=FrozenObject.parse(FINANCE),
            positions=FrozenObject.parse({}),
            owner_positions=FrozenObject.parse({}),
            ledger_basis=FrozenObject.parse({}),
            observed_basis=FrozenObject.parse({}),
        ),
        market=MarketContext(
            quotes=FrozenObject.parse(QUOTES),
            markets=FrozenObject.parse(MARKETS),
            capabilities=(VenueCapability(MARKET, True, True),),
        ),
        policy=RiskPolicy(FrozenObject.parse(POLICY)),
        now_ms=DecisionMs(NOW_MS),
    )


def test_all_three_shells_call_the_one_kernel_function() -> None:
    import copytrade_pm.pm_checkpoint_paper as paper
    import prop_production.checkpoint_admission as admission

    assert research_backtest.step_checkpoint is step_checkpoint
    assert paper.step_checkpoint is step_checkpoint
    assert admission.step_checkpoint is step_checkpoint


def test_the_digest_changes_when_the_decision_changes() -> None:
    state = _state()
    baseline_step = step_checkpoint(state=state, frame=_frame(), inputs=_research_inputs())
    baseline = decision_sha(checkpoint_ms=CHECKPOINT_MS, state=state, step=baseline_step)

    large = CheckpointBatch(
        checkpoint_ms=CheckpointMs(CHECKPOINT_MS),
        outcomes=(
            SourceOutcome(
                owner=OWNER,
                sizing=FixedKSizing(Decimal(K)),
                published_at=KnownAtTs(CHECKPOINT_MS),
                deadline=DecisionMs(NOW_MS),
                legs=(
                    SourceLeg(
                        instrument=NativeInstrument(BTC),
                        market=MARKET,
                        quantity=SIZE * 20,
                        mark=MARK,
                    ),
                ),
                provenance=ProspectiveCWSA(
                    source_cursor=1, chain_sha="a" * 64, packet_sha="b" * 64
                ),
            ),
        ),
    )
    other_step = step_checkpoint(state=state, frame=large, inputs=_research_inputs())
    other = decision_sha(checkpoint_ms=CHECKPOINT_MS, state=state, step=other_step)
    assert baseline != other
    assert net_intent_rows(baseline_step.execution.net_orders) != net_intent_rows(
        other_step.execution.net_orders
    )


def test_golden_market_map_rejects_non_bijective_collision() -> None:
    state = _state()
    step = step_checkpoint(state=state, frame=_frame(), inputs=_research_inputs())
    with pytest.raises(ValueError, match="GOLDEN_MARKET_MAP_COLLISION"):
        decision_sha(
            checkpoint_ms=CHECKPOINT_MS,
            state=state,
            step=step,
            market_map={"BTC": BTC, "WBTC": BTC},
        )


def test_three_engines_end_to_end_golden_parity(tmp_path: Path, monkeypatch) -> None:
    """True end-to-end 3-shell parity over t0 (0) -> t1 (+3 BTC) -> t2 (+1 BTC)."""
    import copytrade_pm.pm_checkpoint_paper as paper
    import prop_production.checkpoint_admission as admission
    from prop_production.checkpoint_receiver import CheckpointSourceReceiver
    from prop_production.copy_mandates import CopyMandate
    from prop_production.evidence import digest as prod_digest
    from prop_production.evidence import sign as prod_sign
    from prop_production.finance import Rulebook as GuardianRulebook
    from prop_production.guardian import Guardian
    from prop_production.journal import Journal
    from prop_production.judge.broker import FakeBroker
    from prop_production.release import ReleaseFloor, ReleaseVerifier, fixture_release

    # ------------------------------------------------------------------ 1. Backtest
    # Build 2 days of qualifying lookback tape (matching lookback_days=2) that
    # flattens cleanly to 0 before T0, then drives T0=0, T1=+3 BTC, T2=+1 BTC
    # at mid=10000 (bid=9999, ask=10001).
    lookback_start = T0 - 2 * DAY_MS
    # 0.078125 (=5/64) and 0.078125*0.8=0.0625 (=1/16) are exact IEEE-754 dyadic
    # floats, so Polars Float64 cum_sum reaches exactly 0 before T0, 3 at T1, 1 at T2.
    lookback = wallet_round_trips(
        WALLET_A,
        MARKET,
        start_ms=lookback_start,
        bars=2 * 288 - 1,
        base_price=10000.0,
        size=0.078125,
    )
    net_lookback = sum(
        Decimal(str(row["size"])) * (1 if row["side"] == "buy" else -1) for row in lookback
    )
    trade_id = len(lookback) + 1
    flatten_bar = T0 - GRID_MS
    fills = list(lookback)
    if net_lookback != 0:
        fills.append(
            _fill(
                flatten_bar,
                "sell" if net_lookback > 0 else "buy",
                float(abs(net_lookback)),
                10000.0,
                trade_id,
            )
        )
        trade_id += 1
    # T0: flat (0 qty) with a price observation at 10000.
    fills.append(_fill(T0, "buy", 0.125, 10000.0, trade_id))
    fills.append(_fill(T0 + 1000, "sell", 0.125, 10000.0, trade_id + 1))
    # T1: open +3 BTC.
    fills.append(_fill(T1, "buy", 3.0, 10000.0, trade_id + 2))
    # T2: partial reduce to +1 BTC.
    fills.append(_fill(T2, "sell", 2.0, 10000.0, trade_id + 3))

    store = build_store(
        tmp_path / "backtest",
        fills=fills,
        equity={WALLET_A: (Decimal(12000), 5)},
        start_ms=lookback_start,
    )
    bt_captured: list[tuple[KernelState, CheckpointBatch, DecisionInputs, Any]] = []
    real_step = research_backtest.step_checkpoint

    def bt_spy(*, state, frame, inputs=None):
        step = real_step(state=state, frame=frame, inputs=inputs)
        bt_captured.append((state, frame, inputs, step))
        return step

    monkeypatch.setattr(research_backtest, "step_checkpoint", bt_spy)

    bt_rulebook = KernelRulebook(
        account_binding=ACCOUNT_ID,
        status="SYNTHETIC",
        initial="10000",
        target="1000",
        daily_loss="500",
        trailing_loss="1000",
        internal_daily_stop="400",
        daily_basis="MAX_BALANCE_EQUITY",
        trailing_basis="INTRADAY_EQUITY",
        timezone="UTC",
        valid_until_ms=10**15,
        evidence_sha="d" * 64,
    )
    bt_market = MarketSpec(
        symbol=MARKET,
        qty_step=Decimal("0.00001"),
        price_tick=Decimal("0.00000001"),
        min_qty=Decimal("0.00001"),
        min_notional=Decimal(10),
        liquidity_notional=LIQUIDITY,
        initial_rate=INITIAL_RATE,
        fee_rate=Decimal(0),
        funding_bound=Decimal(0),
        liquidation_rate=Decimal(0),
        maintenance_rate=INITIAL_RATE,
    )
    bt_config = BacktestConfig(
        account_id=ACCOUNT_ID,
        initial_equity=Decimal(10000),
        lookback_days=2,
        shortfall_bps=SHORTFALL_BPS,
        scale_steps=4,
        made_limit=Decimal(1000),
        pod_loss_budget=Decimal(400),
        hedge_tolerance=Decimal("0.05"),
        minimum_headroom=Decimal(50),
        max_gross_notional=Decimal(20000),
        quote_spread_bps=Decimal(2),
        scenario_shock=Decimal("-0.10"),
        score_config=ScoreConfig(
            min_baseline_order_usd=Decimal(1),
            max_fixed_k=Decimal("0.025"),
            max_euler_cdar_share=Decimal(1),
            max_p95_participation_1m=Decimal(1),
            max_max_participation_1m=Decimal(1),
        ),
        allow_undated=True,
    )
    try:
        bt_report = run_kernel_backtest(
            store,
            window=BacktestWindow(T0, T2 + GRID_MS),
            seats=(SeatSpec("seat_1", Decimal(200), Decimal("1.0")),),
            rulebook=bt_rulebook,
            markets=(bt_market,),
            config=bt_config,
        )
    finally:
        store.close()

    assert len(bt_captured) == 3
    bt_t1_state, bt_t1_frame, _, bt_t1_step = bt_captured[1]
    bt_t2_state, bt_t2_frame, _, bt_t2_step = bt_captured[2]
    assert len(bt_t1_frame.outcomes) == 1
    assert bt_t1_frame.outcomes[0].legs[0].quantity == Decimal(3)
    assert bt_t2_frame.outcomes[0].legs[0].quantity == Decimal(1)
    shared_owner = bt_t1_frame.outcomes[0].owner
    shared_k = bt_t1_frame.outcomes[0].sizing.k
    assert shared_k > 0

    bt_t1_sha = decision_sha(
        checkpoint_ms=T1, state=bt_t1_state, step=bt_t1_step, market_map=MARKET_MAP
    )
    bt_t2_sha = decision_sha(
        checkpoint_ms=T2, state=bt_t2_state, step=bt_t2_step, market_map=MARKET_MAP
    )
    bt_t1_intents = net_intent_rows(bt_t1_step.execution.net_orders, market_map=MARKET_MAP)
    bt_t2_intents = net_intent_rows(bt_t2_step.execution.net_orders, market_map=MARKET_MAP)
    bt_ending = tuple(
        sorted((owner, MARKET_MAP[mkt], qty) for owner, mkt, qty in bt_report.ending_positions)
    )

    # ------------------------------------------------------------------ 2. PM03 Paper
    pm_captured: list[tuple[KernelState, CheckpointBatch, DecisionInputs, Any]] = []

    def pm_spy(*, state, frame, inputs=None):
        step = real_step(state=state, frame=frame, inputs=inputs)
        pm_captured.append((state, frame, inputs, step))
        return step

    monkeypatch.setattr(paper, "step_checkpoint", pm_spy)

    pm_journal = PaperJournal(tmp_path / "paper")
    cwsa_limits = SamplingLimits(
        max_state_age_ms=4000,
        max_acquisition_ms=3000,
        max_dex_skew_ms=2000,
        publication_grace_ms=5000,
    )
    cwsa_scope = CheckpointScope(
        source_id="fixture-leader",
        account_sha=sha256(WALLET_A.encode()).hexdigest(),
        generation="a" * 64,
        selection_sha="b" * 64,
        dexes=("default", "xyz"),
    )
    cwsa_policy = CwsaPolicy(limits=cwsa_limits, measurement_sha="c" * 64)
    pm_journal.register_checkpoint_source(
        cwsa_scope, cwsa_policy, first_checkpoint_ms=T0, at_ms=T0 - 1
    )

    pm_policy = {
        "schema": "checkpoint_paper_policy_v3",
        "max_quote_age_ms": 2000,
        "pod_loss_limit": "400",
        "max_gross_notional": "20000",
        "risk": {k: str(v) for k, v in asdict(PaperRiskPolicy()).items()},
        "shocks": [{BTC: "-0.10"}],
        "costs": {
            "fee_bps": "0",
            "full_spread_bps": "0",
            "slippage_bps": str(SHORTFALL_BPS),
            "impact_bps_at_full_participation": "0",
            "max_participation": "0.1",
            "max_age_seconds": 2,
            "latency_seconds": 0,
        },
    }
    pm_rulebook = replace(
        propr_rulebook("classic_1step", "evaluation", Decimal(10000), observed_at=_stamp(T0 - 1)),
        daily_loss_rate=Decimal("0.04"),
        trailing_loss=Decimal(1000),
        static_loss=None,
        permitted=(BTC_INST,),
        confirmed=True,
    )
    pm_manifest = PaperManifest(
        run_id="parity-run",
        account_id=ACCOUNT_ID,
        portfolio_sha="a" * 64,
        market_manifest_sha="b" * 64,
        release_sha="c" * 64,
        policy_sha=pm_digest(pm_policy),
        selected_at=_stamp(T0 - 1),
        started_at=_stamp(T0 - 1),
        rulebook=pm_rulebook,
    )
    pm_journal.register(pm_manifest)

    authority_key = Ed25519PrivateKey.generate()
    source_key = Ed25519PrivateKey.generate()
    mandate_id = f"{ACCOUNT_ID}:seat_1:g1"
    pm_mandate = SignedMandate.from_fields(
        mandate_id=mandate_id,
        generation=1,
        source_id=cwsa_scope.source_id,
        source_scope_generation=cwsa_scope.generation,
        source_account_sha=cwsa_scope.account_sha,
        account_id=ACCOUNT_ID,
        pod_id="seat_1",
        k=str(shared_k),
        allowed_markets=(BTC,),
        policy_epoch=cwsa_policy.epoch,
        risk_policy_sha=pm_digest(pm_policy),
        rulebook_sha=pm_rulebook.sha,
        effective_from_ms=T0,
        effective_until_ms=T2 + DAY_MS,
    )
    assert pm_mandate.owner == shared_owner
    pm_journal.activate_copy_mandate(
        pm_sign(pm_mandate.data, authority_key, "copy-mandate-v1"),
        authority_key.public_key(),
        account_id=ACCOUNT_ID,
        at_ms=T0 - 1,
    )

    def pm_add_quote(at_ms: int) -> None:
        raw = _ccxt_btc()
        native = parse_ccxt_market(raw, first_known_at=_stamp(T0 - 1000), expected=BTC_INST)
        row = parse_ccxt_book(
            native,
            {
                "symbol": raw["symbol"],
                "timestamp": at_ms,
                "bids": [[float(BID), float(DEPTH)]],
                "asks": [[float(ASK), float(DEPTH)]],
            },
            first_known_at=_stamp(at_ms + 1),
            max_age_seconds=2,
        )
        pm_journal.append_market_quote(row.payload())

    out_t0 = pm_journal.append_checkpoint_sample(
        cwsa_policy.epoch,
        cwsa_sample(
            evidence_root=pm_journal.root,
            scope=cwsa_scope,
            limits=cwsa_limits,
            wallet=WALLET_A,
            coin=MARKET,
            mid=BID,
            boundary=T0,
            qty="0",
        ),
        T0 + 101,
    )
    pkt_t0 = pm_canonical(
        export_checkpoint_packet(
            pm_journal.path,
            cwsa_scope,
            cwsa_policy,
            signer=source_key,
            after_cursor=0,
            issued_ms=T0 + 102,
        )
    )
    assert (
        pm_journal.consume_checkpoint(
            "parity-run",
            shared_owner,
            out_t0.outcome_id,
            at_ms=T0 + 102,
            policy=pm_policy,
            catalog={"BTC/USDC:USDC": _ccxt_btc()},
        )["status"]
        == "BASELINE"
    )

    out_t1 = pm_journal.append_checkpoint_sample(
        cwsa_policy.epoch,
        cwsa_sample(
            evidence_root=pm_journal.root,
            scope=cwsa_scope,
            limits=cwsa_limits,
            wallet=WALLET_A,
            coin=MARKET,
            mid=BID,
            boundary=T1,
            qty="3",
        ),
        T1 + 101,
    )
    pkt_t1 = pm_canonical(
        export_checkpoint_packet(
            pm_journal.path,
            cwsa_scope,
            cwsa_policy,
            signer=source_key,
            after_cursor=2,
            issued_ms=T1 + 102,
        )
    )
    pm_add_quote(T1 + 103)
    res_t1 = pm_journal.consume_checkpoint(
        "parity-run",
        shared_owner,
        out_t1.outcome_id,
        at_ms=T1 + 105,
        policy=pm_policy,
        catalog={"BTC/USDC:USDC": _ccxt_btc()},
    )
    assert res_t1["status"] == "COMMITTED_ONCE"

    out_t2 = pm_journal.append_checkpoint_sample(
        cwsa_policy.epoch,
        cwsa_sample(
            evidence_root=pm_journal.root,
            scope=cwsa_scope,
            limits=cwsa_limits,
            wallet=WALLET_A,
            coin=MARKET,
            mid=BID,
            boundary=T2,
            qty="1",
        ),
        T2 + 101,
    )
    pkt_t2 = pm_canonical(
        export_checkpoint_packet(
            pm_journal.path,
            cwsa_scope,
            cwsa_policy,
            signer=source_key,
            after_cursor=3,
            issued_ms=T2 + 102,
        )
    )
    pm_add_quote(T2 + 103)
    res_t2 = pm_journal.consume_checkpoint(
        "parity-run",
        shared_owner,
        out_t2.outcome_id,
        at_ms=T2 + 105,
        policy=pm_policy,
        catalog={"BTC/USDC:USDC": _ccxt_btc()},
    )
    assert res_t2["status"] == "COMMITTED_ONCE"

    pm_by_cp: dict[int, tuple[KernelState, Any]] = {}
    for state, frame, _, step in pm_captured:
        cp = frame.checkpoint_ms.value
        if cp in pm_by_cp:
            prev_state, prev_step = pm_by_cp[cp]
            assert decision_sha(checkpoint_ms=cp, state=state, step=step) == decision_sha(
                checkpoint_ms=cp, state=prev_state, step=prev_step
            )
        else:
            pm_by_cp[cp] = (state, step)
    assert set(pm_by_cp) == {T1, T2}
    pm_t1_state, pm_t1_step = pm_by_cp[T1]
    pm_t2_state, pm_t2_step = pm_by_cp[T2]
    pm_t1_sha = decision_sha(checkpoint_ms=T1, state=pm_t1_state, step=pm_t1_step)
    pm_t2_sha = decision_sha(checkpoint_ms=T2, state=pm_t2_state, step=pm_t2_step)
    pm_t1_intents = net_intent_rows(pm_t1_step.execution.net_orders)
    pm_t2_intents = net_intent_rows(pm_t2_step.execution.net_orders)

    reopened_pm = PaperJournal(pm_journal.root)
    pm_account = reopened_pm.restore("parity-run")["account"]
    pm_ending = tuple(
        sorted(
            (owner, inst, qty)
            for owner, book in pm_account["pm_quantities"].items()
            for inst, qty in book.items()
            if Decimal(qty) != 0
        )
    )

    # ------------------------------------------------------------------ 3. Guardian
    g_captured: list[tuple[KernelState, CheckpointBatch, DecisionInputs, Any]] = []

    def g_spy(*, state, frame, inputs=None):
        step = real_step(state=state, frame=frame, inputs=inputs)
        if inputs is not None:
            g_captured.append((state, frame, inputs, step))
        return step

    monkeypatch.setattr(admission, "step_checkpoint", g_spy)

    g_rules = GuardianRulebook(
        account_binding=ACCOUNT_ID,
        status="SYNTHETIC",
        initial="10000",
        target="1000",
        daily_loss="500",
        trailing_loss="1000",
        internal_daily_stop="400",
        daily_basis="MAX_BALANCE_EQUITY",
        trailing_basis="INTRADAY_EQUITY",
        timezone="UTC",
        valid_until_ms=10**15,
        evidence_sha="d" * 64,
    )
    g_policy = {
        "schema": "risk_policy_v1",
        "source_mode": "SYNTHETIC",
        "scale_steps": 4,
        "scenarios": [{MARKET: "-0.10"}],
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
        MARKET: {
            "symbol": MARKET,
            "instrument_type": "perp",
            "verified": True,
            "source_mode": "SYNTHETIC",
            "valid_until_ms": 10**15,
            "qty_step": "0.00001",
            "price_tick": "0.00000001",
            "min_qty": "0.00001",
            "min_notional": "10",
            "liquidity_notional": str(LIQUIDITY),
            "initial_rate": str(INITIAL_RATE),
            "fee_rate": "0",
            "funding_bound": "0",
            "liquidation_rate": "0",
            "tiers": [{"cap": str(LIQUIDITY), "rate": str(INITIAL_RATE)}],
            "source_instrument": BTC,
        }
    }
    risk_policy_sha = prod_digest({"policy": g_policy, "markets": g_markets})
    g_mandate_body = SignedMandate.from_fields(
        mandate_id=mandate_id,
        generation=1,
        source_id=cwsa_scope.source_id,
        source_scope_generation=cwsa_scope.generation,
        source_account_sha=cwsa_scope.account_sha,
        account_id=ACCOUNT_ID,
        pod_id="seat_1",
        k=str(shared_k),
        allowed_markets=(BTC,),
        policy_epoch=cwsa_policy.epoch,
        risk_policy_sha=risk_policy_sha,
        rulebook_sha=g_rules.sha,
        effective_from_ms=T0,
        effective_until_ms=T2 + DAY_MS,
    ).data
    g_members = [
        {
            "pod_id": "seat_1",
            "source_identity": cwsa_scope.source_id,
            "identity_known_at_ms": T0 - 1002,
            "account_scope": [cwsa_scope.source_id],
            "allocation": "2000",
            "sizing_policy": "FIXED_K_NATIVE_QUANTITY",
            "k": str(shared_k),
            "allowed_markets": [BTC],
        }
    ]
    release_payload = fixture_release()
    release_payload.update(
        schema="portfolio_release_v2",
        created_at_ms=T0 - 1000,
        selection_known_at_ms=T0 - 1001,
        expires_at_ms=T2 + DAY_MS,
        risk_policy_sha=risk_policy_sha,
        rulebook_sha=g_rules.sha,
        members=g_members,
        mandates=[g_mandate_body],
        checkpoint_policy=cwsa_policy.model_dump(mode="json"),
        checkpoint_scopes={cwsa_scope.source_id: cwsa_scope.model_dump(mode="json")},
        mandate_authority_sha=sha256(
            authority_key.public_key().public_bytes_raw()
        ).hexdigest(),
    )
    release_payload["definition_sha"] = prod_digest(g_members)
    release_payload["source_sha"] = prod_digest(
        {
            k: release_payload[k]
            for k in (
                "research_commit",
                "decision_sha",
                "inventory_sha",
                "definition_sha",
                "selection_known_at_ms",
            )
        }
    )
    release = ReleaseVerifier(
        authority_key.public_key(), "approver", ACCOUNT_ID, risk_policy_sha, g_rules.sha
    ).verify(prod_sign(release_payload, authority_key, "approver"), now_ms=T0 - 10)

    g_root = tmp_path / "guardian"
    g_root.mkdir(mode=0o700)
    floor = ReleaseFloor(g_root / "floor.sqlite3", initialize=True)
    floor.accept(release)
    g_journal = Journal(g_root / "guardian.sqlite3", ACCOUNT_ID, initialize=True)
    broker = FakeBroker(g_root / "broker.sqlite3")
    broker.fill_on_submit = False
    broker.fee_rate = FILL_FEE_RATE
    receiver = CheckpointSourceReceiver(
        g_root / "source.sqlite3",
        release=release,
        source_key=source_key.public_key(),
        wire_version=3,
    )
    try:
        guardian = Guardian(
            g_journal,
            release,
            floor,
            source_key.public_key(),
            g_rules,
            g_policy,
            g_markets,
            broker,
            mandate_key=authority_key.public_key(),
        )
        g_mandate = CopyMandate.parse(g_mandate_body)
        guardian.activate_copy_mandate(
            prod_sign(g_mandate.data, authority_key, "copy-mandate-v1"), now_ms=T0 - 1
        )
        receiver.ingest(pkt_t0, now_ms=T0 + 102)
        guardian.baseline_copy_mandate(shared_owner, receiver, now_ms=T0 + 102)

        def guardian_step_wave(pkt: bytes, outcome_id: str, wave_ms: int, fill_price: str) -> None:
            receiver.ingest(pkt, now_ms=wave_ms)
            guardian.reconcile(wave_ms)
            quotes = {MARKET: {"bid": str(BID), "ask": str(ASK), "known_at_ms": wave_ms}}
            decision = guardian.process_checkpoint(
                receiver, {shared_owner: outcome_id}, quotes=quotes, now_ms=DecisionMs(wave_ms)
            )
            assert decision["status"] == "REVIEWABLE"
            intent_ids = guardian.reserve_checkpoint(
                decision["decision_id"], receiver=receiver, now_ms=wave_ms + 1
            )
            assert len(intent_ids) == 1
            guardian.dispatch(intent_ids[0], wave_ms + 2)
            broker.fill(intent_ids[0], "1", fill_price)
            guardian.reconcile(wave_ms + 3)

        guardian_step_wave(pkt_t1, out_t1.outcome_id, T1 + 102, str(ASK))
        guardian_step_wave(pkt_t2, out_t2.outcome_id, T2 + 102, str(BID))
    finally:
        receiver.close()
        g_journal.close()
        broker.close()
        floor.close()

    g_by_cp: dict[int, tuple[KernelState, Any]] = {}
    for state, frame, _, step in g_captured:
        cp = frame.checkpoint_ms.value
        if cp in g_by_cp:
            prev_state, prev_step = g_by_cp[cp]
            assert decision_sha(
                checkpoint_ms=cp, state=state, step=step, market_map=MARKET_MAP
            ) == decision_sha(
                checkpoint_ms=cp, state=prev_state, step=prev_step, market_map=MARKET_MAP
            )
        else:
            g_by_cp[cp] = (state, step)
    assert set(g_by_cp) == {T1, T2}
    g_t1_state, g_t1_step = g_by_cp[T1]
    g_t2_state, g_t2_step = g_by_cp[T2]
    g_t1_sha = decision_sha(
        checkpoint_ms=T1, state=g_t1_state, step=g_t1_step, market_map=MARKET_MAP
    )
    g_t2_sha = decision_sha(
        checkpoint_ms=T2, state=g_t2_state, step=g_t2_step, market_map=MARKET_MAP
    )
    g_t1_intents = net_intent_rows(g_t1_step.execution.net_orders, market_map=MARKET_MAP)
    g_t2_intents = net_intent_rows(g_t2_step.execution.net_orders, market_map=MARKET_MAP)

    reopened_g = Journal(g_root / "guardian.sqlite3", ACCOUNT_ID, initialize=False)
    try:
        g_snap = reopened_g.snapshot()
    finally:
        reopened_g.close()
    g_ending = tuple(
        sorted(
            (owner, MARKET_MAP[mkt], qty)
            for owner, book in g_snap["pod_positions"].items()
            for mkt, qty in book.items()
            if Decimal(qty) != 0
        )
    )

    # ------------------------------------------------------------------ 4. 3-way assertions
    assert (
        decision_payload(checkpoint_ms=T1, state=bt_t1_state, step=bt_t1_step, market_map=MARKET_MAP)
        == decision_payload(checkpoint_ms=T1, state=pm_t1_state, step=pm_t1_step)
        == decision_payload(
            checkpoint_ms=T1, state=g_t1_state, step=g_t1_step, market_map=MARKET_MAP
        )
    )
    assert (
        decision_payload(checkpoint_ms=T2, state=bt_t2_state, step=bt_t2_step, market_map=MARKET_MAP)
        == decision_payload(checkpoint_ms=T2, state=pm_t2_state, step=pm_t2_step)
        == decision_payload(
            checkpoint_ms=T2, state=g_t2_state, step=g_t2_step, market_map=MARKET_MAP
        )
    )
    assert bt_t1_sha == pm_t1_sha == g_t1_sha
    assert bt_t2_sha == pm_t2_sha == g_t2_sha
    assert bt_t1_sha != bt_t2_sha

    assert bt_t1_intents == pm_t1_intents == g_t1_intents
    assert bt_t2_intents == pm_t2_intents == g_t2_intents
    assert len(bt_t1_intents) == 1 and Decimal(bt_t1_intents[0]["net_delta_qty"]) > 0
    assert len(bt_t2_intents) == 1 and Decimal(bt_t2_intents[0]["net_delta_qty"]) < 0
    assert bt_t2_intents[0]["reduce_only"] is True

    assert bt_ending == pm_ending == g_ending
    assert len(bt_ending) == 1 and Decimal(bt_ending[0][2]) > 0
