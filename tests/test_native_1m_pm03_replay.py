"""Regression checks for causal native CWSA -> local-1m -> PM03 financial replay."""

from __future__ import annotations

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "studies"))
SPEC = importlib.util.spec_from_file_location(
    "native_1m_pm03_replay", ROOT / "studies/native_1m_pm03_replay.py"
)
assert SPEC and SPEC.loader
REPLAY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPLAY
SPEC.loader.exec_module(REPLAY)


def outcome(member: str, checkpoint: int, kind: str, published: int):
    sample = SimpleNamespace(
        receipts=(SimpleNamespace(state_ms=checkpoint + 14_000),),
        limits=SimpleNamespace(max_state_age_ms=45_000),
    )
    return SimpleNamespace(
        source_id=member,
        checkpoint_ms=checkpoint,
        published_ms=published,
        kind=kind,
        signal_id="signal" if kind == "CHANGED" else None,
        sample=sample,
        positions=(),
    )


def test_native_publication_is_event_clock_not_utc_minute_open():
    cp = 1794460200000
    at = cp + 27_000
    assert REPLAY.first_native_decision(at) == at + 1
    assert at + 1 < cp + 59_000
    assert at + 1 < (cp // 60_000 + 1) * 60_000


def test_other_wallet_baseline_preceding_change_is_allowed_without_future_snapshot():
    cp = 1794460200000
    t1 = cp + 300_000
    books = {
        cp: {m: outcome(m, cp, "BASELINE", cp + 26_000 + i) for i, m in enumerate(("A", "B", "C"))},
        t1: {
            "A": outcome("A", t1, "CHANGED", t1 + 27_000),
            "B": outcome("B", t1, "UNCHANGED", t1 + 28_000),
            "C": outcome("C", t1, "UNCHANGED", t1 + 29_000),
        },
    }
    events, stats = REPLAY.build_events(books, ("A", "B", "C"))
    assert len(events) == 1
    assert stats["actionable_native_source_waves"] == 1
    assert events[0].at_ms == t1 + 27_001
    assert set(events[0].changes) == {"A"}


def test_missing_or_unavailable_other_wallet_blocks_native_new_risk():
    cp = 1794460200000
    t1 = cp + 300_000
    books = {
        cp: {m: outcome(m, cp, "BASELINE", cp + 26_000) for m in ("A", "B", "C")},
        t1: {
            "C": outcome("C", t1, "UNAVAILABLE", t1 + 24_000),
            "A": outcome("A", t1, "CHANGED", t1 + 27_000),
        },
    }
    events, stats = REPLAY.build_events(books, ("A", "B", "C"))
    assert not events
    assert stats["blocked_other_wallet_not_yet_verified"] == 1


def test_train_only_coverage_gate_blocks_hindsight_size_that_ignores_sources():
    row = {
        "status": "SCENARIO_ONLY",
        "sampled_minutes": 1000,
        "train_net_usd": 100,
        "worst_daily_loss_usd": 2,
        "peak_drawdown_usd": 4,
        "breach": False,
        "attempts": 2,
    }
    k, reason = REPLAY.choose_train_k({"0.001": row}, 10)
    assert k == Decimal(0)
    assert reason == "NO_POSITIVE_COMPLETE_TRAIN_CANDIDATE"
    k, reason = REPLAY.choose_train_k({"0.001": {**row, "attempts": 8}}, 10)
    assert k == Decimal("0.001")
    assert reason == "CHOSEN_ON_TRAIN_ONLY_EXPLORATORY"


def test_native_pm03_rulebook_has_static_not_trailing_equity_floor():
    from copytrade_pm.pm03_account import Account

    rule = REPLAY.rulebook(("A", "B", "C"), REPLAY.START)
    assert rule.trailing_loss is None
    account = Account.start(rule, REPLAY.START.isoformat())
    floor, parts = account.floor()
    assert floor == Decimal(9700)
    assert parts["static"] == Decimal(9500)
    assert parts["daily"] == Decimal(9700)
    account.equity = Decimal(9499)
    account._check_breach()
    assert account.breached
    assert account.halt_reason == "CONTRACTUAL_EQUITY_FLOOR"


def test_minimum_order_screen_blocks_idealized_dust_fills(monkeypatch):
    from copytrade_pm.pm03_account import Instrument

    start = REPLAY.mill(REPLAY.START)
    key = REPLAY.SOURCE_MARKET_PREFIX + "BTC"
    source = SimpleNamespace(positions=(SimpleNamespace(instrument=key, quantity=Decimal(1)),))
    wave = REPLAY.Wave(start, start + 25_000, {"A": source}, True, ())
    bar = (Decimal(100), Decimal(100), Decimal(100))
    bars = {"BTC": {start - 60_000: bar, start: bar, start + 60_000: bar, start + 120_000: bar}}
    monkeypatch.setattr(REPLAY, "get_marks", lambda _: {key: Decimal(100)})
    strict = REPLAY.simulate(("A",), [wave], bars, k=Decimal("0.0001"), stop_ms=start + 180_000)
    ideal = REPLAY.simulate(
        ("A",), [wave], bars, k=Decimal("0.0001"), min_order_notional=None, stop_ms=start + 180_000
    )
    assert strict["trades"] == 0
    assert strict["stats"]["blocked_below_min_order"] == 1
    assert ideal["trades"] == 1
    assert ideal["stats"]["sub_10_usd_fills"] == 1
    assert ideal["minimum_equity_usd"] <= 10000  # 4dp report rounds dust costs
    assert Instrument.parse(key).symbol == "BTC"


def test_replay_uses_native_account_and_execution_cost_owners():
    from copytrade_pm.pm03_account import Account
    from copytrade_pm.pm03_execution import CostModel

    assert isinstance(REPLAY.model_cost(Decimal(1)), CostModel)
    assert REPLAY.model_cost(Decimal(1)).max_age_seconds == 90
    assert isinstance(
        Account.start(REPLAY.rulebook(("A", "B", "C"), REPLAY.START), REPLAY.START.isoformat()),
        Account,
    )
