"""Adversarial PIT/first-known/holdout and Bybit listing evidence gates."""
from __future__ import annotations

import importlib
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies"))
g = importlib.import_module("pit_guardrails")
audit = importlib.import_module("cft_pit_evidence_audit")


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    p = tmp_path / "registry.sqlite"
    db = sqlite3.connect(p)
    db.executescript(
        """
        CREATE TABLE wallets(wallet_id INTEGER PRIMARY KEY, address TEXT);
        CREATE TABLE monitoring(wallet_id INTEGER PRIMARY KEY, configured INTEGER, known_at TEXT);
        CREATE TABLE monitoring_history(generation_sha TEXT, wallet_id INTEGER,
                configured INTEGER, known_at TEXT);
        CREATE TABLE wallet_aliases(wallet_id INTEGER, alias TEXT, known_at TEXT);
        CREATE TABLE research_selections(known_at TEXT);
        CREATE TABLE exclusions(wallet_id INTEGER, known_at TEXT);
        INSERT INTO wallets VALUES (1,'0x111'),(2,'0x222'),(3,'0x333');
        INSERT INTO monitoring VALUES
            (1,1,'2026-11-04T03:00:00+00:00'),
            (2,1,'2026-11-04T03:00:00+00:00'),
            (3,1,'2026-11-17T03:00:00+00:00');
        INSERT INTO wallet_aliases VALUES
            (1,'OLD','2026-11-04T03:00:00+00:00'),
            (1,'NEW','2026-11-15T03:00:00+00:00'),
            (2,'SECOND','2026-11-04T03:00:00+00:00'),
            (3,'FUTURE','2026-11-17T03:00:00+00:00');
        INSERT INTO exclusions VALUES (2,'2026-11-16T06:00:00+00:00');
        """
    )
    db.commit()
    db.close()
    return p


def test_registry_cannot_backdate_november_import_to_august(registry: Path) -> None:
    past, _ = g.registry_asof(registry, "2026-08-31T23:59:59Z")
    assert past == []
    sept, _ = g.registry_asof(registry, "2026-09-01T00:00:00Z")
    assert sept == []


def test_registry_asof_resolves_alias_only_at_known_time(registry: Path) -> None:
    before, _ = g.registry_asof(registry, "2026-11-05T00:00:00Z")
    assert {x.alias for x in before} == {"OLD", "SECOND"}
    after, sha = g.registry_asof(registry, "2026-11-16T07:00:00Z")
    assert [x.alias for x in after] == ["NEW"]
    assert sha != g.registry_asof(registry, "2026-11-05T00:00:00Z")[1]
    assert len(after[0].identity_sha256) == 64


def test_registry_no_forward_looking_exclusion(registry: Path) -> None:
    before, _ = g.registry_asof(registry, "2026-11-16T05:00:00Z")
    assert "SECOND" in {x.alias for x in before}
    after, _ = g.registry_asof(registry, "2026-11-16T07:00:00Z")
    assert "SECOND" not in {x.alias for x in after}


def test_current_account_value_does_not_drive_historical_selection() -> None:
    rows = [
        {"alias": "A", "net_usd": 900, "maxdd_usd": 100, "net_over_dd": 9.0,
         "operational_eligible": False},
        {"alias": "B", "net_usd": 500, "maxdd_usd": 200, "net_over_dd": 2.5,
         "operational_eligible": True},
        {"alias": "C", "net_usd": 50000, "maxdd_usd": 100, "net_over_dd": 500,
         "operational_eligible": True, "historical_reasons": ["NO_TRAIN_FILLS"]},
    ]
    assert g.train_only_selection(rows) == ["A", "B"]


def test_source_admission_checks_original_receipt_time() -> None:
    rule = g.validate_source_admission
    params = {"source_event_ms": 100_000, "decision_ms": 600_000,
              "execution_ms": 660_000}
    assert rule(**params, first_known_ms=None) == "BLOCKED_MISSING_ORIGINAL_FIRST_KNOWN"
    assert rule(**params, first_known_ms=50_000) == "BLOCKED_SOURCE_CLOCK_ORDER"
    assert rule(**params, first_known_ms=600_001) == "BLOCKED_NOT_KNOWN_AT_DECISION"
    assert rule(**{**params, "execution_ms": 600_000}, first_known_ms=200_000) == (
        "BLOCKED_NOT_FUTURE_EXECUTION"
    )
    assert rule(**{**params, "execution_ms": 650_000}, first_known_ms=200_000) == (
        "BLOCKED_EXECUTION_EARLIER_THAN_1M_DELAY"
    )
    assert rule(**params, first_known_ms=200_000) == "CAUSAL_ADMISSION_ELIGIBLE"


def _instrument(launch: int = 100_000, delivery: int | None = None, status: str = "Trading"):
    return g.BybitListing("BTCUSDT", status, launch, delivery, "0.001", "0.001", "5")


def test_bybit_rejects_future_launch_and_known_delisting() -> None:
    assert g.bybit_asof_status(_instrument(launch=900_000), 600_000) == "NOT_LISTED_AT_DECISION"
    assert g.bybit_asof_status(_instrument(delivery=400_000), 600_000) == "CLOSED_AT_DECISION"
    assert g.bybit_asof_status(_instrument(status="Closed"), 600_000) == (
        "UNKNOWN_CLOSED_WITHOUT_DELIST_TIME"
    )


def test_bybit_requires_complete_past_candle_not_future_current_candle() -> None:
    instrument = _instrument()
    at = 10 * 3_600_000
    assert g.bybit_asof_status(instrument, at) == "RECONSTRUCTED_LISTING_WINDOW_ONLY"
    assert g.bybit_asof_status(
        instrument, at, latest_completed_kline_open_ms=at - 3_600_000
    ) == "RECONSTRUCTED_HISTORICAL_TRADING_EVIDENCE"
    assert g.bybit_asof_status(
        instrument, at, latest_completed_kline_open_ms=at - 1_000
    ) == "BLOCKED_CANDLE_LOOKAHEAD_OR_PRELAUNCH"


def test_missing_historical_listing_remains_unknown() -> None:
    assert g.bybit_asof_status(None, 600_000).startswith("UNKNOWN")
    item = g.parse_bybit_linear_instrument({
        "symbol": "BTCUSDT", "quoteCoin": "USDT", "contractType": "LinearPerpetual",
        "status": "Trading", "launchTime": "1584230400000", "deliveryTime": "0",
        "lotSizeFilter": {"minOrderQty": "0.001", "qtyStep": "0.001", "minNotionalValue": "5"},
    })
    assert item and item.launch_ms == 1584230400000 and item.delivery_ms is None
    assert g.parse_bybit_linear_instrument({
        "symbol": "BTCUSDC", "quoteCoin": "USDC", "contractType": "LinearPerpetual",
    }) is None


def test_historical_oos_can_never_become_prospective() -> None:
    with pytest.raises(ValueError, match="NOT_PROSPECTIVE_HOLDOUT"):
        g.assert_prospective_holdout(
            frozen_at="2026-11-16T14:00:00Z",
            start="2026-09-01T00:00:00Z",
            end="2026-11-01T00:00:00Z",
            now="2026-11-16T14:00:00Z",
        )


def test_future_holdout_must_remain_unread_until_mature() -> None:
    dates = {
        "frozen_at": "2026-11-16T14:00:00Z",
        "start": "2026-11-17T00:05:00Z",
        "end": "2026-12-17T00:05:00Z",
    }
    g.assert_prospective_holdout(**dates, now="2026-11-16T14:00:00Z")
    with pytest.raises(ValueError, match="HOLDOUT_UNMATURED_DO_NOT_READ"):
        g.assert_prospective_holdout(
            **dates, now="2026-11-18T00:05:00Z", evaluation_requested=True
        )
    g.assert_prospective_holdout(
        **dates, now="2026-12-17T00:05:00Z", evaluation_requested=True
    )
    with pytest.raises(ValueError, match="CANNOT_FREEZE_STARTED_HOLDOUT"):
        g.assert_prospective_holdout(
            **dates, now="2026-11-18T00:05:00Z",
        )


def test_forward_freeze_is_exact_and_member_scoped(registry: Path, tmp_path: Path) -> None:
    roster, sha = g.registry_asof(registry, "2026-11-15T04:00:00Z")
    source = tmp_path / "prior.json"
    source.write_text('{"version":1}')
    prior = {"frozen_portfolios": [{
        "aliases": ["NEW", "SECOND"], "train_opt_k": [0.1, 0.2],
    }]}
    frozen = audit._make_forward_freeze(
        prior, source, roster, sha, datetime(2026, 11, 15, 4, tzinfo=UTC),
        {"public_metadata_sha256": "f" * 64},
    )
    assert frozen["holdout_start"] == "2026-11-16T00:05:00+00:00"
    assert frozen["holdout_end_exclusive"] == "2026-12-16T00:05:00+00:00"
    assert frozen["portfolios"][0]["members"][0]["frozen_k"] == 0.1
    payload = {k: v for k, v in frozen.items() if k != "freeze_payload_sha256"}
    assert frozen["freeze_payload_sha256"] == g.sha256_json(payload)
    with pytest.raises(ValueError, match="PROSPECTIVE_MEMBER_NOT_IN_ASOF_REGISTRY"):
        audit._make_forward_freeze(
            {"frozen_portfolios": [{"aliases": ["UNREGISTERED"], "train_opt_k": [0.1]}]},
            source, roster, sha, datetime(2026, 11, 15, 4, tzinfo=UTC), {}
        )


def test_freeze_integrity_survives_alias_change_but_not_wallet_identity_change(
    registry: Path, tmp_path: Path,
) -> None:
    roster, sha = g.registry_asof(registry, "2026-11-15T04:00:00Z")
    path = tmp_path / "before.json"
    path.write_text("{}")
    freeze = audit._make_forward_freeze(
        {"frozen_portfolios": [{"aliases": ["NEW"], "train_opt_k": [0.1]}]},
        path, roster, sha, datetime(2026, 11, 15, 4, tzinfo=UTC), {},
    )
    check = g.verify_forward_freeze(
        freeze, registry, "2026-11-15T04:05:00Z",
    )
    assert check["status"] == "FROZEN_IDENTITIES_VERIFIED"
    assert check["read_holdout_data"] is False
    assert check["original_source_receipts_verified"] is False

    altered = dict(freeze)
    altered["holdout_end_exclusive"] = "2026-11-17T00:05:00+00:00"
    with pytest.raises(ValueError, match="FROZEN_PAYLOAD_TAMPERED"):
        g.verify_forward_freeze(altered, registry, "2026-11-15T04:05:00Z")

    con = sqlite3.connect(registry)
    con.execute("UPDATE wallets SET address='0xDIFFERENT' WHERE wallet_id=1")
    con.commit()
    con.close()
    with pytest.raises(ValueError, match="FROZEN_WALLET_ADDRESS_CHANGED"):
        g.verify_forward_freeze(freeze, registry, "2026-11-15T04:05:00Z")


def test_both_registry_and_source_census_fail_closed_when_missing(registry: Path, tmp_path: Path) -> None:
    census = g.registry_first_receipt(registry)
    assert census["monitoring"]["earliest_known_at"].startswith("2026-11")
    p = tmp_path / "source.sqlite"
    con = sqlite3.connect(p)
    con.executescript("""CREATE TABLE cwsa_outcomes(outcome TEXT);
        INSERT INTO cwsa_outcomes VALUES
        ('{"sample":{"received_ms":1794460227021},"published_ms":1794460227025}');
    """)
    con.commit()
    con.close()
    result = g.source_clock_census(p, "2026-09-01T00:00:00Z")
    assert result["journal_outcomes"] == 1
    assert result["outcomes_received_by_cutoff"] == 0
    assert result["pre_cutoff_receipt_proven"] is False
