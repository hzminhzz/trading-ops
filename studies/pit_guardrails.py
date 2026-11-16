"""Point-in-time proof gates for Hyperliquid-to-Bybit research.

This module distinguishes (1) genuine as-of evidence, (2) historical
reconstructions from today's public records, and (3) missing source receipts.
It does not create a second registry or simulate trading.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MS_DAY = 86_400_000
HISTORICAL_OOS_START = "2026-09-01T00:00:00+00:00"


def utc(value: str | datetime) -> datetime:
    dt = datetime.fromisoformat(value) if isinstance(value, str) else value
    if dt.tzinfo is None:
        raise ValueError("NAIVE_TIMESTAMP_NOT_PIT")
    return dt.astimezone(UTC)


def as_ms(value: str | datetime) -> int:
    return int(utc(value).timestamp() * 1000)


def _open_read_only(path: str | Path) -> sqlite3.Connection:
    target = Path(path).expanduser()
    if not target.is_file():
        raise FileNotFoundError(target)
    return sqlite3.connect(f"file:{target}?mode=ro", uri=True)


@dataclass(frozen=True)
class AsOfWallet:
    wallet_id: int
    alias: str
    identity_sha256: str
    monitoring_known_at: str
    alias_known_at: str | None


def registry_asof(path: str | Path, cutoff: str | datetime) -> tuple[list[AsOfWallet], str]:
    """Reconstruct only identities actually recorded by the cutoff.

    `wallets` alone has no registration clock; the monitoring known_at
    is the minimal existence witness. No fabricated historical first-seen
    date is inferred from fill timestamps or a later import.
    """
    when = utc(cutoff).isoformat()
    con = _open_read_only(path)
    try:
        if con.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("REGISTRY_INTEGRITY_FAILED")
        rows = con.execute(
            """
            SELECT w.wallet_id, lower(w.address), m.known_at,
                (SELECT wa.alias FROM wallet_aliases wa
                 WHERE wa.wallet_id = w.wallet_id AND wa.known_at <= ?
                 ORDER BY wa.known_at DESC, wa.rowid DESC LIMIT 1),
                (SELECT wa.known_at FROM wallet_aliases wa
                 WHERE wa.wallet_id = w.wallet_id AND wa.known_at <= ?
                 ORDER BY wa.known_at DESC, wa.rowid DESC LIMIT 1)
            FROM wallets w JOIN monitoring m USING(wallet_id)
            WHERE m.configured = 1 AND m.known_at <= ?
              AND NOT EXISTS (
                  SELECT 1 FROM exclusions e
                  WHERE e.wallet_id = w.wallet_id AND e.known_at <= ?
              )
            ORDER BY w.wallet_id
            """,
            (when, when, when, when),
        ).fetchall()
    finally:
        con.close()
    roster = [
        AsOfWallet(
            int(wallet_id),
            alias if alias else f"W#{wallet_id}",
            hashlib.sha256(address.encode()).hexdigest(),
            monitor_time,
            alias_time,
        )
        for wallet_id, address, monitor_time, alias, alias_time in rows
    ]
    canonical = "\n".join(
        f"{x.wallet_id}|{x.alias}|{x.identity_sha256}|"
        f"{x.monitoring_known_at}|{x.alias_known_at or ''}" for x in roster
    )
    return roster, hashlib.sha256(canonical.encode()).hexdigest()


def registry_first_receipt(path: str | Path) -> dict[str, Any]:
    """Earliest available known_at, never a guessed wallet discovery time."""
    con = _open_read_only(path)
    try:
        tables = ("monitoring", "monitoring_history", "wallet_aliases", "research_selections")
        fields = {
            "monitoring": "known_at",
            "monitoring_history": "known_at",
            "wallet_aliases": "known_at",
            "research_selections": "known_at",
        }
        data = {}
        for table in tables:
            count, first = con.execute(
                f"SELECT count(*), min({fields[table]}) FROM {table}"
            ).fetchone()
            data[table] = {"count": count, "earliest_known_at": first}
        return data
    finally:
        con.close()


def source_clock_census(path: str | Path, cutoff: str | datetime) -> dict[str, Any]:
    """Read original Chapter-2 receipt/publication clocks; no synthetic fill clock."""
    con = _open_read_only(path)
    try:
        count, earliest_received, earliest_published, before = con.execute(
            """
            SELECT count(*),
                   min(CAST(json_extract(outcome, '$.sample.received_ms') AS INTEGER)),
                   min(CAST(json_extract(outcome, '$.published_ms') AS INTEGER)),
                   sum(CASE WHEN CAST(json_extract(outcome, '$.sample.received_ms')
                            AS INTEGER) <= ? THEN 1 ELSE 0 END)
            FROM cwsa_outcomes
            """,
            (as_ms(cutoff),),
        ).fetchone()
    finally:
        con.close()
    return {
        "journal_outcomes": count,
        "earliest_received_ms": earliest_received,
        "earliest_published_ms": earliest_published,
        "outcomes_received_by_cutoff": before or 0,
        "pre_cutoff_receipt_proven": bool(before),
        "receipt_scope": "CHAPTER2_ONLY_NOT_GLOBAL_WALLET_COVERAGE",
    }


def validate_source_admission(
    *,
    source_event_ms: int,
    first_known_ms: int | None,
    decision_ms: int,
    execution_ms: int,
) -> str:
    """The source must be first-known before decision, and execution strictly later."""
    if first_known_ms is None:
        return "BLOCKED_MISSING_ORIGINAL_FIRST_KNOWN"
    if source_event_ms > first_known_ms:
        return "BLOCKED_SOURCE_CLOCK_ORDER"
    if first_known_ms > decision_ms:
        return "BLOCKED_NOT_KNOWN_AT_DECISION"
    if decision_ms >= execution_ms:
        return "BLOCKED_NOT_FUTURE_EXECUTION"
    if execution_ms < decision_ms + 60_000:
        return "BLOCKED_EXECUTION_EARLIER_THAN_1M_DELAY"
    return "CAUSAL_ADMISSION_ELIGIBLE"


@dataclass(frozen=True)
class BybitListing:
    symbol: str
    status_now: str
    launch_ms: int | None
    delivery_ms: int | None
    min_order_qty_now: str | None
    qty_step_now: str | None
    min_notional_now: str | None


def parse_bybit_linear_instrument(item: dict[str, Any]) -> BybitListing | None:
    if item.get("quoteCoin") != "USDT" or item.get("contractType") != "LinearPerpetual":
        return None
    symbol = item.get("symbol")
    if not isinstance(symbol, str) or not symbol.endswith("USDT"):
        return None

    def timestamp(name: str) -> int | None:
        value = item.get(name)
        try:
            n = int(value)
            return n if n > 0 else None
        except (TypeError, ValueError):
            return None

    lot = item.get("lotSizeFilter") or {}
    return BybitListing(
        symbol=symbol,
        status_now=str(item.get("status") or "UNKNOWN"),
        launch_ms=timestamp("launchTime"),
        delivery_ms=timestamp("deliveryTime"),
        min_order_qty_now=lot.get("minOrderQty"),
        qty_step_now=lot.get("qtyStep"),
        min_notional_now=lot.get("minNotionalValue"),
    )


def bybit_asof_status(
    instrument: BybitListing | None, at_ms: int, *,
    latest_completed_kline_open_ms: int | None = None,
    kline_interval_ms: int = 3_600_000,
) -> str:
    """Historical existence from listing bounds + *completed* exchange candle.

    Candle evidence is not equivalent to a contemporaneously archived
    instrument-status snapshot or to historical minimum-lot parameters.
    """
    if instrument is None:
        return "UNKNOWN_NOT_IN_RECOVERED_INSTRUMENT_CATALOG"
    if instrument.launch_ms is None:
        return "UNKNOWN_NO_LAUNCH_TIME"
    if at_ms < instrument.launch_ms:
        return "NOT_LISTED_AT_DECISION"
    if instrument.delivery_ms is not None and at_ms >= instrument.delivery_ms:
        return "CLOSED_AT_DECISION"
    if instrument.status_now == "Closed" and instrument.delivery_ms is None:
        return "UNKNOWN_CLOSED_WITHOUT_DELIST_TIME"
    if latest_completed_kline_open_ms is None:
        return "RECONSTRUCTED_LISTING_WINDOW_ONLY"
    if (
        latest_completed_kline_open_ms + kline_interval_ms > at_ms
        or latest_completed_kline_open_ms < instrument.launch_ms
    ):
        return "BLOCKED_CANDLE_LOOKAHEAD_OR_PRELAUNCH"
    if at_ms - (latest_completed_kline_open_ms + kline_interval_ms) > MS_DAY:
        return "UNKNOWN_STALE_MARKET_EVIDENCE"
    return "RECONSTRUCTED_HISTORICAL_TRADING_EVIDENCE"


def assert_prospective_holdout(
    *, frozen_at: str | datetime, start: str | datetime,
    end: str | datetime, now: str | datetime,
    evaluation_requested: bool = False,
) -> None:
    frozen, begin, finish, current = (utc(x) for x in (frozen_at, start, end, now))
    if not (frozen < begin < finish):
        raise ValueError("NOT_PROSPECTIVE_HOLDOUT")
    if evaluation_requested and current < finish:
        raise ValueError("HOLDOUT_UNMATURED_DO_NOT_READ")
    if not evaluation_requested and current >= begin:
        raise ValueError("CANNOT_FREEZE_STARTED_HOLDOUT")


def verify_forward_freeze(
    document: dict[str, Any], registry_path: str | Path, now: str | datetime, *,
    evaluation_requested: bool = False,
) -> dict[str, Any]:
    """Verify original freeze digest and wallet *identity*, not mutable alias.

    A changed alias is reported as drift but can never remap a pinned wallet
    to another account. No holdout data is accessed by this function.
    """
    if document.get("schema") != "cft_prospective_holdout_freeze_v1":
        raise ValueError("UNKNOWN_FREEZE_SCHEMA")
    expected_hash = document.get("freeze_payload_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("MISSING_FREEZE_DIGEST")
    original = {k: v for k, v in document.items() if k != "freeze_payload_sha256"}
    if sha256_json(original) != expected_hash:
        raise ValueError("FROZEN_PAYLOAD_TAMPERED")
    frozen = utc(document["created_at"])
    begin = utc(document["holdout_start"])
    end = utc(document["holdout_end_exclusive"])
    if not (frozen < begin < end):
        raise ValueError("NOT_PROSPECTIVE_HOLDOUT")
    if evaluation_requested:
        assert_prospective_holdout(
            frozen_at=frozen, start=begin, end=end, now=now,
            evaluation_requested=True,
        )

    con = _open_read_only(registry_path)
    try:
        current = {}
        for wallet_id, address in con.execute(
            "SELECT wallet_id, lower(address) FROM wallets"
        ):
            current[int(wallet_id)] = hashlib.sha256(address.encode()).hexdigest()
    finally:
        con.close()
    checked = 0
    for portfolio in document.get("portfolios", []):
        for member in portfolio.get("members", []):
            known_id = int(member["registry_id"])
            if known_id not in current:
                raise ValueError("FROZEN_WALLET_IDENTITY_DISAPPEARED")
            if current[known_id] != member["identity_sha256"]:
                raise ValueError("FROZEN_WALLET_ADDRESS_CHANGED")
            checked += 1
    if checked == 0:
        raise ValueError("EMPTY_FREEZE")
    return {
        "status": "FROZEN_IDENTITIES_VERIFIED",
        "member_slots": checked,
        "freeze_sha256": expected_hash,
        "holdout_ends_at": end.isoformat(),
        "evaluation_mature": utc(now) >= end,
        "original_source_receipts_verified": False,
        "read_holdout_data": False,
    }


def train_only_selection(rows: list[dict[str, Any]], top_n: int = 30) -> list[str]:
    """Train-only ranking: never gate on present-day account value or OOS."""
    eligible = [
        r for r in rows if float(r["net_usd"]) > 0.0
        and float(r["maxdd_usd"]) > 0.0
        and not r.get("historical_reasons", [])
    ]
    rank = sorted(
        eligible,
        key=lambda r: (-float(r["net_over_dd"]), -float(r["net_usd"]), str(r["alias"])),
    )
    return [r["alias"] for r in rank[:top_n]]


def sha256_json(doc: dict[str, Any]) -> str:
    blob = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()
