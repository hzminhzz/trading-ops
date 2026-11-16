"""Audit every local selected-wallet May-Aug fill against Bybit launch/delivery bounds.

This is a *retrospective venue-executability* audit. It does not turn
Reservoir exchange timestamps into source first-known clocks, nor does it
replace the Hyperliquid 1m follower price replay.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canonical_replay as cr
from cft_pit_evidence_audit import REGISTRY, ROOT, fetch_bybit_catalog

PRIOR = ROOT / "reports/2026-11-16__cft__10k__exhaustive-v3"
OUTPUT = ROOT / "reports/2026-11-16__cft__10k__pit-evidence-v4__train-listing.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("VERSIONED_LISTING_EVIDENCE_ALREADY_EXISTS")
    records, current_sha = cr.load_registry_universe(
        REGISTRY, include_aliases=("ACT04", "ACT08", "ACT09")
    )
    manifest = json.loads(
        PRIOR.with_name(PRIOR.name + "__inventory.json").read_text()
    )
    # Wallet aliases can change after research was sealed. Never silently
    # re-label a historical replay using the latest alias. Resolve the
    # original W#id fallback by its stable native wallet_id; block all other
    # unresolvable rename/census drift.
    prior_aliases = {row["alias"] for row in manifest["screen"]}
    original_fallbacks = {
        int(label[2:]): label
        for label in prior_aliases
        if label.startswith("W#") and label[2:].isdigit()
    }
    by_addr: dict[str, str] = {}
    alias_drift: list[dict[str, str]] = []
    for rec in records:
        if rec.label in prior_aliases:
            original = rec.label
        elif rec.wallet_id in original_fallbacks:
            original = original_fallbacks[rec.wallet_id]
            alias_drift.append({"original_alias": original, "current_alias": rec.label})
        else:
            raise ValueError("UNRESOLVED_HISTORICAL_WALLET_ALIAS")
        by_addr[rec.address] = original
    if set(by_addr.values()) != prior_aliases:
        raise ValueError("HISTORICAL_WALLET_CENSUS_DRIFT")
    sha = manifest["roster_sha256"]
    paths = [entry["path"] for entry in manifest["source"]["files"]]
    if not paths or any(not Path(p).exists() for p in paths):
        raise FileNotFoundError("SOURCE_PARTITION_MISSING")
    instruments, bybit = fetch_bybit_catalog()
    con = duckdb.connect(":memory:")
    try:
        con.execute("SET threads=3")
        con.execute("SET memory_limit='3GB'")
        con.execute(
            "CREATE TEMP TABLE bybit (symbol VARCHAR, launch_ms BIGINT, delivery_ms BIGINT)"
        )
        con.executemany(
            "INSERT INTO bybit VALUES (?, ?, ?)",
            [(s, m.launch_ms, m.delivery_ms) for s, m in instruments.items()],
        )
        rows = con.execute(
            """
            SELECT lower(f.wallet) wallet, f.coin,
                   count(*) fills,
                   count(*) FILTER (WHERE b.symbol IS NULL) unknown_market,
                   count(*) FILTER (WHERE b.symbol IS NOT NULL
                                   AND b.launch_ms IS NULL) unknown_launch,
                   count(*) FILTER (WHERE b.launch_ms IS NOT NULL
                                   AND epoch_ms(f.exchange_ts) < b.launch_ms) prelaunch,
                   count(*) FILTER (WHERE b.delivery_ms IS NOT NULL
                                   AND epoch_ms(f.exchange_ts) >= b.delivery_ms) postdelist,
                   min(epoch_ms(f.exchange_ts)) first_fill_ms,
                   max(epoch_ms(f.exchange_ts)) last_fill_ms
            FROM read_parquet(?, hive_partitioning=false) f
            LEFT JOIN bybit b ON concat(f.coin, 'USDT') = b.symbol
            WHERE f.dex = 'hyperliquid'
              AND f.exchange_ts >= TIMESTAMPTZ '2026-05-01T00:00:00Z'
              AND f.exchange_ts < TIMESTAMPTZ '2026-09-01T00:00:00Z'
              AND lower(f.wallet) IN (SELECT unnest(?::VARCHAR[]))
            GROUP BY wallet, f.coin
            """, [paths, sorted(by_addr)],
        ).fetchall()
    finally:
        con.close()

    wallets = defaultdict(lambda: {
        "fills": 0, "unknown_market": 0, "unknown_launch": 0,
        "prelaunch": 0, "postdelist": 0, "coins": {},
    })
    for address, coin, fills, unknown_market, unknown_launch, prelaunch, postdelist, first, last in rows:
        if not cr.is_hl_perp_coin(coin):
            continue
        alias = by_addr[address]
        summary = wallets[alias]
        for key, val in (
            ("fills", fills), ("unknown_market", unknown_market),
            ("unknown_launch", unknown_launch), ("prelaunch", prelaunch),
            ("postdelist", postdelist),
        ):
            summary[key] += int(val)
        if prelaunch or postdelist or unknown_market or unknown_launch:
            summary["coins"][coin] = {
                "fills": fills, "prelaunch": prelaunch,
                "postdelist": postdelist, "unknown_market": unknown_market,
                "unknown_launch": unknown_launch,
                "first_fill_ms": first, "last_fill_ms": last,
            }
    old_freeze = json.loads(
        PRIOR.with_name(PRIOR.name + "__freeze.json").read_text()
    )
    finalists = {alias for row in old_freeze["frozen_portfolios"]
                 for alias in row["aliases"]}
    covered = sorted(wallets)
    totals = {
        key: sum(int(row[key]) for row in wallets.values())
        for key in ("fills", "unknown_market", "unknown_launch", "prelaunch", "postdelist")
    }
    output = {
        "schema": "cft_train_bybit_launch_screen_v1",
        "status": "RETROSPECTIVE_LISTING_SCREEN_NOT_PIT_SOURCE",
        "train_window": ["2026-05-01T00:00:00Z", "2026-09-01T00:00:00Z"],
        "roster_sha256": sha,
        "current_roster_sha256": current_sha,
        "historical_label_reconciliation": alias_drift,
        "historical_identity_addresses_digest_at_freeze": "UNPROVEN_OLD_ROSTER_SHA_EXCLUDES_ADDRESSES",
        "archive_sha256": manifest["source"]["archive_sha256"],
        "bybit_catalog_sha256": bybit["public_metadata_sha256"],
        "instrument_source": "CURRENT_TRADING_AND_CLOSED_AT_AUDIT_TIME",
        "price_source": "HYPERLIQUID_LOCAL_1M_UNCHANGED",
        "wallets_with_fills": len(covered),
        "totals": totals,
        "finalist_wallets": sorted(finalists),
        "finalist_exposure": {
            alias: wallets.get(alias, {
                "fills": 0, "unknown_market": 0, "unknown_launch": 0,
                "prelaunch": 0, "postdelist": 0, "coins": {},
            }) for alias in sorted(finalists)
        },
        "affected_wallets": {
            alias: row for alias, row in sorted(wallets.items())
            if row["prelaunch"] or row["postdelist"]
            or row["unknown_market"] or row["unknown_launch"]
        },
        "unproven": [
            "HISTORICAL_BYBIT_LOT_PARAMS",
            "HISTORICAL_TEMPORARY_SUSPENSIONS",
            "HISTORICAL_FIRST_KNOWN_SOURCE_RECEIPTS",
            "PIT_HISTORICAL_REGISTRY",
        ],
        "trade_authorized": False,
    }
    OUTPUT.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "wallets_with_train_fills": len(covered),
        "totals": totals,
        "affected_finalists": {
            a: {k: wallets[a][k] for k in ("prelaunch", "postdelist", "unknown_market")}
            for a in sorted(finalists) if a in wallets
            and any(wallets[a][k] for k in ("prelaunch", "postdelist", "unknown_market"))
        },
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "output": str(OUTPUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
