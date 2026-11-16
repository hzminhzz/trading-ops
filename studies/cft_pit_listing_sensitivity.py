"""Post-hoc listing-window correction sensitivity for the three already-opened CFT pods.

This is NOT an independent OOS rerank. It reuses the canonical position
reconstructor and Guardian simulator and recalibrates half-risk from Train
without looking at OOS when choosing the new k.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canonical_replay as cr
import cft_20261116_exhaustive as prior

ROOT = Path(__file__).resolve().parents[1]
NAME = "2026-11-16__cft__10k__pit-evidence-v4__listing-sensitivity"
DEST = ROOT / "reports" / f"{NAME}.json"
REGISTRY = prior.REGISTRY


def _restore_original_aliases(
    records: list[cr.WalletRecord],
    old_labels: set[str],
) -> list[cr.WalletRecord]:
    remapped = []
    for r in records:
        if r.label in old_labels:
            remapped.append(r)
        elif f"W#{r.wallet_id}" in old_labels:
            remapped.append(dataclasses.replace(r, alias=f"W#{r.wallet_id}"))
        else:
            raise ValueError("UNRESOLVED_OLD_ROSTER_LABEL")
    if {r.label for r in remapped} != old_labels:
        raise ValueError("ROSTER_LABEL_CENSUS_DRIFT")
    return remapped


def _maxdd(step: np.ndarray) -> float:
    eq = np.cumsum(step, dtype=np.float64)
    if not eq.size:
        return 0.0
    peak = np.maximum.accumulate(np.maximum(eq, 0.0))
    return float(np.max(peak - eq))


def _coin_activation(
    listing: dict[str, dict[str, Any]], coins: set[str],
) -> dict[str, int]:
    out = {}
    for coin in coins:
        item = listing.get(f"{coin}USDT")
        if item is None or item["launch_ms"] is None:
            # This strict gate does not silently replace an unknown venue
            # listing with a current market status.
            raise ValueError(f"NO_HISTORICAL_LISTING_BOUNDS:{coin}")
        out[coin] = int(item["launch_ms"])
    return out


def _targets(
    aliases: list[str], sizing: list[float],
    grouped: dict[str, dict[str, Any]], prices: prior.Prices,
    grid: np.ndarray, launches: dict[str, int],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    coins = sorted({coin for name in aliases for coin in grouped[name]})
    member_pos = np.zeros((len(aliases), len(coins), len(grid)), dtype=np.float64)
    for a_idx, alias in enumerate(aliases):
        for c_idx, coin in enumerate(coins):
            data = grouped[alias].get(coin)
            if data is None:
                continue
            r = cr.reconstruct_positions_on_grid(
                data, grid, policy="new",
                activation_ms=max(int(grid[0]), launches[coin]),
            )
            member_pos[a_idx, c_idx] = r.pos
    net = cr.net_member_targets(member_pos, sizing)
    ix = int(np.searchsorted(prior.GRID, int(grid[0])))
    end = ix + len(grid)
    opens = np.vstack([prices.get(coin)[0][ix:end] for coin in coins])
    marks = np.vstack([prices.get(coin)[1][ix:end] for coin in coins])
    return coins, net, opens, marks


def main() -> None:
    if DEST.exists():
        raise FileExistsError("VERSIONED_LISTING_SENSITIVITY_EXISTS")
    inventory = json.loads(
        (ROOT / "reports/2026-11-16__cft__10k__exhaustive-v3__inventory.json").read_text()
    )
    freeze = json.loads(
        (ROOT / "reports/2026-11-16__cft__10k__exhaustive-v3__freeze.json").read_text()
    )
    listing_audit = json.loads(
        (ROOT / "reports/2026-11-16__cft__10k__pit-evidence-v4__audit.json").read_text()
    )
    if inventory["roster_sha256"] != freeze["roster_sha256"]:
        raise ValueError("INPUT_ROSTER_DRIFT")
    old_aliases = {row["alias"] for row in inventory["screen"]}
    current_records, _ = cr.load_registry_universe(
        REGISTRY, include_aliases=prior.TRIAD
    )
    records = _restore_original_aliases(current_records, old_aliases)
    grouped = prior.load_fills(records, [x["path"] for x in inventory["source"]["files"]])
    rows = {row["alias"]: row for row in inventory["screen"]}
    selected = {alias for row in freeze["frozen_portfolios"] for alias in row["aliases"]}
    grouped = {
        alias: {
            coin: grouped[alias][coin]
            for coin in rows[alias]["mapped_coins"]
            if coin in grouped.get(alias, {})
        } for alias in sorted(selected)
    }
    symbols = {coin for books in grouped.values() for coin in books}
    listings = listing_audit["bybit_symbol_evidence"]
    launches = _coin_activation(listings, symbols)
    prices = prior.Prices(prior.GRID)
    train_unit_dd: dict[str, float] = {}
    for alias, coins in sorted(grouped.items()):
        frames, _ = cr.build_unit_replay_by_alias(
            {alias: coins},
            {coin: tuple(v[:prior.CUT] for v in prices.get(coin)) for coin in coins},
            [alias], prior.TRAIN_GRID, policy="new", activation_ms=prior.START,
            symbol_activation_ms=launches,
        )
        solo = np.zeros(len(prior.TRAIN_GRID), dtype=np.float64)
        for frame in frames[alias].values():
            solo += frame["net"].to_numpy()
        train_unit_dd[alias] = _maxdd(solo)
    result = []
    for row in freeze["frozen_portfolios"]:
        aliases = row["aliases"]
        old_k = [float(k) for k in row["train_opt_k"]]
        risk_each = 1000.0 / len(aliases)
        base = [
            (0.50 * risk_each) / max(train_unit_dd[a], 0.01)
            for a in aliases
        ]
        corrected = [min(a, b) for a, b in zip(old_k, base)]
        effects = {}
        for name, k in (("prior_k", old_k), ("corrected_half_risk_k", corrected)):
            effects[name] = {}
            for period, grid in (
                ("train", prior.TRAIN_GRID), ("opened_old_oos", prior.OOS_GRID)
            ):
                coins, targets, opens, marks = _targets(
                    aliases, k, grouped, prices, grid, launches,
                )
                out, _ = cr.simulate_guardian_challenge(
                    coins, targets, opens, marks, grid, cr.GuardianSpec(),
                )
                effects[name][period] = {
                    "modeled_net_usd": round(out.full_window_net_usd, 2),
                    "first_barrier": out.verdict,
                    "peak_dd_usd": round(out.peak_to_trough_dd_usd, 2),
                    "static_floor_dd_usd": round(out.static_floor_dd_usd, 2),
                    "worst_daily_usd": round(out.worst_daily_usd, 2),
                    "sweeps": out.sweeps,
                }
        result.append({
            "aliases": aliases, "prior_frozen_k": old_k,
            "train_unit_maxdd_after_listing_gate": [train_unit_dd[a] for a in aliases],
            "train_recalibrated_half_risk_k": base,
            "conservative_corrected_k": corrected,
            "results": effects,
        })
    output = {
        "status": "RETROSPECTIVE_POSTHOC_LISTING_SENSITIVITY",
        "periods": {"train": ["2026-05-01", "2026-09-01"],
                    "previously_opened_oos": ["2026-09-01", "2026-11-01"]},
        "source_archive_sha256": inventory["source"]["archive_sha256"],
        "bybit_catalog_sha256": listing_audit["bybit_public"].get("public_metadata_sha256"),
        "policy": "strict_new_only; Bybit launch at coin level; local HL 1m execution",
        "source_first_known_at": "UNPROVEN_HISTORICAL",
        "instrument_lot_rule_history": "UNPROVEN",
        "reoptimized_using_oos": False,
        "oos_independence": False,
        "scientifically_qualified": False,
        "economically_qualified": False,
        "results": result,
    }
    DEST.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n")
    for row in result:
        a = row["results"]["prior_k"]["opened_old_oos"]
        b = row["results"]["corrected_half_risk_k"]["opened_old_oos"]
        print(json.dumps({
            "members": row["aliases"],
            "prior_k_oos_net": a["modeled_net_usd"],
            "corrected_k_oos_net": b["modeled_net_usd"],
            "corrected_k": row["conservative_corrected_k"],
        }))


if __name__ == "__main__":
    main()
