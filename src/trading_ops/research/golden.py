"""Canonical decision digest shared by the backtest, PM03 paper and Guardian.

The kernel deliberately returns typed objects, not a hash: a decision identity is
a consumer concern.  This module is the *single* definition of that identity, so
the cross-engine parity test compares the same bytes for all three shells.

``decision_sha`` covers everything the kernel decided for one checkpoint:

* the input ``KernelState`` (owner books, positions, Q15 opening memory),
* the requested notionals and Q15 opening rejections,
* the risk envelope (``status``/``scale``/``worst_loss``/...),
* the execution projection and the resulting ``NetOrderIntent`` rows.

It never covers anything the shells add afterwards (paper cost simulation,
journal writes, broker dispatch), which is exactly the surface where the shells
are allowed to differ.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from typing import Any

from copytrade_kernel import KernelState, KernelStepResult, NetOrderIntent

__all__ = [
    "DECISION_SHA_SCHEMA",
    "decision_payload",
    "decision_sha",
    "ending_state_rows",
    "net_intent_rows",
    "state_payload",
]

DECISION_SHA_SCHEMA = "kernel_checkpoint_decision_v1"


def _canon_decimal(value: Decimal | str) -> str:
    number = value if isinstance(value, Decimal) else Decimal(value)
    normalized = number.normalize()
    if normalized == 0:
        return "0"
    return format(normalized, "f")


def _text(value: Decimal | str | None) -> str | None:
    if value is None:
        return None
    return _canon_decimal(value)


def _validate_market_map(market_map: Mapping[str, str] | None) -> dict[str, str] | None:
    if market_map is None:
        return None
    mapping = dict(market_map)
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("GOLDEN_MARKET_MAP_COLLISION")
    return mapping


def _canon(market: str, mapping: Mapping[str, str] | None) -> str:
    if mapping is None:
        return market
    return mapping.get(market, market)


def _remap_dict(
    data: Mapping[str, Any],
    mapping: Mapping[str, str] | None,
    *,
    numeric_values: bool = False,
) -> dict[str, Any]:
    remapped: dict[str, Any] = {}
    for market, value in data.items():
        canonical = market if mapping is None else mapping.get(market, market)
        if canonical in remapped:
            raise ValueError("GOLDEN_MARKET_MAP_COLLISION")
        remapped[canonical] = (
            _canon_decimal(value) if numeric_values and isinstance(value, (Decimal, str)) else value
        )
    return remapped


def _remap_owner_markets(
    data: Mapping[str, Mapping[str, Any]],
    mapping: Mapping[str, str] | None,
    *,
    numeric_values: bool = False,
) -> dict[str, dict[str, Any]]:
    return {
        owner: _remap_dict(book, mapping, numeric_values=numeric_values)
        for owner, book in data.items()
    }


_RISK_DECIMAL_FIELDS = (
    "scale",
    "gross_notional",
    "worst_loss",
    "cost_reserve",
    "stressed_margin",
    "headroom_after_stress",
    "fidelity",
    "execution_shortfall_bps",
    "tracking_error",
)


def _remap_risk(risk: dict[str, Any], mapping: Mapping[str, str] | None) -> dict[str, Any]:
    remapped = dict(risk)
    for key in _RISK_DECIMAL_FIELDS:
        if isinstance(remapped.get(key), (Decimal, str)):
            remapped[key] = _canon_decimal(remapped[key])
    for key in ("targets", "approved_notionals"):
        if isinstance(remapped.get(key), Mapping):
            remapped[key] = _remap_dict(remapped[key], mapping, numeric_values=True)
    if isinstance(remapped.get("pod_targets"), Mapping):
        remapped["pod_targets"] = _remap_owner_markets(
            remapped["pod_targets"], mapping, numeric_values=True
        )
    if isinstance(remapped.get("pod_losses"), Mapping):
        remapped["pod_losses"] = {
            owner: _canon_decimal(value) if isinstance(value, (Decimal, str)) else value
            for owner, value in remapped["pod_losses"].items()
        }
    return remapped


def _remap_execution_plan(
    plan: dict[str, Any], mapping: Mapping[str, str] | None
) -> dict[str, Any]:
    remapped = dict(plan)
    if isinstance(remapped.get("targets"), Mapping):
        remapped["targets"] = _remap_dict(remapped["targets"], mapping, numeric_values=True)
    if isinstance(remapped.get("reduce_only"), Mapping):
        remapped["reduce_only"] = _remap_dict(remapped["reduce_only"], mapping)
    for key in ("pod_targets", "frozen_pods"):
        if isinstance(remapped.get(key), Mapping):
            remapped[key] = _remap_owner_markets(remapped[key], mapping, numeric_values=True)
    return remapped


def state_payload(
    state: KernelState, *, market_map: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Deterministic, order-independent view of one ``KernelState``."""
    if not isinstance(state, KernelState):
        raise TypeError("GOLDEN_STATE_REQUIRED")
    mapping = _validate_market_map(market_map)
    books = []
    for book in sorted(state.owner_books, key=lambda item: item.owner):
        remapped = _remap_dict(dict(book.positions), mapping)
        books.append(
            {
                "owner": book.owner,
                "positions": [
                    [market, _text(quantity)] for market, quantity in sorted(remapped.items())
                ],
                "opening_memory": [
                    [instrument.key, miss.missed_at.value, _text(miss.qty_at_miss)]
                    for instrument, miss in sorted(book.entry_memory, key=lambda row: row[0].key)
                ],
            }
        )
    return {"owner_books": books}


def ending_state_rows(
    state: KernelState, *, market_map: Mapping[str, str] | None = None
) -> tuple[tuple[str, str, str], ...]:
    """Flat ``(owner, market, quantity)`` rows for reports and assertions."""
    mapping = _validate_market_map(market_map)
    rows: list[tuple[str, str, str]] = []
    for book in sorted(state.owner_books, key=lambda item: item.owner):
        remapped = _remap_dict(dict(book.positions), mapping)
        for market, quantity in sorted(remapped.items()):
            rows.append((book.owner, market, str(quantity)))
    return tuple(rows)


def net_intent_rows(
    intents: tuple[NetOrderIntent, ...],
    *,
    market_map: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Typed net intents as exact strings (the wire shape every shell records)."""
    mapping = _validate_market_map(market_map)
    rows = [
        {
            "market": _canon(intent.market, mapping),
            "net_delta_qty": str(intent.net_delta_qty),
            "reduce_only": intent.reduce_only,
            "owner_allocations": [
                [owner, str(quantity)] for owner, quantity in intent.owner_allocations
            ],
        }
        for intent in intents
    ]
    if mapping is not None:
        if len({row["market"] for row in rows}) != len(rows):
            raise ValueError("GOLDEN_MARKET_MAP_COLLISION")
        rows.sort(key=lambda row: row["market"])
    return tuple(rows)


def decision_payload(
    *,
    checkpoint_ms: int,
    state: KernelState,
    step: KernelStepResult,
    market_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The exact bytes ``decision_sha`` hashes."""
    if type(checkpoint_ms) is not int or checkpoint_ms < 0:
        raise ValueError("GOLDEN_CHECKPOINT_REQUIRED")
    if not isinstance(step, KernelStepResult):
        raise TypeError("GOLDEN_STEP_REQUIRED")
    mapping = _validate_market_map(market_map)
    execution = step.execution
    notionals = []
    for owner, book in step.notionals:
        remapped = _remap_dict(dict(book), mapping)
        items = sorted(remapped.items()) if mapping is not None else list(book)
        notionals.append([owner, [[market, _text(value)] for market, value in items]])
    return {
        "schema": DECISION_SHA_SCHEMA,
        "checkpoint_ms": checkpoint_ms,
        "state": state_payload(state, market_map=mapping),
        "notionals": notionals,
        "opening_rejections": [[owner, list(keys)] for owner, keys in step.opening_rejections],
        "risk": None if execution is None else _remap_risk(execution.risk.data(), mapping),
        "execution_plan": (
            None
            if execution is None or execution.execution_plan is None
            else _remap_execution_plan(execution.execution_plan.data(), mapping)
        ),
        "projection_error": None if execution is None else execution.projection_error,
        "net_orders": (
            []
            if execution is None
            else [dict(row) for row in net_intent_rows(execution.net_orders, market_map=mapping)]
        ),
    }


def decision_sha(
    *,
    checkpoint_ms: int,
    state: KernelState,
    step: KernelStepResult,
    market_map: Mapping[str, str] | None = None,
) -> str:
    """SHA-256 over :func:`decision_payload`; the cross-engine golden identity."""
    payload = json.dumps(
        decision_payload(
            checkpoint_ms=checkpoint_ms, state=state, step=step, market_map=market_map
        ),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class GoldenStep:
    """One recorded checkpoint decision, as every engine must be able to emit."""

    checkpoint_ms: int
    decision_sha: str
    net_intents: tuple[dict[str, Any], ...]
    ending_positions: tuple[tuple[str, str, str], ...]
    status: str | None

    def data(self) -> dict[str, Any]:
        return {
            "checkpoint_ms": self.checkpoint_ms,
            "decision_sha": self.decision_sha,
            "net_intents": [dict(row) for row in self.net_intents],
            "ending_positions": [list(row) for row in self.ending_positions],
            "status": self.status,
        }


def golden_step(
    *,
    checkpoint_ms: int,
    state: KernelState,
    step: KernelStepResult,
    next_state: KernelState,
    market_map: Mapping[str, str] | None = None,
) -> GoldenStep:
    """Record one decision plus the state its intents lead to."""
    mapping = _validate_market_map(market_map)
    status = None if step.execution is None else step.execution.risk.data().get("status")
    return GoldenStep(
        checkpoint_ms=checkpoint_ms,
        decision_sha=decision_sha(
            checkpoint_ms=checkpoint_ms, state=state, step=step, market_map=mapping
        ),
        net_intents=(
            ()
            if step.execution is None
            else net_intent_rows(step.execution.net_orders, market_map=mapping)
        ),
        ending_positions=ending_state_rows(next_state, market_map=mapping),
        status=status,
    )
