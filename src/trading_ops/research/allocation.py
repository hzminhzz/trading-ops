"""Portfolio-construction seam: joint fixed-k allocation decoupled from Stage-1 scoring."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Any

import numpy as np
from copytrade_pm.roster import BenchCandidate, GateFailure, digest

from .scoring import (
    UniverseScorecard,
    euler_marginal_cdar_shares,
    peak_drawdown,
    sizing_gate_failures,
)

__all__ = [
    "AllocationPolicy",
    "PortfolioAllocationPlan",
    "WalletAllocation",
    "allocate_fixed_k_portfolio",
]

_MICRO = Decimal("0.000001")
_SHARE_PLACES = Decimal("0.000000000001")


def _dec(value: float, places: Decimal = _MICRO) -> Decimal:
    if not math.isfinite(value):
        raise ValueError("NONFINITE_ALLOCATION_METRIC")
    return Decimal(repr(float(value))).quantize(places, rounding=ROUND_DOWN)


@dataclass(frozen=True, slots=True)
class AllocationPolicy:
    """Portfolio-level risk budget and capacity constraints for fixed-k sizing."""

    pod_equity_usd: Decimal = Decimal(10000)
    risk_unit_usd: Decimal = Decimal(200)
    max_fixed_k: Decimal = Decimal(10)
    cdar_alpha: float = 0.95
    max_euler_cdar_share: Decimal = Decimal("0.45")
    max_p95_participation: Decimal = Decimal("0.01")
    max_max_participation: Decimal = Decimal("0.05")

    def __post_init__(self) -> None:
        for name in ("pod_equity_usd", "risk_unit_usd", "max_fixed_k"):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
                raise ValueError(f"INVALID_ALLOCATION_POLICY:{name}")
        if self.max_fixed_k > Decimal(10):
            raise ValueError("INVALID_ALLOCATION_POLICY:max_fixed_k")
        for name in ("max_euler_cdar_share", "max_p95_participation", "max_max_participation"):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not Decimal(0) < value <= Decimal(1):
                raise ValueError(f"INVALID_ALLOCATION_POLICY:{name}")
        if not 0.5 <= self.cdar_alpha < 1:
            raise ValueError("INVALID_ALLOCATION_POLICY:cdar_alpha")

    @property
    def data(self) -> dict[str, Any]:
        return {
            "pod_equity_usd": str(self.pod_equity_usd),
            "risk_unit_usd": str(self.risk_unit_usd),
            "max_fixed_k": str(self.max_fixed_k),
            "cdar_alpha": self.cdar_alpha,
            "max_euler_cdar_share": str(self.max_euler_cdar_share),
            "max_p95_participation": str(self.max_p95_participation),
            "max_max_participation": str(self.max_max_participation),
        }


@dataclass(frozen=True, slots=True)
class WalletAllocation:
    """Portfolio-conditioned sizing and constraint outcome for one wallet."""

    wallet: str
    k: Decimal
    marginal_cdar_share: Decimal
    p95_participation: Decimal
    max_participation: Decimal
    eligible: bool
    failures: tuple[GateFailure, ...]

    @property
    def data(self) -> dict[str, Any]:
        return {
            "wallet": self.wallet,
            "k": str(self.k),
            "marginal_cdar_share": str(self.marginal_cdar_share),
            "p95_participation": str(self.p95_participation),
            "max_participation": str(self.max_participation),
            "eligible": self.eligible,
            "failures": [gate.value for gate in self.failures],
        }


@dataclass(frozen=True, slots=True)
class PortfolioAllocationPlan:
    """Joint fixed-k allocation derived from intrinsic wallet scorecards."""

    policy: AllocationPolicy
    allocations: tuple[WalletAllocation, ...]
    source_sha: str

    @property
    def k_by_wallet(self) -> dict[str, Decimal]:
        return {row.wallet: row.k for row in self.allocations if row.eligible}

    @property
    def eligible_wallets(self) -> tuple[str, ...]:
        return tuple(row.wallet for row in self.allocations if row.eligible)

    def to_bench_candidates(
        self,
        scorecard: UniverseScorecard,
        *,
        cluster_distance: Decimal = Decimal(1),
    ) -> tuple[BenchCandidate, ...]:
        """Convert eligible portfolio allocations into roster-ready ``BenchCandidate`` rows."""
        if not isinstance(scorecard, UniverseScorecard):
            raise TypeError("UNIVERSE_SCORECARD_REQUIRED")
        distance = Decimal(cluster_distance)
        if distance < 0:
            raise ValueError("INVALID_CLUSTER_DISTANCE")
        cards_by_wallet = scorecard.by_wallet
        candidates: list[BenchCandidate] = []
        for row in self.allocations:
            if not row.eligible:
                continue
            card = cards_by_wallet.get(row.wallet)
            if card is None or not card.eligible:
                continue
            clamped_cdar = min(max(row.marginal_cdar_share, Decimal(0)), Decimal(1)).quantize(
                _SHARE_PLACES, rounding=ROUND_DOWN
            )
            candidates.append(
                BenchCandidate(
                    wallet=row.wallet,
                    scorecard_sha=card.sha,
                    suggested_k=row.k,
                    marginal_cdar_share=clamped_cdar,
                    cluster_id=card.cluster_id,
                    cluster_distance=distance,
                )
            )
        return tuple(candidates)


def _policy_k(drawdown_usd: float, policy: AllocationPolicy) -> Decimal:
    if drawdown_usd <= 0.0:
        return policy.max_fixed_k
    raw = policy.risk_unit_usd / (Decimal(2) * _dec(drawdown_usd, Decimal("0.000000001")))
    if raw > policy.max_fixed_k:
        return policy.max_fixed_k
    if raw <= 0:
        raise ValueError("INVALID_ALLOCATED_K")
    return raw.quantize(_MICRO, rounding=ROUND_DOWN)


def allocate_fixed_k_portfolio(
    scorecard: UniverseScorecard,
    *,
    policy: AllocationPolicy | None = None,
    wallets: Sequence[str] | None = None,
) -> PortfolioAllocationPlan:
    """Derive portfolio-conditioned fixed-k weights from intrinsic Stage-1 evidence."""
    if not isinstance(scorecard, UniverseScorecard):
        raise TypeError("UNIVERSE_SCORECARD_REQUIRED")
    pol = policy or AllocationPolicy()
    if not isinstance(pol, AllocationPolicy):
        raise TypeError("INVALID_ALLOCATION_POLICY")

    allowed = (
        {wallet.strip().lower() for wallet in wallets} if wallets is not None else None
    )
    selected = sorted(
        (
            card
            for card in scorecard.cards
            if card.eligible and (allowed is None or card.wallet in allowed)
        ),
        key=lambda card: card.wallet,
    )
    if not selected:
        return PortfolioAllocationPlan(
            policy=pol,
            allocations=(),
            source_sha=digest({"scorecard": scorecard.source_sha, "policy": pol.data, "rows": []}),
        )

    returns_rows: list[tuple[float, ...]] = []
    unscaled_participation: list[tuple[float, float]] = []
    ks: list[Decimal] = []
    for card in selected:
        ret = scorecard.wallet_returns(card.wallet)
        unscaled_participation.append(scorecard.wallet_unscaled_participation(card.wallet))
        returns_rows.append(ret)
        dd = peak_drawdown(np.asarray(ret, dtype=np.float64)) * float(pol.pod_equity_usd)
        ks.append(_policy_k(dd, pol))

    matrix = np.array(returns_rows, dtype=np.float64)
    weights = np.array([float(k) for k in ks], dtype=np.float64)
    cdar_shares = euler_marginal_cdar_shares(matrix, weights, pol.cdar_alpha)

    allocations: list[WalletAllocation] = []
    for idx, card in enumerate(selected):
        k = ks[idx]
        raw_p95_unscaled, raw_max_unscaled = unscaled_participation[idx]
        p95_scaled = raw_p95_unscaled * float(k)
        max_scaled = raw_max_unscaled * float(k)
        raw_cdar = float(cdar_shares[idx])
        failures = sizing_gate_failures(
            p95_scaled,
            max_scaled,
            raw_cdar,
            max_p95=pol.max_p95_participation,
            max_max=pol.max_max_participation,
            max_cdar=pol.max_euler_cdar_share,
        )
        allocations.append(
            WalletAllocation(
                wallet=card.wallet,
                k=k,
                marginal_cdar_share=_dec(max(raw_cdar, 0.0), _SHARE_PLACES),
                p95_participation=_dec(max(p95_scaled, 0.0), _SHARE_PLACES),
                max_participation=_dec(max(max_scaled, 0.0), _SHARE_PLACES),
                eligible=not failures,
                failures=failures,
            )
        )

    return PortfolioAllocationPlan(
        policy=pol,
        allocations=tuple(allocations),
        source_sha=digest(
            {
                "scorecard": scorecard.source_sha,
                "policy": pol.data,
                "rows": [row.data for row in allocations],
            }
        ),
    )
