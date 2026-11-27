"""Portfolio-construction seam: joint fixed-k allocation decoupled from Stage-1 scoring."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
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
    portfolio_dd_budget_usd: Decimal | None = None

    def __post_init__(self) -> None:
        for name in ("pod_equity_usd", "risk_unit_usd", "max_fixed_k"):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
                raise ValueError(f"INVALID_ALLOCATION_POLICY:{name}")
        if self.max_fixed_k > Decimal(10):
            raise ValueError("INVALID_ALLOCATION_POLICY:max_fixed_k")
        if self.portfolio_dd_budget_usd is not None and (
            not isinstance(self.portfolio_dd_budget_usd, Decimal)
            or not self.portfolio_dd_budget_usd.is_finite()
            or self.portfolio_dd_budget_usd <= 0
        ):
            raise ValueError("INVALID_ALLOCATION_POLICY:portfolio_dd_budget_usd")
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
            "portfolio_dd_budget_usd": (
                str(self.portfolio_dd_budget_usd)
                if self.portfolio_dd_budget_usd is not None
                else None
            ),
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


def _joint_budget_scale(
    anchor_curve: np.ndarray,
    cand_curve: np.ndarray,
    *,
    pod_equity_usd: float,
    budget_usd: float,
) -> float:
    """Return the largest ``scale`` in ``[0, 1]`` keeping ``anchor + scale * cand`` within budget."""
    if peak_drawdown(anchor_curve) * pod_equity_usd > budget_usd + 1e-9:
        return 0.0
    if peak_drawdown(anchor_curve + cand_curve) * pod_equity_usd <= budget_usd + 1e-9:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if peak_drawdown(anchor_curve + mid * cand_curve) * pod_equity_usd <= budget_usd:
            lo = mid
        else:
            hi = mid
    return lo


def allocate_fixed_k_portfolio(
    scorecard: UniverseScorecard,
    *,
    policy: AllocationPolicy | None = None,
    wallets: Sequence[str] | None = None,
    incumbent_k: Mapping[str, Decimal] | None = None,
) -> PortfolioAllocationPlan:
    """Derive portfolio-conditioned fixed-k weights from intrinsic Stage-1 evidence."""
    if not isinstance(scorecard, UniverseScorecard):
        raise TypeError("UNIVERSE_SCORECARD_REQUIRED")
    pol = policy or AllocationPolicy()
    if not isinstance(pol, AllocationPolicy):
        raise TypeError("INVALID_ALLOCATION_POLICY")

    allowed = {wallet.strip().lower() for wallet in wallets} if wallets is not None else None
    incumbents = {
        wallet.strip().lower(): Decimal(k) for wallet, k in (incumbent_k or {}).items()
    }
    by_wallet = scorecard.by_wallet
    anchor_wallets = sorted(w for w in incumbents if w in by_wallet)
    selected = sorted(
        (
            card
            for card in scorecard.cards
            if card.eligible
            and card.wallet not in incumbents
            and (allowed is None or card.wallet in allowed)
        ),
        key=lambda card: card.wallet,
    )
    if not selected and not anchor_wallets:
        return PortfolioAllocationPlan(
            policy=pol,
            allocations=(),
            source_sha=digest({"scorecard": scorecard.source_sha, "policy": pol.data, "rows": []}),
        )

    ret_by_w: dict[str, tuple[float, ...]] = {}
    part_by_w: dict[str, tuple[float, float]] = {}
    dd_by_w: dict[str, float] = {}
    base_k_by_w: dict[str, Decimal] = {}
    for wallet in anchor_wallets:
        ret = scorecard.wallet_returns(wallet)
        ret_by_w[wallet] = ret
        part_by_w[wallet] = scorecard.wallet_unscaled_participation(wallet)
        dd_by_w[wallet] = peak_drawdown(np.asarray(ret, dtype=np.float64)) * float(
            pol.pod_equity_usd
        )
        base_k_by_w[wallet] = incumbents[wallet]
    for card in selected:
        ret = scorecard.wallet_returns(card.wallet)
        ret_by_w[card.wallet] = ret
        part_by_w[card.wallet] = scorecard.wallet_unscaled_participation(card.wallet)
        dd = peak_drawdown(np.asarray(ret, dtype=np.float64)) * float(pol.pod_equity_usd)
        dd_by_w[card.wallet] = dd
        base_k_by_w[card.wallet] = _policy_k(dd, pol)

    pod_eq = float(pol.pod_equity_usd)
    anchor_matrix = (
        np.array([ret_by_w[w] for w in anchor_wallets], dtype=np.float64)
        if anchor_wallets
        else None
    )
    anchor_weights = (
        np.array([float(base_k_by_w[w]) for w in anchor_wallets], dtype=np.float64)
        if anchor_wallets
        else None
    )
    anchor_curve = (
        (anchor_matrix * anchor_weights[:, None]).sum(axis=0)
        if anchor_matrix is not None and anchor_weights is not None
        else None
    )

    active = list(selected)
    failed_allocations: list[WalletAllocation] = []
    surviving_allocations: list[WalletAllocation] = []

    while active:
        cand_matrix = np.array([ret_by_w[card.wallet] for card in active], dtype=np.float64)
        cand_ks = [base_k_by_w[card.wallet] for card in active]
        cand_weights = np.array([float(k) for k in cand_ks], dtype=np.float64)

        if pol.portfolio_dd_budget_usd is not None:
            budget = float(pol.portfolio_dd_budget_usd)
            zero_curve = np.zeros(cand_matrix.shape[1], dtype=np.float64)
            base_anchor_curve = zero_curve if anchor_curve is None else anchor_curve
            cand_curve = (cand_matrix * cand_weights[:, None]).sum(axis=0)
            scale = _joint_budget_scale(
                base_anchor_curve, cand_curve, pod_equity_usd=pod_eq, budget_usd=budget
            )
            if scale < 1.0:
                scale_dec = Decimal(repr(scale))
                cand_ks = [(k * scale_dec).quantize(_MICRO, rounding=ROUND_DOWN) for k in cand_ks]
                cand_weights = np.array([float(k) for k in cand_ks], dtype=np.float64)

        zero_k_failures = [idx for idx, k in enumerate(cand_ks) if k <= 0]
        if anchor_matrix is not None and anchor_weights is not None:
            full_matrix = np.vstack([anchor_matrix, cand_matrix])
            full_weights = np.concatenate([anchor_weights, cand_weights])
        else:
            full_matrix = cand_matrix
            full_weights = cand_weights

        cdar_shares = euler_marginal_cdar_shares(full_matrix, full_weights, pol.cdar_alpha)
        cand_offset = len(anchor_wallets)
        post_dd = peak_drawdown((full_matrix * full_weights[:, None]).sum(axis=0)) * pod_eq
        budget_violated = (
            pol.portfolio_dd_budget_usd is not None
            and post_dd > float(pol.portfolio_dd_budget_usd) + 1e-9
        )

        step_surviving: list[Any] = []
        step_survivor_rows: list[WalletAllocation] = []
        step_failed = False

        for idx, card in enumerate(active):
            k = cand_ks[idx]
            raw_p95_unscaled, raw_max_unscaled = part_by_w[card.wallet]
            p95_scaled = raw_p95_unscaled * float(k)
            max_scaled = raw_max_unscaled * float(k)
            raw_cdar = float(cdar_shares[cand_offset + idx])
            failures = sizing_gate_failures(
                p95_scaled,
                max_scaled,
                raw_cdar,
                max_p95=pol.max_p95_participation,
                max_max=pol.max_max_participation,
                max_cdar=pol.max_euler_cdar_share,
            )
            if idx in zero_k_failures and GateFailure.MARGINAL_CDAR_LIMIT not in failures:
                failures = (*failures, GateFailure.MARGINAL_CDAR_LIMIT)
            row = WalletAllocation(
                wallet=card.wallet,
                k=k,
                marginal_cdar_share=_dec(max(raw_cdar, 0.0), _SHARE_PLACES),
                p95_participation=_dec(max(p95_scaled, 0.0), _SHARE_PLACES),
                max_participation=_dec(max(max_scaled, 0.0), _SHARE_PLACES),
                eligible=not failures,
                failures=failures,
            )
            if failures:
                failed_allocations.append(row)
                step_failed = True
            else:
                step_surviving.append(card)
                step_survivor_rows.append(row)

        if step_failed:
            active = step_surviving
            continue
        if budget_violated:
            worst_idx = max(
                range(len(active)), key=lambda i: dd_by_w[active[i].wallet] * float(cand_ks[i])
            )
            worst_row = step_survivor_rows[worst_idx]
            failed_allocations.append(
                WalletAllocation(
                    wallet=worst_row.wallet,
                    k=worst_row.k,
                    marginal_cdar_share=worst_row.marginal_cdar_share,
                    p95_participation=worst_row.p95_participation,
                    max_participation=worst_row.max_participation,
                    eligible=False,
                    failures=(*worst_row.failures, GateFailure.MARGINAL_CDAR_LIMIT),
                )
            )
            active = [card for i, card in enumerate(active) if i != worst_idx]
            continue
        surviving_allocations = step_survivor_rows
        break

    allocations = sorted(failed_allocations + surviving_allocations, key=lambda row: row.wallet)

    return PortfolioAllocationPlan(
        policy=pol,
        allocations=tuple(allocations),
        source_sha=digest(
            {
                "scorecard": scorecard.source_sha,
                "policy": pol.data,
                "incumbents": [[w, str(incumbents[w])] for w in anchor_wallets],
                "rows": [row.data for row in allocations],
            }
        ),
    )
