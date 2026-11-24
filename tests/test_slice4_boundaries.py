"""Slice-4 boundary fences: the research path cannot regress into a clone.

These are AST checks, not behaviour checks: they keep the new kernel-backed
research layer honest about *what it may import*. The retired clone surfaces
(``studies.canonical_replay`` Guardian simulation, ``studies.portfolio_risk_policy``
replay loops) must never be reachable from ``trading_ops.research``; a future
change that reaches for them fails here instead of silently re-introducing a
second decision engine.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "src" / "trading_ops" / "research"

#: Modules that re-implemented Guardian/PM03 decisions. New code may not use them.
CLONE_MODULES = {"canonical_replay", "portfolio_risk_policy", "study_data", "pod_constructor"}

#: Everything the research layer is allowed to import (stdlib is checked separately).
ALLOWED_THIRD_PARTY = {
    "copytrade_kernel",
    "copytrade_pm",
    "numpy",
    "polars",
}
ALLOWED_STDLIB_PREFIXES = {
    "__future__",
    "collections",
    "dataclasses",
    "datetime",
    "decimal",
    "hashlib",
    "json",
    "math",
    "pathlib",
    "typing",
}


def _modules() -> list[Path]:
    files = sorted(RESEARCH.glob("*.py"))
    assert files, "research package is missing"
    return files


@pytest.mark.parametrize("path", _modules(), ids=lambda path: path.name)
def test_research_modules_never_import_a_clone(path: Path) -> None:
    tree = ast.parse(path.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & CLONE_MODULES), f"{path.name} imports a retired clone module"


@pytest.mark.parametrize("path", _modules(), ids=lambda path: path.name)
def test_research_modules_only_import_the_shared_contracts(path: Path) -> None:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        roots: set[str] = set()
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            roots = {node.module.split(".")[0]}
        for root in roots:
            assert root in ALLOWED_THIRD_PARTY or root in ALLOWED_STDLIB_PREFIXES, (
                f"{path.name} imports unexpected module {root!r}"
            )


def test_backtest_keeps_a_daily_reconcile_cadence_by_construction() -> None:
    source = (RESEARCH / "backtest.py").read_text()
    assert "RECONCILE_CADENCE_MUST_BE_DAILY" in source
    # The 5m loop must call the kernel, never an inline risk model.
    assert "step_checkpoint(" in source
    tree = ast.parse(source)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "reconcile_roster" in calls


def test_the_guardian_and_pm03_clones_are_gone() -> None:
    """The retired simulators must not come back under any name."""
    import canonical_replay

    for name in (
        "GuardianSpec",
        "ChallengeRunResult",
        "simulate_guardian_challenge",
        "simulate_cwsa_one_shot_replay",
        "reconcile_portfolio_with_pm03",
        "net_member_targets",
        "extract_mc_daily_blocks",
        "classify_and_propose_cwsa_pair",
        "audit_1m_bar_participation",
        "load_registry_universe",
        "load_universe_asof",
        "check_cross_repo_pin_status",
    ):
        assert not hasattr(canonical_replay, name), f"canonical_replay.{name} is a retired clone"
    assert not (ROOT / "studies" / "portfolio_risk_policy.py").exists()


def test_no_study_calls_a_retired_registry_or_replay() -> None:
    retired = (
        "load_registry_universe(",
        "load_universe_asof(",
        "simulate_guardian_challenge(",
        "simulate_cwsa_one_shot_replay(",
        "run_portfolio_policy_replay(",
        "reconcile_portfolio_with_pm03(",
        "wallet_registry as",
    )
    for path in sorted((ROOT / "studies").glob("*.py")):
        text = path.read_text()
        for needle in retired:
            assert needle not in text, f"{path.name} still calls {needle}"
