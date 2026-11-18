"""Cross-repository companion pin, ancestry, and ChallengeRulebookSpec tests."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDIES_DIR = ROOT / "studies"
if str(STUDIES_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIES_DIR))

cr = importlib.import_module("canonical_replay")

PROP_PRODUCTION_ROOT = ROOT.parent / "prop-production"
COPYTRADE_PM_ROOT = ROOT.parent / "copytrade-pm"


def test_cross_repo_pin_ancestry_and_contracts_when_siblings_present() -> None:
    if not PROP_PRODUCTION_ROOT.exists() or not COPYTRADE_PM_ROOT.exists():
        pytest.skip("Sibling prop-production or copytrade-pm checkout not present")

    status = cr.check_cross_repo_pin_status(
        ROOT,
        prop_production_root=PROP_PRODUCTION_ROOT,
        copytrade_pm_root=COPYTRADE_PM_ROOT,
    )
    assert status["ancestry_valid"] is True
    assert status["status"] in (
        "SYNCED",
        "PRODUCTION_PIN_PROMOTION_PENDING",
        "TRADING_OPS_PIN_BEHIND",
    )
    contracts = status["contracts"]
    assert contracts["default_sizing_policy"] == "FIXED_K_NATIVE_QUANTITY"
    assert contracts["trading_authorized"] is False
    assert contracts["execution_mode"] == "SHADOW"


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def test_cross_repo_pin_status_ancestry_directions(tmp_path: Path) -> None:
    pm_repo = tmp_path / "copytrade-pm"
    pm_repo.mkdir()
    _git(pm_repo, "init")
    _git(pm_repo, "config", "user.name", "test")
    _git(pm_repo, "config", "user.email", "test@example.com")
    (pm_repo / "f.txt").write_text("base\n", encoding="utf-8")
    _git(pm_repo, "add", "f.txt")
    _git(pm_repo, "commit", "-m", "c1")
    c1 = _git(pm_repo, "rev-parse", "HEAD")

    (pm_repo / "f.txt").write_text("ahead\n", encoding="utf-8")
    _git(pm_repo, "commit", "-am", "c2")
    c2 = _git(pm_repo, "rev-parse", "HEAD")

    _git(pm_repo, "checkout", "-b", "side", c1)
    (pm_repo / "f.txt").write_text("side\n", encoding="utf-8")
    _git(pm_repo, "commit", "-am", "c3")
    c3 = _git(pm_repo, "rev-parse", "HEAD")

    ops_root = tmp_path / "trading-ops"
    prod_root = tmp_path / "prop-production"
    (prod_root / "src" / "prop_production").mkdir(parents=True)
    ops_root.mkdir()

    for ops_rev, prod_rev, expected_status, expected_valid, expected_ready in (
        (c2, c2, "SYNCED", True, True),
        (c2, c1, "PRODUCTION_PIN_PROMOTION_PENDING", True, False),
        (c1, c2, "TRADING_OPS_PIN_BEHIND", True, False),
        (c3, c2, "DIVERGED_PIN_ERROR", False, False),
    ):
        (ops_root / "pyproject.toml").write_text(
            f'[tool.uv.sources]\ncopytrade-pm = {{ git = "x", rev = "{ops_rev}" }}\n',
            encoding="utf-8",
        )
        (prod_root / "src" / "prop_production" / "compatibility.json").write_text(
            json.dumps({"repositories": {"copytrade-pm": {"commit": prod_rev}}}),
            encoding="utf-8",
        )
        res = cr.check_cross_repo_pin_status(
            ops_root, prop_production_root=prod_root, copytrade_pm_root=pm_repo
        )
        assert res["status"] == expected_status
        assert res["ancestry_valid"] is expected_valid
        assert res["promotion_ready"] is expected_ready


def test_guardian_spec_from_challenge_spec_matches_canonical_prop_firms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not PROP_PRODUCTION_ROOT.exists():
        pytest.skip("Sibling prop-production checkout not present")

    monkeypatch.syspath_prepend(str(PROP_PRODUCTION_ROOT / "src"))
    prop_firms = importlib.import_module("prop_production.prop_firms")

    cft_spec = prop_firms.get_challenge_spec("crypto-fund-trader:two_phase:phase_1")
    gs = cr.GuardianSpec.from_challenge_spec(cft_spec, scale=2.0)
    assert gs.initial_equity == pytest.approx(10_000.0)
    assert gs.profit_target == pytest.approx(800.0)
    assert gs.daily_loss_limit == pytest.approx(500.0)
    assert gs.max_static_dd == pytest.approx(1_000.0)
    assert gs.guardian_sweep_loss == pytest.approx(400.0)
    assert gs.daily_loss_basis == "balance"
    assert gs.daily_loss_rate == pytest.approx(0.05)
    assert gs.guardian_sweep_rate == pytest.approx(0.04)
    assert gs.reset_utc_hour == 0
    assert gs.reset_utc_minute == 5

    hyro_spec = prop_firms.get_challenge_spec("hyrotrader:two_step:phase_1")
    hyro_gs = cr.GuardianSpec.from_challenge_spec(hyro_spec, scale=2.0)
    assert hyro_gs.daily_loss_basis == "equity"
    assert hyro_gs.daily_loss_rate is None

    vanta_eod = prop_firms.get_challenge_spec("vanta-trading:grow:funded")
    with pytest.raises(ValueError, match="UNSUPPORTED_TRAILING_BASIS"):
        cr.GuardianSpec.from_challenge_spec(vanta_eod)
