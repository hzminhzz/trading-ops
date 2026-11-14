"""Contract checks keeping trading-ops strictly read-only and free of broker/secret sprawl."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_gitignore_blocks_secrets_and_private_ledgers() -> None:
    patterns = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    required = {
        ".env",
        ".env.*",
        "*.key",
        "*.pem",
        "*.sqlite",
        "*.sqlite3",
        "*.db",
        "*.duckdb",
        "*.parquet",
        "data/",
    }
    assert required <= patterns


def test_pyproject_does_not_pull_broker_transports_directly() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    deps = " ".join(pyproject["project"]["dependencies"]).lower()
    assert "trading-connector" not in deps
    assert "prop-production" not in deps
    assert "ccxt" not in deps
