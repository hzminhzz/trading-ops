#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

uv run --frozen --group dev ruff check \
    studies/pod_constructor.py \
    studies/canonical_replay.py \
    studies/study_data.py \
    studies/portfolio_risk_policy.py \
    studies/2026-11-16__cft__10k__canonical-full-universe-v4.py \
    tests/test_pod_constructor.py \
    tests/test_canonical_replay.py \
    tests/test_study_data.py \
    tests/test_cft_v4_study.py \
    tests/test_ops_boundaries.py \
    tests/test_cross_repo_pins.py \
    tests/test_phase1_underwriting_and_sizing.py \
    tests/test_phase2_cwsa_liquidity_and_containment.py

uv run --frozen --group dev pytest -q
