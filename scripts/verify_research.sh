#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# The retired Guardian/PM03 simulators (portfolio_risk_policy, canonical_replay's
# simulate_*, the registry universe loaders) and the dated studies that drove
# them were removed in Slice 4; see docs/adr/0004-slice4-kernel-research-path.md.
uv run --frozen --group dev ruff check \
    src/trading_ops \
    studies/pod_constructor.py \
    studies/canonical_replay.py \
    studies/study_data.py \
    studies/2026-11-21__cft__10k__may-nov-disjoint-10pods.py \
    scripts/colab_research_run.py \
    tests/test_pod_constructor.py \
    tests/test_study_data.py \
    tests/test_ops_boundaries.py \
    tests/test_score_universe.py \
    tests/test_kernel_backtest.py \
    tests/test_three_engine_golden_parity.py \
    tests/test_slice4_boundaries.py \
    tests/test_colab_runner.py \
    tests/test_may_nov_disjoint_10pods.py

uv run --frozen --group dev pytest -q
