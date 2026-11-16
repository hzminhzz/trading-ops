#!/usr/bin/env bash
# Scoped deterministic verification gate for shared portfolio research modules.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

uv run --frozen --group dev ruff check \
    studies/pod_constructor.py \
    studies/canonical_replay.py \
    studies/study_data.py \
    studies/2026-11-16__cft__10k__canonical-full-universe-v4.py \
    tests/test_pod_constructor.py \
    tests/test_canonical_replay.py \
    tests/test_study_data.py \
    tests/test_cft_v4_study.py \
    tests/test_ops_boundaries.py

uv run --frozen --group dev pytest -q \
    tests/test_pod_constructor.py \
    tests/test_canonical_replay.py \
    tests/test_study_data.py \
    tests/test_cft_v4_study.py \
    tests/test_ops_boundaries.py

# Guard against accidental drift in repository-wide non-legacy tests too.
uv run --frozen --group dev pytest -q
