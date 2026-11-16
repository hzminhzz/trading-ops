#!/usr/bin/env bash
# Scoped deterministic verification gate for shared portfolio research modules.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

uv run --frozen --group dev ruff check \
    studies/pod_constructor.py \
    studies/canonical_replay.py \
    tests/test_pod_constructor.py \
    tests/test_canonical_replay.py \
    tests/test_ops_boundaries.py

uv run --frozen --group dev pytest -q \
    tests/test_pod_constructor.py \
    tests/test_canonical_replay.py \
    tests/test_ops_boundaries.py

# Guard against accidental drift in repository-wide non-legacy tests too.
uv run --frozen --group dev pytest -q
