# Changelog

All notable changes to `trading-ops` are documented in this file.

## [Phase D & E — Unified Kernel Research Path] - 2026-11-20

### Added
- `src/trading_ops/research/`:
  - `scoring.py`: vectorized Stage-1 PIT universe audit (`score_universe`) using `bars_1m` venue `quote_volume_usd` (`FACT_STORE_BARS_1M_VENUE_VOLUME`, `VENUE_VOLUME_UNKNOWN` exclusion), forward 5m markout against `bars_1m` `close` (`FACT_STORE_BARS_1M_CLOSE_5M`), exact Euler additive marginal CDaR (`EULER_MARGINAL_CDAR_COHORT`), and portfolio-conditioned eligible-subset `bench_from_scorecard`.
  - `backtest.py`: Stage-2 daily-reconciled walk-forward (`run_kernel_backtest`) stepping `copytrade_kernel.step_checkpoint` every 5m, computing `headroom` and `realized_made` via `copytrade_kernel.accounting.account_step`, and preserving `frozen_pods` / draining residuals across `FLAT` bars (`FLAT_WITH_DRAINING_RESIDUALS`).
  - `golden.py`: canonical `decision_sha` / `decision_payload` with bijective `market_map` canonicalization.
- `docs/adr/0002-single-canonical-rulebook-source.md`, `0003-fact-store-discovery.md`, and `0004-slice4-kernel-research-path.md`.
- `tests/test_rulebook_catalog_single_source.py`, `test_score_universe.py`, `test_kernel_backtest.py`, `test_three_engine_golden_parity.py` (true 3-shell end-to-end parity across Backtest, PM03 Paper, and Guardian), and `test_slice4_boundaries.py`.

### Changed
- Repinned `copytrade-kernel` (`prop-production`) and `copytrade-pm[market]` in `pyproject.toml` and `uv.lock` to exact git revisions.
- Switched `studies/pod_constructor.py` rulebooks to `copytrade_kernel.rulebooks.RulebookCatalog` and `studies/study_data.py` fill discovery to `FactStore` manifests (`discover_fact_partitions`).
- Updated legacy registry defaults in `studies/historical_5m_wallet_risk.py`, `studies/cwsa_native_prop10k_audit.py`, and `studies/canonical_replay.py` to the frozen archive path.
- Updated `LESSONS.md`, `scripts/verify_research.sh`, and `tests/test_three_engine_golden_parity.py`.

### Removed
- Deleted local rulebook table `prop_firms_rules.json` and `scripts/check_rulebook_drift.py`.
- Deleted `discover_fill_archive` Parquet globs from `studies/study_data.py`.
- Deleted duplicate Guardian/PM03 simulators and obsolete replay studies (`studies/portfolio_risk_policy.py`, `studies/fast_replay.py`, `studies/bench_replay.py`, `studies/bench_pod_sizing.py`, `canonical_replay.py` Guardian clones, and 9 retired test modules).
