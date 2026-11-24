# Changelog

All notable changes to `trading-ops` are documented in this file.

## [Colab Runner & Onboarding Overhaul] - 2026-11-24

### Added
- `src/trading_ops/research/prop_firm_fees.json` & `fee_schedule.py` (`FeeQuote`, `load_fee_schedule`, `lookup_tier_fee`, `DEFAULT_FEE_SCHEDULE_PATH`): commercial per-tier challenge fee schedule owned by `trading-ops` and keyed against `copytrade_kernel.rulebooks.RulebookCatalog`.
- `scripts/colab_research_run.py` & `tests/test_colab_runner.py`: Google Colab research runner with `rclone` `FactStore` mirroring, remote venv caching, CPU-1 worker reserve, and sanitized artifact checks.
- `studies/2026-11-21__cft__10k__may-nov-disjoint-10pods.py`, `reports/2026-11-21__cft__10k__may-nov-disjoint-10pods*`, `proposals/2026-11-21__cft__10k__may-nov-disjoint-10pods.json`, and `tests/test_may_nov_disjoint_10pods.py`: May–Nov 2026 disjoint 10-pod evaluation on the Phase-D kernel path.

### Changed
- Rewrote `AGENTS.md`, `README.md`, and `runbooks/daily-ops.md` (fixed `facts coverage --store` flag/path) for Phase-D onboarding (`FactStore`, `RosterStore`, `copytrade_kernel`, `src/trading_ops/research/`, and `scripts/colab_research_run.py`).

### Removed
- Cleaned legacy untracked pre-Phase-D studies, fee catalogs (`studies/fee_catalog.py`), and local rulebook caches (`prop_firms_rules.json`) from the working tree.

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
