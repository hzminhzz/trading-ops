# ADR 0004 — Stage-1 scoring and the kernel walk-forward own the research path

**Status:** accepted (Phase D, Slice 4: new research path, cross-engine parity
proof, and the Guardian/PM03 clone removal).

## Context

`trading-ops` research decided portfolio membership and net orders twice: once in
`studies/canonical_replay.py` (`GuardianSpec`, `simulate_guardian_challenge`,
`simulate_cwsa_one_shot_replay`, `reconcile_portfolio_with_pm03`) and once in
`studies/portfolio_risk_policy.py` (`run_portfolio_policy_replay`). Those engines
re-derived sizes, floors, netting and clamping that `prop-production`'s Guardian
owns, and their assumptions drifted from the signed `Q14–Q24` contract — the same
class of drift that produced the `\s` vs `\\s` and Q8/Q11/Q12 bugs.

The settled architecture replaces them with one shape: a vectorized Stage-1
audit that writes roster evidence, and a Stage-2 walk-forward that asks the
roster for seats and the kernel for orders.

## Decision

### 1. `trading_ops.research.score_universe` is the only Stage-1 audit

* Polars/NumPy `float64` inside; every number crosses into exact `Decimal` once,
  at the `WalletScorecard` boundary, and is recorded with
  `RosterStore.record_scorecard` (idempotent by content SHA).
* `score_universe` returns `UniverseScorecard`, whose `basis` names the
  measurement behind every field:
  * `participation = FACT_STORE_BARS_1M_VENUE_VOLUME` — per-wallet uncapped
    p95/max share of PIT `bars_1m` venue `quote_volume_usd` joined on
    `(coin, minute_ms)`. Any wallet with missing or `<= 0` venue quote volume on
    any traded `(coin, minute_ms)` is excluded with `VENUE_VOLUME_UNKNOWN` (never
    falling back to a cohort share). `PARTICIPATION_LIMIT` gates on uncapped
    `raw_p95` / `raw_worst` before clamping `[0, 1]` for `WalletScorecard`.
  * `cdar_share = EULER_MARGINAL_CDAR_COHORT` — exact Euler additive marginal
    CDaR decomposition over `suggested_k`-scaled cohort returns (with peak-at-0
    so shares sum to `1.0` within `1e-9`). `MARGINAL_CDAR_LIMIT` gates on
    `max(share_i, 0.0)` before clamping `[0, 1]` for `WalletScorecard`, and
    `bench_from_scorecard` computes portfolio-conditioned Euler marginal CDaR
    over the eligible bench subset.
  * `markout = FACT_STORE_BARS_1M_CLOSE_5M` — forward 5m markout against PIT
    `bars_1m` `close` prices via sorted `join_asof` by `coin`.
* Missing evidence is **missing**: a wallet with no equity row, missing/zero
  `bars_1m` venue volume, no dated order, or no derivable markout is returned in
  `excluded` with a typed reason and gets no scorecard at all. A scorecard never
  carries a fabricated zero.
* PIT: the slice is filtered by `first_known_at_ms <= as_of`. Reservoir archive
  rows carry no capture clock, so a strict run sees nothing; retrospective
  research must opt in with `allow_undated=True`, which is recorded as
  `pit = RETROSPECTIVE_UNDATED` in the scorecard basis.

### 2. `trading_ops.research.run_kernel_backtest` is the only Stage-2 walk-forward

* One reconciliation per UTC day (the window entry plus every following
  midnight); `BacktestConfig` refuses any cadence other than `DAY_MS`, so the
  `evaluate_universe()`-inside-the-5m-loop bug is structurally impossible.
* Every five-minute decision is `copytrade_kernel.step_checkpoint` with a real
  `DecisionInputs` (quotes, native market specs, risk policy, finance, positions,
  ledger basis). `project()`'s basis contract is enforced by the kernel, so a
  modelled fill that cannot prove its basis fails loudly. When all seats are
  vacant (`FLAT`) while evicted owners still hold open residuals, the loop emits
  `FLAT_WITH_DRAINING_RESIDUALS:<bar>` and still calls `step_checkpoint` with
  `outcomes=()` so `copytrade_kernel.decision.plan_execution` preserves
  `frozen_pods` (Q16/Q21) and re-nets residuals; open residual owners are
  included in `ending_state`.
* Accepted projections are booked as modelled paper fills per owner at the
  executable side plus explicit fee and shortfall; the account book is the sum of
  the owner books (asserted, not assumed).
* Daily floors are evaluated by `copytrade_kernel.account_step` on modelled
  equity; `challenge_pass_rate` is `1.0` only when no floor reason fired.
* Provenance is `RetrospectiveArchive` throughout: this path is research and can
  never authorize trading.

### 3. `decision_sha` is one definition, shared by three engines

`trading_ops.research.golden.decision_sha` hashes the checkpoint, the input
`KernelState`, requested notionals, Q15 rejections, the risk envelope, the
execution projection and the `NetOrderIntent` rows. It intentionally excludes
shell-only effects (paper cost simulation, journal writes, broker dispatch),
which is the only surface where the shells may differ.

`tests/test_three_engine_golden_parity.py` proves, with synthetic evidence over
one shared stream (`t0` flat `0` -> `t1` open `+3 BTC` -> `t2` partial reduce
`+1 BTC`):

1. `run_kernel_backtest`, `PaperJournal` (PM03), and `Guardian` +
   `CheckpointSourceReceiver(wire_version=3)` + `Journal` + `FakeBroker` produce
   byte-identical `decision_payload`, `decision_sha` (including
   `headroom_after_stress` via `copytrade_kernel.accounting.account_step` and
   bijective `market_map` canonicalization into `NativeInstrument`),
   `net_intents`, and persisted `ending_positions` at both `t1` and `t2`;
2. `step_checkpoint` is the *same object* in all three shells, and
   `decision_sha` changes whenever the decision changes.

## Consequences

* Research can no longer produce an order the kernel did not decide.
* Stage-1 is fast (vectorized) and its speed does not leak into Stage-2
  (exact `Decimal`).
* `allow_undated` makes the retrospective/PIT boundary explicit instead of
  implicit in whichever prefix a study happened to glob.

## Removed in this slice

Because `copytrade_pm.wallet_registry*` was archived by the Slice-3 cutover,
`studies/canonical_replay.py` could no longer even import, so the clone removal
became a prerequisite rather than an option. Deleted:

* `studies/canonical_replay.py` — `GuardianSpec`, `ChallengeRunResult`,
  `_execute_guardian_pass`, `simulate_guardian_challenge`, `extract_mc_daily_blocks`,
  `net_member_targets`, `reconcile_portfolio_with_pm03`, `check_cross_repo_pin_status`,
  the registry loaders (`load_registry_universe`, `load_universe_asof`,
  `_compute_inventory_digest`), the whole CWSA one-shot engine
  (`simulate_cwsa_one_shot_replay`, `classify_and_propose_cwsa_pair`, `allocate_*`,
  `estimate_causal_bar_capacity`, `screen_l1_allowed_markets`,
  `audit_1m_bar_participation`, `reconcile_live_executor_and_cwsa`) and the
  Guardian allocation helpers. 3,685 → ~530 lines, all extractors.
* `studies/portfolio_risk_policy.py` — the containment/lifecycle replay engine
  (2,274 lines), and its callers.
* Dated studies whose only purpose was to drive the clones:
  `2026-11-16__cft__10k__canonical-full-universe.py`, its `-v4` variant,
  `2026-11-17__cft__10k__sleeve-vs-upgma.py`,
  `2026-11-17__execution__live-paper-reconciliation.py`,
  `2026-11-18__cft__10k__phase3-pod-shop-ablation.py`, and the registry-based bench
  harnesses `bench_replay.py`, `bench_pod_sizing.py`, `fast_replay.py`. Their
  published `reports/` artifacts stay as historical evidence.
* Tests that pinned clone behaviour: `test_canonical_replay.py`,
  `test_cross_repo_pins.py`, `test_deep_underwriting_falsifiers.py`,
  `test_phase1_underwriting_and_sizing.py`,
  `test_phase2_cwsa_liquidity_and_containment.py`,
  `test_phase3_ablation_and_dedupe.py`, `test_uncapped_l3_default.py`,
  `test_2026_11_16_cft_canonical_study.py`, `test_cft_v4_study.py`.
* `tests/test_study_data.py` now builds `WalletRecord` rows directly instead of
  loading the retired registry, and `tests/test_pod_constructor.py`'s
  study-caller test asserts the Stage-1 invariants against the extractor itself.

`tests/test_slice4_boundaries.py` fences the result: no research module may import
a clone module, the deleted names must stay deleted, and no study may call a
retired registry or replay entry point.

## Consequence the owner must decide

Two dated studies still read the frozen legacy registry SQLite directly and
read-only (`studies/historical_5m_wallet_risk.py`,
`studies/cwsa_native_prop10k_audit.py`). They are historical-evidence readers, not
decision engines, and now point to the frozen archive path
`/home/quant/.local/share/copytrade/registry/archive/legacy_wallet_registry_20261119.sqlite`.

The deleted containment engine (`FREEZE_NEW_RISK`, `FLATTEN/RETIRE`, re-netting,
dual clocks, seat-replacement EV) has **no kernel or roster equivalent**: the
roster carries the Q12 latch, vetoes and draining owners, but it does not simulate
containment. Research that needs containment behaviour must either (a) get a
kernel-level containment contract, or (b) re-introduce it as an explicit,
kernel-consuming policy layer. It must not come back as a second decision engine.
