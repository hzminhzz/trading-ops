# Trading & Portfolio Lessons

Durable lessons from backtests, paper trading, and operational post-mortems in `trading-ops`.

**Agent rule:** Every agent MUST read this file in full at the start of any session before proposing portfolios, calibrating `k`, or running backtests. Append new dated entries when a study or incident reveals a durable rule; keep entries evidence-backed and concise (no secrets, raw `.env`, or wallet PII).

---

## 2026-11-14 — 5m UTC Prop-Challenge Backtest (`A`, `B`, `C`, `TriadChop`, `TriadA`, `2026-09-01..2026-11-10`)

Tested 5 candidate 3-wallet portfolios over `wallet_store` (`2026-09-01..2026-11-10`, 5m UTC, `pod_risk = $166.67` at Hyro 5k scale) against `hyrotrader:two_step:phase_1` ($5k) and `crypto-fund-trader:two_phase:phase_1` (`scale=2`, $10k / $1k DD). Only **`TriadA`** passed both (`+$756.63` Hyro 5k / `+$1,513.26` CFT 10k, positive in both Sep and Oct–Nov, zero breaches).

1. **Screen for multi-regime `net/max_dd`, not single-month PnL.**
   September trended cleanly, making nearly all wallets look profitable, whereas Oct–Nov chop exposed fragile pods (`W028 -$128.02`, `W030 -$52.34`, `W042 -$28.32`, `W031 -$40.86`, `VS066 -$65.65`, `ACT10 -$25.12`). Only **`ACT08`** (`unit_net/max_dd = +1.62`, `Sep +$39.07`, `Oct-Nov +$499.72`) and **`W015`** (`+0.52`, `Sep +$172.99`, `Oct-Nov +$1.15`) stayed positive across both regimes.

2. **Do not stack trend-correlated pods without a chop-resilient anchor.**
   `TriadChop` swapped `TriadA`'s anchor `ACT08` (`+$538.78`) for `ACT10` (`-$25.12`, `Oct-Nov -$283.09`), causing `ACT04`, `ACT09`, and `ACT10` to draw down together in Oct–Nov and breach `DAILY_LOSS`. Build prop portfolios around all-regime anchors (`ACT08`, `W015`) before adding higher-beta sleeves (`W044`, `ACT04`).

3. **Respect leverage risk when sizing `k` at `0.5x` `wallet_max_dd`.**
   Sizing `k` so a 50% wallet drawdown equals `pod_risk` leaves thin margin when multiple pods enter Oct–Nov drawdowns simultaneously (e.g., Portfolio B's `W031` + `CA-005` + `W044` pullback breaching `STATIC_DD`). Stress-test synchronized multi-pod drawdowns and consider `1.0x` `wallet_max_dd` sizing for strict prop limits.

4. **Verify active Hyperliquid perp fill coverage before admitting a pod.**
   `CO-001` had zero HL perp fills/bars in `wallet_store` over the test window, silently wasting 1/3 of Portfolio C's allocation while `VS066` dragged down `W015`. Gate candidate selection on verified active perp execution coverage in the target window.

---

## 2026-11-15 — Handling Flagged / Decayed Wallets in Live Portfolios (`wallet-research` -> `prop-ops`)

When a **live** portfolio contains a wallet flagged by `wallet-research` (`WITHDRAWN`, `DECAYED`, `DORMANT`), **never panic-delete it from `wallet_registry.sqlite`** (that breaks historical provenance). Instead, apply the decision rule by flag and replace via the 3-step workflow:

1. **Decision rule by flag:**
   - **`WITHDRAWN` (`AV < $1,000`):** Set `k = 0` immediately via `prop-ops` (lead trader has no skin in the game; `bybit_executor` cleanly closes their positions on the next 5m wave).
   - **`DECAYED` (active + losing, e.g. `W030`):** Set `k = 0` and replace; do not pay `6.5 bps` turnover on a losing trader.
   - **`DORMANT` with heavy uPnL (e.g. `ACT09`):** Inspect open positions (`clearinghouseState`) and resting `frontendOpenOrders` (TP/SL limits) before cutting. Setting `k = 0` immediately market-closes all attributed exposure on Bybit; if resting TP limits exist and Guardian covers the downside, either keep as-is or halve `k` (`0.00099 -> 0.00050`) to trim unmanaged drag.
2. **3-step replacement workflow:**
   1. **Contain (`prop-ops`):** Lower or zero `k` for the flagged member on the live unit (with explicit approval).
   2. **Replace (`portfolio-research`):** Pair healthy members with fresh Stage-4 candidates from `wallet-research` and verify via 5m replay + Guardian MC.
   3. **Version & Cutover (`prop-ops`):** Register as a new portfolio version (`copytrade-wallet-registry native exclude` / `native register`), deploy to `/opt/prop-production/current`, and restart the executor.

---

## 2026-11-15 — Prop Challenge Selection (Vanta & Propr) + Anti-Detection Execution Hygiene

Monte Carlo (`80k–100k` paths) + Optional Stopping analysis across Vanta Classic (`$5k`/`$10k`) and Propr (`Classic`, `Turbo`, `Pro` 1-Step & `Classic 2-Step`):

1. **Enforce a `-2%` soft daily stop on `3%` Daily DD accounts instead of buying DD add-ons.**
   Uncapped `1%`-risk intraday trading causes `3%` Daily DD to breach prematurely while `5%–6%` static Max DD still has room, flipping fair-coin (`p=0.50`) and slight-edge (`p=0.52`) EV negative. Adding a `-2%` UTC soft daily pause recovers full static pass rates (`33.7%` Vanta, `52.6%` Propr Classic at `p=0.52`) without paying Vanta's `+$23` `5%` Daily DD upgrade.
2. **Choose splits and tiers strictly by expected gross extraction `G = P(pass) × E[Funded Gross]`:**
   - **Vanta Classic (`$10k`, `5%` static DD):** Pure fair-coin is capped at `G <= $166.67` (`1/3 × $500`), so **`80/20` ($81)** dominates when `G < $170`, `90/10` ($98) wins for `$170–$290`, and **`100/0` ($127)** only wins with real positive edge (`G > $290`, i.e., `p >= 52%`). Buy `$10k` over `$5k` (`$127 < 2 × $67`, `2x` sizing precision, avoids cross-account mirroring bans), using `$5k` (`$67`) only as a first live-integration canary since Vanta has no sandbox.
   - **Propr (`$10k`):** With a `-2%` daily guard, **`Classic 1-Step` ($110, `10%` target, `6%` static DD)** dominates `Pro` ($85, `12%`/`5%`), `Turbo` ($50, `9%`/`3%`), and `2-Step` ($100, `5%+10%`, `8%` intraday trailing) across `p = 0.50–0.55`.
3. **Break micro-fill & clock fingerprints on anti-copytrading venues (`Vanta`):**
   - **Notional deadband:** Skip small rebalance deltas below a dollar floor (e.g. `$25–$50` when modifying an open position) to stop echoing every lead-wallet TWAP/trim, while **always bypassing the deadband on full closes (`target == 0`) and halt flattens** so dust never stays open.
   - **Post-wave jitter only (`10–60s`, never pre-bar):** `cwsa_heads` snapshots are only emitted at `next_checkpoint_ms` (5m UTC closes), so pre-5m execution jitter is impossible without shifting upstream sampling; bounded post-wave delay breaks exact `:00/:05` clustering while staying well inside `max_source_lag_ms` (`600s`).

---

## 2026-11-16 — Strict New-Position-Only Research and Retrospective Qualification

1. **Always reset lifecycle admission at the simulated activation boundary.** A position opened before an OOS activation is not a new position; carry a symbol only after a flat-to-open transition or a genuine side flip after activation. The May–Nov 2026 CFT study found five Train-selected candidates with negative Sep–Oct modeled follower net PnL under strict new-only admission. Historical full-target synchronization materially changed PnL and increased drawdown in several portfolios, so the policies must never be conflated.
2. **Keep projected fills and retrospective rosters separate from live-quality financial evidence.** Reservoir `start_position` supports historical quantity reconstruction, but neither source observation known-at time nor actual Bybit execution, funding, min-lot and market impact is proven. Train stale-price checkpoints invalidate precise `k` qualification; handle them by verified backfill or conservative pair-local close before release.
3. **Reject misleading pass probabilities.** Stationary-block Monte Carlo on historical Guardian daily summaries tests sensitivity under resampling, not independent broker-executable paths or a historically frozen wallet-selection release. No candidate is economically or scientifically qualified on these observations. Research artifact: `reports/2026-11-16__cft__10k__may-nov-follower-portfolio.md`.

---

## 2026-11-16 — Canonical Backtest Hardening & Full-Universe `v4` (`May–Nov 2026`, CFT `$10k`)

Hardening `studies/study_data.py`, `studies/canonical_replay.py`, `studies/pod_constructor.py`, and single-pass Parquet-cached `studies/2026-11-16__cft__10k__canonical-full-universe-v4.py` cut runtime from **`2,202.74s` (`36.7m`)** across `96` wallets in `v3` to **`818.61s` (`13.6m`)** across `189` wallets + full OOS stress matrices (`v4`), while uncovering four durable research rules:

1. **Never sort Hyperliquid fills by `trade_id` or filter timestamps in local timezone.**
   `trade_id` is non-monotonic inside same-millisecond bursts (`59,584 / 68,744` position-chain breaks in 30d vs raw arrival sequence), and host local-TZ SQL (`Europe/Berlin`) drops final UTC hours. Always ingest via UTC Polars lazy CSE scans preserving raw Parquet arrival sequence (`with_row_index`) + `start_position` continuity tie-break (`repair_same_ms_fill_order`) and assert UTC window coverage (`assert_utc_coverage`).
2. **Enforce pre-clustering `min_baseline_order_usd = $15` before UPGMA clustering.**
   In `v3` (`min_baseline_order_usd=0.0`), sub-`$15` wallets (`q10_order = $0.18–$12.44`) entered clustering and produced `CPCV PBO = 0.90`. Filtering sub-`$15` wallets at the `2%` portfolio-DD baseline (`$200 / size`) removed `73` wallets at `size=3` (`91` at `size=4`), eliminated `min_lot_warnings`, dropped leak-free **`CPCV PBO` from `0.90` to `0.15`**, and elevated live-deployable `Pod 3` (`W#111, W#152, W#312`, `0.93x` peak leverage, `$15` deadband) to `+$1,128` OOS (`new`, `-$15` static floor DD, `94.46%` 90d MC blended pass).
3. **Separate historical Train eligibility (`scope="historical"`) from present-day deployability (`scope="live"`).**
   Dropping `AV < $1,000` or currently unmapped wallets prior to Train replay injects survivorship lookahead (`scope="live"` alone lifted `CPCV PBO` to `0.70` vs `0.15` across the `189`-wallet historical screen). Keep historical eligibility strictly pre-split, then tag `live_deployable` independently on candidates.
4. **Gate headline OOS passes with deadbands (`$15`/`$30`), `2x`/`3x` cost stress, funding, and 1m participation.**
   `Pod 3` stays strongly profitable under `$30` deadband (`+$1,124` at `1x`), `2x`/`3x` costs (`+$936`/`+$743`), and historical USDT funding carry (`-$10.69`), yet 1m bar-volume diagnostics expose thin-altcoin spikes (`96.96%` of 1m bar volume on `XRP`, `79` trades `>1%`, vs `0.066%` median / `2.21%` p95) that require capacity validation and prospective CWSA paper runs in `copytrade-pm` prior to live authorization.

---

## 2026-11-17 — Research-to-Execution Contract Differences (Q8)

1. **Do not equate strict new-only with checkpoint late entry.** The research replay admits a pre-existing source leg only after a post-activation flat-to-open or genuine flip; the approved checkpoint design permits a later genuine increase to trigger one fresh full-target attempt. Compare them as different policies with their original clocks. See `studies/canonical_replay.py` and `prop-production/docs/ADR-CANONICAL-5M-CHECKPOINT-GRILL-20261111.md`.
2. **Check the signed release and execution path, not just a research `k`.** The current v1 verifier requires `0 < k <= 10`; a zero-k research setting cannot be silently projected. Checkpoint v2 code includes signed pause transitions, but that alone proves neither equivalent leader-following behavior nor deployment or trading authorization.
3. **Accounting reconciliation does not prove liquidity.** `reconcile_portfolio_with_pm03` supplies synthetic volume and 100% participation; the native 1m research replay rejects partial execution. Compare each policy's actual timing, partial-fill and broker-reconciliation assumptions explicitly before claiming equivalence.

---

## 2026-11-17 — PIT Universe Evidence (Q9)

1. **A `known_at` column is not a complete historical universe.** In the examined registry snapshot, cohort, exclusion and monitoring records begin on 2026-11-04, and all 158 `wallet_coverage.first_known` values are `UNKNOWN`. Earlier discovery may exist elsewhere; do not infer it from retrospective fills. The 269-wallet July–November Reservoir backfill is a November-selected cohort, not a PIT July roster.
2. **Do not backdate the v4 freeze.** The original v4 candidates and `k` were frozen on 2026-11-16; a prior-period comparison must re-run the static-v4 method using only evidence available at each cutoff. Previously inspected historical folds are exploratory; a frozen prospective paper comparison is separate.

---

## 2026-11-17 — Episode Underwriting Evidence (Q10)

1. **The current episode audit is not closed-episode ROPN underwriting.** Its extractor includes boundary-open runs without returning censor flags, and its empirical-Bayes shrinkage applies to active-bar Sharpe. See `reports/2026-11-17__research__episode-underwriting__q10-evidence.md`.
2. **Close order is not a causal post-loss entry order.** Overlapping per-symbol episodes may have opened before the previous loss was known; missing win/loss comparison groups currently receive a median fallback rather than `UNKNOWN`. Define causal transitions and evidence requirements before treating the tilt as a discipline gate.

---

## 2026-11-17 — Challenge Frontier Risk Semantics (Q11)

1. **Do not qualify CFT outcomes with an equity-reset proxy for a balance-reset rule.** The research Guardian replay resets its daily baseline to modeled equity and uses fixed dollar limits; CFT describes equity breaches against the 00:05 balance. A synthetic overnight open-loss scenario breaches both plausible balance-based limits while the research replay reports no breach or sweep. Resolve exact account terms and model balance separately before relying on pass/failure rates. See `reports/2026-11-17__research__risk-speed-frontier__q11-evidence.md`.
2. **Separate peak-to-trough DD, static-floor distance and risk-share proxies.** The sizing optimizer's `max_static_dd` is peak-to-trough, and its per-wallet shares are standalone-CDaR-weighted allocation proxies, not actual marginal contributions to net portfolio risk. Do not treat either label as an account-level guarantee.

---

## 2026-11-17 — Lifecycle Retirement and Release (Q12)

1. **An advisory CUSUM kill is not a durable retirement.** `update_roster_episode_cusum` can return `KILL_CONTAIN_0X` and then `OK` on a later episode; a retirement rule needs its own persisted terminal latch. Unit-copy CUSUM baselines are not automatically interchangeable with Q10 ROPN or broker-attributed PnL.
2. **Do not project a research unfreeze into a signed mandate.** The inspected checkpoint mandate contract has no paused-to-active edge, and its test requires a new generation to resume. A clean episode required while follower entries are frozen must come from a separately labeled causal shadow stream, not an imagined real fill. See `reports/2026-11-17__research__lifecycle-thresholds__q12-evidence.md`.

---

## 2026-11-17: Deep Underwriting Gates, Markout Decay & Live Reconciliation

1. **Audit `userFillsByTime` screens for tail truncation (`audit_user_fills_pagination`).** Across `179` cached `dextrabot-top-copy-30d` JSONs, `15` failed completeness (`TRUNCATED_PAGE_CAP=1`, `TAIL_WINDOW_TRUNCATED=9`, `UNVERIFIED_ZERO_FILLS=5`), and `0` carry `start_position`/`trade_id`/`dex`/`seq` for backtesting. Route flagged screens to Reservoir and never backtest vendor API dumps. See `reports/2026-11-17__execution__live-paper-reconciliation.md`.
2. **Filter sub-5m markout decay before clustering.** Across `254` Train-eligible Reservoir wallets (`2026-05-01..2026-09-01`), `108` failed `compute_fill_latency_markouts` (`106` `REJECT_NEGATIVE_5M_MARKOUT`, `2` `REJECT_SUB5M_CHURN`) and `3` failed `right_censored_bagholder`. On untouched strict `new` OOS (`2026-09-01..2026-11-01`, `balance`-reset CFT `$10k`), `Deep_Underwriting_Pod` passed (`+$800.33`, `-$9.17` static DD, `$275.8` break-even fee) vs `Naive_Leaderboard_1_over_DD` (`-$30.85`, `$0.31` break-even fee). Yet Euler `marginal_cdar_shares` (`0.554` on `W#107`), `12` `>5%` 1m capacity breaches (`59.69%` on `VVV`), and `CPCV PBO=0.60` block promotion without marginal caps and liquidity filters. See `reports/2026-11-17__cft__10k__pit-reservoir-oos-benchmark.md` (on branch `feat/deep-underwriting-skills-20261117`).
3. **Track executor lag and fill-price gaps explicitly.** `hydro-5k/executor.sqlite` (`864` waves, `54/54` `FILLED`) shows `37.9s` p50 wave lag, `241.7s` p50 dispatch lag (`+26.99 bps` p90 1m lag drift), and omits realized fill prices/fees (`BLOCKED_EVIDENCE:EXECUTOR_SQLITE_OMITS_FILL_PRICE_AND_FEE`); CWSA Chapter 2 paper DBs (`A/B/C`) hold `1,615` wave receipts each but `0` PM03 `events` (`BLOCKED_EVIDENCE:CWSA_PAPER_EVENTS_EMPTY_NO_SIGNED_MANDATE`).

---

## 2026-11-18 - Phase 3 End-to-End Pod-Shop Ablation & Causal Capacity Audit

Single-pass `197`-day empirical ablation (`studies/2026-11-18__cft__10k__phase3-pod-shop-ablation.py`, `2489.06s`, `5717.64 MB` peak RSS, `0` swap) over `2026-05-01..2026-11-14`:

1. **Euler marginal risk control outperforms Naive and Tier 1 across cost stress.** Across `254` Train-eligible / `126` positive `COPYABLE_DRIFT` wallets, enforcing `max_marginal_cdar_share <= 0.45` selected `W#107, W111, W312` (`full_pipeline_pbo = 0.3158` on `19/20` splits vs fixed-pod `0.80` on `15/20`; note `123`-day pre-split markout filtering preceded splitting). On Sep-Oct OOS CFT `$10k`, Tier 2 reached `PASS` at `1x` (`+$821.98` term / `+$1,811.29` full, `-$36.81` static DD) and `2x` cost (`+$801.78` / `+$1,445.76`), whereas Naive 1/DD (`Whale_H#d50e57, W#286, W#217`) timed out (`-$31.85`) and Tier 1 failed to pass at `2x` cost (`+$712.66` `TIMEOUT`).
2. **Causal L1-L3 mitigation leaves protective-exit 1m capacity breaches non-zero.** Switching to `cwsa_one_shot_v3` cut peak 1m participation from `162.30%` (T2) and `213.38%` (T3-T4) down to `52.67%` (`51.63%` T1), yet `7` (T1), `21` (T2), and `24` (T3-T4) bars still exceeded `5%` ex-post because `allocate_l3_market_step` leaves protective reductions uncapped when `l2_prior_vol_valid=True` and prior liquidity cannot predict bar-`j` volume droughts. Bybit step rounding, `$15` min trade notional, and funding diverged from continuous CWSA on `789-791` bars (`$2,868.27` vs `$1,693.66` Tier 2 full net); live capacity remains `UNPROVEN`.
3. **CUSUM retirement requires causal PIT bench promotion.** On Sep-Oct `Section B`, direct episode CUSUM/MAE retirement retired all `3` initial seats early (`0.00e+00` PM03 conservation error), leaving all `4` `LifecycleArm`s at `+$508.80` T2 / `+$501.19` T3 (`0.48-0.86%` Blended P(Funded) under `LEAVE_RISK_UNUSED`). Promoting `W#284, W#94, W#283` under `PROMOTE_PIT_TOP_BENCH_NEW_POS_ONLY` achieved `PASS` (`45.58%` T2 / `53.28%` T3 P(Funded); post-target continuation finished at `+$643.57` / `+$654.23`). Post-Nov-4 `PIT_STRICT_ASOF` (`W#210, W#315, W083`) returned `-$369.70` / `-$399.26` `RIGHT_CENSORED_AT_ARCHIVE_END` over `9` days.
4. **Rulebook parsing and sparse tensor memory discipline matter.** Parsing `prop_firms_rules.json` required isolating `propr` `1-Step` vs `2-Step` programs, guarding `100/0 default` and `(Grow only)` clauses on `vanta-trading` (`payout_split_pct = 1.0`, `$366.26` Blended breakeven), modeling Hyro's `40%` daily credit cap (`$952.66` breakeven), and blocking `propr:classic_2step:phase_1` fail-closed (`INTRADAY_EQUITY`). Demand-zero sparse 3D CPCV slicing (`_slice_3d_sparse_bars`) prevented multi-GB physical page zeroing that previously drove swap thrashing.

---

## 2026-11-18 - Phase 4 Upstream Release Bridge & Bybit Demo Reconciliation

1. **Match raw-string regexes and strict input types across producer and consumer contracts.** `copytrade_pm.pm_source_contracts.NativeKey` used single `\s` while `prop_production.checkpoint_source.parse_vector` had double-escaped `\\s`, which rejected lowercase `s` coins like `sUSD`; simultaneously enforcing string `k`, duplicate `(account_id, mandate_id)`, live expiry, and pinned dex scope closed producer/consumer validation drift.
2. **Interleave broker fills and funding chronologically across both v1 and checkpoint intents.** Replaying all fills before all funding events triggers false `FUNDING_POSITION_MISMATCH` HALTs when a funding settlement precedes a closing fill in the same `RECONCILE` snapshot, and skipping legacy v1 fills during `validate_attribution` breaks mixed-epoch replay.
3. **Never let zero-fill halt flattens or missing execution summaries silently wedge.** Persist `order_executions` alongside idempotent `filled_quantity`, VWAP `avg_price`, and `exec_fee` updates on `order_intents`, verify terminal `Filled`/`Rejected` status against filled quantity, and append attempt suffixes when retrying zero-fill halt flattens.

---

## 2026-11-20 — Single Canonical Rulebook Source (Slice 1 Kernel Cutover)

`studies/pod_constructor.load_prop_contract_spec` now reads the challenge catalog from
`copytrade_kernel.rulebooks.RulebookCatalog` and only rescales it; `prop_firms_rules.json` and
`scripts/check_rulebook_drift.py` are gone from this repo.

1. **Never re-introduce a local challenge-term table or a drift checker.** The hand-written parser
   (`AMBIGUOUS (...)` clause hints, `(Grow only)` guards, `100/0 default` special cases) was the
   bug source, not the guard against bugs. Terms come from the kernel catalog; a drift script only
   notices divergence after two copies already disagree.
2. **Never mint a fee, target or drawdown to complete a contract.** `challenge_fee_usd` stays
   `None` unless a caller supplies verified pricing, so `evaluate_multiphase_contract_ev` keeps
   reporting `BREAK_EVEN_ONLY_FEE_UNVERIFIED` instead of a fabricated net EV. A missing catalog
   amount raises `PROP_CONTRACT_INCOMPLETE:<key>`.
3. **Know the catalog's coverage before deleting a table.** The kernel catalog carries six firms;
   the retired JSON also held three `e8_markets_perp_*` slugs that no study consumed. E8 terms must
   be added upstream with sources before any E8 research resumes — do not rebuild them here.
4. **Rescale, do not re-derive.** Parity across 36 `(firm, program, account_size, min_days)` cases
   held every numeric field identical to the old loader; only display labels changed (canonical
   `phase_1`/`phase_2` names, selected-program `program_name`).

---

## 2026-11-20 — Fact-Store Discovery (Slice 2)

`studies/study_data.discover_fill_archive` and its `glob` are gone. Studies read
`facts.sqlite` through `discover_fact_partitions` / `copytrade_pm.facts`; the
wallet store's writer is retired.

1. **Never discover historical fills with a glob over named prefixes.** The old
   globs named two of the four Reservoir prefixes, so `reservoir-roster-daily` and
   `liquid12-discovery-20261117/fills_batch2` were invisible to every study, and
   `wallet_store` was bypassed entirely. A manifest names what exists, records how
   many rows each source contributed, and fails loudly when a partition file is
   missing instead of shrinking the window.
2. **A day that was read and empty is not a missing day.** The store closes such a
   day as an explicit zero-row batch; `missing_dates` therefore means "we never had
   this day", which is the only condition that should block a study.
3. **Never assign a point-in-time clock to archive rows.** Reservoir shards record
   no fetch time, so their `first_known_at_ms` is `NULL` and a `scan(as_of=…)`
   cannot see them. A `PIT_STRICT_ASOF` run over such a store returns nothing —
   that is correct behaviour, not an empty-data bug. Only the legacy wallet-store
   catalog `fetched_at` is real evidence, and it is used verbatim.
4. **Repair same-millisecond order once, at ingest.** `arrival_seq` from the store
   is authoritative; do not re-sort by `trade_id` downstream (see the 2026-11-15
   entry: `59,584 / 68,744` chain breaks) and do not re-derive order in a study.
5. **`Decimal(38,12)` truncates below 1e-12.** The migration counts and reports
   those cells; a study that needs sub-1e-12 prices would need a different scale,
   not a silent `Float64` fallback.

---

## 2026-11-20 — Stage-1 Scoring and the Kernel Walk-Forward (Slice 4)

`src/trading_ops/research/` now owns the research path: `score_universe` writes
`RosterStore` scorecards, `run_kernel_backtest` reconciles daily and steps
`copytrade_kernel.step_checkpoint` every five minutes. The Guardian/PM03 simulators
(`studies/canonical_replay.py`'s `simulate_*`, `studies/portfolio_risk_policy.py`,
the registry universe loaders) and the dated studies that drove them are deleted;
see `docs/adr/0004-slice4-kernel-research-path.md`. Their `reports/` artifacts stay
as historical evidence.

1. **Measure real PIT venue participation and Euler marginal CDaR, never silent proxies.**
   `score_universe` queries PIT `bars_1m` venue `quote_volume_usd` (`FACT_STORE_BARS_1M_VENUE_VOLUME`), excludes missing or `<= 0` venue volume via `VENUE_VOLUME_UNKNOWN` without cohort-share fallback, computes forward 5m markout against `bars_1m` `close` (`FACT_STORE_BARS_1M_CLOSE_5M`), and computes exact Euler additive marginal CDaR (`EULER_MARGINAL_CDAR_COHORT`, summing to `1.0` within `1e-9`), gating both before `[0, 1]` clamping.
2. **Missing evidence is an exclusion, never a zero.** A wallet with no `equity`
   row, no dated order, or no derivable 5m markout is returned in `excluded` and
   receives no scorecard — a zero account value would have silently passed the
   `$1k` floor gate in the wrong direction.
3. **A daily reconcile cadence must be enforced by construction.** `BacktestConfig`
   refuses any `reconcile_cadence_ms` other than `DAY_MS`, because the only bug it
   guards is `evaluate_universe()` being called inside the 5m loop (57,600 times
   in a 200-day window).
4. **`NO_TRADE` from the kernel is a finding, not a fixture problem.** An audit
   tape where buys outsize sells accrues a position large enough to breach
   `GROSS_EXPOSURE_CAP`, and the kernel is right to refuse it. Fix the synthetic
   tape or the research `k` cap; never widen the cap to make the test green.
5. **`decision_sha` covers the decision, not the shell.** Checkpoint, input
   `KernelState`, notionals, Q15 rejections, risk envelope, projection and net
   intents are hashed; paper cost simulation, journal writes and broker dispatch
   are not. Two policies that produce the same sized book therefore share a
   digest — that is the intended identity, and why a "policy changed" negative
   test must change the decision, not just a policy field.
6. **Deleting a simulator is only safe when the capability has a replacement.** The
   containment/lifecycle arms (`FREEZE_NEW_RISK`, `FLATTEN/RETIRE`, re-netting,
   dual clocks) existed only in `portfolio_risk_policy`; the roster carries the Q12
   latch and vetoes, not a containment simulator. The deletion is recorded with
   that gap explicit — it must not be re-implemented as a second decision engine.

---

## 2026-11-26 — Stage-1 Follower Execution Stream Defaults (ADR 0005)

Live Hyperliquid hybrid cohort studies (`2026-11-25`) showed four raw-print distortions in Stage-1 scoring when leaders executed maker/TWAP splits or held multi-hour swings on 5m checkpoints; see `docs/adr/0005-follower-execution-stream-stage1.md`:

1. **Score the 5m follower stream, never raw sub-bar splits or uncopiable symbols.** `score_universe` strips `[/:\s]` non-`MarketSpec` tickers (`exclude_non_market_symbols=True`, emitting `UNSCORED_NON_MARKET_SYMBOLS` / `NO_TRADABLE_MARKET_FILLS`) and nets fills into `(wallet, coin_norm, bar_ms)` rebalances with exact token inventory (`position_qty = cum_sum(delta_qty)`) before testing `min_baseline_order_usd = $15`.
2. **Scale participation by `suggested_k` on 5m venue buckets and keep sparse gaps explicit.** `participation_scale_by_k=True` evaluates `suggested_k`-scaled follower impact against 5m-aggregated `bars_1m.quote_volume_usd` (`FACT_STORE_BARS_5M_FOLLOWER_K_SCALED`), excludes `VENUE_VOLUME_UNKNOWN` whenever valid bars `== 0` or missing/zero volume exceeds `max_missing_volume_frac` (`0.02`), and records any within-tolerance gaps explicitly via `TOLERATED_MISSING_VOLUME_BARS:<wallet>:<missing>/<total>` in `UniverseScorecard.notes` (never silent proxies).
3. **Prefer follower trajectory edge (`FOLLOWER_BAR_TRAJECTORY_EDGE`) over `+5m` raw fill markout.** Realized + tail PIT mark-to-market trajectory PnL per dollar of rebalanced turnover retains profitable swing leaders despite `+5m` entry noise while rejecting churning losers with positive `+5m` noise bounces (`NEGATIVE_5M_MARKOUT`).

