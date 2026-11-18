---
firm: crypto-fund-trader
account_usd: 10000
status: RESEARCH
top_candidate: UNPROVEN_RETROSPECTIVE_PHASE3_ABLATION
script: studies/2026-11-18__cft__10k__phase3-pod-shop-ablation.py
---

# Phase 3 pod-shop ablation study across May to November 2026 archives

## Single-pass archive coverage and funnel

- Archive ingestion covered `197` daily files across `2026-05-01T00:00:00Z` to `2026-11-14T00:00:00Z` (`archive_sha256=959a4f195863`, `roster_sha256=59896baa926c`).
- Stage timings in seconds completed in `2489.06` total (discover `0.19`, ingest `5.32`, markouts and replay `330.74`, stage2 and cpcv `289.63`, oos ablations `1041.28`, walk-forward and pit strict `821.9`, peak RSS `5717.64` MB).
- Reservoir funnel retained `261` Train active wallets, `254` historically eligible wallets, `126` positive COPYABLE_DRIFT wallets, and `40` Stage 1 Tier 2 survivors.
- Tier 2 CPCV PBO measured `0.8` across `15` evaluated of `20` total leak-free splits and `0.3158` across `19` evaluated of `20` total full-pipeline splits.
- Section A CPCV note. Wallet eligibility, 5m markout gating (`COPYABLE_DRIFT`), and positive unit-replay pre-filtering ran once across the full 123-day pre-split Train window before CPCV block splitting.

## Section A primary OOS ablations on September to October 2026

| Variant | Mode | Cost | Aliases | Verdict | Term net ($) | Full net ($) | Static DD ($) | Worst day ($) | Capacity status | Max part (%) | CWSA net ($) |
|---|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|
| `Naive_Leaderboard_1_over_DD` | `bar_delta` | 1x | `Whale_H#d50e57, W#286, W#217` | `TIMEOUT` | -31.85 | -31.85 | -76.94 | -408.12 | `CAPACITY_BREACH_GT_5PCT` | 134.3284 | n/a |
| `Deep_Legacy_BarDelta_StandaloneProxy` | `bar_delta` | 1x | `W#107, W111, W312` | `PASS` | 805.14 | 1121.52 | -15.39 | -139.90 | `CAPACITY_BREACH_GT_5PCT` | 59.6929 | n/a |
| `Deep_Euler_Tier1_BarDelta` | `bar_delta` | 1x | `W#107, W111, W312` | `PASS` | 801.32 | 883.48 | -11.71 | -183.01 | `CAPACITY_BREACH_GT_5PCT` | 51.6290 | n/a |
| `Deep_Euler_Tier1_CWSA` | `cwsa_one_shot_v3` | 1x | `W#107, W111, W312` | `PASS` | 800.43 | 879.91 | -12.09 | -183.01 | `CAPACITY_BREACH_GT_5PCT` | 51.6290 | 896.69 |
| `Deep_Euler_Tier1_BarDelta_2xCost` | `bar_delta` | 2x | `W#107, W111, W312` | `TIMEOUT` | 712.66 | 712.66 | -13.16 | -187.50 | `CAPACITY_BREACH_GT_5PCT` | 51.6290 | n/a |
| `Deep_Euler_Tier1_CWSA_2xCost` | `cwsa_one_shot_v3` | 2x | `W#107, W111, W312` | `TIMEOUT` | 714.64 | 714.64 | -12.95 | -187.50 | `CAPACITY_BREACH_GT_5PCT` | 51.6290 | 728.58 |
| `Deep_Euler_Tier2_BarDelta` | `bar_delta` | 1x | `W#107, W111, W312` | `PASS` | 821.98 | 1811.29 | -36.81 | -40.33 | `CAPACITY_BREACH_GT_5PCT` | 162.3017 | n/a |
| `Deep_Euler_Tier2_CWSA` | `cwsa_one_shot_v3` | 1x | `W#107, W111, W312` | `PASS` | 807.61 | 1693.66 | -38.15 | -40.33 | `CAPACITY_BREACH_GT_5PCT` | 52.6712 | 2868.27 |
| `Deep_Euler_Tier2_BarDelta_2xCost` | `bar_delta` | 2x | `W#107, W111, W312` | `PASS` | 801.78 | 1445.76 | -41.38 | -42.72 | `CAPACITY_BREACH_GT_5PCT` | 162.3017 | n/a |
| `Deep_Euler_Tier2_CWSA_2xCost` | `cwsa_one_shot_v3` | 2x | `W#107, W111, W312` | `PASS` | 807.30 | 1356.19 | -40.66 | -42.72 | `CAPACITY_BREACH_GT_5PCT` | 52.6712 | 2350.51 |
| `Deep_Euler_Tier3_BarDelta` | `bar_delta` | 1x | `W#107, W111, W312` | `PASS` | 873.34 | 1863.07 | -48.41 | -53.31 | `CAPACITY_BREACH_GT_5PCT` | 213.3840 | n/a |
| `Deep_Euler_Tier3_CWSA` | `cwsa_one_shot_v3` | 1x | `W#107, W111, W312` | `PASS` | 808.02 | 1505.33 | -50.18 | -53.31 | `CAPACITY_BREACH_GT_5PCT` | 52.6712 | 3675.97 |
| `Deep_Euler_Tier3_BarDelta_2xCost` | `bar_delta` | 2x | `W#107, W111, W312` | `PASS` | 852.69 | 944.08 | -54.41 | -56.43 | `CAPACITY_BREACH_GT_5PCT` | 213.3840 | n/a |
| `Deep_Euler_Tier3_CWSA_2xCost` | `cwsa_one_shot_v3` | 2x | `W#107, W111, W312` | `PASS` | 815.84 | 1201.99 | -53.45 | -56.43 | `CAPACITY_BREACH_GT_5PCT` | 52.6712 | 2999.69 |
| `Deep_Euler_Tier4_BarDelta` | `bar_delta` | 1x | `W#107, W111, W312` | `PASS` | 873.34 | 1863.07 | -48.41 | -53.31 | `CAPACITY_BREACH_GT_5PCT` | 213.3840 | n/a |
| `Deep_Euler_Tier4_CWSA` | `cwsa_one_shot_v3` | 1x | `W#107, W111, W312` | `PASS` | 808.02 | 1505.33 | -50.18 | -53.31 | `CAPACITY_BREACH_GT_5PCT` | 52.6712 | 3675.97 |
| `Deep_Euler_Tier4_BarDelta_2xCost` | `bar_delta` | 2x | `W#107, W111, W312` | `PASS` | 852.69 | 944.08 | -54.41 | -56.43 | `CAPACITY_BREACH_GT_5PCT` | 213.3840 | n/a |
| `Deep_Euler_Tier4_CWSA_2xCost` | `cwsa_one_shot_v3` | 2x | `W#107, W111, W312` | `PASS` | 815.84 | 1201.99 | -53.45 | -56.43 | `CAPACITY_BREACH_GT_5PCT` | 52.6712 | 2999.69 |

## Section B four-arm containment and seat replacement modes

- Section B continuation note. Policy replay continues trading through the full OOS window after reaching the Guardian profit target, and `Static DD ($)` uses the same non-positive floor sign convention as Sections A and C.

| Tier | Policy | Net PnL ($) | Static DD ($) | Guardian verdict | PM03 max err | Bars gt 5 pct | Retired seats | Promoted bench | Blended P(Funded) pct |
|---:|---|---:|---:|---|---:|---:|---|---|---:|
| T2 | `EPISODE_CUSUM_ONLY` | 508.80 | -32.08 | `ACTIVE` | 0.00e+00 | 8 | `W#107, W111, W312` | `none` | 0.72 |
| T2 | `GUARDIAN_MANDATE_PAUSE_STATIC_HOLD` | 508.80 | -32.08 | `ACTIVE` | 0.00e+00 | 8 | `W#107, W111, W312` | `none` | 0.78 |
| T2 | `RESEARCH_LEADER_REDUCE_FREEZE` | 508.80 | -32.08 | `ACTIVE` | 0.00e+00 | 8 | `W#107, W111, W312` | `none` | 0.62 |
| T2 | `NEXT_FLAT_RELEASE` | 508.80 | -32.08 | `ACTIVE` | 0.00e+00 | 8 | `W#107, W111, W312` | `none` | 0.64 |
| T2 | `LEAVE_RISK_UNUSED` | 508.80 | -32.08 | `ACTIVE` | 0.00e+00 | 8 | `W#107, W111, W312` | `none` | 0.48 |
| T2 | `RESIZE_HEALTHY_SEATS` | 544.90 | -32.08 | `ACTIVE` | 0.00e+00 | 8 | `W#107, W111, W312` | `none` | 0.78 |
| T2 | `PROMOTE_PIT_TOP_BENCH_NEW_POS_ONLY` | 643.57 | -32.08 | `PASS` | 0.00e+00 | 8 | `W#107, W111, W312, W#284, W#94, W#283` | `W#284, W#94, W#283` | 45.58 |
| T3 | `EPISODE_CUSUM_ONLY` | 501.19 | -42.19 | `ACTIVE` | 0.00e+00 | 8 | `W#107, W111, W312` | `none` | 0.74 |
| T3 | `GUARDIAN_MANDATE_PAUSE_STATIC_HOLD` | 501.19 | -42.19 | `ACTIVE` | 0.00e+00 | 8 | `W#107, W111, W312` | `none` | 0.86 |
| T3 | `RESEARCH_LEADER_REDUCE_FREEZE` | 501.19 | -42.19 | `ACTIVE` | 0.00e+00 | 8 | `W#107, W111, W312` | `none` | 0.72 |
| T3 | `NEXT_FLAT_RELEASE` | 501.19 | -42.19 | `ACTIVE` | 0.00e+00 | 8 | `W#107, W111, W312` | `none` | 0.76 |
| T3 | `LEAVE_RISK_UNUSED` | 501.19 | -42.19 | `ACTIVE` | 0.00e+00 | 8 | `W#107, W111, W312` | `none` | 0.56 |
| T3 | `RESIZE_HEALTHY_SEATS` | 548.64 | -42.19 | `ACTIVE` | 0.00e+00 | 8 | `W#107, W111, W312` | `none` | 0.80 |
| T3 | `PROMOTE_PIT_TOP_BENCH_NEW_POS_ONLY` | 654.23 | -42.19 | `PASS` | 0.00e+00 | 8 | `W#107, W111, W312, W#284, W#94, W#283` | `W#284, W#94, W#283` | 53.28 |

## Section C rolling walk-forward and post-Nov-4 PIT strict evaluation

- Section C execution note. Rolling walk-forward folds evaluate continuous unit-scale 5m returns without Bybit lot rounding, funding carry, minimum trade notional, or CWSA 1m caps, whereas `pit_strict_nov05` evaluates full Guardian and CWSA replay.

| Fold | Config | Train aliases | OOS status | OOS net frac | OOS static DD frac | Provenance kind |
|---:|---|---|---|---:|---:|---|
| 0 | `Tier2_MarginalEuler` | `W#174, W#47, W312` | `RIGHT_CENSORED_OPEN` | 0.0182 | 0.0026 | `RETROSPECTIVE_ESTIMATE` |
| 0 | `Tier1_StandaloneProxy` | `W#26, W#283, W312` | `RIGHT_CENSORED_OPEN` | 0.0204 | 0.0030 | `RETROSPECTIVE_ESTIMATE` |
| 1 | `Tier2_MarginalEuler` | `W#100, W#97, W111` | `PASS` | 0.3530 | 0.0090 | `RETROSPECTIVE_ESTIMATE` |
| 1 | `Tier1_StandaloneProxy` | `W#174, W#97, W111` | `PASS` | 0.0851 | 0.0024 | `RETROSPECTIVE_ESTIMATE` |
| 2 | `Tier2_MarginalEuler` | `W#315, W#48, W083` | `RIGHT_CENSORED_OPEN` | 0.0229 | 0.0034 | `RETROSPECTIVE_ESTIMATE` |
| 2 | `Tier1_StandaloneProxy` | `W#315, W#48, W083` | `RIGHT_CENSORED_OPEN` | 0.0039 | 0.0059 | `RETROSPECTIVE_ESTIMATE` |

- Post-Nov-4 strict PIT snapshot at `2026-11-05T00:00:00Z` trained over `90` days on `131` positive COPYABLE_DRIFT wallets (`roster_sha256=119c623b7d08`).

| Variant | Mode | Aliases | Verdict | Full net ($) | Static DD ($) | Capacity status |
|---|---|---|---|---:|---:|---|
| `PIT_Strict_Nov05_Tier2_bar_delta` | `bar_delta` | `W#210, W#315, W083` | `RIGHT_CENSORED_AT_ARCHIVE_END` | -369.70 | -369.70 | `CAPACITY_BREACH_GT_5PCT` |
| `PIT_Strict_Nov05_Tier2_cwsa_one_shot_v3` | `cwsa_one_shot_v3` | `W#210, W#315, W083` | `RIGHT_CENSORED_AT_ARCHIVE_END` | -399.26 | -399.26 | `CAPACITY_BREACH_GT_5PCT` |

## Section D multi-firm conditional EV frontier and fail-closed gates

- Blocked gate verification for `propr:classic_2step:phase_1` returned `BLOCKED_UNSUPPORTED_TRAILING_BASIS:INTRADAY_EQUITY` (rulebook eval phases `1-Step=1`, `2-Step=2`).

| Firm slug | Daily basis | OOS verdict | Term net ($) | Full net ($) | Regime | P(Funded) pct [95 pct CI] | Breakeven fee ($) | Fee status | Net EV at 110 ($) |
|---|---|---|---:|---:|---|---|---|---|---:|
| `crypto-fund-trader:two_phase:phase_1` | `balance` | `PASS` | 807.61 | 1536.84 | `Blended` | 61.50 [30.84, 85.13] | 590.98 | `BREAK_EVEN_ONLY_FEE_UNVERIFIED` | 480.98 |
| `crypto-fund-trader:two_phase:phase_1` | `balance` | `PASS` | 807.61 | 1536.84 | `75%-Chop` | 61.60 [30.92, 85.19] | 598.23 | `BREAK_EVEN_ONLY_FEE_UNVERIFIED` | 488.23 |
| `hyrotrader:two_step:phase_1` | `equity` | `PASS` | 1005.12 | 2383.49 | `Blended` | 93.50 [61.14, 99.25] | 952.66 | `BREAK_EVEN_ONLY_FEE_UNVERIFIED` | 842.66 |
| `hyrotrader:two_step:phase_1` | `equity` | `PASS` | 1005.12 | 2383.49 | `75%-Chop` | 93.92 [61.63, 99.33] | 962.84 | `BREAK_EVEN_ONLY_FEE_UNVERIFIED` | 852.84 |
| `vanta-trading:classic_1step_3pct:phase_1` | `balance` | `PASS` | 1005.12 | 1522.53 | `Blended` | 28.84 [9.43, 61.19] | 366.26 | `BREAK_EVEN_ONLY_FEE_UNVERIFIED` | 256.26 |
| `vanta-trading:classic_1step_3pct:phase_1` | `balance` | `PASS` | 1005.12 | 1522.53 | `75%-Chop` | 29.36 [9.71, 61.64] | 365.13 | `BREAK_EVEN_ONLY_FEE_UNVERIFIED` | 255.13 |
| `propr:classic_1step:phase_1` | `balance` | `PASS` | 1005.12 | 1522.53 | `Blended` | 31.06 [10.61, 63.10] | 313.35 | `BREAK_EVEN_ONLY_FEE_UNVERIFIED` | 203.35 |
| `propr:classic_1step:phase_1` | `balance` | `PASS` | 1005.12 | 1522.53 | `75%-Chop` | 29.44 [9.75, 61.71] | 294.94 | `BREAK_EVEN_ONLY_FEE_UNVERIFIED` | 184.94 |

## Qualification and caveats

### Three-line caveat footer
- CFT trading restrictions and account framing require independent compliance confirmation.
- Live broker execution, funding, capacity and PIT source continuity are UNPROVEN.
- No deployments, allocations, orders, service or registry writes occurred.
