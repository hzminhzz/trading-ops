---
firm: crypto-fund-trader
account_usd: 10000
status: RESEARCH
top_candidate: NOT_SCIENTIFICALLY_QUALIFIED
script: studies/cft_20261116_exhaustive.py
---

# CFT canonical 5m portfolio research — May–November 2026

Full roster: 195; SHA256 c4abfa90556928fd9abe7387a7853078d6e73ed6ecf595eeaa6ff5a19f2433fd. Train May 1–Sep 1; OOS Sep 1–Nov 1 (exclusive).

All figures are retrospective modeled follower net returns, not live Bybit fills or PIT portfolio selection.

| Train rank | Members | Frozen k | Train new net/static floor DD | OOS new net/static floor DD | OOS full net/static floor DD | OOS worst daily | OOS sweeps | 90d MC 1x blended pass | Triad A conflict coin-h |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | W#111, W#174, W#179, Whale_0x5f94 | W#111=0.1095061, W#174=1.249331, W#179=0.05938938, Whale_0x5f94=0.0003821126 | +1600 / -66 | +1436 / -14 | +1869 / -77 | -175 | 0 | 92.66% | 1331.2 |
| 2 | W#1, W#174, W#264, Whale_0x5f94 | W#1=0.008003193, W#174=1.249331, W#264=3.322849, Whale_0x5f94=0.0006449343 | +1519 / -72 | +834 / -15 | +933 / -128 | -293 | 0 | 98.10% | 1852.6 |
| 3 | W#174, W#252, W#255, Whale_0x5f94 | W#174=1.249331, W#252=0.07547464, W#255=0.00121063, Whale_0x5f94=0.0006449343 | +1236 / -71 | +839 / -51 | +2498 / -149 | -176 | 0 | 87.50% | 2288.2 |

## Frozen sizing audit
The optimizer was fit exclusively to Train. Primary frozen k is capped at 50% individual wallet risk baselines; uncapped optimizer feasibility is diagnostic only.
- **Pod 1**: W#111 baseline=0.1095061, optimized=0.1095061 (1.0x), W#174 baseline=1.2493308, optimized=1.2493308 (1.0x), W#179 baseline=0.059389376, optimized=0.059389376 (1.0x), Whale_0x5f94 baseline=0.00064493427, optimized=0.00038211257 (0.6x); binding=DailyCVaR_99; uncapped-optimizer-feasible=True; min-lot warnings: W#174: q10_order=$7.52<$10, W#179: q10_order=$8.05<$10.
- **Pod 2**: W#1 baseline=0.0080031931, optimized=0.0080031931 (1.0x), W#174 baseline=1.2493308, optimized=1.2493308 (1.0x), W#264 baseline=3.3228494, optimized=3.3228494 (1.0x), Whale_0x5f94 baseline=0.00064493427, optimized=0.00064493427 (1.0x); binding=DailyCVaR_99; uncapped-optimizer-feasible=True; min-lot warnings: W#174: q10_order=$7.87<$10.
- **Pod 3**: W#174 baseline=1.2493308, optimized=1.2493308 (1.0x), W#252 baseline=0.07547464, optimized=0.07547464 (1.0x), W#255 baseline=0.0015395392, optimized=0.0012106302 (0.8x), Whale_0x5f94 baseline=0.00064493427, optimized=0.00064493427 (1.0x); binding=DailyCVaR_99; uncapped-optimizer-feasible=True; min-lot warnings: W#174: q10_order=$7.70<$10, W#255: q10_order=$2.78<$10.

## Monte Carlo sensitivity: strict new-position-only
Stationary 6-day blocks; 5,000 resamples per cell, 90-day horizon. These figures measure path-resampling sensitivity, not independently executable challenge success.

| Pod | Multiplier | Blended pass / static fail / daily fail | 75%-chop pass / static fail / daily fail | Median pass days (blended) | Avg Guardian sweeps (blended) |
|---|---:|---:|---:|---:|---:|
| 1 | 1.0x | 92.66% / 4.18% / 0.00% | 78.68% / 13.68% / 0.00% | 26.0 | 0.57 |
| 1 | 1.5x | 65.08% / 31.96% / 0.00% | 40.42% / 55.96% / 0.00% | 21.0 | 1.24 |
| 1 | 2.0x | 80.96% / 18.32% / 0.00% | 64.56% / 34.10% / 0.00% | 16.0 | 0.99 |
| 2 | 1.0x | 98.10% / 0.00% / 0.00% | 97.80% / 0.00% / 0.00% | 56.0 | 0.00 |
| 2 | 1.5x | 62.86% / 4.36% / 0.00% | 45.92% / 10.00% / 0.00% | 49.0 | 1.06 |
| 2 | 2.0x | 84.16% / 2.48% / 0.00% | 70.14% / 6.34% / 0.00% | 40.0 | 0.83 |
| 3 | 1.0x | 87.50% / 0.68% / 0.00% | 69.44% / 2.40% / 0.00% | 41.0 | 0.00 |
| 3 | 1.5x | 84.04% / 7.10% / 0.00% | 62.94% / 17.20% / 0.00% | 31.0 | 0.62 |
| 3 | 2.0x | 69.40% / 0.12% / 30.46% | 55.96% / 0.16% / 43.82% | 19.0 | 0.00 |

## OOS cost sensitivity (frozen new-only)
| Pod | 6.5 bps net | 10 bps net | 15 bps net | 15 bps verdict |
|---|---:|---:|---:|---|
| W#111 + W#174 + W#179 + Whale_0x5f94 | +1436 | +1252 | +990 | PASS |
| W#1 + W#174 + W#264 + Whale_0x5f94 | +834 | +746 | +621 | TIMEOUT |
| W#174 + W#252 + W#255 + Whale_0x5f94 | +839 | +744 | +609 | PASS |

## Model and coverage
- 184 complete daily Reservoir archives; bd2badb79973aaf4be37aebaacafe1e484628359e947fa9e51a7ec0f84f33a55 content digest.
- 195 roster wallets screened; 96 passed initial activity/symbol/account-value checks; 128 historically eligible for price screening, 128 price-replayed; 30 Train-shortlisted. Every price-eligible wallet was financially replayed on Train.
- Symbol suitability determined from Train-only fill history; selection did not filter on OOS-only symbols. Current Bybit availability is a retrospective mapping, not a historical snapshot.
- Prices: local Hyperliquid 1m OHLCV, local cached Vike fallback; trade at 1m open after 5m checkpoint and mark by checkpoint T+4 minutes, stale <=180m.
- Source: reconstructed Reservoir start_position + fills; pair-local stale-mark termination via canonical_replay.
- Test: Train-only multi-style clustering, dual-barrier sizing and frozen k; MC 5000 paths, stationary 6-day blocks, 90-day horizon.

## Guardian, PM03 and conflicts
- Candidate 1: first-barrier PASS, peak-DD -254.26, worst daily -174.69, sweep count 0; PM03 drift 0.0; Triad A reverse intervals 45.
- Candidate 2: first-barrier PASS, peak-DD -303.3, worst daily -292.52, sweep count 0; PM03 drift 0.0; Triad A reverse intervals 42.
- Candidate 3: first-barrier PASS, peak-DD -330.82, worst daily -176.09, sweep count 0; PM03 drift -0.0; Triad A reverse intervals 35.
- Triad A reference: new-only full-window 466.73, full-target 1075.84; static k source unverified live.

## Qualification and rank
- **Primary research-only:** Candidate 1 (highest OOS strict-new follower net); **alternative 1:** Candidate 2 (highest 1x modeled blended MC pass, lower OOS net); **alternative 2:** Candidate 3 (positive OOS net, lower modeled pass and greater Triad A conflict). All are research-only.
- All three have historical opposite-position overlaps with Triad A; do not run simultaneously on a reverse-trading-restricted account.
- All historically eligible wallets passed through Train financial replay, subject to documented price coverage. The 30-wallet shortlist and at most 12 structurally distinct candidate combos are bounded exploration, not global portfolio optimality.
- These are Train ranks; no candidate k or ranking used OOS until the freeze file was written.
- All three pods have native-min-lot warnings; primary k respects individual 50%-risk calibration. Bybit minimum quantity/notional and step-size order-rejection simulation remains UNPROVEN.
- If no candidate both survives OOS and has positive OOS net, no portfolio is economically qualified.
- PIT roster and live Bybit fills unavailable; research can support only RETROSPECTIVE_SCENARIO status.
- Full MC matrix, each wallet's eligibility/rejection reasons, trace hashes and all frozen parameters are in the companion JSON files.

## Unmet qualification gates
- CPCV/PBO was not executed. The optimizer evaluated 12 distinct combinations, below the 20–40 target.
- Bybit minimum order quantity, step size and minimum-notional rejection are not integrated into follower replay.
- 10 and 15 bps cost sensitivity is retrospective stress testing, not a new independent holdout.
- Entry-delay, leave-one-out, coin-removal and point-in-time roster stress tests are incomplete.
- Every finalist includes W#174 and Whale_0x5f94, limiting diversification.
- Status: **ENGINEERING_VERIFIED** for scoped research, **RETROSPECTIVE_SCENARIO** for PnL, **NOT SCIENTIFICALLY QUALIFIED**, **NOT ECONOMICALLY QUALIFIED**, **NOT TRADING AUTHORIZED**.

### Three-line caveat footer
- CFT trading restrictions and account framing require independent compliance confirmation.
- Live broker execution, funding, capacity and PIT source continuity are UNPROVEN.
- No deployments, allocations, orders, service or registry writes occurred.
