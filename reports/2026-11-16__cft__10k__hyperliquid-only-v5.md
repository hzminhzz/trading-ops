---
firm: crypto-fund-trader
account_usd: 10000
status: RESEARCH
top_candidate: NOT_SCIENTIFICALLY_QUALIFIED
script: studies/cft_20261116_exhaustive.py
---

# CFT-risk 5m research / hyperliquid market — May–November 2026

Full roster: 195; SHA256 8b61701784f8aa8576381456e484e13dd7f73f6ccab93929b478489f88b0cd02. Train May 1–Sep 1; OOS Sep 1–Nov 1 (exclusive).

All figures are retrospective modeled Hyperliquid-source follower net returns, not live broker fills or PIT portfolio selection.
Backtest venue = hyperliquid; challenge budget is a risk scenario only. Bybit listings and minimum order rules are NOT historical selection inputs when Hyperliquid-only.

| Train rank | Members | Frozen k | Train new net/static floor DD | OOS new net/static floor DD | OOS full net/static floor DD | OOS worst daily | OOS sweeps | 90d MC 1x blended pass | Triad A conflict coin-h |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | W#174, W#257, W#280, W#94 | W#174=1.249331, W#257=0.01671055, W#280=0.08222279, W#94=3.96324 | +2374 / -65 | +1007 / -109 | +854 / -191 | -331 | 0 | 98.94% | 1293.4 |
| 2 | W#1, W#174, W#179, W#280 | W#1=0.008003193, W#174=1.249331, W#179=0.05436783, W#280=0.08222279 | +2170 / -72 | +677 / -158 | +245 / -513 | -351 | 0 | 77.66% | 1399.7 |
| 3 | W#1, W#174, W#94, Whale_0x5f94 | W#1=0.008003193, W#174=1.249331, W#94=3.96324, Whale_0x5f94=0.0006594472 | +1906 / -73 | +1236 / -15 | +1709 / -145 | -272 | 0 | 100.00% | 2026.4 |
| 4 | W#174, W#179, W#257, Whale_0x5f94 | W#174=1.249331, W#179=0.05938938, W#257=0.01671055, Whale_0x5f94=0.0004117575 | +1817 / -34 | +765 / -112 | +1096 / -89 | -360 | 0 | 80.92% | 3010.1 |

## Frozen sizing audit
The optimizer was fit exclusively to Train. Primary frozen k is capped at 50% individual wallet risk baselines; uncapped optimizer feasibility is diagnostic only.
- **Pod 1**: W#174 baseline=1.2493308, optimized=1.2493308 (1.0x), W#257 baseline=0.016710555, optimized=0.016710555 (1.0x), W#280 baseline=0.082222785, optimized=0.082222785 (1.0x), W#94 baseline=3.9632403, optimized=3.9632403 (1.0x); binding=DailyCVaR_99; uncapped-optimizer-feasible=True; min-lot warnings: W#174: q10_order=$6.47<$10.
- **Pod 2**: W#1 baseline=0.0080031931, optimized=0.0080031931 (1.0x), W#174 baseline=1.2493308, optimized=1.2493308 (1.0x), W#179 baseline=0.059389376, optimized=0.054367829 (0.9x), W#280 baseline=0.082222785, optimized=0.082222785 (1.0x); binding=DailyCVaR_99; uncapped-optimizer-feasible=True; min-lot warnings: W#174: q10_order=$7.96<$10, W#179: q10_order=$5.60<$10.
- **Pod 3**: W#1 baseline=0.0080031931, optimized=0.0080031931 (1.0x), W#174 baseline=1.2493308, optimized=1.2493308 (1.0x), W#94 baseline=3.9632403, optimized=3.9632403 (1.0x), Whale_0x5f94 baseline=0.00065944723, optimized=0.00065944723 (1.0x); binding=DailyCVaR_99; uncapped-optimizer-feasible=True; min-lot warnings: W#174: q10_order=$8.77<$10.
- **Pod 4**: W#174 baseline=1.2493308, optimized=1.2493308 (1.0x), W#179 baseline=0.059389376, optimized=0.059389376 (1.0x), W#257 baseline=0.016710555, optimized=0.016710555 (1.0x), Whale_0x5f94 baseline=0.00065944723, optimized=0.00041175748 (0.6x); binding=DailyCVaR_99; uncapped-optimizer-feasible=True; min-lot warnings: W#174: q10_order=$7.22<$10, W#179: q10_order=$9.54<$10.

## Monte Carlo sensitivity: strict new-position-only
Stationary 6-day blocks; 5,000 resamples per cell, 90-day horizon. These figures measure path-resampling sensitivity, not independently executable challenge success.

| Pod | Multiplier | Blended pass / static fail / daily fail | 75%-chop pass / static fail / daily fail | Median pass days (blended) | Avg Guardian sweeps (blended) |
|---|---:|---:|---:|---:|---:|
| 1 | 1.0x | 98.94% / 0.00% / 0.00% | 98.94% / 0.00% / 0.00% | 45.0 | 0.00 |
| 1 | 1.5x | 88.50% / 2.80% / 0.00% | 82.52% / 5.74% / 0.00% | 30.0 | 0.69 |
| 1 | 2.0x | 66.08% / 0.00% / 33.92% | 59.98% / 0.00% / 40.02% | 17.0 | 0.00 |
| 2 | 1.0x | 77.66% / 1.54% / 0.00% | 73.46% / 3.18% / 0.00% | 51.0 | 0.00 |
| 2 | 1.5x | 25.62% / 61.36% / 0.00% | 10.70% / 84.24% / 0.00% | 34.0 | 1.15 |
| 2 | 2.0x | 34.70% / 22.92% / 41.08% | 20.78% / 34.22% / 44.42% | 22.0 | 0.33 |
| 3 | 1.0x | 100.00% / 0.00% / 0.00% | 100.00% / 0.00% / 0.00% | 38.0 | 0.00 |
| 3 | 1.5x | 95.90% / 0.36% / 0.00% | 93.98% / 1.16% / 0.00% | 31.0 | 0.62 |
| 3 | 2.0x | 98.70% / 0.56% / 0.00% | 97.72% / 1.28% / 0.00% | 22.0 | 0.45 |
| 4 | 1.0x | 80.92% / 3.02% / 0.00% | 79.24% / 5.38% / 0.00% | 50.0 | 0.00 |
| 4 | 1.5x | 19.36% / 63.44% / 0.00% | 6.32% / 87.12% / 0.00% | 42.0 | 1.18 |
| 4 | 2.0x | 27.78% / 23.98% / 46.20% | 15.34% / 36.44% / 47.50% | 29.0 | 0.29 |

## OOS cost sensitivity (frozen new-only)
| Pod | 6.5 bps net | 10 bps net | 15 bps net | 15 bps verdict |
|---|---:|---:|---:|---|
| W#174 + W#257 + W#280 + W#94 | +1007 | +881 | +700 | PASS |
| W#1 + W#174 + W#179 + W#280 | +677 | +553 | +375 | TIMEOUT |
| W#1 + W#174 + W#94 + Whale_0x5f94 | +1236 | +1127 | +971 | PASS |
| W#174 + W#179 + W#257 + Whale_0x5f94 | +765 | +635 | +449 | TIMEOUT |

## Model and coverage
- 184 complete daily Reservoir archives; bd2badb79973aaf4be37aebaacafe1e484628359e947fa9e51a7ec0f84f33a55 content digest.
- 195 registry wallets screened; 189 passed Train activity / Hyperliquid-only symbol checks; 189 historically eligible for price screening, 189 price-replayed; 30 Train-shortlisted. Every price-eligible wallet was financially replayed on Train.
- Historical wallet and symbol Train eligibility uses only Hyperliquid main-DEX perp fills before cutoff; no Bybit listing/account-value lookup is performed. OOS-only coins are admitted only AFTER freeze.
- Prices: local Hyperliquid 1m OHLCV, local cached Vike fallback; trade at 1m open after 5m checkpoint and mark by checkpoint T+4 minutes, stale <=180m.
- Source: reconstructed Reservoir start_position + fills; pair-local stale-mark termination via canonical_replay.
- Test: Train-only multi-style clustering, dual-barrier sizing and frozen k; MC 5000 paths, stationary 6-day blocks, 90-day horizon.

## Guardian, PM03 and conflicts
- Candidate 1: first-barrier PASS, peak-DD -419.16, worst daily -330.59, sweep count 0; PM03 drift -0.0; Triad A reverse intervals 41.
- Candidate 2: first-barrier TIMEOUT, peak-DD -827.42, worst daily -350.77, sweep count 0; PM03 drift 0.0; Triad A reverse intervals 34.
- Candidate 3: first-barrier PASS, peak-DD -289.14, worst daily -271.6, sweep count 0; PM03 drift -0.0; Triad A reverse intervals 51.
- Candidate 4: first-barrier TIMEOUT, peak-DD -868.14, worst daily -360.41, sweep count 0; PM03 drift UNPROVEN; Triad A reverse intervals 40.
- Triad A reference: new-only full-window 466.73, full-target 1075.84; static k source unverified live.

## Qualification and Train rank
- The numbered portfolios were ranked and frozen using TRAIN ONLY; the post-hoc OOS winner is not promoted to a new pre-cutoff decision.
- Largest already-opened historical OOS new-only net: candidate 3 (USD +1235.64); descriptive only.
- 4 frozen Train candidates, 16 explored Train combinations. Bounded search, not global portfolio optimality.
- All candidates have historical opposite-position overlaps with Triad A. Never combine them blindly on a reverse-position-restricted account.
- Each historical eligible wallet with Train pricing underwent financial replay; the 30-wallet shortlist is Train-only.
- OOS returns did not tune any frozen candidate k or membership.
- The optimizer's optional $10 notional warnings are diagnostics, not Hyperliquid execution constraints. Primary k respects individual 50%-risk calibration. Actual broker order rejection and follower fill economics remain UNPROVEN.
- If no candidate both survives OOS and has positive OOS net, no portfolio is economically qualified.
- The present-day wallet roster has no pre-September discovery proof, source first-known receipt clocks are unavailable historically, and September–October has already been inspected. This is RETROSPECTIVE_SCENARIO, not an independent prospective OOS.
- Full MC matrix, each wallet's eligibility/rejection reasons, trace hashes and all frozen parameters are in the companion JSON files.

## Unmet qualification gates
- CPCV/PBO was not executed; the bounded Train optimizer explored 16 combinations, below the 20–40 target.
- Venue-specific lot quantities, step sizes, and funding were not integrated into follower replay; none is a selection gate for Hyperliquid-only backtests.
- Cost sensitivities at 10 and 15 bps were tested on frozen memberships and k after OOS; they are stress tests, not new independent holdouts.
- Entry-delay, leave-one-out, coin-removal and independent PIT roster stress checks remain incomplete.
- Wallets shared by every frozen candidate: W#174. Shared-member risk is not proof of independence or broad style diversification.
- Formal statuses: ENGINEERING_VERIFIED for the scoped research code; RETROSPECTIVE_SCENARIO for returns; SCIENTIFICALLY_QUALIFIED=false; ECONOMICALLY_QUALIFIED=false; TRADING_AUTHORIZED=false.

### Three-line caveat footer
- CFT trading restrictions and account framing require independent compliance confirmation.
- Live broker execution, funding, capacity and PIT source continuity are UNPROVEN.
- No deployments, allocations, orders, service or registry writes occurred.
