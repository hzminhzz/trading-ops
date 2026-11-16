---
firm: crypto-fund-trader
account_usd: 10000
status: RESEARCH
top_candidate: NOT_SCIENTIFICALLY_QUALIFIED
script: studies/2026-11-16__cft__10k__canonical-full-universe-v4.py
---

# CFT canonical 5m portfolio research (v4) — May–November 2026

Full roster: 195; SHA256 `c4abfa90556928fd9abe7387a7853078d6e73ed6ecf595eeaa6ff5a19f2433fd`. Train `2026-05-01T00:00Z`–`2026-09-01T00:00Z`; OOS `2026-09-01T00:00Z`–`2026-11-01T00:00Z` (exclusive). Runtime: **684.85s**.

Separated eligibility screens — **historical (`scope=historical`)**: 189 / 195 Train-eligible; **live deployable (`scope=live`, Bybit + HL AV >= $1,000)**: 98 / 195.

All figures are retrospective modeled follower net returns, not live Bybit fills or PIT portfolio selection.

## Candidate summary (primary `$15` deadband, `1x` cost)

| Train rank | Members | Live deployable | Frozen k | Train new net/static DD | OOS new net/static DD | OOS full net/static DD | OOS worst daily | Peak gross lev | OOS sweeps | 90d MC 1x blended pass | Triad A conflict coin-h |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | W#107, W#111, W#190, W#312 | NO | W#107=0.0602298, W#111=0.0464517, W#190=0.00355981, W#312=0.0193923 | +560 / -27 | +632 / -7 | +637 / -18 | -106 | 0.63x | 0 | 68.16% | 54.8 |
| 2 | W#152, W#312, W#94 | NO | W#152=0.00133864, W#312=0.0389313, W#94=0.778943 | +160 / -55 | +125 / -8 | +125 / -8 | -29 | 0.11x | 0 | 0.00% | 236.6 |
| 3 | W#111, W#152, W#312 | YES | W#111=0.108566, W#152=0.00480224, W#312=0.0327334 | +533 / -46 | +1130 / -15 | +1140 / -42 | -163 | 0.93x | 0 | 94.44% | 44.7 |

## Deadband (`$15` vs `$30`) & Cost Multiplier (`1x` / `2x` / `3x`) Stress + 1m Participation

| Pod | Policy | Deadband | 1x net / static DD | 2x net / static DD | 3x net / static DD | Peak gross lev | 1m vol cov | 1m participation (med / p95 / max) | Trades >1% vol |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | new | $15 | +632 / -7 | +483 / -8 | +334 / -9 | 0.63x | 99.6% | 0.043% / 1.22% / 43.18% (VVV) | 71 |
| 1 | new | $30 | +634 / -7 | +486 / -8 | +337 / -9 | 0.62x | 99.6% | 0.047% / 1.34% / 43.18% (VVV) | 71 |
| 1 | full | $15 | +637 / -18 | +487 / -21 | +338 / -24 | 0.63x | 99.6% | 0.043% / 1.23% / 43.18% (VVV) | 72 |
| 1 | full | $30 | +639 / -18 | +490 / -21 | +341 / -24 | 0.62x | 99.6% | 0.047% / 1.35% / 43.18% (VVV) | 72 |
| 2 | new | $15 | +125 / -8 | +104 / -9 | +83 / -10 | 0.11x | 99.7% | 0.025% / 1.13% / 246.67% (XMR) | 19 |
| 2 | new | $30 | +123 / -9 | +103 / -10 | +83 / -10 | 0.11x | 99.6% | 0.022% / 1.28% / 367.83% (XMR) | 17 |
| 2 | full | $15 | +125 / -8 | +104 / -9 | +83 / -10 | 0.11x | 99.7% | 0.025% / 1.13% / 246.67% (XMR) | 19 |
| 2 | full | $30 | +123 / -9 | +103 / -10 | +83 / -10 | 0.11x | 99.6% | 0.022% / 1.28% / 367.83% (XMR) | 17 |
| 3 | new | $15 | +1130 / -15 | +934 / -16 | +738 / -17 | 0.93x | 99.6% | 0.066% / 2.21% / 96.93% (XRP) | 80 |
| 3 | new | $30 | +1125 / -15 | +929 / -16 | +734 / -17 | 0.93x | 99.6% | 0.068% / 2.22% / 96.93% (XRP) | 80 |
| 3 | full | $15 | +1140 / -42 | +944 / -47 | +747 / -51 | 0.93x | 99.6% | 0.067% / 2.23% / 96.93% (XRP) | 82 |
| 3 | full | $30 | +1136 / -42 | +939 / -47 | +742 / -51 | 0.93x | 99.6% | 0.068% / 2.25% / 96.93% (XRP) | 82 |

## Perpetual Funding Sensitivity (`$15` deadband, `1x` cost)

| Pod | Policy | USDT proxy funding ($) | Proxy funding-adj net ($) | Covered coins | +1 bp/8h carry ($) | +1 bp/8h adj net ($) |
|---|---|---:|---:|---:|---:|---:|
| 1 | new | -4.71 | +627.59 | 13/16 | -10.69 | +621.61 |
| 1 | full | -4.81 | +632.11 | 13/16 | -10.92 | +626.01 |
| 2 | new | -2.97 | +121.61 | 15/15 | -5.25 | +119.34 |
| 2 | full | -2.97 | +121.61 | 15/15 | -5.25 | +119.34 |
| 3 | new | -10.81 | +1118.82 | 7/9 | -24.33 | +1105.30 |
| 3 | full | -11.03 | +1129.40 | 7/9 | -24.86 | +1115.57 |

## Frozen sizing & pre-clustering `$15` min-notional audit
The optimizer fits exclusively on Train (`2026-05-01..2026-09-01`). The 50%-wallet-risk baseline equals `2%` portfolio drawdown (`$200` on `$10,000`, split evenly per wallet); optimized `k` is capped at `<= 1.0x` of each wallet's baseline, and wallets with baseline `q10_order < $15` are filtered prior to UPGMA clustering.
- **Pod size 3**: status=`OK`, passed=58, rejected=96 (below `$15` baseline filter=73).
- **Pod size 4**: status=`OK`, passed=45, rejected=109 (below `$15` baseline filter=91).
- **Pod 1**: W#107 baseline=0.06257607, optimized=0.06022984 (0.96x), W#111 baseline=0.08142433, optimized=0.04645172 (0.57x), W#190 baseline=0.003559812, optimized=0.003559812 (1.00x), W#312 baseline=0.02919851, optimized=0.0193923 (0.66x); binding=BaselineCap(DailyCVaR_99); feasible=True; live-excluded={'W#190': ['REJECT_UNMAPPED_MATERIAL', 'WITHDRAWN_LT_1000']}; min-lot warnings: none.
- **Pod 2**: W#152 baseline=0.007173115, optimized=0.001338639 (0.19x), W#312 baseline=0.03893134, optimized=0.03893134 (1.00x), W#94 baseline=2.0731, optimized=0.7789426 (0.38x); binding=BaselineCap(Hard_Daily); feasible=True; live-excluded={'W#94': ['WITHDRAWN_LT_1000']}; min-lot warnings: none.
- **Pod 3**: W#111 baseline=0.1085658, optimized=0.1085658 (1.00x), W#152 baseline=0.007173115, optimized=0.004802245 (0.67x), W#312 baseline=0.03893134, optimized=0.03273342 (0.84x); binding=BaselineCap(DailyCVaR_99); feasible=True; all live-deployable; min-lot warnings: none.
- **Live-deployable subset (`scope=live`) pod audit**: live shortlist=80, live-only candidates=3, live-only CPCV PBO=0.7.

## Monte Carlo sensitivity (`new` policy, `$15` and `$30` deadbands)
Stationary 6-day blocks; 5,000 resamples (or configured paths), 90-day horizon.

| Pod | Deadband | Multiplier | Blended pass / static fail / daily fail | 75%-chop pass / static fail / daily fail | Median pass days (blended) | Avg sweeps (blended) |
|---|---|---:|---:|---:|---:|---:|
| 1 | $15 | 1.0x | 68.16% / 0.00% / 0.00% | 57.84% / 0.00% / 0.00% | 66.0 | 0.00 |
| 1 | $15 | 1.5x | 92.72% / 0.00% / 0.00% | 87.40% / 0.00% / 0.00% | 49.0 | 0.00 |
| 1 | $15 | 2.0x | 97.86% / 0.00% / 0.00% | 95.14% / 0.00% / 0.00% | 38.0 | 0.00 |
| 1 | $30 | 1.0x | 68.58% / 0.00% / 0.00% | 57.74% / 0.00% / 0.00% | 67.0 | 0.00 |
| 1 | $30 | 1.5x | 92.88% / 0.00% / 0.00% | 86.02% / 0.00% / 0.00% | 49.0 | 0.00 |
| 1 | $30 | 2.0x | 97.66% / 0.00% / 0.00% | 93.96% / 0.00% / 0.00% | 38.0 | 0.00 |
| 2 | $15 | 1.0x | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% | -1.0 | 0.00 |
| 2 | $15 | 1.5x | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% | -1.0 | 0.00 |
| 2 | $15 | 2.0x | 0.06% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% | 79.0 | 0.00 |
| 2 | $30 | 1.0x | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% | -1.0 | 0.00 |
| 2 | $30 | 1.5x | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% | -1.0 | 0.00 |
| 2 | $30 | 2.0x | 0.08% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% | 84.0 | 0.00 |
| 3 | $15 | 1.0x | 94.44% / 0.00% / 0.00% | 81.92% / 0.00% / 0.00% | 41.0 | 0.00 |
| 3 | $15 | 1.5x | 98.86% / 0.00% / 0.00% | 92.48% / 0.00% / 0.00% | 28.0 | 0.00 |
| 3 | $15 | 2.0x | 94.10% / 1.48% / 0.00% | 91.24% / 0.84% / 0.00% | 26.0 | 0.53 |
| 3 | $30 | 1.0x | 94.64% / 0.00% / 0.00% | 80.90% / 0.00% / 0.00% | 41.0 | 0.00 |
| 3 | $30 | 1.5x | 98.82% / 0.00% / 0.00% | 92.38% / 0.00% / 0.00% | 28.0 | 0.00 |
| 3 | $30 | 2.0x | 93.60% / 1.68% / 0.00% | 90.64% / 0.62% / 0.00% | 25.0 | 0.54 |

## Model, pipeline, and coverage
- 184 complete daily Reservoir archives; `bd2badb79973aaf4be37aebaacafe1e484628359e947fa9e51a7ec0f84f33a55` content digest.
- UTC Polars lazy scan + CSE bounds: `min_ts_ms=1777593625884`, `max_ts_ms=1793491179279` (strict UTC, zero DuckDB).
- Screening separation: 195 roster wallets -> 189 `scope=historical` eligible vs 98 `scope=live` deployable; 154 positive Train survivors shortlisted (80 live-deployable).
- Leak-free CPCV PBO across frozen candidates: **0.15** (20 purged/embargoed splits).
- Single-pass Train replay reused directly into `build_tensor_from_unit_replays` and persisted to gitignored `data/v4_cache/*.parquet`.

## Guardian, PM03, and Triad A conflicts
- **Candidate 1**: verdict=`TIMEOUT`, peak-DD=-129.53, worst daily=-105.71, sweeps=0, peak lev=0.63x; PM03 7d drift=0.0; Triad A reverse intervals=43 (54.75 coin-h).
- **Candidate 2**: verdict=`TIMEOUT`, peak-DD=-37.76, worst daily=-28.95, sweeps=0, peak lev=0.11x; PM03 7d drift=-0.0; Triad A reverse intervals=24 (236.58 coin-h).
- **Candidate 3**: verdict=`PASS`, peak-DD=-227.94, worst daily=-163.11, sweeps=0, peak lev=0.93x; PM03 7d drift=-0.0; Triad A reverse intervals=10 (44.67 coin-h).
- **Triad A reference (`$15` deadband, `1x` cost)**: `new`=+566.92, `full`=+1132.19; static `k` source unverified live.

## Qualification and caveats
- Qualification gate requires `PBO < 0.50`, positive OOS net under `$15`/`$30` deadbands and `2x` cost stress, `live_deployable=True`, and prospective CWSA paper validation in `copytrade-pm`.
- Companion JSON and gitignored Parquet caches store full screening flags, CPCV logits, daily blocks, and equity curves.

### Three-line caveat footer
- CFT trading restrictions and account framing require independent compliance confirmation.
- Live broker execution, funding, capacity and PIT source continuity are UNPROVEN.
- No deployments, allocations, orders, service or registry writes occurred.
