---
firm: crypto-fund-trader
account_usd: 10000
status: RESEARCH
top_candidate: NOT_SCIENTIFICALLY_QUALIFIED
script: studies/2026-11-16__cft__10k__canonical-full-universe-v4.py
---

# CFT canonical 5m portfolio research (v4) — May–November 2026

Full roster: 195; SHA256 `c4abfa90556928fd9abe7387a7853078d6e73ed6ecf595eeaa6ff5a19f2433fd`. Train `2026-05-01T00:00Z`–`2026-09-01T00:00Z`; OOS `2026-09-01T00:00Z`–`2026-11-01T00:00Z` (exclusive). Runtime: **818.61s**.

Separated eligibility screens — **historical (`scope=historical`)**: 189 / 195 Train-eligible; **live deployable (`scope=live`, Bybit + HL AV >= $1,000)**: 97 / 195.

All figures are retrospective modeled follower net returns, not live Bybit fills or PIT portfolio selection.

## Candidate summary (primary `$15` deadband, `1x` cost)

| Train rank | Members | Live deployable | Frozen k | Train new net/static DD | OOS new net/static DD | OOS full net/static DD | OOS worst daily | Peak gross lev | OOS sweeps | 90d MC 1x blended pass | Triad A conflict coin-h |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | W#107, W#111, W#190, W#312 | NO | W#107=0.0602298, W#111=0.0464517, W#190=0.00355981, W#312=0.0193923 | +552 / -26 | +628 / -7 | +633 / -19 | -105 | 0.62x | 0 | 67.38% | 54.8 |
| 2 | W#152, W#312, W#94 | NO | W#152=0.00133864, W#312=0.0389313, W#94=0.778943 | +156 / -53 | +133 / -9 | +133 / -9 | -27 | 0.10x | 0 | 0.00% | 236.6 |
| 3 | W#111, W#152, W#312 | YES | W#111=0.108566, W#152=0.00480224, W#312=0.0327334 | +528 / -46 | +1128 / -15 | +1139 / -43 | -163 | 0.93x | 0 | 94.46% | 44.7 |

## Deadband (`$15` vs `$30`) & Cost Multiplier (`1x` / `2x` / `3x`) Stress + 1m Participation

| Pod | Policy | Deadband | 1x net / static DD | 2x net / static DD | 3x net / static DD | Peak gross lev | 1m vol cov | 1m participation (med / p95 / max) | Trades >1% vol |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | new | $15 | +628 / -7 | +482 / -8 | +337 / -9 | 0.62x | 99.6% | 0.046% / 1.25% / 43.17% (VVV) | 69 |
| 1 | new | $30 | +629 / -7 | +484 / -8 | +338 / -9 | 0.62x | 99.6% | 0.052% / 1.41% / 43.17% (VVV) | 69 |
| 1 | full | $15 | +633 / -19 | +487 / -21 | +340 / -24 | 0.62x | 99.6% | 0.046% / 1.26% / 43.17% (VVV) | 70 |
| 1 | full | $30 | +634 / -19 | +488 / -21 | +342 / -24 | 0.62x | 99.6% | 0.052% / 1.43% / 43.17% (VVV) | 70 |
| 2 | new | $15 | +133 / -9 | +115 / -9 | +97 / -9 | 0.10x | 99.6% | 0.042% / 1.39% / 250.00% (XMR) | 18 |
| 2 | new | $30 | +131 / -9 | +113 / -10 | +95 / -10 | 0.10x | 99.5% | 0.035% / 1.41% / 361.11% (XMR) | 17 |
| 2 | full | $15 | +133 / -9 | +115 / -9 | +97 / -9 | 0.10x | 99.6% | 0.042% / 1.39% / 250.00% (XMR) | 18 |
| 2 | full | $30 | +131 / -9 | +113 / -10 | +95 / -10 | 0.10x | 99.5% | 0.035% / 1.41% / 361.11% (XMR) | 17 |
| 3 | new | $15 | +1128 / -15 | +936 / -16 | +743 / -17 | 0.93x | 99.6% | 0.066% / 2.21% / 96.96% (XRP) | 79 |
| 3 | new | $30 | +1124 / -15 | +931 / -16 | +738 / -17 | 0.93x | 99.5% | 0.069% / 2.24% / 96.96% (XRP) | 79 |
| 3 | full | $15 | +1139 / -43 | +945 / -47 | +751 / -51 | 0.93x | 99.6% | 0.067% / 2.24% / 96.96% (XRP) | 81 |
| 3 | full | $30 | +1135 / -43 | +941 / -47 | +747 / -51 | 0.93x | 99.6% | 0.069% / 2.26% / 96.96% (XRP) | 81 |

## Perpetual Funding Sensitivity (`$15` deadband, `1x` cost)

| Pod | Policy | USDT proxy funding ($) | Proxy funding-adj net ($) | Covered coins | +1 bp/8h carry ($) | +1 bp/8h adj net ($) |
|---|---|---:|---:|---:|---:|---:|
| 1 | new | -4.64 | +623.61 | 13/16 | -10.55 | +617.70 |
| 1 | full | -4.73 | +628.39 | 13/16 | -10.77 | +622.35 |
| 2 | new | -2.83 | +130.12 | 15/15 | -4.96 | +127.98 |
| 2 | full | -2.83 | +130.12 | 15/15 | -4.96 | +127.98 |
| 3 | new | -10.69 | +1117.76 | 7/9 | -24.16 | +1104.30 |
| 3 | full | -10.91 | +1128.52 | 7/9 | -24.68 | +1114.74 |

## Frozen sizing & pre-clustering `$15` min-notional audit
The optimizer fits exclusively on Train (`2026-05-01..2026-09-01`). The 50%-wallet-risk baseline equals `2%` portfolio drawdown (`$200` on `$10,000`, split evenly per wallet); optimized `k` is capped at `<= 1.0x` of each wallet's baseline, and wallets with baseline `q10_order < $15` are filtered prior to UPGMA clustering.
- **Pod size 3**: status=`OK`, passed=58, rejected=96 (below `$15` baseline filter=73).
- **Pod size 4**: status=`OK`, passed=45, rejected=109 (below `$15` baseline filter=91).
- **Pod 1**: W#107 baseline=0.06257607, optimized=0.06022984 (0.96x), W#111 baseline=0.08142433, optimized=0.04645172 (0.57x), W#190 baseline=0.003559812, optimized=0.003559812 (1.00x), W#312 baseline=0.02919851, optimized=0.0193923 (0.66x); binding=BaselineCap(DailyCVaR_99); feasible=True; live-excluded={'W#190': ['REJECT_UNMAPPED_MATERIAL', 'WITHDRAWN_LT_1000']}; min-lot warnings: none.
- **Pod 2**: W#152 baseline=0.007173115, optimized=0.001338639 (0.19x), W#312 baseline=0.03893134, optimized=0.03893134 (1.00x), W#94 baseline=2.0731, optimized=0.7789426 (0.38x); binding=BaselineCap(Hard_Daily); feasible=True; live-excluded={'W#94': ['WITHDRAWN_LT_1000']}; min-lot warnings: none.
- **Pod 3**: W#111 baseline=0.1085658, optimized=0.1085658 (1.00x), W#152 baseline=0.007173115, optimized=0.004802245 (0.67x), W#312 baseline=0.03893134, optimized=0.03273342 (0.84x); binding=BaselineCap(DailyCVaR_99); feasible=True; all live-deployable; min-lot warnings: none.
- **Live-deployable subset (`scope=live`) pod audit**: live shortlist=79, live-only candidates=3, live-only CPCV PBO=0.7.

## Monte Carlo sensitivity (`new` policy, `$15` and `$30` deadbands)
Stationary 6-day blocks; 5,000 resamples (or configured paths), 90-day horizon.

| Pod | Deadband | Multiplier | Blended pass / static fail / daily fail | 75%-chop pass / static fail / daily fail | Median pass days (blended) | Avg sweeps (blended) |
|---|---|---:|---:|---:|---:|---:|
| 1 | $15 | 1.0x | 67.38% / 0.00% / 0.00% | 57.60% / 0.00% / 0.00% | 67.0 | 0.00 |
| 1 | $15 | 1.5x | 92.60% / 0.00% / 0.00% | 87.34% / 0.00% / 0.00% | 49.0 | 0.00 |
| 1 | $15 | 2.0x | 97.86% / 0.00% / 0.00% | 95.14% / 0.00% / 0.00% | 38.0 | 0.00 |
| 1 | $30 | 1.0x | 67.84% / 0.00% / 0.00% | 57.16% / 0.00% / 0.00% | 67.0 | 0.00 |
| 1 | $30 | 1.5x | 92.70% / 0.00% / 0.00% | 86.20% / 0.00% / 0.00% | 50.0 | 0.00 |
| 1 | $30 | 2.0x | 97.64% / 0.00% / 0.00% | 94.02% / 0.00% / 0.00% | 38.0 | 0.00 |
| 2 | $15 | 1.0x | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% | -1.0 | 0.00 |
| 2 | $15 | 1.5x | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% | -1.0 | 0.00 |
| 2 | $15 | 2.0x | 0.08% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% | 78.5 | 0.00 |
| 2 | $30 | 1.0x | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% | -1.0 | 0.00 |
| 2 | $30 | 1.5x | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% | -1.0 | 0.00 |
| 2 | $30 | 2.0x | 0.12% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% | 85.0 | 0.00 |
| 3 | $15 | 1.0x | 94.46% / 0.00% / 0.00% | 82.30% / 0.00% / 0.00% | 41.0 | 0.00 |
| 3 | $15 | 1.5x | 98.88% / 0.00% / 0.00% | 92.46% / 0.00% / 0.00% | 29.0 | 0.00 |
| 3 | $15 | 2.0x | 94.18% / 1.48% / 0.00% | 91.36% / 0.82% / 0.00% | 26.0 | 0.53 |
| 3 | $30 | 1.0x | 94.68% / 0.00% / 0.00% | 81.20% / 0.00% / 0.00% | 41.0 | 0.00 |
| 3 | $30 | 1.5x | 98.80% / 0.00% / 0.00% | 92.42% / 0.00% / 0.00% | 29.0 | 0.00 |
| 3 | $30 | 2.0x | 93.66% / 1.66% / 0.00% | 90.74% / 0.58% / 0.00% | 26.0 | 0.54 |

## Model, pipeline, and coverage
- 184 complete daily Reservoir archives; `bd2badb79973aaf4be37aebaacafe1e484628359e947fa9e51a7ec0f84f33a55` content digest.
- UTC Polars lazy scan + CSE bounds: `min_ts_ms=1777593625884`, `max_ts_ms=1793491179279` (strict UTC, zero DuckDB).
- Screening separation: 195 roster wallets -> 189 `scope=historical` eligible vs 97 `scope=live` deployable; 154 positive Train survivors shortlisted (79 live-deployable).
- Leak-free CPCV PBO across frozen candidates: **0.15** (20 purged/embargoed splits).
- Single-pass Train replay reused directly into `build_tensor_from_unit_replays` and persisted to gitignored `data/v4_cache/*.parquet`.

## Guardian, PM03, and Triad A conflicts
- **Candidate 1**: verdict=`TIMEOUT`, peak-DD=-129.16, worst daily=-105.14, sweeps=0, peak lev=0.62x; PM03 7d drift=0.0; Triad A reverse intervals=43 (54.75 coin-h).
- **Candidate 2**: verdict=`TIMEOUT`, peak-DD=-34.34, worst daily=-26.9, sweeps=0, peak lev=0.10x; PM03 7d drift=-0.0; Triad A reverse intervals=24 (236.58 coin-h).
- **Candidate 3**: verdict=`PASS`, peak-DD=-227.93, worst daily=-162.85, sweeps=0, peak lev=0.93x; PM03 7d drift=-0.0; Triad A reverse intervals=10 (44.67 coin-h).
- **Triad A reference (`$15` deadband, `1x` cost)**: `new`=+561.89, `full`=+1124.90; static `k` source unverified live.

## Qualification and caveats
- Qualification gate requires `PBO < 0.50`, positive OOS net under `$15`/`$30` deadbands and `2x` cost stress, `live_deployable=True`, and prospective CWSA paper validation in `copytrade-pm`.
- Companion JSON and gitignored Parquet caches store full screening flags, CPCV logits, daily blocks, and equity curves.

### Three-line caveat footer
- CFT trading restrictions and account framing require independent compliance confirmation.
- Live broker execution, funding, capacity and PIT source continuity are UNPROVEN.
- No deployments, allocations, orders, service or registry writes occurred.
