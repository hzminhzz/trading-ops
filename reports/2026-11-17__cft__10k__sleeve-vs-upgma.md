---
firm: crypto-fund-trader
account_usd: 10000
status: RESEARCH
top_candidate: NOT_SCIENTIFICALLY_QUALIFIED
script: studies/2026-11-17__cft__10k__sleeve-vs-upgma.py
---

# Structural episode sleeves vs UPGMA — matched empirical test (CFT $10k)

This is an exploratory retrospective rerun, not an independently preregistered study. The script writes candidate definitions before its OOS replay, but earlier results were already inspected and the gap gate was revised. Figures are modeled, not live or PIT.

## Frozen design

- Train: `2026-05-01T00:00Z`–`2026-09-01T00:00Z`; OOS: `2026-09-01T00:00Z`–`2026-11-01T00:00Z` (exclusive).
- Roster SHA256 `fdfec255964b561f9c1f43eb64339d3a1f02786473974775361de96995ddedcb`; Train-eligible historical `37`; positive Train shortlist `30`; behavioral pass `11`.
- Identical for every method: `pod_size=3`, `$15` min baseline order, `2%` port-DD baseline cap, `$15` deadband, `1.0x` cost, strict `new` OOS.
- Strict `new` OOS is **regenerated**, not sliced: `candidate_targets` calls `reconstruct_positions_on_grid(..., policy='new', activation_ms=OOS_start)` on the raw fills. `UnitCopy5mTensor.slice_time` is not used for OOS.

### Method A — canonical UPGMA (baseline authority)

`stage1_audit_traders` -> `compute_crowding_distance_matrix` -> `cluster_candidates_upgma` into `pod_size` clusters -> top-2 per cluster by overall and chop Net/DD -> Cartesian combos (<=16) -> `size_pod_dual_barrier` -> rank `(feasible, net/DD, net)`.

### Method B — structural episode sleeves (proposed)

Holding-time buckets from **uncensored** per-symbol runs (left-censored at Train start and right-censored at Train end excluded; fallback disclosed per wallet): `intraday_tactical` <= 288 bars, `multi_day_swing` <= 2016, `positional` > 2016; `unlabeled` if active_frac < 0.02 or < 4 runs. Top-3 per sleeve by (chop-positive role tie-break, net/DD, Sharpe, alias). Admit a combo only if every pair has `dist >= 0.25`; an unknown pair (< 5 concurrent active days) additionally requires sized max DD <= 5% and worst day <= 2.5%. Greedy is not used; all admitted combos are sized and ranked deterministically.

### Method C — simple risk-screened baseline

Top-3 behavioral-pass wallets by Train `net/max-DD` (tie: Sharpe, alias), no clustering, same sizer and cap.

### Primary endpoints and decision rule (frozen)

Primary: OOS strict-new `full_window_net_usd`, `full_window_static_floor_dd_usd`, `worst_daily_usd` for each method's top-1 Train-ranked pod. A method dominates another only if OOS net is >=, static floor DD is >= (less negative), and worst day is >= on all three. No winner is declared without dominance over both others. `full` policy is a separately labeled sensitivity.

## Coverage and censoring

- UTC bounds `1777594205446`–`1793491136040`; daily archives `184`; missing source days `[]`.
- Shortlist `30`; behavioral pass `11`; sleeve counts `{"intraday_tactical": 8, "multi_day_swing": 2, "positional": 1}`; unlabeled `0`.
- Censored runs: left `0`, right `26`, uncensored `858`, fallback wallets `0`. The canonical extractor does not expose start/censor flags; labels use the disclosed recomputation.

### Roster drift vs canonical v4 (missing work)

The current live registry exposes **38 configured non-excluded wallets**, not the **195** used by the canonical v4 study (`roster_sha256 c4abfa90...`). Only 37 are Train-eligible, 30 reach the positive Train shortlist, and 11 pass the behavioral audit. The v4 replay is therefore **not reproducible from the current registry**, and this is a reduced-universe matched subtest, not a full-universe replication. The missing work is a re-run on the frozen 195-wallet roster, which is unavailable here without a point-in-time registry snapshot. The 38 configured wallets are the **current live selection**, not a point-in-time historical universe, so the sample is additionally survivorship- and selection-contaminated; every figure is conditional on that roster.

## Reproduction

```bash
cd /home/quant/checkouts/q3-style-sleeves
uv run python studies/2026-11-17__cft__10k__sleeve-vs-upgma.py
```

Runtime `113.66s`. Outputs: `reports/2026-11-17__cft__10k__sleeve-vs-upgma__freeze.json`, `...__results.json`, `...md`, and gitignored `data/v4_cache/...__inventory.json`.

## Results

### Search budgets and method status (frozen audit)

| Method | Status | Budget |
|---|---|---|
| A_upgma | OK | `{"cluster_pick_sizes": [1, 2, 2], "clusters": 3, "combos_admitted": 4, "combos_generated": 4, "combos_sized": 4}` |
| B_sleeve | OK | `{"combos_admitted": 6, "combos_generated": 6, "combos_rejected_crowded": 0, "combos_rejected_infeasible": 0, "combos_rejected_unknown_cap": 0, "picks_per_sleeve": {"intraday_tactical": 3, "multi_day_swing": 2, "positional": 1}, "sleeves_present": 3}` |
| C_baseline | OK | `{"combos_admitted": 1, "combos_generated": 1, "combos_sized": 1}` |

### Train gap and coin-universe diagnostics (selected members)

| Method | Alias | Train gap exits | Train unpriced steps | Train gap flagged | Train coins | OOS coins | OOS coins not in Train |
|---|---|---:|---:|---|---:|---:|---|
| A_upgma | W#107 | 0 | 0 | False | 9 | 9 | - |
| A_upgma | W111 | 0 | 0 | False | 7 | 13 | AVAX, BNB, NEAR, PONS, SAND, ZEC |
| A_upgma | W312 | 0 | 0 | False | 3 | 7 | AVAX, HYPE, NEAR, ZEC |
| B_sleeve | W#117 | 0 | 0 | False | 13 | 15 | PENDLE, UNI |
| B_sleeve | W#264 | 0 | 0 | False | 2 | 5 | DOGE, XMR, XRP |
| B_sleeve | Whale_0x5f94 | 1 | 1 | True | 26 | 29 | ATOM, BNB, ENA |
| C_baseline | W111 | 0 | 0 | False | 7 | 13 | AVAX, BNB, NEAR, PONS, SAND, ZEC |
| C_baseline | W#117 | 0 | 0 | False | 13 | 15 | PENDLE, UNI |
| C_baseline | W#107 | 0 | 0 | False | 9 | 9 | - |

### Train (in-sample, selection-biased)

| Method | Rank | Members | Seats | Train net $ | Train max DD $ | Train worst day $ | Net/DD | Unknown pairs |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| A_upgma | 1 | W#107, W111, W312 | 3 | 540.77 | 58.07 | 31.77 | 9.3131 | n/a |
| A_upgma | 2 | W#174, W111, W312 | 3 | 445.80 | 63.51 | 50.11 | 7.0194 | n/a |
| A_upgma | 3 | W#117, W#174, W312 | 3 | 335.31 | 67.44 | 52.23 | 4.9717 | n/a |
| B_sleeve | 1 | W#117, W#264, Whale_0x5f94 | 3 | 518.79 | 81.88 | 66.95 | 6.3359 | 0 |
| B_sleeve | 2 | M54, W#117, Whale_0x5f94 | 3 | 481.22 | 91.32 | 62.30 | 5.2694 | 0 |
| B_sleeve | 3 | M21, M54, Whale_0x5f94 | 3 | 361.90 | 71.50 | 35.52 | 5.0617 | 0 |
| C_baseline | 1 | W111, W#117, W#107 | 3 | 570.67 | 67.99 | 37.27 | 8.3939 | n/a |

### OOS strict new-position-only (`$15` deadband, `1x` cost)

| Method | Rank | Members | Net $ | Static floor DD $ | Peak-to-trough DD $ | Worst day $ | Verdict | Peak lev | Unpriced steps | Unknown pairs |
|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|
| A_upgma | 1 | W#107, W111, W312 | 797.90 | -9.17 | -213.85 | -171.34 | PASS | 0.75x | 0 | n/a |
| A_upgma | 2 | W#174, W111, W312 | 890.73 | -7.93 | -205.50 | -199.70 | PASS | 0.70x | 0 | n/a |
| A_upgma | 3 | W#117, W#174, W312 | 447.06 | -9.03 | -132.25 | -129.73 | TIMEOUT | 0.57x | 0 | n/a |
| B_sleeve | 1 | W#117, W#264, Whale_0x5f94 | 231.88 | -3.11 | -89.23 | -43.64 | TIMEOUT | 0.14x | 0 | 0 |
| B_sleeve | 2 | M54, W#117, Whale_0x5f94 | 198.37 | -5.60 | -61.66 | -28.72 | TIMEOUT | 0.17x | 1 | 0 |
| B_sleeve | 3 | M21, M54, Whale_0x5f94 | 146.63 | -20.29 | -54.61 | -34.29 | TIMEOUT | 0.19x | 2 | 0 |
| C_baseline | 1 | W111, W#117, W#107 | 765.69 | -9.17 | -211.22 | -169.91 | TIMEOUT | 0.75x | 0 | n/a |

- Boundary check: `new` has **0** nonzero netted targets at the first OOS bar for every method, confirming eligibility is regenerated at the OOS boundary. `full` carries pre-boundary inventory (see table below), so the two policies are not comparable.

### OOS `full` policy sensitivity (not comparable to strict new)

| Method | Members | Net $ | Static floor DD $ | Worst day $ | Boundary nonzero targets |
|---|---|---:|---:|---:|---:|
| A_upgma | W#107, W111, W312 | 804.14 | -24.17 | -171.34 | 2 |
| B_sleeve | W#117, W#264, Whale_0x5f94 | 301.42 | -36.85 | -70.27 | 19 |
| C_baseline | W111, W#117, W#107 | 771.40 | -30.05 | -169.91 | 3 |

### Gap diagnostics (OOS, strict new)

| Method | Unpriced steps | Forced gap exits | Deferred notional $ | Gap invalidated |
|---|---:|---:|---:|---|
| A_upgma | 0 | 0 | 0.00 | False |
| B_sleeve | 0 | 0 | 6094.76 | False |
| C_baseline | 0 | 0 | 0.00 | False |

### Coverage vs exposure gap (not equivalent)

| Coin | mark_valid_frac (all-window grid) |
|---|---:|
| AAVE | 1.0 |
| ADA | 1.0 |
| APT | 1.0 |
| ATOM | n/a |
| AVAX | 1.0 |
| BCH | 1.0 |
| BNB | 1.0 |
| BTC | 1.0 |
| CASHCAT | 0.61305 |
| CRV | 0.99943 |
| DOGE | 1.0 |
| DOT | 1.0 |
| EIGEN | 0.99657 |
| ENA | 1.0 |
| ETH | 1.0 |
| ETHFI | 1.0 |
| GRAM | 0.65874 |
| GRASS | 1.0 |
| HYPE | 1.0 |
| JTO | 0.99592 |
| LIT | 1.0 |
| MON | 0.99855 |
| MORPHO | 1.0 |
| NEAR | 1.0 |
| ONDO | 1.0 |
| OP | 0.99585 |
| PENDLE | 0.99532 |
| PONS | 0.33609 |
| PUMP | 1.0 |
| PURR | 0.99996 |
| SAND | n/a |
| SEI | 0.99783 |
| SOL | 1.0 |
| STRK | 1.0 |
| SUI | 1.0 |
| TRUMP | 0.99528 |
| UNI | 1.0 |
| VVV | 1.0 |
| WLFI | 0.9973 |
| XMR | 1.0 |
| XPL | 1.0 |
| XRP | 1.0 |
| ZEC | 1.0 |
| kPEPE | 0.99581 |

- `mark_valid_frac` is **all-window grid coverage** from `LazyPrices.stats`. `unpriced_open_steps` and `forced_gap_exits` are **exposure-specific**: they count only bars where an open leg meets a data outage. A low `mark_valid_frac` coin does not by itself imply an unpriced open-position step, and a zero exposure-gap count does not prove full grid coverage.

### Structural sleeve pod pair classification (Train)

| Pair | dist | psi_pos | Concurrent days | Common coins | Unknown | Disjoint coins |
|---|---:|---:|---:|---:|---|---|
| W#117 / W#264 | 0.7564 | 0.5939 | 58 | 1 | False | False |
| W#117 / Whale_0x5f94 | 0.4993 | -0.0071 | 46 | 5 | False | False |
| W#264 / Whale_0x5f94 | 0.8371 | -0.2863 | 79 | 2 | False | False |

### Decision-rule outcome

- Pairwise dominance (net, static DD, worst day): `{"A_upgma>C_baseline": false, "C_baseline>A_upgma": false}`.
- Verdict: **GAP_INVALIDATED_NO_COMPARISON**.

### Selection stability under 10% universe drops

- Mean Jaccard vs full-universe selection: A `0.7`, B `0.84` across seeds `[0, 1, 2, 3, 4]`.
- Per-seed: `{"A_upgma": [0.5, 1.0, 0.5, 0.5, 1.0], "B_sleeve": [1.0, 1.0, 1.0, 1.0, 0.2]}`.

## Interpretation (non-causal)

- UPGMA and the unstratified top-rank baseline are close on OOS net (~$798 vs ~$766) and static floor DD (~-$9), and both select overlapping members. The structural sleeve pod has far lower OOS net (~$232) but also far lower static floor DD (~-$3) and worst day (~-$44).
- B top-pod member contributions: `[{"abs_net_share": 0.4654, "alias": "W#117", "avg_weighted_gross_notional_frac": 0.02030628, "capped_k": 0.00649374, "solo_weighted_net_usd": 241.47}, {"abs_net_share": 0.2562, "alias": "W#264", "avg_weighted_gross_notional_frac": 0.00695685, "capped_k": 1.1955260517, "solo_weighted_net_usd": 132.92}, {"abs_net_share": 0.2784, "alias": "Whale_0x5f94", "avg_weighted_gross_notional_frac": 0.02077705, "capped_k": 6.87608e-05, "solo_weighted_net_usd": 144.4}]`. Members below 5% absolute net share: `[]`.
- All sleeve members contribute materially; no dilution claim is made.
- B's top pod is **Train-gap-flagged** (a selected member has forced gap exits or unpriced open steps), and 3/3 of its top candidates are flagged. Its OOS figures are diagnostic only and it is excluded from the dominance verdict.
- Selection stability under 10% universe drops is higher for sleeves (mean Jaccard 0.84 vs 0.70), consistent with the premise that structural labels move less than dendrograms. This does **not** establish superior returns; the OOS net is lower.
- With 11 behavioral-pass wallets, one 61-day OOS window, and no point-in-time roster, the comparison is underpowered. It supports the stability claim and rejects a return-superiority claim; it does not rank the methods for deployment.

## Unknown-pair, gap, and coverage limitations

- **Unknown** means fewer than 5 days on which both wallets were active; this is insufficient evidence, not necessarily zero simultaneous bars. Concurrent activity in **disjoint coins** is a legitimate zero direct cosine, not a missing observation, though it is not proof of low common-factor or tail risk. The canonical matrix initializes no-overlap cosine to neutral; absence of evidence is not evidence of neutrality.
- **Gap gate:** a pod with any `unpriced_open_steps > 0` is marked `gap_invalidated` and excluded from the dominance verdict. Pair-local conservative exits are modeled, not actual forced execution. The Vike fallback is present in this checkout (61 coin files; `STORE_PLUS_VIKE` for covered coins), so the reduced universe is driven by registry roster drift, not price coverage. This study does not claim v4 replay reproducibility.
- OOS is a single 61-day window that earlier research already explored; it is **not** a pristine holdout. The roster is the present-day registry, so historical eligibility is **survivorship-contaminated** and not point-in-time.
- The Train chop window is the last 92 calendar days of Train, not a point-in-time regime label. Chop-positive is used only as a role tie-break; no independently regime-conditioned claim is made.
- `full` policy carries pre-boundary inventory and is shown only as a sensitivity; only the `new` policy regenerates eligibility at the OOS boundary.
- **Coin-universe mismatch:** OOS uses raw `grouped` (canonical v4 behavior) while Train uses `hist_eligible`, so OOS can include coins excluded from Train by price coverage. The table above lists per-member mismatches; this is disclosed, not aligned.
- **Train gap flag:** a selected member with any Train forced gap exit or unpriced open step marks the pod `train_gap_flagged`; such pods are excluded from the dominance verdict because their Train `k` may be gap-contaminated.

## Qualification and caveats

- This is a **matched retrospective comparison**, not causal evidence that sleeve structure produces superior returns. Selecting the OOS winner ex post would itself be selection bias; the frozen dominance rule is reported instead.
- No live broker execution, funding, capacity, min-lot feasibility, or PIT source continuity is proven. No deployments, allocations, orders, service or registry writes occurred.

- CFT trading restrictions and account framing require independent compliance confirmation.
- Live broker execution, funding, capacity and PIT source continuity are UNPROVEN.
- No deployments, allocations, orders, service or registry writes occurred.

