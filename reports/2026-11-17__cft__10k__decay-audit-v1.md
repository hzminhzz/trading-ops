# CFT portfolio research — current-risk veto and OOS failure audit

**Evaluation timestamp:** 2026-11-16 17:16 UTC. This report is a dated **read-only snapshot**, not a continuously refreshed assessment.

**Decision:** **REJECT ALL FOUR V5 PORTFOLIOS FOR CURRENT ADMISSION.** Historical Train/OOS scores remain archival retrospective scenarios. The veto has no effects on live trading or the canonical portfolio registry.

## Cause of the false confidence

- V5's Train was **May 1–August 31, 2026**; the modeled OOS was **September 1–October 31, 2026**. The HyperX screenshots show severe losses later in November, which the historical test never evaluated.
- This OOS has already been inspected in earlier research; it was not a pristine independent holdout. The 5,000-path MC resamples the same favorable two-month trajectory, not the unknown future. The reported **100% pass** for candidate 3 is only a **historical bootstrap scenario**.
- The model ranked candidates on historical follower PnL/DD with no *current wallet-risk admission gate*. It also used a present-day wallet registry lacking a pre-September PIT discovery roster.
- Original **W#94 Train max DD = $31.54**, causing `k = 3.9632403` under the individual 50%-of-risk rule. Its standalone scaled September–October follower net was **+$531.73** before portfolio netting. W#1 was scaled by `k = 0.008003193`, contributing only **−$29.49**. The portfolio's positive period return hid severe untested forward-risk sensitivity.
- Neither historic model nor the MC included the observed November unrealized-risk collapse; the follower's new-position-only policy would *not* necessarily have inherited the same source positions at their leader entry prices.

## Official Hyperliquid current account-state and recent fills

Current snapshot from public `POST /info` `clearinghouseState` and `userFillsByTime` (not a private API key). **Equity = Hyperliquid perpetual margin-summary account value**, not HyperX combined/spot value. The recent `closedPnl` is realized closed-trade PnL, **excluding funding, open PnL and other account cash movements**.

| Wallet | Perp equity | Gross notional | Open uPnL | uPnL loss/equity | Last 30d fills | Last 30d closedPnl | Current research admission |
|---|---:|---:|---:|---:|---:|---:|---|
| W#1 | $72,891 | $405,303 | **−$83,547** | **114.6%** | 5 | −$3,610 | **NO-GO** |
| W#94 | **$190** | $1,629 | **−$139** | **73.2%** | 96 | +$63 | **NO-GO** |
| W#174 | $9,487 | $0 | $0 | 0% | 349 | +$318 | PASS snapshot only (currently flat) |
| W#179 | $37,993 | $91,512 | **−$12,806** | **33.7%** | 902 | +$5,391 | **NO-GO** |
| W#257 | $117,814 | $48,501 | −$622 | 0.5% | 435 | +$4,634 | PASS snapshot only |
| W#280 | **$0** | $0 | $0 | unavailable | 125 | +$951 | **NO-GO** |
| Whale_0x5f94 | $1,736,963 | $12,520,122 | +$3,832,096 | 0% | 228 | +$353,958 | PASS snapshot only (7.2x notional/equity review) |

This is a **current leader risk screen**. It is **not** follower PnL, not an independently audited return series, not a permanent wallet blacklist, and not a trading authorization. W#94's HyperX total account value shown in the screenshot was higher than its native perp margin equity; comparing these without accounting-scope reconciliation is invalid.

## Four portfolio gates

| Historical Train rank | Wallet members | Current admission | Blocked members |
|---|---|---|---|
| 1 | W#174, W#257, W#280, W#94 | **NO-GO** | W#280, W#94 |
| 2 | W#1, W#174, W#179, W#280 | **NO-GO** | W#1, W#179, W#280 |
| 3 | W#1, W#174, W#94, Whale_0x5f94 | **NO-GO** | W#1, W#94 |
| 4 | W#174, W#179, W#257, Whale_0x5f94 | **NO-GO** | W#179 |

Research-only, configurable safety thresholds (not new Guardian rules): reject any member with current perp margin equity below **$1,000**, negative open uPnL exceeding **25%** of current perp equity, missing/malformed official state, or missing/zero recent 30-day fill activity. Flag gross notional above **5x** perp equity for manual review. Any rejected member prevents the entire previously selected portfolio from being recommended for current admission. These thresholds are conservative heuristics, **not statistically validated predictions**.

## Correct workflow

1. Reproduce a wallet's historical follower PnL with Hyperliquid-only 1m and causal five-minute trade timing.
2. Freeze historical members and multipliers **without reading future evaluation outcomes**.
3. **For a recommendation today**, query official account state and recently observed fills; apply the separate date-stamped admission veto and verify present-day source account value, open drawdown and risk posture.
4. Only after current admission, independently validate actual copyable new-position opportunities, portfolio overlap, risk budget and destination execution feasibility. Require forward observations beyond the stale 2026-10-31 cutoff.
5. Never use a 100% bootstrapped historical MC number as a promised challenge pass probability.

**Code:** `studies/cft_current_admission.py`. **Evidence:** `reports/2026-11-17__cft__10k__current-risk-admission-v1.json`. Original V5 freeze and retrospective results remain unchanged.

**Status:** ENGINEERING_VERIFIED (89 full repository tests, 50 scoped research tests, Ruff and diff checks passed); historic OOS RETROSPECTIVE; current recommendations **NO-GO**; SCIENTIFICALLY_QUALIFIED=false; ECONOMICALLY_QUALIFIED=false; TRADING_AUTHORIZED=false. No orders, deployment, Guardian changes, or registry mutations.
