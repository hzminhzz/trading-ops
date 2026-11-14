# $10,000 static-floor challenge — native CWSA + local Hyperliquid 1m PM03 replay

**Run date:** 2026-11-14. **Disposition:** `RESEARCH_PROPOSAL / SCENARIO_ONLY / NO_TRADE`. **Trading authorization:** `false`. This report supersedes prior source-activity rankings, fill-reconstruction experiments and idealized sizing recommendations.

## Decision

The financial replay is operational and reproducible, **but no tested portfolio produces an admissible positive training result after the $10 minimum-order stress, one-shot liquidity and PM03 risk restrictions**. The replay selects `k=0` for all three, meaning **no trade**, not an empirical $0 trading strategy. The shorter, fractional-fill scenario is retained as a *non-executable diagnostic*, not a profitable follower strategy.

All evaluation uses **local Hyperliquid 1-minute OHLCV**. **No Bybit pricing or Bybit fills are used.**

## Data evidence and causal sequence

- **Authentic source:** the canonical CWSA Chapter 2 SQLite journal. The script copies it using SQLite online backup **from a read-only live connection into an ephemeral isolated snapshot**, avoiding races with active source writers. Calls `copytrade_pm.pm_cwsa_chapter2_gate._source_facts`, verifies journal hash chain, typed `CheckpointOutcome`, and validates **166 original Hyperliquid REST receipt/proof bytes** for all observed changes/baselines/recovery-baselines. Never infers flat holdings from gaps.
- **Observed native source overlap:** 2026-11-12 05:10 UTC through 2026-11-13 23:59 UTC, 514 5-minute checkpoint waves, 14 wallets, 136 source change outcomes, 16 UNAVAILABLE and 16 RECOVERY_BASELINE. Native source and finance proofs are different claims.
- **Local historical prices:** `/home/quant/dev/quant/ml4t-data/.data/store/crypto_futures_ohlcv_1m_HYPERLIQUID_<SYMBOL>_PERP/` for 21 instruments and **3 exact native Hyperliquid** instruments GMX/AZTEC/SEI fetched once from the official `/info` `candleSnapshot` endpoint for 2026-11-12..13 to private `trading-ops/data/private/hyperliquid_candles_1m_20261112_13/`. Each 2880-candle series is locally SHA-verified and time-contiguous. Protected `copytrade-pm` and `/home/quant/.local/share/copytrade/` datasets were **not modified**. The replay reads all prices offline.
- **Native source timing:** for each individual `CHANGED` outcome, other selected wallets must have a recent verified book; only after the source's actual `published_ms` and before the native source-age/next UTC checkpoint deadline can an attempt be made. `BASELINE`, `UNCHANGED`, `RECOVERY_BASELINE`, `UNAVAILABLE`, `QUARANTINE`, and unverified/missing wallet evidence do not produce orders.
- **Causal fill proxy:** `published_ms+1ms` as decision/attempt time, using **previous fully completed local 1m bar close** and *previous* traded volume, not the minute's future close/open. Cost model is 4.5bps taker fee, 2bps full spread, 2bps slippage plus 5bps at full participation (sqrt) impact; max participation 10% of previous completed 1m base volume. Explicit research quote-age window 90s, **NOT native production's verified L2 freshness**. One attempt only, no partial catch-up or fabricated multi-minute execution.
- **Native components actually called:** `pm_checkpoint_paper._marks` extracts authentic source prices from hashed original REST receipts; `pm_exposure_sizing.target_from_fixed_k` maps leader signed quantity × native source mark × frozen per-wallet k / local follower mark; `pm03_account.Account` handles cash/equity/MTM and the fixed **$300 daily and $9,500 initial static floor**; `pm03_execution.execute_targets` computes one-shot modeled fills, fees, partial/volume constraints and native account reconciliation. These are in-memory *research invocations*; no signed `PaperJournal.consume_checkpoint` or broker execution is claimed.
- **Explicit additional execution feasibility test:** research-only $10 minimum per external order, **not independently verified exchange-specific lot/minimum**; if any required external residual is below this, the entire event is blocked rather than silently omitting a position. Source accounts and chart data do **not** establish actual Hyperliquid/Bybit executable size, margin or funding.

## Three compared portfolios

| Proposal | Source members | Source changed opportunities | Train-only k recommendation | Admission |
|---|---|---:|---|---|
| **P1 — TriadA (existing sealed research)** | ACT04, ACT08, ACT09 | 20 | **0 (NO TRADE)** | No positive, feasible causal training selection |
| **P2 — Crypto Balanced (unsealed new study)** | ACT04, ACT08, W031 | 23 | **0 (NO TRADE)** | No positive, feasible causal training selection |
| **P3 — Regime alternative (unsealed new study)** | ACT08, ACT09, W031 | 16 | **0 (NO TRADE)** | No positive, feasible causal training selection |

The rosters share wallets and were selected **after observing the historical source window**; they are not independent, or confirmed out-of-sample wallet selections.

### Exploratory economic diagnostic: tiny fractional-size fills **WITHOUT** $10 minimum

These are what the native PM03 model calculates at **uniform k = 0.0001 per wallet**, before enforcing the separate research minimum-order stress. Such fills are **not economically admitted**.

| Metric | P1 TriadA | P2 Balanced | P3 RegimeAlt |
|---|---:|---:|---:|
| Train net (Nov 12 partial day) | −$1.6233 | −$1.6233 | +$0.4925 |
| Later chronological net (Nov 13) | −$15.2608 | −$15.2544 | +$1.8411 |
| Total modeled net | **−$16.8841** | **−$16.8777** | **+$2.3336** |
| Modeled instrument-level fills | 158 | 165 | 99 |
| Modeled fills with notional under $10 | **142 (89.9%)** | **149 (90.3%)** | **98 (99.0%)** |
| Total modeled taker fee | $0.4811 | $0.4826 | $0.0329 |
| Peak-to-trough DD | $26.7816 | $26.7816 | $4.4528 |
| 1×→2×→3× cost-stressed total net | −$16.8841 / −$17.6994 / −$18.5149 | −$16.8777 / −$17.6956 / −$18.5137 | +$2.3336 / +$2.2758 / +$2.2181 |

An earlier exploratory `k=0.0002` P3 variant showed **+$4.6647**, but its quantities were also mostly fractional/dust; do not treat it as an executable result or claim alpha. A `k` search that optimizes PnL by causing most losing opportunities to fail liquidity caps is also invalid. The training selection enforces **≥70% of native eligible source opportunities attempted**, in addition to positive net PnL, training modeled daily loss under $150, peak DD under $250, and no floor breach. Because the short sample and capacity gate were inspected and revised during research, **neither chronological split is independent scientific OOS**.

### Strict minimum-order stress results

After adding **$10 minimum for each required external order**, all portfolio/k candidates failed the pre-specified *research* positive-train/coverage/risk gates. The emitted `k=0` is an **abstention**, not a positive-PnL decision. Under this strict scenario zero orders occur, and the $10,000 balance remains unchanged by definition; this is **not** proof of funded challenge survival.

## Scientific and engineering limitations

1. **Only 42h50m of genuinely overlapping source and historical prices**, including less than one day for train and one day for later evaluation. This is insufficient to judge copytrade alpha, tail risk, regimes, or the chance of meeting the challenge return target.
2. The three portfolios were chosen with full knowledge of this historical sample. This train/chronological-evaluation split tests code and economics but **not an independent portfolio-selection holdout**.
3. **Actual finance authority is absent:** the installed Chapter 2 paper journals have no signed Copy Mandates and no modeled financial effects. The research script uses native PM03 *modules in memory* (not the protected journal, Guardian or connector), and uses hypothetical 1m close fills.
4. **Executable order precision unknown:** research $10 minimum is an explicit conservative proxy, not a verified lot/minimum contract; previous-minute traded volume is not L2 depth. Missing funding/borrow, liquidation/margin behavior, spread variations and real latency prevent claiming broker/follower PnL.
5. Early in this mission, an order-at-next-minute-open experiment was discovered to **violate CWSA source-age expiry**, so it was rejected. The accepted scenario attempts just after original publication but uses **previous-minute close proxy**, which is not a live executable price. No event from a missing/recovery source checkpoint becomes a fresh trade.
6. The original reservoir fills / Sept-Nov historical reconstructed positions are **not silently relabeled as native first-known checkpoints**. The retrospective legacy scripts were retired from active PM source in PR #376.

## Artifacts / verification

All files are in `/home/quant/dev/trading-ops/`, no protected runtime modifications:

- `studies/native_1m_pm03_replay.py` — native source + fixed-k + PM03 local 1m replay
- `studies/fetch_missing_hl_1m.py` — private verified local Hyperliquid OHLCV supplementation, not needed again once present
- `studies/cwsa_native_prop10k_audit.py` — native CWSA original source audit; its `RESEARCH_ROSTERS` definitions are reused
- `tests/test_native_1m_pm03_replay.py` — timing, native halted risk, gaps, k training gate and dust rules
- `reports/native-1m-economic-replay-20261114.json` — machine-readable per-k train/OOS/cost stress/drawdown/coverage
- `proposals/native-hl-1m-three-20261114.json` — three abstaining proposals
- `data/private/hyperliquid_candles_1m_20261112_13/` — **Git-ignored** private price data + manifest SHA
- `pyproject.toml` / `uv.lock` — pinned to `copytrade-pm` merged legacy-retirement commit `1cccd2049e77d39035746ccf4400dbde9c284aca`

Reproduce offline:
```bash
cd /home/quant/dev/trading-ops
uv run --frozen python studies/native_1m_pm03_replay.py
uv run --frozen pytest -q
uv run --frozen ruff check .
```

**Disposition: NO TRADE.** We have actual modeled financial paths on genuine CWSA observations, but no robust, executable native follower alpha, positive challenge survival probability, or independently calibrated per-wallet k. Never promote the fractional +$2.33/+4.66 cases to live/prospective economic qualification.
