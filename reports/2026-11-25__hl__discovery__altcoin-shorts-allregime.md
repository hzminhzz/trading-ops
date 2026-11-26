---
firm: multi (propr:classic_1step, vanta-trading:classic_1step_3pct, hyrotrader:two_step)
account_usd: 10000
status: NO_DEPLOYABLE_POD
provenance: RETROSPECTIVE_RESEARCH_ONLY
top_allregime_copyable: AS#3276acd5 (cleared relaxed-part Stage-1 in Train, OOS & Full; 7/7 months profitable)
top_allregime_uptrend_experts: AS#c1bde03a, AS#6d77ae70, AS#f444942f, AS#3276acd5, AS#0ff64c60
script: studies/2026-11-25__multi__10k__altcoin-short-allregime-study.py
---

# All-Regime Altcoin-Short Hyperliquid Discovery & Uptrend-Resilience Audit (`2026-05-01..2026-11-18`)

**Verdict: `NO_DEPLOYABLE_POD`** (`RETROSPECTIVE_RESEARCH_ONLY` — `0` wallets cleared strict unscaled `ScoreConfig()` in `train_may_aug` or `oos_sep_nov`, and only `2` wallets cleared `RELAXED_PARTICIPATION_SCORE_CONFIG` in `train_may_aug`, short of the 3-seat minimum for `PAPER_ELIGIBLE` Stage-2 kernel pods).

However, expanding Gate 2 from `39` to **`167` wallets** (`4,351,120` fills across `202` UTC days on Google Colab `--high-mem`) uncovered **`38` all-regime short experts** that stayed net-profitable across **Train (`May–Aug`)**, **OOS (`Sep–Nov`)**, and **Uptrend months (`2026-05`, `2026-08 +23.54%`, `2026-09 +30.05%`)** — separating cleanly into **`31` correlated `TRX/PAXG` mirror accounts** and **`7` independent altcoin-short specialists**.

---

## 1. Provenance, Gate 1 & Isolation Disclosure

- **Gate 1 (`roster.sqlite` read-only):** `cft-10k-paper` holds `0` leases, `0` draining owners, `0` bench entries, `0` CUSUM latches, and `0` scorecards (`CLEAN_EMPTY_ROSTER`).
- **Strict PIT (`allow_undated=False`):** Returned **`0` cards** (`basis.pit = STRICT_KNOWN_AT`) over `202` days as of `2026-11-19T00:00:00+00:00`.
- **Retrospective opt-in (`allow_undated=True`):** All Stage-1 audits opt into `allow_undated=True` (`RETROSPECTIVE_UNDATED`).
- **Zero-leak & read-only:** `roster.sqlite` and `prop-production` remain untouched; private address mappings remain in `/home/quant/.local/share/copytrade/research/altcoin_shorts_20261125/private_inputs_v2.json` (`0600`); zero raw `0x...` addresses appear in tracked files.

---

## 2. Gate 2 v2 — Multi-Regime HyperX + HL + Bybit Screen

Queried `POST https://data-api.hyperx.trade/v2/discover/wallets` across 7 passes (`period="all"` with `track_record_days_min=180`, `90d` with `150d`, `30d` with `115–150d`, sorted by `calmar_ratio`, `sharpe_ratio`, and `radar_score_total`; `AV >= $1,500`), filtered via `POST /market/wallet-tags`, and screened on HL `clearinghouseState` + 30d `userFillsByTime` (rate-paced `0.32s`) + `assess_bybit_symbol_screen`:

| Gate 2 Stage | Count | Rejected / Notes |
|---|---:|---|
| HyperX 7-pass discovery (`all`/`90d`/`30d`, short-capable) | **1,077** | Deduplicated across 22 pages |
| HyperX `POST /market/wallet-tags` filter | **1,060** | Rejected **17** (`scalper: 17`) |
| Altcoin-capable pre-filter (`not (primary in CORE_4 and share > 60%)`) | **834** | Removed **226** major-dominated/commodity accounts |
| Top pre-ranked + Run-1 seeds probed on HL `clearinghouseState` | **251** | Prioritized `all_` (`180d`) + non-core4 primary + Calmar/Sharpe |
| HL `clearinghouseState` (`AV >= $1,500` & `uPnL >= -25% AV`) | **233** | Rejected **3** (`AV < $1,500`), **15** (`bagholder_upnl`), **`0` errors** |
| HL 30d `userFillsByTime` + Bybit screen | **219** | `14` HL errors; flags: `TRUNCATED_PAGE_CAP=48`, `TAIL_WINDOW_TRUNCATED=7` |
| Bybit perp + `>= 45%` altcoin + `>= 40%` short (when complete) | **192** | Rejected **16** (`UNMAPPED_MATERIAL`), **10** (`alt < 45%`), **1** (`long_bias_30d`) |
| **Final Gate-2 v2 Shortlist** | **167** | Top `150` + all `39` Run-1 seeds merged cleanly |

---

## 3. Colab Backfill & Market Regime Calibration (`2026-05-01..2026-11-18`)

Executed on Google Colab `--high-mem` (`altcoin-short-allreg3-20261125`, `699.7s`, terminated with `0` active sessions):
- **Reservoir S3 & `FactStore`:** `202 / 202` UTC days fetched (`0` missing S3 days), `4,351,120` fill rows across `167 / 167` wallets, `202` days of `bars_1m` (`missing_bars_coins` contained only `1` spot `@107` and `36` HIP-3 `XYZ:*` equity/commodity symbols outside HL main perps).
- **Equal-weight liquid-altcoin index (`169` HL perps on `>= 150` days):**

| Month | Alt Index Return | BTC Return | Regime Label |
|---|---:|---:|---|
| `2026-05` | **+0.56%** | -5.79% | **UPTREND** |
| `2026-06` | -20.52% | -20.44% | DOWNTREND |
| `2026-07` | -2.71% | +7.26% | DOWNTREND |
| `2026-08` | **+23.54%** | +25.01% | **UPTREND** |
| `2026-09` | **+30.05%** | +6.41% | **UPTREND** |
| `2026-10` | -19.83% | -7.26% | DOWNTREND |
| `2026-11` *(thru 18th)* | -21.93% | -18.70% | DOWNTREND |

`100` of `202` UTC days and `3` of `7` months (`2026-05`, `2026-08 +23.54%`, `2026-09 +30.05%`) were **UPTREND** regimes.

---

## 4. Stage-1 `score_universe` Attrition (`167` Wallets)

| Stage-1 Outcome / Gate | `train_may_aug` (Strict) | `oos_sep_nov` (Strict) | `train_may_aug` (Relaxed Part.) | `oos_sep_nov` (Relaxed Part.) | `full_may_nov` (Relaxed Part.) |
|---|---:|---:|---:|---:|---:|
| **Cohort Input** | **167** | **167** | **167** | **167** | **167** |
| `BASELINE_ORDER_BELOW_FLOOR` (`q10 < $15`) | 92 | 88 | 92 | 88 | 96 |
| `VENUE_VOLUME_UNKNOWN` | 41 | 52 | 41 | 52 | 50 |
| `NO_FILLS_IN_WINDOW` | 7 | 0 | 7 | 0 | 0 |
| **Total Scored** | **27** | **27** | **27** | **27** | **21** |
| Failed: `P95_PARTICIPATION_ABOVE_LIMIT` | 26 | 27 | 18 | 15 | 15 |
| Failed: `NEGATIVE_5M_MARKOUT` | 20 | 18 | 20 | 18 | 17 |
| Failed: `NO_ACTIVE_FILLS` / `COPYABLE_DRIFT` | 8 / 2 | 0 / 0 | 8 / 2 | 0 / 0 | 0 / 0 |
| **Eligible (`failures == ()`)** | **0** | **0** | **2** (`AS#3276acd5`, `AS#821eae6a`) | **4** (`AS#3276acd5`, `AS#0ff64c60`, `AS#a80ed515`, `AS#7688499c`) | **1** (`AS#3276acd5`) |

---

## 5. Independent All-Regime & Uptrend-Proof Altcoin-Short Experts (`7` Independent / `38` Total)

Out of `38` wallets meeting `all_regime_expert` (`train_tot > 0`, `oos_tot > 0`, `train_short > 0`, `oos_short > 0`, `uptrend_months_tot > 0`), **`31`** belong to a single correlated `TRX/PAXG/BNB/BTC/LTC` pair-hedged mirror cluster (which offset uptrend alt-short losses with `PAXG`/BTC hedges), leaving **`7` independent altcoin-short specialists**:

| Alias | AV ($) | Alt % / Short % | Net Total PnL | Alt-Short PnL (Full) | Train Alt-Short | OOS Alt-Short | Uptrend-Months Total / Alt-Short | Uptrend-Days Alt-Short | Profitable Months | 5m Markout (`Train` / `OOS`) | Top Coins | Stage-1 Status / Blocker |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| **`AS#3276acd5`** | $2,896 | 94.0% / 97.3% | +$1,636 | **+$1,712** | **+$500** | **+$1,212** | **+$409** / **+$61** | -$4,417 | **7/7** | **+15.55 / +10.82 bps** | `LIT, STABLE, ARB, STRK` | **ELIGIBLE** (`relaxed_part` Train, OOS, Full; `k_p95=7.6%` OOS) |
| **`AS#c1bde03a`** | $200,496 | 91.5% / 98.2% | +$55,297 | **+$89,796** | **+$38,127** | **+$51,669** | **+$13,238** / **+$30,020** | **+$57,230** | **6/7** *(7/7 alt-sh)* | -0.98 / **+4.70 bps** *(entry)* | `KAITO, HYPE, SAGA, XLM` | `VENUE_VOLUME_UNKNOWN` / `q10 < $15` |
| **`AS#6d77ae70`** | $129,066 | 100% / 100% | **+$103,114** | **+$82,832** | **+$36,559** | **+$46,273** | **+$69,271** / **+$22,537** | -$9,327 | **5/7** *(sat out Jul-Aug)* | -4.84 bps / `VENUE_VOL` | `EIGEN, ZEC, OP, UNI` | `VENUE_VOLUME_UNKNOWN` (OOS/Full) |
| **`AS#f444942f`** | $7,863 | 91.6% / 99.1% | +$38,723 | **+$34,335** | **+$1,391** | **+$32,944** | **+$43,264** / **+$866** | -$54,809 | **4/7** | **+113.1** / -2.94 bps *(entry)* | `ZEC, PURR, FARTCOIN` | `VENUE_VOLUME_UNKNOWN` |
| **`AS#0ff64c60`** | $3,482 | 95.7% / 100% | +$8,353 | +$2,552 | -$321 | **+$2,873** | **+$25,169** / -$3,662 | -$14,176 | 3/7 | **+3.77 / +6.85 bps** | `AVAX, ADA, DOGE, XLM` | **ELIGIBLE** (`relaxed_part` OOS; `k_p95=2.95%`) |
| **`AS#6ae844f8`** | $5,751 | 97.2% / 48.8% | +$3,234 | +$4,484 | +$219 | **+$4,265** | +$2,712 / -$2,691 | -$17,365 | 4/7 | -0.10 / -3.72 bps *(entry)* | `W, SPX, GRIFFAIN, KAITO` | `BASELINE_ORDER_BELOW_FLOOR` (`q10 < $15`) |
| **`AS#81428d8a`** | $6,779 | 71.1% / 53.1% | +$2,569 | +$2,209 | +$412 | +$1,797 | +$196 / -$378 | -$3,372 | 4/7 | -3.50 / **+3.51 bps** | `SOL, ADA, BTC, ETH` | `P95_PARTICIPATION` (`k_p95=3.2%/6.3%`) |

### Key Takeaways on How Experts Profit Through Altcoin Uptrends

1. **Consistent intraday short-fade alpha (`AS#3276acd5`, `AS#c1bde03a`):** Unlike perma-bear gamblers blown out in August (`+23.5%`) and September (`+30.1%`), `AS#3276acd5` (`7/7` months green, `+15.55`/`+10.82 bps` 5m markout) and `AS#c1bde03a` (`7/7` months alt-short green, `+$57.2k` on uptrend days) selectively short overextended altcoins (`LIT`, `KAITO`, `SAGA`, `ARB`, `STRK`) with positive short-entry drift even inside market-wide rallies.
2. **Tactical cash-outs during parabolic squeezes (`AS#6d77ae70`):** `AS#6d77ae70` (`+$103.1k`, Calmar `159`) kept `100%` short bias and sat completely flat (`$0.00` PnL) during July–August (`+23.5%` squeeze) before re-entering shorts in Sep–Nov (`+$46.3k`).
3. **Why no `$10k` prop pod is ready yet (`NO_DEPLOYABLE_POD`):**
   - `AS#3276acd5` is the **only** wallet clearing `RELAXED_PARTICIPATION_SCORE_CONFIG` in both Train and OOS, yet its `suggested_k = 0.509` yields `est_k_scaled_p95 = 7.6%` OOS (`> 1%` thin-bar limit on `LIT`/`STABLE`).
   - `AS#c1bde03a`, `AS#6d77ae70`, and `AS#f444942f` trade thin altcoin bars (`VENUE_VOLUME_UNKNOWN`) or TWAP micro-splits (`q10 < $15`).
   - Optional `ASAR-DIV4` Stage-2 permissive kernel replay timed out past 13m (`22,752` 5m `Decimal` steps across 79 daily `score_universe` calls) and was skipped (`--run-permissive-stage2` off) because `strict_pods=[]` and `diag_k=[]` already settled `NO_DEPLOYABLE_POD`.
