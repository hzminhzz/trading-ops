---
firm: multi (propr:classic_1step, vanta-trading:classic_1step_3pct, hyrotrader:two_step)
account_usd: 10000
status: NO_DEPLOYABLE_POD
provenance: RETROSPECTIVE_RESEARCH_ONLY
top_oos_diag_candidates: AS#7688499c, AS#a80ed515, AS#b56ed959 (k-scaled OOS only; 0 Train-eligible)
top_pnl_specialists: AS#4bfec86b, AS#c1bde03a, AS#f444942f, AS#16d4b95f
script: studies/2026-11-25__multi__10k__altcoin-short-discovery-study.py
---

# Altcoin-Short Hyperliquid Discovery & $10k Prop-Challenge Underwriting (`2026-05-01..2026-11-18`)

**Verdict: `NO_DEPLOYABLE_POD`** (`RETROSPECTIVE_RESEARCH_ONLY` — `0` Train-eligible wallets and `0` OOS-eligible wallets cleared default unscaled `ScoreConfig()` gates, so Stage-2 OOS `run_kernel_backtest` pods were not deployed and no `PAPER_ELIGIBLE` roster proposal was generated).

Under diagnostic relaxed-participation screening (`RELAXED_PARTICIPATION_SCORE_CONFIG`) in the Sep–Nov altcoin-bleed window (`oos_sep_nov`), **`3` candidates (`AS#7688499c`, `AS#a80ed515`, `AS#b56ed959`)** cleared all non-capacity `ScoreConfig` gates (`$15` q10 floor, `$1k` AV floor, `>= 20` markout fills, positive 5m markout, `14d` active, Euler CDaR `<= 0.45`) with `k`-scaled follower p95 participation `< 1%` (`0.15%–0.76%`), while larger pure altcoin-short specialists (`AS#4bfec86b`, `AS#c1bde03a`, `AS#f444942f`) failed either May–Aug Train markout, `< $15` micro-fill q10 floors, or thin-altcoin `VENUE_VOLUME_UNKNOWN` exclusions.

---

## 1. Provenance, Gate 1 & Isolation Disclosure

- **Gate 1 (`roster.sqlite` read-only audit):** Inspected `/home/quant/.local/share/copytrade/roster.sqlite` (`accounts: ["cft-10k-paper"]`, `leases=0`, `draining=0`, `bench=0`, `cusum=0`, `scorecards=0`, `vetoes=0`, status `CLEAN_EMPTY_ROSTER`). Zero live seats or draining positions exist to retire.
- **Strict PIT verification (`allow_undated=False`):** Returned **`0` cards** with `basis.pit = STRICT_KNOWN_AT` over `202` days as of `2026-11-19T00:00:00+00:00`.
- **Retrospective opt-in (`allow_undated=True`):** All Stage-1 audits opt into `allow_undated=True` (`basis.pit = RETROSPECTIVE_UNDATED`, `markout = FACT_STORE_BARS_1M_CLOSE_5M`, `participation = FACT_STORE_BARS_1M_VENUE_VOLUME`, `cdar_share = EULER_MARGINAL_CDAR_COHORT`).
- **Read-only & zero-leak isolation:** `roster.sqlite` and `/home/quant/dev/prop-production` were never mutated; private address maps stay under `/home/quant/.local/share/copytrade/research/altcoin_shorts_20261125/` (`0600`); zero raw `0x...` addresses appear in tracked code, reports, or JSON outputs.

---

## 2. Gate 2 — HyperX Altcoin-Short Discovery & Pre-Backfill Screen

Queried `POST https://data-api.hyperx.trade/v2/discover/wallets` across five keyset-paginated passes targeting short-dominant crypto traders (`primary_dex` in `["main", "hyperliquid"]`, `primary_dex_share_min = 0.80`, `primary_asset_classes = ["crypto"]`, `margin_summary_account_value_min = 5000`, `complete_trade_interval_seconds_medium_min = 300`, `total_short_realized_pnl > max(0, total_long_realized_pnl)`, `profit_factor_min = 1.05`, `roi_min = 0.01`, active `<= 14d`), enriched via `POST /market/wallet-tags`, and screened on live Hyperliquid `clearinghouseState` + 30d `userFillsByTime` + `canonical_replay.assess_bybit_symbol_screen`:

| Gate 2 Stage | Count | Rejected / Notes |
|---|---:|---|
| HyperX 5-pass short-dominant discovery (`30d`/`90d` `short_bias` + `90d` short PnL) | **232** | Deduplicated across 9 pages |
| HyperX `POST /market/wallet-tags` filter | **229** | Rejected **3** (`scalper: 3`) |
| Major-share pre-filter (`primary_coin in CORE_4` & share `> 50%` or commodity excluded) | **143** | Removed **86** major-only/commodity wallets before HL probes |
| HL `clearinghouseState` (`AV >= $5,000` & `uPnL >= -25% AV`) | **133** | Rejected **10** (`bagholder_upnl`), `0` errors |
| HL 30d `userFillsByTime` probed (`audit_user_fills_pagination`) | **94** | `39` HL API 429 errors; flags: `TRUNCATED_PAGE_CAP=18`, `TAIL_WINDOW_TRUNCATED=5` (routed to Reservoir) |
| Bybit linear USDT perp screen + `>= 50%` 30d altcoin notional + `>= 50%` short (when complete) | **39** | Rejected **33** (`REJECT_UNMAPPED_MATERIAL`), **20** (`altcoin_share < 50%`), **2** (`long_bias_30d`) |
| **Final Gate-2 Shortlist** | **39** | All **39** (`AS#<hash8>`) carried to Colab Reservoir backfill |

---

## 3. Gate 3 — Google Colab Reservoir Backfill & `FactStore` Coverage

Executed on Google Colab `--high-mem` (`50 GiB` RAM, `8` vCPUs, session `altcoin-shorts-20261125`, elapsed `717.44s`, subsequently terminated with `0` active sessions) pulling `gdrive:quant-data-backup/facts-mirror` (`2026-05-01..2026-11-18`) + live in-Colab DuckDB S3 scan across all `202` UTC days via `--profile reservoir`:

- **Reservoir S3 completeness (`2026-05-01..2026-11-18`):** `202 / 202` UTC days fetched on Colab (`0` missing S3 days).
- **Cohort `fills` coverage:** `3,028,278` fill rows across **`39 / 39`** shortlisted wallets (`202 / 202` UTC days).
- **Cohort `bars_1m` coverage:** `202 / 202` UTC days across all **`181`** traded coins (`bars_1m_missing_coins = []`).

---

## 4. Stage-1 `score_universe` Gate-Attrition Table

Evaluated across three windows: `train_may_aug` (`2026-05-01..2026-09-01`, `123d`), `oos_sep_nov` (`2026-09-01..2026-11-19`, `79d`), and `full_may_nov_diag` (`2026-05-01..2026-11-19`, `202d`), under both default strict `ScoreConfig()` and diagnostic `RELAXED_PARTICIPATION_SCORE_CONFIG` (`max_p95=1.0`, `max_max=1.0`):

| Stage-1 Outcome / Gate | `train_may_aug` (Strict) | `oos_sep_nov` (Strict) | `full_may_nov` (Strict) | `oos_sep_nov` (Relaxed Part. Diag) |
|---|---:|---:|---:|---:|
| **Shortlisted Cohort Input** | **39** | **39** | **39** | **39** |
| Excluded: `BASELINE_ORDER_BELOW_FLOOR` (`q10 < $15`) | 14 | 14 | 14 | 14 |
| Excluded: `VENUE_VOLUME_UNKNOWN` | 7 | 12 | 14 | 12 |
| Excluded: `NO_FILLS_IN_WINDOW` | 5 | 0 | 0 | 0 |
| **Total Excluded (No Scorecard)** | **26** | **26** | **28** | **26** |
| **Total Scored** | **13** | **13** | **11** | **13** |
| Failed: `P95_PARTICIPATION_ABOVE_LIMIT` | 13 | 13 | 11 | 5 *(raw max > 1.0)* |
| Failed: `NEGATIVE_5M_MARKOUT` | 12 | 6 | 8 | 6 |
| Failed: `NO_ACTIVE_FILLS` (`0` days in 14d) | 2 | 0 | 0 | 0 |
| Failed: `COPYABLE_DRIFT_UNPROVEN` (`< 20` fills) | 1 | 0 | 0 | 0 |
| **Eligible (`failures == ()`)** | **0** | **0** | **0** | **3** (`AS#a80ed515`, `AS#b56ed959`, `AS#7688499c`) |

---

## 5. Top Altcoin-Shorting Experts Discovered (By Category)

### A. Diagnostic OOS Stage-1 Survivors & Positive-5m-Markout Short Experts (`oos_sep_nov`)

These candidates proved positive 5m copyable markout (`FACT_STORE_BARS_1M_CLOSE_5M`), `$15+` q10 clip sizes, complete `bars_1m` volume coverage, and `< 1%` estimated `k`-scaled follower p95 participation in Sep–Nov (`oos_sep_nov`), though none cleared May–Aug Train (`train_may_aug`):

| Alias | AV ($) | OOS Alt % | OOS Short % | OOS Alt-Short PnL ($) | OOS 5m Markout (bps) | OOS Alt-Short Entry 5m (bps) | Est. `k`-Scaled p95 1m | Top Coins | Why Not Train-Eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| **`AS#7688499c`** | $12,641.56 | 100.0% | 92.8% | **+$40,084** | **+1.53** | -5.91 | 0.762% | `ZEC, XMR, WLD` | Train `markout_5m = -18.68 bps` |
| **`AS#a80ed515`** | $68,394.05 | 100.0% | 51.8% | **+$52,962** | **+1.56** | **+3.90** | 0.385% | `XRP` (100%) | Train `VENUE_VOLUME_UNKNOWN` |
| **`AS#b56ed959`** | $186,917.36 | 74.8% | 39.5% | **+$136,750** | **+4.55** | **+5.33** | 0.147% | `TAO, XLM, DOT` | Train `markout_5m = -9.87 bps`; mixed L/S |
| `AS#4bfec86b` *(near-miss)* | $117,958.80 | 100.0% | 98.4% | **+$368,715** | **+2.05** | **+1.72** | 0.277% | `NEAR, ZEC, ARB` | Train `markout_5m = -12.72 bps`; OOS `max_part = 1.000` |
| `AS#c04ed592` *(near-miss)* | $109,106.07 | 100.0% | 100.0% | +$1,528 | **+7.89** | -0.58 | 0.781% | `XPL` (100%) | OOS `raw_max = 1.000145` |

### B. Highest Altcoin-Short Realized/MTM PnL Specialists (Blocked from Prop Copy by Execution Structure)

| Alias | AV ($) | Full Alt % | Full Short % | Full Alt-Short PnL ($) | OOS Alt-Short PnL ($) | Full Alt-Short Entry 5m (bps) | Top Coins | Stage-1 Blocker |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `AS#16d4b95f` | $1,704,166.39 | 92.1% | 89.3% | **+$2,990,405** | **+$955,983** | -4.53 | `HYPE, VVV, PUMP, ENA` | `BASELINE_ORDER_BELOW_FLOOR` (`q10 < $15`, negative 5m markout) |
| `AS#c1bde03a` | $200,495.85 | 91.5% | 98.2% | **+$89,796** | **+$43,154** | **+1.00** (`+4.70` OOS) | `KAITO, HYPE, SAGA, XLM` | `BASELINE_ORDER_BELOW_FLOOR` (`q10 < $15`) |
| `AS#8c910299` | $27,107.17 | 100.0% | 100.0% | **+$83,278** | **+$70,230** | -5.39 | `PURR, PUMP, CASHCAT, FARTCOIN` | `BASELINE_ORDER_BELOW_FLOOR` (`q10 < $15`) |
| `AS#6d77ae70` | $129,065.88 | 100.0% | 100.0% | **+$82,832** | **+$80,638** | +1.07 (`-6.59` OOS) | `EIGEN, ZEC, OP, UNI` | `VENUE_VOLUME_UNKNOWN` |
| `AS#558a4518` | $28,772.53 | 100.0% | 100.0% | **+$53,945** | **+$48,371** | -1.25 | `ZEC, PURR, CASHCAT, FARTCOIN` | `BASELINE_ORDER_BELOW_FLOOR` (`q10 < $15`) |
| `AS#f444942f` | $6,587.38 | 91.6% | 99.1% | **+$34,335** | **+$27,263** | **+12.34** (`-2.94` OOS) | `ZEC, PURR, FARTCOIN` | `VENUE_VOLUME_UNKNOWN` (`4` thin bars) |

---

## 6. Root-Cause Diagnosis & What This Does Not Show

1. **May–Aug vs Sep–Nov regime asymmetry:** `12 / 13` scored Train wallets suffered negative 5m markouts during the May–Aug window (`-2.5` to `-28.7 bps`), while `7 / 13` flipped positive in Sep–Nov (`+1.53` to `+7.89 bps`). Shorting altcoins is structurally regime-dependent rather than an all-weather anchor (cf. `LESSONS.md` 2026-11-14 rule on multi-regime `net/max_dd`).
2. **Thin altcoin 1m bars trigger `VENUE_VOLUME_UNKNOWN` and `PARTICIPATION_LIMIT`:** `12` OOS wallets traded on at least one 1m bar with zero/missing `quote_volume_usd`, and `13 / 13` scored OOS wallets exceeded `1%` unscaled leader 1m participation.
3. **TWAP/micro-split orders trigger `BASELINE_ORDER_BELOW_FLOOR`:** `14 / 39` altcoin-short wallets split entries below `$15` (`q10_order_usd < $15`), making them incompatible with Bybit minimum-notional copy execution (`LESSONS.md` 2026-11-16 rule).
4. **Limits:** Results are `RETROSPECTIVE_RESEARCH_ONLY` (`allow_undated=True`). No Stage-2 OOS `run_kernel_backtest` pod was formed or executed because `0` wallets passed Train eligibility under either strict or `k`-scaled screening.
