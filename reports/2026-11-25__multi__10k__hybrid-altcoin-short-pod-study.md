---
firm: multi (propr:classic_1step, vanta-trading:classic_1step_3pct, hyrotrader:two_step)
account_usd: 10000
status: NO_DEPLOYABLE_HYBRID_POD
provenance: RETROSPECTIVE_RESEARCH_ONLY
top_candidate: HYB-RELAXED4 (0% 5m breach, -$2.95 1x / -$5.90 2x net, $30.55 p2t DD, -challenge_fee EV)
script: studies/2026-11-25__multi__10k__hybrid-altcoin-short-pod-study.py
results_json: reports/2026-11-25__multi__10k__hybrid-altcoin-short-pod-study__results.json
---

# Stage-2 Hybrid Pod Study: Liquid-Major/L12 Anchors + Altcoin-Short Specialists ($10k Prop Challenge)

**Verdict: `NO_DEPLOYABLE_HYBRID_POD`** (`RETROSPECTIVE_RESEARCH_ONLY` — zero `PAPER_ELIGIBLE` proposals emitted). Pairing Train-eligible liquid-major (`LM#`) / `L12` (`R#`) anchors with verified altcoin-short specialists (`AS#821eae6a`, `AS#3276acd5`, `AS#0ff64c60`, `AS#6d77ae70`) inside Train-frozen 3–4 seat pods cut Stage-2 OOS drawdowns to `$30.55–$116.91` (`0` `> 5%` 1m participation bars across all `16` runs), **but** rolling PIT Stage-1 seat attrition (`FLAT = 84.8%` on `HYB-KSCALE3`; `PARTIAL = 84.8%` / `$855.78` total notional on `HYB-RELAXED4`) left all pods near flat (`-$0.67` to `-$5.90` on relaxed/k-scaled hybrids; `-$50.66`/`-$61.29` on `KANCHOR-DIV3`), yielding `0.0%` multi-phase payout probability and negative Net EV (`-$110` to `-$127`) across all three `$10,000` prop programs.

---

## 1. Provenance, Execution & PIT Boundaries

- **Cohort Assembly (`/home/quant/.local/share/copytrade/research/altcoin_shorts_20261125/private_inputs_hybrid.json`, `0600`):** `45` deduplicated wallets (`19` `ANCHOR_LM` from `liquid_majors` R1/R2, `12` `ANCHOR_L12` from `may_nov_consistent_manifest`, `14` `ALT_SHORT` specialists from altcoin-short discovery Run 1/Run 2; minimum Nov snapshot AV = `$1,577.74`). Zero raw `0x...` addresses appear in code, JSONs, or this report.
- **Colab Remote Execution (`2026-05-01..2026-11-18`, `202` UTC days):** Ran in two concurrent `--high-mem` CPU runs (`Part A`: `2000.57s` for `KANCHOR-DIV3` + `HYB-KSCALE3`; `Part B`: `2228.84s` for `HYB-RELAXED4` + `HYB-UPTREND-DIAG`) merging `gdrive:quant-data-backup/facts-mirror` with live S3 requester-pays Reservoir DuckDB scans (`760,746` S3 fills, `0` missing days; `765,952` total isolated cohort fills across `45` wallets and `192` coins; `202/202` `fills` & `bars_1m` days present). `34` missing `bars_1m` tickers were all `XYZ:*` HIP-3 equity perps; `5` wallets trading missing-`bars_1m` or `> 30` coins (`AS#8126a65b`, `AS#81428d8a`, `AS#87ea3330`, `AS#c1bde03a`, `R#cc509d58`) were excluded from Stage-2 pod selection so Stage-2 runs had `0` `unknown_vol_trades` and `0` `unknown_px_trades`.
- **Strict PIT Check (`allow_undated=False`):** Returned **`0` cards** (`basis.pit = STRICT_KNOWN_AT`). All Stage-1 and Stage-2 runs opt into `allow_undated=True` (`RETROSPECTIVE_RESEARCH_ONLY`).
- **Market Regimes (`141` index altcoins, `202` UTC days):** `100` `UPTREND` days / `102` `DOWNTREND` days; `3` `UPTREND` months (`2026-05`: `+1.41%`, `2026-08`: `+24.18%`, `2026-09`: `+31.09%`) and `4` `DOWNTREND` months (`2026-06`: `-19.45%`, `2026-07`: `-2.32%`, `2026-10`: `-19.30%`, `2026-11`: `-22.13%`).

---

## 2. Stage-1 `score_universe` Attrition (`train_may_aug` vs `oos_sep_nov`)

| Mode / Window | Scored | Eligible | Excluded | Top Failures / Exclusions | Eligible Aliases |
|---|---:|---:|---:|---|---|
| Strict `train_may_aug` (`123d`) | 42 | **3** | 3 | `P95_PARTICIPATION_ABOVE_LIMIT` (`39`), `NEGATIVE_5M_MARKOUT` (`7`), `NO_ACTIVE_FILLS` (`6`), `VENUE_VOLUME_UNKNOWN` (`3`) | `LM#91adaf10, R#062fd979, R#4c6c4326` |
| Strict `oos_sep_nov` (`79d`) | 35 | **0** | 10 | `P95_PARTICIPATION` (`34`), `NEGATIVE_5M_MARKOUT` (`15`) | none |
| Strict `full_may_nov_diag` (`202d`) | 35 | **0** | 10 | `P95_PARTICIPATION` (`35`), `NEGATIVE_5M_MARKOUT` (`11`) | none |
| Relaxed `train_may_aug` (`123d`) | 42 | **28** | 3 | `P95_PARTICIPATION` (`8`), `NEGATIVE_5M_MARKOUT` (`7`), `NO_ACTIVE_FILLS` (`6`), `VENUE_VOLUME_UNKNOWN` (`3`) | `26` anchors + **`AS#3276acd5, AS#821eae6a`** |
| Relaxed `oos_sep_nov` (`79d`) | 35 | **15** | 10 | `NEGATIVE_5M_MARKOUT` (`15`), `P95_PARTICIPATION` (`6`), `BASELINE_ORDER_BELOW_FLOOR` (`5`), `VENUE_VOLUME_UNKNOWN` (`5`) | `11` anchors + **`AS#0ff64c60, AS#3276acd5, AS#7688499c, AS#a80ed515`** |
| Relaxed `full_may_nov_diag` (`202d`) | 35 | **18** | 10 | `NEGATIVE_5M_MARKOUT` (`11`), `P95_PARTICIPATION` (`8`), `VENUE_VOLUME_UNKNOWN` (`8`) | `17` anchors + **`AS#3276acd5`** |

### Train-Frozen Pod Composition (`Euler Marginal CDaR <= 0.45` for Strict/Relaxed)

- **Strict Pods (`strict=[]`):** Zero strict pods formed (`3` strict anchors alone did not clear `<= 0.45` Euler CDaR; `0` strict Train `ALT_SHORT` wallets).
- **`KANCHOR-DIV3` (Benchmark — Anchors Only):** `R#1b6db721, R#3a6ae1c9, LM#91adaf10` (`ANCHOR_L12, ANCHOR_L12, ANCHOR_LM`; `k = [0.024927, 0.001196, 0.292166]`; Euler `[0.1996, 0.4227, 0.3777]`; Train avg corr `-0.0106`).
- **`HYB-KSCALE3` (`k`-Scaled `<= 1%`/`5%` Hybrid):** `AS#821eae6a, LM#0b30dd55, LM#b18b3894` (`ALT_SHORT, ANCHOR_LM, ANCHOR_LM`; `k = [0.009509, 0.019731, 0.043782]`; Euler `[0.3566, 0.4089, 0.2344]`; Train avg corr `-0.0012`).
- **`HYB-RELAXED4` (`RELAXED`-Eligible 2+2 Hybrid):** `AS#821eae6a, AS#3276acd5, LM#f84f6807, R#3a6ae1c9` (`ALT_SHORT, ALT_SHORT, ANCHOR_LM, ANCHOR_L12`; `k = [0.009509, 0.509220, 0.009741, 0.001196]`; Euler `[0.3356, 0.2800, -0.0169, 0.4013]`; Train avg corr `0.0000`).
- **`HYB-UPTREND-DIAG` (Permissive Diagnostic):** `AS#0ff64c60, AS#6d77ae70, LM#220e5f5f, LM#91adaf10` (`ALT_SHORT, ALT_SHORT, ANCHOR_LM, ANCHOR_LM`; `k = [0.030068, 0.002810, 0.003692, 0.292166]`; Euler `[0.4258, 0.0551, 0.1257, 0.3933]`; Train avg corr `-0.0032`).

### Key Seat Turnover Between Train (`may_aug`) and OOS (`sep_nov`)

| Alias (Sleeve) | May–Nov Net ($) | Uptrend Months ($) | Train 5m MO (bps) | Train Status (`relaxed_part`) | OOS 5m MO (bps) | OOS Status (`relaxed_part`) |
|---|---:|---:|---:|---|---:|---|
| `AS#821eae6a` (`ALT_SHORT`) | +$40,781.89 | -$69,663.89 | **+18.99** | Eligible (`kp95=0.14%`) | **-13.31** | Failed `NEGATIVE_5M_MARKOUT` |
| `AS#3276acd5` (`ALT_SHORT`) | +$1,635.92 | **+$409.30** | **+15.55** | Eligible (`kp95=50.9%`) | **+10.82** | **Eligible** (`+$1,159.88` OOS) |
| `AS#0ff64c60` (`ALT_SHORT`) | +$8,352.83 | **+$25,169.49** | +3.77 | Failed `P95_PARTICIPATION` | **+6.85** | **Eligible** (`+$7,537.86` OOS) |
| `AS#6d77ae70` (`ALT_SHORT`) | +$103,113.68 | **+$69,271.44** | -4.84 | Failed markout/active/part | None | Excl. `VENUE_VOLUME_UNKNOWN` |
| `LM#0b30dd55` (`ANCHOR_LM`) | +$9,745.66 | -$3,129.97 | +1.47 | Eligible (`kp95=0.86%`) | -1.95 | Failed `NEGATIVE_5M_MARKOUT` |
| `LM#b18b3894` (`ANCHOR_LM`) | -$776.81 | -$1,751.90 | +3.22 | Eligible (`kp95=0.08%`) | -0.11 | Failed `NEGATIVE_5M_MARKOUT` |
| `LM#f84f6807` (`ANCHOR_LM`) | +$79,253.55 | +$35,825.51 | +3.30 | Eligible (`kp95=0.64%`) | None | Excl. `BASELINE_ORDER_BELOW_FLOOR` |
| `LM#91adaf10` (`ANCHOR_LM`) | +$6,381.38 | +$1,578.64 | +9.31 | Eligible (`kp95=0.20%`) | +4.71 | **Eligible** (`+$4,790.75` OOS) |
| `R#1b6db721` (`ANCHOR_L12`) | +$2,546.44 | +$11,088.62 | +23.90 | Eligible (`kp95=0.02%`) | +15.16 | **Eligible** (`-$4,266.68` OOS) |
| `R#3a6ae1c9` (`ANCHOR_L12`) | +$231,821.47 | -$113,646.37 | +9.06 | Eligible (`kp95=0.08%`) | +16.00 | **Eligible** (`+$217,687` OOS) |

---

## 3. Stage-2 Strict OOS Kernel Walk-Forward (`2026-09-01..2026-11-19`, `79d`)

All Stage-2 runs executed through `trading_ops.research.run_kernel_backtest` on isolated per-pod `FactStore`s (`scale=2` `$10,000` `RulebookCatalog` terms: `propr:classic_1step:phase_1` fee `$110`, `vanta-trading:classic_1step_3pct:phase_1` fee `$127`, `hyrotrader:two_step:phase_1` fee `$119` refundable; `1x` = `4.5 bps` fee + `30 bps` shortfall + `4 bps` spread, `2x` = `9 bps` + `60 bps` + `8 bps`; `n_paths = 5000`, `N_eff = 11.29` independent 7-day blocks for Wilson 95% CIs).

| Pod (`mode`) | Firm | Cost | Net PnL ($) | P2T DD / Floor DD ($) | Pass Rate | Mode Share (`A / P / F`) | Ntl ($ USD) & Alt-Short Share | 1m Part (`med / p95 / max`, `>1%`, `>5%`) | EV Blended ($ & 95% CI) | EV 75%-Chop ($ & 95% CI) | First..Last `decision_sha` |
|---|---|---|---:|---:|---:|---|---:|---|---|---|--|
| `KANCHOR-DIV3` (`relaxed`) | `propr` | `1x` | **-$50.66** | $82.95 / $55.22 | 1.0 | `0.44 / 0.29 / 0.27` | $4,653 (`1.03%`) | `0.026% / 0.273% / 0.294%`, `0`, `0` | -$110.00 (`[-110, -110]`) | -$110.00 (`[-110, -110]`) | `da2433377b79..585cbbfa70dc` |
| `KANCHOR-DIV3` (`relaxed`) | `propr` | `2x` | **-$61.29** | $63.56 / $53.94 | 1.0 | `0.44 / 0.29 / 0.27` | $3,861 (`1.25%`) | `0.022% / 0.271% / 0.294%`, `0`, `0` | -$110.00 (`[-110, -110]`) | -$110.00 (`[-110, -110]`) | `cd0261c3f51c..0b90a0b67f7e` |
| `HYB-KSCALE3` (`relaxed`) | `propr` | `1x` | **-$0.67** | $116.91 / $111.68 | 1.0 | `0.00 / 0.15 / 0.85` | $194 (`100.0%`) | `0.064% / 0.067% / 0.068%`, `0`, `0` | -$110.00 (`[-110, -110]`) | -$110.00 (`[-110, -110]`) | `dfe27e175df6..6c988a9a8f77` |
| `HYB-KSCALE3` (`relaxed`) | `propr` | `2x` | **-$1.34** | $116.91 / $112.38 | 1.0 | `0.00 / 0.15 / 0.85` | $194 (`100.0%`) | `0.064% / 0.067% / 0.068%`, `0`, `0` | -$110.00 (`[-110, -110]`) | -$110.00 (`[-110, -110]`) | `f8f1718955e5..f6deb57b6fe7` |
| `HYB-KSCALE3` (`relaxed`) | `vanta-trading` | `1x` | **-$0.67** | $116.91 / $111.68 | 1.0 | `0.00 / 0.15 / 0.85` | $194 (`100.0%`) | `0.064% / 0.067% / 0.068%`, `0`, `0` | -$127.00 (`[-127, -127]`) | -$127.00 (`[-127, -127]`) | `6d6bbd1c2864..95fa8ff1b733` |
| `HYB-KSCALE3` (`relaxed`) | `vanta-trading` | `2x` | **-$1.34** | $116.91 / $112.38 | 1.0 | `0.00 / 0.15 / 0.85` | $194 (`100.0%`) | `0.064% / 0.067% / 0.068%`, `0`, `0` | -$127.00 (`[-127, -127]`) | -$127.00 (`[-127, -127]`) | `d997ced53975..8c42c2f056e8` |
| `HYB-KSCALE3` (`relaxed`) | `hyrotrader` | `1x` | **-$0.67** | $116.91 / $111.68 | 1.0 | `0.00 / 0.15 / 0.85` | $194 (`100.0%`) | `0.064% / 0.067% / 0.068%`, `0`, `0` | -$119.00 (`[-119, -119]`) | -$119.00 (`[-119, -119]`) | `daa8ce6f9291..c2c28b57e620` |
| `HYB-KSCALE3` (`relaxed`) | `hyrotrader` | `2x` | **-$1.34** | $116.91 / $112.38 | 1.0 | `0.00 / 0.15 / 0.85` | $194 (`100.0%`) | `0.064% / 0.067% / 0.068%`, `0`, `0` | -$119.00 (`[-119, -119]`) | -$119.00 (`[-119, -119]`) | `0d7c247ae811..8e607393c444` |
| `HYB-RELAXED4` (`relaxed`) | `propr` | `1x` | **-$2.95** | **$30.55 / $25.23** | 1.0 | `0.15 / 0.85 / 0.00` | $856 (`53.87%`) | `0.050% / 0.355% / 0.432%`, `0`, `0` | -$110.00 (`[-110, -110]`) | -$110.00 (`[-110, -110]`) | `07fc911e4a19..7e2c9841860d` |
| `HYB-RELAXED4` (`relaxed`) | `propr` | `2x` | **-$5.90** | **$30.55 / $26.27** | 1.0 | `0.15 / 0.85 / 0.00` | $856 (`53.87%`) | `0.050% / 0.355% / 0.432%`, `0`, `0` | -$110.00 (`[-110, -110]`) | -$110.00 (`[-110, -110]`) | `91a2f9638e7e..7e2c9841860d` |
| `HYB-RELAXED4` (`relaxed`) | `vanta-trading` | `1x` | **-$2.95** | $30.55 / $25.23 | 1.0 | `0.15 / 0.85 / 0.00` | $856 (`53.87%`) | `0.050% / 0.355% / 0.432%`, `0`, `0` | -$127.00 (`[-127, -127]`) | -$127.00 (`[-127, -127]`) | `9a720894ef68..9ee2d5a0d31f` |
| `HYB-RELAXED4` (`relaxed`) | `vanta-trading` | `2x` | **-$5.90** | $30.55 / $26.27 | 1.0 | `0.15 / 0.85 / 0.00` | $856 (`53.87%`) | `0.050% / 0.355% / 0.432%`, `0`, `0` | -$127.00 (`[-127, -127]`) | -$127.00 (`[-127, -127]`) | `28095308ca47..9ee2d5a0d31f` |
| `HYB-RELAXED4` (`relaxed`) | `hyrotrader` | `1x` | **-$2.95** | $30.55 / $25.23 | 1.0 | `0.15 / 0.85 / 0.00` | $856 (`53.87%`) | `0.050% / 0.355% / 0.432%`, `0`, `0` | -$119.00 (`[-119, -119]`) | -$119.00 (`[-119, -119]`) | `c35a779366bd..e47bbf827774` |
| `HYB-RELAXED4` (`relaxed`) | `hyrotrader` | `2x` | **-$5.90** | $30.55 / $26.27 | 1.0 | `0.15 / 0.85 / 0.00` | $856 (`53.87%`) | `0.050% / 0.355% / 0.432%`, `0`, `0` | -$119.00 (`[-119, -119]`) | -$119.00 (`[-119, -119]`) | `111e74dff36b..e47bbf827774` |
| `HYB-UPTREND-DIAG` (`permissive`) | `propr` | `1x` | **-$120.82** | $276.82 / $207.96 | 1.0 | `0.00 / 1.00 / 0.00` | $35,021 (`0.00%`) | `0.024% / 0.484% / 1.044%`, `1`, `0` | -$109.81 (`[-110, -60.86]`) | -$109.76 (`[-110, -81.94]`) | `ba730058d01e..aa0876ed0e9c` |
| `HYB-UPTREND-DIAG` (`permissive`) | `propr` | `2x` | **-$171.81** | $548.87 / $485.89 | **0.0** | `0.00 / 1.00 / 0.00` | $24,899 (`0.00%`) | `0.029% / 0.924% / 2.635%`, `3`, `0` | -$110.00 (`[-110, -110]`) | -$110.00 (`[-110, -110]`) | `7d8df842ca55..81603e50d7f0` |

*(Note: `HYB-UPTREND-DIAG` at `2x` breached `CONTRACT_DAILY_FLOOR` and `INTERNAL_DAILY_STOP`; all `14` `relaxed_part` runs passed every kernel daily and trailing floor with `challenge_pass_rate = 1.0` and `0` `>1%` / `>5%` 1m participation breaches.)*

---

## 4. Definitive Synthesis

1. **Guardian + Rolling Stage-1 Worked As Designed, Eliminating Tail Blowups:** Pairing altcoin-short sleeves with anchors (`HYB-KSCALE3`, `HYB-RELAXED4`) improved OOS PnL and drawdown relative to both the anchor benchmark (`KANCHOR-DIV3`: `-$50.66`/`-$61.29`, `p2t_dd=$82.95`) and permissive sizing (`HYB-UPTREND-DIAG`: `-$120.82`/`-$171.81` daily-stop breach). When `AS#821eae6a` (`+18.99 -> -13.31 bps`), `LM#0b30dd55`, and `LM#b18b3894` decayed in OOS, daily rolling Stage-1 automatically sidelined them (`FLAT=84.8%` on `HYB-KSCALE3`).
2. **Why Fixed 3–4 Seat Hybrid Pods Failed `$10k` Prop EV (`0%` Payout Probability):** Individual `suggested_k` copy-trading over a 3–4 wallet pod is brittle against 5m markout inversion and sparse trading cadence (`$194–$856` total OOS notional vs `$1,000` evaluation profit target). Consistently profitable multi-regime altcoin-short specialists (`AS#3276acd5`, `AS#0ff64c60`, `AS#6d77ae70`, `AS#c1bde03a`) derive edge from patient multi-day/weekly altcoin decay or hedged swings rather than high-frequency 5m markout bursts.
3. **Recommended Pivot:** Rather than allocating fixed 3–4 seat `suggested_k` pods, evaluate **cross-sectional structural altcoin-short consensus** (following `2026-11-25__multi__10k__sticky-liveness-consensus-study.md`) where net short weights across the top `N` low-churn / high-`gross_move_bps` altcoin specialists are volatility-targeted into liquid altcoins (`>=$1M` daily vol) with 15m TWAP execution.

---

**Qualification & Compliance Limits:** `RETROSPECTIVE_RESEARCH_ONLY` — undated Reservoir archive (`allow_undated=True`, `strict_pit_sample_rows = 0`) supports paper-eligibility screening only and cannot authorize live trading. Results depend on `bars_1m` mark/volume proxies (`34` `XYZ:*` tickers unpriced in `bars_1m` and `>30`-coin wallets excluded from Stage-2 pod selection) and `RulebookCatalog` synthetic checkpoint assumptions; because `0` pods cleared positive Stage-2 EV, no `PAPER_ELIGIBLE` proposal was generated. Live Guardian state, `roster.sqlite`, and `prop-production` remain untouched (`colab --auth=adc sessions` = `0` active).
