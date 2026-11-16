# Trading & Portfolio Lessons

Durable lessons from backtests, paper trading, and operational post-mortems in `trading-ops`.

**Agent rule:** Every agent MUST read this file in full at the start of any session before proposing portfolios, calibrating `k`, or running backtests. Append new dated entries when a study or incident reveals a durable rule; keep entries evidence-backed and concise (no secrets, raw `.env`, or wallet PII).

---

## 2026-11-14 — 5m UTC Prop-Challenge Backtest (`A`, `B`, `C`, `TriadChop`, `TriadA`, `2026-09-01..2026-11-10`)

Tested 5 candidate 3-wallet portfolios over `wallet_store` (`2026-09-01..2026-11-10`, 5m UTC, `pod_risk = $166.67` at Hyro 5k scale) against `hyrotrader:two_step:phase_1` ($5k) and `crypto-fund-trader:two_phase:phase_1` (`scale=2`, $10k / $1k DD). Only **`TriadA`** passed both (`+$756.63` Hyro 5k / `+$1,513.26` CFT 10k, positive in both Sep and Oct–Nov, zero breaches).

1. **Screen for multi-regime `net/max_dd`, not single-month PnL.**
   September trended cleanly, making nearly all wallets look profitable, whereas Oct–Nov chop exposed fragile pods (`W028 -$128.02`, `W030 -$52.34`, `W042 -$28.32`, `W031 -$40.86`, `VS066 -$65.65`, `ACT10 -$25.12`). Only **`ACT08`** (`unit_net/max_dd = +1.62`, `Sep +$39.07`, `Oct-Nov +$499.72`) and **`W015`** (`+0.52`, `Sep +$172.99`, `Oct-Nov +$1.15`) stayed positive across both regimes.

2. **Do not stack trend-correlated pods without a chop-resilient anchor.**
   `TriadChop` swapped `TriadA`'s anchor `ACT08` (`+$538.78`) for `ACT10` (`-$25.12`, `Oct-Nov -$283.09`), causing `ACT04`, `ACT09`, and `ACT10` to draw down together in Oct–Nov and breach `DAILY_LOSS`. Build prop portfolios around all-regime anchors (`ACT08`, `W015`) before adding higher-beta sleeves (`W044`, `ACT04`).

3. **Respect leverage risk when sizing `k` at `0.5x` `wallet_max_dd`.**
   Sizing `k` so a 50% wallet drawdown equals `pod_risk` leaves thin margin when multiple pods enter Oct–Nov drawdowns simultaneously (e.g., Portfolio B's `W031` + `CA-005` + `W044` pullback breaching `STATIC_DD`). Stress-test synchronized multi-pod drawdowns and consider `1.0x` `wallet_max_dd` sizing for strict prop limits.

4. **Verify active Hyperliquid perp fill coverage before admitting a pod.**
   `CO-001` had zero HL perp fills/bars in `wallet_store` over the test window, silently wasting 1/3 of Portfolio C's allocation while `VS066` dragged down `W015`. Gate candidate selection on verified active perp execution coverage in the target window.

---

## 2026-11-15 — Handling Flagged / Decayed Wallets in Live Portfolios (`wallet-research` -> `prop-ops`)

When a **live** portfolio contains a wallet flagged by `wallet-research` (`WITHDRAWN`, `DECAYED`, `DORMANT`), **never panic-delete it from `wallet_registry.sqlite`** (that breaks historical provenance). Instead, apply the decision rule by flag and replace via the 3-step workflow:

1. **Decision rule by flag:**
   - **`WITHDRAWN` (`AV < $1,000`):** Set `k = 0` immediately via `prop-ops` (lead trader has no skin in the game; `bybit_executor` cleanly closes their positions on the next 5m wave).
   - **`DECAYED` (active + losing, e.g. `W030`):** Set `k = 0` and replace; do not pay `6.5 bps` turnover on a losing trader.
   - **`DORMANT` with heavy uPnL (e.g. `ACT09`):** Inspect open positions (`clearinghouseState`) and resting `frontendOpenOrders` (TP/SL limits) before cutting. Setting `k = 0` immediately market-closes all attributed exposure on Bybit; if resting TP limits exist and Guardian covers the downside, either keep as-is or halve `k` (`0.00099 -> 0.00050`) to trim unmanaged drag.
2. **3-step replacement workflow:**
   1. **Contain (`prop-ops`):** Lower or zero `k` for the flagged member on the live unit (with explicit approval).
   2. **Replace (`portfolio-research`):** Pair healthy members with fresh Stage-4 candidates from `wallet-research` and verify via 5m replay + Guardian MC.
   3. **Version & Cutover (`prop-ops`):** Register as a new portfolio version (`copytrade-wallet-registry native exclude` / `native register`), deploy to `/opt/prop-production/current`, and restart the executor.

---

## 2026-11-15 — Prop Challenge Selection (Vanta & Propr) + Anti-Detection Execution Hygiene

Monte Carlo (`80k–100k` paths) + Optional Stopping analysis across Vanta Classic (`$5k`/`$10k`) and Propr (`Classic`, `Turbo`, `Pro` 1-Step & `Classic 2-Step`):

1. **Enforce a `-2%` soft daily stop on `3%` Daily DD accounts instead of buying DD add-ons.**
   Uncapped `1%`-risk intraday trading causes `3%` Daily DD to breach prematurely while `5%–6%` static Max DD still has room, flipping fair-coin (`p=0.50`) and slight-edge (`p=0.52`) EV negative. Adding a `-2%` UTC soft daily pause recovers full static pass rates (`33.7%` Vanta, `52.6%` Propr Classic at `p=0.52`) without paying Vanta's `+$23` `5%` Daily DD upgrade.
2. **Choose splits and tiers strictly by expected gross extraction `G = P(pass) × E[Funded Gross]`:**
   - **Vanta Classic (`$10k`, `5%` static DD):** Pure fair-coin is capped at `G <= $166.67` (`1/3 × $500`), so **`80/20` ($81)** dominates when `G < $170`, `90/10` ($98) wins for `$170–$290`, and **`100/0` ($127)** only wins with real positive edge (`G > $290`, i.e., `p >= 52%`). Buy `$10k` over `$5k` (`$127 < 2 × $67`, `2x` sizing precision, avoids cross-account mirroring bans), using `$5k` (`$67`) only as a first live-integration canary since Vanta has no sandbox.
   - **Propr (`$10k`):** With a `-2%` daily guard, **`Classic 1-Step` ($110, `10%` target, `6%` static DD)** dominates `Pro` ($85, `12%`/`5%`), `Turbo` ($50, `9%`/`3%`), and `2-Step` ($100, `5%+10%`, `8%` intraday trailing) across `p = 0.50–0.55`.
3. **Break micro-fill & clock fingerprints on anti-copytrading venues (`Vanta`):**
   - **Notional deadband:** Skip small rebalance deltas below a dollar floor (e.g. `$25–$50` when modifying an open position) to stop echoing every lead-wallet TWAP/trim, while **always bypassing the deadband on full closes (`target == 0`) and halt flattens** so dust never stays open.
   - **Post-wave jitter only (`10–60s`, never pre-bar):** `cwsa_heads` snapshots are only emitted at `next_checkpoint_ms` (5m UTC closes), so pre-5m execution jitter is impossible without shifting upstream sampling; bounded post-wave delay breaks exact `:00/:05` clustering while staying well inside `max_source_lag_ms` (`600s`).

---

## 2026-11-16 — Strict New-Position-Only Research and Retrospective Qualification

1. **Always reset lifecycle admission at the simulated activation boundary.** A position opened before an OOS activation is not a new position; carry a symbol only after a flat-to-open transition or a genuine side flip after activation. The May–Nov 2026 CFT study found five Train-selected candidates with negative Sep–Oct modeled follower net PnL under strict new-only admission. Historical full-target synchronization materially changed PnL and increased drawdown in several portfolios, so the policies must never be conflated.
2. **Keep projected fills and retrospective rosters separate from live-quality financial evidence.** Reservoir `start_position` supports historical quantity reconstruction, but neither source observation known-at time nor actual Bybit execution, funding, min-lot and market impact is proven. Train stale-price checkpoints invalidate precise `k` qualification; handle them by verified backfill or conservative pair-local close before release.
3. **Reject misleading pass probabilities.** Stationary-block Monte Carlo on historical Guardian daily summaries tests sensitivity under resampling, not independent broker-executable paths or a historically frozen wallet-selection release. No candidate is economically or scientifically qualified on these observations. Research artifact: `reports/2026-11-16__cft__10k__may-nov-follower-portfolio.md`.

---

## 2026-11-16 — Canonical Backtest Hardening & Full-Universe `v4` (`May–Nov 2026`, CFT `$10k`)

Hardening `studies/study_data.py`, `studies/canonical_replay.py`, `studies/pod_constructor.py`, and single-pass Parquet-cached `studies/2026-11-16__cft__10k__canonical-full-universe-v4.py` cut runtime from **`2,202.74s` (`36.7m`)** across `96` wallets in `v3` to **`818.61s` (`13.6m`)** across `189` wallets + full OOS stress matrices (`v4`), while uncovering four durable research rules:

1. **Never sort Hyperliquid fills by `trade_id` or filter timestamps in local timezone.**
   `trade_id` is non-monotonic inside same-millisecond bursts (`59,584 / 68,744` position-chain breaks in 30d vs raw arrival sequence), and host local-TZ SQL (`Europe/Berlin`) drops final UTC hours. Always ingest via UTC Polars lazy CSE scans preserving raw Parquet arrival sequence (`with_row_index`) + `start_position` continuity tie-break (`repair_same_ms_fill_order`) and assert UTC window coverage (`assert_utc_coverage`).
2. **Enforce pre-clustering `min_baseline_order_usd = $15` before UPGMA clustering.**
   In `v3` (`min_baseline_order_usd=0.0`), sub-`$15` wallets (`q10_order = $0.18–$12.44`) entered clustering and produced `CPCV PBO = 0.90`. Filtering sub-`$15` wallets at the `2%` portfolio-DD baseline (`$200 / size`) removed `73` wallets at `size=3` (`91` at `size=4`), eliminated `min_lot_warnings`, dropped leak-free **`CPCV PBO` from `0.90` to `0.15`**, and elevated live-deployable `Pod 3` (`W#111, W#152, W#312`, `0.93x` peak leverage, `$15` deadband) to `+$1,128` OOS (`new`, `-$15` static floor DD, `94.46%` 90d MC blended pass).
3. **Separate historical Train eligibility (`scope="historical"`) from present-day deployability (`scope="live"`).**
   Dropping `AV < $1,000` or currently unmapped wallets prior to Train replay injects survivorship lookahead (`scope="live"` alone lifted `CPCV PBO` to `0.70` vs `0.15` across the `189`-wallet historical screen). Keep historical eligibility strictly pre-split, then tag `live_deployable` independently on candidates.
4. **Gate headline OOS passes with deadbands (`$15`/`$30`), `2x`/`3x` cost stress, funding, and 1m participation.**
   `Pod 3` stays strongly profitable under `$30` deadband (`+$1,124` at `1x`), `2x`/`3x` costs (`+$936`/`+$743`), and historical USDT funding carry (`-$10.69`), yet 1m bar-volume diagnostics expose thin-altcoin spikes (`96.96%` of 1m bar volume on `XRP`, `79` trades `>1%`, vs `0.066%` median / `2.21%` p95) that require capacity validation and prospective CWSA paper runs in `copytrade-pm` prior to live authorization.
