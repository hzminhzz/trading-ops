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

## 2026-11-16 — Full-Tranche CFT Copy-Trading Financial Screening

1. **Do not confuse full registry screening with full financial replay.** The prior study screened 195 wallets but financially replayed only 21. A deterministic 24-wallet convenience sample cannot support an exhaustive portfolio-ranking claim. This study replayed all 128 historically eligible wallets on Train, retaining a per-wallet checkpoint and a separate 30-wallet portfolio shortlist.
2. **Split historical eligibility from current operational suitability.** Present-day Hyperliquid account value and current Bybit listings are not historical point-in-time facts. In the expanded Train ledger, historical eligibility and today's operational constraints are separate fields; current withdrawals never erase past Train observations.
3. **Enforce half-risk sizing at the frozen primary portfolio, not as a comparison only.** The unconstrained optimizer may select multipliers far above each wallet's 50%-risk baseline. Cap primary k to that baseline, preserve uncapped k explicitly as a sensitivity, and never treat unconstrained optimizer feasibility as evidence for capped sizing.
4. **Minimum-order warnings prevent economic qualification.** All three OOS-positive portfolios still have q10 order sizes below the illustrative $10 threshold. Until actual Bybit quantity steps/minimum notional and rejected-order economics are replayed, reported returns and bootstrap pass percentages remain retrospective scenarios rather than executable follower PnL.
5. **Breadth of roster is not breadth of portfolio optimization.** 128 wallets received Train financial replay, but the bounded UPGMA candidate search evaluated 12 unique combinations and three finalists shared W#174 and Whale_0x5f94. Do not describe them as globally optimal; additional cluster diversity and a fresh chronological holdout are needed to strengthen selection claims.
