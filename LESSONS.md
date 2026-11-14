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
