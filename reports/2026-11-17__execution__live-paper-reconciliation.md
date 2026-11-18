---
firm: multi-venue-execution-audit
account_usd: 5000
status: RESEARCH
top_candidate: NONE
script: studies/2026-11-17__execution__live-paper-reconciliation.py
---

# Live Executor, CWSA Paper & Screening Pagination Reconciliation (2026-11-17)

Read-only reconciliation against `executor.sqlite`, CWSA Chapter 2 paper SQLite journals, `ml4t-data` 1m OHLCV bars, and cached `dextrabot-top-copy-30d` JSONs.

## 1. Bybit Executor Timing & 1m Lag Drift (`hydro-5k/executor.sqlite`)

- **Status:** `PARTIAL_TIMING_AND_LAG_DRIFT_VERIFIED`
- **Waves processed:** `864` (wave lag vs checkpoint p50=`37916.0` ms, p90=`41739.8` ms, max=`297006.0` ms)
- **Order intents:** `54` / `54` `FILLED` (`0` non-filled); dispatch lag p50=`241717.0` ms, p90=`286215.0` ms
- **1m dispatch-lag price drift (matched intents=`46`, unmatched=`8`):** mean adverse=`-5.8987` bps, p50=`-11.2109` bps, p90=`26.9926` bps
- **Broker fill price & fee reconciliation:** `BLOCKED_EVIDENCE:EXECUTOR_SQLITE_OMITS_FILL_PRICE_AND_FEE`

## 2. CWSA Chapter 2 Paper Journals

| Paper DB | Wave Receipts | Coverage Complete % | Publish Lag p50 (ms) | Publish Lag p95 (ms) | PM03 Events | Status |
|---|---:|---:|---:|---:|---:|---|
| `A/paper-events.sqlite3` | 1615 | 99.44% | 100478.0 | 4739026.0 | 0 | `BLOCKED_EVIDENCE:CWSA_PAPER_EVENTS_EMPTY_NO_SIGNED_MANDATE` |
| `B/paper-events.sqlite3` | 1615 | 99.26% | 100786.0 | 4739026.0 | 0 | `BLOCKED_EVIDENCE:CWSA_PAPER_EVENTS_EMPTY_NO_SIGNED_MANDATE` |
| `C/paper-events.sqlite3` | 1615 | 99.26% | 85772.0 | 4739026.0 | 0 | `BLOCKED_EVIDENCE:CWSA_PAPER_EVENTS_EMPTY_NO_SIGNED_MANDATE` |

## 3. `userFillsByTime` Pagination Audit (`dextrabot-top-copy-30d`)

- **Files audited:** `179` | **Screening complete:** `164` | **Incomplete:** `15` | **Usable for backtest:** `0`
- **Flags:** `TRUNCATED_PAGE_CAP=1`, `TAIL_WINDOW_TRUNCATED=9`, `UNVERIFIED_ZERO_FILLS=5`

---
_Compliance & Qualification Caveat: Read-only observational diagnostics only; missing broker fill prices/fees (`executor.sqlite`) and empty PM03 paper `events` keep realized slippage calibration `BLOCKED_EVIDENCE` until fill-price persistence and signed mandates are enabled._
