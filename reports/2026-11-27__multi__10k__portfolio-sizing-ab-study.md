---
firm: multi (propr:classic_1step, vanta-trading:classic_1step_3pct, hyrotrader:two_step)
account_usd: 10000
status: NO_QUALIFYING_2X_ARM
provenance: RETROSPECTIVE_RESEARCH_ONLY
top_candidate: ARM_A_INDEP_FIXED_K (2x net=$-730.81, p2t=$888.64)
script: studies/2026-11-27__multi__10k__portfolio-sizing-ab-study.py
results_json: reports/2026-11-27__multi__10k__portfolio-sizing-ab-study__results.json
---

# Controlled A/B Study: Portfolio Fixed-k Allocation vs. Independent Fixed-k ($10k Prop Challenge)

**Verdict: `NO_QUALIFYING_2X_ARM`** (`RETROSPECTIVE_RESEARCH_ONLY` — `strict_pit_cards_count = 0` on undated Reservoir data).

## 1. Cohort & FactStore Provenance

- **Cohort Selection (pre-OOS Train `2026-05-01..2026-08-31`):** `95` equity-backed wallets (`>= $1,000`), `51` with `>= 50` Train fills, **`31`** wallets trading only 100%-complete `bars_1m` standard perp coins (`22` coins, `22,465 / 22,465` 5m bars across OOS `[2026-09-01 - 5m, 2026-11-18)`). Excluded `20` wallets with incomplete `bars_1m` tickers and `3` non-perp tickers.
- **Isolated FactStore Slice (`2026-07-01..2026-11-18`):** `325,538` fills (`141` days), `141` `bars_1m` days.
- **Day-0 (`2026-09-01`, 60d lookback) Stage-1 Comparison:**
  - Legacy `decouple_sizing_gates=False` eligible (`11`): `['W#da87afdf', 'W#5d0580f8', 'W#18a9e538', 'W#2d777aa4', 'W#59f429fd', 'W#358b1426', 'W#690bdb0a', 'W#e54f0b00', 'W#84049b9a', 'W#fb404979', 'W#260f0da9']` (`k={'W#da87afdf': 0.000995, 'W#5d0580f8': 4.5e-05, 'W#18a9e538': 0.218193, 'W#2d777aa4': 1.035742, 'W#59f429fd': 0.010589, 'W#358b1426': 0.017065, 'W#690bdb0a': 0.662308, 'W#e54f0b00': 1.296798, 'W#84049b9a': 0.085572, 'W#fb404979': 1.209162, 'W#260f0da9': 0.391429}`).
  - Decoupled `decouple_sizing_gates=True` eligible (`12`): `['W#a8756f77', 'W#da87afdf', 'W#5d0580f8', 'W#18a9e538', 'W#2d777aa4', 'W#59f429fd', 'W#358b1426', 'W#690bdb0a', 'W#e54f0b00', 'W#84049b9a', 'W#fb404979', 'W#260f0da9']`.
  - Day-0 `$300` allocator (`dd=$298.28`): `k={'W#da87afdf': 0.000814, 'W#5d0580f8': 3.6e-05, 'W#18a9e538': 0.178716, 'W#2d777aa4': 0.84835, 'W#59f429fd': 0.008673, 'W#358b1426': 0.013977, 'W#690bdb0a': 0.54248, 'W#e54f0b00': 1.062175, 'W#84049b9a': 0.070089, 'W#fb404979': 0.990394, 'W#260f0da9': 0.320609}`.
  - Day-0 `$500` allocator (`dd=$366.27`): `k={'W#da87afdf': 0.000995, 'W#5d0580f8': 4.5e-05, 'W#18a9e538': 0.218193, 'W#2d777aa4': 1.035742, 'W#59f429fd': 0.010589, 'W#358b1426': 0.017065, 'W#690bdb0a': 0.662308, 'W#e54f0b00': 1.296798, 'W#84049b9a': 0.085572, 'W#fb404979': 1.209162, 'W#260f0da9': 0.391429}`.

## 2. Controlled 4-Arm × 2-Cost-Stress Kernel Walk-Forward (`2026-09-01..2026-11-18`, 78d)

| Arm | Cost | Net PnL ($) | Costs ($) | Fills | P2T DD ($) | Floor DD ($) | Ret/P2T | Valid / Surv | Mode (`A/P/F`) | 1m Part (`med / p95 / max`, `unk`) | Propr Blended EV ($) |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|
| `ARM_A_INDEP_FIXED_K` | `1x` | **$-950.68** | $505.54 | 129 | $1056.23 | $1056.23 | -0.90x | `True` / `BREACHED` | `0.99 / 0.01 / 0.00` | `0.1194% / 4.1327% / 12.8869%`, unk=1 | $-110.00 |
| `ARM_A_INDEP_FIXED_K` | `2x` | **$-730.81** | $421.94 | 68 | $888.64 | $888.64 | -0.82x | `True` / `BREACHED` | `0.99 / 0.01 / 0.00` | `0.0784% / 1.4871% / 8.1572%`, unk=0 | $-109.28 |
| `ARM_B_PORTFOLIO_DD300` | `1x` | **$+212.43** | $632.97 | 158 | $824.06 | $824.06 | 0.26x | `False` / `BREACHED` | `0.22 / 0.41 / 0.37` | `0.1077% / 3.5042% / 17.5705%`, unk=0 | $-109.76 |
| `ARM_B_PORTFOLIO_DD300` | `2x` | **$+211.91** | $504.61 | 61 | $758.01 | $758.01 | 0.28x | `False` / `BREACHED` | `0.22 / 0.41 / 0.37` | `0.1390% / 5.8885% / 9.4923%`, unk=0 | $-109.66 |
| `ARM_C_PORTFOLIO_DD500` | `1x` | **$+365.28** | $628.39 | 165 | $734.01 | $734.01 | 0.50x | `False` / `BREACHED` | `0.46 / 0.18 / 0.36` | `0.0916% / 3.2931% / 12.8869%`, unk=1 | $-110.00 |
| `ARM_C_PORTFOLIO_DD500` | `2x` | **$+396.36** | $674.07 | 86 | $956.23 | $956.23 | 0.41x | `False` / `BREACHED` | `0.46 / 0.18 / 0.36` | `0.0854% / 4.7012% / 40.6346%`, unk=0 | $-107.93 |
| `ARM_D_PORTFOLIO_DD300_NODUST` | `1x` | **$+212.43** | $632.97 | 158 | $824.06 | $824.06 | 0.26x | `False` / `BREACHED` | `0.22 / 0.41 / 0.37` | `0.1077% / 3.5042% / 17.5705%`, unk=0 | $-109.76 |
| `ARM_D_PORTFOLIO_DD300_NODUST` | `2x` | **$+211.91** | $504.61 | 61 | $758.01 | $758.01 | 0.28x | `False` / `BREACHED` | `0.22 / 0.41 / 0.37` | `0.1390% / 5.8885% / 9.4923%`, unk=0 | $-109.66 |

## 3. Head-to-Head Summary (2x Cost Stress)

- **`ARM_B_PORTFOLIO_DD300` vs `ARM_A_INDEP_FIXED_K` (2x):** Net PnL delta `$+942.72`, P2T DD delta `$-130.63`, Ret/P2T `0.28x` vs `-0.82x`, first..last SHA `449de3fb469d..fcfc8fa32494`.
- **`ARM_C_PORTFOLIO_DD500` vs `ARM_A_INDEP_FIXED_K` (2x):** Net PnL delta `$+1127.17`, P2T DD delta `$+67.59`, Ret/P2T `0.41x` vs `-0.82x`, first..last SHA `4bf93188f47a..7b22cae38664`.
- **`ARM_D_PORTFOLIO_DD300_NODUST` vs `ARM_A_INDEP_FIXED_K` (2x):** Net PnL delta `$+942.72`, P2T DD delta `$-130.63`, Ret/P2T `0.28x` vs `-0.82x`, first..last SHA `345baabfc5dc..6904e97aa278`.

---

**Qualification & Compliance Limits:** `RETROSPECTIVE_RESEARCH_ONLY` — Reservoir fills have `first_known_at_ms = NULL` (`strict_pit_cards_count = 0`) and cannot authorize live trading; equity anchors project historical snapshots into the OOS slice. Arms pass research qualification only when `portfolio_constraints_valid=True`, `survival_no_breach=True`, `capacity_evidence_complete=True` (`0` unknown volume trades, `<=1%` p95 / `<=5%` max 1m participation), and positive `2x`-cost EV. Live `~/.local/share/copytrade/facts`, `roster.sqlite`, and `prop-production` remain strictly untouched and un-mutated.
