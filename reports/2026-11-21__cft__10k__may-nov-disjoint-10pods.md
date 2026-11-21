---
firm: crypto-fund-trader
account_usd: 10000
status: RETROSPECTIVE_RESEARCH_ONLY
top_candidate: MPOD-08_PERMISSIVE_OOS_RETROSPECTIVE
script: studies/2026-11-21__cft__10k__may-nov-disjoint-10pods.py
---

# May-Nov 2026 Disjoint 10-Pod Kernel Research Study (CFT $10k)

## Provenance and Isolation Disclosure

- **Strict PIT check (`allow_undated=False`):** returned `0` cards with `basis.pit=STRICT_KNOWN_AT` over `202` days as of `2026-11-19T00:00:00+00:00`.
- **Retrospective archive opt-in (`allow_undated=True`):** all Stage-1 scorecards and Stage-2 kernel walk-forwards explicitly opt into `allow_undated=True` (`basis.pit=RETROSPECTIVE_UNDATED`) over the slimmed isolated 46-wallet `FactStore` (with carried-forward monthly equity anchors under `known_at_basis=unrecorded`).
- **Isolation guarantee:** no live registry or live roster store was read or mutated; all Stage-1 and Stage-2 cards were written to temporary `RosterStore` instances only.

## Stage-1 Canonical Scoring Summary (46 Wallets, Default `ScoreConfig`)

| Window | As of (UTC) | Lookback (d) | Scored | Eligible | Excluded | Eligible aliases |
|---|---|---:|---:|---:|---:|---|
| `train_may_aug` | `2026-09-01T00:00:00+00:00` | 123 | 26 | 3 | 20 | R#062fd979, R#4c6c4326, R#91adaf10 |
| `prenov_may_oct` | `2026-11-01T00:00:00+00:00` | 184 | 21 | 0 | 25 | none |
| `nov_dump` | `2026-11-19T00:00:00+00:00` | 18 | 38 | 1 | 8 | R#ed9386cc |
| `full_may_nov` | `2026-11-19T00:00:00+00:00` | 202 | 20 | 0 | 26 | none |

### Full May-Nov (`202d`) Exclusions and Gate Failures

| Category | Code | Count | Aliases |
|---|---|---:|---|
| `EXCLUDED` | `BASELINE_ORDER_BELOW_FLOOR` | 11 | DEX-28, DEX-52, R#25396bd1, R#3c73503d, R#53736adc, R#7e9c2dd1, R#8e19f8f8, R#9e6eab28, R#aacd8793, R#e27d2580, R#ef4b034e |
| `EXCLUDED` | `VENUE_VOLUME_UNKNOWN` | 15 | R#0b231f24, R#2725e410, R#521666cd, R#574d8d61, R#6d9f5348, R#7d8dc1e8, R#8cfae98e, R#905a66b0, R#9a9dbaa4, R#cab7364c, R#d7642f60, R#e46e1d2d, R#ed9386cc, R#f1765338, W#120 |
| `GATE_FAIL` | `NEGATIVE_5M_MARKOUT` | 9 | R#115ef038, R#4c6c4326, R#592d0f78, R#876703f9, R#9346ad07, R#a596b968, R#a7bba239, R#bb01edfe, R#f2658c04 |
| `GATE_FAIL` | `P95_PARTICIPATION_ABOVE_LIMIT` | 18 | R#062fd979, R#1b6db721, R#3a6ae1c9, R#3e7de7f0, R#4c6c4326, R#592d0f78, R#5c45c20b, R#60e347aa, R#876703f9, R#8ed53fb4, R#91adaf10, R#9346ad07, R#a443d911, R#a596b968, R#affe732b, R#bb01edfe, R#cc509d58, R#f2658c04 |

### Full May-Nov (`202d`) Per-Wallet Stage-1 Metrics

| Alias | Eligible | Excluded / Failures | Observed | q10 ($) | Markout 5m (bps) | p95 Part | Max Part | CDaR Share | Bench CDaR | Max DD ($) | k | Active 14d | Cluster |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `DEX-28` | `False` | `BASELINE_ORDER_BELOW_FLOOR` | 2387 | None | None | None | None | None | None | None | None | None | None |
| `DEX-52` | `False` | `BASELINE_ORDER_BELOW_FLOOR` | 12188 | None | None | None | None | None | None | None | None | None | None |
| `R#062fd979` | `False` | `P95_PARTICIPATION_ABOVE_LIMIT` | 107 | 29.6548 | 5.8124 | 0.10515942 | 0.18325618 | 0.07043 | None | 311.18 | 0.321351 | 6 | upgma_1 |
| `R#0b231f24` | `False` | `VENUE_VOLUME_UNKNOWN` | 20 | None | None | None | None | None | None | None | None | None | None |
| `R#115ef038` | `False` | `NEGATIVE_5M_MARKOUT` | 177 | 21.2223 | -12.5134 | 0.00709068 | 0.04432691 | 0.113075 | None | 88.26 | 1.132979 | 8 | upgma_1 |
| `R#1b6db721` | `False` | `P95_PARTICIPATION_ABOVE_LIMIT` | 230 | 49.8374 | 22.6866 | 0.01185347 | 0.26421616 | -0.074849 | None | 4011.67 | 0.024927 | 4 | upgma_1 |
| `R#25396bd1` | `False` | `BASELINE_ORDER_BELOW_FLOOR` | 1788 | None | None | None | None | None | None | None | None | None | None |
| `R#2725e410` | `False` | `VENUE_VOLUME_UNKNOWN` | 8 | None | None | None | None | None | None | None | None | None | None |
| `R#3a6ae1c9` | `False` | `P95_PARTICIPATION_ABOVE_LIMIT` | 1027 | 36.5415 | 13.4779 | 0.89447849 | 1.0 | 0.132213 | None | 133983.08 | 0.000746 | 4 | upgma_2 |
| `R#3c73503d` | `False` | `BASELINE_ORDER_BELOW_FLOOR` | 426 | None | None | None | None | None | None | None | None | None | None |
| `R#3e7de7f0` | `False` | `P95_PARTICIPATION_ABOVE_LIMIT` | 6424 | 19.812 | 2.0513 | 0.92367639 | 1.00054318 | 0.016751 | None | 15157.52 | 0.006597 | 9 | upgma_2 |
| `R#4c6c4326` | `False` | `NEGATIVE_5M_MARKOUT, P95_PARTICIPATION_ABOVE_LIMIT` | 122 | 15.1002 | -1.517 | 0.16131219 | 0.24076329 | 0.0 | None | 596.84 | 0.167546 | 2 | upgma_0 |
| `R#521666cd` | `False` | `VENUE_VOLUME_UNKNOWN` | 1 | None | None | None | None | None | None | None | None | None | None |
| `R#53736adc` | `False` | `BASELINE_ORDER_BELOW_FLOOR` | 2131 | None | None | None | None | None | None | None | None | None | None |
| `R#574d8d61` | `False` | `VENUE_VOLUME_UNKNOWN` | 1 | None | None | None | None | None | None | None | None | None | None |
| `R#592d0f78` | `False` | `NEGATIVE_5M_MARKOUT, P95_PARTICIPATION_ABOVE_LIMIT` | 367 | 36.956 | -2.5931 | 1.0 | 1.00014874 | 0.251983 | None | 209.81 | 0.476603 | 9 | upgma_2 |
| `R#5c45c20b` | `False` | `P95_PARTICIPATION_ABOVE_LIMIT` | 359 | 58.6953 | 17.6577 | 0.44656885 | 0.89883343 | 0.134258 | None | 2204.05 | 0.045371 | 5 | upgma_2 |
| `R#60e347aa` | `False` | `P95_PARTICIPATION_ABOVE_LIMIT` | 138 | 32.9235 | 12.1206 | 0.51621757 | 0.64807268 | 0.028909 | None | 2939.55 | 0.034018 | 1 | upgma_0 |
| `R#6d9f5348` | `False` | `VENUE_VOLUME_UNKNOWN` | 1 | None | None | None | None | None | None | None | None | None | None |
| `R#7d8dc1e8` | `False` | `VENUE_VOLUME_UNKNOWN` | 3 | None | None | None | None | None | None | None | None | None | None |
| `R#7e9c2dd1` | `False` | `BASELINE_ORDER_BELOW_FLOOR` | 1961 | None | None | None | None | None | None | None | None | None | None |
| `R#876703f9` | `False` | `NEGATIVE_5M_MARKOUT, P95_PARTICIPATION_ABOVE_LIMIT` | 205 | 50.16 | -19.1836 | 0.98158207 | 1.0 | 0.228474 | None | 2059.88 | 0.048546 | 4 | upgma_0 |
| `R#8cfae98e` | `False` | `VENUE_VOLUME_UNKNOWN` | 9 | None | None | None | None | None | None | None | None | None | None |
| `R#8e19f8f8` | `False` | `BASELINE_ORDER_BELOW_FLOOR` | 151 | None | None | None | None | None | None | None | None | None | None |
| `R#8ed53fb4` | `False` | `P95_PARTICIPATION_ABOVE_LIMIT` | 236 | 38.784 | 2.8749 | 0.37440011 | 0.86092057 | 0.005954 | None | 1252.72 | 0.079826 | 7 | upgma_1 |
| `R#905a66b0` | `False` | `VENUE_VOLUME_UNKNOWN` | 2 | None | None | None | None | None | None | None | None | None | None |
| `R#91adaf10` | `False` | `P95_PARTICIPATION_ABOVE_LIMIT` | 375 | 93.2575 | 6.3529 | 0.01117328 | 0.52606653 | 0.004148 | None | 1522.76 | 0.06567 | 12 | upgma_1 |
| `R#9346ad07` | `False` | `NEGATIVE_5M_MARKOUT, P95_PARTICIPATION_ABOVE_LIMIT` | 116 | 21.5712 | -1.4757 | 0.0553162 | 0.07725749 | -0.033085 | None | 2188.03 | 0.045703 | 3 | upgma_1 |
| `R#9a9dbaa4` | `False` | `VENUE_VOLUME_UNKNOWN` | 1 | None | None | None | None | None | None | None | None | None | None |
| `R#9e6eab28` | `False` | `BASELINE_ORDER_BELOW_FLOOR` | 2227 | None | None | None | None | None | None | None | None | None | None |
| `R#a443d911` | `False` | `P95_PARTICIPATION_ABOVE_LIMIT` | 834 | 24.8336 | 10.6024 | 0.67936501 | 0.98222911 | -0.039929 | None | 4843.65 | 0.020645 | 7 | upgma_1 |
| `R#a596b968` | `False` | `NEGATIVE_5M_MARKOUT, P95_PARTICIPATION_ABOVE_LIMIT` | 84 | 30.6612 | -3.3336 | 0.18541541 | 0.42930977 | -0.003468 | None | 533.68 | 0.187377 | 3 | upgma_2 |
| `R#a7bba239` | `False` | `NEGATIVE_5M_MARKOUT` | 71 | 339.2575 | -2.9262 | 0.00747409 | 0.01587218 | 0.004254 | None | 17193.14 | 0.005816 | 2 | upgma_2 |
| `R#aacd8793` | `False` | `BASELINE_ORDER_BELOW_FLOOR` | 4845 | None | None | None | None | None | None | None | None | None | None |
| `R#affe732b` | `False` | `P95_PARTICIPATION_ABOVE_LIMIT` | 227 | 99.9904 | 8.7938 | 0.16176874 | 0.96153493 | -0.028439 | None | 353.68 | 0.282736 | 7 | upgma_0 |
| `R#bb01edfe` | `False` | `NEGATIVE_5M_MARKOUT, P95_PARTICIPATION_ABOVE_LIMIT` | 754 | 24.248 | -0.411 | 0.09699738 | 0.91974409 | 0.070825 | None | 10396.35 | 0.009618 | 8 | upgma_0 |
| `R#cab7364c` | `False` | `VENUE_VOLUME_UNKNOWN` | 3 | None | None | None | None | None | None | None | None | None | None |
| `R#cc509d58` | `False` | `P95_PARTICIPATION_ABOVE_LIMIT` | 1071 | 20.8753 | 3.8292 | 0.5000203 | 1.0 | 0.100942 | None | 24595.65 | 0.004065 | 2 | upgma_1 |
| `R#d7642f60` | `False` | `VENUE_VOLUME_UNKNOWN` | 1 | None | None | None | None | None | None | None | None | None | None |
| `R#e27d2580` | `False` | `BASELINE_ORDER_BELOW_FLOOR` | 421 | None | None | None | None | None | None | None | None | None | None |
| `R#e46e1d2d` | `False` | `VENUE_VOLUME_UNKNOWN` | 2 | None | None | None | None | None | None | None | None | None | None |
| `R#ed9386cc` | `False` | `VENUE_VOLUME_UNKNOWN` | 20 | None | None | None | None | None | None | None | None | None | None |
| `R#ef4b034e` | `False` | `BASELINE_ORDER_BELOW_FLOOR` | 10192 | None | None | None | None | None | None | None | None | None | None |
| `R#f1765338` | `False` | `VENUE_VOLUME_UNKNOWN` | 1 | None | None | None | None | None | None | None | None | None | None |
| `R#f2658c04` | `False` | `NEGATIVE_5M_MARKOUT, P95_PARTICIPATION_ABOVE_LIMIT` | 66 | 206.7999 | -9.6471 | 0.06367396 | 0.8651202 | 0.017556 | None | 2679.16 | 0.037325 | 2 | upgma_0 |
| `W#120` | `False` | `VENUE_VOLUME_UNKNOWN` | 9 | None | None | None | None | None | None | None | None | None | None |

## Stage-2 Kernel Walk-Forward (`run_kernel_backtest`)

| Pod | Config | Window | Status | Mode Share | Net PnL ($) | End Eq ($) | Pass Rate | Reasons | Decisions | Plans | Cards | First/Last `decision_sha` |
|---|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---|
| `MPOD-01` | `strict_gate` | `nov_dump` | `OK` | `FLAT=1.000` | 0.00 | 10000.00 | 1.00 | `none` | 0 | 18 | 46 | `none..none` |
| `MPOD-01` | `strict_gate` | `sep_nov_oos` | `OK` | `FLAT=1.000` | 0.00 | 10000.00 | 1.00 | `none` | 0 | 79 | 79 | `none..none` |
| `MPOD-01` | `permissive_constituent` | `nov_dump` | `OK` | `FLAT=1.000` | 0.00 | 10000.00 | 1.00 | `none` | 0 | 18 | 36 | `none..none` |
| `MPOD-01` | `permissive_constituent` | `sep_nov_oos` | `OK` | `FLAT=1.000` | 0.00 | 10000.00 | 1.00 | `none` | 0 | 79 | 158 | `none..none` |
| `MPOD-02` | `strict_gate` | `nov_dump` | `OK` | `FLAT=1.000` | 0.00 | 10000.00 | 1.00 | `none` | 0 | 18 | 34 | `none..none` |
| `MPOD-02` | `strict_gate` | `sep_nov_oos` | `OK` | `FLAT=1.000` | 0.00 | 10000.00 | 1.00 | `none` | 0 | 79 | 132 | `none..none` |
| `MPOD-02` | `permissive_constituent` | `nov_dump` | `OK` | `FLAT=0.167, PARTIAL=0.833` | -408.17 | 9591.83 | 0.00 | `CONTRACT_DAILY_FLOOR, INTERNAL_DAILY_STOP` | 120 | 18 | 36 | `3a5af0ffbc3d..9a9ba687e1d7` |
| `MPOD-02` | `permissive_constituent` | `sep_nov_oos` | `OK` | `FLAT=0.139, PARTIAL=0.861` | -110.38 | 9889.62 | 1.00 | `none` | 387 | 79 | 141 | `2149ba49ea66..00bc85473dd1` |
| `MPOD-02` | `permissive_constituent` | `train_may_aug` | `OK` | `FLAT=0.171, PARTIAL=0.829` | -31.12 | 9968.88 | 0.00 | `CONTRACT_DAILY_FLOOR, INTERNAL_DAILY_STOP, TRAILING_FLOOR` | 702 | 123 | 312 | `0d547dd2d729..9db172e31742` |
| `MPOD-03` | `strict_gate` | `nov_dump` | `OK` | `FLAT=1.000` | 0.00 | 10000.00 | 1.00 | `none` | 0 | 18 | 54 | `none..none` |
| `MPOD-03` | `strict_gate` | `sep_nov_oos` | `OK` | `FLAT=1.000` | 0.00 | 10000.00 | 1.00 | `none` | 0 | 79 | 97 | `none..none` |
| `MPOD-03` | `permissive_constituent` | `nov_dump` | `OK` | `FLAT=1.000` | 0.00 | 10000.00 | 1.00 | `none` | 0 | 18 | 18 | `none..none` |
| `MPOD-03` | `permissive_constituent` | `sep_nov_oos` | `OK` | `FLAT=1.000` | 0.00 | 10000.00 | 1.00 | `none` | 0 | 79 | 97 | `none..none` |
| `MPOD-04` | `strict_gate` | `nov_dump` | `OK` | `FLAT=1.000` | 0.00 | 10000.00 | 1.00 | `none` | 0 | 18 | 54 | `none..none` |
| `MPOD-04` | `strict_gate` | `sep_nov_oos` | `OK` | `FLAT=1.000` | 0.00 | 10000.00 | 1.00 | `none` | 0 | 79 | 237 | `none..none` |
| `MPOD-04` | `permissive_constituent` | `nov_dump` | `OK` | `PARTIAL=1.000` | -3.69 | 9996.31 | 1.00 | `none` | 28 | 18 | 54 | `5e99bec1ae9b..c2af2a0db1bf` |
| `MPOD-04` | `permissive_constituent` | `sep_nov_oos` | `OK` | `PARTIAL=1.000` | -71.95 | 9928.05 | 1.00 | `none` | 78 | 79 | 237 | `226961f5abb0..4a24c28f70a2` |
| `MPOD-04` | `permissive_constituent` | `train_may_aug` | `OK` | `ACTIVE=0.138, FLAT=0.203, PARTIAL=0.659` | -57.94 | 9942.06 | 0.00 | `INTERNAL_DAILY_STOP` | 167 | 123 | 361 | `22598c84c613..bf1e9904d5cb` |
| `MPOD-05` | `strict_gate` | `nov_dump` | `OK` | `FLAT=1.000` | 0.00 | 10000.00 | 1.00 | `none` | 0 | 18 | 36 | `none..none` |
| `MPOD-05` | `strict_gate` | `sep_nov_oos` | `OK` | `FLAT=1.000` | 0.00 | 10000.00 | 1.00 | `none` | 0 | 79 | 112 | `none..none` |
| `MPOD-05` | `permissive_constituent` | `nov_dump` | `OK` | `PARTIAL=1.000` | -76.46 | 9923.54 | 1.00 | `none` | 63 | 18 | 30 | `8c405f410f5a..df3c970f31b1` |
| `MPOD-05` | `permissive_constituent` | `sep_nov_oos` | `OK` | `FLAT=0.127, PARTIAL=0.873` | -192.28 | 9807.72 | 1.00 | `none` | 313 | 79 | 112 | `f42c5fc6f986..2c90831fc557` |
| `MPOD-05` | `permissive_constituent` | `train_may_aug` | `OK` | `FLAT=0.911, PARTIAL=0.089` | -8.41 | 9991.59 | 1.00 | `none` | 80 | 123 | 188 | `132d5b847e94..93c8324ec4e5` |
| `MPOD-06` | `strict_gate` | `nov_dump` | `OK` | `FLAT=1.000` | 0.00 | 10000.00 | 1.00 | `none` | 0 | 18 | 20 | `none..none` |
| `MPOD-06` | `strict_gate` | `sep_nov_oos` | `OK` | `FLAT=1.000` | 0.00 | 10000.00 | 1.00 | `none` | 0 | 79 | 79 | `none..none` |
| `MPOD-06` | `permissive_constituent` | `nov_dump` | `OK` | `FLAT=0.222, PARTIAL=0.778` | -10.62 | 9989.38 | 1.00 | `none` | 44 | 18 | 18 | `26cead97e9cb..eb98871d90d2` |
| `MPOD-06` | `permissive_constituent` | `sep_nov_oos` | `OK` | `FLAT=0.101, PARTIAL=0.899` | -12.78 | 9987.22 | 1.00 | `none` | 490 | 79 | 79 | `03d40fceefed..b933e935600e` |
| `MPOD-06` | `permissive_constituent` | `train_may_aug` | `OK` | `FLAT=0.301, PARTIAL=0.699` | -97.54 | 9902.46 | 1.00 | `none` | 1503 | 123 | 272 | `e8d1a8783023..2fbeb5901c0e` |
| `MPOD-07` | `strict_gate` | `nov_dump` | `OK` | `FLAT=1.000` | 0.00 | 10000.00 | 1.00 | `none` | 0 | 18 | 54 | `none..none` |
| `MPOD-07` | `strict_gate` | `sep_nov_oos` | `OK` | `FLAT=1.000` | 0.00 | 10000.00 | 1.00 | `none` | 0 | 79 | 125 | `none..none` |
| `MPOD-07` | `permissive_constituent` | `nov_dump` | `OK` | `PARTIAL=1.000` | -2.04 | 9997.96 | 1.00 | `none` | 112 | 18 | 36 | `92c093fd8efc..a6043c3a6973` |
| `MPOD-07` | `permissive_constituent` | `sep_nov_oos` | `OK` | `FLAT=0.038, PARTIAL=0.962` | -36.92 | 9963.08 | 0.00 | `CONTRACT_DAILY_FLOOR, INTERNAL_DAILY_STOP` | 748 | 79 | 158 | `58130875b74a..ee677390bf96` |
| `MPOD-08` | `strict_gate` | `nov_dump` | `OK` | `FLAT=1.000` | 0.00 | 10000.00 | 1.00 | `none` | 0 | 18 | 36 | `none..none` |
| `MPOD-08` | `strict_gate` | `sep_nov_oos` | `OK` | `FLAT=1.000` | 0.00 | 10000.00 | 1.00 | `none` | 0 | 79 | 124 | `none..none` |
| `MPOD-08` | `permissive_constituent` | `nov_dump` | `OK` | `FLAT=1.000` | 0.00 | 10000.00 | 1.00 | `none` | 0 | 18 | 36 | `none..none` |
| `MPOD-08` | `permissive_constituent` | `sep_nov_oos` | `OK` | `FLAT=0.380, PARTIAL=0.620` | -10.20 | 9989.80 | 1.00 | `none` | 903 | 79 | 176 | `bb49a76d68e0..caa4ad977564` |
| `MPOD-08` | `permissive_constituent` | `train_may_aug` | `OK` | `FLAT=0.455, PARTIAL=0.545` | -19.97 | 9980.03 | 1.00 | `none` | 314 | 123 | 293 | `53576ad87a5a..f1d714b37068` |
| `MPOD-09` | `strict_gate` | `nov_dump` | `OK` | `FLAT=1.000` | 0.00 | 10000.00 | 1.00 | `none` | 0 | 18 | 27 | `none..none` |
| `MPOD-09` | `strict_gate` | `sep_nov_oos` | `OK` | `FLAT=1.000` | 0.00 | 10000.00 | 1.00 | `none` | 0 | 79 | 128 | `none..none` |
| `MPOD-09` | `permissive_constituent` | `nov_dump` | `OK` | `PARTIAL=1.000` | -18.85 | 9981.15 | 1.00 | `none` | 242 | 18 | 36 | `746dc508a0ce..a732b793c268` |
| `MPOD-09` | `permissive_constituent` | `sep_nov_oos` | `OK` | `FLAT=0.722, PARTIAL=0.278` | -28.24 | 9971.76 | 1.00 | `none` | 315 | 79 | 141 | `b994b86ae8d4..94eef0ef93e4` |
| `MPOD-09` | `permissive_constituent` | `train_may_aug` | `OK` | `FLAT=0.203, PARTIAL=0.797` | -76.11 | 9923.89 | 1.00 | `none` | 763 | 123 | 268 | `cd9788e516bb..57f473bfb69e` |
| `MPOD-10` | `strict_gate` | `nov_dump` | `OK` | `FLAT=1.000` | 0.00 | 10000.00 | 1.00 | `none` | 0 | 18 | 54 | `none..none` |
| `MPOD-10` | `strict_gate` | `sep_nov_oos` | `OK` | `FLAT=1.000` | 0.00 | 10000.00 | 1.00 | `none` | 0 | 79 | 193 | `none..none` |
| `MPOD-10` | `permissive_constituent` | `nov_dump` | `OK` | `ACTIVE=0.444, PARTIAL=0.556` | -621.87 | 9378.13 | 0.00 | `CONTRACT_DAILY_FLOOR, INTERNAL_DAILY_STOP` | 45 | 18 | 54 | `f662fd0c07d8..be84152be8f3` |
| `MPOD-10` | `permissive_constituent` | `sep_nov_oos` | `OK` | `ACTIVE=0.278, PARTIAL=0.722` | -546.55 | 9453.45 | 0.00 | `INTERNAL_DAILY_STOP` | 171 | 79 | 237 | `699d84510836..283cf56bb237` |

## Multi-Firm Rulebook Catalog and `sim_inputs.json` Fee Schedule ($10k)

| Firm | Program | Target ($) | Max DD ($) | Daily ($) | Daily Basis | Split | Fee ($) | Refundable | Fee Status |
|---|---|---:|---:|---:|---|---:|---:|---|---|
| `crypto-fund-trader` | `2-Phase Evaluation` | 800 | 1000 | 500 | `balance` | 0.80 | n/a | `False` | `BREAK_EVEN_ONLY_FEE_UNVERIFIED` |
| `hyrotrader` | `Two-Step` | 1000 | 1000 | 500 | `equity` | 0.80 | 119.00 | `True` | `SOURCED_SIM_INPUTS` |
| `vanta-trading` | `Classic (1-Step Evaluation, 3% daily)` | 1000 | 500 | 300 | `balance` | 1.00 | 127.00 | `False` | `SOURCED_SIM_INPUTS` |
| `propr` | `Classic 1-Step` | 1000 | 600 | 300 | `balance` | 0.80 | 110.00 | `False` | `SOURCED_SIM_INPUTS` |

## Qualification and caveats

### Three-line caveat footer
- CFT trading restrictions and account framing require independent compliance confirmation.
- Live broker execution, funding, capacity, and strict PIT source continuity are UNPROVEN.
- No deployments, allocations, orders, service writes, or live registry/roster mutations occurred.
