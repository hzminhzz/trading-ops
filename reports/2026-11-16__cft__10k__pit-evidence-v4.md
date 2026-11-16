---
firm: crypto-fund-trader
account_usd: 10000
status: BLOCKED_HISTORICAL_PIT
top_candidate: NOT_SCIENTIFICALLY_QUALIFIED
script: studies/cft_pit_evidence_audit.py
---

# Point-in-time evidence correction (not a backtest rerank)

Historical Sep 1 as-of registry: **0 provably admitted wallets** from available native metadata.
This is NOT proof that no wallets existed; the recorded metadata arrived later.
Earliest registry monitoring known-at: `2026-11-04T09:36:57.574969+00:00`.
Historical as-of CWSA original receipt rows: **0** (source journal begins later).
Reservoir exchange-time fills are not first-known receipts.

## Retrospective versus contemporary

- 128-wallet train-only ledger: 128 complete; historically profitable wallets that fail today's value gate: 24.
- The Train-only ranking is supplied in audit JSON. It ignores current account value and September-October OOS outcomes.
- September-October 2026 was already examined and cannot be relabeled untouched OOS.
- Current Bybit instrument availability is never silently promoted to historical listing or historical lot-rule proof.
- Historical Bybit reconstruction (available instruments): `{'RECONSTRUCTED_HISTORICAL_TRADING_EVIDENCE': 41}`.
- Positive historical kline evidence only confirms past exchange trading, not historical rulebook/continuous uptime.

## Status

- HISTORICAL_PIT: BLOCKED (no pre-September known-at roster).
- HISTORICAL_SOURCE_FIRST_KNOWN: BLOCKED (Reservoir backfill has no original receipt timestamps).
- HISTORICAL_BYBIT_TRADABILITY: RECONSTRUCTED where launch bounds and completed candles exist; otherwise UNKNOWN.
- HISTORICAL_LOT_RULES: UNKNOWN.
- SEPTEMBER_OCTOBER_INDEPENDENT_OOS: IMPOSSIBLE_TO_RECOVER.
- SCIENTIFICALLY_QUALIFIED=false; ECONOMICALLY_QUALIFIED=false; TRADING_AUTHORIZED=false.

## New prospective untouched holdout

- Freeze: `2026-11-16T14:52:45.781131+00:00`.
- Start: `2026-11-17T00:05:00+00:00` (strictly after freeze).
- Mature: `2026-12-17T00:05:00+00:00`.
- Payload digest: `25f32efa649731c85a68775aa05a8bce9fe091d258f9b323bcaaffcfee419c24`.
- Portfolio definitions and k are frozen without reading future holdout observations.
- Original first-known source collection for these new wallets remains unverified; no watcher/service or orders were enabled.

No changes to Guardian, broker keys, tracking registry, services or orders.
