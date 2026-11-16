---
firm: crypto-fund-trader
account_usd: 10000
status: ENGINEERING_VERIFIED_RETROSPECTIVE_PIT_BLOCKED
top_candidate: NOT_SCIENTIFICALLY_QUALIFIED
script: studies/cft_pit_evidence_audit.py
---

# CFT PIT audit and prospective correction — 16 November 2026

**Decision:** Previous May–November portfolio research must remain `RETROSPECTIVE_SCENARIO`, not PIT-verified, scientifically qualified, or independent OOS. Original September–October performance is not an untouched holdout. No new live trading authority exists.

## Five original defects and verified remedies

| Defect | What the original evidence shows | Implemented correction | Remaining status |
|---|---|---|---|
| Historical point-in-time wallet universe | Native monitoring and aliases are first known on 2026-11-04; as-of 2026-09-01 returns **0 provably registered** wallets | New `registry_asof` admits only wallet identities with original monitoring/alias known-at by cutoff; no implied discovery from old fills; immutable wallet ID and address-digest in forward freeze | **Historical PIT impossible from available metadata** |
| Historical Bybit symbol availability | Current metadata alone did not prove historical trading | Reconstructed `launchTime`/`deliveryTime`, 1h *completed* Bybit klines at Sep 1, per-fill May–Aug scan, optional coin-activation in canonical 5m replay | **Reconstructed listing/trading evidence**, historical temporary suspensions and lot rules unknown |
| Current wallet account value influences history | 24 historically Train-profitable wallets were filtered out by current operational checks | `train_only_selection` uses historical Train statistics and historical exclusions only; never current value or OOS returns | Train-only ranking corrected; old portfolio selection remains ex-post |
| Historical source first-known timestamps | Reservoir fill archives have `exchange_ts`, not an original receipt clock. Inspected CWSA source journal begins in November | `validate_source_admission` blocks missing, late, impossible or after-decision first-known events; original source-clock census is read-only | **May–October receipt provenance not available** |
| Untouched OOS | Historical Sep–Oct had already been opened and rerun | Immutable future freeze, digest/identity checks, holdout maturity gating, old OOS explicitly marked contaminated | **Prospective holdout frozen, no qualified source observation enabled** |

## Underlying inputs and measured coverage

- Original 195-wallet current registry and 128-wallet Train financial ledger kept unchanged and preserved with their source digests.
- Historical registry cutoff: `2026-09-01T00:00:00Z`. **0 wallets** have a native monitoring known-at by that cutoff. This is **not** evidence that no wallet traded or could have been discovered earlier.
- CWSA Chapter 2 source journal: **0** available receipt clocks before historical cutoff; source receipts do exist beginning in November, but coverage is limited to the native owner's selected scope.
- Bybit public V5 market metadata: present-day Trading **and Closed** USDT linear perpetuals, with observed launch and delivery bounds. The account's **private API credentials were not opened**.
- All **41** symbols referenced by the three historical finalists had completed historical Bybit 1-hour kline evidence as of 2026-09-01. This supports exchange trading on that date, **not** continuous historical listing status, source availability, or historical order-size specifications.
- Local Reservoir archive May–August: 604,957 main-DEX fills from **194** wallets with activity (the scan includes registered wallets beyond the 128 Train-eligible subset). Public current Bybit metadata flags **6,854 prelaunch fills**, **15,301 after reconstructed delivery/delist time**, and **2,132 unknown-market fills**. Treat this as venue-screening evidence, not exact real follower missed trades; some markets were already excluded from the prior follower simulation.
- Three aliases changed within the native registry after the earlier study (`W#111`, `W#152`, `W#312` now have explicit aliases). The listing audit uses original wallet IDs and reports the label drift. Because the old roster digest omitted wallet addresses, historical address-identity provenance remains separately unproven.

## Listing-sensitive follower replay (previously opened historical window)

Strict new-only 5m replay and Hyperliquid local 1m OHLCV are unchanged. The canonical reconstructor now admits a symbol only after its historical Bybit launch time and, for strict new-only, only after the original source inventory goes flat and genuinely reopens/flips.

| Frozen portfolio | Original Train modeled net | Venue-gated Train modeled net | Change | Replayed Sep–Oct modeled net (unchanged frozen k) |
|---|---:|---:|---:|---:|
| W#111, W#174, W#179, Whale_0x5f94 | +$1,599.72 | **+$1,491.42** | **-$108.30** | +$1,435.71 |
| W#1, W#174, W#264, Whale_0x5f94 | +$1,518.87 | +$1,518.87 | $0 | +$833.55 |
| W#174, W#252, W#255, Whale_0x5f94 | +$1,235.81 | +$1,235.81 | $0 | +$838.75 |

The corrected Train-only half-risk baseline did **not** require decreasing any of the three previously frozen `k` configurations. This does **not** certify 2026 historical selection, original source availability, Bybit order execution, or OOS independence. This sensitivity uses already-opened OOS data and is therefore **post-hoc only**.

## Frozen future holdout

- Original hypotheses: the same three four-wallet portfolios and their existing frozen `k`; 8 unique wallets, 12 member slots.
- Freeze SHA-256 payload: `25f32efa649731c85a68775aa05a8bce9fe091d258f9b323bcaaffcfee419c24`.
- Starts: `2026-11-17T00:05:00Z`.
- Ends (exclusive): `2026-12-17T00:05:00Z`.
- The freeze contains native wallet IDs and SHA-256 of addresses, **not** raw account addresses or secrets.
- The test gate checks exact frozen digest and every wallet identity. It does not treat alias renaming as an account swap.
- **No forward data has been accessed.** The holdout cannot be evaluated until mature. Source receipt-chain coverage for these members remains **not enabled/unverified**, so maturity alone will not admit economic or scientific qualification.
- No broker order, Guardian policy change, real-account trade or service deployment was performed.

## Reproduction

```bash
uv run --frozen --group dev python studies/cft_holdout_gate.py
uv run --frozen --group dev python studies/cft_holdout_gate.py --evaluation-preflight
uv run --frozen --group dev pytest -q
uv run --frozen --group dev ruff check .
bash scripts/verify_research.sh
```

The other one-time audit scripts are intentionally fail-closed and **refuse to overwrite the versioned outputs**:

- `studies/cft_pit_evidence_audit.py --freeze-forward`
- `studies/cft_train_bybit_listing_audit.py`
- `studies/cft_pit_listing_sensitivity.py`

## Artifacts

- `2026-11-16__cft__10k__pit-evidence-v4__audit.json`: as-of native registry, original source clocks, Train-only selection, historical Bybit evidence
- `2026-11-16__cft__10k__pit-evidence-v4__train-listing.json`: historical market activation versus full selected-wallet Train fill archive
- `2026-11-16__cft__10k__pit-evidence-v4__listing-sensitivity.json`: risk-capped follower replay sensitivity
- `2026-11-16__cft__10k__pit-evidence-v4__forward_freeze.json`: new immutable prospective hypothesis freeze
- `2026-11-16__cft__10k__pit-evidence-v4.md`: initial PIT audit report

**Qualification:** ENGINEERING_VERIFIED for scoped guardrails and retrospective audit; RETROSPECTIVE_SCENARIO for modeled PnL; **NOT SCIENTIFICALLY QUALIFIED**, **NOT ECONOMICALLY QUALIFIED**, **NOT TRADING AUTHORIZED**. Historical point-in-time evidence cannot be manufactured by retrospective queries.
