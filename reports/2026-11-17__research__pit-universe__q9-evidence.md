---
firm: not-applicable
account_usd: null
status: RESEARCH
top_candidate: NONE
script: null
---

# Q9 evidence — is `known_at <= T` enough to rebuild the eligible historical universe?

**Scope:** independent, read-only audit of the private SQLite wallet registry and the canonical wallet-store catalog for the Q9 point-in-time (PIT) universe / walk-forward question. No orders, no registry or service writes, no builds/tests. Registry opened with `mode=ro`; all outputs aggregate-only — no wallet IDs, addresses, aliases or per-wallet rows are reproduced here. Snapshot checkouts: `q9-pit-copytrade`, `q9-pit-tradingops`; registry `~/.local/share/copytrade/registry/wallet_registry.sqlite` (`snapshot_meta.sqlite_native_updated_at = 2026-11-16T18:22:16Z`).

**Answer in one line:** no — `known_at <= T` is necessary but not sufficient. The examined membership, alias, exclusion and monitoring tables begin on 2026-11-04; the stated cohort-based query returns **0 rows** at each tested date through 2026-11-01. This does not prove nobody knew these wallets earlier: the inspected ledger does not establish that earlier discovery history.

## 1. What the registry actually timestamps

Schema inventory (`sqlite_master`), with row counts:

| table | rows | distinct wallets | temporal column | meaning |
|---|---|---|---|---|
| `wallets` | 502 | 502 | **none** | identity only (`venue,network,address`); no `created_at`, no `first_seen` |
| `cohorts` | 7 (all `version=1`) | — | none | `source_sha256` manifest hash only |
| `cohort_memberships` | 306 | 287 | `known_at` | registration receipt time |
| `exclusions` | 295 | 294 | `known_at` | exclusion decision receipt time |
| `monitoring` | 329 | 329 | `known_at` | single frozen snapshot clock |
| `monitoring_history` | 909 | 329 | `known_at` | per-import-generation archive |
| `wallet_coverage` | 158 | 158 | `first_known` | **literal string `UNKNOWN` for all 158 rows** |
| `wallet_aliases` | 190 | 184 | `known_at` | alias binding receipt time |
| `portfolio_memberships` | 79 | — | none | resolved via `portfolio_seals.selection_known_at` |
| `qualification_decisions` | 0 | 0 | `known_at` | empty |

Observed minima for the examined registry fields:

```
cohort_memberships.known_at   2026-11-04T02:49:59.638127+00:00
exclusions.known_at           2026-11-04T02:49:59.638127+00:00
wallet_aliases.known_at       2026-11-04T02:49:59.638127+00:00
monitoring_history.known_at   2026-11-04T03:00:22.406066+00:00
monitoring.known_at           2026-11-04T09:36:57.574969+00:00
import_runs.observed_at       2026-11-04T02:48:00.594296+00:00
research_selections.known_at  2026-11-04T09:06:33+00:00
portfolio_versions.known_at   2026-11-02T07:01:53+00:00   (portfolio definition, not a wallet)
```

All 158 `wallet_coverage` rows record `first_known='UNKNOWN'` and `account_book='UNKNOWN'`; `vendor_pagination` is `UNKNOWN_PARTIAL` for 108 and `VENDOR_COIN_CENSUS_RECONCILED` for 50. These counts do not describe every wallet or prove that no independent receipt exists.

## 2. As-of reconstruction, exact counts

`eligible = cohort members with known_at <= T` minus `wallets with any exclusion known_at <= T`; cutoff is end-of-day UTC. `monitored` = rows in `monitoring` with `known_at <= T`.

| as of | cohort wallets | excluded wallets | eligible | monitored |
|---|---|---|---|---|
| 2026-05-01 | 0 | 0 | **0** | 0 |
| 2026-07-01 | 0 | 0 | **0** | 0 |
| 2026-09-01 | 0 | 0 | **0** | 0 |
| 2026-10-01 | 0 | 0 | **0** | 0 |
| 2026-11-01 | 0 | 0 | **0** | 0 |
| 2026-11-04 | 158 | 22 | 158 | 329 |
| 2026-11-05 | 214 | 22 | 214 | 329 |
| 2026-11-07 | 284 | 32 | 283 | 329 |
| 2026-11-12 | 284 | 83 | 232 | 329 |
| 2026-11-15 | 284 | 137 | **200** | 329 |
| 2026-11-16 | 287 | 294 | 124 | 329 |
| 2026-11-17 | 287 | 294 | 124 | 329 |

The alternative definition — `monitoring.configured=1` minus exclusions — gives 264 / 263 / 249 / **195** / 38 / 38 at the same 11-04/07/12/15/16/17 cutoffs. That 195 is **count-consistent only** with `reports/2026-11-16__cft__10k__canonical-full-universe-v4.md:11` ("Full roster: 195"). A count match does not establish roster identity, and the study's own loader has no as-of filter, so the honest statement is: the study's universe was built from the registry's present state, and its size is consistent with the 2026-11-15 configured-minus-excluded count.

**Scope of the zero result.** "0" above is the answer to the stated query — cohort members minus exclusions, filtered on `known_at` — not a claim that no wallet was known to anyone before 2026-11-01. The registry holds no clock that could express earlier knowledge, so this query cannot distinguish "nothing was known" from "knowledge was not recorded here".

Exclusion growth by reason and receipt date (rows): 2026-11-04 `TRUNCATED_SHORT_SPAN_NANSEN` 14 + `UNDERWATER_BAGHOLDER` 5 + `PASSIVE_MAKER_SCALPER` 3 = 22; 2026-11-07 `INACTIVE_30D_NO_FILLS` 11; 2026-11-11 `EQUITY_UNDER_1000_USD` 13; 2026-11-12 `PERP_EQUITY_UNDER_1000_USD` 38; 2026-11-15 `WALLET_AGE_UNDER_6_MONTHS` 54; 2026-11-16 `DECAY_SCAN_20261116` 157. One wallet carries two reasons (295 rows / 294 wallets); 19 wallets sit in two cohorts.

## 3. Why `known_at` is receipt time, not event time

The contract is explicit in code, not just prose:

- `src/copytrade_pm/wallet_registry.py:1081` — `asof_cutoff()` docstring: *"Use a conservative UTC knowledge cutoff, never historical event time."*
- `src/copytrade_pm/wallet_registry_native.py:569` — `append_exclusions()` docstring: *"Exclusion knowledge begins at the actual receipt time. Retrospective vendor timestamps and arbitrary manifest dates cannot backdate it."* The inserted `known_at` is `now or datetime.now(timezone.utc)`.
- `register_cohort()` in the same file stamps `known_at` with `at = now or datetime.now(...)` for every membership row, and `verify_native_cohort_archives()` asserts `row[1] == version_proof[0]` — membership clocks must equal the evidence-version receipt clock.
- `docs/WALLET-REGISTRY-AGENT-WORKFLOW.md` — *"Use the cutoff when this system knew the identity/selection, not a retrospective fill date."*

Consequence: the 2026-11-04T02:49:59Z block (158 memberships, 22 exclusions, aliases) is the **import receipt**, not when those wallets were discovered or when they began trading. The stamp is literally the wall clock of the import command: `wallet_registry.py:1203` calls `apply(args.db, snapshot, datetime.now(timezone.utc).isoformat())`, and that `when` is written onto every membership, alias, exclusion and monitoring row (`wallet_registry.py:603-605, 616-618, 634-636, 679-688`). `V2_KNOWN_AT = "2026-11-04T09:06:33+00:00"` (`wallet_registry_catalog.py:22`) is likewise a hardcoded selection-decision constant, not a wallet clock.

Three clocks are in play, and only one is in the registry:

| clock | where | meaning |
|---|---|---|
| `ts` / `source_ts` | fills, vendor payloads | **event time** — venue execution (back to 2025-05-25) |
| `known_at` | registry memberships, exclusions, aliases, monitoring | **receipt/decision time** — when this system recorded it |
| `pm01_hyperx_first_local_receipt_v1` | private forward-paper journal (`pm_hyperx_forward.py:106`, `store_first_local_receipt`) | **first local observation** — a genuine first-known clock, outside the registry |

The registry uses a conservative information-receipt convention, not the historical fill time. Reconstructing earlier knowledge requires separately verified evidence; a timestamp alone does not prove the completeness or correctness of every imported decision.

## 4. UNKNOWN is not "not admitted"

- `monitoring.admitted` and `receipt_health` are `UNKNOWN` for **all 329 rows** in the inspected table; this observation does not assert that no other admission evidence exists.
- 264 rows have `configured=1, subscribed=1`; 65 have `configured=0, subscribed=0` (of which 13 are excluded). Configured/subscribed is a tracker-subscription fact, not admission.
- **173 of 502 wallets have no `monitoring` row at all**, including 145 cohort members added on 2026-11-05 and 2026-11-07 (72 + 73). Absence means *not observed by the frozen snapshot*, not *not admitted*.
- `monitoring` has a single recorded `known_at` timestamp, and SQLite lock triggers restrict later writes (`wallet_registry_native.py:33-48`). This table alone is not a complete historical monitoring series.
- `monitoring_history` records four import generations, all stamped 2026-11-04T03:00–09:36. A wallet missing from a generation is not, by itself, proof of an unsubscribe or of flat positions.

## 5. Add / remove / exclusion restoration

- **Add:** possible and timestamped (`cohort_memberships.known_at`). Cohort versions are **monotone**: `register_cohort()` seeds `members` from version−1 and unions new entries, so a later version cannot drop a wallet. All 7 cohorts are still `version=1`.
- **Remove:** the inspected native cohort writer has no removal transition, and the examined snapshot contains no newer cohort version. This bounds the records inspected, not every possible external operation.
- **Exclusion:** append-only, one-way. `append_exclusions()` raises `EXCLUSION_ALREADY_RECORDED_REVIEW_REQUIRED` on any wallet with an existing exclusion row; no `revoked_at`, `restored_at`, `status` or history table exists (`sqlite_master` scan for `revok|restor|removed` returns nothing). `docs/WALLET-REGISTRY-AGENT-WORKFLOW.md` states *"no automatic revocation"*.
- **Restoration:** no restore event appears **in this snapshot**. Across all cohorts, `cohort_memberships.known_at > exclusions.known_at` matches **0 pairs / 0 wallets**; the reverse (membership before exclusion) matches 177 pairs / 163 wallets. Because the schema has no field to encode an un-exclusion, this shows only that none was recorded here — an operational restoration could exist without leaving a trace. An as-of filter therefore behaves as a monotone exclusion mask; it cannot reconstruct a restored state.

## 6. Event time vs knowledge time in the archives

The universe *is* loaded from the registry, but from its **present** state, not an as-of state. `studies/canonical_replay.py:156-215` (`load_registry_universe`) applies no `known_at` filter:

```sql
COALESCE(m.configured, 0) AS configured,
EXISTS(SELECT 1 FROM exclusions e WHERE e.wallet_id = w.wallet_id) AS excluded
```

Both predicates are present-state. Fills do **not** come from the wallet-store catalog: the v4 study calls `sd.discover_fill_archive(START, END, archive_root=paths.archive)` (`studies/2026-11-16__cft__10k__canonical-full-universe-v4.py:127`), which globs **daily Reservoir Parquet** under `~/.local/share/copytrade/research` (`reservoir-full-history-*/hyperliquid/fills/day=*/data.parquet`, `reservoir-monitored-plus-six-*/fills/day=*/data.parquet`) — `studies/study_data.py:24,43-73`. Those archives are a separate authority from the registry and from the wallet store.

What can and cannot be said about time:

- **Event time** is recorded per fill (`ts`, `source_ts`) and reaches back to 2025-05-25. It says when the venue executed, never when this system learned.
- **Knowledge time** for the Reservoir archives is not established by anything I examined. I did not audit their manifests or creation clocks, so no claim is made that May–Nov events were first retrievable on any particular date.
- **Local ingest time** exists for the separate canonical wallet store (`data/private/wallet_store/catalog.sqlite`): its 18,989 fills partitions carry `fetched_at` in a single 2026-11-11 window. This is not proof of first-ever acquisition; the archive-slice manifest refers to an earlier download. The wallet store is not the fills source used by v4.
- **Selection risk.** The wallet-store catalog's `last_ts` values span 2026-09-29 to 2026-11-11. This describes recorded coverage, not proof of inactivity, original discovery or why a wallet entered the dataset. Survivorship risk is not quantified by these values.

## 7. Verdict and what would be required

`known_at <= T` is **necessary but not sufficient**, for three separate reasons:

1. **Empty before 2026-11-04 for this query.** A loop using the examined registry membership and monitoring rows selects nothing before their first recorded receipt. This is a limit of the examined ledger, not proof that no wallet was known earlier.
2. **Conservative receipt, not discovery.** `known_at` is the recorded receipt/decision clock, not the first trade. Filtering it can omit a wallet for which independent evidence establishes an earlier discovery.
3. **Missing first-known evidence.** `wallets` has no creation column and `wallet_coverage.first_known` is `UNKNOWN` for all 158 rows. Other historical receipts may exist and need separate verification.

A defensible PIT universe needs auditable decision-time identity, membership and exclusion evidence—for example a verified historical snapshot or event chain—and data-availability evidence or explicit latency assumptions. When those cannot be established, retain the unknowns and label the result retrospective rather than filling them with later facts.

This is consistent with the repo's own labels: `studies/2026-11-16__cft__10k__canonical-full-universe-v4.py:1553` emits `scientific_status: RETROSPECTIVE_NO_PIT_ROSTER` and `:1676` `historic_known_at_roster: UNPROVEN`; the v4 report line 15 already says the figures are "not … PIT portfolio selection".

## 8. Archive and fold boundaries

- The [269-wallet archive](../../copytrade-pm/docs/RESERVOIR-MONITORED-PLUS-SIX-269-20261112.md#L3) covers 2026-07-01 through 2026-11-11 and was assembled from a November roster. Its catalog records `historical_first_known=false` and `study_class=RETROSPECTIVE_FIXED_MONITORED_PLUS_SIX_NOT_PIT`. Other archives can cover earlier periods but need their own scope and provenance.
- The [v4 candidate freeze](2026-11-16__cft__10k__canonical-full-universe-v4__freeze.json#L10) was created on 2026-11-16, after its May–October study period. Its original frozen members and `k` cannot be projected backward as if known during historical folds. A historical comparison must rerun the static-v4 method at each cutoff using available training data; the later frozen candidate may serve as a prospective research comparator after checking its eligibility. Formal paper financial admission and live authorization remain separate.
- In `[2026-05-01, 2026-11-01)`, a 90-day training period followed by sequential non-overlapping 30-day test windows yields three full test folds; a 60-day training period yields four. Training windows can overlap and test windows are adjacent, so the resulting estimates are not automatically independent. If no contractual 30-day deadline applies, a challenge unresolved at the end of a fold is right-censored, not necessarily a contractual timeout.
- [scikit-learn's time-series cross-validation documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) explains why training must precede evaluation and supports a gap before test; its [leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) warns that model choices and preprocessing cannot fit on test data. These sources do not establish that our already inspected historical folds are fresh.

## Sources

- `~/.local/share/copytrade/registry/wallet_registry.sqlite` (`mode=ro`) — `sqlite_master`, `pragma_table_info`, `snapshot_meta`, `schema_migrations`, and aggregate `COUNT/MIN/MAX/GROUP BY` queries over `wallets`, `cohorts`, `cohort_memberships`, `exclusions`, `monitoring`, `monitoring_history`, `wallet_coverage`, `wallet_aliases`, `portfolio_memberships`, `portfolio_versions`, `research_selections`, `portfolio_seals`, `evidence_artifacts`, `evidence_versions`, `import_runs`, `qualification_decisions`.
- `~/.local/share/copytrade/registry/pre-v6-20261104-e9951b7.sqlite` and `wallet_registry.sqlite.bak-regime-portfolios-20261105T181824Z` (`mode=ro`) — neither examined snapshot contains a pre-2026-11-04 wallet-level clock in the inspected fields.
- `/home/quant/dev/copytrade-pm`: `src/copytrade_pm/wallet_registry.py` (`asof_cutoff`, wallet/cohort as-of readers), `src/copytrade_pm/wallet_registry_native.py` (`LOCK_TRIGGERS`, `register_cohort`, `append_exclusions`, `verify_native_cohort_archives`), `src/copytrade_pm/wallet_registry_catalog.py`, `docs/WALLET-REGISTRY-AGENT-WORKFLOW.md`, `docs/SQLITE-NATIVE-LEGACY-LOCK-20261105.md`, `docs/WALLET-STORE.md`.
- `/home/quant/dev/copytrade-pm/data/private/wallet_store/catalog.sqlite` (`mode=ro`) — `wallets`, `partitions`, aggregate only. This is a **separate local store**, not the fills source used by the v4 study.
- `/home/quant/dev/copytrade-pm/data/private/registry_archive_slice/MANIFEST.json` — archive-slice provenance (`created_utc`, `"sha256-verified against the earlier download"`).
- `/home/quant/dev/trading-ops`: `studies/canonical_replay.py:156-215`, `studies/study_data.py:24,43-73` (`discover_fill_archive`, Reservoir daily Parquet), `studies/2026-11-16__cft__10k__canonical-full-universe-v4.py:127,1553,1676`, `reports/2026-11-16__cft__10k__canonical-full-universe-v4.md`, `LESSONS.md` (2026-11-16 entries 3 and 4).

## Uncertainties

- **Reservoir archive provenance.** The 269-wallet catalog's recorded cohort, dates and retrospective flags were inspected. The full set of daily Reservoir Parquet inputs used by v4 was not comprehensively audited for first-observation clocks; no claim is made about when every event first became available.
- **First fetch vs re-fetch (wallet store).** `partitions` is keyed by `(kind, wallet, day, path)` with a content SHA, so identical re-fetches may not add rows, and the archive-slice manifest refers to an earlier download that left no catalog row. 2026-11-11 is the earliest *recorded* ingest, not proof of first-ever observation.
- **Pre-registry knowledge.** Git decision JSONs, the vendor manifests behind the 2026-11-04 import, and the `pm01_hyperx_first_local_receipt_v1` journal may carry earlier first-known dates. I did not audit them; the registry records only receipt.
- **Journals and tracker state.** Not inspected (out of scope): paper journals, the live tracker subscriber state, and `portfolio_runs`/`portfolio_run_evidence` continuity.
- Counts are a single-snapshot observation at `sqlite_native_updated_at = 2026-11-16T18:22:16Z`; the registry is live and later writes will move them.
