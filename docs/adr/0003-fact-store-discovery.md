# ADR 0003 — Fill discovery reads the fact-store manifest, not an archive glob

**Status:** accepted (Phase D, Slice 2).

## Context

`studies/study_data.discover_fill_archive` globbed two Reservoir prefixes under
`~/.local/share/copytrade/research` and de-duplicated days by preference order.
Three properties of that approach were wrong, not merely slow:

1. **Silent omission.** The glob named `reservoir-full-history-*` and
   `reservoir-monitored-plus-six-*` only. `reservoir-roster-daily` and
   `liquid12-discovery-20261117/fills_batch2` were never read by any study, and
   the canonical wallet store was bypassed entirely — the opposite of what
   LESSONS.md asks for ("inspect local verified 1m … before declaring a gap").
2. **Non-reproducible day selection.** Two prefixes could both publish a day;
   which one won depended on glob order and on an ad-hoc preference list.
3. **No coverage answer.** A day was "present" because a file matched the glob,
   regardless of whether it had rows, and a day with a truncated tail looked
   identical to a day the archive never had.

## Decision

`study_data.discover_fact_partitions(start, end, store_root=…)` reads the
`facts.sqlite` manifest produced by `copytrade_pm.facts` and returns the day
batches for the window. `discover_fill_archive` is deleted; `import glob` is gone
from `study_data`.

* One batch per `(kind, day)` — the store has already applied priority order and
  dedupe, so a duplicate day is structurally impossible and
  `duplicate_date_sources` is always empty.
* `missing_dates` means the store has no batch for the day. A day whose input was
  read and empty is closed as a zero-row batch, so a quiet day is not reported as
  a truncated archive.
* A manifest entry whose file is missing raises `FACTS_PARTITION_FILE_MISSING`
  instead of silently shrinking the window.
* `archive_sha256` stays the SHA-256 of the concatenated per-day file SHAs, so
  existing report fingerprints still identify a content set.

`trading-ops` consumed the sibling `copytrade-pm-phase-d` worktree as an
editable path source during Phase D (the manifest module is stdlib-only and
imports without Polars), and Phase E repins it to the exact `copytrade-pm` git
revision.

## Consequences

* A study now fails loudly (`MISSING_SOURCE_DAYS`) when the facts store has not
  been migrated for part of its window, instead of quietly scoring a shorter
  window.
* Studies read the same bytes the store validated; `Decimal` columns are cast to
  `Float64` only inside the study pipeline, which is the documented Stage-1
  boundary.
* Wallet-store rows now participate in studies. The canonical `dex` label is
  normalised (`HL` → `hyperliquid`) with the raw label retained in `dex_raw`, so
  a `dex == "hyperliquid"` filter no longer drops the legacy source by accident.
* `tests/synthetic_facts.build_facts_store` builds fixtures by running the real
  migration engine over synthetic archive files, so fixture and production paths
  cannot diverge.

## Not changed

`repair_same_ms_fill_order` in `study_data` stays as a read-path safety net for
frames a study assembles itself; the fact store's `arrival_seq` is authoritative
for store-backed data.
