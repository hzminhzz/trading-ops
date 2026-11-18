"""Read-only live/paper reconciliation and DextraBot pagination truncation audit."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDIES_DIR = Path(__file__).resolve().parent
if str(STUDIES_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIES_DIR))

import canonical_replay as cr
import study_data as sd

EXECUTOR_DB = Path.home() / ".local/share/prop-production/hydro-5k/executor.sqlite"
CWSA_PAPER_DBS = [
    Path.home() / ".local/share/copytrade/prospective/cwsa-5m-paper-chapter2-20261112/A/paper-events.sqlite3",
    Path.home() / ".local/share/copytrade/prospective/cwsa-5m-paper-chapter2-20261112/B/paper-events.sqlite3",
    Path.home() / ".local/share/copytrade/prospective/cwsa-5m-paper-chapter2-20261112/C/paper-events.sqlite3",
]
DEXTRABOT_CACHE_DIR = Path.home() / ".local/share/copytrade/research/dextrabot-top-copy-30d"
REPORT_MD = ROOT / "reports/2026-11-17__execution__live-paper-reconciliation.md"
REPORT_JSON = ROOT / "reports/2026-11-17__execution__live-paper-reconciliation.json"


def audit_dextrabot_cache(cache_dir: Path) -> dict[str, object]:
    """Audit cached HL ``userFillsByTime`` files without leaking raw addresses."""
    start_ms = int(datetime(2026, 10, 8, 0, 0, tzinfo=UTC).timestamp() * 1000)
    end_ms = int(datetime(2026, 11, 8, 0, 0, tzinfo=UTC).timestamp() * 1000)
    if not cache_dir.is_dir():
        return {"status": "BLOCKED_EVIDENCE:CACHE_DIR_MISSING", "total_files": 0}

    files = sorted(cache_dir.glob("*.json"))
    complete = 0
    flag_counts = {
        "TRUNCATED_PAGE_CAP": 0,
        "TAIL_WINDOW_TRUNCATED": 0,
        "UNVERIFIED_ZERO_FILLS": 0,
    }
    for fp in files:
        payload = json.loads(fp.read_text(encoding="utf-8"))
        res = sd.audit_user_fills_pagination(payload, start_ms=start_ms, end_ms=end_ms)
        if res["screening_complete"]:
            complete += 1
        for flag in res["flags"]:  # type: ignore[attr-defined]
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    return {
        "status": "COMPLETED",
        "window_utc": "2026-10-08T00:00Z..2026-11-08T00:00Z",
        "total_files": len(files),
        "screening_complete_count": complete,
        "incomplete_count": len(files) - complete,
        "usable_for_backtest_count": 0,
        "flag_counts": flag_counts,
    }


def main() -> int:
    rec = cr.reconcile_live_executor_and_cwsa(
        executor_db=EXECUTOR_DB,
        cwsa_paper_dbs=CWSA_PAPER_DBS,
        store_root=sd.DEFAULT_STORE_ROOT,
    )
    pag = audit_dextrabot_cache(DEXTRABOT_CACHE_DIR)
    payload = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "reconciliation": rec,
        "dextrabot_pagination_audit": pag,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ex = rec["executor"]
    lines = [
        "---",
        "firm: multi-venue-execution-audit",
        "account_usd: 5000",
        "status: RESEARCH",
        "top_candidate: NONE",
        "script: studies/2026-11-17__execution__live-paper-reconciliation.py",
        "---",
        "",
        "# Live Executor, CWSA Paper & Screening Pagination Reconciliation (2026-11-17)",
        "",
        "Read-only reconciliation against `executor.sqlite`, CWSA Chapter 2 paper SQLite journals, `ml4t-data` 1m OHLCV bars, and cached `dextrabot-top-copy-30d` JSONs.",
        "",
        "## 1. Bybit Executor Timing & 1m Lag Drift (`hydro-5k/executor.sqlite`)",
        "",
        f"- **Status:** `{ex.get('status')}`",
        f"- **Waves processed:** `{ex.get('waves_count')}` (wave lag vs checkpoint p50=`{ex.get('wave_lag_ms_p50')}` ms, p90=`{ex.get('wave_lag_ms_p90')}` ms, max=`{ex.get('wave_lag_ms_max')}` ms)",
        f"- **Order intents:** `{ex.get('intents_filled')}` / `{ex.get('intents_total')}` `FILLED` (`{ex.get('intents_non_filled')}` non-filled); dispatch lag p50=`{ex.get('dispatch_lag_ms_p50')}` ms, p90=`{ex.get('dispatch_lag_ms_p90')}` ms",
        f"- **1m dispatch-lag price drift (matched intents=`{ex.get('matched_1m_lag_intents')}`, unmatched=`{ex.get('unmatched_1m_intents')}`):** mean adverse=`{ex.get('mean_adverse_dispatch_lag_bps')}` bps, p50=`{ex.get('p50_dispatch_lag_bps')}` bps, p90=`{ex.get('p90_dispatch_lag_bps')}` bps",
        f"- **Broker fill price & fee reconciliation:** `{ex.get('broker_fill_slippage_status')}`",
        "",
        "## 2. CWSA Chapter 2 Paper Journals",
        "",
        "| Paper DB | Wave Receipts | Coverage Complete % | Publish Lag p50 (ms) | Publish Lag p95 (ms) | PM03 Events | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in rec["cwsa_paper"]:
        cov_pct = round(float(item.get("coverage_complete_frac", 0.0)) * 100.0, 2)
        lines.append(
            f"| `{item.get('db')}` | {item.get('wave_receipts_count')} | {cov_pct}% | "
            f"{item.get('publish_lag_ms_p50')} | {item.get('publish_lag_ms_p95')} | "
            f"{item.get('pm03_events_count')} | `{item.get('cwsa_pm03_fills_status')}` |"
        )

    fc = pag.get("flag_counts", {})
    lines.extend(
        [
            "",
            "## 3. `userFillsByTime` Pagination Audit (`dextrabot-top-copy-30d`)",
            "",
            f"- **Files audited:** `{pag.get('total_files')}` | **Screening complete:** `{pag.get('screening_complete_count')}` | **Incomplete:** `{pag.get('incomplete_count')}` | **Usable for backtest:** `{pag.get('usable_for_backtest_count')}`",
            f"- **Flags:** `TRUNCATED_PAGE_CAP={fc.get('TRUNCATED_PAGE_CAP', 0)}`, `TAIL_WINDOW_TRUNCATED={fc.get('TAIL_WINDOW_TRUNCATED', 0)}`, `UNVERIFIED_ZERO_FILLS={fc.get('UNVERIFIED_ZERO_FILLS', 0)}`",
            "",
            "---",
            "_Compliance & Qualification Caveat: Read-only observational diagnostics only; missing broker fill prices/fees (`executor.sqlite`) and empty PM03 paper `events` keep realized slippage calibration `BLOCKED_EVIDENCE` until fill-price persistence and signed mandates are enabled._",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT_MD} and {REPORT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
