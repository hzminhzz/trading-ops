# trading-ops

Read-only daily trading operations, wallet evaluation, and prop-challenge research workspace.

## Startup steps (mandatory)

1. **Read [LESSONS.md](LESSONS.md) in full** before any research, backtest, or portfolio proposal; append a dated, evidence-backed entry whenever a study reveals a durable rule.
2. **Read [CHANGELOG.md](CHANGELOG.md) and [ADR 0004](docs/adr/0004-slice4-kernel-research-path.md)** before editing research code so changes stay on the Phase-D kernel path.

## Hard guardrails

- **Read-only research only:** Real orders, challenge execution, and Guardian authority belong strictly to `/home/quant/dev/prop-production` + `/home/quant/dev/trading-connector`.
- **Single sources of truth:** Consume facts from `copytrade_pm.facts` (`FactStore`), seats from `copytrade_pm.roster` (`RosterStore`), and rulebooks/accounting/5m decisions from pinned `copytrade_kernel` (`prop-production:packages/copytrade-kernel`). Draft proposals in `proposals/` and promote them upstream to `/home/quant/dev/copytrade-pm` and `/home/quant/dev/prop-production`.
- **No private state in Git:** Keep `.env`, keys, raw broker responses, wallet PII / raw `0x` addresses, and SQLite/Parquet datasets outside Git (runtime stores live in `/home/quant/.local/share/copytrade/`).
- **PIT provenance & sizing:** Reservoir shards carry `first_known_at_ms = NULL`; any run opting into `allow_undated=True` MUST label outputs `RETROSPECTIVE_RESEARCH_ONLY`. Calibrate/freeze `k` strictly pre-evaluation and leave missing `bars_1m` evidence excluded rather than inventing proxies.

## Context pointers

| Branch | Read / run |
|---|---|
| Stage-1 audit (`score_universe`), Stage-2 walk-forward (`run_kernel_backtest`), 3-engine `decision_sha` (`src/trading_ops/research/`) | [ADR 0004](docs/adr/0004-slice4-kernel-research-path.md) |
| Prop-firm rulebooks (`copytrade_kernel.RulebookCatalog`) | [ADR 0002](docs/adr/0002-single-canonical-rulebook-source.md) |
| Fact discovery (`discover_fact_partitions` over `FactStore` manifests) | [ADR 0003](docs/adr/0003-fact-store-discovery.md) |
| Google Colab execution (`rclone` FactStore mirror, venv cache) | `scripts/colab_research_run.py` (`--help` for flags) |
| Daily read-only monitoring & upstream promotion checklist | [runbooks/daily-ops.md](runbooks/daily-ops.md) |
| Verification (`Python >=3.14` via `uv run --frozen --group dev`) | `bash scripts/verify_research.sh` (must pass all tests & ruff) |
| GitHub issues, triage labels, domain glossary/ADRs | [issue-tracker.md](docs/agents/issue-tracker.md), [triage-labels.md](docs/agents/triage-labels.md), [domain.md](docs/agents/domain.md) |
