# trading-ops

Private daily trading operations, wallet evaluation, and prop-challenge research workspace.

## Session Startup (Mandatory)

- **Read [LESSONS.md](LESSONS.md) first:** Before starting any new session, research, backtest, or portfolio proposal in `trading-ops`, read [LESSONS.md](LESSONS.md) in full. Record any new durable lessons learned during the session back into [LESSONS.md](LESSONS.md).
- **Check [CHANGELOG.md](CHANGELOG.md) and ADRs:** Phase D unified the research path onto `FactStore`, `RosterStore`, and `copytrade-kernel`. Read [ADR 0002](docs/adr/0002-single-canonical-rulebook-source.md), [ADR 0003](docs/adr/0003-fact-store-discovery.md), and [ADR 0004](docs/adr/0004-slice4-kernel-research-path.md) before modifying research pipelines.

## Authority & safety boundaries

- **Read-only desk & research only:** Never execute real orders, place prop-firm challenge trades, or bypass Guardian inside `trading-ops`. Real challenge execution belongs strictly to `/home/quant/dev/prop-production` + `/home/quant/dev/trading-connector` under Guardian risk, `HALT`, and explicit authorization.
- **No duplicate sources of truth:** Do not create a second SQLite wallet registry, financial journal, rulebook table, or decision simulator here.
  - Consume market/wallet facts from `copytrade_pm.facts` (`FactStore` / `facts.sqlite`) and roster/seat state from `copytrade_pm.roster` (`RosterStore` / `roster.sqlite`) via pinned package APIs or read-only `copytrade-pm` CLI commands (`facts`, `research`, `roster`, `paper`, `release`, `capabilities`).
  - Consume prop-firm challenge rulebooks, mandate types, accounting (`account_step`), and 5m checkpoint decisions (`step_checkpoint`) strictly from pinned `copytrade_kernel` (`prop-production:packages/copytrade-kernel`).
- **Promote changes upstream:** Portfolio membership, frozen `k`, signed mandates, and risk policies are drafted in `proposals/` and must be sealed, signed, and verified in `/home/quant/dev/copytrade-pm` and `/home/quant/dev/prop-production`.
- **Keep private state out of Git:** Never commit `.env`, API/signing keys, raw broker responses, wallet PII / raw `0x` addresses, or SQLite/Parquet datasets. Runtime stores remain in `/home/quant/.local/share/copytrade/`.

## Layout & workflow

- `src/trading_ops/research/`: Canonical Phase-D research package:
  - `scoring.py`: Vectorized Stage-1 PIT universe audit (`score_universe`) using `FactStore` `bars_1m` venue volume (`VENUE_VOLUME_UNKNOWN` on missing/zero volume), forward 5m markout, and exact Euler additive marginal CDaR, writing `WalletScorecard`s to `RosterStore` (`bench_from_scorecard`).
  - `backtest.py`: Stage-2 daily-reconciled walk-forward (`run_kernel_backtest`) stepping `copytrade_kernel.step_checkpoint` every 5m with `copytrade_kernel.accounting.account_step`.
  - `golden.py`: Shared `decision_sha` / `decision_payload` proven byte-identical across research backtest, PM03 paper, and `prop-production` Guardian (`tests/test_three_engine_golden_parity.py`).
- `studies/`: Portfolio construction (`pod_constructor.py`), `FactStore` discovery (`study_data.py`), and dated study scripts (e.g., `2026-11-21__cft__10k__may-nov-disjoint-10pods.py`). Legacy `portfolio_risk_policy.py`, `fast_replay.py`, and `canonical_replay.py` Guardian clones were removed in Phase D.
- `scripts/colab_research_run.py`: Google Colab research runner with `rclone` `FactStore` mirroring, venv caching, and sanitized artifact download checks.
- `proposals/`: Versioned candidate portfolio, `k`, and risk parameter proposals prior to upstream sealing.
- `reports/`: Sanitized performance and diagnostic summaries (historical pre-Phase-D reports remain as dated archive evidence only).
- `runbooks/`: Daily read-only monitoring and review checklists ([daily-ops.md](runbooks/daily-ops.md)).
- Use `uv run --frozen --group dev` (`Python >=3.14`) and run `bash scripts/verify_research.sh` after edits.
- **PIT vs retrospective provenance:** Reservoir archive shards carry `first_known_at_ms = NULL`; strict PIT (`allow_undated=False`) intentionally returns zero cards over undated shards. Any study that opts into `allow_undated=True` MUST label its output `RETROSPECTIVE_RESEARCH_ONLY` and never treat it as live qualification.
- **No in-sample `k` fitting & explicit gap handling:** Calibrate/freeze `k` strictly before the evaluation window. Never invent quotes or silently fall back to proxy volume/prices when `bars_1m` evidence is missing.

## Agent skills

### Issue tracker

Issues and specs are tracked as GitHub Issues in `hzminhzz/trading-ops` via `gh`. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses the five default triage labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repo (`GLOSSARY.md` and `docs/adr/` at the repo root, created lazily when needed). See `docs/agents/domain.md`.
