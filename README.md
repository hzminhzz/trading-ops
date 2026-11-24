# trading-ops

Private operations workspace for Phase-D wallet research, kernel-backed prop-challenge studies, Colab execution, operating runbooks, and configuration proposals.

## Responsibility split

| Repo | Role | Change cadence |
| :--- | :--- | :--- |
| `copytrade-pm` | Unified `FactStore` (`facts.sqlite`), `RosterStore` (`roster.sqlite`), source observer, PM03 virtual paper, offline signed releases | Engineering changes |
| `prop-production` | Shared `copytrade-kernel` (`packages/copytrade-kernel`), Account Guardian finance, risk checks, HALT, protected execution | Rare, controlled |
| `trading-connector` | Exchange/broker protocol transports | When needed |
| `trading-ops` (this repo) | Stage-1 `score_universe`, Stage-2 `run_kernel_backtest`, Colab runner (`scripts/colab_research_run.py`), studies, proposals, reports, runbooks | Daily |

## Quick start (read-only)

```bash
uv sync
bash scripts/verify_research.sh
uv run copytrade-pm capabilities list
```

## Key entrypoints

- **Operating contract & mandatory lessons:** [AGENTS.md](AGENTS.md) and [LESSONS.md](LESSONS.md)
- **Phase-D architecture & ADRs:** [CHANGELOG.md](CHANGELOG.md), [ADR 0002](docs/adr/0002-single-canonical-rulebook-source.md), [ADR 0003](docs/adr/0003-fact-store-discovery.md), [ADR 0004](docs/adr/0004-slice4-kernel-research-path.md)
- **Research engine:** `src/trading_ops/research/` (`scoring.py`, `backtest.py`, `golden.py`, `fee_schedule.py`, `prop_firm_fees.json`)
- **Synthetic full-pipeline E2E:** [`tests/test_full_pipeline_e2e.py`](tests/test_full_pipeline_e2e.py) (mock wallets -> `Guardian` / `FakeBroker`; establishes `ENGINEERING_VERIFIED` only)
- **Daily checklist:** [runbooks/daily-ops.md](runbooks/daily-ops.md)
