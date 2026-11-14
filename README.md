# trading-ops

Private operations workspace for daily strategy analysis, wallet research, operating runbooks, and configuration proposals.

## Responsibility split

| Repo | Role | Change cadence |
| :--- | :--- | :--- |
| `copytrade-pm` | Source observer, SQLite registry, virtual paper, signed releases | Engineering changes |
| `prop-production` | Guardian finance, risk checks, HALT, protected challenge/live execution | Rare, controlled |
| `trading-connector` | Exchange/broker protocol transports | When needed |
| `trading-ops` (this repo) | Wallet studies, prop-DD modeling, operating checklists, proposals, reports | Daily |

## Quick start (read-only)

```bash
uv sync
uv run pytest && uv run ruff check .
uv run copytrade-pm cwsa chapter2 explain --portfolio A --inspection metadata
```

See [AGENTS.md](AGENTS.md) for strict safety boundaries and [runbooks/daily-ops.md](runbooks/daily-ops.md) for the daily checklist.
