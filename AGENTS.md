# trading-ops

Private daily trading operations, wallet evaluation, and prop-challenge research workspace.

## Authority & safety boundaries

- **Read-only desk & research only:** Never execute real orders, place prop-firm challenge trades, or bypass Guardian inside `trading-ops`. Real challenge execution belongs strictly to `/home/quant/dev/prop-production` + `/home/quant/dev/trading-connector` under Guardian risk, `HALT`, and explicit authorization.
- **No duplicate sources of truth:** Do not create a second SQLite wallet registry, financial journal, or order ledger here. Consume `/home/quant/dev/copytrade-pm` via pinned package APIs or read-only CLI commands (`copytrade-pm cwsa chapter2 ...`).
- **Promote changes upstream:** Portfolio membership, frozen `k`, signed mandates, and risk policies are drafted in `proposals/` and must be sealed, signed, and verified in `copytrade-pm` and `prop-production`.
- **Keep private state out of Git:** Never commit `.env`, API/signing keys, raw broker responses, wallet PII, or SQLite/Parquet datasets. Runtime stores remain in `/home/quant/.local/share/copytrade/`.

## Layout & workflow

- `studies/`: DuckDB/Polars wallet studies and prop-challenge risk/drawdown simulations.
- `proposals/`: Versioned candidate portfolio, `k`, and risk parameter proposals prior to upstream sealing.
- `reports/`: Sanitized performance and diagnostic summaries.
- `runbooks/`: Daily read-only monitoring and review checklists ([daily-ops.md](runbooks/daily-ops.md)).
- Use `uv run` (`Python >=3.14`) and run `uv run pytest && uv run ruff check .` after edits.
