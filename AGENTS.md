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
- **Historical backtest default:** `[2026-07-01T00:00Z, 2026-11-01T00:00Z)`; calibrate/freeze wallet `k` on earlier data (the default train end is October 1), with `k` sized so the modeled wallet max drawdown uses **50% of its allocated risk**. Never fit `k` using the evaluation window.
- **Market-data-gap policy:** a single unpriced pair must not halt the whole portfolio. Inspect local verified 1m, then connected Vike 1m and the verified Reservoir market price source before declaring a gap. Distinguish normal trade-sparse minutes from genuine cutoff. In research only, model a pair-local follower exit with explicit stale-price age, adverse slippage/fees and `executed=false`, and continue independent pairs. No invented quote, no claim of actual forced execution, and no promotion of retrospective source books to PIT/CWSA/trading authorization.
