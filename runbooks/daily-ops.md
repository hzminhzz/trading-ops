# Daily Operations Checklist (Read-Only)

All checks below are strictly read-only. They never mutate financial journals or authorize orders.

## 1. Ops Control Center & Chapter 2 services

- Verify user units (`copytrade-cwsa-5m-source-chapter2`, `copytrade-cwsa-5m-paper-chapter2`, `copytrade-ops-center`):
  ```bash
  systemctl --user status copytrade-cwsa-5m-source-chapter2.service copytrade-cwsa-5m-paper-chapter2.service copytrade-ops-center.service --no-pager
  curl -sf http://127.0.0.1:8888/api/health
  ```
- Inspect unified `FactStore` coverage, `RosterStore` seat books, and release status (read-only):
  ```bash
  uv run copytrade-pm facts coverage --root /home/quant/.local/share/copytrade/phase-d/facts
  uv run copytrade-pm roster show --db /home/quant/.local/share/copytrade/roster.sqlite
  uv run copytrade-pm release chapter2 explain --portfolio A --inspection metadata
  ```
- Inspect paper run and observer health:
  ```bash
  uv run copytrade-pm paper status
  uv run copytrade-pm paper observer-health
  ```

## 2. Guardian / execution side (`prop-production`)

- Check installed Guardian units only when inspecting shadow execution readiness:
  ```bash
  systemctl status prop-production-guardian.service prop-production-observer.service prop-production-bybit-observer.service --no-pager
  ```
- Keep real prop-firm challenge execution, Guardian risk checks, and HALT authority inside `/home/quant/dev/prop-production`.

## 3. Promoting allocation or risk changes

1. Audit candidates and run kernel walk-forwards via `src/trading_ops/research/` (`score_universe`, `run_kernel_backtest`) or `scripts/colab_research_run.py`, recording findings in `studies/`, `reports/`, and `proposals/`.
2. Stage roster updates and sign `portfolio_release_v2` mandates/releases offline in `/home/quant/dev/copytrade-pm` (`copytrade-pm release propose|sign|verify`) on an isolated branch.
3. Verify compatibility pins (`prop-production:src/prop_production/compatibility.json`) and promote runtime releases in `/home/quant/dev/prop-production` only through its reviewed release workflow.
