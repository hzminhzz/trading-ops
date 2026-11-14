# Daily Operations Checklist (Read-Only)

All checks below are strictly read-only. They never mutate financial journals or authorize orders.

## 1. Ops Control Center & Chapter 2 services

- Verify user units (`copytrade-cwsa-5m-source-chapter2`, `copytrade-cwsa-5m-paper-chapter2`, `copytrade-ops-center`):
  ```bash
  systemctl --user status copytrade-cwsa-5m-source-chapter2.service copytrade-cwsa-5m-paper-chapter2.service copytrade-ops-center.service --no-pager
  curl -sf http://127.0.0.1:8888/api/health
  ```
- Inspect scoped portfolio status (metadata first, then evidence if needed):
  ```bash
  uv run copytrade-pm cwsa chapter2 explain --portfolio A --inspection metadata
  uv run copytrade-pm cwsa chapter2 explain --portfolio A --inspection evidence \
    --registry-db /home/quant/.local/share/copytrade/registry/wallet_registry.sqlite \
    --source-db /home/quant/.local/share/copytrade/prospective/cwsa-5m-source-chapter2-20261112/paper-events.sqlite3 \
    --paper-root /home/quant/.local/share/copytrade/prospective/cwsa-5m-paper-chapter2-20261112
  ```

## 2. Guardian / execution side (`prop-production`)

- Check installed Guardian units only when inspecting shadow execution readiness:
  ```bash
  systemctl status prop-production-guardian.service prop-production-observer.service prop-production-bybit-observer.service --no-pager
  ```
- Keep real prop-firm challenge execution, Guardian risk checks, and HALT authority inside `/home/quant/dev/prop-production`.

## 3. Promoting allocation or risk changes

1. Model and document the proposed wallets / `FIXED_K_NATIVE_QUANTITY` `k` / DD bounds in `studies/` and `proposals/`.
2. Seal registry changes and sign mandates/releases in `/home/quant/dev/copytrade-pm` on an isolated branch.
3. Verify compatibility pins and promote runtime releases in `/home/quant/dev/prop-production` only through its reviewed release workflow.
