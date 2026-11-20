# ADR-0002: Single Canonical Rulebook Source (Kernel Catalog, No Local Rulebook Table)

- **Status:** `IMPLEMENTED / ENGINEERING_VERIFIED` (research scope only; no live authorization, no promotion claims).
- **Date:** 2026-11-20
- **Scope:** `trading-ops` research rulebook consumption (`studies/pod_constructor.py`, `studies/2026-11-18__cft__10k__phase3-pod-shop-ablation.py`), against `prop-production/packages/copytrade-kernel` (`copytrade_kernel.rulebooks`).

## 1. Context

`trading-ops` re-published the same challenge terms as `prop-production`/`copytrade-pm` in a private
`prop_firms_rules.json` table, parsed by a hand-written clause reader in `pod_constructor`
(`load_prop_contract_spec`, including `AMBIGUOUS (...)` clause hints, `(Grow only)` guards and
`100/0 default` special cases). Every prior drift bug in this family came from hand-syncing those
dialects (see [LESSONS.md](../../LESSONS.md) 2026-11-18 §4). A `scripts/check_rulebook_drift.py`
regex checker existed in some checkouts to detect divergence after the fact; it caught nothing
structurally because each copy could be internally consistent and mutually different.

The Slice 1 kernel extraction produced `copytrade_kernel.rulebooks.RulebookCatalog`, which already
carries the resolved per-program, per-phase published terms for `crypto-fund-trader`, `propr`,
`hyrotrader`, `vanta-trading`, `northbook` and `hypernova` at a `$5,000` base.

## 2. Decision

1. Research consumes **only** `copytrade_kernel.rulebooks.RulebookCatalog`; `trading-ops` keeps no
   challenge-term data file of its own.
2. `pc.load_prop_contract_spec(...)` is now a thin **rescaled view** of catalog specs: it selects a
   program by name, sorts `phase_1..n`, resolves the funded phase (same program first, then the
   firm's first funded entry), and rescales `initial`/`target`/`daily_loss`/`trailing_loss` to the
   requested account size. It invents nothing: a missing amount raises
   `PROP_CONTRACT_INCOMPLETE:<key>`, an unknown firm raises `PROP_FIRM_NOT_FOUND:<slug>` and an
   unknown program raises `NO_MATCHING_PROP_PHASES:<slug>:<filter>`.
3. **Fees are never derived.** `challenge_fee_usd` stays `None` unless the caller supplies a
   verified price, so contract EV keeps reporting `BREAK_EVEN_ONLY_FEE_UNVERIFIED` rather than a
   fabricated net EV.
4. `prop_firms_rules.json` and `scripts/check_rulebook_drift.py` are deleted; the drift script is
   replaced by a structural single-source test
   (`tests/test_rulebook_catalog_single_source.py`).

## 3. What this does not settle

- **Slice 4 — resolved.** `studies/canonical_replay.py` `GuardianSpec`,
  `simulate_guardian_challenge`, the CWSA one-shot replay and the
  `portfolio_risk_policy` replay loops are deleted, together with the dated studies
  that drove them; the replacement research path and its cross-engine parity proof
  are recorded in [ADR 0004](0004-slice4-kernel-research-path.md). That ADR also
  records the capability the removal dropped (containment/lifecycle simulation),
  which no kernel contract currently covers.
- The kernel catalog currently covers six firms; the deleted JSON also carried three
  `e8_markets_perp_*` slugs. No `trading-ops` study or test consumes E8, so nothing regressed, but
  E8 terms are **not** available through the kernel yet. They must be added upstream (with sources)
  before any E8 research resumes — never re-created locally.
- Fees were previously parsed from the JSON `pricing_usd` block by some call sites; all research
  call sites already passed `challenge_fee_usd=None` or an explicit sensitivity value, so no
  numeric result depended on it.

## 4. Evidence

- Parity harness over 36 `(firm, program, account_size, min_eval_trading_days)` cases: every
  numeric field (`profit_target_usd`, `max_static_dd_usd`, `daily_loss_limit_usd`,
  `daily_loss_basis`, `min_trading_days`, `consistency_max_single_day_pct`, `payout_split_pct`,
  `funded_cycle_days`) matched the pre-cutover JSON loader. Only display labels changed
  (`program_name` now names the selected program; eval phase names are the canonical
  `phase_1`/`phase_2`, which also makes `phase_pass_rates_pct` keys consistent across firms).
- `uv run --frozen pytest -q`: 141 passed, 2 skipped. `uv run --frozen ruff check .`: clean.
