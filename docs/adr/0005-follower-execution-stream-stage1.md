# ADR 0005 — Bar-Aligned `FollowerExecutionStream` Defaults for Stage-1 Scoring

**Status:** accepted (amends [ADR 0004](0004-slice4-kernel-research-path.md) §1).

## Context

In the `may_aug` / `sep_nov` Stage-2 altcoin-short cohort audits (`studies/2026-11-25__multi__10k__hybrid-altcoin-short-pod-study.py`), evaluating Stage-1 rules against raw Hyperliquid fill prints produced four systematic false exclusions/inversions:

1. Stray non-crypto HIP-3 fills (e.g. `XYZ:AAPL`, rejected by `MarketSpec`) triggered downstream symbol errors or missing `bars_1m` failures on otherwise clean crypto wallets.
2. Sub-`$15` TWAP/maker splits inside a single 5m execution bar (`bar_ms = event_ts_ms // 300_000 * 300_000`) drove raw fill `q10_order_usd < $15` even when the follower's net 5m rebalance was hundreds or thousands of dollars.
3. Unscaled 1m notional share (`wallet_usd / bars_1m.quote_volume_usd`) combined with strict zero-tolerance on any 1m print excluded large accounts (`k < 0.10`) and active traders hit by 3–4 isolated zero-volume minute prints out of ~800 bars.
4. Forward `+5m` raw fill markout penalized patient multi-hour swing traders for short-horizon entry dip noise while admitting high-churn losers with positive `+5m` noise bounces and negative cumulative trajectory PnL.

## Decision

Amends [ADR 0004](0004-slice4-kernel-research-path.md) §1 so [`score_universe`](../../src/trading_ops/research/scoring.py) scores wallets through a canonical 5m [`FollowerExecutionStream`](../../src/trading_ops/research/scoring.py) by default while preserving `WalletScorecard` and `ScoreAuditRow` boundary schemas:

* **`exclude_non_market_symbols = True`:** Strip fills on symbols containing `/`, `:`, or whitespace (`_NON_MARKET_SYMBOL_RE`, matching `MarketSpec`) before scoring, emit `UNSCORED_NON_MARKET_SYMBOLS:<n>` in `notes`, and exclude wallets with zero tradable fills under `NO_TRADABLE_MARKET_FILLS`.
* **`net_orders_by_bar = True`:** Aggregate fills into `(wallet, coin_norm, bar_ms)` rebalances with exact token inventory tracking (`position_qty = cum_sum(delta_qty)`, `position_usd = position_qty * bar_price`) and evaluate `min_baseline_order_usd = Decimal(15)` on nonzero follower bar rebalances (`abs_delta_usd > 0`).
* **`participation_scale_by_k = True`, `max_missing_volume_frac = Decimal("0.02")` (`FACT_STORE_BARS_5M_FOLLOWER_K_SCALED`):**
  * Aggregate positive finite PIT `bars_1m.quote_volume_usd` onto 5m `GRID_MS` execution buckets; exclude `VENUE_VOLUME_UNKNOWN` whenever valid buckets `== 0` or missing/zero-volume buckets exceed `max_missing_volume_frac`.
  * Missing evidence remains explicit, never silent: any wallet retained with `0 < missing <= max_missing_volume_frac * total` records `TOLERATED_MISSING_VOLUME_BARS:<wallet>:<missing>/<total>` in `UniverseScorecard.notes`, and omitted buckets are never zero-filled. Setting `max_missing_volume_frac = Decimal(0)` restores zero-tolerance exclusion.
  * Evaluate `p95` and `max` venue share on follower notional `(wallet_usd / quote_volume_usd) * suggested_k` before `[0, 1]` boundary clamping.
* **`expectancy_mode = "trajectory"` (`FOLLOWER_BAR_TRAJECTORY_EDGE`):** Default expectancy divides realized + open-tail PIT mark-to-market trajectory PnL across `(wallet, coin_norm, bar_ms)` transitions by total rebalanced turnover (`* 10_000` bps), counting held transitions against `min_markout_fills`. Opt-in modes `"bar_markout"` (`FOLLOWER_BAR_5M_MARKOUT`) and `"raw_markout"` (`FACT_STORE_BARS_1M_CLOSE_5M`) remain selectable on `ScoreConfig` (used by synthetic `+5m` forward-markout fixture tests that inject pre-`T0` flattening jumps).

## Consequences

* Upstream `copytrade_pm.roster.WalletScorecard` and `GateFailure` contracts stay binary-compatible (`p95_participation_1m`, `markout_5m_bps`, `NEGATIVE_5M_MARKOUT` hold the configured Stage-1 metric and threshold outcome while `UniverseScorecard.basis` and `ScoreConfig.data` pin the exact measurement mode inside `source_sha`).
* With `ScoreConfig(decouple_sizing_gates=True)`, Stage-1 gates evaluate intrinsic wallet evidence only while [`allocate_fixed_k_portfolio`](../../src/trading_ops/research/allocation.py) owns joint `AllocationPolicy` sizing (`portfolio_dd_budget_usd` scaling + surviving-cohort Euler marginal CDaR / participation re-evaluation) and feeds `BacktestConfig.allocation_policy`.
* All ADR-0004 strict-PIT (`first_known_at_ms <= as_of`, `allow_undated`), exclusion-over-zero-fabrication, Euler marginal CDaR, and Stage-2 kernel walk-forward invariants remain unchanged.
