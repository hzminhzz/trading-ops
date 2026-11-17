---
firm: not-applicable
account_usd: null
status: RESEARCH
top_candidate: NONE
script: null
---

# Q10 evidence — episode underwriting and size discipline

**Scope:** read-only inspection of the current research code, a synthetic arithmetic example and a synthetic episode-extractor check. No historical comparison, broker access, registry writes or policy change. The proposed rule is not implemented or scientifically qualified by this note.

## Denominator

For a completed episode, return on peak notional (`ROPN = net_pnl / peak_notional`) is unchanged if its entire position, PnL and costs are uniformly scaled by the same positive factor. It is an equal-episode normalized performance statistic, not proof of pure alpha: sizing within an episode, timing, funding and nonlinear execution costs can still change it. The current tensor uses position quantity times the 5-minute opening price, so its sampled peak is not necessarily the intrabar peak ([tensor construction](../studies/pod_constructor.py#L501), [episode extraction](../studies/pod_constructor.py#L743)).

Two synthetic episodes illustrate why the statistic and fixed-`k` economics differ:

| Win (PnL / peak) | Loss (PnL / peak) | Mean ROPN | Total PnL |
|---|---|---:|---:|
| +$10 / $100 | −$9 / $100 | +0.5% | +$1 |
| +$10 / $100 | −$90 / $1,000 | +0.5% | −$80 |

These are arithmetic examples, not observations or forecasts. A prior-episode median denominator also describes a different statistic: it is not a replay of continuously updated `k`, since real rebalancing, costs and fills depend on the evolving position. For overlapping episodes, any historical reference used to decide an entry must have been observable before that decision.

## Current implementation is not the proposal

- [`reconstruct_execution_episodes`](../studies/pod_constructor.py#L814) orders per-symbol runs by end time but returns no start, end or censor flags. Its existing [test](../tests/test_pod_constructor.py#L125) includes a run still open at the right boundary in the returned episode array. A synthetic call independently reproduced this behavior.
- [`stage1_audit_traders`](../studies/pod_constructor.py#L1033) computes next-episode peak after a losing or winning episode in **close order**, not necessarily a new position opened after the result was known. Overlapping symbol runs can therefore precede the supposed trigger. When a win or loss comparison group is absent, it substitutes the full-episode median; a synthetic four-loss case reports tilt `1.0`, not `UNKNOWN`.
- The present [EB shrinkage](../studies/pod_constructor.py#L1093) is applied to annualized active-bar Sharpe, not to per-episode ROPN. The episode baseline currently uses mean episode PnL ([source](../studies/pod_constructor.py#L1067)). Neither is evidence that an EB-ROPN underwriting implementation exists.
- The prompt's `Spearman = -0.035` could not be located in the visible `trading-ops` studies and reports inspected for this note. Even if reproduced, Spearman measures monotonic association and does not establish independence or eliminate conditional tail exposure ([SciPy](https://docs.scipy.org/doc/scipy-1.15.3/tutorial/stats/hypothesis_spearmanr.html)).

## Implications for a research rule

1. Define episodes per wallet and symbol, including side reversals and entry/exit cost attribution. Use only genuinely uncensored episodes closed and known by the training cutoff in the ROPN score. Report boundary-censored positions and mark their open exposure separately; the completed subset can underrepresent long holds ([NIST definition of censoring](https://itl.nist.gov/div898/handbook/apr/section1/apr131.htm)).
2. Fit any empirical-Bayes pooling only on training candidates, report episode counts and uncertainty, and test its predictive calibration. Pooling shares information between candidates; it does not eliminate model or selection bias ([Stan partial pooling](https://mc-stan.org/rstanarm/articles/pooling.html), [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)).
3. Measure next-entry size only after a loss is actually known; exclude already-open episodes and label insufficient win/loss comparisons `UNKNOWN`. Retain historical 1.50 as a **provisional** research threshold, not a proven universal discipline rule. Likewise, six episodes and a p95/p50 dispersion cut are hypotheses; six episodes do not estimate a stable 95th percentile.
4. Evaluate realized size changes, costs, execution limits and synchronized losses under the actual fixed-`k` follower policy and portfolio constraints. Compare underwriting choices at matched data and constraints on a defensible point-in-time universe; the current historical windows and roster are exploratory ([Q9 PIT audit](2026-11-17__research__pit-universe__q9-evidence.md)).
