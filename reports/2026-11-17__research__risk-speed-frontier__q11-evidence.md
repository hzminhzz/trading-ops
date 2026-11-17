---
firm: crypto-fund-trader
account_usd: 10000
status: RESEARCH
top_candidate: NONE
script: null
---

# Q11 evidence — challenge sizing frontier

**Scope:** read-only contract and research-code analysis, official-source checks and synthetic arithmetic/counterexamples. No historical ladder comparison, broker access, orders or policy changes. Numeric caps are proposed experiment settings, not validated limits or live authorization.

## Account-barrier mismatch must be resolved before the experiment

CFT states that a two-phase breach is checked on equity, while the daily loss limit is calculated from the balance at 00:05 UTC and the overall loss limit is fixed from initial balance ([CFT evaluation process](https://cryptofundtrader.com/evaluation-process/)). The precise contract and calculation for the selected account/add-ons must be confirmed. The firm's [FAQ](https://cryptofundtrader.com/faq/) says no minimum evaluation trading days, while the evaluation-process page says five; pass-speed results must not assume either without resolving the account's terms. Both pages describe an unlimited evaluation period, so a research horizon is not necessarily a contractual timeout.

The research [`GuardianSpec`](../studies/canonical_replay.py#L90) sets a fixed `$500` daily limit and `$400` sweep. Its [daily reset](../studies/canonical_replay.py#L808) assigns `cashday = eq`, which is modeled mark-to-market equity, and its [breach check](../studies/canonical_replay.py#L1043) compares equity with that equity-reset baseline and the fixed dollar limit. This is **not an implementation of the documented balance-reset rule**. It makes no claim about the installed production Guardian.

A deterministic synthetic call to `simulate_guardian_challenge` demonstrated the difference. Zero costs; one position loses $100 and closes before reset, then a new position has a $200 unrealized loss at reset. Reset balance is $9,900 and equity is $9,700. Equity later falls to $9,399. Both $9,900 − $500 = $9,400 and, if the limit is 5% of the reset balance, $9,900 × 0.95 = $9,405 are breached. The research simulator returns its internal `TIMEOUT` marker with **zero sweeps**, because the post-reset loss from equity is only $301. This is a synthetic code counterexample, not a broker execution or a live Guardian finding.

## What the existing sizing fields mean

- The existing v4 [$200 baseline](../studies/2026-11-16__cft__10k__canonical-full-universe-v4.py#L480) apportions a $200 standalone-wallet drawdown budget across members and applies the smallest member cap ratio uniformly. It is not itself a guaranteed $200 account-loss limit.
- The joint optimizer's [`max_static_dd`](../studies/pod_constructor.py#L1476) field comes from peak-to-trough maximum drawdown, not the initial-account static-floor distance. E.g., equity `10000 → 10800 → 10100` has $700 peak-to-trough drawdown but no drawdown below initial equity. Record both metrics under their correct names.
- Its nonnegative [risk-share parameter](../studies/pod_constructor.py#L1520) is based on standalone CDaR; it is a capital-allocation proxy, not a true marginal contribution to net portfolio risk. The existing configuration uses 15%–45% ([source](../studies/pod_constructor.py#L203)). With three proxy shares summing to one, a 35% maximum forces each to be 30%–35%. The optimizer [relaxes infeasible share bounds](../studies/pod_constructor.py#L1562); a research rule described as hard must not silently do so.
- The v4 research reports use the 2% baseline and 5x maximum gross leverage; modeled peak leverage in its highlighted 3-seat candidate was 0.93x, not evidence that a higher leverage cap is safe ([v4 report](2026-11-16__cft__10k__canonical-full-universe-v4.md#L19)).

## Failure probabilities and EV

With zero failures in four genuinely independent, identically distributed attempts, the exact one-sided 95% binomial upper bound for the unknown failure probability is `1 − 0.05**(1/4) = 52.7%`; with zero failures in 29 it is `9.8%`. This calculation assumes independent attempts with the same underlying probability, not resampled versions of one market history ([exact binomial intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html)). A large number of bootstrap draws does not add independent market histories; time-series dependence and small-sample behavior are substantive modeling assumptions ([Federal Reserve research](https://www.federalreserve.gov/pubs/feds/1996/199645/199645pap.pdf)).

Choose a 90-day cash-flow horizon and 60-day sensitivity. Report pass, failure and still-open counts separately; estimate monetary EV only with explicit purchase/reset fees, both evaluation phases, funded-stage reward and cost assumptions. Do not double-count execution costs already included in modeled rewards. A 10% failure ceiling is an optional risk preference, not an empirically demonstrated threshold. If the required economics or failure calibration are missing, show scenario break-even/frontier results and do not name a proven winner.

The Q3 historical comparison is retrospective, uses an unproven PIT roster, and has coverage and selection limitations ([sleeve comparison](2026-11-17__cft__10k__sleeve-vs-upgma.md#L190), [Q9 PIT audit](2026-11-17__research__pit-universe__q9-evidence.md)). Do not select an operating tier from those already-inspected results as if they were an untouched test.
