---
firm: not-applicable
account_usd: null
status: RESEARCH
top_candidate: NONE
script: null
---

# Q8 evidence — venue order lifecycle, reduce-only/netting, and execution-model validation

**Scope:** independent evidence for the Q8 research-to-production bridge. Read-only desk research: no orders, deployments, registry or service writes. Sources fetched 2026-11-17; the banking guidance is an analogy, not an applicable legal mandate.

**Why this note exists:** independently check whether reusing the existing paper engine and projecting a research policy into a signed release preserves execution semantics. Code, venue documentation and model-validation guidance answer different parts of that question.

## 1. Order lifecycle is not a two-state (open/closed) model

Bybit V5 enumerates separate **open** and **closed** statuses, and the closed set is not equivalent to "no fill":

- Open: `New`, `PartiallyFilled`, `Untriggered` (conditional orders).
- Closed: `Rejected`, `PartiallyFilledCanceled`, `Filled`, `Cancelled`, `Triggered`, `Deactivated`.
- `PartiallyFilledCanceled` is annotated **"Only spot has this order status"**.
- `Cancelled` is annotated: *"In derivatives, orders with this status may have an executed qty"*.
- `Deactivated`: *"UTA: Spot tp/sl order, conditional order, OCO order are cancelled before they are triggered"*.
- `Triggered` is an instantaneous transition from `Untriggered` to `New`.

The same page defines the position-mode identifier: `positionIdx` `0` = one-way, `1` = hedge Buy side, `2` = hedge Sell side.

Source: <https://bybit-exchange.github.io/docs/v5/enum>.

Consequences for a derivatives bridge:

- **Terminal status is not a fill quantity.** Mapping `Cancelled` → zero executed quantity mis-states realized risk; quantity must come from cumulative-executed fields.
- **Position mode matters.** The API identifies one-way and hedge sides with different `positionIdx` values; the executor must verify the actual account mode rather than assume either one.

## 2. Execution is bounded, and reduce-only orders can be split

From the Place Order reference:

- Market orders are not unbounded sweeps: *"in order to protect traders against the serious slippage of the Market order, Bybit trading engine will convert the market order into an IOC limit order for matching. If there are no orderbook entries within price slippage limit, the order will not be executed. If there is insufficient liquidity, the order will be cancelled."* The API offers optional `slippageToleranceType` and `slippageTolerance` fields; market orders use `IOC`.
- *"If reduceOnly=true and order qty > max order qty, the order will automatically be split up into multiple orders."*
- `positionIdx` is **required** under hedge mode (`0`/`1`/`2`); `PostOnly` orders are **cancelled** if they would fill immediately.

Source: <https://bybit-exchange.github.io/docs/v5/order/create-order>.

Consequences:

- **A research fill is not a guarantee of a production fill.** Production can return no fill or cancel an order instead of executing at the research price. Point-in-time depth can inform an estimate, but later depth and the actual fill remain uncertain.
- **One logical intent need not equal one venue order.** Auto-splitting changes the order and fill records that need reconciliation; the documented split alone does not establish a different total fee.
- **`reduceOnly` is not a fill guarantee.** Reconcile accepted orders and cumulative fills against current account positions and unresolved orders.

## 3. Independent model-validation authority: validate before use, then compare to realized outcomes

The current US interagency model risk guidance (revised guidance dated April 17, 2026; SR-26-2) states, verbatim:

- *"Validation generally occurs prior to a model's first use."*
- *"Outcomes analysis compares model outputs to corresponding real-world outcomes to assess model performance relative to model objectives and business use."*
- *"Ongoing model monitoring involves an evaluation of the extent to which a model is performing as expected given potential changes in products, exposures, activities, clients, data relevance, or market conditions."*
- *"Using a model beyond its intended purpose introduces additional uncertainty and risk."*

Source: <https://federalreserve.gov/frrs/guidance/supervisory-guidance-on-model-risk-management.htm>.

**Caveat, stated in the document itself:** it *"does not set forth enforceable standards or prescriptive requirements"* and is addressed to banking organizations. It is cited only as a validation-principle reference, not as law or regulation binding this desk; no legal conclusion is drawn.

## What this constrains in the Q8 bridge (challenge to option 4, not confirmation)

1. **PIT-clock parity is necessary but not sufficient.** Two runs agreeing on timestamps says nothing about fill semantics. The gate also needs a realized-vs-modelled comparison (fill rate, partial-fill distribution, fee/slippage error, cancelled-with-execution cases) before promotion.
2. **Do not infer the filled quantity from a request or a status alone.** Reconcile cumulative fills, active orders and actual position, including venue splitting and IOC cancellations.
3. **Engine reuse does not establish realism.** Verify its input, capacity, order-state and fill assumptions for the proposed use.
4. **Position mode must be an explicit, checked input.** A projection that assumes the wrong mode cannot be reliably reconciled against venue positions.

## 4. Independent order-state example

The [FIX Trading Community order-state specification](https://www.fixtrading.org/wp-content/uploads/download-manager-files/FIX-Latest-as-of-EP284-Order-State-Changes.pdf), section B.1.b, explicitly describes executions while cancellation is pending. This is a general protocol example, not a claim that Bybit uses FIX or that a particular account behaved this way. A cancellation request must not be treated as an already-cancelled order or as proof of zero further fills.

## 5. Local contract cross-check

- [`portfolio_release_v1`](../../prop-production/src/prop_production/release.py#L176) has an exact member field set; `FIXED_K_NATIVE_QUANTITY` requires `0 < k <= 10`. This does not prove what other release versions or protective workflows support.
- The [production sizing contract](../../prop-production/docs/SIZING-POLICY.md#L14) distinguishes requested notional from follower quantity and treats targets as proposals independently checked by Guardian.
- [PM03 `execute_targets`](../../copytrade-pm/src/copytrade_pm/pm03_execution.py#L200) is hypothetical paper execution and can report bounded partial fills. It blocks a batch when required quotes or capacity are missing.
- The [trading-ops reconciliation](../studies/canonical_replay.py#L1253) passes synthetic volume of one billion units and `max_participation=1`. Its accounting reconciliation is not a validation of actual liquidity or a 5% participation policy.
- The [checkpoint mandate implementation](../../prop-production/src/prop_production/copy_mandates.py#L225) already defines signed pause/retire transitions that stop new risk while preserving obligations. [Checkpoint admission](../../prop-production/src/prop_production/checkpoint_admission.py#L175) excludes non-active owners and holds their existing books frozen. This is not proof that the complete proposed leader-following freeze is available, deployed or authorized.
- The [approved checkpoint ADR](../../prop-production/docs/ADR-CANONICAL-5M-CHECKPOINT-GRILL-20261111.md#L53) permits bounded re-evaluation before possible dispatch, but not a blind retry after dispatch or an unknown outcome. Its Q15 allows a guarded late entry on a genuine increase; [strict research new-only](../studies/canonical_replay.py#L358) instead waits for a flat-to-open or side-flip event. These are distinct strategies, not equivalent timing details.
- The [native 1m research replay](../studies/native_1m_pm03_replay.py#L431) uses one-shot entries and explicitly rejects a partial execution. PM03's capacity-bounded partial-fill support therefore cannot be treated as a feature of every existing research replay.
- The candidate [compatibility manifest](../../prop-production/src/prop_production/compatibility.json#L14) names v1 as the default release contract, supports v1 and v2 schemas, and records shadow/non-authorized settings. It does not establish the actually installed revision or account permission.
- The [Q3 sleeve comparison](2026-11-17__cft__10k__sleeve-vs-upgma.md#L180) was retrospective and gap-invalidated for the proposed sleeve candidates; it did not prove that a dynamic execution policy increases challenge EV.

## Caveats

- All venue statements are **Bybit-specific** (V5 REST, UTA) and can change; they are not universal exchange behaviour. Other venues differ on partial-cancel status naming, reduce-only rules and market-order protection.
- Documentation describes intended behaviour, not observed behaviour on this account. Nothing here was verified against live responses, and no such verification was performed.
- No claim is made about fees, funding, liquidation, rate limits, capacity or historical depth availability; those are out of scope.
- No private data, credentials, account identifiers or raw broker responses are included.

## Sources

1. Bybit — Enums Definitions (orderStatus, positionIdx): <https://bybit-exchange.github.io/docs/v5/enum>
2. Bybit — Place Order (market→IOC limit, slippage tolerance, reduceOnly split, positionIdx): <https://bybit-exchange.github.io/docs/v5/order/create-order>
3. Federal Reserve/OCC/FDIC — Supervisory Guidance on Model Risk Management (Apr 17, 2026; SR-26-2), non-binding: <https://federalreserve.gov/frrs/guidance/supervisory-guidance-on-model-risk-management.htm>
4. FIX Trading Community — Order State Changes, section B.1.b: <https://www.fixtrading.org/wp-content/uploads/download-manager-files/FIX-Latest-as-of-EP284-Order-State-Changes.pdf>
