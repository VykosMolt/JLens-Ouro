# Independent precision and population-design review

**Recommendation: retain the 160-item population and audited dependency graph, and amend the prospective plan before freeze.** A mandatory benchmark regeneration is not justified by the user's instructions. The current design can test transfer to a genuinely new fixed population, but it does **not** meet the initial precision target or provide a distribution-matched replication.

No new confirmation outcomes were inspected. The JSON companion binds the five reviewed source files and contains the calculations and acceptance conditions.

## Requirement and decision

The user asks for a useful precision basis, plausible dependence, comparable multihop difficulty, explicit distinction between familiar and new relations, and a frozen sample and stopping rule. They prescribe no hard attained half-width and explicitly accept negative or inconclusive replication. The prospective plan itself requires precision to be recomputed when audited dependencies reduce the number of groups.

The audit changes the precision assessment materially: 28 components and effective groups **8.6253**, instead of 80 independent pairs. At group-mean SDs 0.30/0.45/0.60, the planning half-widths are **0.200/0.300/0.400**. The initial approximately 0.10 target at SD 0.45 is unmet. Retaining the design is a transparent decision to tolerate that limitation, not evidence that the target was attained. No assumed discovery effect is needed for this assessment.

The largest component contains 26.25% of all items; the three largest contain 51.25%. A fixed-population equal-item estimate remains meaningful, but generalization and the interval's stability rely heavily on a few families. The planned group means, leave-one-group-out summaries, and intended/control decomposition must accompany the primary interval. This review does not certify finite-sample bootstrap coverage.

## Comparable task versus stronger distribution shift

All questions use a factual clue-to-entity-to-property construction, preserving the broad multihop task form. Equivalent empirical difficulty has not been established. Factual review can reject ambiguity or broken two-hop questions; the model's correctness or either readout's recovery cannot select items.

The 45 familiar requested-relation questions, 19 recombined familiar fact-relation questions and 96 questions with requested predicates absent from discovery form a deliberately mixed population. Most questions therefore involve more than replacing an entity within a familiar requested relation. The overall primary endpoint answers transfer to that mixture; it cannot be presented as the result for new entities alone, and the new-relation items are not interchangeable replication observations.

Restricting the primary to the 45 familiar items would alter the scientific population and leave only 10 original components with effective groups about **3.30**. It is not a valid automatic fix. The prespecified primary and secondary family should remain intact; no favorable subgroup may replace the overall result.

## Why no mandatory regeneration

A 0.10 planning half-width at SD 0.45 requires approximately **77.8 effective independent groups**. Additional paraphrases, repeated templates, or more questions about existing concepts do not provide those groups under the preserved audit rule. Replacing enough items to approach that target would be a substantial new benchmark-construction program; obtaining distinct constructions by inventing many more relation types could further weaken comparability. The current graph cannot be repaired by cosmetic changes, dropping inconvenient dependency edges, or changing the requested equal-item weighting.

The user did not require a decisive or publication-ready result. An honestly limited experiment remains useful evidence about this prespecified population. This recommendation does not assert that more data are computationally unaffordable or that N=160 is resource-optimal; the paid-job resource admission remains a separate check. If a hard ±0.10 precision requirement were imposed, this design would fail it and need substantive redesign before evaluation.

## Concrete amendment before freeze

The blinded dependency audit resolves the 160 candidate questions into 28 connected components, with sizes 42,24,16,10,10,6, four groups of 4 and eighteen groups of 2. The weighted effective group count N²/Σn_g² is 8.6253. For assumed SDs of paired group means 0.30, 0.45 and 0.60, the planning 95% half-widths are approximately 0.200, 0.300 and 0.400. The initial approximately 0.10 precision target at SD 0.45 is not met. These are dependence/variance scenarios, not an assumed effect, a realized confidence interval, or a power guarantee. We retain the fixed 160-question design before any new outcomes because it still estimates transfer to a genuinely new, explicitly defined population; we accept that it may leave the sign and magnitude unresolved. We will not describe this design as achieving the initial precision target, alter dependency edges, change item weights, or add questions after seeing results. Eligibility exclusions require recomputing the graph and planning quantities before final freeze.

The curated relation mixture consists of 45 questions with a familiar requested-relation family, 19 recombining familiar fact relations in a different hop or requested slot, and 96 whose typed requested relation was not represented in the 93 discovery prompts. It therefore tests new-concept/new-question transfer across this fixed mixture, including a substantial relation shift. It is not an interchangeable replication confined to familiar relations. The common factual clue-to-entity-to-property task form provides structural comparability, while equivalent empirical difficulty is not established and will not be enforced by filtering on model outcomes. The primary endpoint remains the prespecified equal-item excess difference; neither the familiar-only subset nor a favorable relation subgroup may replace it. The group bootstrap and group influence summaries are conditional analyses of this curated population and fit, with limited information from large unequal groups; they are not a guarantee of coverage over all possible multihop tasks or relation mixtures.

## Remaining gates

Complete the independent factual, novelty/calibration-overlap, alias/control and token-eligibility checks. Recompute the graph and planning precision after any affected revision or exclusion, then freeze all source bytes and the exact eligible population. Preserve the original fixed-layer equal-item endpoint, paired dependency-group bootstrap, and outcome-blind stopping rule. No additional user approval is required for this documentation amendment within the existing authorization.
