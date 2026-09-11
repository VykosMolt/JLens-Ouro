# Huginn–Ouro comparison contract

Fixed on 8 September 2026 before reading new Huginn outcomes. This document specifies analysis of the already frozen evaluations; it authorizes no new inference, fits, depth search, or deployment changes. The comparison uses Huginn R=8, one N100 fit, both predetermined evaluation seeds, and the preselected Ouro main fit01. Complete, validated outputs are required for both models. Partial seeds or fits cannot become the planned comparison.

The user's original prediction from the 7 September 2026 research request is retained verbatim:

> If J-Lens recovers content outside the native output basis, its advantage over raw logit lens should be larger in Huginn than in Ouro.

The [prospective research-log entry](../../followup_2026-09-07/RESEARCH_LOG.md#prospective-huginn-prediction) records the same directional prediction in different wording and fixes attention to the previously reported R4 raw-readout failure, all recurrent blocks, and the coda baseline. The direction is not changed if the result is null, negative, seed-dependent, or limited to one block/task. Here “advantage” means the **difference in paired recovery effects**, not a ratio with a potentially zero denominator. The previous ten-prompt feasibility suggestion is superseded by the [combined N100 contract](../deployment/combined_contract.json).

## Inputs and common population

Bind the consumed evaluation owners, completion records, source/model/runtime identities, fit identities, all array hashes, and the exact eligibility document before calculating scores. Use the semantic validators in [evaluate_refits.py](../deployment/evaluate_refits.py), [evaluate_controls.py](../deployment/evaluate_controls.py), and [evaluate_huginn.py](../deployment/evaluate_huginn.py), plus the declared combined contract. The completed Huginn population document must equal [huginn_eligibility.json](../deployment/huginn_eligibility.json), whose current byte SHA256 is `153c340591c4c1ff298383a24f2ac1135a157fba08dc9ecb5a82dedc298457af`. No frozen population or control policy is inferred from recovery outcomes.

Both lenses use the same ordered 100 calibration texts, fit01 text-file SHA256 `1242e241951d6a0c99b2520af59e9f748974a5dd37da483d77a6bc8e1cc8e712`, maximum 128 tokens, and the existing skip-first-16/exclude-final-position estimator. Tokenizers produce different token counts, positions, and contexts; matching texts and N does not equate those distributions. Huginn has one calibration initialization recipe, with base `2026090802+i` for paragraph i and the frozen token-dependent seed derivation. Evaluation bases `2026090803` and `2026090804` are two trajectories of that same fitted lens. They are neither independent calibration fits nor 296 independent evaluation items. Ouro's other four N100 fits assess Ouro fit variation elsewhere; they do not replace fit01 or supply missing Huginn fit variance.

Retain all 148 ordered stimulus identities: 93 multihop and 55 arithmetic. Recompute **both** models' scores with the frozen joint slot eligibility and common name controls. Eligible means a supported token form and no leakage under either tokenizer, with arithmetic operation slots excluded. The resulting denominators are 88 multihop items/98 slots and 51 arithmetic items/51 slots. Nine items remain stored with no eligible slot; they do not enter means. The common catalogue has 68 multihop and 20 arithmetic names; multihop `14` and `Vatican` are excluded from the original 70-name catalogue. Use each row's exact `control_indices`: common supported names of the same operation/non-operation kind, excluding **all** of that item's own names. Preserve historical aliases and item weighting.

Keep the original name-axis indices. Huginn's unsupported names and padding are `-1`, explicitly masked before scoring; a negative sentinel is never a rank or a miss that enlarges a denominator. Ouro's saved native catalogue includes the two excluded names, so its precomputed native-population summaries are not this comparison. Both sides use the frozen last token of the common prefix of `encode(prompt)` and `encode(prompt + target)`, their native BOS rules, and evaluation maximum length 512. This does not use a generated continuation. Join by ordered item identity and verify recorded token contexts; never align token IDs across vocabularies. No aggregate population is chosen using either model's correctness or recovery.

## Readout and target semantics

| Model/readout | Source cells | Readout meaning |
| --- | --- | --- |
| Ouro raw | Virtual 0–191: four passes × 48 blocks | Final normalization/head applied directly to the source vector. Native pass exits are 47, 95, 143, 191. |
| Ouro J-Lens | Learned 0–190, target 191 | Averaged source-to-final-residual Jacobian, then the same normalization/head. Displayed cell191 is a known self-identity reference, not a learned map. |
| Huginn raw | Core 0–31: eight passes × four blocks | Final normalization/head applied directly to the core vector. This omits the coda. |
| Huginn J-Lens | Learned core 0–31, target 33 | Transport to the last coda-block residual before final normalization/head. Core31 remains a strict source upstream of normalization and two coda blocks; it is not self-identity. |
| Huginn coda | The same 32 core states | Complete sequence through pre-coda normalization, both attention-bearing coda blocks, final normalization/head, then select the last position. |

Huginn coda sources 3,7,…,31 are native recurrence exits along the paired trajectory. Coda readouts at the other three physical blocks per pass interrupt the core and are diagnostic interventions. Native final logits are the coda readout at source31. Source32, the first coda block, is outside the frozen core-source bank; target33 is not an additional evaluated source. Do not append either as a favorable reference or invent an identity map for core31.

The retained estimator averages, over the 100 calibration texts, the source-position mean of the derivative of the **sum over valid target positions**. It is applied as `J_v @ h_v`; the exported FP16 maps are applied in FP32 before the native readout's dtype conversion. This is a position/context-averaged verbalization map, not an assertion that the transported vector reproduces this item's actual target residual or output prediction. Ouro and Huginn targets have different downstream computation even though both are their final pre-head residuals.

Huginn's coda is a trained, nonlinear decoder with extra attention and compute. Training samples recurrence depth and applies loss after the coda, so the shared coda learns to read completed core states at many depths. That differs from Ouro's weighted losses at every pass boundary, but does not justify describing Huginn as having no intermediate decoding supervision. The published raw-readout failure is block-specific and comes from a different arithmetic/probing setup. [Verified literature and primary-paper links](../../followup_2026-09-07/literature.md#huginns-training-differs-but-its-coda-is-trained-across-depths).

## Scores and fixed summaries

For method m, item i, eligible slot k and cell v, let `h(m,i,k,v)` be whether the minimum full-vocabulary rank over that model's frozen token forms is below ten. Let `c(m,i,k,v)` be the mean of the same event over the slot's exact matched control names. Average slots within each item to form `E(m,i,v) = mean_k[h-c]`, keeping own and control terms separately. The paired lens effect is `D(M,s,i,v) = E(JLens,i,v) - E(raw,i,v)`. Ouro has one selected fit/trajectory; s indexes Huginn's two fixed evaluation seeds. Calculate ranks, events and paired effects before any seed, cell or item averaging.

Report both tasks separately. Primary descriptive displays retain every Huginn core cell and every Ouro cell, with the Ouro identity endpoint marked. Show own, control and excess curves, and the paired J-Lens effect. Use pass × physical-block coordinates and show both Huginn seeds, plus their equal-weight mean. A compact distribution display may summarize the cellwise task means with an empirical CDF; cells are correlated locations, not independent statistical replicates. Also report Huginn's four physical-block means over all eight passes, so the prespecified R4 question cannot hide the other blocks.

The first fixed scalar is the item mean of the difference between Huginn's mean D over all32 core sources and Ouro's mean D over its191 learned sources. It measures the average paired readout advantage over each declared state bank. It removes an oracle maximum, but does not make the banks semantically equivalent. Keep Ouro's full192-cell mean, if shown for historical continuity, separate from this learned-source statistic.

The agreed **same-opportunity sensitivity** uses seven fixed fractions of executed recurrent blocks, `j/8` for j=1,…,7:

| Fraction | Huginn source | Ouro source |
| --- | ---: | ---: |
| 1/8 | 3 | 23 |
| 2/8 | 7 | 47 |
| 3/8 | 11 | 71 |
| 4/8 | 15 | 95 |
| 5/8 | 19 | 119 |
| 6/8 | 23 | 143 |
| 7/8 | 27 | 167 |

Report **mean over these seven cells** and **any-of-seven recovery** separately. For the latter, each own name and each control first takes its own event over the seven selected cells; then average controls, slots and items. Only after scoring each method form J-minus-raw and Huginn-minus-Ouro. Never maximize the paired differences, union the two Huginn seeds, select seven best cells, or maximize a mean control curve. The fraction1 endpoint, Huginn31 versus Ouro191, is shown separately as a native-exit/readout diagnostic and excluded from this learned-map sensitivity because the Ouro J map is identity.

Seven equal opportunities do not equate native recurrence, compute, or semantic depth. All selected Huginn cells end a pass; Ouro23,71,119,167 are within-pass states. The grid therefore directly samples the prespecified Huginn R4 concern and is not neutral across physical block types. Full curves and all-source averages remain primary. The original all-layer maxima, with32 versus192 opportunities (or191 learned Ouro opportunities), are descriptive within-model coverage only; they cannot establish a cross-model ranking.

For each task and both seeds, retain Huginn coda-minus-raw and J-Lens-minus-coda curves and the four physical-block means. These distinguish a raw-head alignment problem from recovery unavailable to the trained coda. Coda success does not demonstrate J-Lens failure, and J-Lens success does not establish causal use of a labelled concept. Keep native pass-end coda points visibly distinct from within-pass interventions.

## Uncertainty and interpretation

For the three cross-model scalars above—all-learned-cell mean, seven-cell mean, any-of-seven—compute the two seed-specific estimates and their equal-weight average. Use10,000 paired item bootstrap draws, root seed2026090808, independently within the two tasks. The same item draw must be shared by every model, method, location, metric and Huginn seed in that task. Keep the two seeds fixed; do not resample fits or treat seeds as a variance estimate. Report percentile pointwise95% intervals and an approximate unstudentized simultaneous interval using the95th percentile of the maximum absolute centered bootstrap error across the six task×metric seed-average contrasts. Any displayed cell intervals remain labelled pointwise; no selected peak is certified by them.

A shared-concept component sensitivity uses the same concept normalization/component rule as the [Ouro analysis contract](CONTRACT.md), rebuilt on the **joint** eligible slots. Resample components within each task, retain all member items, and use ratio means to preserve item weighting. Share component draws across models, seeds and metrics. Record the component memberships/counts. These intervals condition on one fitted map bank per model and two chosen Huginn initializations; they omit calibration, training, and general initial-state uncertainty. Curated prompts and shared templates also limit a sampling interpretation.

A positive cross-model effect is consistent with the recorded readout prediction on these tasks and this R=8 setting. Architecture, width, training data, native capability, tokenization, unequal recurrence, target choice and the learned coda remain competing explanations. Different raw-lens baselines can change an improvement margin through ceilings or floors; therefore own, control and absolute excess accompany every difference. Divergent task/block/seed outcomes are reported as such. The analysis cannot isolate supervision causally, rank the models generally, or claim all Huginn depths. Native next-token agreement and concept recovery are distinct measures; neither substitutes for the other.

## What the frozen arrays permit

| Consumed artifact/fields | Valid use and limitation |
| --- | --- |
| Ouro `common/arrays.npz`: `logitlens_allrank`, `logitlens_rank`, `logitlens_top1`, `exit_top1`; `fits/fit_01/arrays.npz`: `jlens_exit3_allrank`, `jlens_exit3_rank`, `jlens_exit3_top1` | Recompute common-population own/control recovery and paired effects at all192 columns. `allrank` is required for controls. Native exit IDs and top1 agreement are meaningful within Ouro's vocabulary. |
| Huginn `seeds/{seed}/arrays.npz`: `raw_allrank`, `jlens_allrank`, `coda_allrank` and their `*_rank`/`*_top1`, plus `native_top1` | Recompute all three methods' concept scores on the same trajectory. Ranks have shape148×128×32, own-slot ranks148×3×32, top1 arrays148×32. Compare raw/J/coda top1 with same-cell coda or final native predictions within Huginn; use native-exit labels only at3,7,…,31. |
| Huginn `population/eligibility.json`, each seed's `initializations.json` and metadata | Supply ordered identities, eligible slots, common names, exact controls/token forms, and the coupled seed provenance needed for every paired calculation. Existing Huginn summaries are checks, not a replacement for itemwise rank reduction. |
| Huginn `cache.pt`: `H`148×32×5280, `target_states`148×5280, `native_logits`148×65536 | Contains last-position vectors and final native logits. It does **not** retain the complete sequence core states required to rerun the coda, or full raw/J/coda logits at every source. Existing coda rank/top1 arrays already support the comparisons above without inference. |

Do not claim raw/J/coda full-distribution KL, top-k overlap, or ranks of arbitrary native predictions from the Huginn rank bank: those logits/ranks are not saved. Top1 equality is available; rank fields concern the fixed task names. Ouro's existing KL/native-rank diagnostics remain Ouro-only context. Do not compare token IDs or distributions across the two vocabularies. No Huginn generated continuation/correctness record is present in this evaluation; `native_top1` is a next-token prediction, not whole-answer correctness, and Ouro correctness must not be copied onto Huginn.

An eventual analysis export should retain the exact itemwise own/control/excess arrays, paired differences, source selections, eligibility masks, bootstrap memberships, both seed estimates, all fixed summaries, and input hashes.

### Pre-outcome example amendment

As requested before any Huginn outcome, also export `examples.json`. Select ten of the139 jointly eligible items uniformly without replacement using seed2026090809, independently of effect values. Retain every selected prompt, target, intermediate label, eligibility/exclusion record, and all three fixed own/control/excess summaries for Ouro fit01 raw/J-Lens and both seeds of Huginn raw/J-Lens/coda. Separately sample up to five readout-failure items per task, seed2026090810 with a task-specific stream. Define a readout failure in advance as **seed-average Huginn J-Lens own any-of-seven recovery equal to zero**: no eligible own label was recovered in that grid in either seed. Record the complete failure pool and every selected case. This is a conditional descriptive sample of concept-readout misses, not a whole-answer correctness judgement. Neither sample changes a source selection, denominator, aggregate or interval. Keep tasks with no such failure as an explicitly empty pool.

This document is the design artifact; no scientific outcomes were read or computed while writing it.
