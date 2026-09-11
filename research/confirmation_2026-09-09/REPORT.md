# Artifact recovery and prospective Ouro confirmation

**The prospective confirmation supports the preselected loop-4 advantage on new items.** On the frozen 160-question population, Ouro fit01 exceeded the raw logit lens by **+0.232** excess hit@10 across loop 4, physical layers 26–37. The 95% dependency-group bootstrap interval is **[+0.163, +0.329]**. The advantage comes from higher intended-concept recovery, not fewer control hits. All six early-loop secondary contrasts are negative, with simultaneous intervals excluding zero. An independent reconstruction reproduced all 231 compared quantities. The run followed attempt05's failed development check, a bounded diagnostic and a separately frozen correction of that check; every correction was frozen before any new-item outcome. The original freeze and [prior report](../refit_round_2026-09-07/REPORT.md) are unchanged.

1. **What was recovered?** Seven of the sixteen historical estimator binaries survived intact. Ouro fit02's bank was reconstructed from its intact checkpoint and matches the historical whole-file SHA256. Eight binaries remain missing or incomplete: both files for Ouro fits03–05 and both Huginn files. A later storage purge kept only the four confirmation banks; the checkpoints and both partial originals were deleted.
2. **Did the fixed loop-4 advantage generalize?** On this curated population, yes: +0.232 [+0.163, +0.329], against the discovery endpoint of +0.189. The interval is conditional on this fit and population design.
3. **Was any advantage increased intended recovery or reduced control recovery?** Increased intended recovery. fit01 recovered intended concepts at 0.379 against raw's 0.144, a difference of +0.235 [+0.166, +0.332]. Control hits rose slightly (+0.003 [+0.001, +0.004]), so the excess is not driven by fewer control hits.
4. **Did the early-loop deficit generalize?** Yes. All six early-loop contrasts are negative, from −0.088 to −0.611, and their simultaneous 95% intervals exclude zero.
5. **What is supported or unresolved?** Supported: transfer of fit01's fixed-band advantage and of the early-loop deficit to this curated mixture. Not established: absence from pretraining, the model's use of the annotated intermediates, difficulty matched to discovery, or generality beyond this population and fit. The available estimator conversions and fit02 reconstruction are verified. Historical Huginn remains unsupported, and its verification rerun was not run.

## Confirmation result

Frozen [analysis](results/ouro_confirmation_20260911_fixed160_native1/analysis/analysis.json) of the accepted final payload:

| Quantity | Estimate | 95% dependency-group bootstrap interval |
|---|---:|---|
| Primary: fit01 minus raw excess hit@10 | +0.23188 | [+0.16289, +0.32898] |
| fit01 intended recovery | 0.37917 | [0.30444, 0.46563] |
| fit01 control recovery | 0.00344 | [0.00182, 0.00507] |
| Raw intended recovery | 0.14427 | [0.10455, 0.17870] |
| Raw control recovery | 0.00043 | [0.00018, 0.00075] |
| Intended difference | +0.23490 | [+0.16615, +0.33160] |
| Control difference | +0.00301 | [+0.00146, +0.00442] |

All 160 questions were eligible, across 28 dependency groups. Item resampling gives [+0.17971, +0.28706], and leaving out any one group moves the estimate between +0.19730 and +0.24875. Component intervals are descriptive; the primary excess difference is the single inferential endpoint.

The twenty secondary contrasts share simultaneous 95% intervals (bootstrap max-t quantile 3.2255):

| Contrast | Estimate | Simultaneous 95% interval |
|---|---:|---|
| fit01 minus raw, loop 1 fixed-layer mean | −0.08827 | [−0.13642, −0.04013] |
| fit01 minus raw, loop 1 any layer | −0.44407 | [−0.59029, −0.29785] |
| fit01 minus raw, loop 2 fixed-layer mean | −0.14858 | [−0.19541, −0.10175] |
| fit01 minus raw, loop 2 any layer | −0.57366 | [−0.72008, −0.42723] |
| fit01 minus raw, loop 3 fixed-layer mean | −0.15587 | [−0.22404, −0.08770] |
| fit01 minus raw, loop 3 any layer | −0.61084 | [−0.79591, −0.42577] |
| fit01 minus raw, main final third | +0.11543 | [+0.04673, +0.18413] |
| fit01 minus raw, physical layer 32 | +0.24684 | [+0.10717, +0.38650] |
| fit02 minus raw, primary band | +0.22558 | [+0.09523, +0.35593] |
| fit01 minus fit02, primary band | +0.00630 | [−0.00727, +0.01988] |
| Penultimate minus raw, primary band | +0.08804 | [+0.00759, +0.16849] |
| Penultimate minus raw, matched final third | −0.04872 | [−0.08742, −0.01002] |
| Sampled-sum minus raw, primary band | +0.21724 | [+0.09155, +0.34293] |
| Sampled-sum minus raw, matched final third | +0.11439 | [+0.03956, +0.18923] |
| Diagonal minus raw, primary band | +0.05952 | [−0.00340, +0.12244] |
| Diagonal minus raw, matched final third | +0.00171 | [−0.04664, +0.05005] |
| fit01 minus penultimate, primary band | +0.14384 | [+0.07699, +0.21069] |
| fit01 minus penultimate, matched final third | +0.16917 | [+0.09716, +0.24118] |
| Sampled-sum minus diagonal, primary band | +0.15772 | [+0.05796, +0.25748] |
| Sampled-sum minus diagonal, matched final third | +0.11268 | [+0.04515, +0.18022] |

Figures: [primary and components](results/ouro_confirmation_20260911_fixed160_native1/figures/primary_and_components.png), [descriptive layer curves](results/ouro_confirmation_20260911_fixed160_native1/figures/descriptive_layer_curves.png) and [early-loop secondaries](results/ouro_confirmation_20260911_fixed160_native1/figures/early_loop_secondaries.png).

An [independent reconstruction](results/ouro_confirmation_20260911_fixed160_native1/independent_numerical_check_native1.json), which imports no producer code, matched all 231 compared quantities within 1e-12; the largest difference was 8.9×10⁻¹⁶. The frozen checker pins the original run spec, so on this run it stopped at that input check ([record](results/ouro_confirmation_20260911_fixed160_native1/independent_numerical_check.json)). The versioned [copy](reviews/primary_numerical_check_native1.py) adds a pin for the corrected run spec and binds the results to it. This two-line change was made only after confirming that every run-spec field the checker uses is identical to the original freeze.

These results describe transfer to a curated mixture with substantial relation shift: 96 of the 160 questions request relation types absent from discovery. The population has 28 unequal dependency groups (effective count 8.6), and the estimate comes from one fixed calibration fit; fit01 minus fit02 is +0.006 [−0.007, +0.020].

## Corrections and the confirmation run

Two separately frozen corrections precede every new-item outcome and change no scientific input, estimator, scoring or analysis. [precision_v1](corrections/precision_v1/IMPLEMENTATION_REVIEW.md) removed two redundant precision setters that attempt03's strict check rejected. [native_check_v1](corrections/native_check_v1/FREEZE.json) changed only the development check: it now unembeds the whole final-step sequence, which the diagnostic found bit-exact, instead of one residual vector. Its exact-equality criterion is unchanged. A CPU functional test with the real model confirmed the corrected check before freezing; the change itself was not independently reviewed. Run budgets were resized from measured timings: setup 12,800 s, compute 1,200 s and preservation 900 s.

The run used pod `gj9dwkkqtb3e14`, a community RTX 5090, from 23:02Z on 10 September. The four banks uploaded over SSH in 1 h 53 min without interruption. On the GPU, the corrected development check passed all three exact equalities; the controller accepted that stage, and root authorized new-item scoring at 01:00Z. Caching the 160 questions and scoring all six arms took under four minutes. By 01:06Z the controller had retrieved, verified and accepted every stage and the complete final manifest, then deleted the pod; termination was verified at 01:06:59Z.

## Recovery and verification

The [artifact ledger](artifacts/RECOVERY_LEDGER.json) distinguishes complete originals, reconstruction and unresolved losses. All retained complete checkpoints and banks passed hashes, loadability, exact schema, dimensions, dtype and finite-value checks. Independent NumPy conversion checked all 4,001,366,016 entries across the five available estimator arms, including signed-zero representations, with zero bit differences. The fit02 reconstruction contains 801,112,064 FP16 entries and has historical SHA256 `101f31db6aa56d97fbae7ecb3e2f241ef1805000484d0e22f5eba5fc0b9acde8`. This checks the recorded checkpoint-to-bank conversion; it does not establish the scientific validity of every readout interpretation. [Recovery findings](artifacts/FINDINGS.md).

Upstream calibration text and fit identities, downstream bank bindings, four completed evaluation trees, and 23 numerical archives/caches were checked against their actual local files. The Huginn checkpoint retained 1,745,780,736 of 3,568,444,419 expected bytes before the purge deleted it. Searches of the documented local locations found no sufficient checkpoint or paragraph contributions to reconstruct it. Saved activations and scores do not replace the missing derivatives. [Input and evaluation audit](artifacts/RETAINED_INPUTS_FINDINGS.md).

These checks predate the storage purge. Its manifests record the size and SHA-256 of every deleted file, in [this round](PURGE_MANIFEST_2026-09-11.json), the [refit round](../refit_round_2026-09-07/PURGE_MANIFEST_2026-09-11.json) and `/home/moloch/ouro_project/artifacts/jlens/PURGE_MANIFEST_2026-09-11.json`.

## Frozen new-item test

The [prospective plan](PROSPECTIVE_PLAN.md), [benchmark](benchmark/final_candidates.json), eligibility/control population, analysis and executable sources are fixed by the [freeze receipt](FREEZE.json). The run used the separately frozen [precision](corrections/precision_v1/FREEZE.json) and [native-check](corrections/native_check_v1/FREEZE.json) corrections, which change no scientific input. The primary estimator is the original verified Ouro fit01, with the original target, position reduction, normalization, aliases, readout token and full-sort rank semantics. Fit02 and the surviving target/position controls are fixed secondary comparisons; there is no refitting, selection by new score, ensemble or sample extension.

The test contains 160 eligible questions, 80 intermediate identities absent from the discovery benchmark, and 79 equally weighted other-name controls per intended label. An independent reviewer checked both factual hops, novelty and control hygiene before outcomes. Three component facts overlap known calibration text; none is a reused complete two-hop question or discovery intermediate identity. No absence from base-model pretraining is claimed.

The strongest limitation is the population design. Its 28 dependency groups have sizes 42, 24, 16, 10, 10, 6, four groups of 4 and eighteen groups of 2, giving a weighted effective group count of 8.63. Conservative planning scenarios did not achieve the initial approximately ±0.10 precision target. Further, 96 questions request relation types absent from discovery, 45 use familiar requested relations, and 19 recombine familiar relations. This is transfer to a curated mixture with substantial relation shift; matched empirical difficulty is not established.

The single primary 95% interval uses 20,000 paired resamples of whole dependency groups, retaining item weights. Item resampling, group heterogeneity and leave-one-group-out estimates are sensitivity checks. The twenty secondary contrasts share simultaneous 95% intervals; complete layer curves remain descriptive. Neither a favorable secondary nor a relocated peak can replace the primary endpoint. The old five-fit +0.1872 estimate describes calibration stability on the discovery population. The corresponding original fit01 discovery endpoint is +0.18907; neither is pooled with confirmation.

## Native development gate failure

On three historical development items, the worker saves residual states from physical-block hooks, virtual taps and the wrapper forward, plus native final logits and `unembed` of the final residual. It requires `torch.equal` for all three comparisons ([worker](corrections/precision_v1/bundle/evaluation/worker.py), lines 94–97). Attempt05's failed final manifest lists only the development-stage files. They hash-match locally, although the controller handoff never completed. [Independent CPU-only forensics](reviews/attempt05_native_failure_forensics.md) found:

| Comparison | Shape | Differing elements | Maximum absolute difference |
|---|---|---:|---:|
| Virtual vs physical-hook states | [3, 192, 2048] | 0 of 1,179,648 | 0 |
| Virtual vs wrapper states | [3, 192, 2048] | 0 of 1,179,648 | 0 |
| Native vs unembedded logits | [3, 49152] | 112 of 147,456 | 0.03125 |

The mean absolute logit difference is 6.05×10⁻⁶. Values are finite and BF16-representable, no sign differs, and 106 of the 112 differences are one BF16 code step. Top-100 token order is identical for every item. Deeper orderings differ, the earliest at zero-based rank 270 (mars-color), with 1,322–1,391 token ranks changed per item under a stable sort ([documentation review](reviews/attempt05_closure_documentation_review.json)). The precision record matches the frozen original, so the attempt03 defect did not recur.

The forensics proposed shape-dependent BF16 rounding in the final norm or lm_head but could not separate the two; the diagnostic below does.

## Native-logit diagnostic

At the user's instruction, a bounded diagnostic reran the development items on the same GPU type and frozen runtime ([contract](diagnostics/native_logit_v1/CONTRACT.md)). Its analysis rules were fixed before any outcome ([rules](diagnostics/native_logit_v1/run/PRE_RUN_RULES.md)). Independent verification and adversarial review passed with non-blocking findings before the run, and an [independent CPU reanalysis](reviews/diagnostic_native_logit_v1_reanalysis.md) reproduced all 255 recorded comparisons afterwards.

The native forward normalizes each recurrent step's state, gathers each token's exit-step state and applies lm_head to the whole sequence; for every item, the last token's state came from the final step (zero-based 3). The diagnostic recomputed the captured states at several shapes:

| Recomputation | Differing elements against the native tensor |
|---|---|
| Final norm at [2048], [1,2048], [1,1,2048], [S,2048], [1,S,2048], [148,2048], [160,2048] | 0 / 0 / 0 |
| lm_head on one row: [2048], [1,2048], [1,1,2048] | 39 / 28 / 45 |
| lm_head on several rows: [S,2048], [1,S,2048], [148,2048], [160,2048] | 0 / 0 / 0 |
| Norm then lm_head at [1,S,2048] or [160,2048] | 0 / 0 / 0 |
| Attempt05's check: unembed of the single final residual | 39 / 28 / 45 |

Counts are for carnival-ocean / amazon-language / mars-color, and every recomputation agreed with its repeat. At [148,2048] and [160,2048] the target row was placed first or last among the item's own positions, cycled. The attempt05 mismatch recurred exactly. This pod's development tensors are byte-identical to attempt05's, although the lease and provenance records place them on a different machine, and repeated native forwards were bit-exact. The single-row lm_head output differs at exactly attempt05's positions, while the final norm is exact at every tested shape. Exact equality with native logits is therefore achievable by computing norm then lm_head at [1,S,2048] or [160,2048].

These conclusions cover this GPU type, runtime, the three items and the measured shapes. The underlying kernel difference is not identified. Other fill content, the [148,2048] composite and the scorer's [190–192,2048] readout shapes were not measured.

## Preservation, resources and completion

The repaired controller distinguishes persistent setup, completed computation, retrieval and external acceptance. Normal deletion requires the receiver's exact run/manifest receipt after completeness, hash, loadability and numerical checks. Failure tests include restart, stale/missing status, connection failure, corrupt transfers, old acknowledgements and spending stops. A local save/retrieve/verify smoke test passed, and the [actual retry deployment](controller/DEPLOYED_PRIMARY_RETRY_02.json) records all six monitor hashes. All seven workers, the six attempts and the diagnostic, have verified provider shutdown. Attempt03's [deployed controller](controller/DEPLOYED_PRIMARY_ATTEMPT_03.json) and [receiver runtime correction](controller/DEPLOYED_RECEIVER_ATTEMPT_03.json) are recorded. Attempt06's final receipt was accepted before its pod was deleted; earlier cancellations and failure teardowns are not represented as scientific preservation.

The first two primary attempts ended during input transfer. Attempt03 failed its precision preflight after setup; its failed manifest and records are [preserved](resources/ATTEMPT03_PREFLIGHT_FAILURE_PRESERVED.json), and its [diagnosis](resources/PRECISION_PREFLIGHT_DIAGNOSIS.md) led to the reviewed correction. Attempt04 expired during its approval wait and never received scientific inputs.

Attempt05 (pod `i7zq14rj0f0ief`) was created at 14:55:06Z on 10 September under the unchanged $3.77 attempt and $25 combined caps. The user confirmed authorization for this replacement and its export after the attempt ([record](resources/ATTEMPT05_AUTHORIZATION.json)); no contemporaneous record exists. Its upload was interrupted three times ([run log](RUN_LOG.md)). After the user's recorded instruction "Extend the deadline", root [extended only the setup deadline](reviews/attempt05_deadline_amendment_review.md) by 15 minutes; independent review then passed, and the launcher started after the original deadline. Nineteen remote files were [preserved](resources/ATTEMPT05_NATIVE_FAILURE_PRESERVED.json) before teardown was verified at 18:32:19Z.

The diagnostic (pod `mq25wyg0nf9z36`) ran from 21:50Z to 21:57Z on 10 September under a $1.50 cap, with the user's recorded approval. It uploaded only the frozen diagnostic bundle and its worker configuration, and the controller deleted the pod automatically after accepting its results.

Attempt05's conservative charge bound is $2.558589, the diagnostic's $0.080620 and the confirmation run's $1.469640. The combined bound is **$23.195828 against $25**, leaving $1.804172. These are bounds, not invoices. A [final read-only observation](corrections/native_check_v1/run/observation_final.json) after the run lists no pods and zero hourly spending.

The optional full Huginn verification rerun remains resource-blocked: its frozen $6 admission exceeds the remaining $1.804172 under the unchanged $25 cap. The storage purge also deleted its frozen archive and frozen prerequisite cache, so the package can no longer run as frozen; its [frozen plan](huginn_verification/VERIFICATION_PLAN.md) and [freeze receipt](huginn_verification/FREEZE.json) remain as records. Any new estimator will be labeled a rerun, even if its bank hash matches history.
