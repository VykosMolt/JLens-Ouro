# Methods and claim boundary

## Fixed substrate

- Model: `ByteDance/Ouro-2.6B`, revision `1ed04250da1a9936042725d302e81c8fa2ab5abd`.
- Jacobian-lens code: revision `581d398613e5602a5af361e1c34d3a92ea82ba8e`.
- Ouro has 48 shared physical decoder layers applied over four recurrent steps. Human loop 1 is `ut=0`; virtual location is `ut * 48 + physical_layer`.
- The retained primary lens is the final recurrent-exit lens from the historical fit directory. Its producer
  provenance is absent, so the canonical report does not authenticate the fit size or promote an `n=80` claim.

## Lineage gates

The canonical report derives claim status from complete, byte-checked producer
records. A directory name, output-file existence, or a standalone numerical
boolean cannot supply fit size or current-generation status. A current
evaluation must bind its model, lens inputs, evaluation inputs, outputs, and
source generators; the retained evaluation directories currently have no
`provenance.json`, yielding
`LOCAL_OBSERVATIONAL_ARITHMETIC_REPRODUCIBLE_LINEAGE_UNFROZEN`.

The validation gate requires all M1–M5 milestones, the complete comparison set,
`NUMERICAL_AND_PROVENANCE_PASS`, and `HASH_BOUND` model/JLens provenance. The
current retained validator record satisfies that gate and the report records
instrumentation as `SUPPORTED_CURRENT_VALIDATION`. The
standalone probe is accepted only after its hidden-cache, lens-score, design,
runtime, model, and complete project/installed-JLens source chain verifies.
That chain is now current, but the lens input is still classified
`RETAINED_PRE_CUSTODY_EXACT_BYTES_ONLY`, so the probe remains local and
unfrozen. The retained checkpoint JSON is an older flattened
schema and therefore remains unverified.

## Main readout

The retained evaluation contains 93 multihop and 55 order-operations stimuli. After excluding leaked or unscorable task-defined intermediates, the primary populations are:

| population | unique items | clean intermediate slots |
|---|---:|---:|
| multihop | 90 | 100 |
| order-operations, numeric intermediates | 51 | 51 |

For each intermediate, hit@10 is evaluated at every physical layer. The primary statistic is the item-level difference between:

1. own-intermediate hit at any layer within a loop; and
2. the mean of the identical any-layer statistic over same-kind control names.

Multiple intermediate slots are averaged within item before population aggregation or bootstrapping. Paired intervals resample unique items. The eight pointwise intervals are reported as unadjusted; family-wide significance is not claimed.

Model correctness is recomputed from retained continuations with a boundary-aware prefix match. `11` is not correct for target `1`, and `daytime` is not correct for `day`.

## Fit-size analysis

Fits at 8, 32, 56, and 80 prompts are nested prefixes. Thresholded any-layer excess hit@10 is shown alongside the change in the best-within-loop rank. These fits are correlated and do not estimate fixed-size run-to-run variance or behavior at 1000 prompts.

## Transport analysis

For every source location, `transport_report.py` fits

`||J_bar_n||² / d = mu² + sigma² / n`.

`sigma` is modeled RMS prompt-to-prompt scatter. It is not a directly measured mean single-prompt Jacobian norm. Negative fitted moments are reported as invalid and omitted from valid-only aggregates; the historical zero-clipped value is retained only under an explicitly legacy field.

## Arithmetic probe

The family `(a + b) * c = ` contains 648 prompts over 45 unordered `(a,b)` clusters. Five folds keep mirror pairs together. Seventy-two held-out prompts have a label absent from their training fold, so the fair comparison uses 576 fold-trainable prompts from 39 clusters.

For each held-out fold and each method, a physical layer is selected using the other folds and scored only on the held-out fold. The cluster bootstrap repeats layer selection inside every draw. Both uniform 17-way chance and the empirical majority baseline are reported.

The hidden-state cache, lens-score arrays, and CPU design are freshly bound to
the current model/runtime/source bytes. Probe fitting is parallelized over
virtual locations only. Each location forces one native BLAS/OpenMP thread and
retains the same scaler, C grid, fold masks, refit, and rank calculation; this
prevents ambient thread settings from changing discrete predictions. The
remaining lineage limitation is the exact-byte-only pre-custody lens input.

## Explicit limits

- The stimuli score task-defined intermediate token labels; they do not prove causal use of those intermediates.
- No matched 2x2 position-by-reduction estimator experiment has been run.
- No 1000-prompt fit or fixed-n replicate set exists.
- No matched recurrent model without per-step shared-head supervision has been evaluated.
- The old local-versus-eventual analysis mixes 24-prompt local-exit lenses with a 32-prompt eventual-exit lens and is reported only as a refutation of the proposed monotonic convergence pattern.
- Current local artifacts were created before repository custody was repaired; their scientific status remains local and unfrozen even after byte manifests are added.
