# Feasibility of adding independent probe training pairs

7 September 2026. **There is no genuinely new compatible training data in the inspected caches.** Every original outer fold already uses all available non-test pairs from the frozen arithmetic family. A CPU refit of these caches cannot increase the number of independent training pairs beyond 36. The earlier 18-versus-36 contrast remains a reduction of existing training data, not a new-data experiment.

This inspection read the completed follow-up report, actual probe manifests/results, extraction/fold code, and nearby cache metadata before assessing alternatives. It performed file/header inspection, streamed array hashing, and enumeration of pairs/classes only. No model states were extracted, no probe was fitted, and no Huginn operation—including coda-only readout—was run. Any such work remains deferred until the lead confirms that priorities 1–2 (independent Ouro refits, then target/position controls) are settled.

## Existing evidence and search result

The manifested cache is `/home/moloch/ouro_project/artifacts/jlens/probe/n80_v2/gpu_cache.npz`, with archive SHA-256 `4cb03e201dd27b21fd296a9a2b4d98f3254612d1c08c34deecfdd590b081aad6`. Its `H` array is float16 `[648,192,2048]`: last-prompt-token residuals at every physical layer in four Ouro passes. The matching five-fold experiment is in `artifacts/jlens/probe/cv_all648/{design.json,arrays.npz}`. The original target remains the 17 nominal sums 2–18; the common eligible evaluation set is 576 prompts from 39 unordered operand pairs.

Five `gpu_cache.npz` archives were found under `artifacts/jlens/probe/`:

| Directory | Hidden-state payload | Additional independent pairs |
| --- | --- | ---: |
| `n80_v2` | `[648,192,2048]`, float16 | 0 |
| `n80` | identical | 0 |
| `pilot` | identical | 0 |
| `round1_exit3x32` | identical | 0 |
| `history/recovered-2026-09-04` | identical | 0 |

The decompressed `H.npy` entry in **all five archives** has SHA-256 `164b812e9637f381669499f5f490931f4705f411fdc87eeea1fa05102ce0d294`. Older split definitions, lens ranks, compression, or provenance differ; their hidden states are exact duplicates. Merging them would duplicate examples rather than add observations.

The bounded search covered `/home/moloch/ouro_project/artifacts`, `/home/moloch/jacobian-lens/research`, and nearby `/home/moloch/elastic_reasoner`, excluding dependency/model/browser caches. The 974 discovered NumPy archives/arrays partitioned into 29 J-Lens/probe/evaluation files, 3 follow-up derivatives, 109 ARC/debug/trajectory arrays, and 833 Huginn/GSM8K arrays. Filename and code searches for arithmetic/operand/`all648`/probe-cache families found no larger compatible arithmetic feature family. Other nearby pooled candidate-quality features and Huginn features are different inputs or representations; they cannot be substituted for these Ouro last-token residuals. This is a bounded local search result, not an assertion about every possible external artifact. The prior probe auditor independently reported no larger cache; the historically referenced `p3/probe_curve` files were absent, and that account also concerned the same 648 prompts.

## Exact headroom under the frozen design

The generator is `({a} + {b}) * {c} = ` for `a,b ∈ {1,…,9}`, `c ∈ {2,…,9}`. There are 81 ordered pairs and 45 unordered pairs: 9 diagonal and 36 off-diagonal. Each diagonal pair contributes 8 prompts; each off-diagonal pair contributes 16, giving `9×8 + 36×16 = 648`.

The seed-0 five-fold assignment holds out 9 unordered pairs in every fold. All other 36 pairs enter its final classifier refit, including the 8 inner-validation pairs after hyperparameter selection. Hence **45 − 9 − 36 = 0** unused pairs per fold. More multipliers, operand reversals, prompt duplicates, resampling, or importing another fold's held-out pairs cannot supply additional independent pairs while preserving this contract. The 72 currently ineligible test rows also remain held out; moving them into training would change the frozen outer split and class support.

| Outer fold | Current training rows | Eligible test rows | Classes absent from its original training set |
| --- | ---: | ---: | --- |
| 0 | 520 | 104 | 4 |
| 1 | 512 | 136 | none |
| 2 | 528 | 80 | 3, 17, 18 |
| 3 | 512 | 136 | none |
| 4 | 520 | 120 | 2 |

These counts were independently enumerated and checked against the saved `folds`/`labels` arrays. The old layer choices are fixed at physical layers **39, 31, 29, 42, 26**, respectively, in loop 1. Original C values are `.1, 1, 1, 1, 1`. The previous tighter-tolerance result (.611 versus .594) and half/full difference (+.039, interval −.029 to +.109) are already completed results, not unused new-data opportunities.

## Smallest useful future extension with positive integer operands

If new inference later becomes inexpensive enough, preserve all original test pairs, prompt strings, eligible rows, nominal targets, readout positions, selected layers, and evaluation metric. Enlarge only the **training operand domain** to positive integers with `a+b ≤ 18`. This gives

`sum(floor(s/2), s=2,…,18) = 81`

unordered pairs in total. Exactly **36** lie outside the original family: `1 ≤ a < b`, `b ≥ 10`, `a+b ≤ 18`. Their counts by sum 11–18 are `1,2,3,4,5,6,7,8`. There are **no new positive-integer pairs for sums 2–10**. This finite upper bound is an important limitation: a balanced increase for every target class is impossible under positive integer addition with the fixed target set.

Use both orders and the original eight multipliers for these 36 new pairs: **576 new short prompts**, extracted once and reused across folds. To keep each fold's original class support, exclude new sum-17/18 pairs from fold 2; retain the original 576-row evaluation mask even if a future alternative changes support.

| Outer fold | Allowed new pairs | Total training pairs | Allowed new rows | Total training rows |
| --- | ---: | ---: | ---: | ---: |
| 0 | 36 | 72 | 576 | 1,096 |
| 1 | 36 | 72 | 576 | 1,088 |
| 2 | 21 | 57 | 336 | 864 |
| 3 | 36 | 72 | 576 | 1,088 |
| 4 | 36 | 72 | 576 | 1,096 |

This would test upper-sum training augmentation under a declared operand-domain shift; it would not be a clean uniform increase in same-domain data. Only 264/576 eligible test rows have sums 11–18. Before fitting, fix loss weights so the original per-fold class proportions and effective regularization strength do not silently change with the unequal augmentation, and keep the same solver settings in the baseline and augmented comparisons. Retain the frozen selections and use paired evaluation on the same test rows. Any effect remains conditional on those choices.

The economical extraction needs only the five already selected loop-1 locations: float16 `[576,5,2048]` is **11.25 MiB**, versus 432 MiB for all 192 locations. Preserve the model revision, BOS convention, precision, positions, and residual definition, and check a small overlap against the original cache before combining features. The new rows are training inputs with known arithmetic labels; their three-token correctness continuations and lens vocabulary rescoring are unnecessary. One prefill per new prompt suffices. This does not require new J-Lens fitting or a layer sweep.

Historical logs report all 648 old prompts cached in 297, 312, and 325 seconds, including correctness generation. Naively scaling that old path to 576 prompts gives about **4.4–4.8 minutes**; capturing only training features may be cheaper. These are historical timings, not a current hardware measurement or execution commitment. Profile a short fixed batch only after priorities 1–2 are settled, and stop if the actual cost is no longer small.

For comparison, adding `(0,s)` would provide one new pair per class with 272 prompts, but it makes the intermediate equal a visible operand and is a poor data-limitation control. Signed operands can supply genuinely new pairs for all classes: `(-k,s+k)` for `s=2,…,18`, `k=1,2`, with both orders and eight multipliers gives 34 new pairs/544 prompts (10.625 MiB at the five locations). That is a larger change of training distribution, not more data from the original positive single-digit family; it should be a separately declared study, not the default cheap follow-up.

**Recommendation for this round:** record the exact zero headroom in existing features and do no additional cache-only probe refits. If priorities 1–2 leave room for a small inference task, the 36-pair positive extension is concrete and inexpensive in historical terms, with its upper-class/domain limitation reported explicitly. Otherwise defer the probe rather than relabeling repeated old-data refits as a larger-data experiment.
