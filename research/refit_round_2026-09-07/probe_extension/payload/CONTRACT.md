# P4: fixed-test arithmetic probe augmentation

Status: prospective preparation only, 8 September 2026. This document does not start a scientific run or allocate a cloud resource. Execute only after P1 and P2 have been completed and interpreted, and only if the remaining time and the existing combined $25 budget permit the complete experiment. There are no P4 outcomes at contract formation.

The primary question is whether additional, previously unused operand pairs improve the frozen loop-1 probe on its original held-out pairs. Both primary arms use a newly extracted, common feature cache. The historical cache supplies a separate numerical/CPU-path diagnostic, not training or test features for either primary arm. A fresh raw logit-lens reference is included. Historical J-Lens results remain context only.

## 1. Fixed inputs and identities

All paths below are literal local input identities. A staged copy must retain the listed bytes and be explicitly mapped to its canonical input; a similarly named archive is not a substitute. Read NumPy archives with `allow_pickle=False`. Validate identities before consuming inputs and recheck consumed bytes before sealing a result.

| Input and canonical absolute path | SHA-256 | Fields used |
| --- | --- | --- |
| Old features: `/home/moloch/ouro_project/artifacts/jlens/probe/n80_v2/gpu_cache.npz` | `4cb03e201dd27b21fd296a9a2b4d98f3254612d1c08c34deecfdd590b081aad6` | `H`: FP16 `[648,192,2048]`, historical diagnostics only |
| Old feature provenance: `/home/moloch/ouro_project/artifacts/jlens/probe/n80_v2/gpu_cache.provenance.json` | `9b7a4733bdf9094440e05a62aa8d8d6fc9c1325a3de7a6426b1836c14467c16e` | Complete `model_files`, `runtime_versions`, `source_files`, `output` |
| Original CV arrays: `/home/moloch/ouro_project/artifacts/jlens/probe/cv_all648/arrays.npz` | `1ff6157d3bddda1b3fcf782f2ceb98749bc0c5f5d4c1200d172b4b8d04d0c274` | `folds`, `labels`, `chosen_C`, `selection_accuracy`, `probe_rank`, `ll_cand`, `jl_cand` |
| Original CV design: `/home/moloch/ouro_project/artifacts/jlens/probe/cv_all648/design.json` | `28a80d90145f98cdeeb2c80e800e85526e919d696471e1334f73ae73eb6d26ac` | Seed, unordered-pair split, validation ancestry, cache/input identities |
| Saved resamples and original selected scores: `/home/moloch/jacobian-lens/research/followup_2026-09-07/probe/audit_pairing.npz` | `e34cdde8a695a30f6a516aaa73bb6f4c8a2ed004fbd80f8605eff1a4c3a593e7` | `weights`, `clusters`, `counts`, `pid`, `eligible`; historical `probe`, `logit_lens`, `j_lens` |
| Original selection audit: `/home/moloch/jacobian-lens/research/followup_2026-09-07/probe/saved_rank_audit.json` | `1ccaba197a91706b84c5425c4eb9574ef000634ac0519b1a9589f5df137630f1` | Population, frozen per-fold locations, selected C, historical results |
| Historical CPU predictions: `/home/moloch/jacobian-lens/research/followup_2026-09-07/probe/bounded_refit_predictions.npz` | `f5a50cbcae0328ef0cda9892a366a5f3d0593409fac53599b84550d58ae495d0` | `original`, `y`, `eligible`, `folds`, `pid`; other sensitivity arms are not P4 choices |
| Historical CPU audit: `/home/moloch/jacobian-lens/research/followup_2026-09-07/probe/bounded_refit_results.json` | `8aac73e657009e6847b003156896c2f9b90b5bbed175b4b260fd7e493a10fe81` | Existing reproduction and fitting diagnostics, not a new selection source |

The original cache has already exhausted the 45 unordered positive single-digit operand pairs. The other audited caches duplicate these features; no cache merge adds new pair units. This extension therefore requires new inference.

The model is `ByteDance/Ouro-2.6B`, revision `1ed04250da1a9936042725d302e81c8fa2ab5abd`. Its canonical snapshot is `/home/moloch/ouro_project/artifacts/hf_cache/hub/models--ByteDance--Ouro-2.6B/snapshots/1ed04250da1a9936042725d302e81c8fa2ab5abd`. Bind every loaded snapshot file against the complete 13-file `model_files` inventory in the hash-bound old feature provenance, not just these identifying examples:

- `model.safetensors`: `2fdf9805a3510b7dc82959a2db4872ef767a5bdc55520c6a1b54fc3e463c1bba`.
- `modeling_ouro.py`: `c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413`.
- `configuration_ouro.py`: `950443e32929047aa08d02abad2e1888bc1914b3db988d3d675f70787f65dafb`.
- `tokenizer.json`: `fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d`.

Record the actual loaded remote-code paths and hashes, tokenizer inputs, GPU identity, driver/runtime, attention implementation, precision flags and implementation sources. The model geometry must be four recurrent passes, 48 physical blocks per pass and residual width 2,048. Use the current validated Ouro runtime consistently throughout this extraction. Current runtime execution is not represented as the historical GPU's numerical execution.

## 2. Deterministic prompt population

The target is the nominal intermediate class `a+b` among integers 2 through 18. It is not the final product. Every literal prompt is `f"({a} + {b}) * {c} = "`, including the final space. The known product `(a+b)*c` may be retained as metadata; no correctness generation is required.

Rows 0 through 647 retain the original enumeration exactly: `a` ascending 1 through 9, within that `b` ascending 1 through 9, within that `c` ascending 2 through 9. This produces 81 ordered pairs, 45 unordered pairs and 648 prompts.

For the 576 new rows, enumerate the set

`P_new = sorted((a,b) for a in range(1,18) for b in range(a,18) if a+b <= 18 and b >= 10)`.

There are exactly 36 unordered pairs, all with `a < b`. For each pair in this lexicographic order, append orientation `(a,b)` and then `(b,a)`; within each orientation append multipliers 2 through 9 in ascending order. Append these rows after the original 648. The complete population is exactly 1,224 distinct literal prompts. There is no sample selection or RNG for the new population.

New pair counts by sum 11 through 18 are respectively 1, 2, 3, 4, 5, 6, 7 and 8. No new positive-integer pairs exist for sums 2 through 10 while preserving these target classes. Save the complete ordered prompt/operand/label list and its canonical digest before inference. Validate that every new unordered pair is outside every original fold's test pairs, including reversed operand order.

## 3. Frozen folds, classes and locations

For original row `i`, use `folds[i]` and `labels[i]` from the original CV arrays. Independently reconstruct the old seed-0 assignment for validation: enumerate the 45 pairs `(a,b)` with `a=1..9`, `b=a..9`; let `order=default_rng(0).permutation(45)`; assign `fold[order[k]]=k % 5`; give both orders and all multipliers their unordered pair's fold. This reconstruction verifies the saved assignment; it does not choose a new split.

For fold `f`, let `S_f` be the set of labels on original rows with `folds != f`. Baseline training uses exactly those original rows, including the original inner-validation rows that already entered the final classifier. Augmented training adds every new row whose label belongs to `S_f`. Do not repeat inner selection, reserve a different validation subset, add a missing class or move an original held-out row into training. New rows are training-only and are not assigned new test folds.

| Fold, zero based | Original training rows / pairs | Classes absent from original training | Added rows / pairs | Augmented training rows / pairs | Eligible test rows | Probe physical layer | C | Raw logit-lens physical layer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 520 / 36 | 4 | 576 / 36 | 1,096 / 72 | 104 | 39 | 0.1 | 31 |
| 1 | 512 / 36 | none | 576 / 36 | 1,088 / 72 | 136 | 31 | 1 | 25 |
| 2 | 528 / 36 | 3, 17, 18 | 336 / 21 | 864 / 57 | 80 | 29 | 1 | 31 |
| 3 | 512 / 36 | none | 576 / 36 | 1,088 / 72 | 136 | 42 | 1 | 35 |
| 4 | 520 / 36 | 2 | 576 / 36 | 1,096 / 72 | 120 | 26 | 1 | 31 |

Every selected layer is in loop 1 (`current_ut=0`). Physical numbering in this document is one based. The uniform extraction axis is **[25,26,29,31,35,39,42]**, corresponding to virtual indices **[24,25,28,30,34,38,41]**. Save all seven locations for all 1,224 rows; the feature tensor is FP16 `[1224,7,2048]` (35,094,528 payload bytes, 33.46875 MiB). Neither new locations nor new layer selections may be added after extraction or scoring.

The eligible test mask is the exact saved `audit_pairing.npz:eligible`, shape `[648]`, and must equal `labels[i] in S_folds[i]`. It contains 576 rows from 39 complete unordered pairs. The six excluded pairs remain excluded: `(1,1)`, `(1,2)`, `(1,3)`, `(2,2)`, `(8,9)`, `(9,9)`. Their 72 original rows remain available only in the same original training partitions where permitted. Added data must not expand the evaluation mask.

## 4. Common fresh feature extraction and raw reference

Recompute **all** original and new prompts in the specified order on one current runtime/device configuration. Use native batch-one BF16 evaluation, frozen parameters, no gradients, four complete passes and `use_cache=False`, as in the original `OuroLensModel`/`HFLensModel.forward` path. No Jacobian construction or approximation is involved.

Preserve `OuroLensModel.encode(text, max_length=512)`: tokenize the literal prompt with the original tokenizer, truncation length 511, and prepend one explicit BOS exactly as that method does. Save each actual input-ID sequence, length and readout index. These short prompts must not be truncated. The readout is the last token of the complete encoded literal prompt. Do not trim the trailing space, use a prompt/continuation common-prefix position, or import the main J-Lens calibration's T128/skip-16 position rule.

Capture the seven post-block residuals using loop-filtered taps for `current_ut=0`; later recurrent passes must not overwrite them. Store `rec.activations[v][0,-1].half().cpu()` in the declared axis order. Both primary probe arms read the exact same saved fresh feature rows, including the test rows. They may fit different training scalers as specified below. No original cached feature is mixed into a primary training or test matrix.

Raw logit-lens scores use these same freshly stored old-prompt features and the unchanged model's final norm and native output head. For each original prompt, read its full `[7,2048]` FP16 row, cast to float32, and call the native `m.unembed` on that seven-location matrix; the existing implementation converts to the head's BF16 dtype before normalization/head application. Convert resulting logits to float32 for scoring. Keep this readout batching fixed. This is a fresh execution of the existing readout rule, without a claim of historical numerical identity.

For each class `s=2..18`, freeze the ordered token set from `single_token_ids(tokenizer, surface_forms(str(s)))` using the source identities below. Save the literal forms and token IDs before scores are examined; every class must have at least one token. Use the maximum native logit across that class's existing single-token aliases. Score all 17 classes, including classes a particular probe fold cannot train. Do not switch to one-form scoring, vocabulary hit@10, control-adjusted excess, or a new alias rule. Save candidate scores/ranks for the original 648 rows at all seven locations. A fold's raw reference then uses only its fixed location from the table.

There is no fresh J-Lens scoring or fitting in P4. The historical selected J-Lens scores in the frozen inputs refer to the older N80 map. They may be identified as historical context; neither those archived scores nor a new N100 map are a contemporaneous comparator for the fresh probes.

## 5. Fifteen fixed CPU classifier fits

Run five fresh baseline fits, five fresh augmented fits and five diagnostic baseline fits on the historical cache. The diagnostic arm uses exactly the same original training/test rows, selected locations and new weighted CPU fitting path as the fresh baseline. For historical `H[648,192,2048]`, select the corresponding virtual index `physical_layer-1`.

For each arm and fold, convert the selected FP16 features to NumPy float32, preserving all 2,048 coordinates. Fit `StandardScaler(copy=True, with_mean=True, with_std=True)` only on that arm's training rows, passing its sample weights explicitly, then transform training and test features with that scaler. Use the same path even when all weights are one. No test data enter scaling, weighting, fitting or selection.

Baseline and historical-diagnostic weights are one. In augmented fold `f`, let `n_fs` be the original training row count of class `s` and `N_fs` the augmented training row count of that class. Assign **every** augmented-training row of that class, old or new, weight `w_fs=n_fs/N_fs`. This is not a weight applied only to new rows. All included classes have `n_fs>0`. Mathematically, each class's total weight is `n_fs` and the total is the original training-row count. Retain integer counts, the weight formula and the actual floating weight sums so normal arithmetic rounding is visible; no outcome-dependent renormalization is permitted.

Fit `LogisticRegression` with the table's fixed C, `solver="lbfgs"`, `penalty="l2"`, `tol=1e-4`, `max_iter=2000`, `fit_intercept=True`, `class_weight=None`, `warm_start=False`, and otherwise the pinned implementation's defaults. Pass the same sample weights to `.fit`. Use one native BLAS/OpenMP thread via `threadpool_limits(limits=1)` covering both scaling and classifier operations; do not vary thread count across arms. The installed sklearn implementation uses `l2_reg_strength=1/(C*sum(sample_weight))`, so preserving total weight preserves the mean-loss regularization coefficient with the original C.

Retain coefficients, intercepts, classes, scaler statistics, actual settings, iteration counts and convergence diagnostics. A fitting error, non-finite output or convergence warning makes the experiment incomplete. Do not raise iteration limits, tune tolerance/C, change preprocessing or report only successful folds. Numerical differences from the historical classifier are not convergence failures and are not acceptance or tuning gates.

## 6. Metric and exact saved paired resamples

The candidate set is the 17 nominal sums, in ascending order. For probes, place `predict_proba` columns at their `clf.classes_` positions in a 17-column array and leave absent classes at probability zero. For either probe or raw logit lens, the true-class rank is

`rank_i = sum_s(score_i[s] > score_i[true_label_i])`.

Top-1 success is `h_i = 1[rank_i == 0]`. Preserve the strict `>` comparison: a tie for the highest score counts as success. Do not substitute `clf.predict()==label`, whose tie-breaking is a different rule. Evaluate each original row with the classifier and raw-reference location for its own outer fold, and apply the frozen eligible mask to every reported accuracy and contrast.

Point accuracy is the ordinary prompt-weighted mean over the 576 eligible rows, not an equal-weight mean over pairs or folds. The primary point is

`Delta = sum_{i:eligible[i]} (h_augmented[i] - h_fresh_baseline[i]) / 576`.

Use the **saved** arrays from the exact `audit_pairing.npz` above:

- `weights`: integer `[20000,39]`; each row gives bootstrap multiplicities in the saved cluster-column order and sums to 39.
- `clusters`: integer `[39]`, original unordered-pair IDs in that column order.
- `counts`: integer `[39]`, each 8 or 16, summing to 576.
- `pid`: integer `[648]`, original row-to-pair mapping.
- `eligible`: boolean `[648]`, the fixed mask.

Verify these arrays against the original pair enumeration, labels and folds. The source generated draws with `default_rng(20260907)`, sampled 39 eligible pair IDs with replacement, and retained draws containing at least one pair from each of the five outer folds. **Use the retained bytes; do not generate replacement draws, change the seed, resample individual prompts, rebalance classes, or reselect layers within a draw.**

For method `m` and cluster column `g`, set `A_m[g]=sum_{i:pid[i]==clusters[g]} h_m[i]`. These clusters contain only eligible rows. Let `D=weights @ counts`. The 20,000 bootstrap accuracies are `(weights @ A_m)/D`. For a paired contrast use `(weights @ (A_left-A_right))/D`, sharing the exact same draws and denominator. Report the 2.5th and 97.5th percentiles with NumPy's fixed default linear percentile rule. This is the fixed-choice formula used by the saved audit, not its alternative policy that reselects lens layers.

Report primary augmented-minus-fresh-baseline accuracy and interval. Also report the fixed fresh-baseline and augmented accuracies, the fresh raw logit-lens accuracy, and the two probe-minus-fresh-raw contrasts using these same saved paired draws. The raw-gap comparisons are descriptive fixed comparisons, not extra selected primary tests. A confidence interval containing zero does not demonstrate equivalence or that a gap is closed. No equivalence margin or equivalence test is introduced here.

The resamples quantify conditional held-out pair variation for these fixed data, classifiers and locations. They do not refit classifiers, resample the added training pairs, estimate uncertainty over augmentation families, or make the overlapping training folds independent replicates.

## 7. Historical diagnostics and interpretation

Report the historical-cache diagnostic baseline versus the archived original baseline, then the fresh baseline versus that diagnostic baseline. The first comparison identifies sensitivity to the current weighted CPU path/runtime on fixed old features; the second changes the feature cache while holding that CPU path fixed. Preserve per-row ranks/predictions so differences are inspectable. Compare original cached and fresh old-prompt features at the five probe locations descriptively, including bitwise equality and absolute/relative differences computed after promotion from FP16.

There is **no historical feature-error, prediction-agreement or accuracy tolerance** for accepting the primary fresh augmentation comparison. Do not tune, switch runtime/backend, choose a cache or repeat extraction until historical numbers match. Neither diagnostic's scientific outcome is a gate. Source/model/tokenizer/position identity, complete finite data and the declared fitting procedure determine validity.

A positive primary contrast supports the usefulness of this specific upper-sum augmentation on the fixed original test set. A null or negative contrast does not establish that the original probe was not data limited. Only sums 11 through 18 gain pair units, and only 264 of the 576 eligible test rows have these sums. Operands above nine change token/content distributions; class weighting does not remove this shift. The experiment is not a balanced increase in iid original-domain data, a causal test of output-head supervision, or evidence that the labelled intermediate is used in producing the final product. Do not claim the historical gap was explained solely by training-data size.

## 8. Timing, ordering and completion

The hard ceiling is **900 seconds of total marginal P4 GPU-phase wall time**, including any P4 model loading. Start a supervised monotonic timer before the first P4-specific GPU-phase initialization, snapshot/model loading or inference; include fresh feature extraction, raw-head scoring, local output serialization/hashing and cleanup. A pre-existing loaded model can make loading cost zero, but a load cannot be placed outside the timer. No new pod is allocated by this contract. Record all P4 phase timestamps and elapsed time.

Enforce the deadline externally as well as at work boundaries. If the deadline is reached, terminate the P4 work, retain a clearly partial/failure record and mark the experiment incomplete. Do not analyze an interrupted subset, skip rows, shrink the seven-location axis, reduce recurrence, change batch/precision, omit the old-data recomputation, omit a comparison, or select a faster subset to manufacture completion. No scores from an incomplete P4 run may enter the scientific report.

The historical all-purpose cache timings imply approximately 561–614 seconds for 1,224 prompts before model loading; they are a planning proxy, not a current-device benchmark or runtime bound. A 900-second allocation is an explicit limit, not a promise that the complete task fits it.

Actual execution remains after P1/P2 interpretation and a current budget check. It may use spare time in an already planned wait before a Huginn receipt if available; that opportunity is optional, and P4 must not force a delay or displace the higher-priority work. Record all extension elapsed time, including CPU/retrieval/coordination time while the paid resource remains rented. Recompute the later Huginn time/spending projection with that elapsed time and actual charges included. The existing combined **$25 cap** continues to apply; P4 has no separate spending exemption.

Only a complete feature cache and fresh raw scores passing identity/finite/shape checks may enter CPU analysis. Completion then requires all 15 fixed fits, full per-item scores and diagnostics, unchanged input identities and the declared paired summaries. Preserve a manifest binding this contract, the exact implementation/runtime, every consumed source/data artifact and every output byte. Partial files or existence alone are not completion. Historical artifacts and prior P1/P2/Huginn outputs remain separate, unchanged evidence.

## 9. Existing code and local CPU environment

This is a new bounded workflow, not an instruction to run an existing full CLI. Reuse the following source semantics through a thin implementation that supports the declared rows, seven locations and weighted fits; the legacy CLIs assume 648 rows and/or perform additional selection and fitting.

The GPU extraction process should import `load_ouro`/`OuroLensModel` from `ouro_jlens.recurrent`, `ActivationRecorder` from `jlens.hooks`, and the candidate-form helpers from `ouro_jlens.evaldata` directly. Do not import `probe.py`, `probe_cv.py` or the audit entrypoint remotely: they import scikit-learn, which is deliberately absent from the GPU runtime. Their code is a semantic reference for the thin wrapper. Product-generation helpers such as `greedy`/`is_correct` are not required by this feature-only contract. Classifier dependencies remain in the existing local CPU environment.

| Canonical source | SHA-256 | Relevant entrypoints |
| --- | --- | --- |
| `/home/moloch/ouro_project/src/ouro_jlens/probe.py` | `15914aa97704afbe60ba7f335c3ac16a501d57679476c78d96cbc6809cf020d3` | `make_prompts` at line 297; `cache(m,prompts)` at line 315; `lens_candidate_ranks` at line 330. The existing cache also stores all 192 locations and generates correctness continuations; P4 retains its extraction semantics only. |
| `/home/moloch/ouro_project/src/ouro_jlens/probe_cv.py` | `370500ae9749a062651d6e6e4e9756e9d20e19346288174118724b5ebff1870f` | `fold_assignment`, `prompt_folds`, `_scaled`, `_fit_location`, `_rank_of_true`. `_fit_location` searches C and must not be invoked unchanged. |
| `/home/moloch/ouro_project/src/ouro_jlens/recurrent.py` | `4477cde085e9e5965c33ebe737ca6b84148a0bd19562c1fa9e6077168e5a96ed` | `LoopTap`, `OuroLensModel.encode`, `load_ouro`, model revision and full snapshot enumeration |
| `/home/moloch/ouro_project/src/ouro_jlens/evaldata.py` | `2772ddb2cbaa2a7c1fce2a23db14b36fbf9c266aabbd0425cf2c566ed01e6c74` | `surface_forms`, `single_token_ids` |
| `/home/moloch/jacobian-lens/jlens/hooks.py` | `c781d6944fd23396d3fc65a04db1f1db807f6f12cd5912cdbd2fb67eb3508081` | `ActivationRecorder` post-block capture |
| `/home/moloch/jacobian-lens/jlens/hf.py` | `228cf078e4586a7b7f61a6f5064403b8960de337afd19256efa56f04d53e3222` | `HFLensModel.forward`, `unembed`, evaluation mode and frozen parameters |
| `/home/moloch/jacobian-lens/research/followup_2026-09-07/probe/audit.py` | `7b49d87389fe7adb63182569ceff27069b33c0faa52d71260864415dbe61c94c` | `design`, `rank_analysis`, `bounded_refits`; saved pair-resample and fixed-location formulas. Do not run its full entrypoint to repeat previous sensitivity experiments. |

Use the existing local CPU interpreter `/home/moloch/ouro_project/venv/bin/python`, not an installation on the active GPU. Available dependency metadata: NumPy 2.4.4, scikit-learn 1.7.2, SciPy 1.18.0, threadpoolctl 3.6.0. Record the actual interpreter, dependency versions, BLAS implementation and thread settings at execution. The accepted model runtime pins include CPython 3.14.7, Torch `2.12.0.dev20260407+cu128`, Transformers 4.54.1 and NumPy 2.4.4; actual device/attention/precision configuration must also be recorded and fixed for this extraction.

Weighted API/objective evidence is the installed `/home/moloch/ouro_project/venv/lib/python3.14/site-packages/sklearn/preprocessing/_data.py` (`StandardScaler.fit(sample_weight=...)`, SHA-256 `1255040a0eb9c99f58775d55cf34f675cf61539f7c0a0dd0d6a36cb1f0da1e15`) and `/home/moloch/ouro_project/venv/lib/python3.14/site-packages/sklearn/linear_model/_logistic.py` (`sample_weight`, `sw_sum` and `l2_reg_strength`, SHA-256 `981f4e11c9a6fc2b02cd9322f2926849143e4a8bf1819e3a4159a9fce615a00c`). The CPU path uses these supported APIs; it must not patch the solver or change its objective.

No model loading, feature extraction, classifier fitting, new scores or cloud actions were performed to create this contract.
