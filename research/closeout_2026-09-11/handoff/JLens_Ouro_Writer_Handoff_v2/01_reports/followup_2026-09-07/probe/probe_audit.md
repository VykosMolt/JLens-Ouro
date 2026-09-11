# Supervised-probe follow-up, 7 September 2026

The supervised probe still loses to the logit lens in loop 1. I reproduced the saved scores and all five selected loop-1 classifiers exactly, then checked token aliases, layer selection, solver tolerance, one additional regularization value, and a small matched training-size contrast. None explains the gap. Small training data remain a plausible explanation; this follow-up does not establish that more data would close it.

These are analyses performed after the submitted application. The original report and artifacts were left unchanged. The J-Lens scores in this arithmetic experiment use the older, incompletely documented 80-prompt lens, not the later 100-prompt fit used for the main task result.

## What was actually trained and scored

The family is all 648 strings `(a + b) * c = ` for `a,b=1..9` and `c=2..9`. The target is the task-defined intermediate `a+b`, represented as a nominal class among the 17 integers 2–18. It is not the model's final product and does not establish that the model used addition in producing that product.

The frozen features are the same last-prompt-token hidden states for all three methods: `H[648,192,2048]`, stored as float16. There are four recurrent passes through 48 physical layers. The trained probe is multinomial logistic regression after a StandardScaler fitted on training rows. C is chosen from 0.01, 0.1, and 1 using eight inner-validation unordered operand pairs; layer selection uses those same inner scores. Each classifier is then refitted on all 36 outer-training pairs and scored only on its untouched outer fold. No mirror pair crosses a training/test boundary.

The 576 in the write-up is the eligible evaluation count, not the number of prompts in each fitted classifier. The five refits see 520, 512, 528, 512, and 520 prompts, but only **36 distinct unordered operand pairs** each. Inner fitting uses 392–408 prompts from 28 pairs, and selection uses 112–120 prompts from eight pairs. The nominal feature dimension is 2,048. Multipliers and operand order create repeated labels within a pair, so treating all prompts as independent would exaggerate precision.

The 576 eligible prompts come from 39 pairs. Another 72 prompts, from pairs `(1,1)`, `(1,2)`, `(1,3)`, `(2,2)`, `(8,9)`, and `(9,9)`, have a sum absent from their outer-training fold; they remain in the raw arrays and are excluded from the fair comparison. Every reported method uses the same 576-row denominator. The score is candidate-set top-1, not the main experiment's excess vocabulary hit@10. Accuracy is prompt-weighted; uncertainty resamples whole unordered pairs, preserving their 8- or 16-prompt sizes.

The quoted 0.125 majority reference is the global eligible label frequency, equivalently the accuracy of always predicting 10. It is not a majority classifier estimated separately within each training fold. A training-fold majority classifier with smallest-label tie-breaking scores zero on these particular folds; this reflects their uneven label composition and is not a useful replacement headline baseline.

## Independent reproduction and bounded checks

I reconstructed prompt order, folds, inner-validation pairs, eligibility, selected layers, and bootstrap grouping without importing the original evaluator. The reported points reproduce exactly. The primary intervals below use 20,000 paired cluster draws, holding each method's selected classifiers/layers fixed. They are conditional on this fitted experiment and do not include uncertainty from refitting classifiers or lens fits. The original report instead reselects the fixed lenses' layers on each bootstrap draw while fixing the probe's inner choices; I also reproduced that policy independently, with the same conclusions. Both sets of intervals are saved.

| Loop | Probe | Logit lens | Older J-Lens | Probe minus logit lens, paired 95% interval |
|---|---:|---:|---:|---:|
| 1 | 0.5938 | 0.7639 | 0.3924 | −0.1701 [−0.2945, −0.0486] |
| 2 | 0.5590 | 0.5000 | 0.5000 | +0.0590 [−0.0240, +0.1461] |
| 3 | 0.3559 | 0.1788 | 0.1476 | +0.1771 [+0.0587, +0.2939] |
| 4 | 0.2483 | 0.3108 | 0.1198 | −0.0625 [−0.1620, +0.0405] |

The probe's loss is a loop-1 result. It does better than the logit lens in loop 3 on this task, although absolute accuracy is only 0.356. That qualification should accompany any general statement about the supervised probe.

| Check | Loop-1 result | What it tells us |
|---|---|---|
| Refit the five published selected locations | All 648 saved outer-fold ranks reproduce exactly; training top-1 is 1.000 in every fold | No discrepancy found between saved ranks and current fitting code at the locations behind the headline. Poor training fit does not explain the loss. |
| One fixed token form per candidate instead of a maximum over forms | Logit lens 0.7396 versus 0.7639 originally | Alias count is a real asymmetry, but removing it leaves a large native-head advantage. |
| Give lenses the same inner-validation pairs and available labels for layer selection as the probe | Logit lens 0.7899; probe minus logit lens −0.1962 [−0.3204, −0.0725] | The lenses' larger original selection set does not explain their loop-1 advantage. |
| Match inner selection **and** use one form | Logit lens 0.7361; probe difference −0.1424 [−0.2790, −0.0069] | The two identified evaluation asymmetries do not remove the deficit. |
| Tighten lbfgs tolerance from 1e−4 to 1e−6 at fixed C/layers | Probe 0.6111; improvement +0.0174 [+0.0017, +0.0330] | Solver accuracy matters modestly. The remaining difference from the original logit lens is −0.1528 [−0.2761, −0.0352]. |
| Add exactly C=10 at the five fixed layers; use inner validation to choose whether to adopt it | Probe 0.5955; improvement +0.0017 [−0.0106, +0.0158] | Four original choices sit at the upper grid boundary, but one further decade does not explain the gap. Only one fold selects C=10. |
| 18 versus 36 training pairs; preserve every full-training class in the half-sized samples | 0.5544 versus 0.5938; improvement +0.0394 [−0.0288, +0.1094] | The direction is consistent with data limitation, but the interval is inconclusive. |

The training-size check used three predetermined half-sized subsets per fold, identical outer test rows, the original selected layers, and the original C adjusted so the L2 penalty relative to **mean** loss stays constant as sample count changes. Each half subset contains all classes present in the corresponding full training partition. This prevents merely losing target classes from manufacturing an apparent learning curve. The half-size point averages each test prompt's correctness across the three subsets; the interval is paired by test pair and conditional on those subsets. Layers and original C were selected using the full inner data, so this is a conditional sample-reduction check, not a new end-to-end learning-curve benchmark.

The tighter solver used 132–268 iterations instead of 12–27, with no convergence warnings in either run. It changed 36 of the 576 eligible predictions and gained ten correct answers net. The default solver's stopping rule therefore leaves a small measurable sensitivity even when no warning is raised. The original number should stay in the historical account; the improved tolerance is a follow-up result.

The layerwise curves are in [probe_layerwise.png](probe_layerwise.png) and [probe_layerwise.csv](probe_layerwise.csv). They show the loop-1 logit lens advantage over a broad middle-to-late region, alongside probe advantages in later passes. They are descriptive fixed-layer curves, not a license to report a test-selected optimum. Eight random eligible prompts, including failures of every readout, are retained in [random_examples.json](random_examples.json).

## What changed the interpretation

The model's native unembedding is already a strong linear classifier on these exact features. Ouro uses a shared RMS normalization followed by a bias-free linear output head. For one token per candidate, the RMS denominator is a positive scalar common to all candidate scores, so it cancels in the argmax; apart from finite-precision effects, the resulting classifier lies within the supervised probe's linear hypothesis class. StandardScaler is an invertible affine change of the nonconstant feature coordinates. Therefore the learned probe's low score cannot be used to argue that a linear readout lacks access to the intermediate. A useful linear readout is already present in the native head.

The distinction is learning that readout from this small task family versus inheriting it from language-model training. Perfect training accuracy, 36 independent training pairs, a 2,048-dimensional state, and variation across folds all make data limitation plausible. The controlled half/full check is not precise enough to turn that into the explanation. The 17-class target also imposes an avoidable coverage constraint for this arithmetic family; a scalar or structured target would be a different probe experiment, not a reinterpretation of the existing one.

Intermediate output-head supervision is compatible with a strong native head, but this probe audit cannot identify its causal role. A matched training ablation or another model is needed. It also does not show that any labelled intermediate is causally used in producing the answer.

## Concerns and bugs retained in the record

- Historical mirror-pair leakage was real and inflated older probe results. The current folds keep both operand orders together; no pair leakage was found.
- Historical test-selected maxima and cross-fold selection ancestry were corrected before the current arrays. Current supervised choices use only each outer fold's inner partitions. Test-oracle layer maxima remain unsuitable as held-out claims.
- The current outer-fold label exclusion is necessary and correctly implemented. Inner validation still contains 16, 40, 16, 16, and 0 prompts with labels absent from the corresponding inner-training split. Those rows score zero at every probe location; they reduce the useful selection sample and make absolute selection scores incomparable across folds. Removing those constant zero contributions cannot change the probe's selected layer.
- The lenses originally select their layer on more pairs than the probe and maximize over uneven token-alias sets. Both asymmetries were checked directly above; neither explains the main loop-1 gap.
- Default solver tolerance measurably affects held-out predictions despite no convergence warning. A one-decade C extension has negligible effect; this does not rule out every optimization or regularization choice.
- The original bootstrap treats fixed lenses' layer selection and learned-probe selection differently. This is an uncertainty-policy choice, not evidence of test leakage. Fixed-choice and published-policy intervals both preserve the loop-1 deficit and loop-3 advantage.
- The 0.125 majority reference needs the explicit global/constant-10 definition above. Foldwise training majority choices are 9, 10, 9, 8, and 10, all absent from their corresponding test folds. The class-imbalance pattern further limits simple interpretations of this tiny family; it does not invalidate the paired readout comparison.
- The old `probe_cv.py` module docstring still states that being underpowered is **why** the native head wins. That wording exceeds the evidence. Historical audit prose alternately asserted and rejected the data-limited explanation, sometimes comparing different training sets, test sets, label populations, and layer aggregations. The later historical learning curve was also on all 648 prompts with its own test-selected best layers. Its referenced `p3/probe_curve.{py,json,npz,log}` was not found at the stated path; I did not treat that prose as independently verified current evidence.
- The old J-Lens fit has incomplete fitting provenance. Saved array and cache hashes match their manifests, and the five relevant current classifiers reproduce exactly. The source manifest for `probe_report.py` does not match today's file: the archived design records 13,203 bytes while the current file is 13,613 bytes. Other checked CPU-source entries match. The array numbers remain independently reproducible, but the old `VALIDATED_CURRENT_SOURCE...` label is no longer literally true for that reporting source.
- The saved arithmetic correctness vector can support a descriptive subgroup table, but raw generations are not in this cache. I did not independently regenerate or verify those 648 correctness labels; they are not used to explain the supervised-probe deficit here.
- The first run of this follow-up script stopped during JSON writing because unordered-pair entries were NumPy integers. I changed them to Python integers and reran before any fit or metric artifact was reported. This was a serialization failure, not a scoring change.

## Chronological log

- Before refitting: read the complete application draft, current decision ledger, historical probe audit sections, probe implementation, fold/report code, provenance, saved scores, and relevant tests. Located the complete hidden-state cache and verified its identity. Flagged the unsupported causal wording in the source docstring.
- Before seeing new fit results: specified five original selected loop-1 locations only; exact reproduction; tighter tolerance; a single additional C; and 18-versus-36 class-covering pair subsets at fixed effective regularization. Waited for the parent analysis to finish its full-paper reading before starting fits.
- 17:34–17:35 UTC: independently reproduced the reported points, ran the bounded refits, and retained all predictions. A JSON serialization failure was fixed before the successful run. All five original classifiers reproduced saved ranks exactly.
- Before the matched-selection calculation: identified the unequal layer-selection sample sizes and specified the same-inner-pairs/classes check without examining its outcome.
- 17:36–17:37 UTC: matched-selection analysis preserved the loop-1 deficit. Inspected the plotted curves and eight random examples. No broad sweep was run.
- 17:41 UTC: distinguished the global majority reference from a training-fold majority classifier and independently checked the surprising zero against the saved fold-by-label counts. No lens score changed.

## What I would test next

If the supervised-probe question remains worth pursuing, use a larger arithmetic family with many independent generating pairs per sum and freeze test pairs and target classes before varying training size. Keep solver tolerance and validation procedure fixed. This would test whether the gap closes with data without confusing sample size with label coverage or a changing test population. A scalar-sum probe could test whether the nominal classification target is wasteful, but should be reported as a different objective.

Code and complete machine-readable results: [audit.py](audit.py), [saved_rank_audit.json](saved_rank_audit.json), [bounded_refit_results.json](bounded_refit_results.json), and [bounded_refit_predictions.npz](bounded_refit_predictions.npz).
