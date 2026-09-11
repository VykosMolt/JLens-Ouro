---
title: 'J-Lens in Ouro: Early-Pass Deficits and a Confirmed Late-Pass Advantage'
author: 'Jan Kirin'
date: 'Working draft · 11 September 2026'
lang: en-GB
---

## Abstract

Does a Jacobian-based readout reveal intermediate content that a model’s own output head misses during recurrent computation? I compare J-Lens with the raw logit lens in frozen Ouro-2.6B, which applies 48 shared blocks over four recurrent passes. The initial comparison favoured the raw lens on multihop questions in passes 1–3. Layerwise analysis also found a positive region in pass 4, and five calibration fits reproduced its location and magnitude on the discovery set. I then froze one estimator and the layer band before evaluating 160 new two-hop questions with 80 new intermediate concepts. In pass 4, physical layers 26–37, J-Lens improves excess hit@10 by **23.19 percentage points**, with a 95% dependency-group bootstrap interval of **16.29–32.90 points**. Intended-concept recovery is 37.92%, compared with 14.43% for raw readout. The early-pass deficits also transfer. Changing the derivative target or position aggregation substantially reduces the late advantage. Higher-precision scoring and a cross-hardware state replay change the primary estimate by at most 0.42 points. The result supports a localized, estimator-specific improvement in concept recovery, not general J-Lens superiority, improved answer accuracy, or causal use of the recovered content. [S2–S4]

## 1. The question

A readout can miss useful content even when that content is present in a model’s hidden state. The raw logit lens applies the model’s output head directly to an intermediate activation. J-Lens instead first applies a fixed, Jacobian-derived map. It averages how activations affect downstream states across calibration contexts, rather than fitting a predictor to one task’s answers. Its intended target is verbalizable content, not accurate next-token prediction in every context. [1]

Ouro gives this comparison two depth axes: physical layer and recurrent pass. The same layer stack is reused, so a readout that works at one pass need not work equally well at another. The experiment asks where each method recovers a task-annotated intermediate concept more reliably. It does not assume that a recovered word was part of the model’s actual reasoning. [2, S1]

My initial expectation was that J-Lens would provide a broadly better view of intermediate computation. That expectation did not survive the early-pass results. The positive finding came later: a specific region of the final pass where J-Lens outperformed raw readout. This paper separates the exploration that found that region from the new-question test that checked it. [S1–S3]

The distinction matters beyond this particular comparison. Hidden-state readouts could complement observation of visible reasoning, but only after establishing what their scores measure. No monitoring system, deception detector, or chain-of-thought baseline is evaluated here.

## 2. Readouts and measurement

### 2.1 Model and fixed estimators

The model is **ByteDance/Ouro-2.6B**, not the Thinking or RLTT checkpoint. The recorded revision begins `1ed04250`. It is frozen and evaluated with four passes through 48 blocks, a residual width of 2,048, and a vocabulary of 49,152 tokens. I use one-based physical layers and pass numbers; the stored virtual-layer index is $48(r-1)+(\ell-1)$. [S3, S4]

For a captured residual state $h_{i,r,\ell}$ at the recorded readout position of item $i$, let $N$ be the model’s final normalization and $W_U$ its unembedding. The two vocabulary readouts are

$$
z^{\mathrm{raw}}_{i,r,\ell}=W_U N(h_{i,r,\ell}), \qquad
z^{J}_{i,r,\ell}=W_U N(B_{r,\ell}h_{i,r,\ell}).
$$

Here $B_{r,\ell}$ denotes the frozen J-Lens matrix actually applied to the state. Ranking the logits gives the token readout; no task classifier is trained on the confirmation questions. The main banks target the final residual state. The original J-Lens study’s default Sonnet configuration uses a penultimate-layer target, so this is an adaptation, not an exact reproduction of that configuration. [1, S2–S4]

Five Ouro lenses were fitted on 100 calibration paragraphs each. The realized paragraph sets were disjoint, although some source articles overlapped. The first fit, **fit01**, was preselected for confirmation rather than chosen for its score on the new questions. Fit02 was retained as a secondary comparison. [S2, S3]

Two further comparisons examine estimator specification. The target control changes the derivative target from final to penultimate while retaining final normalization and unembedding. The position controls use the same sampled target positions and all 2,048 derivative directions per paragraph: *sampled-sum* combines derivatives over valid source positions, whereas *diagonal* retains only the source position matching the target position. These are distinct estimator constructions, not interchangeable implementations assumed to measure exactly the same thing. [S2]

### 2.2 What counts as recovery

Each confirmation question has one annotated intermediate concept and a fixed set of accepted single-token forms. A concept is recovered when at least one accepted form ranks among the first ten tokens of the **full vocabulary**. This is not a ranking restricted to the candidate names. Each item’s controls are the other 79 intermediate names, equally weighted. [S3, S4]

Let $H_{i,r,\ell}^{m}(a)$ be the hit indicator for name $a$ under method $m$, and $a_i$ the intended name. With control set $C_i$, the fixed-layer excess score is

$$
e_{i,r,\ell}^{m}=H_{i,r,\ell}^{m}(a_i)
-\frac{1}{79}\sum_{c\in C_i}H_{i,r,\ell}^{m}(c).
$$

The primary endpoint is the equal-weight item mean of the paired difference, averaged across the twelve fixed layers in the fourth pass:

$$
\widehat{\Delta}
=\frac{1}{160}\sum_{i=1}^{160}\frac{1}{12}
\sum_{\ell=26}^{37}\left(e_{i,4,\ell}^{J}-e_{i,4,\ell}^{\mathrm{raw}}\right).
$$

This averages fixed-layer measurements. It does **not** ask whether a concept appears anywhere in the band. The historical *any-layer* score separately gives each intended name and each control 48 opportunities for recovery, then averages controls, labels within items, and items. That score answers a different question and is reported separately. [S1, S3]

All effects below are in percentage points. The recovery rates are averages over item–layer cells, not the fraction of questions the model answers correctly. Subtracting control recovery makes the comparison more informative, but does not turn it into a causal test or a calibrated measure of semantic correctness.

## 3. From discovery to a frozen new-question test

### 3.1 What the discovery work established

The discovery evaluation contained 90 eligible multihop items and 51 arithmetic items. Fixed-layer analysis showed that the early multihop loss was not merely an artifact of allowing the raw lens to recover a concept at a favourable late layer. The disadvantage remained in the middle of the stack. Arithmetic was mixed. [S1]

Pass 4 differed. Across five calibration fits, the mean excess advantage in layers 26–37 was **18.72 points**, with a between-fit standard deviation of **0.39 points**. Every fit peaked at physical layer 32. The gain predominantly reflected increased intended-concept recovery. [S2]

That was evidence about calibration variability on the **same questions**, not an independent replication across evaluation populations. Both positive band estimates had simultaneous intervals containing zero, and no positive individual layer cleared the simultaneous shared-concept interval. The band was therefore treated as a discovery to test, not a settled general advantage. [S2]

### 3.2 Confirmation population and inference

The confirmation set contains **160 two-hop questions, 80 intermediate concepts, and two questions per concept**, organised across ten subject domains. All 160 items were eligible. The intermediate concepts were absent from the discovery set. Construction and annotation were blind to method outcomes; factual hops, novelty, and control hygiene were checked before the local freeze. [S3]

This population changes the relation mixture substantially. The final audit records 96 questions using requested relation types absent from the discovery inventory, 45 using familiar requested relations, and 19 recombining familiar relations. This is a transfer test, not a difficulty-matched replication. Three component facts overlap known calibration text. No claim is made that the facts were absent from pretraining. [S3, S4]

Fit01, the pass-4 band, scoring rules, population, and analysis were fixed before any new-item outcome. The stopping rule was one evaluation of that population, with no sample extension or replacement endpoint. “Prospective” here refers to that recorded local freeze; a public preregistration is not supplied with the reports. Runtime-verification corrections and later numerical checks are distinguished in Appendix B. [S3, S4]

The primary interval uses **20,000 bootstrap resamples of whole dependency groups**, seed `2026090901`, with the 2.5th and 97.5th percentiles as limits. The grouping preserves dependence between related questions while retaining the item-weighted target. Twenty prespecified secondary contrasts form a separate simultaneous 95% family using bootstrap max-$t$. Component intervals are descriptive. [S3]

The 28 dependency groups are unequal: sizes 42, 24, 16, 10, 10, 6, four groups of four, and eighteen groups of two. The weight-based effective group count is $160^2/\sum_g n_g^2=8.63$. This is not literally a sample of nine observations, but it warns against treating 160 related questions as 160 independent demonstrations. Uncertainty remains conditional on this grouping and the frozen estimators. [S3, S4]

## 4. Results on new questions

### 4.1 The fixed final-pass advantage transfers

In pass 4, layers 26–37, fit01 exceeds the raw lens by **23.19 points** in excess hit@10, with a 95% dependency-group interval of **16.29–32.90 points**. [S3, S4]

| Quantity | Estimate | 95% group-bootstrap interval |
|:--|--:|:--|
| J-Lens intended recovery | 37.92% | 30.44–46.56% |
| Raw intended recovery | 14.43% | 10.46–17.87% |
| J-Lens control recovery | 0.34% | 0.18–0.51% |
| Raw control recovery | 0.04% | 0.02–0.08% |
| **Paired excess difference** | **+23.19 points** | **+16.29 to +32.90 points** |

*Table 1. Confirmation results for the frozen pass-4 band. Component intervals are descriptive; the final row is the primary inference. Rounded entries need not subtract exactly. Source: [S3].*

The intended-recovery difference is **23.49 points**. Control recovery is also slightly higher under J-Lens, reducing rather than creating its advantage. Thus the positive excess score comes from recovering more intended concepts, not merely suppressing controls. [S3]

Resampling individual questions instead gives an interval of 17.97–28.71 points. Leaving out each dependency group in turn keeps the estimate between 19.73 and 24.88 points. No single omitted group reverses the estimate; that does not establish robustness to every possible grouping. [S3, S4]

Fit02 gives a secondary advantage of **22.56 points**, with a simultaneous interval of **9.52–35.59 points**. The fit01-minus-fit02 difference is 0.63 points, with an interval of −0.73 to +1.99. The two fixed estimators perform similarly here, but this is not a formal equivalence result or an estimate over all possible calibration fits. Discovery and confirmation results are not pooled. [S3]

### 4.2 The early-pass deficit transfers too

All six prespecified early-pass comparisons remain negative on the new population. [S3, S4]

| Pass | Mean over fixed layers | Any-layer recovery |
|:--|:--|:--|
| 1 | −8.83 [−13.64, −4.01] | −44.41 [−59.03, −29.79] |
| 2 | −14.86 [−19.54, −10.18] | −57.37 [−72.01, −42.72] |
| 3 | −15.59 [−22.40, −8.77] | −61.08 [−79.59, −42.58] |

*Table 2. Fit01 minus raw excess hit@10, in percentage points. Fixed-layer means average all 48 physical layers within a pass; any-layer recovery is a separate aggregation. Brackets are simultaneous 95% intervals from the 20-contrast secondary family. Source: [S3].*

The late gain therefore does not overturn the original negative result. Both sides of the pattern transfer: raw readout remains stronger under the early-pass summaries, while J-Lens is stronger in the selected final-pass band. These summaries do not imply that J-Lens loses at every early layer or wins throughout pass 4.

### 4.3 The estimator specification matters

The control comparisons also transfer to the new questions. [S3, S4]

| Contrast in pass 4, layers 26–37 | Difference | Simultaneous 95% interval |
|:--|--:|:--|
| Main fit01 minus penultimate target | +14.38 | [+7.70, +21.07] |
| Sampled-sum minus diagonal | +15.77 | [+5.80, +25.75] |
| Penultimate target minus raw | +8.80 | [+0.76, +16.85] |
| Sampled-sum minus raw | +21.72 | [+9.16, +34.29] |
| Diagonal minus raw | +5.95 | [−0.34, +12.24] |

*Table 3. Estimator controls in percentage points. Comparisons use matched layer support within each control family. These intervals belong to the same 20-contrast family as Table 2. Source: [S3].*

Changing the derivative target reduces the advantage substantially. Restricting the position reduction to the matching source position does so as well. Neither comparison isolates how Ouro reasons: it shows that the measured recovery depends on the lens’s construction. A penultimate-target map can answer a different question from a final-target map without either being an implementation error.

The diagonal-only estimate is smaller and has borderline support. Its interval just excludes zero under one regenerated-state replay, while its magnitude is almost unchanged. It should not be described as either equivalent to raw readout or reliably ineffective. The direct sampled-sum-minus-diagonal comparison is the clearer result. [S4]

## 5. Numerical verification

The confirmation run used BF16 model execution on an RTX 5090. A development check initially failed because applying the output head to one residual row did not reproduce logits computed in a multi-row call. The subsequent verification measured whether this affected the scored comparisons, rather than assuming a small logit discrepancy was harmless. These checks were **post-confirmation**, not prospectively registered. [S3, S4]

On the local GPU, using the retained confirmation states, verified banks, and frozen scoring code reproduced the saved arrays bit for bit across **160 items, six arms, and all scored columns**. This included ranks, top-ten token IDs, saved exit logits, and sampled transported vectors and logits. Calls with 160, 190, 191, and 192 rows matched the executed layout exactly on that GPU. Single-row calls produced some numerical differences but did not change the primary endpoint. [S4]

| Computation | Primary difference | 95% group interval | Change from accepted |
|:--|--:|:--|--:|
| Accepted BF16 path | +23.19 | [+16.29, +32.90] | — |
| Alternative row packing | +23.19 | Unchanged | 0.00 |
| FP64 reference | +22.93 | [+16.17, +32.52] | −0.26 |
| Locally regenerated states | +23.61 | [+16.52, +33.77] | +0.42 |

*Table 4. Numerical sensitivity of the primary endpoint, in percentage points. The FP64 row is a reference readout computation using stored inputs, not an entirely FP64 model. Alternative packing was measured on the local GPU, not in a fresh RTX 5090 run. Source: [S4].*

CPU and GPU FP64 calculations produced identical ranks, hits, and top-ten sets, although their values differed by up to $9\times10^{-14}$. This is higher-precision reference computation, not exact arithmetic. Changing the order of exactly tied logits bounds the accepted primary estimate between **22.60 and 23.35 points**. [S4]

Regenerating states from the prompts on the local GPU was not bit-identical to the original forward pass: 37 of 160 state records matched exactly and 154 greedy continuations matched. Nevertheless, replaying every arm on those states shifted the endpoint by only 0.42 points. The early deficits and direct estimator-control conclusions survive the tested paths. Bit-level state regeneration on another RTX 5090 was not tested. [S4]

A separately implemented checker reconstructed the saved analysis and detected all 13 planted errors in isolated fixtures, including layer shifts, hit changes, missing controls, and changed group labels. “Independent” here means a separate implementation produced by a separate coding agent, not replication by another laboratory. The aggregate checks and the direct scoring replay establish different parts of the verification chain. [S4]

## 6. Interpretation and limits

The supported result is a **localized improvement in reading annotated intermediate concepts from fixed Ouro states**. It generalizes from discovery to the new evaluation population without refitting or selecting a new layer band. The simultaneous transfer of the early deficits makes the result more specific: the method is not uniformly better as computation proceeds.

One possible explanation for the strong raw baseline is Ouro’s training objective, which applies weighted next-token losses at recurrent pass boundaries through a shared output head. This does not supervise every physical block, and the present experiments do not isolate the effect of that objective. A historical Huginn pilot did not support the predicted relative improvement, and its incomplete artifact chain limits its evidential value further. Architecture, calibration, target choice, and executed depth remain entangled. [2, S1, S2, S4]

Recovery is also not use. In exploratory checks, a lens could recover a labelled intermediate on a question the model answered incorrectly. Nor does disagreement between two lenses directly measure recurrent revision: native exits can agree while their readouts disagree sharply. Appendix A retains those observations without promoting them into a causal claim. [S1]

The main population limit is the small number of unequal dependency groups. The reported interval accounts for the declared grouping, not every possible shared fact, relation, or template. The new relation mixture is not difficulty-matched, the labels are restricted to accepted single-token forms, and the evidence concerns one model checkpoint and two retained main calibration fits. The study does not establish performance on arbitrary concepts, other models, or monitoring under adversarial pressure. [S3, S4]

### Reproducibility and artifact loss

The confirmation banks, states, scores, frozen model inputs, and supporting code permit direct rescoring and forward replay as described above. A restoration test from the preservation bundle reproduced the frozen analysis. However, an earlier retrieval failure and a later storage purge destroyed unique historical artifacts, including original FP32 fitting checkpoints. The audit classifies 43 deleted paths, representing 33 distinct artifacts, as unique and lost. Exact reproduction of the original estimator-fitting process is therefore not available. [S2, S4]

This distinction is substantive: the fixed estimators used for the Ouro confirmation can be inspected and evaluated again, but the complete historical fitting process cannot be reconstructed. The Huginn pilot is retained only at the saved-output level. The preservation bundle was still on the same physical disk as the working copies at the time of verification; it was not an independent backup. [S4]

## 7. Conclusion

J-Lens loses to raw readout under the tested early-pass multihop summaries in Ouro, but recovers more annotated intermediate content in a fixed region of the final pass. That advantage transfers to new questions and concepts and survives the measured numerical variations. Its size depends strongly on the derivative target and position aggregation. The finding is about where a particular readout works—not about general superiority, successful reasoning, or causal control.

\newpage

## Appendix A. Exploratory findings kept separate from confirmation

### Exit disagreement

In the September 7 analysis of 148 prompts, native pass-3 and final predictions agreed **85.1%** of the time, with an item-bootstrap interval of **79.1–90.5%**. Two J-Lens variants targeting the current and final exits agreed only **1.45%**, interval **0.58–2.51%**, averaged over physical layers 41–47. Layer 48 was excluded from this comparison because some agreement there follows by construction. One token earlier, native agreement rose to 96.6%, while lens agreement remained 0.39%. These are different comparisons, so the lens gap cannot be read as the amount of model revision. Poor output prediction also does not by itself refute an intermediate-recovery method. [1, S1]

### Passing continuations and arithmetic

The early multihop deficit remained when the discovery items were separated by the historical continuation-pass criterion. That criterion was imperfect: two numerical false positives reduced the original 72 passes out of 148 to 70, while additionally accepting the numeric equivalent of one word-form target gave 71. Four-token continuations and lexical matching do not establish ultimate semantic correctness. The subgroup observations are exploratory and are not part of the prospective primary result. [S1]

A separate supervised arithmetic probe scored 59.4% against the raw lens’s 76.4% in pass 1, but exceeded it in pass 3, 35.6% versus 17.9%. The classifiers trained on only 36 independent operand pairs; a bounded training-size comparison was inconclusive. This used a different task family and candidate-set metric, with incomplete provenance for the older J-Lens comparator. It is not pooled with the concept-recovery results. [S1]

### Huginn pilot

The pilot used one Huginn estimator calibrated on the same 100 texts as the preselected Ouro fit, eight recurrent passes, and two evaluation initializations. Joint tokenizer eligibility retained 88 multihop and 51 arithmetic items. J-Lens-minus-raw estimates were negative under all six prespecified summaries in both initializations. The cross-model difference-of-differences had mixed signs and simultaneous intervals containing zero. The prediction of a larger relative J-Lens advantage in Huginn was not supported. [S2]

This does not isolate supervision. Huginn’s coda is trained after sampled recurrence depths, so it is not an unsupervised-intermediate control. More importantly, the fitted Huginn bank was never retrieved and its partial checkpoint was subsequently lost. The saved statistics remain historical evidence, not a fully verified cross-model replication. [S2, S4]

\newpage

## Appendix B. Verification provenance

The original confirmation freeze is recorded as unchanged. Two separately recorded corrections preceded new-item scoring: `precision_v1` removed redundant TF32 setters, and `native_check_v1` changed the development comparison from a single residual row to a whole-sequence readout. The checker’s accepted run-specification pin was updated after outcomes were known. The final verification reviewed these changes and reports that the scientific input, estimator, endpoint, and analysis were unchanged; runtime budgets also changed and were not fully covered by the correction record’s stated scope. The retrospective audit does not become part of the prospective plan. [S3, S4]

The numerical results in this draft use the accepted BF16 endpoint as primary. FP64 calculations, alternate packing, tie handling, and regenerated states are sensitivity checks, not opportunities to replace it with a preferred value. For diagonal-minus-raw, the accepted estimate is +5.95 points with a simultaneous interval of −0.34 to +12.24; regenerated states give almost the same estimate with a lower limit of +0.04. This is why its support is described as borderline. [S4]

The final report identifies the records under `research/verification_2026-09-11/`, including `FINAL_VERIFICATION.md`, `REPRODUCE.md`, and `report/REPORT_confirmation_verified.md`. The confirmation run identifier is `ouro_confirmation_20260911_fixed160_native1`. These are provenance identifiers reported by the project, not public artifact links verified for this draft.

## References

**[1]** Gurnee, W., Sofroniew, N., Pearce, A., et al. *Verbalizable Representations Form a Global Workspace in Language Models.* Transformer Circuits, Anthropic, 6 July 2026. Methods; Quantitative Methodological Comparisons; Methodological Details and Ablations. https://transformer-circuits.pub/2026/workspace/

**[2]** Zhu, R.-J., Wang, Z., Hua, K., et al. *Scaling Latent Reasoning via Looped Language Models.* arXiv:2510.25741, 2025; version 5 consulted for architectural and objective context. https://arxiv.org/abs/2510.25741

### Internal source records for this draft

**[S1]** J-Lens/Ouro post-submission follow-up, 7 September 2026. Supplied project report. Layerwise discovery analysis, exit comparisons, subgroup checks, and probe audit.

**[S2]** Ouro independent-refit and Huginn-pilot report, completed by 9 September 2026. Supplied project report from the independent-refit round. Calibration fits, estimator controls, pilot results, and retrieval incident.

**[S3]** *Ouro J-Lens prospective confirmation: full report*, 11 September 2026. Supplied project report. Frozen design, population, primary result, and secondary contrasts.

**[S4]** *Final verification report: Ouro J-Lens prospective confirmation*, 11 September 2026. Supplied project report. Numerical replay, report corrections, preservation audit, and reproducibility limits. This record supersedes earlier unmeasured numerical assurances and the description of diagonal-minus-raw as simply inconclusive.
