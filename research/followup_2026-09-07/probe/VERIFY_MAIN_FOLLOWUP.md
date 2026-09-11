# Independent exit/correctness verification

The checked exit and correctness calculations pass. I found no new scoring bug in `analyze_followup.py`. The independent script [verify_main_followup.py](verify_main_followup.py) imports no root-analysis or original-evaluator helpers. It reconstructs scores from the submitted source arrays, verifies the concept-cluster partition independently, and checks 19,382 scalar means, 3,996 interval endpoints, and 15 population/count values. Largest numerical discrepancies are 9.54e−7 in a KL mean and 2.43e−7 in an interval endpoint, from float32 reduction order.

Audited root source SHA-256: `cee07196dfb4a3de061444fd9541724f35431ed2b908e26372f74fab6363f15d`.

## Exit results

Ground truth uses the native output head on each captured loop-end residual, from the **same local evaluation** as the corresponding lenses. This matches the model's fixed-loop exit calculation: the Ouro implementation applies its RMS normalization after each loop and its output head to the selected loop state. Task continuations explicitly use `exit_at_step=3`. I checked this source relationship; this verifier does not rerun the language model or independently regenerate full logits.

Using the pod's ground truth for the laptop's lenses would be incorrect: their saved exit predictions differ on 0, 2, 2, and 4 of 148 items at exits 1–4. The root analysis correctly keeps those evaluations separate.

The saved `local` rank/KL fields mean the lens's fitted `target_ut`, not the current source state's loop. For the final J-Lens and logit lens, `local` and `final` arrays are identical because both were evaluated with `target_ut=3`. The root code correctly leaves unavailable current-exit ranks/KL blank for those methods at loops 1–3. Their current-exit **top-1 agreement** remains computable from saved token IDs.

All local-J layer-48 outputs match their actual exits exactly, with KL zero. The native logit lens also matches at those boundaries, and local J-Lens equals final J-Lens in loop 4 by definition. The 41–47 and 1–47 windows correctly remove this guaranteed boundary; the older 41–48 window is retained separately. Exit denominators are all 148 items, not the 141 concept-eligible items used for the main task comparison.

Actual exit-to-final top-1 agreement rises across loops 1–3:

| Position | Exit 1 versus 4 | Exit 2 versus 4 | Exit 3 versus 4 |
|---|---:|---:|---:|
| Last prompt token | 0.4527 | 0.7297 | 0.8514 |
| One token earlier | 0.7230 | 0.8446 | 0.9662 |

For a compact check, I also averaged loops 1–3 and physical layers 41–47 **within each item before bootstrapping**. Values below are top-1 agreement with paired item 95% intervals:

| Position | Readout | Actual current exit | Actual final exit |
|---|---|---:|---:|
| −1 | Local J-Lens | 0.2989 [0.2616, 0.3369] | 0.2413 [0.2043, 0.2799] |
| −1 | Final J-Lens | 0.0296 [0.0174, 0.0434] | 0.0283 [0.0158, 0.0425] |
| −1 | Logit lens | 0.4363 [0.3858, 0.4881] | 0.3369 [0.2886, 0.3861] |
| −2 | Local J-Lens | 0.3398 [0.2873, 0.3945] | 0.3205 [0.2648, 0.3797] |
| −2 | Final J-Lens | 0.0553 [0.0399, 0.0718] | 0.0537 [0.0386, 0.0701] |
| −2 | Logit lens | 0.4862 [0.4282, 0.5457] | 0.4369 [0.3732, 0.5013] |

In the same windows, local/final J-Lens agreement is 0.01770 [0.00965, 0.02735] at position −1 and 0.001609 [0, 0.004505] at position −2. These numbers cannot be read as direct evidence of recurrent revision: actual model outputs agree much more often. The final J-Lens is particularly poor as an exit-token predictor here. That diagnosis does not, on its own, refute a lens designed to expose verbalizable content rather than predict the actual next token.

For the report's specific **loop 3, layers 41–47, position −1** comparison against the actual final token, the paired differences are:

| Metric | Final J-Lens | Logit lens | Final J-Lens minus logit lens, paired 95% interval |
|---|---:|---:|---:|
| Top-1 agreement | 0.024131 | 0.469112 | −0.444981 [−0.511583, −0.378378] |
| Actual final token in top ten | 0.346525 | 0.745174 | −0.398649 [−0.472973, −0.322394] |

Both use the same 20,000 item draws, seed 20260907, with seven layer indicators averaged within each of the 148 items before subtracting methods. The verifier prints these contrasts with their denominator and seed.

All retained rank metrics, MRR, and KL averages reproduce for both positions and all four windows. KL is directed **lens distribution to model distribution**; actual exit-to-final KL comes from the native head at the loop boundary. The local/final lens KL means reproduce as 4.7711, 4.0976, 6.2995 at −1 and 5.2394, 4.6309, 3.9110 at −2. Full probability vectors/top-k sets are not retained, so rank of the actual top-1 token is verifiable, but a genuine top-k set-overlap statistic cannot be reconstructed from these artifacts.

## Correctness results

The historical criterion has 72 passing continuations among all 148. Eligible main-comparison strata are **36 passing / 54 failing multihop items** and **31 / 20 arithmetic items**. All method means and paired differences reproduce under all three criteria, including the full layerwise means. Controls receive their own maximum over layers before averaging, and multiple own labels are averaged within an item before items are combined.

The stricter numeric boundary rule removes exactly `div-sub-left` (`3.8` versus target `3`) and `nested4-add-mult-add-div` (`3.75` versus `3`): 70/148 pass, with arithmetic strata 29/22. Narrow numeric equivalence additionally accepts `word-parens` (`20.` versus target `twenty`): 71/148 pass, arithmetic 30/21. These are correctly described as scoring sensitivities, not comprehensive semantic judgments.

The bootstrap preserves method pairing within each item. For subgroup interactions, item draws independently resample passing and failing items. The shared-concept sensitivity uses the same sampled concept clusters in both strata, normalizing each stratum's item-weighted mean separately; this correctly retains covariance when a concept occurs in both outcome groups. I independently reconstructed the connected components and reproduced the reported cluster intervals.

Interpretation should remain restrained:

- J-Lens loses at the first three multihop loops in both outcome strata. The negative result is not produced solely by failed model trajectories.
- At multihop loop 3, the J-minus-logit effect is more negative on passing items. The interaction is −0.2575, item interval [−0.4577, −0.0457], concept-cluster interval [−0.4457, −0.0430]. This is one unadjusted exploratory interaction among multiple task/loop comparisons; it is not a general success-monitoring result.
- The arithmetic loop-2 interaction is +0.2498 with item interval [+0.0283, +0.4779], but its concept-cluster interval [−0.0415, +0.5145] includes zero. Under narrow numeric equivalence, even the item interval includes zero: +0.1886 [−0.0355, +0.4165]. A reliable arithmetic success/failure interaction is unresolved.
- Readability differences between passing and failing prompts are associations across different items. They cannot identify whether readable intermediates cause success, or establish an advantage over chain-of-thought monitoring.

## Verification limitations and encountered issues

No new root metric error was found. The already identified decimal-prefix bug and `target_ut` naming trap are real; the follow-up handles both explicitly. Tiny confidence-bound excursions beyond 1 in native identity comparisons are floating-point rounding, not a statistical effect.

I also read the completed `REPORT.md`. Its exit, correctness, and probe values match the checked results, and it appropriately distinguishes exit prediction from J-Lens's verbalizability objective and exploratory subgroup associations from general success claims.

Checks are conditional on the retained fitted lenses and model outputs; they do not assess fit-to-fit variance, rerun the model, or prove that concept clusters remove every dependence among synthetic prompts. Full layerwise interval verification was assigned separately; this script verifies their subgroup means and all any-layer subgroup intervals, rather than duplicating that work.

The verifier initially assumed the saved layer JSON placed tasks at its root. It stopped with a `KeyError`, before checking correctness results. I corrected the lookup to the actual `tasks` container and reran successfully. This was a verifier input-schema mistake, not an analysis discrepancy; no root code or input artifact was edited.
