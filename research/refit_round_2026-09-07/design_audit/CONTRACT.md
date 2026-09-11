# Independent refit and estimator contract

The existing negative result is not just a last-layer artifact. I read the full current report, inspected `layerwise.json` and `item_scores.npz`, and independently recovered the any-layer differences. Multihop mean fixed-layer differences are −0.0530, −0.1015, −0.1314 and +0.0339 across loops; the largest positive old cell is loop 4, physical layer 32, +0.2411. New fits should test whether this shape survives calibration variation, not choose another favorable layer.

This contract follows the actual code and the cached original paper, particularly its Methods and Methodological Details/Pseudocode sections. The released estimator matches the paper's pseudocode. Penultimate targeting is a documented Sonnet variant, not a correction to a proven Ouro implementation error.

## Exact estimators and the inexpensive matched control

Let `t` be target virtual layer 191 (final) or 190 (penultimate), and let `G_l(q,p)` denote the d×d Jacobian of target residual at token q with respect to source residual at token p. Rows index output dimensions. For encoded sequence length T, preserve the current valid set `V={16,…,T−2}` and let m=|V|. BOS counts in these indices. The existing estimator is

`J_l^sum = (1/m) sum_{p in V} sum_{q in V} G_l(q,p)`.

Autodiff with a dimension-one-hot cotangent at **all** valid target tokens computes the inner target sum. It then averages source-token gradients. Causality makes q<p terms zero. This is not a uniform mean over the m(m+1)/2 causal position pairs. Prompt matrices are averaged with equal prompt weights, not weighted by m.

For each prompt, draw **one q uniformly from V** before examining any readout. A single-q VJP returns `g_q[p,:]=G_l(q,p)[output_dimension,:]` at every source token. The same derivative gives two full-matrix estimates:

`S_l(q) = sum_{p in V} G_l(q,p)`

`D_l(q) = G_l(q,q)`.

Then `E_q S_l(q)=J_l^sum`, while `E_q D_l(q)=(1/m)sum_p G_l(p,p)`, the mean same-position Jacobian. Therefore **one shared VJP per output dimension produces both arms**. Each arm uses identical prompts, sampled q values, and all 2,048 output dimensions. Both have exactly the same derivative budget; no stochastic dimension sketch is introduced.

Normalization is easy to get wrong. Sum the source gradients directly. If averaging all m source positions, multiply by m. If averaging only valid p≤q, multiply by their count `k_q=q−16+1`, not by m. Leaving the mean unscaled changes the estimand and, when m varies between prompts, changes their relative weights. Do not introduce the triangular-pair denominator. Retain m, q, k_q, and both per-prompt diagnostics in the ledger.

This is a **sampled-position sum estimator**, not the exact dense estimator from the old N=100 run. The two have the same matrix expectation, but downstream rank accuracy is nonlinear in the estimated matrix. Equal derivative budgets do not make their finite-sample noise identical. Compare sampled-S with the dense-S reference too, and do not attribute a loss caused by additional position sampling to the diagonal reduction.

The historical last-valid-position 2×2 proposal is a different control: target-last/source-mean is not a mean diagonal Jacobian. Also, at the last valid source token, all-valid-target and target-last gradients are identical by causality. Those cells are useful checks but must not be advertised as independent evidence about full-position same-token readout.

## Frozen experiment structure

1. Complete the independently sampled **dense final-target** N=100 fits first. Five fits is a defensible bounded replication block if the benchmark permits it; freeze the count before seeing outcomes. A sixth fit is helpful only if decided from compute cost, not because five disagree.
2. For the target control, compare dense final versus dense penultimate on exactly the same N=100 calibration sets. This changes only the final transformer block in the derivative target. Reuse final dense fits from step 1. Keep the final normalization/unembedding unchanged in both arms; forwarding the penultimate estimate through the final block would undo the intended ablation.
3. For the position control, compute sampled-S and D from the shared single-q VJP above. Use paired q values for final and penultimate targets if both targets are included. The smallest isolated position test needs only the final target; adding penultimate positions is a two-target factorial extension, not required to interpret the first contrast. Freeze which block is affordable before fitting.

All comparison arms within a replicate use the same corpus and tokenization. If compute permits all cells together, a common forward can feed four VJP families: dense-final, dense-penultimate, sampled-q-final, sampled-q-penultimate. The last two each yield both S and D for free. That produces six maps at **four dense-equivalent derivative budgets**, not six. Backpropagating a tuple of both target tensors with nonzero cotangents in one sample **adds** their gradients; it does not recover separate target maps. Separate calls, or separate replicated batch lanes, are required. Free sharing is across source reductions, not across distinguishable target outputs.

Both targets support the common strict source set **0…189**, corresponding to all 48 physical layers in loops 1–3 and layers 1–46 in loop 4. Preserve the full final-target source set 0…190 plus its known self-boundary for historical replication, but restrict paired target contrasts to common supported sources. Never pad penultimate-tail positions 190/191 with identities and count them as learned penultimate readouts. The current fitting wrapper resolves only loop-end targets, and the old evaluator's completion logic inserts identity matrices after a target; both assumptions need explicit handling in the new driver and metadata.

Keep the old eligible items and control averaging. Report full supported depth curves. Freeze summaries as the old any-layer statistic for dense-final replication; first-three-loop middle bands 17–32; and the common-support final-pass late band 33–46 for target contrasts. The old 26–37 positive region can be a named replication region, clearly identified as discovered in the old data. Do not reselect its endpoints or the largest new cell. Recompute the old dense reference over 33–46 when showing that new common-support band.

## Independence means new data draws, not new initialization

J-Lens has no fitted random initialization, optimizer, or learned starting point. `HFLensModel` sets evaluation mode and freezes model parameters. Ouro initializes its recurrent computation from token embeddings, not a random latent state. Repeating a fit on identical prompts, order, and deterministic kernels should reproduce the same map; changing a seed alone does not create a new fit sample.

Calibration-set sampling and, for the position control, sampled q values are the intended randomness. Pin model/tokenizer/corpus/source hashes, attention backend, dtype, dimension batch, thread count, and sequence mask. Independently sample N unique calibration records within each replicate from a declared pool using independent seeds. Overlap between independently drawn sets is allowed, but must be reported; nested prefixes are not independent replicates. If disjoint blocks are chosen instead, describe them as a random partition of a finite pool, with its dependence, not iid corpus samples. Shuffle before partitioning: the retained WikiText file is the first 1,200 qualifying records in source order, and adjacent records may share articles.

Exclude the old 100 calibration records if the claim is specifically new-data replication. Hash **encoded truncated token sequences**, not only full strings, to detect effective duplicates. The retained filtered file does not establish article independence. Fit uncertainty from this pool does not establish robustness to a new corpus distribution. Cache reuse must not transfer accumulated maps between fits; cached deterministic per-prompt derivatives are legitimate, but sampled-q cache keys must include q.

## Numerical acceptance before scientific evaluation

Use a bounded verification ladder, not a blanket test suite:

- A tiny causal float64 model with 3–5 tokens and width 3–4: explicitly materialize every G(q,p). Enumerating every valid q must make mean(S(q)) equal dense stock J and mean(D(q)) equal the diagonal mean, within `1e−10` absolute and relative tolerance. Check exclusions, row orientation, causality, and target/source off-by-one indices. This can falsify all normalization errors above.
- On one fixed short Ouro prompt, compare the new dense path against released `jacobian_for_prompt` for a small fixed output-row set at source virtual layers 0,47,48,95,96,143,144,189 and both targets. Same dtype, backend and batch configuration should produce matching gradients; record max absolute and relative Frobenius errors. Establish any mixed-precision tolerance from duplicate numerical runs before looking at task scores. Never loosen it to admit a favorable result.
- Check batch lanes independently: dimension batch 1 versus the production batch size, identical row indices and q. Each source hook must fire once at its specified `current_ut`; recurrence carries include the RMS normalization between passes. No hooks or `requires_grad` state should leak after the recorder exits.
- For finite differences, choose one source token p and unit directions v (source) and u (target), replace only that source activation by `h±εv`, and compare `[u·z_q(h+εv)−u·z_q(h−εv)]/(2ε)` with `uᵀG(q,p)v`. Use p=q and p<q, plus the forbidden p>q causal zero. Use FP32 for the numerical reference; BF16 finite differences can be dominated by quantization. Two adjacent steps in a predeclared short ε ladder must agree with AD to within 2% for non-negligible derivatives (plus a stated absolute tolerance). If full FP32 does not fit, a copied last-two-block FP32 suffix can verify the target change, while the tiny explicit model verifies full estimator algebra. Label unperformed deep-model finite differences as unverified, not passed.

For target=source, the derivative is exactly identity as a plumbing check; it remains excluded from learned-target comparisons. Dense and sampled map accumulation must reject NaN/Inf and skipped prompts, retaining the intended N exactly. Store sums, counts, per-prompt norms and actual q values; do not replace real per-prompt statistics with modeled scatter or silently discard outliers.

## Two-way uncertainty with few fits

Let `d[r,i,k]` be an item-level paired method/condition contrast at frozen cell or summary k. Average labels and controls within item first, and compute each lens's any-layer maximum before averaging across fits. In particular, neither a maximum of mean layer scores nor evaluation of the averaged lens is the mean performance of independent lenses.

Use 20,000 crossed bootstrap draws. Independently resample R fit **blocks** and N items; use the same sampled fit indices and item indices for every compared method, target, reduction, loop, and layer. A fit block contains the paired calibration set and all its condition maps. The logit lens is the fixed per-item baseline, not R independent measurements. Compute each bootstrap mean from the product of fit and item multiplicities, divided by their corresponding total weight.

For shared-concept sensitivity, replace the item draw by a draw of G concept components. An item's weight is its component's multiplicity, and the denominator is the resulting number of weighted items, preserving the original item-weighted estimand. Use the same sampled components for paired conditions and, if stratifying correctness, both strata. Fit sampling remains independent of concept sampling.

These are approximate crossed/pigeonhole bootstrap intervals, not a finite-sample coverage guarantee. With only five or six fits, the empirical fit distribution has very little support. Report every fit's effect, between-fit SD/range, and leave-one-fit-out means beside any two-way interval. Also show item-only uncertainty conditional on the retained fits. With one fit, fit variance is unidentified; with two or three, avoid treating percentile endpoints as calibrated 95% evidence about a fit population. Five same-sign fits are useful replication evidence, not proof that an unseen fit cannot reverse the sign.

Do not count prompts used to fit a map, source positions, layers, or repeated fit×item cells as independent evaluation units. Additional sampled-q randomness is part of the fit-block variability for the matched position experiment; it is not optimizer initialization variance. Comparing the paired dense and sampled-S maps reveals how much that additional estimator noise changes the outcome.

No new model fit or blanket test run was performed in this design audit.
