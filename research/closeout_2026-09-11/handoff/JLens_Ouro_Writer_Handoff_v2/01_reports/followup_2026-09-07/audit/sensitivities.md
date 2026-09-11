# Bounded metric and dependence sensitivities — 7 September 2026

## Choices recorded before execution, 17:49 UTC

The historical metric and submitted results remain unchanged. Run exactly three
sensitivity analyses using the retained pod N=100 ranks:

1. Change controls only: merge canonical number-word/digit aliases (including
   `third=3`) and case variants, score each resulting control concept by the
   minimum rank over its original names, and exclude every own canonical
   concept. Preserve each true intermediate name's original rank and the
   original within-item averaging. Use the original 90/51 items.
2. Remove only `super-populous-capital` and `firstletter-populous-country` from
   the evaluated items. Keep targets, intermediates, and the historical control
   vocabulary unchanged. This gives 88/51 items.
3. Keep the historical metric and original 90/51 items; resample released-name
   prefix clusters from the previously frozen metadata. Draw the original
   number of clusters with replacement, include all members, and divide by the
   sampled item count, preserving item weighting.

Use 20,000 paired bootstrap draws, seed 0, for each table. Report all four loops
and both tasks. For (1) and (2), retain all 384 paired layerwise mean changes and
their maximum absolute change; do not search for a new favorable layer. Check
whether the earlier multihop deficits and the already observed final-pass
positive window at physical layers 24–39 retain their signs, keeping those
bounds fixed. This window is descriptive and was identified in the original
follow-up curve, not prospectively predicted.

For (3), also report the fixed middle block (17–32) and late block (33–48) with
prefix-cluster intervals. These are pointwise sensitivity intervals, not
multiplicity-adjusted claims. Do not combine the three changes or run further
variants. Results will be appended after execution.

## Results

Each cell is J-Lens minus logit lens in any-layer excess hit@10, followed by its paired 95% interval. The first two sensitivities use item resampling; the third changes only the resampling unit. Intervals are pointwise.

| Sensitivity | Task | n items / clusters | Loop 1 | Loop 2 | Loop 3 | Loop 4 |
|---|---|---:|---|---|---|---|
| alias_controls | multihop | 90 | -0.376 [-0.484, -0.267] | -0.487 [-0.588, -0.384] | -0.415 [-0.520, -0.310] | -0.021 [-0.096, +0.056] |
| alias_controls | order-ops | 51 | -0.255 [-0.362, -0.139] | +0.140 [+0.020, +0.264] | +0.089 [-0.045, +0.226] | -0.021 [-0.072, +0.024] |
| exclude_stale_two | multihop | 88 | -0.363 [-0.474, -0.253] | -0.478 [-0.580, -0.374] | -0.405 [-0.510, -0.300] | -0.034 [-0.109, +0.039] |
| exclude_stale_two | order-ops | 51 | -0.255 [-0.362, -0.139] | +0.140 [+0.020, +0.264] | +0.089 [-0.045, +0.226] | -0.021 [-0.072, +0.024] |
| prefix_clusters | multihop | 90 / 36 | -0.356 [-0.535, -0.191] | -0.479 [-0.660, -0.307] | -0.407 [-0.584, -0.245] | -0.022 [-0.118, +0.061] |
| prefix_clusters | order-ops | 51 / 15 | -0.255 [-0.370, -0.117] | +0.140 [+0.009, +0.288] | +0.089 [-0.073, +0.269] | -0.021 [-0.083, +0.026] |

Changing controls or dropping stale tasks did not reverse the early multihop deficits. Under each change all three early-loop any-layer intervals remain below zero. The fixed loop-4 multihop window at physical layers 24–39 retains a positive mean at each of its 16 layers. This is a sign/shape robustness check, not a new significance claim about that selected window.

| Sensitivity | Task | Maximum absolute layer-effect change | Loop 4 layers 24–39 mean effect | All 16 positive? |
|---|---|---:|---:|---|
| alias_controls | multihop | 0.021965 | +0.154384 | True |
| alias_controls | order-ops | 0.000000 | +0.014517 | False |
| exclude_stale_two | multihop | 0.011917 | +0.149432 | True |
| exclude_stale_two | order-ops | 0.000000 | +0.014517 | False |

All 384 paired layer effects and changes for each of the two metric/population sensitivities, plus changes in each method separately, are retained in `sensitivities.json`. Arithmetic has no change under either sensitivity: its names have no canonical aliases and neither removed item is arithmetic.

Fixed depth blocks under the historical metric, with paired name-prefix-cluster intervals:

| Task | Layers | Clusters | Loop 1 | Loop 2 | Loop 3 | Loop 4 |
|---|---|---:|---|---|---|---|
| multihop | 17-32 | 36 | -0.071 [-0.150, -0.020] | -0.114 [-0.216, -0.045] | -0.154 [-0.276, -0.066] | +0.062 [-0.017, +0.150] |
| multihop | 33-48 | 36 | -0.085 [-0.146, -0.034] | -0.157 [-0.243, -0.086] | -0.172 [-0.266, -0.094] | +0.056 [+0.012, +0.115] |
| order-ops | 17-32 | 15 | -0.363 [-0.502, -0.241] | +0.042 [-0.034, +0.125] | +0.086 [+0.005, +0.183] | +0.043 [-0.042, +0.122] |
| order-ops | 33-48 | 15 | -0.341 [-0.460, -0.238] | +0.062 [-0.021, +0.150] | -0.088 [-0.140, -0.027] | -0.101 [-0.157, -0.055] |

The three checks preserve the descriptive interpretation: a substantial early multihop deficit and a later region of improved relative J-Lens readability. They do not establish a general early-depth advantage, causal use of the labeled concepts, or an explanation involving supervision. Prefix clustering is another imperfect dependence model; the table reports uncertainty under it rather than replacing the original estimand.

Execution checks: alias-aware scoring preserved every true-name hit and any-layer hit; no source artifact was written. The maximum is taken for each control concept before averaging controls in the any-layer statistic. No sensitivities were combined.
