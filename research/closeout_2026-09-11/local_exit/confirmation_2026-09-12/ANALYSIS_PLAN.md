# Analysis plan: own-exit versus final-target J-Lens on the confirmation population

Frozen 2026-09-12 before any readout of the confirmation states with the initial-study banks was computed. This executes Steps 1 and 2 of the bounded proposal in `../LOCAL_EXIT_STATUS.md` (Section 3), at the author's direction, with the author's Hub credential. No lens is fitted; no paid resource is used.

## Data
- States: the accepted confirmation payload cache (`results/common/cache.pt`, sha256 `e78ba9da…`, identical in all seven retained copies), 160 items, 192 virtual layers, plus the stored exit logits. Same tensors the primary endpoint was scored from.
- Population, aliases, controls, dependency groups: the accepted payload `population.json` (own index, 79 control indices, 28 dependency groups), unchanged.
- Existing arms reused from the accepted readouts: `raw.npz`, `fit01.npz` (the restoration test reproduced both bit for bit from these states).
- New banks: initial-study family `n100/exit0` (target virtual 47), `exit1` (95), `exit2` (143), `exit3` (191), each a count-weighted mean of its retrieved shards (`jlens.JacobianLens.merge`, the frozen reference procedure). Every shard is verified against the sha256 recorded in the inventory before use. Equivalence check of the merge procedure: my five-shard merge of exit3 is compared tensor by tensor with the surviving pod merge (`d7c26297…`).

## Readout
Frozen scorer `readouts.readout` with `prepare_readout` from the frozen code (FP32 J·h, BF16 native unembedding with the model's final RMS norm, min over accepted single-token forms, full-vocabulary rank). Each own-exit bank is applied only to the virtual layers it was fitted for (0..target−1), so pass p's band uses bank exit(p−1). exit3 is applied to all 192 layers with the identity at 191, exactly as fit01 was.

## Quantities
Fixed band: physical layers 26–37 of each pass, virtual 48(p−1)+25 … 48(p−1)+36. Per item: intended hit@10 averaged over the 12 layers, control hit@10 averaged over the 79 controls and the 12 layers, excess = intended − control. Paired differences are item-wise means of per-item differences.

## Inference (one family, fixed here)
Six contrasts of excess hit@10 in the band, passes 1–3:
- own-exit(p) − final-target(exit3), p = 1, 2, 3;
- own-exit(p) − raw, p = 1, 2, 3.
Whole-dependency-group bootstrap with the frozen `analyze.resample` (20,000 draws), seed **2026091201**, centered max-t simultaneous 95% intervals over the six contrasts. A contrast is called supported only if its simultaneous interval excludes zero.

## Descriptive (no inference)
Own-exit(p) − fit01 (confirmation-era final-target bank; different calibration, so this does not isolate the target); intended and control components of every arm; per-layer excess curves for passes 1–3; exit3 versus fit01 and versus raw in the pass-4 band (initial-study final-target bank on the confirmation population); group-percentile intervals for descriptive quantities with the same draws.

## Checks that must pass before results are reported
1. All 13 shard files match their recorded sha256 and byte counts.
2. exit3 merged from shards equals the pod merge within FP16 rounding (report max absolute difference and the fraction of exactly equal entries).
3. Reusing the frozen scorer on these states with the fit01 bank reproduces the accepted `fit01.npz` ranks bit for bit (sanity check of the environment; the restoration test already did this on the SSD copy).
4. Pass-4 identity: with exit3, band 169–180 readouts are the bank's own-exit readouts by construction; no separate own-exit arm exists for pass 4.

## Labeling
A post-confirmation analysis on the already-used confirmation population with an estimator family (initial-study recipe: 12-article calibration prefix, dim_batch 32, B300 pod) that differs from the confirmation fit01 family. It is matched within its own family, so own-exit − exit3 isolates the target; own-exit − raw does not depend on any fit01 quantity. It does not change the primary endpoint or any prospectively registered result. Whatever the outcome, it is reported.
