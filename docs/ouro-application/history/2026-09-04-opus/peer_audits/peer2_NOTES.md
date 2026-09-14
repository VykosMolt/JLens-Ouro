# PEER2 notes — inference, metrics, write-up (C3–C8)

Working dir /home/moloch/ouro_project. Interpreter `venv/bin/python`.
Independent re-implementation of the metric (no `analyze.py` import) lives at
`/tmp/claude-1000/-home-moloch-ouro-project/6c2195c9-8eae-43c2-bdf6-39cd670c7b37/scratchpad/p2/recompute.py`.
RESULTS.md snapshot used for number-checking: `.../scratchpad/p2/RESULTS_snapshot_0005.md`
(MAIN is editing the live file; section numbers shifted by +1 after it added a
"Relation to prior work" section, so section names below are by content).

## 1. Number check against artifacts — status so far

All numbers below recomputed from `artifacts/jlens/eval/round1_exit3x32/{arrays.npz,items.json,task_names.json}`.

MATCH (exact, to the printed precision):
- lens-free exits: KL mean [3.828,0.647,0.086,0], median [4.081,0.220,0.031,0];
  exit top-1==final multihop [0.39,0.67,0.80,1.0], arithmetic [0.49,0.843,0.961,1.0];
  intermediate-is-exit-top-1 arithmetic [0.176,0.078,0.059,0.039]. (C2)
- any-layer excess table: multihop J [0.004,-0.010,0.101,0.481] / logit [0.380,0.478,0.523,0.505];
  arithmetic J [0.069,0.279,0.205,0.317] / logit [0.471,0.330,0.259,0.363]. (C3)
- local vs eventual mean-over-48-layers excess, all 18 cells of the section-5 table:
  multihop local [0.065,0.107,0.167], eventual [-0.000,-0.007,0.016], logit [0.053,0.095,0.139];
  arithmetic local [0.152,0.053,0.051], eventual [0.006,0.096,0.087], logit [0.236,0.118,0.094]. (C3)
- top-1 agreement local vs eventual, last 8 layers: 0.0017 / 0.0177 / 0.0084 -> "0.002, 0.018, 0.008". (C3)
- fit-size sweep, all 8 rows, matches the *current* `fitsize_n*/summary.json`. (C5)
- cross-loop matrices, variance shares (multihop 0.986/0.011/0.003 -> 0.99/0.01/0.00;
  arithmetic 0.330/0.157/0.513 -> 0.33/0.16/0.51) and all six contrast point estimates. (C6)
- probe/lens per-loop maxima [0.665,0.653,0.341,0.330] / [0.892,0.619,0.176,0.403] /
  [0.551,0.562,0.222,0.148]. (C8)

MISMATCHES / STALE:
- **M1 (stale artifact, referenced by the write-up).** `artifacts/jlens/eval/fitsize_summary.md`
  and `artifacts/jlens/eval/fig6_fit_size.png` (both mtime Sep 3 08:37) predate the
  Sep 3 23:46 re-run of `analyze.py` on the fitsize dirs. They give multihop
  [0.004,0.014,0.051,0.435] / [0.013,-0.001,0.096,0.437] / [0.003,-0.021,0.092,0.426] /
  [0.006,-0.020,0.090,0.426], against the current summary.json values
  [0.005,0.017,0.064,0.479] / [0.003,-0.011,0.100,0.481] / [0.003,-0.023,0.094,0.469] /
  [0.007,-0.021,0.093,0.469] that RESULTS.md reports. RESULTS.md is right and the
  figure it points a reader at is wrong (loop-4 multihop 0.435 vs 0.479).
  Fix: rerun `venv/bin/python src/ouro_jlens/fitsize_report.py`.
- **M2 (minor).** "Pearson r 0.43 to 0.77" for local-vs-logit per-layer excess: the actual
  range over the six (task, loop) cells is 0.453 to 0.765. Lower bound should read 0.45.
  Mean-absolute-difference "0.05 to 0.10" is actually 0.049 to 0.089.
- **M3.** "the paper uses 1000 and calls ~100 usable" and other prose numbers not checked here.

## 2. The metric — is "excess hit@10 over matched controls" fair?

### 2a. Per-layer excess: fair. Label-shuffle null agrees.
The stated null is "the other intermediate names of the same task and same kind".
I built a proper label-shuffle null instead: reassign each scored slot a name drawn
uniformly from the *own-names actually used by other slots of the same task and kind*
(rejecting names the item owns), 1000 draws, aggregated by item exactly as the real
statistic is. This weights control names by how often they really occur as answers,
which the analyze.py control does not.

    (script: scratchpad/p2, permutation null on mean-over-layers hit@10)

Result: the C3 numbers barely move.
  multihop  local  loops1-3  ctrl-excess +0.065/+0.107/+0.167  perm-excess +0.062/+0.104/+0.162 (p<0.001)
  multihop  event. loops1-4  -0.000/-0.007/+0.016/+0.191       -0.002/-0.012/+0.011/+0.184
                                                                p=0.95 / 0.98 / 0.11 / <0.001
  multihop  logit  loops1-4  +0.053/+0.095/+0.139/+0.166       +0.049/+0.091/+0.135/+0.162 (all p<0.001)
  arith.    event. loops1-4  +0.006/+0.096/+0.087/+0.092       +0.003/+0.094/+0.078/+0.083
  arith.    logit  loops1-4  +0.236/+0.118/+0.094/+0.134       +0.220/+0.105/+0.077/+0.116
So for the *per-layer* metric the matched control is not smuggling anything, and the
eventual-exit lens's loop-1/2 nulls are confirmed null by a permutation test (p=0.95, 0.98).
C3 survives this attack.

### 2b. The ANY-LAYER ("pass@10") control is biased low. Real defect.
`analyze.py:any_layer` and `cross_loop_summary` compute the control as
`max_over_layers( mean_over_control_names( rank<10 ) )` while the own score is
`max_over_layers( rank<10 )` and is only then averaged. Max of a mean <= mean of maxes
(Jensen), so the control is systematically too small and every any-layer excess is
inflated. Scoring the control the same way the own name is scored
(`mean_over_names( max_over_layers( rank<10 ) )`) gives:

| task | lens | analyze.py excess | like-for-like excess |
|---|---|---|---|
| multihop | eventual J | [0.004,-0.010,0.101,0.481] | [-0.002,-0.014,0.089,0.464] |
| multihop | logit | [0.380,0.478,0.523,0.505] | [0.358,0.458,0.500,0.483] |
| arithmetic | eventual J | [0.069,0.279,0.205,0.317] | [0.059,0.247,0.184,**0.172**] |
| arithmetic | logit | [0.471,0.330,0.259,0.363] | [**0.306**,**0.152**,**0.131**,**0.213**] |

Multihop conclusions are unaffected in kind. Arithmetic is materially overstated: the
loop-4 eventual-exit excess nearly halves (0.317 -> 0.172) and every logit-lens
arithmetic number drops by a third to a half. The arithmetic rows of the any-layer
table (and any prose derived from them) should be restated.
Proposed diff to `analyze.py` (MAIN owns the file):
    def per_loop(v):  # in any_layer()
        return by_item(v.reshape(len(v), N_UT, N_LAYER).max(2), sc["items"]).mean(0)...
  -> the `control` passed in must be per-control-name, not the pre-averaged mean.
  i.e. `scores()` should also return `ctrl_any` computed as
     `(allrank[i, others] < K).reshape(nc, N_UT, N_LAYER).max(3).mean(0)`.
The per-layer (`loc_maps`) path is unaffected because mean and threshold commute there.

---

## 3. REPLY TO MAIN (00:30) — your fix is correct; your C4 retraction is over-correction

### 3a. analyze.py implementation: VERIFIED CORRECT.
Read the new `scores` / `_split_layers` / `_any_layer_per_name` / `_drop_layer_axis` /
`any_layer` / `cross_loop_summary`. Shape handling is right on both paths:
- `_split_layers` keys off `shape[-1] == 192`; the cross-loop tensor's trailing axis is 48
  so it is left alone and `.max(-1)` collapses the layer axis of `[n_ctrl, 4, 4, 48]`
  to `[n_ctrl, 4, 4]`. Correct.
- `_drop_layer_axis` returns `(slots, 4)` for the 192 case and `(slots, 4, 4)` for the
  cross-loop case. Correct. (No ambiguity because 192 != 48; a comment saying so would
  be worth one line if N_LAYER ever changes.)
- `per_loop`'s `v[:, None] ... [:, 0]` trick is right but obscure; consider
  `_split_layers(v).max(-1)` directly.
- `loc_maps` correctly skips `control_any`; the per-layer path is untouched.
Numeric check: every number in the regenerated `round1_exit3x32/summary.md` equals the
values I computed independently before you changed anything (multihop control_any
[0.035,0.036,0.056,0.092], arithmetic [0.196,0.204,0.169,0.632], cross-loop
-0.110/-0.045/-0.174 and 0.98/0.01/0.00, arithmetic +0.005/+0.011/-0.002 and
0.10/0.23/0.67). Main-table any-layer excess now equals the cross-loop diagonal to 3dp
on both tasks. Fix accepted.

### 3b. C4: do NOT retract it outright. WEAKENED, not FALSE.
You are reading two point estimates without their uncertainty. Paired bootstrap over
items, 5000 draws, of (eventual-exit J-lens minus logit lens):

  arithmetic, corrected any-layer excess, J - LL per loop:
      [-0.247, +0.095, +0.053, -0.041]
      CI95 lo [-0.377, -0.041, -0.083, -0.115]
      CI95 hi [-0.116, +0.228, +0.183, +0.015]
  -> the loops-2/3 reversals do not exclude zero. They are point estimates inside noise
     on 51 items.

  arithmetic, candidate-set top-1 any layer (NO control at all, so immune to this whole
  Jensen problem), J - LL per loop:
      [-0.098, -0.078, -0.157, -0.196]  CI95 hi [+0.039, +0.078, 0.000, -0.078]
  -> the logit lens is ahead at every loop on the control-free metric.
  multihop, same metric: [-0.456, -0.559, -0.504, -0.115], all four CIs exclude zero.

  Per-loop MAXIMUM over layers of excess and of cand-top-1: the J-lens does not exceed
  the logit lens at any loop on either task under either metric.

The location-by-location statement was never true and should not have been claimed:
the J-lens beats the logit lens at 72/192 individual locations on arithmetic (22 of the
48 layers of loop 2) even before the fix.

Honest wording: "the eventual-exit lens does not beat the logit lens at any loop on
multihop (all four CIs exclude zero) nor on any per-loop aggregate on arithmetic; on
arithmetic loops 2-3 the corrected any-layer excess point estimate favours the J-lens,
but the difference is within noise on 51 items and reverses on the control-free
candidate-set metric." Retracting C4 to FALSE overstates the evidence in the other
direction, and the note that the arithmetic control is near ceiling (65% of control
names hit@10 under the logit lens at loop 2) belongs next to it: excess is a poor
discriminator there.

### 3c. Timing for the CV probe: NOT a schedule risk.
Benchmarked `LogisticRegression(max_iter=2000)` on the real cached features: 0.03-0.10 s
per fit (lbfgs converges in 13-42 iterations, nowhere near max_iter). 192 locations x
5 folds x (3 C values + 1 refit) ~ 3800 fits ~ 8-12 min CPU with OMP_NUM_THREADS=6.
The GPU part (scoring both lenses on all 648 prompts instead of 176) is ~15 min
including model + lens load. Do not cut it.

---

## 4. C5 — FIT SIZE. WEAKENED. hit@10 is floored; the ranks underneath are not flat.

The write-up infers "the lens has converged / the deficit is not a sampling artifact"
from a flat thresholded statistic. At loops 1-2 that statistic sits at 0.01-0.04, i.e.
close to its floor, so flatness is weak evidence. Looking under it:

Median rank of the true intermediate under the eventual-exit lens, best layer within the
loop, per fit size (multihop, 90 items):
  loop 1: 767 -> 443 -> 412 -> 566   (n = 8, 32, 56, 80)
  loop 2: 896 -> 545 -> 547 -> 402
  loop 3: 230 -> 132 -> 128 ->  96      hit@100: 0.37 -> 0.44 -> 0.43 -> 0.51
  loop 4:   8 ->   8 ->  10 ->  10      (already converged)

Paired bootstrap over items (the same 148 items in the same order in all four dirs),
5000 draws, of the improvement from n=8 to n=80:
  mean d log10(rank+1), positive = better with more data
    multihop   loop1 +0.096 CI [-0.028,+0.219]   loop2 +0.227 [+0.142,+0.310]
               loop3 +0.244 [+0.135,+0.354]      loop4 +0.019 [-0.013,+0.053]
    arithmetic loop1 +0.308 [+0.209,+0.407]      loop2 +0.387 [+0.272,+0.505]
               loop3 +0.315 [+0.215,+0.413]      loop4 +0.038 [-0.011,+0.090]
  AUC(own rank beats control rank), monotone in n at loops 2-4 on both tasks;
    arithmetic loop2 0.668->0.707, loop3 0.672->0.705, loop4 0.667->0.696, all CIs
    excluding zero for the n80-n8 difference.

So: 6 of 8 (task, loop) cells show a statistically significant rank improvement over the
8->80 range, at ranks far above the hit@10 threshold, while hit@10 does not move. Loop 4
is flat at the rank level too, exactly as a converged lens should be. That pattern is the
opposite of "converged by eight prompts": it says the lens is converged where its horizon
is short and is still improving where it is long, which is where the claim lives.

What survives: **multihop loop 1** is genuinely flat at BOTH levels (rank CI includes
zero, hit@10 zero throughout), so the loop-1 multihop blind spot specifically is not a
sampling artifact over this range. Everything else should be stated as "hit@10 does not
move over 8-80 prompts, though the underlying ranks do improve, so this range does not
bound what a 1000-prompt fit would do." The 1000-prompt rented run is therefore not
merely "extending the curve", as section 10 says: it is the test.
Caveat on the sweep design: the four fits are nested prefixes of one prompt pool
(n8 subset of n32 subset of n56 subset of n80), so the four points are strongly
correlated and the sweep understates run-to-run variability at fixed n.

## 5. C6 — CROSS-LOOP. Decomposition correct; HOLDS on multihop; state effect does NOT
   survive an honest multiple-comparison correction.

Checked line by line:
- variance share: standard one-observation-per-cell two-way decomposition of the 4x4 item
  -mean matrix; ss_row + ss_col + ss_int = ss_tot exactly. Computed correctly.
- contrast balance: all three are balanced. `state_only` compares {fit in 2,3,4 x state 1}
  against the 6 off-diagonal cells among loops 2-4, and each fit loop appears with weight
  1/3 on both sides. `fit_only` is the mirror, balanced on state loop. `h1_preregistered`
  is 6 cells vs the same 6. No confound between the two factors.
- bootstrap unit: `boot_ci` resamples rows of `ex_items` = items. Correct unit.
- rank readouts are scale-invariant (jlens `unembed` applies the final RMSNorm before the
  head), so the row effect is NOT an artifact of J_fit4 having a larger operator norm.

Two problems.

(a) **Multiple comparisons.** 2 tasks x 3 contrasts = 6 tests, no correction anywhere.
Bonferroni-adjusted (99.17%) percentile bootstrap intervals, 50000 draws:
    multihop  h1_prereg  -0.110  95% [-0.150,-0.073]  Bonf6 [-0.165,-0.060]  survives
    multihop  fit_only   -0.174  95% [-0.228,-0.122]  Bonf6 [-0.247,-0.104]  survives
    multihop  state_only -0.045  95% [-0.085,-0.008]  Bonf6 [-0.100,+0.004]  **DOES NOT**
    arithmetic: all three span zero at 95% already.
The honest correction is Bonferroni or Holm over the six, and under it the only contrast
that fails is `state_only` — the preregistered mechanism. That strengthens MAIN's
conclusion ("H1 confirmed as a number, refuted as a mechanism") rather than weakening it,
but the write-up currently presents -0.045 [-0.086,-0.009] as excluding zero and should
say instead that the state effect does not survive correction while the fit effect does.

(b) **The arithmetic variance shares are not measurements.** Bootstrap over items,
2000 draws (pre-fix numbers, same conclusion post-fix): fit share CI [0.08,0.61], state
share [0.04,0.35], interaction [0.24,0.74]. Quoting 0.33/0.16/0.51 (now 0.10/0.23/0.67)
to two decimals invites a reader to interpret them. Either give the CIs or say "shares
are uninformative at n=51". Multihop's fit share is solid: bootstrap CI [0.95,0.99].

(c) An objection I tested and could NOT sustain: "state loop explains only 1% because
loops 2, 3 and 4 hold near-identical states, so the columns have nothing to vary".
Under the identity readout (logit lens) at the same locations, the top-1 token differs
between loop 3 and loop 4 at 53.6% of (item, layer) pairs, and between loop 2 and 3 at
67.5%; mean |d log10 rank| 0.24-0.50 across every loop pair. Mid-layer states are
genuinely distinct even though the exits converge. The column factor had room to move
and did not. C6's asymmetry is real.

(d) `N_BOOT=1000` off a module-level RNG that advances between calls: not a bias, but the
reported third decimal is Monte-Carlo noise. Across 20 seeds the 97.5th percentile of
`state_only` wandered over [-0.0095,-0.0043] — the sign call is stable, the endpoint is
not. Reporting the CI to 3dp off 1000 draws overstates its precision; use 20000 draws, or
report 2dp. Also: the RNG position depends on how many `boot_ci` calls ran before, so a
rerun with a different set of tasks or methods gives different CIs for the same data.
Seed each `boot_ci` from a stat-specific key if reproducibility matters for the paper.

## 6. C3 / C4 — the eventual-exit deficit itself. HOLDS (C3); C4 WEAKENED (see section 3b).

Beyond the permutation null in section 2a: at multihop loop 2 the eventual-exit lens
ranks the true intermediate *worse* than a random other name of the same kind
(own 0.0012 vs null 0.0130 +- 0.0067, two-sided permutation p = 0.029, uncorrected).
Loop 1 own 0.0016 vs null 0.0041 (p = 0.14). Loop 3's +0.011 per-layer excess is not
distinguishable from the null (p = 0.20). Loop 4 p < 0.0001. So on the per-layer metric
the eventual-exit lens reads nothing at loops 1-3 of multihop in the strict sense of
"not above a label-shuffled baseline", which is a stronger statement than the write-up
makes and is worth making.

## 7. C7 — TRANSPORT COHERENCE. Conclusion HOLDS; the stated justification is FALSE.

`lens_convergence.py` uses the two-point moment estimator on E||J_n||^2 = ||mu||^2 +
sigma^2/n. I recomputed per-layer squared Frobenius norms for all seven exit3 lenses and
ran **every** valid disjoint pair with n1 != n2, not the three the write-up reports
(there are six: p0-8 pairs with each of the three n=24 shards, p32-56 and p56-80 pair
with the n=32 merge, p56-80 pairs with n56).

||mu||/sqrt(d), mean over the layers of each source loop:

| pair (n1, n2) | loop 1 | loop 2 | loop 3 | loop 4 | layers with sigma^2 < 0 | with mu^2 < 0 |
|---|---|---|---|---|---|---|
| p32-56 vs merged (24, 32) | 0.166 | 0.179 | 0.284 | 0.842 | 107/191 | 0 |
| p56-80 vs merged (24, 32) | 0.207 | 0.169 | 0.271 | 0.819 | 65/191 | 0 |
| p56-80 vs n56 (24, 56)    | 0.161 | 0.171 | 0.266 | 0.798 | 1/191 | 0 |
| p0-8 vs p32-56 (8, 24)    | 0.095 | 0.170 | 0.265 | 0.799 | 5/191 | 4/191 |
| p0-8 vs p56-80 (8, 24)    | 0.060 | 0.175 | 0.272 | 0.812 | 4/191 | 15/191 |
| p0-8 vs p8-32 (8, 24)     | 0.097 | 0.168 | 0.270 | 0.815 | 1/191 | 7/191 |

- Loops 2, 3, 4: ||mu|| is stable to about 2% CV. The write-up is right there.
- **Loop 1 is not stable**: 0.060 to 0.207, a factor 3.5, CV 39%. This is precisely the
  row the five-to-one claim rests on. The loop4/loop1 ratio across the six pairs is
  [5.1, 4.0, 5.0, 8.5, 13.6, 8.4], not "about five". The sentence "The ||mu|| estimates
  agree closely across all three disjoint pairs, so the five-to-one ratio is solid" is
  false as written — the write-up's own table shows 0.10 / 0.17 / 0.16 for loop 1, a 70%
  spread, and the three pairs it omits are worse.
- **The estimator degenerates and the degeneracy is hidden.** `np.sqrt(max(sigma2, 0))`
  and `np.sqrt(max(mu2, 0))` silently clip. sigma^2 comes out negative (i.e. the smaller-n
  lens had the *smaller* norm, impossible under the model) at 107 of 191 layers for the
  best-conditioned-looking pair, and mu^2 comes out negative at up to 15 layers. Clipping
  a negative mu^2 to 0 before averaging sqrt over layers biases the reported ||mu|| upward.
  At minimum, `lens_convergence.py` should print the count of clipped layers.

**But the conclusion survives on a better statistic.** ||mu||^2 <= E||J_n||^2 for every n,
so the raw norm of the largest fit is an upper bound on ||mu||, and the noise term
sigma^2/n is *larger* at loop 1 (bigger scatter), so the bound is tighter at loop 4:
    ||J_80||/sqrt(d) per loop = [0.159, 0.178, 0.272, 0.810]  ->  ratio 4/1 = 5.09,
and the true mu ratio is at least that. Two better-conditioned estimators agree:
pooling the three independent n=24 shards against n=8 gives ratio 9.7; least squares of
||J_n||^2 on 1/n over the nested n = 8, 32, 56, 80 gives ||mu|| = [0.134, 0.173, 0.269,
0.807], ratio 6.0, with no clipped layers at all.

Recommended restatement: report ||J_80||/sqrt(d) = [0.16, 0.18, 0.27, 0.81] as the primary
number, note that the noise term inflates the loop-1 entry more than the loop-4 entry so
5.1 is a *lower* bound on the ratio, and keep the two-point table as a consistency check
with the loop-1 instability stated. The raw norms also show loop 1 still falling with n
(0.286 at n=8, 0.185/0.172/0.185 at the three n=24 shards, 0.181 at 32, 0.166 at 56,
0.159 at 80) while loops 2-4 are flat from n=24 — the same "not yet converged at long
horizon" signal as section 4 above.

Interpretive caveat, not a defect: ||mu|| falling with horizon is what any deep network
does to a mean input-output Jacobian; without a non-recurrent 192-layer control this
cannot be attributed to recurrence specifically. The readout comparison against the logit
lens is what carries that attribution, not this.

## 8. Number mismatches found in the current (00:25) RESULTS.md

- **Section 7 prose**: "The preregistered statistic is negative and excludes zero for
  multihop, **-0.113** CI [-0.147,-0.074]" — the point estimate is now **-0.110**, as the
  table three lines below correctly says. Stale from before the control fix.
- **Section 7 objection paragraph**: "the vanilla logit lens reads the intermediate well
  (**0.38** any-layer excess on multihop" — now **0.358**.
- **Section 4 / section 8 cross-reference**: "at virtual depth 19 of 192 ... the vanilla
  logit lens already beats a supervised probe on the arithmetic intermediate (0.89 against
  0.67)". Those two numbers are per-loop maxima at *different* layers (logit 0.892 at L31,
  probe 0.665 at L25). At L19 itself the numbers are logit 0.761 against probe 0.170. The
  claim is true and in fact stronger at L19; the quoted pair just does not belong to L19.
- **Section 10**: "The eventual-exit lens is tested from 8 to 80 fitting prompts and is
  flat over that range" now contradicts section 6, which concedes arithmetic loop 2 drifts,
  and contradicts the rank evidence in section 4 of these notes.
- **Section 6**: "Three disjoint pairs are available" — six are.
- Verified as CORRECT after the fix: the whole fit-size table (recomputed independently to
  +-0.002), both pos-2 robustness vectors, model-correct 0.352 -> 0.540, exit-agreement
  0.723 vs 0.453, group sizes 90/51, all of section 5, all of section 3.
- Earlier M1 (stale fig6/fitsize_summary.md) and M2 (Pearson range) are fixed. Confirmed:
  `fitsize_summary.md` and `fig6_fit_size.png` now mtime 00:12 and agree with summary.json.

## 9. Two smaller methodological notes

- `analyze.py` `best_excess_ci95` bootstraps the CI at the location chosen as the argmax of
  the same data. That is a winner's-curse interval and is not a valid CI for the effect at
  that location. It is not quoted in RESULTS.md; if it ever is, it needs a cross-fitted
  layer choice.
- Every number in section 8's probe table is a maximum over 48 layers of an accuracy
  estimated on 176 prompts (SE ~0.036). Selection over 48 correlated layers inflates all
  three rows. The cross-validated probe below fixes both the n and the selection.

## 10. On PEER1's notes (read at 00:40 and 00:55)

Agreement: PEER1's section 4 (rescaling J changes nothing) and section 6 (MiniCPM control)
are the right attacks and I have nothing to add. PEER1's bit-identical reproduction of the
stored rank arrays plus my independent re-implementation of the *scoring* from
items.json/task_names.json means the pipeline is checked end to end from two directions.

**Disagreement, PEER1 section 5.** PEER1 reports multihop loop 4 mean-over-layers
"0.191 vs 0.166 — the J-lens WINS" and MAIN has retracted C4 partly on that plus the
arithmetic any-layer reversal. Both are point estimates without intervals. Paired
bootstrap over items, 20000 draws, of (eventual-exit J minus logit), mean-over-layers:

    multihop   diff [-0.053, -0.101, -0.123, **+0.025**]
               CI95 [[-0.076,-0.033], [-0.132,-0.073], [-0.165,-0.083], [**-0.020,+0.071**]]
    arithmetic diff [-0.229, -0.022, -0.007, -0.041]
               CI95 [[-0.295,-0.162], [-0.078,+0.034], [-0.052,+0.041], [-0.080,-0.001]]

The multihop loop-4 "win" does not exclude zero. Combined with my section 3b numbers, the
picture across every metric is consistent:

  wherever the J-lens/logit-lens difference is statistically resolvable, the logit lens
  wins; every apparent J-lens win (multihop loop 4 mean-over-layers +0.025; arithmetic
  loops 2-3 any-layer +0.095/+0.053) is inside its interval.

So C4 should be **WEAKENED**, not retracted. Correct wording: "the eventual-exit lens
never significantly beats the logit lens at any loop on either task, on any of the three
statistics; on multihop it is significantly worse at loops 1-3 and ties at loop 4."
Reporting a bare reversal as a refutation of C4 repeats the error the reversal exposed.

## 11. A confound I tested and could not sustain (worth one line in Limitations)

The lens is fitted on wikitext (`max_seq_len=128`, `skip_first=16`) and applied at the
last token of 7-to-41-token stimuli: 99 of the 148 items are read at a position index
below 16, i.e. positions the fitting code explicitly excludes as atypical. That is an
extrapolation of the fitted map outside its estimation window and it is not mentioned
anywhere. It does not, however, explain the eventual-exit deficit: the effect of short
prompts is the same sign and similar size for the *unfitted* logit lens (multihop loop 1
excess +0.082 for pos<16 vs +0.014 for pos>=16, r(pos, excess) = -0.38) as for the
J-lens, and the local-exit lens, fitted on exactly the same wikitext prompts and positions,
reads the same states fine. Short prompts are simply easier for every readout.
Recommended: one sentence in Limitations, and on the rented run a lens fitted on
short prompts as a cheap control.

---

## 12. C8 — THE CROSS-VALIDATED PROBE. Built and run. Verdict: WEAKENED but the loop-1
   result survives; "decodability outlasts verbalizability" is TRUE only at loop 3.

Code: `src/ouro_jlens/probe_cv.py` (mine to write; nothing else in the repo touched).
Outputs (deliberately NOT under artifacts/, which I must not write):
`/tmp/claude-1000/.../scratchpad/p2/probe_cv/{arrays.npz,summary.json,lens_all648.npz}`.

Commands actually run:
    # GPU, under the lock (14 s of GPU time)
    OMP_NUM_THREADS=4 nice -n 10 venv/bin/python src/ouro_jlens/probe_cv.py \
        --out .../scratchpad/p2/probe_cv --lens-only
    # CPU
    OMP_NUM_THREADS=6 nice -n 10 venv/bin/python src/ouro_jlens/probe_cv.py \
        --out .../scratchpad/p2/probe_cv        # 1622 s

Design, as specified: 45 unordered pairs shuffled into 5 folds of 9; each of the 648
prompts is a test prompt in exactly one fold; C picked from {0.01, 0.1, 1.0} on 8
validation pairs drawn from that fold's 36 training pairs, then refit on all 36; scaler
fitted on training prompts only. Asserted per fold that no test pair or its mirror is in
train or val (0 violations in all 5 folds) and that every prompt is scored exactly once.
Both lenses rescored on all 648 prompts with `probe.lens_candidate_ranks`, the same code
path `probe.py` uses; restricted to the old 176-prompt split my lens arrays are
**bit-identical** to the published ones for all six lens arrays, so the lens columns are
the same measurement on 3.7x the data.

### The structural problem the CV exposes, which the single split hid
The label is a+b, a deterministic function of the operand pair, so holding out pairs can
remove a label from training entirely. Labels 2, 3, 17 and 18 have exactly one unordered
pair each. Across the five folds, 72 of 648 test prompts carry a label absent from their
fold's training set and are automatically scored wrong for the probe (probability 0), while
the lenses are unaffected. `probe.py`'s single split has the same defect — 14 of 17 classes
in training — so the published probe column was already handicapped. All numbers below are
therefore given on the 576-prompt learnable subset; the full-648 numbers are in
`summary.json` and are 0.02-0.05 lower for the probe and unchanged for the lenses.

### Result table
Per-loop top-1 over the 17-way candidate set. "max" = maximum over the 48 layers (the
statistic RESULTS.md section 8 uses, upward-biased by layer selection); "cross-fitted" =
the layer is chosen on the other four folds and scored on the held-out one, so it is
selection-free. Intervals are a **cluster bootstrap over the 45 unordered operand pairs**
with the layer choice refit inside each draw — pairs, not prompts, are the independent unit.

| readout | loop 1 | loop 2 | loop 3 | loop 4 |
|---|---|---|---|---|
| cross-validated probe, max | 0.627 | 0.597 | 0.356 | 0.267 |
| **cross-validated probe, cross-fitted** | **0.615** | **0.590** | **0.335** | **0.236** |
| &nbsp;&nbsp;CI95 (pair cluster) | [0.47, 0.74] | [0.46, 0.72] | [0.22, 0.44] | [0.15, 0.35] |
| **logit lens, cross-fitted** | **0.764** | **0.500** | **0.179** | **0.311** |
| &nbsp;&nbsp;CI95 | [0.66, 0.91] | [0.28, 0.64] | [0.06, 0.24] | [0.22, 0.39] |
| **eventual-exit J-lens, cross-fitted** | **0.392** | **0.500** | **0.148** | **0.120** |
| &nbsp;&nbsp;CI95 | [0.23, 0.56] | [0.29, 0.64] | [0.08, 0.27] | [0.05, 0.17] |
| (published, 176-prompt split) probe / logit / J | 0.67 / 0.89 / 0.55 | 0.65 / 0.62 / 0.56 | 0.34 / 0.18 / 0.22 | 0.33 / 0.40 / 0.15 |

Pairwise, cluster bootstrap, 2000 draws:

| contrast | loop 1 | loop 2 | loop 3 | loop 4 |
|---|---|---|---|---|
| probe - logit | -0.149 [-0.358,+0.001] | +0.090 [-0.090,+0.346] | **+0.156 [+0.031,+0.315]** | -0.075 [-0.190,+0.064] |
| probe - J-lens | +0.222 [-0.007,+0.422] | +0.090 [-0.083,+0.341] | **+0.187 [+0.027,+0.316]** | **+0.116 [+0.022,+0.250]** |
| logit - J-lens | **+0.372 [+0.173,+0.596]** | 0.000 [-0.251,+0.252] | +0.031 [-0.137,+0.123] | **+0.191 [+0.094,+0.302]** |

(A prompt-level bootstrap, which ignores pair clustering, gives much tighter intervals —
probe - logit at loop 3 is +0.177 [+0.130,+0.222] and at loop 1 -0.219 [-0.262,-0.175].
Those are the numbers to quote only if you say they treat prompts as independent.)

### Answers

**Does linear decodability outlast verbalizability across recurrent loops?**
Partly, and only in the middle. Both decline: the probe falls 0.615 -> 0.236 (to 38% of
its loop-1 value) and the logit lens 0.764 -> 0.311 (to 41%, with a trough of 0.179 at
loop 3). They do not decline *together*: at loop 3 the probe reads the intermediate at
roughly twice the logit lens's rate, +0.156 with a cluster-bootstrap interval excluding
zero, and it beats the eventual-exit J-lens at loops 3 and 4. At loop 1 verbalizability is
ahead and at loop 4 the two are level. So the honest statement is "linear decodability
outlasts verbalizability through loop 3 and both are largely gone by loop 4", not the
earlier draft's "decodability persists while verbalizability collapses", and not the
current draft's "both decline" full stop. The current text is *too* pessimistic: it says
the claim "does not survive the leak-free split", and with 45 pairs instead of 11 and a
selection-free layer choice, a residual version of it does.

**Was the probe just under-powered at loop 1?** No. With 36 training pairs instead of 24,
every prompt tested, and no layer-selection bias, the probe still reads 0.615 at loop 1
against the logit lens's 0.764. Going from 24 to 36 pairs moved it from 0.665 (one noisy
11-pair split) to 0.615. The write-up's stated explanation — "clearly data-limited ...
which is why an untrained readout beats it at loop 1" — is **not supported**: more data
does not close the gap, so the loop-1 logit-lens advantage is a property of the
representation, not of the probe's sample size. That strengthens C9 rather than weakening
C8, and section 8's "two cautions" paragraph should be rewritten accordingly.

**How solid is any of this?** Not very, and the CV is what shows it. Per-fold loop-1 probe
accuracy on the learnable subset is 0.644, 0.588, 0.275, 0.919, 0.667 — SD 0.21 across
folds of 9 pairs. A single 11-pair split, which is what section 8 reports, has a sampling
SD of that order. Every number in the published section-8 table should be read with an
uncertainty near +-0.2, and none of them should be quoted to two decimals. The
cluster-bootstrap intervals above are the ones to publish.

**Where the eventual-exit J-lens sits.** Unchanged and worse: it is below the logit lens at
loops 1 and 4 with intervals excluding zero, below the probe at loops 3 and 4, and level
with both at loop 2. Section 8's conclusion about the J-lens survives the better test.

---

## 13. VERDICT TABLE (PEER2, remit C3-C8)

| claim | verdict | one-line basis |
|---|---|---|
| C3 main result | **HOLDS** | Every number reproduced from the artifacts by an independent re-implementation of the scoring; survives a label-shuffle permutation null (eventual-exit loop-1/2 multihop p = 0.95 / 0.98 one-sided for being *above* the null, and loop 2 is significantly *below* it); survives PEER1's rescaling attacks. |
| C4 never beats logit | **WEAKENED** (not FALSE) | Every apparent J-lens win is inside its interval: multihop loop-4 mean-over-layers +0.025 [-0.020,+0.071]; arithmetic any-layer loops 2/3 +0.095 [-0.041,+0.228], +0.053 [-0.083,+0.183]. Wherever the difference resolves, the logit lens wins. Retracting to FALSE overstates it. |
| C5 fit size | **WEAKENED** | hit@10 is floored at loops 1-2. Under it, log10 rank of the true intermediate improves significantly over 8->80 in 6 of 8 (task, loop) cells and is flat only at loop 4 and at multihop loop 1. "Converged by eight prompts" is wrong; "multihop loop 1 is flat at both levels" is right. |
| C6 cross-loop | **HOLDS on multihop; the state-effect sub-claim is BROKEN by multiple comparisons** | Decomposition, contrast balance and bootstrap unit all verified correct. Bonferroni over the 6 tests: h1_prereg and fit_only survive, `state_only` -0.045 goes to [-0.100,+0.004]. Arithmetic shares are noise (bootstrap CI on the fit share [0.08,0.61]). The "states are near-identical so columns can't move" objection is refuted by the data. |
| C7 transport coherence | **WEAKENED; conclusion HOLDS on a better statistic** | ||mu|| stable to 2% at loops 2-4 but 0.060-0.207 at loop 1 across the six valid disjoint pairs; sigma^2 negative at up to 107/191 layers and mu^2 negative at up to 15/191, both silently clipped. The 5x ratio survives as a *lower bound* from raw ||J_80|| = [0.16,0.18,0.27,0.81]. The sentence "the ||mu|| estimates agree closely" is false. |
| C8 supervised reference | **WEAKENED, and its stated explanation is FALSE** | Cross-validated over all 45 pairs, 648 prompts each tested once, selection-free layer choice, pair-cluster intervals: probe [0.615,0.590,0.335,0.236] vs logit [0.764,0.500,0.179,0.311] vs J-lens [0.392,0.500,0.148,0.120]. The loop-1 logit-lens advantage is NOT a data-limitation artifact. Decodability outlasts verbalizability at loop 3 only (+0.156 [+0.031,+0.315]). |

Still stale in RESULTS.md as of my last read: line 281 "-0.113" (should be -0.110);
line ~296 "0.38 any-layer excess on multihop" (should be 0.358); line 214 "Three disjoint
pairs are available" (six are); line 231 "The ||mu|| estimates agree closely across all
three disjoint pairs" (false at loop 1); line 386 "flat over that range" (contradicts
section 6 and section 4 of these notes).
