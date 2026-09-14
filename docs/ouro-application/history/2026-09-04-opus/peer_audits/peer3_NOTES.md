# PEER3 notes — auditing the auditors (PEER1, PEER2, and MAIN's own recent edits)

Scope: I do not re-derive C1-C10. I check the *checks*. Scripts in `scratchpad/p3/`,
interpreter `venv/bin/python`, everything below is CPU (I have not taken the GPU lock;
none of this needed it).

Tooling I built and validated first, because every verdict below rests on it:
- `p3/score.py` — independent re-implementation of the excess/any-layer/candidate metric
  straight from `items.json` + `task_names.json`. Does NOT import `analyze.py`.
- `p3/cpuread.py` — CPU replica of `jlens.HFLensModel.unembed` (bf16 cast, Ouro RMSNorm
  eps 1e-6, `lm_head`) plus `jlens.vis._ranks_of`, loading only `lm_head.weight` and
  `model.norm.weight` out of the safetensors. Lets me re-score arbitrary transported
  states without the model, the GPU, or the lock.

---

## 0. PEER1's "bit-identical" claim — CONFIRMED, and it matters

    venv/bin/python -c "... np.array_equal(rescale_allrank.npz[base], arrays.npz[jlens_exit3_allrank])"

`base` vs stored `jlens_exit3_allrank`: identical, 0 differing entries.
`logit` vs stored `logitlens_allrank`: identical, 0 differing entries.
So `p1/H.npy` is the same cache the published run used and PEER1's rescoring is the same
measurement. Everything in PEER1 section 4 is comparable to the paper. Good.

My own CPU readout path reproduces the same aggregates to +-0.002 on mean-over-layers and
+-0.026 on any-layer (bf16 logit ties break differently off the GPU; the max over 48
layers amplifies that). Adequate for the 0.00-vs-0.08 contrasts below, not for bit checks.

## 1. PEER1 section 4, the whitening result — **PARTLY REFUTED / OVERSTATED**

I reproduce PEER1's table exactly from their own arrays with my independent scorer:

    multihop mean-over-layers excess   base [-0.000,-0.007, 0.016, 0.191]
                                       orth [-0.000, 0.006, 0.079, 0.175]
                                      logit [ 0.053, 0.095, 0.139, 0.166]
    multihop raw any-layer pass@10     base [0.033,0.022,0.144,0.556]
                                       orth [0.022,0.111,0.489,0.611]

(so PEER1's quoted "0.144 -> 0.489" is the raw hit, not the excess; the like-for-like
any-layer *excess* goes 0.089 -> 0.428. Say which.)

Three problems.

### 1a. "Same picture on order-ops" is FALSE. Whitening DESTROYS arithmetic loop 2.

    order-ops numeric, mean-over-layers excess
      base [0.006, 0.096, 0.087, 0.092]
      orth [0.002, 0.005, 0.041, 0.106]      <- loop 2 falls 0.096 -> 0.005 (20x), loop 3 halves
    order-ops numeric, any-layer excess
      base [0.059, 0.247, 0.184, 0.172]
      orth [0.065, 0.081, 0.243, 0.228]      <- loop 2 falls 0.247 -> 0.081

On the second of the two tasks, whitening does the opposite of rescuing at loop 2 and only
helps at loop 3. PEER1's sentence "Same picture on order-ops" is not supported by the data
in their own `rescale_allrank.npz`. If MAIN has written "whitening rescues loop 3" into the
paper as a general statement, it is a multihop-only statement with an arithmetic
counter-example.

### 1b. Whitening does not "rescue" the readout, it partially un-breaks it — and lands
###     BELOW the do-nothing baseline.

multihop loop 3: base 0.016, orth 0.079, **identity (= logit lens) 0.139**. The whitened
map is still only 57% of what you get by applying no map at all. Same on any-layer excess:
0.428 vs 0.500. So the loop-3 content was never in doubt (the logit lens reads it), and the
correct statement is "whitening recovers about half of the damage the fitted J does at
loop 3, and none of it at loop 1", not "rescues".

### 1c. The loop-1 half of MAIN's split — "a genuine transport failure at loop 1" — is
###     contradicted by the cross-loop matrix already in `arrays.npz`.

`summary.json -> cross_loop -> multihop -> excess_hit10_fit_by_state` (rows = fit loop,
cols = state loop), which I recomputed independently and match exactly:

    fit L1 [-0.002,  0.018,  0.015,  0.028]
    fit L2 [-0.014, -0.014,  0.017,  0.060]
    fit L3 [ 0.042,  0.054,  0.089,  0.106]
    fit L4 [ 0.422,  0.447,  0.485,  0.464]     <- J fitted at loop 4, applied to a LOOP-1 state: 0.422

A loop-1 state read through the loop-4-fitted Jacobian scores 0.422 any-layer excess. The
loop-1 content is present and linearly transportable; what fails is the *map fitted at
loop 1*, not the state. "Genuine transport failure at loop 1" is therefore the wrong
gloss if it is meant to say the content cannot be transported. It can be, by a different
linear map, at 0.422.

### 1d. What is actually going on (cheap, no SVD needed)

Frobenius cosine between each fitted J and the identity, `cos = tr(J)/(||J||_F sqrt(d))`
(chance for a random 2048x2048 matrix is 1/sqrt(d) = 0.022):

    loop 1  mean 0.069  (min 0.051, max 0.082)   ||J||/sqrt(d) 0.181
    loop 2  mean 0.097                            0.183
    loop 3  mean 0.134                            0.277
    loop 4  mean 0.471  (min 0.132, max 1.000)    0.827
    share of ||J||_F^2 explained by the best scalar multiple of I: [0.005, 0.010, 0.018, 0.287]

The loop-4 J is strongly identity-aligned and therefore behaves like the logit lens; the
loop-1 J is barely distinguishable from a random rotation in this respect. That is C9's
mechanism as a number, and it also predicts the whitening result without any appeal to
"conditioning": the polar factor of a near-random-direction matrix is a near-random
rotation, so it cannot recover a readout that lives in the identity direction.

### 1e. THE CONTROLS. A whitened Jacobian is no longer a lens. **REFUTED.**

    OMP_NUM_THREADS=10 nice -n 15 venv/bin/python p3/run_whiten.py \
        --variants orth orth_l4 orth_shuf randorth --out p3/w_controls.npz --threads 10
    (+ --variants orth_shuf --seed 1 / --seed 2 for the two extra shuffles)

All scored with my CPU readout, so all rows are mutually comparable; `identity` and `base`
reproduce the GPU pipeline's aggregates to +-0.002 (mean-over-layers).

multihop:

| variant | what it is | mean-over-layers excess | any-layer excess |
|---|---|---|---|
| identity | no map at all (= logit lens) | [0.053, 0.094, 0.139, 0.165] | [0.358, 0.459, 0.500, 0.494] |
| base | J as fitted | [-0.000, -0.006, 0.016, 0.189] | [-0.001, -0.013, 0.090, 0.464] |
| orth | polar factor of the J fitted **here** (PEER1) | [-0.000, 0.006, 0.078, 0.174] | [-0.002, 0.086, **0.428**, 0.517] |
| orth_l4 | polar factor of the J fitted at **(loop 4, same L)** | [**0.059**, 0.091, 0.132, 0.174] | [**0.337**, 0.471, 0.497, 0.517] |
| orth_shuf | polar factor from a **random other** location, seed 0/1/2 | [0.019, 0.029, 0.059, 0.082] | [0.285, 0.369, **0.519**, 0.543] |
| orth_shuf s1 | | [0.017, 0.042, 0.047, 0.087] | [0.294, 0.436, 0.438, 0.477] |
| orth_shuf s2 | | [0.014, 0.022, 0.060, 0.079] | [0.317, 0.359, 0.491, 0.510] |
| randorth | Haar-random orthogonal per location | [0.000, 0.000, -0.000, 0.000] | [0.011, 0.013, -0.003, 0.012] |

Read the loop-1 and loop-3 columns.

- **Loop 1.** The whitened *correct* Jacobian reads nothing (-0.002 any-layer excess). The
  whitened Jacobian from **a different fit loop** reads it at 0.337 (orth_l4) and a
  whitened Jacobian from a **random other location** reads it at 0.285/0.294/0.317 across
  three seeds. So the loop-1 null is not "the content cannot be transported"; a
  well-conditioned map fitted somewhere else transports it fine. This is the same thing
  the cross-loop matrix says (1c) and it is now shown for the whitened maps too.
- **Loop 3.** PEER1's "rescue" (0.428) is **not specific to the loop-3 Jacobian**. Three
  random shuffles give 0.519 / 0.438 / 0.491 — as good or better. The rescue therefore
  reports nothing about loop-3 content; any fitted Jacobian's polar factor does it.
- **But it is not pure conditioning either.** `randorth` — a genuinely random orthogonal
  matrix, perfectly conditioned, unit singular values — reads exactly nothing everywhere.
  So "any well-conditioned map rescues it" is false. What the fitted Jacobians share, and
  a Haar matrix does not, is partial alignment with the identity (section 1d), and that is
  what survives whitening.
- `orth_l4` reproduces the logit lens almost exactly ([0.059, 0.091, 0.132, 0.174] against
  identity's [0.053, 0.094, 0.139, 0.165]). That is the cleanest single statement of what
  the loop-4 Jacobian is: a scaled near-identity dressed up as a transport map.

**Answer to the question PEER1 did not ask.** A whitened Jacobian is not a lens in any
useful sense. It does not report location-specific content (a wrong-location whitened map
does as well or better), it never exceeds the do-nothing baseline, and it does not survive
the one thing it would need to survive to be interpretable — being replaced by a whitened
map from somewhere else. The result belongs in a methods appendix as "the early-loop null
is not a magnitude or conditioning artifact", not in the paper as a claim that the loop-3
content is present-but-badly-conditioned. It cannot support a two-regime split.

## 4. PEER2 section 2b as MAIN implemented it — **CONFIRMED, cross-loop path included**

    OMP_NUM_THREADS=4 venv/bin/python  (p3 script, independent of analyze.py)

Every cell of the cross-loop tables reproduces MAIN's `summary.json` exactly on both
tasks: `excess_hit10_fit_by_state`, `hit10_fit_by_state`, `control10_fit_by_state`,
variance shares (mine 0.984/0.012/0.004 vs MAIN 0.98/0.01/0.00), and all six contrast
point estimates (multihop h1 -0.1098 / state -0.0452 / fit -0.1744). The trailing-axis
dispatch in `_split_layers` / `_drop_layer_axis` does the right thing on the
`[n_ctrl, 4, 4, 48]` tensor: 48 != 192 so the reshape is skipped and `.max(-1)` collapses
the layer axis, giving a per-name any-layer control. The main-table any-layer excess now
equals the cross-loop diagonal to 3dp on both tasks, as it must.

One cosmetic defect: `summary.md`/`summary.json` print `pass10_per_loop` and
`control_pass10_per_loop` each rounded to 3dp, so a reader who subtracts them gets
multihop loop 3 = 0.088 where the exact excess is 0.089. Harmless, but if RESULTS.md
quotes a subtraction of the two printed vectors it will be off by 0.001 in places.

## 5. PEER2's cross-validated probe `probe_cv.py` — no leakage, but a **REAL structural
##    defect that biases C8's comparison against the probe**

Leakage checks (mine, on the actual fold assignment):
- 0 unordered pairs span train and test in any fold.
- 0 ordered pairs whose mirror sits in a different fold. The (3,5)/(5,3) bug is genuinely
  fixed; `prompt_folds` keys on `(min(a,b), max(a,b))`, which is correct.
- `StandardScaler` is fit on training rows only, in both the C-selection fit and the
  refit. No scaler leakage.

The defect: **11.1% of test prompts carry a label that does not occur anywhere in their
fold's training set, so the probe cannot possibly get them right, while the two lenses
have no such handicap.**

    fold 0: label 4 missing from train+val   -> 24 test prompts unpredictable
    fold 2: labels 3, 17, 18 missing         -> 40
    fold 4: label 2 missing                  ->  8
    total 72/648 = 11.11%; the CV probe's top-1 is capped at 0.889, the lenses at 1.0

Cause: with a+b in 2..18 and 45 unordered pairs, labels 2, 3, 17, 18 each come from a
single pair, and 4, 5, 15, 16 from two — fold 0 happened to draw both of label 4's pairs.
`_rank_of_true` handles the absent class gracefully (probability 0) but that is scored as
a miss, so the handicap is silent.

Effect on the headline (per-loop max over layers, candidate top-1):

    readout                    all 648 (as reported)      predictable-only (n=576)
    cross-validated probe      [0.557 0.531 0.316 0.238]  [0.627 0.597 0.356 0.267]
    logit lens                 [0.853 0.511 0.174 0.310]  [0.845 0.500 0.179 0.311]
    eventual-exit J-lens       [0.423 0.480 0.164 0.111]  [0.431 0.500 0.175 0.120]

    paired bootstrap over prompts, (probe - logit lens) at each readout's own best layer:
      all 648:      loop1 -0.296 [-0.341,-0.253]  loop2 +0.020 [-0.019,+0.059]
                    loop3 +0.142 [+0.100,+0.185]  loop4 -0.073 [-0.116,-0.029]
      predictable:  loop1 -0.219 [-0.260,-0.175]  loop2 +0.097 [+0.062,+0.132]
                    loop3 +0.177 [+0.130,+0.224]  loop4 -0.043 [-0.090,+0.002]

**C8's headline survives**: the logit lens beats the probe at loop 1 by a wide margin
either way, and the margin is a real 0.219 after the handicap is removed. But loop 2 flips
from "tie" to "probe significantly ahead" and loop 4 flips from "logit lens significantly
ahead" to "tie". Any per-loop prose beyond loop 1 must be restated, and the probe's
absolute numbers as published understate it by ~0.06-0.07.

Two smaller things in the same file:
- `"chance": 1/len(LABELS)` = 0.059 is reported, but the label distribution is far from
  uniform (label 10 has five generating pairs). The majority-class rate is 0.111 — nearly
  double the quoted chance, and above the J-lens's loop-4 number of 0.111/0.120.
- `per_loop_max` picks the layer on the same prompts it scores, on all three readouts.
  PEER2 already flagged this in their section 9 and their `cv_report.py` has a
  cross-fitted variant; it should be the reported number, not the appendix.

STATUS 01:05: items 2 (position reduction), 3 (MiniCPM), 6 (peer-vs-peer consistency)
and the RESULTS.md number audit still open. Whitening controls running.

---

## 2. PEER1 section 7 / 8c / 9, the position-reduction A/B — implementation CORRECT,
##    conclusion **OVERSTATED**: the A/B is confounded with sample size and I have
##    measured the size of the confound.

### 2a. The implementation is right. Both branches check out.

`p1/gpu5.py`, `fit_ab`. Read line by line against `jlens/fitting.py:jacobian_for_prompt`.

- summed branch: `cot[bi[:nd,None], vp[None,:], d0+bi[:nd,None]] = 1.0` then
  `g[:nd, vp, :].float().mean(1)`. Identical to stock: cotangent at every valid target
  position, gradient averaged over valid source positions. Their own verify run reports
  `rel_fro=0.000e+00, max_abs=0.000e+00` at v=16 and v=184. I add an independent check:
  PEER1's `logit` column in `ab_allrank.npz` reproduces the published per-layer logit-lens
  excess **exactly** at all 12 locations on both tasks, so their whole scoring path is the
  published one. **Bit-exactness claim CONFIRMED** (caveat: verified at 2 of 191 layers,
  1 prompt, seq 24).
- diag branch: `cot[bi[:nd], P, d0+bi[:nd]] = 1.0` then `g[:nd, P, :].float()`. That is
  exactly `dh_tgt[P]/dh_src[P]` with P the last valid position. **CONFIRMED as asked.**
- `later_share == 0` in `locality.json` is a real free correctness check on the gradient
  plumbing (causality forbids anything else). Good.

### 2b. The A/B is not a matched control. The diagonal arm has ~31x fewer samples.

The summed arm averages the gradient over all valid source positions: seq 48,
`skip_first=16`, so 31 positions per prompt x 3 prompts = 93 position-samples. The diag
arm uses **one** position per prompt: 3 samples. The two arms therefore differ in the
reduction *and* in the effective sample size, and `E||J_n||^2 = ||mu||^2 + sigma^2/n` —
the very model section 6 is built on — says a smaller n inflates the norm and degrades
the map. PEER1's own norm table is the signature of this: diag/summed norm ratio is
2.4-3.3x at loop 1 (where sigma dominates) and 0.96-1.08x at loop 4 (where mu dominates).
That is what a sample-size effect looks like, not what a reduction effect looks like.

### 2c. I measured how much n=3 costs, using PEER1's own arrays.

    OMP_NUM_THREADS=4 venv/bin/python (p3, scoring p1/ab_allrank.npz against
      artifacts/jlens/eval/round1_exit3x32/arrays.npz at the same 12 virtual locations)

Per-layer excess hit@10, multihop, PEER1's 3-prompt summed lens vs the published
32-prompt summed lens at the same locations:

| location | summed n=3 | summed n=32 (published) | logit (both, identical) |
|---|---|---|---|
| loop4 L16 | 0.048 | **0.126** | 0.225 |
| loop4 L32 | 0.219 | 0.269 | 0.090 |
| loop4 L40 | 0.243 | 0.285 | 0.302 |

order-ops is worse: loop2 L40 0.045 (n=3) against 0.178 (n=32); loop3 L16 0.060 against
0.118; loop2 L16 0.087 against 0.011 in the other direction.

So **at n=3 the summed estimator itself loses up to 60% of its readout where a readout
exists**, and the diagonal arm sits at 3 position-samples against the summed arm's 93.

### 2d. The specific number that is confounded

PEER1's causal sentence is "The diagonal map at loop 1 has 2.4-3x the Frobenius norm and
puts the target token 60% further down the vocabulary" (median own-name rank at loop1 L16:
summed 11573, diag 18349, a factor 1.59). PEER2 section 4 measured that the **summed**
estimator's own mean d log10(rank) improves by +0.10 to +0.39 going from n=8 to n=80,
i.e. a factor 1.3 to 2.5 in rank purely from sample size. **The 1.59x gap PEER1 attributes
to the reduction is inside the range attributable to sample count alone.** The median-rank
argument does not carry the conclusion.

### 2e. What survives, and the free fix

What survives: a 3-prompt, one-position-per-prompt diagonal estimator does not read
multihop loop 1 (exactly 0.000 excess: no task name enters the top 10 for any item). That
is a true and useful negative. What does NOT survive is the inference from it to "the
position-summed reduction is not what produces the early-loop null", because the
comparison changes two things at once and I have shown the second one is large.

**Proposed diff to `p1/gpu5.py`, zero extra GPU cost, gives a clean 2x2.** The source
reduction is free — both reductions come out of the *same* backward — so record both on
each pass:

```python
    gs = torch.autograd.grad(tgt, srcs, cot, retain_graph=keep)
    for v, g in zip(srcv, gs):
        acc[est + "_srcmean"][v][d0:d0+nd] += g[:nd, vp, :].float().mean(1).cpu()
        acc[est + "_srcP"   ][v][d0:d0+nd] += g[:nd, P,  :].float().cpu()
```

That yields four estimators from the two backwards already being run:

| | source: mean over valid p | source: p = P only |
|---|---|---|
| **target: summed over valid p'** | stock jlens (`jacobian_for_prompt`) | new |
| **target: P only** | new — *this is the matched control* | PEER1's `diag` |

The cell "target P only, source mean over p" has the **same** number of source samples as
stock and differs from it only in the target reduction. Comparing stock against that cell
isolates the position-summed cotangent with no sample-size confound. Comparing that cell
against `diag` isolates the source reduction. Until one of those two comparisons is run,
"horizon governs transport, not the estimator" is UNTESTED, not confirmed.

**What I would tell MAIN about the rented run.** I agree with PEER1's operational
recommendation (do not restructure the run around this) but not with their epistemic one.
The paper should say "a same-position estimator fitted on 3 prompts also reads nothing at
loop 1, though at that sample size the estimator loses most of its readout even where one
exists, so this rules out only the strongest form of the artifact hypothesis." It should
not say "the result is not an artifact of the paper's position-summed cotangent
reduction". The 2x2 above costs the same as the A/B already run and would settle it; it is
a ~25-minute local job, not rented time.

### 1f. The whitened family decomposes completely, and it is a FIT-LOOP effect only

I ran `orth_lK` for K = 1..4: at every one of the 192 locations, use the polar factor of
the J fitted at (loop K, that same physical layer), and apply it to every loop's state.

multihop **any-layer excess**, rows = which loop the source Jacobian was fitted at,
columns = which loop's state it is applied to:

| whitened J fitted at | state loop 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| loop 1 (`orth_l1`) | -0.002 | -0.009 | 0.027 | 0.042 |
| loop 2 (`orth_l2`) | 0.054 | 0.086 | 0.150 | 0.200 |
| loop 3 (`orth_l3`) | 0.191 | 0.323 | 0.428 | 0.438 |
| loop 4 (`orth_l4`) | 0.337 | 0.471 | 0.497 | 0.517 |
| (identity, no map) | 0.358 | 0.459 | 0.500 | 0.494 |

PEER1's `orth` lens is exactly the **diagonal** of this table: [-0.002, 0.086, 0.428,
0.517]. Check each cell against their reported vector — it is the diagonal, cell for cell.

So "whitening rescues loop 3 but not loop 1" is not a statement about loop 3 versus loop 1
at all. It is the statement that `orth_l3` reads well **from every state** and `orth_l1`
reads nothing **from every state**, and PEER1's lens happens to use `orth_l3` at loop 3 and
`orth_l1` at loop 1. The row effect is the whole effect; there is no column effect. That is
C6's fit-loop dominance again, now shown to survive whitening, and it leaves nothing for
"a conditioning problem at loop 3" to refer to.

Independent validation of my pipeline while I was at it: `raw_l4` (the *unwhitened* loop-4
J applied to every loop) gives any-layer excess [0.389, 0.447, 0.485, 0.464] against the
published `xloop_allrank` row fit-4 of [0.422, 0.447, 0.485, 0.464] — three of four cells
exact, the fourth inside my CPU-readout noise.

---

## 3. PEER1 section 6, the MiniCPM control — **CONFIRMED, and stronger than they stated**,
##    but their raw comparison is confounded and the fix changes the argument they should make

### 3a. The matching objections do not bite. I checked all three.

    OMP_NUM_THREADS=4 venv/bin/python (p3, scoring p1/control_allrank.npz against
      round1_exit3x32/arrays.npz on item-matched slots)

- **Control name sets are identical**, not merely similar: `control_task_names.json` and
  `round1_exit3x32/task_names.json` are the *same lists* — multihop 70 names, order-ops 20
  names, `==` True on both. The "different control name sets" worry is empty.
- **Items are the same 148 in the same order**, and the scored slot sets are identical on
  multihop (100 vs 100, all shared). On order-ops MiniCPM has 54 slots to Ouro's 51; the
  3 extra are items (`nested-sub-add-mult`, `add-add-add`, `square-mult`) whose
  intermediate Ouro's tokenizer flags as **leaked** (the digit appears in the prompt token
  ids) and MiniCPM's does not. Those 3 items are, if anything, *easier* for MiniCPM, so the
  asymmetry does not bias against it.
- Re-scored on the identical 51 shared slots, MiniCPM's arithmetic mean-over-layers excess
  is +0.022 (PEER1 reported +0.020 on their 54) and it is **0.000 at every depth from 10%
  to 80%** while Ouro loop 1 reads +0.489 at 40%. **The 54-vs-51 asymmetry drives nothing.**
- The 40-vs-48-layer "fractional depth" worry cuts PEER1's way: matched on the *weight*
  stack MiniCPM reads nothing until 90%; matched on *total computation* Ouro's loop-1 L19
  is 10% and MiniCPM's 10% is L4, where it also reads nothing. Either matching gives the
  same answer, so PEER1's choice is the conservative one.

### 3b. But PEER1 never measured whether MiniCPM can do the task, and it partly cannot.

`gpu3.py` hard-codes `"correct": False` for every MiniCPM item, so the model-competence
robustness check RESULTS.md section 9 applies to Ouro was never applied to the control.
It matters: MiniCPM's **peak** excess over all 40 of its layers is +0.235 on arithmetic
against Ouro loop 1's +0.510, and its any-layer hit@10 is 0.412 against Ouro loop 1's
0.902. A raw comparison at matched depth therefore conflates "reads it later" with "reads
it less, everywhere".

### 3c. The fix, and it strengthens the conclusion

Normalise each model's depth profile by its **own** peak excess, which removes the level
difference entirely and leaves only the shape. Depth fraction at which each model first
reaches 25% / 50% of its own peak:

| | multihop | arithmetic |
|---|---|---|
| MiniCPM (40 layers) | 0.79 / 0.79 | 0.85 / 0.87 |
| Ouro loop 1 (48 layers) | **0.34 / 0.40** | **0.38 / 0.38** |
| Ouro loop 4 | 0.00 / 0.00 | 0.00 / 0.00 |

The shape difference is large and survives the competence normalisation on both tasks —
including multihop, where PEER1 called the mean-over-layers "a tie" and therefore said the
mechanism was "not uniformly demonstrated". On the normalised profile multihop is not a
tie: MiniCPM is flat to 79% depth and Ouro loop 1 is at 43% of its own peak by 40%.
**PEER1 undersold their own result by comparing levels instead of shapes.** The right
statistic for the paper is the normalised profile, and it should be reported with the
absolute-level caveat stated (MiniCPM's arithmetic peak is less than half Ouro loop 1's).

Everything PEER1 says about this being one control model, differently trained and
tokenized, and about Huginn being the clean comparison, stands. "Consistent with C9", not
"demonstrates".

---

## 6. Peer-versus-peer consistency

### 6a. RESOLVED before I got there: C4. PEER1 withdrew, PEER2 was right.
Both now agree the J-lens never *significantly* beats the logit lens. I reproduced the
whole thing independently and confirm both the point estimates and the intervals
(section 7a below). Nothing left to arbitrate.

### 6b. LIVE: PEER1 section 9 leans on a quantity PEER2 has shown is n-sensitive.
PEER1's median-rank argument (summed 11573 vs diag 18349 at loop1 L16, "60% further down")
is offered as the reason the norm difference does not matter. PEER2 section 4 measured that
the *same* median-rank statistic moves by a factor 1.3-2.5 from n=8 to n=80 for the summed
estimator alone. PEER1's diag arm is at n=3 with one source position per prompt. The two
notes cannot both be taken at face value: if rank is a sample-size-sensitive statistic
(PEER2), it cannot adjudicate a comparison whose two arms differ 31-fold in sample size
(PEER1). Detail and the free fix in section 2 above.

### 6c. LIVE: PEER1 section 3 cites a single-prompt gradient norm as independent
###     confirmation of C7. It is not, and if it were it would refute RESULTS.md section 6.

PEER1: "its norm (0.176 vs 0.880 for the loop-4 source) reproduces the 5x transport ratio
of C7 from a completely different measurement."

That is one prompt, one output dimension. C7 is a claim about ‖mu‖, the mean *over*
prompts. Decomposing E‖J_n‖²/d = ‖mu‖²/d + sigma²/(n d) by least squares on the nested
fits n = 8, 32, 56, 80 (PEER2's estimator, no clipping needed):

| loop | ‖mu‖/√d | per-prompt scatter sigma/√d | expected single-prompt ‖J₁‖/√d |
|---|---|---|---|
| 1 | 0.134 | 0.708 | 0.721 |
| 2 | 0.173 | 0.324 | 0.368 |
| 3 | 0.269 | 0.321 | 0.419 |
| 4 | 0.807 | 0.715 | 1.105 |
| **loop4 / loop1 ratio** | **6.04** | **1.01** | **1.53** |

A single-prompt Jacobian at loop 1 is 84% noise (sigma 0.708 against mu 0.134). Its
expected loop4/loop1 norm ratio is 1.53, not 5. PEER1's single sample landed on 5.0 by
chance in a distribution whose mean ratio is 1.5. It confirms nothing about C7.

Worse, if that measurement *did* generalise it would contradict the central sentence of
RESULTS.md section 6 — "Per-prompt Jacobians from loop-1 states are not themselves small;
they simply do not share a direction across prompts". The table above says that sentence is
**exactly right**: per-prompt scatter is 0.708 at loop 1 against 0.715 at loop 4, ratio
1.01, while ‖mu‖ differs 6-fold. **MAIN's mechanism claim in section 6 is CONFIRMED** by an
estimator nobody had run, and PEER1 should drop the "independent confirmation" sentence
rather than have it stand next to a claim it contradicts.

### 6d. Agreement I checked and can second
- Both peers concluded the `skip_first=16` fit/eval position mismatch does not explain the
  null, by different routes (PEER1 splits by item length, PEER2 by readout position). Their
  numbers differ slightly because the split variable differs; the conclusion is the same
  and I have no reason to doubt it.
- PEER2's least-squares ‖mu‖ = [0.134, 0.173, 0.269, 0.807], ratio 6.0: I reproduce it
  exactly (table above). PEER2 section 7 CONFIRMED.
- PEER2's Pearson range 0.453-0.765 and mean|diff| 0.049-0.089: I reproduce both exactly.

### 6e. A precision issue neither peer raised, which I checked and can close
The whole pipeline scores ranks on **bf16** logits, which carry only ~2 259 distinct values
across the 49 152-token vocabulary, so ranks are tie-block quantised. I measured the block
width against rank:

    rank    10 -> median 1 token shares that logit (p90 2)
    rank   500 -> 11 (p90 17)
    rank  1000 -> 18 (p90 26)
    rank 10000 -> 25 (p90 61)

So hit@10 is safe (a tie can displace it by at most ~4 places), PEER2's C5 rank analysis at
median ranks 96-896 is safe (+-9 against changes of 300+), and PEER1's tail ranks are safe
(+-30 against a gap of 6 800). **Not a defect.** Two consequences worth one line in
methods: `_ranks_of` breaks ties by token index, so ranks are deterministic but arbitrary
inside a block; and no rank statistic in the tail should be quoted to better than about
+-30. Also: my CPU replica of `unembed` agrees with the GPU on top-1 at 97-99% of locations
but on only ~8% of *exact tail ranks*, entirely because of this — a reader who tries to
reproduce the rank arrays off-GPU will see the same thing and should not conclude the
pipeline is broken.

---

## 7. MAIN's own recent work: the section 4 bootstrap, and every number against artifacts

### 7a. Section 4 paired bootstrap — **CONFIRMED, exactly, including the intervals.**

    OMP_NUM_THREADS=4 venv/bin/python (p3/score.py, independent of analyze.py;
      paired bootstrap over items, 5000 draws, np.random.default_rng(0))

| task | loop 1 | loop 2 | loop 3 | loop 4 |
|---|---|---|---|---|
| multihop (mine) | -0.360 [-0.459,-0.262] | -0.473 [-0.573,-0.371] | -0.411 [-0.513,-0.307] | -0.019 [-0.094,+0.058] |
| multihop (RESULTS.md) | -0.360 [-0.459,-0.262] | -0.473 [-0.573,-0.371] | -0.411 [-0.513,-0.307] | -0.019 [-0.094,+0.058] |
| arithmetic (mine) | -0.247 [-0.382,-0.106] | +0.095 [-0.044,+0.231] | +0.053 [-0.080,+0.193] | -0.041 [-0.115,+0.015] |
| arithmetic (RESULTS.md) | -0.247 [-0.382,-0.106] | +0.095 [-0.044,+0.231] | +0.053 [-0.080,+0.193] | -0.041 [-0.115,+0.015] |

Every digit. Same for the section-5 mean-over-layers version (-0.053/-0.101/-0.123/+0.025
multihop, CIs matching). The pairing is correct: `own_slots` returns the same
(item, name) order for both lenses, so the difference is genuinely paired, and `by_item`
puts items as the resampling unit. **No defect found.**

One methodological note, which is PEER2's own criticism of section 7 applied to section 4:
these are 8 uncorrected tests and MAIN writes "the logit lens wins significantly in four of
the eight cells". Under Bonferroni over 8 the three multihop early cells are far enough
from zero to survive; arithmetic loop 1 (-0.247 [-0.382,-0.106]) is the marginal one. If
section 7 gets a multiple-comparison sentence, section 4 should get the same one.

### 7b. Numbers verified correct against `artifacts/jlens/**`

All recomputed by me from `arrays.npz` / `summary.json` / the lens `.pt` files, not read
out of a summary MAIN wrote:

- Section 3, every cell: KL mean [3.828, 0.647, 0.086, 0], median [4.081, 0.220, 0.031, 0];
  exit==final multihop [0.39, 0.67, 0.80, 1.0], arithmetic [0.49, 0.843, 0.961, 1.0];
  intermediate-is-top-1 arithmetic [0.176, 0.078, 0.059, 0.039]. Stimulus counts 93 / 55.
- Section 4 any-layer table, both lenses, both tasks. Section 5 all 18 cells.
  Pearson r 0.453-0.765, mean|diff| 0.049-0.089 (so "0.45 to 0.77" and "0.05 to 0.09" are
  both right as written).
- Section 5's "45 percent": pooled over all 148 items, exit-1 top-1 == final top-1 is
  0.453. Correct.
- Section 6 fit-size table: all 8 rows match the four `fitsize_n*/summary.json` exactly.
- Section 6 ‖J‖/√d at 80 prompts [0.159, 0.178, 0.272, 0.810], ratio 5.09: correct, with
  the note that the loop-4 entry averages the 47 *fitted* layers (v=191 is the target and
  has no J); including an identity there gives 0.814. Worth one word in the caption.
- Section 7: every cell of the 4x4 matrix, all six contrasts, both variance-share triples,
  and the main-table/cross-loop-diagonal identity. See section 4 of these notes.
- Section 9: pos-2 multihop J [0.004, 0.007, 0.047, 0.379] and logit
  [0.296, 0.325, 0.455, 0.453] — exact; "72 percent" at pos-2 = 0.723 — exact;
  model-correct 0.352 -> 0.540 — exact.
- Probe family: `model_accuracy_all` 0.779, so "78 percent" is right.

### 7c. **RETRACTED BY ME AT 01:40. Section 8 DOES reproduce.** Read this before the
###      version of 7c that was here for 25 minutes.

I wrote at 01:20 that section 8's table could not be reproduced from any artifact. **That
was my error and I withdraw it in full.** I had read PEER2's notes at their 00:45 revision;
their section 12, written at 00:49, states the aggregation and I did not re-read before
publishing. The table is the **cross-fitted layer choice** (layer picked on the other four
folds, scored on the held-out one) evaluated on the **576-prompt learnable subset**. I
tried each of those corrections singly and neither alone matches; applying both does:

| readout | cross-fitted, all 648 | cross-fitted, predictable 576 | RESULTS.md section 8 |
|---|---|---|---|
| probe | [0.546, 0.525, 0.298, 0.210] | **[0.615, 0.590, 0.335, 0.236]** | [0.615, 0.590, 0.335, 0.236] |
| logit lens | [0.799, 0.511, 0.174, 0.310] | **[0.764, 0.500, 0.179, 0.311]** | [0.764, 0.500, 0.179, 0.311] |
| eventual-exit J-lens | [0.369, 0.469, 0.136, 0.111] | **[0.392, 0.500, 0.148, 0.120]** | [0.392, 0.500, 0.148, 0.120] |

All twelve cells exact. The per-fold spread claim reproduces too: cross-fitted loop-1 probe
accuracy by fold on the 576 subset is [0.558, 0.559, **0.275**, **0.919**, 0.608], sd 0.205,
against RESULTS.md's "ranges from 0.275 to 0.919, a standard deviation of 0.21". So MAIN has
already applied *both* of the corrections I was about to demand (the label-coverage
restriction and a selection-free layer choice), which is more than the text admits.

**What survives is a documentation defect, not a numbers defect.** Section 8's method
paragraph says "every one of the 648 prompts tested exactly once ... Candidate-set top-1,
with a pair-clustered bootstrap". It does **not** say the table is restricted to the 576
learnable prompts, and it does **not** say the layer is cross-fitted. Both are load-bearing:
on all 648 with a per-loop max — the reading the text supports — the same arrays give probe
[0.557, 0.531, 0.316, 0.238] and logit [0.853, 0.511, 0.174, 0.310], and the loop-1 gap is
-0.296 rather than -0.149. A reviewer who follows the stated method gets different numbers
and concludes the paper is wrong. Two sentences fix it, and they should be added.

**What does still stand from my original 7c, and it is the important half.** MAIN quotes
the loop-1 difference as -0.149 [-0.358, +0.001], which does not exclude zero, and then
asserts flatly "The vanilla logit lens beats a supervised probe at loop 1, 0.764 against
0.615", and leans on that assertion twice more as evidence for the C9 mechanism (section 4
"0.764 against 0.615, section 8"; section 7 "on arithmetic it beats a supervised probe,
0.764 against 0.615"). The paper asserts a difference its own quoted interval does not
support. PEER2 is explicit that the pair-clustered interval is the honest one and that the
prompt-level interval (-0.219 [-0.262, -0.175], which does exclude zero) may be quoted only
with the independence caveat stated. Pick one and say which: either soften the three
assertions to match the pair-clustered interval, or quote the prompt-level interval with its
caveat. As it stands, section 4 and section 7 are stronger than section 8's own statistics.

### 7d. The 11% label-coverage handicap — **already corrected by MAIN, disclosure incomplete**

I confirm the handicap independently: folds 0 / 2 / 4 lose labels 4 / {3, 17, 18} / 2, for
24 + 40 + 8 = 72 of 648 test prompts (11.11%), capping the probe at 0.889 while both lenses
are uncapped; the cause is that labels 2, 3, 17 and 18 come from exactly one unordered pair
each and labels 4, 5, 15, 16 from two (fold 0 drew both of label 4's). Section 8 discloses
the 72 and, per 7c, the table already excludes them. So the correction is done. What is
missing is the sentence saying the table is on the 576, plus the note that `chance` is
quoted as 1/17 = 0.059 while the **majority-class rate is 0.111** — which is above the
eventual-exit J-lens's own loop-4 number of 0.120 and level with it inside noise. That last
one is worth stating: at loop 4 the J-lens does not beat predicting the most common sum.
I have pinned the 72 in `utilities/tests/unit/test_peer3_jlens_audit.py` so a change to the
fold seed cannot move it silently.

### 7e. "More training data moved the probe DOWN" is confounded — and **PEER2 and MAIN
###      made the same inference independently**, so nobody caught it. Curve in section 11.

Section 8: "Giving the probe half again as many training pairs moved it *down*, from 0.665
on the single split to 0.615 here, so the logit lens's first-loop advantage is a property
of the representation rather than of the sample. An earlier draft explained the gap as the
probe being data-limited; that explanation is false and has been removed."

0.665 and 0.615 are not comparable. They differ in the training-pair count (24 -> 36), the
**test set** (176 prompts from 11 pairs -> all 648), the number of labels missing from
training (3 of 17 in the single split -> a varying set per fold, 11.1% of prompts), and the
aggregation. PEER2's own `cv_report.py` output shows the confound directly: restricted to
probe.py's original 176-prompt test split, the *cross-validated* probe scores
[0.540, 0.545, 0.347, 0.301] against the published single-split [0.665, 0.653, 0.341,
0.330] — so the drop is mostly still there on the same test prompts, but that is now a
comparison of two different probes on a test set one of them was tuned around. Deleting the
data-limited explanation may well be right; this particular comparison does not establish
it, and a claim that a *supervised* method does not improve with more data deserves a clean
learning curve (fit the CV probe at 12 / 24 / 36 training pairs, same folds) rather than
one before/after pair. That is a 10-minute CPU job on the existing cache and I would run it
before letting the sentence stand.

---

## 8. Two significance facts nobody had put a CI on, both of which matter for section 5

    OMP_NUM_THREADS=4 venv/bin/python (p3, bootstrap over items, 20000 draws, seed 0)

Eventual-exit J-lens excess against its own matched control, with intervals:

| task | statistic | loop 1 | loop 2 | loop 3 | loop 4 |
|---|---|---|---|---|---|
| multihop | mean-over-layers | -0.0004 [-0.0020,+0.0017] | **-0.0065 [-0.0101,-0.0032]** | +0.0159 [-0.0033,+0.0439] | +0.191 [+0.141,+0.244] |
| multihop | any-layer | -0.002 [-0.034,+0.038] | -0.014 [-0.041,+0.021] | +0.089 [+0.022,+0.165] | +0.464 [+0.364,+0.564] |
| arithmetic | mean-over-layers | +0.006 [-0.001,+0.015] | +0.096 [+0.048,+0.148] | +0.087 [+0.034,+0.147] | +0.093 [+0.038,+0.152] |

Two consequences.

**8a. PEER2's "anti-informative at multihop loop 2" is CONFIRMED, on the metric the paper
actually reports.** PEER2 got p = 0.029 from a label-shuffle null; I get -0.0065 CI
[-0.0101, -0.0032] from a bootstrap over items on the mean-over-layers excess, which
excludes zero on the same side. Two different nulls agree. It does *not* hold on the
any-layer statistic ([-0.041, +0.021]), so if MAIN uses it, say which statistic.

**8b. The multihop loop-3 eventual-exit reading, +0.0159, does not exclude zero on
mean-over-layers** ([-0.0033, +0.0439]) though it does on any-layer ([+0.022, +0.165]).
This is the baseline PEER1's whitening result moves (0.016 -> 0.079). A "conditioning
problem at loop 3" is being asserted about a starting value that is itself not
distinguishable from zero on that statistic. One more reason not to build a two-regime
split on it.

---

## 9. Reproducibility gaps in the peers' own work (not defects, but the reviewer will ask)

- PEER2's permutation null (their 2a), their Bonferroni recomputation (5a), their C7
  six-pair table (7) and their fit-size rank bootstrap (4) have **no script on disk**.
  `p2/recompute.py` is a clean independent reimplementation of the *per-layer* metric only
  — it has no any-layer / `ctrl_any` path and no bootstrap. I re-derived their any-layer
  numbers, their least-squares ‖mu‖, their Pearson range and their loop-2 anti-informative
  result independently and all four agree, so I have no reason to doubt the rest; but
  nothing in `p2/` would let a third party re-run the permutation null.
- PEER1's `p1/gpu2.py` was superseded by `gpu2b.py` and both write the *same* output file
  `rescale_allrank.npz`; only gpu2b's version survives. gpu2.py computes the SVD in
  float64 and gpu2b in float32, so the surviving `orth` arrays are the float32 ones. It
  does not change any conclusion (I recomputed the polar factors in float64 and get the
  same picture) but two scripts writing one filename should not both be cited.
- PEER1's bit-exactness verification of the summed branch covers 2 of 191 source layers,
  on 1 prompt at seq 24. It passed exactly, and their `logit` column independently
  reproduces the published per-layer logit lens at all 12 locations, so I am satisfied —
  but "bit-for-bit" should be stated with its scope.

---

## 10. Two C10 items I checked while I was in the launch path (PEER1 owns C10; these are additions)

**10a. PEER1's section 11a is already FIXED and the fix works.** `validate.py:248` now reads
`sys.exit(0 if ok else 1)` and `run_b300.sh` has `set -euo pipefail`, so a failed milestone
aborts before any paid fit. I also checked the one idiom in that script that could have
defeated `set -e` — `[[ -f $out ]] && continue` inside the shard loop — and it is exempt
under POSIX/bash rules (verified by running it): the loop survives a false test.

**10b. NEW: the rented run's documented follow-up cannot reproduce section 8.**
`HANDOFF_B300.md` "After it finishes" says to run

    python src/ouro_jlens/probe.py --lens .../n100/exit3.pt --out artifacts/jlens/probe/n100
    python src/ouro_jlens/analyze.py --eval ... --probe artifacts/jlens/probe/n100

That regenerates the **single-split** probe, which section 8 no longer uses — section 8 is
now the cross-validated `probe_cv.py`. And the two are not interchangeable:

- `analyze.py:probe_summary` reads `meta.json` and the keys `jlens_cand_rank` /
  `logitlens_cand_rank`. `probe_cv.py` writes `summary.json` and the keys `jl_cand` /
  `ll_cand`. `analyze.py --probe <probe_cv dir>` raises FileNotFoundError on `meta.json`.
- `probe_cv.py` needs `probe.py`'s `gpu_cache.npz` (648 x 192 x 2048 fp16, 510 MB) *and*
  its own GPU `--lens-only` pass to rescore both lenses on all 648 prompts with the new
  n=100 lens. Neither is in the handoff.

The correct follow-up is three commands, not two, and the middle one needs a GPU:

    python src/ouro_jlens/probe.py --lens <n100/exit3.pt> --out artifacts/jlens/probe/n100
    python src/ouro_jlens/probe_cv.py --cache artifacts/jlens/probe/n100/gpu_cache.npz \
        --lens <n100/exit3.pt> --out artifacts/jlens/probe/n100_cv --lens-only     # GPU
    python src/ouro_jlens/probe_cv.py --cache artifacts/jlens/probe/n100/gpu_cache.npz \
        --lens <n100/exit3.pt> --out artifacts/jlens/probe/n100_cv                 # CPU

It runs on the local 12 GB card, so it is not a launch blocker — but as written the handoff
would leave the paper's section 8 stuck at the local n=80 lens after the money is spent.
Also worth deciding before launch: whether `probe.py`'s GPU cache should be produced on the
pod (it is ~15 min of GPU there against ~15 min locally, and it needs the n=100 lens on the
card anyway for the `--lens-only` pass) — doing both on the pod is one line in
`run_b300.sh` and removes a 20 GB lens download from the critical path.

**10c. Not a defect, just verify it before you rent:** `setup_b300.sh` clones
`https://github.com/anthropics/jacobian-lens.git`, which is also the origin of the local
checkout. If that repo is not public from the pod, setup dies at line 8 after you are
already paying. One `git ls-remote` from an unauthenticated shell settles it.

---

## 11. **The probe IS data-limited. The retraction in RESULTS.md section 8 — and PEER2's
##     section 12 conclusion that it rests on — are both REFUTED.**

This is the finding I would most want acted on. PEER2 wrote "Was the probe just
under-powered at loop 1? No. ... Going from 24 to 36 pairs moved it from 0.665 to 0.615 ...
more data does not close the gap"; MAIN copied that into section 8 and deleted the
data-limited explanation on the strength of it. Two agents agreeing on one confounded
before/after is exactly the failure mode this swarm exists to catch, so I ran the curve.

I ran the clean learning curve I said I would, because section 8 deleted an explanation on
the strength of a confounded before/after.

    nohup env OMP_NUM_THREADS=6 nice -n 15 venv/bin/python p3/probe_curve.py

Design: probe_cv's own 5 folds, the **same test prompts**, the **same fixed 8-pair
validation set per fold** for choosing C, the same 48 loop-1 locations, the same
`StandardScaler` + `LogisticRegression(max_iter=2000)` + `C_GRID`. The *only* thing that
varies is how many of the 28 non-validation training pairs the probe sees, taken as nested
prefixes of one permutation. 9.5 min on 6 cores.

| training pairs | loop-1 top-1 at its own best layer | pair-clustered CI95 | mean over L19..L47 |
|---|---|---|---|
| 7 | 0.185 (L41) | [0.089, 0.296] | 0.157 |
| 14 | 0.349 (L29) | [0.220, 0.487] | 0.300 |
| 21 | 0.383 (L41) | [0.249, 0.519] | 0.332 |
| 28 | 0.443 (L37) | [0.310, 0.575] | 0.382 |
| 36 (probe_cv's full probe, refit on train+val, same all-648 statistic) | 0.557 | | |

Five points, monotone across the whole range, and the top point is the very configuration
PEER2 and MAIN called "more data, which moved it down": 0.185 -> 0.349 -> 0.383 -> 0.443 ->
0.557 at 7 / 14 / 21 / 28 / 36 training pairs, every number on all 648 test prompts at each
probe's own best loop-1 layer.

Paired over prompts, pair-clustered bootstrap:

    28 pairs - 7 pairs   +0.258 [+0.117, +0.402]   excludes zero
    28 pairs - 14 pairs  +0.094 [+0.011, +0.192]   excludes zero
    28 pairs - 21 pairs  +0.060 [-0.004, +0.134]   still positive, still climbing at the top

**Monotone, significant, and not saturated at the largest training set the family allows.**

So section 8's sentence — "Giving the probe half again as many training pairs moved it
*down*, from 0.665 on the single split to 0.615 here, so the logit lens's first-loop
advantage is a property of the representation rather than of the sample. An earlier draft
explained the gap as the probe being data-limited; that explanation is false and has been
removed" — is wrong, and the deleted explanation was right. The 0.665 -> 0.615 comparison
changed the training size, the test set (176 prompts from 11 pairs -> all 648), the
fraction of test prompts whose label is absent from training, and the aggregation, all at
once; the drop came from the test set, not from the training size.

This does **not** overturn the headline. The logit lens still beats the probe at loop 1 by
-0.219 [-0.260, -0.175] on the predictable subset, and the probe would need roughly to
double again to close that. But the *reason* stated in the paper is wrong, and it is the
reason that carries the mechanism argument in sections 4 and 7. The defensible wording is:

  "The logit lens beats the cross-validated probe at loop 1 by 0.22 (CI excludes zero) on
  the 45-pair family. The probe is still improving with training pairs over the whole range
  the family allows (+0.094 CI [+0.011, +0.192] from 14 to 28 pairs), so this is a gap at
  the sample sizes available, not a demonstration that no supervised probe could close it.
  A larger operand family would settle it and is not in this submission."

Recommend deleting "that explanation is false and has been removed" and the "not a power
artifact" sentence in section 4. Data and script: `p3/probe_curve.{json,npz,py,log}`.
