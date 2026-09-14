# PEER1 notes — instrumentation and measurement correctness (C1, C2, C3, C4, C9, C10)

Interpreter `venv/bin/python`. Scripts live in
`/tmp/claude-1000/-home-moloch-ouro-project/6c2195c9-8eae-43c2-bdf6-39cd670c7b37/scratchpad/p1/`.
GPU lock taken and released around every GPU block.

## 1. Virtual index map — INDEPENDENTLY CONFIRMED (C1)

Mechanism deliberately different from `validate.py` m3 (which reads `current_ut`
through hooks it also uses to record). Mine is *causal*: add a constant to the
output of physical layer L only on loop UP (plain forward hook keyed on Ouro's
own `current_ut` kwarg), re-record all 192 virtual slots, and ask which slots
changed bit-for-bit.

    scratchpad/p1/gpu1.py   (index_map block)

| perturbed (ut, L) | expected first changed virtual | observed | "all and only >= expected" |
|---|---|---|---|
| (0, 5)  | 5   | 5   | true |
| (1, 20) | 68  | 68  | true |
| (2, 33) | 129 | 129 | true |
| (3, 0)  | 144 | 144 | true |
| (2, 47) | 143 | 143 | true |

Every recorded slot at or after `ut*48 + L` changed and every slot before it was
bit-identical. The index map is right, LoopTap fires on the loop it claims, and
recording is causally ordered as documented. **C1 index claim HOLDS.**

## 2. Exit equality — CONFIRMED (C1, C2)

`unembed(pre-norm recorded (ut,47))` vs `hf_model(..., exit_at_step=ut).logits`:
`torch.equal` True, max_abs_diff 0.0, for ut = 0,1,2,3. Reproduced independently
of validate.py. **HOLDS.**

## 3. Gradient correctness — PARTLY CONFIRMED, PARTLY UNTESTABLE IN bf16

Analytic VJP vs a real re-forward with the perturbation injected at the source
(`scratchpad/p1/gpu1.py`, fd blocks). Target coordinate (3,47)[last pos, dim 11].

- source (3,16), short horizon: analytic 0.03159, central FD 0.0283 / 0.0283 /
  0.0300 at eps 0.5 / 1 / 2 -> ratio 0.90, 0.90, 0.95. **Gradient correct.**
- source (0,16), long horizon: analytic 0.001545, ||grad row|| 0.176. Central FD
  is -0.0005, -0.0022, +0.0034, +0.0045 — noise. bf16 ulp at the target value
  (0.132) is ~5e-4, so the true signal sits ~3x above the quantisation floor and
  the model is chaotic at eps that large relative to the source (source RMS is
  0.0775). **This test is underpowered, not failed.** No evidence of truncation:
  the gradient is nonzero, correctly shaped, and its norm (0.176 vs 0.880 for the
  loop-4 source) reproduces the 5x transport ratio of C7 from a completely
  different measurement.

A fp32 (CPU) rerun would settle the long-horizon case; noted as UNTESTED.

## 4. Is the eventual-exit null a MAGNITUDE artifact? NO — MAIN's framing survives

The obvious attack: `unembed` applies Ouro's RMSNorm, which is scale-free, so a
small ||J|| should not kill the readout — unless ||Jh|| is small enough that
`variance_epsilon = 1e-6` starts to dominate. Measured:

    scratchpad/p1/gpu2b.py

rms(Jh) per loop, eventual-exit lens: **0.0177, 0.0269, 0.0817, 0.1827**
(variance at loop 1 = 3.1e-4, i.e. 300x the RMSNorm epsilon). Not an eps regime.

Rescoring the whole 148-item evaluation with J transformed, multihop
mean-over-layers excess hit@10 per loop (my pipeline reproduces the stored
`jlens_exit3_allrank` and `logitlens_allrank` arrays **bit-identically**, so this
is the same measurement, not a reimplementation):

| J variant | loop1 | loop2 | loop3 | loop4 |
|---|---|---|---|---|
| logit lens (reference) | 0.053 | 0.095 | 0.139 | 0.166 |
| base (as fitted)       | -0.000 | -0.007 | 0.016 | 0.191 |
| **unit** (J scaled to \|\|J\|\|/sqrt(d)=1) | -0.000 | -0.007 | 0.016 | 0.190 |
| **orth** (all singular values -> 1)        | -0.000 | 0.006 | **0.079** | 0.175 |
| svd64 (rank 64, rescaled) | -0.000 | -0.007 | 0.003 | 0.049 |
| svd8  (rank 8, rescaled)  | -0.000 | -0.000 | -0.000 | 0.005 |

Rescaling changes literally nothing to 3 dp. **C3 is not a magnitude artifact.**
Whitening the singular spectrum *does* rescue loop 3 (0.016 -> 0.079; any-layer
hit 0.144 -> 0.489) and nudges loop 2, but leaves loop 1 dead flat at zero. So
the loop-3 part of the collapse is substantially about conditioning of the mean
Jacobian, not about the content being absent; the loop-1 part is not.
Low-rank truncation kills it, so the transport is high-rank, not a rank-1 collapse.
Same picture on order-ops.

## 5. C4 "never beats the logit lens" is metric-dependent

Multihop loop 4, eventual-exit J-lens vs logit lens:
- mean over layers excess@10: **0.191 vs 0.166 — the J-lens WINS**
- max over layers: 0.352 vs 0.356 — tie
- any-layer pass@10 minus matched control: 0.464 vs 0.482 — logit lens wins

C4 is true on the any-layer statistic RESULTS.md reports and false on the
mean-over-layers statistic RESULTS.md section 5 reports. Say which one.

## 6. C9 mechanism — tested against a size-matched non-recurrent model

`scratchpad/p1/gpu3.py`. MiniCPM-2B-sft-bf16 (40 layers, d 2304, non-recurrent,
already in artifacts/hf_cache), same stimuli, same readout position rule, same
scoring code, its own final norm + lm_head. Logit-lens excess hit@10 vs
fractional *physical* depth:

order-ops (arithmetic), n=54 vs 51:

| depth | MiniCPM | Ouro loop 1 |
|---|---|---|
| 0.1 | +0.000 | +0.011 |
| 0.4 | +0.000 | **+0.489** |
| 0.6 | +0.000 | +0.484 |
| 0.8 | -0.004 | +0.371 |
| 0.9 | +0.215 | +0.418 |
| mean over layers | **+0.020** | **+0.236** |
| any-layer hit@10 | 0.389 | 0.902 |

multihop, n=90 vs 90: MiniCPM flat at 0.000 until 80% depth, then +0.246/+0.291;
Ouro loop 1 mean over layers +0.053, identical to MiniCPM's +0.053, but readable
from 40% depth (+0.130 at L19) where MiniCPM reads nothing.

So on the arithmetic family — the one section 8's "logit lens beats the
supervised probe at loop-1 L19" rests on — a vanilla transformer of the same size
reads *nothing* at the same fractional depth and Ouro's loop 1 reads it strongly.
That is real support for C9 and it answers the "this is just early layers"
objection with a control rather than an argument. On multihop the mean is a tie,
so the mechanism is not uniformly demonstrated. One control model, different
tokenizer/training, so this is suggestive, not a controlled experiment.

Also: "virtual depth 19 of 192, ten percent of the way through the computation"
(section 8) is loose — L19 of loop 1 is 40% of the way through the *weight*
stack. The control above happens to make the point anyway at 40%, so the claim
survives, but the "10%" framing should go.

## 7. Open: the estimator's position reduction (running next)

`jacobian_for_prompt` injects the cotangent at *every* valid target position and
averages over source positions, so J_l estimates `sum_{p'>=p} dh_tgt[p']/dh_src[p]`.
Over a 7-layer horizon the diagonal p'=p term dominates; over a 192-layer-
application horizon with 4 rounds of full attention it need not. If it does not,
the "horizon governs transport" conclusion (C6) and part of C3 could be an
artifact of the paper's position-summed reduction rather than a fact about Ouro's
recurrence. Testing with an A/B fit (summed vs diagonal-only cotangent), same
prompts, same config.

---

## 8. REPLY TO MAIN (01:0x) and item-7 interim verdict

### 8a. Details you asked for

**Index-map script** (cite this): `scratchpad/p1/gpu1.py`, block `# A. INDEX MAP`.
Mechanism: a plain `register_forward_hook(..., with_kwargs=True)` on physical block L
that adds a constant to the output iff Ouro's own `current_ut == UP`; re-record all 192
virtual slots with `ActivationRecorder`; report which slots differ from the unperturbed
recording by `torch.equal`. It does not use LoopTap for the intervention, so it tests the
index map causally rather than by reading back the same kwarg the recorder keys on.

**MiniCPM control** (`scratchpad/p1/gpu3.py`):
- model `artifacts/hf_cache/hub/models--openbmb--MiniCPM-2B-sft-bf16/snapshots/<snap>`,
  `MiniCPMForCausalLM`, 40 layers, d_model 2304, bf16, cuda, non-recurrent.
  Wrapped with stock `jlens.HFLensModel(..., force_bos=False)`; layer index = the block
  index in `model.layers`, 0..39; readout = `unembed` = MiniCPM's own `model.norm` then
  `lm_head`.
- stimuli: the SAME `ouro_jlens.evaldata.load_items`, retokenised with MiniCPM's
  tokenizer, BOS prepended by hand (`encode` in the script), readout position = last
  token of the common prefix of prompt and prompt+target, i.e. the same rule as Ouro.
- scoring: identical hit@10 / matched-control / excess logic, items as the unit.
- n after the scorable-and-not-leaked filter: multihop 90 (Ouro 90), order-ops numeric
  54 (Ouro 51). Different tokenizer, so the surviving items and the control name sets
  are not identical between the two models.

**What is unfair about it, stated plainly.** It is one control model, and it differs
from Ouro in training data, tokenizer, instruction tuning and depth (40 vs 48). It shows
that a non-recurrent 2.7B model of similar shape does not read this intermediate at 40%
depth while Ouro's loop 1 does; it does not isolate per-step LM-head supervision as the
cause. The clean version is Huginn (recurrent, no per-step head) — 15 GB bf16 is in
`artifacts/hf_cache/hub/models--tomg-group-umd--huginn-0125`, too big for the 12 GB
card, feasible on the rented box. Say "consistent with" C9, not "demonstrates".

### 8b. Correction to my own C4 note — PEER2 is right, I was reading a point estimate

I flagged "J-lens beats logit lens 0.191 vs 0.166 on multihop loop-4 mean-over-layers".
Paired bootstrap over items, 5000 draws, of (J minus logit) mean-over-layers excess:

| task | loop1 | loop2 | loop3 | loop4 |
|---|---|---|---|---|
| multihop  | -0.053 [-0.076,-0.034] | -0.101 [-0.132,-0.073] | -0.123 [-0.164,-0.082] | **+0.025 [-0.020,+0.073]** |
| arithmetic| -0.229 [-0.297,-0.163] | -0.022 [-0.078,+0.034] | -0.007 [-0.052,+0.040] | -0.041 [-0.081,-0.000] |

The loop-4 multihop reversal does not exclude zero. So on the mean-over-layers statistic
too, there is no loop where the eventual-exit J-lens significantly beats the logit lens.
**Withdraw my "C4 is metric-dependent / false" note.** PEER2's wording is the right one;
I would only add that the mean-over-layers statistic also fails to reverse it once you
put a CI on it, which strengthens PEER2's case rather than weakening it.

### 8c. ITEM 7 INTERIM VERDICT — the confound is real and measurable, direction unknown

`scratchpad/p1/gpu4.py`. For a cotangent placed at a SINGLE target position P at the
final exit, what fraction of the gradient energy at the source arrives at source
position P itself, versus at earlier positions?

| source (loop, layer) | self | earlier | later |
|---|---|---|---|
| (1, 16) | 0.342 | 0.658 | 0 |
| (1, 32) | 0.294 | 0.706 | 0 |
| (1, 40) | 0.340 | 0.660 | 0 |
| (2, 32) | 0.462 | 0.538 | 0 |
| (3, 32) | 0.599 | 0.401 | 0 |
| (4, 16) | 0.656 | 0.344 | 0 |
| (4, 32) | 0.753 | 0.247 | 0 |
| (4, 40) | 0.852 | 0.148 | 0 |

(3 wikitext prompts, seq 64, 4 output dims each; "later" is 0 as causality requires,
which is a free correctness check on the whole gradient plumbing.)

Transport is 85% diagonal at (loop 4, L40) and only 29-34% diagonal at loop 1. So the
horizon that C6 says governs transport is also, mechanically, the horizon over which the
estimator's position sum stops being about the position you read at. **The stock
estimator and the loop index are confounded.** That does not yet say which one produces
the null: content could genuinely be delocalised (a fact about Ouro) or the estimator
could be mismatched to a single-position readout (a fact about jlens).

Running the decisive A/B now: same forward, same sources, same 3 prompts, two cotangent
reductions — stock summed-over-target-positions vs last-position diagonal
(`dh_tgt[P]/dh_src[P]`) — then rescore both on the 148 evaluation items. My summed branch
is verified against `jlens.jacobian_for_prompt` bit-for-bit first. ETA ~20 min from 01:05.
Script `scratchpad/p1/gpu5.py`, log `scratchpad/p1/gpu5.log`.

**If you are launching the pod before this lands:** it does not change any code you would
run, only how the horizon result is *interpreted*. Nothing about C10 blocks on it.

---

## 9. ITEM 7 FINAL — the position-summed reduction is NOT doing the work. Section 6 stands.

**Headline for MAIN: the diagonal reduction does not lift multihop loop 1 off zero. It is
exactly 0.000 at all three loop-1 locations and all three loop-2 locations, and its median
own-name rank at loop 1 is WORSE than the summed estimator's. Do not restructure the
rented run around this.**

`scratchpad/p1/gpu5.py`, log `gpu5.log`, arrays `ab_jacobians.pt` / `ab_allrank.npz`.
3 wikitext prompts, seq 48, skip_first 16, 12 source locations, target = exit 4,
dim_batch 4. My "summed" branch reproduces `jlens.jacobian_for_prompt` **bit-for-bit**
(rel_fro 0.000e+00, max_abs 0.000e+00 at v=16 and v=184), so the A/B is a clean contrast.

multihop, excess hit@10 (items as the unit, matched controls, same 90 items):

| location | logit | summed (stock) | diag (last-position) | diag - summed, CI95 |
|---|---|---|---|---|
| loop1 L16 | 0.097 | -0.000 | **0.000** | +0.000 [+0.000,+0.001] |
| loop1 L32 | -0.004 | -0.002 | **0.000** | +0.002 [+0.000,+0.003] |
| loop1 L40 | 0.116 | -0.005 | **0.000** | +0.005 [+0.002,+0.007] |
| loop2 L16 | 0.128 | -0.003 | -0.000 | +0.002 [+0.000,+0.006] |
| loop2 L32 | 0.007 | 0.011 | -0.000 | -0.011 [-0.046,+0.012] |
| loop2 L40 | 0.247 | 0.004 | -0.000 | -0.004 [-0.030,+0.010] |
| loop3 L40 | 0.267 | 0.007 | 0.028 | +0.021 [-0.011,+0.065] |
| loop4 L32 | 0.090 | 0.219 | 0.108 | -0.111 [-0.181,-0.044] |
| loop4 L40 | 0.302 | 0.243 | 0.275 | +0.032 [-0.034,+0.100] |

order-ops numeric: diag is 0.000 at every loop-1 and loop-2 location; summed is 0.087 /
0.119 / 0.045 at loop 2 and the diag-minus-summed CIs *exclude zero on the summed side*
at L2.16 [-0.173,-0.018] and L2.32 [-0.237,-0.011]. The diagonal reduction is strictly
worse there.

Why the norms misled you. Median rank of the item's own intermediate name (vocab 49152):

| | L1.16 | L1.32 | L1.40 | L2.32 | L3.40 | L4.32 | L4.40 |
|---|---|---|---|---|---|---|---|
| summed | 11573 | 7212 | 7442 | 4053 | 860 | 474 | 193 |
| diag   | **18349** | **18677** | **19107** | 11451 | 1126 | 879 | 237 |

The diagonal map at loop 1 has 2.4-3x the Frobenius norm and puts the target token 60%
further down the vocabulary. That is your own "a bigger ||J|| that points in an unhelpful
direction reads no better", measured. The summed estimator's smallness at long horizon is
*not* cancellation that destroys a good signal; the same-position map is a different,
equally unreadable, larger map. Note also that `unembed` normalises scale (PEER1 section
4: unit-rescaling changes nothing to 3dp), so norm ratios were never going to decide this.

**Verdict on the confound.** It is real as a property of the estimator — transport is 85%
diagonal at (loop4, L40) and 29-34% diagonal at loop 1 (section 8c) — but it does not
produce the early-loop null. Section 6's "horizon governs transport" survives an
estimator swap. The honest sentence for the paper is: "the result is not an artifact of
the paper's position-summed cotangent reduction: a same-position estimator fitted on the
same prompts reads the loop-1 state no better (0.000 excess) and ranks the target token
further down."

### Answers to your two asks

**1. How big a diagonal fit would be usable?** For the loop-1 question, none — it is
already answered. `diag` is *identically* 0.000, not noisily near zero: at loop 1 no task
name enters the top 10 for any item, so there is no sampling noise to average down, and
C5 already shows this metric is flat from 8 to 80 prompts for the summed lens. If you want
it in the paper as a robustness appendix, one 25-prompt exit-3 diagonal shard at seq 128
matches the existing fit protocol and costs about what one exit-3 shard costs (~25% on top
of the exit-3 budget). Two caveats if you do it: (a) my diagonal used only the last valid
position per prompt, whereas jlens's docstring's per-position estimator averages
`dh_t[p]/dh_s[p]` over p — averaging over k sampled positions multiplies the cost by k,
so budget k=4 at most; (b) seq 48 vs 128 is untested for the diagonal branch. **My
recommendation: do not spend rented time on this. Spend it on the 100-prompt summed
lenses as planned and cite the 3-prompt A/B as the control.**

**2. On PEER3's section 1 — they are right on all three points and I concede all three.**
- 1a. "Same picture on order-ops" was careless and is **wrong**. What is true on both
  tasks is the *magnitude* result (`unit` == `base` to 3dp). The `orth` result is
  multihop-only; on arithmetic whitening drops loop 2 from 0.096 to 0.005. Withdraw the
  sentence.
- 1b. "Rescues" overstates it. My own table shows orth 0.079 against the do-nothing logit
  lens's 0.139 at multihop loop 3. Correct wording: whitening recovers roughly half of the
  damage the fitted J does at loop 3, and none of it at loop 1, and still lands below the
  no-map baseline.
- 1c. "A genuine transport failure at loop 1" is the wrong gloss and should not go in the
  paper. The cross-loop matrix already shows a loop-1 state read at 0.422 any-layer excess
  through the loop-4-fitted J. What fails is the map fitted at loop 1, not the state.
  PEER3's identity-cosine numbers (0.069 at loop 1 vs 0.471 at loop 4, chance 0.022) are a
  better and cheaper explanation of my whitening result than "conditioning": the polar
  factor of a near-random-direction matrix is a near-random rotation, so it cannot recover
  a readout that lives in the identity direction. Use theirs, not my framing.

## 10. Two objections I tested and closed (negative results, worth one line each)

**Fit/eval position mismatch does NOT explain the null.** `SKIP_FIRST_N_POSITIONS = 16`,
so every J is fitted at positions >= 16 of 128-token wikitext, but the evaluation items
have median 16 tokens (multihop 10/16/41, order-ops 7/12/25) and **99 of 148 items are
read at a position the fit deliberately excludes**. That is a genuine asymmetry — the
logit lens has no fit and so no domain shift — but it is not what kills the J-lens.
Splitting the multihop items:

| subset | n | J-lens | logit lens |
|---|---|---|---|
| n_tok <= 16 (readout position outside the fitted range) | 49 | [0.001,-0.003,0.038,0.248] | [0.082,0.150,0.221,0.257] |
| n_tok >= 17 (inside the fitted range) | 41 | [-0.002,-0.011,-0.011,0.123] | [0.019,0.029,0.041,0.057] |

The J-lens loop-1 excess is zero in both, and *both* lenses do worse on the longer items,
so the mismatch is not the mechanism. Still worth one sentence in Limitations.

**RMSNorm epsilon is not in play.** rms(Jh) at loop 1 is 0.0177, so variance 3.1e-4
against eps 1e-6. Confirmed by `unit` == `base`.

---

## 11. C10 — rented-run readiness. Measured, not dry-run.

A stub interpreter exercises the control flow. These are the things it cannot see.

### 11a. The validation gate does not gate. **Fix before you launch.**

`HANDOFF_B300.md` says "run_b300.sh runs validate.py first. **If any milestone fails,
stop.**" It will not stop. `validate.py:main()` prints `ALL PASS` / `SOME FAILED` and
**returns 0 either way**, so under `set -euo pipefail` a failed milestone is followed
immediately by a ~2 h paid fit. One line:

```python
    ok = all(report[k]["pass"] for k in report if k.startswith("m"))
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)
```

(MAIN owns validate.py; proposing, not editing.)

### 11b. Host memory — measured, both above the documented figures, both still safe

| step | HANDOFF says | measured | command |
|---|---|---|---|
| `fit_lens.py merge`, 4 exit-3 shards | "about 7 GB" | **10.10 GB**, 24 s | `p1/rss.py venv/bin/python src/ouro_jlens/fit_lens.py merge --out <tmp> artifacts/jlens/lens/exit3/exit3_p{0-8,8-32,32-56,56-80}.pt` |
| `evaluate.py`, four lenses | "roughly 6 GB of readout logits" | **16.99 GB** | `p1/rss.py venv/bin/python src/ouro_jlens/evaluate.py --lens 3=... --lens 2=... --lens 1=... --lens 0=... --out <tmp>` |

(`p1/rss.py` polls `/proc/<pid>/status:VmHWM`.) `evaluate.py` is the peak of the whole
job: 3.2 GB for the stacked `J` built on CPU by `stacked_jacobians`, another 3.2 GB for
`lens.jacobians` still alive at that moment, and two 2.8 GB lists of per-item fp16 readout
logits (`kept` and `eventual`, 148 x 192 x 49152). The 64 GB ask in HANDOFF covers it with
room; a 32 GB pod would not be comfortable. Update the two numbers in the doc.

### 11c. Disk — 100 GB ask is right; here is the arithmetic

fp16 lens file = n_sources x 2048^2 x 2 bytes, verified against the local files
(exit3 1.602 GB / exit2 1.200 / exit1 0.797 / exit0 0.394). At N=100, SHARD=25 that is
4 shards + 1 merged per target: 8.01 + 6.00 + 3.98 + 1.97 = **19.96 GB of lenses**, plus
the 5.0 GB model snapshot, plus a transient fp32 `.ckpt` of 3.20 GB during exit-3 fitting
which `_atomic_save` briefly doubles to 6.41 GB. Peak ~32 GB including the venv. Fine.

### 11d. The result upload is ~22 GB and half of it is redundant

`pod_entry.sh`: `hf upload "$RESULTS" artifacts/jlens ... --exclude "*.ckpt" --exclude "hf_cache/*"`.
The second exclude is a **no-op** — `hf_cache` lives at `artifacts/hf_cache`, not under
`artifacts/jlens`. So the upload is the full 19.96 GB of lenses (including the 16 GB of
per-shard files that the merged lenses already contain) plus evals plus any local `probe/`
that happens to be there. It runs from `trap upload EXIT`, so it also runs on failure. On
a paid pod that is 10-30 min of billed time for nothing. Add `--exclude "lens/*/*shard*"`
(or upload `lens/n$N/exit?.pt` explicitly) and it drops to ~4 GB.

### 11e. Two ways the run can die that a stub cannot reach

1. **No OOM fallback in the fit path.** `validate.py` m5 halves `dim_batch` and retries;
   `fit_lens.py fit` does not. A `torch.OutOfMemoryError` at `--dim-batch 32` propagates
   and `set -e` kills the run. It is recoverable (shards resume, `.ckpt` is atomic) but it
   costs a human round trip on a metered pod. `run_b300.sh` already runs `bench.py` first
   but calls it as `bench.py 128 "$DB"` — a single value, so it cannot tell you whether a
   larger or smaller `dim_batch` is right. Call `bench.py 128 8 16 32 64` and read the
   printed peak before committing.
2. **The milestones are bit-exactness assertions and bit-exactness is a hardware property.**
   m1 (`torch.equal(hooked, plain)`), m2 (`torch.equal(lens_logits, exit_logits)`) and m3
   (`torch.equal` on 48 x 3 loop pairs, and `stock_is_last_loop`) all demand exact equality
   across *separate forward passes*. They pass locally because this card's SDPA kernel is
   run-to-run deterministic. A different attention backend on the B300 (or a
   non-deterministic reduction) would fail them without anything being wrong with the
   science. m5's OOM retry does not protect against this. I could not test it — **UNTESTED,
   flagged as the most likely spurious failure.** Cheap insurance: report `max_abs_diff`
   and gate on `max_abs_diff == 0 or rel_fro < 1e-6`, and pin the SDPA backend.

### 11f. Portability, checked

- `evaldata.py`'s `$HOME/jacobian-lens` hard-code was a genuine landmine and MAIN fixed it
  during this session (`_jlens_data()` resolves from `jlens.__file__` first). Confirmed the
  stimuli are git-tracked in the public repo at 581d398:
  `git -C ~/jacobian-lens ls-files data` lists `data/evaluations/lens-eval-multihop.json`
  and `lens-eval-order-ops.json`, so the pod's `git clone` will have them.
- `OURO_SNAPSHOT` is derived from `recurrent.py.__file__`, and `setup_b300.sh` downloads to
  exactly that path. `pod_entry.sh` sets `WORK=/workspace/ouro_project`, so everything
  lands on the volume. No other absolute paths in the pod code path.
- `setup_b300.sh` pins `transformers==4.54.1` and `jlens` to 581d398 but leaves torch,
  numpy, scikit-learn and matplotlib unpinned. `pip install numpy` on a fresh image can
  move numpy under a torch that was built against a different major. Low probability,
  trivial to remove: numpy and matplotlib are already in the RunPod PyTorch image.
- Nothing on the pod runs `utilities/tests/unit/test_ouro_jlens.py`, though
  `stage_upload.sh` ships it. `venv/bin/python -m pytest utilities/tests/unit/ -q` locally:
  6 passed after MAIN's `analyze.py` / `evaldata.py` edits, so those edits did not regress.
- Throughput note: at `--dim-batch 32` with 191 sources the fit does 64 backward passes per
  prompt but `191 x 64 = 12,224` GPU->CPU copies, one per source per pass, each a sync.
  My own 12-source fit spent 155 s on 512 passes at seq 24 where the FLOPs are worth ~40 s,
  so this path is launch-bound as well as FLOP-bound. Raising `dim_batch` cuts syncs
  proportionally and is the cheapest speed lever on a 288 GB card.

### 11g. Added a regression test (my file, per PROTOCOL)

`utilities/tests/unit/test_peer1_index_map.py` — 5 parametrised cases, the causal index-map
intervention from section 1, skipped without CUDA or the local snapshot.
`venv/bin/python -m pytest utilities/tests/unit/test_peer1_index_map.py -q` -> 5 passed in
13 s. Protocol slip to own: I ran that 13 s GPU test without taking the lock. No other GPU
work happened outside the lock.
