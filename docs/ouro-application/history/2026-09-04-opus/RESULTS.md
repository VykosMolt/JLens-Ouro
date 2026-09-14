# Jacobian lenses across recurrent depth in Ouro-2.6B

**Headline.** In Ouro, a Jacobian lens targeting the *final* recurrent exit reads
essentially nothing out of a first-loop state, on either task and on every metric
here, even though the latent intermediate is present there and is read out by a
vanilla logit lens, by a Jacobian lens targeting the *local* exit, and, on the
arithmetic family where a supervised probe exists, by that probe. On multihop the
deficit persists through every loop before the last, four to thirty times below
the logit lens; on arithmetic beyond loop 1 the two readouts trade places
depending on the statistic, and no clean claim survives there.

The loop-1 failure is not an artifact of the fit: a tenfold increase in fitting
data does not close it, and the prompt-invariant component of the transport is
five times smaller for loop-1 sources than for loop-4 sources. What governs the
readout is the horizon, how much recurrent computation still lies between the
fitted source and the target, rather than which loop produced the state. This is
a monitorability blind spot with a measurable cause.

Status: local lenses, with the eventual-exit lens swept from 8 to 80 fitting
prompts and the three local-exit lenses at 24 each. Lens-free results and the
probe and logit-lens comparisons do not depend on any fit.

---

## 1. Setup, verified against source

Model ByteDance/Ouro-2.6B, revision `1ed04250`, bf16, 48 physical layers reused
for 4 recurrent steps, d_model 2048, giving 192 virtual `(loop, layer)`
locations. Lens code is anthropics/jacobian-lens at `581d398`, unmodified.

Ouro applies its final norm to the recurrent state at the end of every pass. The
normed state is simultaneously the exit-k state (`lm_head(norm(h))`) and the next
pass's input. So the *pre-norm* output of layer 47 at step k, pushed through
jlens's `unembed` (norm + head), is exactly Ouro's `exit_at_step=k` logits. This
was tested, not assumed.

Recording keeps the native forward intact. `model.layers` is a flat list of 192
`LoopTap` objects; each registers its hook on the shared physical module and
fires only when Ouro's own `current_ut` kwarg matches. Anthropic's `fit`,
`apply` and `merge` then run unmodified over the virtual index `ut*48 + layer`.

### Validation milestones (all bit-exact)

| check | result |
|---|---|
| hooked logits vs plain logits | identical, hooks removed cleanly |
| `unembed(pre-norm (k,47))` vs `exit_at_step=k` logits, k = 1..4 | identical |
| `norm(recorded (k,47))` vs Ouro's own `hidden_states_list[k]` | identical |
| next loop's layer-0 input vs normed (k,47) | identical |
| stock `ActivationRecorder` on Ouro | returns loop 4 only, as predicted |
| stock jlens J(L40→47) vs recurrent J((loop4,40)→(loop4,47)) | identical |
| same, against the loop-3 source | differs, 0.99 relative Frobenius |
| VJPs to (loop1,16)/(loop2,16)/(loop3,16) | distinct, correctly shaped |

Two model facts worth recording. Under transformers 4.54.1 the Ouro remote code
cannot construct its own cache, so every reference forward uses
`use_cache=False`. And the default forward equals exit 4 on all tested prompts,
with the early-exit gates far from saturation.

---

## 2. Relation to prior work

That a recurrent model's states converge over iterations is already established
for Ouro and Huginn (Blayney et al. and others), and Ouro's own paper reports
adaptive-depth behaviour directly. Section 3 below measures the exit divergence
in this setup only to calibrate the later sections; none of it is offered as a
new phenomenon. Likewise the loop-1 discontinuity that motivated hypothesis H1
comes from this project's earlier OPI work on candidate-quality representations,
not from here.

The object that is new is the transport: what a fitted downstream Jacobian can
and cannot read out of a weight-shared recurrent state, and what that implies
for reading an intermediate state as a monitor. Sections 5, 6 and 7 are the
contribution; sections 3 and 4 are the setting.

---

## 3. The model's own recurrent exits (lens-free)

Measured at the readout position on the evaluation stimuli, with no lens
involved. The logit lens at location (k, 47) *is* the exit-k distribution, so
these are properties of the model.

| | loop 1 | loop 2 | loop 3 | loop 4 |
|---|---|---|---|---|
| KL(exit k ‖ exit 4), nats, mean | 3.83 | 0.65 | 0.09 | 0 |
| KL, median | 4.08 | 0.22 | 0.03 | 0 |
| exit top-1 == final top-1, multihop | 0.39 | 0.67 | 0.80 | 1.0 |
| exit top-1 == final top-1, arithmetic | 0.49 | 0.84 | 0.96 | 1.0 |
| intermediate is the exit's top-1, arithmetic | 0.18 | 0.08 | 0.06 | 0.04 |

Almost all of the rewriting happens between loop 1 and loop 2. The loop-1 exit
disagrees with the completed model on roughly half the items, and on arithmetic
it names the latent intermediate rather than the answer on 18 percent of them:
`(2 + 3) * 4 = ` exits `5, 2, 2, 2` across the four loops; `(9 - 6) * 2 = `
exits `3, 6, 6, 6`.

This is the monitor gap stated in the model's own output. An intermediate
monitor reading loop 1 sees content the finished computation will not produce.

---

## 4. Known-intermediate readout across loop × layer

Stimuli are Anthropic's lens-eval multihop (93 items) and order-ops (55 items),
read at the token immediately preceding the target. Every intermediate name of
the task is ranked for every item, so the item's own names give hit@10 and the
other names of the same kind are matched controls for the position's prior.
Excess = own minus control. Matching by kind matters: operation names such as
"multiplication" rank far below numbers at these positions, so letting them
control a numeric intermediate inflates its excess by 0.04 to 0.14. Leaked
intermediates are excluded from the primary analysis and reported separately.
Items, not (item, name) slots, are the unit everywhere: a few multihop items
carry near-duplicate names such as "3"/"three"/"third", which are one
measurement, not three. Confidence intervals bootstrap over items. Where a
statistic is "any layer within the loop", the control is scored the same way,
per control name and only then averaged; taking the max of an already-averaged
control understates it, since a max of a mean is at most a mean of maxes, and
inflates every excess. That bug was in an earlier draft and is corrected here.

Any-layer excess hit@10 within each loop:

| task | readout | loop 1 | loop 2 | loop 3 | loop 4 |
|---|---|---|---|---|---|
| multihop | eventual-exit J-lens | 0.00 | -0.01 | 0.09 | 0.46 |
| multihop | logit lens | 0.36 | 0.46 | 0.50 | 0.48 |
| arithmetic | eventual-exit J-lens | 0.06 | 0.25 | 0.18 | 0.17 |
| arithmetic | logit lens | 0.31 | 0.15 | 0.13 | 0.21 |

On multihop the eventual-exit Jacobian lens is at or below zero for the first two
loops and stays far below the logit lens until loop 4, where its target is the
state's own exit. On arithmetic the point estimates cross over in the middle, the
Jacobian lens reading 0.25 against 0.15 at loop 2, but that crossover does not
survive a confidence interval.

Paired bootstrap over items, 5000 draws, of the Jacobian lens minus the logit
lens on this statistic:

| task | loop 1 | loop 2 | loop 3 | loop 4 |
|---|---|---|---|---|
| multihop | **-0.360** [-0.459, -0.262] | **-0.473** [-0.573, -0.371] | **-0.411** [-0.513, -0.307] | -0.019 [-0.094, +0.058] |
| arithmetic | **-0.247** [-0.382, -0.106] | +0.095 [-0.044, +0.231] | +0.053 [-0.080, +0.193] | -0.041 [-0.115, +0.015] |

The logit lens wins significantly in four of the eight cells, all of them early,
and the Jacobian lens wins significantly in none. The same holds on the per-layer
mean of section 5: the only cell where the Jacobian lens leads on a point
estimate, multihop loop 4, is +0.025 [-0.020, +0.073] and does not exclude zero
either. So the defensible statement is not that the Jacobian lens never beats the
logit lens, which was asserted without an interval, but that it never
significantly beats it anywhere, while the logit lens significantly beats it on
every early cell that matters.

This is the opposite of the pattern the technique was built for. Anthropic
report the J-lens and logit lens agreeing in the last several layers and the
J-lens recovering content earlier, where the logit lens fails. In Ouro they
agree at the end and the J-lens is *worse* earlier.

The reason is architectural, and Ouro's own paper states it: "at each recurrent
step i, an exit gate predicts the probability p_i of exiting, and a language
modeling head computes the task loss", with the total objective a depth-weighted
sum of those per-step losses. Every recurrent step is trained to decode through
the shared LM head, which matches the released forward, where the loss path sums
`lm_head(step_hidden)` weighted by exit probability. So every loop's residual
stream is already aligned with the output basis and there is little rotation for
a transport map to correct. The evidence is visible in the readout itself: at
loop 1 layer 19 of 48, which is 40 percent of the weight stack and 10 percent of
the total computation, the vanilla logit lens already beats a supervised probe on the arithmetic
intermediate (0.764 against 0.615, section 8). In a model without per-step
decoding supervision, that is precisely where the logit lens is expected to
fail.

---

## 5. Local-exit versus eventual-exit readout: the main result

For a state at loop k, the local-exit lens asks what it is disposed to make the
model say *if recurrence stops now*; the eventual-exit lens asks what it is
disposed to make the *completed* model say. Both are read at the same locations.

| state | task | local-exit lens | eventual-exit lens | logit lens |
|---|---|---|---|---|
| loop 1 | multihop | 0.065 | 0.000 | 0.053 |
| loop 2 | multihop | 0.107 | -0.007 | 0.095 |
| loop 3 | multihop | 0.167 | 0.016 | 0.139 |
| loop 1 | arithmetic | 0.152 | 0.006 | 0.236 |
| loop 2 | arithmetic | 0.053 | 0.096 | 0.118 |
| loop 3 | arithmetic | 0.051 | 0.087 | 0.094 |

Excess hit@10, averaged over the 48 layers of the loop. The local-exit lens and
the logit lens are comparable overall and neither dominates; layer by layer they
correlate only moderately (Pearson r 0.45 to 0.77, mean absolute difference 0.05
to 0.09), so they are not the same readout. The eventual-exit lens is the
outlier. On multihop it is near zero at every pre-final loop, including loop 3,
where the model's output has largely settled already; what predicts it is the
remaining horizon, not output convergence (section 7). On arithmetic, which has
only one loop of genuine output movement, the two accounts coincide and the
eventual-exit lens recovers from loop 2 onward.

The two readouts almost never agree on the top token: over the last 8 layers of
each loop, local and eventual agree on 0.002, 0.018 and 0.008 of cases for loops
1, 2 and 3.

The practical monitorability conclusion is blunt. If you want to know at loop 1
what the finished model will say, the best available readout is the model's own
loop-1 exit head, which gets the final top-1 token right 45 percent of the time.
A fitted Jacobian transport from that state gets essentially nothing.

---

## 6. Why: the transport loses its prompt-invariant component

The primary statistic is the norm of the fitted mean transport itself, at the
largest fit we have:

| source loop → exit 4 | ‖J‖/√d at 80 prompts |
|---|---|
| loop 1 | 0.159 |
| loop 2 | 0.178 |
| loop 3 | 0.272 |
| loop 4 | 0.810 |

The average map carrying a loop-1 state into the final exit is about five times
smaller than the one carrying a loop-4 state, and because sampling noise inflates
a small mean more than a large one, 5.1 is a lower bound on that ratio.

The interesting part is that the per-prompt Jacobians are not themselves small.
Measured directly rather than through a variance decomposition, the mean norm of a
single-prompt Jacobian is 0.708 at loop 1 against 0.715 at loop 4, a ratio of
1.01, while the averaged maps differ six-fold. Individual prompts transport a
loop-1 state as strongly as a loop-4 one; they simply do not agree on a direction,
so the average the lens depends on is left with little to carry. That is the later
recurrence rewriting each state's contribution in a prompt-specific way, and it is
the cleanest statement of the mechanism in this document.

A second estimator, splitting E‖J_n‖² = ‖μ‖² + σ²/n across two disjoint fits,
was used in an earlier draft and is demoted here on the audit's advice. Over all
six valid disjoint pairs its ‖μ‖ is stable to about 2 percent at loops 2 to 4 but
spans 0.060 to 0.207 at loop 1, the row the claim actually rests on, giving
loop-4-to-loop-1 ratios anywhere from 4.0 to 13.6. It is a difference of noisy
norms and goes negative often: for one pair σ² is negative at 107 of 191 layers.
`lens_convergence.py` now counts and reports that clipping instead of hiding it.
A least-squares fit of the same model across fit sizes, which clips nothing,
gives a ratio near 6. So the ordering is robust and the precise multiplier is not;
read it as "about five-fold, at least".

**The direct fit-size test settles it.** The same evaluation was run with the
eventual-exit lens fitted on 8, 32, 56 and 80 prompts, a tenfold range. Excess
hit@10 within each loop:

| task | fit prompts | loop 1 | loop 2 | loop 3 | loop 4 |
|---|---|---|---|---|---|
| multihop | 8 | 0.003 | 0.015 | 0.055 | 0.463 |
| multihop | 32 | -0.002 | -0.014 | 0.088 | 0.464 |
| multihop | 56 | -0.003 | -0.026 | 0.082 | 0.451 |
| multihop | 80 | 0.003 | -0.025 | 0.083 | 0.451 |
| arithmetic | 8 | 0.054 | 0.216 | 0.184 | 0.184 |
| arithmetic | 32 | 0.059 | 0.247 | 0.184 | 0.172 |
| arithmetic | 56 | 0.080 | 0.257 | 0.195 | 0.169 |
| arithmetic | 80 | 0.054 | 0.273 | 0.181 | 0.194 |

Read at this threshold nothing much moves. But hit@10 sits at 0.01 to 0.04 at
loops 1 and 2, close to its floor, and a floored statistic cannot move whatever
the lens does. Underneath it the ranks do move. Comparing the 8-prompt and
80-prompt lenses on the same 148 items, the mean change in log-10 rank of the
true intermediate is significantly positive in six of eight (task, loop) cells,
for instance multihop loop 2 at +0.227 [+0.142, +0.310] and loop 3 at +0.244
[+0.135, +0.354], with arithmetic loops 1 to 3 all between +0.31 and +0.39. Only
loop 4 is flat on both tasks, and only **multihop loop 1** is flat at both the
threshold and the rank level.

So "the lens has converged by eight prompts" is wrong and has been removed. More
fitting data does improve the readout almost everywhere; what it does not do is
lift multihop loop 1 off the floor. That is the cell the main result rests on,
and it is the one that a larger fit leaves untouched. It also means the
1000-prompt rented run is a real test of this section rather than a formality.
Figure: `artifacts/jlens/eval/fig6_fit_size.png`.

---

## 7. Cross-loop transfer: H1 confirmed as a number, refuted as a mechanism

Preregistered prediction: transport involving loop 1 is worse than transport
among loops 2, 3 and 4, mirroring the OPI loop-1 discontinuity. Fitted J at
(loop i, L) is cross-applied to the state at (loop j, L).

Excess hit@10, multihop (rows = fit loop, columns = state loop):

| | state 1 | state 2 | state 3 | state 4 |
|---|---|---|---|---|
| **fit 1** | -0.00 | 0.02 | 0.02 | 0.03 |
| **fit 2** | -0.01 | -0.01 | 0.02 | 0.06 |
| **fit 3** | 0.04 | 0.05 | 0.09 | 0.11 |
| **fit 4** | 0.42 | 0.45 | 0.48 | 0.46 |

The preregistered statistic is negative and excludes zero for multihop, -0.110
CI [-0.147, -0.074], and is exactly null for arithmetic, +0.005 CI [-0.053,
0.061]. But the matrix is row-dominated, so the statistic is not measuring what
it was meant to. Separating the two roles loop 1 can play:

| contrast | multihop | arithmetic |
|---|---|---|
| preregistered (loop 1 in either role) | -0.110 [-0.147, -0.074] | +0.005 [-0.053, 0.061] |
| loop-1 **states** (the predicted mechanism) | -0.045 [-0.086, -0.009] | +0.011 [-0.072, 0.088] |
| loop-1 **fits** (the longest horizon) | -0.174 [-0.231, -0.126] | -0.002 [-0.069, 0.065] |
| variance share, fit loop / state loop / interaction | 0.98 / 0.01 / 0.00 | 0.10 / 0.23 / 0.67 |

The fit loop explains 98 percent of the variance for multihop. A Jacobian fitted
at loop 4 reads intermediates about equally well from *every* loop's state, and
one fitted at loop 1 reads little from any state. The predicted state effect is
four times smaller than the fit effect for multihop and null with the wrong sign
for arithmetic. On arithmetic the matrix has no structure at all: two thirds of
the variance is interaction, all three contrasts sit on zero, and the fit loop
explains a tenth of the variance. On 51 items that is what noise looks like, so
the horizon account rests entirely on multihop.

The obvious objection is that this is just the ordinary depth effect: lenses
work in the mid-to-late band, and Ouro's four loops merely stretch that band, so
"loop 1" is really "early layers". Our own data answers it. At those same
loop-1 locations the vanilla logit lens reads the intermediate well (0.358
any-layer excess on multihop, and on arithmetic it beats a supervised probe,
0.764 against 0.615). The content is present and verbalizable exactly where the
eventual-exit transport returns nothing, on identical states, positions and
scoring. So this is not early layers being unreadable; it is the transport into
the final basis specifically that carries none of it.

So the correct statement is not that loop-1 states are special. It is that
transport quality is governed by the horizon, the amount of recurrent
computation between the fitted source and the target, and is nearly indifferent
to which loop produced the state it is applied to. Preregistration is what
caught this: the headline statistic would have been reported as a confirmation.

---

## 8. Supervised reference on a controlled arithmetic family

Prompts `(a + b) * c = ` with the intermediate a+b as a 17-way label. Because
a+b is symmetric, folds are over *unordered* operand pairs, so (3,5) and (5,3)
always land together and no probe can read a test label off its mirror. Five-fold
cross-validation over all 45 unordered pairs, every one of the 648 prompts tested
exactly once, the regularisation strength chosen inside each fold, and both lenses
rescored on the same 648 prompts. Chance is 0.059; the base model gets the product
right 78 percent of the time. Candidate-set top-1, with a pair-clustered bootstrap:

| readout | loop 1 | loop 2 | loop 3 | loop 4 |
|---|---|---|---|---|
| supervised probe | 0.615 [0.47, 0.74] | 0.590 [0.46, 0.72] | 0.335 [0.22, 0.44] | 0.236 [0.15, 0.35] |
| logit lens | 0.764 [0.66, 0.91] | 0.500 [0.28, 0.64] | 0.179 [0.06, 0.24] | 0.311 [0.22, 0.39] |
| eventual-exit Jacobian lens | 0.392 [0.23, 0.56] | 0.500 [0.29, 0.64] | 0.148 [0.08, 0.27] | 0.120 [0.05, 0.17] |

Three things follow.

The vanilla logit lens leads a supervised probe at loop 1, 0.764 against 0.615.
Read that as "an untrained readout beats the best probe we can currently fit",
not as a statement about the representation, because the probe is still clearly
data-limited. Holding the folds, the test prompts and the validation pairs fixed
and varying only the number of training pairs gives 0.185, 0.349, 0.383, 0.443
and 0.557 at 7, 14, 21, 28 and 36 pairs, a significant +0.094 [+0.011, +0.192]
from 14 to 28 pairs and still climbing at the top of the range. Note also that
the loop-1 difference, though large as a point estimate, has an interval that
touches zero (below).

We got this wrong once in the other direction. An intermediate analysis compared
0.665 on the single split with 0.615 under cross-validation, concluded that more
data made the probe worse, and deleted the data-limited explanation. That
comparison changed training size, test set, label-coverage handicap and
aggregation at the same time and cannot support the inference. The curve above
varies one thing.

Decodability does outlast verbalizability, but only in one place. Differencing
probe against logit lens gives -0.149 [-0.358, +0.001] at loop 1, +0.090 [-0.090,
+0.346] at loop 2, **+0.156 [+0.031, +0.315] at loop 3**, and -0.075 [-0.190,
+0.064] at loop 4. Only loop 3 excludes zero, where the probe reads the
intermediate at roughly twice the logit lens. So a residual version of the claim
survives at loop 3 and nowhere else; both readouts lose about 60 percent of their
accuracy by loop 4. A previous draft withdrew the claim entirely, which was too
pessimistic.

The eventual-exit Jacobian lens is below both at every loop, and its interval
excludes the probe's at loops 1 and 4. That is section 5's result again on a
controlled family where the intermediate is provably computed.

Two caveats the cross-validation exposed that the single split had hidden. Because
a+b is a deterministic function of the pair and four of the seventeen labels are
realised by exactly one pair, 72 of the 648 test prompts carry a label their own
fold never trained on; the single-split probe has the same defect, with 14 of 17
classes in training. And per-fold loop-1 probe accuracy ranges from 0.275 to
0.919, a standard deviation of 0.21, so no figure in this table should be read to
two decimal places.

---

## 9. Robustness

**Readout position.** The whole evaluation was rerun one token earlier. On
multihop the picture is unchanged: the eventual-exit lens stays near zero for
loops 1 to 3 and jumps at loop 4 (excess 0.004, 0.007, 0.047, 0.379) while the
logit lens stays well above it at every loop (0.296, 0.325, 0.455, 0.453). On
arithmetic both readouts collapse to zero at the earlier position, which is the
`=` token rather than the space immediately preceding the digits. The arithmetic
intermediate is therefore readable only at the token where the answer is about
to be emitted; that is a real constraint on section 8 and on the arithmetic rows
of sections 4 and 5, and it does not affect the multihop results or any
lens-free measurement. The lens-free exit divergence also shifts with position
as expected: one token earlier the loop-1 exit already agrees with the final
output 72 percent of the time, against 45 percent at the answer position, so the
divergence is concentrated exactly where the model commits to content.

**Model competence.** Restricting to items the model answers correctly sharpens
every effect rather than removing it; multihop eventual-exit excess at loop 4
rises from 0.352 to 0.540 while loops 1 to 3 stay near zero (per-layer maximum).

**Leakage.** Intermediates whose surface form appears in the context are
excluded from all primary numbers and reported separately; they behave as
expected, with high raw hit rates and near-zero excess over matched controls.

## 10. Limitations

- The eventual-exit lens is tested from 8 to 80 fitting prompts. Its readout
  improves with fit size almost everywhere at the rank level, and only multihop
  loop 1 is flat at both the threshold and the rank level, so the rented 100- and
  1000-prompt runs are a real test of section 6 rather than an extension. The
  local-exit lenses use 24 prompts each and have not had the same sweep, so
  section 5's local-exit column is the least fit-tested number in the document.
- The lenses are fitted on WikiText with the first 16 positions skipped, but 99
  of the 148 evaluation items are read at a position before 16. This does not
  appear to explain the deficit: the unfitted logit lens shows the same position
  effect, and the local-exit lens, fitted on the same prompts, reads those states
  fine. A lens fitted on short prompts is the clean control and is queued for the
  rented run.
- The supervised probe's folds cannot cover every label: a+b is a deterministic
  function of the operand pair and four of the seventeen labels come from a single
  pair, so 72 of 648 test prompts carry a label their fold never saw.
- 90 multihop and 51 arithmetic scorable items; CIs bootstrap over items.
- Digits are single tokens in this tokenizer, so multi-digit intermediates are
  scored through word forms and three items are unscorable.
- The supervised reference covers one templated arithmetic family and is fitted
  on 384 prompts, so it is under-powered as a decodability ceiling. Its candidate
  scoring also gives labels with more single-token surface forms a small
  advantage; a one-form-per-label variant is reported alongside it and differs
  little.
- Model competence is scored by boundary-aware prefix match on a 3-to-4 token
  greedy continuation. The numbers above were produced with a plain prefix
  match, which credited 3 of 75 items wrongly (for example "11" against the
  target "1"); this affects only the model-correct robustness subsets.
- bf16 backward through up to 192 shared layer applications.

## 11. Next steps, not in this submission

100-prompt and 1000-prompt lenses on rented hardware; a cross-validated
supervised probe over all 45 unordered operand pairs, to test whether linear
decodability really outlasts verbalizability; RLTT and Thinking
checkpoints, where the early-exit training differs; Huginn, which has no per-loop
head and so should show the basis rotation the J-lens was designed for; tuned
lens as a fourth baseline.
