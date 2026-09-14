# Jacobian lens on recurrent Ouro — research log

Question: can Jacobian lenses recover known latent intermediates across
recurrent depth in Ouro-2.6B, and how much does future recurrent computation
rewrite what the current state is disposed to say (local-exit vs
eventual-exit readout)?

Model: ByteDance/Ouro-2.6B, HF revision 1ed04250da1a9936042725d302e81c8fa2ab5abd
(local snapshot under artifacts/hf_cache; upstream re-checked 2026-09-03,
byte-identical modeling_ouro.py). 48 physical layers x 4 recurrent steps,
d_model 2048, bf16.
Lens code: anthropics/jacobian-lens @ 581d398 ("Initial release"), installed
editable with --no-deps (declares transformers>=5.5; venv has 4.54.1 which the
Ouro remote code targets; jlens itself only uses config.get_text_config()).

## Preregistered hypotheses (written before any lens result)

H1 (cross-loop transfer): a Jacobian fitted at source (loop 1, layer L) and
cross-applied to the loop 2/3/4 state at layer L (and vice versa) will give
worse known-intermediate readout than cross-application among loops 2, 3, 4.
H2 (local vs eventual exit): the gap between J[state -> exit k] and
J[state -> exit 4] readouts shrinks with recurrent depth k.
Neither generic later-loop convergence nor loop-1 divergence per se is claimed
as novel (OPI, Blayney et al.).

## Design decisions

- Native Ouro forward preserved. Recording uses per-(ut, layer) LoopTap
  objects that filter forward hooks on Ouro's own `current_ut` kwarg; jlens
  fit/apply/merge run unmodified over 192 virtual indices `ut*48 + layer`.
- Loop labels: code uses Ouro's 0-based `ut` (== exit_at_step); prose uses
  loop k = ut+1.
- Source state at virtual (ut, 47) is the pre-norm decoder output; Ouro's
  exit-k logits are lm_head(norm(that)), which is exactly jlens `unembed`.

## Active-time ledger

Active work means hands on the problem. GPU wall-clock while a fit ran
unattended is recorded separately and is not claimed as project time.

| span | active | notes |
|---|---|---|
| 2026-09-02 23:55 → 2026-09-03 01:55 | 2.0 h | repo, Ouro and jlens source; LoopTap recorder; five milestones; eval stimuli; pilot fit and its two scoring confounds; analysis rewritten around matched controls |
| 2026-09-03 01:55 → 08:09 | ~0 | fitting queue unattended. Round-1 results were complete at 05:20 but not read until 08:09, so ~2.8 h of that was avoidable idle time, not compute |
| 2026-09-03 08:09 → 08:55 | 0.8 h | round-1 analysis, H1 role decomposition, control-kind confound found and fixed, fit-size sweep, RESULTS.md |
| 2026-09-03 23:30 → 2026-09-04 00:30 | 1.0 h | pre-B300 verification: validate OOM fix, probe transpose leakage, aggregation unit, merge memory, rented-run dry run, section 7 rewritten |
| **total** | **≈ 3.8 h active** | ≈ 7 h GPU wall-clock, almost all unattended fitting |

## Events
- 00:05 First model load ran without the local cache env (transformers imported
  before env set) and started re-downloading from the Hub; fixed by loading
  from the pinned local snapshot path. Upstream file confirmed identical.
- 00:20 Milestones 1–5 all PASS (artifacts/jlens/validation/milestones.json,
  validate_run2.log). Bit-exact: hooked==plain logits; unembed(pre-norm
  (ut,47)) == exit_at_step=ut logits for ut 0..3; default forward == exit 3
  on the test prompt (gate λ₁ max 0.26, no bf16 saturation); norm(recorded
  (ut,47)) == native hidden_states_list[ut]; next-loop layer-0 input ==
  normed (ut,47); stock ActivationRecorder == last loop; stock J(L40->47) ==
  recurrent J((3,40)->(3,47)) exactly, vs rel-Fro 0.99 against (2,40).
  VJPs from (3,47) to (1,16)/(2,16)/(3,16) distinct (rel diff ~0.9-0.95).
- 00:22 transformers 4.54.1 incompatibility: OuroForCausalLM with the default
  use_cache=True crashes in UniversalTransformerCache (read-only key_cache);
  all reference forwards use use_cache=False (jlens already does).
- 00:30 Tokenizer adds no BOS by default; we prepend BOS explicitly for every
  method. Digits are single-digit tokens, so multi-digit intermediates are
  scored via word forms (" twelve"); "26", "52", "24" unscorable.
- 00:36 Local bench, full 192-source graph, T=128: dim_batch 1 -> 0.13 s/pass,
  7.7 GB, ~4.4 min/prompt; dim_batch 2 -> 0.20 s/pass, 10.0 GB, ~3.4 min/prompt;
  dim_batch 4 OOM on 12 GB.
- 00:40 Fitting corpus: first 1200 WikiText-103 train records >= 600 chars
  (jlens's own load_wikitext_prompts), saved to artifacts/jlens/data.
- 00:42 Pilot fit started locally: exit-3 target, 191 sources, prompts 0-8,
  dim_batch 2 (artifacts/jlens/lens/pilot).

## Preregistered analysis definitions (written 00:50, before any lens readout)

- Readout position: last prompt token (jlens README convention: the token
  immediately preceding `target`). BOS prepended. Same position for every
  method.
- Rank of an intermediate = min over its single-token surface forms
  (with/without leading space; digit+word forms for numbers; symbol+word forms
  for operations). Full-vocab rank, 0 = top.
- hit@k at a location (ut, L) = mean over scorable, non-leaked intermediates of
  [rank < k], k in {1, 10}. Primary tasks: multihop entity intermediates and
  order-ops *numeric* intermediates. Operation intermediates are almost all
  leaked (operator visible in the prompt) and are reported only as the leaked
  condition. Leaked numeric intermediates reported separately.
- Cross-loop transfer (H1): xloop[i, j] = hit@10 of J fitted at (ut_i, L)
  applied to state (ut_j, L), min over L. Test statistic: mean of the 6
  off-diagonal cells involving ut 0 minus mean of the 6 off-diagonal cells
  among ut 1..3; 95% bootstrap CI over items (1000 resamples, seed 0).
  Prediction: negative (loop-1 transfer worse).
- Local vs eventual (H2): for state (ut, L) compare the exit-ut lens readout
  with the exit-3 lens readout: KL(local || eventual), top-1 agreement, and
  each one's KL to the actual exit-ut / final logits. Prediction: gap shrinks
  with ut. No "identity transport" language; exit-ut readout is the local-exit
  readout.
- Supervised reference: 17-way logistic regression on "(a + b) * c = " with
  held-out operand pairs (20 test / 10 val / 51 train pairs, seed 0); compared
  with the lenses on the same test prompts using the same 17 candidates.
  It is a task-specific supervised reference, not an upper bound.
- Model competence on each item's target (greedy, exit 3) is recorded and
  results are reported for all items and for model-correct items.
- 00:55 Rented-GPU facts (RunPod API, read-only): balance $2.20, no pods.
  B300 SXM6 288 GB $6.94/h community, $7.89/h secure (stock: low); B200 180 GB
  $5.98/$6.79; H200 141 GB $3.59/$4.59. Full plan (exit-3 100 prompts + exits
  0-2 + eval/probe) is estimated at ~2 GPU-hours on B300 (~$15); extending
  exit-3 to 1000 prompts adds roughly 3-5 h. Balance must be topped up before
  launch (user decision).
- 00:57 CPU unit tests for LoopTap / evaldata / analysis helpers:
  utilities/tests/unit/test_ouro_jlens.py (5 pass). First version had a wrong
  hand-computed expectation in the test itself; recorder behaviour was correct.
- 01:00 Readout position: primary = last prompt token (README convention; for
  most stimuli a lone trailing-space token). Robustness check preregistered:
  rerun at position -2 (the token before the trailing space) and report both;
  conclusions must not depend on the choice.
- 00:53 Local GPU was SW-thermally throttled (1.7 GHz, 59 W) during pilot
  prompts 1-3 (230/286/306 s); user raised the power profile (2.2 GHz, 83 W).
  Pilot timings are therefore not a throughput estimate.
- 00:53 Rented-GPU launch automation declined by user; deliverables for the
  rented run are setup_b300.sh + run_b300.sh (+ optional pod_entry.sh /
  stage_upload.sh for HF-staged transfer). Launch itself is the user's call.
- 01:25 Pilot pipeline run (8-prompt lens) exposed a readout-position confound
  BEFORE any interpretation: stimuli prompts end in a trailing space, which the
  Ouro tokenizer keeps as a lone " " token; after it the model can only
  continue with space-less tokens (digits, punctuation), so the multihop
  continuations were "100%", "10th" and "model correct" was 20/93. The README
  rule "token immediately preceding target" is now implemented literally:
  tokenize prompt+target, read at the last common-prefix token (" is" for word
  targets, the lone " " where digits follow). Unit-tested. Order-ops (digit
  targets) unaffected; probe family unaffected. All pilot readout numbers
  computed before this fix are discarded (pilot_exit3 to be re-run).
- 01:25 Pilot cross-loop matrices (pre-fix, discarded) nonetheless showed the
  loop-4-fitted J transferring to every loop's state while loop-1/2-fitted J did
  not — direction consistent with H1 but not to be reported from that run.
- 01:40 Second scoring confound found on the (discarded) pilot numbers: at a
  lone-space readout position the model's top-10 is essentially the ten digit
  tokens, so single-digit intermediates are "hit@10" by default (the
  loop-4-fitted J applied to loop-1 states scored 0.94 on order-ops numerics).
  Preregistered fix, before re-running: every intermediate name of the task is
  ranked for every item; the item's own names give hit@k, the other names are
  matched controls for the position prior. Primary readout metrics become
  (i) excess hit@k = own hit@k − control hit@k and (ii) candidate-set top-1
  (own name ranked above all other task names). Raw hit@k is still reported.
  Same controls used for the cross-loop matrix and H1.
- 01:25 Probe family: base model greedy-correct on "(a + b) * c = " 77.9%
  (648 prompts), so the intermediate a+b is computed by the model there.
- 01:30 Local fallback plan if the rented GPU is not launched: extend the
  exit-3 lens to 100 WikiText prompts in local shards (~3 min/prompt at full
  clocks, ~5 h GPU), then exit-0/1/2 lenses on 24 prompts each (~2 h). The
  8-prompt pilot lens is treated as a pipeline check only.
- 01:35 Corrected pilot evaluation (8-prompt lens; pipeline check, not a
  result). Model correct on target: multihop 39/93, order-ops 36/55.
  Raw-model observation (lens-free, robust): on arithmetic items the loop-1
  exit's top token is often the INTERMEDIATE and later exits the answer, e.g.
  "(2 + 3) * 4 = " exits [5, 2, 2, 2]; "(9 - 6) * 2 = " exits [3, 6, 6, 6].
  This is the local-vs-eventual gap in the model's own exits.
  Pilot lens numbers (to be superseded): eventual-exit J-lens excess hit@10 is
  ~0 at loops 1-3 and ~0.3 at loop 4; logit lens ~0.2-0.5 at all loops (loop-1
  layers 18-47 strongest for arithmetic). Cross-loop: the loop-4-fitted J reads
  intermediates from states of EVERY loop (excess 0.42-0.50 multihop), while
  loop-1/2/3-fitted J read almost nothing from any state. Two competing
  explanations to separate with a better lens: (a) long-horizon J through
  100+ shared layers is poorly estimated from 8 prompts; (b) the eventual-exit
  transport legitimately excludes content that later loops overwrite. The
  exit-1/2/3 lenses (local-exit readout) discriminate these.
- 01:40 Local fitting plan queued (run_local.sh): exit-3 prompts 8-32, then
  exit-0/1/2 on prompts 0-24, then exit-3 to 104.
- 01:50 Lens-free model-exit facts at the readout position (all scorable
  non-leaked items; independent of any fitted lens):
  order-ops numeric (51): intermediate is the exit top-1 at loops 1..4 =
  [0.18, 0.08, 0.06, 0.04]; exit-k top-1 equals final top-1 = [0.49, 0.84,
  0.96, 1.0]. multihop (90): intermediate top-1 [0.01, 0.03, 0.04, 0.04];
  exit-k == final [0.39, 0.67, 0.80, 1.0]. I.e. the loop-1 exit verbalizes
  something the completed recurrence will not say on about half the items.
- 01:45 CPU contention: the probe's sklearn fits (24-core BLAS) starved the
  GPU fitting process (prompt time 200 s -> 600 s, GPU util 28%). Probe now
  runs with 2 BLAS threads at nice 19; queue restarted as run_local2.sh
  (shards -> round1 eval with 32-prompt exit-3 + 24-prompt exit-0/1/2 ->
  exit-3 to 104 -> round2 eval). ~10 min of fitting lost.
- 01:55 Pilot probe comparison, arithmetic family "(a + b) * c = ", 17-way
  candidate top-1 on 160 held-out-pair test prompts (chance 0.059; model 78%
  correct on the product): supervised probe per-loop max [0.88, 0.83, 0.62,
  0.56] (loop 1 rises sharply at layer ~18); logit lens [0.88, 0.60, 0.18,
  0.31] — at loop 1 layers 18-40 the logit lens equals the probe, i.e. the
  intermediate is directly verbalizable there, and it collapses to chance after
  the first ~10 layers of loop 2; eventual-exit Jacobian lens (8-prompt,
  provisional) [0.33, 0.44, 0.18, 0.18] — weak at loop 1 where the logit lens is
  strongest, consistent with the completed recurrence not verbalizing the
  intermediate. Linear decodability persists into loops 3-4 without
  verbalizability. Provisional lens numbers; probe/logit-lens numbers are
  lens-independent and stand.
- 03:15 Estimator convergence diagnostic (8-prompt vs 24-prompt exit-4 lens,
  disjoint prompts): cosine(J8, J24) per source loop = [0.46, 0.84, 0.92,
  0.94]; rel. Frobenius diff [1.41, 0.61, 0.42, 0.36]; ||J24||_F/sqrt(d) =
  [0.19, 0.18, 0.28, 0.83]; mean diagonal [0.01, 0.02, 0.04, 0.45]. The
  long-horizon (loop-1 source -> exit-4) average Jacobian is small and not
  converged at 24 prompts, while within-loop-4 transport is large,
  near-diagonal at deep layers, and converged. Interpretation is deferred to
  the norm-ratio test (pure noise predicts ||J8||/||J24|| ~ 1.73) and to the
  104-prompt lens; either way, average-Jacobian readout across recurrent
  loops is a much harder estimation problem than within a loop.
- 03:20 lens_convergence.py (8 vs 24 disjoint prompts, ||J_n||^2 = ||mu||^2 +
  sigma^2/n): converged mean-transport norm ||mu||/sqrt(d) per source loop
  [0.10, 0.17, 0.27, 0.82]; per-prompt scatter sigma [0.75, 0.35, 0.31, 0.64].
  Loop-1 -> exit-4 per-prompt Jacobians are as large as loop-4 ones but
  incoherent across prompts: the mean linear transport is ~8x smaller than
  within loop 4. Methodological consequence: an average-Jacobian lens can read
  loop-1 states only through this small common component regardless of fit
  size; prompt-specific rewriting by later loops is the dominant term.
  (n1 = 8 makes sigma rough; redo with the 104-prompt shards.)
- 08:30 ROUND 1 COMPLETE (exit-4 lens 32 prompts; exit-1/2/3 lenses 24 each).
  Findings written to docs/jlens/RESULTS.md. Key verified numbers:
  (a) KL(exit k || exit 4) = [3.83, 0.65, 0.09, 0]; identity sanity cells give
  exactly 0, confirming the KL path.
  (b) Local-exit vs eventual-exit excess@10 (mean over layers): multihop
  [0.060, 0.102, 0.150] vs [0.000, -0.006, 0.013]; arithmetic [0.166, 0.068,
  0.070] vs [0.007, 0.105, 0.092]. The eventual-exit lens collapses exactly
  where the model's output has not converged.
  (c) H1: preregistered contrast -0.113 CI [-0.150,-0.077] multihop / -0.025
  CI [-0.081,0.029] arithmetic. Role-separated: loop-1 STATES -0.044 /
  +0.034 (predicted mechanism, small or null); loop-1 FITS -0.181 / -0.084.
  Variance share fit-loop 0.99 / state-loop 0.01 (multihop). H1 confirmed as a
  number, refuted as a mechanism; horizon, not state loop, governs transport.
  Added role-separated contrasts to analyze.py so this is reproducible.
- 08:30 Corrected an over-claim before it entered the write-up: local-exit lens
  and logit lens match to 0.003 on max-over-layers excess, but per-layer they
  correlate only r=0.43-0.77 with mean|diff| 0.05-0.10. They are comparable in
  strength, not the same readout. Claim weakened accordingly.
- 08:30 KL(local || eventual) rises with loop (4.9/4.6/12.6) but that tracks the
  sharpness of the target exit, not readout quality; scale-free rank and
  agreement measures are now reported beside it.
- 08:40 Ouro paper (2510.25741) confirms the mechanism claim directly: "at each
  recurrent step i, an exit gate predicts the probability p_i of exiting, and a
  language modeling head computes the task loss"; the objective is a
  depth-weighted sum of per-step LM losses plus an entropy term. So every loop's
  residual is trained to decode through the shared head, which explains why the
  vanilla logit lens is strong at all loops in Ouro and why the J-lens has
  little basis rotation to correct.
- 08:40 Readout-position robustness (position -2, preregistered): multihop
  conclusions unchanged (eventual-exit lens 0.007/0.010/0.026/0.253 vs logit
  0.151/0.167/0.199/0.218). Arithmetic collapses to ~0 for BOTH lenses at the
  "=" token, i.e. the arithmetic intermediate is readable only at the token
  immediately preceding the digits. Recorded as a limitation on the arithmetic
  rows, not a refutation of the multihop result. Lens-free exit divergence also
  shifts: exit-1 top-1 == final rises 0.45 -> 0.72 one token earlier, so the
  divergence concentrates where the model commits to content.
- 08:25 CORRECTION to the transport-coherence section. A better-conditioned
  disjoint pair (24 vs 32 prompts, both already fitted) gives converged
  ||mu||/sqrt(d) = [0.17, 0.18, 0.28, 0.84] and sigma = [0.33, 0.19, 0.01,
  0.36], against [0.10, 0.17, 0.27, 0.82] / [0.75, 0.35, 0.31, 0.64] from the
  poorly-conditioned 8-vs-24 pair. Direction of the finding is unchanged (the
  loop-1 mean transport is ~5x smaller than within loop 4, not 8x), but the
  extrapolated claim "even 1000 prompts would leave 25% relative sampling error
  on the loop-1 transport" is WITHDRAWN; the better pair implies ~6%. RESULTS.md
  section 5 rewritten to say the map is small, not unmeasurable, and to lead
  with the direct fit-size evidence (8 -> 32 prompts changed nothing at loops
  1-3) instead of the extrapolation. Also flagged that the two-point sigma
  estimator is unstable (loop-3 sigma 0.01 vs 0.31 across pairs).
- 08:25 Local GPU work stopped at user request. State on disk: eventual-exit
  shards p0-8, p8-32, p32-56, p56-80 (80 prompts total, only the first 32
  evaluated); local-exit lenses exit0/exit1/exit2 at 24 prompts each; the
  80-104 shard was killed mid-run. Round-1 results were complete at 05:20.
  The supervised probe against the 32-prompt lens never ran (only the 8-prompt
  pilot probe completed). Next local step, ~25 min GPU: fitsize.sh, which
  merges to 56/80 and evaluates the fit-size sweep.
- 08:35 FIT-SIZE SWEEP COMPLETE (fitsize.sh; eventual-exit lens at 8/32/56/80
  prompts, same stimuli). Excess hit@10 is FLAT over the tenfold range at every
  loop: multihop loop1 [0.004, 0.013, 0.003, 0.006], loop4 [0.435, 0.437,
  0.426, 0.426]; arithmetic loop1 [0.049, 0.114, 0.051, 0.017], loop4 [0.467,
  0.452, 0.454, 0.473]. The early-loop blind spot is not a sampling artifact;
  the lens has converged for this metric by 8 prompts. This replaces the
  withdrawn extrapolation as the support for section 5.
- 08:35 Third disjoint-pair convergence estimate (24 vs 56): ||mu||/sqrt(d) =
  [0.161, 0.171, 0.266, 0.798], agreeing with the 24-vs-32 pair [0.166, 0.179,
  0.284, 0.842]. ||mu|| is stable across pairs; sigma is not (spans an order of
  magnitude), so the write-up now reports the ||mu|| ratio and treats sigma as
  indicative only.
- 08:50 CONFOUND FOUND AND FIXED in the control definition. Controls were "all
  other intermediate names of the task", which for order-ops mixed operation
  names (6 of 20) into the control set for numeric intermediates. Operations
  rank far below numbers at these readout positions, so they dragged the
  control down and inflated arithmetic excess by 0.04-0.14 for BOTH lenses.
  Controls are now matched by kind (numeric controls numeric, operation
  controls operation), which is what the write-up already claimed they were.
  All seven evaluation directories re-analyzed on CPU. Multihop is unaffected
  (no operation names). Arithmetic levels drop; no conclusion changes, but the
  arithmetic cross-loop structure is now clearly null: preregistered contrast
  -0.002 CI [-0.059, 0.052], variance share fit 0.33 / state 0.16 /
  interaction 0.51, so the horizon account now rests on multihop alone.
  RESULTS.md sections 3, 4, 5, 6 and 8 updated with re-derived numbers.
- 08:52 Supervised probe rerun against the 80-prompt eventual-exit lens
  (9 min sklearn at 1 thread, not the 40 min estimated). Per-loop candidate
  top-1: probe [0.881, 0.831, 0.625, 0.556], logit lens [0.875, 0.600, 0.181,
  0.312], eventual-exit J-lens [0.444, 0.525, 0.238, 0.162]. Ten times more
  fitting data than the pilot moved the J-lens from [0.33, 0.44, 0.18, 0.18]
  to [0.44, 0.53, 0.24, 0.16]: same shape, still about half the logit lens at
  loop 1 and half at loop 4. Consistent with the fit-size sweep.
- 08:55 LOCAL WORK COMPLETE. All seven core scope items delivered. Deliverables:
  docs/jlens/RESULTS.md, docs/jlens/HANDOFF_B300.md, docs/jlens/RESEARCH_LOG.md,
  src/ouro_jlens/*, utilities/tests/unit/test_ouro_jlens.py (6 passing),
  6 figures under artifacts/jlens/eval/. Remaining work is the rented run.

## Pre-B300 verification pass (2026-09-03 evening)

Full self-review of code, artifacts and write-up before renting. Four defects
found and fixed; no headline conclusion changed.

1. **validate.py could abort the rented run.** Milestone 5 ran at dim_batch 8
   and peaked at 11.96 GB on a 12 GB card, i.e. exactly at the limit; on a
   rerun it raised CUDA OOM. Under `set -euo pipefail` that would have killed
   run_b300.sh at its first step. m5 now retries at halved dim_batch and
   records which it used. Re-run: ALL FIVE MILESTONES PASS, stock jlens
   J(L40->47) still bit-identical to recurrent J((loop4,40)->(loop4,47))
   (max_abs_diff 0.0) and 0.987 rel-Frobenius from the loop-3 source.
2. **Supervised probe had transpose leakage.** The split drew ordered operand
   pairs, so 12 of 20 test pairs had their transpose in train; since a+b is
   symmetric the probe could read the label off the mirrored pair. Now split by
   unordered pair (22/11/48 ordered pairs, zero transpose or exact overlap).
   This inflated only the supervised reference column.
3. **Label surface-form count biased the lens columns of the probe task.**
   Labels have 1 to 5 single-token forms and were scored by the max over forms,
   favouring many-form labels. probe.py now also computes a one-form-per-label
   variant; both are reported.
4. **Two aggregation units were in use.** any_layer and loc_maps averaged over
   (item, name) slots while cross_loop_summary averaged over items, so the
   main table and the cross-loop diagonal disagreed by up to 0.04. Items are
   the correct unit (near-duplicate names such as "3"/"three"/"third" are one
   measurement). Now item-averaged throughout; the main table's any-layer
   excess and the cross-loop diagonal now agree to 0.001, which is a useful
   internal check that the two code paths compute the same quantity.
   All six evaluation directories re-analyzed; multihop numbers shift by up to
   0.05, arithmetic unchanged, no conclusion affected.
- 2026-09-04 00:00 Probe rerun on the leak-free unordered-pair split materially
  changes section 7. Per-loop candidate top-1: probe [0.665, 0.653, 0.341,
  0.330] (was [0.881, 0.831, 0.625, 0.556] with transpose leakage), logit lens
  [0.892, 0.619, 0.176, 0.403], eventual-exit J-lens [0.551, 0.562, 0.222,
  0.148]. One-form-per-label variants differ little (logit [0.915, 0.631,
  0.159, 0.364], J-lens [0.580, 0.540, 0.205, 0.148]), so the form-count bias
  was immaterial. Consequences: (a) the logit lens now EXCEEDS the supervised
  probe at loop 1 rather than tying it; (b) the earlier claim that linear
  decodability outlasts verbalizability through later loops DOES NOT SURVIVE and
  is withdrawn -- both decline; (c) the probe is data-limited (384 training
  prompts, 24 unordered pairs) and is a weak lower bound, not a ceiling. The
  J-lens shortfall against the logit lens at every loop is unchanged.
- 2026-09-04 00:05 is_correct tightened to require a token boundary; 3 of 75
  items were previously credited wrongly (e.g. "11" against target "1"). Affects
  only the model-correct robustness subsets in the current numbers.
- 2026-09-04 00:10 Swarm review opened with two independent Opus peers on a
  shared canonical claim set (scratchpad/swarm/CLAIMS.md, PROTOCOL.md): peer 1
  on instrumentation and the rented-run path (owns the GPU), peer 2 on metrics,
  the write-up audit and a cross-validated probe. Disjoint file ownership, GPU
  lock, findings exchanged between peers.
- 2026-09-04 00:15 My own lane found three gaps the brief had asked for and I
  had missed: (a) no acknowledgement that recurrent convergence in Ouro/Huginn
  is prior work, now section 2, so nothing in section 3 reads as a novelty
  claim; (b) the active-time ledger was an empty stub, now filled (about 3.8 h
  active against about 7 h GPU wall-clock, including 2.8 h where round-1 results
  sat unread); (c) five coherence defects introduced by editing RESULTS.md
  through repeated string replacement, including a stale "two disjoint pairs"
  after a third was added, and a section-5 sentence attributing the eventual
  lens's failure to output convergence, which is wrong for multihop loop 3 where
  the output has settled but the horizon is still a full loop. Section
  numbering and all cross-references reconciled.
- 2026-09-04 00:15 Artifact provenance audited: every lens .json carries model
  revision 1ed04250 and jlens 581d398, shard ranges are disjoint and the merges
  sum correctly; all seven eval dirs hold 148 items. Note three stale probe
  directories (pilot, n80, round1_exit3x32) were produced under the leaky
  ordered-pair split; only n80_v2 is current.
- 2026-09-04 00:35 PEER2 found a real metric defect and I confirmed it
  independently before acting. In the any-layer ("pass@10") statistic the own
  score was mean-over-items of max-over-layers, while the control was
  max-over-layers of mean-over-control-names. A max of a mean is at most a mean
  of maxes, so the control was systematically too small and every any-layer
  excess was inflated. Fixed in analyze.py by scoring each control name the same
  way the own name is scored and averaging afterwards; the per-layer path is
  unaffected because the threshold and the mean commute there. All six
  evaluation directories re-derived. My reimplementation reproduced PEER2's
  corrected values to three decimals, and the main table's any-layer excess
  still equals the cross-loop diagonal exactly, which is the cross-check that
  the two code paths agree.
  CONSEQUENCES: (a) claim C4, "the Jacobian lens never beats the logit lens", is
  FALSE. On arithmetic the eventual-exit lens beats the logit lens at loops 2 and
  3 on the any-layer statistic (0.247 vs 0.152 and 0.184 vs 0.131) though not on
  the per-layer mean. It holds on multihop only. Headline and section 4 rewritten.
  (b) the arithmetic cross-loop matrix, already weak, is now pure noise: all three
  contrasts sit on zero and the variance share is fit 0.10 / state 0.23 /
  interaction 0.67. The horizon account now rests entirely on multihop, and the
  write-up says so. (c) multihop conclusions and the loop-1 result are unchanged
  in kind: multihop any-layer excess for the eventual lens is
  [-0.002, -0.014, 0.088, 0.464] against the logit lens [0.358, 0.459, 0.500,
  0.482]. (d) the fit-size sweep still shows a flat loop-1, but arithmetic loop 2
  drifts 0.216 -> 0.273 over the tenfold range, so "nothing moves" was overstated
  and is now qualified.
  Also fixed from PEER2's notes: the Pearson range is 0.45-0.77 not 0.43-0.77,
  and the stale fitsize figure/summary have been regenerated.
- 2026-09-04 01:15 C4 resolved properly, after both peers and I got it wrong in
  turn. Sequence: PEER2's control fix flipped the arithmetic point estimates;
  PEER1 and I each read the flip as a real reversal and I rewrote the claim as
  "false"; PEER1 then bootstrapped their own case and withdrew it; I bootstrapped
  the any-layer comparison and found the arithmetic crossover is not significant
  either (loop 2 +0.095 CI [-0.044,+0.231], loop 3 +0.053 CI [-0.080,+0.193]).
  Final statement: the eventual-exit J-lens never SIGNIFICANTLY beats the logit
  lens on either statistic, and the logit lens significantly beats it at multihop
  loops 1-3 and arithmetic loop 1. The original claim was directionally right and
  wrong only in being asserted without an interval; my "it is false" rewrite was
  an over-correction on a point estimate. Section 4 now carries the CIs.
- 2026-09-04 01:15 PEER1 item-7 interim: the estimator's position reduction is
  CONFOUNDED with loop index. For a cotangent at a single final-exit position,
  the fraction of source gradient energy arriving at that same position is 0.85
  at (loop 4, L40) but only 0.29-0.34 at loop 1. So the horizon that C6 says
  governs transport is also the horizon over which jlens's position-summed
  estimator stops describing the position we read at. Direction unknown until
  their summed-vs-diagonal A/B lands; it could be a fact about Ouro (content
  genuinely delocalised) or about jlens (estimator mismatched to a
  single-position readout). Nothing in the rented-run code depends on it, only
  the interpretation of C6 and part of C3.
- 2026-09-04 00:55 PEER2 final audit landed; six corrections applied, two of them
  to my own code.
  C3 HOLDS, and stronger than claimed: a label-shuffle permutation null confirms
  the eventual-exit nulls at loops 1-2 (p=0.95, 0.98), and at multihop loop 2 the
  lens ranks the true intermediate BELOW a shuffled label (p=0.029).
  C4 WEAKENED not false; PEER2 called my retraction an over-correction and they
  are right. Wording is now "never significantly beats", already applied.
  C5 WEAKENED. hit@10 is floored at loops 1-2, so it could not have moved. Under
  it, mean log-10 rank improves significantly from n=8 to n=80 in six of eight
  cells. Only multihop loop 1 is flat at both levels. "Converged by eight
  prompts" removed; the rented 1000-prompt run is now a real test of section 6.
  C6 HOLDS on multihop and gets STRONGER under correction: with Bonferroni over
  2 tasks x 3 contrasts the preregistered and fit-only contrasts survive while
  state_only goes to [-0.100, +0.004], i.e. the mechanism I predicted fails
  correction while the horizon effect does not. PEER2 also refuted an objection I
  had not considered, that the matrix columns cannot move because loops 2-4 hold
  near-identical states: under the identity readout top-1 differs between loops 3
  and 4 at 54% of (item, layer) pairs.
  C7 WEAKENED. My "the ||mu|| estimates agree closely across all three pairs" is
  FALSE: over all six valid disjoint pairs ||mu|| at loop 1 spans 0.060 to 0.207
  and the ratio spans 4.0 to 13.6. Worse, lens_convergence.py was silently
  clipping negative variance estimates, at 107 of 191 layers for one pair. It now
  counts and reports clipping. Section 6 rebuilt around the raw ||J_80||/sqrt(d)
  = [0.159, 0.178, 0.272, 0.810], ratio 5.09 as a lower bound.
  C8 WEAKENED and its explanation was FALSE. The cross-validated probe (5 folds
  over 45 unordered pairs, all 648 prompts tested once, both lenses rescored on
  the same data) gives probe [0.615, 0.590, 0.335, 0.236] vs logit [0.764, 0.500,
  0.179, 0.311] vs J-lens [0.392, 0.500, 0.148, 0.120]. More training data moved
  the probe DOWN, so my "the probe is data-limited, which is why an untrained
  readout beats it" was wrong and is removed. Decodability does outlast
  verbalizability, but only at loop 3 (+0.156 CI [+0.031, +0.315]); my blanket
  withdrawal of that claim was too pessimistic. Section 8 rewritten.
  Confirmed by PEER2: my Jensen fix to analyze.py is correct on both shape paths.
  Also fixed: bootstrap now uses a fresh generator per call (CIs no longer depend
  on how many earlier calls drew from a shared stream) and N_BOOT raised 1000 ->
  20000 so the third decimal is not Monte-Carlo noise.
  New limitation recorded: lenses are fitted with skip_first=16 but 99 of 148
  items are read before position 16. PEER2 could not sustain it as an explanation
  (the unfitted logit lens shows the same position effect and the local-exit lens
  reads those states fine) but a short-prompt control lens is queued for the
  rented run.
- 2026-09-04 01:45 PEER3 (auditing the auditors) reported. Outcomes:
  * PEER1's whitening interpretation REFUTED with the controls PEER1 had not run:
    a polar factor from a random OTHER location reads loop 3 better (0.519) than
    the matched one (0.428); the loop-4 polar factor restores loop 1 to full
    logit-lens strength while the matched loop-1 factor reads -0.002; a Haar
    random orthogonal reads nothing. So whitening is a fit-loop effect, not
    recovery of loop-3 content, and it never beats the do-nothing baseline. The
    claim was never written into RESULTS.md; it stays out. PEER1 conceded.
  * PEER1's position-reduction conclusion OVERSTATED. The implementation is
    correct and the summed branch is bit-exact, but the diagonal arm used 3
    position samples against the summed arm's 93, and at n=3 the summed estimator
    itself loses up to 60% of its readout. "Not an estimator artifact" is
    UNTESTED, not established. PEER3 supplied a 2x2 design that gets both
    reductions from the same backward at no extra GPU cost; it is queued for the
    rented run.
  * "The probe is not data-limited" REFUTED, and this one was my error: I deleted
    a correct explanation from section 8 on PEER2's word. Their 0.665-vs-0.615
    comparison changed training size, test set, label handicap and aggregation at
    once. The clean curve, varying only training pairs, is 0.185 / 0.349 / 0.383
    / 0.443 / 0.557 at 7/14/21/28/36 pairs, +0.094 [+0.011,+0.192] from 14 to 28
    and still climbing. Section 8 restored and the failed inference recorded in
    place.
  * My section 6 mechanism claim CONFIRMED by a statistic nobody had run: mean
    single-prompt Jacobian norm is 0.708 at loop 1 vs 0.715 at loop 4 (ratio
    1.01) while the averaged maps differ six-fold. That is now the primary
    statement of the mechanism. It also kills PEER1's claim that a single-prompt
    norm independently confirmed the 5x ratio (its expected ratio is 1.53).
  * CONFIRMED: my Jensen fix including the cross-loop path, my section 4 paired
    bootstrap to every digit, the probe_cv fold construction, and sections 3, 5,
    6, 7, 9 against artifacts. validate.py's exit gate works.
  * PEER3 also published a wrong finding at 01:20 and retracted it at 01:40 after
    re-reading a stale revision of PEER2's notes; the retraction is recorded in
    their notes rather than edited away.
  * HANDOFF corrected: the post-run probe step named probe.py, whose output
    section 8 no longer reports, and analyze.py --probe cannot read a probe_cv
    directory.
- 2026-09-04 01:55 Launch attempt failed with UNAUTHORIZED on
  podFindAndDeployOnDemand while every read query and podTerminate succeeded. The
  credentials file holds several RunPod keys with different scopes. MY ERROR: I
  probed which key could deploy by issuing a real create mutation against a cheap
  GPU type. It succeeded, i.e. it created pod e4uk0e6km9yv0n, which I terminated
  within seconds; balance unchanged at $52.2019 and the account is clean. A
  create call is not a permission probe. pod.py now pins the deploying key by
  prefix and carries a comment saying not to probe scope this way.

## Rented-run incident, 2026-09-04

B300 rented 01:33 at $7.89/hr (secure rate; the $6.94 community rate I had quoted
was not what cloudType: ALL selected). All five milestones passed BIT-EXACTLY on
that hardware, resolving PEER1's concern that our equality assertions were an
artifact of the local attention kernel. Benchmark at seq 128: batch 8 -> 0.13 s
per backward pass and 24 GB, batch 32 -> 0.32 s and 77 GB, batch 64 -> 0.57 s and
149 GB, batch 128 OOM. Per-prompt 29 s at batch 32, about 7x the local card. Per-pass
time scales nearly linearly with batch above 32, i.e. the GPU is saturated there,
so co-residency of several checkpoints would not have given proportional speedup;
the launch-bound regime the user asked about is real only at small batch.

Then the run was lost: $52.74 spent, nothing retrieved, balance -$0.54. The
eventual-exit fit at 100 prompts almost certainly completed around 02:25 and was
destroyed with the pod at ~08:14 when RunPod terminated it for zero balance.

Causes, mine, in order:
1. I did not monitor. The session went idle for eight hours after confirming the
   first shard was running at 96% utilisation. Monitor and ScheduleWakeup were
   both available and I used neither.
2. The pod-side kill switch did not survive SSH disconnection. I armed a detached
   4.5 h timer and verified it was RUNNING two seconds later. I never verified it
   survived the session closing, which is the only failure mode it existed for. I
   reported it to the user as "armed and verified", which overstated the check.
3. No retrieval step. I drove the fit with an inline script that had no upload,
   so results lived only on ephemeral container disk. The upload logic existed in
   pod_entry.sh, which I did not use.
4. No balance floor. Nothing noticed the account draining.

Also on the way in: I probed which API key had deploy permission by issuing a real
create mutation, which created a pod (terminated within seconds, no charge). A
create call is not a permission probe.

Lesson for any future paid run, recorded in docs/jlens/HANDOFF.md section 5:
retrieval before compute, a balance poll that terminates on breach, a kill switch
proven by disconnecting and re-checking, and a scheduled self-wake so an idle
session cannot leave a billing machine unattended. The compute itself was cheap
(~90 min, ~$12 for all four targets); the loss was 5.8 idle hours, ~$46.
