# Application evidence pack (inputs for the human-written submission)

Purpose: everything verified that the MATS 12.0 write-up may draw on, mapped onto
Neel Nanda's rubric. This is NOT the submission. The executive summary and the
form answers must be written by the applicant in their own voice (the
application doc says LLM-written summaries are a significant negative signal).
Numbers below are copied from `artifacts/jlens/final/analysis.json` and
`docs/jlens/RESULTS.md`; do not state anything stronger than the claim status
shown.

Deadline: Fri Sept 4 11:59pm PT = Sat Sept 5 08:59 CEST. The doc states
"extensions available until Sept 11".

## 1. What Neel asks for (from the application doc)

Deliverables
- Application form Qs: read first, used as a filter. Prioritise. Concrete:
  name the model, the key experiment, the surprising number.
- Google Doc, anyone-with-link. First 1-3 pages = executive summary, max 600
  words, WITH graphs. Then enough detail to follow without reading code.
- Code optional (he feeds it to agents). Time-tracking screenshot encouraged.
- 20 h research + 2 h for exec summary/form. Not counted: GPU rental/setup,
  waiting for jobs, form answers, prior general learning. Counted: project
  code, project-specific reading, analysis, thinking, write-up.

Rubric (in his words)
- Clarity: "If I understand what you're claiming, what evidence you're
  providing, and think that evidence supports your conclusion, that instantly
  puts you in the top 20%."
- Skepticism: "Most research results are false, especially the exciting ones."
  Negative/inconclusive results well analysed beat weak positives.
- Baselines: random/probe/logit lens. Compare.
- Sanity-check your agents: "I want scholars with value add over prompting
  Claude myself." Document what you verified and how; he checks numbers
  against the write-up.
- Show raw examples (randomly selected) if data quality matters.
- Narrative over chronology; one or two insights deep, not ten shallow.
- Recent models only. Simplicity. Honest limitations. Plausible over ambitious.

Problem fit (verbatim from his list, "Improved Interpretability Methods"):
"J-Lens ... From a scientific perspective, what is J-Lens actually doing? How
much better is it really than logit lens and tuned lens and why? How much does
it hallucinate?" Also "Monitoring" under Applied Interpretability.

## 2. The story (candidate framing; rewrite)

Question: can a Jacobian lens read task-defined intermediate variables out of a
recurrent-depth model (Ouro-2.6B: 48 shared layers x 4 recurrent loops = 192
virtual locations), and does a monitor reading an early loop see what the
completed recurrence will eventually say?

Headline (descriptive, local, lineage unfrozen):
1. The prompt-averaged final-exit J-lens reads verbalisable intermediates
   WORSE than the plain logit lens at every pre-final loop on multihop, and at
   loop 1 on arithmetic. It only catches up at the final loop.
2. The local-vs-eventual gap does NOT shrink with recurrent depth (preregistered
   H2 refuted): mean KL(local readout || eventual readout) is 4.9, 4.6, 6.5
   nats at loops 1-3; last-eight-layer top-1 agreement 0.002-0.018.
3. Cross-loop: the readout is determined by the loop the Jacobian was fitted
   at (fit-loop variance share 0.98/0.99), not by which loop produced the
   state (0.01). Descriptive association, cause not established.
4. Plausible (not demonstrated) mechanism: Ouro trains every recurrent exit to
   decode through the shared lm_head, so the logit lens is "trained in" at
   every loop, while a Jacobian to the final exit averages over prompt-specific
   routes that later recurrence rewrites.

Why Neel might care: a direct empirical answer to "how much better is J-Lens
than logit lens, and why" on a model class where the answer flips; a concrete
monitoring blind spot; a negative preregistered result honestly reported.

## 3. Verified numbers (copy exactly; statuses in brackets)

Model: ByteDance/Ouro-2.6B rev 1ed04250da1a9936042725d302e81c8fa2ab5abd.
J-lens code rev 581d398613e5602a5af361e1c34d3a92ea82ba8e. 192 virtual layers.
Stimuli: Anthropic's released multihop (93 -> 90 clean items, 100 slots) and
order-operations numeric (55 -> 51 items). Prompt-leakage exclusions applied.

Main readout, any-layer excess hit@10 over matched same-kind controls
[SUPPORTED_LOCAL_UNFROZEN]:

| population | readout | L1 | L2 | L3 | L4 |
|---|---|---:|---:|---:|---:|
| multihop (90) | J-lens | +0.003 | -0.025 | +0.084 | +0.451 |
| | logit lens | +0.358 | +0.458 | +0.500 | +0.483 |
| | J minus logit | -0.355 | -0.483 | -0.416 | -0.032 |
| order-ops (51) | J-lens | +0.054 | +0.273 | +0.181 | +0.193 |
| | logit lens | +0.306 | +0.152 | +0.131 | +0.213 |
| | J minus logit | -0.252 | +0.121 | +0.050 | -0.020 |

Paired item-bootstrap 95% CI for J minus logit (unadjusted, 8 intervals):
multihop [-0.460,-0.250] [-0.583,-0.382] [-0.519,-0.313] [-0.105,+0.039];
arithmetic [-0.354,-0.136] [-0.005,+0.246] [-0.081,+0.186] [-0.069,+0.023].
Say "below zero at multihop loops 1-3 and arithmetic loop 1; unresolved
elsewhere". Do not say "never beats"; say "never significantly beats".
J-lens is not blind: arithmetic candidate-set top-1 = 0.608 at loop 1.

Lens-free exit agreement (top-1 of exit k vs final exit, per item):
multihop 0.409 / 0.656 / 0.785 / 1.000; order-ops 0.527 / 0.855 / 0.964 / 1.000.

Local-exit vs eventual-exit [REFUTED_PRE_FINAL for "gap shrinks"]:
| loop | mean KL local||eventual | last-8-layer top-1 agreement | local-minus-eventual excess |
|---|---:|---:|---:|
| 1 | 4.903 | 0.002 | +0.065 |
| 2 | 4.567 | 0.018 | +0.114 |
| 3 | 6.526 | 0.008 | +0.151 |
Caveat to state: local lenses were fit on 24 prompts, eventual on 32.

Cross-loop transfer [descriptive]: multihop fit-loop share 0.98 (n32) / 0.99
(n80), state-loop share 0.01; arithmetic unstable (0.10 / 0.44, interaction).

Fit-size ladder (nested 8/32/56/80) [INCONCLUSIVE for n=1000]: multihop
loop-1 excess stays ~0 (+0.096 thresholded variant), loop 4 stays ~0.45; more
fitting prompts do not recover early-loop readout. Fits are nested, not
independent replicates. Fit-size labels are not authenticated by provenance.

Supervised probe baseline (arithmetic (a+b)*c, 17-way, 576 fold-trainable
prompts / 39 unordered pair clusters, layer chosen on other folds)
[SUPPORTED_LOCAL_UNFROZEN; family-wide INCONCLUSIVE]:
| readout | L1 | L2 | L3 | L4 |
|---|---:|---:|---:|---:|
| supervised probe | 0.594 | 0.559 | 0.356 | 0.248 |
| logit lens | 0.764 | 0.500 | 0.179 | 0.311 |
| J-lens | 0.392 | 0.500 | 0.148 | 0.120 |
Chance 0.059; majority baseline 0.125 (fair population). J-lens loop-4 does
not exceed the majority baseline. J minus logit below zero at L1
[-0.548,-0.221] and L4 [-0.284,-0.105]. Probe minus logit below zero at L1,
above zero at L3 [+0.061,+0.342]. Pointwise intervals only.

Transport [INCONCLUSIVE]: modeled mean-map norm by loop [0.134, 0.173, 0.269,
0.807] (loop-4/loop-1 ratio ~5). Modeled scatter, not measured per-prompt
norms; two invalid moments excluded. Do not describe as single-prompt norms.

Instrumentation [SUPPORTED_CURRENT_VALIDATION]: M1-M5, 18/18 required
comparisons bit-exact (hooked==plain forward, exit equality at all four loops,
recurrent identity, distinct VJPs incl. loop 1, stock consistency).
Re-run on the B300 (pod 9, 2026-09-05 04:03 CEST): pass=True,
numerical_pass=True, bit_exact=True, 18/18 - published to
Vykos/ouro-jlens-results/jlens-b300-20260905-0359/artifacts/validation/.

B300 100-prompt fit: IN PROGRESS at time of writing. Fill in or state
"not completed by submission" - do not imply otherwise.

## 4. Figures for the executive summary

- artifacts/jlens/eval/round1_exit3x32/fig4_local_vs_eventual.png  (headline:
  KL, agreement, local vs eventual excess by loop)
- artifacts/jlens/eval/round1_exit3x32/fig1_readout_heatmaps.png  (loop x layer
  heatmaps; J-lens vs logit lens; excess over matched controls)
- artifacts/jlens/eval/round1_exit3x32/fig5_probe_vs_lens.png  (baselines:
  probe vs J-lens vs logit lens, per loop)
- artifacts/jlens/eval/fig6_fit_size.png  (does more data recover early-loop
  readout: no)
Label axes/loops in captions; a reader has zero context.

## 5. Sanity-checking narrative (write this in; he weights it heavily)

- Any-layer control bug (Jensen): control used max(mean over names) while own
  score used mean(max over layers) -> every excess inflated. Found in peer
  review, fixed in analyze.py, all six evaluation directories re-derived.
- Probe leakage: initial fold split by ordered (a,b) pairs leaked mirror pairs;
  fixed to unordered-pair clusters (648 -> 576 fair prompts, 39 clusters).
- Correctness matching: prefix match counted "11" as correct for "1" and
  "daytime" for "day"; replaced with boundary-aware match (72/148 correct).
- Retracted overclaims after audit: "J-lens never beats logit lens" -> "never
  significantly beats"; "0.708/0.715 single-prompt norms" were modeled scatter
  with two invalid moments silently zero-clipped.
- Recorder validated bit-exact against the stock forward (M1-M5) before any
  lens was fit; validation replicated on the B300.
- Preregistered H1/H2; H2 refuted and reported as such.
- Item-level (not slot-level) bootstrap; intervals labelled unadjusted;
  matched controls by intermediate kind; leakage-confounded stimuli excluded.
- Independent re-derivation: headline any-layer numbers recomputed from
  arrays.npz outside the pipeline (state this only if you actually did it).

## 6. Limitations to state plainly

- One model; the effect may be specific to per-step shared-head training.
  No matched recurrent model without that supervision was evaluated.
- Fits use <=80 prompts (nested); no independent fixed-n replicates; the
  100-prompt B300 fit is [in progress / not completed].
- Eight pointwise CIs, no family-wide adjustment; "unresolved" != "equal".
- Intermediates are task-defined tokens; causal use is not shown.
- The estimator 2x2 (position-summed vs per-position; mean vs alternatives)
  control was not run, so "estimator-specific" is a hypothesis.
- Evaluation lineage for the local artifacts is unfrozen (no producer
  provenance); the B300 run is the provenance-bound replication.
- A paid B300 run on 2026-09-04 was lost to an unsupervised session (~$52);
  the controller was rebuilt with fail-closed supervision before rerun.

## 7. "What I would do next" (short list he likes)

- Matched 2x2 estimator control; fixed-n replicate fits; 1000-prompt fit.
- A recurrent model without per-step shared-head supervision, or an Ouro
  ablation, to test the mechanism.
- Tuned lens baseline (he names it explicitly).
- Per-prompt Jacobian statistics to test prompt-specific rewriting directly.

## 8. Form-question raw material

- Model: Ouro-2.6B (ByteDance, 2025 recurrent/looped transformer), 4 loops.
- Key experiment: task-defined intermediate readout at all 192 locations with
  J-lens vs logit lens vs supervised probe, matched controls, item bootstrap.
- Surprising number: multihop J-minus-logit excess -0.483 at loop 2
  (CI [-0.583,-0.382]); local->eventual KL 4.6-6.5 nats that does not shrink.
- Biggest limitation: one model, <=80-prompt fits, unadjusted CIs, cause not
  demonstrated.
- Tools: Claude Code (Fable), Codex (Sol) for the controller rebuild, three
  independent reviewer agents; every headline number re-derived by hand or
  peer-checked.

## 9. Time accounting (applicant fills in)

Counted: recorder + validation code, estimator integration, evaluation and
analysis code, reading Ouro/J-lens papers, analysis, write-up.
Not counted: RunPod rental/controller engineering, waiting on fits, form.
Report honestly; a Toggl-style screenshot is encouraged, not required.
