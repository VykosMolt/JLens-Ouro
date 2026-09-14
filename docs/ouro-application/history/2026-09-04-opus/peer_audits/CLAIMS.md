# Canonical claim set — Jacobian lens on recurrent Ouro-2.6B

Everything below is asserted in docs/jlens/RESULTS.md. The job of the swarm is
to falsify these, not to confirm them. A claim survives only if someone tried
hard to break it and failed. Write what you actually ran.

Repo: /home/moloch/ouro_project. Interpreter: venv/bin/python (or `source venv/bin/activate`).
Model: local snapshot, artifacts/hf_cache/hub/models--ByteDance--Ouro-2.6B/snapshots/1ed04250...
Lens library: ~/jacobian-lens @ 581d398, installed editable, UNMODIFIED. Read it.
Ouro modeling code: the snapshot's modeling_ouro.py. Read it. 48 physical layers
x 4 recurrent steps ("loops"), d_model 2048. Virtual index = ut*48 + layer, ut 0-based,
loop k = ut k-1.

C1  INSTRUMENTATION. jlens's ActivationRecorder keys by hooked index, so on Ouro's
    native forward each loop overwrites the last. src/ouro_jlens/recurrent.py wraps each
    (loop, layer) as a LoopTap that fires only when Ouro's own current_ut kwarg matches.
    Claim: recording is non-interfering and correctly addressed; unembed(pre-norm output
    of (ut,47)) equals the model's own exit_at_step=ut logits bit-exactly for all four ut;
    stock jlens on physical layer L equals the recurrent (loop4,L) Jacobian exactly.
    Evidence: src/ouro_jlens/validate.py, artifacts/jlens/validation/milestones.json.

C2  LENS-FREE. KL(exit k || exit 4) = [3.83, 0.65, 0.09, 0] nats. Exit-k top-1 equals the
    final top-1 on [0.39,0.67,0.80,1.0] of multihop and [0.49,0.84,0.96,1.0] of arithmetic
    items. On arithmetic the loop-1 exit's top token IS the latent intermediate 18% of the
    time. Depends on no fitted lens.

C3  MAIN RESULT. A Jacobian lens targeting the FINAL exit reads ~nothing from early
    recurrent states, while the same states are read fine by a local-exit Jacobian lens and
    by the vanilla logit lens. multihop mean-over-layers excess hit@10, loops 1-3:
    local-exit [0.065, 0.107, 0.167], eventual-exit [0.000, -0.007, 0.016], logit
    [0.053, 0.095, 0.139].

C4  The eventual-exit J-lens never beats the logit lens at any loop on either task.

C5  FIT SIZE. 8 -> 80 fitting prompts changes nothing: multihop any-layer excess loop 1
    [0.005, 0.003, 0.003, 0.007], loop 4 [0.479, 0.481, 0.469, 0.469].

C6  CROSS-LOOP. Matrix of (J fitted at loop i) applied to (state at loop j) is row-dominated:
    variance share fit-loop 0.99, state-loop 0.01 (multihop). The preregistered H1 contrast
    is -0.113 CI [-0.150,-0.077] but the state-only contrast is only -0.044 CI [-0.084,-0.007]
    and the fit-only contrast is -0.181 CI [-0.237,-0.133]. Conclusion drawn: H1 confirmed as
    a number, refuted as a mechanism; horizon governs transport, not which loop's state.

C7  TRANSPORT COHERENCE. Across three disjoint fit pairs the converged mean transport
    ||mu||/sqrt(d) into the final exit is ~[0.16, 0.17, 0.27, 0.82] for source loops 1-4,
    i.e. ~5x smaller for loop-1 sources. Per-prompt scatter is comparable across loops.

C8  SUPERVISED REFERENCE. On "(a + b) * c = " with a leak-free unordered-pair split, per-loop
    candidate top-1: probe [0.67,0.65,0.34,0.33], logit lens [0.89,0.62,0.18,0.40], J-lens
    [0.55,0.56,0.22,0.15]. The logit lens EXCEEDS the supervised probe at loop 1.

C9  MECHANISM. Ouro trains every recurrent step to decode through the shared lm_head
    (paper: "at each recurrent step i, an exit gate predicts the probability p_i of exiting,
    and a language modeling head computes the task loss"), so every loop's residual is
    already output-aligned and there is little basis rotation for a transport map to correct.
    This is offered as the explanation for C4.

C10 RENTED-RUN READINESS. src/ouro_jlens/setup_b300.sh then run_b300.sh 100 32 will run to
    completion on a fresh single-GPU pod and is resumable shard-by-shard.

## Known, already-fixed defects (do not re-report; DO check the fixes are right)
- readout position now the token preceding the target (was a trailing-space token)
- controls matched by kind (numeric vs operation), was mixed
- items not (item,name) slots as the aggregation unit
- probe split by unordered pair (was ordered -> transpose leakage)
- validate.py milestone 5 retries on OOM
- merge folds pairwise (was loading all shards at once)
- is_correct requires a token boundary
