# JLens operational handoff

Status: `LOCAL_OBSERVATIONAL_ARITHMETIC_REPRODUCIBLE_LINEAGE_UNFROZEN`.

This handoff supersedes the Opus-era handoff preserved under
`history/2026-09-04-opus/`. It records operational state only; scientific
values and claim statuses live in `artifacts/jlens/final/analysis.json`.

## Established now

- The canonical report reproduces the retained arithmetic table, but the
  evaluation directories have no complete `evaluate.py` provenance record.
  Their lineage is therefore
  `LOCAL_OBSERVATIONAL_ARITHMETIC_REPRODUCIBLE_LINEAGE_UNFROZEN`; the fit size
  is not authenticated.
- The current validator JSON passes the complete M1–M5 and 18-comparison gate
  with `NUMERICAL_AND_PROVENANCE_PASS` and `HASH_BOUND` model/JLens byte
  provenance; the generated instrumentation status is
  `SUPPORTED_CURRENT_VALIDATION`.
- The retained arithmetic-probe arrays and Base/Thinking/RLTT checkpoint
  measurements are earlier-source/local evidence. They have not been
  re-certified against the repaired source/runtime and are excluded from the
  new B300 paid verifier and application claim. Their regeneration is not a
  launch requirement.
- The independent arithmetic verifier reports
  `MECHANICAL_ARITHMETIC_PASS` with mechanical-only scope; this does not
  establish provenance or scientific claim acceptance.
- The original documents and peer notes are byte-preserved and hash-checked.
- Fit, merge, evaluate, report, and publication paths now use the repaired
  validation and publication gates. Unsafe or incomplete resume state fails
  closed.

## Scientific endpoint

The retained final-exit average Jacobian lens is below the logit lens for
task-defined multihop intermediate-token readout at loops 1–3 and arithmetic at
loop 1. This is descriptive local, estimator-specific arithmetic with
unfrozen evaluation lineage; it is not evidence that the intermediate is absent
or causally erased.

The local/eventual convergence prediction is `REFUTED_PRE_FINAL`. Cross-loop
transfer is row-dominated on multihop in the retained directories nominally
labeled 32 and 80, but those fit-size labels are not authenticated and the
cause remains unidentified. Transport scatter is modeled from nested averages,
not directly measured per-prompt Jacobians.

## Deliberately open

- 1000-prompt or replicated fixed-n fitting: `NOT_RUN`.
- Matched position-by-reduction 2x2 estimator control: `NOT_RUN`.
- Direct per-prompt sufficient statistics: `NOT_RETAINED`.
- Matched recurrent-model/ablation evidence for architecture causality:
  `NOT_ESTABLISHED`.
- B300 result from the failed paid run: `NOT_RETAINED`.
- Submission readiness: `NOT_ESTABLISHED`.

None of these states may be upgraded from filenames, elapsed time, or a future
pod transcript.

## Rental boundary

No cloud pod was launched during this repair. The old recipe is rejected. Paid
execution remains a separate campaign requiring final review, explicit user
signoff, positive balance, hard budget/runtime limits, detached supervision,
results extraction, and verified termination. The new paid verifier covers the
B300 validation, paid evaluation/lens lineage, transport recomputation, and
canonical report regeneration; the retained probe and checkpoint material is
not part of that claim.

The non-negotiable code limits are a $5.00 minimum account reserve, $30.00
maximum spend, and 12,600-second provider TTL. The worker deadline is 900
seconds before the provider deadline. Before leasing, the results repository
must pass a write-read-list canary. HF authentication uses a protected,
explicit token channel sourced from a 0600 file, not ambient credentials.

No-cost full-tensor recovery uses
`venv/bin/python src/ouro_jlens/pod.py recover --state <state-file>` after the
normal successful controller has retrieved the small application bundle.
Failed runs still retrieve every available recovery shard immediately.

## Safe next command

Run the complete 14-suite normal and optimized commands in `REPRODUCE.md`.
Then read `RESULTS.md` and `REPRODUCE.md`. If further empirical work is desired,
freeze a new experiment contract and budget first; do not revive the archived
probe/checkpoint commands. Passing these checks establishes readiness only;
the pod still requires explicit user signoff.
