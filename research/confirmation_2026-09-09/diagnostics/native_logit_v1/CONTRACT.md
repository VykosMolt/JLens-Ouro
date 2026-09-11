# Native-logit diagnostic v1 — frozen contract

Fixed by root before implementation, following the user's instruction "Run the diagnostic first". Facts and file:line citations: `/tmp/claude-1000/-home-moloch-jacobian-lens/18b73cc8-31a8-4b0b-b8c9-be0441298f55/scratchpad/diagnostic_exploration.md`, [attempt05 forensics](../../reviews/attempt05_native_failure_forensics.md).

## Objective

On the attempt05 GPU type (community RTX 5090) and the frozen precision_v1 runtime, using only the three historical development items:

1. rerun attempt05's native development comparison unchanged and record the result;
2. localize any native-versus-recomputed logit difference to the final RMSNorm, the lm_head or both, by recomputing each at defined shapes from tensors captured in the native forward;
3. determine whether any defined recomputation reproduces native final logits bit-exactly for all three items, repeatably.

The diagnostic produces evidence for a later protocol decision. It does not change, re-evaluate or relax the frozen confirmation gate.

## Scope

In scope:
- This directory: bundle, run spec, output contract, verifier, worker, run config, tests, runbook, freeze receipt and reviews.
- One new lease under `cloud_leases/`, created by the unchanged controller.
- Post-run analysis and documentation.

Out of scope: confirmation items, lens banks, Huginn, controller source, `resources/prior_debit.json`, and every existing frozen, lease or evidence file.

## Repository facts

- **Native forward.**
  - The final norm runs at the end of each of four recurrent steps.
  - With `early_exit_threshold` 1.0, the forward stacks the four normed step states, gathers one per token and applies lm_head to `[1,S,2048]`, keeping all positions.
- **RMSNorm.** It upcasts to float32, takes the mean of squares and rsqrt, casts to BF16, then multiplies by a BF16 weight.
- **lm_head and unembed.**
  - lm_head is an untied BF16 `Linear(2048→49152)`.
  - `unembed` uses the same norm and lm_head objects.
  - `virtual[191][0,-1]` is the final step's norm input for the last token.
- **Shapes reaching norm and head.**

  | Path | Shape |
  |---|---|
  | Native | `[1,S,2048]`, with S = 20, 15, 13 |
  | Gate recomputation | `[2048]` |
  | Scorer exits | `[160,2048]` |
  | Scorer readouts | `[190–192,2048]` |
  | Historical evaluation | `[148,2048]` |

- **Controller.** The controller can run a bankless run with its own run spec, `stages: {}` contract, pinned `validate_outputs` verifier and worker, provided the final manifest is `complete`. A failed final is never accepted and bills until the watch deadline.
- **Admission.** All three must hold:
  - balance ≥ cap + 2.528676;
  - (setup + compute + preservation + 600)/3600 × 0.7064384 + 0.15 ≤ min(25 − ledger total, cap);
  - cap ≤ 4.0.

## Measurements

All measurements run on the GPU inside one worker process, for each item.

- **M0 — runtime gate.** Precision_v1 `worker.py`'s source, snapshot, imported-module and full precision-dictionary checks, unchanged. A failure publishes `failed`.
- **M1 — attempt05 reproduction.** Run the unchanged `readouts.native_development`. Record its three `torch.equal` results and differing-element counts.
- **M2 — instrumented native forward.** Same call and activation recorder as M1, plus read-only hooks capturing:
  - every final-norm input and output;
  - the lm_head input and output;
  - `output.logits`.

  Identify the last token's exit step by matching the lm_head input row bitwise against each step's norm output.
- **M3 — recomputations,** each performed twice:
  - **Norm:** apply it to the exit step's last-token norm input at `[2048]`, `[1,2048]`, `[1,1,2048]`, `[S,2048]` and `[1,S,2048]`. Also at `[148,2048]` and `[160,2048]`, with the target row first and last and the other rows filled from the item's own positions, cycled.
  - **lm_head:** apply it to the native lm_head input row at the same shapes.
  - **Composites:**
    - `model.unembed` of the FP32 `virtual[191]` row, which is the attempt05 path;
    - lm_head∘norm at `[1,1,2048]`, `[1,S,2048]` and `[160,2048]`.
- **M4 — determinism.** Repeat M2 once to test run-to-run determinism.

Every conclusion rests on exact comparisons: `torch.equal` and bitwise element counts. Maximum and mean absolute differences and top-1/top-10 agreement are descriptive only. Mismatches are data: once M0 passes, the worker publishes `complete`.

## Outputs

Small files, under 50 MB in total:
- a runtime and provenance record (precision dictionary, versions, GPU identity);
- M1 tensors, plus captured and recomputed tensors in their produced dtypes;
- a comparison record.

The verifier recomputes every recorded comparison from the saved tensors and does not fail on mismatches.

## Invariants

1. **Controller.** Sources are byte-identical to the six pinned attempt05 hashes. No controller edits.
2. **Runtime.** Identical to precision_v1: image digest, `requirements.lock`, `environment.json`, model manifest and revision, bootstrap environment, and shared library sources. The precision dictionary must equal the frozen original before any measurement.
3. **Items.** Exactly precision_v1's `development_item_names`, loaded through the unchanged old-item path. The bundle contains no `population.json`, no `benchmark.json`, no bank and no confirmation prompt text.
4. **Stages.** `bank_records: {}` and `stages: {}`. Never run `parallel_upload.py` or `accept-development`.
5. **Frozen files.** Existing frozen bundles, FREEZE receipts, `prior_debit.json`, attempt01–05 lease directories and preserved evidence stay byte-identical. Check their hashes before and after.
6. **Hooks.** Hooks only read: detach or clone, and return None.
7. **Spending.**
   - Job cap ≤ $1.50; setup, compute and preservation budgets ≤ 2400, 1200 and 900 s.
   - Before create: a fresh observation showing zero pods and zero spend, and recorded user approval covering creation and export to the provider-assigned address.
   - Teardown verified by three absence confirmations, then a final read-only account observation.

## Forbidden shortcuts

- Editing frozen files, `prior_debit.json` or controller code.
- Uploading a precision_v1 bundle or any bank.
- Using tolerances in any equality conclusion.
- Presenting mocks or CPU outputs as GPU evidence.
- Relaxing the precision check, or using a different GPU type.
- Retrying after a paid failure without new user approval.
- Skipping verification or review.

## Acceptance before any paid resource

- **A1 — freeze.** A receipt hashing every file freezes the bundle, run spec, contract, verifier, worker, run config and runbook. Archive members match the directory.
- **A2 — admission.** The unchanged controller's offline plan admits the run config with a job cap ≤ $1.50 against the current ledger.
- **A3 — verifier tests.** The verifier accepts a conforming synthetic output set. It rejects:
  - missing or extra files;
  - wrong shapes or dtypes;
  - non-finite values;
  - binding or precision mismatch;
  - non-complete manifests;
  - comparison records inconsistent with the tensors.
- **A4 — CPU pipeline run.** The measurement code runs end to end on CPU, using the real local snapshot if memory allows, otherwise a small random-weight model built from the pinned `modeling_ouro.py`. It is labeled pipeline validation only.
- **A5 — lifecycle smoke test.** Run it with the unchanged controller and an isolated ledger: publish, retrieve, verify, accept, mock deletion.
- **A6 — bundle audit.** No bank, population, benchmark or confirmation prompt, and shared files byte-identical to precision_v1.
- **A7 — review.** Independent verification and adversarial review pass, with findings repaired and re-verified.

## Acceptance after the run

- The complete manifest is retrieved, hash-verified and accepted.
- The precision dictionary equals the frozen original.
- All measurements are present for the three items.
- Termination is verified, and the final observation shows no pods and zero hourly spend.
- Spend stays within the cap.
- An independent CPU reanalysis reproduces every recorded comparison.

## Assumptions

- Community RTX 5090 capacity is available.
- The pinned image, wheels and model revision still download.
- The balance is at least cap + 2.528676 at creation.
- SSH is adequate for small transfers.

## Stop and report to the user if

- the controller would need code changes;
- admission fails;
- the GPU type is unavailable;
- an approval or permission is denied;
- the pod fails setup, the runtime gate or any safety limit (tear down, preserve and report; do not retry);
- any invariant would be violated.
