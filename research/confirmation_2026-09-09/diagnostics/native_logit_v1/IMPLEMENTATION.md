# Native-logit diagnostic v1: implementation record

This record covers the implementation of [CONTRACT.md](CONTRACT.md) up to, but not including, A7. The work used no network, provider, SSH or GPU. It created no lease and ran no controller `create`, `watch` or `terminate`, no `parallel_upload.py` and no `accept-development`. [FREEZE.json](FREEZE.json) is not bound by itself, and neither is this file.

## Deliverables

| Deliverable | SHA-256 | Notes |
|---|---|---|
| `FREEZE.json` | `178f2f924d1421102184199842a4b06f20466ee1bff96ddd2be2e60e932970ea` | Binds all 45 bundle files, the archive, `CONTRACT.md`, `RUNBOOK.md`, `build.py`, `observe_account.py`, `run_config.json`, `tests/*` and `evidence/*` |
| `bundle.tar.gz` | `2d0e2a09304b1d2a2da478c6e5adea58ad2af023ee343030c72773d42a1a5032` | 205,065 bytes; 45 members, all `bundle/…` regular files |
| `bundle/evaluation/worker.py` | `cdacad32d927f4961c2bafb25a7bc919dff205d457698e3ad2092c93247baad5` | New: M0 to M4 |
| `bundle/evaluation/validate_outputs.py` | `0afb9d9031b1ace14435abbd318a19defc1ff7688724606c522b28311b0b64f8` | New verifier; the worker imports its comparison code |
| `bundle/frozen/run_spec.json` | `a9e54b9a94de17ee4c6681094851c5268df5dc8bc57dd270ce6f2d2442e583d6` | See below |
| `bundle/frozen/output_contract.json` | `d143f7d8a378e3f813af35671f91470cb628320df497f103fa13358d50aa9cf7` | Five files, `stages: {}`, checks `loadability` and `numerical` |
| `run_config.json` | `289843502053d11bbd07d0acd7460d5949564e141a2ea32466a92336e77168a5` | Budgets: setup 2,400 s, compute 1,200 s, preservation 900 s, transfer 600 s; job cap $1.50 passed on the command line |
| `RUNBOOK.md` | `51a6c92b15c7f02a3d22a5da690d9d9d62cb5dceec00cd5f609f65bec76b9a04` | Paid-run commands and failure branches |
| `build.py` | `08ff4e2e93e2183684f3c68ba55cbfd031898c2eedd36eb2e8815721de7c37d2` | Subcommands `bundle`, `archive`, `freeze`, `verify` |
| `observe_account.py` | `4a9d531896da81df8dd1dddefb073b8f1338c9cab0eaae4016eeff65dc6979c3` | Read-only account and pod observation |
| `tests/inventory.py` | `ff630c39053c98ba1cb37302c6d4c8c3cf233a2fbc7abde412d87a31c89fa3e3` | Invariant 5 check |
| `tests/admission_plan.py` | `c34bbeedca934afdde2573564ee7541ae716b0fdba383e300f372ba6a5aa11e4` | Produces `evidence/A2_admission_plan.json` (`3c9da310…`) |
| `tests/verifier_tests.py` | `586dbebf071a9ca42fdcd9d64c86cf16fed3f7c8f4b33f835e7e0f440dc92e6e` | Produces `evidence/A3_verifier_tests.json` (`ca11a80c…`) |
| `tests/cpu_pipeline.py` | `38e6d66c514bc36743b1f9dfb58e4f5435ac07755db237d4dc90ac6afda24115` | Produces `evidence/A4_cpu_pipeline.json` (`09da96d4…`) |
| `tests/lifecycle_smoke.py` | `61e28db7bec9db64e1ece188d1fa9d8f96206a227ed347b4084fee3a8abaea2f` | Produces `evidence/A5_lifecycle_smoke.json` (`fa4979fb…`) |
| `tests/audit_bundle.py` | `a07eaad9c4aa22e6452bb66330bfae634bbcb664add11fca9e14461141363c4e` | Produces `evidence/A6_bundle_audit.json` (`36d38eee…`) |
| `tests/common.py` | `e3007629d266c2349d084192c4552ffb3f0ef4812f6445cfdf6cbc854cd64e95` | Shared test helpers |

The run spec contains:
- `bank_records: {}`;
- a source record for every non-frozen bundle file (43);
- records of `CONTRACT.md` and the precision_v1 `FREEZE.json`;
- `development_item_names`, `model_identity`, `model_record` and `original_precision`, copied verbatim from precision_v1.

It contains no population, benchmark or arm fields.

The bundle consists of:
- the 41 shared files, byte-identical to the precision_v1 freeze: `controller/` (7 files), `legacy/`, `ouro_project/`, `repo/`, and `evaluation/{artifacts,bootstrap,download_model,launch,readouts}.py`;
- the two new sources;
- the two frozen JSON files.

## Measurement design

The run order is M0, then M1 over all three items, then for each item M2, M3 and M4.

- **M0** keeps the precision_v1 worker's statements for four checks, unchanged:
  - sources;
  - environment and snapshot;
  - imported modules;
  - whole-dictionary precision.
- **M1** is the unchanged `readouts.native_development`.
- **M2** makes the M1 call `hf_model(input_ids=ids, use_cache=False)` inside `ActivationRecorder(model.layers, range(192))`.
  - Forward hooks on `hf_model.model.norm` and `hf_model.lm_head` keep `.detach()` references and return None. They allocate nothing during the forward.
  - The worker requires four norm calls and one lm_head call.
  - Captures are copied to CPU only after the forward returns.
- **Exit step:** the unique step whose last-token norm-output row equals the lm_head input row bitwise.
- **M3** builds every input on CPU from the M2 captures, then moves it to the device as a fresh contiguous copy.
  - It saves only the target row(s) of each output, in the produced dtype.
  - The whole sequence runs twice: norm, lm_head, lm_head∘norm, then `unembed(virtual191)`.
- **M4** repeats M2.
- **Records:** `comparisons.json` holds 255 records.
  - Each has `torch_equal` and `differing_elements` (an int16 or int32 bit-view count). These decide conclusions.
  - Each also has `max_abs_diff`, `mean_abs_diff` (elementwise float64 and `math.fsum`, so recomputation is exact), and, for vocabulary rows, top-1 and top-10 agreement. These are descriptive.
- **Verifier checks:**
  - a complete final manifest;
  - an exact file set on disk;
  - payload hashes;
  - binding, sources, model, precision and GPU name;
  - schema, dtypes, shapes and finiteness;
  - equality of the whole comparison record with its recomputation.

  It never fails on measured mismatches.

## Acceptance evidence (A1–A6)

All commands use `/home/moloch/ouro_project/venv/bin/python -B`. Temporary work directories were under the session scratchpad.

**A1 (freeze).** `build.py bundle`, then `build.py archive`, then (after the tests) `build.py freeze` and `build.py verify`. Result: `"status": "verified"`.
- The archive members equal the directory byte-for-byte.
- Every evidence file carrying `bundle_records` equals the frozen bundle.

**A2 (admission).** `tests/admission_plan.py --work DIR` ran the unchanged `lease.py plan` from a byte-identical copy. Result: admitted.
- Balance: $5.9138717458 from `resources/attempt05_final_account_observation.json`.
- Prior bound $21.645567963766442; remaining $3.354432036233561.
- 5,100 s gives a bound of $1.1507876712328766, which is at most min(remaining, $1.50).
- Minimum balance: $4.028675539145375.
- The ledger (120 entries) hashed identically before and after.

**A3 (verifier tests).** `tests/verifier_tests.py --work DIR` ran on synthetic random tensors, each case through `artifact_handoff.semantic_validate` of the unchanged controller on the pinned verifier path.
- The conforming set passed, including 158 of 255 unequal comparisons.
- All 25 mutations were rejected, each with its expected message:
  - missing or extra files, in the manifest or on disk;
  - payload bytes differing from the manifest;
  - wrong shape (capture, recomputation);
  - wrong dtype (capture, M1);
  - non-finite values (capture, M1);
  - a missing recomputation;
  - provenance-binding and run-spec-binding mismatch;
  - source, precision and GPU mismatch;
  - failed, stopped and stage manifests;
  - a flipped equality, a changed count, a removed record, a changed exit step, and a tensor changed after its record was written.

**A4 (CPU pipeline).** `tests/cpu_pipeline.py --work DIR` is pipeline validation only: CPU kernels on the real local snapshot, not GPU evidence. MemAvailable before loading was 22.1 GiB against a required 16 GiB. Result: passed in 27 s.
- `_environment` and `_snapshot` passed, and the precision dictionary equals the frozen original.
- The CUDA-only `_evaluation_runtime` did not run.
- Token counts: 20, 15 and 13. The exit step was 3, uniquely matched, for every item and both runs.
- Payloads totalled 36,921,970 bytes, and the saved record equals its recomputation from the reloaded tensors.
- On CPU, 34 records were unequal: M1 native versus unembedded, and lm_head at `[2048]`, `[1,2048]` and `[1,1,2048]`, including their composites. This is CPU behaviour only.

**A5 (lifecycle).** `tests/lifecycle_smoke.py --work DIR --cpu-results A4/results` used fixture provenance, pod, account, SSH and provider. Result: passed.
- The controller ran from a byte-identical temporary copy.
- **Ledger isolation:** `lease.LEDGER` was patched in-process with `unittest.mock.patch.object` to a temporary ledger. No pinned file was edited. The real ledger hashed identically before and after.
- `make_plan` read the real `run_config.json` and `prior_debit.json`.
- The worker's `Publisher` and `save_outputs` published a complete final manifest. `sync_results` with `LocalTransport` retrieved it, the pinned verifier passed it, and it was accepted.
- The controller's own `test_lifecycle.watch_fixture` then mock-deleted it with class `accepted_artifacts`, followed by three absence confirmations and status `terminated`.

**A6 (audit).** `tests/audit_bundle.py`. Result: passed.
- The 41 shared files equal both the precision_v1 freeze and its bundle.
- The controller equals both the attempt05 LEASE pins and `attempt_05/monitor`.
- There is no population, benchmark, plan, bank or tensor file, and `bank_records` and `stages` are empty.
- No bundle byte contains any of the 160 confirmation prompts, searched in raw, ASCII-JSON and UTF-8-JSON forms.
- The development items are historical `lens-eval-multihop` items.

**Invariant 5 (this session).** Every entry under the round root except this directory was hashed before the first execution and again after the freeze. The result was 5,146 entries, none added, removed or changed. The frozen `tests/inventory.py` reproduces the same inventory and reports `unchanged`. No `__pycache__` exists in this directory.

## Deviations and interpretations

1. `evaluation/download_model.py` is shipped although the delegation's shared-file list omitted it. `bootstrap.py:61` executes it, and it is byte-identical to precision_v1.
2. `controller/` includes `test_lifecycle.py`, as precision_v1's controller does. The six pinned sources are identical to attempt05.
3. The contract names only the shape for the `lm_head∘norm` composite at `[160,2048]`. It is recomputed with the target row both first and last, like the norm and lm_head cases.
4. If the exit-step match is not unique, the recomputations use step 3, which is the input of `virtual[191]`. The matching steps are recorded. This keeps "complete once M0 passes" intact; it did not occur on CPU.
5. M3 inputs are fresh contiguous device copies of the CPU-captured tensors, not the live forward tensors. M3 saves only target rows, not whole recomputed outputs.
6. M2 omits `native_development`'s per-block physical hooks; the contract specifies the same call and recorder only. M2 and M4 logits and `virtual191` are compared with M1.
7. The verifier also requires the GPU name `NVIDIA GeForce RTX 5090`, the value recorded by attempt05.
8. Worker edits relative to precision_v1 are limited to the following. All M0 check statements are otherwise unchanged.
   - Removed: the bank loop, the benchmark and population copies, the development gate and every confirmation stage (invariants 3 and 4).
   - Removed: unused imports.
   - The stop message was shortened.
9. The A4 random-weight fallback is not implemented. Memory sufficed, so the script refuses below 16 GiB instead.
10. The lease root is `cloud_leases/diagnostic_native_logit_v1`, a diagnostic name rather than `attempt_06`.

## Open risks

- **GPU payload size.** On the pod, `logits` and `lm_head_output` are separate CPU copies; on CPU they shared storage. The estimated total is about 46.4 MB, under the 50 MB contract limit but with a small margin.
- **Auto-termination blocked.** A complete manifest the verifier rejects (for example a non-finite GPU value, or another GPU name string) is never accepted. The watch deadline then deletes at +4,500 s, about $0.88, unless root runs teardown (RUNBOOK F8).
- **Launcher cannot resume.** The unchanged `launch.py` cannot resume once the bundle has been extracted if `BOOTSTRAP_STARTED.json` was never written (RUNBOOK F5).
- **Untested observation tool.** `observe_account.py` could not be run: the network was forbidden. Its calls are those of the attempt05 observer.
- **Provider and account.** The provider TTL has never been observed to fire. Community-host SSH has been unstable. The live balance may have changed since 18:43Z.
- **Future admissions.** This lease will be debited into every later admission.
- **A7 not done.** No independent verification or adversarial review has run.
