# Native-logit diagnostic v1: adversarial review (A7, part 2)

**Verdict: PASS_WITH_FINDINGS.** There are no critical or major findings. Repair the minor runbook findings, or explicitly accept them, before step 1.

Reviewed freeze: `FREEZE.json` sha256 `178f2f924d1421102184199842a4b06f20466ee1bff96ddd2be2e60e932970ea`. At review time every bound file matched it, and there were no unrecorded files.

The review was read-only, with no network, provider, SSH or GPU use. Structured record: [diagnostic_native_logit_v1_review.json](diagnostic_native_logit_v1_review.json).

## 1. Scientific validity

As implemented, M1–M4 can separate final-norm from lm_head shape dependence and can decide bit-exact reproducibility. No recorded conclusion uses a tolerance.

- **Layout.** The native norm input (the residual sum) and the native lm_head input (gather, then squeeze) are fresh contiguous `[1,S,2048]` tensors at offset 0. M3 inputs are also fresh contiguous device copies.
  - RMSNorm upcasts to a fresh FP32 tensor before its only reduction, so input alignment cannot shift the reduction grouping.
  - lm_head reaches `mm` for every shape, with the same transposed weight view.
- **Allocator.** The M2 hooks keep `.detach()` aliases of tensors that are already retained, so they allocate nothing during the forward. Allocator blocks are page-aligned and 512-byte rounded. This is inference, not a GPU measurement.
- **Transfers.** All transfers are bitwise exact. `unembed_fp32_virtual191` reproduces `readouts.py:79` exactly.
- **Exit gather.** The exit step is identified bitwise, and the lm_head family uses the captured gathered rows.
- **Placement and repeats.** First and last placement are covered at 148 and 160 rows. M3 runs the whole sequence twice; M4 repeats M2 after M3. Both are within one process.
- **Comparison record.** The worker's record is recomputed exactly across machines: double arithmetic, stable sort, `fsum`, integer counts and JSON float repr.
- **Remaining ambiguities.** Signed zero (F-03) and the M2-versus-M1 link (F-06), both resolved by analysis rules.

## 2. Deviations

| # | Decision |
|---|---|
| 1 `download_model.py` | Accept: `bootstrap.py:61` needs it. |
| 2 `test_lifecycle.py` | Accept: precision_v1 file set; not pinned; not run on the pod. |
| 3 composite `[160]` first and last | Accept: superset of the contract. |
| 4 fallback to step 3 | Accept, with a rule: if `M2_matching_steps` is not exactly `[exit_step]`, treat the exit step as unidentified and do not interpret the norm or composite families for that item. |
| 5 fresh copies; target rows only | Accept. |
| 6 no physical hooks in M2 | Accept, with rule F-06. |
| 7 GPU-name check | Accept: false rejection is very unlikely, and a worker-side check would bill the same. |
| 8 worker edits | Accept. Wording nit: the `readouts` import moved after model load, and two comments were removed. |
| 9 no random-weight fallback | Accept: memory sufficed. |
| 10 lease root name | Accept: both the primary and the Huginn debit chains glob `*/LEASE.json` with no name or run_id constraint. |

## 3. Spending and safety: confirmed

- **Create gates.** The order is observation, approval, plan, notice, create. Create re-checks zero pods, zero hourly spend and the balance, and checks the notice twice.
- **Admission.** 5,100 s gives $1.1507876712, within min($3.3544, $1.50). The minimum balance is $4.0286755391.
- **Deadlines.** The runbook table equals `make_plan`.
- **Worst-case automatic spend.** About $0.88 at the watch deadline, and about $1.00 at the provider TTL.
- **Upload path.** Only the bundle, worker config and an empty `BANKS_READY` go to the pod. There is no `parallel_upload.py` path.
- **Interpreter.** The watcher and verifier inherit `$PY` through systemd `sys.executable`.
- **Payload.** About 46.4 MB, and deterministic: 30,941,184 tensor bytes in `tensors.pt` plus 15.34 MB in `native.pt`. Retrieval within 600 s needs about 77 kB/s.
- **F8 likelihood.** Every verifier check is deterministic. The realistic F8 driver is transfer instability; the GPU-name check adds negligible risk.

## 4. Findings

| ID | Sev. | Location | Problem | Required fix |
|---|---|---|---|---|
| F-01 | minor | RUNBOOK.md:221-225; lease.py:812-824, 865-869, 880, 649-659 | F4 claims the watcher reconciles every non-supply error. Pre-`try` errors (notice expiry, stopped watcher, bundle mismatch, readiness or systemd failure) never set halt, so the lease only terminates ($0) at +4,500 s and blocks the ledger meanwhile. If the watcher never started, the lease stays `pending` indefinitely, and T is impossible without a `pod_id`. An uncertain create with no pod finalizes at +5,100 s with about a $1.00 debit. | Split F4 by `mutation_phase`: not_started (watcher active: wait; inactive: stop and report), in_flight/uncertain (expected waits and debit; T if a pod is listed), rejected. |
| F-02 | minor | RUNBOOK.md:229-233 | P omits `/workspace/jlens/results`, so a worker killed before publishing loses its partial payloads. | Add `results` to P. |
| F-03 | minor | validate_outputs.py:55-56 | `torch.equal(+0,-0)` is True while the bit count is 1 (verified locally), so the two "decisive" fields can disagree. | Analysis rule: bit-exact means `differing_elements == 0`; `torch_equal` answers the attempt05 gate question; report any disagreement. |
| F-04 | minor | tests/common.py:48-50 and every test's `save_evidence` | Tests overwrite frozen `evidence/*.json`. Re-running them, as A7 verification may, breaks `build.py verify` at step 0: fail-safe, but it invites an unreviewed re-freeze. | After verification, require `build.py verify` against sha `178f2f92…`. Future tests should write exclusively. |
| F-05 | minor | RUNBOOK.md:9-20 and later blocks | Steps depend on shell variables and `field()`, but root's shells don't persist state, so an incident branch could break. | Prepend the setup block (and derive HOST, PORT and POD) in every step. |
| F-06 | minor | worker.py:22-45; validate_outputs.py:81-83 | The M3 references are M2 captures. M1 has physical hooks that M2 lacks, so an M2≠M1 difference is unattributable and localization would not transfer to attempt05. | Rule: link to attempt05 only if the M2~M1 logits and virtual191 records are bit-exact for all items. |
| F-07 | minor | RUNBOOK.md:48-55 | The approval omits the remaining authorization after the lease and the carry-forward debit. | Add: about $2.47 (watch deadline), $2.35 (TTL) or $1.85 (cap) remaining, debited in every later admission. |
| F-08 | minor | RUNBOOK.md:25-27, 189-199 | Any root edit under R during the run (RUN_LOG, reviews) triggers F11. The inventory excludes D, whose freeze is never re-checked after the run. | Forbid R writes outside D/run and the lease directory between steps 0 and 10; add `build.py verify` to step 10. |
| F-09 | minor | worker.py:30, 44 | `lm_head_output` is `output.logits`: 9.44 MB of duplicate bytes, and a tautological comparison. | Optional; drop it only if re-freezing for another reason, otherwise document the record as a capture check. |
| F-10 | minor | RUNBOOK.md:97-122 | ssh is bound before sshd is up, so an immediate launch likely fails, and reruns are unpaced. | Poll `$SSH cat /workspace/jlens/provider_lease_name` until it prints the name; pace reruns. |
| F-11 | nit | RUNBOOK.md:189 | `$POD` is unset on early branches. | Use `pod_id` from `field`; omit `--pod-id` when null. |
| F-12 | nit | RUNBOOK.md:218, 241-244 | The T document uses `'x'`, but F10 says "rerun T". | Rerun only `lease.py terminate`. |
| F-13 | nit | RUNBOOK.md:152, 216 | `sync` prints `[]` when its lock is busy. | Judge F8 by `retrieval_in_progress` and `retrieval_last_error`. |
| F-14 | nit | validate_outputs.py:83-90 | No record links `virtual191` to `norm_input[3,0,-1]`. | CPU reanalysis: compare them bitwise and report the exit step per item. |
| F-15 | nit | validate_outputs.py:151-152; worker.py:77 | A tautological source check, and a redundant `copy=True`. | None required. |
| F-16 | nit | RUNBOOK.md:35-44; observe_account.py:50-53 | The step 1 observation may be stale by create. `--out` is opened after the network reads, and non-APIError exceptions write no file. | Repeat step 1 if more than about 10 minutes pass; use unused output names. |
| F-17 | nit | FREEZE.json; RUNBOOK.md:59-61; worker.py:169 | `lease_root` and `job_cap` are not checked against create; the approval doesn't bind the A7 reports; `config_record` is not cross-checked. | Put the A7 report hashes in USER_APPROVAL.json; check `config_record` against `BOOTSTRAP_STARTED.json` post-run. |

## 5. Integrity

- **Frozen and confirmation files.** Nothing in the design reaches confirmation items, banks, frozen precision_v1 files, `prior_debit.json` or the attempt01–05 directories.
- **Write locations.** All run writes go to `D/run`, the new lease directory, and the pod.
- **Provenance binding.** The chain is FREEZE → archive record → LEASE `bundle_record`, then run spec and contract hashes → binding → manifest → provenance, and the verifier checks it.
- **FREEZE coverage.** FREEZE covers the bundle, archive, contract, runbook, build, observer, run config, tests and evidence. It does not cover IMPLEMENTATION.md or the A7 reports (F-17).

## 6. Checks not performed

- No GPU run. Claims about CUDA reduction kernels, cuBLAS and cuBLASLt heuristics and allocator alignment are inference.
- No network, provider, SSH or systemd use. `observe_account.py`, `create`, `watch`, `launch`, `sync` and `terminate` were not executed.
- A2–A6 and `build.py` were not re-run: executable verification belongs to the separate verifier, and re-running would overwrite evidence (F-04). Frozen hashes were instead checked read-only.
- Not checked: image, wheel and model availability, RTX 5090 stock, live balance, and provider TTL behaviour.
- `test_lifecycle.watch_fixture` and the unchanged legacy modules were not reviewed in full.
