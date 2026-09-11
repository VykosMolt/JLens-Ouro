# Native-logit diagnostic v1: independent verification (A7, part 1)

**Verdict: PASS_WITH_FINDINGS.** Every A1–A6 claim was reproduced. Every recorded comparison was recomputed exactly with separate code, and ledger accounting is sound. No protected file changed. The findings are three low-severity runbook and test hazards and four informational notes. None blocks the paid run. Fixing F1 or F2 requires editing `RUNBOOK.md`, which means re-freezing.

Details and exact values: [diagnostic_native_logit_v1_verification.json](diagnostic_native_logit_v1_verification.json).

## Scope and method

- **Target:** `diagnostics/native_logit_v1`, with `FREEZE.json` SHA-256 `178f2f92…970ea`.
- **Interpreter and isolation:** `/home/moloch/ouro_project/venv/bin/python -B`, on CPU. Every run executed under `unshare -rn`, so none had network access.
- **Outputs:** everything went to the scratch directory `…/scratchpad/verify_diag`.
- **Test evidence:** the frozen tests write into `D/evidence`. To avoid that, they were re-run unchanged with `common.EVIDENCE` redirected in-process (see F3).

## Findings

| ID | Severity | Where | Finding | Fix |
|---|---|---|---|---|
| F1 | low | `RUNBOOK.md:189` (reached from :217, :222, :247) | `POD` is set only once SSH is bound (:100). In no-pod branches (F4 `create_rejected`, F4 with no pod listed, early F9 or T), `--pod-id $POD` expands to nothing. Reproduced offline: argparse exits 2, and no final observation is written. | Use `${POD:+--pod-id $POD}`, then re-freeze. |
| F2 | low | `RUNBOOK.md:82-84` | Invariant 7 requires recorded approval before create, but only step order enforces it. The create command does not test for `$RUN/USER_APPROVAL.json`, and `lease.py create` (827–902) takes no approval input. | Prefix the create line with `test -s $RUN/USER_APPROVAL.json &&`, then re-freeze. Otherwise root must keep the order manually. |
| F3 | low | `tests/common.py:48-50`; `build.py:163` | `save_evidence` overwrites the frozen `D/evidence` in place. Re-running any of A2–A6 breaks `build.py verify`, and A2 embeds a fresh lease name and timestamps. | Never re-run the tests in place; redirect `EVIDENCE`. If the tests are edited anyway, make the write exclusive. |
| F4 | info | `IMPLEMENTATION.md:124-138` | The file lists **ten** deviations, not eight. All ten match the code. | None. |
| F5 | info | `bundle/evaluation/validate_outputs.py:55-56` | `torch_equal` treats −0.0 and +0.0 as equal. An adversarial case records `torch_equal=true` with `differing_elements=1`. | Judge objective 3 (bit-exact) by `differing_elements == 0`, not by `torch_equal`. |
| F6 | info | `bundle/evaluation/validate_outputs.py:117-118` | `token_ids` are not bound to the development items. A same-length substitution, with a consistently regenerated record, is accepted. | In the post-run CPU reanalysis, retokenize the three items (S = 20, 15, 13). |
| F7 | info | `RUNBOOK.md:27, 190-192, 219` | Step 10's compare allows additions only under the new lease directory. Any `RUN_LOG.md` or review write between steps 0 and 10 triggers a false F11. This was observed with the concurrent review files. | Take the step-0 inventory after all A7 reports are written, and defer log writes until after step 10. |

## Checks performed

1. **Freeze (A1).**
   - `build.py verify` reported `verified` both before and after all runs.
   - Independent hashing:
     - the 45 archive members are regular files with safe paths;
     - the archive, the directory and `FREEZE.json` agree;
     - the outside-bundle files, the evidence and the parent freeze all match;
     - `D` contains no unbound files.
   - The 41 shared files are byte-identical to the precision_v1 bundle and to its freeze.
   - The six controller files equal the attempt05 `controller_sources` pins, and `test_lifecycle.py` is identical too.
   - The run spec binds all 43 non-frozen files, `CONTRACT.md`, the parent freeze and the four inherited fields.

2. **A2 admission.** Admitted, with the following values:

   | Quantity | Value |
   |---|---|
   | Bound | $1.1507876712328766 for 5,100 s |
   | Prior | $21.64556796376644 |
   | Remaining | $3.354432036233561 |
   | Minimum balance | $4.028675539145375 |

   - The re-run's evidence differs from the frozen A2 only in lease name, deadlines and timestamps.
   - The RUNBOOK step-3 plan, run as written, gives identical accounting. A balance of 4.0286 is rejected.
   - The ledger was checked before and after every controller run and was identical: SHA-256 of all 120 files, plus type, mode, size and mtime of all 152 entries.

3. **A3 verifier tests.**
   - The conforming set is accepted and all 25 mutations are rejected with their expected messages. The evidence is identical to the frozen A3.
   - I added 15 adversarial cases of my own. All behaved as predicted:
     - Rejected: reordered records, a tampered signed-zero count, reordered items, an extra run key, a non-finite composite, a geometry mismatch, a missing GPU record (`KeyError`, still never accepted), an inconsistent token count, an extra top-level key, and stale M1 rows.
     - Accepted:
       - a consistent signed-zero record (F5);
       - same-length token-ID substitution (F6);
       - a symlinked payload; the verifier alone accepts it, but the controller's `verify_payload` rejects linked paths first;
       - no exit-step match (deviation 4's fallback);
       - `torch_equal` recorded as `1`, which is harmless.

4. **A4 CPU pipeline** (pipeline validation only).
   - Passed on the real snapshot, with payload hashes identical to the frozen evidence (36,921,970 bytes).
   - Exit step 3 was uniquely matched for every item and run; 34 records are unequal on CPU.

5. **A5 lifecycle.** Passed with an isolated ledger:
   - accepted, then deleted with class `accepted_artifacts`;
   - three absence confirmations, then status `terminated`;
   - the real ledger was unchanged.

6. **A6 audit.** Passed, with evidence identical to the frozen A6.

7. **Independent recomputation.**
   - **Method:** separate code on raw bytes, with a hand-written bf16 decoder that was self-checked against torch's exact cast. The bundle's verifier was never imported.
   - **Records:**
     - all 255 names are in identical order;
     - every record equals my recomputation;
     - exit steps agree, and every token's exit step is 3 on CPU;
     - all shapes, dtypes and finiteness values match the contract;
     - all 23 M1–M4 coverage checks pass.
   - **M3 construction:** every shaped input was rebuilt independently and recomputed from the snapshot's `model.norm.weight` and `lm_head.weight`. All 23 families × 2 repeats × 3 items reproduce the saved rows bit-for-bit.
   - **Layout:** the captured inputs are contiguous with storage offset 0, so deviation 5 is benign.
   - **GPU payload estimate:** 46,360,421 bytes plus about 16 KB, under the 50 MB limit.
   - **Tooling note:** the first pass of my script had a decoder bug. Its self-check caught it, and every figure above comes from the corrected run.

8. **Ledger accounting.**
   - **Enumeration:** `prior_debit` enumerates `LEDGER.glob('*/LEASE.json')` (`lease.py:238`), where `LEDGER` is this round's `cloud_leases`.
   - **Simulation** on a patched scratch ledger:
     - `cloud_leases/diagnostic_native_logit_v1` is accepted as a lease root;
     - while the lease is not terminated, it blocks every other admission and create;
     - once terminated, it is counted exactly once, and later plans see six debits;
     - the attempt_01–05 records are unchanged;
     - a name collision and a foreign prior-debit record are both rejected.
   - **Collisions:** none. Controller writes outside the new root are limited to append-mode opens of the existing `.lease_creation.lock`.

9. **Nothing else changed.**
   - All 11 named files match every recorded value in `ATTEMPT05_SPENDING_AND_SHUTDOWN.json`, both FREEZE receipts, the lease records and the diagnostic freeze and run spec. Their hashes were identical before and after my runs.
   - A full-round SHA-256 inventory taken before any execution differs afterwards only by two added files: `reviews/diagnostic_native_logit_v1_review.{json,md}`. Another process wrote them during this verification (23:31–23:32 +0200); I did not read them.
   - No file changed or was removed, and `D` has no `__pycache__`.

10. **Runbook.**
    - **Executed offline:**
      - step 0, with `RUN` in scratch: `verify` passed and the inventory has 5,146 entries;
      - step 3;
      - step 10's compare logic;
      - step 10's argument handling (F1).
    - **Gating:**
      - steps 0–1 are local or read-only;
      - step 2 is the approval checkpoint;
      - create (the only billing mutation), steps 5–10, and branches P and T all come after it;
      - F1–F3 stop before creation.
    - **Consistency:** deadlines, emergencies, the balance floor, worker phases, the acceptance path and branches F5–F10 all match the code.
    - **Setup budget:** the long historical setups were bank uploads. Without banks, bootstrap-to-setup took about 4–10 min, so 2,400 s is adequate.

## Checks not performed

- **Live RUNBOOK steps:** steps 1, 2 and 4–10, and branches P and T. They need the network, provider, SSH or payment, all forbidden here.
- **Live `observe_account.py` observation:** only its argument handling was exercised.
- **GPU execution:** the CUDA-only M0 runtime load and all GPU bit behaviour are unverified.
- **On-pod work:** bootstrap, wheel install, model download and SSH transfer timing.
- **The controller's full self-test suite:** not requested; only `watch_fixture` ran, via A5. The controller is byte-identical to the attempt05 pins.
- **The concurrently written review files:** not read, to keep this verification independent.
