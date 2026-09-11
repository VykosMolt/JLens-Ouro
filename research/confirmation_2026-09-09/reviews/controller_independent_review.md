# Independent controller review

Status: **passed** for the exact controller sources and offline evidence below. **New confirmation results exposed: false.** No GPU, SSH or provider operation was performed during this review.

The receiver and deletion gates implement the frozen protocol. Setup completion is durable for the full lease/pod/run binding. Missing, false or failed later status reads cannot reopen setup or bypass early STOP, local acceptance or emergency decisions. Closed stages are preserved while later work can continue. Terminal computation and local retrieval acceptance remain separate.

Normal deletion requires the receiver's exact current final receipt, manifest, output contract and pinned semantic verifier, explicitly successful loadability/numerical checks, and fresh hashes of every accepted payload. Stage receipts, stopped/failed manifests and worker-supplied status cannot authorize normal deletion. Corrupt published generations are preserved in quarantine and rebuilt without transfers into published payloads; the stale receipt stays invalid until a new fully checked receipt is bound. The final deletion call rechecks the full binding and receipt pointer.

The prior-debit reader verifies historical records and rejects reducing the carried-forward debit below their summed or cumulative bounds. Completed new leases are added once; duplicate or unreconciled leases block another creation. Run, account, contract, verifier, source, accounting and bound pod/machine identities are immutable under state transactions. Deadlines cannot be extended, rates cannot decrease, and prior debit, incremental ceiling, full $25 cap and independent provider TTL remain enforced. A budget emergency records preservation/deletion evidence before mutation and bypasses the collector lock; normal deletion takes that lock nonblockingly. Every deletion retains account and exact owned-pod checks plus three bound absence confirmations.

The semantic verifier is rehashed before creation and before/after validation. Its pinned source bytes are compiled directly, so cached bytecode cannot replace those bytes. The actual scientific verifier and output contract remain part of the separately frozen experiment; this review establishes the controller's enforcement of them.

## Independent execution

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/moloch/ouro_project/venv/bin/python research/confirmation_2026-09-09/controller/test_lifecycle.py
```

**38 cases passed** in 2.586 seconds. Independent proof: `/tmp/confirmation-controller-proof-3ab9qhyr/PROOF.json`. SHA256: `6e96ff816147a64b313027c2da00568fbeb7adbb00a6d98d778c401f3dd4a5ec`. All seven proof source records were rechecked against the files after this run. The implementer's durable equivalent suite is [controller/proofs/lifecycle_v1/PROOF.json](../controller/proofs/lifecycle_v1/PROOF.json).

The suite exercises actual watcher/state/receiver paths with mocked provider/SSH boundaries: persistent setup and repeated status errors; concurrent updates and stale observations; early STOP and emergency deletion during collection; account/owned-resource/absence guards; exact stale receipt binding and contracted path membership; stage reuse; interrupted/truncated or corrupt-prefix resume; double collectors; quarantine repair; six publication/receipt crash points; source mutation and pruning during transfer; hash-consistent unloadable, wrong-shape, wrong-dtype, nonfinite and numerically wrong tensors; prior-debit and uncertain-create barriers; immutable IDs/deadlines and changed verifier before creation; explicit teardown and rejected worker-forged acknowledgement; failed-terminal preservation; bounded process cleanup; metadata atime handling; and the actual bootstrap Python snippet executed in a temporary filesystem.

Its real CPU smoke saves a tiny checkpoint/final bank with the existing primitives, interrupts a copy, resumes the same partial staging, loads and validates the exact FP32-to-FP16 conversion, publishes and consumes a receiver receipt, and follows accepted-result deletion through three mocked absence confirmations despite a worker-status timeout.

Separate reviewer-authored temporary probes independently reproduced the initial receiver failures and then verified their repairs: stale receipt rejection, corrupt manifest recovery, and receiptless damaged-payload recovery with quarantine retention. A separate tiny CPU save/copy/load smoke rejected a fully hash-consistent wrong final mean specifically at the FP16 N100 conversion check. Separate actual-watch probes verified setup-latched timeouts cause neither STOP nor deletion, early STOP survives a timeout, accepted outputs permit normal deletion despite timeout, and pending transfers permit only a documented emergency deletion at the deadline. Exact mocked deletion IDs and three absence checks passed.

## Review findings resolved

The final source resolves seven findings from this review: unrecoverable corrupt accepted generations; the PENDING-unlink crash gap; metadata reads rejecting access-time changes; status exceptions bypassing STOP/normal teardown; understated historical debit declarations; normal deletion blocking on a collector across the emergency deadline; and missing last-moment verifier-file checks before creation. The later immutable-state, direct-source compilation and final mutation-binding safeguards were also inspected. The corrected confirmation bootstrap prefix passed the executable fixture.

No unresolved implementation blocker remains in this controller scope. This is local lifecycle evidence; it does not establish live provider reliability or large-artifact transfer throughput. The scientific job's separately frozen contract, verifier, runtime and preservation budget remain necessary inputs to deployment.

## Exact reviewed source records

| File | Bytes | SHA256 |
|---|---:|---|
| `artifact_handoff.py` | 29606 | `879a9b6bd9c282dc5bd46f2fb86e20a7d63511940e86f0c7e5c3911d97d58154` |
| `lease.py` | 56750 | `c59d303693253017e56f44daebd261fbf3fe38b765602c9411902eaaa9248366` |
| `pod_entry.sh` | 1223 | `49770f538686d3798547742b57838fca2ba9246ccda054b8691da6ff28bf5cad` |
| `run_refits.py` | 51999 | `a8841cd456434b5582e578f115cbeff62829af89e1809a43a189a5dba0ff0fd4` |
| `runpod_api.py` | 9601 | `6124d239c0e1ec6b7c8e86e3f845716b17bd67943423f88416a9cae771051849` |
| `test_lifecycle.py` | 35975 | `7dc4d0d328ccb4a5661d4362179fbfef2d7d2f81ab6f14197d0f47ef444fa7de` |
| `transport.py` | 2638 | `acec7a803383119706ee852d8c0b21c4380df0b91b16aa4c198462c6d7bd199a` |

Also reviewed: [PROTOCOL.md](../controller/PROTOCOL.md), [design_review.md](../controller/design_review.md), the actual prior-debit snapshot and the historical setup/termination mechanism. This reviewer did not edit controller sources or historical evidence.
