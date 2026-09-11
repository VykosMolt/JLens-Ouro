# Separate Huginn verification phase controller

This controller is prepared for independent review. It has not admitted a real Huginn debit, created a worker, or spent funds. The primary controller, primary debit, scientific bundle, and historical records remain unchanged.

## Frozen resource contract

The original combined ceiling remains $25. The original prior bound is pinned at $13.851844033145376, and the original primary job ceiling remains $4 in its unchanged source and debit. This separate phase permits at most $6 across the entire Huginn phase, including all retries after carrying every finalized primary lease bound exactly once. The original account balance floor is retained exactly; create also preserves the stronger balance floor inherited from the actual starting balance and chosen job ceiling.

The proposed resource allocation is 3,600 seconds setup; 20,880 seconds work (3,600 native gate, 16,380 N=100 fit, 900 historical readouts); 4,680 seconds preservation; and 600 seconds termination. Transfer timeout may not exceed preservation. At the unchanged full price ceiling of $0.69/hour plus 120 GB storage at $0.10/GB-month divided by 730 hours, 29,760 seconds plus $0.15 latency reserve costs **$5.989890410958905**. This is a planning bound, not an invoice. The scientific wrapper separately decides whether remaining work time can admit the full N=100 calculation.

## Phase admission

`resources/prepare_phase_debit.py` requires the actual finalized primary lease root and a fresh account/pod observation. It writes an exclusive new `resources/prior_debit.json` only if:

- The pinned original debit retains its complete historical lease records, $13.851844033145376 bound, $4 primary ceiling and original balance floor.
- Every `PRIMARY_LEDGER/*/LEASE.json` is included exactly once, terminated, account-bound and absence-reconciled. Duplicate names/pods, nonterminal states, altered prior-debit references, omitted records, understated charges and inconsistent previous-lease accounting fail.
- One exact primary complete final receipt freshly passes the original primary `artifact_handoff.accepted_for_current_run(..., rehash=True)`. The original primary `absence_can_finish` predicate is also checked on every primary lease.
- The supplied account snapshot is no more than five minutes old, matches the original account, lists no pods and reports no active spend. Create repeats live read/write account identity, pod, active-spend and balance checks.

Primary source bytes are checked against the frozen six-source baseline before execution. An isolated Python process directly compiles those verified bytes into modules with their true primary file locations. It cannot accidentally invoke the Huginn handoff module or substitute an old bytecode cache. No primary verifier is imported through an H-relative location.

The H controller repeats these prerequisite checks itself; the helper's success is not authorization to create. H retries use their separate canonical `cloud_leases` ledger and the same immutable debit path/hash. Every reconciled H lease charge is added exactly once; an unreconciled H lease, duplicate state, changed debit or attempted replacement snapshot fails.

## Full rehash and immediate creation check

The full primary payload inventory is captured **before** acceptance rehash and must be identical **after** it. The inventory rejects links, nonregular files, missing/extra files or directories and records device, inode, size, mtime_ns and ctime_ns. The proof retains original per-file hashes, prerequisite file hashes, the selected receipt, and the full-rehash UTC time.

The full check finishes before the H billing/deadline origin is chosen and before the watcher is armed. Immediately before the create mutation, `begin_create` verifies all exact source, receipt, manifest, validation, original/historical/primary lease and snapshot hashes, re-enumerates the primary ledger, and compares the same payload inventory. It also rechecks all H retry debit records. Each prior H state must satisfy the unchanged absence predicate, elapsed-time/rate charge lower bound, combined/prior consistency, and a complete nonoverlapping predecessor chain. Each H state is parsed from the exact byte string that is hashed for its returned record, with descriptor/path stability guards, final record rechecks and a repeated complete ledger inventory. Every prior H charge reduces the remaining aggregate $6 phase allowance; a retry cannot request a fresh $6. Any change fails before the mutation intent transitions. A successful check records `primary_admission_rechecked_utc`; the original full-rehash time remains in the immutable admission proof. A proof older than five minutes fails.

Both primary and H creation locks are held around H creation, preventing a concurrent primary create from changing the prerequisite population. Acquiring the existing primary controller's lock does not alter its lease/debit records. The original 30-second watcher readiness check is retained unchanged; full payload hashing is not repeated inside that short window.

## Focused source changes

All admission runtime logic lives in H `lease.py`. The existing six-file `SOURCE_NAMES`, source binding and monitor-copy closure therefore include every executable admission function. H `artifact_handoff.py` also implements the bounded receiver below. The other four runtime files (`transport.py`, `run_refits.py`, `runpod_api.py`, `pod_entry.sh`) are byte-identical copies of primary. No new runtime import falls outside the existing six-source closure.

Only five existing lease functions change: `prior_debit` enforces phase prerequisites and retry accounting; `make_plan` uses the H ceiling and records the freshness proof after the full check; `begin_create` applies the immediate prerequisite check; `create` coordinates the primary creation lock and repeats live funding/resource checks; `change` makes the admission proof immutable. The canonical H ledger is separate. Provider identity, prices, normal acceptance, emergency termination, STOP behavior, the bounded transport cleanup utility, fixed lease deadlines and absence reconciliation functions remain unchanged.

The standalone debit preparation helper and resource declaration are separately source-pinned. They are not silently omitted executable dependencies: the controller repeats admission through its own bound `lease.py` implementation.

## Bounded parallel preservation

H `SSHTransport.transfer` retrieves at most **eight ranges concurrently**, each no larger than **64 MiB**, using the existing bounded subprocess transport and exact bound SSH endpoint/options. Remote readers accept only canonical relative result paths, require the exact `provider_lease_name`, enforce the fixed receiver deadline, and check regular-file size plus opened-descriptor/path identity before and after each exact-length range read. They perform no remote writes.

The receiver's global deadline is the earlier of its original transfer allowance and the lease's `watch_deadline_utc`, leaving the original 600-second controller termination reserve. Metadata and original before/after full-source snapshots retain their verification semantics and have their subprocess timeouts clamped to this same deadline. No work, preservation or provider deadline is extended.

**Worker STOP and the work deadline stop computation, while preservation continues.** The range reader therefore permits worker STOP. Local SIGINT/SIGTERM, an explicit durable retrieval halt, a durable lease halt or terminating/terminated state, changed binding/SSH/deadlines, a range error, or the retrieval deadline cancels collection and kills the receiver's own started process groups. Synchronized registration also kills a child first registered after cancellation; the original transport cleanup proves quiescence, including descendants that ignore SIGTERM. Emergency teardown remains independent of the collector lock.

Completed ranges are stored under the staging generation's `partials/chunks/<whole-file-sha>/<offset>.part`, with immutable whole-file/offset/length/chunk-hash records. Cached bytes must rehash against those records before reuse. A crash between publishing metadata and publishing a chunk can resume by refetching exactly that range. An interruption retains completed verified chunks and publishes no incomplete file.

Assembly uses the declared offset order, rechecks chunk descriptor/path identity and hashes, fsyncs a staging temporary, verifies the full manifest byte count and SHA, rereads the same open temporary, then atomically publishes and fsyncs the destination directory. Completed chunk data is removed only after whole-file verification. Existing accepted-stage reuse, complete source snapshots, full local payload validation, the outside semantic verifier, acceptance receipts and emergency deletion rules remain unchanged. The temporary cache and assembly occupy receiver disk only; no new worker chunk copies are created. Eight range responses contain at most 512 MiB of payload; subprocess buffering can transiently duplicate these bytes, in addition to Python overhead.

## Verification

The copied lifecycle suite retains the original real tiny CPU save/load/conversion/receipt fixtures and fake provider/SSH boundaries. Its phase fixtures construct a separately source-bound primary run through the original primary receiver in an isolated interpreter, including its actual final receipt and accepted payload. No test touches real primary records or calls a provider.

The added tests cover omitted/duplicated historical and primary debit records; a newly added or duplicate actual primary ledger state; nonterminal and unaccepted primary runs; modified receipt despite a spoofed H handoff; same-size payload corruption under both full rehash and immediate inventory checking; mutation between full rehash and its final inventory; same-byte inode replacement; changed primary source before import; account mismatch/pods/spend and balance-floor changes; the exact planned $5.989890 cost, H $6 and total $25 bounds; controller rejection before deployment intent; immutable H retry accounting; exact-byte ledger replacement/membership races; actual eight-stream local subprocess retrieval; STOP-preservation behavior; interrupted chunk reuse; wrong owner, changed bytes, short ranges, corrupt cache and descriptor-parent swaps; and deadline/halt/identity/late-registration cancellation including SIGTERM-ignoring descendants. The original lifecycle cases continue to check normal acceptance and hard emergency paths.

Run with:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/moloch/ouro_project/venv/bin/python research/confirmation_2026-09-09/huginn_verification/controller/test_lifecycle.py --output-dir /tmp/huginn-controller-proof-review
```

The current proof and exact source records are named in `SOURCE_RECORDS.json`; the earlier `proofs/phase_v1/PROOF.json` remains an immutable record of the initial revision. Source and unchanged-function comparisons are recorded in `SOURCE_RECORDS.json`. Independent parent review and later scientific-wrapper/bundle/receiver integration remain separate gates. No admitted debit has been written as part of this preparation.
