# Prospective controller and artifact-handoff design review

Status: recommendation for root to freeze, 9 September 2026. This review changes only this document. No implementation, tests, deserialization, provider call, SSH, deployment, or external safeguard change was performed. All September 7–9 original evidence and controllers remain read-only.

## Finding and scope

The preserved setup-latch candidate is a valid starting point, but it is not the full repair requested for this round. Its seven passing synthetic cases cover setup identity, persistence, restart, missing/false status, and the historical receipt branch. They do not prove manifest completeness, independent tensor verification, or a safe artifact-transfer lifecycle.

The incident selected the setup-timeout branch despite a preserved `complete/setup_complete=true` observation. Retrieval was still incomplete. The historical receipt was absent. The incident does not establish the precise SSH error/deletion ordering. Evidence: [incident review](../../refit_round_2026-09-07/monitoring/attempt_06/termination_incident/attempt06_termination_incident_review.json).

The frozen base controller SHA-256 is `a7f5d92cf62f4e1d40773c6edb16c07811bb4568f2dc23018d91d9852b340992`; the candidate is `5493d28c8a74b6740d4a3edbbccc5229d00446357d3c192d51cb46bbbfaa1de3`. These source hashes were checked in this review. The [candidate and proof](../../refit_round_2026-09-07/monitoring/attempt_06/controller_correction/lease_setup_latch_README.md) remain prospective evidence.

Concrete remaining weaknesses in the old source:

| Surface | Current behavior | Required change in a new controller only |
| --- | --- | --- |
| `lease.py:240–251, 716` | Setup depends on one current SSH status observation. | Incorporate the accepted durable latch, bound to the current lease and pod. |
| `lease.py:281–323` | Copies all files currently present in a stopped tree. Its receipt explicitly leaves scientific completeness separate. | Verify an independently specified output contract and exact run/manifest, then locally load and validate the preserved artifacts. |
| `lease.py:335–345` | Any existing `RETRIEVAL_VERIFIED.json` suppresses collection. | Only a validated receipt for the current run, pod, contract, and manifest can suppress collection. Preserve rejected receipts as evidence. |
| `lease.py:720–723` | A receipt with matching lease name and `status=passed` allows deletion. | Require the complete acceptance predicate below; a worker status or transfer-only receipt never authorizes normal deletion. |
| `lease.py:730–735` | Three generic supervision exceptions can delete a completed worker. | Missing/transient worker status and collector/validator errors preserve the worker and retry; a hard budget emergency remains independently enforceable. |
| `lease.py:156–186, 591–610` | Fixed reserves; historical spend is summed only from the one configured ledger. | Carry the reconciled prior-round debit into the new ledger explicitly; use a measured early preservation reserve without extending the hard deadline. |
| `cloud_worker.py:305–309` | Final status binds evaluation completion records but no full artifact manifest. | Publish immutable per-stage artifact manifests promptly and one terminal manifest with complete expected-output closure. |

## Minimum state mechanism

Keep provider lifecycle `status` separate from scientific and retrieval state. Do not overload the existing `running/terminating/terminated` field. Create a new schema before any new lease and bind every scientific/receipt record to:

`account_id`, `lease_name`, `pod_id`, fresh `run_id`, `run_spec_sha256`, `output_contract_sha256`, and pinned producer/validator source records.

The run ID and contracts are fixed in the local intent before work. Pod identity is filled only through the existing owned-provider observation path. A status observed with one identity cannot be committed after that identity changes.

Use the existing `locked()` / `change(root, action)` transaction for all state transitions: reread the latest state under `.state.lock`, check the observation's identity, mutate only owned fields, then atomically write. This preserves concurrent halt/rate updates and the historical maximum rate floor.

| Durable field | Who may establish it | Meaning and transition |
| --- | --- | --- |
| `setup_completion` | Local watcher after an identity-checked `setup_complete is True` observation | Monotonic for the exact lease/pod/run. A restart, `None`, timeout, or later `False` does not clear it. Apply the setup deadline only while it is absent. |
| `computation_finished` | Local watcher after validating the terminal producer marker and its manifest binding | Records `outcome=complete/stopped/failed`, terminal marker record, final manifest digest, and local observed time. It establishes quiescence, not retrieval acceptance. |
| `retrieval_in_progress` | Local collector under a separate collection lock | Records manifest digest, stable staging generation, attempt ID, progress/error timestamps, and completed file records. It persists across process failure and is resumable; it is not a reason to delete. |
| `retrieval_accepted` | Local verifier outside the paid worker | Records the exact acceptance-receipt record after all checks succeed. It is immutable for this final manifest. It cannot be set by worker JSON or a collector spawn. |

Treat a new manifest or run identity as a new handoff generation, never as permission to reuse an old receipt. Once the final manifest is bound, a conflicting final marker is an error requiring reconciliation; it must not silently replace the deletion target. A durable terminal binding allows retries when later status reads are missing. Retries still verify the immutable manifest and source ownership, rather than requiring every retry to reread a healthy ephemeral status file.

## Exact output contract and acceptance

Before paid work, root freezes an output contract naming the required artifact roles and validators for this run: estimator inputs and their provenance, requested ranks/caches, item/label/tokenization files, run identity, configuration, completion records, logs, and any newly produced checkpoint/final bank. Dynamic generation paths are resolved only through validated owner/pointer/seal chains. A worker-supplied directory listing alone is not a completeness specification.

The terminal manifest contains the exact safe relative paths, byte lengths, SHA-256 records, artifact roles, expected schema/geometry, run bindings, terminal outcome, and required external prerequisites. It excludes itself and the later local receipt to avoid a hash cycle. Mutable logs become final only after the writing subprocesses have stopped; earlier log copies are labeled snapshots. No path traversal, absolute path, symlink, device, socket, or unrecorded file is accepted in the published scientific generation.

The local acceptance process must establish all of these, in order:

1. Run, lease/pod, contracts, producer completion, and prerequisite chains match the local intent. Required roles are complete; no required checkpoint, bank, cache, rank file, or metadata member is omitted merely because it is missing remotely.
2. The source manifest and bindings are unchanged before and after transfer. Every required local path exists as a regular file and its actual byte length and SHA-256 match the manifest. The local published set exactly equals the manifest's payload set.
3. The copied files load outside the worker with the pinned readers. For existing fit banks, reuse `run_refits._read_checkpoint` and `_read_final`: verify pointer/seal/metadata/count/shape/dtype/finiteness and the exact FP16 mean of the bound FP32 checkpoint. Use `weights_only=True`, CPU mapping, and mmap where the existing readers support it. Reuse relevant sealed evaluation readers for ranks/caches (`allow_pickle=False` for NPZ); new confirmation outputs need the contract-specific semantic validator, not an unmodified N100 evaluator with the wrong item population.
4. Perform the frozen numerical checks appropriate to each role: finite and correctly shaped tensors, source and item/slot axes, rank/sentinel/control denominators and bound summaries; for a newly produced bank, exact checkpoint-to-bank conversion. A hash-only copy proof or producer claim does not satisfy this step. Never deserialize a partial or unverified artifact.
5. Publish an immutable local generation and an exclusive, fully flushed acceptance receipt only after all checks pass. The receipt binds the manifest, every accepted payload record, exact local generation, validator sources/configuration and validation-report records, run identity, and `status=passed` with explicit scope. A partial preservation receipt uses a different schema/status and cannot impersonate full acceptance.

The watcher uses one `accepted_for_current_run(state)` predicate both to skip collection and to authorize normal deletion. It verifies the local receipt record, exact current identity/contract/final-manifest match, passed required checks and referenced validation records. Immediately before normal deletion, under the local handoff/state coordination protocol, recheck the published payload records against the accepted manifest. Do not accept mere filename existence or timestamps as a substitute for hashes. Published generations are never transfer destinations and cooperating processes never modify them; local corruption or a missing member invalidates deletion eligibility and resumes repair. The normal deletion reason records the accepted receipt and manifest hashes.

For a stopped/failed computation that lacks planned outputs, preserve all committed artifacts and report the missing roles explicitly. It does not receive full scientific retrieval acceptance. Root can explicitly authorize teardown after reviewing that salvage record, or the documented hard budget emergency can terminate it. This avoids laundering an incomplete result into a complete receipt by shrinking the manifest after failure.

## Resumable transfer and prompt preservation

Use one stable local staging directory per `(run_id, manifest_sha256)`, with an exclusive collector lock and a manifest-bound `PENDING` record. Resume this directory after a collector restart; do not create a fresh timestamp destination for every retry. Keep transfer temporaries in a dedicated partial directory. Use rsync's existing partial-transfer mechanism with an exact NUL-delimited files list, bounded transport/process-group cleanup, and per-file atomic replacement; never use in-place writes into accepted files. A mismatched partial file is retransferred and then hashed, not accepted because its length matches.

After all staged files pass verification, publish the generation with a same-filesystem atomic rename into a unique final location, fsync the containing directory, then publish the exclusive acceptance receipt. If publication is interrupted, a restart may discover and reverify a complete generation, but it must not infer acceptance from a partially written receipt. Preserve failed staging generations and diagnostic records without merging them into an accepted tree.

Start preservation as each immutable stage commits, even while later computation is running. Poll committed owner/pointer/seal records and copy their closed generations and required metadata, then record stage-specific local acceptance. Begin with small run/contract/manifest files and completed evaluation outputs; schedule large bank/checkpoint transfers as soon as they exist. A final manifest can refer to already verified immutable local generations, which are rechecked before final acceptance. Do not recopy all banks only at the end of the paid lease.

The runner must not prune a committed generation that a pending manifest references. Prefer keeping the current completed checkpoint and final bank until their local preservation acknowledgement; old checkpoint pruning needs either a local acknowledgement or a contract-declared retention rule that cannot remove a pending transfer source. No worker-side acceptance credential or provider key is needed.

## Budget and deletion policy

The combined ceiling remains **$25**, including all previous attempts and this round. The historical incident reports `$13.851844033145376` as a conservative controller bound, not settled billing. Root is separately reconciling actual resources and accounting; this design does not replace that work or treat a remaining account balance as a fresh budget.

Changing `LEDGER` must not reset spending. Freeze a reconciled prior-round debit record, its provenance hash, and the accounted prior lease IDs in the new round. Every new create adds that nondecreasing base debit to all new lease bounds exactly once. Refuse an unreconciled prior lease, overlapping lease IDs, or missing debit snapshot. Do not credit lost artifacts or a failed computation back into the budget.

Preserve the fixed provider deletion deadline and the hard all-in price floor; no status or receipt extends them. Compute an earlier stop-admission deadline from a conservative bound on outstanding bytes and effective transfer rate, local hashing/load/numerical-check time, retry allowance, worker quiescence time, and the existing termination/polling/billing margins. Before each paid stage, require room for its worst-case compute plus preservation reserve. Completed-stage copying reduces the outstanding byte bound. A small smoke gives functional evidence; transfer reserve also needs conservative large-artifact sizing and observed throughput. If useful computation plus preservation cannot fit, do not launch paid science.

At the early deadline, request bounded computation STOP and start/continue preservation. It is not normal deletion permission. Once setup is latched, transient/missing worker status or retrieval/validation errors update diagnostics and retry; they cannot reopen setup or trigger the historical three-error deletion rule. Provider account/ownership mismatches block unsafe mutations and require reconciliation; they do not permit deleting an unrelated pod.

Normal deletion requires full current-run local acceptance, or an explicit user/root-authorized teardown recorded with its scope. A true spending/balance deadline emergency may delete the owned worker without acceptance to enforce the existing cap. Before that mutation, durably record `termination_class=budget_emergency`, triggering budget/deadline observations, manifest/receipt status, known preserved/missing roles, and best available transfer progress; do not spend extra time hashing the remote tree at the emergency boundary. Start best-effort salvage earlier. The provider deadline is still a last-resort independent safeguard, and local loss of connectivity cannot be made a guarantee of recovery.

The deletion routine should take a structured authorization class (`accepted_artifacts`, `explicit_teardown`, or `budget_emergency`) and validate its current-state preconditions while retaining the old account/owned-pod checks, mutation lock, and repeated absence confirmation. A generic reason string must not bypass the acceptance gate. Setup failure/provisioning timeout should request stop and preserve available evidence; before a pod exists the existing durable halt/create barrier still prevents a later deployment mutation. Any additional automatic destructive reason needs root's explicit mechanism decision before implementation.

## Proposed source changes after root freezes this mechanism

| New-round source/responsibility | Bounded implementation |
| --- | --- |
| `controller/lease.py` | New isolated copy derived from the pinned latch candidate; new ledger/debit binding and schema, durable terminal/retrieval fields, receipt predicate, lifecycle-aware error handling, early reserve/admission rule, structured termination authorization. Preserve owned-resource/create/absence protections. |
| `controller/artifact_handoff.py` | One local collection/verification command: frozen output contract, exact manifest, stable staging, bounded resume, local role validators, immutable publication, and acceptance predicate shared with the watcher. |
| New confirmation worker/orchestrator | Publish run-bound setup and immutable stage/terminal manifests, stop before preservation reserve, keep referenced completed generations, and start no next scientific stage unless admitted. Do not retrofit historical `cloud_worker.py`. |
| New-round pinned dependency/bundle manifest | Reuse read-only `run_refits` integrity/serialization helpers and relevant evaluator readers; pin their bytes and all new controller/handoff sources in the launched monitor. Correct any path constants in the new copy only. |
| Existing or one focused new controller fixture harness | Extend actual watcher/state/collector-path synthetic fixtures, plus the real local tiny-tensor smoke below. Avoid a broad duplicate test suite. |

Reusable source evidence: [lease state transactions](../../refit_round_2026-09-07/deployment/lease.py), [sealed generation readers and durability helpers](../../refit_round_2026-09-07/deployment/run_refits.py), and [bounded small handoff](../../refit_round_2026-09-07/monitoring/collect_huginn_handoff06.py). The last provides exact expected paths, before/after bindings, process cleanup, and exclusive receipt publication; its fixed attempt06 identities, 22-file scope, timestamp-only staging and hash-only acceptance must not be reused unchanged.

## Required executable evidence before paid science

Run the real proposed watcher/collector/state paths against temporary directories and fake provider/SSH boundaries. Assert the absence or exact identity of every deletion call; checking helper return values alone is insufficient.

| Fixture | Required observed result |
| --- | --- |
| Setup success, watcher restart after setup deadline, missing/false/timeout status | Durable setup survives; no setup deletion; no fabricated acceptance. Control with no validated setup still stops bounded setup. |
| Concurrent state update and stale observation | Halt/rate fields survive; old lease/pod/run observation cannot persist setup, terminal manifest, or acceptance. |
| Terminal completion followed by three transport failures; collector restart | Completion remains latched, collection retries from the same manifest staging, no generic three-error deletion. |
| Interrupted transfer, truncated or same-length-corrupt file, missing required bank/cache/metadata | No accepted receipt and no normal deletion. Resume replaces bad partials; correct local bytes then pass. |
| Old receipt, wrong run/pod/contract/manifest, receipt copied from another generation, modified accepted file | No skip of required collection and no deletion. Existing stale evidence remains preserved. |
| Hash-valid but unloadable tensor; incorrect shape/dtype/nonfinite values; wrong FP32-to-FP16 mean | Local semantic/numerical validation fails even though the remote/local hashes agree; no acceptance. |
| Source manifest/pointer change during transfer; pending source pruning | No mixed-generation acceptance. Pin current manifest and preserve/retry its immutable generation. |
| Crash before/after generation publication and receipt fsync/link; double collector | Restart reuses or reverifies durable state; exactly one immutable accepted generation/receipt; no partial marker or duplicate conflicting acceptance. |
| Accepted complete run, then status unavailable | Exact accepted generation and receipt permit normal owned-pod deletion and three bound absence confirmations. |
| Early reserve reached; no acceptance | STOP and preservation begin, with no normal deletion; later accepted receipt permits deletion. |
| Actual budget deadline while transfer pending or local checks fail | Structured budget-emergency record precedes owned-pod deletion; missing artifacts remain explicitly unaccepted; no cap extension. |
| New ledger with prior debit plus a new attempt; restart/retry/create race | Combined debit never resets or double-counts, cannot decrease, and no second paid lease is created from an uncertain first mutation. |

Finally run a small real local save/copy/load/verify smoke using tiny CPU tensors, not real model weights or outcome data: create one owned sealed checkpoint and final bank with the existing save primitives, publish a run-bound manifest, interrupt one copy, restart into the same staging, verify hashes, reload using the real readers, check the exact converted mean, and publish/consume the real acceptance receipt. Then flip one serialized artifact byte and separately use a hash-consistent numerically wrong bank; both must fail acceptance. Record source pins, file records, transitions, and the absence of provider/network actions. Root and an independent reviewer must assess this evidence and the frozen mechanism before any new paid science.
