# B300 handoff — repaired prelaunch candidate

The original rental recipe is preserved at `history/2026-09-04-opus/HANDOFF_B300.md`. Do not run it.

The 2026-09-04 paid run spent approximately `$52.74` and retained no B300 validation or fit artifact. The old recipe created a paid pod without durable supervision, uploaded only on process exit, excluded unique shards, and did not terminate on successful completion.

No pod may be created or started without explicit user signoff. The controller
does not treat a successful remote process as a successful experiment until
termination, extraction, and the exact paid-result gate all succeed.
The paid subcommand also requires `--confirm-launch` to equal the exact fresh
lease-attempt ID; omission or mismatch stops before stage or account access.

## Launch boundary

Paid execution remains blocked until the current candidate has passed the
complete normal and optimized suites, independent verification, adversarial
review, paid-readiness adjudication, deterministic clean-stage construction,
private-Hub stage upload/readback, and a final account/price/no-live-pod check.
Passing those gates establishes readiness only; it is not permission to start
a pod.

The launch claim is deliberately narrower than the earlier application work:
the retained arithmetic probe and Base/Thinking/RLTT checkpoint measurements
are earlier-source/local evidence, are not re-certified as current-source
artifacts here, and are excluded from the new B300 paid verifier and
application claim. Their regeneration is not launch-critical.

The launch contract requires all of the following:

- the exact supervised-controller fault-injection suite passes;
- the account has sufficient positive balance above both reserve and maximum-budget requirements;
- a staged bundle manifest verifies locally and remotely;
- before any lease, the controller downloads that immutable stage, verifies
  its clean-commit source policy, and takes its bootstrap entrypoint from the
  verified archive rather than the live local checkout;
- the detached user service and linger preflight pass;
- every completed shard and sidecar is published under an immutable run ID and locally acknowledged by SHA-256;
- repeated API uncertainty, balance floor, budget, and runtime limits terminate the exact pod;
- termination is verified after success, failure, or controller exception.

The hard code-enforced account/runtime bounds are: minimum account reserve
`$5.00`, maximum spend `$30.00`, provider TTL `12,600` seconds, and worker
deadline `900` seconds before the provider deadline. The paid defaults use
those exact bounds: a 12,600-second provider TTL, 10,800-second minimum usable
worker window, $30 spend cap, and $5 reserve. Before a
lease, the results repository must pass a write-read-list canary: one
run-scoped record is written, read back byte-for-byte, and found by listing.

HF authentication uses a protected non-ambient channel. The controller reads a
0600 token file, passes the token explicitly to the publisher/API or one-use
bootstrap, keeps it out of command arguments and durable state, and the remote
entrypoint removes the bootstrap variable before setup and execution.
Large uploads use a descriptor-anchored `/proc/self/fd` pathname, preserving
the owned inode while keeping Hugging Face's Xet path active; BinaryIO fallback
to the serial HTTP uploader is rejected by regression coverage. Xet high-
performance mode is enabled only when `/proc/meminfo` reports at least 64 GiB
of host RAM, matching Hugging Face's published requirement.

## Result extraction and custody

- Every remotely accepted payload is installed at an immutable run-scoped path
  with an SHA-256/size/kind receipt. Payload and receipt are read back before an
  acknowledgement is accepted. A transient post-commit readback outage is
  retried; an unknown outcome never permits a blind overwrite.
- The final run emits an exact schema-2 expected inventory. The controller
  parses its lens target/shard intervals, prefix lenses, evaluation tags and
  files, reports, validation products, verifier verdict, markers, current
  attempt heartbeat, run log, checkpoint-recovery status, and cumulative
  receipt index. Broad file-kind presence is insufficient.
- Hub receipt listings are retried and unioned across observations. A stale or
  incomplete listing can delay or fail extraction, but cannot become a success.
- A successful paid run immediately downloads the application bundle: all
  reports, validation, evaluations/arrays/plots, logs, status records,
  inventory, and lens sidecars. It validates the complete remote receipt union
  but deliberately defers the large `lens/*.pt` payloads. The 25 lens tensors
  total about 27.98 GB (26.06 GiB): 17 prompt-slice recovery shards, four
  merged lenses, and four target-3 prefix lenses. They are not model-weight
  shards and are not required to write the application.
- The large tensors remain remotely acknowledged and digest-bound. The
  no-cost `recover` command can later download every one and run the full local
  paid-verifier replay. Failed or termination-uncertain runs still retrieve
  every available shard/checkpoint immediately, because those bytes are the
  recovery product rather than successful-run duplication.
- Downloads are hash-checked before installation and checked again from their
  installed paths. Exact receipt JSON files are retained separately from the
  payload tree, together with a content-addressed custody snapshot labeled
  `application`, `full_replay`, or `partial`.
- Stage archives are materialized member-by-member in a private temporary
  directory, fully verified, fsynced, and installed with an atomic Linux
  no-replace rename. Existing workspaces, links, collisions, partial
  extraction, special tar members, and traversal are rejected.
- Every paid attempt uses an absent workspace and a fresh attempt identity.
  Earlier terminal heartbeats cannot satisfy the current attempt. A valid old
  output/sidecar pair is never silently adopted from a reused local tree.
- The paid verifier binds the exact container image, current B300 CUDA device
  record, model/source/lens/evaluation lineage, canonical report regeneration,
  and artifact run ID before its verdict can enter the expected inventory.

The no-cost full-replay recovery surface is
`venv/bin/python src/ouro_jlens/pod.py recover --state <state-file>`.
It uses the terminated lease's exact state, token fingerprint, stage, remote
receipt inventory, and local roots without accessing RunPod.

## Honest failure limits

No software can recover bytes that a provider hard-kills before they have been
written and acknowledged remotely. A hard kill can therefore lose the one
currently executing shard/evaluation and its most recent pod-local checkpoint.
Every artifact acknowledged before that point remains independently
discoverable and recoverable from its receipt, even without an exit trap or
terminal index. Network, provider, or listing uncertainty produces a failed or
partial custody snapshot; it never upgrades the run to complete.

Paid same-lineage retry is disabled. Attempt-variant validation, inventory, and
verification bytes currently share fixed artifact-relative names, so reusing an
artifact lineage could conflict before shard restoration begins. A failed run's
acknowledged tensors and sealed checkpoints remain fully recoverable offline,
but any new paid attempt must use one fresh ID for both the lease and artifact
lineage. This may sacrifice recomputation after a hard kill, but prevents mixed
or nondeterministically overwritten evidence.

RunPod's requested GPU identity plus the CUDA device record establish the
operational B300 check available to this workflow. They are not a cryptographic
hardware attestation by the provider, and the documentation does not claim one.

The current safe commands are the no-cloud verification and dry-run commands in
`REPRODUCE.md`. The supervised `run` command is withheld until the readiness
gates pass and the user explicitly authorizes launch.
