# Confirmation controller implementation v1

The new controller implements the frozen [protocol](PROTOCOL.md) in a separate directory. Historical controllers, receipts, and experiments remain unchanged. No cloud or SSH deployment was performed by this implementation task.

The setup latch, terminal manifest, transfer progress, stage acceptance, and final receiver acceptance are separate durable records. Every scientific transition binds the exact run, lease/pod, contracts, and six pinned controller sources. Transactional updates cannot change those identities, reduce prior accounting, or extend a deadline. A worker status cannot set local acceptance.

The receiver copies declared immutable stages as soon as they appear, resumes the same manifest-keyed staging directory, verifies source and local hashes, runs the pinned local semantic verifier from its checked source bytes, and publishes a durable acceptance receipt. Final manifests can reuse already verified stage files. Damaged published generations move intact to quarantine with their previous receipt record and failure evidence before a fully reverified replacement is published; old receipt pointers do not validate the replacement.

Normal deletion requires full final acceptance and fresh payload hashes. Explicit teardown requires the local root-authored document in the protocol. A fixed spending/balance emergency records known preservation and missing-file evidence before deleting the owned resource, without waiting for the collector lock. Missing status, repeated transport failures, setup/provisioning timeout, and the early preservation deadline do not independently authorize deletion. Existing provider-account, ownership, create-race, mutation-lock, and three-absence checks remain in place.

The prior debit file is hashed and its historical lease records verified. New lease costs are added once, the combined cap stays $25, and the requested job limit cannot exceed the prior snapshot's initial $4 ceiling. Planning reserves declared setup, compute, preservation, a 600-second termination margin, and $0.15 billing margin. The bootstrap now accepts the new `jlens-confirm-` lease-name prefix.

The [source/version record](SOURCE_RECORDS.json) binds the candidate base, copied dependencies, current sources, and [durable proof](proofs/lifecycle_v1/PROOF.json). Twenty-two provider/ownership/absence protection functions retain their exact abstract syntax from the reviewed latch candidate. The proof passed **38 cases**, including an actual tiny CPU sealed-checkpoint save, interrupted copy, resume, hash verification, loadability, exact FP32-to-FP16 conversion, receiver acceptance, fake normal deletion, and three absence confirmations. It also covers the independently identified crash-recovery, stale acknowledgement, dtype, post-transfer source mutation, and bootstrap cases.

The [independent review](../reviews/controller_independent_review.md) reran the 38 cases and reconciled the exact source pins. The implementation is locally and independently verified; root still controls the prospective scientific worker, contracts, local model-specific verifier, final budget admission, and any actual deployment.

Run the standalone smoke with:

```sh
PYTHONDONTWRITEBYTECODE=1 /home/moloch/ouro_project/venv/bin/python research/confirmation_2026-09-09/controller/test_lifecycle.py --smoke
```

Omit `--smoke` for the complete fixture suite. Use a fresh `--output-dir` to preserve another proof version. These commands load only synthetic 2×2 CPU tensors and use fake provider/SSH boundaries.
