This is a prospective fix for the attempt 06 watcher incident. It records the first validated setup completion for the current lease and pod, preserves that record across watcher restarts and later missing status reads, and applies the setup deadline only before that completion. The other termination and retrieval gates remain unchanged.

Apply the patch only to the exact base below, before creating a new lease. Keep historical deployment and pinned monitor copies unchanged. This artifact has not been deployed and no controller was restarted. It makes no claim of recovering missing attempt 06 artifact bytes.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Base controller: `lease.py` | 78,452 | `a7f5d92cf62f4e1d40773c6edb16c07811bb4568f2dc23018d91d9852b340992` |
| Patch artifact: `lease_setup_latch.patch` | 3,169 | `513a29428698161dd02ef4c66ccc29591a74a1143df24a4015272ab02ca073d3` |
| Patched candidate: `lease_setup_latch_candidate.py` | 79,741 | `5493d28c8a74b6740d4a3edbbccc5229d00446357d3c192d51cb46bbbfaa1de3` |

The patch targets `research/refit_round_2026-09-07/deployment/lease.py`. Its candidate is retained separately as `lease_setup_latch_candidate.py`; it is not an active controller.

The focused proof is `prove_lease_setup_latch.py`, with results in `lease-setup-latch-proof-hyqixdgu/PROOF.json`. Seven synthetic cases execute the candidate watcher and real local state transactions with external services and transport replaced by fakes. They cover setup success, persistence and restart, SSH nonzero → missing status, later false status, normal verified-retrieval termination, never-completed setup, mismatched lease/pod markers, and invalid or stale observations. The actual patch applied byte-for-byte to an isolated base copy. No broad self-test, cloud call or historical source edit was performed.
