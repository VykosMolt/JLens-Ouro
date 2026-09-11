# Native-logit diagnostic v1 — pre-run rules

Fixed by root after A7 and before any paid resource or outcome. The A7 reports are [review](../../../reviews/diagnostic_native_logit_v1_review.md) and [verification](../../../reviews/diagnostic_native_logit_v1_verification.md). Both returned PASS_WITH_FINDINGS, with nothing blocking. The frozen [runbook](../RUNBOOK.md) applies, with the execution changes below.

## Analysis rules

1. Bit-exact means `differing_elements == 0`. `torch.equal` alone is not enough, because it treats −0.0 and +0.0 as equal.
2. If M1's logits are bit-exact, attempt05's failure did not recur on this pod, and that is the primary finding. M3 localizes the attempt05 mismatch only if, for every item, M2's native logits and its last-token virtual191 state are both bit-exact with M1's. Otherwise, M3 describes M2's forward only.
3. An item's exit step counts as identified only if exactly one step's norm output matches the lm_head input row and that step is the one used. Otherwise, that item's recomputations are reported as conditional on step 3.
4. The norm or the lm_head is shape-sensitive on this runtime if its native-shape recomputation is bit-exact with the native output while some other shape is not.
5. A defined recomputation achieves exact equality only if, for all three items and in both repeats, its composite is bit-exact with the native logits, and M2 and M4 are also bit-exact. If M2 and M4 differ, exact equality is not achievable on this runtime.
6. The CPU reanalysis retokenizes the three items (S = 20, 15, 13), compares virtual191 with `norm_input[3,0,-1]`, and recomputes every comparison from the accepted tensors.
7. Magnitudes and top-k agreement are descriptive only. Conclusions cover only this GPU type, runtime, items and the measured shapes.

## Execution changes to the runbook

- Every command repeats the shell-setup block. Step 0 uses `mkdir -p`.
- Between steps 0 and 10, nothing is written under the round directory outside this diagnostic directory.
- **Step 2:** `USER_APPROVAL.json` also records the SHA-256 of both A7 reports. If the approval is more than 10 minutes old at create, repeat step 1.
- **Step 4:** runs only if `USER_APPROVAL.json` exists. F4 is judged by `mutation_phase`. Before deploy there is no pod to tear down, so report it and never rerun create.
- **Before step 6:** poll SSH until `/workspace/jlens/provider_lease_name` prints the lease name, spacing out reruns.
- **F8:** judge by `retrieval_in_progress` and `retrieval_last_error`, because `sync` prints `[]` while another sync holds the lock.
- **P:** also copies `/workspace/jlens/results`.
- **F10:** reruns only `lease.py terminate`, with the same document.
- **Step 10:** passes `--pod-id` only if a pod was bound, runs `build.py verify`, and checks `config_record` against `BOOTSTRAP_STARTED.json`.
- Tests are never rerun inside this directory.
