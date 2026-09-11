# Claim-change log

Compares the [claims table](../../confirmation_2026-09-09/CLAIMS.md) as of 11 September 2026 with the post-confirmation verification. Nothing in the original record was edited.

## Existing claims

| # | Claim | Before | After | What changed |
|---:|---|---|---|---|
| 1 | All historical estimator binaries were preserved | Contradicted | Contradicted | Purge reconciled: of 189 deleted files, 126 have verified duplicates, 1 is reconstructible, 43 are unique and lost and 19 are unresolved. Three of the seven original complete binaries survive, plus the fit02 reconstruction. |
| 2 | Ouro fit02 can be recovered without another fit | Supported as reconstruction | Unchanged | The reconstructed bank hashes to the historical value and loads. Its source checkpoint was purged, so the conversion can no longer be rerun. |
| 3 | Available Ouro banks implement their recorded checkpoint conversion | Supported for retained artifacts | Unchanged; record only | All FP32 checkpoints were purged, so the zero-difference audit remains a saved record (level 1) and cannot be repeated. |
| 4 | Historical Huginn matrices can be recovered from retained scores or logs | Unsupported | Unchanged | No bank, checkpoint or seed cache survives. |
| 5 | A newly seeded-identical Huginn rerun would be the recovered original | Not an allowed inference | Unchanged | — |
| 6 | Ouro's loop-4 layers 26–37 advantage generalizes to new items | Supported for this population and fit01 | Unchanged | Now also backed by: exact rescoring from retained states and banks, with the endpoint unchanged; FP64 reference +0.22928 (−0.00260); regenerated states +0.23608 (+0.00419); tie-order bounds +0.22604 to +0.23352. The lowest interval bound among the replay and reference paths is +0.16170. |
| 7 | Any positive excess means increased intended-concept recovery | Not by excess alone; here intended recovery increased | Unchanged | Intended difference +0.23229 (FP64) and +0.23906 (regenerated); control difference changes by at most 0.00003. |
| 8 | The new benchmark is a matched-distribution replication | Unsupported | Unchanged | Relation mixture restated from source: 96 not represented, 45 familiar requested family, 19 familiar relation recombined. |
| 9 | Calibration stability on old items establishes new-item replication | Unsupported | Unchanged | Old-item rescoring is unavailable: the historical cache was purged, and fits 03–05 banks were never recovered. |
| 10 | The early multihop deficit transfers | Supported | Unchanged | All six contrasts stay negative, with simultaneous intervals excluding zero, on every tested path. |
| 11 | The final target and cross-position aggregation each add to the advantage | Supported by secondary contrasts | Supported; one sub-statement reworded | fit01 − penultimate and sampled-sum − diagonal hold on every path. **"Diagonal minus raw includes zero" is replaced by "diagonal minus raw is borderline":** its lower bound is −0.00340 (accepted), −0.00400 (FP64) and +0.00037 (regenerated). |
| 12 | The corrected worker's precision settings match the frozen record on the live GPU | Supported | Unchanged | The local replays also matched the full dictionary. This still does not imply logit equality. |
| 13 | Native final logits equal unembedding of one final residual vector on the deployed runtime | Contradicted for the development items | Unchanged | Consistent local evidence, on a different GPU: single-row unembedding differs for all 160 items there, by at most 50 logits and with no hit change. |
| 14 | The mismatch arises in lm_head applied to a single row | Supported for the development items on the RTX 5090 | Unchanged | The local GPU reproduces the retained 5090 outputs exactly at every retained shape, including the single-row differences. |
| 15 | Exact native-logit equality is achievable with a defined recomputation | Supported, and passed on the confirmation run | Unchanged, scope extended off-device | Locally, native logits equal the scorer's [192,2048] call, the [160,2048] exit call and the whole sequence for all 160 items. The 190/191/192-row development shapes equal native. None of this was re-measured on an RTX 5090. |
| 16 | Top-k agreement means the native gate effectively passed | Not an allowed inference | Unchanged | — |
| 17 | Historical Huginn supports the predicted relative improvement | Unsupported | Unchanged | Saved scores (level 1) only. |
| 18 | Controller tests alone establish deployed preservation | Insufficient alone; demonstrated live | Unchanged | — |

## New or corrected statements

| Statement | Status | Evidence |
|---|---|---|
| "The untested scorer shapes do not affect the results" | Replaced by a measured statement | Against exact FP64 arithmetic, the executed path raised the primary by 0.00260. On a GPU of the same architecture and software, 160/190/191/192-row calls give bit-identical logits, and single-row calls change no endpoint quantity. Parity on an RTX 5090 for non-executed layouts was not measured. |
| The checker pin was "made only after confirming that every run-spec field the checker uses is identical" | Corrected | Every scientific field it reads is identical; the run ID differs. The pin was written after outcomes were computed. It was first reviewed in this pass (pass with findings). |
| native_check_v1 "changes nothing else" | Corrected | It also resized run budgets (non-scientific) and changed provenance fields. `worker.py` differs from the original freeze through precision_v1. |
| Rescoring from inputs (level 2) | New: executed | All six arms and 160 items reproduced exactly on the local GPU, and repeated for raw and fit01 from the preservation bundle. |
| Replay from prompts (level 3) | New: executed, not bit-identical | 37 of 160 items regenerate bit-identically on the local GPU; the endpoint moves by +0.00419. |
| A verified preservation bundle exists | New, with a limit | 7,380 core files (14.97 GB), restoration and loading checked. It is on the same physical disk as the originals, so this is not a backup. |
