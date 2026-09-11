# Artifact preservation audit after the 2026-09-10 storage purge

Audit date: 2026-09-11. Independent, read-only audit. Nothing was modified, moved, copied or deleted outside this directory. No network access was used and no GPU computation was run. Repository text was treated as evidence, not instructions. The machine-readable per-file classification is in [ARTIFACT_AUDIT.json](ARTIFACT_AUDIT.json).

## Bottom line

- **Surviving banks.** The four confirmation banks survive intact. All four hash to their recorded SHA-256, load on CPU (`weights_only=True`, `mmap=True`), and map exactly onto the five bank arms of the six evaluated arms.
- **Confirmation payload.** The accepted payload is complete and self-consistent: 8 accepted directories and 71 file hashes, with no mismatch or extra file. Rerunning the frozen analysis in memory on it reproduced `analysis.json` exactly, and the saved resample arrays bit for bit.
- **What the purge destroyed.** Across the three roots it deleted 189 files (98,410,290,077 bytes). Of these:
  - 126 are verified duplicates.
  - 1 is exactly reconstructible.
  - 43 files (33 distinct contents) are unique and lost.
  - 19 files (15 distinct contents) are unresolved, because the only copy may be in a private Hugging Face repository that cannot be checked offline.
- **Contents lost with no copy anywhere:**
  - all four complete Ouro FP32 N100 checkpoints;
  - the truncated Huginn checkpoint;
  - both historical Huginn evaluation caches;
  - the historical Ouro evaluation state cache;
  - the frozen Huginn verification archive;
  - the application-era jlens lens files and probe caches;
  - three optimization benchmark binaries.
- **Evidence levels for the confirmation claims:**
  - Level 1 (tables from saved scores) is available, and was executed here for the primary analysis.
  - Level 2 (rescoring from banks and retained states) is available in principle, because the payload keeps its own state cache.
  - Level 3 (replay) is available in principle. The weights are verified locally, but replay needs an RTX 5090-class GPU and the pinned runtime.
  - Level 4 (re-fitting) is only partly available. Calibration inputs and code survive, but every FP32 checkpoint is gone, so a new fit would be a new estimator.
- **Historical claims** have lost level 2 entirely: the historical Ouro and Huginn caches are gone.
- **Storage.** Everything that survives sits on one physical NVMe disk. There is no second physical device.

## Method

1. **Purge manifests.** I read the three `PURGE_MANIFEST_2026-09-11.json` files. Each listed deleted-file count and byte total reconciles with its `deleted_bytes` field.
2. **Current existence.** I checked every deleted path. Only the 118 upload chunks exist again.
3. **Size search.** I walked all regular files under `/home/moloch` and matched their exact sizes against every deleted size. I then hashed only the candidates: HF cache blobs, model blobs, the re-created chunks and the kept banks. A supplementary search outside home found no size match for any lost or unresolved file. It covered `/tmp` (tmpfs), `/var`, `/opt`, `/usr/local`, and all files over 50 MiB on `/`.
4. **Bank checks.** I hashed each kept bank whole and in 64 MiB slices, and hashed the first 179,527,232 bytes of the fit02 bank. The slices were compared with each chunk `MANIFEST.json` and with the purge manifest.
5. **Bank loading.** I loaded the banks on CPU with the project venv (torch `2.12.0.dev20260407+cu128`), without materializing whole banks.
6. **Payload checks.** I verified every accepted stage directory:
   - canonical manifest digest, receipt and validation records;
   - the NUL-separated `FILES_FROM` list;
   - the verifier source record;
   - every listed file's hash.
7. **Level-1 checks.** I reran the frozen `analyze()` in memory and checked `development/native.pt` equalities on CPU.
8. **Retrieval records.** I traced lineage and remote-copy records through the jlens retrieval receipts and lens sidecars.

## 1. Classification of deleted files

Class definitions used:

| Class | Meaning |
|---|---|
| (a) verified duplicate | An intact copy exists now, including re-materialization at the original path. Its SHA-256 and size were recomputed in this audit and match the purge manifest. |
| (b) reconstructible | Exact bytes are derivable from retained inputs whose hashes were verified here, and the derived bytes' hash was checked. |
| (c) unique and lost | No copy found and no verified derivation. There is also no local record of a remote copy. |
| (d) unresolved | No local copy, but a local record points to a remote copy that cannot be verified offline. |

### Counts and bytes per root

| Root | (a) verified duplicate | (b) reconstructible | (c) unique and lost | (d) unresolved | Total (= manifest) |
|---|---|---|---|---|---|
| `/home/moloch/ouro_project/artifacts/jlens` (purge 22:06:42Z) | 6 / 9,613,689,744 | 0 / 0 | 20 / 21,635,005,017 | 19 / 19,168,670,984 | 45 / 50,417,365,745 |
| `refit_round_2026-09-07` (purge 22:06:42Z) | 1 / 1,602,276,737 | 1 / 179,527,232 | 14 / 31,082,524,066 | 0 / 0 | 16 / 32,864,328,035 |
| `confirmation_2026-09-09` (purge 22:17:17Z) | 119 / 13,238,185,196 | 0 / 0 | 9 / 1,890,411,101 | 0 / 0 | 128 / 15,128,596,297 |
| **All** | **126 / 24,454,151,677** | **1 / 179,527,232** | **43 / 54,607,940,184** | **19 / 19,168,670,984** | **189 / 98,410,290,077** |

Distinct contents: class (c) holds 33 distinct SHA-256 values and class (d) holds 15.

The 118 confirmation upload chunks are counted as (a). They were re-materialized at their original paths at 2026-09-10T23:03:41–55Z, after the purge, and all 118 hash-match. They are also independently reconstructible: every 64 MiB slice of the kept banks matches both the chunk manifests and the purge manifest. If you prefer to count them as (b), the confirmation root reads (a) 1 / 5,336,011,242 and (b) 118 / 7,902,173,954.

### Class (a): verified duplicates

| Deleted file(s) | Surviving verified copy |
|---|---|
| jlens `lens/b300_jlens-b300-20260905-0359/exit3.pt` and `retrieved/jlens-b300-20260905-0359/lens/n100/exit3.pt` (d7c26297…) | `~/.cache/huggingface/hub/models--Vykos--ouro-jlens-results/blobs/d7c26297…` |
| jlens `lens/n100/exit3_shard_0000_0008.pt`, `_0008_0032`, `_0032_0056`, `_0056_0080` (04e902ac…, 8779292e…, 2c6a15f0…, f96a7142…) | Same HF cache, blobs with those names |
| refit `retrieved/ouro/fit_01/final/.../lens.pt` (90f01f6a…) | Kept fit01 bank in `prefetch_v3_20260909T134030Z_649dff12/staging/...` |
| confirmation `resources/model_snapshot/1ed04250.../model.safetensors` (2fdf9805…) | `/home/moloch/ouro_project/artifacts/hf_cache/hub/models--ByteDance--Ouro-2.6B/blobs/2fdf9805…` (the snapshot symlink resolves to it; all 13 snapshot files match `model_manifest.json` 377d9e75…) |
| confirmation `resources/input_chunks/<bank>/*.part` (118 files) | Re-materialized at the same paths, hash-verified; also derivable by 64 MiB slicing of the verified banks |

### Class (b): reconstructible

| Deleted file | Reconstruction |
|---|---|
| refit `prefetch_v3_20260909T144501Z_646f716f/staging/ouro/fit_02/final/cursor_000100_f707cb4efc844266aacff02c81ced5d0/lens.pt` (179,527,232 bytes, 43bf646d…) | `head -c 179527232 confirmation_2026-09-09/artifacts/reconstructed/ouro/fit_02/lens.pt`. The prefix hash was verified as 43bf646d… in this audit. The input bank was verified as 101f31db…. |

### Class (c): unique and lost (43 files)

**jlens root (20).** None of these had a size match anywhere, and none has a remote receipt.

| File | Bytes | SHA-256 (prefix) | Note |
|---|---|---|---|
| `lens/exit0/exit0_p0-24.pt` | 394,278,314 | 36b3d5ee | Application-era lens; listed as `retained_raw_inputs` in jlens `MANIFEST.json` |
| `lens/exit1/exit1_p0-24.pt` | 796,944,858 | c9484f22 | same |
| `lens/exit2/exit2_p0-24.pt` | 1,199,611,445 | 772878a1 | same |
| `lens/exit3/exit3_merged.pt` | 1,602,278,298 | 44cc08ce | same |
| `lens/exit3/exit3_n56.pt` | 1,602,277,707 | beb0592c | same |
| `lens/exit3/exit3_n80.pt` | 1,602,277,707 | cd49c026 | same |
| `lens/exit3/exit3_p0-8.pt` | 1,602,277,904 | e5fd1d3c | same; identical bytes also deleted at `lens/pilot/` |
| `lens/exit3/exit3_p32-56.pt` | 1,602,278,298 | caa6fc6b | same |
| `lens/exit3/exit3_p56-80.pt` | 1,602,278,298 | 669faf13 | same |
| `lens/exit3/exit3_p8-32.pt` | 1,602,278,101 | fdc10b6a | same |
| `lens/pilot/exit3_p0-8.pt` | 1,602,277,904 | e5fd1d3c | duplicate of the file above; both deleted |
| `lens/n100/exit0.pt` | 394,278,685 | 3693469e | Local merge of four exit0 shards; the shards are class (d) |
| `lens/n100/exit1.pt` | 796,945,565 | 12c1a434 | Local merge of four exit1 shards; the shards are class (d) |
| `lens/n100/exit2.pt` | 1,199,612,552 | a5e6e1e7 | Local merge of four exit2 shards; the shards are class (d) |
| `lens/n100/exit3.pt` | 1,602,279,480 | d3bc7c42 | Local merge of five exit3 shards. Another merge of the identical five inputs survives as d7c26297… at the same size but with different bytes, so byte-exact re-merge is not expected. Tensor equivalence to d7c26297… is untested. |
| `probe/history/recovered-2026-09-04/gpu_cache.npz` | 510,421,966 | bfbb9308 | Historical probe cache (MANIFEST group `historical_evidence`) |
| `probe/n80/gpu_cache.npz` | 510,101,594 | 912a0a09 | Probe cache |
| `probe/n80_v2/gpu_cache.npz` | 392,103,153 | 4cb03e20 | Probe cache. MANIFEST records a different file at this path (392,103,392 bytes, 0e8418c9…), so the manifested bytes were already gone before the purge. |
| `probe/pilot/gpu_cache.npz` | 510,101,594 | f07bbade | Probe cache |
| `probe/round1_exit3x32/gpu_cache.npz` | 510,101,594 | ceacedaa | Probe cache |

**Refit root (14).** All paths are relative to `refit_round_2026-09-07/`.

| File | Bytes | SHA-256 (prefix) | Note |
|---|---|---|---|
| `cloud_leases/attempt_06/retrieved/ouro/fit_01/checkpoints/cursor_000100_04725c9f.../state.pt` | 3,204,501,185 | aecd994c | Complete Ouro fit01 FP32 N100 checkpoint; sole copy |
| `cloud_leases/attempt_06/retrieved/ouro/fit_02/checkpoints/cursor_000100_c77feb54.../state.pt` | 3,204,501,185 | 474c6ca7 | Complete fit02 checkpoint; the source of the fit02 reconstruction |
| `cloud_leases/attempt_06/retrieved/controls/ouro_penultimate/fit_01/checkpoints/cursor_000100_cfc6042d.../state.pt` | 3,187,723,651 | cc61b21e | Complete penultimate checkpoint |
| `cloud_leases/attempt_06/retrieved/controls/ouro_positions/fit_01/checkpoints/cursor_000100_14e68ba4.../state.pt` | 6,409,003,339 | e5f43582 | Complete positions checkpoint (both arms) |
| `cloud_leases/attempt_06/prefetch_20260909T120245Z/staging/controls/ouro_penultimate/fit_01/checkpoints/.../state.pt` | 1,779,204,096 | 97a3059d | Truncated prefix copy of cc61b21e, which is itself lost |
| `cloud_leases/attempt_06/prefetch_v2_20260909T123125Z_58b81f3b/staging/controls/ouro_positions/fit_01/checkpoints/.../state.pt` | 3,975,348,504 | c4912ff4 | Truncated prefix copy of e5f43582, which is itself lost |
| `cloud_leases/attempt_06/retrieved/huginn/fit_01/checkpoints/cursor_000100_a540b860.../state.pt` | 1,745,780,736 | 099a99fd | Only Huginn checkpoint evidence; truncated (1,745,780,736 of 3,568,444,419 bytes) |
| `monitoring/attempt_06/huginn_handoff_20260909T153351Z_659a482c/results/huginn_evaluation/seeds/2026090803/cache.pt` | 141,949,409 | 226437a4 | Historical Huginn seed cache; all 5 copies deleted |
| `monitoring/attempt_06/huginn_handoff_20260909T153351Z_659a482c/results/huginn_evaluation/seeds/2026090804/cache.pt` | 141,949,409 | de18bf96 | Historical Huginn seed cache; all 3 copies deleted |
| `monitoring/attempt_06/main_only_20260909T071754Z/results/ouro_evaluation/common/cache.pt` | 407,374,621 | fb51d115 | Historical Ouro evaluation state cache; all 4 copies deleted |
| `monitoring/attempt_06/ouro_handoff_20260909T110503Z/results/ouro_evaluation/common/cache.pt` | 407,374,621 | fb51d115 | Identical copy of the above |
| `optimization/baseline_full.pt` | 3,204,502,175 | 167fc1ed | Fitting-throughput benchmark output; JSON/log summaries retained |
| `optimization/optimized_full_b8.pt` | 3,204,463,663 | a9c16a7d | same |
| `optimization/offload_reference_b8_v2.npz` | 68,847,472 | d54751a1 | same |

**Confirmation root (9).** All paths are relative to `confirmation_2026-09-09/huginn_verification/`.

| File | Bytes | SHA-256 (prefix) | Note |
|---|---|---|---|
| `bundle.tar.gz` | 223,965,405 | 2d3b4cae | Frozen Huginn verification archive. Its member `prerequisites/ouro_evaluation/common/cache.pt` (fb51d115…) is lost everywhere. |
| `bundle/prerequisites/ouro_evaluation/common/cache.pt` | 407,374,621 | fb51d115 | Copy of the lost historical Ouro cache |
| `resources/contract_probe/prerequisites/ouro_evaluation/common/cache.pt` | 407,374,621 | fb51d115 | same |
| `reviews/.historical-comparison-fixture-3swbyq12/synthetic_accepted/results/fit/lens.pt` | 141,949,409 | 226437a4 | Synthetic fixture copy with the lost Huginn cache bytes |
| `reviews/.historical-comparison-fixture-3swbyq12/.../readouts/seeds/2026090803/cache.pt` | 141,949,409 | 226437a4 | same |
| `reviews/.historical-comparison-fixture-3swbyq12/.../readouts/seeds/2026090804/cache.pt` | 141,949,409 | de18bf96 | same |
| `reviews/.historical-comparison-fixture-gut_do_k/synthetic_accepted/results/fit/lens.pt` | 141,949,409 | 226437a4 | same |
| `reviews/.historical-comparison-fixture-gut_do_k/.../readouts/seeds/2026090803/cache.pt` | 141,949,409 | 226437a4 | same |
| `reviews/.historical-comparison-fixture-gut_do_k/.../readouts/seeds/2026090804/cache.pt` | 141,949,409 | de18bf96 | same |

### Class (d): unresolved (19 files, all in the jlens root)

For 18 of these files, the reason is the same. A local retrieval receipt under `retrieved/jlens-b300-20260905-0359.receipts/lens/n100/` records the exact size and SHA-256 at a `remote_path` under `jlens-b300-20260905-0359/artifacts/lens/n100/`. That path is in the private Hub repository `Vykos/ouro-jlens-results`, named as `results_repository` in `runs/jlens-b300-20260905-0359/state.json`. No local copy exists, and whether the remote copy still exists and is intact cannot be verified offline.

| File | Bytes | SHA-256 (prefix) |
|---|---|---|
| `retrieved/jlens-b300-20260905-0359/lens/n100/exit0_shard_0000_0025.pt` | 394,279,661 | 48041804 |
| `retrieved/.../exit0_shard_0025_0050.pt` | 394,279,661 | e4872d1c |
| `retrieved/.../exit0_shard_0050_0075.pt` | 394,279,661 | a9c8a74d |
| `retrieved/.../exit0_shard_0075_0100.pt` | 394,279,661 | fa4823d7 |
| `retrieved/.../exit1_shard_0000_0025.pt` | 796,947,245 | 4da502ff |
| `retrieved/.../exit1_shard_0025_0050.pt` | 796,947,245 | 5a8126e1 |
| `retrieved/.../exit1_shard_0050_0075.pt` | 796,947,245 | 654928cb |
| `retrieved/.../exit1_shard_0075_0100.pt` | 796,947,245 | 3f5c7ef5 |
| `retrieved/.../exit2_shard_0000_0025.pt` | 1,199,615,000 | 6ed26f99 |
| `retrieved/.../exit2_shard_0025_0050.pt` | 1,199,615,000 | 922d036d |
| `retrieved/.../exit2_shard_0050_0075.pt` | 1,199,615,000 | e25cafe5 |
| `retrieved/.../exit2_shard_0075_0100.pt` | 1,199,615,000 | d9788ead |
| `lens/n100/exit2_shard_0000_0025.pt` | 1,199,615,000 | 6ed26f99 (same bytes as the retrieved copy) |
| `lens/n100/exit2_shard_0025_0050.pt` | 1,199,615,000 | 922d036d |
| `lens/n100/exit2_shard_0050_0075.pt` | 1,199,615,000 | e25cafe5 |
| `lens/n100/exit2_shard_0075_0100.pt` | 1,199,615,000 | d9788ead |
| `retrieved/.../exit3_n32.pt` | 1,602,280,332 | eea0d692 |
| `retrieved/.../exit3_n56.pt` | 1,602,280,332 | 9ec2189d |
| `lens/n100/exit3_shard_0080_0100.pt` | 1,602,282,696 | a803d687 |

The last row, `exit3_shard_0080_0100.pt`, has a different reason. It has no receipt and no HF cache entry. Its four sibling exit3 shards are present in the local HF cache at successive repository revisions. That suggests, but does not establish, that this fifth shard was uploaded.

## 2. Surviving banks

### Hash verification

All four hash to their recorded SHA-256, with mtimes unchanged since the purge manifests. Every 64 MiB slice matches the chunk manifests: 120 slices, including 118 purge-recorded ones.

| Kept bank | Bytes | SHA-256 | Result |
|---|---|---|---|
| `refit_round_2026-09-07/cloud_leases/attempt_06/prefetch_v3_20260909T134030Z_649dff12/staging/ouro/fit_01/final/cursor_000100_e090a066bc3b4630891f195e7e7d1d29/lens.pt` | 1,602,276,737 | 90f01f6a… | match |
| `confirmation_2026-09-09/artifacts/reconstructed/ouro/fit_02/lens.pt` | 1,602,276,737 | 101f31db… | match |
| `refit_round_2026-09-07/cloud_leases/attempt_06/retrieved/controls/ouro_penultimate/fit_01/final/cursor_000100_6d4d010756ab4a1cbca52509c897410c/lens.pt` | 1,593,887,875 | dc6354df… | match |
| `refit_round_2026-09-07/cloud_leases/attempt_06/retrieved/controls/ouro_positions/fit_01/final/cursor_000100_435bacee56d94ff988ff069b59f2c987/lens.pt` | 3,204,553,347 | b8e8b7d2… | match |

### CPU load: keys, shapes and dtypes

Finiteness was sampled on the first, middle and last matrices only. All sampled matrices were finite, all tensors were on CPU, none required gradients, and stat was unchanged after loading.

| File | Top-level keys | `J` structure |
|---|---|---|
| fit01 (90f01f6a) | `J`, `n_prompts`=100, `d_model`=2048, `source_layers` (191) | dict int 0..190, 191 × (2048, 2048) float16 (801,112,064 elements) |
| fit02 (101f31db) | same | dict int 0..190, 191 × (2048, 2048) float16 |
| penultimate (dc6354df) | `J`, `n_prompts`=100, `d_model`=2048, `source_layers` (190) | dict int 0..189, 190 × (2048, 2048) float16 (796,917,760 elements) |
| positions (b8e8b7d2) | `J`, `n_prompts`=100, `d_model`=2048, `source_layers` (191), `schema_version`=1, `kind`=`named_jacobian_lenses`, `bank_arms`=[`sampled_sum`, `diagonal`] | `J['sampled_sum']` and `J['diagonal']`, each dict int 0..190, 191 × (2048, 2048) float16 |

### Bank-to-arm mapping

The mapping comes from `corrections/native_check_v1/bundle/frozen/run_spec.json` `bank_records` and `bundle/evaluation/worker.py` lines 129–140. That run_spec's SHA-256, 002b91cb…, equals the payload binding. Support columns come from `measurement.py`.

| Evaluated arm | Bank file | Worker path | Selection | Target layer | Support columns |
|---|---|---|---|---|---|
| raw | none | none | not loaded | none | 192 |
| fit01 | 90f01f6a… | `banks/fit01.pt` | `state['J']` | 191 | 192 |
| fit02 | 101f31db… (exact historical reconstruction) | `banks/fit02.pt` | `state['J']` | 191 | 192 |
| penultimate | dc6354df… | `banks/penultimate.pt` | `state['J']` | 190 | 190 |
| sampled_sum | b8e8b7d2… | `banks/positions.pt` | `state['J']['sampled_sum']` | 191 | 191 |
| diagonal | b8e8b7d2… (same file) | `banks/positions.pt` | `state['J']['diagonal']` | 191 | 191 |

Four files give five bank arms and six evaluated arms. They do not correspond to four methods:
- fit01 and fit02 are two calibration fits of the same estimator.
- penultimate changes the target.
- sampled_sum and diagonal are two aggregation variants stored in one file.
- raw uses no bank.

## 3. Confirmation payload and evidence levels

### Payload verification

The payload is `confirmation_2026-09-09/cloud_leases/attempt_06/handoff/accepted/ouro_confirmation_20260911_fixed160_native1/`. It holds 8 directories: 7 stages plus the final. Every check passed for every directory:

- **Manifest identity:** the SHA-256 of canonical manifest JSON equals the directory name and `RECEIPT.manifest_sha256`.
- **Manifest bytes:** `RECEIPT.manifest_record` matches the MANIFEST.json bytes.
- **Receipt contents:** receipt `files` and `binding` equal the manifest's, and the receipt status is `passed`.
- **Validation:** the `VALIDATION.json` record matches, its status is `passed`, and its checked files equal the manifest files.
- **Pending and file lists:** `RECEIPT_PENDING_*` is byte-identical to `RECEIPT.json`, and `FILES_FROM` (NUL-separated) equals the manifest file list.
- **Verifier:** the verifier `validate_outputs.py` matches its record.
- **Files:** every listed file hash-matches, and there are no extra files.

| Stage | Files verified | Bytes | Accepted (UTC) |
|---|---|---|---|
| development | 5 | 16,732,156 | 2026-09-11T00:59:28Z |
| raw | 8 | 413,584,153 | 01:04:34Z |
| fit02 | 8 | 413,202,594 | 01:05:11Z |
| diagonal | 8 | 411,908,636 | 01:05:24Z |
| penultimate | 8 | 410,858,543 | 01:05:37Z |
| sampled_sum | 8 | 412,058,984 | 01:05:51Z |
| fit01 | 8 | 413,237,708 | 01:06:04Z |
| final (ff7c0769…) | 18 | 503,729,173 | 01:06:24Z |

Further facts about the payload:
- Each relative path has one hash across all stages, so stage copies are identical. That includes seven copies of `common/cache.pt` (377,492,133 bytes, e78ba9da…).
- The frozen run_spec equals the binding, and the output contract is 10cfeacc….
- The handoff staging directory is empty.
- The confirmation run was frozen at 2026-09-10T22:56:18Z, after the purge.

Two checks were executed in this audit:
- **Level-1 reanalysis.** Frozen `analyze()` run in memory on the final directory reproduced `results/.../analysis/analysis.json` exactly. All five arrays in `item_statistics_and_resamples.npz` matched bit for bit. The result is primary +0.23188 [+0.16289, +0.32898], on 160 items in 28 groups.
- **Development native check.** On CPU, `development/native.pt` gave virtual == physical == wrapper states and native == unembedded logits (`torch.equal`) for all three development items.

### What each evidence level depends on

These shared dependencies apply to the confirmation-run claims.

- **Level 1: tables from saved scores.**
  - Needs the hash-verified `readouts/*.npz` and `population.json`, plus frozen `analyze.py` and `measurement.py`.
  - Available, and executed here.
- **Level 2: rescoring from verified banks and retained states.**
  - Needs the payload `common/cache.pt` (H 160×192×2048 FP32, exit_logits 160×4×49152) and the verified banks.
  - Also needs the verified Ouro-2.6B weights, because readout applies the model's final norm and BF16 unembedding, plus the bundle code.
  - Available in principle, not executed.
  - Bitwise rank equality away from the original RTX 5090 BF16 `lm_head` is not established; the native_logit_v1 diagnostic shows shape- and kernel-dependent BF16 logits.
- **Level 3: replay from frozen model inputs.**
  - The model inputs are in place:
    - the Ouro-2.6B snapshot 1ed04250, whose 13 files match `model_manifest.json` (weights blob verified);
    - the frozen population, benchmark and run_spec;
    - the bundle code.
  - It also needs the pinned runtime and an RTX 5090-class GPU (capability 12.0) for bit-level comparability.
    - The runtime is pinned by `bundle/legacy/requirements.lock` with wheel hashes; installing it needs the network.
    - The local venv merely reports matching torch, numpy, transformers, tokenizers and safetensors versions.
  - Available in principle; not executed, and not possible here.
- **Level 4: reproducing estimator fitting.**
  - Calibration JSONs (`refit_round_2026-09-07/audit/calibration_fit_01..05.json`, `deployment/huginn_calibration.json`), fitter code and model survive.
  - Every FP32 N100 checkpoint is gone.
  - A re-fit needs GPU time and budget and would be a new estimator. `optimization/RESULTS.md` shows BF16 Jacobians change with batch geometry, so byte identity with the historical bank is not established.

| # | Claim (confirmation `CLAIMS.md`) | L1 saved scores | L2 rescoring | L3 replay | L4 fitting |
|---|---|---|---|---|---|
| 1 | All historical estimator binaries were preserved (Contradicted) | Records only: ledger, inventory, purge manifests, this audit | n/a | n/a | n/a. Now only 3 of 7 original complete binaries physically survive, plus the fit02 reconstruction; all 4 complete FP32 checkpoints, the duplicate fit01 final and the partial copies were deleted. |
| 2 | fit02 recovered without another fit (Supported as reconstruction) | Available: bank re-hashed = historical 101f31db; `tensor_audit.json` | Available in principle (fit02 bank + cache + model) | As shared L3 | Conversion lost: source checkpoint 474c6ca7 purged; the purged partial final is re-derivable as a prefix (verified) |
| 3 | Banks implement recorded conversion (Supported for retained) | Record only (`tensor_audit.json`) | n/a | n/a | Lost: all four FP32 checkpoints purged; re-audit impossible |
| 4 | Huginn matrices recoverable (Unsupported) | Record only (`huginn_retention.json`) | Unavailable | Unavailable for J-Lens | Original unrecoverable: truncated checkpoint purged |
| 5 | Seeded-identical Huginn rerun = original (Not allowed) | n/a | n/a | Rerun inputs present: huginn-0125 snapshot bb6621b6 matches all 16 manifest entries (verified today) | Rerun only: calibration and code retained, GPU needed. The frozen archive is purged. The plan's elementwise state comparison is impossible because both seed caches are purged; ranks and initializations remain comparable. |
| 6 | Loop-4 advantage generalizes (+0.23188) | **Executed** | Available in principle (fit01 bank; raw needs none) | As shared L3 | Partial: `calibration_fit_01.json` retained; checkpoint aecd994c purged |
| 7 | Excess shows intended recovery | **Executed** (components) | As #6 | As #6 | As #6 |
| 8 | Matched-distribution replication (Unsupported) | Available (benchmark, population, review) | n/a | n/a | n/a |
| 9 | Old-item calibration stability ≠ new-item replication (Unsupported) | New items executed; old items available from 19 historical NPZ (current hashes present in `RETAINED_INPUTS_SUPPLEMENT.json`) | New items available in principle. Old items unavailable: the historical Ouro common cache fb51d115 is purged in all 4 copies, and fits03–05 banks were never recovered. | New items as #6; old items fit01/fit02 only | Partial: calibration_fit_01..05 retained, all checkpoints purged |
| 10 | Early multihop deficit transfers | **Executed** (secondary family) | As #6 | As #6 | As #6 |
| 11 | Final target and aggregation add to the advantage | **Executed** | Available in principle (penultimate dc6354df; positions b8e8b7d2 holds both arms; fit01) | As shared L3 | Partial: control checkpoints cc61b21e and e5f43582 purged |
| 12 | Precision settings match on the live GPU | Records (payload `provenance.json` vs run_spec; forensics) | n/a | Re-observation needs the RTX 5090 runtime | n/a |
| 13 | Native logits = single-vector unembedding (Contradicted) | Preserved evidence (`resources/ATTEMPT05_NATIVE_FAILURE_PRESERVED.json`, diagnostics records; nothing under diagnostics was purged) | n/a | Needs the deployed GPU type | n/a |
| 14 | Mismatch is in single-row `lm_head` | Records (reanalysis) | n/a | Needs the deployed GPU type | n/a |
| 15 | Exact native equality achievable; passed on confirmation | **Executed** on CPU (`development/native.pt`) | n/a | Needs GPU | n/a |
| 16 | Top-k agreement ≠ gate pass (Not allowed) | Records | n/a | n/a | n/a |
| 17 | Historical Huginn supports prediction (Unsupported) | Available: both seed `arrays.npz`, summaries, initializations, seals; `analysis/huginn_run01` | Unavailable: no bank, both caches purged | J-Lens arm unavailable; raw and coda arms possible with the verified snapshot and a GPU | Rerun only |
| 18 | Deployed preservation demonstrated live | **Executed** (8 directories, 71 hashes, all records consistent) | n/a | n/a | n/a |

## 4. Historical Huginn pilot

**Status.** The evaluation completed, but the bank is unrecoverable and only level-1 evidence remains.
- `huginn_retention.json` status: `metadata_authenticated_bank_unrecoverable`.
- Setup: N100, R8, one fit, two fixed trajectories (seeds 2026090803/04).
- J-Lens minus raw is negative in all six fixed task/summary combinations in both seeds.
- The relative-improvement prediction is unresolved, because all six simultaneous intervals include zero.
- Still retained:
  - both seeds' `arrays.npz`, summaries, initializations and seals;
  - the authenticated OWNER, COMPLETE and LATEST records, checkpoint seal and metadata;
  - the 100 diagnostic row identities;
  - `huginn_calibration.json`;
  - the huginn-0125 model snapshot, verified today against its manifest.

**Missing estimator.**
- **Final bank:** never retrieved (expected 1,784,226,499 bytes, SHA-256 7eddc849…), and the final seal and metadata are absent.
- **Checkpoint:** the FP32 N100 checkpoint arrived truncated at 1,745,780,736 of 3,568,444,419 bytes, with 1,822,663,683 bytes missing. It was never deserialized, and no per-paragraph contributions exist.
- **Purge:** the purge then deleted:
  - that truncated checkpoint (099a99fd…);
  - both evaluation seed caches, in every copy;
  - the frozen verification archive.

**Conversion limitation.**
- **Unchecked relation:** the relation "FP16 bank = FP32 N100 sum / 100" is only the producer's validation assertion, per the refit REPORT's Huginn validation and fit provenance section and the retrieval claim row.
- **Contrast with Ouro:** neither the complete checkpoint nor the bank was recovered, so the relation was never independently repeated locally. For the Ouro arms, a NumPy re-conversion compared 4,001,366,016 entries with zero differences.
- **Consequence:** the relation can no longer be checked for the original estimator. The J-Lens arm cannot be rescored or replayed, and any new fit is a "newly rerun estimator".

## 5. Storage

- **Disk:** one physical disk, `nvme0n1`, a WD PC SN8000S 1 TB (serial 25363H803914).
  - `p1` 1 GiB vfat on `/boot`: 464 M used, 559 M free.
  - `p2` 952.9 GiB ext4 on `/`: 937 G size, 514 G used, 376 G free (58%).
- **Swap:** `zram0` is 4 G of RAM-backed swap. `/swapfile` is 32 G and sits on `/`.
- **tmpfs:** `/tmp` is 16 G (2.4 G used), and `/dev/shm` and `/run` are also tmpfs.
- **Other storage:** no RAID, and no removable or network storage is mounted.
- **No second physical device exists.** Every surviving copy is on this single device: the banks, the payload, both model snapshots, the HF cache and the re-materialized chunks. Verified duplicates therefore give no protection against device failure.

## Observations

- **Where the kept banks are.** The four kept banks are three in the refit root and one in the confirmation root. The jlens root kept no large file.
- **jlens MANIFEST.** jlens `MANIFEST.json` still lists the deleted application-era lens files as `retained_raw_inputs`, and the recovered probe cache as `historical_evidence`. Its `manifest verify` reproduction command would now fail on those entries; this is an inference, not run.
- **Final chunks never deleted.** Two final chunks under 50 MiB were never deleted and match their chunk manifests: `b8e8b7d2…/003154116608.part` and `dc6354df…/001543503872.part`.
- **Local GPU.** `nvidia-smi -L` lists an RTX 5070 Ti Laptop GPU. It was not used. The confirmation ran on an RTX 5090.

## Proposed recoveries (none performed)

1. **Second copy.** Copy the following to a second physical device or an independent remote, then verify each by SHA-256:
   - the four banks (8.0 GB);
   - the final accepted directory `ff7c0769…` (0.5 GB);
   - the confirmation root's small files and bundles;
   - the 19 historical NPZ files and the handoff records;
   - the Ouro-2.6B snapshot (5.3 GB) and the huginn-0125 snapshot (15.7 GB).
2. **Hub retrieval.** With network access and the Hub token:
   - List `Vykos/ouro-jlens-results`.
   - Fetch the 18 receipt-recorded shard and prefix files, and check whether `exit3_shard_0080_0100.pt` exists.
   - Verify each against its receipt SHA-256 before reclassifying the class (d) items.
   - The merged `lens/n100/exit{0,1,2}.pt` could then be re-merged numerically, though not byte-identically.
3. **Document the purge in jlens.** Record in the jlens documentation that the purge removed MANIFEST-listed retained inputs, so manifest verification does not later fail without explanation.
4. **Chunks are derived data.** Treat the 7.9 GB of re-materialized chunks as derived: they are regenerable from the banks by 64 MiB slicing.
5. **Historical rescoring.** If historical Ouro rescoring is needed, regenerate the evaluation cache by GPU replay. Label it as new and compare it with the retained historical `arrays.npz`; do not assume bit identity.
6. **No known recovery.** Nothing is known to recover the following, beyond remote copies with no local record:
   - the FP32 checkpoints;
   - the Huginn checkpoint and caches;
   - the optimization binaries;
   - the application-era jlens lens files and probe caches.

## Limits of this audit

- **Remote copies** were not checked, since no network was used.
- **Levels 2–4** were not executed. Their availability is stated from verified inputs and code, not from reruns.
- **Sampled finiteness only.** Bank finiteness was sampled, not exhaustive; exhaustive checks are recorded in the earlier `tensor_audit.json`.
- **Historical NPZ integrity** was checked by confirming that each current SHA-256 appears in the supplement's records, not by re-running the supplement's structured validation.
- **Search coverage.** The search was limited to locally mounted filesystems.
