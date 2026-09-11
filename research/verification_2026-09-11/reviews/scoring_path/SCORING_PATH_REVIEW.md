# Scoring-path review: ouro_confirmation_20260911_fixed160_native1

Date: 2026-09-11.

Mode:
- Read-only and CPU-only, with no model forward pass, GPU use or network access.
- The only files written are this file and SCORING_PATH_REVIEW.json.

Evidence labels:
- **measured**: I computed it from retained files.
- **code**: read from the executed source.
- **inference**: reasoned, but not directly observed.

## Abbreviations and identity checks

| Symbol | Path |
|---|---|
| R | /home/moloch/jacobian-lens/research/confirmation_2026-09-09 |
| B | R/corrections/native_check_v1/bundle (executed code) |
| ACC | R/cloud_leases/attempt_06/handoff/accepted/ouro_confirmation_20260911_fixed160_native1/ff7c0769b2b0dbe2e53b120daf7f0729434fe65e54c013288771f5052ffd4cef/results |
| MO | R/resources/model_snapshot/1ed04250da1a9936042725d302e81c8fa2ab5abd/modeling_ouro.py |

In the citations below, bare file names refer to files under B:
- worker.py, readouts.py, measurement.py and validate_outputs.py are in B/evaluation.
- evaluate.py, recurrent.py and evaldata.py are in B/ouro_project/src/ouro_jlens.
- evaluate_refits.py, evaluate_controls.py and run_refits.py are in B/legacy.
- lens.py, hf.py, hooks.py and vis.py are in B/repo/jlens.
- analyze.py is in B/analysis, and run_spec.json is in B/frozen.
- "Plan" is R/PROSPECTIVE_PLAN.md, which is byte-identical to B/frozen/PROSPECTIVE_PLAN.md.

Identity checks (measured):
- **Source pins.** All 50 files in B match `run_spec.json` `source_records`. `ACC/provenance.json` `actual_worker_source` and all 8 imported-module records equal those pins.
- **Model sources.**
  - MO has sha256 c5c68fbb…. It equals `B/legacy/model_manifest.json:48-50`.
  - configuration_ouro.py has sha256 950443e3…. It equals `model_manifest.json:33-35`.
- **Accepted copies.** Every file in all 8 accepted directories (7 stages plus final) rehashes to its MANIFEST record. Every stage record equals the final manifest record.
- **Banks.** The five bank records, which name four distinct files at `receiver_path`, rehash to `run_spec.json:1168` `bank_records`.
- **Verifier.** The receipt's verifier record equals `B/evaluation/validate_outputs.py` and its run_spec pin (10057 bytes, 588dbf72…).
- **Runtime** (ACC/provenance.json):
  - RTX 5090, capability 12.0.
  - torch 2.12.0.dev20260407+cu128 and transformers 4.54.1.
  - The `kernels` package is not installed.
  - Attention is SDPA.

## 1. Executed path per arm

### 1a. Hidden states H (shared by all six arms)

**Caching loop.**
- `worker.py:122-126` calls `evaluate.cache_states(model, [item], position=-1)` once per item, then concatenates the results.
- `evaluate.py:480` allocates H as float32.
- `evaluate.py:484-486` records `rec.activations[v][0, -1]` for v = 0..191, stacks them, and applies `.float().cpu()`.

**Forward.**
- The forward is the Ouro text module with `use_cache=False` (`hf.py:163-164`).
- Batch size is 1 and sequences are 13–35 tokens.
- Parameters are BF16 on CUDA (`evaluate_refits.py:612, 615-619`).
- SDPA is required (`evaluate_refits.py:627-629`).

**Taps.**
- A LoopTap fires only when `kwargs["current_ut"] == ut` (`recurrent.py:73-78`).
- The virtual index is v = 48·ut + layer, zero-based (`recurrent.py:88-98`).
- The recorder stores the block's output tensor (`hooks.py:49-54`).
- The decoder layer returns the post-residual tensor (`MO:423`).
- The final norm is applied after each loop (`MO:606`). Virtual states 47, 95, 143 and 191 are therefore pre-norm block outputs.

**Position.**
- The readout uses the last token of the common prefix of encode(prompt) and encode(prompt+target) (`evaldata.py:96-103`).
- BOS is prepended (`recurrent.py:114-117`).
- The worker checks these tokens against the frozen token IDs (`worker.py:113-116`).

**Stored.**
- `ACC/common/cache.pt` holds `H` as [160,192,2048] float32 on CPU (`worker.py:128`).
- Measured: `H.to(bfloat16).float() == H` holds bitwise. H therefore holds the GPU BF16 values without loss.
- Greedy continuations are stored but not scored (`evaluate.py:487`).

### 1b. Banks

The bank file comes from `run_spec.json` `bank_records`. The key is chosen at worker.py:135, the target at worker.py:137, and the rows passed at worker.py:139.

| Arm | Worker file | Key used | Keys present (FP16 [2048,2048]) | Target | J rows passed | Identity row passed |
|---|---|---|---|---|---|---|
| raw | none | none (J=None) | none | none | none: transported = h (`evaluate.py:549`) | none |
| fit01 | banks/fit01.pt | `state['J']` | 0–190 | 191 | 192 | 191 = I |
| fit02 | banks/fit02.pt | `state['J']` | 0–190 | 191 | 192 | 191 = I |
| penultimate | banks/penultimate.pt | `state['J']` | 0–189 | 190 | 190 | none |
| sampled_sum | banks/positions.pt | `state['J']['sampled_sum']` | 0–190 | 191 | 191 | none (row 191 sliced off) |
| diagonal | banks/positions.pt | `state['J']['diagonal']` | 0–190 | 191 | 191 | none (row 191 sliced off) |

Casts, in execution order (code):
1. `worker.py:79` hashes the worker-side bank. `worker.py:134` then loads it with `torch.load(..., map_location='cpu', weights_only=True, mmap=True)`, giving FP16 tensors on CPU.
2. `lens.py:40` applies `J.float()` to every key, giving FP32 on CPU.
3. `evaluate.py:600` builds `torch.eye(2048).repeat(192,1,1)` as FP32 on CPU. Lines 601-605 set `J[v] = lens.jacobians[v]` for each source layer. Line 606 moves the stack to CUDA as FP32 [192,2048,2048].
4. `evaluate_refits.py:297-299` keeps all 192 rows for target 191 and `full[:190]` for target 190.
5. `worker.py:139` slices `plan.jacobians[:columns]`. Here `columns = len(SUPPORT[arm])` (`measurement.py:8-9`), which is 192/192/192/190/191/191.

For every arm, column index = virtual index = bank key.

### 1c. Transport

- **Call.**
  - `readouts.py:33-34` passes `states[i:i+1, :C]` and the J stack.
  - `evaluate.py:548` moves `h = H[0]` to CUDA as FP32 [C,2048].
  - `evaluate.py:549` runs `torch.einsum("vde,ve->vd", J, h)` in FP32 on CUDA, one call per item covering all C columns.
  - Inference: PyTorch lowers this to a batched matmul with batch size C. The kernel was not recorded.
- **Precision switches.**
  - They were captured once, at model load, by `run_refits._precision` (`evaluate_refits.py:631`; `run_refits.py:856-888`).
  - `worker.py:74-75` required them to equal `run_spec.json:1249-1297`:
    - `float32_matmul_precision` "highest".
    - `matmul.allow_tf32` false and `matmul.fp32_precision` "none".
    - `CUBLAS_WORKSPACE_CONFIG` ":4096:8".
    - Deterministic algorithms off.
  - The executed sources contain no setter, only getters at `run_refits.py:863-887`.
  - Inference: TF32 was therefore off during transport. The switches were not re-recorded at readout time.
- **Identity row** (fit01/fit02, measured): `transported[191]` equals `H[i,191]` bitwise for items 0–2.

### 1d. Unembedding

- **Path.** `readouts.py:17` calls `hf.py:166-174`, which computes `lm_head(final_norm(residual.to(bf16).to(cuda)))`.
- **Softcap.** There is none: the Ouro config sets no `final_logit_softcapping` (`hf.py:128-130, 172-173`).
- **RMSNorm** (`MO:357-362`).
  - The BF16 input is cast to float32.
  - It computes the mean of squares, then rsqrt with eps 1e-6.
  - It casts back to BF16 and multiplies by the BF16 weight.
  - The `use_kernel_forward_from_hub` decorator (`MO:347`) is an identity when `kernels` is missing (transformers `hub_kernels.py:66-73`), and the pod's package list lacks `kernels`.
- **lm_head.** A BF16 `nn.Linear` with no bias, untied (`tie_word_embeddings` false).
- **Rows per call.** Each item gets one 2-D call of shape [C,2048] per arm. C is 192 for raw, fit01 and fit02; 190 for penultimate; and 191 for sampled_sum and diagonal.
- **BF16 rounding of lens-arm states.**
  - The lens arms' FP32 J·h is rounded to BF16 before the norm (`hf.py:170`).
  - Measured: for fit01 item 0 at v170, all 2048 elements change, by at most 0.37% relative.
  - For raw the cast is lossless, because H is already BF16.
- **Logit resolution.**
  - `evaluate.py:550` applies `.float()`, so the logits are float32 holding BF16 values.
  - Measured: the largest minimum gap between distinct values among the top 200 is 1/32 for raw and 1/16 for the lens arms.

### 1e. Ranking

- **Rank computation.**
  - `evaluate.py:553` calls `TaskNames.ranks` (`evaluate.py:514-517`), which calls `_ranks_of` (`vis.py:98-125`).
  - Because `chunk_size` is 256 and C ≤ 192, a single `argsort(dim=-1, descending=True)` runs over [C,49152] (`vis.py:119`).
  - The inverse permutation is built by `scatter_` (`vis.py:121`) and gathered at every alias token (`vis.py:123`).
  - Each name takes the minimum over its forms (`evaluate.py:517`). Output is padded to 128 names with -1 (`evaluate.py:523-526, 554`).
- **Name catalogue.** It is built from the population items (`worker.py:118-120`; `evaluate_refits.py:258-268`) and checked against the population names.
- **Ties** (measured).
  - In the saved full sorts, all 7,863,091 adjacent equal-logit pairs are ordered by ascending token ID.
  - Exactly 10 tokens have rank < 10.
  - When a tie straddles zero-based positions 9 and 10, the lower token IDs count as hits.
  - In 42 of 168 sampled rows, the logit at position 9 equals the logit at position 10.
- **Top-10 capture.** `readouts.py:18-19` runs a separate argsort on the same float32 tensor. For every item, name and column, a runtime check requires top-10 membership to equal allrank < 10 (`readouts.py:38-42`).
- **top1.** It is computed by `argmax` (`evaluate.py:558`) and is not scored. Measured: top1 equals `top10_ids[...,0]` in all 183,680 cells of the six arms.

## 2. Packing across arms

- **Row counts differ by arm.** Unembedding uses 192 rows for raw, fit01 and fit02; 190 for penultimate; and 191 for sampled_sum and diagonal. The einsum batch size follows the same pattern.
  - The primary contrast (fit01 − raw) uses 192 rows on both sides.
  - Secondary contrasts involving penultimate, sampled_sum or diagonal compare 190- or 191-row calls with 192-row calls.
  - The historical control evaluator sliced the same way (`evaluate_controls.py:370-380`).
- **Virtual 191 is an identity endpoint for fit01 and fit02.**
  - Code: J[191] = I (`evaluate.py:600-605`), and the banks have no key 191.
  - Measured:
    - `transported[191]` equals `H[i,191]` bitwise for items 0–2, and the logits at 191 equal raw's.
    - For all 160 items, column 191 equals raw's exactly in 8 arrays: allrank, rank, top1, kl_to_final, kl_to_local, rank_of_final_top1, rank_of_local_top1 and top10_ids.
- **Unsupported tails change packing.**
  - Penultimate drops J rows and H columns 190–191.
  - Sampled_sum and diagonal drop row and column 191 (`worker.py:131, 139`; `readouts.py:33`).
- **Layout evidence.**
  - Measured, 192 rows against 160 rows:
    - The 192-row raw logits at v = 47, 95, 143 and 191 equal the 160-row exit logits bitwise for items 0–2, with 0 of 49,152 elements differing in each of 12 rows.
    - `kl_to_final[:,191]` is exactly 0.0 for 160/160 items in raw, fit01 and fit02.
    - top1 at columns 47, 95 and 143 equals the exit argmax for 160/160 items.
  - No retained row permits this comparison for 190- or 191-row calls, so those layouts are unmeasured.

## 3. Exit logits

- **Computation.**
  - `worker.py:127` makes one `model.unembed(states[:, exit_index(ut)])` call per ut on [160,2048], with all items packed together. It uses the same unembed path as the arms.
  - `exit_index` = 48·ut + 47 (`recurrent.py:111-112`), giving rows 47, 95, 143 and 191.
  - They are stored as [160,4,49152] float32 (`worker.py:128`), and the values are BF16-representable (measured).
  - The historical producer used [148,2048] (`evaluate_refits.py:672-675`).
- **Use.**
  - `readouts.py:34` passes `target_ut = n_ut − 1 = 3`, so `final` and `local` are both `exit_logits[i,3]` (`evaluate.py:551`).
  - They feed only kl_to_final, kl_to_local, rank_of_final_top1 and rank_of_local_top1 (`evaluate.py:559-562`).
  - Measured: for all arms, kl_to_final equals kl_to_local, and the two rank arrays are identical.
- **They affect neither hit@10 nor eligibility.**
  - allrank and top-10 come only from the arm's own logits (`evaluate.py:550-554`).
  - analyze.py reads only `allrank` (`analyze.py:39-41`).
  - Eligibility comes from `population.json` rows (`measurement.py:23-24`; `analyze.py:42-43`).
  - The validator checks exit_logits only for shape and finiteness (`validate_outputs.py:78`).

## 4. Score definition compared with the plan

| Element | Implementation | Plan | Result |
|---|---|---|---|
| Intended hit I | min alias rank < 10 (`evaluate.py:517`; `measurement.py:31`) | Plan:9 | match |
| Control C | `(allrank[i, controls] < 10).mean(axis=0)`, where controls = every name not in own_index (`measurement.py:28, 32`) | Plan:9, :17 | match |
| Aliases | surface forms × {with, without leading space}, single-token forms only (`evaldata.py:50-67, 86-93`) | Plan:17 | match |
| Eligibility | frozen rows (scorable and not leaked), rechecked by `worker.py:115-116` and `validate_outputs.py:56` | Plan:17 | match |
| Within-item aggregation | `/ len(slots)` (`measurement.py:31-32`) | Plan:9 | match |
| Item weights | `allvalues.mean(axis=0)` (`analyze.py:48, 81`); bootstrap as sum/count (`analyze.py:28`) | Plan:9, :35 | match |
| Band | `range(3*48+25, 3*48+37)` (`measurement.py:6`); `run_spec.json:1298-1314` | Plan:11 | match |
| Fixed-layer vs any-layer | primary = mean of the per-layer excess difference over 12 columns (`measurement.py:35-39`; `analyze.py:47`); any-layer only in early_loop{1,2,3}_any_layer (`measurement.py:46-47, 63-73`; `analyze.py:66-67`) | Plan:9, :49 | match |

- **Population** (measured):
  - 160 rows and 80 names. Every row has one intermediate, is eligible, and has 79 control indices.
  - Forms per name: 1 form for 56 names, 2 for 17, 3 for 5 and 4 for 2.
  - There are no cross-concept token collisions.
- **Index convention.**
  - The band covers zero-based virtual indices 169–180.
  - That is zero-based ut 3 and zero-based physical blocks 25–36, which is one-based loop 4, physical layers 26–37.
  - Virtual index v is the output of physical block v mod 48 in loop v // 48.
- **Leak check** (measured).
  - The check uses readout-context IDs, which omit one trailing " " token in 131 of 160 rows.
  - Retokenizing all prompts locally reproduced the frozen token IDs.
  - No alias appears in the full prompt without also appearing in the context. There is no effect.
- **fit01 final third.** The fit01_final_third region (176–191) includes column 191, where fit01 ≡ raw. That column contributes exactly 0 to the contrast, as declared at Plan:11.

No difference was found between the implemented score and the frozen plan.

## 5. What the saved checks exercised

### validate_outputs.py (the receiver)

**Coverage.**
- Only items 0, 1 and 2 are sampled (`readouts.py:6`; `validate_outputs.py:106`).
- The sampled layers are (0, 47, 95, 143, 170, 176, 181, 189, 190, 191) intersected with each arm's support (`readouts.py:7`; `validate_outputs.py:116`).
  - That gives 10 layers for raw, fit01 and fit02; 8 for penultimate; and 9 for sampled_sum and diagonal: 168 in total.
  - Only 170 and 176 fall inside the band.

**FP64 transport check** (`validate_outputs.py:131-134`).
- The reference is `bank[v].numpy().astype(float64) @ h` on CPU. For raw, and for fit01/fit02 at 191, the reference is `h` itself.
- It is compared with the saved GPU FP32 transported vector at rtol = atol = 2e-5.
- The operation, precision and shape all differ from the scorer's: a single FP64 NumPy vector here, against an FP32 CUDA einsum over [C,2048,2048].
- It covers J·h only, and it uses the scorer's own H.

**Full-sort check** (`validate_outputs.py:118-130`). It verifies:
- the saved order is a permutation (:120);
- it sorts the saved logits (:123);
- its first 10 equal `top10_ids` (:127);
- NumPy argmax equals top1 (:128);
- allrank equals the per-name minimum of the inverse permutation (:124, :130).

All of these use logits and order produced on the GPU.

**Not exercised.**
- For no arm, item or layer are the logits recomputed from the transported state (BF16 cast, RMSNorm, lm_head, row layout).
- Ranks are never checked for invariance to tie order or layout.

**Population-wide.** For every item, column and name, allrank < 10 equals top-10 alias membership, and own-rank slots equal the name bank (`validate_outputs.py:97-102`).

**My measurements on the 132 non-identity sampled transported vectors.**
- GPU FP32 against a CPU FP32 einsum: 235,422 of 270,336 elements differ bitwise.
- After BF16 rounding, GPU against the FP64 reference: 51 elements differ.
- After BF16 rounding, GPU against CPU FP32: 93 elements differ.
- The largest |GPU − FP64| difference is 2.0e-6.

### Independent checker (R/reviews/primary_numerical_check_native1.py)

- **Inputs and rebuild.**
  - It reads only `allrank`, `rank` and `top10_ids` (:378).
  - It sets hits = allrank < 10 (:207) and cross-checks them against `top10_ids` (:208-210).
  - It rebuilds weights, components, bootstraps and secondaries (:284-316).
- **Scope.** It does not recompute ranks, logits or transport; it aggregates saved ranks.
- **Result.** Status passed, with 231 comparisons and a maximum difference of 8.9e-16. Its NPZ input records equal the final manifest's.
- **Frozen original.** R/reviews/primary_numerical_check.py failed at its run_spec pin (R/results/ouro_confirmation_20260911_fixed160_native1/independent_numerical_check.json). The native1 copy differs by 2 lines: the added pin and its use.

## 6. Retained arrays (ACC)

C is 192 for raw, fit01 and fit02; 190 for penultimate; and 191 for sampled_sum and diagonal. Shapes and dtypes were measured.

| File | Key | Shape | Dtype | Content |
|---|---|---|---|---|
| common/cache.pt | H | [160,192,2048] | float32 (BF16-exact) | pre-final-norm block outputs at the readout position |
| common/cache.pt | exit_logits | [160,4,49152] | float32 (BF16-exact) | per-loop exit logits, packed as [160,2048] |
| common/cache.pt | continuations | list[160] | str | greedy continuations; not scored |
| development/native.pt | virtual_states, physical_states, wrapper_states | [3,192,2048] | float32 | development items only |
| development/native.pt | native_logits, unembedded_logits | [3,49152] | float32 | unembedded as [1,S,2048] |
| development/native.pt | item_names | list[3] | str | carnival-ocean, amazon-language, mars-color |
| readouts/{arm}.npz | rank | [160,3,C] | int32 | own-slot ranks |
| readouts/{arm}.npz | allrank | [160,128,C] | int32 | min-alias rank for 80 names; slots 80–127 are -1 |
| readouts/{arm}.npz | top1 | [160,C] | int64 | argmax |
| readouts/{arm}.npz | kl_to_final, kl_to_local | [160,C] | float32 | both against exit 3; identical |
| readouts/{arm}.npz | rank_of_final_top1, rank_of_local_top1 | [160,C] | int32 | identical |
| readouts/{arm}.npz | top10_ids | [160,C,10] | int32 | first 10 entries of the descending argsort |
| readouts/{arm}_samples.pt | list of 3 dicts | — | — | see below |

Each samples dict holds:
- `item_index`: 0, 1 or 2.
- `virtual_indices`: 10, 8 or 9 entries.
- `transported`: [k,2048] float32, taken before the BF16 cast.
- `logits`: [k,49152] float32 holding BF16 values.
- `sorted_ids`: [k,49152] int32.

**Not retained.**
- Logits for non-sampled cells (`evaluate.py:565`; discarded at `readouts.py:35`).
- Transported states outside the samples.
- The J stack.
- Model weights.

**Replay feasibility.**
- A readout-only replay of all 160 items needs no model forward pass. It does need:
  - H, which is retained and BF16-exact;
  - the banks, which are present and were rehashed today;
  - `model.norm.weight` and `lm_head.weight` from model.safetensors (5,336,011,242 bytes, sha256 2fdf9805…).
- **Weights are not on this machine.**
  - The HF cache and R/resources/model_snapshot hold only config and tokenizer files.
  - The only multi-GB safetensors files found belong to ouro_rltt_local, a different model.
- Replay is therefore possible in principle but cannot be run locally now.
- A bitwise replay would also need the same GPU BF16 kernels and C-row layout. A CPU replay would be approximate (see the transport measurements in section 5).

## 7. Mismatches and dependence on batching or precision

No mismatch was found in the score definition. The findings below concern verification scope and numerical sensitivity.

**F1. The unembedding layout for control arms is unverified.** Status: confirmed gap.
- **Where.** Penultimate uses 190 rows and sampled_sum/diagonal use 191 (`worker.py:131, 143`; `evaluate.py:550`).
  - The development gate checks only [1,S,2048] (`readouts.py:72`; `worker.py:94-97`).
  - Retained data show [192] and [160] agree on exit rows.
  - No evidence exists for 190 or 191 rows.
- **Why it matters.** The diagnostic showed that single-row BF16 lm_head output changes 28–45 of 49,152 logits by up to 0.03125 (see the reanalysis in R/reviews).
- **Check.**
  - On the same GPU type, unembed the saved H and bank transports at C = 190/191 and at C = 192.
  - Compare the logits and allrank bitwise.
  - This needs a GPU and the model weights, and cannot be done from retained data.
- **Effect.**
  - No effect on the primary endpoint, where both arms use 192 rows.
  - Possible effect only on near-tie cells in the 10 secondary contrasts involving penultimate, sampled_sum or diagonal; direction unknown.

**F2. Receiver validation never recomputes logits.** Status: confirmed.
- **Where.** `validate_outputs.py:114-136`.
- **What it misses.** The BF16 cast, RMSNorm, lm_head and row layout are unverified for every arm.
- **Check.** Recompute norm∘lm_head on the sampled transported vectors and compare with the saved logits. This needs the weights.
- **Effect.** It adds none by itself; it leaves F1 and F4 undetected if present.

**F3. Hit@10 depends on token-ID order at BF16 ties.** Status: confirmed, small.
- **Where.** `vis.py:119`; `readouts.py:18`. The logits are BF16-resolution.
- **Tie frequency** (measured). Positions 9 and 10 tie in 42 of 168 sampled rows.
- **Ambiguous cells** (measured). 3 of 13,440 sampled name cells would change hit status under a different tie order:
  - raw, item 0, v143, "helium" (control): saved rank 10, possible 9–10.
  - fit02, item 0, v95, "Venus" (own): saved rank 10, possible 8–11.
  - fit02, item 2, v189, "Voyager" (control): saved rank 11, possible 9–11.
- None of the three is in fit01 or in the 169–180 band.
- **Near-boundary own cells in the band** (1,920 per arm):
  - fit01: 37 at rank 9–10, of which 22 are at rank 9.
  - raw: 18 at rank 9–10, of which 9 are at rank 9.
- **Adversarial envelope on the primary estimate (0.23188).** Flip every band cell (own and control, both arms) whose saved rank lies in the window, against or for fit01:

  | Window | Primary range |
  |---|---|
  | [9,10] | [0.2154, 0.2447] |
  | [8,11] | [0.1999, 0.2591] |
  | [6,13] | [0.1606, 0.2802] |

  These are worst-case bounds, not estimates.

**F4. BF16 rounding and kernel sensitivity of lens-arm transport.** Status: confirmed, small.
- **Where.** `hf.py:170` rounds the FP32 J·h to BF16. The raw arm's states are already BF16-exact.
- **Measured.**
  - 87% of FP32 transported elements differ bitwise between GPU and CPU.
  - After BF16 rounding, about 0.02–0.03% differ.
  - The effect of einsum batch size (190/191/192) was not measured (inference).
- **Effect.** It can only move hits in near-tie cells (see F3); it is by design and shared with the historical scorer.

**F5. The development gate's layout differs from the scorer's.** Status: confirmed.
- **Where.** `readouts.py:72` unembeds 3-D [1,S,2048]; the scorer unembeds 2-D [C,2048].
- **What bridges them.**
  - For 192-row arms, the link to native logits relies on the diagnostic's [160,2048] result (development items, another pod).
  - This run adds a measured [192] = [160] identity on exit rows.
- **Effect.** None is shown for 192-row arms; see F1 for the others.

**F6. Diagnostics outside the score.** Status: confirmed, no score impact.
- **Local references.**
  - kl_to_local and rank_of_local_top1 reference native exit 3 for every arm (`readouts.py:34`).
  - The historical control evaluator instead referenced the target-state logits, via `target_ut = n_ut` and appended references (`evaluate_controls.py:377-383`).
- **Exit packing.** Exit logits are packed as [160] rows here, against [148] historically.
- **Plan:19.** Its claim of unchanged geometry holds for the scoring arrays.

**F7. Precision switches were recorded only at model load.** Status: confirmed.
- **Where.** `evaluate_refits.py:631`; `worker.py:73-75`.
- **Why it is low risk.** The executed code contains no setter.

**F8. The leak check uses the readout context rather than the full prompt.** Status: confirmed, no effect.
- **Where.** `evaldata.py:96-103`; `worker.py:115`.
- **Measured.** Retokenization shows no alias in the dropped trailing space.

## Checks run

- Read every executed scoring source listed above, the plan, `run_spec` and both checkers.
- Rehashed the bundle sources, accepted stage and final files, banks and model source files.
- Loaded the retained arrays on CPU to confirm shapes and dtypes.
- Checked that H and exit_logits are BF16-exact.
- Compared column 191 of fit01 and fit02 with raw (all arrays, all items).
- Compared 192-row raw logits with 160-row exit logits (samples, and KL = 0 at column 191 for all items).
- Analyzed tie order and tie ambiguity on all 168 full-sort samples.
- Compared FP64 and CPU FP32 transports of the sampled vectors, including BF16 rounding.
- Computed the near-boundary envelope on the primary band.
- Retokenized the prompts for the leak check.
- Searched the filesystem for model weights.
