# Native-logit diagnostic v1: independent CPU reanalysis

**Verdict: PASS.** Every recorded comparison reproduces from the accepted tensors. The verdict covers that post-run criterion and tasks 1–5. It does not cover termination, the final account observation or spend, which were not checked.

Scripts:
- `reanalyze.py` and `retokenize.py`, in `/tmp/claude-1000/-home-moloch-jacobian-lens/18b73cc8-31a8-4b0b-b8c9-be0441298f55/scratchpad/reanalysis/`;
- CPU-only, with no bundle code imported.

Structured record: [diagnostic_native_logit_v1_reanalysis.json](diagnostic_native_logit_v1_reanalysis.json).

## Verification

- **Hashes.**
  - All five payload files match MANIFEST.json and RECEIPT.json in bytes and SHA-256, with no extra files.
  - The manifest's canonical hash equals the directory name and the receipt.
  - The receipt's verifier record equals the diagnostic's `validate_outputs.py`.
  - `run_spec.json` is byte-identical to the frozen bundle copy.
- **Precision and GPU.**
  - `provenance.runtime.precision` equals `run_spec.original_precision`.
  - The GPU is an NVIDIA GeForce RTX 5090 (uuid 05552ecd…, capability 12.0).
  - The admission plan names the COMMUNITY cloud.
- **Comparisons.**
  - My own code recomputed all 255 records, in the expected order, plus the `exit_steps` block.
  - It checked `dtype`, `shape`, `torch_equal`, `differing_elements`, `max_abs_diff`, `mean_abs_diff`, `top1_equal` and `top10_set_equal`.
  - There were 0 disagreements. No record has `torch_equal` true with differing bits.
  - Negative control: my counter detects a single flipped bit, and −0.0 against +0.0.
- **Rule 6.**
  - The tokenizer and data files match their recorded hashes.
  - Retokenization gives S = 20, 15 and 13, with token IDs equal to the saved IDs.
  - `virtual191` equals `norm_input[3,0,-1]` in 0 differing bits for every item, in both M2 and M4.
- **Cross-machine.**
  - This pod's `development/native.pt` is byte-identical to attempt05's (sha256 aa6a79bc…; pods mq25wyg0nf9z36 and i7zq14rj0f0ief, with different GPU UUIDs).
  - All five M1 tensors differ in 0 elements, so M1 is identical across machines.

## Rule conclusions

| Rule | Conclusion | Evidence |
|---|---|---|
| 1 | `differing_elements` decides. | No record has `torch_equal` and the bit count in conflict. |
| 2 | **Attempt05's failure recurred. M3 localizes it.** | M1 native vs unembedded: 39/28/45 differing. M2 logits[0,−1] vs M1 native: 0/0/0. M2 virtual191 vs M1: 0/0/0. |
| 3 | **Exit step identified: step 3 for all items.** | Only step 3 matches the lm_head input row, in both M2 and M4, and it is the step used. |
| 4 | **Norm: not shown to be shape-sensitive. lm_head: shape-sensitive.** | All nine norm shapes are exact. For lm_head, `[1,S,2048]` is exact while `[2048]`, `[1,2048]` and `[1,1,2048]` are not. |
| 5 | **M2 and M4 are bit-exact. Exact equality holds for `lm_head∘norm` at `[1,S,2048]`, `[160,2048]/first` and `[160,2048]/last`.** | These composites differ in 0 elements for all items in both repeats, and M4 vs M2 is 0 on all six tensors. `[1,1,2048]` and `unembed_fp32_virtual191` fail (39/28/45). |
| 7 | Magnitudes are descriptive. | M1 max \|Δ\| is 0.03125 for each item. Top-1 and top-10 agree. Scope is RTX 5090, precision_v1, the three items and the measured shapes. |

M3 differing elements against native, for items carnival-ocean / amazon-language / mars-color. Repeat #2 is identical to #1, and #1~#2 is 0 everywhere.

| Recomputation | Differing |
|---|---|
| norm, all nine shapes | 0 / 0 / 0 |
| lm_head `[2048]`, `[1,2048]`, `[1,1,2048]` | 39 / 28 / 45 |
| lm_head `[S,2048]`, `[1,S,2048]`, `[148,2048]` first/last, `[160,2048]` first/last | 0 / 0 / 0 |
| lm_head∘norm `[1,1,2048]` | 39 / 28 / 45 |
| lm_head∘norm `[1,S,2048]`, `[160,2048]` first/last | 0 / 0 / 0 |
| unembed_fp32_virtual191 (the attempt05 path) | 39 / 28 / 45 |

**How M3 pins the mismatch on lm_head.**
- For every item, the single-row lm_head output, `lm_head∘norm[1,1,2048]` and `unembed_fp32_virtual191` are bitwise identical.
- Their differing positions equal M1's.
- The M3 unembed reproduces M1's unembedded logits in 0 elements.
- The norm contributes no difference at any measured shape.

**Scorer shapes.**
- `[160,2048]`: the norm, lm_head and composite are all bit-exact, with the target row first or last.
- `[148,2048]`: the norm and lm_head are each bit-exact on their own. The composite at `[148,2048]` was not measured, and no rule decides it.
- Fill rows are the item's own positions, cycled. Other fill content and the `[190–192,2048]` readouts were not measured, and no rule decides them.

## Discrepancies

None.

Observation, not a discrepancy:
- The native.pt bytes are identical to attempt05's.
- The bundle and its archive contain no `.pt`, `.pth` or `.safetensors` file, and the worker source record matches.
- The tensors alone cannot distinguish a deterministic recomputation from a copy.

## Checks not performed

- Termination, absence confirmations, the final zero-pod and zero-spend observation, and spend within cap.
- Invariant 5 before/after frozen-file hashes. Controller hashes beyond the binding records.
- GPU re-execution. Tensors were compared as saved, and not regenerated from weights on CPU.
- Independent proof of GPU origin beyond the provenance record.
- The bundle's `validate_outputs` verifier, not used by instruction.
