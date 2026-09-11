# Review: native_check_v1 correction and native1 checker pin

Reviewer: independent first review (read-only; CPU only; no network, no GPU).
Round root R = /home/moloch/jacobian-lens/research/confirmation_2026-09-09.

## Verdicts

| Claim | Verdict |
|---|---|
| 1. native_check_v1 changed only the development-check recomputation plus run identity/provenance | **PASS_WITH_FINDINGS** |
| 2. primary_numerical_check_native1.py differs only by accepting the corrected run-spec pin | **PASS_WITH_FINDINGS** |

No blocking finding. The scientific content of both claims holds under every check below. The findings concern scope statements that are slightly too broad, reduced auditability, and missing verification records for the modified checker.

## What was checked, with primary evidence

### Integrity of the three freezes
- **Freeze records.** Every entry in `R/FREEZE.json`, `corrections/precision_v1/FREEZE.json` and `corrections/native_check_v1/FREEZE.json` matches its bundle on disk and its `bundle.tar.gz` members. The counts are 57, 58 and 59 files, with no extra or missing files.
- **Supporting records.** The builder records, run-config records and parent-freeze chain all match.
- **Hash cross-references.** The native_check_v1 FREEZE sha256 is `fc74b0c2…`. It equals `run/USER_APPROVAL.json:7` and RUN_LOG.md:207.

### Byte diff of every file across the three bundles (sha256 of all files)

| File | orig→precision_v1 | precision_v1→native_check_v1 |
|---|---|---|
| evaluation/readouts.py | same | **changed** (`native_development` only) |
| evaluation/worker.py | changed (TF32 setters removed) | same |
| frozen/implementation_review.json | changed | same |
| frozen/precision_correction.json | added | same |
| frozen/native_check_correction.json | – | added |
| frozen/run_spec.json | changed | changed |
| frozen/output_contract.json | changed (run_id, run_spec_sha256) | changed (run_id, run_spec_sha256) |
| all other 50 files | same | same |

Byte-identical across all three bundles:
- benchmark.json, population.json, annotation_review.json and PROSPECTIVE_PLAN.md;
- measurement.py, analyze.py, plot.py and validate_outputs.py;
- the ouro_jlens, jlens and legacy code, and the controller.

**Structured diff of the run spec**, precision_v1 → native_check_v1:
- `run_id` and `frozen_utc` changed;
- `source_records['evaluation/readouts.py']` changed;
- `native_check_correction_record` was added.

**Fields identical from the original to native_check_v1:** model_identity, model_record, original_precision, bank_records, benchmark_record, population_record, plan_record, annotation_review_record, planned_items, planned_concepts, development_item_names, primary, secondary_specs, uncertainty, arm_support, dependency_groups and schema.

**Correction record.** Every record in `native_check_correction.json` matches its file on disk:
- diagnostic receipt, comparisons and reanalysis;
- parent freeze;
- original and corrected readouts.

**Readers of the changed fields.** These fields are read only by identity and provenance gates: `validate_outputs.py:40,60`, `worker.py:39,70` and `bootstrap.py:21`. No scoring or analysis code reads them.

### The new `native_development`
The diff is the only change in `evaluation/readouts.py`. Line 72 now computes `model.unembed(rec.activations[191])[0,-1]`. It replaces `model.unembed(virtual[-1].to(device))`.

**Is `rec.activations[191]` the right state?** It is the output of physical layer 47 when `current_ut == 3`:
- `recurrent.py:66-78,88-90` sets this up, with virtual index = ut*48+layer;
- in Ouro, that tensor is exactly the input to `self.norm` at the end of UT step 3 (`modeling_ouro.py:592-607`);
- the norm is out-of-place (`modeling_ouro.py:357-362`), and no layer runs after it, so the captured tensor is not mutated.

**The native head.** `config.json:81` sets `early_exit_threshold: 1.0`. The HF forward therefore takes the gather path (`modeling_ouro.py:780-800`): per-position exit steps, then `lm_head` on a `[1,S,2048]` tensor. For the three development items the reanalysis (Rule 3) found the last-token exit is step 3.

**`model.unembed`** is `lm_head(norm(x))`, with `.to()` a no-op and no softcap (`hf.py:166-174`; `final_logit_softcapping` is absent from the config). It uses the same `model.model.norm` and `lm_head` modules as the native path.

**The comparison is not trivial.** It compares `output.logits[0,-1]` from the native gather-exit head against a separate `norm`+`lm_head` call on the captured tensor. A wrong tap (index or UT step) or a different exit step would change the logits and fail `torch.equal`.

**Exact equality is retained.** `worker.py:94-97` and `validate_outputs.py:72-73` are byte-identical to both parents.

**Tensor evidence** (CPU load of the saved `development/native.pt` from attempt05 (precision_v1, failed) and from attempt_06 (native1, accepted)):
- `virtual_states`, `physical_states`, `wrapper_states` and `native_logits`: attempt05 and attempt_06 are identical in 0 differing elements.
- `unembedded_logits`: they differ in 112 elements (39/28/45 per item; max |Δ| 0.03125). These are exactly attempt05's failing counts.
- attempt_06 `unembedded_logits` equals attempt05 `native_logits` bitwise.
- attempt_06 passes all three equalities. attempt05 fails only native==unembedded.
- The saved `virtual_states[:,191]` round-trips through BF16 exactly, so the old single-row input was bitwise the same state.

Conclusion: the correction changed only the kernel shape of the recomputation. The states and native logits are unchanged, and the check still tests the same state.

### Executed payload vs the native_check_v1 freeze
- **Run spec.** `results/run_spec.json` is byte-identical to the frozen run spec (`002b91cb…`).
- **Population and benchmark.** `population.json` and `benchmark.json` are byte-identical to all three bundles.
- **Provenance.**
  - `source_records`, `model_record` and `bank_records` equal the spec;
  - `runtime.precision` equals `original_precision`;
  - `runtime.model` equals `model_identity`;
  - `imported_sources` match the spec;
  - `binding.controller_sources` match the FREEZE controller files;
  - `binding.run_spec_sha256` is `002b91cb…` and `output_contract_sha256` is `10cfeacc…`.
- **Lease and receipt.** `LEASE.json` binds the native_check_v1 bundle (`92db22f8…`), run_config (`c957c74a…`), run spec and semantic verifier (`588dbf72…`). The final MANIFEST file records (18) all verify.

### The checker
- **Diff.** The diff between `reviews/primary_numerical_check.py` and `…_native1.py` is exactly two lines: it adds a FROZEN pin for `corrections/native_check_v1/bundle/frozen/run_spec.json` (line 33), and selects that pin for `results.run_spec` (line 365). All numerical, aggregation, bootstrap, weighting and self-test code is byte-identical.
- **Run-spec fields read:** `primary.virtual_indices` (366), `uncertainty` (368), `secondary_specs` (369), `arm_support` (377), `dependency_groups` (384) and `run_id` (388). All are identical between the original and corrected spec except `run_id`.
- **Pinned root files.** `analysis/analyze.py` and `evaluation/measurement.py` equal the native_check_v1 bundle copies.
- **Reproduction runs.** The scripts were run unmodified from `reviews/` against scratch copies of the payload, with output in the scratchpad:

  | Run | Result |
  |---|---|
  | native1, exact payload | passed, 231 comparisons, max 8.88e-16 |
  | native1, run_spec with `uncertainty.seed+1` | **failed** at `results.run_spec.frozen_record` |
  | native1, original run_spec substituted | **failed** at `results.run_spec.frozen_record` |
  | frozen checker, original run_spec substituted | passed, 230 comparisons; the non-pin rows equal native1's, and the reconstruction differs only in `run_id` |
  | frozen checker, exact payload | failed at the pin (reproduces the recorded failure) |
  | native1 `--self-test` | passed, 11 cases |

  So the pin change admits no altered run spec, and population and benchmark remain pinned to the original bytes. The numerical verdict does not depend on any changed field.

## Findings

### Claim 1

**F1-1 (minor; confirmed). Run budgets changed outside the stated scope.**
- **Location:**
  - `corrections/native_check_v1/build_correction.py:1`, `:17`, `:53`, `:75`;
  - `corrections/native_check_v1/run_config.json:2,4,16`.
- **Evidence:**
  - Relative to precision_v1, the run config changes setup 12300→12800 s, compute 3600→1200 s and preservation 1800→900 s.
  - It is bound by FREEZE `controller_run_config` and by `LEASE.json`.
  - The builder docstring says "nothing else changes", and the correction record scope says every other gate and rule is unchanged. Neither mentions budgets. REPORT.md:60 and RUN_LOG.md:207 do disclose them.
  - Budgets feed only lease deadlines and cost bounds (`controller/lease.py:268-282`), not scoring.
- **Correction:** state the controller-budget change in the claim, the builder docstring and the correction record scope, as a non-scientific change.

**F1-2 (minor; confirmed). Relative to the original freeze, the scoring orchestrator is not byte-unchanged.**
- **Location:** `corrections/precision_v1/bundle/evaluation/worker.py:59-60`, and the run spec's `implementation_review_record` and `precision_correction_record`.
- **Evidence:**
  - `worker.py` (which runs caching, exit unembedding and `readout`) differs from the original by removing the two TF32 setters. This is inherited from precision_v1, not introduced by native_check_v1.
  - Its effect is bounded by `worker.py:74-75`. The payload's `runtime.precision` equals `original_precision`.
- **Correction:** qualify "scoring and readout code unchanged relative to the original freeze" with "except the separately frozen precision_v1 `worker.py` change".

**F1-3 (minor; suspected). "As the native head does" is exact for shape, not necessarily for the other rows' values.**
- **Location:**
  - `corrections/native_check_v1/bundle/evaluation/readouts.py:70-71`;
  - `native_check_correction.json` `scope`;
  - `modeling_ouro.py:780-800`;
  - `config.json:81`.
- **Evidence:**
  - With `early_exit_threshold=1.0` the native head unembeds a per-position gathered exit tensor.
  - The recomputation unembeds the step-3 sequence for all positions.
  - Exit step 3 was established only for the last token of the three items.
  - If bf16 gate saturation makes earlier positions exit earlier, the other rows differ in value. The `[0,-1]` comparison is unaffected, because the kernel choice depends on shape.
- **Settle by:** a GPU run recording `exit_steps` for all positions of the development items. The weights were purged locally.
- **Correction:** reword to "unembeds the step-3 final-norm input at the native head's input shape `[1,S,2048]`".

**F1-4 (minor; confirmed). The saved evidence no longer contains the recomputation's input.**
- **Location:** `readouts.py:69,72,84-86`.
- **Evidence:** only the last-token row (`virtual_states[:,191]`) is saved. The corrected check needs the full `[1,S,2048]` step-3 sequence, which is not saved. Before the correction, the recomputation input was exactly the saved row.
- **Correction:** document this reduced auditability. Future versions could save the final-step sequence with `native.pt`.

**F1-5 (minor; confirmed). The retained gate certifies a narrower equivalence.**
- **Location:** `readouts.py:72`; `readouts.py:17`; `worker.py:127`; `reviews/diagnostic_native_logit_v1_reanalysis.md:63-66`.
- **Evidence:**
  - Had the original gate passed, it would also have certified that single-row BF16 `model.unembed` reproduces native logits.
  - The corrected gate certifies only the shape-matched composite.
  - The scorer calls `model.unembed` at other shapes (`[columns,2048]` and `[160,2048]`), and `[190–192,2048]` was not measured.
  - The frozen plan (PROSPECTIVE_PLAN.md:55) does not require scorer-native bit equality, so this is not a scope violation.
- **Correction:** state explicitly in the correction record that the gate validates state extraction and the native exit, not scorer-shape unembedding equality.

**F1-6 (minor; confirmed). No independent review before execution.**
- **Location:** REPORT.md:60; RUN_LOG.md:207.
- **Evidence:** the change ran on the GPU with only a self-reported CPU functional test. This review is the first.
- **Correction:** bind this review to the freeze record (`fc74b0c2…`) as a post-run review, not a pre-run gate.

### Claim 2

**F2-1 (should-fix; confirmed). The modified checker has no bound self-test proof or review, and was created after outcomes were visible.**
- **Location:**
  - `reviews/primary_numerical_check_native1.py` (sha256 `f24372fa…`);
  - `reviews/primary_numerical_check_proof.json`;
  - `reviews/primary_numerical_check_root_review.json`.
- **Evidence:**
  - The self-test proof and the root review "passed_for_actual_result_use" both bind only the frozen code record `89a7309a…`.
  - `f24372fa…` appears only in its own output record.
  - The file mtime is 01:09:37.419Z, 0.1 s before the passing record (01:09:37.536Z) and after the failing run (01:07:07Z) and the analysis output.
  - REPORT.md:54 cites the native1 record as the independent reconstruction.
  - The self-test does pass: I ran it and got 11 cases.
- **Correction:** publish a self-test proof and a review record bound to `f24372fa…`. State in REPORT.md that the pin was added after the new-item outcomes were computed.

**F2-2 (minor; confirmed). The claim's field list is literally false for `run_id`.**
- **Location:**
  - `reviews/primary_numerical_check_native1.py:388`;
  - REPORT.md:54 ("every run-spec field the checker uses is identical");
  - RUN_LOG.md:212.
- **Evidence:** `spec['run_id']` is read and differs (`ouro_confirmation_20260909_fixed160` vs `ouro_confirmation_20260911_fixed160_native1`). It is only echoed into `reconstruction.run_id`: the frozen checker on the original spec differs from native1 only in that key.
- **Correction:** reword to "every scientific run-spec field it reads; `run_id` is read only as an output label".

**F2-3 (minor; confirmed). The checker does not encode why the new pin is safe.**
- **Location:** `reviews/primary_numerical_check_native1.py:31-39,358-365`.
- **Evidence:**
  - Safety rests on a single hand-added hash.
  - The original-spec pin (line 32) is still hashed but compared to nothing in the results, so it is vestigial.
  - Nothing ties the corrected spec to the original, whether field-level or through the native_check_v1/precision_v1 FREEZE parent chain.
  - It is correct for this run (I verified the fields), but the checker would equally accept a pinned spec that changed science.
- **Correction:** assert that the results spec equals the original spec on all keys except {run_id, frozen_utc, source_records, implementation_review_record, precision_correction_record, native_check_correction_record}. Alternatively, pin `corrections/native_check_v1/FREEZE.json` and verify its parent chain.

**F2-4 (minor; confirmed; pre-existing, not introduced by the diff). The reconstruction's array hashes are not bit-reproducible.**
- **Location:**
  - `reviews/primary_numerical_check_native1.py:275-277,402-404`;
  - `independent_numerical_check_native1.json` `reconstruction.reconstructed_array_records`.
- **Evidence:**
  - My rerun passed with the same 231 checks.
  - `family_bootstrap` sha256 was `3b156c37…` against the recorded `9c260f5f…`.
  - `primary_item_resampling_95_interval[0]` differs in the 17th significant digit (0.17971271756329119 vs 0.17971271756329113).
  - The cause is BLAS matmul variation; it is within the 1e-12 tolerance.
- **Correction:** do not use `reconstructed_array_records` hashes as bit-exact fingerprints. Compare within tolerance.

## Not checked
- GPU re-execution, and recomputing native or unembedded logits from weights (the weights were purged).
- Whether non-last positions exit before step 3 (F1-3).
- The analysis producer (`analyze.py`) run itself, beyond the independent reconstruction.
- Spending and termination records.
