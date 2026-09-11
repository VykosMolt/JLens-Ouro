# Huginn N100 verification rerun feasibility

**A complete rerun is financially plausible within a separately admitted $6 job, after primary completion and external acceptance.** At the stipulated all-in ceiling of $0.70644/hour, the proposed 8 h 16 min envelope plus the existing controller's $0.15 billing-latency margin costs $5.989904. Spending the primary's full $4 planning cap would leave $7.148 of the combined authorization; spending another $6 leaves $1.148 unallocated. Admission must use the actual final primary debit and available account balance.

This is a design assessment. No Huginn weights were loaded, no GPU was initialized, and no model outcomes, provider requests or paid work were produced. The only executable preparation was the original `run_huginn.py --plan`, which passed the complete retained historical Ouro/control prerequisites and reported `model_loaded=false`, `gpu_initialized=false`. The output path `/planned/huginn_verification` was not created. No frozen Ouro/controller source was changed.

## Recorded runtime and proposed headroom

The evidence concerns an RTX 5090, matching the controller's GPU class. All 100 retained diagnostic rows preserve width 5280, 32 sources, target 33, B8 and 660 derivative passes. The 100 compute-only durations sum to 12,075.023 seconds; checkpoint accumulation/write/hash/pruning occurs outside those timings. The completed fit's observed per-paragraph bound is 130.483836 seconds, so100 times that bound is3.624551 hours. The former20.49-hour arithmetic proposal predates these actual measurements and is not the relevant runtime basis.

The full engineering preflight measured1,023.660 seconds for the native reference,119.176 seconds for the optimized paragraph and8.608 seconds for checkpoint I/O. All892,108,800 derivative values matched bitwise. Native and optimized CUDA peaks were12.368 and 12.656 GB; native offload peaked at4.474 GB of host payload. The unchanged preflight requires at least48 GB of effective host/cgroup headroom. Both historical 148-item readout seeds took18.860 and 18.263 seconds inside their item loops; model loading, validation, hashing and sealing are additional work.

| Stage | Planning allowance | Basis |
| --- | ---: | --- |
| Runtime installation and complete snapshots | 60 min | Allowance; no authenticated complete setup elapsed time was established |
| Native gate and external tensor acceptance | 60 min | About19 min measured derivatives, plus retained evidence transfer/load/comparison |
| Complete100-paragraph fit and checkpointing | 273 min |4.55 h, slightly above125% of100×130.484 seconds |
| Both original readout seeds, model load and sealing | 15 min |37.123 seconds measured item loops, with operational headroom |
| Final retrieval and independent verification | 78 min | About5.641 GB;47 min at2 MB/s leaves time for hashes, loading and numerical checks |
| Provider termination reserve | 10 min | Existing controller constant |
| **Total** | **496 min** | **$5.839904 at the rate ceiling, plus$0.15=$5.989904** |

Using the historical fit bound without its25% margin, with the same other allowances, gives about$5.336. Both values are conditional planning calculations. No complete historical estimator-transfer throughput was established. The final design must measure actual setup, gate and transfer performance, then admit all 100 paragraphs only if their conservative remaining duration, both readouts and all preservation/termination reserves fit. The conservative formula is `100 × 1.25 × max(new complete preflight paragraph+checkpoint time, 130.483836 seconds)`, plus900 seconds for readout work, with preservation and termination reserved separately. No partial calibration size replacesN100.

## Exact inputs and initialization

Reuse `tomg-group-umd/huginn-0125` revision `bb6621b65e90b6a4b9b29ef88dc83866d450470c`, all 16 manifest files totaling 15,651,519,639 bytes, and the original BF16/runtime switches. The original evaluator also checks complete membership of the Ouro snapshot while rebuilding tokenizer eligibility. Supply its authentic 5,341,718,984-byte snapshot as well: about20.997 GB of combined model inputs. The120 GB controller disk allocation has room for these inputs and the proposed retained tensors, subject to an actual free-space check during setup.

Reuse the 100 paragraphs in `audit/calibration_fit_01.json`, their order and exact bytes; `huginn_calibration.json`; the original run specification, combined contract, environment, lockfile, model manifests and eligibility document. Keep R8, four core blocks per recurrence, two prelude/two coda blocks, width 5280, learned sources0–31, target 33, T128, skip16, B8 and the original dense CUDA-graph estimator with its existing lossless saved-tensor compression. Within each paragraph the estimator is `(1/|V|) Σq∈V Σp∈V G(q,p)`, with `V=16..T−2`. Accumulate the 100 paragraph matrices in FP32 and divide by100 once before FP16 conversion.

Calibration paragraph index `i` uses base seed `2026090802+i`, namespace `calibration`, and the existing canonical-JSON hash recipe over version, base seed, namespace and actual input IDs. Take the first eight SHA256 bytes big-endian modulo 2^63. Preserve the native discarded `randn` draw followed by truncated-normal initialization and native scaling, in the original dtype/device. Draw one prompt state and clone identical states into every derivative lane. Test-time noise remains zero. Check every actual token hash, length, valid-position count and initialization record against all 100 frozen rows.

The engineering paragraph remains `ouro_jlens.bench.TEXT`, namespace `benchmark`, base seed 2026090899, with historical derived seed8544906854076094770. Evaluation uses the original two base seeds 2026090803 and2026090804, namespace `evaluation`, and all148 old items. The same state trajectory supports raw, J-Lens and trained-coda readouts; coda receives the full sequence. Only cells3,7,…,31 are native recurrence exits. Retain the original joint-tokenizer eligibility, aliases, control weights and rank convention.

## Reuse and the remaining admission work

The unchanged `run_huginn.py --plan` successfully checked the historical five-fit main and complete-control evaluation directories at `monitoring/attempt_06/ouro_handoff_20260909T110503Z/results`. These validators consume the preserved evaluation evidence and do not require missing fit03–05 estimator binaries. Their historical prerequisite gate must be supplemented by the exact externally accepted **new primary** final receipt and reconciled primary termination before any Huginn launch.

Keep the original package layout `bundle/repo/research/refit_round_2026-09-07/deployment`, with all 42 source records from the old run specification and the corresponding `bundle/ouro_project/src` tree. Copy all five small calibration JSON files required by the original `_contract` validator, even though only fit01 is computed. The flat `bundle/legacy` arrangement used by the Ouro confirmation wrapper does not preserve Huginn's relative source-contract resolution. Retain the complete old Ouro/control evaluation prerequisite trees and their exact owners/seals.

The paired JSON supplies complete, unexecuted argument vectors for these existing entry points:

1. `preflight_gpu.py --kind huginn`: original native primal and complete-width derivative parity, under a bounded process timeout.
2. `run_huginn.py`: original complete N100 run, explicit old prerequisite directories, fresh output root and frozen work deadline. A900-second fitting reserve protects the readout stage.
3. `evaluate_huginn.py`: original fit directory, both authentic snapshots and original eligibility; both fixed seeds, with the normal 600-second reserve.
4. `evaluate_huginn.py --validate-output`: completed readout validation before external acceptance.

There are two concrete pieces to freeze separately. First, the current immutable controller/debit enforces `initial_ouro_job_incremental_cap_usd=4`, so it correctly rejects a $6 job. A reviewed Huginn phase ceiling and fully reconciled ledger/debit mechanism are required; do not overwrite or relax the existing Ouro record. Second, the old preflight compares its tensors and then discards them, retaining hashes/assertions. A narrowly source-bound preservation wrapper must retain actual gate tensors if receiver-side tensor acceptance is required. Neither change is implemented or admitted by this document.

## Verification and preservation contract

Preserve native/adapter primal states, coupled initial states and logits, plus the complete native and optimized engineering matrix banks. The two matrix banks contain about 7.137 GB, and retaining both full primal paths and initial state adds about 1.283 GB. An approximately8.420 GB gate handoff needs about 28 min at5 MB/s, making the 60-minute gate allowance plausible alongside the measured 19-minute compute. Require actual measured throughput and external hash/load/bitwise comparisons before admitting the 100-paragraph fit. Keep the benchmark separate from calibration accumulation.

After N100, immediately publish the sealed final FP32 checkpoint, final FP16 bank, owner, pointers, seals and all 100 diagnostic rows as a distinct stage. Historical serialized sizes are 3,568,444,419 bytes for the checkpoint and 1,784,226,499 bytes for the bank. The complete old readout handoff was 288,099,459 bytes. These motivate the approximately5.641 GB final transfer reserve; new serialization metadata may differ slightly.

Outside the worker, hash every exact manifest member, load both tensors with CPU `weights_only`/mmap, validate complete shapes/source order/finite values/N100/prefix/seed diagnostics, and independently compare **all 892,108,800** FP16 entries against NumPy FP32 division by100 followed by FP16 conversion and literal uint16-bit comparison. Do not accept only a producer assertion. Preserve both original readout seed stages and require final external semantic acceptance before normal provider deletion. A stopped partial fit remains explicitly incomplete.

For reproducibility, compare the new final bank's hash with the retained historical final-bank hash `7eddc849bca857a1406a901d8e4998b6bf90580b09f6ac6b326690ef378dc596`. Compare old/new initialization records, retained states/native logits and raw/J/coda ranks/top1 for both seeds, reporting every discrepancy. Whole-checkpoint bytes are expected to carry a different fit-identity hash because the runtime identity changes; this is not a matrix reproducibility test. The original runtime did not enable deterministic algorithms, so identical scientific inputs do not guarantee identical final bits on a new machine.

A complete accepted new chain supplies an independently verified N100 rerun. It remains a distinct record. If the bank hash differs, the absent old matrix values cannot be compared directly, and the old retrieval limitation remains. Do not tune, repeat until favorable, or change seeds, architectures, depths, calibration count, bands or the original historical comparison contract in response.

All source and evidence records used here, measured values, budget arithmetic, staging requirements and command templates are retained in `huginn_rerun_feasibility.json`.
