# Exact-estimator optimization — 7 September 2026

The completed local benchmark improved from **210.69 seconds to 192.30 seconds per paragraph**, an **8.7% runtime reduction** (1.096× throughput). This uses the compressed native batch-8 fitter. Keeping the original batch size of two gives **202.80 seconds**, a **3.7% reduction**, with every one of the 801,112,064 FP32 entries numerically equal to the saved stock reference. These are measured complete paragraphs, not projected full-width timings.

The larger improvement is memory. A full-depth replicated batch of 32 now executes in **11.12 GB** on the 12 GB laptop GPU. The historical uncompressed B32 benchmark reported approximately 77 GB on B300; that is a cross-device comparison, not a same-device memory measurement. On the same local device at B2, compression reduced the measured peak from 9.77 GB to 7.70 GB. Larger batches did not keep getting faster: B8 was the best compressed configuration in the local sweep.

No paid GPU was rented and no scientific calibration lens was fitted. The original application and historical research artifacts remain unchanged. The five independent N100 fits, target/position controls and conditional Huginn work are still pending.

| Complete fixed-paragraph run | Seconds | Peak allocated GPU GB | Evidence |
| --- | ---: | ---: | --- |
| Original stock B2 | 215.034 | 9.962 | [Initial complete reference](baseline_full.json) |
| Stock B2, repeated after sustained GPU work | 210.690 | 9.962 | [Warm reference](stock_full_b2_warm.json); all reference bits reproduced |
| CUDA replay and gradient edges, B2 | 202.804 | 9.769 | [Complete comparison](optimized_full_b2.json); all entries numerically equal |
| CUDA replay, gradient edges and exact saved-tensor compression, B8 | 192.301 | 8.360 | [Complete run](optimized_full_b8.json); all entries finite, separate matching-B8 reference checks |

All runs use the pinned Ouro checkpoint, BF16 weights, four loops/192 virtual decoder events, target 191, sources 0–190, width 2048, the same T128 cost paragraph, valid positions 16–126, FP32 source means and FP32 CPU matrices. Timers include the forward, graph preparation, all derivative directions, source reduction and host assignment. They exclude loading the checkpoint, scanning completed matrices for finiteness and saving. The B8 save took 2.295 seconds. The GPU reached 87°C during sustained work; the adjacent warm stock run helps avoid crediting operating-temperature differences to the optimizer. These are individual benchmark runs, not a statistical speed estimate.

The implementation retains ordinary replicated-primal `autograd.grad`. It records public gradient edges instead of retaining all source values, verifies duplicate saved-tensor lanes byte for byte, reconstructs their exact original layouts when needed, captures backward plus the original FP32 source-position means, and transfers one stacked row bank per direction batch. CPU outputs are fully assigned, so initializing a 3.2 GB bank to zero is unnecessary. Partial direction batches use the upstream slice shape. Graphs and hooks are released on completion or failure. Compression is useful only when its memory savings justify the additional reconstruction copies.

The final implementation also reduces discarded warmups from three to one. A [cold-process B8 check](warmup_one_check_v2.json) verified that no backward preceded construction, then checked first/middle/last, repeated and zero cotangents against eager execution and the saved full result using literal FP32 bits. The complete timings above include the earlier three warmups, so this final small saving is not counted in the reported speedup.

Changing B can change Ouro's BF16 results substantially even in the original estimator. The same-fixture B8-versus-B2 diagnostic differs by 4.22% in relative Frobenius norm over the first two rows at all sources. Therefore the B8 speed comparison is a throughput comparison, not a claim that a B8 Jacobian equals B2. The completed B8 result was checked against the native **B8** calculation. The independent reference stores complete saved tensors on CPU, preserving strides/offsets and keeping weights on GPU; it first passes a mandatory literal-bit B2 gate. All 9,388,032 compared values match literal FP32 bits across all 191 sources in three eight-row groups at the beginning, middle and end, plus repeated/zero cotangent checks. See [reference evidence](offload_reference_b8_v2.json) and [final independent verification](FINAL_VERIFICATION.json). This is 24 distinct output rows, not a complete uncompressed B8 reference.

The full B2 comparison used `torch.equal`: its JSON's older `bitwise_equal` label means numerical equality, including equality of signed zeros. Subsequent reference and sample comparisons use integer views to check literal FP32 bits. The integrated CPU checks cover 192 complete cases with changing lengths, targets, batches, FP32/FP64/BF16, compression and gradient-edge options; 48 cases actually compress saved tensors. A separate independent check covers tied recurrent blocks, the actual Ouro loop-tap mechanism and loop-end normalization. See [integrated checks](integrated_final_cpu.json) and [recurrent-edge checks](edge_recorder_cpu.json).

Single-primal vmap, compact batched perturbations, a math-attention replacement and Inductor fusion were investigated and not adopted. Vmap changed matching-primal sampled derivatives by 6.6% globally; changing native primal/backward batch 1 to 2 changed them by 151.7%. The Inductor suffix changed values even with BF16 casts preserved and FMA fusion disabled. A numerically exact FX/eager compiled backward was slower. These failures remain in [the contract and audit](CONTRACT.md), the benchmark files and [the chronological log](../RUN_LOG.md). They do not establish that a scientific J-Lens finding changed.

The optimized engine is available explicitly through [the refit wrapper](../fit_estimators.py): `dense_engine="cuda_graph", compress_saved_tensors=True`, with a fixed, recorded `dim_batch`. The existing nonfinite-output guard is preserved. The released library default is unchanged. The CPU implementation and sampled-position controls continue to work without CUDA. All five future independent fits must use the same recorded numerical configuration; choosing a batch or backend from favorable recovery scores would confound the replication.

To reproduce the complete optimized B8 benchmark from the repository, using the existing model environment:

```bash
PYTHONDONTWRITEBYTECODE=1 HF_MODULES_CACHE=/tmp/jlens-refit-hf-modules \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
/home/moloch/ouro_project/venv/bin/python \
research/refit_round_2026-09-07/optimization/benchmark_optimized.py \
--engine cuda_graph --batch 8 --compress \
--output /tmp/optimized_b8_repeat.json
```

The memory result removes the main capacity reason for preferring a 48 GB rental. It makes the quoted 24 GB RTX 4090 a serious first benchmark candidate again. It does not establish cloud throughput or a new dollar cost: the card should be chosen by complete-prompt seconds multiplied by the actual hourly quote. CPU offload is used for verification, not recommended as the production fast path.
