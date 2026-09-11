Optimization contract, proposed v1, 2026-09-07. This fixes the intended computation and proposes numerical acceptance checks before inspecting optimizer parity arrays or new scientific recovery results. The root researcher makes the final acceptance decision. This audit changes no active fitting code and runs no GPU work.

The main task remains five independently sampled N100 **dense final-target** fits using the frozen calibration files. Each fit targets virtual layer 191 and returns all 191 matrices for source locations 0–190. Ouro has 48 physical decoder blocks and four recurrent passes. Reducing N, truncation length, the output-direction set, the source-layer set, or the number of recurrent passes would change the experiment, not optimize this computation. Penultimate-target, sampled-position, diagonal-position, and Huginn experiments remain separate scope decisions.

For one encoded paragraph of length T, define V = {16, ..., T−2} and M = |V|. The pinned tokenizer prepends BOS after truncating to 127 tokens; T can be below 128. If h_l(p,j) is the residual after source decoder block l and h_t(q,i) the residual after target decoder block t, the matrix is

`J_l[i,j] = (1/M) Σ_{p∈V} Σ_{q∈V} ∂h_t(q,i)/∂h_l(p,j)`.

Rows are target/output coordinates i and columns are source/input coordinates j. There is one source-position divisor M and no second target-position divisor. The matrix is neither a same-position Jacobian nor a Jacobian at an averaged hidden state. The target is the decoder-block output **before** that loop's final model normalization. The fitted map is the equally weighted arithmetic mean of the 100 per-paragraph matrices; paragraph length does not change paragraph weight.

The current reverse-mode method uses 2,048 output-coordinate directions. With direction batch B it performs `ceil(2048/B)` reverse calls, with all requested source derivatives shared in each traversal. There is no extra factor of 191 reverse traversals. Changes to the direction chunk size must cover each output row exactly once, including a partial final chunk. The model is in evaluation mode with frozen weights; parameter gradients and cache updates are unnecessary and must remain disabled.

A single-primal batched VJP has primal tensors `[1,T,2048]`, cotangents `[B,1,T,2048]`, and returned source gradients `[B,1,T,2048]`. The extra leading dimension enumerates derivative directions, not calibration paragraphs. For direction r, the cotangent is one at every valid target position and output coordinate r, and zero elsewhere. Reduce valid source positions on the T axis in float32; do not average over the derivative axis. A single primal can differ numerically from a replicated primal because batch shape may select different floating-point kernels, even when the real-arithmetic expression is identical.

The most promising additional mechanism is to differentiate with respect to compact **source perturbations** instead of asking autograd to return every full source-position gradient. Introduce one independent float32 vector δ_l of length 2,048 at each source event:

`h'_l(p) = cast_to_original_dtype(h_l(p).float() + 1[p∈V] * δ_l / M)`.

Evaluate at all δ = 0, and return target functional `f_i(δ) = Σ_{q∈V} h_t(q,i).float()`. The derivative `∂f_i/∂δ_l,j` is J_l[i,j] under the same autodiff algebra. Perturbations at different source locations are independent inputs; later zero-valued perturbations do not remove the propagation of earlier ones. A reverse VJP over the 2,048 target outputs is required. Forward-mode differentiation over the 191×2,048 source inputs would introduce a much larger direction count.

This compact formulation changes the requested output of each backward call from 191 full token-gradient tensors to 191 short row tensors. Internal token adjoints still have to propagate through attention and the MLP. The mechanism offers memory and dispatch opportunities; it does not remove the mathematical derivative workload. At B32, the original requested source-gradient bank has approximately 3.204 GB of bf16 elements; 191 compact float32 row blocks have approximately 50.069 MB. These are payload counts, not measured allocator peaks.

The following details are essential for compact perturbations:

- `h.float() + 0` must be cast back immediately to the original residual dtype before the next block. Leaving the residual in float32 changes the forward computation. Check zero-perturbation primals at every virtual layer, every loop boundary, and the target.
- Keep δ, its position mask, and the 1/M scaling in float32. Do not round the scale or perturbation to bf16 before its gradient is accumulated. Source scaling before a broadcast reduction can sum terms in a different float32 order from the stock `.float().mean(...)`; this small numerical difference still needs measurement.
- Use a **float32 target sum**, with no target mean. Its derivative through the cast provides the stock constant-one cotangent at valid target positions.
- The δ inputs already supply differentiable roots. Do not call `requires_grad_()` from `ActivationRecorder(start_graph_at=...)` inside a `torch.func` transform; such mutation is incompatible with functional transforms. Use perturbation hooks that return the modified tensor and remove themselves reliably after every call, including failures.
- Do not detach between blocks or recurrent passes. A source perturbation at virtual layer 47 must propagate through the intervening model norm and then through loop 2.
- Preserve the output-layout conversion: a functional Jacobian with shape `[output_coordinate, source_layer, input_coordinate]` must become `[source_layer, output_coordinate, input_coordinate]` without transposing the individual matrices.

The pinned Ouro implementation has several specific constraints. Its configuration has 16 query heads and **16 KV heads**, each of dimension 128: this checkpoint uses ordinary multi-head attention, not grouped-query attention. The source code supports KV repetition, but a general GQA implementation or benchmark is outside the current checkpoint's requirements. A claimed general optimizer should reject unsupported head configurations explicitly or test them separately.

Attention uses the selected Transformers attention backend, with causal masking, no dropout during evaluation, and rotary embeddings. A `torch.autograd.grad(..., is_grads_batched=True)` call may encounter missing or fallback batching rules in the actual SDPA backward kernel. Record the runtime backend and fallback warnings. Replacing fused SDPA with a math/eager implementation is a distinct engine candidate with its own primal, derivative, memory, and timing checks; it is not automatically numerically identical or faster. Keep the causal mask, scaling, head layout, rotary positions, and disabled cache unchanged.

Each physical Ouro decoder block contains four RMS normalizations: before attention, after attention before the residual addition, before the MLP, and after the MLP before the second residual addition. RMS variance and reciprocal square root are computed in float32, with dtype conversion and weight multiplication in a particular order. A manual reverse formula must preserve that order's autodiff behavior. The SiLU-gated MLP and both residual branches must contribute. The model also applies `self.norm` after **every** 48-block loop, then feeds that normalized state to the next loop. A blockwise reverse must include norms at boundaries 47→48, 95→96, and 143→144. It must exclude the final post-target norm when the requested target is decoder output 191. The same physical weights are reused across loops; numerical states and hook identities are different.

The second, conditional mechanism worth a bounded probe is a **compiled or fused frozen-block reverse**, starting with one representative Ouro block. Frozen linear reverse is `G_input = G_output @ W` for `Y = X @ W.T`; no weight gradient is needed. That identity alone does not reduce the dominant multiplication count because stock autograd already freezes the weights. Potential gains come from fusing RMSNorm/SiLU/residual pointwise backward operations, avoiding unnecessary intermediate tensors, and batching the adjoints efficiently. A whole hand-written attention/model reverse is not yet justified. First compare one block, then a two-block suffix crossing a loop norm, and only proceed if conformance passes and end-to-end measurements show a substantial gain over the best batched/graph candidate. Do not assume a generic textbook RMSNorm or attention derivative reproduces the runtime's casting and fused-kernel behavior.

Proposed acceptance cases and thresholds are below. Algebraic checks, numerical engine checks, and scientific readout checks are separate evidence. Passing a small row sample is enough to continue benchmarking, not enough to claim the complete fitter is verified.

| Case | Fixed comparison | Acceptance |
| --- | --- | --- |
| Explicit causal algebra | Existing CPU causal toy and an explicitly materialized position-by-position Jacobian; include nonlinear cross-position mixing | Float32 API maps: `atol=2e-7`, `rtol=1e-6`, matching the existing estimator audit; correct orientation and exactly one divisor M |
| Chunk and shape edges | Width 5; B=1,2,4,8; multiple valid lengths; empty-valid-set rejection | Every output row written once; final partial chunk correct; source keys, shape, dtype, counts and errors match reference |
| Recurrent boundaries | Small model with tied blocks, nontrivial loop-end RMSNorm, sources before/after a boundary, final and penultimate targets | Same explicit-Jacobian tolerance; dropping carry normalization or including a post-target normalization must fail |
| Zero perturbation | Stock and compact-δ execution using the same primal batch, backend, dtype and actual paragraph | Every captured virtual residual and normalized carry numerically identical; all source/target dtypes unchanged; no lingering hooks or parameter gradients |
| Local hand/compiled block | FP32 representative block, then two-block suffix crossing a loop norm; baseline frozen weights | `atol=1e-6`, `rtol=1e-5`; compare all adjoint inputs, not only output values; no TF32/backend change hidden in comparison |
| BF16 screening on actual Ouro | All 191 sources, fixed output rows spanning the width including the final row; compare reference and candidate on the same fixed non-evaluation paragraph | Report per-source relative Frobenius error, cosine, max absolute error and primal error; finite values throughout. Proposed numerical screen: relative error ≤0.005 per source, cosine ≥0.99998 for nonzero rows, and elementwise `atol=1e-5`, `rtol=0.016` |
| Complete selected engine | At least one full 2,048-row paragraph on all 191 sources; then a short multi-paragraph mean with different lengths; compare against stock | Same numerical thresholds, plus exact N and paragraph ordering; inspect worst layers and row errors, not only one pooled number. A row-sample pass cannot substitute for this case |
| Reuse and cleanup | Repeated calls with different paragraphs, direction chunks, and valid lengths; test a failure path | No stale primal graph, cotangent, δ, mask or graph-capture buffer; graph released after completion; hooks removed; weights remain frozen |
| Saved/readout behavior | Reload the completed comparison maps using the existing lens format; apply both to fixed already-saved residuals | Report all changed readout ranks/top-k decisions and their score margins. Do not claim bitwise or discrete-output equivalence if any differ; resolve whether differences exceed the existing serialization/engine variation before promoting the fitter |

The BF16 numbers are proposed engineering tolerances, not a theorem about downstream rank stability. They must be adopted or revised with a stated precision rationale before inspecting scientific recovery outcomes. A failure must not be repaired by choosing a tolerance or backend because it makes J-Lens perform better. If changing the primal batch or backend itself exceeds the numerical screen, separate that change from the backward implementation and compare against a matching-primal reference. Record that an engine changed rather than silently calling it a bitwise replacement.

Performance comparisons must keep paragraph bytes, token lengths, all 191 sources, target 191, output directions, dtypes, active backend and CPU settings explicit. Synchronize CUDA around wall timers; GPU events alone omit host dispatch. Report cold load/compile/capture cost, warm full-width prompt wall time, reduction/transfer and checkpoint/save time, maximum allocated/reserved GPU memory, and host memory where available. Use identical full-width workloads for the selected engine and the reference, and a warm repetition set; a handful of output rows is a microbenchmark. Account for warm-up cost over the intended 500 paragraphs instead of assuming an indefinitely long run.

The current [local profile](../profile_cost.json) measured 176.384 ms inside `autograd.grad` and 6.434 ms for reduction/copy/assignment per B2 sweep. Thus the former is 96.48% of those measured phases, but it includes CPU dispatch and is not a measurement of GPU-kernel-only time. Eliminating the entire latter phase would give at most a 1.0365× sweep speedup in that profile. The one measured forward was 0.659 seconds versus 187.206 seconds extrapolated across full-width sweeps, so avoiding forward replication is useful chiefly when its memory savings permit better backward batching. CPU norm/change/accumulation timings totaled about 0.677 seconds on mostly zero full-size matrices; checkpoint I/O was not measured. Those data do not justify prioritizing a broad CPU/transfer rewrite over the backward engine.

References inspected: [stock estimator](/home/moloch/jacobian-lens/jlens/fitting.py:100), [accumulation/checkpoint loop](/home/moloch/jacobian-lens/jlens/fitting.py:280), [hook implementation](/home/moloch/jacobian-lens/jlens/hooks.py:49), [dense/sampled wrapper](../fit_estimators.py:38), [Ouro wrapper](/home/moloch/ouro_project/src/ouro_jlens/recurrent.py:85), pinned Ouro configuration and `modeling_ouro.py`, and [the prior explicit estimator audit](../design_audit/verify_estimators.py). No optimizer is accepted by this document alone.

Independent parity inspection of the first legacy-vmap benchmark (performed after writing v1 above). All three arrays `batched_2`, `batched_8`, and `batched_32` in `bench_vjp.npz` are bitwise identical to one another. Each has shape `[191,2,2048]` and contains only the first two output-coordinate rows. Compared with `replicated_2`, their global relative Frobenius error is **1.51559664**, maximum per-source error **2.22785941** at virtual source 31, median per-source error **0.69266840**, and minimum source cosine **−0.26279075**. The maximum absolute difference is **0.86328125**. All 191 sources fail the proposed relative-error screen; 741,458 of 782,336 entries fail the proposed elementwise BF16 screen. These are not rounding-level differences. The benchmark execution flag `pass:true` checks successful execution, not parity.

The sampled tensors show no simple swap of the two rows: at source 190, the row-aligned cosines are approximately 0.999980 and 0.999983, while crossed-row cosines are about −0.00274. This is only a diagnostic. The cause is still unresolved between changed primal batch/backend behavior and batched-backward behavior; an ordinary scalar VJP on the same single primal, and a batched VJP on the same replicated primal, are the necessary controls before drawing that conclusion.

Inspected NPZ SHA-256: `a1032a91478137ab295daeea50df1da34e50fca4aef99952468b6ec1b62b364c`. Shared batched-array payload SHA-256: `6b2666254180714318ecefb33b0c43d34b0ca57044f3df5cf9ea212f17ea566d`. All comparisons used float64 accumulation on CPU. No model run or scientific recovery evaluation was performed.

The complete per-source relative Frobenius errors below apply to all three batched candidates. Physical layers are numbered 1–48; a source index is `(loop−1)*48 + physical_layer−1`. The final physical layer of loop 4 is the target and has no fitted source matrix.

| Physical layer | Loop 1 | Loop 2 | Loop 3 | Loop 4 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.79897355 | 1.80398075 | 0.52289818 | 0.11546969 |
| 2 | 1.81910584 | 1.81455642 | 0.57539719 | 0.12138124 |
| 3 | 1.83854692 | 1.80155953 | 0.72159010 | 0.14246636 |
| 4 | 1.86345984 | 1.80738810 | 0.79873530 | 0.16013919 |
| 5 | 1.86877429 | 1.80052824 | 0.85448352 | 0.16286382 |
| 6 | 1.90414455 | 1.79676350 | 0.86009372 | 0.15946936 |
| 7 | 1.94590140 | 1.79288205 | 0.83780284 | 0.15681150 |
| 8 | 1.97061790 | 1.70188272 | 0.82910547 | 0.15469819 |
| 9 | 1.98057778 | 1.74651805 | 0.81740190 | 0.14763331 |
| 10 | 1.94499156 | 1.78715351 | 0.80828443 | 0.14550276 |
| 11 | 2.05082421 | 1.76706833 | 0.78717288 | 0.14087494 |
| 12 | 2.13038849 | 1.71059650 | 0.74031260 | 0.13417626 |
| 13 | 2.13771684 | 1.65552149 | 0.67490758 | 0.12687154 |
| 14 | 2.20016711 | 1.59312071 | 0.63445869 | 0.12205419 |
| 15 | 2.15140413 | 1.57458741 | 0.58004162 | 0.12020059 |
| 16 | 2.05396260 | 1.56526311 | 0.47941372 | 0.11228975 |
| 17 | 1.98897725 | 1.60053684 | 0.43590482 | 0.11003071 |
| 18 | 1.93120647 | 1.47896570 | 0.41323676 | 0.09814654 |
| 19 | 1.86754931 | 1.42068624 | 0.36172857 | 0.08810820 |
| 20 | 1.93840841 | 1.43215352 | 0.35202181 | 0.09242682 |
| 21 | 1.95821029 | 1.42315242 | 0.34806559 | 0.07987019 |
| 22 | 1.95154780 | 1.41408751 | 0.34442029 | 0.07649729 |
| 23 | 1.93740034 | 1.38993624 | 0.34065695 | 0.07125563 |
| 24 | 1.87499179 | 1.34047694 | 0.34219040 | 0.06621399 |
| 25 | 1.89032658 | 1.43722730 | 0.37618322 | 0.06054217 |
| 26 | 1.92586554 | 1.33745551 | 0.30358674 | 0.05160571 |
| 27 | 1.99036810 | 1.30366689 | 0.26307411 | 0.04622374 |
| 28 | 2.04434784 | 1.26929850 | 0.24269111 | 0.04430996 |
| 29 | 2.09868891 | 1.28556731 | 0.23149468 | 0.04271651 |
| 30 | 2.04699901 | 1.20964717 | 0.21977746 | 0.04159961 |
| 31 | 2.19996247 | 1.14190273 | 0.21316337 | 0.04057411 |
| 32 | 2.22785941 | 1.08052696 | 0.20490782 | 0.03984266 |
| 33 | 2.18804849 | 1.06312697 | 0.20059086 | 0.03786880 |
| 34 | 2.21124858 | 0.91095481 | 0.19054698 | 0.03646774 |
| 35 | 2.14530718 | 0.85935705 | 0.18084856 | 0.03514536 |
| 36 | 2.12416132 | 0.78736565 | 0.17302920 | 0.03359081 |
| 37 | 2.11596390 | 0.71722011 | 0.16356751 | 0.03163686 |
| 38 | 2.07340010 | 0.69266840 | 0.15510547 | 0.02884214 |
| 39 | 2.02233670 | 0.68053280 | 0.15170151 | 0.02616360 |
| 40 | 1.99121324 | 0.67444079 | 0.14810924 | 0.02231020 |
| 41 | 1.98810307 | 0.67256508 | 0.14682645 | 0.01938376 |
| 42 | 1.98148432 | 0.66677693 | 0.14665118 | 0.01715986 |
| 43 | 1.99456021 | 0.66087628 | 0.14489668 | 0.01518709 |
| 44 | 1.98074157 | 0.65510989 | 0.14390782 | 0.01293011 |
| 45 | 1.96531976 | 0.65195769 | 0.14278377 | 0.01071734 |
| 46 | 1.88360581 | 0.66516874 | 0.14008461 | 0.00879185 |
| 47 | 1.81779840 | 0.63770535 | 0.13521839 | 0.00638845 |
| 48 | 1.77003549 | 0.61476319 | 0.13337161 | — |

Complete-reference cross-check: the separately completed `baseline_full.pt` contains exactly 191 CPU float32 matrices of shape `[2048,2048]`. A CPU-only `torch.load(weights_only=True, mmap=True)` comparison found its first two rows at **every** source bitwise equal to `bench_vjp.npz:replicated_2`: all 782,336 checked values match, with zero absolute or relative error. The three single-primal batched candidates retain the failures above against this complete reference. This verifies repeatability for the sampled rows; it is not a repeatability scan of all 801,112,064 matrix entries. The complete reference JSON reports 215.034 seconds for the full paragraph, plus 1.958 seconds for saving, versus 182.864 seconds extrapolated from the earlier few-sweep reference benchmark. Use completed full-width timings when available.

Additional numerical isolation audit, 2026-09-07. Independent NumPy comparisons of `bench_reduced.npz` and the earlier `bench_vjp.npz`, with float64 accumulation on CPU, separate the following effects. Each comparison covers all 191 sources but only output rows 0 and 1 of the same repeated-sentence benchmark paragraph, for 782,336 float32 entries. Every sampled tensor is finite.

| Comparison, with denominator from the second method | Global relative Frobenius error | Largest source error | Minimum source cosine | Sources failing relative error ≤0.005 |
| --- | ---: | ---: | ---: | ---: |
| Native scalar B1 / native replicated B2 | 1.51719078635 | 2.22934310052 at source 31 | −0.26779007422 | 191/191 |
| Compact-δ scalar B1 / native scalar B1 | 0; bitwise identical | 0 | 1 | 0/191 |
| Legacy vmap on B1 / native scalar B1 | 0.06608092380 | 0.10160270202 at source 0 | 0.99483348158 | 166/191 |
| Compact-δ vmap on B1 / native scalar B1 | 0.06608092380 | 0.10160270202 at source 0 | 0.99483348158 | 166/191 |

All four compact-δ direction chunks, G=2,8,32,64, are bitwise identical to one another and to the earlier legacy-vmap tensors. Thus the compact source-position reduction reproduces the corresponding scalar and batched engines exactly on these sampled rows; it does not cause the observed gap between those engines. For the shared vmap/scalar gap, maximum absolute error is 0.10910612345, 385,696 entries fail the proposed elementwise criterion, and 161 sources fail the cosine criterion. Its source-relative errors at 0,47,95,143,190 are respectively 0.10160270, 0.06242371, 0.01451281, 0.00845852, and 0.00034875. The growth over longer reverse paths is consistent with amplification of arithmetic differences, but does not identify the originating operation.

The `primal_errors` object has exactly 192 named decoder events and one loop-norm key containing four events: **all 196 comparisons report finite, bitwise-equal values and zero numerical errors**. The benchmark checks event-list lengths before comparing each pair, so the result does not silently omit extra recurrent calls. These are comparisons of **native B1 against zero-δ B1**. They do not compare native B1 against B2. The benchmark casts both sides to float32 before comparison and does not save the per-event dtype or original tensors; it therefore establishes equality of values after that cast, rather than independently certifying original dtype metadata. The injection source explicitly casts back to the original hidden dtype. Future direct event assertions should retain shape and dtype checks as well as value checks.

The B1/B2 result changes both the native forward batch shape and native backward matrix shapes. Consequently these files alone cannot assign its 1.51719 error wholly to changed forward states. Nor do they prove a particular CUDA kernel is faulty. The same-B1 scalar/vmap comparison does establish an additional difference introduced by the transformed reverse path. The log records a missing batching rule for `aten::_scaled_dot_product_efficient_attention_backward`, which establishes a fallback in this execution; that warning describes performance and is not evidence that attention, rather than a GEMM or another reduction, produces the numerical discrepancy. The old `bench_vjp.json` records script SHA `82a013dcef5ecaab27a35f9c6aa18ece5b3f7ccc2a2c3532bc1eba80d6fd0c0f`; the current script has since acquired additional command-line controls, so use the saved run metadata and tensors when identifying that run.

The installed build is `2.12.0.dev20260407+cu128`, commit `6a13e444ee88996ff01cd2bab41d7f2857291646`. A fresh CPU-only process reported the flags below; CUDA remained uninitialized before and after inspection. These are fresh-process defaults, not a reconstruction of every setting in the completed benchmark process. Neither saved benchmark JSON contains a complete precision-flag snapshot.

| Control | Observed default | Diagnostic relevance |
| --- | --- | --- |
| `get_float32_matmul_precision()`; `matmul.allow_tf32` | `highest`; `False` | Controls float32 matmul arithmetic; it does not upgrade BF16 matmuls to FP32 |
| `matmul.fp32_precision` | `none` | Record the effective legacy and new settings; do not interpret this string alone as enabled TF32 |
| `matmul.allow_bf16_reduced_precision_reduction` | `True` | Permits reduced-precision intermediate GEMM reductions |
| `matmul.allow_bf16_reduced_precision_reduction_split_k` | `True` | Separate split-K heuristic permission in this installed build |
| FP16 reduction and split-K; FP16 accumulation | `True`, `True`; `False` | Different dtype controls; not a substitute for the BF16 control |
| Flash, memory-efficient, math and cuDNN SDPA enabled | All `True` | These permit candidates; they do not record which kernel actually executed |
| `fp16_bf16_reduction_math_sdp_allowed()` | `False` | Applies to math SDPA, not the observed efficient-attention kernel |
| Deterministic algorithms enabled | `False` | Repeat a fixed case before attributing differences to a changed method |

The installed `torch/backends/cuda/__init__.py` parses a boolean BF16-reduction assignment as `(value, True)`. Therefore `allow_bf16_reduced_precision_reduction = False` disables reduced-precision accumulation while retaining split-K permission; `(False, False)` disables both. The matching PyTorch 2.12 API documentation identifies the second setting as a control on cuBLASLt split-K heuristics. A negative result with the boolean setting alone does not test the second control. Neither setting guarantees equality between different tensor shapes. [PyTorch backend controls](https://docs.pytorch.org/docs/2.12/backends.html#torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)

PyTorch documents that batched and sliced computations can differ despite mathematical equivalence, and that BF16 GEMMs can truncate intermediate accumulations unless reduced-precision reduction is disabled. This supports testing the arithmetic path; it does not establish the cause or acceptable size of this model's measured error. [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-gemms) SDPA selects among implementations based on inputs, and backend fusion can change numerical results. A forced math-backend comparison must apply the same backend to both reference and candidate and retain the existing causal mask; its FP32 intermediates are a distinct control. [PyTorch SDPA documentation](https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.scaled_dot_product_attention.html)

The bounded distinguishing checks, in order, are:

1. Repeat ordinary scalar VJPs on the same native B1 graph, then compare scalar against vmap with **G=1** and G=2 on that same graph. G=1 distinguishes transform/lowering changes from the presence of multiple derivative directions. Separately capture all 196 native forward events for B1, B2 and historical B32, compare the first lane across batches, and check equality between duplicate lanes. This localizes whether forward drift is already present and where it first appears; no full Jacobian is needed for the forward comparison.
2. Repeat the fixed small direction comparison with BF16 reduction set to `False`, then `(False, False)`, keeping the selected value on both sides and through forward and backward. Log both getters. Keep FP32 matmul at its effective highest setting. Changing flags only after constructing one side's primal would confound this test.
3. If the discrepancy remains, compare one fixed SDPA backend on both sides, first efficient and then math, recording actual operator names and fallback warnings. If a backend affects the gap, isolate one Ouro block or attention operation at identical saved inputs and cotangents before replacing the model's attention implementation. A flags-only fix or a fallback warning is insufficient to claim an attention root cause.
4. For a proposed replacement, retain the full-width and variable-length mean checks above. For comparison with historical maps, add a matched-paragraph B32/B2 control with the same checkpoint, device, backend and flags; otherwise describe an explicit numerical-engine difference. None of these controls should be selected based on scientific recovery scores.

The historical N100 maps used native replicated B32. New native B2 maps and historical B32 maps cannot be assumed interchangeable merely because their real-arithmetic estimator formula agrees. New calibration lists also use a different sampling policy from the historical prefix. A difference between historical and new fitted maps would therefore combine calibration selection and numerical-engine effects unless those factors are controlled. All five new fits must use one frozen numerical engine to interpret their between-fit spread as calibration variation conditional on that engine. The present two-row, one-paragraph evidence neither establishes how a full N100 mean changes nor overturns any saved recovery metric; it exposes a material comparability issue that must remain visible in the research conclusions.

Inspected hashes: `bench_reduced.npz` SHA-256 `724804ad73285eae4b6d6275247070f0de4874bafe95aac4523117542cf650b3`; `bench_reduced.json` `85438d3e461625bad653001db8455f062b23b2f4a547c0e97c7f58ada7bc4b59`; `bench_reduced.py` `b4faf69cef32d77144227a00e3da93710248f480a52903ca9a7a1eacdb0220a6`; inspected compact candidate `52ccc2d20bfdfe407fb856397f50416ba948cdbf552836ac14b0af34f7297386`. This audit launched no GPU work, changed no fitting code, and leaves the original acceptance thresholds intact.
