# Huginn feasibility inspection, 7 September 2026

**Decision: do not run Huginn J-Lens now.** The checkpoint is locally complete at the structural level, and its native modules expose the necessary states. However, this is a custom adapter task with several correctness traps, and the lead's environment checks found no available CUDA device and an NVIDIA driver communication failure. No checkpoint was loaded, no tensor/model experiment was run, and no cloud resource was requested during this inspection. The prospective prediction in [RESEARCH_LOG.md](RESEARCH_LOG.md#prospective-huginn-prediction) and [literature.md](literature.md#implications-fixed-before-a-new-huginn-result) is unchanged; there is no new Huginn scientific result.

## Cached checkpoint

Inspected `/home/moloch/ouro_project/artifacts/hf_cache/hub/models--tomg-group-umd--huginn-0125/snapshots/bb6621b65e90b6a4b9b29ef88dc83866d450470c`. `refs/main` names this revision. Config, remote model/config code, tokenizer files, generation config, weight index, and all four indexed safetensors shards are present; every snapshot symlink resolves.

Standard-library inspection of the safetensors headers found 77 tensor entries, all F32. Every indexed tensor occurs in its designated shard, with contiguous data offsets, and all four actual file lengths equal their header-derived expected lengths. The summed tensor payload equals the index's `total_size`: **15,645,600,384 bytes**; the four complete files occupy approximately 14.571 GiB. This is structural completeness, not a recomputation of the weight blobs' cryptographic hashes or a successful loading test.

The stored tensors include a positional-frequency buffer and both embedding/head keys. With the declared weight tying, the source/config imply 3,564,976,800 unique parameters. The large on-disk size reflects F32 storage and the additional head entry; it does not mean the native model has 7.8B parameters.

| File | SHA-256 of inspected small file |
| --- | --- |
| `config.json` | `e9fe79df06a783ca33a76038c59a715b6a62f15c4a5b17b9681e697fea46c79c` |
| `model.safetensors.index.json` | `a28061573e97ba3645b482c11d82ba903b40942de0113472b30afb5a3cf4c1eb` |
| `raven_config_minimal.py` | `36a39939e76dc9e8f7563187c6498e012ca7f64a3d86b32ff1daa7f4c41a4278` |
| `raven_modeling_minimal.py` | `a1d447da93c605a6dc11fbc17cbc7185663f4a41ae12335b34b5847edf0f84aa` |

The config specifies width 5,280, intermediate width 17,920, 55 attention and KV heads, head dimension 96, vocabulary 65,536, two prelude blocks, four recurrent blocks, and two coda blocks. It declares mean recurrence 32 and test-time noise defaults to zero in the accompanying config class.

## Native boundaries and the adapter contract

The following claims refer to the pinned `raven_modeling_minimal.py`, not a generic Huginn implementation.

1. **Repeated modules require invocation-aware taps.** Each physical `transformer.core_block[b]` runs once per recurrent pass. Its output is already the block's `norm_4` output. The existing `jlens.hooks.ActivationRecorder` overwrites earlier invocations if given the same physical module. Reuse the idea of Ouro's `LoopTap`, with a distinct virtual index for every `(pass, block)` and immediate filtering on the native `step_idx` argument. With two prelude blocks, core calls have indices `2 + 4*r + b` for zero-based `r,b`; coda calls use `-1,-2`. Check this mapping against captured call counts before fitting. Do not detach taps used for Jacobians.

2. **Scalar recurrence depth disables the required gradient path.** `iterate_forward` interprets scalar `num_steps=R` as `R` no-grad passes and zero grad passes (lines 736–759). `eval()` also defaults to this path. The explicit pair `(0, R)` uses the grad-capable path, but the native outer forward still computes the LM head even when `output_details["return_logits"]` is false (lines 699–716). `iterate_one_step` is decorated with `torch.no_grad()`. These are unsuitable defaults for the fitting protocol.

3. **A thin custom forward can reuse the native operations.** Implement `LensModel` directly; ordinary `from_hf` layout selection cannot turn the `ModuleDict` and repeated stack into a callable flat decoder. Reuse native embedding/prelude, `core_block_forward`, norm, and coda operations; omit the LM head during `forward`. Frozen weights plus the recorder's earliest-source leaf can bound the retained graph. Preserve adapter injection of the prelude state at every pass. Alternatively, use the native `(0,R)` route first for parity testing, accepting its unnecessary head cost. Keep checkpointing and compilation off for the first validation: checkpoint recomputation can refire hooks, and native checkpointing disables RNG preservation. Neither complication is needed to establish basic correctness.

4. **Keep the two readout paths distinct.** The trajectory is

   `s_(r,b) -> remaining core -> ln_f -> coda[0] -> coda[1] -> ln_f -> lm_head`.

   For a final-output J-Lens, target the output of the last coda block, **before** its subsequent `ln_f`; then `unembed(h) = lm_head(ln_f(h))`. Raw logit lens at a recurrent block uses that same norm/head directly. Native `predict_from_latents` instead applies `ln_f`, both attention-bearing coda blocks, another `ln_f`, and the head. Feed it the raw core-block output: `forward().latent_states` is already normalized and detached. Passing that returned field straight to `predict_from_latents` adds an extra pre-coda normalization. Supply the complete sequence state and original positions to the coda, then select evaluation positions. A coda readout at R4 is the native exit at that pass; a coda readout at R1–R3 is a diagnostic intervention, not a native model exit. No token-vector-only replacement can reproduce coda attention.

5. **Random initialization must be coupled across replicated rows.** Native initialization samples a truncated normal with standard deviation `sqrt(2/(5*d))`, truncated at ±3 standard deviations, then scales by `sqrt(d)` (lines 802–811). Merely calling `manual_seed` before a batch still gives its rows different states. `jacobian_for_prompt` assigns different output dimensions to replicated rows, so different states would splice rows from different Jacobians. Draw one state per prompt and replicate it identically across the `dim_batch` rows; persist the base seed and state-generation specification. Reuse the same initial state for every paired readout and native-exit comparison, and assert `test_time_noise == 0`. The nearby runner's `init_state` provides the correct distribution and explicit generators, but its independently sampled per-row states must be adapted for this estimator.

6. **Use unpadded single prompts in the first contract.** Native `forward` currently sets `prepared_attn_mask = None`, ignoring its supplied attention mask, while `embed_inputs` takes a different masking route. Unpadded single prompts, replicated only for output dimensions, avoid this mismatch and preserve causal attention. Tokenize with a recorded explicit BOS convention, no KV cache, and complete position coverage. Do not inherit the older probe's masked-mean pooling: concept recovery here requires position-specific states and logits.

The existing [`bg_v3_huginn_probe.py`](/home/moloch/ouro_project/utilities/tests/manual/bg_v3_huginn_probe.py) captures only the final core block, runs in `inference_mode`, detaches states, and pools tokens. It verifies a useful extraction pattern, but neither its features nor its forward wrapper implement J-Lens. [`runner/model.py`](/home/moloch/elastic_reasoner/runner/model.py) contributes pinned loading and explicit RNG ideas; it also uses inference mode and detached traces. No change to the J-Lens estimator itself appears necessary if a correct virtual-layer adapter is supplied, but that judgment is based on inspection and still needs executable validation.

## Resource implications

For the existing estimator, each prompt entails one forward on `dim_batch` replicated rows and `ceil(d/dim_batch)` backward calls. Increasing `dim_batch` trades memory for fewer calls without reducing total backward arithmetic.

| Quantity | Ouro, d=2,048 | Huginn, d=5,280 | Ratio |
| --- | ---: | ---: | ---: |
| Entries in one dense Jacobian | 4,194,304 | 27,878,400 | 6.647 |
| One float32 Jacobian | 16 MiB | 106.35 MiB | 6.647 |
| One float16 saved Jacobian | 8 MiB | 53.17 MiB | 6.647 |
| Backward calls per prompt, batch 8 | 256 | 660 | 2.578 |
| Backward calls per prompt, batch 4 | 512 | 1,320 | 2.578 |

At eight Huginn passes there are 32 recurrent source cells and 36 executed transformer blocks including prelude/coda. Their float32 Jacobian matrices alone total **3.323 GiB**; fitting can simultaneously hold the running sum and per-prompt matrices, and merging/saving can create further copies. At 16/32 passes the corresponding single-set totals are 6.647/13.293 GiB. A single BF16 residual of shape `[8,128,5280]` is 10.31 MiB; the autograd graph saves many tensors per block, including the wider MLP activations. This residual size is not an estimate of peak graph memory.

The config gives about 0.791 GFLOP per token for one block's linear operations, plus 0.112 GFLOP per recurrence for its adapter. At eight passes, the forward's linear operations including its head total about 30.05 GFLOP per token, excluding attention, norms, and elementwise work. These are arithmetic counts, not measured fitting time. Width increases both derivative directions and each block's arithmetic; Huginn also executes fewer blocks than the four-pass Ouro setup. There is no defensible single runtime multiplier from width alone.

The pre-existing [`huginn_profile.json`](/home/moloch/elastic_reasoner/profile/huginn_profile.json), produced by [`profile_huginn.py`](/home/moloch/elastic_reasoner/profile/profile_huginn.py), identifies this same snapshot and model-code SHA-256. It records PyTorch `2.12.0.dev20260407+cu128`, Transformers `4.54.1`, BF16, and an RTX 5070 Ti Laptop GPU with 11,813 MiB total memory. Loaded weights used 6,801 MiB. An R=8, 528-token prefill used 7,737 MiB peak and about 0.614 seconds; recorded decode cases at R=8/32 succeeded. Those were inference and cached-generation measurements. They do **not** establish that a replicated retained graph plus hundreds of backward calls fits or completes within a bounded budget. The device is not currently usable according to the lead's checks.

## Smallest defensible future pilot

This is a proposed bounded contract, not authorization to start compute or a claim that it will fit.

1. **Engineering gate:** after local CUDA is restored, use the pinned snapshot with BF16 weights, no cache, no checkpointing, no compilation, and unpadded prompts. Check native-versus-adapter logits at fixed input states; exact hook counts at every virtual cell; identical activations across replicated rows; finite nonzero source-to-coda VJPs; and agreement of selected Jacobian rows between dimension batches 1 and 2. Include a numerical directional-derivative check on a small, gradient-capable version of the native architecture before relying on BF16 checkpoint derivatives. Do not inspect task recovery to decide whether the adapter passes. Stop on failed parity, disconnected gradients, or inconsistent rows.

2. **Fixed scientific scope:** R=8, all four blocks at every pass, with the final coda-block residual as the one target. This is a low-compute pilot at a declared depth, not a replication of the probing paper's R=16 experiment or a test of all Huginn depths. Fit the same first ten archived WikiText texts from `wikitext_prompts_b08601e.json`, maximum 128 tokens, skip the first 16 and the final position, preserving the current estimator. Evaluate the complete submitted concept dataset with its existing labels, controls, leak exclusions, and score. Freeze the exact eligible intersection under both tokenizers before reading new recovery results; report exclusions. Use paired raw logit lens and J-Lens at every cell, coda diagnostics at every cell with R1–R3 labeled as interventions, and actual exits after each R4. Preserve the full sequence through every coda evaluation. Fit with a fixed per-prompt state seed and evaluate two predetermined state seeds, reporting them separately as a limited stochastic sensitivity.

3. **Comparable claims:** report complete cell curves and predeclared depth averages with paired item/label-cluster uncertainty. Do not pick a block or pass after observing the results. Any-layer maxima offer different numbers of opportunities in Huginn and Ouro, so they cannot alone establish the cross-model prediction. A quantitative cross-model pilot comparison must use an Ouro lens with the same ten calibration texts and matching eligibility; the submitted N=100 Ouro lens is additional context, not a matched calibration-size baseline. Retain the recorded prediction that the paired J-Lens advantage over raw readout should be larger in Huginn. A null or negative result stays so. The pilot cannot isolate the causal contribution of intermediate supervision or establish convergence of a ten-prompt Jacobian average.

4. **Compute gate:** cap local engineering/profiling at 20 GPU minutes. Measure graph memory and a small fixed set of backward calls at the planned R and sequence length, allowing only dimension batch 2 and then 1 as a memory fallback. Before producing a fitted lens, project the complete ten-prompt fit, two-seed evaluation, and writes against a two-hour local GPU ceiling and available host memory/storage. If the projection exceeds that ceiling, or the graph does not fit, stop and record the failed feasibility gate. A partly fitted lens or reduced task set must not silently become the planned scientific comparison. No paid fallback is part of this contract.

This inspection therefore supports preparing a small adapter and its parity checks later; it does not support starting a broad Huginn fit under the current conditions.
