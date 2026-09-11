# Primary runtime feasibility evidence

This report covers the frozen run `ouro_confirmation_20260909_fixed160` and the retained RTX 5090 Ouro evaluation. It is a measured proxy for the evaluator-only work. No new model forward, provider operation, or confirmation result was produced while collecting this evidence. The retained old evaluation was complete; the first primary attempt was stopped during setup before `BANKS_READY` or an artifact index.

## Relevant surface

- The frozen run specification is `research/confirmation_2026-09-09/bundle/frozen/run_spec.json` (18,643 bytes, SHA256 `8a4a3e6725d811d0b2584561a277d5071f097c551f40c46be9f6773a7d61181e`). It declares 160 planned items, 80 concepts, development items `carnival-ocean`, `amazon-language`, and `mars-color`, and five bank-backed arms: `fit01`, `fit02`, `penultimate`, `sampled_sum`, and `diagonal`. The raw arm makes six readout arms in total.
- The retained baseline is rooted at `research/refit_round_2026-09-07/monitoring/attempt_06/ouro_handoff_20260909T110503Z/results`. It contains the complete old main evaluation (`items=148`, five fits, `files=24`, elapsed `158.2429125609342` seconds) and complete controls evaluation (`items=148`, three control readouts, `files=19`, elapsed `23.064808867988177` seconds).
- The old runtime metadata records NVIDIA GeForce RTX 5090, capability `[12,0]`, 170 multiprocessors, `33668857856` bytes of VRAM, CUDA 12.8, cuDNN runtime 92000, eight CPU threads, one interop thread, SDPA, and `torch 2.12.0.dev20260407+cu128` in `ouro_evaluation/common/metadata.json` (SHA256 `fd420918419427467cd899564d7c7cc7c2e4e55f192f8d9698eb4d1fec125540`).
- The immutable old model manifest is `research/confirmation_2026-09-09/bundle/legacy/model_manifest.json`: 13 files, total model input bytes `5341718984`, model `ByteDance/Ouro-2.6B`, revision `1ed04250da1a9936042725d302e81c8fa2ab5abd`. The model was loaded for the retained old run; no new model load was performed for this review.
- The retry admission record is `research/confirmation_2026-09-09/resources/primary_retry_admission_plan.json`. It currently declares `setup_budget_seconds=12300` (3 h 25 min), `compute_budget_seconds=3600`, preservation reserve 1800 seconds, and transfer timeout 900 seconds. `RUN_LOG.md:70-72` describes the same planned retry and records that attempt01 stopped before computation.

## Established behavior

- `bundle/legacy/evaluate_refits.py:_evaluation_runtime` (lines 602-637) initializes one frozen BF16 Ouro model on CUDA, checks geometry `(4,48,192,2048)`, checks evaluation mode and device/dtype, verifies the remote implementation hashes, and requires SDPA. `_evaluate` (lines 640-745) starts one elapsed timer, loads all 148 items, calls `cache_states` once, validates FP32 state geometry `[148,192,2048]`, computes four native exit logits at virtual indices `[47,95,143,191]`, retains target-state logits for 190 and 191, computes raw logit-lens arrays, and then loops fit IDs 1 through 5. The old log confirms the stage sequence `cache_states`, `raw_logit_lens`, and `fit_readout` IDs 1–5.
- `bundle/legacy/evaluate_controls.py:_evaluate` (lines 469-575) starts a separate timer and model runtime, loads the shared main cache by mmap, validates the `[148,192,2048]` state geometry and finite FP32 tensors, checks target-191 bitwise parity with the native final exit, revalidates common/raw/fit01 arrays, and loops `ouro_penultimate` dense, `ouro_positions` `sampled_sum`, and `ouro_positions` `diagonal` (lines 504-545). Its old log contains exactly those three control stages.
- `bundle/evaluation/measurement.py` declares the new frozen arms and support (lines 5-9): raw and fit01/fit02 use virtual 0–191, penultimate uses 0–189, and sampled-sum/diagonal use 0–190. The same file says the scorer has no model fitting or selection (line 1).
- `bundle/evaluation/worker.py` loads one model/runtime and checks frozen sources (lines 38-75), runs the three old development items through `readouts.native_development` (lines 90-108), publishes the development tensors and waits for an externally bound acceptance receipt, then caches the 160 new items one at a time (lines 109-129). It loops raw plus all five bank-backed arms, loading each bank with CPU mmap and preparing its readout (lines 130-141), then uses the per-item wrapper and publishes each arm (lines 142-153).
- `bundle/evaluation/readouts.py:readout` (lines 27-48) invokes the original `readout_arrays` once per item with a one-item state slice, captures top-10 IDs, and stores numerical samples for item indices 0, 1, and 2. `native_development` (lines 50-84) records physical hooks, virtual activations, wrapper activations, native logits, and unembedded logits for each development item. The worker compares virtual/physical/wrapper states and native/unembedded logits bitwise before waiting for acceptance (worker lines 90-108).
- The new worker consumes the five existing lens banks; it does not fit them. The retained old fitting logs are a separate historical cost context. Summed per-prompt elapsed rows are `32416.77566821559` seconds for old main fits 1–5 and `13831.362243451294` seconds for the old controls, `46248.13791166688` seconds in total. Those numbers must not be added to an evaluator-only compute budget when the frozen worker is using precomputed banks.

## Execution or data paths

The closest six-arm baseline is the old main process: one model/cache pass followed by raw plus five independent fit readouts. Scaling its complete process elapsed time from 148 to 160 items gives

`158.2429125609342 * (160/148) = 171.07341898479373 seconds`.

The old controls process supplies a separate operational envelope for penultimate and position-style readouts. Scaling its complete process elapsed time gives

`23.064808867988177 * (160/148) = 24.934928505933165 seconds`.

Adding those two scaled values gives `196.0083474907269` seconds. This is an upper-envelope proxy rather than an exact new-worker prediction: controls starts another model runtime, reloads/revalidates the shared cache, and rechecks raw/fit01, while the new worker keeps one model and one cache. Conversely, the new worker has unmeasured three-item native development, external acceptance waiting, per-item wrapper calls, six per-arm saves, and six publication/manifest operations that the old elapsed records do not isolate.

The old common cache is `ouro_evaluation/common/cache.pt`, 407,374,621 bytes, SHA256 `fb51d115b231f482c86233bbda74055660b79d7fdf952e6d7006f8ed9bd3e3f4`. Its FP32 tensor payload is 407,371,776 bytes: `[148,192,2048]` states (232,783,872 bytes), `[148,4,49152]` exit logits (116,391,936 bytes), and two `[148,49152]` target-state tensors (29,097,984 bytes each). A linear 160-item cache with the same four tensor fields would have payload 440,401,920 bytes (420 MiB) and a serialized-size estimate of 440,404,765 bytes using the old 2,845-byte archive overhead. The actual frozen worker line 129 writes only `H`, `exit_logits`, and continuations, so its new cache omits the two old target-state tensors; continuation and serialization overhead are not measured.

The retained old output arrays establish the row and support geometry but do not provide stage timings: common, fit01, and fit02 arrays use `[148,3,192]` and `[148,128,192]`; penultimate uses `[148,3,190]` and `[148,128,190]`; sampled-sum and diagonal use `[148,3,191]` and `[148,128,191]`. Their exact NPZ records are respectively common 3,982,267 bytes (`0169d2d18a55f18eab14ee2e14891936b1da1263e5e8e4d9da5166180ce0ca15`), fit01 3,794,670 (`904ff358c453fc527cde6833f3fec321db174c5c8f03df57dd9168142aa759`), fit02 3,752,472 (`ee766346bbced5a6ae380cd974443c45f14e632bfbab52dd5c65f61a5b5a5065`), penultimate 3,813,311 (`1cef42112bb2859fa4de26dcf60559b5cae90a11892ac9e4633f859d45546c7d`), sampled-sum 3,812,126 (`ec0a67b30847985eb475e04753c8069d7031e087c699df80c221397247859cd5`), and diagonal 3,723,655 (`fbb4efa8eb0e7534dec8096480ae96eded7911339ef3c37e3b7be63b48f60d48`).

Using the `196.0083474907269` second envelope only as the comparison point, the candidate evaluator budgets have these arithmetic margins:

| `compute_budget_seconds` | margin over scaled envelope | envelope multiples |
| ---: | ---: | ---: |
| 600 | 403.9916525092731 s | 3.061094120128664× |
| 1200 | 1003.9916525092731 s | 6.122188240257327× |
| 1800 | 1603.9916525092731 s | 9.183282360385991× |

These margins compare only against retained evaluator process elapsed values. They do not establish a hard completion bound: no old stage-level timings, native three-item duration, new per-item wrapper/publication timing, or future serialized output size exists. The retry setup clock is separate from this compute comparison; the current retry plan records a long setup allowance and one hour of evaluator compute.

## Tests and validation surface

- Retained old receipts are complete and hash-stable: `ouro_evaluation/COMPLETE.json` reports status complete, 148 items, 24 files, and elapsed 158.2429125609342 seconds; its record is 887 bytes, SHA256 `eeaa3d7e6fe74f59b9d52e6061a841d1942042766c7e8d1cc614db9657eb3d7e`. `controls_evaluation/COMPLETE.json` reports status complete, 148 items, 19 files, and elapsed 23.064808867988177 seconds; its record is 666 bytes, SHA256 `6dceceac5aa6449b89b60878b9c68df3ba735e8fbb5a74dcb418159bcdb2c04f`.
- The old source validates model geometry, frozen BF16 parameters, remote source hashes, SDPA, common-cache finite FP32 values, target-191/native-exit equality, raw and fit-array rank support, and complete section seals. The new source validates frozen input-bank hashes, population tokenization, development bitwise equality, per-item rank/top-10 equivalence, and immutable per-arm publication.
- Read-only checks performed for this report were SHA256/size verification of retained receipts and arrays, NPZ key/shape/dtype inspection, Torch archive member-size inspection for the common cache, and independent scaling arithmetic. No model was loaded and no GPU was initialized by these checks.
- The source-level CPU self-tests and synthetic receiver tests documented by the implementation review exercise scorer and acceptance logic, but they are not runtime measurements for a native GPU forward or for the new three-item development loop.

## Contradictions

- The old complete main run has 148 items and raw plus five old fit IDs, whereas the frozen new run has 160 items and raw plus fit01, fit02, penultimate, sampled-sum, and diagonal. The six-arm count is comparable; arm composition and cache reuse are not identical.
- The old controls evaluator is a separate process that reloads the model and cache. Summing scaled main and controls elapsed values therefore double-counts some work for the one-process new worker, while omitting new wrapper/publication and development costs. The sum is retained as an operational envelope, not a measured combined runtime.
- The old common cache contains target-state logits 190 and 191 and the controls validator requires them (`evaluate_controls.py:446-466`); the frozen new worker's cache save contains only `H`, `exit_logits`, and continuations (`worker.py:122-129`). The old controls validator cannot consume that new cache shape without an additional producer or compatibility step.
- `results/environment.json` and the nested runtime environment in `ouro_evaluation/common/metadata.json` report different Linux platform strings (`Linux-7.2.3-1-cachyos...` versus `Linux-6.8.0-111-generic...`) while both retain the same CUDA 12.8/torch development family. The retained metadata does not explain this capture difference.
- Historical fit logs make the full estimator-plus-readout workload much larger than any 600/1200/1800-second budget, but the frozen worker is explicitly bank-consuming and has no fitting call. Treating those fitting rows as part of this evaluator budget would describe a different workload.

## Missing evidence

- No old stage-level timings separate model load, tokenizer/item load, state caching, raw readout, each fit readout, validation, or final sealing. The available old elapsed values are process totals.
- No retained timing exists for the new worker's native development pass over the three old items, external development artifact transfer/acceptance, per-item wrapper readout, `npz`/Torch saves, or publication manifest hashing.
- No new RTX 5090 execution, development tensor receipt, new-item activation cache, readout array, or confirmation result exists. Attempt01 ended during setup; `RUN_LOG.md:72` explicitly records no `BANKS_READY` and no artifact index.
- No measured 160-item serialized cache or readout sizes exist. The 440,401,920-byte same-field cache figure is a linear geometry estimate; the actual new cache omits target tensors and has unmeasured continuation text.
- No evidence establishes that setup duration, provider scheduling, model download, disk staging, or network transfer remains within the retry clock. The recorded 64 MiB transfer measurements are setup/transport observations, not evaluator compute timings.
- No independent timing establishes GPU-to-GPU variance, CUDA warm-up behavior, or whether the old runtime metadata's platform discrepancy changes throughput.

## Exact paths and symbols

- `research/confirmation_2026-09-09/bundle/evaluation/worker.py:38-75,90-108,109-153` — frozen worker source; SHA256 `2db64e6d598623bb87dde39d55e7de67dcab762c0a148face40d6dea98e045d0`.
- `research/confirmation_2026-09-09/bundle/evaluation/readouts.py:27-48,50-84` — per-item readout and native development functions; SHA256 `2d1762f126e9f1da9a9b7aabc2039de3697bbc5fc5b385ab6a034aa2dc931226`.
- `research/confirmation_2026-09-09/bundle/evaluation/measurement.py:5-9` — arm and support declarations; SHA256 `06c615b29285e0e4e10196ae8f15b015d5c8a362f5e7704adb38ce6e275799fe`.
- `research/confirmation_2026-09-09/bundle/legacy/evaluate_refits.py:602-745` — old runtime, cache, raw, and five-fit loop; source SHA256 `63345907f22b713763857f910e4dc56afa261979ef1da55f2f5943754d7b0b17`.
- `research/confirmation_2026-09-09/bundle/legacy/evaluate_controls.py:446-575` — old cache validation, target parity, and three-control loop; source SHA256 `cbe9dc654712cf6b7f88648b5fb53cf1079df3d8b2ef33c3bd90d8ce4625886c`.
- `research/refit_round_2026-09-07/monitoring/attempt_06/ouro_handoff_20260909T110503Z/results/logs/ouro_evaluation.log` and `controls_evaluation.log` — retained stage markers and COMPLETE records.
- `research/refit_round_2026-09-07/monitoring/attempt_06/ouro_handoff_20260909T110503Z/results/ouro_evaluation/common/cache.pt`, `common/metadata.json`, `ouro_evaluation/COMPLETE.json`, and `controls_evaluation/COMPLETE.json` — retained cache/runtime/receipt evidence.
- `research/confirmation_2026-09-09/resources/primary_retry_admission_plan.json` and `research/confirmation_2026-09-09/RUN_LOG.md:70-72` — current retry clocks and preserved incomplete-attempt status.
