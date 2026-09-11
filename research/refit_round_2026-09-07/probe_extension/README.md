# P4 bounded workflow

`CONTRACT.md` is frozen at SHA-256 `b92411f0a4551870cc3a3a7be6d4d5fa168a3eec97b124f6634292eb211c6b8b`. This directory contains prospective implementation and local CPU checks. It does not authorize execution: P1/P2 interpretation, remaining time, the existing combined $25 cap and priority of Huginn remain the root operator's gates.

`workflow.py` has no model/scientific work on import. It uses the existing Ouro/J-Lens modules directly for native extraction and the installed weighted sklearn APIs locally. It never imports the legacy probe, CV or audit entrypoints on the GPU. Every output directory and receipt must be new. Changed implementation bytes require a new plan.

## Prepare and review locally

```bash
/home/moloch/ouro_project/venv/bin/python -B research/refit_round_2026-09-07/probe_extension/workflow.py self-test \
  --proof research/refit_round_2026-09-07/probe_extension/SELF_TEST.json

/home/moloch/ouro_project/venv/bin/python -B research/refit_round_2026-09-07/probe_extension/workflow.py prepare \
  --output research/refit_round_2026-09-07/probe_extension/prepared

/home/moloch/ouro_project/venv/bin/python -B research/refit_round_2026-09-07/probe_extension/workflow.py validate \
  --plan research/refit_round_2026-09-07/probe_extension/prepared/plan.json
```

Preparation hashes all eight literal historical artifacts and the declared reference sources. It reads only the small CV, saved-pairing and historical prediction arrays to validate ancestry, classes, counts, pair leakage, selected locations/C and all 20,000 retained draws. It does not decompress historical `H` or fit a classifier. The plan preserves the complete historical provenance bytes and the full 13-file model inventory. Any difference in an unused historical provenance source is identified explicitly; every consumed source remains hash checked.

Self-test uses small synthetic CPU features, three tiny classifiers (one deliberately emitting a convergence warning), fake recurrent blocks and the real source hooks/encode method. It checks class/weight objectives, tied ranks, missing classes, unequal pair denominators and exact retained-draw replay, output/source tampering, incomplete and externally expired/failed completion, token metadata serialization and extraction geometry. It does not load model weights or call CUDA. Existing completed proof paths should be inspected rather than overwritten; use a fresh path for a repeat.

The completed local evidence is `SELF_TEST.json` (128 passing synthetic checks), `TOKENIZER_PREFLIGHT.json` (all 1,224 actual local-tokenizer encodings and all 17 alias sets), and `PREPARATION_PROOF.json` (bound preparation and packaging identities). The tokenizer check used the native encode method on CPU with offline loading, checked the six tokenizer input files before and after, retained complete literal-prompt readout positions, and exercised actual `AddedToken` metadata serialization. Encoded lengths were 12–13 tokens; CUDA remained uninitialized. These checks establish prospective implementation behavior, not model/GPU timing or scientific outcomes.

Preparation preserves the archived `src/ouro_jlens/evaluate.py` identity `ceb76d742ebee87f74ade09780b04cb5f43f5cda7256420ae6d6dc036280858d` (43,821 bytes) and records its current identity `9118bd8f48925930e56c851028958dd62b372c2d9b507d141fc9bf9af90e9c22` (63,535 bytes). That entrypoint is not consumed by P4. The preparation false-refusal fix keeps its historical provenance intact and reports the difference, while still enforcing every declared current source pin and consumed dependency identity.

## Small standalone GPU payload

The only newly staged files are:

- `workflow.py` and the unchanged `CONTRACT.md`;
- `prepared/plan.json`, `prompts.json`, `model_manifest.json` and `historical_provenance.json`.

Keep those four metadata files together beside `plan.json`. Preserve `workflow.py` and `CONTRACT.md` beside each other. The plan maps canonical local historical paths to their hashes; those paths are not opened on the GPU. No historical feature arrays, sklearn packages or new model files need uploading. A local payload can be assembled in a new directory with ordinary file copies; hash it and compare its source/metadata bytes before transfer. The already-staged Ouro source, J-Lens source (including its existing `data/evaluations` directory used at `evaldata` import) and model snapshot supply the runtime inputs.

The prepared local bundle is `payload.tar.gz`; its six files and archive hash are listed in `PAYLOAD_MANIFEST.json`. Archive paths are relative to the intended P4 directory. The manifest and local proof files remain local. A relocated copy of the packed workflow and plan is validated before the preparation proof is written.

## Bounded GPU command, after the root gates

Paths below illustrate the existing `/workspace/jlens` layout; the root operator must supply its actual reviewed staged paths. No cloud resource is allocated by this command. Retain the current worker environment:

```bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HOME=/workspace/jlens/runtime_cache/huggingface HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

timeout --signal=TERM --kill-after=5s 895s \
  /workspace/jlens/venv/bin/python -B /workspace/jlens/p4/workflow.py supervise-extract \
  --python /workspace/jlens/venv/bin/python \
  --snapshot /workspace/jlens/models/ouro \
  --ouro-src /workspace/jlens/bundle/ouro_project/src \
  --jlens-root /workspace/jlens/bundle/jacobian-lens \
  --plan /workspace/jlens/p4/prepared/plan.json \
  --output /workspace/jlens/p4_results/cache \
  --deadline /workspace/jlens/p4_results/deadline.json \
  --receipt /workspace/jlens/p4_results/supervisor.json
```

Create only the parent `p4_results` directory beforehand. The supervisor starts a monotonic 900-second receipt before child Python initialization, launches the extraction child in its own process group, and kills that group at the deadline. The child also enforces its receipt with a timer and work-boundary checks. Loading, all 1,224 native B1 BF16 forwards, 648 batch-seven native raw readouts, serialization, hashes and model cleanup occur inside that phase. The supervisor verifies the child's complete seal before writing its external successful receipt.

The root also launches the entire supervisor under `timeout --signal=TERM --kill-after=5s 895s ...`, covering supervisor initialization and all post-child checks within the 900-second total cap. The conservative 895-second TERM boundary reserves cancellation time. Supervisor SIGTERM/SIGHUP handlers immediately kill the extraction process group and write an incomplete receipt; short-lived synthetic process tests exercise both paths. A missing, late, failed or incomplete supervisor receipt makes analysis refuse the cache, even if a child happened to write `COMPLETE.json` just before termination. Existence of partial arrays or a child marker is insufficient. `manifest.json:gpu_elapsed_seconds` is a child pre-seal checkpoint; the supervisor receipt supplies the validated whole-phase elapsed time.

The direct child CLI is the same argument set without `--python`/`--receipt`, using `extract` in place of `supervise-extract`. It requires an already-started deadline receipt and cannot by itself produce an analysis-acceptable external receipt. This entrypoint is provided for supervised execution, not to bypass the outer gate.

## Retrieve and analyze locally

Retrieve the entire sealed cache directory, `supervisor.json`, `deadline.json`, and the command's stdout/stderr log. Account for GPU extraction, CPU/retrieval/coordination elapsed time while the resource remains rented in the combined budget and updated Huginn forecast.

```bash
/home/moloch/ouro_project/venv/bin/python -B research/refit_round_2026-09-07/probe_extension/workflow.py validate \
  --plan research/refit_round_2026-09-07/probe_extension/prepared/plan.json \
  --cache /absolute/retrieved/p4/cache \
  --receipt /absolute/retrieved/p4/supervisor.json

/home/moloch/ouro_project/venv/bin/python -B research/refit_round_2026-09-07/probe_extension/workflow.py analyze \
  --plan research/refit_round_2026-09-07/probe_extension/prepared/plan.json \
  --cache /absolute/retrieved/p4/cache \
  --receipt /absolute/retrieved/p4/supervisor.json \
  --output research/refit_round_2026-09-07/probe_extension/analysis_run01
```

Analysis performs exactly five fresh baselines, five augmented fits and five historical-cache diagnostic baselines. Every scaler and solver receives the same per-row weights, with one native thread covering both. Augmented class total weights preserve their original counts; the output records actual floating sums and the scaler's effective float32 weight sum. The common fresh test matrix, fixed classes/layers/C and saved eligible mask remain fixed. Any fitting error, nonfinite value or convergence warning marks the whole analysis incomplete.

The sealed CPU output contains 15 coefficient/scaler/training-row/weight archives, full per-item scores/ranks/predictions, all retained paired bootstrap draws and denominators, the primary and descriptive contrasts, historical numerical diagnostics and runtime/source identities. `results.json` explains the conditional uncertainty and restricted upper-sum interpretation. No extraction, fitting or scientific result is supplied by the preparation proof.
