# Reproduction commands

Run from `/home/moloch/jacobian-lens/research/verification_2026-09-11`. Every script refuses to overwrite its output, so point `--out` or `--roots` at a new path when repeating a step.

```sh
PY=/home/moloch/ouro_project/venv/bin/python   # Python 3.14.7, torch 2.12.0.dev20260407+cu128; matches legacy/environment.json
GPU="env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_ALLOC_CONF=expandable_segments:True"
```

GPU steps ran on an RTX 5070 Ti Laptop GPU. Other GPUs may differ in the last bits.

| Step | Command | Output |
|---|---|---|
| Run specification | `python3 inventory/run_specification.py` | `RUN_SPECIFICATION.json` |
| Readout replay (level 2), 9 layouts | `$GPU $PY replay/replay_scorer.py --device cuda --out replay/local_cuda_rtx5070ti` | about 12 min |
| FP64 reference on CPU | `env OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES= $PY replay/replay_scorer.py --device cpu --layouts fp64 --out replay/local_cpu_fp64` | about 8 min |
| Propagation through the frozen analysis | `$PY replay/compare_paths.py --run local_cuda_rtx5070ti=$PWD/replay/local_cuda_rtx5070ti --roots $PWD/replay/propagation_local_cuda --out $PWD/replay/sensitivity_local_cuda.json` | analogous for `local_cpu_fp64` |
| Scoring-shape inputs | `$GPU $PY -B gpu_shape_v1/prepare_inputs.py` | `gpu_shape_v1/inputs/` |
| Native output, sample rows, development shapes, regenerated states | `$GPU $PY -B gpu_shape_v1/run_local.py --out gpu_shape_v1/local_rtx5070ti` | about 4 min |
| Replay from regenerated states (level 3) | `$GPU $PY replay/replay_scorer.py --device cuda --layouts executed --states $PWD/gpu_shape_v1/local_rtx5070ti/regenerated_states.pt --out $PWD/replay/local_cuda_regenerated_states`, then `compare_paths.py` as above | `replay/sensitivity_local_regenerated.json` |
| Independent checker, saved outputs | `$PY checker/endpoint_checker.py --population $A/population.json --readouts $A/readouts --expected-analysis $SAVED/analysis.json --out NEW.json` | `checker/saved_outputs.json` |
| Checker fixtures | `$PY checker/endpoint_checker.py --self-test --out NEW.json` | `checker/self_test.json` |
| Checker, one propagation root | `$PY checker/endpoint_checker.py --population ROOT/population.json --readouts ROOT/readouts --expected-analysis ROOT/analysis.json --out NEW.json` | `checker/replay_checks/` |
| Metrics | `python3 -B metrics/build_metrics.py` | `metrics/VERIFICATION_METRICS.json` |
| Preservation bundle | `python3 -B preservation/build_bundle.py core`, later `... verification` | `preservation/BUNDLE_MANIFEST_*.json` |
| Restoration and loading | `$GPU $PY -B preservation/restore_check.py` | `preservation/RESTORATION_CHECK.json` |
| Ledger | `python3 -B preservation/build_ledger.py` | `preservation/PRESERVATION_LEDGER.json` |
| Protected roots unchanged | `python3 -B inventory/compare_protected.py` | `inventory/protected_after_comparison.json` |
| Read-only account observation | `$PY -B ../confirmation_2026-09-09/diagnostics/native_logit_v1/observe_account.py --out NEW.json` | `spending/observation_end.json` |

In the checker rows:
- `A=../confirmation_2026-09-09/cloud_leases/attempt_06/handoff/accepted/ouro_confirmation_20260911_fixed160_native1/ff7c0769b2b0dbe2e53b120daf7f0729434fe65e54c013288771f5052ffd4cef/results`
- `SAVED=../confirmation_2026-09-09/results/ouro_confirmation_20260911_fixed160_native1/analysis`

The checker imports only numpy and the standard library.
