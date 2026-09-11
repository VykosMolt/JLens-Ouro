# Precision preflight failure: attempt 03

The failure is caused by the worker's redundant legacy matmul precision setter.
The frozen historical precision record is internally consistent. In a fresh local
process using the same `torch 2.12.0.dev20260407+cu128` build and the exact frozen
precision environment, every field returned by the pinned legacy `_precision`
helper matched `original_precision`. Executing the two assignments in frozen
`evaluation/worker.py:60–61` produced exactly one difference:

| Field | Frozen original / fresh defaults | After worker assignments |
| --- | --- | --- |
| `matmul.fp32_precision` | `"none"` | `"ieee"` |

The assignment `torch.backends.cuda.matmul.allow_tf32=False` changes the newer
precision state even though the legacy boolean was already false. The full equality
check at worker lines 75–76 correctly rejects the resulting record. CUDA remained
uninitialized in the diagnostic; no model or scientific outcome was inspected.
The complete source records, method, and observed output are in
`PRECISION_PREFLIGHT_DIAGNOSIS.json`.

The historical source is
`research/refit_round_2026-09-07/monitoring/attempt_06/ouro_handoff_20260909T110503Z/results/ouro_evaluation/common/metadata.json`.
Its `runtime.precision` exactly equals the current frozen `original_precision`.
The legacy helper at `bundle/legacy/run_refits.py:856` records both the boolean and
the newer precision field; `bundle/legacy/evaluate_refits.py:631` uses that helper.

## Current binding is terminal

Attempt 03 has locally latched failed final manifest
`fd042fa7cf1938d08d1e65ff4700890e210d572bd33534622fe511b4d9b6ffb0`,
with an empty file set. This error occurs before development artifact production
and before any confirmation state cache. A retry under that binding is unsupported:

- `Publisher.__init__` in `bundle/evaluation/artifacts.py` rejects an existing
  `artifact_index.json`.
- `remember_manifest` in the pinned controller rejects a changed final manifest
  and replacement of the latched terminal manifest SHA.
- Final receipt acceptance requires outcome `complete`; the failed outcome cannot
  be promoted to a success receipt.

Preserve the failed publication and its binding. Do not delete/reset publication
state or alter the current frozen worker to reuse the attempt.

## Minimal prospective correction

Create an explicitly versioned corrected bundle, preserving the original bundle
and failure evidence. Remove both redundant `allow_tf32` assignments from its new
worker: the fresh pinned defaults already reproduce the entire original precision
record. Removing only the matmul assignment addresses the measured difference;
removing both avoids needlessly mutating already matching historical settings.

Retain the original precision record and full dictionary equality check unchanged,
including its environment and SDPA fields. Retain the exact benchmark, population,
bank hashes, estimands, analysis, native state/logit equality, and external
development acceptance gates. Bind and verify the corrected source prospectively
before a new run; require actual full precision agreement before development or
confirmation states. The correction does not justify weakening any precision or
native gate.

This document reports independent local reproduction. The root agent separately
preserves the actual remote failure logs and runtime observations and handles
termination of the failed lease. No frozen file, lease, pod, or service was changed
while producing this diagnosis.
