# Proposed compute budget — 7 September 2026

**Status: rejected by the user.** This earlier proposal is preserved below. See [the cost correction](COST_REVISION.md) for the essential replication block and B200/B300 comparison. No spending amount in this document is approved.

**Recommend $100 total: an Ouro spending cap of $75 and a separate conditional $25 allowance for Huginn.** The working estimate for Ouro is $20–40, pending a complete-prompt benchmark. These are proposed allocations, not a claim of user approval or guaranteed completion within either allowance.

The [refreshed public catalogue](gpu_catalogue_budget.json), retrieved at 19:52:49 UTC, still quotes one RTX 6000 Ada 48 GB Community GPU at $0.74/hour, with low stock, 78 GB host RAM and 14 vCPUs. The earlier $0.84/hour Secure offer is not available in this newer response; it is only a price sensitivity below. No account balance was queried and no Pod was created.

## Experiment scope priced before any new recovery result

- Five independently sampled N100 dense final-target fits, retaining all supported sources across all four Ouro loops.
- Five dense penultimate-target fits on those same calibration sets, comparing targets on their common supported source positions.
- Five final-target sampled-position runs on those same sets, each producing both the sum and diagonal reductions from a shared VJP stream.

This is 1,500 per-prompt Jacobian computations and 20 fitted maps. The position pair requires one backward stream, not two. It does require extra CPU reduction, matrix accumulation and output storage. The count is fixed for this budget before outcomes; it will not be chosen by whether a positive band appears. This is not a full target-by-position factorial sweep.

## Cost arithmetic

| Assumed average seconds per prompt computation | Sequential GPU hours for 1,500 | At $0.74/hour | At $0.84/hour |
|---:|---:|---:|---:|
| 45 | 18.75 | $13.88 | $15.75 |
| 90 | 37.50 | $27.75 | $31.50 |
| 180 | 75.00 | $55.50 | $63.00 |

**None of those Ada timings has been measured.** They are planning scenarios covering the full computations, including the extra position-arm work. The $20–40 working estimate adds setup, numerical validation, evaluation and storage to the faster scenarios. The $75 cap allows a substantially slower run and some failed attempts. Historical concurrent B300 process timers were not treated as billable wall time.

Twenty final outputs require approximately 32 GB if saved in FP16, or 64 GB in FP32, plus model weights, checkpoints and evaluation artifacts. The allocation includes storage, retrieval and shutdown time. Local report writing and CPU analysis do not require an idle rented GPU.

The first cloud work should measure complete prompts and validate the real Ouro gradients before committing the full block. Update the runtime forecast using only timing and memory, without consulting task-recovery outcomes. Preserve the five-fit scientific block; if the measured cost exceeds the remaining allocation, resolve that before starting a block that cannot finish. Any authorized spending cap must include all Pods and storage rather than only the fitter's timers.

Huginn's $25 is a conditional pilot allowance, not a promise that its adapter or matched experiment will succeed. Ouro comes first. Use the allowance only after gradient, initialization, coda, tokenizer and calibration-separation checks pass and its own measured cost fits the allowance. The existing prospective prediction remains unchanged. No causal swap is included.

An independent arithmetic audit checked the derivative-family count, shared-VJP accounting, cost scenarios and matrix sizes. The full executable run/provenance manifest is still to be finalized before fitting.
