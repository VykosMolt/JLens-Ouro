The successful retained B300 final-target fit shards total **6,703.860 fit-process seconds for 100 prompts**, or **67.0386 seconds per prompt**. **This is a sum of process timers, not observed pod wall time or a billable GPU duration.** The metadata creation times and fit durations overlap substantially, consistent with concurrent fitting work. These timers must not be multiplied by a provider hourly price to claim an observed bill or a reliable cost forecast.

The measurements come from successful `kind=fit` sidecars, not from the old short benchmark or failed runs. Each final-target shard records matching requested and fitted prompt counts. They use Ouro-2.6B, 48 physical layers and four recurrent passes, final virtual target 191, **all 191 source locations 0–190**, T128, BOS prepended, `skip_first=16`, `dim_batch=32`, and checkpointing every 10 prompts. This is the full graph; it is not a final-pass-only fit. The source loader uses bfloat16 model weights, while per-prompt Jacobians and accumulated sums use float32 on CPU and final saved lenses default to fp16.

| Final-target shard, under `/home/moloch/ouro_project/artifacts/jlens/lens/n100/` | N | Fit-process seconds | Seconds / prompt | Metadata creation time, UTC |
| --- | ---: | ---: | ---: | --- |
| [exit3_shard_0000_0008.json](/home/moloch/ouro_project/artifacts/jlens/lens/n100/exit3_shard_0000_0008.json) | 8 | 520.961 | 65.120125 | 2026-09-05 02:45:30 |
| [exit3_shard_0008_0032.json](/home/moloch/ouro_project/artifacts/jlens/lens/n100/exit3_shard_0008_0032.json) | 24 | 1,479.947 | 61.664458 | 2026-09-05 02:41:29 |
| [exit3_shard_0032_0056.json](/home/moloch/ouro_project/artifacts/jlens/lens/n100/exit3_shard_0032_0056.json) | 24 | 1,671.823 | 69.659292 | 2026-09-05 02:51:55 |
| [exit3_shard_0056_0080.json](/home/moloch/ouro_project/artifacts/jlens/lens/n100/exit3_shard_0056_0080.json) | 24 | 1,664.777 | 69.365708 | 2026-09-05 02:54:38 |
| [exit3_shard_0080_0100.json](/home/moloch/ouro_project/artifacts/jlens/lens/n100/exit3_shard_0080_0100.json) | 20 | 1,366.352 | 68.317600 | 2026-09-05 03:06:34 |
| Total / N-weighted mean | 100 | 6,703.860 | 67.038600 | — |

The five shard means range from **61.664 to 69.659 seconds per prompt**, with median **68.318** and N-weighted population standard deviation **3.241 seconds**. This describes dispersion among shard averages. Successful B300 raw per-prompt fit logs were not found among the retained task-specific logs, so within-shard per-prompt dispersion cannot be recovered. No confidence interval for individual prompt timing is implied.

For example, creation time plus the recorded duration for `[8,32)` spans approximately 02:41:29–03:06:09 UTC, while `[0,8)` spans 02:45:30–02:54:11 and `[32,56)` spans 02:51:55–03:19:47. Creation occurs shortly before the timer starts, not at a separately recorded scheduling event; these are approximate intervals, sufficient to flag overlap but not to reconstruct GPU utilization or the complete pod schedule.

The other successful N100 target fits show the expected increasing process time with target depth. All use T128, `dim_batch=32`, `skip_first=16`, and four contiguous N25 shards. The source range is always every virtual location before the selected target, not a restriction to that target's recurrent pass.

| Target exit, human numbering | Virtual target | Source locations | Sum of fit-process seconds, N100 | Seconds / prompt |
| --- | ---: | --- | ---: | ---: |
| Loop 1 | 47 | 0–46 | 1,783.799 | 17.83799 |
| Loop 2 | 95 | 0–94 | 3,370.198 | 33.70198 |
| Loop 3 | 143 | 0–142 | 5,231.279 | 52.31279 |
| Loop 4 | 191 | 0–190 | 6,703.860 | 67.03860 |

The loop-1 and loop-2 raw shard sidecars are in [the retrieved N100 directory](/home/moloch/ouro_project/artifacts/jlens/retrieved/jlens-b300-20260905-0359/lens/n100); loop-3 and loop-4 sidecars are also in [the local N100 directory](/home/moloch/ouro_project/artifacts/jlens/lens/n100). Duplicate copies were checked for identical timing, counts, configuration, creation times, and output digests before deduplication. No historical final-pass-only fit timing was found.

The [wrapper timer](/home/moloch/ouro_project/src/ouro_jlens/fit_lens.py:1326) starts immediately before `jlens.fit` and stops immediately after it returns. It includes prompt encoding, forward and backward computation, GPU-to-CPU Jacobian row transfers, CPU norm/change diagnostics, accumulation, periodic and final running-sum checkpoints, and conversion from the sum to the fitted mean. It excludes model loading, model/source/prompt hashing before the fit, final lens serialization and output hashing, publication/retrieval, and subsequent evaluation. Those excluded overheads do not have separate retained timing measurements. A final-target checkpoint contains about 3.204 GB of float32 matrix payload; a final saved shard is about 1.602 GB. Disk and transfer overhead therefore cannot be assumed negligible.

The inspected timer and estimator source bytes match the hashes in the successful fit sidecars: `fit_lens.py` SHA-256 `007517db44a77957479757a9e6f1f6b67375c64cd8d90df682b9f6f2a8b3c6b0`, `jlens/fitting.py` `5be8959db8efc34cee41ed677beba84e21ba3c9e3ccb958bdbc1600c86b5e080`, and `jlens/lens.py` `e231e7d3a6c8e8f7791b53705a34342d0bba376a127a82376eaf6ec30ca11808`.

The merged [exit3.json](/home/moloch/ouro_project/artifacts/jlens/lens/n100/exit3.json) contains `seconds=520.961`, inherited from its first N8 shard. That field is **not the total N100 fitting time**. Aggregate timing above is calculated from distinct raw fit sidecars only.

The successful-run hardware fields record **NVIDIA B300 SXM6 AC** and a historical **$7.89/hour** rate. Only GPU identity, hourly rate, and image fields were extracted from the retained run record; no account balance or credential fields were reported. The [worker entry script](/home/moloch/ouro_project/src/ouro_jlens/pod_entry.sh:244) requires one visible B300 device. The archived task-specific hardware notes identify the card as 288 GB. These are historical resource records, not a current offer.

The [submitted application](/home/moloch/ouro_project/docs/jlens/APPLICATION_DRAFT.md:70) documents the successful run's CPU fix: reduce PyTorch's 192 threads to **eight**. Its approximate post-fix “40 seconds” figure is not a substitute for the complete fit-sidecar timers above. The retained successful-fit sidecars do not record per-process CPU thread counts, OpenMP/MKL environment variables, CPU allocation, or inter-op thread settings, so those details cannot be independently reconstructed. The checked-in historical fit/launch scripts do not themselves contain an eight-thread cap; it was an operational setting and must be set explicitly in a new run.

For memory planning only, [an archived task-specific benchmark note](/home/moloch/ouro_project/docs/jlens/history/2026-09-04-opus/RESEARCH_LOG.md:509) records full-graph B300 T128 measurements: `dim_batch=8` peaked at **24 GB**, batch 32 at **77 GB**, batch 64 at **149 GB**, and batch 128 ran out of memory. It reports 0.13/0.32/0.57 seconds per sampled backward pass respectively. These are secondary records of an earlier short benchmark; the raw benchmark output was not found retained. The [benchmark implementation](/home/moloch/ouro_project/src/ouro_jlens/bench.py:99) estimates prompt time from a few backward sweeps and omits full-fit CPU diagnostics, checkpointing, final saving, and publication. It is not equivalent to the successful full-fit timer.

No task-specific measured RTX 4090, RTX 5090, RTX 6000 Ada, RTX PRO 6000, A100, or H100 fit timings were readily discoverable. Relative throughput or feasible dimension batches on those cards remain projections until a comparable fit is timed. This audit launched no jobs and queried no live billable resources.
