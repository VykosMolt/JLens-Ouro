# Cost correction after the rejected budget

The user rejected the proposed $100 allocation. **No $100 spending authority exists, and no cloud job has been launched.** The previous [budget](BUDGET.md) is retained as the proposal that was rejected.

The earlier answer combined the required five independent fits with two additional five-run control families, padded an unmeasured $20–40 working estimate to a $75 cap, and added a speculative $25 Huginn reserve. It answered a question about likely cost with an oversized all-stage allowance. The research priorities already call for the main replication first; that should also be how the spending is staged.

## Same essential workload on three cards

The essential replication block is **five independent N100 final-target fits**, covering all four Ouro passes: 500 prompt Jacobian computations. Its fit count remains selected before outcomes. Controls and Huginn are later stages, not included in this block's price.

| Card | Current on-demand rate | Timing scenario for five main fits | Fitting-only cost scenario |
|---|---:|---:|---:|
| RTX 6000 Ada, Community | $0.74/hour | 6.25–12.5 hours | $4.63–9.25 |
| B200, Secure | $6.79/hour | 4–6 hours, assuming comparable B300 throughput | $27.16–40.74 |
| B300, Secure | $7.89/hour | Approximately 4–6 hours | $31.56–47.34 |

These are **planning scenarios, not measured completion-time intervals or spending caps**. They exclude separately unmeasured setup, evaluation, final artifact retrieval and storage. Rates were checked against [RunPod pricing](https://www.runpod.io/pricing) and the [19:52:49 UTC public API snapshot](gpu_catalogue_budget.json).

Ada's scenario retains the unmeasured 45–90 seconds per complete prompt computation assumed in the earlier budget; dividing the expanded 1,500-prompt plan by three gives the smaller number. The GPU has not been benchmarked for this code, so the comparison does not prove Ada's actual dollars per completed fit.

B300 has better empirical context. The five historical shards that together formed one N100 final-target fit have an approximate union of overlapping created-time-plus-duration intervals of 2,871.352 seconds, or 47.85 minutes. Repeating that throughput for five new N100 fits gives about 3.99 hours and $31.47 before other overhead. The original benchmark/research prose reports roughly 29–40 seconds per prompt, also suggesting about four to six hours for 500 prompts. These secondary timings do not establish an exact new-run duration. Scheduling, CPU settings, checkpointing and other simultaneous tasks differed.

Summing those old shard timers instead gives 6,703.86 process seconds per historical N100, which would extrapolate to $73.46 for five at $7.89/hour. **That is not a measured B300 bill or a valid upper bound.** The shard runs overlapped. The whole successful historical lease lasted approximately 3h14m from pod binding to a termination attempt, but it included other targets and evaluation and has no exact provider-deletion timestamp in these records. It cannot be charged entirely to one final-target fit. See [the timing audit](audit/GPU_TIMINGS.md).

No B200 fitting measurement exists here. NVIDIA lists B200 and B300 at the same 2,250 dense BF16 TFLOP/s and up to 8 TB/s bandwidth: [Exemplar table](https://github.com/NVIDIA/exemplar-performance/blob/main/README.md#peak-theoretical-throughput), [HGX specifications](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html). At equal actual throughput, B200 costs 13.94% less. B300 needs over 16.20% more throughput to offset its higher rate. Both have sufficient nominal memory for the historical batch-64 peak of approximately 149 GB; that fit remains to be verified on a B200. Extra B300 memory alone is not evidence of a speedup for this job.

Each Ouro prompt computes all 2,048 output directions of the Jacobian. Dimension batch 16 requires 128 backward calls per prompt; batching does not reduce the number of derivative directions. This explains why the operation is more costly than the dataset's 500-prompt size suggests, but does not justify paying for unmeasured contingencies or optional stages upfront.

The practical next stage is a timed complete-prompt preflight, followed by the five-fit replication block only when its measured projection fits an agreed small allocation. The original target/position questions remain pending; no controls or scientific claims are silently removed to make the estimate appear smaller.
