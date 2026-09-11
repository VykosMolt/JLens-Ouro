# GPU and budget after optimization — 8 September 2026

**Choose one RTX 4090 24 GB on RunPod Community Cloud, quoted at $0.34/hour. Set aside $10 total for the five independent N100 final-target Ouro fits and their evaluation.** This is a proposed spending cap, not an approved rental or a guaranteed completion price. No Pod has been created.

The [public catalogue](gpu_catalogue_after_optimization.json), refreshed at 07:49:05 UTC on 8 September, reports low stock and a lowest-price configuration with 50 GB host RAM and 20 vCPUs. Actual deployment must retain the quoted rate and sufficient resources. The earlier 48 GB RTX 6000 Ada recommendation depended on activation capacity. The [optimized fitter](optimization/RESULTS.md) completes native B8 in 8.36 GB and executes B32 in 11.12 GB on the local GPU. A 24 GB card is therefore the better starting point for a cost-focused benchmark. This is a justified selection, not a measured cloud cost optimum.

The refreshed Community quotes are $0.34/hour for RTX 4090, $0.69 for RTX 5090 and $0.74 for RTX 6000 Ada. The larger cards would need about 2.03× and 2.18× the 4090's actual throughput to cost less. Capacity no longer supplies the earlier reason to favor Ada. The existing [hardware comparison](GPU_HARDWARE.md) distinguishes the relevant dense BF16 arithmetic from inference marketing figures.

The $10 scope is exactly five independently sampled N100 dense final-target fits: 500 paragraph Jacobians, all four recurrent passes, all 191 strict source layers and all 2,048 output directions. It includes an initial rental benchmark, fitting, setup, evaluation and artifact retrieval/storage. It does not bundle target/position controls or Huginn into this first stage.

There is no measured RTX 4090 runtime yet. A transparent planning anchor is to assume it completes each paragraph only as fast as the measured local B8 run:

| Component | Calculation | Cost |
| --- | --- | ---: |
| Full Jacobian computation | 500 × 192.3013439 seconds = 26.7085 hours, at $0.34/hour | $9.08 |
| Setup, evaluation and retrieval allowance | Two additional GPU-hours; an allowance, not a measured duration | $0.68 |
| Temporary storage | 50 GB for about 28.71 hours at $0.10/GB/month | About $0.20 |
| Total planning anchor | Before any speed advantage on the 4090 | **About $9.96** |

Storage uses an approximate 30-day month for this calculation. RunPod documents per-second billing for compute and container/volume storage, $0.10/GB/month while running, and no ingress/egress fee in its [Pod pricing documentation](https://docs.runpod.io/pods/pricing). Transfer time can still consume rented GPU time, which is included in the allowance. The selected 50 GB disk allocation is separate from the quoted 50 GB of host RAM.

This anchor leaves little margin at laptop speed and is not an upper bound on cloud runtime. A faster 4090 run would reduce the bill, but a $4–6 expected bill is not yet supported by a measured cloud benchmark. The first complete rented-GPU paragraph must establish the projection before the full replication proceeds; the total spending cap remains $10 unless changed by the user.

The 192.30-second anchor applies to B8. It cannot be transferred to B32: batch size affects BF16 numerical results, and the B32 local result was only a short timing projection. Validate the selected cloud engine against a matching-batch reference and fix batch, precision and software before examining scientific outcomes. Do not change those settings between independent fits to obtain a favorable recovery result.

This proposal replaces the earlier GPU preference and the rejected $100 all-stage budget for the immediate replication stage. It does not claim any new scientific fit or cloud measurement has been completed.
