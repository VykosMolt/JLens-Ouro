# RunPod GPU selection — 7 September 2026

**Choose one RTX 6000 Ada 48 GB on Community Cloud, quoted at $0.74/hour.** It is the strongest starting candidate for completing the current Ouro Jacobian fits quickly at a low cost. The RTX 4090 at $0.34/hour is the close alternative. We have not established a measured cost winner between them.

The recommendation uses current public catalogue data and this implementation's retained-graph requirements. Account credits did not enter the comparison. No Pod was created and no fitting result was generated during this selection.

## Current offers

The [public API snapshot](gpu_catalogue_details.json) was retrieved at **19:04:59 UTC**. Prices are USD per GPU-hour for one on-demand GPU, before storage. The API reports stock categories; it does not reserve a machine. Community stock was `Low` for each listed Community offer.

| GPU | VRAM | Community/hour | Secure/hour | Assessment for the Ouro fits |
|---|---:|---:|---:|---|
| RTX 4090 | 24 GB | $0.34 | $0.74 | Strong candidate for lowest dollars per fit |
| RTX 5090 | 32 GB | $0.69 | $0.99 | Higher bandwidth, but less attractive BF16 compute per dollar |
| **RTX 6000 Ada** | **48 GB** | **$0.74** | **$0.84** | **Selected: more batching room and high BF16 compute per dollar** |
| L40S | 48 GB | $0.79 | $1.09 | Similar class; slightly more expensive |
| A100 SXM | 80 GB | $1.39 | $1.59 | Extra memory is not required for this Ouro configuration |
| RTX PRO 6000 Blackwell Server | 96 GB | $1.69 | $2.09 | More capacity than this stage needs |
| H100 SXM | 80 GB | — | $3.49 | No evidence that this implementation recovers its price premium |

The first [catalogue snapshot](gpu_catalogue.json), retrieved at 19:03:00 UTC, quoted B300 at $7.89/hour with low stock. The later response returned no B300 offer. RTX 3090 was also available at $0.22/hour, but has substantially less BF16 compute than 4090 at the same 24 GB capacity. These are live catalogue observations, not guarantees about a subsequent deployment. The [RunPod pricing page](https://www.runpod.io/pricing) displays different default cloud prices, so using that page alone would miss the cheaper Community offers.

## Why the Ada

This job computes dense BF16 derivatives through 192 virtual layers, with T up to 128 and width 2048. It retains the forward graph and repeats backward passes over batches of output dimensions. FP4 inference TOPS and sparse tensor numbers do not describe this computation.

NVIDIA's tables give **165.2, 209.5 and 364 dense BF16 TFLOP/s with FP32 accumulation** for RTX 4090, RTX 5090 and RTX 6000 Ada respectively. At the quoted Community prices, that is approximately **486, 304 and 492 peak TFLOP/s per dollar/hour**. These are hardware ratios, not observed application throughput. Sources: [GeForce architecture, Table 3](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf) and [RTX PRO architecture, Table 4](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf).

The Ada and 4090 therefore offer almost the same theoretical compute per dollar. The Ada combines that with twice the VRAM. Archived Ouro benchmark notes reported approximately 24 GB at dimension batch 8 and 77 GB at batch 32; the new laptop preflight measured 9.96 GB at batch 2 and exhausted its 12 GB card at batch 4. A conservative starting batch is **16 on Ada**, with 8 as fallback. A 24 GB card may need batch 4–8. These cloud-card batch sizes are estimates to validate, not successful measurements. Larger batches reduce the number of backward calls while preserving the derivative budget.

The quoted Ada offer included 78 GB host RAM and 14 vCPUs, compared with 41 GB and 8 vCPUs on the 4090 offer. Both have plausible host capacity for the current fits. Cap PyTorch/BLAS threads at eight initially: excessive CPU threading caused a major slowdown in the historical B300 work. More host RAM also leaves room for the two CPU matrix collections in the position-control estimator.

## What could change the choice

Ada must complete the same work **2.176× faster** than the $0.34/hour 4090 to cost less. Its dense BF16 peak is 2.203× higher, so the theoretical value margin is only **1.24%**. Its memory bandwidth is slightly lower: 960 versus 1008 GB/s. If bandwidth, CPU overhead or transfers dominate, the 4090 could be cheaper. More batching room does not remove Jacobian entries or total transfer bytes. We should describe Ada as the selected starting card, not a proven optimum.

Before committing the full frozen experiment to a machine, time the existing retained-graph preflight plus a complete calibration prompt, including CPU reduction and accumulation. Select the dimension batch using memory and timing only. Preserve T, source coverage, target and derivative counts. A genuine two-card comparison would use the same calibration prompts and compare elapsed seconds × hourly price; task-recovery outcomes must not influence that choice. It is not necessary to benchmark the entire catalogue.

The historical B300 evidence does not supply that comparison. Its successful final-target shard timers sum to 6,703.86 seconds for 100 prompts, but their intervals overlap: **summed fitting-process time is not billable Pod wall time**. No retained benchmark measures this exact job on 4090, 5090 or RTX 6000 Ada. See [the timing audit](audit/GPU_TIMINGS.md) and [the hardware audit](GPU_HARDWARE.md).

This choice covers the current Ouro refits. A later Huginn pilot has different graph and matrix sizes and remains conditional on the Ouro results and adapter validation.
