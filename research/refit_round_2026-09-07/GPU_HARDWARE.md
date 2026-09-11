# GPU hardware evidence for the next Ouro Jacobian fit

Checked 2026-09-07. **RTX 6000 Ada 48 GB Community at $0.74/hour is a reasonable first candidate for speed and cost together. RTX 4090 24 GB at $0.34/hour remains the essential cost comparator.** This is a hardware-informed choice, not a measured optimum. The larger card must complete the same fit over 2.176 times faster to cost less.

Prices below come from the read-only RunPod catalogue snapshot [gpu_catalogue.json](gpu_catalogue.json), retrieved at `2026-09-07T19:03:00.968999+00:00`. They are the lowest quoted one-GPU on-demand rates in that snapshot, using Community where quoted and Secure otherwise. No credits enter this comparison. Availability, host resources and the final quoted configuration still belong to the catalogue decision; the hardware specifications below do not establish them.

| GPU | VRAM | Memory bandwidth, GB/s | Dense BF16 Tensor TFLOPS, FP32 accumulation | Catalogue $/hour | Dense peak / hourly price | Board power limit |
|---|---:|---:|---:|---:|---:|---:|
| RTX 3090 | 24 GB | 936 | 71.2 | 0.22 | 324 | 350 W |
| RTX 4090 | 24 GB | 1008 | 165.2 | 0.34 | 486 | 450 W |
| RTX 5090 | 32 GB | 1792 | 209.5 | 0.69 | 304 | 575 W |
| RTX 6000 Ada | 48 GB | 960 | 364 | 0.74 | 492 | 300 W |
| A100 SXM 80 GB | 80 GB | 2039 | 312 | 1.39 | 224 | 400 W standard |
| RTX PRO 6000 Blackwell Server | 96 GB | 1597 | See qualification below | 1.69 | — | Up to 600 W |
| RTX PRO 6000 Blackwell Workstation | 96 GB | 1792 | 503.8 | 1.69 | 298 | 600 W |
| H100 SXM | 80 GB | 3350 | 989 | 3.49 | 283 | Up to 700 W |
| B300 SXM | 288 GB advertised | Up to 8000 | 2250 | 7.89 | 285 | Configuration dependent |

The last numeric comparison column is an index of theoretical hardware capacity per quoted hourly rate. It is **not** measured Jacobians per dollar.

The GeForce figures are from NVIDIA's [RTX Blackwell architecture whitepaper, Table 3, printed pages 46–48](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf). The BF16 row explicitly specifies FP32 accumulation. It gives dense/sparse pairs of 71.2/142.4, 165.2/330.4 and 209.5/419. Accordingly, using 419 as the RTX 5090's dense BF16 peak would double the relevant number. The higher FP4/FP8 and FP16-accumulation figures describe different arithmetic.

The Ada and Blackwell Workstation figures come from NVIDIA's [RTX PRO Blackwell architecture whitepaper, Table 4, printed pages 45–47](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf). Ada's 364 is dense BF16 with FP32 accumulation; 728 is sparse. The 96 GB Workstation row is included because the saved catalogue lists that distinct model at the same Community rate as Server Edition. Its specifications should not be silently transferred to Server Edition.

The [A100 datasheet, page 1](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf) gives 312 dense versus 624 sparse BF16 and distinguishes SXM 80 GB bandwidth from PCIe. H100's memory and power are from the [H100 specifications](https://www.nvidia.com/en-eu/data-center/h100/); that page's 1979 BF16 figure is marked sparse. NVIDIA's [Exemplar Performance reference table](https://github.com/NVIDIA/exemplar-performance/blob/main/README.md#peak-theoretical-throughput) explicitly supplies non-sparse BF16 peaks of 989 for H100 and 2250 for B300. B300's advertised memory and bandwidth are from the [HGX AI Factory components table](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html). Usable memory and bandwidth can differ by deployment; the Exemplar reference system describes 270 GB and 7.7 TB/s for its B300 configuration.

For Server Edition, the [NVIDIA product page](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/) advertises 1 PFLOP for FP16/BF16 without an adjacent sparsity qualification. The linked [December 2025 NVIDIA datasheet, page 3](https://dam-cdn.nvd.orangelogic.com/AssetLink/707m1632ypg4du1fj3ci1jo3h4w1k78j.pdf) confirms 96 GB, 1597 GB/s, 120 FP32 TFLOPS and configurable power up to 600 W, but omits BF16. This bounded review therefore does not promote that rounded 1 PFLOP advertisement into a verified dense BF16 value or substitute the Workstation's 503.8. This uncertainty is immaterial to the main Ada-versus-4090 recommendation.

The practical judgment follows from three comparisons:

1. **Ada versus 4090:** the price ratio is 2.17647; the dense BF16 peak ratio is 2.20339. Ada's peak-per-dollar advantage is only 1.24%. Its bandwidth ratio is 0.95238. The case for trying Ada first rests on doubled VRAM enabling a larger VJP batch and fewer serial backward calls. The strongest counterargument is that bandwidth, launch overhead, host copying or poor kernel utilization could leave the 4090 substantially cheaper. A larger batch does not eliminate Jacobian entries or guarantee fewer total transfer bytes.
2. **5090 versus 4090:** the 5090 costs 2.02941 times as much, while its relevant tensor peak and memory bandwidth are 1.26816 and 1.77778 times as large. Those specifications alone do not support choosing it for lower cost. Extra memory and implementation-specific behavior would need to provide the remaining benefit.
3. **Larger accelerators:** A100, H100 and B300 offer more memory, but their quoted prices require over 4.09, 10.26 and 23.21 times the 4090's fit throughput, respectively, to be cheaper. They need a measured utilization or batching benefit; their inference marketing benchmarks do not answer this Jacobian workload question. L40S has the same 48 GB capacity as Ada and lower 864 GB/s bandwidth at a higher $0.79 Community rate, so these facts supply no reason to prefer it initially. [NVIDIA L40S specifications](https://www.nvidia.com/en-us/data-center/l40s/)

The decisive measurement is elapsed time and peak memory for the **same complete Jacobian workload**, including its normal CPU reductions and copies, at each feasible batch size. Warm-up and model download time should be reported separately. Record the actual GPU, power limit, software, attention implementation and host allocation. The choice should be revised if measured dollars per completed fit disagree with this starting recommendation. No GPU benchmark or paid action was performed for this hardware note.
