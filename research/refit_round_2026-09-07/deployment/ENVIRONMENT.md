# Exact-runtime portability evidence

The exact Torch and Triton wheel objects remain available on the original PyTorch download host. This identifies an exact-version bootstrap route; a fresh environment and rental GPU have not yet been validated. No wheel archives, model weights or packages were downloaded or installed during this audit.

The successful public responses were observed on **8 September 2026 at 09:28:41–42 UTC**. Full observations and version pins are in [environment_portability.json](environment_portability.json).

## Installed runtime

The accepted runtime is CPython **3.14.7**, Torch **2.12.0.dev20260407+cu128** (git `6a13e444ee88996ff01cd2bab41d7f2857291646`), Transformers **4.54.1**, NumPy **2.4.4**, tokenizers **0.21.4**, safetensors **0.7.0**, huggingface-hub **0.36.2** and accelerate **1.13.0**. Torch's wheel tag is `cp314-cp314-manylinux_2_28_x86_64`: use normal GIL-enabled CPython 3.14 on x86_64 Linux with glibc 2.28 or later.

Torch reports CUDA **12.8** and cuDNN runtime **92000**; installed CUDA packages include cuDNN **9.20.0.48** and cuBLAS **12.8.4.1**. The JSON records the complete CUDA library package closure and Torch build configuration. CUDA remained uninitialized throughout the CPU inspection.

The venv points to `/usr/bin/python3.14`. Its creation file records 3.14.4 while the current interpreter is 3.14.7, so copying the venv alone would omit its interpreter and standard library. The [official Python 3.14.7 release](https://www.python.org/downloads/release/python-3147/) is available; no specific Linux container image or interpreter build was created or certified here.

## Exact wheel evidence

| Package | Official exact wheel | HEAD bytes | Published metadata |
| --- | --- | ---: | --- |
| Torch | [2.12.0.dev20260407+cu128, cp314](https://download.pytorch.org/whl/nightly/cu128/torch-2.12.0.dev20260407%2Bcu128-cp314-cp314-manylinux_2_28_x86_64.whl) | 832,494,036 | Matches installed metadata byte for byte |
| Triton | [3.7.0+git9c288bc5, cp314](https://download.pytorch.org/whl/nightly/triton-3.7.0%2Bgit9c288bc5-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl) | 292,180,125 | Matches installed metadata byte for byte |

Only HEAD responses and the small `.whl.metadata` sidecars were read. Sidecar SHA256 values are:

- Torch: `bb8cc0a6d990b45cf1a2fb7a8115f9204bf12012dbb9fcb2bf398d84568931b5`.
- Triton: `15be25c2c8ced19d19fad33dc94ce3fab99254f0546bb0a8b4f7496ea5c091db`.

These are **metadata hashes, not archive hashes**. No wheel SHA256 has been acquired; multipart ETags must not be presented as SHA256 values.

The [nightly Torch index](https://download.pytorch.org/whl/nightly/cu128/torch/) lists the next-day build, while the exact April 7 object remains reachable at its direct original-host URL. The matching Triton object is also retained despite absence from the inspected [Triton index](https://download.pytorch.org/whl/nightly/cu128/triton/). Corresponding `download-r2.pytorch.org` HEAD checks returned 403. No cached wheel files were found in the inspected pip/uv caches or readable temporary/project paths; the JSON records exclusions and inaccessible-directory counts.

## Portable setup recommendation

Use a pinned CPython 3.14.7 Linux environment with the exact Torch/Triton URLs and dependency versions recorded in the JSON. The Torch/Transformers/NumPy closure contains **44 packages**, with all active installed requirements satisfied. Retaining accelerate and psutil gives **46 packages**, matching existing model-loading package availability. The JSON supplies both exact version pins and a requirements recommendation using direct wheel URLs. Other package archive availability and hashes remain to be checked during staging. An unavailable pin must fail explicitly rather than select a newer version.

The accepted engine uses ordinary autograd, gradient edges, saved-tensor hooks and CUDA graph replay. It does not require CMake, Ninja, NVCC, `torch.compile`, custom Triton compilation, flash-attn, torchvision, torchaudio or Inductor caches. Triton remains pinned because Torch declares it as a runtime dependency. Frozen calibration JSON lists remove the need for datasets/pyarrow on this fitting path. Huginn adapter dependencies are outside this Ouro audit.

The local `jlens 0.1.0` metadata declares `transformers>=5.5`, conflicting with the accepted 4.54.1 runtime; `pip check` confirms that single conflict. Stage the exact `jlens` and `ouro_jlens` sources through `PYTHONPATH`, or install local packages with `--no-deps` after fixing runtime dependencies. A normal dependency-resolving install could upgrade Transformers and change the validated runtime.

## Remaining checks

The exact wheel route has not been installed in a fresh environment. Verify archive hashes, a concrete interpreter/base-image identity and the complete package installation during staging. The rental then needs matching-batch numerical parity, memory and complete-prompt timing checks with driver/device, precision/attention flags and source hashes recorded. Identical package versions on a 4090 do not establish identical kernels or numerical results to the laptop. Validate any changed runtime or numerical configuration and keep the selected configuration fixed across all five N100 fits.
