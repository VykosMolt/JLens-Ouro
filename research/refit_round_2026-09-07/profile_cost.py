"""Short local timing decomposition; no fit or task-recovery evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, "/home/moloch/ouro_project/src")
from ouro_jlens.bench import TEXT
from ouro_jlens.recurrent import load_ouro
from jlens.hooks import ActivationRecorder


def main() -> None:
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available()
    result = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": __doc__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dim_batch": 2,
        "warmup_sweeps": 2,
        "measured_sweeps": 6,
        "cpu_threads": 8,
    }
    t0 = time.perf_counter()
    model = load_ouro()
    result["load_seconds"] = time.perf_counter() - t0
    sources = list(range(model.n_layers - 1))
    width = model.d_model
    ids = model.encode(TEXT, max_length=128)
    result.update(
        source_count=len(sources),
        width=width,
        sequence_length=ids.shape[1],
        full_prompt_backward_calls=math.ceil(width / 2),
    )
    t0 = time.perf_counter()
    matrices = {layer: torch.zeros(width, width) for layer in sources}
    result["cpu_zero_matrices_seconds"] = time.perf_counter() - t0
    measurements = []
    torch.cuda.reset_peak_memory_stats()
    with ActivationRecorder(model.layers, at=range(model.n_layers), start_graph_at=0) as rec:
        t0 = time.perf_counter()
        with torch.enable_grad():
            model.forward(ids.expand(2, -1))
        torch.cuda.synchronize()
        result["forward_seconds"] = time.perf_counter() - t0
        target = rec.activations[model.n_layers - 1]
        inputs = [rec.activations[layer] for layer in sources]
        positions = torch.arange(16, ids.shape[1] - 1, device=target.device)
        lanes = torch.arange(2, device=target.device)
        cotangent = torch.zeros_like(target)
        for sweep in range(8):
            first_row = sweep * 2
            cotangent.zero_()
            cotangent[lanes[:, None], positions[None, :], (first_row + lanes)[:, None]] = 1
            torch.cuda.synchronize()
            start = time.perf_counter()
            grads = torch.autograd.grad(target, inputs, cotangent, retain_graph=True)
            torch.cuda.synchronize()
            after_backward = time.perf_counter()
            for layer, grad in zip(sources, grads, strict=True):
                rows = grad[:, positions, :].float().mean(dim=1)
                matrices[layer][first_row : first_row + 2, :] = rows.cpu()
            torch.cuda.synchronize()
            after_rows = time.perf_counter()
            measurements.append({
                "sweep": sweep,
                "backward_seconds": after_backward - start,
                "reduce_copy_assign_seconds": after_rows - after_backward,
                "total_seconds": after_rows - start,
            })
            del grads, grad, rows
    result["peak_cuda_gb"] = torch.cuda.max_memory_allocated() / 1e9
    result["sweeps"] = measurements
    result["mean_after_warmup"] = {
        key: statistics.mean(row[key] for row in measurements[2:])
        for key in ("backward_seconds", "reduce_copy_assign_seconds", "total_seconds")
    }
    result["extrapolated_full_prompt_sweeps_seconds"] = (
        result["mean_after_warmup"]["total_seconds"] * result["full_prompt_backward_calls"]
    )
    # The matrix shape/CPU operations match fitting. Only 16 rows were filled;
    # their values are not a fitted estimator and are neither saved nor scored.
    t0 = time.perf_counter()
    running_sum = {layer: matrix.clone() for layer, matrix in matrices.items()}
    result["cpu_clone_running_sum_seconds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = max(matrix.norm().item() for matrix in matrices.values()) / math.sqrt(width)
    result["cpu_prompt_norm_seconds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = max(
        ((matrices[layer] - running_sum[layer] / 10).norm()
         / (11 * (running_sum[layer] / 10).norm())).item()
        for layer in sources
    )
    result["cpu_running_change_seconds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    for layer in sources:
        running_sum[layer] += matrices[layer]
    result["cpu_accumulate_seconds"] = time.perf_counter() - t0
    result["cpu_timing_qualification"] = "Full matrix shapes, only 16 rows filled; no lens result. Checkpoint I/O not timed."
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
