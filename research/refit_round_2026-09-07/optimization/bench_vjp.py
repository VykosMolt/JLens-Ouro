"""Compare repeated primal batches with one-primal batched VJPs, without fitting."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/moloch/ouro_project/src")
from ouro_jlens.bench import TEXT
from ouro_jlens.recurrent import load_ouro
from jlens.hooks import ActivationRecorder


def measure(model, mode, batch, sweeps=3, target_layer=None):
    target_layer = model.n_layers - 1 if target_layer is None else target_layer
    sources = list(range(target_layer))
    ids = model.encode(TEXT, max_length=128)
    primal_batch = 1 if mode == "batched" else batch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    sample = None
    records = []
    with ActivationRecorder(model.layers, at=[*sources, target_layer], start_graph_at=0) as rec:
        with torch.enable_grad():
            model.forward(ids.expand(primal_batch, -1))
        torch.cuda.synchronize()
        forward = time.perf_counter() - start
        target = rec.activations[target_layer]
        source_values = [rec.activations[layer] for layer in sources]
        positions = torch.arange(16, ids.shape[1] - 1, device=target.device)
        lanes = torch.arange(batch, device=target.device)
        shape = (batch, *target.shape) if mode == "batched" else target.shape
        cotangent = torch.zeros(shape, dtype=target.dtype, device=target.device)
        for sweep in range(sweeps):
            first = sweep * batch
            cotangent.zero_()
            if mode == "batched":
                cotangent[lanes[:, None], 0, positions[None, :], (first + lanes)[:, None]] = 1
            else:
                cotangent[lanes[:, None], positions[None, :], (first + lanes)[:, None]] = 1
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            grads = torch.autograd.grad(
                target, source_values, cotangent, retain_graph=True,
                is_grads_batched=mode == "batched",
            )
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            rows = []
            for grad in grads:
                value = grad[:, 0] if mode == "batched" else grad
                rows.append(value[:, positions, :].float().mean(1).cpu())
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            if sweep == 0:
                sample = torch.stack([row[:min(2, batch)] for row in rows]).numpy()
            records.append({"backward": t1 - t0, "reduce_copy": t2 - t1, "total": t2 - t0})
            del grads, grad, value, rows
    mean = {key: statistics.mean(row[key] for row in records[1:]) for key in records[0]}
    return {
        "mode": mode, "batch": batch, "source_count": len(sources), "target": target_layer,
        "forward_seconds": forward, "sweeps": records, "mean_after_warmup": mean,
        "estimated_prompt_seconds": mean["total"] * ((model.d_model + batch - 1) // batch),
        "peak_cuda_gb": torch.cuda.max_memory_allocated() / 1e9,
        "sample_finite": bool(np.isfinite(sample).all()),
    }, sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, nargs="+", default=[2, 8, 32])
    parser.add_argument("--reference-batch", type=int, default=2)
    parser.add_argument("--attention", choices=["native", "math"], default="native")
    parser.add_argument("--fallback-warnings", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available()
    if args.fallback_warnings:
        torch._C._debug_only_display_vmap_fallback_warnings(True)
    model = load_ouro()
    result = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": __doc__, "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "torch_git": torch.version.git_version,
        "attention": model.hf_model.config._attn_implementation,
        "sdpa_override": args.attention,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "records": [],
    }
    samples = {}
    for mode, batch in [("replicated", args.reference_batch), *[("batched", b) for b in args.batches]]:
        label = f"{mode}_{batch}"
        try:
            attention_context = (
                torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
                if args.attention == "math" else nullcontext()
            )
            with attention_context:
                record, sample = measure(model, mode, batch)
            record["pass"] = True
            samples[label] = sample
        except Exception as exc:
            record = {"mode": mode, "batch": batch, "pass": False,
                      "error": f"{type(exc).__name__}: {exc}"}
            traceback.print_exc()
        result["records"].append(record)
        print(json.dumps(record), flush=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        np.savez(args.output.with_suffix(".npz"), **samples)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
