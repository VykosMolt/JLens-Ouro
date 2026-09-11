"""Numerical isolation and short timing of the modern reduced-source candidate."""

from __future__ import annotations

import gc
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
from reduced_vjp_candidate import prepare_reduced_vjp


def difference(actual, expected):
    actual, expected = actual.double(), expected.double()
    delta = actual - expected
    norms = expected.flatten(1).norm(dim=1).clamp_min(1e-30)
    errors = delta.flatten(1).norm(dim=1) / norms
    return {
        "bitwise": torch.equal(actual, expected),
        "finite": bool(torch.isfinite(actual).all().item()),
        "max_absolute": delta.abs().max().item(),
        "relative_frobenius": delta.norm().item() / expected.norm().clamp_min(1e-30).item(),
        "max_layer_relative": errors.max().item(),
        "max_layer": int(errors.argmax().item()),
    }


def reference_one(model):
    ids = model.encode(TEXT, max_length=128)
    norms = []
    handle = model._final_norm.register_forward_hook(lambda module, args, output: norms.append(output.detach().cpu()))
    try:
        with ActivationRecorder(model.layers, at=range(192), start_graph_at=0) as rec:
            model.forward(ids)
            primal = {f"layer:{layer}": (rec.activations[layer].detach().cpu(),) for layer in range(192)}
            primal["module:loop_norm"] = tuple(norms)
            target = rec.activations[191]
            sources = [rec.activations[layer] for layer in range(191)]
            arrays = []
            for dim in (0, 1):
                cot = torch.zeros_like(target)
                cot[:, 16:-1, dim] = 1
                gradients = torch.autograd.grad(target, sources, cot, retain_graph=True)
                arrays.append(torch.stack([grad[0, 16:-1].float().mean(0).cpu() for grad in gradients]))
            return torch.stack(arrays, dim=1), primal
    finally:
        handle.remove()


def main():
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch._C._debug_only_display_vmap_fallback_warnings(True)
    here = Path(__file__).resolve().parent
    model = load_ouro()
    result = {"started_utc": datetime.now(timezone.utc).isoformat(), "purpose": __doc__,
              "device": torch.cuda.get_device_name(0), "attention": model.hf_model.config._attn_implementation}
    samples = {}
    reference, primals = reference_one(model)
    samples["native_scalar_b1"] = reference.numpy()
    previous = np.load(here / "bench_vjp.npz")
    result["native_b1_vs_b2"] = difference(reference, torch.from_numpy(previous["replicated_2"]))
    result["legacy_batched_vs_native_b1"] = difference(torch.from_numpy(previous["batched_2"]), reference)
    gc.collect()
    torch.cuda.empty_cache()
    with prepare_reduced_vjp(model, TEXT, list(range(191)), audit_primal=True,
                             audit_modules={"loop_norm": model._final_norm}) as context:
        primal_errors = {}
        for name, expected in primals.items():
            actual = context.primal_audit[name]
            assert len(actual) == len(expected), name
            pairs = [difference(a.detach().cpu().float(), e.float()) for a, e in zip(actual, expected)]
            primal_errors[name] = pairs
        result["primal_errors"] = primal_errors
        scalar = []
        for dim in (0, 1):
            cot = torch.zeros(model.d_model, device=context.device)
            cot[dim] = 1
            values = context.apply_scalar(cot)
            scalar.append(torch.stack([values[layer].cpu() for layer in range(191)]))
        scalar = torch.stack(scalar, dim=1)
        samples["reduced_scalar"] = scalar.numpy()
        result["reduced_scalar_vs_native_b1"] = difference(scalar, reference)
        values = context.rows(0, 2)
        batched = torch.stack([values[layer].cpu() for layer in range(191)])
        samples["reduced_batched_2"] = batched.numpy()
        result["reduced_batched_vs_scalar"] = difference(batched, scalar)
        result["reduced_batched_vs_native_b1"] = difference(batched, reference)
    del context, values, primals
    gc.collect()
    torch.cuda.empty_cache()
    result["timings"] = []
    with prepare_reduced_vjp(model, TEXT, list(range(191))) as context:
        for batch in (2, 8, 32, 64):
            torch.cuda.reset_peak_memory_stats()
            rows_times = []
            try:
                for sweep in range(3):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    values = context.rows(sweep * batch, batch)
                    torch.cuda.synchronize()
                    after = time.perf_counter()
                    cpu = torch.stack(list(values.values())).cpu()
                    torch.cuda.synchronize()
                    rows_times.append({"backward": after - start, "total": time.perf_counter() - start})
                    if sweep == 0:
                        samples[f"reduced_batched_{batch}"] = cpu[:, :2].numpy()
                    del values, cpu
                mean = {key: statistics.mean(row[key] for row in rows_times[1:]) for key in rows_times[0]}
                record = {"batch": batch, "sweeps": rows_times, "mean_after_warmup": mean,
                          "estimated_prompt_seconds": mean["total"] * ((2048 + batch - 1) // batch),
                          "peak_cuda_gb": torch.cuda.max_memory_allocated() / 1e9, "executed": True}
            except Exception as exc:
                traceback.print_exc()
                record = {"batch": batch, "executed": False, "error": f"{type(exc).__name__}: {exc}"}
            result["timings"].append(record)
            print(json.dumps(record), flush=True)
            (here / "bench_reduced.json").write_text(json.dumps(result, indent=2) + "\n")
            np.savez(here / "bench_reduced.npz", **samples)
            gc.collect()
            torch.cuda.empty_cache()
    print(json.dumps({key: value for key, value in result.items() if key not in ("timings", "primal_errors")}), flush=True)


if __name__ == "__main__":
    main()
