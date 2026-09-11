"""Bounded native-batch memory/capture probe; no full fit or scientific result."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
import traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/moloch/ouro_project/src")
from ouro_jlens.bench import TEXT
from ouro_jlens.recurrent import load_ouro

from cuda_graph_candidate import CapturedVJP, DenseCotangentWriter, dense_source_means
from optimized_fitting import SourceEdgeRecorder
from saved_tensor_candidate import ReplicatedSavedTensorHooks


def measure(model, batch, compress, sweeps=6):
    ids = model.encode(TEXT, max_length=128)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    storage = (ReplicatedSavedTensorHooks(model, batch_size=batch, seq_len=ids.shape[1])
               if compress else None)
    result = {"batch": batch, "compress": compress, "source_edges": True,
              "source_count": 191, "target": 191, "seq_len": int(ids.shape[1])}
    torch.cuda.reset_peak_memory_stats()
    with torch.cuda.stream(stream), torch.enable_grad(), SourceEdgeRecorder(
        model.layers, sources=list(range(191)), target=191
    ) as recorder:
        started = time.perf_counter()
        with storage if storage is not None else nullcontext():
            model.forward(ids.expand(batch, -1))
        stream.synchronize()
        result["forward_seconds"] = time.perf_counter() - started
        result["after_forward_gb"] = torch.cuda.memory_allocated() / 1e9
        target = recorder.activations[191]
        sources = tuple(recorder.edges[layer] for layer in range(191))
        cot = torch.zeros_like(target)
        writer = DenseCotangentWriter(cot)
        writer.write(0)

        def reducer(grads):
            return dense_source_means(grads, writer.valid_positions)

        def eager():
            return reducer(torch.autograd.grad(target, sources, cot, retain_graph=True))[0]

        expected = eager().cpu()
        times = []
        for sweep in range(sweeps):
            writer.write(sweep * batch)
            stream.synchronize()
            started = time.perf_counter()
            rows = eager().cpu()
            stream.synchronize()
            times.append(time.perf_counter() - started)
            del rows
        result["eager_times"] = times
        result["eager_mean"] = statistics.mean(times[1:])
        result["eager_estimated_full_backward_seconds"] = result["eager_mean"] * ((2048 + batch - 1) // batch)
        result["eager_peak_cuda_gb"] = torch.cuda.max_memory_allocated() / 1e9
        writer.write(0)
        with CapturedVJP(target, sources, cot, stream=stream, reducer=reducer) as captured:
            result["warmup_seconds"] = captured.warmup_seconds
            result["capture_seconds"] = captured.capture_seconds
            actual = captured.replay()[0].cpu()
            result["capture_same_graph_bitwise"] = torch.equal(actual, expected)
            sample = actual[:, :2].numpy()
            del actual, expected
            static = DenseCotangentWriter(captured.cotangent)
            times = []
            for sweep in range(sweeps):
                static.write(sweep * batch)
                stream.synchronize()
                started = time.perf_counter()
                rows = captured.replay()[0].cpu()
                stream.synchronize()
                times.append(time.perf_counter() - started)
                del rows
            result["capture_times"] = times
            result["capture_mean"] = statistics.mean(times[1:])
            result["capture_estimated_full_prompt_seconds"] = (
                result["forward_seconds"] + result["warmup_seconds"] + result["capture_seconds"]
                + result["capture_mean"] * ((2048 + batch - 1) // batch))
            result["peak_cuda_gb"] = torch.cuda.max_memory_allocated() / 1e9
    result["storage"] = storage.stats() if storage is not None else None
    return result, sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[2, 8, 16, 32])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    model = load_ouro()
    result = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": __doc__, "device": torch.cuda.get_device_name(0), "records": []}
    reference = np.load(Path(__file__).with_name("bench_vjp.npz"))["replicated_2"]
    samples = {}
    for batch, compress in [(2, False), *[(batch, True) for batch in args.batches]]:
        try:
            record, sample = measure(model, batch, compress)
            record["executed"] = True
            record["sample_all_finite"] = bool(np.isfinite(sample).all())
            if batch == 2:
                record["stock_b2_first_rows_bitwise"] = bool(np.array_equal(reference, sample))
            samples[f"b{batch}_compress{compress}"] = sample
        except Exception as exc:
            traceback.print_exc()
            record = {"batch": batch, "compress": compress, "executed": False,
                      "error": f"{type(exc).__name__}: {exc}"}
        result["records"].append(record)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        np.savez(args.output.with_suffix(".npz"), **samples)
        print(json.dumps({key: value for key, value in record.items() if key != "storage"}), flush=True)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
