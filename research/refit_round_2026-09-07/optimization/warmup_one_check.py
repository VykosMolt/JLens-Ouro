"""Cold-process Ouro B8 smoke check for exactly one discarded CUDA warmup.

Run explicitly with --run after reserving the GPU. No earlier autograd.grad is
allowed, including during model loading. Selected captured rows are copied to
CPU, then the graph is released before matching eager VJPs run on the retained
primal. This is a sampled warmup check, not a complete fit or engine validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for location in (str(REPO), "/home/moloch/ouro_project/src", str(HERE)):
    if location not in sys.path:
        sys.path.insert(0, location)


def write_result(path, result):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def compare(actual, expected):
    import torch

    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("comparison tensors must have matching shape and dtype")
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    unequal = actual.view(torch.int32) != expected.view(torch.int32)
    record = {"bitwise_equal": not bool(unequal.any()), "finite": finite,
              "values_compared": actual.numel(), "different_bits_count": int(unequal.sum())}
    if record["different_bits_count"]:
        record["first_mismatch_source_lane_hidden"] = unequal.nonzero()[0].tolist()
        if finite:
            record["max_absolute"] = (actual.double() - expected.double()).abs().max().item()
    return record


def run(args):
    import torch
    from ouro_jlens import recurrent
    from ouro_jlens.bench import TEXT

    from benchmark_optimized import (
        check_reference_metadata,
        load_reference_metadata,
        precision_snapshot,
        source_hashes,
        validate_maps,
    )
    from cuda_graph_candidate import CapturedVJP, DenseCotangentWriter, dense_source_means
    from optimized_fitting import SourceEdgeRecorder
    from saved_tensor_candidate import ReplicatedSavedTensorHooks

    if torch.cuda.is_initialized():
        raise RuntimeError("start this driver in a fresh process with CUDA uninitialized")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    sources, target, batch, length, skip_first = list(range(191)), 191, 8, 128, 16
    hashes = source_hashes()
    hashes[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    hashes["ouro_jlens/recurrent.py"] = hashlib.sha256(Path(recurrent.__file__).read_bytes()).hexdigest()
    result = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "purpose": "Cold-process warmup=1; first/middle/last B8 rows, repeat and zero",
        "cuda_initialized_on_entry": False, "source_sha256": hashes,
        "torch": str(torch.__version__), "torch_git": torch.version.git_version,
        "cpu_threads": torch.get_num_threads(), "cpu_interop_threads": torch.get_num_interop_threads(),
        "dim_batch": batch, "max_seq_len": length, "skip_first": skip_first,
        "target": target, "source_count": len(sources), "source_layers": sources,
        "compress": True, "source_edges": True, "engine": "cuda_graph", "warmup_calls": 1,
        "prompt_sha256": hashlib.sha256(TEXT.encode()).hexdigest(),
        "calibration_count_in_research": 0, "full_width_complete": False,
        "reference": {"path": str(args.reference), "status": "unavailable"},
        "comparisons": [],
    }
    started = time.perf_counter()
    original_grad = torch.autograd.grad
    phase = "before_capture"
    calls = {"before_capture": 0, "constructor": 0, "replay": 0, "eager": 0}

    def counted_grad(*grad_args, **grad_kwargs):
        calls[phase] += 1
        if phase == "before_capture":
            raise RuntimeError("an autograd.grad call preceded the cold capture constructor")
        return original_grad(*grad_args, **grad_kwargs)

    write_result(args.output, result)
    captured = None
    stream = None
    try:
        with patch.object(torch.autograd, "grad", counted_grad):
            metadata = None
            reference = None
            if args.reference.is_file():
                metadata, provenance = load_reference_metadata(args.reference)
                check_reference_metadata(metadata, {name: result[name] for name in (
                    "torch", "torch_git", "dim_batch", "max_seq_len", "skip_first",
                    "target", "source_count", "source_layers", "compress", "source_edges",
                    "engine", "prompt_sha256",
                )})
                check_reference_metadata(metadata, {"full_width_complete": True})
                result["reference"].update(metadata=provenance, saved_status=metadata.get("status"),
                                           artifact_bytes=args.reference.stat().st_size,
                                           source_sha256=metadata.get("source_sha256"))
            load_started = time.perf_counter()
            model = recurrent.load_ouro()
            ids = model.encode(TEXT, max_length=length)
            if tuple(ids.shape) != (1, length) or model.d_model != 2048 or model.n_layers != 192:
                raise ValueError("the frozen Ouro T128/D2048/192-layer fixture changed")
            result.update(load_seconds=time.perf_counter() - load_started,
                          device=torch.cuda.get_device_name(ids.device), model_revision=model.model_revision,
                          attention=model.hf_model.config._attn_implementation, d_model=model.d_model,
                          model_dtype=str(next(model.hf_model.parameters()).dtype),
                          model_training=model.hf_model.training, precision_flags=precision_snapshot(),
                          sequence_length=length, n_valid=length - skip_first - 1,
                          token_ids_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest())
            if result["model_training"] or result["model_dtype"] != "torch.bfloat16":
                raise ValueError("the frozen eval/BF16 model configuration changed")
            if metadata is not None:
                check_reference_metadata(metadata, {name: result[name] for name in (
                    "device", "model_revision", "attention", "d_model", "sequence_length",
                    "n_valid", "precision_flags",
                )})
                reference = torch.load(args.reference, map_location="cpu", mmap=True, weights_only=True)
                validate_maps(reference, model.d_model, "reference")
                result["reference"]["status"] = "metadata_validated"
            storage = ReplicatedSavedTensorHooks(model, batch_size=batch, seq_len=length)
            stream = torch.cuda.Stream(device=ids.device)
            stream.wait_stream(torch.cuda.current_stream(ids.device))
            torch.cuda.reset_peak_memory_stats(ids.device)
            cases = [("first", 0), ("middle", 1024), ("tail", 2040), ("repeat_first", 0), ("zero", None)]
            with torch.cuda.stream(stream), torch.enable_grad(), SourceEdgeRecorder(
                model.layers, sources=sources, target=target,
            ) as recorder:
                forward_started = time.perf_counter()
                with storage:
                    model.forward(ids.expand(batch, -1))
                stream.synchronize()
                result["forward_seconds"] = time.perf_counter() - forward_started
                if set(recorder.metadata) != {*sources, target}:
                    raise RuntimeError("source/target hooks did not fire exactly as requested")
                if any(meta != ((batch, length, model.d_model), ids.device, torch.bfloat16)
                       for meta in recorder.metadata.values()):
                    raise RuntimeError("source/target metadata differs from the frozen fixture")
                activation = recorder.activations[target]
                edges = tuple(recorder.edges[layer] for layer in sources)
                cotangent = torch.zeros_like(activation)
                writer = DenseCotangentWriter(cotangent, skip_first=skip_first)

                def reduce(grads):
                    return dense_source_means(grads, writer.valid_positions)

                writer.write(0)
                phase = "constructor"
                captured = CapturedVJP(activation, edges, cotangent, stream=stream,
                                      warmup=1, reducer=reduce)
                if calls["constructor"] != 2:
                    raise RuntimeError("expected exactly one warmup and one capture VJP")
                result.update(warmup_seconds=captured.warmup_seconds, capture_seconds=captured.capture_seconds)
                static_writer = DenseCotangentWriter(captured.cotangent, skip_first=skip_first)
                phase = "replay"
                captured_rows = []
                for _, row in cases:
                    if row is None:
                        captured.cotangent.zero_()
                    else:
                        static_writer.write(row)
                    captured_rows.append(captured.replay()[0].cpu())
                result["replay_calls"] = captured.replay_calls
                captured.close()
                captured = None
                # Release every remaining reference to the graph's cotangent.
                del static_writer
                phase = "eager"
                for (name, row), actual in zip(cases, captured_rows, strict=True):
                    if row is None:
                        cotangent.zero_()
                    else:
                        writer.write(row)
                    grads = torch.autograd.grad(activation, edges, cotangent,
                                                retain_graph=True, create_graph=False, allow_unused=False)
                    expected = reduce(grads)[0].cpu()
                    del grads
                    record = {"case": name, "dim_start": row, "eager": compare(actual, expected)}
                    if reference is not None and name in ("first", "middle", "tail"):
                        saved = torch.stack([reference[layer][row:row + batch] for layer in sources])
                        record["saved_artifact"] = compare(actual, saved)
                    result["comparisons"].append(record)
                result["repeat"] = compare(captured_rows[3], captured_rows[0])
                result["zero"] = compare(captured_rows[4], torch.zeros_like(captured_rows[4]))
                stream.synchronize()
            if calls != {"before_capture": 0, "constructor": 2, "replay": 0, "eager": 5}:
                raise RuntimeError(f"unexpected autograd.grad call counts: {calls}")
            result["storage"] = storage.stats()
            result["storage_counter_scope"] = "Python warmup/capture/eager calls; replays do not increment unpack counters"
            checks = [record[kind] for record in result["comparisons"]
                      for kind in ("eager", "saved_artifact") if kind in record]
            checks.extend((result["repeat"], result["zero"]))
            passed = all(check["bitwise_equal"] and check["finite"] for check in checks)
            result["status"] = "passed" if passed else "mismatch"
            result["reference"]["status"] = ("sampled_rows_passed" if all(
                record["saved_artifact"]["bitwise_equal"] and record["saved_artifact"]["finite"]
                for record in result["comparisons"] if "saved_artifact" in record
            ) else "sampled_rows_mismatch") if reference is not None else "unavailable"
    except Exception as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        try:
            if captured is not None:
                captured.close()
            if stream is not None:
                stream.synchronize()
        except Exception as error:
            result.update(status="error", cleanup_error=f"{type(error).__name__}: {error}")
        result.update(autograd_grad_calls=calls, elapsed_seconds=time.perf_counter() - started,
                      completed_utc=datetime.now(timezone.utc).isoformat())
        if torch.cuda.is_initialized():
            result.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        write_result(args.output, result)
    print(json.dumps({name: result[name] for name in (
        "status", "autograd_grad_calls", "reference", "elapsed_seconds",
    )}, allow_nan=False), flush=True)
    return 0 if result["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run one cold-process GPU smoke check")
    parser.add_argument("--output", type=Path, default=HERE / "warmup_one_check.json")
    parser.add_argument("--reference", type=Path, default=HERE / "optimized_full_b8.pt")
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    if args.output.resolve() in {args.reference.resolve(), args.reference.with_suffix(".json").resolve(), Path(__file__).resolve()}:
        parser.error("output must not overwrite the reference, its metadata, or this driver")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
