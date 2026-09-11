"""Complete same-shape fitter benchmark on the fixed Ouro fixture.

No calibration/recovery example enters this benchmark. A complete reference
file is compared entry by entry using mmap only after its metadata matches.
Passing this one-paragraph comparison does not by itself accept an engine for
the research fits. Failed comparisons are explicit and exit nonzero.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from jlens.fitting import valid_position_mask

sys.path.insert(0, "/home/moloch/ouro_project/src")
from ouro_jlens.bench import TEXT
from ouro_jlens.recurrent import load_ouro

from optimized_fitting import jacobian_for_prompt


SOURCE_LAYERS = list(range(191))
TARGET_LAYER = 191
MAX_SEQ_LEN = 128
SKIP_FIRST = 16


class ReferenceValidationError(ValueError):
    """The supplied reference does not describe this exact comparison."""


def source_hashes():
    here = Path(__file__).resolve().parent
    names = ("benchmark_optimized.py", "optimized_fitting.py",
             "cuda_graph_candidate.py", "saved_tensor_candidate.py")
    result = {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in names}
    import jlens.fitting
    result["jlens/fitting.py"] = hashlib.sha256(Path(jlens.fitting.__file__).read_bytes()).hexdigest()
    return result


def precision_snapshot():
    """Read supported controls without changing flags or initializing CUDA."""
    def available(getter):
        try:
            return getter()
        except (AttributeError, RuntimeError) as error:
            return {"unavailable": type(error).__name__}

    matmul = torch.backends.cuda.matmul
    attributes = ("allow_tf32", "fp32_precision",
                  "allow_bf16_reduced_precision_reduction",
                  "allow_bf16_reduced_precision_reduction_split_k",
                  "allow_fp16_reduced_precision_reduction",
                  "allow_fp16_reduced_precision_reduction_split_k",
                  "allow_fp16_accumulation")
    sdpa = ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled",
            "cudnn_sdp_enabled", "fp16_bf16_reduction_math_sdp_allowed")
    return {
        "float32_matmul_precision": available(torch.get_float32_matmul_precision),
        "matmul": {name: available(lambda name=name: getattr(matmul, name)) for name in attributes},
        "sdpa": {name: available(lambda name=name: getattr(torch.backends.cuda, name)()) for name in sdpa},
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "sdpa_scope": "enabled implementations; not a trace of the selected runtime kernel",
    }


def load_reference_metadata(reference):
    path = reference.with_suffix(".json")
    if not reference.is_file():
        raise ReferenceValidationError(f"reference tensor file does not exist: {reference}")
    try:
        raw = path.read_bytes()
        metadata = json.loads(raw)
    except (OSError, ValueError) as error:
        raise ReferenceValidationError(f"cannot read reference metadata {path}: {error}") from error
    if not isinstance(metadata, dict):
        raise ReferenceValidationError("reference metadata must be a JSON object")
    if metadata.get("all_finite") is not True:
        raise ReferenceValidationError("reference metadata must report all_finite=true")
    if metadata.get("status") not in (None, "passed", "completed_unverified"):
        raise ReferenceValidationError(f"reference has an incomplete or failed status: {metadata.get('status')!r}")
    if "sequence_length" not in metadata and "seq_len" in metadata:
        metadata["sequence_length"] = metadata["seq_len"]
    if "seq_len" in metadata and metadata.get("sequence_length") != metadata["seq_len"]:
        raise ReferenceValidationError("reference sequence_length and seq_len disagree")
    return metadata, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def check_reference_metadata(metadata, expected):
    mismatches = {}
    for name, value in expected.items():
        observed = metadata.get(name)
        # A JSON boolean must not pass an integer comparison as 0 or 1.
        if type(observed) is not type(value) or observed != value:
            mismatches[name] = {"expected": value, "observed": observed}
    if mismatches:
        raise ReferenceValidationError(f"reference metadata mismatch: {json.dumps(mismatches)}")


def validate_maps(maps, d_model, label):
    error_type = ReferenceValidationError if label == "reference" else ValueError
    if not isinstance(maps, dict) or any(type(layer) is not int for layer in maps):
        raise error_type(f"{label} must be a dictionary with integer source keys")
    if set(maps) != set(SOURCE_LAYERS):
        raise error_type(f"{label} source keys must be exactly 0..190")
    for layer, matrix in maps.items():
        if (not isinstance(matrix, torch.Tensor) or matrix.shape != (d_model, d_model)
                or matrix.dtype != torch.float32 or matrix.device.type != "cpu"):
            raise error_type(f"{label} source {layer} must be CPU float32 [{d_model},{d_model}]")


def compare_maps(maps, reference, d_model):
    validate_maps(reference, d_model, "reference")
    parity = []
    for layer, matrix in maps.items():
        expected = reference[layer]
        finite = bool(torch.isfinite(expected).all().item())
        record = {"layer": layer, "reference_finite": finite,
                  "numerically_equal": torch.equal(matrix, expected),
                  "bitwise_equal": torch.equal(matrix.view(torch.int32), expected.view(torch.int32))}
        if finite and not record["bitwise_equal"]:
            delta = matrix.double() - expected.double()
            record.update(max_absolute=delta.abs().max().item(),
                          relative_frobenius=delta.norm().item() / expected.double().norm().clamp_min(1e-30).item())
        parity.append(record)
    return parity


def write_result(path, info):
    path.write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("stock", "eager", "cuda_graph"), default="cuda_graph")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()
    if args.engine == "stock" and args.compress:
        parser.error("stock reference does not apply saved-tensor compression")
    if args.save and args.save.resolve() == args.output.resolve():
        parser.error("--save and --output must be different paths")
    if args.reference:
        protected = {args.reference.resolve(), args.reference.with_suffix(".json").resolve()}
        if args.output.resolve() in protected or (args.save and args.save.resolve() in protected):
            parser.error("benchmark outputs must not overwrite the reference or its metadata")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    info = {"started_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": __doc__, "torch": torch.__version__,
            "torch_git": torch.version.git_version, "source_sha256": source_hashes(),
            "precision_flags": precision_snapshot(),
            "cpu_threads": torch.get_num_threads(), "cpu_interop_threads": torch.get_num_interop_threads(),
            "prompt_sha256": hashlib.sha256(TEXT.encode()).hexdigest(),
            "calibration_count_in_research": 0, "compress": args.compress,
            "engine": args.engine, "dim_batch": args.batch, "target": TARGET_LAYER,
            "source_count": len(SOURCE_LAYERS), "source_layers": SOURCE_LAYERS,
            "max_seq_len": MAX_SEQ_LEN, "skip_first": SKIP_FIRST,
            "status": "running", "full_width_complete": False,
            "reference_comparison_passed": False,
            "reference": str(args.reference) if args.reference else None,
            "reference_comparison_scope": "one complete fixed paragraph at matching batch, checkpoint, target and positions; not engine adoption"}
    write_result(args.output, info)
    try:
        if args.batch < 1:
            raise ValueError("batch must be positive")
        reference_metadata = None
        if args.reference:
            reference_metadata, info["reference_metadata"] = load_reference_metadata(args.reference)
            check_reference_metadata(reference_metadata, {
                name: info[name] for name in ("dim_batch", "target", "source_count", "prompt_sha256")
            })
        model = load_ouro()
        info.update(device=torch.cuda.get_device_name(0), model_revision=model.model_revision,
                    attention=model.hf_model.config._attn_implementation,
                    d_model=model.d_model)
        length = int(model.encode(TEXT, max_length=MAX_SEQ_LEN).shape[1])
        n_valid = int(valid_position_mask(length, skip_first=SKIP_FIRST).sum())
        info.update(sequence_length=length, n_valid=n_valid)
        # Loading and encoding can set runtime options. Snapshot the effective
        # flags again immediately before the measured computation.
        info["precision_flags"] = precision_snapshot()
        if reference_metadata is not None:
            check_reference_metadata(reference_metadata, {
                name: info[name] for name in ("model_revision", "sequence_length", "n_valid")
            })
            info["reference_metadata"]["validation"] = "passed_before_full_compute"
        storage = None
        if args.compress:
            from saved_tensor_candidate import ReplicatedSavedTensorHooks
            storage = ReplicatedSavedTensorHooks(model, batch_size=args.batch, seq_len=length)
        write_result(args.output, info)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        if args.engine == "stock":
            from jlens.fitting import jacobian_for_prompt as stock
            maps, observed_length, observed_valid = stock(
                model, TEXT, SOURCE_LAYERS, target_layer=TARGET_LAYER, dim_batch=args.batch,
                max_seq_len=MAX_SEQ_LEN, skip_first=SKIP_FIRST,
            )
        else:
            maps, observed_length, observed_valid = jacobian_for_prompt(
                model, TEXT, SOURCE_LAYERS, target_layer=TARGET_LAYER, dim_batch=args.batch,
                max_seq_len=MAX_SEQ_LEN, skip_first=SKIP_FIRST, engine=args.engine,
                storage_context=storage, diagnostics=info,
            )
        torch.cuda.synchronize()
        info["complete_prompt_seconds"] = time.perf_counter() - started
        info["peak_cuda_gb"] = torch.cuda.max_memory_allocated() / 1e9
        info["peak_cuda_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
        if (observed_length, observed_valid) != (length, n_valid):
            raise ValueError("fitter token/valid-position counts changed after preflight")
        validate_maps(maps, model.d_model, "candidate")
        if storage is not None:
            info["storage"] = storage.stats()
            info["storage_counter_scope"] = "Python warmup/capture/eager calls; CUDA graph replays do not increment unpack counters"
        info["all_finite"] = all(torch.isfinite(matrix).all().item() for matrix in maps.values())
        info["full_width_complete"] = True
        info["values_written"] = sum(matrix.numel() for matrix in maps.values())
        info["status"] = "computed" if info["all_finite"] else "nonfinite"
        write_result(args.output, info)
        if args.reference and info["all_finite"]:
            reference = torch.load(args.reference, map_location="cpu", mmap=True, weights_only=True)
            parity = compare_maps(maps, reference, model.d_model)
            info["parity"] = parity
            info["reference_all_finite"] = all(item["reference_finite"] for item in parity)
            info["all_values_bitwise_equal"] = all(item["bitwise_equal"] for item in parity)
            info["values_compared"] = info["values_written"]
            info["reference_comparison_passed"] = info["reference_all_finite"] and info["all_values_bitwise_equal"]
            info["status"] = ("passed" if info["reference_comparison_passed"] else
                              "mismatch" if info["reference_all_finite"] else "nonfinite")
        elif info["all_finite"]:
            info["status"] = "completed_unverified"
        if args.save and info["all_finite"]:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            torch.save(maps, args.save)
            info["save_seconds"] = time.perf_counter() - started
            info["artifact_bytes"] = args.save.stat().st_size
            info["saved_artifact_status"] = info["status"]
        elif args.save:
            info["save_skipped"] = "candidate contains nonfinite values"
    except Exception as error:
        info["status"] = "reference_validation_failed" if isinstance(error, ReferenceValidationError) else "error"
        info["reference_comparison_passed"] = False
        info["error"] = f"{type(error).__name__}: {error}"
        info["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_result(args.output, info)
        print(json.dumps(info, allow_nan=False), flush=True)
        raise
    info["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_result(args.output, info)
    print(json.dumps(info, allow_nan=False), flush=True)
    if info["status"] in ("nonfinite", "mismatch"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
