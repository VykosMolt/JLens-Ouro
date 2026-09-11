"""Full-width native-versus-optimized checks on one fixed non-task paragraph.

Every row at the same B=8 is compared literally. The native reference keeps
the original forward, activation recorder, autograd inputs and estimator; its
saved activations may be moved to CPU with the separately tested exact-layout
offloader. No calibration mean or recovery score is produced here.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import run_refits as runner


GIB = 1024 ** 3
MIN_HOST_BYTES = 48_000_000_000  # Provider host-RAM catalogue uses decimal GB.


def _host_memory(*, proc_root=Path("/proc"), cgroup_root=Path("/sys/fs/cgroup")):
    """Conservative available RAM, including visible v1/v2 ancestor limits.

    cgroup usage includes reclaimable file cache, so its headroom can be lower
    than the amount the kernel could eventually reclaim. A native reference
    uses that lower bound instead of treating host-wide MemAvailable as a
    container allowance.
    """
    fields = {}
    for line in (proc_root / "meminfo").read_text().splitlines():
        parts = line.split()
        if parts and parts[0] in {"MemTotal:", "MemAvailable:"}:
            if len(parts) != 3 or parts[2] != "kB":
                raise ValueError("unexpected /proc/meminfo byte units")
            fields[parts[0][:-1]] = int(parts[1]) * 1024
    if (set(fields) != {"MemTotal", "MemAvailable"} or fields["MemTotal"] <= 0
            or not 0 <= fields["MemAvailable"] <= fields["MemTotal"]):
        raise ValueError("cannot establish available physical host memory")
    candidates = {
        (cgroup_root, "memory.max", "memory.current"),
        (cgroup_root / "unified", "memory.max", "memory.current"),
        (cgroup_root / "memory", "memory.limit_in_bytes", "memory.usage_in_bytes"),
        (cgroup_root, "memory.limit_in_bytes", "memory.usage_in_bytes"),
    }
    for line in (proc_root / "self/cgroup").read_text().splitlines():
        pieces = line.split(":", 2)
        if len(pieces) != 3:
            raise ValueError("malformed process cgroup membership")
        _, controllers, relative = pieces
        if not relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError("noncanonical process cgroup membership")
        if not controllers:
            bases = (cgroup_root, cgroup_root / "unified")
            limit_name, usage_name = "memory.max", "memory.current"
        elif "memory" in controllers.split(","):
            bases = (cgroup_root / "memory", cgroup_root)
            limit_name, usage_name = "memory.limit_in_bytes", "memory.usage_in_bytes"
        else:
            continue
        for base in bases:
            current = base / relative.lstrip("/")
            while True:
                candidates.add((current, limit_name, usage_name))
                if current == base:
                    break
                current = current.parent
    constraints = []
    total, available = fields["MemTotal"], fields["MemAvailable"]
    for directory, limit_name, usage_name in sorted(candidates):
        limit_path = directory / limit_name
        if not limit_path.exists():
            continue
        raw = limit_path.read_text().strip()
        usage = int((directory / usage_name).read_text().strip())
        limit = None if raw == "max" else int(raw)
        if usage < 0 or (limit is not None and limit <= 0):
            raise ValueError("invalid cgroup memory limit or usage")
        headroom = None if limit is None else max(0, limit - usage)
        constraints.append({"path": str(directory), "limit_bytes": limit,
                            "usage_bytes": usage, "headroom_bytes": headroom})
        if limit is not None:
            total, available = min(total, limit), min(available, headroom)
    return {"physical_total_bytes": fields["MemTotal"], "physical_available_bytes": fields["MemAvailable"],
            "effective_total_bytes": total, "effective_available_bytes": available,
            "cgroup_constraints": constraints,
            "scope": "snapshot of physical MemAvailable and visible cgroup ancestor headroom; cgroup file cache is counted conservatively"}


def _offload_budget(*, paired, proc_root=Path("/proc"), cgroup_root=Path("/sys/fs/cgroup")):
    memory = _host_memory(proc_root=proc_root, cgroup_root=cgroup_root)
    if memory["effective_total_bytes"] < MIN_HOST_BYTES:
        raise RuntimeError("full-width native preflight requires at least 48 GB (48e9 bytes) of effective host/cgroup RAM")
    # Native/optimized paired banks and the I/O clone can each occupy about
    # 12 GiB in their respective phases. Single-bank Huginn uses under 7 GiB.
    # This reserved non-payload allowance includes the 4 GiB live host reserve.
    non_payload = (12 if paired else 8) * GIB
    maximum = min(24 * GIB, memory["effective_available_bytes"] - non_payload)
    if maximum <= 0:
        raise RuntimeError("insufficient host/cgroup headroom for the full-width reference and its matrix banks")
    return {"max_host_bytes": maximum, "reserve_host_bytes": 4 * GIB,
            "non_payload_headroom_bytes": non_payload, "minimum_host_bytes": MIN_HOST_BYTES,
            "memory": memory}


def _huginn_benchmark_runtime(model, runtime):
    model.state_seed, model.seed_namespace = 2026090899, "benchmark"
    return {**runtime, "initialization": {
        **runtime["initialization"], "base_seed": model.state_seed,
        "seed_namespace": model.seed_namespace,
        "base_seed_rule": "fixed for the engineering paragraph; no calibration paragraph index",
    }}


def _source(path, contract):
    records = {str(contract["source_paths"][(row["root"], row["path"])].absolute()): row["sha256"]
               for row in contract["sources"]}
    path = runner._no_links(path)
    if str(path) not in records or runner._file_hash(path) != records[str(path)]:
        raise ValueError(f"preflight code is not bound by the run specification: {path}")


def _compare(torch, actual, native, sources, width):
    if set(actual) != set(native):
        raise ValueError("native and optimized estimator arms differ")
    identity = {"source_layers": sources, "d_model": width, "bank_arms": list(actual)}
    runner._banks(torch, actual, identity, torch.float32, "optimized benchmark")
    runner._banks(torch, native, identity, torch.float32, "native benchmark")
    records = []
    for arm in actual:
        digest = hashlib.sha256()
        for layer in sources:
            a, b = actual[arm][layer], native[arm][layer]
            if not torch.equal(a.view(torch.int32), b.view(torch.int32)):
                delta = (a - b).abs()
                raise ValueError(f"native parity failed for {arm}, source{layer}: maximum absolute difference{delta.max().item()}")
            digest.update(a.numpy().tobytes())
        records.append({"arm": arm, "source_count": len(sources), "output_directions": width,
                        "values_compared": len(sources) * width * width,
                        "bitwise_equal": True, "matrix_bits_sha256": digest.hexdigest()})
    return records


def _io_timing(torch, maps, directory):
    """Measure FP32 accumulation, writes/fsync, hashing and old-state pruning."""
    old, new = directory / "previous_benchmark.pt", directory / "current_benchmark.pt"
    for path in (old, new):
        if path.exists():
            raise ValueError("preflight I/O scratch file already exists")
    with old.open("xb") as handle:
        torch.save({"benchmark_only": True, "maps": maps}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    old_record = runner._record(old)
    started = time.monotonic()
    summed = {arm: {layer: tensor.clone().add_(tensor) for layer, tensor in bank.items()}
              for arm, bank in maps.items()}
    if not all(torch.isfinite(tensor).all().item() for bank in summed.values() for tensor in bank.values()):
        raise ValueError("benchmark FP32 sum is nonfinite")
    with new.open("xb") as handle:
        torch.save({"benchmark_only": True, "maps": summed}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    runner._fsync_dir(directory)
    record = runner._record(new)
    runner._verify_record(old, old_record)
    old.unlink()
    runner._fsync_dir(directory)
    elapsed = time.monotonic() - started
    # These are scratch benchmark matrices, not a calibration checkpoint.
    new.unlink()
    runner._fsync_dir(directory)
    return {"accumulate_write_hash_prune_seconds": elapsed, "state_bytes": record["bytes"],
            "state_sha256": record["sha256"], "scratch_removed": True}


def _huginn_primal(torch, model, text):
    ids = model.encode(text, max_length=128).expand(8, -1).clone()
    initial = model.sample_initial_state(ids)
    if not torch.equal(initial, initial[:1].expand_as(initial)):
        raise ValueError("Huginn initial states differ across derivative lanes")
    def capture(native):
        values, counts, handles = {}, {}, []
        for layer, tap in enumerate(model.layers):
            def hook(module, args, output, layer=layer):
                counts[layer] = counts.get(layer, 0) + 1
                values[layer] = output.detach().clone()
            handles.append(tap.register_forward_hook(hook))
        try:
            with torch.no_grad():
                output = (model.native_logits(ids, input_states=initial) if native
                          else model.unembed(model.forward(ids, input_states=initial)))
            if counts != {layer: 1 for layer in range(34)}:
                raise ValueError("Huginn virtual cell counts differ from the native topology")
            return values, output
        finally:
            for handle in handles:
                handle.remove()
    observed, logits = capture(False)
    expected, native_logits = capture(True)
    for layer in observed:
        if not torch.equal(observed[layer].view(torch.int16), expected[layer].view(torch.int16)):
            raise ValueError(f"Huginn adapter/native primal mismatch at cell{layer}")
    if not torch.equal(logits.view(torch.int32), native_logits.view(torch.int32)):
        raise ValueError("Huginn adapter/native logits differ")
    return {"all_34_cells_bitwise_equal": True, "logits_bitwise_equal": True,
            "coupled_initial_states": True, "batch": 8, "sequence_length": ids.shape[1]}


class _NativeHuginn:
    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)

    def forward(self, ids):
        return self.model.native_logits(ids)


def execute(args, main_contract, prepared=None, prerequisites=None):
    import torch
    initial_host_budget = _offload_budget(paired=args.kind == "ouro")
    if args.kind == "ouro":
        model, estimators, runtime = runner._load_ouro_runtime(args, main_contract)
    else:
        import run_huginn
        model, estimators, runtime = run_huginn._load_runtime(args, prepared, prerequisites)
        runtime = _huginn_benchmark_runtime(model, runtime)
    import ouro_jlens.bench as bench_module
    TEXT = bench_module.TEXT
    import control_estimators
    from optimization.bench_offload_reference import ExactCPUOffload
    import optimization.bench_offload_reference as offload_module
    for module in (control_estimators, offload_module, bench_module):
        _source(Path(module.__file__), main_contract)
    result = {"schema_version": 1, "kind": args.kind, "status": "running",
              "scope": "fixed engineering paragraph; no scientific calibration or task recovery",
              "runtime": runtime, "prompt_sha256": hashlib.sha256(TEXT.encode()).hexdigest(),
              "benchmark_source": {"module": bench_module.__name__, "path": str(Path(bench_module.__file__).absolute()),
                                   "sha256": runner._file_hash(bench_module.__file__)},
              "initial_host_budget": initial_host_budget,
              "dim_batch": 8, "max_seq_len": 128, "skip_first": 16, "profiles": []}
    if args.kind == "huginn":
        result["native_primal"] = _huginn_primal(torch, model, TEXT)
        result["benchmark_initialization"] = model.initialization_metadata(model.encode(TEXT, max_length=128))
        native_model = _NativeHuginn(model)
        profiles = [("huginn_r8", 33, list(range(32)), False)]
    else:
        native_model = model
        profiles = [("ouro_main", 191, list(range(191)), False),
                    ("ouro_penultimate", 190, list(range(190)), False),
                    ("ouro_positions", 191, list(range(191)), True)]
    for name, target, sources, paired in profiles:
        common = dict(target_layer=target, dim_batch=8, max_seq_len=128, skip_first=16)
        row = {"name": name, "target_layer": target, "source_layers": sources,
               "q": 64 if paired else None}
        budget = _offload_budget(paired=paired)
        row["host_offload_budget"] = budget
        offloader = ExactCPUOffload(native_model, pin_memory=False,
                                   max_host_bytes=budget["max_host_bytes"],
                                   reserve_host_bytes=budget["reserve_host_bytes"])
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with offloader:
            native, length, count = estimators.jacobians_for_prompt(
                native_model, TEXT, sources, mode="sampled" if paired else "dense",
                q=64 if paired else None, dense_engine="stock", **common)
        torch.cuda.synchronize()
        row.update(native_seconds=time.monotonic() - started,
                   native_peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                   native_offloader=offloader.stats(), sequence_length=length, n_valid=count)
        if offloader.live_host_bytes != 0:
            raise ValueError("native benchmark retained saved activation payloads")
        gc.collect()
        torch.cuda.empty_cache()
        details = {}
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        if paired:
            observed, other_length, other_count = control_estimators.paired_jacobians_for_prompt(
                model, TEXT, sources, q=64, engine="cuda_graph", compress_saved_tensors=True,
                diagnostics=details, **common)
        else:
            observed, other_length, other_count = estimators.jacobians_for_prompt(
                model, TEXT, sources, mode="dense", dense_engine="cuda_graph",
                compress_saved_tensors=True, diagnostics=details, **common)
        torch.cuda.synchronize()
        row.update(optimized_seconds=time.monotonic() - started,
                   optimized_peak_cuda_bytes=torch.cuda.max_memory_allocated(), optimized_diagnostics=details)
        if (length, count) != (other_length, other_count):
            raise ValueError("native and optimized token eligibility differs")
        row["parity"] = _compare(torch, observed, native, sources, model.d_model)
        del native
        gc.collect()
        row["io"] = _io_timing(torch, observed, args.output_dir)
        row["prompt_with_checkpoint_seconds"] = row["optimized_seconds"] + row["io"]["accumulate_write_hash_prune_seconds"]
        result["profiles"].append(row)
        runner._new_json(args.output_dir / f"{name}.json", row)
        del observed
        gc.collect()
        torch.cuda.empty_cache()
    result["status"] = "passed"
    runner._new_json(args.output_dir / "COMPLETE.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("ouro", "huginn"), required=True)
    parser.add_argument("--run-spec", type=Path, default=runner.HERE / "run_spec.json")
    parser.add_argument("--combined-contract", type=Path, default=runner.HERE / "combined_contract.json")
    parser.add_argument("--calibration", type=Path, default=runner.HERE / "huginn_calibration.json")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--ouro-src", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ouro-evaluation-dir", type=Path)
    parser.add_argument("--controls-evaluation-dir", type=Path)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("preflight output must be a new directory")
    main_contract = runner._contract(args.run_spec, args.ouro_src, list(range(1, 6)))
    _source(Path(__file__), main_contract)
    prepared = prerequisites = None
    if args.kind == "huginn":
        import run_huginn
        prepared = run_huginn.contract(args)
        prerequisites = run_huginn._evaluation_gate(args, prepared)
    runner._mkdir(args.output_dir)
    print(json.dumps(execute(args, main_contract, prepared, prerequisites), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
