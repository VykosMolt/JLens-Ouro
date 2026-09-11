"""Bounded local Ouro compiled-autograd trial; run explicitly with --run.

Each stage runs in its own process group. The supervisor terminates the whole
group on timeout, >22 GiB process-tree RSS, or <3 GiB system available memory.
Suffix Inductor parity must pass before full Inductor compilation. A failed or
bounded full attempt can be followed by a full compiled-autograd FX/eager check.
No scientific fitting, model rewriting, or task-score evaluation occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "bench_compiled_ouro.json"
LOG = HERE / "bench_compiled_ouro.log"
PROMPTS = (
    "The scientist carefully studied the properties of iron and copper. " * 30,
    "A researcher compared oxygen, copper, and iron in a careful experiment. " * 30,
)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def worker(stage: str, path: Path) -> None:
    import traceback

    sys.path.insert(0, "/home/moloch/ouro_project/src")
    sys.path.insert(0, str(HERE))
    import torch
    import torch._inductor.config as inductor_config
    from compiled_candidate import CompiledVJP, benchmark_retained
    from jlens.hooks import ActivationRecorder
    from ouro_jlens.recurrent import load_ouro

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    sources = [188, 189, 190] if stage == "suffix_inductor" else list(range(191))
    backend = "eager" if stage == "full_fx_eager" else "inductor"
    result = {
        "stage": stage, "status": "running", "phase": "load",
        "backend": backend, "source_layers": sources, "target_layer": 191,
        "dim_batch": 2, "max_seq_len": 128, "skip_first": 16,
        "torch": torch.__version__, "torch_git": torch.version.git_version,
        "source_sha256": {
            name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in ("compiled_candidate.py", "cuda_graph_candidate.py", "bench_compiled_ouro.py")
        },
        "config": {"donated_buffer": False, "backward_pass_autocast": "off", "emulate_precision_casts": True, "triton_enable_fp_fusion": False, "compile_threads": 1, "triton_cudagraphs": False},
        "config_evidence": "installed torch/_inductor/codegen/triton.py sets enable_fp_fusion = not emulate_precision_casts",
        "primals": [],
    }
    write_json(path, result)
    try:
        model = load_ouro()
        result["model_revision"] = model.model_revision
        result["attention_implementation"] = model.hf_model.config._attn_implementation
        result["model_dtype"] = str(next(model.hf_model.parameters()).dtype)
        result["model_training"] = model.hf_model.training
        result["load_seconds"] = time.perf_counter() - started
        result["device"] = torch.cuda.get_device_name()
        with CompiledVJP(backend=backend) as compiled:
            for prompt_index, prompt in enumerate(PROMPTS):
                result["phase"] = f"primal_{prompt_index}"
                write_json(path, result)
                with (
                    torch.enable_grad(),
                    ActivationRecorder(model.layers, at=[*sources, 191], start_graph_at=min(sources)) as recorder,
                ):
                    ids = model.encode(prompt, max_length=128)
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    forward_started = time.perf_counter()
                    model.forward(ids.expand(2, -1))
                    torch.cuda.synchronize()
                    forward_seconds = time.perf_counter() - forward_started
                    result["phase"] = f"compile_and_measure_{prompt_index}"
                    write_json(path, result)
                    measured = benchmark_retained(
                        compiled, recorder.activations[191],
                        [recorder.activations[source] for source in sources],
                        skip_first=16, measure_passes=6,
                    )
                    measured.update({
                        "prompt_index": prompt_index,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "token_ids": ids[0].cpu().tolist(),
                        "forward_seconds": forward_seconds,
                        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                    })
                    measured["estimated_100_prompt_compute_seconds_if_cache_reused"] = 100 * forward_seconds + measured["estimated_100_prompt_backward_seconds_if_cache_reused"]
                    measured["estimated_100_prompt_eager_compute_seconds"] = 100 * forward_seconds + measured["estimated_100_prompt_eager_backward_seconds"]
                    result["primals"].append(measured)
                    write_json(path, result)
                    print(json.dumps({"stage": stage, "prompt": prompt_index, "status": measured["status"], "first_compiled_seconds": measured["first_compiled_call_seconds"], "eager_seconds_per_pass": measured["eager_seconds_per_pass"], "compiled_seconds_per_pass": measured["compiled_seconds_per_pass"], "graphs": measured["total_graphs_captured"]}), flush=True)
                del recorder
                if measured["status"] != "passed":
                    break
        result["status"] = "passed" if len(result["primals"]) == 2 and all(item["status"] == "passed" for item in result["primals"]) else "numerical_mismatch"
        result["config_restored_emulate_precision_casts"] = inductor_config.emulate_precision_casts
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], flush=True)
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        write_json(path, result)


def available_memory() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("cannot read system MemAvailable")


def process_tree_rss(pid: int) -> int:
    pending, seen, total = [pid], set(), 0
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            for line in Path(f"/proc/{current}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
            children = Path(f"/proc/{current}/task/{current}/children").read_text().split()
            pending.extend(int(child) for child in children)
        except (FileNotFoundError, ProcessLookupError):
            pass
    return total


def stop_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def supervise_stage(stage: str, *, limit: int, log) -> dict:
    stage_path = Path(f"/tmp/jlens-{stage}-{os.getpid()}.json")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", stage, "--stage-output", str(stage_path)]
    started = time.perf_counter()
    print(f"START {stage}: wall limit {limit}s", flush=True)
    log.write(f"\nSTART {stage}: wall limit {limit}s\n")
    log.flush()
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    peak_rss, min_available, stop_reason = 0, available_memory(), None
    try:
        while process.poll() is None:
            elapsed = time.perf_counter() - started
            rss, available = process_tree_rss(process.pid), available_memory()
            peak_rss, min_available = max(peak_rss, rss), min(min_available, available)
            if elapsed > limit:
                stop_reason = f"wall limit {limit}s"
            elif rss > 22 * 1024**3:
                stop_reason = "process-tree RSS exceeded 22 GiB"
            elif available < 3 * 1024**3:
                stop_reason = "system MemAvailable fell below 3 GiB"
            if stop_reason:
                stop_group(process)
                break
            time.sleep(0.5)
    finally:
        if process.poll() is None:
            stop_group(process)
    result = json.loads(stage_path.read_text()) if stage_path.exists() else {"stage": stage, "status": "failed", "error": "worker produced no result"}
    result.update({"process_exit_code": process.returncode, "supervised_seconds": time.perf_counter() - started, "peak_process_tree_rss_bytes": peak_rss, "minimum_system_available_bytes": min_available})
    if stop_reason:
        result.update({"status": "resource_bound", "stop_reason": stop_reason})
    elif process.returncode != 0:
        result["status"] = "failed"
    print(f"FINISH {stage}: {result['status']} in {result['supervised_seconds']:.1f}s", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", choices=("suffix_inductor", "full_inductor", "full_fx_eager"))
    parser.add_argument("--stage-output", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.stage_output is None:
            parser.error("--worker requires --stage-output")
        worker(args.worker, args.stage_output)
        return
    if not args.run:
        print("No benchmark run; --run requires the assigned exclusive GPU slot.")
        return
    result = {"status": "running", "scope": "local exact-estimator performance trial; no scientific fit or task scoring", "stages": []}
    write_json(OUTPUT, result)
    with LOG.open("w") as log:
        suffix = supervise_stage("suffix_inductor", limit=120, log=log)
        result["stages"].append(suffix)
        write_json(OUTPUT, result)
        full = None
        if suffix["status"] == "passed":
            full = supervise_stage("full_inductor", limit=300, log=log)
            result["stages"].append(full)
            write_json(OUTPUT, result)
        if full is None or full["status"] != "passed":
            fallback = supervise_stage("full_fx_eager", limit=120, log=log)
            result["stages"].append(fallback)
            write_json(OUTPUT, result)
    result["status"] = "complete"
    write_json(OUTPUT, result)
    print(json.dumps({"status": "complete", "stages": [{"stage": stage["stage"], "status": stage["status"]} for stage in result["stages"]]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
