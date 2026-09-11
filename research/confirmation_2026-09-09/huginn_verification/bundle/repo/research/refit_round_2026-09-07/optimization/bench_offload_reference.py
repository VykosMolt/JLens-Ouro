"""Sampled SAME-B native references with exact full-tensor CPU offload.

This is a reference driver, not a fit or a complete B4/B8 Jacobian reference.
It uses the stock ActivationRecorder/all source tensors and ordinary
autograd.grad at the original replicated batch. No gradient edges, delta
inputs, vmap, lane compression, backend override or tolerance changes occur.

The installed public save_on_cpu(pin_memory=True) allocates CPU tensors with
torch.empty(size), which makes transposed saves contiguous, and offloads each
saved weight. Instead, exact_cpu_offload below copies each full activation into
a CPU backing allocation with its original shape/stride/storage offset, then
reconstructs the same layout on its original device. Persistent weight/buffer
storages stay on their original device. Unsupported overlapping/non-dense
saves stay there too, with explicit fallback statistics. This is first-order,
frozen-model execution: do not mutate saved values or weights after packing.
The driver uses unpinned CPU backing and blocking copies to avoid pinned-host
allocator rounding/cache overhead. This reference prioritizes memory and exact
values over transfer throughput.

No GPU is initialized on import or by --self-test. Only the root operator should
run the GPU driver after reserving a slot. B4 is the default; B8 requires enough
available RAM and passes both a projected and a live host-payload budget.
The B2 offloaded first two rows must match the saved stock B2 reference exactly
before the requested-batch samples are attempted. Outputs are NPZ + JSON with
explicit sampled row indices, provenance, resource measurements and gate status.

Primary hook contract:
https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html
Installed implementation checked in torch/autograd/graph.py:264-420.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import sys
import time
import traceback
import weakref
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from jlens.fitting import valid_position_mask  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402

GIB = 1024**3


def _storage_key(tensor):
    return tensor.device.type, tensor.device.index, tensor.untyped_storage().data_ptr()


def _dense_nonoverlapping(tensor):
    expected = 1
    for stride, size in sorted((stride, size) for size, stride in zip(tensor.shape, tensor.stride(), strict=True) if size > 1):
        if stride != expected:
            return False
        expected *= size
    return True


def _bitwise_equal(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    integer = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[left.element_size()]
    return torch.equal(left.detach().view(integer), right.detach().view(integer))


def _available_memory_bytes():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("cannot determine available host memory from /proc/meminfo")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class HostMemoryBudgetExceeded(RuntimeError):
    pass


@dataclass
class OffloadedTensor:
    cpu_value: torch.Tensor
    original_device: torch.device
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    storage_elements: int
    pinned: bool


class ExactCPUOffload:
    """Offload full saved activations without changing their runtime layouts.

    Scope only around the original native forward. Pack holds an independent
    CPU allocation; unpack creates an ephemeral allocation on the original
    device, with no persistent device cache. Same-object/version reuse avoids
    repeated copies of the very same saved object; it never compares or merges
    different batch lanes. Weak cache entries retain no original tensor/storage.
    A payload finalizer tracks currently live CPU allocation bytes. The caller
    still measures RSS/available RAM because pinned allocator caches and other
    process allocations are outside this payload accounting.
    """

    def __init__(self, model, *, pin_memory=False, max_host_bytes, reserve_host_bytes=0):
        if isinstance(max_host_bytes, bool) or not isinstance(max_host_bytes, Integral) or max_host_bytes < 1:
            raise ValueError("max_host_bytes must be a positive integer")
        if isinstance(reserve_host_bytes, bool) or not isinstance(reserve_host_bytes, Integral) or reserve_host_bytes < 0:
            raise ValueError("reserve_host_bytes must be a nonnegative integer")
        modules = [model] if isinstance(model, torch.nn.Module) else [getattr(model, name, None) for name in ("_hf_model", "hf_model", "_text_module")]
        modules = [module for module in modules if isinstance(module, torch.nn.Module)]
        if not modules:
            raise ValueError("model must expose an nn.Module for persistent-storage exclusion")
        self._persistent = {_storage_key(tensor) for module in modules for tensor in [*module.parameters(), *module.buffers()] if tensor.layout == torch.strided and tensor.numel() > 0}
        self.pin_memory = bool(pin_memory)
        self.max_host_bytes = int(max_host_bytes)
        self.reserve_host_bytes = int(reserve_host_bytes)
        self._counts = Counter()
        self._layouts = Counter()
        self._cache = {}
        self._hooks = None
        self.live_host_bytes = 0
        self.peak_host_bytes = 0

    @property
    def active(self):
        return self._hooks is not None

    def _release(self, nbytes):
        self.live_host_bytes -= nbytes
        self._counts["released_payloads"] += 1

    def _keep(self, tensor, reason):
        self._counts[f"keep:{reason}"] += 1
        self._counts[f"keep_bytes:{reason}"] += tensor.numel() * tensor.element_size()
        return tensor.detach()

    def pack(self, tensor):
        self._counts["pack_calls"] += 1
        if tensor.layout != torch.strided or tensor.is_quantized or tensor.numel() == 0 or tensor.is_conj() or tensor.is_neg():
            return self._keep(tensor, "unsupported")
        if _storage_key(tensor) in self._persistent:
            return self._keep(tensor, "persistent_storage")
        if not _dense_nonoverlapping(tensor):
            return self._keep(tensor, "non_dense_or_overlapping")
        key = id(tensor), tensor._version
        cached = self._cache.get(key)
        if cached is not None and cached[0]() is tensor:
            payload = cached[1]()
            if payload is not None:
                self._counts["reused_payloads"] += 1
                return payload
        shape, stride, offset = tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset()
        elements = offset + 1 + sum((size - 1) * step for size, step in zip(shape, stride, strict=True))
        nbytes = elements * tensor.element_size()
        if self.live_host_bytes + nbytes > self.max_host_bytes:
            raise HostMemoryBudgetExceeded(f"exact_cpu_offload would retain {self.live_host_bytes + nbytes} host bytes, above budget {self.max_host_bytes}")
        if self.reserve_host_bytes and self._counts["pack_calls"] % 64 == 0:
            available = _available_memory_bytes()
            if available - nbytes < self.reserve_host_bytes:
                raise HostMemoryBudgetExceeded(f"available host RAM {available} would fall below reserve {self.reserve_host_bytes}")
        pinned = self.pin_memory and tensor.device.type == "cuda"
        backing = torch.empty(elements, dtype=tensor.dtype, device="cpu", pin_memory=pinned)
        cpu_value = backing.as_strided(shape, stride, offset)
        cpu_value.copy_(tensor.detach(), non_blocking=pinned)
        payload = OffloadedTensor(cpu_value, tensor.device, shape, stride, offset, elements, pinned)
        self.live_host_bytes += nbytes
        self.peak_host_bytes = max(self.peak_host_bytes, self.live_host_bytes)
        self._counts["offloaded_payloads"] += 1
        self._counts["host_bytes_allocated_total"] += nbytes
        self._counts["offloaded_logical_bytes_total"] += tensor.numel() * tensor.element_size()
        self._counts["max_reconstruction_bytes"] = max(self._counts["max_reconstruction_bytes"], nbytes)
        self._layouts[str((shape, stride, offset))] += 1
        self._cache[key] = (weakref.ref(tensor), weakref.ref(payload))
        weakref.finalize(payload, self._release, nbytes)
        return payload

    def unpack(self, payload):
        self._counts["unpack_calls"] += 1
        if isinstance(payload, torch.Tensor):
            return payload
        with torch.no_grad():
            backing = torch.empty(payload.storage_elements, dtype=payload.cpu_value.dtype, device=payload.original_device)
            value = backing.as_strided(payload.shape, payload.stride, payload.storage_offset)
            value.copy_(payload.cpu_value, non_blocking=payload.pinned)
        self._counts["reconstructions"] += 1
        self._counts["reconstructed_bytes_total"] += backing.numel() * backing.element_size()
        return value

    def stats(self):
        return {"engine": "exact_cpu_offload", "pin_memory": self.pin_memory, "max_host_bytes": self.max_host_bytes, "reserve_host_bytes": self.reserve_host_bytes, "live_host_payload_bytes": self.live_host_bytes, "peak_host_payload_bytes": self.peak_host_bytes, "counts": dict(self._counts), "offloaded_layout_counts": dict(self._layouts), "scope": "full original tensors; weights/buffers and unsupported layouts stay on original device; payload bytes exclude allocator caches"}

    def __enter__(self):
        if self.active:
            raise RuntimeError("offloader is already active")
        hooks = torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)
        hooks.__enter__()
        self._hooks = hooks
        return self

    def __exit__(self, *exc):
        hooks, self._hooks = self._hooks, None
        if hooks is not None:
            hooks.__exit__(*exc)


def sampled_reference(model, prompt, *, batch, starts, max_host_bytes, reserve_host_bytes, repeat_first=False, include_zero=False):
    """Original native B-lane VJPs, exact stock source reduction, selected rows."""
    if model.n_layers != 192 or model.d_model != 2048:
        raise ValueError("reference driver requires the pinned 192-event, width-2048 Ouro model")
    sources = list(range(191))
    ids = model.encode(prompt, max_length=128)
    seq_len = ids.shape[1]
    mask = valid_position_mask(seq_len, skip_first=16)
    n_valid = int(mask.sum())
    schedule = [(f"start_{start}", start) for start in starts]
    if repeat_first:
        schedule.append(("repeat_start_0", 0))
    if include_zero:
        schedule.append(("zero", None))
    if not schedule or any(start is not None and (start < 0 or start + batch > 2048) for _, start in schedule):
        raise ValueError("sample row schedule is out of range")
    offload = ExactCPUOffload(model, pin_memory=False, max_host_bytes=max_host_bytes, reserve_host_bytes=reserve_host_bytes)
    samples, sweeps = {}, []
    record = {"engine": "exact_cpu_offload", "batch": batch, "target": 191, "source_layers": sources, "seq_len": seq_len, "valid_positions": mask.nonzero().flatten().tolist(), "n_valid": n_valid, "complete_jacobian_reference": False, "calibration_count_in_research": 0}
    device = ids.device
    torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    try:
        with ActivationRecorder(model.layers, at=[*sources, 191], start_graph_at=0) as recorder, torch.enable_grad():
            with offload:
                model.forward(ids.expand(batch, -1))
            _sync(device)  # Every original forward/save is complete before backward.
            record["forward_seconds"] = time.perf_counter() - started
            record["after_forward_cuda_bytes"] = torch.cuda.memory_allocated(device)
            record["after_forward_host_payload_bytes"] = offload.live_host_bytes
            target = recorder.activations[191]
            source_values = [recorder.activations[layer] for layer in sources]
            positions = mask.nonzero(as_tuple=True)[0].to(target.device)
            lanes = torch.arange(batch, device=target.device)
            cotangent = torch.zeros_like(target)
            for index, (label, start) in enumerate(schedule):
                cotangent.zero_()
                if start is not None:
                    cotangent[lanes[:, None], positions[None, :], start + lanes[:, None]] = 1.0
                _sync(device)
                t0 = time.perf_counter()
                grads = torch.autograd.grad(target, source_values, grad_outputs=cotangent, retain_graph=index < len(schedule) - 1, create_graph=False)
                _sync(device)
                t1 = time.perf_counter()
                rows = []
                for grad in grads:
                    # Preserve the upstream mean operation and its original B
                    # shape. Only the selected direction groups are retained.
                    rows.append(grad[:, positions.to(grad.device), :].float().mean(dim=1).cpu())
                bank = torch.stack(rows)
                _sync(device)
                if not torch.isfinite(bank).all():
                    raise RuntimeError(f"nonfinite original offloaded reference at {label}")
                if label == "repeat_start_0" and not _bitwise_equal(bank, torch.from_numpy(samples["start_0"])):
                    raise RuntimeError("repeated start-zero reference differs bitwise on the same retained primal")
                if start is None and torch.count_nonzero(bank).item() != 0:
                    raise RuntimeError("zero cotangent produced nonzero reference rows")
                samples[label] = bank.numpy()
                sweep = {"label": label, "output_rows": [] if start is None else list(range(start, start + batch)), "stored_shape": list(bank.shape), "backward_seconds": t1 - t0, "reduction_copy_seconds": time.perf_counter() - t1, "all_finite": True}
                sweeps.append(sweep)
                print(json.dumps({"batch": batch, **sweep}), flush=True)
                del grads, grad, rows, bank
            record["peak_cuda_bytes"] = torch.cuda.max_memory_allocated(device)
            record["sweeps"] = sweeps
            record["offload"] = offload.stats()
        del target, source_values, recorder, cotangent
    except Exception as error:
        error.add_note(json.dumps({"batch": batch, "offload_at_failure": offload.stats()}))
        raise
    gc.collect()
    record["host_payload_bytes_after_graph_release"] = offload.live_host_bytes
    if offload.live_host_bytes != 0:
        raise RuntimeError("offloaded payloads remain live after reference graph release")
    return samples, record


def _load_b2_baseline(path):
    if path.suffix == ".pt":
        maps = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if set(maps) != set(range(191)):
            raise ValueError("stock baseline must contain all 191 source maps")
        reference = torch.stack([maps[layer][:2].clone() for layer in range(191)])
        del maps
    else:
        with np.load(path, allow_pickle=False) as saved:
            reference = torch.from_numpy(saved["replicated_2"].copy())
    if reference.shape != (191, 2, 2048) or reference.dtype != torch.float32 or not torch.isfinite(reference).all():
        raise ValueError("stock B2 baseline slice has invalid shape/dtype/values")
    return reference, {"path": str(path.resolve()), "file_bytes": path.stat().st_size, "used_slice": "all191sources, output rows0:2, all2048input coordinates", "used_slice_sha256": hashlib.sha256(reference.numpy().tobytes()).hexdigest()}


def _self_test():
    from jlens.fitting import jacobian_for_prompt
    from tests.tiny import TinyDecoder

    torch.set_num_threads(1)
    holder = torch.nn.Linear(5, 5, bias=False).eval().requires_grad_(False)
    offload = ExactCPUOffload(holder, pin_memory=False, max_host_bytes=4 * 1024**2)
    base = torch.randn(3, 7, 5)
    backing = torch.empty(base.numel() + 4)
    offset = backing.as_strided(base.shape, base.stride(), 4)
    offset.copy_(base)
    layouts = [base, base.transpose(0, 1), base.transpose(1, 2), base.flatten(0, 1), base.flatten(0, 1).T, offset]
    for original in layouts:
        packed = offload.pack(original)
        assert isinstance(packed, OffloadedTensor)
        assert _storage_key(packed.cpu_value) != _storage_key(original)
        restored = offload.unpack(packed)
        assert _bitwise_equal(restored, original)
        assert restored.stride() == original.stride() and restored.storage_offset() == original.storage_offset()
        assert packed.cpu_value.stride() == original.stride() and packed.cpu_value.storage_offset() == original.storage_offset()
    assert isinstance(offload.pack(holder.weight.T), torch.Tensor)
    assert isinstance(offload.pack(base[:, ::2]), torch.Tensor)
    assert isinstance(offload.pack(base[:1].expand_as(base)), torch.Tensor)
    original = base.clone()
    reference = weakref.ref(original)
    payload = offload.pack(original)
    assert offload.pack(original) is payload
    del original
    gc.collect()
    assert reference() is None and _bitwise_equal(offload.unpack(payload), base)
    limited = ExactCPUOffload(holder, pin_memory=False, max_host_bytes=1)
    try:
        limited.pack(base)
    except HostMemoryBudgetExceeded:
        pass
    else:
        raise AssertionError("host payload budget was not enforced")

    class CausalBlock(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.linear = torch.nn.Linear(width, width, bias=False)
            self.norm = torch.nn.LayerNorm(width)

        def forward(self, hidden):
            positions = torch.arange(1, hidden.shape[1] + 1, device=hidden.device, dtype=hidden.dtype).view(1, -1, 1)
            context = hidden.cumsum(dim=1) / positions
            return hidden + 0.1 * torch.nn.functional.silu(self.linear(self.norm(context)))

    cases = []
    for dtype in (torch.float32, torch.bfloat16):
        model = TinyDecoder(n_layers=4, d_model=5, seed=42)
        model.layers = torch.nn.ModuleList([CausalBlock(5) for _ in range(4)])
        model.to(dtype).eval().requires_grad_(False)
        for batch in (2, 4):
            kwargs = dict(model=model, prompt="CPU exact offload parity", source_layers=[0, 1, 2], target_layer=3, dim_batch=batch, max_seq_len=7, skip_first=1)
            expected, _, _ = jacobian_for_prompt(**kwargs)
            original_forward = model.forward
            exact = ExactCPUOffload(model, pin_memory=False, max_host_bytes=8 * 1024**2)

            def forward(ids):
                with exact:
                    return original_forward(ids)

            model.forward = forward
            try:
                actual, _, _ = jacobian_for_prompt(**kwargs)
            finally:
                model.forward = original_forward
            assert all(_bitwise_equal(actual[layer], expected[layer]) for layer in expected)
            assert not exact.active and all(not block._forward_hooks for block in model.layers)
            gc.collect()
            assert exact.live_host_bytes == 0
            cases.append({"dtype": str(dtype), "batch": batch, "all_rows_all_sources_bitwise_equal": True, "peak_host_payload_bytes": exact.peak_host_bytes})
    try:
        with offload:
            raise RuntimeError("injected exception")
    except RuntimeError:
        pass
    assert not offload.active
    return {"device": "cpu", "engine": "exact_cpu_offload", "layout_roundtrips": len(layouts), "full_jacobian_cases": cases, "host_budget_guard": "pass", "original_tensor_released": True, "exception_cleanup": "pass", "cuda_initialized": torch.cuda.is_initialized()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--batch", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path, help="baseline_full.pt (mmap) or bench_vjp.npz; defaults to the available stock baseline")
    parser.add_argument("--host-reserve-gib", type=float, default=4.0)
    parser.add_argument("--max-offload-gib", type=float, help="optional additional cap; never exceeds available RAM minus reserve")
    parser.add_argument("--include-zero", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(_self_test(), indent=2))
        return
    if args.host_reserve_gib < 1 or not np.isfinite(args.host_reserve_gib):
        parser.error("host reserve must be finite and at least 1 GiB")
    if args.max_offload_gib is not None and (args.max_offload_gib <= 0 or not np.isfinite(args.max_offload_gib)):
        parser.error("max offload GiB must be finite and positive")
    output = args.output or HERE / f"offload_reference_b{args.batch}.json"
    json_path, npz_path = output.with_suffix(".json"), output.with_suffix(".npz")
    baseline = args.baseline or (HERE / "baseline_full.pt" if (HERE / "baseline_full.pt").exists() else HERE / "bench_vjp.npz")
    reference, reference_info = _load_b2_baseline(baseline)
    sys.path.insert(0, "/home/moloch/ouro_project/src")
    from ouro_jlens.bench import TEXT
    from ouro_jlens.recurrent import OURO_REVISION, load_ouro

    baseline_manifest = baseline.with_suffix(".json")
    if baseline.suffix == ".pt" and baseline_manifest.exists():
        raw_manifest = baseline_manifest.read_bytes()
        metadata = json.loads(raw_manifest)
        expected = {"dim_batch": 2, "target": 191, "source_count": 191, "model_revision": OURO_REVISION, "prompt_sha256": hashlib.sha256(TEXT.encode()).hexdigest()}
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"stock baseline manifest {key} does not match this reference: {metadata.get(key)!r} != {value!r}")
        reference_info["manifest_path"] = str(baseline_manifest.resolve())
        reference_info["manifest_sha256"] = hashlib.sha256(raw_manifest).hexdigest()

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU driver requires an available, reserved CUDA device")
    samples = {}
    result = {"status": "running", "engine": "exact_cpu_offload", "purpose": __doc__, "started_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "torch": torch.__version__, "torch_git": torch.version.git_version, "prompt_sha256": hashlib.sha256(TEXT.encode()).hexdigest(), "requested_batch": args.batch, "baseline": reference_info, "complete_jacobian_reference": False, "calibration_count_in_research": 0, "records": []}

    def persist(*, write_arrays=True):
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, indent=2) + "\n")
        if samples and write_arrays:
            np.savez(npz_path, **samples)

    reserve = int(args.host_reserve_gib * GIB)

    def budget():
        available = _available_memory_bytes()
        limit = available - reserve
        if args.max_offload_gib is not None:
            limit = min(limit, int(args.max_offload_gib * GIB))
        if limit < GIB:
            raise HostMemoryBudgetExceeded(f"only {available} available host bytes; no safe offload budget after reserve")
        return limit, available

    try:
        model = load_ouro()
        if model.model_revision != OURO_REVISION:
            raise RuntimeError("loaded model revision differs from the pinned Ouro revision")
        result.update({"device": torch.cuda.get_device_name(0), "model_revision": model.model_revision, "attention": model.hf_model.config._attn_implementation, "float32_matmul_precision": torch.get_float32_matmul_precision(), "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "hooks_source_sha256": hashlib.sha256(Path(sys.modules[ActivationRecorder.__module__].__file__).read_bytes()).hexdigest()})
        gc.collect()
        limit, available = budget()
        result["b2_gate_initial_host_available_bytes"] = available
        result["b2_gate_host_budget_bytes"] = limit
        persist()
        gate_samples, gate = sampled_reference(model, TEXT, batch=2, starts=[0], max_host_bytes=limit, reserve_host_bytes=reserve)
        actual = torch.from_numpy(gate_samples["start_0"])
        exact = _bitwise_equal(actual, reference)
        delta = actual.double() - reference.double()
        result["b2_gate"] = {"bitwise_equal": exact, "max_absolute_error": delta.abs().max().item(), "relative_frobenius_error": delta.norm().item() / reference.double().norm().clamp_min(1e-30).item()}
        result["records"].append(gate)
        samples["b2_gate_start_0"] = gate_samples["start_0"]
        persist()
        if not exact:
            raise RuntimeError("mandatory B2 offload gate failed; requested-batch reference was not attempted")
        del gate_samples, actual, reference, delta
        gc.collect()
        torch.cuda.empty_cache()
        # The graph has released every offloaded payload (checked above), but
        # glibc may keep several GiB of freed pageable arenas after the B2 gate.
        # Return those free pages before applying the unchanged B8 RAM guard.
        # No live tensor storage or GPU allocation is discarded by malloc_trim.
        result["host_available_before_allocator_trim_bytes"] = _available_memory_bytes()
        libc = ctypes.CDLL(None)
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim.argtypes = [ctypes.c_size_t]
            libc.malloc_trim.restype = ctypes.c_int
            result["malloc_trim_return"] = libc.malloc_trim(0)
        result["host_available_after_allocator_trim_bytes"] = _available_memory_bytes()
        limit, available = budget()
        projected = int(gate["offload"]["peak_host_payload_bytes"] * (args.batch / 2) * 1.10)
        result["requested_initial_host_available_bytes"] = available
        result["requested_host_budget_bytes"] = limit
        result["projected_host_payload_bytes_with_10pct_margin"] = projected
        persist()
        if projected > limit:
            raise HostMemoryBudgetExceeded(f"B{args.batch} projects {projected} host payload bytes including margin, above safe budget {limit}; choose a smaller batch or free RAM")
        chosen, record = sampled_reference(model, TEXT, batch=args.batch, starts=[0, 1024, 2048 - args.batch], max_host_bytes=limit, reserve_host_bytes=reserve, repeat_first=True, include_zero=args.include_zero)
        samples.update({f"native_b{args.batch}_{label}": value for label, value in chosen.items()})
        # Convenience slice for existing two-row comparison scripts. Other keys
        # retain every lane in the sampled middle/tail groups, including row2047.
        samples[f"native_b{args.batch}_first_two"] = chosen["start_0"][:, :2]
        result["records"].append(record)
        result["status"] = "complete_sampled_reference"
        result["npz_path"] = str(npz_path.resolve())
        persist()
        result["npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
        persist(write_arrays=False)
        print(json.dumps({"status": result["status"], "batch": args.batch, "b2_gate": result["b2_gate"], "complete_jacobian_reference": False, "output": str(json_path)}), flush=True)
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        result["error_notes"] = getattr(error, "__notes__", [])
        persist()
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
