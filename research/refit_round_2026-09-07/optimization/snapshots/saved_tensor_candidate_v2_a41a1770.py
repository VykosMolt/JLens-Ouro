"""Conservative identical-lane compression of an ORIGINAL replicated primal.

This candidate uses saved_tensors_hooks only around the native forward. It
does not alter the forward batch, attention backend, activation hooks, or
ordinary autograd.grad calls. No vmap, source perturbations, or GPU work is
performed on import or by --self-test.

Only unambiguous explicit-B / flattened-B*T dense strided tensors whose every
lane is bytewise equal are compressed. Parameters and buffers are excluded by
underlying storage pointer, including transposed/offset aliases. Packed values
hold a cloned lane and metadata, never the original storage. Unpack materializes
the exact original size, stride, storage offset, dtype, device, and value bits.
No reconstructed full tensor is cached between operations or backward calls.

Savings counters describe packed payloads, not measured allocator savings:
other aliases, including ActivationRecorder's requested residuals, can keep an
original storage alive. Equality checks synchronize once per candidate tensor
on CUDA, and every unpack allocates/copies. Real memory and throughput benefit
must be measured. Saved tensors/weights must not be mutated after packing;
saved-tensor hooks do not reproduce all native version-counter checks.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import weakref
from collections import Counter
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _storage_key(tensor: torch.Tensor) -> tuple[str, int | None, int]:
    return tensor.device.type, tensor.device.index, tensor.untyped_storage().data_ptr()


def _dense_nonoverlapping(tensor: torch.Tensor) -> bool:
    # A permutation of a dense contiguous allocation is safe. Skip expanded,
    # overlapping, stepped, negative-stride or otherwise ambiguous layouts.
    expected = 1
    for stride, size in sorted((stride, size) for size, stride in zip(tensor.shape, tensor.stride(), strict=True) if size > 1):
        if stride != expected:
            return False
        expected *= size
    return True


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    # Reinterpret at the same element size: unlike float equality this catches
    # signed-zero differences and preserves/checks NaN payloads. Same-size dtype
    # views also work for dense transposes without making a contiguous copy.
    dtype = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[tensor.element_size()]
    return tensor.detach().view(dtype)


def _bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.shape == right.shape and left.dtype == right.dtype and torch.equal(_bits(left), _bits(right))


@dataclass
class PackedLane:
    lane: torch.Tensor
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    axis: int
    flattened: bool
    batch_size: int
    seq_len: int

    @property
    def reconstruction_elements(self) -> int:
        return self.storage_offset + 1 + sum((size - 1) * stride for size, stride in zip(self.shape, self.stride, strict=True))


class ReplicatedSavedTensorHooks:
    """Context manager for one replicated native forward, with reusable unpack.

    Example::

        compressor = ReplicatedSavedTensorHooks(model, batch_size=B, seq_len=T)
        with ActivationRecorder(...) as recorder, torch.enable_grad():
            with compressor:
                model.forward(ids.expand(B, -1))
            # Ordinary autograd.grad, same original target/source shapes.
            rows = torch.autograd.grad(..., retain_graph=True)
        print(compressor.stats())

    ``model`` may be an nn.Module or the existing HFLensModel/Ouro wrapper.
    Only the context's forward saves are packed; unpack callbacks remain valid
    after exit for every retained backward. Context nesting follows PyTorch's
    innermost-hook behavior. One instance cannot be entered concurrently.
    """

    def __init__(self, model: Any, *, batch_size: int, seq_len: int, min_savings_bytes: int = 4096) -> None:
        self.batch_size = _integer(batch_size, "batch_size", 1)
        self.seq_len = _integer(seq_len, "seq_len", 1)
        self.min_savings_bytes = _integer(min_savings_bytes, "min_savings_bytes", 1)
        modules = [model] if isinstance(model, torch.nn.Module) else [getattr(model, name, None) for name in ("_hf_model", "hf_model", "_text_module")]
        modules = [module for module in modules if isinstance(module, torch.nn.Module)]
        if not modules:
            raise ValueError("model must expose an nn.Module so parameter/buffer storage can be excluded")
        self._persistent_storages = {
            _storage_key(tensor)
            for module in modules
            for tensor in [*module.parameters(), *module.buffers()]
            if tensor.layout == torch.strided and tensor.numel() > 0
        }
        self._counts = Counter()
        self._shapes = Counter()
        # Both references are weak: this cache neither retains original tensor
        # storage nor prolongs packed payload lifetime beyond the autograd graph.
        # Exact object identity + version avoids allocator-pointer reuse errors.
        self._reuse: dict[tuple[int, int], tuple[Any, Any]] = {}
        self._hooks = None

    @property
    def active(self) -> bool:
        return self._hooks is not None

    def _skip(self, tensor: torch.Tensor, reason: str) -> torch.Tensor:
        self._counts[f"skip:{reason}"] += 1
        return tensor.detach()  # Avoid a tensor/grad_fn reference cycle.

    def _layout(self, tensor: torch.Tensor) -> tuple[int, bool] | None:
        shape = tuple(tensor.shape)
        explicit = [axis for axis, size in enumerate(shape) if size == self.batch_size and len(shape) >= 3 and any(other != axis and length == self.seq_len for other, length in enumerate(shape))]
        flattened = [axis for axis, size in enumerate(shape) if size == self.batch_size * self.seq_len and len(shape) >= 2]
        candidates = [(axis, False) for axis in explicit] + [(axis, True) for axis in flattened]
        return candidates[0] if len(candidates) == 1 else None

    def pack(self, tensor: torch.Tensor) -> PackedLane | torch.Tensor:
        self._counts["pack_calls"] += 1
        if self.batch_size == 1:
            return self._skip(tensor, "batch_one")
        if tensor.layout != torch.strided or tensor.is_quantized or tensor.numel() == 0 or tensor.is_conj() or tensor.is_neg():
            return self._skip(tensor, "unsupported_tensor")
        if _storage_key(tensor) in self._persistent_storages:
            return self._skip(tensor, "persistent_storage")
        if tensor.element_size() not in (1, 2, 4, 8) or not _dense_nonoverlapping(tensor):
            return self._skip(tensor, "non_dense_or_unsupported_dtype")
        layout = self._layout(tensor)
        if layout is None:
            return self._skip(tensor, "ambiguous_or_unrecognized_shape")
        full_bytes = tensor.numel() * tensor.element_size()
        lane_bytes = full_bytes // self.batch_size
        if full_bytes - lane_bytes < self.min_savings_bytes:
            return self._skip(tensor, "small_savings")
        key = (id(tensor), tensor._version)
        cached = self._reuse.get(key)
        if cached is not None and cached[0]() is tensor:
            packed = cached[1]()
            if packed is not None:
                self._counts["reused_payloads"] += 1
                return packed
        axis, flattened = layout
        logical = tensor.unflatten(axis, (self.batch_size, self.seq_len)) if flattened else tensor
        first = logical.select(axis, 0)
        first_bits = _bits(first)
        # One broadcast comparison/reduction covers every remaining lane, with
        # one host decision. The temporary bool tensor is transient and keeps
        # no original storage alive after pack; no per-lane launch loop is used.
        remaining_bits = _bits(logical.narrow(axis, 1, self.batch_size - 1))
        equal = (remaining_bits == first_bits.unsqueeze(axis)).all()
        self._counts["lane_comparisons"] += self.batch_size - 1
        self._counts["equality_reductions"] += 1
        if not bool(equal.item()):
            return self._skip(tensor, "lanes_differ")
        # clone is mandatory even when first is already contiguous: a view or
        # contiguous() alone could retain the full original B-lane allocation.
        lane = first.detach().clone(memory_format=torch.contiguous_format)
        packed = PackedLane(lane, tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset(), axis, flattened, self.batch_size, self.seq_len)
        self._reuse[key] = (weakref.ref(tensor), weakref.ref(packed))
        self._counts["compressed_payloads"] += 1
        self._counts["compressed_logical_bytes"] += full_bytes
        self._counts["packed_lane_bytes"] += lane.untyped_storage().nbytes()
        self._counts["nominal_logical_bytes_saved"] += full_bytes - lane.untyped_storage().nbytes()
        self._counts["max_reconstruction_bytes"] = max(self._counts["max_reconstruction_bytes"], packed.reconstruction_elements * lane.element_size())
        self._shapes[str((tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset(), axis, flattened))] += 1
        return packed

    def unpack(self, packed: PackedLane | torch.Tensor) -> torch.Tensor:
        self._counts["unpack_calls"] += 1
        if isinstance(packed, torch.Tensor):
            return packed
        with torch.no_grad():
            storage = torch.empty(packed.reconstruction_elements, dtype=packed.lane.dtype, device=packed.lane.device)
            output = storage.as_strided(packed.shape, packed.stride, packed.storage_offset)
            logical = output.unflatten(packed.axis, (packed.batch_size, packed.seq_len)) if packed.flattened else output
            logical.copy_(packed.lane.unsqueeze(packed.axis).expand_as(logical))
        self._counts["materializations"] += 1
        self._counts["materialized_bytes_total"] += storage.numel() * storage.element_size()
        return output

    def stats(self) -> dict[str, Any]:
        return {"batch_size": self.batch_size, "seq_len": self.seq_len, "min_savings_bytes": self.min_savings_bytes, "active": self.active, "counts": dict(self._counts), "compressed_layout_counts": dict(self._shapes), "byte_count_scope": "logical payload accounting, not measured live allocator savings"}

    def __enter__(self) -> ReplicatedSavedTensorHooks:
        if self.active:
            raise RuntimeError("saved-tensor compressor is already active")
        hooks = torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)
        hooks.__enter__()
        self._hooks = hooks
        return self

    def __exit__(self, *exc) -> None:
        hooks, self._hooks = self._hooks, None
        if hooks is not None:
            hooks.__exit__(*exc)


def _self_test() -> dict[str, Any]:
    from types import SimpleNamespace

    from jlens.fitting import jacobian_for_prompt
    from tests.tiny import TinyDecoder

    torch.set_num_threads(1)
    torch.manual_seed(20260907)
    holder = torch.nn.Linear(5, 5, bias=False).eval().requires_grad_(False)
    compressor = ReplicatedSavedTensorHooks(holder, batch_size=3, seq_len=7, min_savings_bytes=1)
    base = torch.randn(7, 5).unsqueeze(0).repeat(3, 1, 1)
    flat = base.flatten(0, 1)
    offset_storage = torch.empty(base.numel() + 4)
    offset = offset_storage.as_strided(base.shape, base.stride(), 4)
    offset.copy_(base)
    layouts = {"B_T_D": base, "T_B_D": base.transpose(0, 1), "B_D_T": base.transpose(1, 2), "BT_D": flat, "D_BT": flat.T, "offset": offset}
    roundtrips = []
    for name, original in layouts.items():
        packed = compressor.pack(original)
        assert isinstance(packed, PackedLane), (name, compressor.stats())
        assert _storage_key(packed.lane) != _storage_key(original)
        restored = compressor.unpack(packed)
        assert _bitwise_equal(original, restored)
        assert restored.stride() == original.stride() and restored.storage_offset() == original.storage_offset()
        assert _storage_key(restored) != _storage_key(packed.lane)
        roundtrips.append({"layout": name, "shape": list(original.shape), "stride": list(original.stride()), "offset": original.storage_offset(), "value_bits_and_layout_exact": True})

    # Special values require bit equality, not allclose or float torch.equal.
    special = base.clone()
    special[:, 0, 0] = float("nan")
    packed = compressor.pack(special)
    assert isinstance(packed, PackedLane) and _bitwise_equal(special, compressor.unpack(packed))
    signed_zero = torch.zeros_like(base)
    signed_zero[1, 0, 0] = -0.0
    assert isinstance(compressor.pack(signed_zero), torch.Tensor)
    different = base.clone()
    different[2, 4, 1] += 1
    assert isinstance(compressor.pack(different), torch.Tensor)
    expanded = base[:1].expand_as(base)
    assert isinstance(compressor.pack(expanded), torch.Tensor)
    assert isinstance(compressor.pack(base[:, ::2]), torch.Tensor)
    assert isinstance(compressor.pack(torch.ones(3, 3, 7)), torch.Tensor)
    assert isinstance(compressor.pack(holder.weight.T), torch.Tensor)
    assert isinstance(compressor.pack(holder.weight.flatten()[2:]), torch.Tensor)

    def make_disposable_payload():
        original = base.clone()
        tensor_ref = weakref.ref(original)
        storage_ref = weakref.ref(original.untyped_storage())
        assert tensor_ref() is original and storage_ref() is not None
        payload = compressor.pack(original)
        assert isinstance(payload, PackedLane)
        comparisons = compressor.stats()["counts"]["equality_reductions"]
        assert compressor.pack(original) is payload
        assert compressor.stats()["counts"]["equality_reductions"] == comparisons
        return payload, tensor_ref, storage_ref

    disposable, tensor_ref, storage_ref = make_disposable_payload()
    gc.collect()
    assert tensor_ref() is None and storage_ref() is None
    assert _bitwise_equal(compressor.unpack(disposable), base)

    class CausalBlock(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.mix = torch.nn.Linear(width, width, bias=False)
            self.gate = torch.nn.Linear(width, width, bias=False)

        def forward(self, hidden, *, current_ut):
            scores = hidden.float() @ hidden.float().transpose(-1, -2)
            mask = torch.ones(hidden.shape[1], hidden.shape[1], dtype=torch.bool).tril()
            probabilities = scores.masked_fill(~mask, float("-inf")).softmax(-1)
            mixed = (probabilities @ self.mix(hidden).float()).to(hidden.dtype)
            return (hidden + 0.15 * torch.nn.functional.silu(self.gate(mixed)),)

    class LoopTap:
        def __init__(self, block, ut):
            self.block, self.ut = block, ut

        def register_forward_hook(self, hook):
            def filtered(module, args, kwargs, output):
                if kwargs["current_ut"] == self.ut:
                    hook(module, args, output)

            return self.block.register_forward_hook(filtered, with_kwargs=True)

    class RecurrentTiny(TinyDecoder):
        def __init__(self):
            super().__init__(n_layers=4, d_model=5, seed=11)
            del self.layers
            self.blocks = torch.nn.ModuleList([CausalBlock(5), CausalBlock(5)])
            self.layers = [LoopTap(block, ut) for ut in range(2) for block in self.blocks]
            self.compressor = None
            self.fail_forward = False

        def _forward(self, input_ids):
            hidden = self.embed_tokens(input_ids)
            for ut in range(2):
                for block in self.blocks:
                    hidden = block(hidden, current_ut=ut)[0]
                    if self.fail_forward:
                        raise RuntimeError("injected forward failure")
                if ut == 0:
                    hidden = self.norm(hidden)
            return SimpleNamespace(last_hidden_state=hidden)

        def forward(self, input_ids):
            if self.compressor is None:
                return self._forward(input_ids)
            with self.compressor:
                return self._forward(input_ids)

    parity = []
    for dtype in (torch.float32, torch.bfloat16):
        model = RecurrentTiny().to(dtype).eval().requires_grad_(False)
        for batch in (1, 2, 4, 8):
            for target in (2, 3):
                kwargs = dict(model=model, prompt="replicated CPU parity", source_layers=list(range(target)), target_layer=target, max_seq_len=7, skip_first=1, dim_batch=batch)
                model.compressor = None
                reference, _, _ = jacobian_for_prompt(**kwargs)
                model.compressor = ReplicatedSavedTensorHooks(model, batch_size=batch, seq_len=7, min_savings_bytes=1)
                candidate, _, _ = jacobian_for_prompt(**kwargs)
                assert all(_bitwise_equal(candidate[layer], reference[layer]) for layer in reference)
                assert not model.compressor.active
                assert all(not block._forward_hooks for block in model.blocks)
                assert all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters())
                counts = model.compressor.stats()["counts"]
                if batch > 1:
                    assert counts.get("compressed_payloads", 0) > 0 and counts.get("materializations", 0) > 0
                parity.append({"dtype": str(dtype), "batch": batch, "target": target, "full_jacobians_bitwise_equal": True, "counts": counts})

    model.fail_forward = True
    try:
        jacobian_for_prompt(model, "failing prompt", [0, 1, 2], dim_batch=8, max_seq_len=7, skip_first=1)
    except RuntimeError as error:
        assert "injected forward failure" in str(error)
    else:
        raise AssertionError("expected a forward exception")
    assert not model.compressor.active and all(not block._forward_hooks for block in model.blocks)
    before = model.compressor.stats()["counts"].get("pack_calls", 0)
    x = torch.randn(3, requires_grad=True)
    torch.autograd.grad((x * x).sum(), x)
    assert model.compressor.stats()["counts"].get("pack_calls", 0) == before
    return {"device": "cpu", "torch_version": torch.__version__, "roundtrips": roundtrips, "original_tensor_and_storage_released": True, "nan_payload_and_signed_zero_guards": "pass", "full_jacobian_parity": parity, "exception_cleanup": "pass", "cuda_initialized": torch.cuda.is_initialized()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run CPU-only layout, release, and exact-Jacobian checks")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(_self_test(), indent=2))
    else:
        parser.print_help()
