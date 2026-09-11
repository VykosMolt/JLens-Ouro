"""Experimental exact reduced-source VJPs with modern ``torch.func`` batching.

This opt-in research candidate does not alter jlens or select an attention
backend. Importing it and ``--self-test`` do not use CUDA or load Ouro.

For valid positions V, M = |V|, a zero FP32 vector delta_l[D] is injected after
each requested source block as ``(h.float() + mask * delta_l / M).to(h.dtype)``.
The function returns the FP32 SUM of target residuals over V. Consequently,
d f / d delta_l is exactly the existing sum-over-target / mean-over-source
estimator in real arithmetic. Floating-point reduction order can differ;
actual-model numerical and performance parity remain separate acceptance gates.

One native batch-one forward constructs ``torch.func.vjp``. Its retained
pullback is then vmapped over successive one-hot output cotangents, using
modern FuncTorchBatched rules rather than legacy ``is_grads_batched`` rules.
No source-position gradient tensors are returned to Python. Fused attention
backward may still lack suitable batching rules; errors propagate without
silently switching backends. The caller must provide a frozen eval model on
one device and must keep weights unchanged until the context is closed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from jlens.fitting import _check_layer_indices, valid_position_mask  # noqa: E402
from jlens.protocol import LensModel  # noqa: E402


def _integer(value: Any, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _hidden(output: Any) -> torch.Tensor:
    hidden = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(hidden, torch.Tensor):
        raise TypeError("hooked block must return a tensor or a tuple/list with tensor first")
    return hidden


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        if hasattr(output, "_fields"):
            return type(output)(hidden, *output[1:])
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    return hidden


def _register_hook(layer: Any, hook: Callable[..., Any]) -> Any:
    # Ouro's LoopTap filters current_ut but drops a modifying hook's return
    # value. Attach to the same physical block, preserving both the filter and
    # the replacement output. Ordinary layer hooks retain their normal API.
    if hasattr(layer, "block") and hasattr(layer, "ut"):
        def filtered(module, args, kwargs, output):
            if kwargs.get("current_ut") == layer.ut:
                return hook(module, args, output)
            return None

        return layer.block.register_forward_hook(filtered, with_kwargs=True)
    return layer.register_forward_hook(hook)


@dataclass
class ReducedVJPContext:
    """One retained primal, with hooks already removed.

    ``rows(start, batch)`` returns {source: FP32[actual_batch, D]} on the
    model's device. A final partial batch is truncated. ``apply_cotangents``
    also accepts arbitrary FP32[K,D] cotangents for parity/linearity checks.
    Outputs own their storage, unlike CUDA-graph borrowed output buffers.
    Calls are first-order only and may be repeated in any order. ``close``
    drops the retained graph; no extra backward or device synchronization is
    performed. The context must not be used concurrently with model mutation.
    """

    source_layers: tuple[int, ...]
    target_layer: int
    seq_len: int
    n_valid_positions: int
    d_model: int
    target_sum: torch.Tensor
    _pullback: Callable[..., Any] | None
    primal_audit: dict[str, tuple[torch.Tensor, ...]] = field(default_factory=dict)

    @property
    def device(self) -> torch.device:
        return self.target_sum.device

    @property
    def closed(self) -> bool:
        return self._pullback is None

    def apply_cotangents(self, cotangents: torch.Tensor) -> dict[int, torch.Tensor]:
        if self.closed:
            raise RuntimeError("ReducedVJPContext is closed")
        if (
            not isinstance(cotangents, torch.Tensor)
            or cotangents.ndim != 2
            or cotangents.shape[0] < 1
            or cotangents.shape[1] != self.d_model
            or cotangents.dtype != torch.float32
            or cotangents.device != self.device
        ):
            raise ValueError("cotangents must be nonempty FP32[K,D] on the target device")
        pullback = self._pullback
        assert pullback is not None

        def single(cotangent):
            # torch.func.vjp defaults create_graph to ambient grad mode. Force
            # first-order gradients while retaining this one primal for reuse.
            return pullback(cotangent, retain_graph=True, create_graph=False)[0]

        rows = torch.vmap(single, randomness="error")(cotangents)
        return dict(zip(self.source_layers, rows, strict=True))

    def apply_scalar(self, cotangent: torch.Tensor) -> dict[int, torch.Tensor]:
        """Ordinary unbatched VJP on this exact primal, for engine parity checks."""
        if self.closed:
            raise RuntimeError("ReducedVJPContext is closed")
        if (
            not isinstance(cotangent, torch.Tensor)
            or cotangent.shape != (self.d_model,)
            or cotangent.dtype != torch.float32
            or cotangent.device != self.device
        ):
            raise ValueError("cotangent must be FP32[D] on the target device")
        assert self._pullback is not None
        rows = self._pullback(cotangent, retain_graph=True, create_graph=False)[0]
        return dict(zip(self.source_layers, rows, strict=True))

    def rows(self, dim_start: int, dim_batch: int) -> dict[int, torch.Tensor]:
        if self.closed:
            raise RuntimeError("ReducedVJPContext is closed")
        dim_start = _integer(dim_start, "dim_start", 0)
        dim_batch = _integer(dim_batch, "dim_batch", 1)
        if dim_start >= self.d_model:
            raise ValueError(f"dim_start must be < d_model={self.d_model}")
        n_rows = min(dim_batch, self.d_model - dim_start)
        cotangents = torch.zeros(n_rows, self.d_model, device=self.device, dtype=torch.float32)
        indices = torch.arange(n_rows, device=self.device)
        cotangents[indices, dim_start + indices] = 1.0
        return self.apply_cotangents(cotangents)

    def close(self) -> None:
        self._pullback = None

    def __enter__(self) -> ReducedVJPContext:
        if self.closed:
            raise RuntimeError("ReducedVJPContext is closed")
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def prepare_reduced_vjp(
    model: LensModel,
    prompt: str,
    source_layers: Sequence[int] | None = None,
    *,
    target_layer: int | None = None,
    max_seq_len: int = 128,
    skip_first: int = 16,
    audit_primal: bool = False,
    audit_modules: dict[str, torch.nn.Module] | None = None,
) -> ReducedVJPContext:
    """Construct the exact reduced function and one reusable modern pullback.

    This runs exactly one model forward at batch size one. The model's native
    attention and loop-carry behavior are preserved. Sources must precede the
    target, and each requested virtual hook must fire exactly once. Hooks are
    removed in a finally block, including forward/transform failures. Frozen
    parameters need no ``requires_grad_`` calls inside the function transform.
    ``audit_primal`` saves detached residuals at every virtual block; optional
    ``audit_modules`` additionally saves every invocation of named modules
    such as Ouro's loop-end norm. These diagnostics add memory/clone work and
    must be disabled for throughput benchmarks.
    """
    n_layers = _integer(model.n_layers, "model.n_layers", 1)
    d_model = _integer(model.d_model, "model.d_model", 1)
    max_seq_len = _integer(max_seq_len, "max_seq_len", 1)
    skip_first = _integer(skip_first, "skip_first", 0)
    if target_layer is not None:
        target_layer = _integer(target_layer, "target_layer")
    if source_layers is not None:
        source_layers = [_integer(layer, "source layer") for layer in source_layers]
    sources, target = _check_layer_indices(source_layers, target_layer, n_layers)
    if not sources:
        raise ValueError("at least one source layer preceding the target is required")

    input_ids = model.encode(prompt, max_length=max_seq_len)
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("model.encode must return a tensor with shape [1, sequence]")
    seq_len = input_ids.shape[1]
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    n_valid = int(position_mask.sum())
    positions = position_mask.nonzero(as_tuple=True)[0].to(input_ids.device)
    mask = position_mask.to(device=input_ids.device, dtype=torch.float32).view(1, seq_len, 1)
    zeros = tuple(torch.zeros(d_model, device=input_ids.device, dtype=torch.float32) for _ in sources)

    def reduced_function(deltas):
        handles = []
        counts = {layer: 0 for layer in [*sources, target]}
        captured = []
        audit = {}

        def check_hidden(output):
            hidden = _hidden(output)
            if hidden.shape != (1, seq_len, d_model) or hidden.device != input_ids.device:
                raise ValueError("all hooked residuals must have shape [1,S,D] on the input device")
            if hidden.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError("candidate supports FP16, BF16 or FP32 residuals only")
            return hidden

        def source_hook(layer, delta):
            def inject(module, args, output):
                counts[layer] += 1
                hidden = check_hidden(output)
                injected = (hidden.float() + mask * (delta / n_valid).view(1, 1, d_model)).to(hidden.dtype)
                return _replace_hidden(output, injected)

            return inject

        def target_hook(module, args, output):
            counts[target] += 1
            hidden = check_hidden(output)
            if not hidden.requires_grad:
                raise RuntimeError("target is not differentiable; check model no_grad/inference decorators")
            captured.append(hidden.index_select(1, positions).float().sum(dim=1).squeeze(0))

        def audit_hook(name):
            audit[name] = []

            def observe(module, args, output):
                audit[name].append(_hidden(output).detach().clone())

            return observe

        try:
            for layer, delta in zip(sources, deltas, strict=True):
                handles.append(_register_hook(model.layers[layer], source_hook(layer, delta)))
            handles.append(_register_hook(model.layers[target], target_hook))
            if audit_primal:
                for layer in range(n_layers):
                    handles.append(_register_hook(model.layers[layer], audit_hook(f"layer:{layer}")))
            for name, module in (audit_modules or {}).items():
                handles.append(module.register_forward_hook(audit_hook(f"module:{name}")))
            model.forward(input_ids)
            if any(count != 1 for count in counts.values()):
                raise RuntimeError(f"each requested virtual layer must execute exactly once: {counts}")
            return captured[0], {name: tuple(values) for name, values in audit.items()}
        finally:
            for handle in handles:
                handle.remove()

    target_sum, pullback, primal_audit = torch.func.vjp(reduced_function, zeros, has_aux=True)
    return ReducedVJPContext(
        source_layers=tuple(sources),
        target_layer=target,
        seq_len=seq_len,
        n_valid_positions=n_valid,
        d_model=d_model,
        target_sum=target_sum.detach(),
        _pullback=pullback,
        primal_audit=primal_audit,
    )


def _self_test() -> dict[str, Any]:
    """CPU parity against the upstream estimator; no model checkpoint needed."""
    from types import SimpleNamespace

    from jlens.fitting import jacobian_for_prompt
    from tests.tiny import TinyDecoder

    torch.set_num_threads(1)

    class CausalBlock(torch.nn.Module):
        def __init__(self, width, output_kind):
            super().__init__()
            self.mix = torch.nn.Linear(width, width, bias=False)
            self.gate = torch.nn.Linear(width, width, bias=False)
            self.output_kind = output_kind

        def forward(self, hidden, *, current_ut=None):
            scores = hidden.float() @ hidden.float().transpose(-1, -2)
            causal = torch.ones(hidden.shape[1], hidden.shape[1], dtype=torch.bool).tril()
            attention = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
            mixed = (attention @ self.mix(hidden).float()).to(hidden.dtype)
            result = hidden + 0.15 * torch.nn.functional.silu(self.gate(mixed))
            return (result, None) if self.output_kind == "tuple" else [result, None]

    class DroppingLoopTap:
        def __init__(self, block, ut):
            self.block, self.ut = block, ut

        def register_forward_hook(self, hook):
            def filtered(module, args, kwargs, output):
                if kwargs["current_ut"] == self.ut:
                    hook(module, args, output)  # Mirrors the actual Ouro tap.

            return self.block.register_forward_hook(filtered, with_kwargs=True)

    class RecurrentTiny(TinyDecoder):
        def __init__(self):
            super().__init__(n_layers=4, d_model=5, seed=19)
            del self.layers
            self.blocks = torch.nn.ModuleList([CausalBlock(5, "tuple"), CausalBlock(5, "list")])
            self.layers = [DroppingLoopTap(block, ut) for ut in range(2) for block in self.blocks]
            self.fail_forward = False

        def forward(self, input_ids):
            hidden = self.embed_tokens(input_ids)
            for ut in range(2):
                for block in self.blocks:
                    hidden = block(hidden, current_ut=ut)[0]
                    if self.fail_forward:
                        raise RuntimeError("injected forward failure")
                if ut == 0:
                    hidden = self.norm(hidden)  # Carry boundary must remain in the derivative.
            return SimpleNamespace(last_hidden_state=hidden)

    prompt = "a causal CPU parity prompt"
    results = []
    models = [("linear_fp32", TinyDecoder(d_model=5)), ("recurrent_causal_fp32", RecurrentTiny()), ("recurrent_causal_bf16", RecurrentTiny().to(torch.bfloat16))]
    for name, model in models:
        model.eval().requires_grad_(False)
        for target, sources, skip in [(3, [0, 1, 2], 2), (2, [0, 1], 0)]:
            dense, seq_len, n_valid = jacobian_for_prompt(
                model, prompt, sources, target_layer=target, dim_batch=3, max_seq_len=13, skip_first=skip
            )
            input_ids = model.encode(prompt, max_length=13)
            positions = valid_position_mask(seq_len, skip_first=skip).nonzero(as_tuple=True)[0]
            native_audit = {}
            handles = []

            def audit_hook(name):
                native_audit[name] = []

                def observe(module, args, output):
                    native_audit[name].append(_hidden(output).detach().clone())

                return observe

            for layer in range(model.n_layers):
                handles.append(_register_hook(model.layers[layer], audit_hook(f"layer:{layer}")))
            handles.append(model.norm.register_forward_hook(audit_hook("module:carry_norm")))
            try:
                with torch.no_grad():
                    model.forward(input_ids)
            finally:
                for handle in handles:
                    handle.remove()
            native_sum = native_audit[f"layer:{target}"][0].index_select(1, positions).float().sum(dim=1).squeeze(0)
            with prepare_reduced_vjp(
                model, prompt, sources, target_layer=target, max_seq_len=13, skip_first=skip,
                audit_primal=True, audit_modules={"carry_norm": model.norm},
            ) as context:
                assert native_audit.keys() == context.primal_audit.keys()
                for key, native_values in native_audit.items():
                    assert len(native_values) == len(context.primal_audit[key])
                    for native, reduced in zip(native_values, context.primal_audit[key], strict=True):
                        torch.testing.assert_close(reduced, native, rtol=0, atol=0)
                        assert reduced.dtype == native.dtype and not reduced.requires_grad
                torch.testing.assert_close(context.target_sum, native_sum, rtol=0, atol=0)
                assert context.seq_len == seq_len and context.n_valid_positions == n_valid
                assembled = {layer: torch.empty_like(dense[layer]) for layer in sources}
                for start in range(0, model.d_model, 3):
                    for layer, rows in context.rows(start, 3).items():
                        assembled[layer][start : start + len(rows)] = rows
                max_abs = max(float((assembled[layer] - dense[layer]).abs().max()) for layer in sources)
                # CPU BF16 batched/replicated matmuls can differ at their BF16
                # rounding steps. FP32 should match to normal reduction error.
                tolerance = dict(rtol=0.02, atol=0.003) if "bf16" in name else dict(rtol=1e-6, atol=2e-7)
                for layer in sources:
                    torch.testing.assert_close(assembled[layer], dense[layer], **tolerance)
                    torch.testing.assert_close(context.rows(3, 3)[layer], assembled[layer][3:5], rtol=0, atol=0)
                same_primal_scalar_max_abs = 0.0
                for row_index in range(model.d_model):
                    cotangent = torch.zeros(model.d_model)
                    cotangent[row_index] = 1.0
                    scalar = context.apply_scalar(cotangent)
                    for layer in sources:
                        same_primal_scalar_max_abs = max(same_primal_scalar_max_abs, float((scalar[layer] - assembled[layer][row_index]).abs().max()))
                        torch.testing.assert_close(scalar[layer], assembled[layer][row_index], **tolerance)
                for batch in (1, 2, 4, 8):
                    for start in range(0, model.d_model, batch):
                        for layer, rows in context.rows(start, batch).items():
                            torch.testing.assert_close(rows, assembled[layer][start : start + len(rows)], **tolerance)
                zero_rows = context.apply_cotangents(torch.zeros(2, model.d_model))
                assert all(torch.count_nonzero(rows) == 0 for rows in zero_rows.values())
                random_cotangents = torch.randn(2, model.d_model)
                combined = context.apply_cotangents(random_cotangents)
                for layer in sources:
                    # BF16 kernels round a combined cotangent at each backward
                    # operation; equality to a sum of rounded rows is approximate.
                    torch.testing.assert_close(combined[layer], random_cotangents @ assembled[layer], **tolerance)
                assert all(not rows.requires_grad for rows in combined.values())
                results.append({"model": name, "target": target, "sources": sources, "max_abs_error": max_abs, "same_primal_scalar_max_abs_error": same_primal_scalar_max_abs, "every_virtual_primal_and_carry_bitwise_equal": True})
            try:
                context.rows(0, 1)
            except RuntimeError:
                pass
            else:
                raise AssertionError("closed context remained usable")

    # Materialize the full [target_position, output, source_position, input]
    # Jacobian using a directly written suffix, independent of both estimators'
    # hooks and reductions. The suffix explicitly includes the tied-loop norm.
    model = models[1][1]
    explicit_checks = []
    for length, skip in ((5, 1), (6, 0), (7, 5)):
        ids = model.encode(prompt, max_length=length)
        positions = valid_position_mask(length, skip_first=skip).nonzero(as_tuple=True)[0]
        for target in (2, 3):
            sources = list(range(target))
            native_sources = {}
            with torch.no_grad():
                hidden = model.embed_tokens(ids)
                for virtual in range(4):
                    hidden = model.blocks[virtual % 2](hidden, current_ut=virtual // 2)[0]
                    native_sources[virtual] = hidden[0].detach()
                    if virtual == 1:
                        hidden = model.norm(hidden)

            def suffix(hidden, source):
                hidden = hidden.unsqueeze(0)
                for virtual in range(source + 1, target + 1):
                    if virtual == 2:
                        hidden = model.norm(hidden)
                    hidden = model.blocks[virtual % 2](hidden, current_ut=virtual // 2)[0]
                return hidden[0]

            with prepare_reduced_vjp(model, prompt, sources, target_layer=target, max_seq_len=length, skip_first=skip) as context:
                actual = context.rows(0, 8)
                worst = 0.0
                for source in sources:
                    full = torch.autograd.functional.jacobian(lambda h: suffix(h, source), native_sources[source])
                    expected = full.index_select(0, positions).index_select(2, positions).sum(dim=0).sum(dim=1) / len(positions)
                    torch.testing.assert_close(actual[source], expected, rtol=1e-6, atol=2e-7)
                    worst = max(worst, float((actual[source] - expected).abs().max()))
                explicit_checks.append({"length": length, "skip_first": skip, "target": target, "max_abs_error": worst})

    for sources, target in [([3], 3), ([2], 1), ([], 3), ([True], 3), ([0], False), (None, 0)]:
        try:
            prepare_reduced_vjp(model, prompt, sources, target_layer=target, skip_first=0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid layers: {sources}, {target}")
    try:
        prepare_reduced_vjp(model, "x", [0], target_layer=3, skip_first=2)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted an empty valid-position set")
    model.fail_forward = True
    try:
        prepare_reduced_vjp(model, prompt, [0, 1, 2], target_layer=3, skip_first=0)
    except RuntimeError as error:
        assert "injected forward failure" in str(error)
    else:
        raise AssertionError("expected injected forward failure")
    assert all(len(block._forward_hooks) == 0 for block in model.blocks)
    return {"device": "cpu", "torch_version": torch.__version__, "width": 5, "dimension_batches": [1, 2, 4, 8], "checks": results, "explicit_position_jacobian_checks": explicit_checks, "invalid_layer_guards": "pass", "empty_position_guard": "pass", "exception_hook_cleanup": "pass", "cuda_initialized": torch.cuda.is_initialized()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run CPU-only tiny-model parity checks")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(_self_test(), indent=2))
    else:
        parser.print_help()
