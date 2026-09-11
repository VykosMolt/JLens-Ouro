"""Paired single-position controls using the accepted replicated VJP engine.

For a caller-selected q in V, one backward stream yields both
``sampled_sum = sum_{p in V} d h_target[q] / d h_source[p]`` and
``diagonal = d h_target[q] / d h_source[q]``. There is no source-position
divisor: averaging sampled_sum over uniformly sampled q recovers the dense
estimator. Calibration selection, q draws and prompt averaging are external.

The native B-lane forward and all derivative directions are preserved. Only
the two source reductions use FP32; completed matrices live on CPU. CUDA
replay captures the backward and both reductions together, with one stacked
row transfer per direction chunk. No compilation or fallback is performed.
Importing this module or running --self-test does not use CUDA or load weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Sequence
from contextlib import nullcontext
from numbers import Integral
from pathlib import Path

import torch

ROUND = Path(__file__).resolve().parents[1]
REPO = ROUND.parents[1]
for path in (REPO, ROUND):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from jlens.fitting import _check_layer_indices, valid_position_mask  # noqa: E402
from optimization.cuda_graph_candidate import CapturedVJP  # noqa: E402
from optimization.optimized_fitting import SourceEdgeRecorder  # noqa: E402
from optimization.saved_tensor_candidate import ReplicatedSavedTensorHooks  # noqa: E402

ARMS = ("sampled_sum", "diagonal")


def _integer(name: str, value: object, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def paired_jacobians_for_prompt(
    model,
    prompt: str,
    source_layers: Sequence[int],
    *,
    q: int,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = 16,
    engine: str = "cuda_graph",
    compress_saved_tensors: bool = False,
    capture_warmup: int = 1,
    diagnostics: dict | None = None,
) -> tuple[dict[str, dict[int, torch.Tensor]], int, int]:
    """Return both CPU FP32 matrix banks, encoded length and number of valid q.

    ``q`` is an absolute position in the actual encoded, truncated sequence,
    including any BOS. It must belong to V = {skip_first, ..., length - 2}.
    Layer indices follow the released fitter; all sources precede the target.
    The caller supplies a deterministic model in evaluation mode with frozen
    parameters. This function does not alter model dtype, mode or parameters.

    ``engine='eager'`` supports CPU and CUDA. ``'cuda_graph'`` requires one
    CUDA device and uses the same explicit stream for primal and backward.
    A partial final chunk slices source gradients before either reduction,
    while retaining the original full B-lane forward and backward shapes.
    Optional saved-tensor hooks reconstruct exact native values and layouts.
    Nonfinite reduced maps raise before return; hooks and capture resources
    are cleaned up on failures. CUDA parity and performance require a separate
    real-device check; the bundled self-test exercises only the CPU eager path.
    """
    if engine not in ("eager", "cuda_graph"):
        raise ValueError("engine must be 'eager' or 'cuda_graph'")
    if type(compress_saved_tensors) is not bool:
        raise ValueError("compress_saved_tensors must be a boolean")
    dim_batch = _integer("dim_batch", dim_batch, 1)
    max_seq_len = _integer("max_seq_len", max_seq_len, 1)
    skip_first = _integer("skip_first", skip_first, 0)
    capture_warmup = _integer("capture_warmup", capture_warmup, 1)
    q = _integer("q", q, 0)
    d_model = _integer("model.d_model", model.d_model, 1)
    n_layers = _integer("model.n_layers", model.n_layers, 1)
    if source_layers is None:
        raise ValueError("source_layers must be an explicit nonempty sequence")
    sources = [_integer("source layer", layer) for layer in source_layers]
    if target_layer is not None:
        target_layer = _integer("target_layer", target_layer)
    sources, target = _check_layer_indices(sources, target_layer, n_layers)
    input_ids = model.encode(prompt, max_length=max_seq_len)
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("model.encode must return a tensor with shape [1, tokens]")
    seq_len = int(input_ids.shape[1])
    mask = valid_position_mask(seq_len, skip_first=skip_first)
    if q >= seq_len or not bool(mask[q]):
        raise ValueError(f"q={q} is not in valid positions [{skip_first}, {seq_len - 2}]")
    n_valid = int(mask.sum())
    device = input_ids.device
    if engine == "cuda_graph" and device.type != "cuda":
        raise ValueError("CUDA graph engine requires a CUDA model")

    info = diagnostics if diagnostics is not None else {}
    info.update(engine=engine, source_edges=True, dim_batch=dim_batch,
                sequence_length=seq_len, n_valid=n_valid, q=q, target=target,
                source_count=len(sources), n_passes=math.ceil(d_model / dim_batch),
                bank_arms=list(ARMS), warmup_seconds=0.0, capture_seconds=0.0,
                warmup_calls=0, completed_row_chunks=0)
    started = time.perf_counter()
    # Every entry is assigned before return. The two banks share a contiguous
    # allocation, so each direction chunk requires only one device-to-host copy.
    bank = torch.empty((len(ARMS), len(sources), d_model, d_model),
                       dtype=torch.float32, device="cpu")
    info["allocate_cpu_seconds"] = time.perf_counter() - started
    storage = (ReplicatedSavedTensorHooks(model, batch_size=dim_batch, seq_len=seq_len)
               if compress_saved_tensors else None)
    stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    captured = None
    try:
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream(device))
        with (
            torch.cuda.stream(stream) if stream is not None else nullcontext(),
            torch.enable_grad(),
            SourceEdgeRecorder(model.layers, sources=sources, target=target) as recorder,
        ):
            started = time.perf_counter()
            with storage if storage is not None else nullcontext():
                model.forward(input_ids.expand(dim_batch, -1))
            if stream is not None:
                stream.synchronize()
            info["forward_seconds"] = time.perf_counter() - started
            missing = {*sources, target} - recorder.metadata.keys()
            if missing:
                raise RuntimeError(f"layer hooks did not fire: {sorted(missing)}")
            if any(meta[1] != device for meta in recorder.metadata.values()):
                raise ValueError("paired fitter requires all activations on one device")
            if any(meta[0] != (dim_batch, seq_len, d_model) for meta in recorder.metadata.values()):
                raise ValueError("source/target activation shape is inconsistent")
            target_activation = recorder.activations[target]
            edges = tuple(recorder.edges[layer] for layer in sources)
            cotangent = torch.zeros_like(target_activation)
            positions = mask.nonzero(as_tuple=True)[0].to(device)
            batch_indices = torch.arange(dim_batch, device=device)

            def write(buffer, dim_start):
                n_dims = min(dim_batch, d_model - dim_start)
                lanes = batch_indices[:n_dims]
                buffer.zero_()
                buffer[lanes, q, dim_start + lanes] = 1

            def reduce(grads):
                sum_rows, diagonal_rows = [], []
                for grad in grads:
                    sum_rows.append(grad[:, positions, :].float().sum(dim=1))
                    diagonal_rows.append(grad[:, q, :].float())
                return (torch.stack((torch.stack(sum_rows), torch.stack(diagonal_rows))),)

            def eager_rows(n_dims):
                grads = torch.autograd.grad(target_activation, edges, cotangent,
                                            retain_graph=True, create_graph=False,
                                            allow_unused=False)
                return reduce(tuple(grad[:n_dims] for grad in grads))[0]

            if engine == "cuda_graph" and dim_batch <= d_model:
                write(cotangent, 0)
                captured = CapturedVJP(target_activation, edges, cotangent,
                                      stream=stream, reducer=reduce, warmup=capture_warmup)
                info.update(warmup_seconds=captured.warmup_seconds,
                            capture_seconds=captured.capture_seconds,
                            warmup_calls=captured.warmup_calls)
            if stream is not None:
                stream.synchronize()
            started = time.perf_counter()
            for dim_start in range(0, d_model, dim_batch):
                n_dims = min(dim_batch, d_model - dim_start)
                if captured is not None and n_dims == dim_batch:
                    write(captured.cotangent, dim_start)
                    rows = captured.replay()[0]
                else:
                    write(cotangent, dim_start)
                    rows = eager_rows(n_dims)
                host_rows = rows.cpu()
                if not bool(torch.isfinite(host_rows).all()):
                    raise FloatingPointError(
                        f"nonfinite paired matrix rows at target {target}, q={q}, dim_start={dim_start}"
                    )
                bank[:, :, dim_start:dim_start + n_dims, :].copy_(host_rows)
                info["completed_row_chunks"] += 1
                del rows, host_rows
            if stream is not None:
                stream.synchronize()
            info["all_rows_seconds"] = time.perf_counter() - started
    finally:
        try:
            if captured is not None:
                captured.close()
        finally:
            if stream is not None:
                stream.synchronize()
            if storage is not None:
                info["saved_tensor_compression"] = storage.stats()
    return {
        arm: {layer: bank[arm_index, source_index]
              for source_index, layer in enumerate(sources)}
        for arm_index, arm in enumerate(ARMS)
    }, seq_len, n_valid


def _self_test() -> dict:
    """Independent causal oracle, original sampled parity and failure cleanup."""
    import copy
    from unittest.mock import patch

    from design_audit.verify_estimators import (
        BadGradientBlock,
        CausalBlock,
        check_hooks_clean,
        close_error,
        explicit_blocks,
        require_raises,
    )
    from fit_estimators import jacobians_for_prompt
    from tests.tiny import TinyDecoder

    torch.set_num_threads(1)
    cases = []
    oracle_errors = []
    compressed_payloads = 0
    # Lower only the storage-size threshold so these tiny fixtures actually
    # exercise exact packing and reconstruction, rather than the small skip.
    hooks_type = ReplicatedSavedTensorHooks
    compressors = []

    def tiny_hooks(*args, **kwargs):
        hooks = hooks_type(*args, **kwargs, min_savings_bytes=1)
        compressors.append(hooks)
        return hooks

    for dtype in (torch.float64, torch.float32, torch.bfloat16):
        model = TinyDecoder(n_layers=4, d_model=5, seed=20260907)
        model.layers = torch.nn.ModuleList([CausalBlock(5) for _ in range(4)])
        model.to(dtype).eval().requires_grad_(False)
        for prompt in ("ab", "abcd", "abcdefgh"):
            length = int(model.encode(prompt, max_length=5).shape[1])
            valid = valid_position_mask(length, skip_first=1).nonzero().flatten()
            for target in (1, 2, 3):
                sources = list(range(target))
                exact = ({source: explicit_blocks(model, prompt, source, target)
                          for source in sources} if dtype == torch.float64 else None)
                for q in valid.tolist():
                    for batch in (1, 2, 8):
                        kwargs = dict(model=model, prompt=prompt, source_layers=sources,
                                      target_layer=target, dim_batch=batch, max_seq_len=5,
                                      skip_first=1, q=q)
                        expected, expected_length, expected_valid = jacobians_for_prompt(
                            **kwargs, mode="sampled")
                        for compress in (False, True):
                            details = {}
                            with (
                                patch(__name__ + ".ReplicatedSavedTensorHooks", tiny_hooks),
                                patch("torch.autograd.grad", wraps=torch.autograd.grad) as backward,
                                patch.object(model, "forward", wraps=model.forward) as forward,
                            ):
                                maps, actual_length, actual_valid = paired_jacobians_for_prompt(
                                    **kwargs, engine="eager", compress_saved_tensors=compress,
                                    diagnostics=details)
                            assert backward.call_count == math.ceil(5 / batch)
                            assert forward.call_count == 1
                            assert tuple(forward.call_args.args[0].shape) == (batch, length)
                            assert (actual_length, actual_valid) == (expected_length, expected_valid)
                            assert set(maps) == set(ARMS)
                            for arm in ARMS:
                                assert set(maps[arm]) == set(sources)
                                for source, matrix in maps[arm].items():
                                    assert matrix.dtype == torch.float32 and matrix.device.type == "cpu"
                                    assert matrix.shape == (5, 5) and not matrix.requires_grad
                                    assert torch.isfinite(matrix).all()
                                    assert torch.equal(matrix, expected[arm][source])
                                    if exact is not None:
                                        g = exact[source]
                                        oracle = (g[q].index_select(1, valid).sum(dim=1)
                                                  if arm == "sampled_sum" else g[q, :, q, :])
                                        oracle_errors.append(close_error(matrix, oracle))
                            stats = details.get("saved_tensor_compression", {})
                            compressed_payloads += stats.get("counts", {}).get("compressed_payloads", 0)
                            assert not stats.get("active", False)
                            check_hooks_clean(model)
                            cases.append(dict(dtype=str(dtype), length=length, truncated=len(prompt) > 4,
                                              q=q, target=target, batch=batch, compressed=compress,
                                              bitwise_equal_to_original=True))

    assert compressed_payloads > 0
    assert all(not hooks.active for hooks in compressors)
    kwargs = dict(model=model, prompt="abcd", source_layers=[0, 1, 2],
                  target_layer=3, dim_batch=2, max_seq_len=5, skip_first=1,
                  q=2, engine="eager")
    invalid = [
        {name: value}
        for name, values in (
            ("q", (None, True, 1.5, -1, 0, 4, 5)),
            ("dim_batch", (True, 0, -1, 1.5)),
            ("max_seq_len", (0, True, 1.5)),
            ("skip_first", (-1, True, 4)),
            ("capture_warmup", (0, True, 1.5)),
            ("source_layers", (None, [], [True], [0.5], [3], [-5])),
            ("target_layer", (True, 1.5, 4)),
            ("engine", ("stock", "cuda_graph")),
            ("compress_saved_tensors", (1, None)),
        ) for value in values
    ]
    invalid.append({"prompt": ""})
    for change in invalid:
        require_raises(ValueError, paired_jacobians_for_prompt, **{**kwargs, **change})
        check_hooks_clean(model)
    positive, _, _ = paired_jacobians_for_prompt(**kwargs)
    negative, _, _ = paired_jacobians_for_prompt(
        **{**kwargs, "source_layers": [-4, -3, -2, -4], "target_layer": -1})
    assert all(torch.equal(positive[arm][source], negative[arm][source])
               for arm in ARMS for source in positive[arm])
    for malformed in (torch.zeros(5, dtype=torch.long), torch.zeros(2, 5, dtype=torch.long), None):
        with patch.object(model, "encode", return_value=malformed):
            require_raises(ValueError, paired_jacobians_for_prompt, **kwargs)

    broken = copy.deepcopy(model)
    broken.forward = lambda ids: None
    require_raises(RuntimeError, paired_jacobians_for_prompt, **{**kwargs, "model": broken})
    assert all(not block._forward_hooks for block in broken.layers)
    bad = copy.deepcopy(model)
    bad.layers[1] = BadGradientBlock()
    for compress in (False, True):
        with patch(__name__ + ".ReplicatedSavedTensorHooks", tiny_hooks):
            require_raises(FloatingPointError, paired_jacobians_for_prompt,
                           **{**kwargs, "model": bad, "source_layers": [0], "target_layer": 1,
                              "compress_saved_tensors": compress})
        check_hooks_clean(bad)

    class FiniteOverflowBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden):
            return hidden.clone()

        @staticmethod
        def backward(ctx, grad):
            # Each raw value is finite; summing three valid source positions
            # overflows FP32. The paired call must reject the entire result.
            return torch.full_like(grad, torch.finfo(grad.dtype).max)

    class OverflowBlock(torch.nn.Module):
        def forward(self, hidden):
            return FiniteOverflowBackward.apply(hidden)

    overflow = copy.deepcopy(model).float()
    overflow.layers[1] = OverflowBlock()
    require_raises(FloatingPointError, paired_jacobians_for_prompt,
                   **{**kwargs, "model": overflow, "source_layers": [0], "target_layer": 1})
    check_hooks_clean(overflow)
    ordinary_backward = torch.autograd.grad
    backward_count = 0

    def failed_backward(*args, **options):
        nonlocal backward_count
        backward_count += 1
        if backward_count == 2:
            raise RuntimeError("injected backward failure after one complete chunk")
        return ordinary_backward(*args, **options)

    details = {}
    with (patch("torch.autograd.grad", side_effect=failed_backward),
          patch(__name__ + ".ReplicatedSavedTensorHooks", tiny_hooks)):
        require_raises(RuntimeError, paired_jacobians_for_prompt,
                       **{**kwargs, "compress_saved_tensors": True, "diagnostics": details})
    assert backward_count == 2 and details["completed_row_chunks"] == 1
    check_hooks_clean(model)
    ordinary_forward = model.forward

    def failed_forward(ids):
        ordinary_forward(ids)
        raise RuntimeError("injected forward failure")

    with (patch.object(model, "forward", side_effect=failed_forward),
          patch(__name__ + ".ReplicatedSavedTensorHooks", tiny_hooks)):
        require_raises(RuntimeError, paired_jacobians_for_prompt,
                       **{**kwargs, "compress_saved_tensors": True})
    check_hooks_clean(model)
    assert all(not hooks.active for hooks in compressors)
    return {
        "status": "passed",
        "scope": "CPU causal fixtures only; no real model, GPU or CUDA graph validation",
        "torch_version": torch.__version__,
        "case_count": len(cases), "cases": cases,
        "max_absolute_explicit_float64_oracle_error": max(oracle_errors),
        "explicit_oracle_comparisons": len(oracle_errors),
        "original_sampled_bitwise_parity": True,
        "native_batch_and_shared_backward_count": True,
        "partial_and_oversized_direction_batches": True,
        "compressed_payloads_tested": compressed_payloads,
        "invalid_cases_rejected": len(invalid) + 3,
        "negative_layer_indices": "passed",
        "nonfinite_and_missing_hook_failures": "passed",
        "fp32_source_reduction_overflow_rejected": True,
        "partial_result_backward_failure_cleanup": "passed",
        "forward_failure_hook_and_storage_cleanup": "passed",
        "source_sha256": {
            str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), ROUND / "fit_estimators.py",
                         ROUND / "design_audit/verify_estimators.py",
                         ROUND / "optimization/optimized_fitting.py",
                         ROUND / "optimization/cuda_graph_candidate.py",
                         ROUND / "optimization/saved_tensor_candidate.py",
                         REPO / "jlens/fitting.py", REPO / "jlens/hooks.py", REPO / "tests/tiny.py")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.self_test:
        parser.error("use --self-test for CPU-only verification; fitting is driven by the runner")
    rendered = json.dumps(_self_test(), indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
