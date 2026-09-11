"""Opt-in compiled-autograd candidate for unchanged retained-graph VJPs.

This uses the installed torch 2.12 private compiled_autograd._enable API. It
compiles the existing autograd.grad backward, not the model forward or hooks.
Import and default CLI execution perform no CUDA work. --cpu-check is a tiny
CPU feasibility proof; real GPU use is an explicit caller action.

Keep one CompiledVJP context around all prompts to permit compiler-cache reuse.
Different shapes, source/target topology, dtype or backend may recompile. The
100-prompt projection is conditional on reuse at the measured shape; it is not
an observed 100-prompt run. The first-call timer includes compilation or cache
retrieval and execution. No full model rewrite or silent eager fallback occurs.

donated_buffer=False preserves saved tensors for retain_graph=True. Static BF16
weights remain BF16: backward_pass_autocast='off' adds no autocast context.
emulate_precision_casts=True preserves eager intermediate low-precision casts
that Inductor otherwise deliberately elides when fusing. This does not guarantee
bitwise agreement for all fused kernels; the real-data benchmark defaults to a
strict bitwise acceptance criterion. CUDA graphs are disabled in this candidate
so fusion can be measured independently of the separate capture candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import torch
import torch._dynamo.config as dynamo_config
import torch._functorch.config as functorch_config
import torch._inductor.config as inductor_config
from torch._dynamo import compiled_autograd

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
REPO = ROUND.parents[1]
for directory in (REPO, HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from cuda_graph_candidate import DenseCotangentWriter, dense_source_means  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402


class CompiledVJP:
    """Scoped Inductor compiler around ordinary retained autograd.grad calls.

    Example: ``with CompiledVJP() as compiled:`` followed by repeated
    ``compiled.grad(target, sources, cotangent)``. Build and dispose of each
    prompt's normal recorder/primal separately inside that context. Returned
    gradients retain their original source shapes and order. All global config
    changes are scoped and restored when the context exits, including failures.
    """

    def __init__(self, *, backend: str = "inductor") -> None:
        if backend not in ("inductor", "eager"):
            raise ValueError("backend must be 'inductor' or 'eager'")
        self.backend = backend
        self.graphs: list[dict] = []
        self._stack: ExitStack | None = None

    def _compiler(self, graph):
        self.graphs.append({"nodes": len(list(graph.graph.nodes))})
        if self.backend == "eager":
            # This tests compiled-autograd capture alone; it is not an
            # Inductor speed result and is labeled separately by the caller.
            return graph.forward
        return torch.compile(graph, backend="inductor", fullgraph=True, dynamic=False)

    def __enter__(self) -> CompiledVJP:
        if self._stack is not None:
            raise RuntimeError("CompiledVJP context is already active")
        if not hasattr(compiled_autograd, "_enable"):
            raise RuntimeError("this torch build lacks the verified compiled_autograd._enable API")
        stack = ExitStack()
        try:
            stack.enter_context(functorch_config.patch(
                donated_buffer=False, backward_pass_autocast="off",
            ))
            stack.enter_context(inductor_config.patch({
                "compile_threads": 1,
                "triton.cudagraphs": False,
                "emulate_precision_casts": True,
            }))
            stack.enter_context(dynamo_config.patch(suppress_errors=False))
            stack.enter_context(compiled_autograd._enable(self._compiler, dynamic=False))
        except Exception:
            stack.close()
            raise
        self._stack = stack
        return self

    def grad(self, target, sources, cotangent, *, is_grads_batched: bool = False):
        if self._stack is None:
            raise RuntimeError("use grad inside the CompiledVJP context")
        return torch.autograd.grad(
            target, tuple(sources), grad_outputs=cotangent,
            retain_graph=True, create_graph=False, allow_unused=False,
            is_grads_batched=is_grads_batched,
        )

    def __exit__(self, *exc) -> None:
        stack, self._stack = self._stack, None
        assert stack is not None
        stack.close()


def _sync(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        torch.cuda.current_stream(tensor.device).synchronize()


def _compare(actual, expected, *, atol: float, rtol: float) -> dict:
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    difference = actual.double() - expected.double()
    mismatches = (actual != expected).nonzero()
    first_index = mismatches[0].tolist() if mismatches.numel() else None
    first_actual = actual[tuple(first_index)].item() if first_index is not None else None
    first_expected = expected[tuple(first_index)].item() if first_index is not None else None
    return {
        "finite": finite,
        "bitwise_equal": torch.equal(actual, expected),
        "within_tolerance": finite and bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
        "max_abs": difference.abs().max().item() if finite else None,
        "relative_frobenius": difference.norm().item() / max(expected.double().norm().item(), 1e-30) if finite else None,
        "mismatched_values": len(mismatches),
        "first_difference_index": first_index,
        "first_actual": first_actual if first_actual is None or math.isfinite(first_actual) else None,
        "first_expected": first_expected if first_expected is None or math.isfinite(first_expected) else None,
    }


def benchmark_retained(
    compiled: CompiledVJP,
    target: torch.Tensor,
    sources,
    *,
    skip_first: int = 16,
    dim_batch: int | None = None,
    is_grads_batched: bool = False,
    measure_passes: int = 6,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict:
    """Bounded eager/compiled comparison on an existing primal; no fitting.

    Both timings include exact fp32 valid-source means and a stacked CPU copy.
    The caller owns forward time, model lifetime and exclusive GPU scheduling.
    Default tolerances require bitwise-equivalent rows; only the CPU float64
    proof uses its predeclared 1e-10 allowance for compiler rounding. No finite
    difference or full FP32 Ouro proof is implied by the bounded row checks.
    """
    if measure_passes < 1:
        raise ValueError("measure_passes must be positive")
    sources = tuple(sources)
    if is_grads_batched:
        if dim_batch is None or dim_batch < 1:
            raise ValueError("batched gradients need an explicit positive dim_batch")
        cotangent = torch.zeros((dim_batch, *target.shape), dtype=target.dtype, device=target.device)
    else:
        if dim_batch is not None and dim_batch != target.shape[0]:
            raise ValueError("dim_batch must match the existing replicated primal")
        dim_batch = target.shape[0]
        cotangent = torch.zeros_like(target)
    writer = DenseCotangentWriter(cotangent, skip_first=skip_first, is_grads_batched=is_grads_batched)

    def rows(use_compiler: bool):
        if use_compiler:
            grads = compiled.grad(target, sources, cotangent, is_grads_batched=is_grads_batched)
        else:
            with compiled_autograd._disable():
                grads = torch.autograd.grad(
                    target, sources, grad_outputs=cotangent,
                    retain_graph=True, create_graph=False,
                    is_grads_batched=is_grads_batched,
                )
        return dense_source_means(grads, writer.valid_positions, is_grads_batched=is_grads_batched)[0].cpu()

    writer.write(0)
    rows(False)
    starts = list(range(0, writer.width, dim_batch))
    _sync(target)
    started = time.perf_counter()
    for index in range(measure_passes):
        writer.write(starts[index % len(starts)])
        measured = rows(False)
        del measured
    _sync(target)
    eager_seconds = (time.perf_counter() - started) / measure_passes
    graphs_before = len(compiled.graphs)
    writer.write(0)
    _sync(target)
    started = time.perf_counter()
    first = rows(True)
    _sync(target)
    first_seconds = time.perf_counter() - started
    initial = _compare(first, rows(False), atol=atol, rtol=rtol)
    del first
    parity = [{"dim_start": 0, **initial}]
    for start in (writer.width // 2, writer.width - 1, 0):
        n_dims = writer.write(start)
        parity.append({"dim_start": start, "n_dims": n_dims, **_compare(rows(True), rows(False), atol=atol, rtol=rtol)})
    cotangent.zero_()
    zero = rows(True)
    zero_parity = _compare(zero, rows(False), atol=atol, rtol=rtol)
    zero_parity["all_zero"] = bool(torch.count_nonzero(zero).item() == 0)
    del zero
    _sync(target)
    started = time.perf_counter()
    for index in range(measure_passes):
        writer.write(starts[index % len(starts)])
        measured = rows(True)
        del measured
    _sync(target)
    compiled_seconds = (time.perf_counter() - started) / measure_passes
    overhead = max(0.0, first_seconds - compiled_seconds)
    n_passes = math.ceil(writer.width / dim_batch)
    result = {
        "backend": compiled.backend,
        "target_shape": list(target.shape), "n_sources": len(sources),
        "dtype": str(target.dtype), "device": str(target.device),
        "is_grads_batched": is_grads_batched,
        "n_valid": writer.n_valid, "dim_batch": dim_batch,
        "parity_atol": atol, "parity_rtol": rtol,
        "graphs_added": compiled.graphs[graphs_before:],
        "total_graphs_captured": len(compiled.graphs),
        "eager_seconds_per_pass": eager_seconds,
        "first_compiled_call_seconds": first_seconds,
        "compiled_seconds_per_pass": compiled_seconds,
        "estimated_first_call_overhead_seconds": overhead,
        "parity": parity,
        "zero_after_nonzero": zero_parity,
        "estimated_100_prompt_backward_seconds_if_cache_reused": overhead + 100 * n_passes * compiled_seconds,
        "estimated_100_prompt_eager_backward_seconds": 100 * n_passes * eager_seconds,
        "projection_excludes": "forward, model load, checkpoint I/O; assumes unchanged graph/shape and reusable compiler cache",
    }
    result["status"] = "passed" if all(item["within_tolerance"] for item in parity) and zero_parity["all_zero"] else "numerical_mismatch"
    return result


def verify_cpu() -> dict:
    sys.path.insert(0, str(ROUND / "design_audit"))
    from verify_estimators import causal_model

    torch.set_num_threads(1)
    model = causal_model()  # float64 reference; 1e-10 fixed before this run
    cuda_before = torch.cuda.is_initialized()
    defaults = (functorch_config.donated_buffer, functorch_config.backward_pass_autocast, inductor_config.emulate_precision_casts)
    results = []
    with CompiledVJP() as compiled:
        for prompt in ("abcd", "pqrs"):
            with ActivationRecorder(model.layers, at=[0, 1, 2, 3], start_graph_at=0) as recorder:
                model.forward(model.encode(prompt, max_length=5).expand(2, -1))
            measured = benchmark_retained(
                compiled, recorder.activations[3], [recorder.activations[i] for i in range(3)],
                skip_first=1, measure_passes=6, atol=1e-10, rtol=1e-10,
            )
            measured["prompt"] = prompt
            assert measured["status"] == "passed"
            results.append(measured)
        assert len(compiled.graphs) == 1, "same-shaped fresh primals unexpectedly recompiled"
    assert defaults == (functorch_config.donated_buffer, functorch_config.backward_pass_autocast, inductor_config.emulate_precision_casts)
    assert torch.cuda.is_initialized() == cuda_before
    assert all(not block._forward_hooks for block in model.layers)
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters())
    return {"status": "passed", "scope": "CPU float64 causal model only", "cuda_initialized": cuda_before, "config_restored": True, "fresh_primals": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"status": "not_run", "torch": torch.__version__, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "note": "GPU use requires an explicit caller invocation; default execution does not fit or compile"}
    if args.cpu_check:
        result["cpu_check"] = verify_cpu()
        result["status"] = result["cpu_check"]["status"]
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
