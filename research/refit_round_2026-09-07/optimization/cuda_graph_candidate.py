"""Opt-in CUDA graph candidate for repeated VJPs of one retained primal.

Importing this module or running it without --run-cuda-smoke does not use CUDA.
No Ouro model is loaded here. A caller with an assigned GPU slot can use
benchmark_dense_prompt on its already loaded model before adopting anything.

The forward MUST run on the same non-default stream used for capture: autograd
nodes retain forward-stream affinity. Every replay uses the same primal and
source/target tensors but new values in a fixed cotangent buffer. The graph
captures backward and, optionally, the exact existing source-position mean.
There is no dimension sketch, new normalization, target change, or fit average.

Constraints and failure modes:
* All graph inputs, outputs and backward operations must be on one CUDA device.
* Forward, warmup, capture and replay use one explicit non-default stream.
* CPU synchronization, host copies, Python data-dependent backward logic, or
  unsupported kernels can prevent capture. Backward Python hooks execute at
  capture time, not once per replay; do not depend on their side effects.
* The graph private memory pool adds to the retained primal's memory. OOM is a
  failed candidate, not permission to reduce layers, tokens, width or precision.
* Captured output buffers are overwritten on the next replay. Reduce or copy
  them first. Never mutate/release saved primal tensors or update model weights.
* Every captured backward retains the autograd graph. close() synchronizes and
  resets the CUDA graph; releasing the caller's recorder/primal releases the
  remaining autograd graph. There is no extra backward for cleanup.
* Capture errors propagate; rebuild the primal for an eager fallback. This code
  does not change process-global stale-stream or capture-error policy flags.

Reference constraints were checked against installed torch 2.12.0.dev20260407
and the primary documentation:
https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graph-semantics
https://docs.nvidia.com/dl-cuda-graph/troubleshooting/capture-failures.html
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from numbers import Integral
from pathlib import Path

import torch

ROUND = Path(__file__).resolve().parents[1]
REPO = ROUND.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from jlens.fitting import _check_layer_indices, valid_position_mask  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from jlens.protocol import LensModel  # noqa: E402

TensorTuple = tuple[torch.Tensor, ...]


class CaptureUnavailableError(RuntimeError):
    """This retained graph could not be captured under the frozen conditions."""


class CapturedVJP:
    """Capture one backward, then replay with a mutable fixed-address cotangent.

    Construct inside ``with torch.cuda.stream(stream):`` after creating the
    entire primal on that stream. ``stream`` must be explicit and non-default.
    The supplied cotangent is cloned once; subsequently either write to the
    ``cotangent`` property and call ``replay()``, or call ``replay(new_values)``.

    ``reducer`` is optional GPU-only postprocessing executed within capture.
    It must return a nonempty tuple of CUDA tensors. Without it, replay returns
    raw per-source gradients in the source order. With ``dense_source_means``
    it returns one stacked [source, direction, hidden] tensor, avoiding a Python
    loop over all source layers on each replay. ``is_grads_batched`` supports a
    separately validated single-primal vmap path without forcing it on users.

    Warmup calls are discarded and reported as overhead; only the requested
    output rows enter the lens. No GPU work from capture is consumed before the
    first replay. Use a context manager or explicitly call close().
    """

    def __init__(
        self,
        target: torch.Tensor,
        sources: Sequence[torch.Tensor | torch.autograd.graph.GradientEdge],
        cotangent: torch.Tensor,
        *,
        stream: torch.cuda.Stream,
        warmup: int = 3,
        is_grads_batched: bool = False,
        reducer: Callable[[TensorTuple], TensorTuple] | None = None,
    ) -> None:
        if not target.is_cuda:
            raise ValueError("CapturedVJP requires a CUDA target")
        if not sources or not target.requires_grad:
            raise ValueError("target must require gradients and sources must be nonempty")
        if any(not isinstance(source, torch.autograd.graph.GradientEdge)
               and (not source.requires_grad or source.device != target.device)
               for source in sources):
            raise ValueError("sources must be gradient edges or tensors requiring gradients on the target CUDA device")
        if cotangent.device != target.device or cotangent.dtype != target.dtype:
            raise ValueError("cotangent must have the target's device and dtype")
        expected = cotangent.shape[1:] if is_grads_batched else cotangent.shape
        if tuple(expected) != tuple(target.shape):
            raise ValueError("cotangent shape does not match the requested VJP layout")
        if isinstance(warmup, bool) or not isinstance(warmup, Integral) or warmup < 1:
            raise ValueError("warmup must be a positive integer")
        if stream != torch.cuda.current_stream(target.device):
            raise ValueError("construct CapturedVJP inside its explicit CUDA stream context")
        if stream == torch.cuda.default_stream(target.device):
            raise ValueError("capture requires a non-default stream; build the primal there too")

        self.stream = stream
        self.device = target.device
        self._target: torch.Tensor | None = target
        self._sources = tuple(sources)
        self._cotangent: torch.Tensor | None = cotangent.detach().clone()
        self._reducer = reducer
        self._is_grads_batched = bool(is_grads_batched)
        self._outputs: TensorTuple = ()
        self._graph: torch.cuda.CUDAGraph | None = torch.cuda.CUDAGraph()
        self._closed = False
        self.warmup_calls = int(warmup)
        self.replay_calls = 0
        started = time.perf_counter()
        phase = "warmup"
        try:
            for _ in range(self.warmup_calls):
                warm_outputs = self._run()
                del warm_outputs
            stream.synchronize()
            self.warmup_seconds = time.perf_counter() - started
            phase = "capture"
            started = time.perf_counter()
            # The outer stream context restores state even if an installed
            # torch.cuda.graph.__exit__ raises before exiting its own context.
            with torch.cuda.stream(stream):
                with torch.cuda.graph(self._graph, stream=stream, capture_error_mode="global"):
                    self._outputs = self._run()
            self.capture_seconds = time.perf_counter() - started
            if not self._outputs or any(t.device != self.device for t in self._outputs):
                raise ValueError("reducer must return nonempty CUDA outputs on the target device")
        except Exception as error:
            with suppress(Exception):
                self._graph.reset()
            self._drop_references()
            raise CaptureUnavailableError(
                f"CUDA VJP {phase} failed: {type(error).__name__}: {error}. "
                "Rebuild the primal before trying an eager fallback."
            ) from error

    def _run(self) -> TensorTuple:
        grads = torch.autograd.grad(
            self._target,
            self._sources,
            grad_outputs=self._cotangent,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
            is_grads_batched=self._is_grads_batched,
        )
        outputs = grads if self._reducer is None else self._reducer(grads)
        if not isinstance(outputs, tuple) or not outputs:
            raise ValueError("reducer must return a nonempty tuple of tensors")
        return outputs

    @property
    def cotangent(self) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("CapturedVJP is closed")
        assert self._cotangent is not None
        return self._cotangent

    def replay(self, cotangent: torch.Tensor | None = None) -> TensorTuple:
        """Queue replay on the construction stream; outputs are borrowed buffers."""
        if self._closed:
            raise RuntimeError("CapturedVJP is closed")
        if torch.cuda.current_stream(self.device) != self.stream:
            raise ValueError("replay and output consumption must use the capture stream")
        if cotangent is not None:
            static = self.cotangent
            if cotangent.shape != static.shape or cotangent.dtype != static.dtype or cotangent.device != static.device:
                raise ValueError("replay cotangent shape, dtype and device must remain fixed")
            static.copy_(cotangent)
        assert self._graph is not None
        self._graph.replay()
        self.replay_calls += 1
        return self._outputs

    def _drop_references(self) -> None:
        self._outputs = ()
        self._sources = ()
        self._target = None
        self._cotangent = None
        self._reducer = None
        self._graph = None
        self._closed = True

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.stream.synchronize()
        finally:
            try:
                assert self._graph is not None
                self._graph.reset()
            finally:
                self._drop_references()

    def __enter__(self) -> CapturedVJP:
        if self._closed:
            raise RuntimeError("CapturedVJP is closed")
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class DenseCotangentWriter:
    """Write exact dense one-hot direction batches, clearing stale/unused lanes."""

    def __init__(
        self, cotangent: torch.Tensor, *, skip_first: int = 16,
        is_grads_batched: bool = False,
    ) -> None:
        if cotangent.requires_grad:
            raise ValueError("cotangent must not require gradients")
        if is_grads_batched:
            if cotangent.ndim != 4 or cotangent.shape[1] != 1:
                raise ValueError("single-primal batched VJPs require [directions, 1, token, hidden]")
            self.view = cotangent[:, 0]
        else:
            if cotangent.ndim != 3:
                raise ValueError("replicated-primal VJPs require [directions, token, hidden]")
            self.view = cotangent
        self.cotangent = cotangent
        self.dim_batch, seq_len, self.width = self.view.shape
        if self.dim_batch < 1 or self.width < 1:
            raise ValueError("cotangent directions and width must be nonzero")
        mask = valid_position_mask(seq_len, skip_first=skip_first)
        self.n_valid = int(mask.sum())
        self.valid_positions = mask.nonzero(as_tuple=True)[0].to(cotangent.device)
        self.lanes = torch.arange(self.dim_batch, device=cotangent.device)

    def write(self, dim_start: int) -> int:
        if isinstance(dim_start, bool) or not isinstance(dim_start, Integral) or not 0 <= dim_start < self.width:
            raise ValueError("dim_start must be an integer within the output width")
        n_dims = min(self.dim_batch, self.width - int(dim_start))
        lanes = self.lanes[:n_dims]
        self.cotangent.zero_()
        self.view[
            lanes[:, None], self.valid_positions[None, :], dim_start + lanes[:, None]
        ] = 1
        return n_dims


def dense_source_means(
    grads: TensorTuple, valid_positions: torch.Tensor, *, is_grads_batched: bool = False,
) -> TensorTuple:
    """Exact upstream fp32 source-token mean, stacked for one transfer per pass."""
    rows = []
    for grad in grads:
        if is_grads_batched:
            if grad.ndim != 4 or grad.shape[1] != 1:
                raise ValueError("expected one primal batch lane in batched source gradients")
            grad = grad[:, 0]
        rows.append(grad[:, valid_positions, :].float().mean(dim=1))
    return (torch.stack(rows),)


def _difference(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    difference = actual.double() - expected.double()
    denominator = expected.double().norm().item()
    return {
        "bitwise_equal": torch.equal(actual, expected),
        "finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
        "max_abs": difference.abs().max().item(),
        "relative_frobenius": difference.norm().item() / max(denominator, 1e-30),
    }


def benchmark_dense_prompt(
    model: LensModel,
    prompt: str,
    source_layers: Sequence[int],
    *,
    target_layer: int | None = None,
    dim_batch: int = 2,
    max_seq_len: int = 128,
    skip_first: int = 16,
    is_grads_batched: bool = False,
    measure_passes: int = 12,
) -> dict:
    """Opt-in bounded eager/capture parity and cost check; does not fit a lens.

    The baseline uses the same primal, cotangents and fp32 source mean. Both
    timings include changing direction batches and one stacked D2H transfer per
    pass. Capture/warmup costs are reported separately and included in the full
    width cost estimate. Strict bitwise parity is required; mismatches are
    reported without adapting numerical tolerances. No saved scientific metric
    is evaluated. The caller schedules exclusive local GPU access.
    """
    if isinstance(dim_batch, bool) or not isinstance(dim_batch, Integral) or dim_batch < 1:
        raise ValueError("dim_batch must be a positive integer")
    if isinstance(measure_passes, bool) or not isinstance(measure_passes, Integral) or measure_passes < 1:
        raise ValueError("measure_passes must be a positive integer")
    sources, target = _check_layer_indices(source_layers, target_layer, model.n_layers)
    if not sources:
        raise ValueError("at least one strict source layer is required")
    input_ids = model.encode(prompt, max_length=max_seq_len)
    if not input_ids.is_cuda:
        raise ValueError("benchmark_dense_prompt requires an already loaded CUDA model")
    device = input_ids.device
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    result = {
        "target_layer": target, "source_layers": sources, "dim_batch": dim_batch,
        "seq_len": int(input_ids.shape[1]), "skip_first": skip_first,
        "is_grads_batched": bool(is_grads_batched), "measure_passes": measure_passes,
        "d_model": model.d_model, "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device),
    }
    with (
        torch.cuda.stream(stream),
        torch.enable_grad(),
        ActivationRecorder(model.layers, at=[*sources, target], start_graph_at=min(sources)) as recorder,
    ):
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        model.forward(input_ids if is_grads_batched else input_ids.expand(dim_batch, -1))
        stream.synchronize()
        result["forward_seconds"] = time.perf_counter() - started
        target_activation = recorder.activations[target]
        source_activations = tuple(recorder.activations[layer] for layer in sources)
        shape = (dim_batch, *target_activation.shape) if is_grads_batched else target_activation.shape
        cotangent = torch.zeros(shape, dtype=target_activation.dtype, device=device)
        writer = DenseCotangentWriter(cotangent, skip_first=skip_first, is_grads_batched=is_grads_batched)
        result["n_valid_positions"] = writer.n_valid
        writer.write(0)

        def reduce(grads: TensorTuple) -> TensorTuple:
            return dense_source_means(grads, writer.valid_positions, is_grads_batched=is_grads_batched)

        def eager_rows() -> torch.Tensor:
            grads = torch.autograd.grad(
                target_activation, source_activations, grad_outputs=cotangent,
                retain_graph=True, is_grads_batched=is_grads_batched,
            )
            return reduce(grads)[0]

        # Warm eager kernels before baseline timing, and record duplicate-run
        # scatter before testing graph replay. It is never used to relax parity.
        first = eager_rows().cpu()
        duplicate = eager_rows().cpu()
        result["eager_duplicate"] = _difference(duplicate, first)
        del first, duplicate
        starts = list(range(0, model.d_model, dim_batch))
        stream.synchronize()
        started = time.perf_counter()
        for index in range(measure_passes):
            writer.write(starts[index % len(starts)])
            measured = eager_rows().cpu()
            del measured
        stream.synchronize()
        result["eager_seconds_per_pass"] = (time.perf_counter() - started) / measure_passes
        result["eager_peak_bytes"] = torch.cuda.max_memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)

        with CapturedVJP(
            target_activation, source_activations, cotangent, stream=stream,
            is_grads_batched=is_grads_batched, reducer=reduce,
        ) as captured:
            result["capture_seconds"] = captured.capture_seconds
            result["capture_warmup_seconds"] = captured.warmup_seconds
            result["capture_warmup_calls"] = captured.warmup_calls
            parity = []
            for start in (0, model.d_model // 2, model.d_model - 1, 0):
                n_dims = writer.write(start)
                expected = eager_rows().cpu()
                actual = captured.replay(cotangent)[0].cpu()
                parity.append({"dim_start": start, "n_dims": n_dims, **_difference(actual, expected)})
                del expected, actual
            cotangent.zero_()
            expected = eager_rows().cpu()
            actual = captured.replay(cotangent)[0].cpu()
            zero = _difference(actual, expected)
            zero["captured_all_zero"] = bool(torch.count_nonzero(actual).item() == 0)
            result["zero_after_nonzero"] = zero
            del expected, actual
            result["replay_parity"] = parity
            if not all(item["finite"] and item["bitwise_equal"] for item in parity) or not zero["captured_all_zero"]:
                result["status"] = "numerical_mismatch"
                return result

            static_writer = DenseCotangentWriter(captured.cotangent, skip_first=skip_first, is_grads_batched=is_grads_batched)
            stream.synchronize()
            started = time.perf_counter()
            for index in range(measure_passes):
                static_writer.write(starts[index % len(starts)])
                measured = captured.replay()[0].cpu()
                del measured
            stream.synchronize()
            result["captured_seconds_per_pass"] = (time.perf_counter() - started) / measure_passes
            result["capture_peak_bytes"] = torch.cuda.max_memory_allocated(device)
            result["replay_calls_in_check"] = captured.replay_calls
            n_passes = math.ceil(model.d_model / dim_batch)
            result["estimated_eager_prompt_seconds"] = result["forward_seconds"] + n_passes * result["eager_seconds_per_pass"]
            result["estimated_captured_prompt_seconds"] = (
                result["forward_seconds"] + result["capture_seconds"] + result["capture_warmup_seconds"]
                + n_passes * result["captured_seconds_per_pass"]
            )
            result["estimated_prompt_speedup"] = result["estimated_eager_prompt_seconds"] / result["estimated_captured_prompt_seconds"]
            result["status"] = "passed"
    result["cleanup_complete"] = True
    return result


def verify_helpers_cpu() -> dict:
    """Exercise row updates and exact reductions without initializing CUDA."""
    cuda_before = torch.cuda.is_initialized()
    for is_batched in (False, True):
        shape = (2, 1, 5, 3) if is_batched else (2, 5, 3)
        cotangent = torch.full(shape, 7.0, dtype=torch.float64)
        writer = DenseCotangentWriter(cotangent, skip_first=1, is_grads_batched=is_batched)
        for start in (0, 2, 1, 0):
            n_dims = writer.write(start)
            expected = torch.zeros(2, 5, 3, dtype=torch.float64)
            for lane in range(n_dims):
                expected[lane, 1:4, start + lane] = 1
            assert torch.equal(writer.view, expected)
        raw = torch.arange(30, dtype=torch.float64).reshape(2, 5, 3)
        grads = (raw.unsqueeze(1), raw.unsqueeze(1) * -2) if is_batched else (raw, raw * -2)
        means = dense_source_means(grads, writer.valid_positions, is_grads_batched=is_batched)[0]
        expected = torch.stack((raw[:, 1:4].float().mean(1), (raw * -2)[:, 1:4].float().mean(1)))
        assert torch.equal(means, expected)
        assert means.shape == (2, 2, 3) and means.dtype == torch.float32
    try:
        target = torch.zeros(2, 5, 3, requires_grad=True)
        CapturedVJP(target, [target], torch.zeros_like(target), stream=None)
    except ValueError as error:
        assert "CUDA target" in str(error)
    else:
        raise AssertionError("CPU graph capture must be rejected")
    assert torch.cuda.is_initialized() == cuda_before
    return {"status": "passed", "cuda_initialized": cuda_before, "layouts": ["replicated", "single_primal_batched"], "partial_lane_and_stale_cotangent_checks": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-check", action="store_true")
    parser.add_argument("--run-cuda-smoke", action="store_true", help="Explicitly use CUDA for tiny-model checks; schedule GPU access first")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "status": "not_run", "note": "CUDA execution requires --run-cuda-smoke or an explicit benchmark_dense_prompt call"}
    if args.cpu_check:
        result["cpu_helpers"] = verify_helpers_cpu()
    if args.run_cuda_smoke:
        sys.path.insert(0, str(ROUND / "design_audit"))
        from verify_estimators import causal_model

        torch.set_num_threads(1)
        model = causal_model().cuda()
        result["cuda_smoke"] = []
        for is_batched in (False, True):
            try:
                measured = benchmark_dense_prompt(
                    model, "abcd", [0, 1, 2], target_layer=3, dim_batch=2,
                    max_seq_len=5, skip_first=1, is_grads_batched=is_batched, measure_passes=6,
                )
            except Exception as error:
                measured = {"status": "failed", "is_grads_batched": is_batched, "error": f"{type(error).__name__}: {error}"}
            result["cuda_smoke"].append(measured)
        result["status"] = "passed" if all(item["status"] == "passed" for item in result["cuda_smoke"]) else "failed"
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
