"""Same-shape dense fitter with optional CUDA replay and exact storage hooks.

The original replicated prompt, derivative directions, BF16 operator shapes,
source-position fp32 means and CPU fp32 output format are preserved. No model
forward is compiled. ``engine='cuda_graph'`` captures the retained backward
plus the existing per-source means; the resulting rows use one host transfer
per direction batch. Its primal must be built on the capture stream.

This research implementation is deliberately separate from the released fitter.
Use the benchmark evidence for the exact model/device/batch before adoption.
Increasing dim_batch can change BF16 numerical results even without this code.
The caller may supply a validated saved-tensor context via ``storage_context``;
it applies to the forward only and must preserve all unpacked values/layouts.
No fallback, precision change, skipped row, or altered fit count is automatic.
"""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from numbers import Integral
from typing import ContextManager

import torch

from jlens.fitting import _check_layer_indices, valid_position_mask
from jlens.hooks import ActivationRecorder

try:
    from .cuda_graph_candidate import CapturedVJP, DenseCotangentWriter, dense_source_means
except ImportError:
    from cuda_graph_candidate import CapturedVJP, DenseCotangentWriter, dense_source_means


class SourceEdgeRecorder(ActivationRecorder):
    """Keep gradient destinations and metadata while releasing source values.

    PyTorch's public get_gradient_edge is equivalent to the Tensor input of
    autograd.grad. Saved tensors still belong to the ordinary forward graph;
    optional storage hooks control their storage independently. In particular,
    this does not detach the recurrent state or change its derivatives.
    """

    def __init__(self, blocks, *, sources, target):
        super().__init__(blocks, at=[*sources, target], start_graph_at=min(sources))
        self.target = target
        self.edges = {}
        self.metadata = {}

    def _make_hook(self, index):
        def hook(module, args, output):
            tensor = output if torch.is_tensor(output) else output[0]
            if index == self._start_graph_at:
                tensor.requires_grad_(True)
            self.metadata[index] = (tuple(tensor.shape), tensor.device, tensor.dtype)
            if index == self.target:
                self.activations[index] = tensor
            else:
                self.edges[index] = torch.autograd.graph.get_gradient_edge(tensor)
        return hook


def jacobian_for_prompt(
    model,
    prompt: str,
    source_layers,
    *,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = 16,
    engine: str = "cuda_graph",
    storage_context: ContextManager | None = None,
    source_edges: bool = True,
    capture_warmup: int = 1,
    diagnostics: dict | None = None,
):
    """Return stock-compatible ``(CPU fp32 matrices, sequence length, |V|)``.

    ``engine='eager'`` uses ordinary autograd and batches only the row transfer.
    ``engine='cuda_graph'`` requires a single CUDA device. A final partial batch
    uses eager reduction with exactly the upstream slice shape. ``diagnostics``
    is populated with measured phase times; the function never writes files.
    All output entries are filled before return; failures raise and clean up.
    """
    if engine not in ("eager", "cuda_graph"):
        raise ValueError("engine must be 'eager' or 'cuda_graph'")
    for name, value, minimum in (
        ("dim_batch", dim_batch, 1),
        ("max_seq_len", max_seq_len, 1),
        ("skip_first", skip_first, 0),
        ("capture_warmup", capture_warmup, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    sources, target = _check_layer_indices(source_layers, target_layer, model.n_layers)
    if not sources:
        raise ValueError("at least one source layer is required")
    input_ids = model.encode(prompt, max_length=max_seq_len)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("model.encode must return [1, tokens]")
    seq_len = int(input_ids.shape[1])
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    n_valid = int(position_mask.sum())
    d_model = model.d_model
    if engine == "cuda_graph" and not input_ids.is_cuda:
        raise ValueError("CUDA graph engine requires a CUDA model")
    device = input_ids.device
    is_cuda = device.type == "cuda"
    stream = torch.cuda.Stream(device=device) if is_cuda else None
    if stream is not None:
        stream.wait_stream(torch.cuda.current_stream(device))
    info = diagnostics if diagnostics is not None else {}
    info.update(engine=engine, source_edges=source_edges, dim_batch=int(dim_batch), sequence_length=seq_len,
                n_valid=n_valid, source_count=len(sources), target=target,
                n_passes=math.ceil(d_model / dim_batch))
    started = time.perf_counter()
    # A contiguous bank permits a single CPU copy per chunk. Every element is
    # assigned exactly once, so a 3.2-GB zero-fill is unnecessary for Ouro.
    bank = torch.empty((len(sources), d_model, d_model), dtype=torch.float32)
    info["allocate_cpu_seconds"] = time.perf_counter() - started
    captured = None
    try:
        with (
            torch.cuda.stream(stream) if stream is not None else nullcontext(),
            torch.enable_grad(),
            (SourceEdgeRecorder(model.layers, sources=sources, target=target)
             if source_edges else ActivationRecorder(model.layers, at=[*sources, target],
                                                      start_graph_at=min(sources))) as recorder,
        ):
            started = time.perf_counter()
            with storage_context if storage_context is not None else nullcontext():
                model.forward(input_ids.expand(dim_batch, -1))
            if stream is not None:
                stream.synchronize()
            info["forward_seconds"] = time.perf_counter() - started
            metadata = (recorder.metadata if source_edges else
                        {layer: (tuple(tensor.shape), tensor.device, tensor.dtype)
                         for layer, tensor in recorder.activations.items()})
            missing = {*sources, target} - metadata.keys()
            if missing:
                raise RuntimeError(f"layer hooks did not fire: {sorted(missing)}")
            target_activation = recorder.activations[target]
            activations = tuple((recorder.edges if source_edges else recorder.activations)[layer]
                                for layer in sources)
            if any(meta[1] != device for meta in metadata.values()):
                raise ValueError("optimized fitter requires all activations on one device")
            if any(meta[0] != (dim_batch, seq_len, d_model) for meta in metadata.values()):
                raise ValueError("source/target activation shape is inconsistent")
            cotangent = torch.zeros_like(target_activation)
            writer = DenseCotangentWriter(cotangent, skip_first=skip_first)

            def reduce(grads):
                return dense_source_means(grads, writer.valid_positions)

            def eager_rows(n_dims):
                grads = torch.autograd.grad(target_activation, activations, cotangent,
                                            retain_graph=True, create_graph=False)
                return reduce(tuple(grad[:n_dims] for grad in grads))[0]

            info["warmup_seconds"] = 0.0
            info["capture_seconds"] = 0.0
            info["warmup_calls"] = 0
            if engine == "cuda_graph" and dim_batch <= d_model:
                writer.write(0)
                captured = CapturedVJP(target_activation, activations, cotangent,
                                      stream=stream, reducer=reduce, warmup=capture_warmup)
                info["warmup_seconds"] = captured.warmup_seconds
                info["capture_seconds"] = captured.capture_seconds
                info["warmup_calls"] = captured.warmup_calls
                static_writer = DenseCotangentWriter(captured.cotangent, skip_first=skip_first)
            if stream is not None:
                stream.synchronize()
            started = time.perf_counter()
            for dim_start in range(0, d_model, dim_batch):
                n_dims = min(dim_batch, d_model - dim_start)
                if captured is not None and n_dims == dim_batch:
                    static_writer.write(dim_start)
                    rows = captured.replay()[0]
                else:
                    writer.write(dim_start)
                    rows = eager_rows(n_dims)
                bank[:, dim_start:dim_start + n_dims, :].copy_(rows.cpu())
                del rows
            if stream is not None:
                stream.synchronize()
            info["all_rows_seconds"] = time.perf_counter() - started
    finally:
        if captured is not None:
            captured.close()
        if stream is not None:
            # All returned tensors are on CPU; completion here also prevents
            # freed graph buffers from being reused before side-stream work.
            stream.synchronize()
    return {layer: bank[index] for index, layer in enumerate(sources)}, seq_len, n_valid
