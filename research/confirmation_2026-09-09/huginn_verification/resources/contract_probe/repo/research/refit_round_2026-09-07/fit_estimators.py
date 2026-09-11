"""Bounded per-prompt estimators for the independent Ouro refit round.

The dense arm delegates to the released estimator. The sampled arm computes
two reductions of the same single-position VJP. Position selection and
calibration-set averaging belong to the caller; this module does neither.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from importlib import import_module
from numbers import Integral

import torch

from jlens.fitting import (
    SKIP_FIRST_N_POSITIONS,
    _check_layer_indices,
    jacobian_for_prompt,
    valid_position_mask,
)
from jlens.hooks import ActivationRecorder
from jlens.protocol import LensModel

logger = logging.getLogger(__name__)


def _integer(name: str, value: object, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def jacobians_for_prompt(
    model: LensModel,
    prompt: str,
    source_layers: Sequence[int],
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    mode: str = "dense",
    q: int | None = None,
    dense_engine: str = "stock",
    compress_saved_tensors: bool = False,
    diagnostics: dict | None = None,
) -> tuple[dict[str, dict[int, torch.Tensor]], int, int]:
    """Return ``(maps, seq_len, n_valid)`` with CPU float32 d×d matrices.

    ``mode='dense'`` returns ``maps['dense']``. ``dense_engine='stock'`` uses
    unmodified upstream :func:`jlens.fitting.jacobian_for_prompt`. Explicit
    ``'eager'`` or ``'cuda_graph'`` selects the research optimized fitter and
    permits lossless saved-tensor compression. Runtime diagnostics can be
    collected in the supplied dictionary. ``q`` must then be None. Batch size
    and engine belong in fit provenance; BF16 outputs can depend on batch size.

    ``mode='sampled'`` requires an explicit token position ``q`` drawn by the
    caller uniformly from V = {skip_first, ..., seq_len - 2}. It returns
    ``maps['sampled_sum'][l] = sum_{p in V} G_l(q,p)`` and
    ``maps['diagonal'][l] = G_l(q,q)`` from one shared VJP per output dimension.
    Thus averaging sampled_sum over uniform q recovers the dense estimator,
    ``(1/|V|) sum_{p,q in V} G_l(q,p)``; averaging diagonal recovers the mean
    same-position Jacobian. There is no additional source-position divisor.

    Layer indices follow upstream normalization, including negative indices;
    every resolved source must precede the target. q is an absolute position
    in the actual encoded, truncated sequence, including any BOS. No random
    draw, identity padding, prompt skipping, or outlier removal occurs here.
    A deterministic model in evaluation mode is required by LensModel.

    Memory follows upstream: one retained forward graph and one batch of
    gradients, with all completed matrices held on CPU. The sampled arm
    returns two CPU matrices per source. Nonfinite gradients raise immediately
    in that arm; dense output matrices are checked after upstream returns.
    """
    if mode not in ("dense", "sampled"):
        raise ValueError(f"mode must be 'dense' or 'sampled', got {mode!r}")
    if dense_engine not in ("stock", "eager", "cuda_graph"):
        raise ValueError("dense_engine must be 'stock', 'eager', or 'cuda_graph'")
    if type(compress_saved_tensors) is not bool:
        raise ValueError("compress_saved_tensors must be a boolean")
    if mode == "sampled" and (dense_engine != "stock" or compress_saved_tensors):
        raise ValueError("optimized dense options do not apply to sampled estimators")
    if compress_saved_tensors and dense_engine == "stock":
        raise ValueError("saved-tensor compression requires an explicit optimized dense engine")
    dim_batch = _integer("dim_batch", dim_batch, minimum=1)
    max_seq_len = _integer("max_seq_len", max_seq_len, minimum=1)
    skip_first = _integer("skip_first", skip_first, minimum=0)
    if source_layers is None:
        raise ValueError("source_layers must be an explicit nonempty sequence")
    sources = [_integer("source layer", layer) for layer in source_layers]
    if target_layer is not None:
        target_layer = _integer("target_layer", target_layer)
    sources, target = _check_layer_indices(sources, target_layer, model.n_layers)

    if mode == "dense":
        if q is not None:
            raise ValueError("q must be None for mode='dense'")
        compute = jacobian_for_prompt
        execution_options = {}
        if dense_engine != "stock":
            prefix = f"{__package__}." if __package__ else ""
            compute = import_module(f"{prefix}optimization.optimized_fitting").jacobian_for_prompt
            storage = None
            if compress_saved_tensors:
                hooks = import_module(f"{prefix}optimization.saved_tensor_candidate").ReplicatedSavedTensorHooks
                length = int(model.encode(prompt, max_length=max_seq_len).shape[1])
                storage = hooks(model, batch_size=dim_batch, seq_len=length)
            execution_options = dict(engine=dense_engine, storage_context=storage,
                                     diagnostics=diagnostics)
        dense, seq_len, n_valid = compute(
            model,
            prompt,
            sources,
            target_layer=target,
            dim_batch=dim_batch,
            max_seq_len=max_seq_len,
            skip_first=skip_first,
            **execution_options,
        )
        for layer, matrix in dense.items():
            if not torch.isfinite(matrix).all().item():
                raise FloatingPointError(f"nonfinite dense matrix at source {layer}")
        return {"dense": dense}, seq_len, n_valid

    q = _integer("q", q, minimum=0)
    input_ids = model.encode(prompt, max_length=max_seq_len)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("model.encode must return input_ids with shape [1, seq_len]")
    seq_len = int(input_ids.shape[1])
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    if q >= seq_len or not position_mask[q].item():
        raise ValueError(
            f"q={q} is not valid for encoded seq_len={seq_len}, "
            f"skip_first={skip_first}; valid positions end at {seq_len - 2}"
        )
    n_valid = int(position_mask.sum())
    d_model = model.d_model
    maps = {
        reduction: {
            layer: torch.empty(d_model, d_model, dtype=torch.float32, device="cpu")
            for layer in sources
        }
        for reduction in ("sampled_sum", "diagonal")
    }
    n_passes = math.ceil(d_model / dim_batch)

    with (
        ActivationRecorder(
            model.layers,
            at=[*sources, target],
            start_graph_at=min(sources),
        ) as recorder,
        torch.enable_grad(),
    ):
        model.forward(input_ids.expand(dim_batch, -1))
        missing = {*sources, target} - recorder.activations.keys()
        if missing:
            raise RuntimeError(f"requested layer hooks did not fire: {sorted(missing)}")
        target_activation = recorder.activations[target]
        source_activations = [recorder.activations[layer] for layer in sources]
        for layer in [*sources, target]:
            shape = tuple(recorder.activations[layer].shape)
            if shape != (dim_batch, seq_len, d_model):
                raise ValueError(f"unexpected activation shape at layer {layer}: {shape}")

        valid_positions = position_mask.nonzero(as_tuple=True)[0]
        positions_by_device = {
            device: valid_positions.to(device)
            for device in {activation.device for activation in source_activations}
        }
        batch_indices = torch.arange(dim_batch, device=target_activation.device)
        cotangent = torch.zeros_like(target_activation)
        for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
            n_dims = min(dim_batch, d_model - dim_start)
            lanes = batch_indices[:n_dims]
            cotangent.zero_()
            cotangent[lanes, q, dim_start + lanes] = 1.0
            grads = torch.autograd.grad(
                outputs=target_activation,
                inputs=source_activations,
                grad_outputs=cotangent,
                retain_graph=pass_idx < n_passes - 1,
            )
            for layer, grad in zip(sources, grads, strict=True):
                if not torch.isfinite(grad).all().item():
                    raise FloatingPointError(
                        f"nonfinite sampled gradient at source {layer}, "
                        f"target {target}, q={q}, dim_start={dim_start}"
                    )
                positions = positions_by_device[grad.device]
                sum_rows = grad[:n_dims, positions, :].float().sum(dim=1)
                diagonal_rows = grad[:n_dims, q, :].float()
                if not torch.isfinite(sum_rows).all().item():
                    raise FloatingPointError(
                        f"sampled source reduction overflow at source {layer}, "
                        f"target {target}, q={q}, dim_start={dim_start}"
                    )
                maps["sampled_sum"][layer][dim_start : dim_start + n_dims].copy_(
                    sum_rows.cpu()
                )
                maps["diagonal"][layer][dim_start : dim_start + n_dims].copy_(
                    diagonal_rows.cpu()
                )
            del grads, grad, sum_rows, diagonal_rows
            if pass_idx % 100 == 0 or pass_idx == n_passes - 1:
                logger.debug(
                    "sampled target=%d q=%d pass=%d/%d dimensions=%d:%d",
                    target,
                    q,
                    pass_idx + 1,
                    n_passes,
                    dim_start,
                    dim_start + n_dims,
                )

    return maps, seq_len, n_valid
