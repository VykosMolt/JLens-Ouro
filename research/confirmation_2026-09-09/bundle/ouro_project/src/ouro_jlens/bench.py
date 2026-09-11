"""Measure retained-graph memory and backward time for a full JLens fit.

The command is a preflight, not a best-effort logger: a requested dimension
batch is either measured or recorded as failed, and the process exits nonzero
when none of the requested configurations can run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from jlens.hooks import ActivationRecorder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ouro_jlens.recurrent import load_ouro  # noqa: E402

TEXT = " ".join(
    [
        "The committee reviewed the proposal and returned detailed comments on every section."
    ]
    * 20
)


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _cuda_memory_allocated() -> float:
    return torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def _cuda_peak_memory() -> float:
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def _empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def bench(m: Any, dim_batch: int, seq_len: int, n_passes: int = 4) -> dict[str, Any]:
    """Run one benchmark configuration and return measured values.

    ``torch.OutOfMemoryError`` (and the common CUDA ``RuntimeError`` spelling)
    is intentionally allowed to reach the caller so the configuration runner
    can mark just this configuration failed and try the next one.
    """

    if isinstance(dim_batch, bool) or not isinstance(dim_batch, int) or dim_batch <= 0:
        raise ValueError(f"dim_batch must be > 0, got {dim_batch!r}")
    if isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len <= 1:
        raise ValueError(f"seq_len must be > 1, got {seq_len!r}")
    if isinstance(n_passes, bool) or not isinstance(n_passes, int) or n_passes <= 0:
        raise ValueError(f"n_passes must be > 0, got {n_passes!r}")

    ids = m.encode(TEXT, max_length=seq_len)
    _empty_cuda_cache()
    _reset_peak_memory()
    base = _cuda_memory_allocated()
    sources = list(range(m.n_layers - 1))
    t0 = time.perf_counter()
    with ActivationRecorder(
        m.layers, at=range(m.n_layers), start_graph_at=0
    ) as rec, torch.enable_grad():
        m.forward(ids.expand(dim_batch, -1))
        _cuda_sync()
        t_fwd = time.perf_counter() - t0
        target = rec.activations[m.n_layers - 1]
        srcs = [rec.activations[i] for i in sources]
        cot = torch.zeros_like(target)
        cot[:, 16:-1, 0] = 1.0
        times: list[float] = []
        for _ in range(n_passes):
            t1 = time.perf_counter()
            grads = torch.autograd.grad(target, srcs, cot, retain_graph=True)
            rows = [g[:, 16:-1].float().mean(1).cpu() for g in grads]
            _cuda_sync()
            times.append(time.perf_counter() - t1)
            del grads, rows
    peak = _cuda_peak_memory()
    # Do not use a warm-up pass as the advertised throughput when one was
    # requested; with n_passes=1 the only available measurement is still
    # useful and is reported honestly.
    warmup = times[0] if times else 0.0
    measured = times[1:] if len(times) > 1 else times
    per_pass = sum(measured) / max(1, len(measured))
    per_prompt = per_pass * (m.d_model / dim_batch)
    result = {
        "dim_batch": dim_batch,
        "seq_len": int(ids.shape[1]),
        "forward_seconds": t_fwd,
        "backward_seconds": per_pass,
        "warmup_backward_seconds": warmup,
        "estimated_prompt_minutes": per_prompt / 60,
        "peak_gb": peak,
        "weights_gb": base,
        "pass": True,
    }
    print(
        f"dim_batch={dim_batch} T={ids.shape[1]}: fwd {t_fwd:.2f}s, "
        f"backward/pass {per_pass:.2f}s (first {warmup:.2f}s), "
        f"est per-prompt {per_prompt / 60:.1f} min, peak {peak:.2f} GB "
        f"(weights {base:.2f} GB)",
        flush=True,
    )
    return result


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def run_configs(m: Any, seq_len: int, dim_batches: Iterable[int]) -> list[dict[str, Any]]:
    """Run each requested configuration, preserving failures as evidence."""

    results: list[dict[str, Any]] = []
    for dim_batch in dim_batches:
        try:
            results.append(bench(m, dim_batch, seq_len))
        except Exception as exc:
            # A preflight configuration can fail for reasons other than
            # allocation (for example an unsupported backend). Preserve that
            # failure and continue so the caller can distinguish "some
            # configuration worked" from "all requested configurations
            # failed".
            _empty_cuda_cache()
            result = {
                "dim_batch": dim_batch,
                "seq_len": seq_len,
                "pass": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            results.append(result)
            label = "OOM" if _is_oom(exc) else "FAILED"
            print(f"dim_batch={dim_batch}: {label} ({exc})", flush=True)
    return results


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("seq_len", nargs="?", type=int, default=128)
    parser.add_argument("dim_batches", nargs="*", type=int, default=[1, 2, 4])
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.seq_len <= 1:
        parser.error("seq_len must be > 1")
    if not args.dim_batches or any(db <= 0 for db in args.dim_batches):
        parser.error("at least one positive dim_batch is required")
    model = load_ouro()
    results = run_configs(model, args.seq_len, args.dim_batches)
    if not any(result.get("pass") for result in results):
        print("all requested benchmark configurations failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
