"""CPU-only, explicit-Jacobian verification of the new sampled estimator.

Run from any directory with a Python environment containing this repo's test
dependencies. JSON evidence is printed to stdout; an optional --output path
stores the same evidence. This does not load or fit Ouro.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
REPO = ROUND.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROUND))

from fit_estimators import jacobians_for_prompt  # noqa: E402
from jlens.fitting import jacobian_for_prompt, valid_position_mask  # noqa: E402
from tests.tiny import TinyDecoder  # noqa: E402

# Fixed before execution. The API must return fp32 even for a float64 model;
# hence 1e-10 checks apply to explicit float64 algebra, and 1e-6 to API maps.
ALGEBRA_ATOL = 1e-10
MAP_ATOL = 2e-7
MAP_RTOL = 1e-6
FD_STEPS = (1e-2, 3e-3, 1e-3, 3e-4)
FD_RTOL = 0.02
FD_ATOL = 1e-5


class CausalBlock(nn.Module):
    """Nonlinear token mixing ensures off-diagonal token blocks are nonzero."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.local = nn.Linear(width, width, bias=False)
        self.context = nn.Linear(width, width, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        count = torch.arange(
            1, hidden.shape[1] + 1, device=hidden.device, dtype=hidden.dtype
        )[None, :, None]
        context = hidden.cumsum(dim=1) / count
        return hidden + 0.3 * torch.tanh(self.local(hidden) + self.context(context))


def causal_model() -> TinyDecoder:
    model = TinyDecoder(n_layers=4, d_model=3, seed=20260907)
    model.layers = nn.ModuleList([CausalBlock(3) for _ in range(4)])
    model.double().eval()
    model.requires_grad_(False)
    return model


def activation(model: TinyDecoder, prompt: str, layer: int) -> torch.Tensor:
    with torch.no_grad():
        hidden = model.embed_tokens(model.encode(prompt, max_length=5))
        for block in model.layers[: layer + 1]:
            hidden = block(hidden)
    return hidden[0].detach()


def suffix(model: TinyDecoder, hidden: torch.Tensor, source: int, target: int):
    hidden = hidden.unsqueeze(0)
    for block in model.layers[source + 1 : target + 1]:
        hidden = block(hidden)
    return hidden[0]


def explicit_blocks(model: TinyDecoder, prompt: str, source: int, target: int):
    hidden = activation(model, prompt, source).requires_grad_(True)
    # The source residual and suffix are formed directly, without recorder
    # hooks or either estimator's batching and position-reduction path.
    return torch.autograd.functional.jacobian(
        lambda h: suffix(model, h, source, target), hidden
    ).detach()


def close_error(actual, expected, *, atol=MAP_ATOL, rtol=MAP_RTOL) -> float:
    actual, expected = actual.double(), expected.double()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return float((actual - expected).abs().max())


def check_hooks_clean(model: TinyDecoder) -> None:
    assert all(not block._forward_hooks for block in model.layers)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not model.forward(model.encode("abcd", max_length=5)).last_hidden_state.requires_grad


def require_raises(exception, function, **kwargs) -> None:
    try:
        function(**kwargs)
    except exception:
        return
    raise AssertionError(f"expected {exception.__name__}: {kwargs}")


def numerical_checks(model: TinyDecoder) -> dict:
    prompt, seq_len, skip_first = "abcd", 5, 1
    valid = valid_position_mask(seq_len, skip_first=skip_first).nonzero().flatten()
    evidence = []
    worst_api_error, worst_algebra_error = 0.0, 0.0
    wrong_normalization_errors = []
    for target in (1, 2, 3):
        sources = list(range(target))
        kwargs = dict(
            model=model,
            prompt=prompt,
            source_layers=sources,
            target_layer=target,
            dim_batch=2,
            max_seq_len=seq_len,
            skip_first=skip_first,
        )
        wrapped, observed_len, n_valid = jacobians_for_prompt(**kwargs, mode="dense")
        upstream, upstream_len, upstream_valid = jacobian_for_prompt(**kwargs)
        assert (observed_len, n_valid) == (upstream_len, upstream_valid) == (5, 3)
        assert set(wrapped) == {"dense"} and set(wrapped["dense"]) == set(sources)
        for source in sources:
            assert torch.equal(wrapped["dense"][source], upstream[source])
        full = {source: explicit_blocks(model, prompt, source, target) for source in sources}
        sampled = []
        for q in valid.tolist():
            with patch("torch.autograd.grad", wraps=torch.autograd.grad) as counted:
                maps, sampled_len, sampled_n = jacobians_for_prompt(
                    **kwargs, mode="sampled", q=q
                )
            assert counted.call_count == math.ceil(model.d_model / kwargs["dim_batch"])
            assert (sampled_len, sampled_n) == (5, 3)
            assert set(maps) == {"sampled_sum", "diagonal"}
            sampled.append(maps)
            for reduction in maps.values():
                assert set(reduction) == set(sources)
                for matrix in reduction.values():
                    assert matrix.dtype == torch.float32 and matrix.device.type == "cpu"
                    assert not matrix.requires_grad and matrix.shape == (3, 3)
            for source in sources:
                g = full[source]
                expected_sum = g[q].index_select(1, valid).sum(dim=1)
                expected_diagonal = g[q, :, q, :]
                worst_api_error = max(
                    worst_api_error,
                    close_error(maps["sampled_sum"][source], expected_sum),
                    close_error(maps["diagonal"][source], expected_diagonal),
                )
            kwargs_one = {**kwargs, "dim_batch": 1}
            batch_one, _, _ = jacobians_for_prompt(**kwargs_one, mode="sampled", q=q)
            for reduction in maps:
                for source in sources:
                    worst_api_error = max(
                        worst_api_error,
                        close_error(maps[reduction][source], batch_one[reduction][source]),
                    )
            check_hooks_clean(model)

        for source in sources:
            g = full[source]
            dense_expected = (
                g.index_select(0, valid).index_select(2, valid).sum(dim=0).sum(dim=1)
                / n_valid
            )
            sum_exact_mean = torch.stack(
                [g[q].index_select(1, valid).sum(dim=1) for q in valid]
            ).mean(dim=0)
            diag_expected = torch.stack([g[q, :, q, :] for q in valid]).mean(dim=0)
            worst_algebra_error = max(
                worst_algebra_error,
                close_error(sum_exact_mean, dense_expected, atol=ALGEBRA_ATOL, rtol=ALGEBRA_ATOL),
            )
            sum_api_mean = torch.stack([maps["sampled_sum"][source].double() for maps in sampled]).mean(dim=0)
            diag_api_mean = torch.stack([maps["diagonal"][source].double() for maps in sampled]).mean(dim=0)
            worst_api_error = max(
                worst_api_error,
                close_error(wrapped["dense"][source], dense_expected),
                close_error(sum_api_mean, dense_expected),
                close_error(diag_api_mean, diag_expected),
            )
            # Deliberately wrong source MEAN fails: this would silently scale
            # the estimand by 1/|V| and change prompt weights when |V| varies.
            wrong_relative = float((sum_api_mean / n_valid - dense_expected).norm() / dense_expected.norm())
            assert wrong_relative > 0.6
            wrong_normalization_errors.append(wrong_relative)
            for q in range(seq_len):
                for p in range(q + 1, seq_len):
                    assert torch.count_nonzero(g[q, :, p, :]).item() == 0
            assert g[3, :, 1, :].norm().item() > 1e-3
        evidence.append({"target": target, "sources": sources, "q_enumerated": valid.tolist()})

    # A one-position valid set and a set including BOS both exercise endpoints.
    for skip_first, q in ((3, 3), (0, 0)):
        kwargs = dict(
            model=model, prompt=prompt, source_layers=[2], target_layer=3,
            dim_batch=2, max_seq_len=5, skip_first=skip_first,
        )
        sampled, _, n_valid = jacobians_for_prompt(**kwargs, mode="sampled", q=q)
        g = explicit_blocks(model, prompt, 2, 3)
        valid_edge = valid_position_mask(5, skip_first=skip_first).nonzero().flatten()
        close_error(sampled["sampled_sum"][2], g[q].index_select(1, valid_edge).sum(dim=1))
        if n_valid == 1:
            dense, _, _ = jacobians_for_prompt(**kwargs, mode="dense")
            close_error(sampled["sampled_sum"][2], sampled["diagonal"][2])
            close_error(sampled["sampled_sum"][2], dense["dense"][2])
    return {
        "target_source_position_cases": evidence,
        "dense_upstream_parity": "bitwise equal",
        "max_float64_algebra_abs_error": worst_algebra_error,
        "max_api_fp32_abs_error": worst_api_error,
        "wrong_source_mean_foil_relative_error_min": min(wrong_normalization_errors),
        "dimension_batches_checked": [1, 2],
        "width": 3,
        "vjp_calls_per_sampled_prompt": 2,
        "causal_forbidden_blocks": "exact zero",
    }


def boundary_checks(model: TinyDecoder) -> dict:
    kwargs = dict(
        model=model, prompt="abcdefghi", source_layers=[0, 1], target_layer=3,
        dim_batch=2, max_seq_len=5, skip_first=1, mode="sampled", q=2,
    )
    failures = [
        {"source_layers": []}, {"source_layers": [4]}, {"source_layers": [-5]},
        {"source_layers": [3]}, {"source_layers": [2], "target_layer": 2},
        {"source_layers": [2], "target_layer": 1}, {"target_layer": 4},
        {"target_layer": -5}, {"target_layer": 0}, {"source_layers": [True]},
        {"source_layers": None}, {"q": None}, {"q": -1}, {"q": 0}, {"q": 4},
        {"q": 5}, {"q": 8}, {"q": 2.0}, {"q": True}, {"dim_batch": 0},
        {"dim_batch": -1}, {"dim_batch": True}, {"max_seq_len": 0},
        {"skip_first": -1}, {"prompt": "x"}, {"mode": "unknown"},
        {"mode": "dense", "q": 2},
    ]
    for overrides in failures:
        require_raises(ValueError, jacobians_for_prompt, **{**kwargs, **overrides})
        check_hooks_clean(model)
    positive, _, _ = jacobians_for_prompt(**kwargs)
    negative, _, _ = jacobians_for_prompt(
        **{**kwargs, "source_layers": [-4, -3, -4], "target_layer": -1}
    )
    for reduction in positive:
        assert set(negative[reduction]) == {0, 1}
        for source in positive[reduction]:
            close_error(positive[reduction][source], negative[reduction][source], atol=0, rtol=0)
    # Repeated forwards leave neither hooks nor model parameter gradients.
    check_hooks_clean(model)
    return {"invalid_cases_rejected": len(failures), "negative_layer_resolution": "passed"}


class NonfiniteBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden):
        return hidden.clone()

    @staticmethod
    def backward(ctx, grad):
        bad = grad.clone()
        bad[:, 1, 0] = float("nan")
        return bad


class BadGradientBlock(nn.Module):
    def forward(self, hidden):
        return NonfiniteBackward.apply(hidden)


def failure_cleanup_checks(model: TinyDecoder) -> dict:
    bad = copy.deepcopy(model)
    bad.layers[1] = BadGradientBlock()
    kwargs = dict(
        model=bad, prompt="abcd", source_layers=[0], target_layer=1,
        dim_batch=2, max_seq_len=5, skip_first=1,
    )
    for mode, q in (("dense", None), ("sampled", 2)):
        require_raises(FloatingPointError, jacobians_for_prompt, **kwargs, mode=mode, q=q)
        check_hooks_clean(bad)
    missing = copy.deepcopy(model)
    # Keep the declared depth but omit the requested terminal block from forward.
    missing.layers = nn.ModuleList(list(missing.layers[:3]) + [nn.Identity()])
    missing.forward = lambda ids: None
    require_raises(
        RuntimeError, jacobians_for_prompt,
        **{**kwargs, "model": missing, "target_layer": 3, "mode": "sampled", "q": 2},
    )
    assert all(not block._forward_hooks for block in missing.layers)
    return {"nonfinite_dense_and_sampled_rejected": True, "hooks_clean_after_exceptions": True}


def finite_difference_checks(model: TinyDecoder) -> list[dict]:
    # FP32 checks match the predeclared numerical contract; float64 full blocks
    # above establish the estimator algebra separately.
    model = copy.deepcopy(model).float()
    u = torch.tensor([0.31, -0.72, 0.62])
    v = torch.tensor([-0.81, 0.44, 0.39])
    u, v = u / u.norm(), v / v.norm()
    results = []
    for source, target in ((0, 2), (0, 3), (2, 3)):
        hidden = activation(model, "abcd", source)
        g = explicit_blocks(model, "abcd", source, target)
        for p, q, relation in ((2, 2, "same position"), (1, 3, "past source"), (3, 1, "forbidden future source")):
            automatic = float(u @ g[q, :, p, :] @ v)
            estimates, passed = [], []
            for epsilon in FD_STEPS:
                shift = torch.zeros_like(hidden)
                shift[p] = epsilon * v
                plus = suffix(model, hidden + shift, source, target)[q]
                minus = suffix(model, hidden - shift, source, target)[q]
                estimate = float(u @ (plus - minus) / (2 * epsilon))
                error = abs(estimate - automatic)
                acceptable = error <= FD_ATOL + FD_RTOL * abs(automatic)
                estimates.append({"epsilon": epsilon, "estimate": estimate, "abs_error": error, "passed": acceptable})
                passed.append(acceptable)
            assert any(a and b for a, b in zip(passed, passed[1:], strict=False))
            if p > q:
                assert automatic == 0 and all(item["estimate"] == 0 for item in estimates)
            else:
                assert abs(automatic) > 1e-3
            results.append({
                "source": source, "target": target, "p": p, "q": q,
                "relation": relation, "automatic_derivative": automatic,
                "steps": estimates,
            })
    return results


def optimized_eager_checks() -> dict:
    """Exercise complete output banks, partial chunks, lengths and cleanup."""
    sys.path.insert(0, str(ROUND / "optimization"))
    from optimized_fitting import jacobian_for_prompt as optimized
    from saved_tensor_candidate import ReplicatedSavedTensorHooks

    cases = []
    for dtype in (torch.float32, torch.float64, torch.bfloat16):
        model = TinyDecoder(n_layers=4, d_model=5, seed=20260907)
        model.layers = nn.ModuleList([CausalBlock(5) for _ in range(4)])
        model.to(dtype).eval().requires_grad_(False)
        for prompt in ("abcd", "uvwxyz"):
            for target in (2, 3):
                for batch in (1, 2, 4, 8):
                    kwargs = dict(model=model, prompt=prompt,
                                  source_layers=list(range(target)), target_layer=target,
                                  dim_batch=batch, max_seq_len=7, skip_first=1)
                    expected, length, n_valid = jacobian_for_prompt(**kwargs)
                    for edges in (False, True):
                        for compress in (False, True):
                            details = {}
                            storage = (ReplicatedSavedTensorHooks(model, batch_size=batch,
                                       seq_len=length, min_savings_bytes=1) if compress else None)
                            actual, observed_len, observed_valid = optimized(
                                **kwargs, engine="eager", diagnostics=details,
                                source_edges=edges, storage_context=storage)
                            assert (observed_len, observed_valid) == (length, n_valid)
                            assert set(actual) == set(expected)
                            for layer, matrix in actual.items():
                                assert matrix.dtype == torch.float32 and matrix.device.type == "cpu"
                                assert matrix.shape == (5, 5) and torch.isfinite(matrix).all()
                                assert torch.equal(matrix, expected[layer])
                            payloads = storage.stats()["counts"].get("compressed_payloads", 0) if storage else 0
                            check_hooks_clean(model)
                            cases.append(dict(dtype=str(dtype), length=length, target=target,
                                              batch=batch, source_edges=edges, compress=compress,
                                              compressed_payloads=payloads,
                                              bitwise_equal=True))
        kwargs = dict(model=model, prompt="abcd", source_layers=[0, 1, 2],
                      target_layer=3, max_seq_len=5, skip_first=1, engine="eager")
        for bad_batch in (0, -1, True, 1.5):
            require_raises(ValueError, optimized, **kwargs, dim_batch=bad_batch)
        broken = copy.deepcopy(model)
        broken.forward = lambda ids: None
        require_raises(RuntimeError, optimized, **{**kwargs, "model": broken}, dim_batch=2)
        assert all(not block._forward_hooks for block in broken.layers)
        wrapped_kwargs = {name: value for name, value in kwargs.items() if name != "engine"}
        expected, expected_len, expected_valid = jacobians_for_prompt(**wrapped_kwargs, dim_batch=2)
        actual, actual_len, actual_valid = jacobians_for_prompt(
            **wrapped_kwargs, dim_batch=2, dense_engine="eager", compress_saved_tensors=True)
        assert (actual_len, actual_valid) == (expected_len, expected_valid)
        assert all(torch.equal(actual["dense"][layer], expected["dense"][layer]) for layer in expected["dense"])
        bad = copy.deepcopy(model)
        bad.layers[1] = BadGradientBlock()
        require_raises(FloatingPointError, jacobians_for_prompt,
                       **{**wrapped_kwargs, "model": bad}, dim_batch=2, dense_engine="eager")
        check_hooks_clean(bad)
    assert sum(case["compressed_payloads"] for case in cases) > 0
    return {"cases": cases, "partial_chunks_and_oversized_batch": True,
            "missing_hooks_cleanup": True, "invalid_batches_rejected": True,
            "wrapper_parity_and_nonfinite_guard": True,
            "scope": "eager CPU path only; CUDA replay requires separate real-device checks"}


def runner_checks() -> dict:
    """Exercise actual N100 checkpoint/resume orchestration with tiny matrices."""
    from deployment.run_refits import run_fit

    prompts = [f"independent calibration paragraph {index}" for index in range(100)]
    lengths = [20 + index % 5 for index in range(100)]
    valid = [length - 17 for length in lengths]
    sources = (0, 1)
    base = torch.tensor([[1.0, -2.0], [0.5, 4.0]], dtype=torch.float32)
    expected = {layer: (50.5 * base + 0.25 * layer).half() for layer in sources}
    results = []

    def new_compute(calls, *, bad=None):
        def compute(prompt, index, diagnostics):
            assert prompt == prompts[index]
            calls.append(index)
            diagnostics["fixture_index"] = index
            maps = {layer: (index + 1) * base + 0.25 * layer for layer in sources}
            length = lengths[index]
            if bad == "nonfinite":
                maps[0][0, 0] = float("nan")
            elif bad == "shape":
                maps[0] = maps[0][:1]
            elif bad == "keys":
                del maps[1]
            elif bad == "token_length":
                length += 1
            elif bad == "overflow":
                maps = {layer: torch.full((2, 2), torch.finfo(torch.float32).max) for layer in sources}
            return maps, length, valid[index]
        return compute

    def check_lens(result):
        assert result["status"] == "complete"
        assert result["n_done"] == result["next_idx"] == 100
        artifact = torch.load(result["lens_path"], map_location="cpu", weights_only=True)
        assert artifact["n_prompts"] == 100 and artifact["d_model"] == 2
        assert artifact["source_layers"] == list(sources)
        assert set(artifact["J"]) == set(sources)
        for layer in sources:
            assert artifact["J"][layer].dtype == torch.float16
            assert torch.equal(artifact["J"][layer], expected[layer])

    def must_fail(call):
        try:
            call()
        except (ValueError, RuntimeError, OSError):
            return
        raise AssertionError("runner accepted inconsistent or corrupted data")

    with tempfile.TemporaryDirectory(prefix="jlens-runner-checks-") as directory:
        root = Path(directory)

        def invoke(name, calls, **overrides):
            kwargs = dict(
                prompts=prompts, token_lengths=lengths, n_valid=valid, fit_id=1,
                identity={"fixture": "n100-equal-prompt-weight-v1"},
                output_dir=root / name, compute=new_compute(calls),
                d_model=2, source_layers=sources, cpu_test=True,
            )
            kwargs.update(overrides)
            return run_fit(**kwargs)

        calls = []
        stopped = invoke("resume", calls, stop_requested=lambda: len(calls) >= 37)
        assert stopped["status"] == "stopped"
        assert stopped["n_done"] == stopped["next_idx"] == 37
        assert calls == list(range(37))
        checkpoint = Path(stopped["checkpoint_path"])
        assert checkpoint.is_file()
        for label, change in (
            ("identity", {"identity": {"fixture": "changed"}}),
            ("prompt", {"prompts": ["changed first paragraph", *prompts[1:]]}),
            ("token_rule", {"token_lengths": [lengths[0] + 1, *lengths[1:]]}),
        ):
            failed_calls = []
            must_fail(lambda change=change: invoke("resume", failed_calls, **change))
            assert failed_calls == []
            results.append({"case": "resume_rejects_" + label, "passed": True})

        resumed_calls = []
        finished = invoke("resume", resumed_calls)
        assert resumed_calls == list(range(37, 100))
        check_lens(finished)
        noop_calls = []
        rerun = invoke("resume", noop_calls)
        check_lens(rerun)
        assert rerun["completed_noop"] is True and noop_calls == []
        results.append({"case": "stop37_resume_N100_and_verified_noop", "passed": True})

        calls = []
        invoke("locking", calls, stop_requested=lambda: len(calls) >= 3)
        lock_path = root / "locking/fit_01/.lock"
        lock_script = (
            "import fcntl,sys\n"
            "with open(sys.argv[1], 'a+b') as lock:\n"
            " fcntl.flock(lock, fcntl.LOCK_EX)\n"
            " print('locked', flush=True)\n"
            " sys.stdin.readline()\n"
        )
        holder = subprocess.Popen(
            [sys.executable, "-c", lock_script, str(lock_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            assert holder.stdout.readline().strip() == "locked"
            failed_calls = []
            must_fail(lambda: invoke("locking", failed_calls))
            assert failed_calls == []
        finally:
            holder.communicate("release\n", timeout=10)
        assert holder.returncode == 0
        resumed_calls = []
        check_lens(invoke("locking", resumed_calls))
        assert resumed_calls == list(range(3, 100))
        results.append({"case": "concurrent_process_lock_then_resume", "passed": True})

        # Sealed checkpoints and final artifacts must fail before recomputation
        # when their stored bytes no longer match their digest.
        calls = []
        damaged = invoke("corrupt_checkpoint", calls, stop_requested=lambda: len(calls) >= 3)
        path = Path(damaged["checkpoint_path"])
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 1
        path.write_bytes(data)
        failed_calls = []
        must_fail(lambda: invoke("corrupt_checkpoint", failed_calls))
        assert failed_calls == []
        results.append({"case": "corrupt_checkpoint_rejected", "passed": True})
        path = Path(finished["lens_path"])
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 1
        path.write_bytes(data)
        failed_calls = []
        must_fail(lambda: invoke("resume", failed_calls))
        assert failed_calls == []
        results.append({"case": "corrupt_final_artifact_rejected", "passed": True})

        stages = (
            "after_checkpoint_state", "after_checkpoint_metadata",
            "after_checkpoint_seal", "after_checkpoint_pointer",
            "after_final_state", "after_final_metadata", "after_final_seal", "after_final_pointer",
        )
        for stage in stages:
            calls = []
            fired = []

            def interrupt(actual, details):
                if actual == stage and (stage.startswith("after_final") or len(calls) == 3):
                    fired.append(actual)
                    raise RuntimeError("injected process interruption at " + actual)

            try:
                invoke(stage, calls, fault_hook=interrupt)
            except RuntimeError as error:
                assert "injected process interruption" in str(error)
            else:
                raise AssertionError("fault stage did not interrupt: " + stage)
            assert fired == [stage]
            resumed_calls = []
            completed = invoke(stage, resumed_calls)
            check_lens(completed)
            if stage.startswith("after_checkpoint"):
                assert resumed_calls in (list(range(2, 100)), list(range(3, 100)))
            else:
                assert resumed_calls == []
            results.append({"case": "crash_resume_" + stage, "passed": True})

        for count in (99, 101):
            calls = []
            bad_prompts = prompts[:count] if count < 100 else [*prompts, "extra paragraph"]
            must_fail(lambda: invoke(f"count{count}", calls, prompts=bad_prompts))
            assert calls == []
            results.append({"case": f"reject_count_{count}", "passed": True})
        for bad in ("nonfinite", "shape", "keys", "token_length", "overflow"):
            calls = []
            must_fail(lambda: invoke("bad_" + bad, calls, compute=new_compute(calls, bad=bad)))
            assert calls == ([0, 1] if bad == "overflow" else [0])
            results.append({"case": "reject_" + bad, "passed": True})
        calls = []
        stopped = invoke(
            "deadline", calls,
            deadline_utc=datetime.now(timezone.utc) - timedelta(seconds=1),
        )
        assert stopped["status"] == "stopped" and stopped["n_done"] == 0 and calls == []
        results.append({"case": "deadline_stops_before_first_prompt", "passed": True})

        # Paired controls share a cursor and one transaction. A missing arm or
        # a failure before the pointer must never advance only one estimator.
        def paired_compute(calls, *, missing=False):
            base_compute = new_compute(calls)
            def compute(prompt, index, diagnostics):
                dense, length, count = base_compute(prompt, index, diagnostics)
                banks = {"sampled_sum": dense,
                         "diagonal": {layer: matrix * -2 for layer, matrix in dense.items()}}
                if missing:
                    del banks["diagonal"]
                return banks, length, count
            return compute

        calls = []
        must_fail(lambda: invoke("pair_missing", calls, production_profile="ouro_positions",
                                 compute=paired_compute(calls, missing=True)))
        assert calls == [0] and not (root / "pair_missing/fit_01/LATEST.json").exists()
        results.append({"case": "paired_missing_arm_never_commits", "passed": True})
        for stage in ("after_checkpoint_state", "after_checkpoint_pointer", "after_final_seal"):
            calls = []
            def pair_fault(actual, details):
                if actual == stage and (stage.startswith("after_final") or len(calls) == 3):
                    raise RuntimeError("injected paired interruption")
            must_fail(lambda: invoke("pair_" + stage, calls, production_profile="ouro_positions",
                                     compute=paired_compute(calls), fault_hook=pair_fault))
            resumed_calls = []
            completed = invoke("pair_" + stage, resumed_calls, production_profile="ouro_positions",
                               compute=paired_compute(resumed_calls))
            payload = torch.load(completed["lens_path"], weights_only=True)
            assert payload["kind"] == "named_jacobian_lenses"
            assert payload["bank_arms"] == ["sampled_sum", "diagonal"]
            for layer in sources:
                assert torch.equal(payload["J"]["sampled_sum"][layer], expected[layer])
                assert torch.equal(payload["J"]["diagonal"][layer], expected[layer] * -2)
            assert resumed_calls == (list(range(2, 100)) if stage == "after_checkpoint_state"
                                     else list(range(3, 100)) if stage == "after_checkpoint_pointer" else [])
            noop_calls = []
            noop = invoke("pair_" + stage, noop_calls, production_profile="ouro_positions",
                          compute=paired_compute(noop_calls))
            assert noop["completed_noop"] and not noop_calls
            changed_calls = []
            must_fail(lambda: invoke("pair_" + stage, changed_calls, production_profile="ouro_penultimate"))
            assert not changed_calls
            results.append({"case": "paired_crash_resume_" + stage, "passed": True})
        for profile in ("ouro_penultimate", "huginn_r8"):
            calls = []
            check_lens(invoke(profile, calls, production_profile=profile))
            assert calls == list(range(100))
            # Tiny geometry is never accepted as a production execution.
            rejected_calls = []
            must_fail(lambda: invoke(profile + "_bad", rejected_calls, production_profile=profile, cpu_test=False))
            assert not rejected_calls
            results.append({"case": profile + "_typed_geometry_and_mean", "passed": True})

    return {
        "status": "passed", "scope": "actual N100 orchestration with CPU 2x2 matrices only",
        "equal_prompt_weight_with_varying_valid_positions": True,
        "source_sha256": {
            str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), ROUND / "deployment/run_refits.py", REPO / "jlens/lens.py")
        },
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-optimized", action="store_true")
    parser.add_argument("--check-runner", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.check_runner:
        rendered = json.dumps(runner_checks(), indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            args.output.write_text(rendered)
        print(rendered, end="")
        return
    model = causal_model()
    result = {
        "status": "passed",
        "scope": "CPU tiny causal model only; no real Ouro fit or derivative check",
        "torch_version": torch.__version__,
        "source_sha256": {
            str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (ROUND / "fit_estimators.py", Path(__file__), REPO / "jlens/fitting.py", REPO / "jlens/hooks.py", REPO / "tests/tiny.py")
        },
        "tolerances": {
            "float64_algebra_abs_and_relative": ALGEBRA_ATOL,
            "api_fp32_absolute": MAP_ATOL,
            "api_fp32_relative": MAP_RTOL,
            "fp32_finite_difference_absolute": FD_ATOL,
            "fp32_finite_difference_relative": FD_RTOL,
            "finite_difference_adjacent_passing_steps_required": 2,
        },
        "numerical_checks": numerical_checks(model),
        "boundary_checks": boundary_checks(model),
        "failure_cleanup_checks": failure_cleanup_checks(model),
        "fp32_finite_differences": finite_difference_checks(model),
    }
    if args.check_optimized:
        result["optimized_eager"] = optimized_eager_checks()
        for name in ("optimized_fitting.py", "cuda_graph_candidate.py", "saved_tensor_candidate.py"):
            path = ROUND / "optimization" / name
            result["source_sha256"][str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
