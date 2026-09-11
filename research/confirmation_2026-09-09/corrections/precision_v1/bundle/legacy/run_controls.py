"""Run the frozen, paired Ouro controls after the five main fits complete.

The penultimate map and the jointly checkpointed position arms each reuse all
100 paragraphs of preselected fit01. No score or partial mean selects a fit.
Plan mode is standard-library-only and does not create an output directory.
"""
from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path
import random
import signal
import sys

import run_refits as runner


def _source(path, name, main):
    key = ("jlens", "research/refit_round_2026-09-07/deployment/" + name)
    records = {(row["root"], row["path"]): row for row in main["sources"]}
    if key not in records or key not in main["source_paths"]:
        raise ValueError(f"control run spec does not bind {name}")
    if runner._file_hash(path) != records[key]["sha256"]:
        raise ValueError(f"control source differs from the frozen run spec: {name}")


def contract(args):
    main = runner._contract(args.run_spec, args.ouro_src, list(range(1, 6)))
    for name, path in (("run_controls.py", Path(__file__)),
                       ("control_estimators.py", Path(__file__).with_name("control_estimators.py")),
                       ("combined_contract.json", args.combined_contract)):
        _source(path, name, main)
    combined = runner._json(runner._no_links(args.combined_contract))
    fit = main["fits"][0]
    controls = combined["controls"]
    if (combined.get("schema_version") != 1 or combined.get("budget_usd") != 25
            or controls.get("calibration_fit_id") != 1
            or controls.get("calibration_seed") != fit["seed"]
            or controls.get("calibration_sha256") != fit["sha256"]
            or controls.get("n_prompts") != 100):
        raise ValueError("combined control contract differs from preselected fit01 N100")
    penult, positions = controls["penultimate"], controls["positions"]
    if (penult != {"target_layer": 190, "source_layers": list(range(190)), "mode": "dense"}
            or positions.get("target_layer") != 191
            or positions.get("source_layers") != list(range(191))
            or positions.get("arms") != ["sampled_sum", "diagonal"]
            or controls.get("common_target_support") != list(range(190))
            or controls.get("common_position_support") != list(range(191))):
        raise ValueError("control geometry or shared scoring support has changed")
    rng = random.Random(runner._integer("q_seed", positions["q_seed"]))
    if positions["q"] != [rng.randrange(16, length - 1) for length in fit["token_lengths"]]:
        raise ValueError("the saved paired q positions differ from their frozen uniform draws")
    return main, combined


def _complete_main(root, expected, main_contract):
    """Verify the five whole means before starting a conditional control."""
    import torch
    shared = runner._json(runner._no_links(root / "RUN_IDENTITY.json"))
    if (shared.get("schema_version") != 1 or shared.get("kind") != "ouro_n100_run"
            or shared.get("identity_sha256") != runner._digest(shared.get("identity"))):
        raise ValueError("invalid shared main run identity")
    original = shared["identity"]
    for key in ("run_spec_sha256", "model", "snapshot_files", "source_files", "environment",
                "gpu", "precision", "settings", "attention_implementation"):
        if original.get(key) != expected.get(key):
            raise ValueError(f"control and main fitting runtimes differ: {key}")
    for fit_id in range(1, 6):
        fit = next(fit for fit in main_contract["fits"] if fit["fit_id"] == fit_id)
        directory = root / f"fit_{fit_id:02d}"
        owner = runner._json(runner._no_links(directory / "OWNER.json"))
        identity = runner._fit_identity(
            fit["prompts"], fit["token_lengths"], fit["n_valid"], fit_id,
            {**original, "calibration": {key: value for key, value in fit.items() if key != "prompts"}},
            2048, list(range(191)), False)
        if owner != {"schema_version": 1, "fit_identity_sha256": runner._digest(identity), "identity": identity}:
            raise ValueError(f"main fit {fit_id} has a different production identity")
        checkpoint = runner._read_checkpoint(torch, directory, identity)
        if runner._read_final(torch, directory, identity, checkpoint) is None:
            raise ValueError(f"main fit {fit_id} is not complete")
        del checkpoint
        gc.collect()


def execute(args, main, combined, deadline):
    model, estimators, runtime = runner._load_ouro_runtime(args, main)
    _complete_main(args.main_output_dir, runtime, main)
    import control_estimators
    _source(Path(control_estimators.__file__), "control_estimators.py", main)
    runtime = {**runtime, "combined_contract_sha256": runner._file_hash(args.combined_contract)}
    runner._bind_run_identity(args.output_dir, runtime, kind="ouro_controls_n100_run")
    fit = main["fits"][0]
    for index, prompt in enumerate(fit["prompts"]):
        length = int(model.encode(prompt, max_length=128).shape[1])
        if length != fit["token_lengths"][index] or length - 17 != fit["n_valid"][index]:
            raise ValueError(f"loaded tokenizer changes control paragraph {index}")
    stop = {"requested": False}
    handlers = {}
    def request_stop(signum, frame):
        stop["requested"] = True
    for signum in (signal.SIGTERM, signal.SIGINT):
        handlers[signum] = signal.signal(signum, request_stop)
    try:
        for profile_name in ("ouro_penultimate", "ouro_positions"):
            profile = runner.PROFILES[profile_name]
            settings = {**runner.SETTINGS, "source_layers": profile["source_layers"],
                        "target_layer": profile["target_layer"],
                        "mode": "paired_sampled" if profile["bank_arms"] else "dense"}
            def compute(prompt, index, diagnostics):
                options = dict(target_layer=profile["target_layer"], dim_batch=8,
                               max_seq_len=128, skip_first=16, compress_saved_tensors=True,
                               diagnostics=diagnostics)
                if profile["bank_arms"] is not None:
                    q = combined["controls"]["positions"]["q"][index]
                    diagnostics["frozen_q"] = q
                    return control_estimators.paired_jacobians_for_prompt(
                        model, prompt, profile["source_layers"], q=q, engine="cuda_graph", **options)
                maps, length, count = estimators.jacobians_for_prompt(
                    model, prompt, profile["source_layers"], mode="dense", dense_engine="cuda_graph", **options)
                if set(maps) != {"dense"}:
                    raise ValueError("penultimate fit received unexpected estimator arms")
                return maps["dense"], length, count
            identity = {**runtime, "settings": settings, "production_profile": profile_name,
                        "control_contract": combined["controls"],
                        "calibration": {key: value for key, value in fit.items() if key != "prompts"}}
            result = runner.run_fit(
                prompts=fit["prompts"], token_lengths=fit["token_lengths"], n_valid=fit["n_valid"],
                fit_id=1, identity=identity, output_dir=args.output_dir / profile_name,
                compute=compute, source_layers=profile["source_layers"], d_model=profile["d_model"],
                production_profile=profile_name, stop_requested=lambda: stop["requested"],
                deadline_utc=deadline, reserve_seconds=args.reserve_seconds,
                initial_prompt_bound_seconds=args.initial_prompt_bound_seconds)
            print(runner.json.dumps({"profile": profile_name, **result}), flush=True)
            gc.collect()
            if result["status"] != "complete":
                return 0
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-spec", type=Path, default=runner.HERE / "run_spec.json")
    parser.add_argument("--combined-contract", type=Path, default=runner.HERE / "combined_contract.json")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--ouro-src", type=Path, required=True)
    parser.add_argument("--main-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--stop-at-utc")
    parser.add_argument("--reserve-seconds", type=float, default=600.0)
    parser.add_argument("--initial-prompt-bound-seconds", type=float, default=600.0)
    args = parser.parse_args(argv)
    deadline = runner._deadline(args.stop_at_utc)
    if not args.plan and deadline is None:
        parser.error("execution requires --stop-at-utc")
    for name in ("reserve_seconds", "initial_prompt_bound_seconds"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"{name} must be finite and nonnegative")
    main_contract, combined = contract(args)
    if args.plan:
        print(runner.json.dumps({"status": "plan", "model_loaded": False, "gpu_initialized": False,
              "run_spec_sha256": main_contract["spec_sha256"],
              "combined_contract_sha256": runner._file_hash(args.combined_contract),
              "profiles": {key: runner.PROFILES[key] for key in ("ouro_penultimate", "ouro_positions")},
              "calibration_fit_id": 1, "n_prompts_each": 100}, indent=2))
        return 0
    with runner._output_lock(args.output_dir):
        return execute(args, main_contract, combined, deadline)


if __name__ == "__main__":
    raise SystemExit(main())
