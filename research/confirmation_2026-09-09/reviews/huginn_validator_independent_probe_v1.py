#!/usr/bin/env python3
"""Independent CPU checks of the reviewed Huginn receiver integration."""
from __future__ import annotations
import argparse
import copy
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from datetime import datetime, timezone


def rec(path):
    raw = Path(path).read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    repo = args.repo.resolve()
    result = {"schema": "huginn_validator_independent_probe.v1", "status": "running",
              "utc": datetime.now(timezone.utc).isoformat(), "source": rec(source),
              "scope": "CPU fixtures and authenticated historical producer schemas; no model weights or new outcomes",
              "checks": []}
    root = Path(tempfile.mkdtemp(prefix="huginn-validator-independent-"))
    result["temporary_root"] = str(root)
    bundle = root / "bundle"
    bundle.mkdir()
    probe = repo / "research/confirmation_2026-09-09/huginn_verification/resources/contract_probe"
    for name in ("repo", "ouro_project"):
        shutil.copytree(probe / name, bundle / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (bundle / "evaluation").mkdir()
    copied_source = bundle / "evaluation/validate_outputs.py"
    shutil.copyfile(source, copied_source)
    assert rec(copied_source) == result["source"]
    spec = {"schema": "huginn_verification_run.v1", "source_records": {
        path.relative_to(bundle).as_posix(): rec(path)
        for path in bundle.rglob("*") if path.is_file()}}
    deployment = bundle / "repo/research/refit_round_2026-09-07/deployment"
    spec["historical_run_spec_record"] = rec(deployment / "run_spec.json")
    module_spec = importlib.util.spec_from_file_location("h_validator_independent", copied_source)
    v = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = v
    exec(compile(copied_source.read_bytes(), str(copied_source), "exec"), v.__dict__)
    spec["readout_paths"] = list(v.EXPECTED_READOUT_PATHS)
    import torch
    import numpy as np
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    def check(name, **details):
        result["checks"].append({"name": name, "status": "passed", **details})

    def reject(name, call):
        try:
            call()
        except ValueError as error:
            check(name, error=str(error))
        else:
            raise AssertionError("Invalid input was accepted: " + name)

    previous_cwd = Path.cwd()
    unrelated = root / "unrelated_cwd"
    unrelated.mkdir()
    os.chdir(unrelated)
    legacy_h, legacy_ref, legacy_run, legacy_h_run, legacy_paths = v._load_legacy(spec)
    try:
        assert all(path.is_relative_to(bundle) for path in legacy_paths.values())
        legacy_args = v._legacy_contract_args(spec, legacy_paths)
        assert legacy_args.ouro_src == bundle / "ouro_project/src"
        prepared = legacy_h_run.contract(legacy_args)
        check("relocated_exact_bundle_original_contract_from_unrelated_cwd",
              source_paths={name: str(path.relative_to(bundle)) for name, path in legacy_paths.items()},
              run_spec_sha256=prepared["main"]["spec_sha256"],
              n_prompts=len(prepared["main"]["fits"][0]["prompts"]))

        historical = repo / "research/refit_round_2026-09-07"
        owner_path = historical / "cloud_leases/attempt_06/retrieved/huginn/fit_01/OWNER.json"
        owner = json.loads(owner_path.read_text())
        owner_kwargs = {"combined_sha256": prepared["combined_sha256"],
                        "calibration_sha256": prepared["calibration_sha256"]}
        def validate_owner(payload):
            return legacy_h.validate_huginn_owner(
                payload, prepared["main"], prepared["combined"], prepared["manifest"],
                prepared["calibration"], **owner_kwargs)
        validate_owner(owner)
        check("authentic_original_owner_validates_portably", owner_record=rec(owner_path))
        altered_owner = copy.deepcopy(owner)
        altered_owner["identity"]["runtime"]["initialization"]["base_seed"] += 1
        altered_owner["fit_identity_sha256"] = v.digest(altered_owner["identity"])
        reject("resealed_owner_wrong_calibration_seed_rejected", lambda: validate_owner(altered_owner))

        init_root = root / "initialization_fixture"
        init_root.mkdir()
        lens_path = init_root / "lens.pt"
        lens_path.write_bytes(b"not-loaded-by-initialization-check")
        diagnostics = []
        for i, row in enumerate(prepared["calibration"]["rows"]):
            diagnostics.append({"index": i, "initialization": {
                "base_seed": row["base_seed"], "namespace": "calibration",
                "prompt_state_seed": row["prompt_state_seed"],
                "input_ids_sha256": row["input_ids_sha256"], "seed_recipe": v.SEED_RULE,
                "dtype": "torch.bfloat16", "test_time_noise": 0,
                "one_prompt_shape": [1, row["token_length"], 5280],
                "generator_device": "cuda:0", "native_draw": v.FIT_NATIVE_DRAW,
                "lane_coupling": "contiguous copies of the same one-prompt state"}})
        def seal_diagnostics(rows):
            metadata = init_root / "metadata.json"
            metadata.write_bytes(v.canonical({"diagnostics": rows}))
            (init_root / "SEAL.json").write_bytes(v.canonical({"files": {"metadata.json": rec(metadata)}}))
        seal_diagnostics(diagnostics)
        actual = legacy_h._fit_initializations(lens_path, prepared["calibration"])
        assert actual == {"rows": 100, "initializations_sha256": v.digest([x["initialization"] for x in diagnostics])}
        check("all_100_frozen_initialization_records_validated", **actual)
        for field, wrong in (("prompt_state_seed", diagnostics[73]["initialization"]["prompt_state_seed"] + 1),
                             ("input_ids_sha256", "0" * 64), ("dtype", "torch.float32")):
            altered = copy.deepcopy(diagnostics)
            altered[73]["initialization"][field] = wrong
            seal_diagnostics(altered)
            reject("resealed_paragraph_73_wrong_" + field + "_rejected",
                   lambda: legacy_h._fit_initializations(lens_path, prepared["calibration"]))
    finally:
        v._unload_legacy(legacy_paths)
        os.chdir(previous_cwd)

    # The readout consumer runs over actual authenticated historical cache and
    # rank files. No model is loaded and no new-model outcome is accessed.
    old_readouts = repo / "research/refit_round_2026-09-07/monitoring/attempt_06/huginn_handoff_20260909T153351Z_659a482c/results/huginn_evaluation"
    readout_root = root / "readout_fixture"
    paths = []
    for relative in v.EXPECTED_READOUT_PATHS:
        target = readout_root / "readouts" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(old_readouts / relative, target)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copyfile(old_readouts / relative, target)
        paths.append("readouts/" + relative)
    original_readout_owner = json.loads((old_readouts / "OWNER.json").read_text())
    fit_record = original_readout_owner["identity"]["fit"]
    fit = {"fit_record": fit_record, "fit_identity_sha256": fit_record["fit_identity_sha256"]}
    observed = v._validate_readouts(readout_root, spec, paths, fit)
    assert observed["files"] == 17
    check("new_wrapper_original_validator_accepts_all_17_authentic_historical_readout_files", **observed)
    bad_fit = copy.deepcopy(fit)
    bad_fit["fit_record"]["fit_identity_sha256"] = "0" * 64
    reject("readout_different_fit_generation_rejected",
           lambda: v._readout_identity_checks(readout_root, spec, bad_fit))

    # Exercise conversion at signed zeros, subnormals, round-to-nearest ties,
    # and ordinary magnitudes; then alter only a signed zero bit.
    base = np.array([0.0, -0.0, np.nextafter(np.float32(0), np.float32(1)),
                     -np.nextafter(np.float32(0), np.float32(1)), 1.0, -1.0,
                     100.0, -100.0, 100.048828125, -100.048828125,
                     0.0000059604644775390625, -0.0000059604644775390625,
                     6550400.0, -6550400.0, 3.1415927, -3.1415927], dtype=np.float32)
    total = torch.from_numpy(base.reshape(4, 4).copy())
    saved = (total / 100).half()
    samples = []
    count = v._independent_fp16_conversion({0: total}, {0: saved}, sources=(0,), width=4, samples=samples)
    assert count == 16 and all(x["numpy_fp16_bits"] == x["saved_fp16_bits"] for x in samples)
    check("numpy_conversion_signed_zeros_subnormals_ties_and_large_finite", entries=count, samples=samples)
    altered_saved = saved.clone()
    altered_saved.view(torch.int16)[0, 0] = -32768
    reject("conversion_changed_signed_zero_rejected", lambda: v._independent_fp16_conversion(
        {0: total}, {0: altered_saved}, sources=(0,), width=4))

    # Resolve trusted paths first, then replace source bytes before compile.
    # The changed code would create a marker if executed.
    selected_paths = v._legacy_source_paths(spec)
    selected = selected_paths["run_refits"]
    original_raw = selected.read_bytes()
    marker = root / "unexpected_executed_source"
    selected.write_bytes(original_raw + ("\nfrom pathlib import Path\nPath(" + repr(str(marker)) + ").touch()\n").encode())
    original_locator = v._legacy_source_paths
    v._legacy_source_paths = lambda _: selected_paths
    try:
        reject("source_replacement_between_selection_and_compile_rejected", lambda: v._load_legacy(spec))
        assert not marker.exists()
    finally:
        selected.write_bytes(original_raw)
        v._legacy_source_paths = original_locator

    assert rec(source) == result["source"] == rec(copied_source)
    assert not torch.cuda.is_initialized()
    check("no_source_change_and_no_cuda_initialization")
    result["status"] = "passed"
    result["script_record"] = rec(Path(__file__))
    result["cuda_initialized"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
    print(json.dumps({"status": "passed", "checks": len(result["checks"]), "proof": str(args.output), "proof_record": rec(args.output)}))


if __name__ == "__main__":
    main()
