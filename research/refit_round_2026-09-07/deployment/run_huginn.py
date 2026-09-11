"""Run the frozen Huginn R=8 N100 fit after verified Ouro evaluations.

Plan mode uses only the standard library and can report missing evaluation
prerequisites. Production verifies every local Huginn snapshot file before
loading weights, preserves the native B=8 dense derivative engine, and binds
each paragraph's token IDs and initialization seed to the frozen calibration.
The shared checkpoint runner publishes an FP16 lens only after all 100 inputs.
This program neither rents hardware nor controls the provider's lease deadline.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import inspect
import json
import math
import signal
import sys
from pathlib import Path

try:
    from . import run_refits as runner
except ImportError:
    import run_refits as runner


HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
REPO = HERE.parents[2]
REPO_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
SEED_NAMESPACE = "calibration"
SEED_RECIPE = "SHA256(canonical JSON(version, base_seed, namespace, input_ids)); first 8 bytes big-endian modulo 2^63"
SETTINGS = {
    "n_ut": 8, "n_physical": 4, "n_prelude": 2, "n_coda": 2,
    "d_model": 5280, "target_layer": 33, "source_layers": list(range(32)),
    "mode": "dense", "dense_engine": "cuda_graph", "compress_saved_tensors": True,
    "dim_batch": 8, "max_seq_len": 128, "skip_first": 16,
    "accumulation_dtype": "float32", "saved_dtype": "float16", "checkpoint_every": 1,
}
MAIN_SECTIONS = ["common", *[f"fits/fit_{fit_id:02d}" for fit_id in range(1, 6)]]
CONTROL_SECTIONS = ["penultimate", "positions/sampled_sum", "positions/diagonal", "comparisons"]


def _same(left, right):
    return runner._canonical(left) == runner._canonical(right)


def _source(path, name, main_contract):
    """Check actual imported/input bytes against the frozen logical source."""
    key = ("jlens", "research/refit_round_2026-09-07/deployment/" + name)
    records = {(row["root"], row["path"]): row for row in main_contract["sources"]}
    if key not in records or key not in main_contract["source_paths"]:
        raise ValueError(f"run specification does not bind Huginn input {name}")
    path = runner._no_links(path)
    if runner._file_hash(path) != records[key]["sha256"]:
        raise ValueError(f"Huginn input differs from the frozen run specification: {name}")
    return {"path": str(path), "sha256": records[key]["sha256"]}


def _manifest(manifest):
    if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1 or manifest.get("repo_id") != REPO_ID
            or manifest.get("revision") != REVISION or not isinstance(manifest.get("files"), list)
            or not manifest["files"]):
        raise ValueError("Huginn snapshot manifest has an unexpected identity or schema")
    records = {}
    for row in manifest["files"]:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise ValueError("invalid Huginn snapshot file record")
        name = runner._relative(row["path"]).as_posix()
        runner._integer("snapshot file bytes", row["bytes"])
        if (name in records or not isinstance(row["sha256"], str)
                or runner._SHA256.fullmatch(row["sha256"]) is None):
            raise ValueError("duplicate Huginn snapshot path or invalid file hash")
        records[name] = row
    runner._integer("snapshot total bytes", manifest.get("total_bytes"), 1)
    if (sum(row["bytes"] for row in records.values()) != manifest["total_bytes"]
            or not {"config.json", "raven_config_minimal.py", "raven_modeling_minimal.py",
                    "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
                    "special_tokens_map.json"} <= set(records)
            or not any(name.endswith(".safetensors") for name in records)):
        raise ValueError("Huginn snapshot manifest is incomplete or has a wrong total")


def _calibration(calibration, combined, fit, manifest_sha256):
    huginn = combined.get("huginn") if isinstance(combined, dict) else None
    expected = {
        "repo_id": REPO_ID, "revision": REVISION, "n_prompts": 100,
        "calibration_fit_id": 1, "calibration_sha256": fit["sha256"],
        "n_recurrences": 8, "source_layers": list(range(32)), "target_layer": 33,
        "d_model": 5280, "dim_batch": 8, "max_seq_len": 128, "skip_first": 16,
    }
    if (not isinstance(huginn, dict)
            or not all(_same(huginn.get(key), value) for key, value in expected.items())
            or type(combined.get("schema_version")) is not int or combined["schema_version"] != 1
            or not _same(combined.get("budget_usd"), 25)):
        raise ValueError("combined contract changes the frozen Huginn N100 R8 geometry")
    base = runner._integer("initialization_base_seed", huginn.get("initialization_base_seed"))
    if base + 99 >= 2 ** 63:
        raise ValueError("Huginn paragraph base seeds must fit in 63 bits")
    expected = {
        "schema_version": 1, "n_prompts": 100, "calibration_fit_id": 1,
        "calibration_sha256": fit["sha256"], "model_manifest_sha256": manifest_sha256,
        "tokenizer_bos_id": 65504, "max_seq_len": 128, "skip_first": 16,
        "seed_namespace": SEED_NAMESPACE,
    }
    if (not isinstance(calibration, dict)
            or not all(_same(calibration.get(key), value) for key, value in expected.items())
            or not isinstance(calibration.get("rows"), list) or len(calibration["rows"]) != 100):
        raise ValueError("Huginn calibration does not bind the frozen fit01 inputs")
    for index, row in enumerate(calibration["rows"]):
        if (not isinstance(row, dict)
                or set(row) != {"index", "token_length", "n_valid", "input_ids_sha256",
                                "base_seed", "prompt_state_seed"}):
            raise ValueError("invalid Huginn calibration row")
        for name, minimum in (("index", 0), ("token_length", 18), ("n_valid", 1),
                              ("base_seed", 0), ("prompt_state_seed", 0)):
            runner._integer(name, row[name], minimum)
        if (row["index"] != index or row["token_length"] > 128
                or row["n_valid"] != row["token_length"] - 17
                or row["base_seed"] != base + index or row["prompt_state_seed"] >= 2 ** 63
                or not isinstance(row["input_ids_sha256"], str)
                or runner._SHA256.fullmatch(row["input_ids_sha256"]) is None):
            raise ValueError(f"Huginn calibration row {index} violates token/seed rules")
    return base


def contract(args):
    """Validate frozen metadata and source bytes without importing torch."""
    main = runner._contract(args.run_spec, args.ouro_src, list(range(1, 6)))
    manifest_path = HERE / "huginn_model_manifest.json"
    for name, path in (
        ("run_huginn.py", Path(__file__)), ("run_refits.py", Path(runner.__file__)),
        ("huginn_adapter.py", HERE / "huginn_adapter.py"),
        ("huginn_model_manifest.json", manifest_path),
        ("huginn_calibration.json", args.calibration),
        ("combined_contract.json", args.combined_contract),
        ("evaluate_refits.py", HERE / "evaluate_refits.py"),
        ("evaluate_controls.py", HERE / "evaluate_controls.py"),
    ):
        _source(path, name, main)
    manifest = runner._json(manifest_path)
    _manifest(manifest)
    combined = runner._json(runner._no_links(args.combined_contract))
    calibration = runner._json(runner._no_links(args.calibration))
    fit = next(fit for fit in main["fits"] if fit["fit_id"] == 1)
    manifest_sha256 = runner._file_hash(manifest_path)
    base = _calibration(calibration, combined, fit, manifest_sha256)
    return {
        "main": main, "combined": combined, "calibration": calibration, "fit": fit,
        "manifest": manifest, "manifest_sha256": manifest_sha256, "base_seed": base,
        "combined_sha256": runner._file_hash(args.combined_contract),
        "calibration_sha256": runner._file_hash(args.calibration),
    }


def _evaluation_modules(main_contract):
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    modules = []
    for name in ("evaluate_refits", "evaluate_controls"):
        module = importlib.import_module(name)
        _source(Path(module.__file__), name + ".py", main_contract)
        modules.append(module)
    return tuple(modules)


def _main_evaluation(root, evaluator, main_contract):
    validator = getattr(evaluator, "validate_completed_main", None)
    if not callable(validator):
        raise RuntimeError("the frozen evaluator lacks its main completion-binding validator")
    checked = validator(root, main_contract, require_all=True)
    if (checked["owner"].get("schema") != "ouro_refit_evaluation.v1"
            or not _same(checked["owner"]["identity"].get("fit_ids"), list(range(1, 6)))
            or [row.get("section") for row in checked["complete"]["sections"]] != MAIN_SECTIONS):
        raise ValueError("Huginn requires the complete frozen five-fit Ouro evaluation")
    return checked


def _evaluation_gate(args, prepared):
    """Validate complete, semantically bound Ouro artifacts before model load."""
    if args.ouro_evaluation_dir is None or args.controls_evaluation_dir is None:
        raise ValueError("Huginn execution requires both explicit Ouro evaluation directories")
    _, control_evaluator = _evaluation_modules(prepared["main"])
    main = _main_evaluation(args.ouro_evaluation_dir, control_evaluator, prepared["main"])
    # The control producer owns its semantic profile/calibration validation.
    # Absence of that validator is a hard failure, never a schema-only bypass.
    validate_controls = getattr(control_evaluator, "validate_completed_controls", None)
    if not callable(validate_controls):
        raise RuntimeError("the frozen controls evaluator lacks its completion-binding validator")
    controls = validate_controls(args.controls_evaluation_dir, prepared["main"],
                                 prepared["combined"], main)
    identity = controls["owner"]["identity"]
    expected = {
        "kind": "ouro_controls_n100_readout", "run_spec_sha256": prepared["main"]["spec_sha256"],
        "combined_contract_sha256": prepared["combined_sha256"],
        "source_files": prepared["main"]["sources"], "fit_ids": [1],
        "control_profiles": ["ouro_penultimate", "ouro_positions"],
        "main_evaluation_owner_sha256": runner._digest(main["owner"]), "sections": CONTROL_SECTIONS,
    }
    if (controls["owner"].get("schema") != "ouro_controls_evaluation.v1"
            or not all(_same(identity.get(key), value) for key, value in expected.items())
            or [row.get("section") for row in controls["complete"]["sections"]] != CONTROL_SECTIONS):
        raise ValueError("Huginn requires complete controls paired to this five-fit Ouro evaluation")
    records = {}
    for name, root, checked in (("ouro", args.ouro_evaluation_dir, main),
                                ("controls", args.controls_evaluation_dir, controls)):
        root = runner._no_links(root)
        owner = checked["owner"]
        records[name] = {
            "schema": owner["schema"], "identity_sha256": owner["identity_sha256"],
            "owner_sha256": runner._digest(owner),
            "owner_file": runner._record(root / "OWNER.json"),
            "complete_file": runner._record(root / "COMPLETE.json"),
        }
    return records


def _paragraph_initialization(model, prompt, row):
    """Check actual tokens and the independently frozen prompt-seed recipe."""
    model.state_seed = row["base_seed"]
    if model.seed_namespace != SEED_NAMESPACE:
        raise ValueError("Huginn fitting initialization namespace changed")
    input_ids = model.encode(prompt, max_length=128)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Huginn encoder must return one unpadded prompt")
    tokens = input_ids[0].detach().cpu().tolist()
    length = len(tokens)
    ids_sha256 = runner._digest(tokens)
    if (length != row["token_length"] or length - 17 != row["n_valid"]
            or ids_sha256 != row["input_ids_sha256"]):
        raise ValueError(f"actual Huginn tokens differ for paragraph {row['index']}")
    recipe = {"version": 1, "base_seed": row["base_seed"],
              "namespace": SEED_NAMESPACE, "input_ids": tokens}
    seed = int.from_bytes(hashlib.sha256(runner._canonical(recipe)).digest()[:8], "big") % (2 ** 63)
    if seed != row["prompt_state_seed"]:
        raise ValueError(f"frozen Huginn initialization seed differs for paragraph {row['index']}")
    metadata = model.initialization_metadata(input_ids)
    expected = {
        "base_seed": row["base_seed"], "namespace": SEED_NAMESPACE,
        "input_ids_sha256": ids_sha256, "prompt_state_seed": seed, "seed_recipe": SEED_RECIPE,
        "one_prompt_shape": [1, length, model.d_model], "generator_device": str(model.input_device),
        "dtype": str(model.dtype), "test_time_noise": 0,
    }
    if (not isinstance(metadata, dict)
            or not all(_same(metadata.get(key), value) for key, value in expected.items())):
        raise ValueError(f"actual Huginn initialization metadata differs for paragraph {row['index']}")
    return json.loads(runner._canonical(metadata))


def _load_runtime(args, prepared, prerequisites):
    snapshot_files = runner._snapshot(args.snapshot, prepared["manifest"])
    runner._runtime_cache(args.output_dir)
    import torch
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    environment = runner._environment(torch, prepared["main"]["environment"])
    if not torch.cuda.is_available():
        raise RuntimeError("the fixed Huginn production engine requires an available CUDA GPU")
    estimators, _, imported_sources = runner._imports(prepared["main"], args.ouro_src)
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    adapter = importlib.import_module("huginn_adapter")
    record = _source(Path(adapter.__file__), "huginn_adapter.py", prepared["main"])
    _source(Path(runner.__file__), "run_refits.py", prepared["main"])
    _source(Path(__file__), "run_huginn.py", prepared["main"])
    imported_sources.append({"module": "huginn_adapter", **record})
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    gpu = {"index": index, "name": properties.name, "total_memory": properties.total_memory,
           "capability": [properties.major, properties.minor],
           "uuid": str(getattr(properties, "uuid", "unavailable")),
           "multiprocessor_count": properties.multi_processor_count}
    precision = runner._precision(torch)
    model = adapter.load_huginn(args.snapshot, device=f"cuda:{index}", dtype=torch.bfloat16,
                               state_seed=prepared["base_seed"], seed_namespace=SEED_NAMESPACE)
    if ((model.n_ut, model.n_physical, model.n_prelude, model.n_coda, model.n_layers, model.d_model)
            != (8, 4, 2, 2, 34, 5280) or tuple(model.source_layers) != tuple(range(32))
            or model.target_layer != 33 or len(model.layers) != 34
            or model.declared_revision != REVISION):
        raise ValueError("loaded Huginn geometry differs from the frozen R8 fit")
    if (any(module.training for module in model.hf_model.modules())
            or any(parameter.requires_grad or parameter.dtype != torch.bfloat16
                   or parameter.device != torch.device("cuda", index)
                   for parameter in model.hf_model.parameters())):
        raise ValueError("Huginn production parameters must be frozen BF16 on the selected CUDA device")
    remote = []
    for cls in (type(model.hf_model), type(model.hf_model.config)):
        path = Path(inspect.getfile(cls))
        digest = runner._file_hash(path)
        if path.name not in snapshot_files or digest != snapshot_files[path.name]["sha256"]:
            raise ValueError(f"loaded Huginn native implementation differs from the snapshot: {path}")
        remote.append({"class": f"{cls.__module__}.{cls.__name__}", "path": str(path), "sha256": digest})
    if any(name not in snapshot_files or digest != snapshot_files[name]["sha256"]
           for name, digest in model.verified_metadata_sha256.items()):
        raise ValueError("Huginn loader metadata differs from the complete snapshot manifest")
    identity = {
        "run_spec_sha256": prepared["main"]["spec_sha256"],
        "combined_contract_sha256": prepared["combined_sha256"],
        "huginn_calibration_sha256": prepared["calibration_sha256"],
        "model": {"repo_id": REPO_ID, "revision": REVISION, "manifest_sha256": prepared["manifest_sha256"]},
        "snapshot_path": str(args.snapshot.absolute()), "snapshot_files": snapshot_files,
        "source_files": prepared["main"]["sources"], "imported_sources": imported_sources,
        "remote_implementations": remote, "environment": environment, "gpu": gpu,
        "precision": precision, "settings": SETTINGS,
        "initialization": {"base_seed": prepared["base_seed"], "seed_namespace": SEED_NAMESPACE,
                           "base_seed_rule": "base_seed + zero-based paragraph index", "prompt_seed_rule": SEED_RECIPE},
        "prerequisites": prerequisites,
        "attention_implementation": str(getattr(model.hf_model.config, "_attn_implementation", None)),
    }
    return model, estimators, identity


def _compute(model, estimators, calibration):
    def compute(prompt, index, diagnostics):
        row = calibration["rows"][index]
        metadata = _paragraph_initialization(model, prompt, row)
        maps, length, count = estimators.jacobians_for_prompt(
            model, prompt, SETTINGS["source_layers"], target_layer=33, dim_batch=8,
            max_seq_len=128, skip_first=16, mode="dense", dense_engine="cuda_graph",
            compress_saved_tensors=True, diagnostics=diagnostics)
        if set(maps) != {"dense"}:
            raise ValueError("Huginn dense engine returned unexpected estimator arms")
        if not _same(model.last_initialization, metadata):
            raise ValueError(f"Huginn forward used another initialization for paragraph {index}")
        diagnostics["initialization"] = metadata
        return maps["dense"], length, count
    return compute


def execute(args, prepared, prerequisites, deadline):
    if (runner._utc_now().timestamp() + args.reserve_seconds + args.initial_prompt_bound_seconds
            >= deadline.timestamp()):
        return {"status": "stopped", "reason": "insufficient deadline margin before model load",
                "model_loaded": False, "n_done": None}
    model, estimators, runtime = _load_runtime(args, prepared, prerequisites)
    runner._bind_run_identity(args.output_dir, runtime, kind="huginn_r8_n100_run")
    fit = prepared["fit"]
    for prompt, row in zip(fit["prompts"], prepared["calibration"]["rows"], strict=True):
        _paragraph_initialization(model, prompt, row)
    stop = {"requested": False}
    handlers = {}

    def request_stop(signum, frame):
        stop["requested"] = True

    for signum in (signal.SIGTERM, signal.SIGINT):
        handlers[signum] = signal.signal(signum, request_stop)
    try:
        return runner.run_fit(
            prompts=fit["prompts"],
            token_lengths=[row["token_length"] for row in prepared["calibration"]["rows"]],
            n_valid=[row["n_valid"] for row in prepared["calibration"]["rows"]],
            fit_id=1, identity={**runtime, "calibration": {
                "origin": {key: value for key, value in fit.items() if key != "prompts"},
                "huginn": prepared["calibration"],
            }}, output_dir=args.output_dir, compute=_compute(model, estimators, prepared["calibration"]),
            d_model=5280, source_layers=tuple(range(32)), production_profile="huginn_r8",
            stop_requested=lambda: stop["requested"], deadline_utc=deadline,
            reserve_seconds=args.reserve_seconds, initial_prompt_bound_seconds=args.initial_prompt_bound_seconds)
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        gc.collect()


def _plan(args, prepared, deadline):
    directories = {"ouro": args.ouro_evaluation_dir, "controls": args.controls_evaluation_dir}
    missing = [name for name, path in directories.items()
               if path is None or not (path / "COMPLETE.json").is_file()]
    prerequisites = ({"status": "missing", "missing": missing} if missing
                     else {"status": "verified", "records": _evaluation_gate(args, prepared)})
    return {
        "status": "plan", "model_loaded": False, "gpu_initialized": False,
        "run_spec_sha256": prepared["main"]["spec_sha256"],
        "combined_contract_sha256": prepared["combined_sha256"],
        "huginn_calibration_sha256": prepared["calibration_sha256"],
        "model": {"repo_id": REPO_ID, "revision": REVISION,
                  "manifest_sha256": prepared["manifest_sha256"],
                  "snapshot_bytes_to_verify_before_load": prepared["manifest"]["total_bytes"]},
        "settings": SETTINGS, "calibration_fit_id": 1, "n_prompts": 100,
        "initialization_base_seed": prepared["base_seed"], "seed_namespace": SEED_NAMESPACE,
        "token_length_range": [min(row["token_length"] for row in prepared["calibration"]["rows"]),
                               max(row["token_length"] for row in prepared["calibration"]["rows"])],
        "evaluation_prerequisites": prerequisites,
        "stop_at_utc": deadline.isoformat() if deadline else None,
        "snapshot": str(args.snapshot.absolute()), "output_dir": str(args.output_dir.absolute()),
    }


def _self_test():
    """Exercise seed guards and real checkpoint/resume using CPU fixtures."""
    import copy
    from datetime import timedelta
    import subprocess
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import patch

    # -S makes third-party packages unavailable. Test the public plan dispatch
    # with frozen-input loading substituted by a small declared fixture.
    script = """
import importlib.util,json,pathlib,sys,types
path=pathlib.Path(sys.argv[1])
sys.path.insert(0,str(path.parent))
spec=importlib.util.spec_from_file_location('huginn_plan_fixture',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
prepared={'main':{'spec_sha256':'cpu-fixture'},'combined_sha256':'cpu-fixture',
 'calibration_sha256':'cpu-fixture','manifest_sha256':'cpu-fixture',
 'manifest':{'total_bytes':1},'base_seed':2026090802,
 'calibration':{'rows':[{'token_length':20} for _ in range(100)]}}
module.contract=lambda args:prepared
result=module.main(['--plan','--snapshot','missing-snapshot','--ouro-src','missing-source',
 '--run-spec','missing-spec','--combined-contract','missing-combined',
 '--calibration','missing-calibration','--output-dir',sys.argv[2]])
assert result==0 and 'torch' not in sys.modules
assert not pathlib.Path(sys.argv[2]).exists()
"""
    with tempfile.TemporaryDirectory(prefix="jlens-huginn-plan-test-") as directory:
        result = subprocess.run([sys.executable, "-S", "-c", script, str(Path(__file__).resolve()),
                                 str(Path(directory) / "must_not_exist")],
                                text=True, capture_output=True, check=True)
        plan = json.loads(result.stdout)
        assert plan["evaluation_prerequisites"] == {"status": "missing", "missing": ["ouro", "controls"]}
        assert not plan["model_loaded"] and not plan["gpu_initialized"]

    import torch
    torch.set_num_threads(1)
    prompts = [f"CPU Huginn calibration paragraph {index:03d}" + " padding" * (index % 3)
               for index in range(100)]
    base = 2026090802

    class FixtureModel:
        d_model = 2
        dtype = torch.float32
        input_device = torch.device("cpu")
        seed_namespace = SEED_NAMESPACE
        state_seed = 0
        last_initialization = None

        def encode(self, prompt, *, max_length):
            return torch.tensor([[65504, *list(prompt.encode("utf-8"))[:max_length - 1]]], dtype=torch.long)

        def initialization_metadata(self, ids):
            tokens = ids[0].tolist()
            digest = hashlib.sha256(runner._canonical({
                "version": 1, "base_seed": self.state_seed, "namespace": self.seed_namespace,
                "input_ids": tokens,
            })).digest()
            return {
                "base_seed": self.state_seed, "namespace": self.seed_namespace,
                "input_ids_sha256": runner._digest(tokens),
                "prompt_state_seed": int.from_bytes(digest[:8], "big") % (2 ** 63),
                "seed_recipe": SEED_RECIPE, "one_prompt_shape": [1, len(tokens), 2],
                "generator_device": "cpu", "dtype": "torch.float32", "test_time_noise": 0,
                "native_draw": "CPU orchestration fixture; no native model or state draw",
                "lane_coupling": "CPU orchestration fixture",
            }

    model = FixtureModel()
    rows = []
    for index, prompt in enumerate(prompts):
        model.state_seed = base + index
        ids = model.encode(prompt, max_length=128)
        metadata = model.initialization_metadata(ids)
        rows.append({"index": index, "token_length": int(ids.shape[1]),
                     "n_valid": int(ids.shape[1]) - 17, "base_seed": base + index,
                     "input_ids_sha256": metadata["input_ids_sha256"],
                     "prompt_state_seed": metadata["prompt_state_seed"]})
    fit = {"fit_id": 1, "seed": 2026090701, "n_prompts": 100,
           "sha256": "f" * 64, "prompts": prompts, "prompt_path": "/cpu-fixture/fit01.json",
           "token_lengths": [row["token_length"] for row in rows],
           "n_valid": [row["n_valid"] for row in rows]}
    combined = {"schema_version": 1, "budget_usd": 25, "huginn": {
        "repo_id": REPO_ID, "revision": REVISION, "n_prompts": 100,
        "calibration_fit_id": 1, "calibration_sha256": fit["sha256"], "n_recurrences": 8,
        "source_layers": list(range(32)), "target_layer": 33, "d_model": 5280,
        "dim_batch": 8, "max_seq_len": 128, "skip_first": 16, "initialization_base_seed": base,
    }}
    calibration = {"schema_version": 1, "n_prompts": 100, "calibration_fit_id": 1,
                   "calibration_sha256": fit["sha256"], "model_manifest_sha256": "a" * 64,
                   "tokenizer_bos_id": 65504, "max_seq_len": 128, "skip_first": 16,
                   "seed_namespace": SEED_NAMESPACE, "rows": rows}
    assert _calibration(calibration, combined, fit, "a" * 64) == base
    checked = []

    def rejects(name, function, *args, **kwargs):
        try:
            function(*args, **kwargs)
        except (ValueError, RuntimeError):
            checked.append({"case": name, "passed": True})
        else:
            raise AssertionError("invalid fixture was accepted: " + name)

    rejects("reject_absent_evaluation_directories", _evaluation_gate,
            SimpleNamespace(ouro_evaluation_dir=None, controls_evaluation_dir=None), {})
    rejects("reject_absent_semantic_main_validator", _main_evaluation,
            Path("unused"), SimpleNamespace(), {})

    def incomplete_main(root, main_contract, *, require_all):
        assert require_all is True
        raise ValueError("incomplete main evaluation fixture")

    with patch(__name__ + "._evaluation_modules", return_value=(None, SimpleNamespace(
        validate_completed_main=incomplete_main,
    ))):
        rejects("require_all_five_main_evaluations", _evaluation_gate,
                SimpleNamespace(ouro_evaluation_dir=Path("unused"), controls_evaluation_dir=Path("unused")),
                {"main": {}})

    for name, value in (("index", 1), ("token_length", 129), ("n_valid", 1),
                        ("base_seed", base + 1), ("prompt_state_seed", 2 ** 63),
                        ("input_ids_sha256", "invalid")):
        changed = copy.deepcopy(calibration)
        changed["rows"][0][name] = value
        rejects("reject_calibration_" + name, _calibration, changed, combined, fit, "a" * 64)
    changed = copy.deepcopy(calibration)
    changed["rows"] = changed["rows"][:-1]
    rejects("reject_N99_calibration", _calibration, changed, combined, fit, "a" * 64)
    changed = copy.deepcopy(combined)
    changed["huginn"]["n_recurrences"] = 4
    rejects("reject_recurrence_change", _calibration, calibration, changed, fit, "a" * 64)
    changed = copy.deepcopy(calibration)
    changed["seed_namespace"] = "evaluation"
    rejects("reject_calibration_evaluation_namespace", _calibration, changed, combined, fit, "a" * 64)
    for index in (99, 0, 37):
        metadata = _paragraph_initialization(model, prompts[index], rows[index])
        assert model.state_seed == base + index and metadata["prompt_state_seed"] == rows[index]["prompt_state_seed"]
    rejects("reject_actual_token_change", _paragraph_initialization, model, prompts[0] + " changed", rows[0])
    wrong = {**rows[0], "prompt_state_seed": rows[0]["prompt_state_seed"] + 1}
    rejects("reject_frozen_seed_change", _paragraph_initialization, model, prompts[0], wrong)
    model.seed_namespace = "evaluation"
    rejects("reject_model_namespace_change", _paragraph_initialization, model, prompts[0], rows[0])
    model.seed_namespace = SEED_NAMESPACE
    actual_metadata = model.initialization_metadata
    with patch.object(model, "initialization_metadata", side_effect=lambda ids: {
        **actual_metadata(ids), "prompt_state_seed": 1,
    }):
        rejects("reject_actual_seed_metadata_change", _paragraph_initialization, model, prompts[0], rows[0])

    calls = []
    matrix = torch.tensor([[1.0, -2.0], [0.5, 4.0]])

    class FixtureEstimator:
        wrong_initialization = False
        wrong_arms = False

        def jacobians_for_prompt(self, actual_model, prompt, sources, **options):
            assert actual_model is model and sources == list(range(32))
            expected = {"target_layer": 33, "dim_batch": 8, "max_seq_len": 128, "skip_first": 16,
                        "mode": "dense", "dense_engine": "cuda_graph", "compress_saved_tensors": True}
            assert {key: value for key, value in options.items() if key != "diagnostics"} == expected
            index = prompts.index(prompt)
            assert model.state_seed == base + index
            calls.append(index)
            model.last_initialization = model.initialization_metadata(model.encode(prompt, max_length=128))
            if self.wrong_initialization:
                model.last_initialization["prompt_state_seed"] += 1
            bank = {source: (index + 1) * matrix + 0.25 * source for source in sources}
            return {"wrong" if self.wrong_arms else "dense": bank}, rows[index]["token_length"], rows[index]["n_valid"]

    estimator = FixtureEstimator()
    compute = _compute(model, estimator, calibration)
    estimator.wrong_initialization = True
    rejects("reject_forward_initialization_mismatch", compute, prompts[0], 0, {})
    estimator.wrong_initialization = False
    estimator.wrong_arms = True
    rejects("reject_wrong_estimator_arm", compute, prompts[0], 0, {})
    estimator.wrong_arms = False
    calls.clear()
    prepared = {"fit": fit, "calibration": calibration}
    runtime = {"fixture": "CPU N100 driver seed/resume test; no native model or GPU"}
    real_run_fit = runner.run_fit
    stop_after = 37
    forwarded_options = []

    def fixture_run_fit(**options):
        assert options["d_model"] == 5280 and tuple(options["source_layers"]) == tuple(range(32))
        assert options["fit_id"] == 1 and options["production_profile"] == "huginn_r8"
        assert options.get("cpu_test", False) is False
        forwarded_options.append({key: options[key] for key in ("d_model", "fit_id", "production_profile")})
        # The production call above is checked first. Only this explicit CPU
        # fixture substitutes 2x2 matrices and labels every saved owner cpu_test.
        options.update(d_model=2, cpu_test=True, stop_requested=lambda: len(calls) >= stop_after)
        return real_run_fit(**options)

    with tempfile.TemporaryDirectory(prefix="jlens-huginn-runner-test-") as directory:
        args = SimpleNamespace(output_dir=Path(directory) / "cpu_execution",
                               reserve_seconds=0.0, initial_prompt_bound_seconds=0.0)
        deadline = runner._utc_now() + timedelta(days=1)
        with (patch(__name__ + "._load_runtime", return_value=(model, estimator, runtime)),
              patch.object(runner, "run_fit", fixture_run_fit)):
            first = execute(args, prepared, {"fixture": True}, deadline)
            assert first["status"] == "stopped" and first["n_done"] == 37
            assert calls == list(range(37))
            stop_after = 1000
            completed = execute(args, prepared, {"fixture": True}, deadline)
            assert completed["status"] == "complete" and calls == list(range(100))
            noop = execute(args, prepared, {"fixture": True}, deadline)
            assert noop["completed_noop"] and calls == list(range(100))
        saved = torch.load(completed["lens_path"], map_location="cpu", weights_only=True)
        assert set(saved) == {"J", "n_prompts", "source_layers", "d_model"}
        for source in range(32):
            assert torch.equal(saved["J"][source], (50.5 * matrix + 0.25 * source).half())
        owner = runner._json(args.output_dir / "fit_01/OWNER.json")
        assert owner["identity"]["cpu_test"] is True and owner["identity"]["production_profile"] == "huginn_r8"
        checkpoint = runner._read_checkpoint(torch, args.output_dir / "fit_01", owner["identity"])
        for index, diagnostic in enumerate(checkpoint["diagnostics"]):
            initialization = diagnostic["initialization"]
            assert initialization["base_seed"] == base + index
            assert initialization["prompt_state_seed"] == rows[index]["prompt_state_seed"]
            assert initialization["input_ids_sha256"] == rows[index]["input_ids_sha256"]
        with patch(__name__ + "._load_runtime", side_effect=AssertionError("past deadline loaded a model")):
            stopped = execute(args, prepared, {}, runner._utc_now() - timedelta(seconds=1))
        assert stopped["status"] == "stopped" and stopped["model_loaded"] is False
    return {
        "status": "passed", "scope": "CPU 2x2 synthetic estimator and real N100 checkpoint runner only; no weights, native derivative or GPU validation",
        "plan_works_without_site_packages": True, "plan_creates_no_output": True,
        "plan_reports_absent_evaluation_prerequisites": True,
        "guard_cases": checked, "guard_case_count": len(checked),
        "N100_stop37_resume_noop": True, "all_100_prompt_seeds_preserved": True,
        "all_32_final_means_exact": True, "production_geometry_passed_to_runner": forwarded_options,
        "test_artifacts_labeled_cpu_test": True, "past_deadline_never_loads_model": True,
        "source_sha256": {str(path.relative_to(REPO)): runner._file_hash(path)
                          for path in (Path(__file__).resolve(), Path(runner.__file__).resolve())},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "ouro-src", "run-spec", "combined-contract", "calibration", "output-dir",
                 "ouro-evaluation-dir", "controls-evaluation-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-output", type=Path)
    parser.add_argument("--stop-at-utc")
    parser.add_argument("--reserve-seconds", type=float, default=600.0)
    parser.add_argument("--initial-prompt-bound-seconds", type=float, default=1200.0)
    args = parser.parse_args(argv)
    if args.self_test:
        if args.plan:
            parser.error("--self-test and --plan are separate CPU entry points")
        rendered = json.dumps(_self_test(), indent=2, allow_nan=False) + "\n"
        if args.self_test_output is not None:
            args.self_test_output.write_text(rendered)
        print(rendered, end="")
        return 0
    for name in ("snapshot", "ouro_src", "run_spec", "combined_contract", "calibration", "output_dir"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    deadline = runner._deadline(args.stop_at_utc)
    if not args.plan and deadline is None:
        parser.error("execution requires --stop-at-utc")
    if not args.plan and (args.ouro_evaluation_dir is None or args.controls_evaluation_dir is None):
        parser.error("execution requires --ouro-evaluation-dir and --controls-evaluation-dir")
    for name in ("reserve_seconds", "initial_prompt_bound_seconds"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    prepared = contract(args)
    if args.plan:
        print(json.dumps(_plan(args, prepared, deadline), indent=2, allow_nan=False))
        return 0
    prerequisites = _evaluation_gate(args, prepared)
    with runner._output_lock(args.output_dir):
        result = execute(args, prepared, prerequisites, deadline)
    print(json.dumps({"profile": "huginn_r8", **result}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
