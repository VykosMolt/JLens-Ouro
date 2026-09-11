"""Evaluate the two frozen Ouro controls against main fit01 and its raw lens.

All five main evaluations must already be complete. This reader validates the
new profile OWNER/LATEST/COMPLETE records, reuses the sealed main activation and
native-logit cache, and exports only learned control columns. Target comparison
uses 0..189; position comparison uses 0..190. There is one conditional calibration
draw and no fit-uncertainty estimate here. No cloud action is performed.

Import, completion validation, and --help use only the standard library.
--self-test exercises CPU fixtures; production inference requires explicit paths.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import time

import evaluate_refits as common
import run_controls
import run_refits as runner


SCHEMA = "ouro_controls_evaluation.v1"
PROFILES = ["ouro_penultimate", "ouro_positions"]
SECTIONS = ["penultimate", "positions/sampled_sum", "positions/diagonal", "comparisons"]
TARGET_SUPPORT = list(range(190))
POSITION_SUPPORT = list(range(191))
TARGET_LATE = list(range(176, 190))
POSITION_LATE = list(range(176, 191))
_MAIN_RUNTIME_COMPARISON = (
    "run_spec_sha256", "model", "snapshot_files", "source_files", "environment",
    "gpu", "precision", "attention_implementation",
)


def _combined_sha(main_contract):
    key = ("jlens", "research/refit_round_2026-09-07/deployment/combined_contract.json")
    records = {(row["root"], row["path"]): row["sha256"] for row in main_contract["sources"]}
    if key not in records or key not in main_contract["source_paths"]:
        raise ValueError("run specification does not bind the combined contract")
    path = runner._no_links(main_contract["source_paths"][key])
    if runner._file_hash(path) != records[key]:
        raise ValueError("combined contract content differs from its frozen source record")
    return records[key], path


def _bound_combined(main_contract, combined):
    digest, path = _combined_sha(main_contract)
    if not common._same(runner._json(path), combined):
        raise ValueError("provided combined contract differs from the hashed document")
    controls = combined.get("controls", {})
    if (controls.get("common_target_support") != TARGET_SUPPORT
            or controls.get("common_position_support") != POSITION_SUPPORT
            or controls.get("loop4_late_band_target_contrast") != TARGET_LATE
            or controls.get("loop4_late_band_position_contrast") != POSITION_LATE
            or controls.get("historical_loop4_final_third") != list(range(176, 192))):
        raise ValueError("paired source support or the frozen late band changed")
    return digest


def _record_owner(record):
    if (not isinstance(record, dict) or not isinstance(record.get("identity"), dict)
            or record.get("fit_identity_sha256") != runner._digest(record["identity"])
            or record.get("fit_id") != record["identity"].get("fit_id")
            or record.get("n_prompts") != 100):
        raise ValueError("embedded fit identity is incomplete or inconsistent")
    owner = {"schema_version": 1, "fit_identity_sha256": record["fit_identity_sha256"],
             "identity": record["identity"]}
    owner_bytes = runner._canonical(owner) + b"\n"
    if record.get("owner") != {"bytes": len(owner_bytes), "sha256": hashlib.sha256(owner_bytes).hexdigest()}:
        raise ValueError("embedded fit identity differs from the producer OWNER byte binding")
    for key in ("checkpoint_pointer", "complete_pointer"):
        pointer = record.get(key)
        if (not isinstance(pointer, dict) or pointer.get("n_done") != 100
                or pointer.get("next_idx") != 100
                or pointer.get("fit_identity_sha256") != owner["fit_identity_sha256"]):
            raise ValueError("embedded fit does not reference complete N100 generations")
    return owner


def _validate_evaluation_runtime(runtime, main_contract):
    if not isinstance(runtime, dict):
        raise ValueError("evaluation lacks its actual runtime record")
    snapshot = {row["path"]: {key: row[key] for key in ("bytes", "sha256")}
                for row in main_contract["manifest"]["files"]}
    if (not common._same(runtime.get("model"), main_contract["spec"]["model"])
            or not common._same(runtime.get("snapshot_files"), snapshot)
            or runtime.get("attention_implementation") != "sdpa"
            or not isinstance(runtime.get("gpu"), dict) or not runtime["gpu"].get("name")
            or not isinstance(runtime.get("precision"), dict) or not runtime["precision"]):
        raise ValueError("evaluation model, GPU, precision, or attention identity changed")
    environment = runtime.get("environment", {})
    for key in ("packages", "torch_git", "python_version", "cuda_build_version", "cudnn_runtime_integer"):
        if not common._same(environment.get(key), main_contract["environment"].get(key)):
            raise ValueError(f"evaluation runtime lacks the pinned {key}")
    sources = {(row["root"], row["path"]): row["sha256"] for row in main_contract["sources"]}
    imported = runtime.get("imported_sources")
    expected_modules = {
        "ouro_jlens", "ouro_jlens.evaluate", "ouro_jlens.evaldata", "ouro_jlens.fit_lens",
        "ouro_jlens.evidence", "ouro_jlens.recurrent", "jlens", "jlens._logging",
        "jlens.fitting", "jlens.hooks", "jlens.hf", "jlens.lens", "jlens.protocol", "jlens.vis",
    }
    if (not isinstance(imported, list) or len(imported) != len(expected_modules)
            or {row.get("module") for row in imported if isinstance(row, dict)} != expected_modules):
        raise ValueError("evaluation imported-source membership changed")
    for row in imported:
        if (set(row) != {"module", "root", "path", "sha256"}
                or row["sha256"] != sources.get((row["root"], row["path"]))):
            raise ValueError("evaluation imported an unmanifested source hash")
    remote = runtime.get("remote_implementations")
    if (not isinstance(remote, list) or len(remote) != 2
            or {row.get("sha256") for row in remote if isinstance(row, dict)}
            != {snapshot[name]["sha256"] for name in ("modeling_ouro.py", "configuration_ouro.py")}):
        raise ValueError("evaluation remote implementations changed")


def validate_completed_main(root, main_contract, *, require_all=True):
    """Semantically validate the sealed main output without loading tensors."""
    checked = common.validate_output(root)
    identity = checked["owner"]["identity"]
    fit_ids = identity.get("fit_ids")
    expected = {
        "kind": "ouro_main_n100_readout", "run_spec_sha256": main_contract["spec_sha256"],
        "source_files": main_contract["sources"], "model": main_contract["spec"]["model"],
        "tasks": list(common.TASKS), "position": -1, "greedy_steps": 4,
        "target_layer": 191, "learned_sources": list(range(191)),
        "virtual_indices": list(range(192)), "known_identity_indices": [191],
    }
    if any(not common._same(identity.get(key), value) for key, value in expected.items()):
        raise ValueError("main evaluation does not match the frozen source/model/readout contract")
    if (not isinstance(fit_ids, list) or fit_ids != sorted(set(fit_ids)) or 1 not in fit_ids
            or (require_all and fit_ids != [1, 2, 3, 4, 5])
            or identity.get("sections") != ["common", *(f"fits/fit_{fit_id:02d}" for fit_id in fit_ids)]):
        raise ValueError("all five main fit evaluations must complete before controls")
    records = identity.get("fits")
    if (not isinstance(records, list) or len(records) != len(fit_ids)
            or [record.get("fit_id") for record in records] != fit_ids):
        raise ValueError("main evaluation fit membership is inconsistent")
    common_reference = checked["complete"]["sections"][0]
    for record in records:
        owner = _record_owner(record)
        common._validate_owner_binding(owner, main_contract)
        metadata = runner._json(Path(root) / f"fits/fit_{record['fit_id']:02d}/metadata.json")
        if (metadata.get("kind") != "independent_fit_readout" or metadata.get("fit") != record
                or metadata.get("common") != common_reference or metadata.get("n_prompts") != 100
                or metadata.get("readout") != {"target_layer": 191, "virtual_indices": list(range(192)),
                                               "learned_sources": list(range(191)), "identity_indices": [191]}):
            raise ValueError("main fit section is not paired to the declared common cache")
    metadata = runner._json(Path(root) / "common/metadata.json")
    population = metadata.get("population", {})
    for key, value in {
        "items": 148, "tasks": common.TASK_COUNTS, "population_sha256": common.POPULATION_SHA256,
        "task_names_sha256": common.TASK_NAMES_SHA256, "token_forms_sha256": common.TOKEN_FORMS_SHA256,
        "eligible_items": {"multihop": 90, "order-ops": 51},
        "eligible_slots": {"multihop": 100, "order-ops": 51},
    }.items():
        if not common._same(population.get(key), value):
            raise ValueError(f"main evaluation population changed: {key}")
    if (metadata.get("kind") != "common_readout" or metadata.get("position") != -1
            or metadata.get("virtual_indices") != list(range(192))
            or metadata.get("native_exit_indices") != [47, 95, 143, 191]
            or metadata.get("target_state_indices") != [190, 191]):
        raise ValueError("main cache lacks the complete native/target-state geometry")
    _validate_evaluation_runtime(metadata.get("runtime"), main_contract)
    return checked


def validate_control_owner(owner, main_contract, combined, profile_name, *, main_checked):
    """Reconstruct one exact production control identity using logical bindings."""
    if profile_name not in PROFILES:
        raise ValueError("unknown Ouro control profile")
    combined_sha = _bound_combined(main_contract, combined)
    identity = owner.get("identity")
    if (not isinstance(identity, dict) or owner.get("schema_version") != 1
            or owner.get("fit_identity_sha256") != runner._digest(identity)):
        raise ValueError("invalid control owner digest")
    runtime = identity.get("runtime")
    additions = {"combined_contract_sha256", "production_profile", "control_contract"}
    main_keys = set(main_checked["owner"]["identity"]["fits"][0]["identity"]["runtime"])
    if not isinstance(runtime, dict) or set(runtime) != main_keys | additions:
        raise ValueError("control runtime membership differs from its production schema")
    profile = runner.PROFILES[profile_name]
    fit = next(fit for fit in main_contract["fits"] if fit["fit_id"] == 1)
    expected = runner._fit_identity(fit["prompts"], fit["token_lengths"], fit["n_valid"], 1,
                                   runtime, 2048, profile["source_layers"], False,
                                   production_profile=profile_name)
    if not common._same(identity, expected):
        raise ValueError("control is not the complete preselected N100 profile")
    settings = {**runner.SETTINGS, "source_layers": profile["source_layers"],
                "target_layer": profile["target_layer"],
                "mode": "paired_sampled" if profile["bank_arms"] else "dense"}
    for key, value in {
        "combined_contract_sha256": combined_sha, "production_profile": profile_name,
        "control_contract": combined["controls"], "settings": settings,
    }.items():
        if not common._same(runtime.get(key), value):
            raise ValueError(f"control runtime differs from the frozen {key}")
    # Reuse the strict main source/model/environment/calibration validator after
    # removing only the explicitly validated profile extension. Old producer
    # paths remain descriptive and are never opened.
    base = {key: value for key, value in runtime.items() if key not in additions}
    base["settings"] = runner.SETTINGS
    base_identity = runner._fit_identity(fit["prompts"], fit["token_lengths"], fit["n_valid"], 1,
                                        base, 2048, tuple(range(191)), False)
    common._validate_owner_binding({"schema_version": 1, "identity": base_identity,
                                    "fit_identity_sha256": runner._digest(base_identity)}, main_contract)
    main_runtime = main_checked["owner"]["identity"]["fits"][0]["identity"]["runtime"]
    if any(not common._same(runtime[key], main_runtime[key]) for key in _MAIN_RUNTIME_COMPARISON):
        raise ValueError("control and main fit01 have different fitting runtime bindings")
    return fit


def validate_completed_controls(root, main_contract, combined, main_checked):
    """Read-only standard-library completion gate for the subsequent Huginn run."""
    checked = common.validate_output(root, schema=SCHEMA)
    identity = checked["owner"]["identity"]
    expected = {
        "kind": "ouro_controls_n100_readout", "run_spec_sha256": main_contract["spec_sha256"],
        "combined_contract_sha256": _bound_combined(main_contract, combined), "fit_ids": [1],
        "control_profiles": PROFILES, "sections": SECTIONS,
        "main_evaluation_owner_sha256": runner._digest(main_checked["owner"]),
        "main_evaluation_complete_sha256": runner._digest(main_checked["complete"]),
        "source_files": main_contract["sources"], "model": main_contract["spec"]["model"],
        "tasks": list(common.TASKS), "position": -1,
        "common_target_support": TARGET_SUPPORT, "common_position_support": POSITION_SUPPORT,
        "target_contrast_late_band": TARGET_LATE,
        "position_contrast_late_band": POSITION_LATE,
    }
    if any(not common._same(identity.get(key), value) for key, value in expected.items()):
        raise ValueError("completed controls do not match the declared main/paired contract")
    records = identity.get("control_fits")
    if not isinstance(records, dict) or set(records) != set(PROFILES):
        raise ValueError("completed controls must bind both fitted profiles")
    for profile, record in records.items():
        validate_control_owner(_record_owner(record), main_contract, combined, profile, main_checked=main_checked)
        if profile == "ouro_positions" and record.get("frozen_q_sha256") != runner._digest(combined["controls"]["positions"]["q"]):
            raise ValueError("position fit completion lacks its validated shared q ledger")
    common_reference = main_checked["complete"]["sections"][0]
    runtime = None
    for section, profile, arm, support, target in (
        ("penultimate", "ouro_penultimate", "dense", TARGET_SUPPORT, 190),
        ("positions/sampled_sum", "ouro_positions", "sampled_sum", POSITION_SUPPORT, 191),
        ("positions/diagonal", "ouro_positions", "diagonal", POSITION_SUPPORT, 191),
    ):
        metadata = runner._json(Path(root) / section / "metadata.json")
        expected = {"kind": "control_readout", "profile": profile, "arm": arm, "control_fit": records[profile],
                    "common": common_reference, "position": -1,
                    "readout": {"target_layer": target, "virtual_indices": support,
                                "learned_sources": support, "identity_indices": []}}
        if any(not common._same(metadata.get(key), value) for key, value in expected.items()):
            raise ValueError(f"control section metadata changed: {section}")
        if not isinstance(metadata.get("evaluation_runtime"), dict):
            raise ValueError("control section lacks its actual inference runtime")
        if runtime is not None and not common._same(metadata["evaluation_runtime"], runtime):
            raise ValueError("control readouts did not share one inference runtime")
        runtime = metadata["evaluation_runtime"]
        _validate_evaluation_runtime(runtime, main_contract)
        if runtime.get("shared_target_state_parity") != {
            "190": "bitwise_equal_to_sealed_main_target_state",
            "191": "bitwise_equal_to_sealed_main_target_state",
        }:
            raise ValueError("control readout does not reproduce the sealed native target states")
    metadata = runner._json(Path(root) / "comparisons/metadata.json")
    summary = runner._json(Path(root) / "comparisons/summary.json")
    for document in (metadata, summary):
        if (document.get("conditional_on_fit_id") != 1 or document.get("n_independent_control_fits") != 1
                or document.get("target_support") != TARGET_SUPPORT
                or document.get("position_support") != POSITION_SUPPORT
                or document.get("target_late_band") != TARGET_LATE
                or document.get("position_late_band") != POSITION_LATE):
            raise ValueError("paired comparison support or conditional-fit interpretation changed")
    if metadata.get("common") != common_reference:
        raise ValueError("paired comparisons are not bound to the shared main cache")
    return checked


def validate_inputs(args):
    main, combined = run_controls.contract(args)
    for name in ("evaluate_controls.py", "evaluate_refits.py"):
        run_controls._source(Path(__file__).with_name(name), name, main)
    common._evaluation_sources(main)
    _bound_combined(main, combined)
    main_checked = validate_completed_main(args.main_evaluation_dir, main)
    entries = []
    import torch
    for profile, path in zip(PROFILES, (args.penultimate_fit_dir, args.positions_fit_dir)):
        fit_dir, owner = common._owner(path)
        validate_control_owner(owner, main, combined, profile, main_checked=main_checked)
        lens_path, record = common._read_generation(torch, fit_dir, owner)
        if profile == "ouro_positions":
            checkpoint = runner._read_checkpoint(torch, fit_dir, owner["identity"])
            if [row.get("frozen_q") for row in checkpoint["diagnostics"]] != combined["controls"]["positions"]["q"]:
                raise ValueError("position checkpoint did not record the frozen q for every paragraph")
            record["frozen_q_sha256"] = runner._digest(combined["controls"]["positions"]["q"])
            del checkpoint
        entries.append({"profile": profile, "fit_dir": fit_dir, "owner": owner,
                        "lens_path": lens_path, "record": record})
        gc.collect()
    snapshot_files = runner._snapshot(args.snapshot, main["manifest"])
    return main, combined, main_checked, snapshot_files, entries


def score_rank_arrays(allrank, rows, names, virtual_indices, *, bands):
    """Historical own/control recovery with explicit real columns and masks.

    Each control takes its own any-layer maximum before control averaging.
    Ineligible items have zero-valued storage and an explicit false mask.
    """
    import numpy as np
    if (not virtual_indices or len(set(virtual_indices)) != len(virtual_indices)
            or any(type(index) is not int for index in virtual_indices)
            or allrank.shape != (len(rows), 128, len(virtual_indices))
            or not np.issubdtype(allrank.dtype, np.integer)):
        raise ValueError("rank bank must have its declared item/name/virtual support")
    columns = {}
    for name, support in bands.items():
        if not support or len(set(support)) != len(support) or not set(support) <= set(virtual_indices):
            raise ValueError("score bands must be unique subsets of learned columns")
        columns[name] = [virtual_indices.index(index) for index in support]
    size = (len(rows), len(virtual_indices))
    result = {"eligible": np.zeros(len(rows), dtype=np.bool_),
              "own_layer": np.zeros(size, dtype=np.float64), "control_layer": np.zeros(size, dtype=np.float64),
              "own_regions": np.zeros((len(rows), len(bands)), dtype=np.float64),
              "control_regions": np.zeros((len(rows), len(bands)), dtype=np.float64)}
    for item_index, row in enumerate(rows):
        count = len(names[row["task"]])
        if np.any(allrank[item_index, :count] < 0) or np.any(allrank[item_index, count:] != -1):
            raise ValueError("real task names cannot contain unsupported rank columns")
        slots = [slot for slot, eligible in enumerate(row["eligible"]) if eligible]
        if not slots:
            continue
        result["eligible"][item_index] = True
        for slot in slots:
            own = row["own_index"][slot]
            controls = row["control_indices"][slot]
            if (type(own) is not int or not 0 <= own < count or not controls
                    or len(set(controls)) != len(controls)
                    or any(type(index) is not int or not 0 <= index < count
                           or index in row["own_index"] for index in controls)):
                raise ValueError("eligible slot has an invalid own/control catalogue")
            own_rank = allrank[item_index, own]
            control_rank = allrank[item_index, controls]
            own_hits = (own_rank >= 0) & (own_rank < 10)
            control_hits = (control_rank >= 0) & (control_rank < 10)
            result["own_layer"][item_index] += own_hits / len(slots)
            result["control_layer"][item_index] += control_hits.mean(axis=0) / len(slots)
            for region_index, selected in enumerate(columns.values()):
                result["own_regions"][item_index, region_index] += own_hits[selected].any() / len(slots)
                result["control_regions"][item_index, region_index] += control_hits[:, selected].any(axis=1).mean() / len(slots)
    result["delta_layer"] = result["own_layer"] - result["control_layer"]
    result["delta_regions"] = result["own_regions"] - result["control_regions"]
    return result


def _load_arrays(path, prefix):
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key[len(prefix):]: archive[key].copy() for key in archive.files if key.startswith(prefix)}
    if not arrays:
        raise ValueError(f"no readout arrays with prefix {prefix!r}")
    return arrays


def _control_readout(evaluator, model, items, states, lens, target, exit_logits, target_logits):
    """Original rank math; explicit extra diagnostic reference is the fit target."""
    import torch
    support = list(range(target))
    prepared = common.prepare_readout(model, lens, target_layer=target, source_layers=support, evaluator=evaluator)
    plan = common.PreparedReadout(prepared.jacobians[:target], target, support, support, [])
    view = common._ReadoutView(model, target)
    references = torch.cat([exit_logits, target_logits.to(model.input_device)[:, None]], dim=1)
    with torch.no_grad():
        arrays, kept = evaluator.readout_arrays(view, items, states[:, :target], plan.jacobians,
                                                references, model.n_ut)
    del kept, references
    arrays["kl_to_target_state"] = arrays.pop("kl_to_local")
    arrays["rank_of_target_state_top1"] = arrays.pop("rank_of_local_top1")
    return arrays, plan.metadata()


def _comparisons(main_arrays, raw_arrays, control_arrays, rows, names):
    import numpy as np
    banks = {}
    for comparison, support, late, arms in (
        ("target", TARGET_SUPPORT, TARGET_LATE, {"main": (main_arrays, list(range(192))),
                                                "raw": (raw_arrays, list(range(192))),
                                                "penultimate": (control_arrays["penultimate"], TARGET_SUPPORT)}),
        ("position", POSITION_SUPPORT, POSITION_LATE, {"main": (main_arrays, list(range(192))),
                                                      "raw": (raw_arrays, list(range(192))),
                                                      "sampled_sum": (control_arrays["sampled_sum"], POSITION_SUPPORT),
                                                      "diagonal": (control_arrays["diagonal"], POSITION_SUPPORT)}),
    ):
        for arm, (arrays, existing) in arms.items():
            selected = common.slice_readout_arrays(arrays, virtual_indices=existing, keep=support)
            banks[f"{comparison}_{arm}"] = score_rank_arrays(selected["allrank"], rows, names, support,
                                                              bands={"all_common": support, "loop4_late": late})
    pairs = {
        "target_main_minus_penultimate": ("target_main", "target_penultimate"),
        "target_main_minus_raw": ("target_main", "target_raw"),
        "target_penultimate_minus_raw": ("target_penultimate", "target_raw"),
        "position_sampled_sum_minus_diagonal": ("position_sampled_sum", "position_diagonal"),
        "position_main_minus_sampled_sum": ("position_main", "position_sampled_sum"),
        "position_main_minus_raw": ("position_main", "position_raw"),
        "position_sampled_sum_minus_raw": ("position_sampled_sum", "position_raw"),
        "position_diagonal_minus_raw": ("position_diagonal", "position_raw"),
    }
    arrays = {f"{arm}_{field}": value for arm, bank in banks.items() for field, value in bank.items()}
    for name, (left, right) in pairs.items():
        if not np.array_equal(banks[left]["eligible"], banks[right]["eligible"]):
            raise ValueError("paired score populations differ")
        for field in ("own_layer", "control_layer", "delta_layer", "own_regions", "control_regions", "delta_regions"):
            arrays[f"contrast_{name}_{field}"] = banks[left][field] - banks[right][field]
    summary = {
        "conditional_on_fit_id": 1, "n_independent_control_fits": 1,
        "target_support": TARGET_SUPPORT, "position_support": POSITION_SUPPORT, "target_late_band": TARGET_LATE,
        "position_late_band": POSITION_LATE, "regions": ["all_common", "loop4_late"],
        "uncertainty": "descriptive paired item effects conditional on fit01; no fit-uncertainty interval",
        "scores": {}, "contrasts": {},
    }
    for task in common.TASKS:
        summary["scores"][task], summary["contrasts"][task] = {}, {}
        for stratum in ("all", "correct", "incorrect"):
            selected = np.asarray([row["task"] == task and (stratum == "all" or bool(row["correct"]) == (stratum == "correct")) for row in rows])
            task_scores, task_contrasts = {}, {}
            for arm, bank in banks.items():
                mask = selected & bank["eligible"]
                task_scores[arm] = {"eligible_items": int(mask.sum()), **{
                    field: bank[field][mask].mean(axis=0).tolist() if mask.any() else None
                    for field in bank if field != "eligible"}}
            for name, (left, right) in pairs.items():
                mask = selected & banks[left]["eligible"]
                task_contrasts[name] = {"eligible_items": int(mask.sum()), **{
                    field: arrays[f"contrast_{name}_{field}"][mask].mean(axis=0).tolist() if mask.any() else None
                    for field in banks[left] if field != "eligible"}}
            summary["scores"][task][stratum] = task_scores
            summary["contrasts"][task][stratum] = task_contrasts
    return arrays, summary


def _cache(torch, main_root):
    cache = torch.load(main_root / "common/cache.pt", map_location="cpu", weights_only=True, mmap=True)
    required = {"schema", "H", "exit_logits", "target_state_logits", "virtual_indices",
                "native_exit_indices", "position", "population_sha256"}
    if (not isinstance(cache, dict) or set(cache) != required or cache["schema"] != common.SCHEMA
            or cache["virtual_indices"] != list(range(192)) or cache["native_exit_indices"] != [47, 95, 143, 191]
            or cache["position"] != -1 or cache["population_sha256"] != common.POPULATION_SHA256):
        raise ValueError("common cache schema or source population changed")
    states, exits, targets = cache["H"], cache["exit_logits"], cache["target_state_logits"]
    if (not isinstance(states, torch.Tensor) or tuple(states.shape) != (148, 192, 2048)
            or not isinstance(exits, torch.Tensor) or exits.ndim != 3 or tuple(exits.shape[:2]) != (148, 4)
            or not isinstance(targets, dict) or set(targets) != {190, 191}
            or any(not isinstance(value, torch.Tensor) or tuple(value.shape) != (148, exits.shape[2])
                   for value in targets.values())):
        raise ValueError("common activation/native-logit cache geometry changed")
    for value in (states, exits, *targets.values()):
        if value.device.type != "cpu" or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError("common cache must contain finite FP32 CPU tensors")
    if not torch.equal(targets[191].view(torch.int32), exits[:, 3].view(torch.int32)):
        raise ValueError("target191 cache disagrees with native final-exit logits")
    return cache


def _evaluate(args, main, combined, main_checked, snapshot_files, entries):
    import torch
    started = time.monotonic()
    identity = {
        "kind": "ouro_controls_n100_readout", "run_spec_sha256": main["spec_sha256"],
        "combined_contract_sha256": _bound_combined(main, combined), "fit_ids": [1],
        "control_profiles": PROFILES, "sections": SECTIONS,
        "main_evaluation_owner_sha256": runner._digest(main_checked["owner"]),
        "main_evaluation_complete_sha256": runner._digest(main_checked["complete"]),
        "control_fits": {entry["profile"]: entry["record"] for entry in entries},
        "source_files": main["sources"], "model": main["spec"]["model"], "tasks": list(common.TASKS),
        "position": -1, "common_target_support": TARGET_SUPPORT, "common_position_support": POSITION_SUPPORT,
        "target_contrast_late_band": TARGET_LATE, "position_contrast_late_band": POSITION_LATE,
        "created_utc": runner._utc_now().isoformat(),
    }
    out = common._new_output(args.out, identity, schema=SCHEMA)
    main_root = runner._no_links(args.main_evaluation_dir)
    common_reference = main_checked["complete"]["sections"][0]
    with common._runtime_scope(out.parent):
        evaluator, recurrent, imported = common._imports(main, args.ouro_src)
        model, runtime = common._evaluation_runtime(torch, main, snapshot_files, args.snapshot, recurrent, imported)
        stored_rows = runner._json(main_root / "common/items.json")
        stored_names = runner._json(main_root / "common/task_names.json")
        stored_forms = runner._json(main_root / "common/token_forms.json")
        items = evaluator.load_items(model.tokenizer, common.TASKS, encode=lambda text: model.encode(text)[0].tolist())
        with common.task_names(evaluator, items) as registry, torch.no_grad():
            rows = common._item_rows(model, items, registry, evaluator)
            names, forms, population = common._population(rows, registry)
            if names != stored_names or forms != stored_forms or len(stored_rows) != len(rows):
                raise ValueError("control tokenizer/name population differs from the shared main cache")
            for current, stored in zip(rows, stored_rows):
                if (not isinstance(stored, dict) or not common._same(current, {key: stored.get(key) for key in current})
                        or type(stored.get("correct")) is not bool):
                    raise ValueError("control item alignment, eligibility, or matched catalogue differs from main")
            rows = stored_rows
            cache = _cache(torch, main_root)
            states, targets = cache["H"], cache["target_state_logits"]
            exits = cache["exit_logits"].to(model.input_device)
            parity = {}
            for target in (190, 191):
                actual = model.unembed(states[:, target].to(model.input_device)).float().cpu()
                if not torch.equal(actual.view(torch.int32), targets[target].view(torch.int32)):
                    raise ValueError("current unembedding does not reproduce the sealed target-state logits exactly")
                parity[str(target)] = "bitwise_equal_to_sealed_main_target_state"
            runtime["shared_target_state_parity"] = parity
            raw_arrays = _load_arrays(main_root / "common/arrays.npz", "logitlens_")
            main_arrays = _load_arrays(main_root / "fits/fit_01/arrays.npz", "jlens_exit3_")
            common._validate_arrays(raw_arrays, rows, names, list(range(192)))
            common._validate_arrays(main_arrays, rows, names, list(range(192)))
            control_arrays, section_records = {}, []
            for entry in entries:
                profile = entry["profile"]
                runner._verify_record(entry["lens_path"], entry["record"]["lens"])
                state = torch.load(entry["lens_path"], map_location="cpu", weights_only=True, mmap=True)
                arms = ["dense"] if profile == "ouro_penultimate" else ["sampled_sum", "diagonal"]
                target = runner.PROFILES[profile]["target_layer"]
                for arm in arms:
                    print(json.dumps({"stage": "control_readout", "profile": profile, "arm": arm}), flush=True)
                    bank = state["J"] if arm == "dense" else state["J"][arm]
                    lens = importlib.import_module("jlens").JacobianLens(bank, n_prompts=100, d_model=2048)
                    arrays, readout = _control_readout(evaluator, model, items, states, lens, target, exits, targets[target])
                    common._validate_arrays(arrays, rows, names, readout["virtual_indices"])
                    section = "penultimate" if arm == "dense" else f"positions/{arm}"
                    directory = runner._mkdir(out / section)
                    common._write_npz(directory / "arrays.npz", arrays)
                    section_records.append(common._seal_section(out, section, {
                        "kind": "control_readout", "profile": profile, "arm": arm, "control_fit": entry["record"],
                        "common": common_reference, "position": -1, "readout": readout,
                        "evaluation_runtime": runtime,
                        "diagnostics": "final reference is native UT3; target_state reference is the actual fitted target190 or191",
                    }))
                    control_arrays["penultimate" if arm == "dense" else arm] = arrays
                    del lens, bank
                    gc.collect()
                    torch.cuda.empty_cache()
                del state
            scores, summary = _comparisons(main_arrays, raw_arrays, control_arrays, rows, names)
            comparison_dir = runner._mkdir(out / "comparisons")
            common._write_npz(comparison_dir / "scores.npz", scores)
            paired_ranks = {}
            for label, support in (("target", TARGET_SUPPORT), ("position", POSITION_SUPPORT)):
                for arm, arrays in (("main", main_arrays), ("raw", raw_arrays)):
                    selected = common.slice_readout_arrays(arrays, virtual_indices=list(range(192)), keep=support)
                    paired_ranks.update({f"{label}_{arm}_{key}": value for key, value in selected.items()})
            common._write_npz(comparison_dir / "paired_ranks.npz", paired_ranks)
            runner._new_json(comparison_dir / "items.json", rows)
            runner._new_json(comparison_dir / "task_names.json", names)
            runner._new_json(comparison_dir / "token_forms.json", forms)
            runner._new_json(comparison_dir / "summary.json", summary)
            section_records.append(common._seal_section(out, "comparisons", {
                "kind": "paired_control_comparisons", "common": common_reference,
                "conditional_on_fit_id": 1, "n_independent_control_fits": 1,
                "target_support": TARGET_SUPPORT, "position_support": POSITION_SUPPORT,
                "target_late_band": TARGET_LATE, "position_late_band": POSITION_LATE,
                "population": population, "regions": ["all_common", "loop4_late"],
                "main_fit": main_checked["owner"]["identity"]["fits"][0],
                "control_sections": section_records.copy(),
                "score_rule": "(rank>=0)&(rank<10); matched controls take per-control any-layer maxima before averaging; slots then items weighted equally",
                "missing_items": "score arrays store zero with eligible=false; summaries exclude those items",
                "fit_uncertainty": "not estimable from the one preselected conditional calibration draw",
            }))
            common._complete_output(out, section_records)
    checked = validate_completed_controls(out, main, combined, main_checked)
    return {"status": "complete", "schema": SCHEMA, "out": str(out), "files": checked["files"],
            "control_profiles": PROFILES, "items": 148, "fit_ids": [1],
            "elapsed_seconds": time.monotonic() - started, "complete": runner._record(out / "COMPLETE.json")}


def _self_test(ouro_src):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU proof must start without CUDA initialization")
    checks = []

    def require(condition, name):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def rejects(action, name, errors=(ValueError, FileNotFoundError, FileExistsError)):
        try:
            action()
        except errors:
            checks.append(name)
        else:
            raise AssertionError(f"did not reject: {name}")

    names = {"multihop": ["a", "b", "c"], "order-ops": ["1", "2", "3"]}
    rows = [{"task": task, "own_index": [0, -1, -1], "eligible": [eligible],
             "control_indices": [[1, 2]], "correct": correct}
            for task, eligible, correct in (("multihop", True, True), ("multihop", False, False), ("order-ops", True, False))]
    ranks = np.full((3, 128, 192), -1, dtype=np.int32)
    ranks[:, :3] = 20
    ranks[0, 0, 0] = 9
    ranks[0, 0, 1] = 10
    ranks[0, 1, 5] = 9
    ranks[0, 2, 180] = 9
    ranks[2, 0, 190] = 9
    scored = score_rank_arrays(ranks[..., :191], rows, names, POSITION_SUPPORT,
                               bands={"all_common": POSITION_SUPPORT, "loop4_late": POSITION_LATE})
    require(scored["eligible"].tolist() == [True, False, True], "ineligible items excluded with an explicit mask")
    require(scored["own_layer"][0, :2].tolist() == [1.0, 0.0], "zero-based rank9 hits and rank10 misses")
    require(scored["control_regions"][0, 0] == 1.0 and scored["control_layer"][0].max() == 0.5,
            "each control takes its maximum before control averaging")
    target_scored = score_rank_arrays(ranks[..., :190], rows, names, TARGET_SUPPORT,
                                      bands={"all_common": TARGET_SUPPORT, "loop4_late": TARGET_LATE})
    require(target_scored["own_regions"][2, 0] == 0.0 and scored["own_regions"][2, 0] == 1.0,
            "source190 removed from both target arms before any-layer maximum")
    broken = ranks[..., :190].copy()
    broken[0, 0, 189] = -1
    rejects(lambda: score_rank_arrays(broken, rows, names, TARGET_SUPPORT, bands={"late": TARGET_LATE}),
            "unsupported negative real-name ranks cannot become hits")
    rejects(lambda: score_rank_arrays(ranks[..., :190], rows, names, TARGET_SUPPORT, bands={"bad": [190]}),
            "unsupported comparison band rejected")
    raw = ranks.copy()
    raw[:, :3] = 20
    scores, summary = _comparisons({"allrank": ranks}, {"allrank": raw}, {
        "penultimate": {"allrank": ranks[..., :190]}, "sampled_sum": {"allrank": ranks[..., :191]},
        "diagonal": {"allrank": raw[..., :191]},
    }, rows, names)
    require(np.array_equal(scores["contrast_target_main_minus_penultimate_delta_layer"], np.zeros((3, 190))),
            "identical paired maps produce exactly zero target contrast")
    require(summary["contrasts"]["order-ops"]["all"]["position_sampled_sum_minus_diagonal"]["own_regions"][0] == 1.0,
            "position paired effect retains supported source190")
    require(summary["scores"]["multihop"]["incorrect"]["target_main"]["eligible_items"] == 0
            and summary["scores"]["multihop"]["incorrect"]["target_main"]["own_layer"] is None,
            "empty correctness stratum does not manufacture an estimate")

    sys.path[:0] = [str(runner.REPO), str(Path(ouro_src).absolute())]
    from tests.tiny import TinyDecoder
    from ouro_jlens import evaluate as evaluator
    from ouro_jlens.evaldata import Item
    from jlens import JacobianLens
    model = TinyDecoder(n_layers=192, d_model=4, vocab_size=32).eval().requires_grad_(False)
    model.n_ut, model.n_physical = 4, 48
    model.exit_index = lambda ut: ut * 48 + 47
    items = [Item("one", "multihop", "prompt", "answer", ["a"], [0, 1, 2], {"a": [4]}, {"a": False}),
             Item("two", "order-ops", "other", "answer", ["1"], [0, 3, 4], {"1": [5]}, {"1": False})]
    states = torch.randn(2, 192, 4)
    with common.task_names(evaluator, items), torch.no_grad():
        exits = torch.stack([model.unembed(states[:, model.exit_index(ut)]).float() for ut in range(4)], dim=1)
        targets = {target: model.unembed(states[:, target]).float() for target in (190, 191)}
        for target in (190, 191):
            lens = JacobianLens({layer: torch.eye(4) for layer in range(target)}, n_prompts=100, d_model=4)
            arrays, readout = _control_readout(evaluator, model, items, states, lens, target, exits, targets[target])
            require(arrays["allrank"].shape[-1] == target and readout["identity_indices"] == [],
                    f"target{target} exposes learned source columns only")
            require("kl_to_local" not in arrays and "rank_of_local_top1" not in arrays,
                    f"target{target} diagnostics name the actual target state")
            direct = evaluator.kl(model.unembed(states[0, :target]).float(), targets[target][0]).numpy()
            require(np.array_equal(arrays["kl_to_target_state"][0], direct),
                    f"target{target} diagnostic matches direct original KL")
        require(not torch.equal(targets[190], targets[191]), "target diagnostic fixture distinguishes190 from191")

    with tempfile.TemporaryDirectory(prefix="ouro-controls-cpu-") as temporary:
        temporary = Path(temporary)
        prompts = [f"Named-bank CPU paragraph {index}." for index in range(100)]
        lengths, valid = [128] * 100, [111] * 100
        # Symbolic production identities exercise the strict profile binder
        # without allocating a production-width matrix or loading a model.
        combined_path = runner.HERE / "combined_contract.json"
        combined = runner._json(combined_path)
        fit = {"fit_id": 1, "seed": 2026090701, "n_prompts": 100, "prompts_path": "fit01.json",
               "sha256": combined["controls"]["calibration_sha256"], "token_lengths": lengths,
               "n_valid": valid, "prompts": prompts, "prompt_path": "/relocated/fit01.json"}
        declared = [{"root": key[0], "path": key[1], "sha256": "a" * 64}
                    for key in sorted(set(common._fit_import_keys().values()))]
        combined_key = ("jlens", combined_path.relative_to(runner.REPO).as_posix())
        declared.append({"root": combined_key[0], "path": combined_key[1], "sha256": runner._file_hash(combined_path)})
        snapshot_rows = [{"path": name, "bytes": 1, "sha256": char * 64}
                         for name, char in (("modeling_ouro.py", "b"), ("configuration_ouro.py", "c"))]
        environment = {"packages": {"torch": "fixture"}, "torch_git": "fixture", "python_version": "fixture",
                       "cuda_build_version": "fixture", "cudnn_runtime_integer": 1}
        contract = {"fits": [fit], "spec_sha256": "d" * 64, "spec": {"model": {"fixture": True}},
                    "sources": declared, "source_paths": {combined_key: combined_path},
                    "environment": environment, "manifest": {"files": snapshot_rows}}
        runtime = {
            "run_spec_sha256": contract["spec_sha256"], "model": contract["spec"]["model"],
            "snapshot_path": "/deleted/producer/snapshot",
            "snapshot_files": {row["path"]: {key: row[key] for key in ("bytes", "sha256")} for row in snapshot_rows},
            "source_files": declared, "settings": runner.SETTINGS, "environment": environment,
            "gpu": {"name": "producer GPU", "uuid": "producer UUID"}, "precision": {"fixture": True},
            "attention_implementation": "sdpa", "calibration": {key: value for key, value in fit.items() if key != "prompts"},
            "imported_sources": [{"module": name, "path": f"/deleted/producer/{name}.py", "sha256": "a" * 64}
                                 for name in common._fit_import_keys()],
            "remote_implementations": [{"class": "fixture." + row["path"], "path": "/deleted/producer/" + row["path"],
                                         "sha256": row["sha256"]} for row in snapshot_rows],
        }
        runtime["calibration"]["prompt_path"] = "/deleted/producer/fit01.json"
        main_identity = runner._fit_identity(prompts, lengths, valid, 1, runtime, 2048, tuple(range(191)), False)
        main_checked = {"owner": {"identity": {"fits": [{"identity": main_identity}]}}}
        for profile_name in PROFILES:
            profile = runner.PROFILES[profile_name]
            extended = {**runtime, "combined_contract_sha256": runner._file_hash(combined_path),
                        "production_profile": profile_name, "control_contract": combined["controls"],
                        "settings": {**runner.SETTINGS, "source_layers": profile["source_layers"],
                                     "target_layer": profile["target_layer"],
                                     "mode": "paired_sampled" if profile["bank_arms"] else "dense"}}
            profile_identity = runner._fit_identity(prompts, lengths, valid, 1, extended, 2048, profile["source_layers"], False,
                                                    production_profile=profile_name)
            profile_owner = {"schema_version": 1, "identity": profile_identity,
                             "fit_identity_sha256": runner._digest(profile_identity)}
            require(validate_control_owner(profile_owner, contract, combined, profile_name, main_checked=main_checked)["fit_id"] == 1,
                    f"{profile_name} exact relocated production identity accepted")
            altered = json.loads(runner._canonical(profile_owner))
            altered["identity"]["runtime"]["control_contract"]["positions"]["q"][0] += 1
            altered["fit_identity_sha256"] = runner._digest(altered["identity"])
            rejects(lambda: validate_control_owner(altered, contract, combined, profile_name, main_checked=main_checked),
                    f"{profile_name} changed shared q contract rejected")
            altered = json.loads(runner._canonical(profile_owner))
            altered["identity"]["runtime"]["gpu"]["uuid"] = "different fitting GPU"
            altered["fit_identity_sha256"] = runner._digest(altered["identity"])
            rejects(lambda: validate_control_owner(altered, contract, combined, profile_name, main_checked=main_checked),
                    f"{profile_name} fitting runtime mismatch rejected")
        identity = runner._fit_identity(prompts, lengths, valid, 1, {"fixture": True}, 4, (0, 1), True,
                                        production_profile="ouro_positions")
        sums = {arm: {layer: torch.eye(4) * scale for layer in (0, 1)}
                for arm, scale in (("sampled_sum", 100.0), ("diagonal", 50.0))}
        diagnostics = [{"index": index, "prompt_sha256": identity["prompt_sha256"][index],
                        "token_length": 128, "n_valid": 111, "frozen_q": 16} for index in range(100)]
        with runner._owned_fit(temporary / "producer", 1, identity) as fit_dir:
            pointer, _ = runner._commit(torch, fit_dir, identity, sums, 100, diagnostics, "checkpoint")
            final_pointer, final_dir = runner._commit(torch, fit_dir, identity, sums, 100, diagnostics, "final", checkpoint_pointer=pointer)
        fit_dir, owner = common._owner(fit_dir)
        final, record = common._read_generation(torch, fit_dir, owner)
        require(record["n_prompts"] == 100, "both named banks pass producer sealed FP16-mean validation")
        require(_record_owner(record) == owner, "embedded producer OWNER bytes and complete pointers validate")
        state = torch.load(final, weights_only=True)
        state["J"]["diagonal"][1][0, 0] += 0.25
        with final.open("wb") as handle:
            torch.save(state, handle)
        seal = runner._json(final_dir / "SEAL.json")
        seal["files"]["lens.pt"] = runner._record(final)
        runner._atomic_json(final_dir / "SEAL.json", seal)
        final_pointer["seal"] = runner._record(final_dir / "SEAL.json")
        runner._atomic_json(fit_dir / "COMPLETE.json", final_pointer)
        rejects(lambda: common._read_generation(torch, fit_dir, owner),
                "changed named arm rejected even after consistent file-hash resealing")

        output = common._new_output(temporary / "evaluation", {"fixture": True, "sections": SECTIONS}, schema=SCHEMA)
        references = []
        for section in SECTIONS:
            directory = runner._mkdir(output / section)
            common._write_npz(directory / "arrays.npz", {"ranks": ranks[..., :190]})
            references.append(common._seal_section(output, section, {"fixture": True}))
        rejects(lambda: common._complete_output(output, references[:-1]), "controls cannot complete without comparisons")
        require(not (output / "COMPLETE.json").exists(), "incomplete control output has no COMPLETE")
        common._complete_output(output, references)
        require(common.validate_output(output, schema=SCHEMA)["owner"]["identity"]["sections"] == SECTIONS,
                "all four ordered control sections seal and validate")
        rejects(lambda: common.validate_output(output), "control completion cannot masquerade as main completion")
    require(not torch.cuda.is_initialized(), "CUDA remained uninitialized")
    return {"status": "passed", "n_checks": len(checks), "checks": checks,
            "source_sha256": runner._file_hash(__file__), "target_support": TARGET_SUPPORT,
            "target_late_band": TARGET_LATE, "position_support": POSITION_SUPPORT,
            "cuda_initialized": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-spec", type=Path, default=runner.HERE / "run_spec.json")
    parser.add_argument("--combined-contract", type=Path, default=runner.HERE / "combined_contract.json")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--ouro-src", type=Path)
    parser.add_argument("--main-evaluation-dir", type=Path)
    parser.add_argument("--penultimate-fit-dir", type=Path)
    parser.add_argument("--positions-fit-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    if argv is None and len(sys.argv) == 1:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    if args.ouro_src is None:
        parser.error("--ouro-src is required")
    if args.self_test:
        if args.validate_only or any((args.main_evaluation_dir, args.penultimate_fit_dir, args.positions_fit_dir)):
            parser.error("CPU self-test cannot consume scientific fit/evaluation artifacts")
        result = _self_test(args.ouro_src)
        if args.out is not None:
            runner._new_json(args.out, result)
    else:
        if (any(value is None for value in (args.snapshot, args.main_evaluation_dir,
                                            args.penultimate_fit_dir, args.positions_fit_dir))
                or (args.out is None and not args.validate_only)):
            parser.error("production requires --snapshot, --main-evaluation-dir, both --*-fit-dir paths and --out")
        import torch
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        prepared = validate_inputs(args)
        result = ({"status": "validated", "control_profiles": PROFILES, "cuda_initialized": torch.cuda.is_initialized()}
                  if args.validate_only else _evaluate(args, *prepared))
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
