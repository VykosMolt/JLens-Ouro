"""CPU comparison of the frozen Huginn R8 lens and Ouro main fit01.

The associated HUGINN_COMPARISON_CONTRACT.md fixes the population, operators,
source selections and conditional uncertainty before Huginn outcomes. This
reader performs no inference, fitting, model loading or cloud action. Plan/help
use the standard library; numerical execution requires completed sealed inputs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_refits as ouro

io, evaluation, controls = ouro.io, ouro.evaluation, ouro.controls
import evaluate_huginn as huginn

SCHEMA = "huginn_ouro_comparison.v1"
PREDICTION = "If J-Lens recovers content outside the native output basis, its advantage over raw logit lens should be larger in Huginn than in Ouro."
TASKS = ouro.TASKS
SEEDS = (2026090803, 2026090804)
N_BOOT, BOOT_SEED = 10000, 2026090808
EXAMPLE_SEED, FAILURE_SEED = 2026090809, 2026090810
ELIGIBILITY_SHA256 = "153c340591c4c1ff298383a24f2ac1135a157fba08dc9ecb5a82dedc298457af"
COUNTS = {"items": 148, "tasks": {"multihop": 93, "order-ops": 55},
          "eligible_items": {"multihop": 88, "order-ops": 51},
          "eligible_slots": {"multihop": 98, "order-ops": 51},
          "control_names": {"multihop": 68, "order-ops": 20}}
SUPPORT = {"ouro": list(range(192)), "huginn": list(range(32))}
LEARNED = {"ouro": list(range(191)), "huginn": list(range(32))}
GRID = {"ouro": [24 * j - 1 for j in range(1, 8)], "huginn": [4 * j - 1 for j in range(1, 8)]}
EXITS = {"ouro": [47, 95, 143, 191], "huginn": list(range(3, 32, 4))}
METHODS = {"ouro": ("raw", "jlens"), "huginn": ("raw", "jlens", "coda")}
REGIONS = ("all_displayed", "all_learned", "seven")
METRICS = ("all_learned_mean", "seven_mean", "any_of_seven")
MODES = ("conditional_item", "conditional_component")
OUTPUT_FILES = {"OWNER.json", "report.json", "population.json", "examples.json", "paired_items.npz",
                "cells.csv", "metrics.csv", "native_agreement.csv"}


def np_module():
    return ouro.numpy()


def _record_shape(record):
    return (isinstance(record, dict) and set(record) == {"bytes", "sha256"}
            and type(record["bytes"]) is int and record["bytes"] > 0
            and isinstance(record["sha256"], str) and io._SHA256.fullmatch(record["sha256"]) is not None)


def population(plan, original_rows, names, forms):
    """Independently reconstruct the frozen common slots and control catalogue."""
    if (plan.get("schema") != huginn.ELIGIBILITY_SCHEMA or plan.get("tasks") != list(TASKS)
            or plan.get("evaluation_base_seeds") != list(SEEDS)
            or plan.get("evaluation_max_length") != 512 or plan.get("seed_namespace") != "evaluation"
            or plan.get("seed_rule") != huginn.SEED_RULE or plan.get("task_names") != names
            or plan.get("token_forms", {}).get("ouro") != forms or len(plan.get("rows", [])) != 148):
        raise ValueError("common population schema, names, forms or seed policy changed")
    native_forms = plan["token_forms"]["huginn"]
    common = {task: [index for index, name in enumerate(names[task])
                     if forms[task].get(name) and native_forms[task].get(name)] for task in TASKS}
    excluded = {task: [name for index, name in enumerate(names[task]) if index not in common[task]] for task in TASKS}
    if (common != plan.get("common_name_indices") or excluded != plan.get("excluded_control_names")
            or excluded != {"multihop": ["14", "Vatican"], "order-ops": []}):
        raise ValueError("common supported control names changed")
    rows = []
    for index, (old, row) in enumerate(zip(original_rows, plan["rows"])):
        for key in ("name", "task", "prompt", "target", "intermediates", "own_index"):
            if not ouro._same(row.get(key), old[key]):
                raise ValueError("common item order or labels differ from Ouro: " + str(index))
        for key in ("scorable", "leaked", "eligible", "token_ids", "n_tokens", "readout_token", "intermediate_tokens"):
            if not ouro._same(row.get("ouro", {}).get(key), old[key]):
                raise ValueError("common Ouro context differs from its sealed evaluation: " + str(index))
        own = {value for value in row["own_index"] if value >= 0}
        yes, pools, reasons = [], [], []
        for slot, label in enumerate(row["intermediates"]):
            why = []
            for model in ("ouro", "huginn"):
                view = row[model]
                if not view["scorable"][slot]:
                    why.append(model + ":no_single_token_form")
                if view["leaked"][slot]:
                    why.append(model + ":form_in_readout_context")
            if row["task"] == "order-ops" and label in ouro.OPERATIONS:
                why.append("arithmetic:operation_slot_excluded")
            pool = [value for value in common[row["task"]] if value not in own
                    and ((names[row["task"]][value] in ouro.OPERATIONS) == (label in ouro.OPERATIONS))]
            if not why and not pool:
                raise ValueError("eligible common slot lacks matched controls")
            yes.append(not why)
            pools.append(pool)
            reasons.append(why)
        if (row.get("eligible") != yes or row.get("control_indices") != pools
                or row.get("exclusion_reasons") != reasons):
            raise ValueError("joint eligibility, exclusions or matched controls were overridden")
        rows.append({**row, "eligible": yes, "control_indices": pools})
    counts = {"items": len(rows), "tasks": {task: sum(row["task"] == task for row in rows) for task in TASKS},
              "eligible_items": {task: sum(row["task"] == task and any(row["eligible"]) for row in rows) for task in TASKS},
              "eligible_slots": {task: sum(sum(row["eligible"]) for row in rows if row["task"] == task) for task in TASKS},
              "control_names": {task: len(common[task]) for task in TASKS}}
    if counts != COUNTS or plan.get("counts") != counts:
        raise ValueError("common population is not the frozen139 eligible items")
    return rows


def _prior_records(artifacts):
    result = {}
    for label, key in (("ouro", "main"), ("controls", "controls")):
        artifact = artifacts[key]
        owner = io._json(Path(artifact["root"]) / "OWNER.json")
        result[label] = {"schema": owner["schema"], "identity_sha256": owner["identity_sha256"],
                         "owner_sha256": io._digest(owner), "owner_file": artifact["files"]["OWNER.json"],
                         "complete_file": artifact["files"]["COMPLETE.json"]}
    return result


def _runtime(runtime, main, manifest):
    expected_keys = {"environment", "gpu", "precision", "imported_sources", "remote_implementations",
                     "attention_implementation", "snapshot_files", "snapshot_path_at_evaluation"}
    if not isinstance(runtime, dict) or set(runtime) != expected_keys:
        raise ValueError("Huginn evaluation runtime schema changed")
    snapshot = {row["path"]: {key: row[key] for key in ("bytes", "sha256")} for row in manifest["files"]}
    if (runtime["snapshot_files"] != snapshot or not isinstance(runtime["snapshot_path_at_evaluation"], str)
            or not isinstance(runtime["gpu"], dict) or not runtime["gpu"].get("name")
            or not isinstance(runtime["precision"], dict) or not runtime["precision"]
            or not isinstance(runtime["attention_implementation"], str)):
        raise ValueError("Huginn actual snapshot, GPU, attention or precision binding changed")
    for field in ("packages", "torch_git", "python_version", "cuda_build_version", "cudnn_runtime_integer"):
        if not ouro._same(runtime["environment"].get(field), main["environment"].get(field)):
            raise ValueError("Huginn evaluation environment differs from the frozen " + field)
    modules = {
        "ouro_jlens", "ouro_jlens.evaluate", "ouro_jlens.evaldata", "ouro_jlens.fit_lens",
        "ouro_jlens.evidence", "ouro_jlens.recurrent", "jlens", "jlens._logging", "jlens.fitting",
        "jlens.hooks", "jlens.hf", "jlens.lens", "jlens.protocol", "jlens.vis", "huginn_adapter",
    }
    imported = runtime["imported_sources"]
    module_paths = {module: ("ouro_src" if module.startswith("ouro_jlens") else "jlens",
        module.replace(".", "/") + ("/__init__.py" if "." not in module else ".py")) for module in modules}
    module_paths["huginn_adapter"] = ("jlens", (ouro.DEPLOYMENT / "huginn_adapter.py").relative_to(ouro.REPO).as_posix())
    declared = {(row["root"], row["path"]): row["sha256"] for row in main["sources"]}
    if (not isinstance(imported, list) or len(imported) != len(modules)
            or {row.get("module") for row in imported if isinstance(row, dict)} != modules):
        raise ValueError("Huginn evaluation imported-source membership changed")
    for row in imported:
        if (set(row) != {"module", "root", "path", "sha256"}
                or (row["root"], row["path"]) != module_paths[row["module"]]
                or row["sha256"] != declared.get((row["root"], row["path"]))):
            raise ValueError("Huginn evaluation imported an unmanifested source")
    remote = runtime["remote_implementations"]
    if (not isinstance(remote, list) or len(remote) != 2
            or any(not isinstance(row, dict) or set(row) != {"class", "path", "sha256"}
                   or not isinstance(row["class"], str) or not isinstance(row["path"], str) for row in remote)
            or {row["sha256"] for row in remote}
            != {snapshot[name]["sha256"] for name in ("raven_modeling_minimal.py", "raven_config_minimal.py")}):
        raise ValueError("Huginn evaluation native implementation hashes changed")


def bind_huginn(checked, root, args, main, combined, manifest, calibration, plan, prior):
    """Bind the completed readout to its actual N100 owner and prerequisites."""
    identity = checked["owner"]["identity"]
    expected = {"kind": "huginn_r8_n100_readout", "run_spec_sha256": main["spec_sha256"],
                "combined_contract_sha256": io._file_hash(args.combined_contract),
                "source_files": main["sources"], "model": {key: combined["huginn"][key]
                    for key in ("repo_id", "revision", "manifest_sha256")},
                "eligibility_file": io._record(args.eligibility), "eligibility_payload_sha256": io._digest(plan),
                "evaluation_base_seeds": list(SEEDS), "sections": huginn.SECTIONS,
                "methods": list(huginn.METHODS), "learned_sources": SUPPORT["huginn"],
                "target_layer": 33, "position": -1, "evaluation_max_length": 512}
    if any(not ouro._same(identity.get(key), value) for key, value in expected.items()):
        raise ValueError("Huginn completed evaluation differs from the declared comparison inputs")
    record = identity.get("fit")
    owner = controls._record_owner(record)
    huginn.validate_huginn_owner(owner, main, combined, manifest, calibration,
                                 combined_sha256=expected["combined_contract_sha256"],
                                 calibration_sha256=io._file_hash(args.huginn_calibration))
    initialization = record.get("initialization_validation")
    if (not _record_shape(record.get("lens")) or not isinstance(initialization, dict)
            or set(initialization) != {"rows", "initializations_sha256"} or initialization["rows"] != 100
            or not isinstance(initialization["initializations_sha256"], str)
            or io._SHA256.fullmatch(initialization["initializations_sha256"]) is None
            or owner["identity"]["runtime"]["prerequisites"] != prior):
        raise ValueError("Huginn lens, initialization or preceding Ouro/control provenance changed")
    population_metadata = io._json(root / "population/metadata.json")
    if population_metadata != {"kind": "frozen_common_eligibility", "counts": COUNTS}:
        raise ValueError("Huginn population metadata changed")
    shared_runtime = None
    for seed in SEEDS:
        metadata = io._json(root / "seeds" / str(seed) / "metadata.json")
        required = {"status": "complete", "n_items_done": 148, "seed": seed, "position": -1,
                    "virtual_indices": SUPPORT["huginn"], "native_exit_indices": EXITS["huginn"],
                    "coda_policy": "full raw sequence -> pre-coda norm -> two coda blocks -> final norm/head; select last logits afterwards",
                    "within_pass_coda": "diagnostic interrupted-core intervention; only physical block3 is a native recurrence exit",
                    "readout_policy": "native raw and JLens last-state vectors use final norm/head; maps are FP32 at application",
                    "matrix_policy": "one preselected N100 map bank; the two seeds measure state-initialization sensitivity"}
        if any(not ouro._same(metadata.get(key), value) for key, value in required.items()):
            raise ValueError("Huginn completed seed changed its native/coda/readout semantics")
        runtime = metadata.get("runtime")
        _runtime(runtime, main, manifest)
        if shared_runtime is not None and not ouro._same(runtime, shared_runtime):
            raise ValueError("the two Huginn seeds did not share one inference runtime")
        shared_runtime = runtime


def load_inputs(args):
    """No model objects: complete semantic gates, frozen population, sealed ranks."""
    np = np_module()
    # This reuses the full Ouro/P2 semantic reader, including fresh paired-score
    # reconstruction, without running that reader's statistical analysis.
    original = ouro.load_inputs(args)
    main = original["contract"]
    hmain, combined, manifest, calibration = huginn._contract(args)
    if (hmain["spec_sha256"] != main["spec_sha256"]
            or not ouro._same(hmain["fits"][0], main["fits"][0])):
        raise ValueError("Huginn and Ouro do not bind the same preselected calibration fit01")
    if io._file_hash(args.eligibility) != ELIGIBILITY_SHA256:
        raise ValueError("the pre-outcome eligibility file bytes changed")
    plan = io._json(io._no_links(args.eligibility))
    rows = population(plan, original["rows"], original["names"], original["forms"])
    root = io._no_links(args.huginn)
    checked = huginn.validate_output(root)
    if not ouro._same(io._json(root / "population/eligibility.json"), plan):
        raise ValueError("Huginn stored population differs from the frozen common population")
    bind_huginn(checked, root, args, main, combined, manifest, calibration, plan,
                 _prior_records(original["artifacts"]))
    artifacts = {**original["artifacts"], "huginn": {"root": str(root), "files": ouro.artifact_records(root, checked)}}
    consumed = dict(original["consumed"])
    # Huginn's semantic validator reads both caches as well as rank/JSON files.
    # Keep all those actual input records for the final stability check.
    for relative, record in artifacts["huginn"]["files"].items():
        io._verify_record(root / relative, record)
        consumed[str(root / relative)] = record
    main_root = Path(artifacts["main"]["root"])
    arrays = {"ouro": {}, "huginn": {}}
    for method, relative, prefix in (("raw", "common/arrays.npz", "logitlens_"),
                                      ("jlens", "fits/fit_01/arrays.npz", "jlens_exit3_")):
        arrays["ouro"][method] = controls._load_arrays(main_root / relative, prefix)
        evaluation._validate_arrays(arrays["ouro"][method], original["rows"], original["names"], SUPPORT["ouro"])
    with np.load(main_root / "common/arrays.npz", allow_pickle=False) as archive:
        arrays["ouro"]["exit_top1"] = archive["exit_top1"].copy()
    for seed in SEEDS:
        with np.load(root / "seeds" / str(seed) / "arrays.npz", allow_pickle=False) as archive:
            bank = {key: archive[key].copy() for key in archive.files}
        huginn._validate_arrays(bank, plan)
        arrays["huginn"][seed] = bank
    return {"contract": main, "rows": rows, "plan": plan, "arrays": arrays,
            "artifacts": artifacts, "consumed": consumed}


def score(ranks, rows, plan, model):
    """Compact only supported names to reuse the original strict rank reducer."""
    np = np_module()
    support = SUPPORT[model]
    if ranks.shape != (len(rows), 128, len(support)) or ranks.dtype != np.int32:
        raise ValueError("source rank axes differ from the frozen readout geometry")
    names = {task: [plan["task_names"][task][i] for i in plan["common_name_indices"][task]] for task in TASKS}
    indices = {task: {old: new for new, old in enumerate(plan["common_name_indices"][task])} for task in TASKS}
    selected = np.full_like(ranks, -1)
    compact_rows = []
    for i, row in enumerate(rows):
        common = plan["common_name_indices"][row["task"]]
        selected[i, :len(common)] = ranks[i, common]
        remap = indices[row["task"]]
        compact_rows.append({**row, "own_index": [remap.get(index, -1) for index in row["own_index"]],
                             "control_indices": [[remap[index] for index in pool] for pool in row["control_indices"]]})
    return controls.score_rank_arrays(selected, compact_rows, names, support,
                                       bands={"all_displayed": support, "all_learned": LEARNED[model], "seven": GRID[model]})


def method_metrics(bank, model, field):
    np = np_module()
    layer, regions = bank[field + "_layer"], bank[field + "_regions"]
    return np.stack((layer[..., LEARNED[model]].mean(axis=-1), layer[..., GRID[model]].mean(axis=-1),
                     regions[..., REGIONS.index("seven")]), axis=-1)


def bootstrap_design(rows, *, draws=N_BOOT):
    np = np_module()
    design = {}
    for task_index, task in enumerate(TASKS):
        indices = np.asarray([i for i, row in enumerate(rows) if row["task"] == task and any(row["eligible"])], np.int64)
        groups = ouro.concept_components(rows, indices)
        item_draws, weights = ouro._membership(len(indices), draws, np.random.SeedSequence(BOOT_SEED, spawn_key=(1, task_index)))
        component_draws, component_weights = ouro._membership(len(indices), draws,
            np.random.SeedSequence(BOOT_SEED, spawn_key=(2, task_index)), groups)
        design[task] = {"indices": indices, "groups": groups, "item_indices": item_draws,
                        "component_indices": component_draws, "item_weights": weights,
                        "component_weights": component_weights}
    return design


def calculate(inputs, design):
    np = np_module()
    rows, plan, arrays = inputs["rows"], inputs["plan"], inputs["arrays"]
    banks = {"ouro": {}, "huginn": {}}
    saved = {}
    for model in METHODS:
        for seed in ((None,) if model == "ouro" else SEEDS):
            destination = banks[model] if seed is None else banks[model].setdefault(seed, {})
            for method in METHODS[model]:
                ranks = (arrays[model][method]["allrank"] if seed is None
                         else arrays[model][seed][method + "_allrank"])
                bank = score(ranks, rows, plan, model)
                destination[method] = bank
                prefix = model + ("" if seed is None else "_" + str(seed)) + "_" + method
                saved.update({prefix + "_" + field: value for field, value in bank.items()})
                for field in ("own", "control", "delta"):
                    saved[prefix + "_" + field + "_metrics"] = method_metrics(bank, model, field)
            pairs = {"jlens_minus_raw": ("jlens", "raw")}
            if model == "huginn":
                pairs.update(coda_minus_raw=("coda", "raw"), jlens_minus_coda=("jlens", "coda"))
            prefix = model + ("" if seed is None else "_" + str(seed))
            for pair, (left, right) in pairs.items():
                saved.update({prefix + "_" + pair + "_" + field: destination[left][field] - destination[right][field]
                              for field in destination[left] if field != "eligible"})
    paired = {}
    for model in METHODS:
        per_seed = []
        for seed in ((None,) if model == "ouro" else SEEDS):
            values = banks[model] if seed is None else banks[model][seed]
            per_seed.append(method_metrics(values["jlens"], model, "delta") - method_metrics(values["raw"], model, "delta"))
        paired[model] = per_seed[0] if model == "ouro" else np.stack(per_seed)
    cross = paired["huginn"] - paired["ouro"][None]
    seed_average = cross.mean(axis=0)
    saved.update(ouro_paired_metrics=paired["ouro"], huginn_paired_metrics=paired["huginn"],
                 cross_model_paired_metrics=cross, cross_model_seed_average_metrics=seed_average)
    hgrid = np.stack([banks["huginn"][seed]["jlens"]["delta_layer"][:, GRID["huginn"]]
                     - banks["huginn"][seed]["raw"]["delta_layer"][:, GRID["huginn"]] for seed in SEEDS])
    ogrid = banks["ouro"]["jlens"]["delta_layer"][:, GRID["ouro"]] - banks["ouro"]["raw"]["delta_layer"][:, GRID["ouro"]]
    saved["cross_model_paired_seven_cells"] = hgrid - ogrid[None]
    points, draws = {}, {}
    populations = {}
    for task in TASKS:
        sample = design[task]
        indices, groups = sample["indices"], sample["groups"]
        bank = seed_average[indices]
        points[task] = bank.mean(axis=0)
        draws[task] = {"conditional_item": sample["item_weights"] @ bank,
                       "conditional_component": sample["component_weights"] @ bank}
        sizes = np.bincount(groups)
        populations[task] = {"item_indices": indices, "eligible_items": len(indices),
            "eligible_slots": sum(sum(rows[int(i)]["eligible"]) for i in indices),
            "component_membership": groups, "component_sizes": sizes, "component_count": len(sizes)}
        for field in ("indices", "groups", "item_indices", "component_indices"):
            saved["bootstrap_" + task + "_" + field] = sample[field]
    intervals, radii = ouro.family_intervals(points, draws, MODES)
    comparison = {"metric_ids": list(METRICS), "conditional_on_fit_id": 1,
                  "calibration_fits_per_model": 1, "evaluation_base_seeds": list(SEEDS),
                  "family": {"contrasts": sum(len(point) for point in points.values()), "simultaneous_radius95": radii},
                  "tasks": {}}
    for task in TASKS:
        indices = design[task]["indices"]
        comparison["tasks"][task] = {"mean": points[task], "per_seed": cross[:, indices].mean(axis=1),
            "ouro_paired_effect": paired["ouro"][indices].mean(axis=0),
            "huginn_paired_effect_per_seed": paired["huginn"][:, indices].mean(axis=1),
            "seven_cell_difference_per_seed": saved["cross_model_paired_seven_cells"][:, indices].mean(axis=1),
            "intervals": intervals[task]}
    if comparison["family"]["contrasts"] != 6:
        raise ValueError("comparison must retain the six prespecified task-by-metric contrasts")
    return {"populations": populations, "comparison": comparison,
            "readouts": readout_summaries(banks, design), "native_agreement": native_agreement(arrays, design)}, saved


def readout_summaries(banks, design):
    np = np_module()
    result = {model: {} for model in METHODS}
    for model in METHODS:
        seed_values = ("fixed",) if model == "ouro" else (*SEEDS, "seed_average")
        for seed in seed_values:
            source = (banks["ouro"] if model == "ouro" else banks["huginn"].get(seed))
            if source is None:
                source = {method: {field: np.mean([banks["huginn"][value][method][field] for value in SEEDS], axis=0)
                                   for field in banks["huginn"][SEEDS[0]][method] if field != "eligible"}
                          for method in METHODS[model]}
            views = {method: bank for method, bank in source.items()}
            pairs = {"jlens_minus_raw": ("jlens", "raw")}
            if model == "huginn":
                pairs.update(coda_minus_raw=("coda", "raw"), jlens_minus_coda=("jlens", "coda"))
            for name, (left, right) in pairs.items():
                views[name] = {field: source[left][field] - source[right][field]
                               for field in source[left] if field != "eligible"}
            per_task = {}
            for task in TASKS:
                indices = design[task]["indices"]
                per_task[task] = {}
                for method, bank in views.items():
                    item = {field: bank[field][indices].mean(axis=0)
                            for field in bank if field != "eligible"}
                    item["source_indices"] = SUPPORT[model]
                    item["native_exit_indices"] = EXITS[model]
                    item["known_identity_indices"] = [191] if model == "ouro" and method.startswith("jlens") else []
                    if model == "huginn":
                        item["physical_block_means"] = {field: item[field].reshape(8, 4).mean(axis=0)
                                                         for field in ("own_layer", "control_layer", "delta_layer")}
                    values = item["delta_layer"][LEARNED[model]]
                    item["learned_cell_empirical_cdf"] = {"sorted_cell_means": np.sort(values),
                                                         "cdf": np.arange(1, len(values) + 1) / len(values),
                                                         "unit": "correlated source locations, not independent observations"}
                    per_task[task][method] = item
            result[model][str(seed)] = per_task
    return result


def native_agreement(arrays, design):
    np = np_module()
    result = {"ouro": {}, "huginn": {}}
    for task in TASKS:
        indices = design[task]["indices"]
        final = arrays["ouro"]["exit_top1"][:, 3]
        result["ouro"][task] = {method: {"to_final_native":
            (arrays["ouro"][method]["top1"][indices] == final[indices, None]).mean(axis=0)} for method in METHODS["ouro"]}
    for seed in SEEDS:
        bank = arrays["huginn"][seed]
        result["huginn"][str(seed)] = {}
        for task in TASKS:
            indices = design[task]["indices"]
            result["huginn"][str(seed)][task] = {method: {
                "to_final_native": (bank[method + "_top1"][indices] == bank["native_top1"][indices, None]).mean(axis=0),
                "to_same_source_coda": (bank[method + "_top1"][indices] == bank["coda_top1"][indices]).mean(axis=0),
            } for method in METHODS["huginn"]}
    return result


def examples(inputs, saved):
    """Frozen uniform examples and separately labelled concept-readout misses."""
    np = np_module()
    rows = inputs["rows"]
    eligible = np.asarray([i for i, row in enumerate(rows) if any(row["eligible"])], np.int64)
    selected = np.random.default_rng(EXAMPLE_SEED).choice(eligible, min(10, len(eligible)), replace=False)
    def record(index):
        index = int(index)
        row = rows[index]
        values = {"ouro": {}, "huginn": {}}
        for model in METHODS:
            for seed in ((None,) if model == "ouro" else SEEDS):
                target = values[model] if seed is None else values[model].setdefault(str(seed), {})
                for method in METHODS[model]:
                    prefix = model + ("" if seed is None else "_" + str(seed)) + "_" + method
                    target[method] = {field: saved[prefix + "_" + field + "_metrics"][index]
                                      for field in ("own", "control", "delta")}
        return {"index": index, **{key: row[key] for key in ("name", "task", "prompt", "target", "intermediates", "eligible", "exclusion_reasons")},
                "readout_metrics": values, "cross_model_paired_metrics_per_seed": saved["cross_model_paired_metrics"][:, index]}
    own_any = np.mean([saved[f"huginn_{seed}_jlens_own_metrics"][:, METRICS.index("any_of_seven")] for seed in SEEDS], axis=0)
    failures = {}
    for task_index, task in enumerate(TASKS):
        pool = np.asarray([index for index in eligible if rows[int(index)]["task"] == task and own_any[index] == 0.0], np.int64)
        chosen = np.random.default_rng(np.random.SeedSequence(FAILURE_SEED, spawn_key=(task_index,))).choice(pool, min(5, len(pool)), replace=False)
        failures[task] = {"eligible_readout_failure_count": len(pool), "eligible_readout_failure_indices": pool,
                          "selected_indices": chosen, "examples": [record(index) for index in chosen]}
    return {"metric_ids": METRICS, "method_field_delta": "own recovery minus matched-control recovery",
            "interpretation": "descriptive item examples; selection never changes metrics, cells, seeds or denominators",
            "random": {"seed": EXAMPLE_SEED, "eligible_population_size": len(eligible), "selected_indices": selected,
                       "selection": "ten jointly eligible items uniformly without replacement, independent of outcomes",
                       "examples": [record(index) for index in selected]},
            "readout_failures": {"seed": FAILURE_SEED,
                "definition": "seed-average Huginn J-Lens own any-of-seven recovery equals zero; no eligible own label is recovered on the seven cells in either seed",
                "interpretation": "conditional descriptive concept-readout misses, not whole-answer incorrectness",
                "tasks": failures}}


def write_tables(out, result):
    cells, native, metrics = [], [], []
    for model, seeds in result["readouts"].items():
        for seed, tasks in seeds.items():
            for task, methods in tasks.items():
                for method, bank in methods.items():
                    for column in SUPPORT[model]:
                        width = 48 if model == "ouro" else 4
                        cells.append({"model": model, "seed": seed, "task": task, "method_or_contrast": method,
                            "source_index": column, "pass_one_based": column // width + 1,
                            "physical_block_one_based": column % width + 1,
                            "native_pass_end": column in EXITS[model], "ouro_identity_reference": model == "ouro" and column == 191,
                            "within_pass_coda_intervention": model == "huginn" and column not in EXITS[model] and "coda" in method,
                            "own": float(bank["own_layer"][column]), "control": float(bank["control_layer"][column]),
                            "excess": float(bank["delta_layer"][column])})
    for task, methods in result["native_agreement"]["ouro"].items():
        for method, fields in methods.items():
            for column in SUPPORT["ouro"]:
                native.append({"model": "ouro", "seed": "fixed", "task": task, "method": method,
                    "source_index": column, "native_pass_end": column in EXITS["ouro"],
                    "to_final_native": float(fields["to_final_native"][column]), "to_same_source_coda": ""})
    for seed, tasks in result["native_agreement"]["huginn"].items():
        for task, methods in tasks.items():
            for method, fields in methods.items():
                for column in SUPPORT["huginn"]:
                    native.append({"model": "huginn", "seed": seed, "task": task, "method": method,
                        "source_index": column, "native_pass_end": column in EXITS["huginn"],
                        **{field: float(values[column]) for field, values in fields.items()}})
    for task, bank in result["comparison"]["tasks"].items():
        for column, name in enumerate(METRICS):
            metrics.append({"task": task, "metric": name, "mean_over_two_fixed_seeds": float(bank["mean"][column]),
                "seed_2026090803": float(bank["per_seed"][0, column]), "seed_2026090804": float(bank["per_seed"][1, column]),
                "ouro_fit01_paired_effect": float(bank["ouro_paired_effect"][column]),
                **ouro._interval_columns(bank["intervals"], column)})
    for filename, rows in (("cells.csv", cells), ("metrics.csv", metrics), ("native_agreement.csv", native)):
        ouro._csv(out / filename, rows)


def validate_output(root):
    root = io._no_links(root)
    owner = io._json(root / "OWNER.json")
    complete = io._json(root / "COMPLETE.json")
    if (set(owner) != {"schema", "identity", "identity_sha256"} or owner["schema"] != SCHEMA
            or owner["identity_sha256"] != io._digest(owner["identity"])
            or owner["identity"].get("kind") != "huginn_ouro_fit01_common_population"
            or owner["identity"].get("evaluation_base_seeds") != list(SEEDS)
            or owner["identity"].get("fit_ids") != {"ouro": 1, "huginn": 1}
            or owner["identity"].get("eligibility_file_sha256") != ELIGIBILITY_SHA256
            or set(complete) != {"schema", "identity_sha256", "files"}
            or complete["schema"] != SCHEMA or complete["identity_sha256"] != owner["identity_sha256"]
            or set(complete["files"]) != OUTPUT_FILES):
        raise ValueError("invalid completed Huginn comparison identity")
    for name, record in complete["files"].items():
        io._verify_record(root / name, record)
    actual = set()
    for path in root.iterdir():
        io._no_links(path)
        if not path.is_file():
            raise ValueError("comparison output contains a nonregular member")
        actual.add(path.name)
    if actual != OUTPUT_FILES | {"COMPLETE.json"}:
        raise ValueError("comparison has missing or unsealed files")
    return {"status": "passed", "files": len(actual), "identity_sha256": owner["identity_sha256"]}


def analyze(args):
    np = np_module()
    out = io._no_links(args.output)
    for input_root in (args.main, args.controls, args.huginn):
        input_root = io._no_links(input_root)
        if out == input_root or input_root in out.parents:
            raise ValueError("comparison output cannot be inside a sealed input directory")
    if out.exists():
        raise FileExistsError("comparison requires a new output directory")
    argument_records = {str(io._no_links(path)): io._record(path) for path in
        (args.run_spec, args.combined_contract, args.huginn_manifest, args.huginn_calibration, args.eligibility)}
    authored = {str(path): io._record(path) for path in (Path(__file__), HERE / "HUGINN_COMPARISON_CONTRACT.md",
                                                       HERE / "analyze_refits.py", HERE / "CONTRACT.md")}
    inputs = load_inputs(args)
    for path, record in argument_records.items():
        io._verify_record(Path(path), record)
    contract = inputs["contract"]
    if argument_records[str(io._no_links(args.run_spec))]["sha256"] != contract["spec_sha256"]:
        raise ValueError("run specification changed between capture and semantic loading")
    sources = {str(path): {"bytes": path.stat().st_size, "sha256": next(row["sha256"] for row in contract["sources"]
        if (row["root"], row["path"]) == key)} for key, path in contract["source_paths"].items()}
    for path, record in {**authored, **argument_records}.items():
        if path in sources and sources[path] != record:
            raise ValueError("captured source document conflicts with the declared frozen record")
        sources[path] = record
    design = bootstrap_design(inputs["rows"])
    result, saved = calculate(inputs, design)
    selected_examples = examples(inputs, saved)
    identity = {"kind": "huginn_ouro_fit01_common_population", "fit_ids": {"ouro": 1, "huginn": 1},
                "run_spec_sha256": contract["spec_sha256"], "evaluation_base_seeds": list(SEEDS),
                "eligibility_file_sha256": ELIGIBILITY_SHA256, "bootstrap_seed": BOOT_SEED, "bootstrap_draws": N_BOOT,
                "example_seed": EXAMPLE_SEED, "readout_failure_seed": FAILURE_SEED,
                "source_records": sources, "evaluation_artifacts": inputs["artifacts"],
                "python": sys.version, "numpy": np.__version__}
    report = {"schema": SCHEMA, "created_utc": io._utc_now().isoformat(), "prediction_verbatim": PREDICTION,
        "counts": COUNTS, "learned_sources": LEARNED, "displayed_sources": SUPPORT, "seven_sources": GRID,
        "region_order": REGIONS, "native_exit_indices": EXITS, "known_identity": {"ouro": [191], "huginn": []},
        "uncertainty": {"draws": N_BOOT, "seed": BOOT_SEED, "modes": MODES,
            "seed_streams": {"items": [1, "task_index"], "components": [2, "task_index"]},
            "paired_across": "models, methods, metrics, locations and both fixed Huginn seeds",
            "fit_resampling": False, "seed_resampling": False,
            "interval_rule": "percentile pointwise95; unstudentized max-absolute-centered-error simultaneous95 over six seed-average contrasts",
            "limitations": "conditional on one N100 fit per model and two fixed Huginn initializations; component ratio means preserve item weighting"},
        "interpretation": {"same_opportunity": "seven fixed sources; fractions do not equate compute, native recurrence or semantic depth",
            "coda": "trained nonlinear decoder; only pass-end coda readouts are native exits; other cells are interrupted-core interventions",
            "target": "Ouro191 is self-identity; Huginn31 remains a strict source upstream of target33 after the coda",
            "native_agreement": "within-model token IDs on jointly eligible items; next-token agreement is not whole-answer correctness",
            "unavailable": "no Huginn per-source full-distribution KL or top-k overlap; no cross-vocabulary token-ID comparison",
            "claim_scope": "descriptive readout effects on the frozen tasks and horizons; no causal supervision inference or general model ranking"},
        "examples": "examples.json", **result}
    io._no_links(out).mkdir(parents=True, exist_ok=False)
    io._no_links(out)
    io._new_json(out / "OWNER.json", {"schema": SCHEMA, "identity": identity, "identity_sha256": io._digest(identity)})
    io._new_json(out / "report.json", ouro.plain(report))
    io._new_json(out / "population.json", inputs["plan"])
    io._new_json(out / "examples.json", ouro.plain(selected_examples))
    evaluation._write_npz(out / "paired_items.npz", saved)
    write_tables(out, result)
    for path, record in {**inputs["consumed"], **sources}.items():
        io._verify_record(Path(path), record)
    records = {name: io._record(out / name) for name in OUTPUT_FILES}
    io._new_json(out / "COMPLETE.json", {"schema": SCHEMA, "identity_sha256": io._digest(identity), "files": records})
    return {**validate_output(out), "output": str(out), "scientific_claims": "review the complete retained estimates and limitations"}


def _synthetic_huginn_arrays(plan):
    """Deterministic rank fixtures; these are not model readouts."""
    np = np_module()
    result = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        bank = {}
        for method in METHODS["huginn"]:
            ranks = np.full((148, 128, 32), -1, np.int32)
            own = np.full((148, 3, 32), -1, np.int32)
            for index, row in enumerate(plan["rows"]):
                common = plan["common_name_indices"][row["task"]]
                ranks[index, common] = rng.integers(0, 90, (len(common), 32), dtype=np.int32)
                for slot, target in enumerate(row["own_index"]):
                    if target >= 0:
                        own[index, slot] = ranks[index, target]
            bank[method + "_allrank"], bank[method + "_rank"] = ranks, own
            bank[method + "_top1"] = rng.integers(0, 2, (148, 32), dtype=np.int64)
        bank["native_top1"] = np.zeros(148, np.int64)
        bank["coda_top1"][:, 31] = 0
        result[seed] = bank
    return result


def _sealed_huginn_fixture(root, args, main, combined, manifest, calibration, plan, arrays, prior):
    """Symbolic N100 provenance and compact CPU tensors; no fitted-map claim."""
    import torch
    import run_huginn as implementation
    declared = {(row["root"], row["path"]): row["sha256"] for row in main["sources"]}
    snapshot = {row["path"]: {key: row[key] for key in ("bytes", "sha256")} for row in manifest["files"]}
    import_keys = {**evaluation._fit_import_keys(), "huginn_adapter":
        ("jlens", (ouro.DEPLOYMENT / "huginn_adapter.py").relative_to(ouro.REPO).as_posix())}
    fit = main["fits"][0]
    runtime = {"run_spec_sha256": main["spec_sha256"], "combined_contract_sha256": io._file_hash(args.combined_contract),
        "huginn_calibration_sha256": io._file_hash(args.huginn_calibration),
        "model": {key: combined["huginn"][key] for key in ("repo_id", "revision", "manifest_sha256")},
        "snapshot_path": "/symbolic-cpu-fixture/no-checkpoint", "snapshot_files": snapshot,
        "source_files": main["sources"], "environment": main["environment"],
        "gpu": {"name": "symbolic CPU fixture; no inference"}, "precision": {"symbolic_cpu_fixture": True},
        "settings": implementation.SETTINGS,
        "initialization": {"base_seed": 2026090802, "seed_namespace": "calibration",
                           "base_seed_rule": "base_seed + zero-based paragraph index", "prompt_seed_rule": huginn.SEED_RULE},
        "prerequisites": prior, "attention_implementation": "sdpa",
        "calibration": {"origin": {key: value for key, value in fit.items() if key != "prompts"}, "huginn": calibration},
        "imported_sources": [{"module": module, "path": "/deleted/producer/" + module, "sha256": declared[key]}
                             for module, key in import_keys.items()],
        "remote_implementations": [{"class": "symbolic." + name, "path": "/deleted/producer/" + name,
                                     "sha256": snapshot[name]["sha256"]}
                                    for name in ("raven_modeling_minimal.py", "raven_config_minimal.py")]}
    fit_identity = io._fit_identity(fit["prompts"], [row["token_length"] for row in calibration["rows"]],
        [row["n_valid"] for row in calibration["rows"]], 1, runtime, 5280, tuple(range(32)), False,
        production_profile="huginn_r8")
    fit_owner = {"schema_version": 1, "identity": fit_identity, "fit_identity_sha256": io._digest(fit_identity)}
    encoded = io._canonical(fit_owner) + b"\n"
    pointer = {"n_done": 100, "next_idx": 100, "fit_identity_sha256": fit_owner["fit_identity_sha256"]}
    record = {"fit_id": 1, "n_prompts": 100, "identity": fit_identity,
        "fit_identity_sha256": fit_owner["fit_identity_sha256"],
        "owner": {"bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()},
        "checkpoint_pointer": pointer, "complete_pointer": pointer,
        "lens": {"bytes": 1, "sha256": "e" * 64},
        "initialization_validation": {"rows": 100, "initializations_sha256": io._digest(calibration["rows"])}}
    identity = {"kind": "huginn_r8_n100_readout", "run_spec_sha256": main["spec_sha256"],
        "combined_contract_sha256": io._file_hash(args.combined_contract), "source_files": main["sources"],
        "fit": record, "model": runtime["model"], "eligibility_file": io._record(args.eligibility),
        "eligibility_payload_sha256": io._digest(plan), "evaluation_base_seeds": list(SEEDS),
        "sections": huginn.SECTIONS, "methods": list(huginn.METHODS), "learned_sources": SUPPORT["huginn"],
        "target_layer": 33, "position": -1, "evaluation_max_length": 512, "symbolic_cpu_fixture": True}
    out = evaluation._new_output(root, identity, schema=huginn.SCHEMA)
    directory = io._mkdir(out / "population")
    io._new_json(directory / "eligibility.json", plan)
    sections = [evaluation._seal_section(out, "population", {"kind": "frozen_common_eligibility", "counts": COUNTS})]
    modules = ("ouro_jlens", "ouro_jlens.evaluate", "ouro_jlens.evaldata", "ouro_jlens.fit_lens",
               "ouro_jlens.evidence", "ouro_jlens.recurrent", "jlens", "jlens._logging", "jlens.fitting",
               "jlens.hooks", "jlens.hf", "jlens.lens", "jlens.protocol", "jlens.vis", "huginn_adapter")
    imported = []
    for module in modules:
        key = (("jlens", (ouro.DEPLOYMENT / "huginn_adapter.py").relative_to(ouro.REPO).as_posix())
               if module == "huginn_adapter" else ("ouro_src" if module.startswith("ouro_jlens") else "jlens",
                   module.replace(".", "/") + ("/__init__.py" if "." not in module else ".py")))
        imported.append({"module": module, "root": key[0], "path": key[1], "sha256": declared[key]})
    evaluation_runtime = {key: runtime[key] for key in ("environment", "gpu", "precision", "remote_implementations",
                                                      "attention_implementation", "snapshot_files")}
    evaluation_runtime.update(imported_sources=imported, snapshot_path_at_evaluation="/symbolic-cpu-fixture/no-checkpoint")
    for seed in SEEDS:
        section = "seeds/" + str(seed)
        directory = io._mkdir(out / section)
        evaluation._write_npz(directory / "arrays.npz", arrays[seed])
        scalar = torch.zeros(1, dtype=torch.float32)
        evaluation._write_torch(directory / "cache.pt", {"H": scalar.expand(148, 32, 5280),
            "target_states": scalar.expand(148, 5280), "native_logits": scalar.expand(148, 65536)})
        initializations = []
        for index, row in enumerate(plan["rows"]):
            recipe = {"version": 1, "base_seed": seed, "namespace": "evaluation", "input_ids": row["huginn"]["token_ids"]}
            initializations.append({"index": index, "name": row["name"], "task": row["task"],
                "base_seed": seed, "namespace": "evaluation", "prompt_state_seed": int.from_bytes(bytes.fromhex(io._digest(recipe))[:8], "big") % (2 ** 63),
                "input_ids_sha256": io._digest(row["huginn"]["token_ids"]), "seed_recipe": huginn.SEED_RULE})
        io._new_json(directory / "initializations.json", initializations)
        io._new_json(directory / "summaries.json", huginn._summaries(arrays[seed], plan))
        sections.append(evaluation._seal_section(out, section, {"status": "complete", "n_items_done": 148, "seed": seed,
            "runtime": evaluation_runtime, "position": -1, "virtual_indices": SUPPORT["huginn"], "native_exit_indices": EXITS["huginn"],
            "coda_policy": "full raw sequence -> pre-coda norm -> two coda blocks -> final norm/head; select last logits afterwards",
            "within_pass_coda": "diagnostic interrupted-core intervention; only physical block3 is a native recurrence exit",
            "readout_policy": "native raw and JLens last-state vectors use final norm/head; maps are FP32 at application",
            "matrix_policy": "one preselected N100 map bank; the two seeds measure state-initialization sensitivity"}))
    evaluation._complete_output(out, sections)
    return out, evaluation_runtime


def self_test(historical_main):
    """Literal CPU score reductions and sealed symbolic inputs, never a model."""
    import torch
    from unittest.mock import patch
    np = np_module()
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU proof must start without CUDA initialization")
    checks, comparisons, maximum = [], 0, 0.0
    def require(condition, name):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    def close(left, right, name, tolerance=3e-14):
        nonlocal comparisons, maximum
        a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise AssertionError(name + ": invalid comparison arrays")
        error = float(np.max(np.abs(a - b))) if a.size else 0.0
        comparisons += a.size
        maximum = max(maximum, error)
        require(error <= tolerance, name)
    def rejects(action, name):
        try:
            action()
        except (ValueError, OSError, KeyError):
            checks.append(name)
        else:
            raise AssertionError("did not reject: " + name)
    def direct(ranks, rows, model):
        support = SUPPORT[model]
        result = {"eligible": np.asarray([any(row["eligible"]) for row in rows]),
                  **{field + "_layer": np.zeros((len(rows), len(support))) for field in ("own", "control")},
                  **{field + "_regions": np.zeros((len(rows), 3)) for field in ("own", "control")}}
        for index, row in enumerate(rows):
            slots = [slot for slot, yes in enumerate(row["eligible"]) if yes]
            for slot in slots:
                own = ranks[index, row["own_index"][slot]] < 10
                ctr = ranks[index, row["control_indices"][slot]] < 10
                result["own_layer"][index] += own.astype(float) / len(slots)
                result["control_layer"][index] += ctr.mean(axis=0) / len(slots)
                for region, cells in enumerate((SUPPORT[model], LEARNED[model], GRID[model])):
                    result["own_regions"][index, region] += float(any(own[cells])) / len(slots)
                    result["control_regions"][index, region] += sum(bool(any(hit[cells])) for hit in ctr) / len(ctr) / len(slots)
        for kind in ("layer", "regions"):
            result["delta_" + kind] = result["own_" + kind] - result["control_" + kind]
        return result

    plan = io._json(ouro.DEPLOYMENT / "huginn_eligibility.json")
    require(io._file_hash(ouro.DEPLOYMENT / "huginn_eligibility.json") == ELIGIBILITY_SHA256, "pre-outcome common eligibility byte identity")
    historical_main = io._no_links(historical_main)
    old_rows = io._json(historical_main / "items.json")
    names = io._json(historical_main / "task_names.json")
    forms = plan["token_forms"]["ouro"]
    original_rows = ouro.population(old_rows, names, forms, stored_labels=False)
    # Old archives predate explicit token-ID fields. Their frozen native input
    # identity is checked above; the pre-inference plan supplies those fields.
    for old, row in zip(original_rows, plan["rows"]):
        old.update({key: row["ouro"][key] for key in ("token_ids", "intermediate_tokens")})
    rows = population(plan, original_rows, names, forms)
    require({task: sum(row["task"] == task and any(row["eligible"]) for row in rows) for task in TASKS} == COUNTS["eligible_items"],
            "joint population has88 multihop and51 arithmetic eligible items")
    changed = copy.deepcopy(plan)
    changed["rows"][0]["eligible"][0] = not changed["rows"][0]["eligible"][0]
    rejects(lambda: population(changed, original_rows, names, forms), "joint eligibility cannot be overridden")
    changed = copy.deepcopy(plan)
    changed["common_name_indices"]["multihop"].append(37)
    rejects(lambda: population(changed, original_rows, names, forms), "unsupported multihop name37 cannot enter common controls")
    changed = copy.deepcopy(plan)
    changed["rows"][0]["control_indices"][0] = []
    rejects(lambda: population(changed, original_rows, names, forms), "stored controls cannot replace the common matched catalogue")
    manifest = io._json(ouro.HISTORICAL / "results/inputs.json")
    old_hashes = {Path(row["path"]).name: row["sha256"] for row in manifest["input_files"]
                  if Path(row["path"]).parent == ouro.HISTORICAL_MAIN}
    for filename in ("arrays.npz", "items.json", "task_names.json"):
        require(io._file_hash(historical_main / filename) == old_hashes[filename], "retained historical SHA256 " + filename)
    archive_arrays = {}
    with np.load(historical_main / "arrays.npz", allow_pickle=False) as archive:
        for method, prefix in (("raw", "logitlens_"), ("jlens", "jlens_exit3_")):
            archive_arrays[method] = {key: archive[prefix + key].copy() for key in ("allrank", "rank", "top1")}
            bank, expected = score(archive_arrays[method]["allrank"], rows, plan, "ouro"), direct(archive_arrays[method]["allrank"], rows, "ouro")
            for field in expected:
                close(bank[field], expected[field], "historical Ouro common-name reweight " + method + " " + field)
    synthetic = _synthetic_huginn_arrays(plan)
    for seed in SEEDS:
        huginn._validate_arrays(synthetic[seed], plan)
        for method in METHODS["huginn"]:
            ranks = synthetic[seed][method + "_allrank"]
            bank, expected = score(ranks, rows, plan, "huginn"), direct(ranks, rows, "huginn")
            for field in expected:
                close(bank[field], expected[field], "sparse original-name Huginn rank reduction " + str(seed) + " " + method + " " + field)
    require(GRID == {"ouro": [23,47,71,95,119,143,167], "huginn": [3,7,11,15,19,23,27]}, "seven source opportunities match the literal frozen grid")
    design = bootstrap_design(rows)
    require([len(np.unique(design[task]["groups"])) for task in TASKS] == [62, 14], "components rebuilt on joint eligibility have62/14 groups")
    require(all(design[task]["item_indices"].shape == (N_BOOT, COUNTS["eligible_items"][task]) for task in TASKS), "all10000 paired item draws retained")
    for task in TASKS:
        sample = design[task]
        values = np.arange(len(sample["indices"]), dtype=float) ** 2
        literal = np.asarray([values[draw].mean() for draw in sample["item_indices"][:31]])
        close(sample["item_weights"][:31] @ values, literal, "literal paired item draws " + task, tolerance=2e-12)
        literal = np.asarray([np.concatenate([values[sample["groups"] == group] for group in draw]).mean()
                              for draw in sample["component_indices"][:31]])
        close(sample["component_weights"][:31] @ values, literal, "whole-component ratio weighting " + task, tolerance=2e-12)

    # Two-item toy makes averaging seedwise any-events distinguishable from
    # pooling/unioning opportunities over seeds, and makes source31 nonidentity.
    toy_plan = {"task_names": {task: ["own", "c1", "c2", "unsupported"] for task in TASKS},
                "common_name_indices": {task: [0, 1, 2] for task in TASKS}}
    toy_rows = [{"task": task, "intermediates": ["own"], "own_index": [0,-1,-1],
                 "eligible": [True], "control_indices": [[1,2]]} for task in TASKS]
    def toy_rank(model):
        value = np.full((2,128,len(SUPPORT[model])), -1, np.int32)
        value[:, :3] = 20
        return value
    miss = toy_rank("huginn")
    hits = [miss.copy(), miss.copy()]
    hits[0][0,0,3], hits[1][1,0,7] = 9, 9
    any_by_seed = np.stack([method_metrics(score(value, toy_rows, toy_plan, "huginn"), "huginn", "own")[:,2] for value in hits])
    close(any_by_seed.mean(axis=0), [0.5,0.5], "two fixed seeds average separate any-of-seven events")
    require(np.all((np.stack(hits)[:,:,0][:,:,GRID["huginn"]] < 10).any(axis=(0,2))), "toy union differs from the required seedwise average")
    controls_hit = miss.copy()
    controls_hit[0,1,3], controls_hit[0,2,7] = 9, 9
    bank = score(controls_hit, toy_rows, toy_plan, "huginn")
    require(bank["control_regions"][0,2] == 1.0 and bank["control_layer"][0].max() == 0.5,
            "each control takes any-of-seven before averaging controls")
    endpoint = toy_rank("ouro")
    endpoint[:,0,191] = 9
    close(method_metrics(score(endpoint, toy_rows, toy_plan, "ouro"), "ouro", "own"), np.zeros((2,3)), "Ouro191 identity endpoint excluded from all three learned comparisons")
    endpoint = toy_rank("huginn")
    endpoint[:,0,31] = 9
    close(method_metrics(score(endpoint, toy_rows, toy_plan, "huginn"), "huginn", "own"), [[1/32,0,0],[1/32,0,0]], "Huginn31 remains a learned source but is outside seven-point sensitivity")

    args = SimpleNamespace(run_spec=ouro.DEPLOYMENT / "run_spec.json", combined_contract=ouro.DEPLOYMENT / "combined_contract.json",
        huginn_manifest=ouro.DEPLOYMENT / "huginn_model_manifest.json", huginn_calibration=ouro.DEPLOYMENT / "huginn_calibration.json",
        eligibility=ouro.DEPLOYMENT / "huginn_eligibility.json", ouro_src=Path("/home/moloch/ouro_project/src"))
    contract = io._contract(args.run_spec, args.ouro_src, ouro.FIT_IDS)
    _, combined, model_manifest, calibration = huginn._contract(args)
    with tempfile.TemporaryDirectory(prefix="huginn-comparison-cpu-") as temporary:
        root = Path(temporary)
        original_args = ouro._sealed_cpu_fixture(root / "ouro-fixture", original_rows, names, archive_arrays["raw"], contract)
        args.main, args.controls = original_args.main, original_args.controls
        original = ouro.load_inputs(args)
        prior = _prior_records(original["artifacts"])
        args.huginn, runtime = _sealed_huginn_fixture(root / "huginn", args, contract, combined, model_manifest, calibration, plan, synthetic, prior)
        args.output = root / "analysis"
        loaded = load_inputs(args)
        require(loaded["plan"] == plan, "unmocked completed Ouro/control/Huginn semantic readers accept the bound synthetic bundle")
        result, saved = calculate(loaded, design)
        close(saved["cross_model_seed_average_metrics"], saved["cross_model_paired_metrics"].mean(axis=0), "cross-model item effects average both fixed seeds after scoring")
        require(result["comparison"]["family"]["contrasts"] == 6, "exact six task-by-metric seed-average family")
        for task in TASKS:
            sample = design[task]
            values = saved["cross_model_seed_average_metrics"][sample["indices"]]
            close(result["comparison"]["tasks"][task]["mean"], values.mean(axis=0), "cross-model task effect uses equal item weights " + task)
        points = {task: result["comparison"]["tasks"][task]["mean"] for task in TASKS}
        for mode, weight_key in (("conditional_item", "item_weights"), ("conditional_component", "component_weights")):
            samples = {task: design[task][weight_key] @ saved["cross_model_seed_average_metrics"][design[task]["indices"]] for task in TASKS}
            joined = np.concatenate([samples[task] - points[task] for task in TASKS], axis=1)
            expected_radius = np.quantile(np.max(np.abs(joined), axis=1), .95)
            close(result["comparison"]["family"]["simultaneous_radius95"][mode], expected_radius, "six-contrast max-centered family " + mode)
            for task in TASKS:
                close(result["comparison"]["tasks"][task]["intervals"][mode]["pointwise95"],
                      np.quantile(samples[task],[.025,.975],axis=0).T, "percentile conditional interval " + task + " " + mode)
        first = examples(loaded, saved)
        require(first["random"]["selected_indices"].tolist() == [13,36,90,48,133,24,110,71,111,0],
                "ten uniform examples match their frozen pre-outcome metadata selection")
        shifted = {key: value + 100 if key == "cross_model_paired_metrics" else value for key, value in saved.items()}
        require(np.array_equal(first["random"]["selected_indices"], examples(loaded, shifted)["random"]["selected_indices"]), "uniform random examples do not depend on effect values")
        for task in TASKS:
            pool = first["readout_failures"]["tasks"][task]["eligible_readout_failure_indices"]
            require(all(rows[int(index)]["task"] == task and all(saved[f"huginn_{seed}_jlens_own_metrics"][index,2] == 0 for seed in SEEDS) for index in pool),
                    "failure pool is a paired concept-readout miss, not answer correctness " + task)
        changed = copy.deepcopy(runtime)
        a = next(row for row in changed["imported_sources"] if row["module"] == "jlens.hf")
        b = next(row for row in changed["imported_sources"] if row["module"] == "jlens.lens")
        for key in ("root", "path", "sha256"):
            a[key], b[key] = b[key], a[key]
        rejects(lambda: _runtime(changed, contract, model_manifest), "swapped valid imported file records cannot change module-to-path association")
        changed = copy.deepcopy(runtime)
        a = next(row for row in changed["imported_sources"] if row["module"] == "jlens.hf")
        b = next(row for row in changed["imported_sources"] if row["module"] == "jlens.lens")
        a.update({key:b[key] for key in ("root","path","sha256")})
        rejects(lambda: _runtime(changed, contract, model_manifest), "duplicated valid source file cannot impersonate a different module")
        checked = evaluation.validate_output(args.huginn, schema=huginn.SCHEMA)
        for key, bad in (("run_spec_sha256", "f"*64), ("source_files", []), ("target_layer",31), ("evaluation_base_seeds",[SEEDS[0]])):
            changed = copy.deepcopy(checked)
            changed["owner"]["identity"][key] = bad
            rejects(lambda: bind_huginn(changed,args.huginn,args,contract,combined,model_manifest,calibration,plan,prior), "Huginn semantic identity rejects changed " + key)
        changed_prior = copy.deepcopy(prior)
        changed_prior["ouro"]["owner_file"]["sha256"] = "f"*64
        rejects(lambda: bind_huginn(checked,args.huginn,args,contract,combined,model_manifest,calibration,plan,changed_prior), "Huginn fit cannot pair with a different sealed Ouro prerequisite")
        result = analyze(args)
        require(result["status"] == "passed" and result["files"] == 9, "complete10000-draw CPU analysis writes and validates all nine sealed artifacts")
        report = io._json(args.output / "report.json")
        require(report["prediction_verbatim"] == PREDICTION and report["comparison"]["family"]["contrasts"] == 6,
                "output retains exact user prediction and six-contrast family")
        require((args.output / "examples.json").is_file(), "random examples and defined readout failures are retained")
        rejects(lambda: analyze(args), "completed comparison cannot be overwritten")
        for input_root in (args.main,args.controls,args.huginn):
            nested = copy.copy(args)
            nested.output = input_root / "forbidden-analysis"
            rejects(lambda: analyze(nested), "output cannot modify sealed input tree " + input_root.name)
            require(not nested.output.exists(), "rejected nested output leaves input untouched " + input_root.name)
        local_document = root / "argument-combined.json"
        local_document.write_bytes(args.combined_contract.read_bytes())
        changed_document = copy.copy(args)
        changed_document.combined_contract = local_document
        changed_document.output = root / "changed-argument-output"
        original_loader = load_inputs
        def mutate_after_load(received):
            # The Huginn source binder intentionally requires canonical source
            # paths. Validate the identical canonical document, then mutate
            # only this temporary argument copy after the real reader returns.
            actual = copy.copy(received)
            actual.combined_contract = args.combined_contract
            require(local_document.read_bytes() == args.combined_contract.read_bytes(),
                    "temporary argument document matches the bytes used by the real semantic reader")
            loaded_value = original_loader(actual)
            with local_document.open("ab") as handle:
                handle.write(b"\n")
            return loaded_value
        with patch(__name__ + ".load_inputs", side_effect=mutate_after_load):
            rejects(lambda: analyze(changed_document), "captured argument mutation after semantic loading is rejected")
        require(not (changed_document.output / "COMPLETE.json").exists(),
                "a changed argument document cannot replace its captured expected hash")
        metadata_path = args.huginn / "seeds" / str(SEEDS[0]) / "metadata.json"
        original_bytes = metadata_path.read_bytes()
        original_writer = write_tables
        unstable = copy.copy(args)
        unstable.output = root / "unstable-output"
        def mutate_after_tables(*values):
            original_writer(*values)
            with metadata_path.open("ab") as handle:
                handle.write(b"\n")
        try:
            with patch(__name__ + ".write_tables", side_effect=mutate_after_tables):
                rejects(lambda: analyze(unstable), "Huginn metadata mutation during analysis prevents completion")
            require(not (unstable.output / "COMPLETE.json").exists(), "changed consumed metadata leaves no completed analysis claim")
        finally:
            metadata_path.write_bytes(original_bytes)
        (args.output / "extra").write_text("unsealed CPU fixture")
        rejects(lambda: validate_output(args.output), "unsealed output extra rejected")
        (args.output / "extra").unlink()
        with (args.output / "report.json").open("ab") as handle:
            handle.write(b"changed")
        rejects(lambda: validate_output(args.output), "changed output bytes rejected")
        (args.huginn / "COMPLETE.json").unlink()
        rejects(lambda: load_inputs(args), "partial Huginn evaluation cannot become the comparison")
    require(not torch.cuda.is_initialized(), "CUDA remained uninitialized; no model/checkpoint was loaded")
    return {"status": "passed", "schema": "huginn_comparison_cpu_proof.v1", "check_count": len(checks),
        "checks": checks, "scalar_comparisons": int(comparisons), "maximum_absolute_error": maximum,
        "scope": "CPU toy scores, retained historical Ouro rank reweight, and synthetic sealed Huginn fixtures only; no new model outcome",
        "bootstrap_draws": N_BOOT, "fixed_random_example_indices": first["random"]["selected_indices"].tolist(),
        "common_component_counts": {task: len(np.unique(design[task]["groups"])) for task in TASKS},
        "cuda_initialized": False, "source_records": {name:io._record(HERE/name) for name in
            ("analyze_huginn.py","HUGINN_COMPARISON_CONTRACT.md","analyze_refits.py","CONTRACT.md")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan")
    check = commands.add_parser("validate")
    check.add_argument("output", type=Path)
    test = commands.add_parser("self-test")
    test.add_argument("--historical-main", type=Path, default=ouro.HISTORICAL_MAIN)
    test.add_argument("--proof", type=Path)
    run = commands.add_parser("analyze")
    for name in ("main", "controls", "huginn", "output", "ouro-src"):
        run.add_argument("--" + name, type=Path, required=True)
    for flag, filename in (("run-spec", "run_spec.json"), ("combined-contract", "combined_contract.json"),
                           ("huginn-manifest", "huginn_model_manifest.json"), ("huginn-calibration", "huginn_calibration.json"),
                           ("eligibility", "huginn_eligibility.json")):
        run.add_argument("--" + flag, type=Path, default=ouro.DEPLOYMENT / filename)
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = {"schema": SCHEMA, "prediction_verbatim": PREDICTION, "counts": COUNTS,
                  "fit_ids": {"ouro": 1, "huginn": 1}, "evaluation_base_seeds": list(SEEDS),
                  "learned_sources": LEARNED, "seven_sources": GRID, "metric_ids": METRICS,
                  "bootstrap_draws": N_BOOT, "bootstrap_seed": BOOT_SEED, "simultaneous_family": 6,
                  "output_files": sorted(OUTPUT_FILES | {"COMPLETE.json"}), "new_inference": False, "cloud_actions": 0}
    elif args.command == "validate":
        result = validate_output(args.output)
    elif args.command == "self-test":
        result = self_test(args.historical_main)
        if args.proof is not None:
            io._new_json(args.proof, result)
    else:
        result = analyze(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
