"""CPU analysis of the five sealed Ouro refits and conditional fit01 controls.

See CONTRACT.md for frozen scores, locations, uncertainty families and examples.
Plan/help import no numerical framework; self-test uses historical data and CPU
fixtures only. Analyze writes a new immutable output directory.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
DEPLOYMENT = ROUND / "deployment"
REPO = ROUND.parents[1]
HISTORICAL = REPO / "research/followup_2026-09-07"
HISTORICAL_MAIN = Path("/home/moloch/ouro_project/artifacts/jlens/retrieved/jlens-b300-20260905-0359/eval/n100_exit3")
sys.path.insert(0, str(DEPLOYMENT))
import run_refits as io
import evaluate_refits as evaluation
import evaluate_controls as controls

SCHEMA = "ouro_refit_analysis.v1"
FIT_IDS = [1, 2, 3, 4, 5]
TASKS = ("multihop", "order-ops")
N_BOOT = 10000
SEED = 2026090805
EXAMPLE_SEED = 2026090806
FAILURE_SEED = 2026090807
OPERATIONS = {"addition", "subtraction", "multiplication", "division", "mod", "squared"}
NUMBER_WORDS = dict(zip("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(), range(21)))
NUMBER_WORDS["third"] = 3
# All endpoints here are one-based and inclusive.
BANDS = {"early": (1, 16), "middle": (17, 32), "final_third": (33, 48),
         "historical_local": (26, 37), "full_loop": (1, 48)}
PAIRS = (
    "target_main_minus_penultimate", "target_main_minus_raw", "target_penultimate_minus_raw",
    "position_sampled_sum_minus_diagonal", "position_main_minus_sampled_sum",
    "position_main_minus_raw", "position_sampled_sum_minus_raw", "position_diagonal_minus_raw",
)
MODES = ("crossed_fit_item", "conditional_item", "conditional_fit", "crossed_fit_component")
OUTPUT_FILES = {"OWNER.json", "report.json", "control_comparisons_summary.json", "examples.json",
                "paired_items.npz", "main_cells.csv", "main_metrics.csv", "per_fit_metrics.csv", "control_metrics.csv"}


def numpy():
    import numpy as np
    return np


def _same(left, right):
    return io._canonical(left) == io._canonical(right)


def historical_correct(continuation, target):
    """The frozen producer's token-boundary prefix rule, without importing torch."""
    value, answer = continuation.strip().strip('"').lower(), target.strip().lower()
    if not value.startswith(answer):
        return False
    rest = value[len(answer):]
    return rest == "" or not rest[0].isalnum()


def population(rows, names, forms=None, *, stored_labels=True):
    """Bind actual historical rows, then reconstruct every scoring decision."""
    if (not isinstance(rows, list) or len(rows) != 148
            or io._digest([{key: row[key] for key in evaluation.POPULATION_FIELDS} for row in rows]) != evaluation.POPULATION_SHA256
            or io._digest(names) != evaluation.TASK_NAMES_SHA256
            or (forms is not None and io._digest(forms) != evaluation.TOKEN_FORMS_SHA256)):
        raise ValueError("actual item population, task names or token forms changed")
    result = []
    for row in rows:
        catalogue = names[row["task"]]
        own = {index for index in row["own_index"] if index >= 0}
        for slot, index in enumerate(row["own_index"]):
            if index >= 0 and (slot >= len(row["intermediates"]) or catalogue[index] != row["intermediates"][slot]):
                raise ValueError("own indices do not identify the frozen labels")
        eligible, pools = [], []
        for slot, label in enumerate(row["intermediates"]):
            eligible.append(bool(row["scorable"][slot]) and not row["leaked"][slot]
                            and (row["task"] == "multihop" or label not in OPERATIONS))
            pools.append([index for index, other in enumerate(catalogue)
                          if index not in own and ((other in OPERATIONS) == (label in OPERATIONS))]
                         if row["scorable"][slot] else [])
        if any(yes and not pool for yes, pool in zip(eligible, pools)):
            raise ValueError("eligible target lacks matched controls")
        if stored_labels and (not _same(row.get("eligible"), eligible) or not _same(row.get("control_indices"), pools)):
            raise ValueError("stored eligibility or matched controls differ from historical rules")
        if (not isinstance(row.get("continuation"), str) or type(row.get("correct")) is not bool
                or row["correct"] != historical_correct(row["continuation"], row["target"])):
            raise ValueError("stored correctness differs from the sealed continuation and original rule")
        result.append({**row, "eligible": eligible, "control_indices": pools})
    if ({task: sum(row["task"] == task and any(row["eligible"]) for row in result) for task in TASKS}
            != {"multihop": 90, "order-ops": 51}):
        raise ValueError("scoring denominator changed")
    return result


def concept_components(rows, indices):
    """Historical shared-eligible-concept components; preserve original item weights."""
    np = numpy()
    parents, seen = list(range(len(indices))), {}
    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    for local, index in enumerate(indices):
        row = rows[int(index)]
        for slot, yes in enumerate(row["eligible"]):
            if not yes:
                continue
            name = row["intermediates"][slot]
            label = str(NUMBER_WORDS.get(name.lower(), int(name) if name.isdigit() else name.lower()))
            if label in seen:
                parents[find(local)] = find(seen[label])
            else:
                seen[label] = local
    _, groups = np.unique([find(index) for index in range(len(indices))], return_inverse=True)
    return groups


def fixed_metrics(support, *, include_slope=False, regions=None):
    definitions = []
    for loop in range(4):
        for name, (first, last) in BANDS.items():
            selected = [index for index in range(loop * 48 + first - 1, loop * 48 + last) if index in support]
            if not selected:
                raise ValueError("fixed band lacks common source support")
            definitions.append({"id": f"loop{loop + 1}_{name}_mean", "kind": "fixed_mean", "loop": loop + 1,
                                "band": name, "virtual_indices": selected,
                                "historical_selected": name == "historical_local"})
    if regions is None:
        regions = {f"loop{loop + 1}_any_layer": list(range(loop * 48, (loop + 1) * 48)) for loop in range(4)}
    for region, selected in regions.items():
        if not selected or not set(selected) <= set(support):
            raise ValueError("any-layer metric lacks common source support")
        definitions.append({"id": region, "kind": "any_layer", "virtual_indices": selected})
    if include_slope:
        for loop in range(4):
            definitions.append({"id": f"loop{loop + 1}_depth_slope", "kind": "depth_slope", "loop": loop + 1,
                                "virtual_indices": list(range(loop * 48, (loop + 1) * 48))})
    return definitions


def metric_values(layer, region, support, definitions):
    np = numpy()
    columns = {index: column for column, index in enumerate(support)}
    values, region_index = [], 0
    for definition in definitions:
        selected = [columns[index] for index in definition["virtual_indices"]]
        if definition["kind"] == "fixed_mean":
            values.append(layer[..., selected].mean(axis=-1))
        elif definition["kind"] == "any_layer":
            values.append(region[..., region_index])
            region_index += 1
        else:
            x = np.linspace(0.0, 1.0, len(selected))
            x -= x.mean()
            values.append(layer[..., selected] @ x / (x @ x))
    if region_index != region.shape[-1]:
        raise ValueError("region scores do not match declared any-layer metrics")
    return np.stack(values, axis=-1)


def _membership(n, draws, seed, group=None):
    np = numpy()
    if type(n) is not int or n < 1 or type(draws) is not int or draws < 1:
        raise ValueError("bootstrap populations and draw counts must be positive integers")
    size = n if group is None else int(np.max(group)) + 1
    chosen = np.random.default_rng(seed).integers(0, size, size=(draws, size))
    counts = np.zeros((draws, size), dtype=np.float64)
    np.add.at(counts, (np.arange(draws)[:, None], chosen), 1.0)
    weights = counts if group is None else counts[:, group]
    return chosen, weights / weights.sum(axis=1, keepdims=True)


def bootstrap_design(rows, *, draws=N_BOOT):
    np = numpy()
    fits, fit_weights = _membership(5, draws, np.random.SeedSequence(SEED, spawn_key=(0,)))
    result = {"fit_indices": fits, "fit_weights": fit_weights, "tasks": {}}
    for task_index, task in enumerate(TASKS):
        indices = np.asarray([i for i, row in enumerate(rows) if row["task"] == task and any(row["eligible"])], dtype=np.int64)
        groups = concept_components(rows, indices)
        selected, weights = _membership(len(indices), draws, np.random.SeedSequence(SEED, spawn_key=(1, task_index)))
        selected_groups, component_weights = _membership(len(indices), draws, np.random.SeedSequence(SEED, spawn_key=(2, task_index)), groups)
        result["tasks"][task] = {"indices": indices, "groups": groups, "item_indices": selected,
                                 "component_indices": selected_groups, "item_weights": weights,
                                 "component_weights": component_weights}
    return result


def crossed_means(values, fit_weights, item_weights):
    """Same fit and item choices apply jointly to every already-paired metric."""
    np = numpy()
    if (values.ndim != 3 or fit_weights.shape[1] != values.shape[0]
            or item_weights.shape[1] != values.shape[1] or fit_weights.shape[0] != item_weights.shape[0]
            or not np.isfinite(values).all()):
        raise ValueError("crossed bootstrap axes or finite values are invalid")
    draws = len(fit_weights)
    result = np.empty((draws, values.shape[2]), dtype=np.float64)
    for begin in range(0, draws, 512):
        end = min(begin + 512, draws)
        weights = (fit_weights[begin:end, :, None] * item_weights[begin:end, None, :]).reshape(end - begin, -1)
        result[begin:end] = weights @ values.reshape(-1, values.shape[2])
    return result


def bootstrap(values, design):
    fit_weights = design["fit_weights"]
    result = {}
    for task in TASKS:
        bank, sample = values[task], design["tasks"][task]
        result[task] = {
            "crossed_fit_item": crossed_means(bank, fit_weights, sample["item_weights"]),
            "conditional_item": sample["item_weights"] @ bank.mean(axis=0),
            "conditional_fit": fit_weights @ bank.mean(axis=1),
            "crossed_fit_component": crossed_means(bank, fit_weights, sample["component_weights"]),
        }
    return result


def family_intervals(points, draws, modes):
    """Join tasks before selecting the centered-error simultaneous radius."""
    np = numpy()
    result, radii = {task: {} for task in TASKS}, {}
    for mode in modes:
        errors = np.stack([np.abs(draws[task][mode] - points[task]).max(axis=1) for task in TASKS])
        radius = float(np.quantile(errors.max(axis=0), 0.95))
        radii[mode] = radius
        for task in TASKS:
            result[task][mode] = {"pointwise95": np.quantile(draws[task][mode], [0.025, 0.975], axis=0).T,
                                  "simultaneous95": np.stack((points[task] - radius, points[task] + radius), axis=-1)}
    return result, radii


def fit_summary(values):
    np = numpy()
    if values.shape[0] != 5:
        raise ValueError("main fit summaries require all five calibration fits")
    per_fit = values.mean(axis=1)
    return {"per_fit": per_fit, "mean": per_fit.mean(axis=0), "fit_sd": per_fit.std(axis=0, ddof=1),
            "fit_min": per_fit.min(axis=0), "fit_max": per_fit.max(axis=0),
            "positive_fits": (per_fit > 0).sum(axis=0), "negative_fits": (per_fit < 0).sum(axis=0),
            "zero_fits": (per_fit == 0).sum(axis=0)}


def learned_peak(curve):
    np = numpy()
    values = np.asarray(curve)[144:191]
    if values.shape != (47,) or not np.isfinite(values).all():
        raise ValueError("loop4 peak requires all 47 learned columns")
    best = float(values.max())
    ties = (np.flatnonzero(values == best) + 144).tolist()
    positive = np.flatnonzero(values > 0) + 144
    runs = []
    for index in positive.tolist():
        if runs and runs[-1][1] + 1 == index:
            runs[-1][1] = index
        else:
            runs.append([index, index])
    return {"maximum": best, "tied_virtual_indices": ties,
            "tied_one_based_physical_layers": [index - 143 for index in ties],
            "positive_virtual_runs": runs,
            "positive_physical_runs": [[start - 143, stop - 143] for start, stop in runs],
            "interpretation": "descriptive learned-map locations; identity191 excluded; no selected-cell pointwise interval"}


def artifact_records(root, checked):
    records = {"OWNER.json": io._record(root / "OWNER.json"), "COMPLETE.json": io._record(root / "COMPLETE.json")}
    for section in checked["complete"]["sections"]:
        name = section["section"]
        records[name + "/SEAL.json"] = section["seal"]
        seal = io._json(root / name / "SEAL.json")
        records.update({name + "/" + relative: record for relative, record in seal["files"].items()})
    return records


def load_inputs(args):
    """Check complete scientific identities and actual bytes before numerical work."""
    np = numpy()
    contract = io._contract(args.run_spec, args.ouro_src, FIT_IDS)
    evaluation._evaluation_sources(contract)
    sources = {(row["root"], row["path"]): row["sha256"] for row in contract["sources"]}
    for name in ("run_refits.py", "evaluate_refits.py", "evaluate_controls.py", "run_controls.py"):
        path = DEPLOYMENT / name
        key = ("jlens", path.relative_to(REPO).as_posix())
        if key not in sources or io._file_hash(path) != sources[key]:
            raise ValueError("analysis helper is not the evaluator's frozen source: " + name)
    combined = io._json(io._no_links(args.combined_contract))
    if (controls.TARGET_SUPPORT != list(range(190)) or controls.POSITION_SUPPORT != list(range(191))
            or controls.TARGET_LATE != list(range(176, 190))
            or controls.POSITION_LATE != list(range(176, 191))):
        raise ValueError("control supports do not implement the corrected historical final-third intersection")
    main_root, control_root = io._no_links(args.main), io._no_links(args.controls)
    main_checked = controls.validate_completed_main(main_root, contract, require_all=True)
    control_checked = controls.validate_completed_controls(control_root, contract, combined, main_checked)
    roots = {"main": main_root, "controls": control_root}
    records = {"main": artifact_records(main_root, main_checked), "controls": artifact_records(control_root, control_checked)}
    consumed = {}
    def path(family, relative):
        record = records[family][relative]
        result = roots[family] / relative
        io._verify_record(result, record)
        consumed[str(result)] = record
        return result
    # The semantic gates also consume owners, section metadata and seals.
    # Keep their original records in the final input-stability check.
    for family in roots:
        for relative in records[family]:
            if relative.endswith(".json"):
                path(family, relative)
    rows = io._json(path("main", "common/items.json"))
    names = io._json(path("main", "common/task_names.json"))
    forms = io._json(path("main", "common/token_forms.json"))
    rows = population(rows, names, forms)
    for filename, expected in (("items.json", rows), ("task_names.json", names), ("token_forms.json", forms)):
        if not _same(io._json(path("controls", "comparisons/" + filename)), expected):
            raise ValueError("control comparison population is not paired with the main evaluation")
    def readout(family, relative, prefix, support):
        arrays = controls._load_arrays(path(family, relative), prefix)
        evaluation._validate_arrays(arrays, rows, names, support)
        return arrays
    support = list(range(192))
    regions = {f"loop{loop + 1}_any_layer": list(range(loop * 48, (loop + 1) * 48)) for loop in range(4)}
    raw_arrays = readout("main", "common/arrays.npz", "logitlens_", support)
    with np.load(path("main", "common/arrays.npz"), allow_pickle=False) as archive:
        exits = archive["exit_top1"]
        if exits.shape != (148, 4) or not np.array_equal(exits, raw_arrays["top1"][:, [47, 95, 143, 191]]):
            raise ValueError("raw boundary predictions differ from the sealed native exits")
    raw = controls.score_rank_arrays(raw_arrays["allrank"], rows, names, support, bands=regions)
    fit_scores, first_arrays = [], None
    for fit_id in FIT_IDS:
        arrays = readout("main", f"fits/fit_{fit_id:02d}/arrays.npz", "jlens_exit3_", support)
        for key in ("allrank", "rank", "top1"):
            if not np.array_equal(arrays[key][..., 191], raw_arrays[key][..., 191]):
                raise ValueError("main final target is not the declared raw identity reference")
        score = controls.score_rank_arrays(arrays["allrank"], rows, names, support, bands=regions)
        if not np.array_equal(score["eligible"], raw["eligible"]):
            raise ValueError("fit and raw eligible populations differ")
        fit_scores.append(score)
        if fit_id == 1:
            first_arrays = arrays
    fitted = {key: np.stack([score[key] for score in fit_scores]) for key in raw if key != "eligible"}
    control_arrays = {arm: readout("controls", section + "/arrays.npz", "", columns) for arm, section, columns in (
        ("penultimate", "penultimate", controls.TARGET_SUPPORT),
        ("sampled_sum", "positions/sampled_sum", controls.POSITION_SUPPORT),
        ("diagonal", "positions/diagonal", controls.POSITION_SUPPORT))}
    comparison_scores, comparison_summary = controls._comparisons(first_arrays, raw_arrays, control_arrays, rows, names)
    with np.load(path("controls", "comparisons/scores.npz"), allow_pickle=False) as archive:
        if set(archive.files) != set(comparison_scores) or any(not np.array_equal(archive[key], value) for key, value in comparison_scores.items()):
            raise ValueError("stored control scores differ from the sealed rank arrays")
    if not _same(io._json(path("controls", "comparisons/summary.json")), comparison_summary):
        raise ValueError("stored control comparison summary differs from recomputed scores")
    with np.load(path("controls", "comparisons/paired_ranks.npz"), allow_pickle=False) as archive:
        for comparison, columns in (("target", controls.TARGET_SUPPORT), ("position", controls.POSITION_SUPPORT)):
            for arm, arrays in (("main", first_arrays), ("raw", raw_arrays)):
                for key, value in arrays.items():
                    stored = f"{comparison}_{arm}_{key}"
                    if stored not in archive or not np.array_equal(archive[stored], value[..., columns]):
                        raise ValueError("retained paired comparison ranks differ from main fit01/raw")
    return {"contract": contract, "rows": rows, "names": names, "forms": forms, "raw": raw, "fits": fitted,
            "comparison_scores": comparison_scores, "comparison_summary": comparison_summary, "consumed": consumed,
            "artifacts": {family: {"root": str(roots[family]), "files": records[family]} for family in roots}}


def main_analysis(inputs, design):
    np = numpy()
    raw, fitted = inputs["raw"], inputs["fits"]
    paired_layer = fitted["delta_layer"] - raw["delta_layer"][None]
    paired_regions = fitted["delta_regions"] - raw["delta_regions"][None]
    definitions = fixed_metrics(list(range(192)), include_slope=True)
    paired_metrics = metric_values(paired_layer, paired_regions, list(range(192)), definitions)
    values, layer_stats, metric_stats = {}, {}, {}
    for task in TASKS:
        indices = design["tasks"][task]["indices"]
        values[task] = np.concatenate((paired_layer[:, indices], paired_metrics[:, indices]), axis=-1)
        layer_stats[task], metric_stats[task] = fit_summary(paired_layer[:, indices]), fit_summary(paired_metrics[:, indices])
    samples = bootstrap(values, design)
    intervals, families = {}, {}
    for name, selected, statistics in (("cells", slice(0, 192), layer_stats), ("metrics", slice(192, None), metric_stats)):
        subset = {task: {mode: samples[task][mode][:, selected] for mode in MODES} for task in TASKS}
        points = {task: statistics[task]["mean"] for task in TASKS}
        intervals[name], radius = family_intervals(points, subset, MODES)
        families[name] = {"contrasts": sum(len(point) for point in points.values()), "simultaneous_radius95": radius}
    result = {"metric_definitions": definitions, "families": families, "tasks": {}}
    for task in TASKS:
        indices = design["tasks"][task]["indices"]
        stats = layer_stats[task]
        result["tasks"][task] = {
            "eligible_items": len(indices), "eligible_slots": sum(sum(inputs["rows"][int(index)]["eligible"]) for index in indices),
            "cells": {**stats, "intervals": intervals["cells"][task]},
            "metrics": {**metric_stats[task], "intervals": intervals["metrics"][task]},
            "method_cells": {"jlens": {field: fit_summary(fitted[field][:, indices]) for field in ("own_layer", "control_layer", "delta_layer")},
                             "raw": {field: raw[field][indices].mean(axis=0) for field in ("own_layer", "control_layer", "delta_layer")}},
            "loop4_learned_peaks_per_fit": [{"fit_id": fit_id, **learned_peak(stats["per_fit"][position])}
                                             for position, fit_id in enumerate(FIT_IDS)],
            "loop4_learned_peak_of_fit_mean": learned_peak(stats["mean"]),
        }
        # A selected grand-mean peak gets only the prespecified full-cell-family band.
        peak = result["tasks"][task]["loop4_learned_peak_of_fit_mean"]
        peak["simultaneous95_at_tied_locations"] = {mode: intervals["cells"][task][mode]["simultaneous95"][peak["tied_virtual_indices"]]
                                                   for mode in MODES}
    saved = {"main_paired_delta_layer": paired_layer, "main_paired_delta_regions": paired_regions,
             "main_paired_metrics": paired_metrics,
             **{"main_jlens_" + key: value for key, value in fitted.items()},
             **{"main_raw_" + key: value for key, value in raw.items()}}
    return result, saved


def control_analysis(inputs, design):
    np = numpy()
    scores = inputs["comparison_scores"]
    definitions, banks = [], []
    for pair in PAIRS:
        target = pair.startswith("target_")
        support = controls.TARGET_SUPPORT if target else controls.POSITION_SUPPORT
        late = controls.TARGET_LATE if target else controls.POSITION_LATE
        declared = fixed_metrics(support, regions={"all_common_any_layer": support, "loop4_late_any_layer": late})
        layer, region = (scores[f"contrast_{pair}_{field}"] for field in ("delta_layer", "delta_regions"))
        banks.append(metric_values(layer, region, support, declared))
        definitions.extend({**definition, "contrast": pair, "id": pair + ":" + definition["id"]} for definition in declared)
    values = np.concatenate(banks, axis=-1)
    points, samples = {}, {}
    for task in TASKS:
        sample = design["tasks"][task]
        bank = values[sample["indices"]]
        points[task] = bank.mean(axis=0)
        samples[task] = {"conditional_item": sample["item_weights"] @ bank,
                         "conditional_component": sample["component_weights"] @ bank}
    intervals, radii = family_intervals(points, samples, ("conditional_item", "conditional_component"))
    return {"conditional_on_fit_id": 1, "n_independent_control_fits": 1, "fit_sd": None,
            "interpretation": "paired control effects conditional on fit01; no control fit-variance estimate",
            "metric_definitions": definitions, "family": {"contrasts": len(definitions) * len(TASKS), "simultaneous_radius95": radii},
            "tasks": {task: {"mean": points[task], "intervals": intervals[task]} for task in TASKS}}, {
                "control_paired_metrics": values, **{"control_" + key: value for key, value in scores.items()}}


def examples(inputs, design, paired_metrics, definitions):
    np = numpy()
    rows = inputs["rows"]
    eligible = np.asarray([i for i, row in enumerate(rows) if any(row["eligible"])], dtype=np.int64)
    chosen = np.random.default_rng(EXAMPLE_SEED).choice(eligible, min(10, len(eligible)), replace=False)
    def record(index):
        row = rows[int(index)]
        return {"index": int(index), **{key: row[key] for key in ("name", "task", "prompt", "target", "intermediates", "continuation", "correct", "eligible")},
                "paired_metric_values_per_fit": paired_metrics[:, int(index)]}
    failures = {}
    for task_index, task in enumerate(TASKS):
        pool = np.asarray([i for i in eligible if rows[int(i)]["task"] == task and not rows[int(i)]["correct"]], dtype=np.int64)
        selected = np.random.default_rng(np.random.SeedSequence(FAILURE_SEED, spawn_key=(task_index,))).choice(pool, min(5, len(pool)), replace=False)
        failures[task] = {"eligible_failure_count": len(pool), "eligible_failure_indices": pool,
                          "selected_indices": selected, "examples": [record(index) for index in selected]}
    return {"interpretation": "descriptive examples; no selection changes an aggregate or chooses a location",
            "metric_definitions": definitions, "fit_ids": FIT_IDS,
            "random": {"seed": EXAMPLE_SEED, "eligible_population_size": len(eligible), "selected_indices": chosen,
                       "selection": "uniform without replacement among all eligible items, before effect inspection",
                       "examples": [record(index) for index in chosen]},
            "failures": {"seed": FAILURE_SEED, "selection": "separate outcome-conditioned descriptive sample; no effect filtering",
                         "tasks": failures}}


def plain(value):
    np = numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def _csv(path, rows):
    with io._no_links(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    io._fsync_dir(path.parent)


def _interval_columns(statistics, column):
    result = {}
    for mode, estimates in statistics.items():
        for field, label in (("pointwise95", "pointwise"), ("simultaneous95", "simultaneous")):
            low, high = estimates[field][column]
            result[f"{mode}_{label}_low"] = float(low)
            result[f"{mode}_{label}_high"] = float(high)
    return result


def write_tables(out, p1, p2):
    curves, metrics, per_fit, conditional = [], [], [], []
    for task in TASKS:
        result = p1["tasks"][task]
        for column in range(192):
            stat = result["cells"]
            row = {"task": task, "virtual_index": column, "loop": column // 48 + 1,
                   "physical_layer_one_based": column % 48 + 1, "known_identity": column == 191,
                   **{key: float(stat[key][column]) for key in ("mean", "fit_sd", "fit_min", "fit_max")},
                   **{key: int(stat[key][column]) for key in ("positive_fits", "negative_fits", "zero_fits")}}
            for field in ("own_layer", "control_layer", "delta_layer"):
                row["jlens_" + field] = float(result["method_cells"]["jlens"][field]["mean"][column])
                row["raw_" + field] = float(result["method_cells"]["raw"][field][column])
            curves.append({**row, **_interval_columns(stat["intervals"], column)})
        for column, definition in enumerate(p1["metric_definitions"]):
            stat = result["metrics"]
            row = {"task": task, "metric": definition["id"], "kind": definition["kind"],
                   "virtual_indices": " ".join(map(str, definition["virtual_indices"])),
                   **{key: float(stat[key][column]) for key in ("mean", "fit_sd", "fit_min", "fit_max")},
                   **{key: int(stat[key][column]) for key in ("positive_fits", "negative_fits", "zero_fits")}}
            metrics.append({**row, **_interval_columns(stat["intervals"], column)})
            for position, fit_id in enumerate(FIT_IDS):
                value = float(stat["per_fit"][position, column])
                per_fit.append({"task": task, "fit_id": fit_id, "metric": definition["id"],
                                "value": value, "sign": "positive" if value > 0 else "negative" if value < 0 else "zero"})
        for column, definition in enumerate(p2["metric_definitions"]):
            stat = p2["tasks"][task]
            conditional.append({"task": task, "conditional_fit_id": 1, "contrast": definition["contrast"],
                                "metric": definition["id"], "kind": definition["kind"],
                                "virtual_indices": " ".join(map(str, definition["virtual_indices"])),
                                "mean": float(stat["mean"][column]), **_interval_columns(stat["intervals"], column)})
    for name, rows in (("main_cells.csv", curves), ("main_metrics.csv", metrics),
                       ("per_fit_metrics.csv", per_fit), ("control_metrics.csv", conditional)):
        _csv(out / name, rows)


def validate_output(root):
    root = io._no_links(root)
    owner = io._json(io._no_links(root / "OWNER.json"))
    complete = io._json(io._no_links(root / "COMPLETE.json"))
    if (set(owner) != {"schema", "identity", "identity_sha256"} or owner["schema"] != SCHEMA
            or owner["identity_sha256"] != io._digest(owner["identity"])
            or set(complete) != {"schema", "identity_sha256", "files"}
            or complete["schema"] != SCHEMA or complete["identity_sha256"] != owner["identity_sha256"]
            or owner["identity"].get("kind") != "five_fit_ouro_and_conditional_controls"
            or owner["identity"].get("fit_ids") != FIT_IDS
            or not isinstance(complete["files"], dict) or set(complete["files"]) != OUTPUT_FILES):
        raise ValueError("invalid completed analysis identity")
    expected = {"COMPLETE.json"}
    for name, record in complete["files"].items():
        if len(io._relative(name).parts) != 1 or name == "COMPLETE.json":
            raise ValueError("analysis receipt has an invalid member")
        io._verify_record(root / name, record)
        expected.add(name)
    actual = set()
    for path in root.iterdir():
        io._no_links(path)
        if not path.is_file():
            raise ValueError("analysis output contains a nonregular member")
        actual.add(path.name)
    if actual != expected:
        raise ValueError("analysis contains missing or unsealed files")
    return {"status": "passed", "identity_sha256": owner["identity_sha256"], "files": len(expected)}


def analyze(args):
    np = numpy()
    out = io._no_links(args.output)
    for input_root in (io._no_links(args.main), io._no_links(args.controls)):
        if out == input_root or input_root in out.parents:
            raise ValueError("analysis output cannot be inside a sealed input directory")
    if out.exists():
        raise FileExistsError("analysis requires a new output directory")
    authored = {str(path): io._record(path) for path in (Path(__file__), HERE / "CONTRACT.md")}
    argument_records = {str(io._no_links(path)): io._record(path)
                        for path in (args.run_spec, args.combined_contract)}
    inputs = load_inputs(args)
    contract = inputs["contract"]
    if argument_records[str(io._no_links(args.run_spec))]["sha256"] != contract["spec_sha256"]:
        raise ValueError("run specification changed while its inputs were consumed")
    source_records = {str(path): {"bytes": path.stat().st_size, "sha256": next(
        row["sha256"] for row in contract["sources"] if (row["root"], row["path"]) == key)}
        for key, path in contract["source_paths"].items()}
    for group in (authored, argument_records):
        for path, record in group.items():
            if path in source_records and source_records[path] != record:
                raise ValueError("conflicting frozen and preconsumption source records: " + path)
            source_records[path] = record
    for path, record in argument_records.items():
        io._verify_record(Path(path), record)
    design = bootstrap_design(inputs["rows"])
    p1, saved = main_analysis(inputs, design)
    p2, paired_controls = control_analysis(inputs, design)
    selected = examples(inputs, design, saved["main_paired_metrics"], p1["metric_definitions"])
    saved.update(paired_controls)
    saved["bootstrap_fit_indices"] = design["fit_indices"]
    populations = {}
    for task in TASKS:
        sample = design["tasks"][task]
        for key in ("indices", "groups", "item_indices", "component_indices"):
            saved[f"bootstrap_{task}_{key}"] = sample[key]
        sizes = np.bincount(sample["groups"])
        populations[task] = {"item_indices": sample["indices"], "component_membership": sample["groups"],
                             "component_count": len(sizes), "component_sizes": sizes,
                             "largest_component_fraction": float(sizes.max() / sizes.sum()),
                             "limitations": "shared eligible concepts only; curated prompts and templates are not an established IID sample"}
    identity = {"kind": "five_fit_ouro_and_conditional_controls", "run_spec_sha256": contract["spec_sha256"],
                "fit_ids": FIT_IDS, "bootstrap_draws": N_BOOT, "bootstrap_seed": SEED,
                "analysis_sources": authored, "frozen_source_records": source_records,
                "evaluation_artifacts": inputs["artifacts"],
                "python": sys.version, "numpy": np.__version__}
    report = {"schema": SCHEMA, "created_utc": io._utc_now().isoformat(), "fit_ids": FIT_IDS,
              "score": "paired difference of historical item excess; each fit scored before aggregation",
              "uncertainty": {"draws": N_BOOT, "seed": SEED, "fit_count": 5,
                              "shared_fit_draws_across_tasks": True, "shared_item_draws_across_fits_and_methods": True,
                              "pointwise": "percentile95", "simultaneous": "95th percentile of maximum absolute centered bootstrap error; unstudentized",
                              "seed_streams": {"fits": [0], "items": [1, "task_index"], "components": [2, "task_index"]},
                              "limits": "five-fit uncertainty is coarsely estimated; approximate intervals and concept sensitivity are not population guarantees"},
              "populations": populations, "P1": p1, "P2": p2,
              "verified_control_summary": "control_comparisons_summary.json",
              "examples": "examples.json", "paired_item_data": "paired_items.npz"}
    # Numerical work and full input validation finish before claiming an output.
    io._no_links(out).mkdir(parents=True, exist_ok=False)
    io._no_links(out)
    io._new_json(out / "OWNER.json", {"schema": SCHEMA, "identity": identity, "identity_sha256": io._digest(identity)})
    io._new_json(out / "report.json", plain(report))
    io._new_json(out / "control_comparisons_summary.json", inputs["comparison_summary"])
    io._new_json(out / "examples.json", plain(selected))
    evaluation._write_npz(out / "paired_items.npz", saved)
    write_tables(out, p1, p2)
    for path, record in {**inputs["consumed"], **source_records}.items():
        io._verify_record(Path(path), record)
    records = {path.name: io._record(path) for path in sorted(out.iterdir())}
    io._new_json(out / "COMPLETE.json", {"schema": SCHEMA, "identity_sha256": io._digest(identity), "files": records})
    return {**validate_output(out), "output": str(out), "scientific_claims": "requires review of the retained estimates and limitations"}


def _sealed_cpu_fixture(root, rows, names, raw_arrays, contract):
    """Symbolic owners plus repeated historical raw arrays; no fitted-model claim."""
    from types import SimpleNamespace
    forms = io._json(HISTORICAL / "audit/control_forms.json")["token_forms"]
    combined_path = DEPLOYMENT / "combined_contract.json"
    combined = io._json(combined_path)
    declared = {(row["root"], row["path"]): row["sha256"] for row in contract["sources"]}
    snapshot = {row["path"]: {key: row[key] for key in ("bytes", "sha256")} for row in contract["manifest"]["files"]}
    runtime = {"run_spec_sha256": contract["spec_sha256"], "model": contract["spec"]["model"],
               "snapshot_path": "/symbolic-cpu-fixture/no-model", "snapshot_files": snapshot,
               "source_files": contract["sources"], "settings": io.SETTINGS, "environment": contract["environment"],
               "gpu": {"name": "symbolic CPU fixture; no inference"}, "precision": {"symbolic_cpu_fixture": True},
               "attention_implementation": "sdpa",
               "imported_sources": [{"module": module, "path": "/symbolic-cpu-fixture/" + module,
                                     "sha256": declared[key]} for module, key in evaluation._fit_import_keys().items()],
               "remote_implementations": [{"class": "symbolic_cpu_fixture." + name, "path": "/symbolic-cpu-fixture/" + name,
                                            "sha256": snapshot[name]["sha256"]} for name in ("modeling_ouro.py", "configuration_ouro.py")]}
    def fit_record(fit, profile=None):
        settings = io.SETTINGS
        source_layers = list(range(191))
        state = {**runtime, "calibration": {key: value for key, value in fit.items() if key != "prompts"}}
        if profile is not None:
            selected = io.PROFILES[profile]
            source_layers = selected["source_layers"]
            settings = {**settings, "source_layers": source_layers, "target_layer": selected["target_layer"],
                        "mode": "paired_sampled" if selected["bank_arms"] else "dense"}
            state.update(combined_contract_sha256=io._file_hash(combined_path), production_profile=profile,
                         control_contract=combined["controls"], settings=settings)
        identity = io._fit_identity(fit["prompts"], fit["token_lengths"], fit["n_valid"], fit["fit_id"], state,
                                    2048, source_layers, False, **({"production_profile": profile} if profile else {}))
        owner = {"schema_version": 1, "identity": identity, "fit_identity_sha256": io._digest(identity)}
        encoded = io._canonical(owner) + b"\n"
        pointer = {"n_done": 100, "next_idx": 100, "fit_identity_sha256": owner["fit_identity_sha256"]}
        return {"fit_id": fit["fit_id"], "n_prompts": 100, "identity": identity,
                "fit_identity_sha256": owner["fit_identity_sha256"],
                "owner": {"bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()},
                "checkpoint_pointer": pointer, "complete_pointer": pointer}
    fits = [fit_record(fit) for fit in contract["fits"]]
    modules = ("ouro_jlens", "ouro_jlens.evaluate", "ouro_jlens.evaldata", "ouro_jlens.fit_lens",
               "ouro_jlens.evidence", "ouro_jlens.recurrent", "jlens", "jlens._logging", "jlens.fitting",
               "jlens.hooks", "jlens.hf", "jlens.lens", "jlens.protocol", "jlens.vis")
    imported = []
    for module in modules:
        family = "ouro_src" if module.startswith("ouro_jlens") else "jlens"
        path = module.replace(".", "/") + ("/__init__.py" if "." not in module else ".py")
        imported.append({"module": module, "root": family, "path": path, "sha256": declared[(family, path)]})
    evaluation_runtime = {**runtime, "imported_sources": imported}
    identity = {"kind": "ouro_main_n100_readout", "run_spec_sha256": contract["spec_sha256"],
                "source_files": contract["sources"], "model": contract["spec"]["model"],
                "fit_ids": FIT_IDS, "fits": fits, "tasks": list(TASKS), "position": -1, "greedy_steps": 4,
                "target_layer": 191, "learned_sources": list(range(191)), "virtual_indices": list(range(192)),
                "known_identity_indices": [191], "sections": ["common", *(f"fits/fit_{fit:02d}" for fit in FIT_IDS)],
                "symbolic_cpu_fixture": True}
    main_root = evaluation._new_output(root / "main", identity)
    directory = io._mkdir(main_root / "common")
    evaluation._write_npz(directory / "arrays.npz", {"exit_top1": raw_arrays["top1"][:, [47, 95, 143, 191]],
                                                     **{"logitlens_" + key: value for key, value in raw_arrays.items()}})
    for name, value in (("items", rows), ("task_names", names), ("token_forms", forms)):
        io._new_json(directory / (name + ".json"), value)
    observed = {"items": 148, "tasks": evaluation.TASK_COUNTS, "population_sha256": evaluation.POPULATION_SHA256,
                "task_names_sha256": evaluation.TASK_NAMES_SHA256, "token_forms_sha256": evaluation.TOKEN_FORMS_SHA256,
                "eligible_items": {"multihop": 90, "order-ops": 51}, "eligible_slots": {"multihop": 100, "order-ops": 51}}
    common = evaluation._seal_section(main_root, "common", {"kind": "common_readout", "runtime": evaluation_runtime,
        "population": observed, "position": -1, "virtual_indices": list(range(192)),
        "native_exit_indices": [47, 95, 143, 191], "target_state_indices": [190, 191]})
    sections = [common]
    for record in fits:
        section = f"fits/fit_{record['fit_id']:02d}"
        directory = io._mkdir(main_root / section)
        evaluation._write_npz(directory / "arrays.npz", {"jlens_exit3_" + key: value for key, value in raw_arrays.items()})
        sections.append(evaluation._seal_section(main_root, section, {"kind": "independent_fit_readout", "fit": record,
            "common": common, "n_prompts": 100, "readout": {"target_layer": 191, "virtual_indices": list(range(192)),
                                                             "learned_sources": list(range(191)), "identity_indices": [191]}}))
    evaluation._complete_output(main_root, sections)
    checked = controls.validate_completed_main(main_root, contract)
    records = {profile: fit_record(contract["fits"][0], profile) for profile in controls.PROFILES}
    records["ouro_positions"]["frozen_q_sha256"] = io._digest(combined["controls"]["positions"]["q"])
    identity = {"kind": "ouro_controls_n100_readout", "run_spec_sha256": contract["spec_sha256"],
                "combined_contract_sha256": io._file_hash(combined_path), "fit_ids": [1], "control_fits": records,
                "control_profiles": controls.PROFILES, "sections": controls.SECTIONS,
                "main_evaluation_owner_sha256": io._digest(checked["owner"]), "main_evaluation_complete_sha256": io._digest(checked["complete"]),
                "source_files": contract["sources"], "model": contract["spec"]["model"], "tasks": list(TASKS), "position": -1,
                "common_target_support": controls.TARGET_SUPPORT, "common_position_support": controls.POSITION_SUPPORT,
                "target_contrast_late_band": controls.TARGET_LATE, "position_contrast_late_band": controls.POSITION_LATE,
                "symbolic_cpu_fixture": True}
    control_root = evaluation._new_output(root / "controls", identity, schema=controls.SCHEMA)
    sections, banks = [], {}
    evaluation_runtime = {**evaluation_runtime, "shared_target_state_parity": {
        "190": "bitwise_equal_to_sealed_main_target_state", "191": "bitwise_equal_to_sealed_main_target_state"}}
    for name, profile, arm, support, target in (("penultimate", "ouro_penultimate", "dense", controls.TARGET_SUPPORT, 190),
                                               ("positions/sampled_sum", "ouro_positions", "sampled_sum", controls.POSITION_SUPPORT, 191),
                                               ("positions/diagonal", "ouro_positions", "diagonal", controls.POSITION_SUPPORT, 191)):
        arrays = {key: value[..., support] for key, value in raw_arrays.items()}
        banks["penultimate" if arm == "dense" else arm] = arrays
        directory = io._mkdir(control_root / name)
        evaluation._write_npz(directory / "arrays.npz", arrays)
        sections.append(evaluation._seal_section(control_root, name, {"kind": "control_readout", "profile": profile,
            "arm": arm, "control_fit": records[profile], "common": common, "position": -1,
            "evaluation_runtime": evaluation_runtime, "readout": {"target_layer": target, "virtual_indices": support,
                                                                   "learned_sources": support, "identity_indices": []}}))
    scores, summary = controls._comparisons(raw_arrays, raw_arrays, banks, rows, names)
    directory = io._mkdir(control_root / "comparisons")
    evaluation._write_npz(directory / "scores.npz", scores)
    evaluation._write_npz(directory / "paired_ranks.npz", {f"{kind}_{arm}_{key}": value[..., support]
        for kind, support in (("target", controls.TARGET_SUPPORT), ("position", controls.POSITION_SUPPORT))
        for arm in ("main", "raw") for key, value in raw_arrays.items()})
    for name, value in (("items", rows), ("task_names", names), ("token_forms", forms), ("summary", summary)):
        io._new_json(directory / (name + ".json"), value)
    sections.append(evaluation._seal_section(control_root, "comparisons", {"common": common, "conditional_on_fit_id": 1,
        "n_independent_control_fits": 1, "target_support": controls.TARGET_SUPPORT, "position_support": controls.POSITION_SUPPORT,
        "target_late_band": controls.TARGET_LATE, "position_late_band": controls.POSITION_LATE}))
    evaluation._complete_output(control_root, sections)
    return SimpleNamespace(main=main_root, controls=control_root, run_spec=Path(contract["spec_path"]),
                           combined_contract=combined_path, ouro_src=Path("/home/moloch/ouro_project/src"), output=root / "analysis")


def self_test(historical_main):
    """Independent retained-score comparisons plus literal small CPU resamples."""
    from unittest.mock import patch
    np = numpy()
    checks = []
    maximum = 0.0
    comparisons = 0
    def check(value, name):
        if not value:
            raise AssertionError(name)
        checks.append(name)
    def close(left, right, name, tolerance=2e-14):
        nonlocal maximum, comparisons
        left, right = np.asarray(left), np.asarray(right)
        if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
            raise AssertionError(name + " invalid comparison")
        error = float(np.max(np.abs(left - right))) if left.size else 0.0
        maximum = max(maximum, error)
        comparisons += left.size
        check(error <= tolerance, name)
    def rejects(operation, name):
        try:
            operation()
        except (ValueError, OSError, KeyError):
            checks.append(name)
        else:
            raise AssertionError(name + " was accepted")

    historical_main = io._no_links(historical_main)
    historical_manifest = io._json(HISTORICAL / "results/inputs.json")
    historical_records = {Path(row["path"]).name: row["sha256"] for row in historical_manifest["input_files"]
                          if Path(row["path"]).parent == HISTORICAL_MAIN}
    for name in ("arrays.npz", "items.json", "task_names.json"):
        check(io._file_hash(historical_main / name) == historical_records[name], "historical input SHA256 " + name)
    check(io._file_hash(HISTORICAL / "analyze_followup.py") == historical_manifest["generator_sha256"], "historical scoring source SHA256")
    inventory = {row["path"]: row["sha256"] for row in io._json(HISTORICAL / "MANIFEST.json")["files"]}
    check(io._file_hash(HISTORICAL / "results/item_scores.npz") == inventory["results/item_scores.npz"], "retained historical item-score SHA256")
    old_rows = io._json(historical_main / "items.json")
    names = io._json(historical_main / "task_names.json")
    rows = population(old_rows, names, stored_labels=False)
    loop_regions = {f"loop{loop + 1}_any_layer": list(range(loop * 48, (loop + 1) * 48)) for loop in range(4)}
    scores = {}
    with np.load(historical_main / "arrays.npz", allow_pickle=False) as archive, \
            np.load(HISTORICAL / "results/item_scores.npz", allow_pickle=False) as retained:
        for method in ("jlens_exit3", "logitlens"):
            scores[method] = controls.score_rank_arrays(archive[method + "_allrank"], rows, names, list(range(192)), bands=loop_regions)
            for task in TASKS:
                indices = np.asarray([i for i, row in enumerate(rows) if row["task"] == task and any(row["eligible"])])
                check(np.array_equal(indices, retained[task + "_indices"]), "historical eligible item order " + task + " " + method)
                for current, old in (("own_layer", "hit"), ("control_layer", "control"), ("delta_layer", "excess"),
                                     ("own_regions", "any_hit"), ("control_regions", "any_control"), ("delta_regions", "any_excess")):
                    expected = retained[f"{task}_{method}_{old}"]
                    close(scores[method][current][indices].reshape(expected.shape), expected, "historical score " + task + " " + method + " " + old)
                groups = concept_components(rows, indices)
                old_groups = retained[task + "_concept_clusters"]
                check(np.array_equal(groups[:, None] == groups, old_groups[:, None] == old_groups), "historical concept connectivity " + task + " " + method)
        check([len(np.unique(concept_components(rows, np.asarray([i for i, row in enumerate(rows) if row["task"] == task and any(row["eligible"])]))))
               for task in TASKS] == [64, 14], "historical component counts64/14")
    changed = copy.deepcopy(rows)
    changed[0]["eligible"][0] = not changed[0]["eligible"][0]
    rejects(lambda: population(changed, names), "stored eligibility cannot override the historical labels")
    changed = copy.deepcopy(rows)
    changed[0]["control_indices"][0] = []
    rejects(lambda: population(changed, names), "stored controls cannot override the historical catalogue")
    changed = copy.deepcopy(rows)
    changed[0]["correct"] = not changed[0]["correct"]
    rejects(lambda: population(changed, names), "stored failure label must match its original continuation rule")
    changed = copy.deepcopy(rows)
    changed[0]["prompt"] += "changed"
    rejects(lambda: population(changed, names), "changed core population bytes are rejected")

    definitions = fixed_metrics(list(range(192)), include_slope=True)
    check(len(definitions) == 28, "P1 has the declared28 fixed metrics")
    declared = {row["id"]: row for row in definitions}
    check(declared["loop1_middle_mean"]["virtual_indices"] == list(range(16, 32))
          and declared["loop4_historical_local_mean"]["virtual_indices"] == list(range(169, 181))
          and declared["loop4_final_third_mean"]["virtual_indices"] == list(range(176, 192)), "one-based historical bands map to exact virtual indices")
    for stop in (190, 191):
        declared_control = {row["id"]: row for row in fixed_metrics(list(range(stop)), regions={"common": list(range(stop)), "late": list(range(176, stop))})}
        check(declared_control["loop4_final_third_mean"]["virtual_indices"] == list(range(176, stop)), "P2 late intersection excludes unsupported columns " + str(stop))
    curve = np.full(192, -1.0)
    curve[191] = 0.0
    peak = learned_peak(curve)
    check(peak["maximum"] == -1 and 191 not in peak["tied_virtual_indices"] and not peak["positive_virtual_runs"], "known identity cannot masquerade as the best learned map")
    curve[150:153], curve[180] = 1.0, 2.0
    check(learned_peak(curve)["positive_virtual_runs"] == [[150, 152], [180, 180]], "descriptive positive runs retain disjoint locations")

    rng = np.random.default_rng(113)
    values = rng.normal(size=(5, 4, 7))
    fit_indices, fit_weights = _membership(5, 17, 11)
    item_indices, item_weights = _membership(4, 17, 12)
    actual = crossed_means(values, fit_weights, item_weights)
    literal = np.stack([values[fit_indices[i]][:, item_indices[i]].mean(axis=(0, 1)) for i in range(17)])
    close(actual, literal, "crossed means equal literal independently sampled fit/item rectangles")
    close(crossed_means(np.zeros_like(values), fit_weights, item_weights), np.zeros_like(actual), "paired constant nuisance cancels in every resample")
    close(crossed_means(np.full_like(values, 0.375), fit_weights, item_weights), np.full_like(actual, 0.375), "constant paired effect survives all bootstrap memberships")
    component_indices, weights = _membership(3, 17, 13, np.asarray([0, 0, 1]))
    v = np.asarray([0.0, 2.0, 10.0])
    literal = np.asarray([np.concatenate([v[np.asarray([0, 0, 1]) == group] for group in selected]).mean() for selected in component_indices])
    close(weights @ v, literal, "component ratio means preserve original item weighting")
    counts = np.asarray([0, 1, 2, 3, 4], dtype=np.float64)[:, None, None] - 2
    stats = fit_summary(np.repeat(counts, 3, axis=1))
    close(stats["fit_sd"], np.asarray([np.sqrt(2.5)]), "fit spread uses sample SD with four degrees of freedom")
    check(stats["positive_fits"].tolist() == [2] and stats["negative_fits"].tolist() == [2]
          and stats["zero_fits"].tolist() == [1], "per-fit signs preserve positive negative and zero counts")
    one = np.asarray([[0.0, 1.0], [1.0, -1.0], [0.5, -0.5]])
    points = {task: np.zeros(2) for task in TASKS}
    samples = {"multihop": {"mode": one}, "order-ops": {"mode": 2 * one}}
    intervals, radii = family_intervals(points, samples, ("mode",))
    radius = np.quantile(np.abs(np.concatenate((one, 2 * one), axis=1)).max(axis=1), 0.95)
    close(np.asarray([radii["mode"]]), np.asarray([radius]), "simultaneous family joins tasks before its max-centered radius")
    close(intervals["multihop"]["mode"]["pointwise95"], np.quantile(one, [0.025, 0.975], axis=0).T, "pointwise intervals remain percentile intervals")

    design = bootstrap_design(rows, draws=64)
    repeated = bootstrap_design(rows, draws=64)
    check(np.array_equal(design["fit_indices"], repeated["fit_indices"])
          and all(np.array_equal(design["tasks"][task]["item_indices"], repeated["tasks"][task]["item_indices"]) for task in TASKS), "seeded fit and item membership repeats exactly")
    raw = scores["logitlens"]
    fixture = {"rows": rows, "raw": raw, "fits": {key: np.repeat(value[None], 5, axis=0) for key, value in raw.items() if key != "eligible"}}
    p1, saved = main_analysis(fixture, design)
    check(p1["families"]["cells"]["contrasts"] == 384 and p1["families"]["metrics"]["contrasts"] == 56, "P1 uncertainty families have384 cells and56 metrics")
    for task in TASKS:
        for family in ("cells", "metrics"):
            stat = p1["tasks"][task][family]
            check(np.all(stat["mean"] == 0) and np.all(stat["fit_sd"] == 0)
                  and all(np.all(interval == 0) for mode in stat["intervals"].values() for interval in mode.values()), "identical lenses cancel through all P1 summaries and uncertainty " + task + " " + family)
    first = examples(fixture, design, saved["main_paired_metrics"], definitions)
    second = examples(fixture, design, saved["main_paired_metrics"] + 100, definitions)
    check(np.array_equal(first["random"]["selected_indices"], second["random"]["selected_indices"])
          and all(np.array_equal(first["failures"]["tasks"][task]["selected_indices"], second["failures"]["tasks"][task]["selected_indices"]) for task in TASKS), "example membership does not depend on effect values")
    full = bootstrap_design(rows)
    check(full["fit_indices"].shape == (10000, 5) and all(len(full["tasks"][task]["item_indices"]) == 10000 for task in TASKS), "production design retains all10000 five-fit draws")

    # The full reader and writer run on symbolically bound CPU fixtures. All five
    # banks and all controls are the same retained raw array, so every contrast
    # must cancel. This creates no new fitted lens or model outcome.
    contract = io._contract(DEPLOYMENT / "run_spec.json", Path("/home/moloch/ouro_project/src"), FIT_IDS)
    with np.load(historical_main / "arrays.npz", allow_pickle=False) as archive:
        raw_arrays = {key: archive["logitlens_" + key].copy() for key in ("allrank", "rank", "top1")}
    with tempfile.TemporaryDirectory(prefix="jlens-analysis-reader-") as temporary:
        args = _sealed_cpu_fixture(Path(temporary), rows, names, raw_arrays, contract)
        result = analyze(args)
        check(result["status"] == "passed" and result["files"] == 10, "real sealed-input semantic gates and full10000-draw output complete on symbolic CPU fixture")
        report = io._json(args.output / "report.json")
        check(report["P2"]["family"]["contrasts"] == 352
              and all(np.count_nonzero(report["P1"]["tasks"][task]["cells"]["mean"]) == 0
                      and np.count_nonzero(report["P2"]["tasks"][task]["mean"]) == 0 for task in TASKS),
              "paired reader/writer preserves exact cancellation across all five fits and eight controls")
        rejects(lambda: analyze(args), "completed analysis cannot be overwritten")
        for input_root in (args.main, args.controls):
            nested = copy.copy(args)
            nested.output = input_root / "analysis"
            rejects(lambda: analyze(nested), "analysis cannot modify a sealed input tree " + input_root.name)
            check(not nested.output.exists(), "rejected nested output leaves the sealed input tree unchanged " + input_root.name)
        unstable = copy.copy(args)
        unstable.output = Path(temporary) / "unstable-analysis"
        metadata_path = args.main / "common/metadata.json"
        original_metadata = metadata_path.read_bytes()
        original_write_tables = write_tables
        def changed_metadata(*arguments):
            original_write_tables(*arguments)
            with metadata_path.open("ab") as handle:
                handle.write(b"\n")
        try:
            with patch(__name__ + ".write_tables", side_effect=changed_metadata):
                rejects(lambda: analyze(unstable), "metadata changed during analysis prevents completion")
            check(not (unstable.output / "COMPLETE.json").exists(), "unstable input metadata never receives an output completion seal")
        finally:
            metadata_path.write_bytes(original_metadata)
        unstable_document = copy.copy(args)
        unstable_document.output = Path(temporary) / "unstable-contract-analysis"
        unstable_document.combined_contract = Path(temporary) / "combined_contract.json"
        unstable_document.combined_contract.write_bytes(args.combined_contract.read_bytes())
        original_load_inputs = load_inputs
        document_changed = [False]
        def changed_document(namespace):
            result = original_load_inputs(namespace)
            with namespace.combined_contract.open("ab") as handle:
                handle.write(b"\n")
            document_changed[0] = True
            return result
        with patch(__name__ + ".load_inputs", side_effect=changed_document):
            rejects(lambda: analyze(unstable_document), "a contract document changed during loading cannot acquire a fresh expected hash")
        check(document_changed[0], "contract mutation occurs after the real sealed-input semantic reader passes")
        check(not (unstable_document.output / "COMPLETE.json").exists(), "changed argument document never receives an output completion seal")
        section = args.controls / "comparisons"
        summary = io._json(section / "summary.json")
        summary["contrasts"]["multihop"]["all"]["target_main_minus_raw"]["delta_layer"][0] += 0.25
        io._atomic_json(section / "summary.json", summary)
        seal = io._json(section / "SEAL.json")
        seal["files"]["summary.json"] = io._record(section / "summary.json")
        io._atomic_json(section / "SEAL.json", seal)
        complete = io._json(args.controls / "COMPLETE.json")
        complete["sections"][-1]["seal"] = io._record(section / "SEAL.json")
        io._atomic_json(args.controls / "COMPLETE.json", complete)
        rejects(lambda: load_inputs(args), "consistently resealed false control summary is rejected by score recomputation")
        with (args.main / "common/arrays.npz").open("ab") as handle:
            handle.write(b"tampered")
        rejects(lambda: load_inputs(args), "changed sealed input rank bytes are rejected before scoring")

    with tempfile.TemporaryDirectory(prefix="jlens-analysis-test-") as temporary:
        root = Path(temporary)
        identity = {"kind": "five_fit_ouro_and_conditional_controls", "fit_ids": FIT_IDS}
        io._new_json(root / "OWNER.json", {"schema": SCHEMA, "identity": identity, "identity_sha256": io._digest(identity)})
        for name in OUTPUT_FILES - {"OWNER.json"}:
            (root / name).write_bytes(b"sealed CPU fixture\n")
        completion = {"schema": SCHEMA, "identity_sha256": io._digest(identity),
                      "files": {name: io._record(root / name) for name in OUTPUT_FILES}}
        io._new_json(root / "COMPLETE.json", completion)
        check(validate_output(root)["status"] == "passed", "complete artifact membership and hashes pass")
        (root / "extra").write_text("unsealed")
        rejects(lambda: validate_output(root), "unsealed output extra rejected")
        (root / "extra").unlink()
        (root / "report.json").write_text("corrupted")
        rejects(lambda: validate_output(root), "corrupted output bytes rejected")
        (root / "COMPLETE.json").unlink()
        rejects(lambda: validate_output(root), "partial output without completion rejected")
    return {"status": "passed", "scope": "CPU fixtures and retained historical scoring only; no new scientific outcomes",
            "checks": checks, "check_count": len(checks), "scalar_comparisons": int(comparisons),
            "maximum_absolute_error": maximum, "bootstrap_draws": N_BOOT,
            "fixed_random_example_indices": first["random"]["selected_indices"].tolist(),
            "source_records": {name: io._record(HERE / name) for name in ("analyze_refits.py", "CONTRACT.md")},
            "historical_input_sha256": historical_records, "numpy": np.__version__}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan")
    run = commands.add_parser("analyze")
    run.add_argument("--main", type=Path, required=True, help="sealed five-fit main evaluation")
    run.add_argument("--controls", type=Path, required=True, help="sealed paired fit01 control evaluation")
    run.add_argument("--run-spec", type=Path, default=DEPLOYMENT / "run_spec.json")
    run.add_argument("--combined-contract", type=Path, default=DEPLOYMENT / "combined_contract.json")
    run.add_argument("--ouro-src", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    test = commands.add_parser("self-test")
    test.add_argument("--historical-main", type=Path, default=HISTORICAL_MAIN)
    check = commands.add_parser("validate")
    check.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = {"schema": SCHEMA, "fit_ids": FIT_IDS, "bootstrap_draws": N_BOOT, "bootstrap_seed": SEED,
                  "bands_one_based_inclusive": BANDS, "cell_family": 384, "metric_family": 56,
                  "control_metric_family": 352, "known_identity_virtual_index": 191,
                  "target_late": list(range(176, 190)), "position_late": list(range(176, 191)),
                  "requires": "complete sealed main and control evaluations plus their frozen source/run contracts",
                  "output_files": sorted(OUTPUT_FILES | {"COMPLETE.json"}), "external_actions": 0, "new_scientific_outcomes": False}
    elif args.command == "self-test":
        result = self_test(args.historical_main)
    elif args.command == "validate":
        result = validate_output(args.output)
    else:
        result = analyze(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
