"""Interim P1-only orchestration of the unchanged, frozen Ouro analysis.

Only completed main evaluations are consumed. No control results are supplied
or invented, and this output never claims the combined P1/P2 analysis complete.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import analyze_refits as frozen

io = frozen.io
evaluation = frozen.evaluation
controls = frozen.controls
HERE = Path(__file__).resolve().parent
SCHEMA = "ouro_refit_main_analysis.v1"
KIND = "five_fit_ouro_main_only"
SCOPE = "Interim P1-only analysis; P2 and the combined Ouro analysis remain separate."
OUTPUT_FILES = {"OWNER.json", "report.json", "examples.json", "paired_items.npz",
                "main_cells.csv", "main_metrics.csv", "per_fit_metrics.csv"}


def load_inputs(args):
    """Preserve the frozen loader's source gates and main-side semantic checks."""
    np = frozen.numpy()
    contract = io._contract(args.run_spec, args.ouro_src, frozen.FIT_IDS)
    evaluation._evaluation_sources(contract)
    sources = {(row["root"], row["path"]): row["sha256"] for row in contract["sources"]}
    for name in ("run_refits.py", "evaluate_refits.py", "evaluate_controls.py", "run_controls.py"):
        path = frozen.DEPLOYMENT / name
        key = ("jlens", path.relative_to(frozen.REPO).as_posix())
        if key not in sources or io._file_hash(path) != sources[key]:
            raise ValueError("analysis helper is not the evaluator's frozen source: " + name)
    combined_path = io._no_links(args.combined_contract)
    combined_sha = controls._bound_combined(contract, io._json(combined_path))
    if io._file_hash(combined_path) != combined_sha:
        raise ValueError("provided combined contract bytes differ from the frozen document")
    root = io._no_links(args.main)
    checked = controls.validate_completed_main(root, contract, require_all=True)
    records = frozen.artifact_records(root, checked)
    consumed = {}

    def path(relative):
        record = records[relative]
        result = root / relative
        io._verify_record(result, record)
        consumed[str(result)] = record
        return result

    # The semantic gate hashes every sealed file, including opaque cache.pt.
    # Retain all those original records for the final input-stability check.
    for relative in records:
        path(relative)
    rows = io._json(path("common/items.json"))
    names = io._json(path("common/task_names.json"))
    forms = io._json(path("common/token_forms.json"))
    rows = frozen.population(rows, names, forms)

    def readout(relative, prefix):
        arrays = controls._load_arrays(path(relative), prefix)
        evaluation._validate_arrays(arrays, rows, names, list(range(192)))
        return arrays

    support = list(range(192))
    regions = {f"loop{loop + 1}_any_layer": list(range(loop * 48, (loop + 1) * 48)) for loop in range(4)}
    raw_arrays = readout("common/arrays.npz", "logitlens_")
    with np.load(path("common/arrays.npz"), allow_pickle=False) as archive:
        exits = archive["exit_top1"]
        if exits.shape != (148, 4) or not np.array_equal(exits, raw_arrays["top1"][:, [47, 95, 143, 191]]):
            raise ValueError("raw boundary predictions differ from the sealed native exits")
    raw = controls.score_rank_arrays(raw_arrays["allrank"], rows, names, support, bands=regions)
    fit_scores = []
    for fit_id in frozen.FIT_IDS:
        arrays = readout(f"fits/fit_{fit_id:02d}/arrays.npz", "jlens_exit3_")
        for key in ("allrank", "rank", "top1"):
            if not np.array_equal(arrays[key][..., 191], raw_arrays[key][..., 191]):
                raise ValueError("main final target is not the declared raw identity reference")
        score = controls.score_rank_arrays(arrays["allrank"], rows, names, support, bands=regions)
        if not np.array_equal(score["eligible"], raw["eligible"]):
            raise ValueError("fit and raw eligible populations differ")
        fit_scores.append(score)
    fitted = {key: np.stack([score[key] for score in fit_scores]) for key in raw if key != "eligible"}
    return {"contract": contract, "rows": rows, "names": names, "forms": forms,
            "raw": raw, "fits": fitted, "consumed": consumed,
            "artifacts": {"main": {"root": str(root), "files": records}}}


def write_tables(out, p1):
    """The unchanged main-table projection, without a synthetic P2 argument."""
    curves, metrics, per_fit = [], [], []
    for task in frozen.TASKS:
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
            curves.append({**row, **frozen._interval_columns(stat["intervals"], column)})
        for column, definition in enumerate(p1["metric_definitions"]):
            stat = result["metrics"]
            row = {"task": task, "metric": definition["id"], "kind": definition["kind"],
                   "virtual_indices": " ".join(map(str, definition["virtual_indices"])),
                   **{key: float(stat[key][column]) for key in ("mean", "fit_sd", "fit_min", "fit_max")},
                   **{key: int(stat[key][column]) for key in ("positive_fits", "negative_fits", "zero_fits")}}
            metrics.append({**row, **frozen._interval_columns(stat["intervals"], column)})
            for position, fit_id in enumerate(frozen.FIT_IDS):
                value = float(stat["per_fit"][position, column])
                per_fit.append({"task": task, "fit_id": fit_id, "metric": definition["id"],
                                "value": value, "sign": "positive" if value > 0 else "negative" if value < 0 else "zero"})
    for name, rows in (("main_cells.csv", curves), ("main_metrics.csv", metrics), ("per_fit_metrics.csv", per_fit)):
        frozen._csv(out / name, rows)


def validate_output(root):
    root = io._no_links(root)
    owner = io._json(io._no_links(root / "OWNER.json"))
    complete = io._json(io._no_links(root / "COMPLETE.json"))
    if (set(owner) != {"schema", "identity", "identity_sha256"} or owner["schema"] != SCHEMA
            or owner["identity_sha256"] != io._digest(owner["identity"])
            or set(complete) != {"schema", "identity_sha256", "files"}
            or complete["schema"] != SCHEMA or complete["identity_sha256"] != owner["identity_sha256"]
            or owner["identity"].get("kind") != KIND or owner["identity"].get("scope") != SCOPE
            or owner["identity"].get("fit_ids") != frozen.FIT_IDS
            or owner["identity"].get("bootstrap_draws") != frozen.N_BOOT
            or owner["identity"].get("bootstrap_seed") != frozen.SEED
            or not isinstance(complete["files"], dict) or set(complete["files"]) != OUTPUT_FILES):
        raise ValueError("invalid completed P1-only analysis identity")
    for name, record in complete["files"].items():
        io._verify_record(root / name, record)
    actual = set()
    for path in root.iterdir():
        io._no_links(path)
        if not path.is_file():
            raise ValueError("P1-only output contains a nonregular member")
        actual.add(path.name)
    if actual != OUTPUT_FILES | {"COMPLETE.json"}:
        raise ValueError("P1-only output contains missing or unsealed files")
    report = io._json(root / "report.json")
    if report.get("scope") != SCOPE or "P1" not in report or "P2" in report:
        raise ValueError("P1-only report has the wrong analysis scope")
    return {"status": "passed", "scope": SCOPE, "identity_sha256": owner["identity_sha256"], "files": len(actual)}


def analyze(args):
    np = frozen.numpy()
    out, main_root = io._no_links(args.output), io._no_links(args.main)
    if out == main_root or main_root in out.parents:
        raise ValueError("analysis output cannot be inside a sealed input directory")
    if out.exists():
        raise FileExistsError("analysis requires a new output directory")
    authored = {str(path): io._record(path) for path in (Path(__file__), Path(frozen.__file__), HERE / "CONTRACT.md")}
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
    design = frozen.bootstrap_design(inputs["rows"])
    p1, saved = frozen.main_analysis(inputs, design)
    selected = frozen.examples(inputs, design, saved["main_paired_metrics"], p1["metric_definitions"])
    saved["bootstrap_fit_indices"] = design["fit_indices"]
    populations = {}
    for task in frozen.TASKS:
        sample = design["tasks"][task]
        for key in ("indices", "groups", "item_indices", "component_indices"):
            saved[f"bootstrap_{task}_{key}"] = sample[key]
        sizes = np.bincount(sample["groups"])
        populations[task] = {"item_indices": sample["indices"], "component_membership": sample["groups"],
                             "component_count": len(sizes), "component_sizes": sizes,
                             "largest_component_fraction": float(sizes.max() / sizes.sum()),
                             "limitations": "shared eligible concepts only; curated prompts and templates are not an established IID sample"}
    identity = {"kind": KIND, "scope": SCOPE, "run_spec_sha256": contract["spec_sha256"],
                "fit_ids": frozen.FIT_IDS, "bootstrap_draws": frozen.N_BOOT, "bootstrap_seed": frozen.SEED,
                "analysis_sources": authored, "frozen_source_records": source_records,
                "evaluation_artifacts": inputs["artifacts"], "python": sys.version, "numpy": np.__version__}
    report = {"schema": SCHEMA, "scope": SCOPE, "created_utc": io._utc_now().isoformat(), "fit_ids": frozen.FIT_IDS,
              "score": "paired difference of historical item excess; each fit scored before aggregation",
              "uncertainty": {"draws": frozen.N_BOOT, "seed": frozen.SEED, "fit_count": 5,
                              "shared_fit_draws_across_tasks": True, "shared_item_draws_across_fits_and_methods": True,
                              "pointwise": "percentile95", "simultaneous": "95th percentile of maximum absolute centered bootstrap error; unstudentized",
                              "seed_streams": {"fits": [0], "items": [1, "task_index"], "components": [2, "task_index"]},
                              "limits": "five-fit uncertainty is coarsely estimated; approximate intervals and concept sensitivity are not population guarantees"},
              "populations": populations, "P1": p1, "examples": "examples.json", "paired_item_data": "paired_items.npz"}
    io._no_links(out).mkdir(parents=True, exist_ok=False)
    io._no_links(out)
    io._new_json(out / "OWNER.json", {"schema": SCHEMA, "identity": identity, "identity_sha256": io._digest(identity)})
    io._new_json(out / "report.json", frozen.plain(report))
    io._new_json(out / "examples.json", frozen.plain(selected))
    evaluation._write_npz(out / "paired_items.npz", saved)
    write_tables(out, p1)
    for path, record in {**inputs["consumed"], **source_records}.items():
        io._verify_record(Path(path), record)
    records = {path.name: io._record(path) for path in sorted(out.iterdir())}
    io._new_json(out / "COMPLETE.json", {"schema": SCHEMA, "identity_sha256": io._digest(identity), "files": records})
    return {**validate_output(out), "output": str(out), "scientific_claims": "requires review of the retained estimates and limitations"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("analyze")
    run.add_argument("--main", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--ouro-src", type=Path, required=True)
    run.add_argument("--run-spec", type=Path, default=frozen.DEPLOYMENT / "run_spec.json")
    run.add_argument("--combined-contract", type=Path, default=frozen.DEPLOYMENT / "combined_contract.json")
    check = commands.add_parser("validate")
    check.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    result = validate_output(args.output) if args.command == "validate" else analyze(args)
    print(io._canonical(result).decode())


if __name__ == "__main__":
    main()
