"""Evaluate one sealed Huginn R8 N100 lens on the frozen shared population.

Prepare eligibility before inference with --prepare-eligibility. Production
requires that exact file in the source manifest, both fixed evaluation seeds,
all 148 item IDs and all 32 core locations. Coda readouts consume the complete
sequence; only pass-end coda readouts are native recurrence exits. Outputs are
new immutable directories, with no COMPLETE marker after an interrupted seed.
This program does not rent, upload, terminate, or select a model or depth.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import gc
import importlib
import inspect
import json
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

try:
    from . import evaluate_refits as ref
    from . import run_refits as runner
except ImportError:
    import evaluate_refits as ref
    import run_refits as runner

HERE, REPO = ref.HERE, ref.REPO
TASKS = ref.TASKS
SCHEMA = "huginn_r8_evaluation.v1"
ELIGIBILITY_SCHEMA = "huginn_common_eligibility.v1"
HUGINN_REPO_ID = "tomg-group-umd/huginn-0125"
SEEDS = (2026090803, 2026090804)
SOURCES = tuple(range(32))
METHODS = ("raw", "jlens", "coda")
SECTIONS = ["population", *(f"seeds/{seed}" for seed in SEEDS)]
SEED_RULE = "SHA256(canonical JSON(version, base_seed, namespace, input_ids)); first 8 bytes big-endian modulo 2^63"


def _adapter():
    if __package__:
        return importlib.import_module(f"{__package__}.huginn_adapter")
    return importlib.import_module("huginn_adapter")


def _preparation_sources(ouro_src):
    """Portable source records, without a circular run_spec/eligibility hash."""
    roots = {"jlens": REPO, "ouro_src": Path(ouro_src).absolute()}
    keys = ref._evaluation_source_keys() | {
        ("jlens", Path(__file__).relative_to(REPO).as_posix()),
        ("jlens", (HERE / "huginn_adapter.py").relative_to(REPO).as_posix()),
        ("jlens", (HERE / "run_refits.py").relative_to(REPO).as_posix()),
        *(("jlens", path.relative_to(REPO).as_posix()) for path in (REPO / "jlens").rglob("*.py")),
    }
    records = [{"root": root, "path": path,
                "sha256": runner._file_hash(runner._no_links(roots[root] / path))}
               for root, path in sorted(keys)]
    for task, digest in ref.STIMULUS_HASHES.items():
        if runner._file_hash(REPO / f"data/evaluations/lens-eval-{task}.json") != digest:
            raise ValueError("the frozen stimulus bytes changed")
    return records


def _tokenizer_files(snapshot, manifest_path):
    """Check every small snapshot input; weight bytes are checked at execution."""
    snapshot = Path(snapshot).absolute()
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ValueError("tokenizer snapshot must be an explicit local directory")
    manifest_path = runner._no_links(manifest_path)
    manifest = runner._json(manifest_path)
    files = manifest.get("files")
    if manifest.get("schema_version") != 1 or not isinstance(files, list) or not files:
        raise ValueError("invalid tokenizer model manifest")
    seen, small = set(), {}
    for record in files:
        relative = runner._relative(record["path"])
        if relative.as_posix() in seen:
            raise ValueError("duplicate tokenizer manifest input")
        seen.add(relative.as_posix())
        path = snapshot / relative
        if any((snapshot / Path(*relative.parts[:i])).is_symlink()
               for i in range(1, len(relative.parts))):
            raise ValueError("snapshot directory links are not supported")
        if not path.is_file():
            raise ValueError(f"snapshot input is missing: {relative}")
        if path.suffix != ".safetensors":
            actual = runner._record(path.resolve())
            if actual != {"bytes": record["bytes"], "sha256": record["sha256"]}:
                raise ValueError(f"tokenizer metadata changed: {relative}")
            small[relative.as_posix()] = actual
    actual_names = set()
    for path in snapshot.rglob("*"):
        if path.is_dir():
            if path.is_symlink():
                raise ValueError("snapshot directory links are not supported")
        elif path.is_file():
            actual_names.add(path.relative_to(snapshot).as_posix())
        else:
            raise ValueError("snapshot contains an unsupported entry")
    if actual_names != seen:
        raise ValueError("snapshot has extra or missing tokenizer/model inputs")
    return {"repo_id": manifest["repo_id"], "revision": manifest["revision"],
            "manifest_sha256": runner._file_hash(manifest_path), "metadata_files": small}


def _build_population(ouro_tokenizer, huginn_tokenizer, ouro_items, huginn_items,
                      evaluator, *, historical=True):
    """Freeze eligibility and matched controls before any state or rank exists."""
    operations = set(importlib.import_module("ouro_jlens.evaldata").OPERATIONS)
    with ref.task_names(evaluator, ouro_items) as registry:
        ouro_rows = ref._item_rows(SimpleNamespace(tokenizer=ouro_tokenizer), ouro_items, registry, evaluator)
        names, ouro_forms, original = ref._population(ouro_rows, registry, historical=historical)
    with ref.task_names(evaluator, huginn_items) as registry:
        huginn_rows = ref._item_rows(SimpleNamespace(tokenizer=huginn_tokenizer), huginn_items, registry, evaluator)
        _, huginn_forms, native = ref._population(huginn_rows, registry, historical=False)
    fields = ("name", "task", "prompt", "target", "intermediates")
    if (len(ouro_rows) != len(huginn_rows)
            or any(not ref._same({key: a[key] for key in fields}, {key: b[key] for key in fields})
                   for a, b in zip(ouro_rows, huginn_rows))):
        raise ValueError("tokenizers did not preserve the same ordered stimulus identities")
    common = {task: [index for index, name in enumerate(names[task])
                     if ouro_forms[task].get(name) and huginn_forms[task].get(name)] for task in TASKS}
    rows = []
    for a, b in zip(ouro_rows, huginn_rows):
        task = a["task"]
        own = set(index for index in a["own_index"] if index >= 0)
        reasons, eligible, controls = [], [], []
        for slot, name in enumerate(a["intermediates"]):
            why = []
            for family, row in (("ouro", a), ("huginn", b)):
                if not row["scorable"][slot]:
                    why.append(f"{family}:no_single_token_form")
                if row["leaked"][slot]:
                    why.append(f"{family}:form_in_readout_context")
            if task == "order-ops" and name in operations:
                why.append("arithmetic:operation_slot_excluded")
            pool = [index for index in common[task] if index not in own
                    and ((names[task][index] in operations) == (name in operations))]
            if not why and not pool:
                raise ValueError("jointly eligible target has no matched common controls")
            reasons.append(why)
            eligible.append(not why)
            controls.append(pool)
        rows.append({**{key: a[key] for key in fields}, "own_index": a["own_index"],
                     "ouro": {key: a[key] for key in ("scorable", "leaked", "eligible", "token_ids", "n_tokens", "readout_token", "intermediate_tokens")},
                     "huginn": {key: b[key] for key in ("scorable", "leaked", "eligible", "token_ids", "n_tokens", "readout_token", "intermediate_tokens")},
                     "eligible": eligible, "exclusion_reasons": reasons, "control_indices": controls})
    counts = {"items": len(rows), "tasks": {task: sum(row["task"] == task for row in rows) for task in TASKS},
              "eligible_items": {task: sum(row["task"] == task and any(row["eligible"]) for row in rows) for task in TASKS},
              "eligible_slots": {task: sum(sum(row["eligible"]) for row in rows if row["task"] == task) for task in TASKS},
              "control_names": {task: len(common[task]) for task in TASKS}}
    return {"rows": rows, "task_names": names, "common_name_indices": common,
            "excluded_control_names": {task: [name for index, name in enumerate(names[task])
                                               if index not in common[task]] for task in TASKS},
            "token_forms": {"ouro": ouro_forms, "huginn": huginn_forms},
            "original_ouro_population": original, "native_huginn_population": native, "counts": counts}


def prepare_eligibility(*, ouro_snapshot, huginn_snapshot, ouro_src,
                        ouro_manifest=HERE / "model_manifest.json",
                        huginn_manifest=HERE / "huginn_model_manifest.json"):
    """CPU-only preparation. No model weights, fitted maps, states or scores."""
    import torch
    import transformers
    sources = _preparation_sources(ouro_src)
    tokenizers = {"ouro": _tokenizer_files(ouro_snapshot, ouro_manifest),
                  "huginn": _tokenizer_files(huginn_snapshot, huginn_manifest)}
    sys.path[:0] = [str(REPO), str(Path(ouro_src).absolute())]
    evaluator = importlib.import_module("ouro_jlens.evaluate")
    recurrent = importlib.import_module("ouro_jlens.recurrent")
    for module in (evaluator, recurrent, importlib.import_module("ouro_jlens.evaldata")):
        actual = Path(module.__file__).absolute()
        expected = Path(ouro_src).absolute() / "ouro_jlens" / actual.name
        if actual != expected:
            raise ValueError("eligibility preparation imported a different source tree")
    adapter = _adapter()
    adapter.verify_pinned_metadata(huginn_snapshot)
    if (tokenizers["ouro"]["repo_id"] != "ByteDance/Ouro-2.6B"
            or tokenizers["ouro"]["revision"] != runner.REVISION
            or tokenizers["huginn"]["repo_id"] != HUGINN_REPO_ID
            or tokenizers["huginn"]["revision"] != adapter.REVISION):
        raise ValueError("eligibility tokenizers do not identify the two pinned models")
    ouro = transformers.AutoTokenizer.from_pretrained(str(ouro_snapshot), local_files_only=True)
    huginn = transformers.PreTrainedTokenizerFast.from_pretrained(str(huginn_snapshot), local_files_only=True)
    proxy = SimpleNamespace(tokenizer=ouro, input_device=torch.device("cpu"))
    ouro_encode = lambda text: recurrent.OuroLensModel.encode(proxy, text, max_length=512)[0].tolist()
    huginn_encode = lambda text: huginn(text, add_special_tokens=True, truncation=True, max_length=512,
                                      padding=False, return_attention_mask=False).input_ids
    ouro_items = evaluator.load_items(ouro, TASKS, encode=ouro_encode)
    huginn_items = evaluator.load_items(huginn, TASKS, encode=huginn_encode)
    population = _build_population(ouro, huginn, ouro_items, huginn_items, evaluator)
    if any(not row["huginn"]["token_ids"] or row["huginn"]["token_ids"][0] != 65504
           or huginn.pad_token_id in row["huginn"]["token_ids"] for row in population["rows"]):
        raise ValueError("Huginn eligibility contexts violate native BOS/no-padding semantics")
    plan = {"schema": ELIGIBILITY_SCHEMA, "tasks": list(TASKS), "evaluation_max_length": 512,
            "stimulus_sha256": ref.STIMULUS_HASHES, "sources": sources, "tokenizers": tokenizers,
            "evaluation_base_seeds": list(SEEDS), "seed_namespace": "evaluation", "seed_rule": SEED_RULE,
            "position_policy": "last token of common prefix encode(prompt), encode(prompt + target); each tokenizer uses its native BOS rule; total length <=512",
            "eligibility_policy": "single-token scorable and not leaked under both tokenizers; retain arithmetic numeric slots only; no item selected by correctness or recovery",
            "control_policy": "common supported task names only; match operation/non-operation kind; exclude all own names; retain existing aliases",
            "score_policy": "zero-based full-vocabulary rank<10; minimum over all native token forms; average eligible slots within each item before averaging items",
            **population}
    return plan, (evaluator, huginn, huginn_items)


def _source_record(path, contract):
    path = runner._no_links(path)
    key = ("jlens", path.relative_to(REPO).as_posix())
    declared = {(row["root"], row["path"]): row["sha256"] for row in contract["sources"]}
    if key not in declared or runner._file_hash(path) != declared[key]:
        raise ValueError(f"run specification does not bind the exact input: {path.name}")
    return runner._record(path)


def _contract(args):
    main = runner._contract(args.run_spec, args.ouro_src, [1])
    ref._evaluation_sources(main)
    for path in (Path(__file__), HERE / "huginn_adapter.py", HERE / "run_huginn.py",
                 args.combined_contract, args.huginn_calibration, args.eligibility, args.huginn_manifest):
        _source_record(path, main)
    combined = runner._json(args.combined_contract)
    huginn = combined["huginn"]
    adapter = _adapter()
    expected = {"repo_id": HUGINN_REPO_ID, "revision": adapter.REVISION,
                "n_prompts": 100, "calibration_fit_id": 1,
                "calibration_sha256": main["fits"][0]["sha256"], "n_recurrences": 8,
                "source_layers": list(SOURCES), "target_layer": 33, "d_model": 5280,
                "max_seq_len": 128, "skip_first": 16, "initialization_base_seed": 2026090802,
                "evaluation_base_seeds": list(SEEDS), "manifest_sha256": runner._file_hash(args.huginn_manifest)}
    if (combined.get("schema_version") != 1 or combined.get("budget_usd") != 25
            or any(not ref._same(huginn.get(key), value) for key, value in expected.items())):
        raise ValueError("Huginn geometry, calibration, initialization or model contract changed")
    manifest = runner._json(args.huginn_manifest)
    calibration = runner._json(args.huginn_calibration)
    rows = calibration.get("rows")
    if (manifest.get("repo_id") != HUGINN_REPO_ID or manifest.get("revision") != adapter.REVISION
            or calibration.get("schema_version") != 1 or calibration.get("n_prompts") != 100
            or calibration.get("calibration_fit_id") != 1
            or calibration.get("calibration_sha256") != main["fits"][0]["sha256"]
            or calibration.get("model_manifest_sha256") != huginn["manifest_sha256"]
            or calibration.get("max_seq_len") != 128 or calibration.get("skip_first") != 16
            or calibration.get("seed_namespace") != "calibration"
            or not isinstance(rows, list) or len(rows) != 100):
        raise ValueError("Huginn token calibration does not bind the frozen N100 paragraphs")
    for index, row in enumerate(rows):
        if (row.get("index") != index or type(row.get("token_length")) is not int
                or not 18 <= row["token_length"] <= 128 or row.get("n_valid") != row["token_length"] - 17
                or row.get("base_seed") != 2026090802 + index):
            raise ValueError("Huginn calibration token or seed row changed")
    implementation = importlib.import_module(f"{__package__}.run_huginn" if __package__ else "run_huginn")
    _source_record(Path(implementation.__file__), main)
    implementation._manifest(manifest)
    implementation._calibration(calibration, combined, main["fits"][0], runner._file_hash(args.huginn_manifest))
    return main, combined, manifest, calibration


def validate_huginn_owner(owner, main, combined, manifest, calibration, *, combined_sha256, calibration_sha256):
    """Check portable logical bindings without opening producer absolute paths."""
    if (not isinstance(owner, dict) or set(owner) != {"schema_version", "fit_identity_sha256", "identity"}
            or type(owner["schema_version"]) is not int or owner["schema_version"] != 1
            or owner["fit_identity_sha256"] != runner._digest(owner["identity"])):
        raise ValueError("invalid Huginn OWNER identity")
    fit, identity = main["fits"][0], owner["identity"]
    runtime = identity.get("runtime")
    keys = {"run_spec_sha256", "combined_contract_sha256", "huginn_calibration_sha256", "model",
            "snapshot_path", "snapshot_files", "source_files", "imported_sources", "remote_implementations",
            "environment", "gpu", "precision", "settings", "initialization", "prerequisites",
            "attention_implementation", "calibration"}
    if not isinstance(runtime, dict) or set(runtime) != keys:
        raise ValueError("Huginn fitting runtime is incomplete or has an unknown schema")
    lengths = [row["token_length"] for row in calibration["rows"]]
    valid = [row["n_valid"] for row in calibration["rows"]]
    expected_identity = runner._fit_identity(fit["prompts"], lengths, valid, 1, runtime,
                                            5280, SOURCES, False, production_profile="huginn_r8")
    if not ref._same(identity, expected_identity):
        raise ValueError("Huginn OWNER does not describe the fixed R8 N100 map bank")
    implementation = importlib.import_module(f"{__package__}.run_huginn" if __package__ else "run_huginn")
    _source_record(Path(implementation.__file__), main)
    expected = {"run_spec_sha256": main["spec_sha256"], "combined_contract_sha256": combined_sha256,
                "huginn_calibration_sha256": calibration_sha256,
                "model": {key: combined["huginn"][key] for key in ("repo_id", "revision", "manifest_sha256")},
                "source_files": main["sources"], "settings": implementation.SETTINGS,
                "snapshot_files": {row["path"]: {key: row[key] for key in ("bytes", "sha256")} for row in manifest["files"]},
                "initialization": {"base_seed": 2026090802, "seed_namespace": "calibration",
                                   "base_seed_rule": "base_seed + zero-based paragraph index", "prompt_seed_rule": SEED_RULE}}
    for name, value in expected.items():
        if not ref._same(runtime[name], value):
            raise ValueError(f"Huginn fitting runtime {name} differs from the frozen inputs")
    bound = runtime["calibration"]
    origin = {key: value for key, value in fit.items() if key != "prompts"}
    if (not isinstance(bound, dict) or set(bound) != {"origin", "huginn"}
            or not isinstance(bound["origin"], dict) or set(bound["origin"]) != set(origin)
            or not isinstance(bound["origin"].get("prompt_path"), str)
            or not ref._same({k: v for k, v in bound["origin"].items() if k != "prompt_path"},
                            {k: v for k, v in origin.items() if k != "prompt_path"})
            or not ref._same(bound["huginn"], calibration)):
        raise ValueError("Huginn calibration identity changed")
    for key in ("packages", "torch_git", "python_version", "cuda_build_version", "cudnn_runtime_integer"):
        if not ref._same(runtime["environment"].get(key), main["environment"].get(key)):
            raise ValueError(f"Huginn fitting environment changed: {key}")
    if (not isinstance(runtime["gpu"], dict) or not runtime["gpu"].get("name")
            or not isinstance(runtime["precision"], dict) or not runtime["precision"]
            or not isinstance(runtime["attention_implementation"], str)):
        raise ValueError("Huginn fitting hardware/precision identity is missing")
    imports = {**ref._fit_import_keys(), "huginn_adapter": ("jlens", (HERE / "huginn_adapter.py").relative_to(REPO).as_posix())}
    declared = {(row["root"], row["path"]): row["sha256"] for row in main["sources"]}
    actual = runtime["imported_sources"]
    if (not isinstance(actual, list) or len(actual) != len(imports)
            or {row.get("module") for row in actual if isinstance(row, dict)} != set(imports)):
        raise ValueError("Huginn fitting imported-source membership changed")
    for row in actual:
        if (set(row) != {"module", "path", "sha256"} or not isinstance(row["path"], str)
                or row["sha256"] != declared[imports[row["module"]]]):
            raise ValueError("Huginn fitting imported-source hash changed")
    remote = runtime["remote_implementations"]
    remote_hashes = {expected["snapshot_files"][name]["sha256"] for name in ("raven_modeling_minimal.py", "raven_config_minimal.py")}
    if (not isinstance(remote, list) or len(remote) != 2
            or any(not isinstance(row, dict) or set(row) != {"class", "path", "sha256"}
                   or not isinstance(row["class"], str) or not isinstance(row["path"], str) for row in remote)
            or {row["sha256"] for row in remote} != remote_hashes):
        raise ValueError("Huginn fitting native implementation hashes changed")
    prior = runtime["prerequisites"]
    if not isinstance(prior, dict) or set(prior) != {"ouro", "controls"}:
        raise ValueError("Huginn fit lacks the preceding Ouro/control evidence")
    for name, schema in (("ouro", ref.SCHEMA), ("controls", "ouro_controls_evaluation.v1")):
        record = prior[name]
        if (not isinstance(record, dict) or set(record) != {"schema", "identity_sha256", "owner_sha256", "owner_file", "complete_file"}
                or record["schema"] != schema
                or any(not isinstance(record[key], str) or runner._SHA256.fullmatch(record[key]) is None
                       for key in ("identity_sha256", "owner_sha256"))):
            raise ValueError("Huginn prerequisite binding is malformed")
        for key in ("owner_file", "complete_file"):
            value = record[key]
            if (not isinstance(value, dict) or set(value) != {"bytes", "sha256"}
                    or type(value["bytes"]) is not int or value["bytes"] <= 0
                    or not isinstance(value["sha256"], str) or runner._SHA256.fullmatch(value["sha256"]) is None):
                raise ValueError("Huginn prerequisite file record is malformed")


def _fit_initializations(lens_path, calibration):
    """Check the sealed producer's actual per-paragraph state seed records."""
    seal = runner._json(runner._no_links(lens_path.parent / "SEAL.json"))
    metadata_path = runner._no_links(lens_path.parent / "metadata.json")
    runner._verify_record(metadata_path, seal["files"]["metadata.json"])
    metadata = runner._json(metadata_path)
    diagnostics = metadata["diagnostics"]
    if len(diagnostics) != 100:
        raise ValueError("Huginn fitting state provenance must retain all 100 paragraphs")
    for index, (diagnostic, row) in enumerate(zip(diagnostics, calibration["rows"])):
        actual = diagnostic.get("initialization")
        expected = {"base_seed": row["base_seed"], "namespace": "calibration",
                    "prompt_state_seed": row["prompt_state_seed"], "input_ids_sha256": row["input_ids_sha256"],
                    "seed_recipe": SEED_RULE, "dtype": "torch.bfloat16", "test_time_noise": 0,
                    "one_prompt_shape": [1, row["token_length"], 5280]}
        if (diagnostic.get("index") != index or not isinstance(actual, dict)
                or any(not ref._same(actual.get(key), value) for key, value in expected.items())):
            raise ValueError("a completed Huginn paragraph used a different initialization")
    return {"rows": 100, "initializations_sha256": runner._digest([row["initialization"] for row in diagnostics])}


@contextmanager
def _states(model, input_ids):
    records, counts, handles = {}, [0] * 34, []
    def hook(index):
        def capture(module, args, output):
            counts[index] += 1
            records[index] = output
        return capture
    try:
        for index, tap in enumerate(model.layers):
            handles.append(tap.register_forward_hook(hook(index)))
        target = model.forward(input_ids)
        if counts != [1] * 34:
            raise ValueError(f"Huginn evaluation hook invocation counts changed: {counts}")
    finally:
        for handle in handles:
            handle.remove()
    # Hooks are removed before the diagnostic coda forwards below.
    yield target, records


def _common_registry(evaluator, items, plan):
    masked = []
    for item in items:
        changed = copy.copy(item)
        common = {plan["task_names"][item.task][index] for index in plan["common_name_indices"][item.task]}
        changed.intermediate_tokens = {name: list(forms) if name in common else []
                                       for name, forms in item.intermediate_tokens.items()}
        masked.append(changed)
    return {task: evaluator.TaskNames(masked, task) for task in TASKS}


def _summaries(arrays, plan):
    """Preserve the original slot/item weighting and per-control any-layer rule."""
    import numpy as np
    summaries = {}
    for method in METHODS:
        ranks = arrays[f"{method}_allrank"]
        task_rows = {task: [] for task in TASKS}
        for index, row in enumerate(plan["rows"]):
            own, control, any_own, any_control = [], [], [], []
            for slot, eligible in enumerate(row["eligible"]):
                if not eligible:
                    continue
                target = ranks[index, row["own_index"][slot]]
                controls = ranks[index, row["control_indices"][slot]]
                if np.any(target < 0) or np.any(controls < 0) or len(controls) == 0:
                    raise ValueError("unsupported name entered a score denominator")
                own.append(target < 10)
                control.append((controls < 10).mean(axis=0))
                any_own.append(bool((target < 10).any()))
                any_control.append(float((controls < 10).any(axis=1).mean()))
            if own:
                task_rows[row["task"]].append((np.mean(own, axis=0), np.mean(control, axis=0),
                                               float(np.mean(any_own)), float(np.mean(any_control))))
        summaries[method] = {}
        for task, values in task_rows.items():
            if len(values) != plan["counts"]["eligible_items"][task] or not values:
                raise ValueError("score denominator differs from the frozen eligibility plan")
            own = np.mean([row[0] for row in values], axis=0)
            control = np.mean([row[1] for row in values], axis=0)
            any_own = float(np.mean([row[2] for row in values]))
            any_control = float(np.mean([row[3] for row in values]))
            summaries[method][task] = {"n_items": len(values), "own_curve": own.tolist(),
                                      "control_curve": control.tolist(), "excess_curve": (own - control).tolist(),
                                      "any_layer_own": any_own, "any_layer_control": any_control,
                                      "any_layer_excess": any_own - any_control, "any_layer_opportunities": 32}
    return summaries


def _validate_arrays(arrays, plan):
    import numpy as np
    expected = {f"{method}_{key}" for method in METHODS for key in ("allrank", "rank", "top1")}
    expected.add("native_top1")
    if set(arrays) != expected:
        raise ValueError("Huginn arrays have missing or unknown readout fields")
    n = len(plan["rows"])
    if (arrays["native_top1"].shape != (n,) or arrays["native_top1"].dtype != np.int64
            or np.any(arrays["native_top1"] < 0) or np.any(arrays["native_top1"] >= 65536)):
        raise ValueError("native top1 geometry changed")
    for method in METHODS:
        ranks, own, top1 = (arrays[f"{method}_{name}"] for name in ("allrank", "rank", "top1"))
        if (ranks.shape != (n, 128, 32) or ranks.dtype != np.int32
                or own.shape != (n, 3, 32) or own.dtype != np.int32
                or top1.shape != (n, 32) or top1.dtype != np.int64):
            raise ValueError("Huginn rank/name/core geometry changed")
        for index, row in enumerate(plan["rows"]):
            supported = np.zeros(128, dtype=bool)
            supported[plan["common_name_indices"][row["task"]]] = True
            if (np.any(ranks[index, supported] < 0) or np.any(ranks[index, supported] >= 65536)
                    or np.any(ranks[index, ~supported] != -1)
                    or np.any(top1[index] < 0) or np.any(top1[index] >= 65536)):
                raise ValueError("common name support or explicit unsupported mask changed")
            for slot, name_index in enumerate(row["own_index"]):
                expected_row = ranks[index, name_index] if name_index >= 0 else np.full(32, -1, np.int32)
                if not np.array_equal(own[index, slot], expected_row):
                    raise ValueError("own-slot ranks do not reference the original name catalogue")
    if not np.array_equal(arrays["coda_top1"][:, 31], arrays["native_top1"]):
        raise ValueError("final native coda top1 does not match the final-pass coda readout")


def evaluate_seed(model, jacobians, items, plan, evaluator, *, seed,
                  stop_requested=lambda: False, deadline=None, reserve_seconds=600.0, cpu_test=False):
    """One complete seed, with shared states for all three readout operators."""
    import numpy as np
    import torch
    if (seed not in SEEDS or type(seed) is not int or tuple(model.source_layers) != SOURCES
            or model.target_layer != 33 or model.n_layers != 34 or model.n_ut != 8
            or (not cpu_test and (model.d_model != 5280 or len(items) != 148))):
        raise ValueError("evaluation must retain both fixed seeds and the full declared Huginn geometry")
    if (not isinstance(jacobians, torch.Tensor) or jacobians.dtype != torch.float32
            or tuple(jacobians.shape) != (32, model.d_model, model.d_model)
            or jacobians.device != model.input_device or not torch.isfinite(jacobians).all()):
        raise ValueError("Huginn readout requires the complete finite FP32 map bank")
    if len(items) != len(plan["rows"]):
        raise ValueError("evaluation item count differs from its frozen plan")
    model.state_seed, model.seed_namespace = seed, "evaluation"
    registry = _common_registry(evaluator, items, plan)
    for task in TASKS:
        expected = [plan["task_names"][task][index] for index in plan["common_name_indices"][task]]
        if registry[task].names != expected:
            raise ValueError("common task-name order changed")
        if not ref._same({name: sorted(forms) for name, forms in registry[task].forms.items()},
                        {name: plan["token_forms"]["huginn"][task][name] for name in expected}):
            raise ValueError("runtime Huginn single-token forms changed")
    n = len(items)
    arrays = {f"{method}_{name}": np.full(shape, -1, dtype)
              for method in METHODS for name, shape, dtype in (
                  ("allrank", (n, 128, 32), np.int32), ("rank", (n, 3, 32), np.int32),
                  ("top1", (n, 32), np.int64))}
    arrays["native_top1"] = np.full(n, -1, np.int64)
    states = torch.empty((n, 32, model.d_model), dtype=torch.float32)
    target_states = torch.empty((n, model.d_model), dtype=torch.float32)
    native_logits = []
    initializations = []
    started, maximum_item = time.monotonic(), 30.0
    def save_logits(index, row, method, logits):
        if logits.ndim != 2 or logits.shape[0] != 32 or not torch.isfinite(logits).all():
            raise ValueError("Huginn readout logits are nonfinite or have changed core geometry")
        ranks = registry[row["task"]].ranks(logits)
        indices = plan["common_name_indices"][row["task"]]
        arrays[f"{method}_allrank"][index, indices] = ranks
        for slot, own_index in enumerate(row["own_index"]):
            if own_index >= 0:
                arrays[f"{method}_rank"][index, slot] = arrays[f"{method}_allrank"][index, own_index]
        arrays[f"{method}_top1"][index] = logits.argmax(-1).cpu().numpy()
    with torch.no_grad():
        for index, (item, row) in enumerate(zip(items, plan["rows"])):
            if (stop_requested() or (deadline is not None
                    and (deadline - runner._utc_now()).total_seconds() <= reserve_seconds + maximum_item)):
                return {"status": "stopped", "n_items_done": index, "seed": seed,
                        "arrays": {key: value[:index] for key, value in arrays.items()},
                        "cache": {"H": states[:index], "target_states": target_states[:index],
                                  "native_logits": torch.stack(native_logits) if native_logits else torch.empty(0)},
                        "initializations": initializations, "elapsed_seconds": time.monotonic() - started}
            before = time.monotonic()
            if ((item.name, item.task, item.prompt, item.target, item.intermediates, list(item.token_ids))
                    != (row["name"], row["task"], row["prompt"], row["target"], row["intermediates"], row["huginn"]["token_ids"])
                    or not ref._same(item.intermediate_tokens, row["huginn"]["intermediate_tokens"])
                    or not ref._same([bool(item.leaked[name]) for name in item.intermediates], row["huginn"]["leaked"])):
                raise ValueError("a runtime Huginn input differs from the frozen eligibility context")
            ids = torch.tensor([item.token_ids], dtype=torch.long, device=model.input_device)
            initialization = model.initialization_metadata(ids)
            with _states(model, ids) as (target, captured):
                if not ref._same(model.last_initialization, initialization):
                    raise ValueError("actual Huginn initial state recipe differs from its prechecked seed")
                h = torch.stack([captured[cell][0, -1].float() for cell in SOURCES])
                if not torch.isfinite(h).all() or not torch.isfinite(target).all():
                    raise ValueError("Huginn evaluation states contain nonfinite values")
                states[index], target_states[index] = h.cpu(), target[0, -1].float().cpu()
                native = model.unembed(target)[0, -1].float()
                native_logits.append(native.cpu())
                arrays["native_top1"][index] = int(native.argmax().item())
                save_logits(index, row, "raw", model.unembed(h).float())
                transported = torch.einsum("vde,ve->vd", jacobians, h)
                save_logits(index, row, "jlens", model.unembed(transported).float())
                coda = torch.stack([model.coda_readout(captured[cell], ids)[0, -1].float() for cell in SOURCES])
                if not torch.equal(coda[31], native):
                    raise ValueError("complete-sequence final-pass coda logits differ from the actual final coda")
                save_logits(index, row, "coda", coda)
            initializations.append({"index": index, "name": item.name, "task": item.task, **initialization})
            maximum_item = max(maximum_item, time.monotonic() - before)
            print(json.dumps({"stage": "huginn_readout", "seed": seed, "items_done": index + 1,
                              "items_total": n, "elapsed_seconds": time.monotonic() - started}), flush=True)
    _validate_arrays(arrays, plan)
    return {"status": "complete", "n_items_done": n, "seed": seed, "arrays": arrays,
            "cache": {"H": states, "target_states": target_states, "native_logits": torch.stack(native_logits)},
            "initializations": initializations, "summaries": _summaries(arrays, plan),
            "elapsed_seconds": time.monotonic() - started}


def _evaluation_runtime(torch, args, main, snapshot_files, imported):
    environment = runner._environment(torch, main["environment"])
    if not torch.cuda.is_available():
        raise RuntimeError("Huginn evaluation requires an available CUDA GPU")
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    gpu = {"index": index, "name": properties.name, "total_memory": properties.total_memory,
           "capability": [properties.major, properties.minor], "uuid": str(getattr(properties, "uuid", "unavailable")),
           "multiprocessor_count": properties.multi_processor_count}
    model = _adapter().load_huginn(args.huginn_snapshot, device=f"cuda:{index}", dtype=torch.bfloat16,
                                   state_seed=SEEDS[0], seed_namespace="evaluation")
    if (any(module.training for module in model.hf_model.modules())
            or any(parameter.requires_grad or parameter.dtype != torch.bfloat16
                   or parameter.device != torch.device("cuda", index) for parameter in model.hf_model.parameters())):
        raise ValueError("Huginn evaluation requires frozen BF16 parameters on the chosen GPU")
    remote = []
    for cls in (type(model.hf_model), type(model.hf_model.config)):
        path = Path(inspect.getfile(cls))
        digest = runner._file_hash(path)
        if path.name not in snapshot_files or snapshot_files[path.name]["sha256"] != digest:
            raise ValueError("loaded Huginn native implementation differs from its pinned snapshot")
        remote.append({"class": f"{cls.__module__}.{cls.__name__}", "path": str(path), "sha256": digest})
    return model, {"environment": environment, "gpu": gpu, "precision": runner._precision(torch),
                   "imported_sources": imported, "remote_implementations": remote,
                   "attention_implementation": str(getattr(model.hf_model.config, "_attn_implementation", None)),
                   "snapshot_files": snapshot_files, "snapshot_path_at_evaluation": str(Path(args.huginn_snapshot).absolute())}


def validate_output(path):
    """Verify all seals and the mandatory two-seed/full-core array semantics."""
    import numpy as np
    import torch
    checked = ref.validate_output(path, schema=SCHEMA)
    root = runner._no_links(path)
    identity = checked["owner"]["identity"]
    if (identity.get("sections") != SECTIONS or identity.get("evaluation_base_seeds") != list(SEEDS)
            or identity.get("learned_sources") != list(SOURCES) or identity.get("target_layer") != 33
            or identity.get("methods") != list(METHODS) or identity.get("evaluation_max_length") != 512):
        raise ValueError("completed Huginn evaluation has a different seed/source/target contract")
    plan = runner._json(root / "population" / "eligibility.json")
    if (plan.get("schema") != ELIGIBILITY_SCHEMA or len(plan.get("rows", [])) != 148
            or runner._digest(plan) != identity.get("eligibility_payload_sha256")
            or plan.get("counts", {}).get("tasks") != ref.TASK_COUNTS
            or plan.get("evaluation_base_seeds") != list(SEEDS)
            or plan.get("evaluation_max_length") != 512):
        raise ValueError("completed Huginn eligibility population differs from its owner")
    for seed in SEEDS:
        section = root / "seeds" / str(seed)
        metadata = runner._json(section / "metadata.json")
        if (metadata.get("status") != "complete" or metadata.get("seed") != seed
                or metadata.get("n_items_done") != 148 or metadata.get("native_exit_indices") != list(range(3, 32, 4))):
            raise ValueError("a completed Huginn seed is partial or has changed exit semantics")
        with np.load(section / "arrays.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        _validate_arrays(arrays, plan)
        cache = torch.load(section / "cache.pt", map_location="cpu", weights_only=True, mmap=True)
        expected_shapes = {"H": (148, 32, 5280), "target_states": (148, 5280), "native_logits": (148, 65536)}
        if not isinstance(cache, dict) or set(cache) != set(expected_shapes):
            raise ValueError("Huginn state cache fields changed")
        for key, shape in expected_shapes.items():
            value = cache[key]
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                    or value.device.type != "cpu" or value.requires_grad or tuple(value.shape) != shape
                    or not torch.isfinite(value).all()):
                raise ValueError("Huginn state cache is partial, nonfinite or has changed geometry")
        if not np.array_equal(cache["native_logits"].argmax(-1).numpy(), arrays["native_top1"]):
            raise ValueError("Huginn stored native predictions differ from their actual logits")
        del cache
        if not ref._same(runner._json(section / "summaries.json"), _summaries(arrays, plan)):
            raise ValueError("Huginn saved summaries do not equal the frozen rank/control reduction")
        initializations = runner._json(section / "initializations.json")
        if len(initializations) != 148:
            raise ValueError("Huginn initial-state provenance is incomplete")
        for index, (row, recorded) in enumerate(zip(plan["rows"], initializations)):
            recipe = {"version": 1, "base_seed": seed, "namespace": "evaluation", "input_ids": row["huginn"]["token_ids"]}
            actual_seed = int.from_bytes(bytes.fromhex(runner._digest(recipe))[:8], "big") % (2 ** 63)
            if any(recorded.get(key) != value for key, value in {
                    "index": index, "name": row["name"], "task": row["task"], "base_seed": seed,
                    "namespace": "evaluation", "prompt_state_seed": actual_seed,
                    "input_ids_sha256": runner._digest(row["huginn"]["token_ids"]), "seed_recipe": SEED_RULE}.items()):
                raise ValueError("a Huginn item has a changed initial-state seed binding")
    return checked


def _execute(args, main, combined, manifest, calibration, plan, deadline):
    import torch
    fit_dir, owner = ref._owner(args.fit_dir)
    validate_huginn_owner(owner, main, combined, manifest, calibration,
                          combined_sha256=runner._file_hash(args.combined_contract),
                          calibration_sha256=runner._file_hash(args.huginn_calibration))
    snapshot_files = runner._snapshot(args.huginn_snapshot, manifest)
    lens_path, fit_record = ref._read_generation(torch, fit_dir, owner)
    fit_record["initialization_validation"] = _fit_initializations(lens_path, calibration)
    if args.validate_only:
        return {"status": "validated", "fit_identity_sha256": owner["fit_identity_sha256"],
                "eligibility": plan["counts"], "cuda_initialized": torch.cuda.is_initialized()}
    if (deadline - runner._utc_now()).total_seconds() <= args.reserve_seconds + 30:
        return {"status": "stopped", "reason": "deadline reserve before model loading", "complete": False}
    identity = {"kind": "huginn_r8_n100_readout", "run_spec_sha256": main["spec_sha256"],
                "combined_contract_sha256": runner._file_hash(args.combined_contract),
                "source_files": main["sources"], "fit": fit_record,
                "model": {key: combined["huginn"][key] for key in ("repo_id", "revision", "manifest_sha256")},
                "eligibility_file": runner._record(args.eligibility), "eligibility_payload_sha256": runner._digest(plan),
                "evaluation_base_seeds": list(SEEDS), "sections": SECTIONS, "methods": list(METHODS),
                "learned_sources": list(SOURCES), "target_layer": 33, "position": -1,
                "evaluation_max_length": 512, "created_utc": runner._utc_now().isoformat()}
    out = ref._new_output(args.out, identity, schema=SCHEMA)
    population = runner._mkdir(out / "population")
    runner._new_json(population / "eligibility.json", plan)
    sections = [ref._seal_section(out, "population", {"kind": "frozen_common_eligibility", "counts": plan["counts"]})]
    stopped = [False]
    old_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    try:
        with ref._runtime_scope(out.parent):
            evaluator, _, imported = ref._imports(main, args.ouro_src)
            imported.append({"module": "huginn_adapter", "root": "jlens",
                             "path": (HERE / "huginn_adapter.py").relative_to(REPO).as_posix(),
                             "sha256": runner._file_hash(HERE / "huginn_adapter.py")})
            model, runtime = _evaluation_runtime(torch, args, main, snapshot_files, imported)
            # Rebuild the native contexts using the actual loaded tokenizer.
            items = evaluator.load_items(model.tokenizer, TASKS,
                                         encode=lambda text: model.encode(text, max_length=512)[0].tolist())
            runner._verify_record(lens_path, fit_record["lens"])
            lens = importlib.import_module("jlens").JacobianLens.load(str(lens_path))
            if (lens.n_prompts != 100 or lens.d_model != 5280 or tuple(lens.source_layers) != SOURCES):
                raise ValueError("loaded Huginn lens does not retain the complete R8 N100 bank")
            jacobians = torch.stack([lens.jacobians[cell].float() for cell in SOURCES]).to(model.input_device)
            del lens
            gc.collect()
            for seed in SEEDS:
                result = evaluate_seed(model, jacobians, items, plan, evaluator, seed=seed,
                                       stop_requested=lambda: stopped[0], deadline=deadline,
                                       reserve_seconds=args.reserve_seconds)
                section = f"seeds/{seed}"
                path = runner._mkdir(out / section)
                ref._write_npz(path / "arrays.npz", result.pop("arrays"))
                ref._write_torch(path / "cache.pt", result.pop("cache"))
                runner._new_json(path / "initializations.json", result.pop("initializations"))
                if result["status"] != "complete":
                    runner._new_json(path / "INCOMPLETE.json", result)
                    return {"status": "stopped", "out": str(out), "seed": seed,
                            "n_items_done": result["n_items_done"], "complete": False}
                runner._new_json(path / "summaries.json", result.pop("summaries"))
                sections.append(ref._seal_section(out, section, {
                    **result, "runtime": runtime, "position": -1, "virtual_indices": list(SOURCES),
                    "native_exit_indices": list(range(3, 32, 4)),
                    "coda_policy": "full raw sequence -> pre-coda norm -> two coda blocks -> final norm/head; select last logits afterwards",
                    "within_pass_coda": "diagnostic interrupted-core intervention; only physical block3 is a native recurrence exit",
                    "readout_policy": "native raw and JLens last-state vectors use final norm/head; maps are FP32 at application",
                    "matrix_policy": "one preselected N100 map bank; the two seeds measure state-initialization sensitivity"}))
            ref._complete_output(out, sections)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    checked = validate_output(out)
    return {"status": "complete", "schema": SCHEMA, "out": str(out), "items": 148,
            "evaluation_base_seeds": list(SEEDS), "files": checked["files"],
            "complete": runner._record(out / "COMPLETE.json")}


def self_test(*, huginn_snapshot, ouro_src):
    """Native tiny CPU inference plus independent rank/control bookkeeping."""
    import numpy as np
    import torch
    sys.path[:0] = [str(REPO), str(Path(ouro_src).absolute())]
    evaluator = importlib.import_module("ouro_jlens.evaluate")
    Item = importlib.import_module("ouro_jlens.evaldata").Item
    adapter = _adapter()
    RavenConfig, RavenForCausalLM = adapter._native_classes(huginn_snapshot)
    class Tokenizer:
        bos_token_id, eos_token_id, pad_token_id = 1, 2, 31
        def decode(self, ids):
            return ":".join(str(value) for value in ids)
    tokenizer = Tokenizer()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260908)
        config = RavenConfig(n_embd=8, n_heads=2, intermediate_size=16, vocab_size=32,
                             block_size=16, bos_token_id=1, eos_token_id=2, pad_token_id=31)
        native = RavenForCausalLM(config)
        native.freqs_cis = native._precompute_freqs_cis()
        with torch.no_grad():
            native.transformer.ln_f.weight.copy_(torch.linspace(0.65, 1.35, 8))
    model = adapter.HuginnLensModel(native, tokenizer, cpu_test=True)
    ouro_items = []
    for index, (task, name, forms) in enumerate((
            ("multihop", "A", [3, 4]), ("multihop", "B", [5]), ("multihop", "C", [6]),
            ("order-ops", "1", [7]), ("order-ops", "2", [8]), ("order-ops", "3", [9]))):
        item = Item(f"fixture_{index}", task, f"Fixture prompt {index}", " answer", [name],
                    [1, 18, 19 + index], {name: forms}, {name: False})
        ouro_items.append(item)
    huginn_items = copy.deepcopy(ouro_items)
    huginn_items[0].token_ids[1] = 3
    huginn_items[0].leaked["A"] = True
    huginn_items[2].intermediate_tokens["C"] = []
    plan = _build_population(tokenizer, tokenizer, ouro_items, huginn_items, evaluator, historical=False)
    checks = []
    def require(value, label):
        if not value:
            raise AssertionError(label)
        checks.append(label)
    def rejects(call, label):
        try:
            call()
        except (ValueError, AssertionError):
            checks.append(label)
        else:
            raise AssertionError(label)
    require(plan["counts"]["eligible_items"] == {"multihop": 1, "order-ops": 3}, "joint eligibility excludes either-tokenizer leakage and unsupported forms")
    require(plan["excluded_control_names"] == {"multihop": ["C"], "order-ops": []}, "shared control catalogue excludes unsupported labels before scoring")
    require(plan["rows"][1]["control_indices"] == [[0]], "common controls exclude every own label")
    matrix = torch.eye(8).expand(32, -1, -1).clone()
    random_before = torch.random.get_rng_state().clone()
    before_hooks = [len(module._forward_hooks) for module in model.hf_model.modules()]
    first = evaluate_seed(model, matrix, huginn_items, plan, evaluator, seed=SEEDS[0], cpu_test=True)
    second = evaluate_seed(model, matrix, huginn_items, plan, evaluator, seed=SEEDS[1], cpu_test=True)
    repeated = evaluate_seed(model, matrix, huginn_items, plan, evaluator, seed=SEEDS[0], cpu_test=True)
    require(torch.equal(random_before, torch.random.get_rng_state()), "evaluation uses private initial-state generators without changing global RNG")
    require(first["status"] == second["status"] == "complete", "both frozen seeds evaluate every fixture item and all 32 core cells")
    require(all(np.array_equal(first["arrays"][key], repeated["arrays"][key]) for key in first["arrays"])
            and torch.equal(first["cache"]["H"], repeated["cache"]["H"]), "same seed repeats bitwise state and rank arrays")
    require(all(np.array_equal(first["arrays"][f"raw_{key}"], first["arrays"][f"jlens_{key}"])
                for key in ("rank", "allrank", "top1")), "identity transport reproduces raw logits ranks exactly")
    require(all(a["prompt_state_seed"] != b["prompt_state_seed"]
                for a, b in zip(first["initializations"], second["initializations"])), "second base seed changes every recorded initial-state seed")
    require(before_hooks == [len(module._forward_hooks) for module in model.hf_model.modules()], "evaluation and diagnostic coda calls leave no hooks")
    with torch.no_grad():
        ids = torch.tensor([huginn_items[0].token_ids])
        model.state_seed = SEEDS[0]
        native_logits = model.native_logits(ids)[0, -1]
    require(torch.equal(native_logits, first["cache"]["native_logits"][0]), "cached actual native final logits match direct native inference")
    stopped = evaluate_seed(model, matrix, huginn_items, plan, evaluator, seed=SEEDS[0],
                            stop_requested=lambda: True, cpu_test=True)
    require(stopped["status"] == "stopped" and stopped["n_items_done"] == 0, "stop before first item produces no completed seed")
    expired = evaluate_seed(model, matrix, huginn_items, plan, evaluator, seed=SEEDS[0],
                            deadline=runner._utc_now(), cpu_test=True)
    require(expired["status"] == "stopped" and expired["n_items_done"] == 0, "expired deadline prevents the first forward")
    rejects(lambda: evaluate_seed(model, matrix[:31], huginn_items, plan, evaluator, seed=SEEDS[0], cpu_test=True), "missing learned core map rejected")
    rejects(lambda: evaluate_seed(model, matrix, huginn_items, plan, evaluator, seed=1, cpu_test=True), "unfrozen evaluation seed rejected")
    changed_items = copy.deepcopy(huginn_items)
    changed_items[0].token_ids[-1] += 1
    rejects(lambda: evaluate_seed(model, matrix, changed_items, plan, evaluator, seed=SEEDS[0], cpu_test=True), "changed readout context rejected before forward")
    changed_arrays = {key: value.copy() for key, value in first["arrays"].items()}
    changed_arrays["raw_allrank"][0, 2] = 0
    rejects(lambda: _validate_arrays(changed_arrays, plan), "unsupported control name cannot enter stored ranks")
    # Two controls recover at disjoint cells: average their individual any-cell
    # events (1), rather than take the maximum of their mean hit curve (1/2).
    score_plan = copy.deepcopy(plan)
    for row in score_plan["rows"][4:]:
        row["eligible"] = [False]
    score_plan["counts"]["eligible_items"]["order-ops"] = 1
    synthetic = {key: value.copy() for key, value in first["arrays"].items()}
    for method in METHODS:
        for index, row in enumerate(plan["rows"]):
            synthetic[f"{method}_allrank"][index, plan["common_name_indices"][row["task"]]] = 31
        synthetic[f"{method}_allrank"][3, 1, 0] = 0
        synthetic[f"{method}_allrank"][3, 2, 1] = 0
    summary = _summaries(synthetic, score_plan)["raw"]["order-ops"]
    require(summary["any_layer_control"] == 1.0 and max(summary["control_curve"]) == 0.5,
            "any-layer controls are reduced separately before the control mean")
    require(not torch.cuda.is_initialized(), "CPU proof never initializes CUDA")
    return {"status": "pass", "checks": checks, "counts": plan["counts"],
            "native_model": "verified Raven source with tiny random CPU parameters; no checkpoint weight load",
            "native_final_logits_bitwise": True, "cuda_initialized": torch.cuda.is_initialized(),
            "source_sha256": runner._file_hash(Path(__file__))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-spec", type=Path, default=HERE / "run_spec.json")
    parser.add_argument("--combined-contract", type=Path, default=HERE / "combined_contract.json")
    parser.add_argument("--huginn-calibration", type=Path, default=HERE / "huginn_calibration.json")
    parser.add_argument("--ouro-manifest", type=Path, default=HERE / "model_manifest.json")
    parser.add_argument("--huginn-manifest", type=Path, default=HERE / "huginn_model_manifest.json")
    parser.add_argument("--eligibility", type=Path, default=HERE / "huginn_eligibility.json")
    parser.add_argument("--ouro-snapshot", type=Path)
    parser.add_argument("--huginn-snapshot", type=Path)
    parser.add_argument("--ouro-src", type=Path)
    parser.add_argument("--fit-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--stop-at-utc")
    parser.add_argument("--reserve-seconds", type=float, default=600.0)
    parser.add_argument("--prepare-eligibility", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validate-output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    if args.self_test:
        if args.huginn_snapshot is None or args.ouro_src is None:
            parser.error("--self-test requires --huginn-snapshot and --ouro-src")
        import torch
        torch.set_num_threads(1)
        result = self_test(huginn_snapshot=args.huginn_snapshot, ouro_src=args.ouro_src)
        if args.out is not None:
            runner._new_json(args.out, result)
    elif args.validate_output is not None:
        checked = validate_output(args.validate_output)
        result = {"status": "validated", "files": checked["files"], "schema": SCHEMA}
    else:
        if args.ouro_src is None or args.ouro_snapshot is None or args.huginn_snapshot is None:
            parser.error("--ouro-src, --ouro-snapshot and --huginn-snapshot are required")
        if not args.prepare_eligibility and (args.fit_dir is None or (not args.validate_only and (args.out is None or args.stop_at_utc is None))):
            parser.error("execution requires --fit-dir, --out and --stop-at-utc; --validate-only requires --fit-dir")
        if args.prepare_eligibility and (args.fit_dir is not None or args.validate_only):
            parser.error("eligibility preparation cannot consume fit outcomes")
        import math
        if not math.isfinite(args.reserve_seconds) or args.reserve_seconds < 0:
            parser.error("--reserve-seconds must be finite and nonnegative")
        deadline = runner._deadline(args.stop_at_utc)
        import torch
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        plan, prepared = prepare_eligibility(ouro_snapshot=args.ouro_snapshot, huginn_snapshot=args.huginn_snapshot,
                                             ouro_src=args.ouro_src, ouro_manifest=args.ouro_manifest,
                                             huginn_manifest=args.huginn_manifest)
        if args.prepare_eligibility:
            runner._new_json(args.eligibility, plan)
            result = {"status": "prepared", "counts": plan["counts"], "excluded_control_names": plan["excluded_control_names"],
                      "eligibility": runner._record(args.eligibility), "cuda_initialized": torch.cuda.is_initialized()}
        else:
            if not ref._same(plan, runner._json(runner._no_links(args.eligibility))):
                raise ValueError("current tokenizers, sources or eligibility differ from the pre-inference frozen file")
            contract = _contract(args)
            result = _execute(args, *contract, plan, deadline)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
