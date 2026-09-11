"""Evaluate sealed Ouro N100 refits with the unchanged concept readout rules.

Production accepts the main target-191 profile only. It verifies OWNER,
LATEST, and COMPLETE with run_refits before loading one lens at a time. The
fitting GPU identity is retained as evidence; inference has its own recorded
GPU/runtime. There is no rental, upload, legacy-sidecar fabrication, plotting,
or bootstrap here. Import and --self-test do not initialize CUDA.

The geometry-only prepare_readout API also supports a future penultimate
control: it exposes only learned columns 0..189 for target190. It does not
authorize another artifact schema. Main target191 additionally exposes the
known self-identity at191 to preserve the historical 192-column readout.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

try:
    from . import run_refits as runner
except ImportError:
    import run_refits as runner


HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
REPO = HERE.parents[2]
TASKS = ("multihop", "order-ops")
TASK_COUNTS = {"multihop": 93, "order-ops": 55}
STIMULUS_HASHES = {
    "multihop": "50b7e4c9255291c0ca2a8e94615be9f44531fa57bb1a844e4f9616056d987416",
    "order-ops": "b203206d16ff628152cc86f3838604e06cb54776f3e14fa1c34f150db8bc7560",
}
# Canonical hashes of the retained historical population; generation outputs
# and model correctness are deliberately excluded from this input identity.
POPULATION_FIELDS = (
    "name", "task", "prompt", "target", "intermediates", "own_index",
    "scorable", "leaked", "n_tokens", "readout_token",
)
POPULATION_SHA256 = "97214457b3b0208aaedef34d588287a4e6d6eb8d9fccc8dbafbf30d51134a1f9"
TASK_NAMES_SHA256 = "596b81d1846a58513fb6069f7d4f1bac1358ca7d0162e7571e34378761c00ead"
TOKEN_FORMS_SHA256 = "6e4a262eded8b6b2ab4538625571f4b8d7fae3b9a046dc874f6ab1801ce48f90"
SCHEMA = "ouro_refit_evaluation.v1"


def _same(left, right):
    return runner._canonical(left) == runner._canonical(right)


def _evaluation_source_keys():
    return {
        ("jlens", Path(__file__).relative_to(REPO).as_posix()),
        ("jlens", "jlens/data/slice_vis.html"),
        *(("ouro_src", f"ouro_jlens/{name}.py") for name in (
            "__init__", "evaluate", "evaldata", "fit_lens", "evidence", "recurrent",
        )),
        *(("jlens", f"data/evaluations/lens-eval-{task}.json") for task in TASKS),
    }


def _evaluation_sources(contract):
    missing = _evaluation_source_keys() - set(contract["source_paths"])
    if missing:
        raise ValueError(f"run_spec omits evaluation inputs: {sorted(missing)}")
    for task, expected in STIMULUS_HASHES.items():
        key = ("jlens", f"data/evaluations/lens-eval-{task}.json")
        if runner._file_hash(contract["source_paths"][key]) != expected:
            raise ValueError(f"the frozen {task} stimulus bytes changed")


def _owner(path):
    path = runner._no_links(path)
    if not path.is_dir():
        raise ValueError(f"fit directory is missing: {path}")
    owner_path = runner._no_links(path / "OWNER.json")
    owner = runner._json(owner_path)
    if (not isinstance(owner, dict)
            or set(owner) != {"schema_version", "fit_identity_sha256", "identity"}
            or type(owner["schema_version"]) is not int or owner["schema_version"] != 1
            or not isinstance(owner["identity"], dict)
            or owner["fit_identity_sha256"] != runner._digest(owner["identity"])):
        raise ValueError(f"invalid OWNER identity: {owner_path}")
    return path, owner


def _fit_import_keys():
    prefix = ROUND.relative_to(REPO).as_posix()
    return {
        "fit_estimators": ("jlens", f"{prefix}/fit_estimators.py"),
        **{f"optimization.{name}": ("jlens", f"{prefix}/optimization/{name}.py")
           for name in ("optimized_fitting", "cuda_graph_candidate", "saved_tensor_candidate")},
        **{f"ouro_jlens.{name}": ("ouro_src", f"ouro_jlens/{name}.py")
           for name in ("recurrent", "evidence")},
        "jlens": ("jlens", "jlens/__init__.py"),
        **{f"jlens.{name}": ("jlens", f"jlens/{name}.py")
           for name in ("fitting", "hooks", "hf", "lens", "protocol")},
    }


def _validate_owner_binding(owner, contract):
    """Validate logical bindings; producer absolute paths are never opened."""
    identity = owner["identity"]
    fit_id = identity.get("fit_id")
    if type(fit_id) is not int or fit_id not in range(1, 6):
        raise ValueError("fit identity must select one of the five frozen fits")
    fit = next((entry for entry in contract["fits"] if entry["fit_id"] == fit_id), None)
    if fit is None:
        raise ValueError("fit identity is absent from the supplied run specification")
    runtime = identity.get("runtime")
    required = {
        "run_spec_sha256", "model", "snapshot_path", "snapshot_files", "source_files",
        "imported_sources", "remote_implementations", "environment", "gpu", "precision",
        "settings", "attention_implementation", "calibration",
    }
    if not isinstance(runtime, dict) or set(runtime) != required:
        raise ValueError("the fitting runtime binding is incomplete or has an unknown schema")
    expected_identity = runner._fit_identity(
        fit["prompts"], fit["token_lengths"], fit["n_valid"], fit_id, runtime,
        2048, tuple(range(191)), False,
    )
    if not _same(identity, expected_identity):
        raise ValueError("OWNER does not describe the complete main N100 population/geometry")
    for name, expected in {
        "run_spec_sha256": contract["spec_sha256"], "model": contract["spec"]["model"],
        "source_files": contract["sources"], "settings": runner.SETTINGS,
    }.items():
        if not _same(runtime[name], expected):
            raise ValueError(f"fitting runtime {name} differs from the supplied run specification")
    # prompt_path records where the producer read the list. Its hash, seed,
    # ordered paragraphs and token lengths are checked at the relocated path.
    calibration = runtime["calibration"]
    expected_calibration = {key: value for key, value in fit.items() if key != "prompts"}
    if (not isinstance(calibration, dict) or set(calibration) != set(expected_calibration)
            or not isinstance(calibration.get("prompt_path"), str)
            or not _same({k: v for k, v in calibration.items() if k != "prompt_path"},
                         {k: v for k, v in expected_calibration.items() if k != "prompt_path"})):
        raise ValueError("fitting calibration binding differs from the frozen input list")
    snapshot_files = {row["path"]: {"bytes": row["bytes"], "sha256": row["sha256"]}
                      for row in contract["manifest"]["files"]}
    if not _same(runtime["snapshot_files"], snapshot_files):
        raise ValueError("fitting snapshot file hashes differ from the pinned model")
    environment = runtime["environment"]
    expected_environment = contract["environment"]
    for name in ("packages", "torch_git", "python_version", "cuda_build_version", "cudnn_runtime_integer"):
        if (not isinstance(environment, dict) or name not in environment
                or not _same(environment[name], expected_environment.get(name))):
            raise ValueError(f"fitting environment lacks the pinned {name}")
    if (not isinstance(runtime["gpu"], dict) or not runtime["gpu"].get("name")
            or not isinstance(runtime["precision"], dict) or not runtime["precision"]
            or runtime["attention_implementation"] != "sdpa"):
        raise ValueError("fitting GPU, precision, or SDPA identity is missing")
    declared = {(row["root"], row["path"]): row["sha256"] for row in contract["sources"]}
    imported = runtime["imported_sources"]
    import_keys = _fit_import_keys()
    if (not isinstance(imported, list)
            or {row.get("module") for row in imported if isinstance(row, dict)} != set(import_keys)
            or len(imported) != len(import_keys)):
        raise ValueError("fitting imported-source membership changed")
    for row in imported:
        if (set(row) != {"module", "path", "sha256"} or not isinstance(row["path"], str)
                or row["sha256"] != declared[import_keys[row["module"]]]):
            raise ValueError("fitting imported-source hash differs from its logical source")
    remote = runtime["remote_implementations"]
    if not isinstance(remote, list) or len(remote) != 2:
        raise ValueError("fitting remote implementation identity is missing")
    expected_remote = {snapshot_files[name]["sha256"] for name in ("modeling_ouro.py", "configuration_ouro.py")}
    if (any(not isinstance(row, dict) or set(row) != {"class", "path", "sha256"}
            or not isinstance(row["class"], str) or not isinstance(row["path"], str) for row in remote)
            or {row["sha256"] for row in remote} != expected_remote):
        raise ValueError("fitting remote implementation hashes differ from the snapshot")
    return fit


def _read_generation(torch, fit_dir, owner):
    """Use the producer's validators, including FP16 == sealed FP32 sum/100."""
    identity = owner["identity"]
    checkpoint = runner._read_checkpoint(torch, fit_dir, identity)
    final = runner._read_final(torch, fit_dir, identity, checkpoint)
    if final is None:
        raise ValueError(f"no committed COMPLETE generation: {fit_dir}")
    complete = runner._json(fit_dir / "COMPLETE.json")
    seal = runner._json(final.parent / "SEAL.json")
    record = {
        "fit_id": identity["fit_id"], "fit_identity_sha256": owner["fit_identity_sha256"],
        "identity": identity, "fit_dir_at_evaluation": str(fit_dir),
        "owner": runner._record(fit_dir / "OWNER.json"),
        "latest": runner._record(fit_dir / "LATEST.json"),
        "complete": runner._record(fit_dir / "COMPLETE.json"),
        "checkpoint_pointer": checkpoint["pointer"], "complete_pointer": complete,
        "lens": seal["files"]["lens.pt"], "n_prompts": 100,
        "validation": "producer checkpoint/final checks, literal FP16 bits of FP32 sum/100",
    }
    del checkpoint
    return final, record


def validate_inputs(*, run_spec, ouro_src, snapshot, fit_dirs):
    """Read and verify inputs on CPU; no model, CUDA initialization, or writes."""
    if not fit_dirs or len(fit_dirs) > 5:
        raise ValueError("supply one to five completed fit directories")
    entries = [_owner(path) for path in fit_dirs]
    fit_ids = [owner["identity"].get("fit_id") for _, owner in entries]
    if any(type(value) is not int for value in fit_ids) or len(set(fit_ids)) != len(fit_ids):
        raise ValueError("fit IDs must be distinct integers")
    contract = runner._contract(run_spec, ouro_src, fit_ids)
    _evaluation_sources(contract)
    shared = None
    for _, owner in entries:
        _validate_owner_binding(owner, contract)
        runtime = {key: value for key, value in owner["identity"]["runtime"].items() if key != "calibration"}
        if shared is not None and not _same(runtime, shared):
            raise ValueError("the requested fits do not share one frozen fitting runtime")
        shared = runtime
    snapshot_files = runner._snapshot(snapshot, contract["manifest"])
    import torch
    validated = []
    for fit_dir, owner in sorted(entries, key=lambda row: row[1]["identity"]["fit_id"]):
        final, record = _read_generation(torch, fit_dir, owner)
        validated.append({"fit_dir": fit_dir, "owner": owner, "lens_path": final, "record": record})
        gc.collect()
    return contract, snapshot_files, validated


def _imports(contract, ouro_src):
    sys.path[:0] = [str(REPO), str(Path(ouro_src).absolute())]
    names = (
        "ouro_jlens", "ouro_jlens.evaluate", "ouro_jlens.evaldata", "ouro_jlens.fit_lens",
        "ouro_jlens.evidence", "ouro_jlens.recurrent", "jlens", "jlens._logging",
        "jlens.fitting", "jlens.hooks", "jlens.hf", "jlens.lens", "jlens.protocol", "jlens.vis",
    )
    expected = {str(path.absolute()): key for key, path in contract["source_paths"].items()}
    hashes = {(row["root"], row["path"]): row["sha256"] for row in contract["sources"]}
    modules, records = {}, []
    for name in names:
        module = importlib.import_module(name)
        path = runner._no_links(module.__file__)
        key = expected.get(str(path))
        if key is None or runner._file_hash(path) != hashes[key]:
            raise ValueError(f"evaluation imported an unmanifested module: {name}: {path}")
        modules[name] = module
        records.append({"module": name, "root": key[0], "path": key[1], "sha256": hashes[key]})
    return modules["ouro_jlens.evaluate"], modules["ouro_jlens.recurrent"], records


@contextmanager
def task_names(evaluator, items, tasks=TASKS):
    """Restore the legacy module-global registry, including on an exception."""
    previous = dict(evaluator.TASK_NAMES)
    try:
        evaluator.TASK_NAMES.clear()
        evaluator.TASK_NAMES.update({task: evaluator.TaskNames(items, task) for task in tasks})
        yield evaluator.TASK_NAMES
    finally:
        evaluator.TASK_NAMES.clear()
        evaluator.TASK_NAMES.update(previous)


@dataclass
class PreparedReadout:
    jacobians: Any
    target_layer: int
    virtual_indices: list[int]
    learned_sources: list[int]
    identity_indices: list[int]

    def metadata(self):
        return {name: getattr(self, name) for name in (
            "target_layer", "virtual_indices", "learned_sources", "identity_indices",
        )}


def prepare_readout(model, lens, *, target_layer, source_layers, evaluator):
    """Geometry only: retain final self-identity, omit every other unfitted row.

    This helper does not validate any artifact schema. The main CLI separately
    requires the existing main N100 owner and rejects control-profile owners.
    """
    if (type(target_layer) is not int or not 0 < target_layer < model.n_layers
            or any(type(layer) is not int for layer in source_layers)
            or list(source_layers) != list(range(target_layer))
            or list(lens.source_layers) != list(source_layers)):
        raise ValueError("readout requires exact contiguous learned sources strictly before its target")
    full = evaluator.stacked_jacobians(model, lens, target_layer)
    final = target_layer == model.n_layers - 1
    indices = list(range(model.n_layers)) if final else list(source_layers)
    selected = full if final else full[:target_layer]
    return PreparedReadout(selected, target_layer, indices, list(source_layers), [target_layer] if final else [])


class _ReadoutView:
    def __init__(self, model, n_layers):
        self.model, self.n_layers = model, n_layers

    def __getattr__(self, name):
        return getattr(self.model, name)


def readout_for_plan(evaluator, model, items, states, plan, exit_logits):
    """Use the original rank math with only the declared virtual columns.

    Legacy *_local diagnostics continue to reference native final exit UT3;
    explicit target-state logits are stored separately. They are not relabeled
    as penultimate-target diagnostics by this geometry helper.
    """
    if states.shape[1] != model.n_layers:
        raise ValueError("state cache must retain the complete original virtual axis")
    selected = states if len(plan.virtual_indices) == model.n_layers else states[:, :len(plan.virtual_indices)]
    view = model if len(plan.virtual_indices) == model.n_layers else _ReadoutView(model, len(plan.virtual_indices))
    result, kept = evaluator.readout_arrays(
        view, items, selected, plan.jacobians, exit_logits, model.n_ut - 1,
    )
    del kept
    return result


def slice_readout_arrays(arrays, *, virtual_indices, keep):
    """Select real columns explicitly; never create -1 placeholders for layers."""
    import numpy as np
    if (not virtual_indices or len(set(virtual_indices)) != len(virtual_indices)
            or not keep or len(set(keep)) != len(keep)
            or any(type(v) is not int for v in [*virtual_indices, *keep])
            or not set(keep) <= set(virtual_indices)):
        raise ValueError("requested virtual support must be a unique subset of existing columns")
    columns = [virtual_indices.index(value) for value in keep]
    result = {}
    for name, value in arrays.items():
        if not isinstance(value, np.ndarray) or value.ndim < 2 or value.shape[-1] != len(virtual_indices):
            raise ValueError(f"{name} does not have the declared readout-column axis")
        result[name] = value[..., columns].copy()
    return result


def _validate_arrays(arrays, items, names, virtual_indices, *, prefix=""):
    import numpy as np
    n, columns = len(items), len(virtual_indices)
    ranks = arrays[prefix + "allrank"]
    if ranks.shape != (n, 128, columns) or ranks.dtype != np.int32:
        raise ValueError("allrank has the wrong item/name/virtual shape or dtype")
    own = arrays[prefix + "rank"]
    if own.shape != (n, 3, columns) or own.dtype != np.int32:
        raise ValueError("own ranks have the wrong item/slot/virtual shape or dtype")
    for value in arrays.values():
        if value.shape[0] != n or value.shape[-1] != columns:
            raise ValueError("readout arrays disagree on item or virtual support")
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError("readout arrays contain nonfinite values")
    for index, item in enumerate(items):
        count = len(names[item["task"]])
        if np.any(ranks[index, :count] < 0) or np.any(ranks[index, count:] != -1):
            raise ValueError("a real name has unsupported columns or name padding changed")
        for slot, name_index in enumerate(item["own_index"]):
            expected = ranks[index, name_index] if name_index >= 0 else np.full(columns, -1, np.int32)
            if not np.array_equal(own[index, slot], expected):
                raise ValueError("own-slot ranks disagree with the task-name rank bank")


def _item_rows(model, items, registry, evaluator):
    operations = set(importlib.import_module("ouro_jlens.evaldata").OPERATIONS)
    rows = []
    for item in items:
        task = registry[item.task]
        own_index = task.own_index(item)
        own = {index for index in own_index if index >= 0}
        intermediates = item.intermediates[:evaluator.MAX_INTER]
        eligible = [bool(item.intermediate_tokens[name]) and not item.leaked[name]
                    and (item.task == "multihop" or name not in operations) for name in intermediates]
        controls = [
            [index for index, other in enumerate(task.names) if index not in own
             and ((other in operations) == (name in operations))]
            if item.intermediate_tokens[name] else []
            for name in intermediates
        ]
        if any(yes and not control for yes, control in zip(eligible, controls)):
            raise ValueError("an eligible slot has no historical matched controls")
        rows.append({
            "name": item.name, "task": item.task, "prompt": item.prompt, "target": item.target,
            "intermediates": intermediates, "own_index": own_index,
            "scorable": [bool(item.intermediate_tokens[name]) for name in intermediates],
            "leaked": [bool(item.leaked[name]) for name in intermediates],
            "n_tokens": len(item.token_ids), "readout_token": model.tokenizer.decode([item.token_ids[-1]]),
            "token_ids": list(item.token_ids), "readout_position": -1,
            "intermediate_tokens": {name: item.intermediate_tokens[name] for name in intermediates},
            "eligible": eligible, "control_indices": controls,
        })
    return rows


def _population(rows, registry, *, historical=True):
    names = {task: list(value.names) for task, value in registry.items()}
    forms = {task: {name: sorted(ids) for name, ids in value.forms.items()} for task, value in registry.items()}
    identity = [{key: row[key] for key in POPULATION_FIELDS} for row in rows]
    observed = {
        "items": len(rows), "tasks": {task: sum(row["task"] == task for row in rows) for task in names},
        "population_sha256": runner._digest(identity), "task_names_sha256": runner._digest(names),
        "token_forms_sha256": runner._digest(forms),
        "eligible_items": {task: sum(row["task"] == task and any(row["eligible"]) for row in rows) for task in names},
        "eligible_slots": {task: sum(sum(row["eligible"]) for row in rows if row["task"] == task) for task in names},
        "control_policy": "other task names of matching operation/non-operation kind; exclude all own names; retain historical aliases",
        "score_policy": "zero-based full-vocabulary rank<10; min over token forms; slots averaged within item; any-layer per control before averaging controls",
    }
    if historical and (observed["items"] != 148 or observed["tasks"] != TASK_COUNTS
                       or observed["population_sha256"] != POPULATION_SHA256
                       or observed["task_names_sha256"] != TASK_NAMES_SHA256
                       or observed["token_forms_sha256"] != TOKEN_FORMS_SHA256
                       or observed["eligible_items"] != {"multihop": 90, "order-ops": 51}):
        raise ValueError("tokenization, names, leakage, or item eligibility changed from the frozen population")
    return names, forms, observed


def _new_output(path, identity, *, schema=SCHEMA):
    """Own a new directory exclusively; never resume or replace an evaluation."""
    _requested_sections(identity)
    path = runner._no_links(path)
    runner._mkdir(path.parent)
    path.mkdir(exist_ok=False)
    runner._fsync_dir(path.parent)
    runner._new_json(path / "OWNER.json", {
        "schema": schema, "identity_sha256": runner._digest(identity), "identity": identity,
    })
    return path


def _requested_sections(identity):
    sections = identity.get("sections") if isinstance(identity, dict) else None
    if (not isinstance(sections, list) or not sections
            or any(not isinstance(name, str) for name in sections)
            or len(set(sections)) != len(sections)):
        raise ValueError("evaluation identity must declare all ordered, unique sections")
    for name in sections:
        runner._relative(name)
        if name in {"OWNER.json", "COMPLETE.json"}:
            raise ValueError("a section cannot use a reserved output filename")
    if any(left != right and right.startswith(left + "/") for left in sections for right in sections):
        raise ValueError("evaluation sections cannot contain one another")
    return sections


def _output_owner(root, *, schema=None):
    root = runner._no_links(root)
    owner = runner._json(runner._no_links(root / "OWNER.json"))
    if (not isinstance(owner, dict) or set(owner) != {"schema", "identity_sha256", "identity"}
            or not isinstance(owner["schema"], str) or not isinstance(owner["identity"], dict)
            or owner["identity_sha256"] != runner._digest(owner["identity"])
            or (schema is not None and owner["schema"] != schema)):
        raise ValueError("invalid evaluation owner")
    _requested_sections(owner["identity"])
    return owner


def _write_npz(path, arrays):
    import numpy as np
    if (not arrays or any(not isinstance(name, str) or not isinstance(value, np.ndarray)
                          or value.dtype.hasobject for name, value in arrays.items())):
        raise ValueError("NPZ outputs must be nonempty named arrays without object data")
    path = runner._no_links(path)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    runner._fsync_dir(path.parent)


def _write_torch(path, value):
    import torch
    path = runner._no_links(path)
    with path.open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    runner._fsync_dir(path.parent)


def _seal_section(root, section, metadata):
    """Seal an existing flat section after exclusively writing its metadata."""
    root = runner._no_links(root)
    owner = _output_owner(root)
    section = runner._relative(section).as_posix()
    path = runner._no_links(root / section)
    if not path.is_dir():
        raise ValueError("a section must exist before it is sealed")
    runner._new_json(path / "metadata.json", metadata)
    files = {}
    for entry in sorted(path.iterdir()):
        runner._no_links(entry)
        if not entry.is_file() or entry.name == "SEAL.json":
            raise ValueError("sections must contain new regular payload files only")
        files[entry.name] = runner._record(entry)
    runner._new_json(path / "SEAL.json", {
        "schema": owner["schema"], "kind": "section", "section": section,
        "identity_sha256": owner["identity_sha256"], "files": files,
    })
    return {"section": section, "seal": runner._record(path / "SEAL.json")}


def _verify_section(root, record, owner):
    if not isinstance(record, dict) or set(record) != {"section", "seal"}:
        raise ValueError("invalid evaluation section reference")
    section = runner._relative(record["section"]).as_posix()
    path = runner._no_links(root / section)
    runner._verify_record(path / "SEAL.json", record["seal"])
    seal = runner._json(path / "SEAL.json")
    expected = {"schema": owner["schema"], "kind": "section", "section": section,
                "identity_sha256": owner["identity_sha256"]}
    if (not isinstance(seal, dict) or set(seal) != {*expected, "files"}
            or any(seal[key] != value for key, value in expected.items())
            or not isinstance(seal["files"], dict) or "metadata.json" not in seal["files"]):
        raise ValueError("invalid evaluation section seal")
    files = {f"{section}/SEAL.json"}
    for name, file_record in seal["files"].items():
        relative = runner._relative(name)
        if len(relative.parts) != 1 or name == "SEAL.json":
            raise ValueError("a section file must be a direct, non-seal member")
        runner._verify_record(path / name, file_record)
        files.add(f"{section}/{name}")
    return files


def _complete_output(root, sections, *, fault_hook=None):
    """Publish COMPLETE only after every requested section is durable and valid."""
    root = runner._no_links(root)
    owner = _output_owner(root)
    if (not isinstance(sections, list) or any(not isinstance(row, dict) for row in sections)
            or [row.get("section") for row in sections] != _requested_sections(owner["identity"])):
        raise ValueError("completion must include every requested section in the declared order")
    allowed = {"OWNER.json"}
    for record in sections:
        allowed.update(_verify_section(root, record, owner))
    _check_members(root, allowed)
    complete = {"schema": owner["schema"], "kind": "complete",
                "identity_sha256": owner["identity_sha256"], "sections": sections}
    if fault_hook is not None:
        fault_hook("before_complete", root)
    runner._new_json(root / "COMPLETE.json", complete)
    return complete


def _check_members(root, allowed):
    actual = set()
    for entry in root.rglob("*"):
        runner._no_links(entry)
        if entry.is_file():
            actual.add(entry.relative_to(root).as_posix())
        elif not entry.is_dir():
            raise ValueError("evaluation contains a nonregular entry")
    if actual != allowed:
        raise ValueError("evaluation contains unsealed or missing files")


def validate_output(root, *, schema=SCHEMA):
    """Verify a portable completed evaluation without opening producer paths."""
    root = runner._no_links(root)
    owner = _output_owner(root, schema=schema)
    complete = runner._json(runner._no_links(root / "COMPLETE.json"))
    expected = {"schema": owner["schema"], "kind": "complete",
                "identity_sha256": owner["identity_sha256"]}
    if (not isinstance(complete, dict) or set(complete) != {*expected, "sections"}
            or any(complete[key] != value for key, value in expected.items())
            or not isinstance(complete["sections"], list) or not complete["sections"]):
        raise ValueError("invalid completed evaluation manifest")
    if (any(not isinstance(row, dict) for row in complete["sections"])
            or [row.get("section") for row in complete["sections"]] != _requested_sections(owner["identity"])):
        raise ValueError("completed evaluation omits or reorders requested sections")
    allowed = {"OWNER.json", "COMPLETE.json"}
    sections = set()
    for record in complete["sections"]:
        if not isinstance(record, dict) or record.get("section") in sections:
            raise ValueError("duplicate or invalid evaluation section")
        sections.add(record.get("section"))
        allowed.update(_verify_section(root, record, owner))
    _check_members(root, allowed)
    return {"owner": owner, "complete": complete, "files": len(allowed)}


@contextmanager
def _runtime_scope(parent):
    """Keep generated import caches outside the immutable evaluation bundle."""
    previous_environment = dict(os.environ)
    previous_tempdir, previous_bytecode = tempfile.tempdir, sys.dont_write_bytecode
    try:
        with tempfile.TemporaryDirectory(prefix=".ouro-evaluation-runtime-", dir=parent) as temporary:
            runner._runtime_cache(temporary)
            yield
    finally:
        os.environ.clear()
        os.environ.update(previous_environment)
        tempfile.tempdir, sys.dont_write_bytecode = previous_tempdir, previous_bytecode


def _evaluation_runtime(torch, contract, snapshot_files, snapshot, recurrent, imported):
    environment = runner._environment(torch, contract["environment"])
    if not torch.cuda.is_available():
        raise RuntimeError("Ouro evaluation requires an available CUDA GPU")
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    gpu = {"index": index, "name": properties.name, "total_memory": properties.total_memory,
           "capability": [properties.major, properties.minor],
           "uuid": str(getattr(properties, "uuid", "unavailable")),
           "multiprocessor_count": properties.multi_processor_count}
    model = recurrent.load_ouro(path=snapshot, device=f"cuda:{index}", dtype=torch.bfloat16)
    if (model.n_ut, model.n_physical, model.n_layers, model.d_model) != (4, 48, 192, 2048):
        raise ValueError("evaluation model geometry differs from the fitted model")
    if (any(module.training for module in model.hf_model.modules())
            or any(parameter.requires_grad or parameter.dtype != torch.bfloat16
                   or parameter.device != torch.device("cuda", index)
                   for parameter in model.hf_model.parameters())):
        raise ValueError("evaluation requires frozen BF16 parameters in evaluation mode on the chosen GPU")
    remote = []
    for cls in (type(model.hf_model), type(model.hf_model.config)):
        path = Path(inspect.getfile(cls))
        digest = runner._file_hash(path)
        if path.name not in snapshot_files or digest != snapshot_files[path.name]["sha256"]:
            raise ValueError(f"loaded evaluation remote implementation changed: {path}")
        remote.append({"class": f"{cls.__module__}.{cls.__name__}", "path": str(path), "sha256": digest})
    attention = str(getattr(model.hf_model.config, "_attn_implementation", None))
    if attention != "sdpa":
        raise ValueError("evaluation must use the pinned SDPA implementation setting")
    return model, {
        "environment": environment, "gpu": gpu, "precision": runner._precision(torch),
        "imported_sources": imported, "remote_implementations": remote,
        "attention_implementation": attention,
        "model": contract["spec"]["model"], "snapshot_files": snapshot_files,
        "snapshot_path_at_evaluation": str(Path(snapshot).absolute()),
        "fit_gpu_matching": "fitting GPU identity is preserved; evaluation records its own GPU",
    }


def _evaluate(args, contract, snapshot_files, fits):
    import torch
    identity = {
        "kind": "ouro_main_n100_readout", "run_spec_sha256": contract["spec_sha256"],
        "source_files": contract["sources"], "model": contract["spec"]["model"],
        "fit_ids": [entry["record"]["fit_id"] for entry in fits],
        "fits": [entry["record"] for entry in fits], "tasks": list(TASKS),
        "sections": ["common", *(f"fits/fit_{entry['record']['fit_id']:02d}" for entry in fits)],
        "position": -1, "greedy_steps": 4, "target_layer": 191,
        "learned_sources": list(range(191)), "virtual_indices": list(range(192)),
        "known_identity_indices": [191],
        "matrix_policy": "read out each independent fit; never average matrices",
        "created_utc": runner._utc_now().isoformat(),
    }
    out = _new_output(args.out, identity)
    started = time.monotonic()
    with _runtime_scope(out.parent):
        evaluator, recurrent, imported = _imports(contract, args.ouro_src)
        model, runtime = _evaluation_runtime(torch, contract, snapshot_files, args.snapshot, recurrent, imported)
        items = evaluator.load_items(model.tokenizer, TASKS, encode=lambda text: model.encode(text)[0].tolist())
        with task_names(evaluator, items) as registry, torch.no_grad():
            if any(len(value.names) > evaluator.MAX_NAMES for value in registry.values()):
                raise ValueError("task-name population exceeds the retained 128-name storage axis")
            rows = _item_rows(model, items, registry, evaluator)
            names, forms, population = _population(rows, registry)
            print(json.dumps({"stage": "cache_states", "items": len(items)}), flush=True)
            states, continuations = evaluator.cache_states(model, items, position=-1)
            if (tuple(states.shape) != (148, 192, 2048) or states.dtype != torch.float32
                    or not torch.isfinite(states).all()):
                raise ValueError("common activation cache has invalid geometry, dtype, or values")
            # Preserve the original exit-state unembedding, including Ouro's
            # native loop boundaries, and keep the unnormalized 190 target too.
            exit_logits = torch.stack([
                model.unembed(states[:, model.exit_index(ut)].to(model.input_device)).float()
                for ut in range(model.n_ut)
            ], dim=1)
            target_logits = {
                190: model.unembed(states[:, 190].to(model.input_device)).float().cpu(),
                191: exit_logits[:, 3].cpu(),
            }
            if (not torch.isfinite(exit_logits).all()
                    or any(not torch.isfinite(value).all() for value in target_logits.values())):
                raise ValueError("native/target-state logits contain nonfinite values")
            exit_top1 = exit_logits.argmax(-1).cpu().numpy()
            for index, (row, continuation) in enumerate(zip(rows, continuations)):
                row.update(continuation=continuation, correct=evaluator.is_correct(continuation, row["target"]),
                           exit_top1=[model.tokenizer.decode([token]) for token in exit_top1[index]])
            print(json.dumps({"stage": "raw_logit_lens", "virtual_columns": 192}), flush=True)
            raw, kept = evaluator.readout_arrays(model, items, states, None, exit_logits, model.n_ut - 1)
            del kept
            _validate_arrays(raw, rows, names, list(range(192)))
            common_dir = runner._mkdir(out / "common")
            _write_npz(common_dir / "arrays.npz", {
                "exit_top1": exit_top1, **{f"logitlens_{key}": value for key, value in raw.items()},
            })
            runner._new_json(common_dir / "items.json", rows)
            runner._new_json(common_dir / "task_names.json", names)
            runner._new_json(common_dir / "token_forms.json", forms)
            _write_torch(common_dir / "cache.pt", {
                "schema": SCHEMA, "H": states, "exit_logits": exit_logits.cpu(),
                "target_state_logits": target_logits, "virtual_indices": list(range(192)),
                "native_exit_indices": [model.exit_index(ut) for ut in range(model.n_ut)],
                "position": -1, "population_sha256": population["population_sha256"],
            })
            common = _seal_section(out, "common", {
                "kind": "common_readout", "runtime": runtime, "population": population,
                "virtual_indices": list(range(192)), "position": -1,
                "native_exit_indices": [47, 95, 143, 191], "target_state_indices": [190, 191],
                "raw_logit_lens": "original readout_arrays with J=None; reused by every fit",
                "local_diagnostics": "legacy local and final both reference native final exit UT3",
                "kept_logits": "full per-location vocabulary logits are discarded after original rank computation",
                "correct_items": {task: sum(row["task"] == task and row["correct"] for row in rows) for task in TASKS},
            })
            del raw, target_logits
            sections = [common]
            for entry in fits:
                fit_id = entry["record"]["fit_id"]
                print(json.dumps({"stage": "fit_readout", "fit_id": fit_id}), flush=True)
                # Recheck the selected immutable payload immediately before
                # loading. The committed generation was numerically validated
                # against its FP32 checkpoint during preflight.
                runner._verify_record(entry["lens_path"], entry["record"]["lens"])
                lens = importlib.import_module("jlens").JacobianLens.load(str(entry["lens_path"]))
                plan = prepare_readout(model, lens, target_layer=191, source_layers=list(range(191)), evaluator=evaluator)
                del lens
                gc.collect()
                arrays = readout_for_plan(evaluator, model, items, states, plan, exit_logits)
                _validate_arrays(arrays, rows, names, plan.virtual_indices)
                section = f"fits/fit_{fit_id:02d}"
                fit_out = runner._mkdir(out / section)
                _write_npz(fit_out / "arrays.npz", {f"jlens_exit3_{key}": value for key, value in arrays.items()})
                sections.append(_seal_section(out, section, {
                    "kind": "independent_fit_readout", "fit": entry["record"],
                    "common": common, "readout": plan.metadata(), "position": -1,
                    "array_prefix": "jlens_exit3_", "n_prompts": 100,
                    "local_diagnostics": "legacy local and final both reference native final exit UT3",
                }))
                del arrays, plan
                gc.collect()
                torch.cuda.empty_cache()
            _complete_output(out, sections)
    checked = validate_output(out)
    return {"status": "complete", "schema": SCHEMA, "out": str(out),
            "fit_ids": identity["fit_ids"], "items": 148,
            "files": checked["files"], "elapsed_seconds": time.monotonic() - started,
            "complete": runner._record(out / "COMPLETE.json")}


def _self_test(ouro_src, *, snapshot=None):
    """CPU fixtures exercise original readout math and real producer seals."""
    import numpy as np
    import torch
    from types import SimpleNamespace
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU self-test must begin without CUDA initialization")
    source_paths = {
        ("jlens", path.relative_to(REPO).as_posix()): path for path in (REPO / "jlens").rglob("*.py")
    }
    for name in ("__init__", "evaluate", "evaldata", "fit_lens", "evidence", "recurrent"):
        source_paths[("ouro_src", f"ouro_jlens/{name}.py")] = Path(ouro_src).absolute() / f"ouro_jlens/{name}.py"
    sources = [{"root": key[0], "path": key[1], "sha256": runner._file_hash(path)} for key, path in source_paths.items()]
    evaluator, recurrent, imported = _imports({"source_paths": source_paths, "sources": sources}, ouro_src)
    from tests.tiny import TinyDecoder
    from ouro_jlens.evaldata import Item
    from jlens import JacobianLens
    model = TinyDecoder(n_layers=192, d_model=4, vocab_size=32).eval().requires_grad_(False)
    model.n_ut, model.n_physical = 4, 48
    model.exit_index = lambda ut: ut * 48 + 47

    class NativeFacade:
        def __call__(self, ids, **kwargs):
            return SimpleNamespace(logits=model.unembed(model.forward(ids).last_hidden_state))

    model.hf_model = NativeFacade()
    specifications = [
        ("a", "multihop", {"alpha": [4, 5], "missing": []}, {"alpha": False, "missing": False}),
        ("b", "multihop", {"beta": [6]}, {"beta": True}),
        ("c", "order-ops", {"1": [8], "addition": [9]}, {"1": False, "addition": False}),
        ("d", "order-ops", {"2": [10], "subtraction": [11]}, {"2": False, "subtraction": False}),
    ]
    items = [Item(name, task, "prompt " + name, "answer", list(forms),
                  model.encode("prompt " + name)[0].tolist(), forms, leaked)
             for name, task, forms, leaked in specifications]
    checks = []

    def require(condition, name):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def rejects(action, name, errors=(ValueError, FileExistsError, FileNotFoundError)):
        try:
            action()
        except errors:
            checks.append(name)
        else:
            raise AssertionError(f"did not reject: {name}")

    previous = dict(evaluator.TASK_NAMES)
    with task_names(evaluator, items) as registry, torch.no_grad():
        rows = _item_rows(model, items, registry, evaluator)
        names, forms, population = _population(rows, registry, historical=False)
        require(population["eligible_items"] == {"multihop": 1, "order-ops": 2}, "fixture eligible denominators")
        require(rows[2]["control_indices"] == [[2], [3]], "operation-matched controls exclude all own names")
        states, continuations = evaluator.cache_states(model, items)
        exits = torch.stack([model.unembed(states[:, model.exit_index(ut)]).float() for ut in range(4)], dim=1)
        matrices = {layer: torch.eye(4) + (layer + 1) * 0.0001 for layer in range(191)}
        lens = JacobianLens(matrices, n_prompts=100, d_model=4)
        plan = prepare_readout(model, lens, target_layer=191, source_layers=list(range(191)), evaluator=evaluator)
        expected, kept = evaluator.readout_arrays(model, items, states, evaluator.stacked_jacobians(model, lens, 191), exits, 3)
        del kept
        observed = readout_for_plan(evaluator, model, items, states, plan, exits)
        require(expected.keys() == observed.keys() and all(np.array_equal(expected[key], observed[key]) for key in expected),
                "all seven final-target arrays bitwise equal to original readout")
        require(plan.virtual_indices == list(range(192)) and plan.identity_indices == [191], "final known self-identity retained")
        _validate_arrays(observed, rows, names, plan.virtual_indices)
        raw, kept = evaluator.readout_arrays(model, items, states, None, exits, 3)
        del kept
        penult = prepare_readout(model, JacobianLens({key: value for key, value in matrices.items() if key < 190},
                                 n_prompts=100, d_model=4), target_layer=190, source_layers=list(range(190)), evaluator=evaluator)
        penult_arrays = readout_for_plan(evaluator, model, items, states, penult, exits)
        _validate_arrays(penult_arrays, rows, names, penult.virtual_indices)
        require(penult.virtual_indices == list(range(190)) and not penult.identity_indices
                and penult_arrays["allrank"].shape == (4, 128, 190), "penultimate exports learned columns only")
        sliced = slice_readout_arrays(raw, virtual_indices=list(range(192)), keep=list(range(190)))
        require(all(np.array_equal(sliced[key], value[..., :190]) for key, value in raw.items()), "paired raw baseline uses explicit common columns")
        rejects(lambda: slice_readout_arrays(penult_arrays, virtual_indices=list(range(190)), keep=[190]), "unsupported layer selection rejected")
        broken = {key: value.copy() for key, value in penult_arrays.items()}
        broken["allrank"][0, 0, 189] = -1
        rejects(lambda: _validate_arrays(broken, rows, names, penult.virtual_indices), "negative rank for a real name rejected")
        rejects(lambda: prepare_readout(model, lens, target_layer=190, source_layers=list(range(191)), evaluator=evaluator), "unfitted target geometry rejected")
    require(evaluator.TASK_NAMES == previous, "legacy task registry restored normally")
    try:
        with task_names(evaluator, items):
            raise LookupError("fixture interruption")
    except LookupError:
        pass
    require(evaluator.TASK_NAMES == previous, "legacy task registry restored on exception")

    prompts = [f"Distinct CPU paragraph {index}." for index in range(100)]
    lengths, valid = [128] * 100, [111] * 100
    # Producer paths are deliberately nonexistent. Only relocated logical
    # sources and pinned content identities may be compared by this reader.
    fit = {"fit_id": 1, "seed": 2026090701, "n_prompts": 100, "prompts_path": "fit01.json",
           "sha256": "a" * 64, "token_lengths": lengths, "n_valid": valid,
           "prompts": prompts, "prompt_path": "/relocated/frozen/fit01.json"}
    declared = [{"root": key[0], "path": key[1], "sha256": "a" * 64} for key in sorted(set(_fit_import_keys().values()))]
    environment = {"packages": {"torch": "fixture"}, "torch_git": "fixture", "python_version": "fixture",
                   "cuda_build_version": "fixture", "cudnn_runtime_integer": 1}
    snapshot_rows = [{"path": name, "bytes": 1, "sha256": char * 64}
                     for name, char in (("modeling_ouro.py", "b"), ("configuration_ouro.py", "c"))]
    contract = {"fits": [fit], "spec_sha256": "d" * 64, "spec": {"model": {"fixture": True}},
                "sources": declared, "environment": environment, "manifest": {"files": snapshot_rows}}
    runtime = {
        "run_spec_sha256": contract["spec_sha256"], "model": contract["spec"]["model"],
        "snapshot_path": "/deleted/producer/snapshot", "snapshot_files": {row["path"]: {key: row[key] for key in ("bytes", "sha256")} for row in snapshot_rows},
        "source_files": declared, "settings": runner.SETTINGS, "environment": environment,
        "gpu": {"name": "producer GPU", "uuid": "producer UUID"}, "precision": {"fixture": True},
        "attention_implementation": "sdpa", "calibration": {key: value for key, value in fit.items() if key != "prompts"},
        "imported_sources": [{"module": name, "path": f"/deleted/producer/{name}.py", "sha256": "a" * 64} for name in _fit_import_keys()],
        "remote_implementations": [{"class": "fixture." + row["path"], "path": "/deleted/producer/" + row["path"], "sha256": row["sha256"]} for row in snapshot_rows],
    }
    runtime["calibration"]["prompt_path"] = "/deleted/producer/fit01.json"
    identity = runner._fit_identity(prompts, lengths, valid, 1, runtime, 2048, tuple(range(191)), False)
    owner = {"schema_version": 1, "fit_identity_sha256": runner._digest(identity), "identity": identity}
    require(_validate_owner_binding(owner, contract)["fit_id"] == 1, "relocated producer paths accepted by logical content binding")
    altered = json.loads(runner._canonical(owner))
    altered["identity"]["runtime"]["source_files"][0]["sha256"] = "f" * 64
    rejects(lambda: _validate_owner_binding(altered, contract), "changed fitting source binding rejected")

    with tempfile.TemporaryDirectory(prefix="ouro-readout-cpu-") as temporary:
        temporary = Path(temporary)
        tiny_identity = runner._fit_identity(prompts, lengths, valid, 1, {"fixture": True}, 4, tuple(range(191)), True)
        diagnostics = [{"index": index, "prompt_sha256": tiny_identity["prompt_sha256"][index],
                        "token_length": lengths[index], "n_valid": valid[index]} for index in range(100)]
        sums = {layer: value * 100 for layer, value in matrices.items()}
        with runner._owned_fit(temporary / "producer", 1, tiny_identity) as fit_dir:
            pointer, _ = runner._commit(torch, fit_dir, tiny_identity, sums, 100, diagnostics, "checkpoint")
            runner._commit(torch, fit_dir, tiny_identity, sums, 100, diagnostics, "final", checkpoint_pointer=pointer)
        fit_dir, tiny_owner = _owner(fit_dir)
        final, record = _read_generation(torch, fit_dir, tiny_owner)
        require(record["n_prompts"] == 100 and final.name == "lens.pt", "producer OWNER/LATEST/COMPLETE and FP16 mean validation")
        rejects(lambda: _validate_owner_binding(tiny_owner, contract), "CPU fit owner forbidden in production evaluation")
        with final.open("ab") as handle:
            handle.write(b"corrupt")
        rejects(lambda: _read_generation(torch, fit_dir, tiny_owner), "altered sealed lens rejected before deserialization")

        request = {"fixture": True, "sections": ["common", "fits/fit_01"]}
        output = _new_output(temporary / "evaluation", request)
        section_records = []
        for section in request["sections"]:
            directory = runner._mkdir(output / section)
            _write_npz(directory / "arrays.npz", observed)
            if section == "common":
                _write_torch(directory / "cache.pt", {"H": states, "exit_logits": exits})
                runner._new_json(directory / "items.json", rows)
            section_records.append(_seal_section(output, section, {"fixture": True}))
        rejects(lambda: _complete_output(output, section_records[:1]), "partial requested section list cannot complete")

        def interrupt(stage, path):
            raise InterruptedError("fixture termination immediately before publication")

        rejects(lambda: _complete_output(output, section_records, fault_hook=interrupt),
                "publication interruption retained", errors=(InterruptedError,))
        require(not (output / "COMPLETE.json").exists(), "interrupted evaluation has no completion claim")
        _complete_output(output, section_records)
        checked = validate_output(output)
        require(checked["owner"]["identity"] == request, "complete portable output validates")
        rejects(lambda: _new_output(output, request), "existing output cannot be replaced")
        rejects(lambda: _write_npz(output / "common/arrays.npz", observed), "existing payload cannot be replaced")
        (output / "common/extra.txt").write_text("unsealed", encoding="utf-8")
        rejects(lambda: validate_output(output), "unsealed extra payload rejected")
        (output / "common/extra.txt").unlink()
        with (output / "common/arrays.npz").open("ab") as handle:
            handle.write(b"corrupt")
        rejects(lambda: validate_output(output), "altered evaluation payload rejected")

    historical = {"status": "not_requested", "reason": "pass --snapshot for a CPU-only tokenizer/population check"}
    if snapshot is not None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
        population_model = SimpleNamespace(tokenizer=tokenizer, input_device=torch.device("cpu"))
        population_model.encode = lambda text: recurrent.OuroLensModel.encode(population_model, text)
        real_items = evaluator.load_items(tokenizer, TASKS, encode=lambda text: population_model.encode(text)[0].tolist())
        with task_names(evaluator, real_items) as registry:
            real_rows = _item_rows(population_model, real_items, registry, evaluator)
            _, _, historical = _population(real_rows, registry)
        require(historical["items"] == 148, "all frozen historical items/forms/eligibility hashes match")
    require(not torch.cuda.is_initialized(), "CUDA remained uninitialized")
    return {"status": "passed", "checks": checks, "n_checks": len(checks),
            "original_readout_arrays": list(observed), "main_columns": 192, "penultimate_columns": 190,
            "fixture_population": population, "historical_population": historical,
            "imported_sources": imported, "source_sha256": runner._file_hash(__file__),
            "cuda_initialized": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-spec", type=Path, default=HERE / "run_spec.json")
    parser.add_argument("--snapshot", type=Path, help="explicit local pinned Ouro snapshot")
    parser.add_argument("--ouro-src", type=Path, help="explicit source directory containing ouro_jlens")
    parser.add_argument("--fit-dir", type=Path, nargs="+", action="append", help="one or more main N100 fit directories; may be repeated")
    parser.add_argument("--out", type=Path, help="new production output directory, or new CPU-proof JSON file")
    parser.add_argument("--validate-only", action="store_true", help="verify complete fit inputs on CPU; no inference or output")
    parser.add_argument("--self-test", action="store_true", help="run CPU fixtures without loading model weights")
    if argv is None and len(sys.argv) == 1:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    if args.ouro_src is None:
        parser.error("--ouro-src is required")
    if args.self_test:
        if args.fit_dir or args.validate_only:
            parser.error("--self-test cannot consume production fit directories")
        result = _self_test(args.ouro_src, snapshot=args.snapshot)
        if args.out is not None:
            runner._new_json(args.out, result)
    else:
        if args.snapshot is None or not args.fit_dir or (not args.validate_only and args.out is None):
            parser.error("production requires --snapshot, --fit-dir and --out (except --validate-only)")
        import torch
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        contract, snapshot_files, fits = validate_inputs(
            run_spec=args.run_spec, ouro_src=args.ouro_src, snapshot=args.snapshot,
            fit_dirs=[path for group in args.fit_dir for path in group],
        )
        result = ({"status": "validated", "fit_ids": [entry["record"]["fit_id"] for entry in fits],
                   "cuda_initialized": torch.cuda.is_initialized(), "run_spec_sha256": contract["spec_sha256"]}
                  if args.validate_only else _evaluate(args, contract, snapshot_files, fits))
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
