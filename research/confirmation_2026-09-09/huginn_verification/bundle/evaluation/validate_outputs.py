#!/usr/bin/env python3
"""Receiver-side validation for the separately rerun Huginn chain.

The worker writes ordinary payload files and the controller invokes this module
after the payload has been copied.  This validator never edits the payload.  It
checks the retained native gate, reconstructs the producer's sealed fit in a
temporary directory next to the payload, and then invokes the pinned historical
readout validator from a private package namespace.  The private namespace is
needed because the controller itself has a module named ``run_refits``.

The public callable intentionally has the controller's generic signature::

    validate_outputs(root=Path(...), manifest=..., contract=...)

``--self-test`` exercises the byte-level helpers with tiny CPU tensors.  It
does not load model weights or initialize CUDA.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
import types
import shutil
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


SCHEMA = "confirmation_validation.v1"
CONTRACT_SCHEMA = "confirmation_output_contract.v1"
FIT_EVIDENCE_SCHEMA = "huginn_verification_fit_evidence.v1"
HUGINN_SCHEMA = "huginn_r8_evaluation.v1"
HUGINN_ELIGIBILITY_SCHEMA = "huginn_common_eligibility.v1"
HUGINN_REPO_ID = "tomg-group-umd/huginn-0125"
HUGINN_REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
HUGINN_PROMPT_SHA256 = "e3283ab440bddc5d9947d69fb0ea46f721c171e90a84c3604f25222b1b9f2a02"
HUGINN_INPUT_IDS_SHA256 = "03c3c4d0607eaad7acd4e7cfb56d990dd47c9518b6f1775ded2479c4ee1ac5e8"
HUGINN_PROMPT_STATE_SEED = 8544906854076094770
HUGINN_BENCHMARK_SOURCE_SHA256 = "08cf7986c036019382e9e26f107b51f0f53b12f4df46b4bd2188d55b2aee577d"
HISTORICAL_PROMPT_BOUND = 130.48383641405962
FIT_SAFETY_MULTIPLIER = 1.25
FIT_READOUT_RESERVE = 900.0
FIT_NATIVE_DRAW = "randn([1,S,D]); overwrite trunc_normal(std=sqrt(2/(5D)), bounds=+-3std); multiply sqrt(D) in native dtype"
SEEDS = (2026090803, 2026090804)
SOURCES = tuple(range(32))
METHODS = ("raw", "jlens", "coda")
N_PROMPTS = 100
WIDTH = 5280
N_LAYERS = 34
VOCAB = 65536
DIM_BATCH = 8
MAX_SEQ_LEN = 128
SKIP_FIRST = 16
HISTORICAL_BANK_SHA256 = "7eddc849bca857a1406a901d8e4998b6bf90580b09f6ac6b326690ef378dc596"
SEED_RULE = (
    "SHA256(canonical JSON(version, base_seed, namespace, input_ids)); "
    "first 8 bytes big-endian modulo 2^63"
)
EXPECTED_READOUT_PATHS = [
    "OWNER.json",
    "COMPLETE.json",
    "population/eligibility.json",
    "population/metadata.json",
    "population/SEAL.json",
    *[
        f"seeds/{seed}/{name}"
        for seed in SEEDS
        for name in (
            "cache.pt",
            "arrays.npz",
            "initializations.json",
            "metadata.json",
            "summaries.json",
            "SEAL.json",
        )
    ],
]
EXPECTED_BASE_PATHS = [
    "run_spec.json",
    "provenance.json",
    "logs/native_gate.log",
    "gate/primal.pt",
    "gate/native_matrices.pt",
    "gate/optimized_matrices.pt",
    "gate/START.json",
    "gate/huginn_r8_START.json",
    "gate/huginn_r8.json",
    "gate/COMPLETE.json",
]


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _safe_relative(value: str) -> str:
    require(isinstance(value, str), "relative path must be a string")
    path = PurePosixPath(value)
    require(
        bool(path.parts)
        and not path.is_absolute()
        and all(part not in ("", ".", "..") for part in path.parts)
        and path.as_posix() == value
        and "\\" not in value,
        f"invalid relative path: {value!r}",
    )
    return value


def _regular(path: Path) -> Path:
    """Return a regular, unlinked path without resolving it."""
    path = Path(path)
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(mode), f"linked evidence path: {current}")
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode), f"evidence is not a regular file: {path}")
    return path


def record(path: Path) -> dict[str, Any]:
    path = _regular(path)
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": hasher.hexdigest()}


def _valid_record(value: Any, *, name: str = "file record") -> dict[str, Any]:
    require(isinstance(value, dict) and set(value) == {"bytes", "sha256"}, f"invalid {name}")
    require(type(value["bytes"]) is int and value["bytes"] > 0, f"invalid {name} byte count")
    require(
        isinstance(value["sha256"], str)
        and len(value["sha256"]) == 64
        and all(character in "0123456789abcdef" for character in value["sha256"]),
        f"invalid {name} SHA256",
    )
    return value


def _verify_record(path: Path, expected: dict[str, Any], *, name: str = "evidence") -> dict[str, Any]:
    _valid_record(expected, name=f"{name} record")
    actual = record(path)
    require(actual == expected, f"{name} bytes differ from its record")
    return actual


def read_json(path: Path) -> Any:
    return json.loads(_regular(path).read_text(encoding="utf-8"))


def load_torch(path: Path) -> Any:
    _regular(path)
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def _finite_tensor(
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    name: str,
    contiguous: bool = False,
) -> torch.Tensor:
    require(
        isinstance(value, torch.Tensor)
        and value.layout == torch.strided
        and value.device.type == "cpu"
        and value.dtype == dtype
        and tuple(value.shape) == shape
        and not value.requires_grad,
        f"{name} has invalid dtype, device or shape",
    )
    if contiguous:
        require(value.is_contiguous(), f"{name} is not contiguous")
    require(bool(torch.isfinite(value).all()), f"{name} contains nonfinite values")
    return value


def _bitwise_equal(left: torch.Tensor, right: torch.Tensor, *, name: str) -> None:
    require(left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape), f"{name} geometry differs")
    try:
        equal = torch.equal(left.view(torch.int16 if left.element_size() == 2 else torch.int32),
                           right.view(torch.int16 if right.element_size() == 2 else torch.int32))
    except RuntimeError as error:
        raise ValueError(f"{name} cannot be compared as raw bits") from error
    require(equal, f"{name} differ at the raw-bit level")


def _matrix_bit_hash(bank: dict[int, torch.Tensor], sources: tuple[int, ...]) -> str:
    hasher = hashlib.sha256()
    for layer in sources:
        hasher.update(bank[layer].numpy().tobytes())
    return hasher.hexdigest()


def _validate_maps(
    payload: Any,
    *,
    name: str,
    sources: tuple[int, ...] = SOURCES,
    width: int = WIDTH,
) -> tuple[dict[int, torch.Tensor], str]:
    require(isinstance(payload, dict) and set(payload) == {"maps", "sources", "width"}, f"{name} schema changed")
    require(payload["sources"] == list(sources) and payload["width"] == width, f"{name} geometry changed")
    maps = payload["maps"]
    require(isinstance(maps, dict) and set(maps) == {"dense"}, f"{name} estimator arm changed")
    bank = maps["dense"]
    require(isinstance(bank, dict) and set(bank) == set(sources), f"{name} source set changed")
    for layer in sources:
        _finite_tensor(bank[layer], (width, width), torch.float32, name=f"{name}[{layer}]", contiguous=True)
    return bank, _matrix_bit_hash(bank, sources)


def _validate_primal(
    payload: Any,
    *,
    width: int = WIDTH,
    vocab: int = VOCAB,
    n_layers: int = N_LAYERS,
    batch: int = DIM_BATCH,
    sequence_length: int | None = None,
) -> dict[str, Any]:
    expected = {
        "input_ids",
        "initial_states",
        "adapter_states",
        "native_states",
        "adapter_logits",
        "native_logits",
        "initialization",
    }
    require(isinstance(payload, dict) and set(payload) == expected, "native primal schema changed")
    ids = _finite_tensor(
        payload["input_ids"],
        (batch, payload["input_ids"].shape[1]) if isinstance(payload["input_ids"], torch.Tensor) and payload["input_ids"].ndim == 2 else (0, 0),
        torch.long,
        name="native primal input IDs",
        contiguous=True,
    )
    seq = int(ids.shape[1])
    require(sequence_length is None or seq == sequence_length, "native primal sequence length changed")
    require(0 < seq <= MAX_SEQ_LEN, "native primal sequence length is outside the contract")
    require(torch.equal(ids, ids[:1].expand_as(ids)), "native primal lanes use different input IDs")
    tokens = ids[0].tolist()
    require(tokens[0] == 65504 and min(tokens) >= 0 and max(tokens) < vocab, "native primal token contract changed")

    initial = _finite_tensor(payload["initial_states"], (batch, seq, width), torch.bfloat16,
                             name="native primal initial states", contiguous=True)
    _bitwise_equal(initial, initial[:1].expand_as(initial), name="coupled native initial states")

    for field in ("adapter_states", "native_states"):
        states = payload[field]
        require(isinstance(states, dict) and set(states) == set(range(n_layers)), f"{field} cell set changed")
        for layer in range(n_layers):
            _finite_tensor(states[layer], (batch, seq, width), torch.bfloat16,
                           name=f"{field}[{layer}]", contiguous=True)
    for layer in range(n_layers):
        _bitwise_equal(payload["adapter_states"][layer], payload["native_states"][layer],
                       name=f"adapter/native cell {layer}")

    adapter_logits = _finite_tensor(payload["adapter_logits"], (batch, seq, vocab), torch.float32,
                                    name="adapter logits", contiguous=True)
    native_logits = _finite_tensor(payload["native_logits"], (batch, seq, vocab), torch.float32,
                                   name="native logits", contiguous=True)
    _bitwise_equal(adapter_logits, native_logits, name="adapter/native logits")

    initialization = payload["initialization"]
    require(isinstance(initialization, dict), "native primal initialization metadata is missing")
    expected_seed_recipe = SEED_RULE
    require(initialization.get("base_seed") == 2026090899, "native primal benchmark seed changed")
    require(initialization.get("namespace") == "benchmark", "native primal benchmark namespace changed")
    recipe = {"version": 1, "base_seed": initialization["base_seed"],
              "namespace": initialization["namespace"], "input_ids": tokens}
    expected_prompt_seed = int.from_bytes(hashlib.sha256(canonical(recipe)).digest()[:8], "big") % (2 ** 63)
    require(initialization.get("prompt_state_seed") == expected_prompt_seed,
            "native primal prompt seed differs from its recipe")
    require(isinstance(initialization.get("generator_device"), str)
            and initialization["generator_device"].startswith("cuda:"),
            "native primal generator device changed")
    require(initialization.get("input_ids_sha256") == digest(tokens), "native primal token digest changed")
    require(initialization.get("seed_recipe") == expected_seed_recipe, "native primal seed recipe changed")
    require(initialization.get("one_prompt_shape") == [1, seq, width], "native primal initialization shape changed")
    require(initialization.get("dtype") == "torch.bfloat16", "native primal initialization dtype changed")
    require(initialization.get("test_time_noise") == 0, "native primal test-time noise changed")
    require(initialization.get("lane_coupling") == "contiguous copies of the same one-prompt state",
            "native primal lane coupling metadata changed")
    return {"batch": batch, "sequence_length": seq, "cells": n_layers, "vocab": vocab}


def _expected_readout_paths(spec: dict[str, Any]) -> list[str]:
    paths = spec.get("readout_paths")
    require(paths == EXPECTED_READOUT_PATHS, "Huginn readout file contract changed")
    return list(paths)


def _spec_record(spec: dict[str, Any], suffix: str) -> dict[str, Any]:
    records = spec.get("source_records")
    require(isinstance(records, dict), "Huginn run specification source records are missing")
    matches = [value for key, value in records.items() if key.endswith(suffix)]
    require(len(matches) == 1, f"missing unique frozen source record: {suffix}")
    return _valid_record(matches[0], name=f"source {suffix}")


def _validate_spec_and_manifest(root: Path, manifest: dict[str, Any], contract: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    require(isinstance(contract, dict) and contract.get("schema") == CONTRACT_SCHEMA, "Huginn output contract schema changed")
    require(contract.get("required_checks") == ["loadability", "numerical"], "Huginn semantic checks changed")
    require(isinstance(manifest, dict) and manifest.get("schema") == "confirmation_artifact_manifest.v1",
            "Huginn artifact manifest schema changed")
    require(manifest.get("outcome") == "complete", "incomplete Huginn output cannot be accepted")
    require(manifest.get("kind") in ("stage", "final"), "invalid Huginn artifact kind")
    binding = manifest.get("binding")
    require(isinstance(binding, dict), "Huginn manifest binding is missing")
    require(binding.get("run_spec_sha256") == contract.get("run_spec_sha256"), "run specification binding differs")
    require(binding.get("run_id") == contract.get("run_id"), "Huginn run identity differs")

    spec = read_json(root / "run_spec.json")
    require(isinstance(spec, dict) and spec.get("schema") == "huginn_verification_run.v1",
            "Huginn run specification schema changed")
    require(record(root / "run_spec.json")["sha256"] == contract["run_spec_sha256"],
            "Huginn run specification bytes differ from the contract")
    require(spec.get("run_id") == contract.get("run_id") == binding.get("run_id"), "Huginn run ID differs")
    require(spec.get("category") == "newly_rerun_estimator", "Huginn output category changed")
    require(spec.get("historical_bank_sha256") == HISTORICAL_BANK_SHA256, "historical bank comparison hash changed")
    require(spec.get("n_prompts") == N_PROMPTS and spec.get("source_layers") == list(SOURCES)
            and spec.get("target_layer") == 33 and spec.get("d_model") == WIDTH,
            "Huginn fit geometry changed")
    require(spec.get("evaluation_seeds") == list(SEEDS) and spec.get("readout_items") == 148,
            "Huginn readout seed or population contract changed")
    paths = sorted(manifest.get("files", {}))
    require(isinstance(manifest.get("files"), dict), "Huginn manifest files are missing")
    expected = set(contract.get("files", {}))
    if manifest["kind"] == "stage":
        stage = manifest.get("stage_id")
        require(stage in contract.get("stages", {}), "unknown Huginn artifact stage")
        expected = set(contract["stages"][stage])
    else:
        require(manifest.get("stage_id") is None, "final Huginn artifact has a stage ID")
    require(set(paths) == expected, "Huginn artifact membership differs from its contract")
    for path in paths:
        _safe_relative(path)
        _valid_record(manifest["files"][path], name=f"manifest {path}")
    return spec, paths


def _validate_provenance(root: Path, spec: dict[str, Any], manifest: dict[str, Any]) -> None:
    provenance = read_json(root / "provenance.json")
    require(isinstance(provenance, dict), "Huginn provenance is not an object")
    require(provenance.get("binding") == manifest.get("binding"), "Huginn provenance binding differs")
    require(provenance.get("category") == "newly_rerun_estimator" and provenance.get("n_prompts") == N_PROMPTS,
            "Huginn provenance category or prompt count changed")
    require(provenance.get("source_records") == spec.get("source_records"), "Huginn executed source pins differ")
    worker_record = spec.get("source_records", {}).get("evaluation/worker.py")
    if worker_record is not None:
        require(provenance.get("actual_worker_source") == worker_record, "Huginn worker source pin differs")


def _runtime_source_checks(runtime: dict[str, Any], spec: dict[str, Any]) -> None:
    """Bind the executable source identities recorded by the GPU gate.

    The gate is produced in a separate environment, so its absolute paths are
    intentionally ignored.  The byte digests are compared with the frozen
    source records instead; this keeps the check portable while retaining the
    source binding.
    """
    imported = runtime.get("imported_sources")
    require(isinstance(imported, list) and len(imported) == 13,
            "native gate imported source inventory changed")
    expected_suffixes = {
        "fit_estimators": "fit_estimators.py",
        "optimization.optimized_fitting": "optimization/optimized_fitting.py",
        "optimization.cuda_graph_candidate": "optimization/cuda_graph_candidate.py",
        "optimization.saved_tensor_candidate": "optimization/saved_tensor_candidate.py",
        "ouro_jlens.recurrent": "ouro_project/src/ouro_jlens/recurrent.py",
        "ouro_jlens.evidence": "ouro_project/src/ouro_jlens/evidence.py",
        "jlens": "jlens/__init__.py",
        "jlens.fitting": "jlens/fitting.py",
        "jlens.hooks": "jlens/hooks.py",
        "jlens.hf": "jlens/hf.py",
        "jlens.lens": "jlens/lens.py",
        "jlens.protocol": "jlens/protocol.py",
        "huginn_adapter": "deployment/huginn_adapter.py",
    }
    observed_modules: set[str] = set()
    for entry in imported:
        require(isinstance(entry, dict)
                and set(entry) == {"module", "path", "sha256"}
                and isinstance(entry.get("module"), str)
                and isinstance(entry.get("path"), str),
                "native gate imported source record changed")
        module = entry["module"]
        require(module in expected_suffixes and module not in observed_modules,
                "native gate imported source set changed")
        observed_modules.add(module)
        expected = _spec_record(spec, expected_suffixes[module])
        require(entry["sha256"] == expected["sha256"],
                f"native gate source digest changed: {module}")
    require(observed_modules == set(expected_suffixes),
            "native gate imported source set is incomplete")


def _validate_gate_runtime(start: dict[str, Any], complete: dict[str, Any],
                           spec: dict[str, Any]) -> None:
    """Validate model, runtime, source, and benchmark provenance receipts."""
    require(start.get("prompt_sha256") == HUGINN_PROMPT_SHA256
            and complete.get("prompt_sha256") == HUGINN_PROMPT_SHA256,
            "native gate engineering prompt changed")
    source = start.get("benchmark_source")
    complete_source = complete.get("benchmark_source")
    require(isinstance(source, dict) and set(source) == {"module", "path", "sha256"}
            and source.get("module") == "ouro_jlens.bench"
            and isinstance(source.get("path"), str)
            and source.get("sha256") == HUGINN_BENCHMARK_SOURCE_SHA256,
            "native gate benchmark source changed")
    require(complete_source == source, "native gate benchmark source changed between phases")
    require(start.get("native_primal") is None
            and start.get("benchmark_initialization") is None,
            "native gate start receipt claims completed benchmark work")

    runtime = start.get("runtime")
    require(isinstance(runtime, dict), "native gate runtime receipt is missing")
    require(complete.get("runtime") == runtime,
            "native gate runtime changed between start and completion")
    model = runtime.get("model")
    require(isinstance(model, dict)
            and model.get("repo_id") == HUGINN_REPO_ID
            and model.get("revision") == HUGINN_REVISION,
            "native gate model identity changed")
    model_manifest = _spec_record(spec, "/deployment/huginn_model_manifest.json")
    require(model.get("manifest_sha256") == model_manifest["sha256"],
            "native gate Huginn manifest binding changed")
    require(runtime.get("run_spec_sha256") == _spec_record(spec, "/deployment/run_spec.json")["sha256"]
            and runtime.get("combined_contract_sha256") == _spec_record(spec, "/deployment/combined_contract.json")["sha256"]
            and runtime.get("huginn_calibration_sha256") == _spec_record(spec, "/deployment/huginn_calibration.json")["sha256"],
            "native gate deployment input binding changed")
    require(runtime.get("attention_implementation") == "sdpa",
            "native gate attention implementation changed")
    require(runtime.get("initialization") == {
        "base_seed": 2026090899,
        "base_seed_rule": "fixed for the engineering paragraph; no calibration paragraph index",
        "prompt_seed_rule": SEED_RULE,
        "seed_namespace": "benchmark",
    }, "native gate initialization recipe changed")
    require(runtime.get("settings") == {
        "accumulation_dtype": "float32",
        "checkpoint_every": 1,
        "compress_saved_tensors": True,
        "d_model": WIDTH,
        "dense_engine": "cuda_graph",
        "dim_batch": DIM_BATCH,
        "max_seq_len": MAX_SEQ_LEN,
        "mode": "dense",
        "n_coda": 2,
        "n_physical": 4,
        "n_prelude": 2,
        "n_ut": 8,
        "saved_dtype": "float16",
        "skip_first": SKIP_FIRST,
        "source_layers": list(SOURCES),
        "target_layer": 33,
    }, "native gate runtime settings changed")
    _runtime_source_checks(runtime, spec)


def _validate_benchmark_initialization(initialization: Any, sequence_length: int) -> None:
    require(isinstance(initialization, dict), "native gate benchmark initialization is missing")
    require(initialization == {
        "base_seed": 2026090899,
        "dtype": "torch.bfloat16",
        "generator_device": "cuda:0",
        "input_ids_sha256": HUGINN_INPUT_IDS_SHA256,
        "lane_coupling": "contiguous copies of the same one-prompt state",
        "namespace": "benchmark",
        "native_draw": FIT_NATIVE_DRAW,
        "one_prompt_shape": [1, sequence_length, WIDTH],
        "prompt_state_seed": HUGINN_PROMPT_STATE_SEED,
        "seed_recipe": SEED_RULE,
        "test_time_noise": 0,
    }, "native gate benchmark initialization changed")


def _validate_gate(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    start = read_json(root / "gate/START.json")
    start_h = read_json(root / "gate/huginn_r8_START.json")
    profile = read_json(root / "gate/huginn_r8.json")
    complete = read_json(root / "gate/COMPLETE.json")
    _validate_gate_runtime(start, complete, spec)
    require(start.get("kind") == "huginn" and start.get("status") == "running", "native gate start record changed")
    require(start.get("dim_batch") == DIM_BATCH and start.get("max_seq_len") == MAX_SEQ_LEN
            and start.get("skip_first") == SKIP_FIRST, "native gate settings changed")
    require(start_h.get("name") == "huginn_r8" and start_h.get("target_layer") == 33
            and start_h.get("source_layers") == list(SOURCES), "native gate profile contract changed")
    require(complete.get("kind") == "huginn" and complete.get("status") == "passed",
            "native gate did not pass")
    native_primal = complete.get("native_primal")
    require(isinstance(native_primal, dict)
            and native_primal.get("all_34_cells_bitwise_equal") is True
            and native_primal.get("logits_bitwise_equal") is True
            and native_primal.get("coupled_initial_states") is True
            and native_primal.get("batch") == DIM_BATCH
            and native_primal.get("sequence_length") == MAX_SEQ_LEN,
            "native gate primal receipt is incomplete")
    profiles = complete.get("profiles")
    require(isinstance(profiles, list) and len(profiles) == 1, "native gate profile count changed")
    observed = profiles[0]
    require(observed.get("name") == "huginn_r8" and observed.get("target_layer") == 33
            and observed.get("source_layers") == list(SOURCES) and observed.get("sequence_length") == MAX_SEQ_LEN,
            "native gate profile geometry changed")
    require(isinstance(observed.get("parity"), list) and len(observed["parity"]) == 1,
            "native gate parity receipt is missing")
    parity = observed["parity"][0]
    require(parity.get("arm") == "dense" and parity.get("bitwise_equal") is True
            and parity.get("source_count") == len(SOURCES)
            and parity.get("output_directions") == WIDTH
            and parity.get("values_compared") == len(SOURCES) * WIDTH * WIDTH,
            "native gate parity scope changed")
    require(observed.get("n_valid") == 111, "native gate valid-token count changed")
    diagnostics = observed.get("optimized_diagnostics")
    require(isinstance(diagnostics, dict)
            and diagnostics.get("dim_batch") == DIM_BATCH
            and diagnostics.get("sequence_length") == MAX_SEQ_LEN
            and diagnostics.get("n_valid") == 111
            and diagnostics.get("source_count") == len(SOURCES)
            and diagnostics.get("target") == 33
            and diagnostics.get("source_edges") is True
            and diagnostics.get("n_passes") == 660,
            "native gate optimized benchmark recipe changed")
    _validate_benchmark_initialization(
        complete.get("benchmark_initialization"), observed["sequence_length"])
    require(profile == observed, "native gate profile was changed after completion")

    primal = load_torch(root / "gate/primal.pt")
    primal_details = _validate_primal(primal, sequence_length=observed["sequence_length"])
    # The retained primal payload carries the actual token-derived state.  It
    # must be the exact state described by the completed engineering receipt;
    # checking the two metadata objects independently would leave that link
    # forgeable.
    require(primal["initialization"] == complete["benchmark_initialization"],
            "retained primal initialization differs from the completed gate receipt")
    native_payload = load_torch(root / "gate/native_matrices.pt")
    optimized_payload = load_torch(root / "gate/optimized_matrices.pt")
    native, native_hash = _validate_maps(native_payload, name="native matrices")
    optimized, optimized_hash = _validate_maps(optimized_payload, name="optimized matrices")
    require(native_hash == optimized_hash, "native and optimized matrix bits differ")
    for layer in SOURCES:
        _bitwise_equal(native[layer], optimized[layer], name=f"native/optimized matrix {layer}")
    require(parity.get("matrix_bits_sha256") == native_hash, "native matrix bit hash differs from gate receipt")
    del native_payload, optimized_payload, native, optimized, primal
    return {"primal": primal_details, "matrix_values_compared": len(SOURCES) * WIDTH * WIDTH,
            "matrix_bits_sha256": native_hash}


def _decode_evidence_files(evidence: dict[str, Any]) -> dict[str, bytes]:
    files = evidence.get("files")
    require(isinstance(files, dict) and files, "fit evidence file records are missing")
    decoded: dict[str, bytes] = {}
    for relative, entry in files.items():
        _safe_relative(relative)
        require(isinstance(entry, dict) and set(entry) == {"record", "base64"},
                f"fit evidence entry changed: {relative}")
        expected = _valid_record(entry["record"], name=f"fit evidence {relative}")
        encoded = entry["base64"]
        require(isinstance(encoded, str) and len(encoded) <= 4 * 1024 * 1024,
                f"fit evidence text is unbounded: {relative}")
        try:
            raw = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (ValueError, UnicodeEncodeError, binascii.Error) as error:
            raise ValueError(f"invalid base64 fit evidence: {relative}") from error
        require(len(raw) == expected["bytes"] and _sha256_bytes(raw) == expected["sha256"],
                f"fit evidence bytes differ: {relative}")
        decoded[relative] = raw
    return decoded


def _write_decoded(root: Path, files: dict[str, bytes]) -> None:
    for relative, raw in files.items():
        path = root / PurePosixPath(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(raw)


def _link_fixed(source: Path, target: Path) -> None:
    _regular(source)
    require(not target.exists(), f"temporary fit path already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, target, follow_symlinks=False)
    # Opening the hard link read-only keeps the operation observational.  Do
    # not chmod it: permissions belong to the shared payload inode.
    descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    os.close(descriptor)


def _pointer_and_generation(pointer: dict[str, Any], *, kind: str, identity_sha: str) -> tuple[str, str]:
    required = {"schema_version", "kind", "generation", "seal", "fit_identity_sha256", "n_done", "next_idx",
                "observed_prompt_bound_seconds"}
    require(isinstance(pointer, dict) and set(pointer) == required, f"invalid {kind} fit pointer")
    require(pointer["schema_version"] == 1 and pointer["kind"] == kind
            and pointer["fit_identity_sha256"] == identity_sha
            and pointer["n_done"] == 100 and pointer["next_idx"] == 100,
            f"{kind} pointer is not a complete N100 generation")
    relative = _safe_relative(pointer["generation"])
    parent = "checkpoints" if kind == "checkpoint" else "final"
    pieces = PurePosixPath(relative).parts
    require(len(pieces) == 2 and pieces[0] == parent and pieces[1].startswith("cursor_000100_"),
            f"{kind} generation path changed")
    return relative, pointer["generation"]


def _fit_identity_checks(identity: Any, spec: dict[str, Any]) -> None:
    require(isinstance(identity, dict), "fit identity is missing")
    require(identity.get("kind") == "huginn_r8_n100_fit" and identity.get("production_profile") == "huginn_r8",
            "fit profile changed")
    require(identity.get("fit_id") == 1 and identity.get("cpu_test") is False
            and identity.get("n_prompts") == N_PROMPTS and identity.get("d_model") == WIDTH,
            "fit prompt or geometry identity changed")
    require(identity.get("source_layers") == list(SOURCES) and identity.get("bank_arms") is None,
            "fit source bank changed")
    require(identity.get("saved_dtype") == "float16" and identity.get("checkpoint_every") == 1
            and identity.get("accumulation") == "equal-paragraph FP32 sum, divided by 100 once",
            "fit accumulation contract changed")
    profile = identity.get("profile")
    require(isinstance(profile, dict) and profile.get("d_model") == WIDTH
            and profile.get("source_layers") == list(SOURCES) and profile.get("target_layer") == 33,
            "fit production profile geometry changed")
    require(len(identity.get("prompt_sha256", [])) == N_PROMPTS
            and len(identity.get("token_lengths", [])) == N_PROMPTS
            and len(identity.get("n_valid", [])) == N_PROMPTS,
            "fit N100 provenance is incomplete")
    require(spec.get("n_prompts") == identity.get("n_prompts")
            and spec.get("source_layers") == identity.get("source_layers")
            and spec.get("d_model") == identity.get("d_model"),
            "fit identity is not bound to the frozen run specification")


def _fit_metadata_initialization_digest(final_dir: Path) -> dict[str, Any]:
    metadata = read_json(final_dir / "metadata.json")
    diagnostics = metadata.get("diagnostics")
    require(isinstance(diagnostics, list) and len(diagnostics) == N_PROMPTS,
            "fit final metadata does not retain all N100 diagnostics")
    initializations = []
    for index, row in enumerate(diagnostics):
        require(isinstance(row, dict) and row.get("index") == index and isinstance(row.get("initialization"), dict),
                "fit initialization provenance is incomplete")
        initializations.append(row["initialization"])
    return {"rows": N_PROMPTS, "initializations_sha256": digest(initializations)}


def _stage_manifest_sha256(manifest: dict[str, Any], contract: dict[str, Any],
                           stage: str) -> str:
    """Reconstruct a previously published stage manifest from a later one."""
    require(isinstance(manifest, dict) and isinstance(contract, dict),
            "stage manifest inputs are missing")
    paths = contract.get("stages", {}).get(stage)
    require(isinstance(paths, list) and paths == sorted(set(paths)),
            f"invalid {stage} stage membership")
    files = manifest.get("files")
    require(isinstance(files, dict) and set(paths) <= set(files),
            f"later manifest omits {stage} stage files")
    reconstructed = {
        "schema": "confirmation_artifact_manifest.v1",
        "binding": manifest.get("binding"),
        "kind": "stage",
        "stage_id": stage,
        "outcome": "complete",
        "files": {path: files[path] for path in paths},
    }
    return digest(reconstructed)


def _validate_fit_admission(root: Path, manifest: dict[str, Any],
                            contract: dict[str, Any]) -> dict[str, Any]:
    admission = read_json(root / "fit_admission.json")
    # ``measured_preflight_prompt_seconds`` is a copied decision input, not a
    # free-standing timing claim.  The completed gate profile is retained in
    # the same payload and is the sole producer of this measurement.
    gate_profile = read_json(root / "gate/huginn_r8.json")
    require(isinstance(gate_profile, dict), "native gate profile is not an object")
    measured_profile = gate_profile.get("prompt_with_checkpoint_seconds")
    require(type(measured_profile) in (int, float) and not isinstance(measured_profile, bool)
            and math.isfinite(float(measured_profile)) and float(measured_profile) > 0,
            "native gate prompt timing is missing or nonfinite")
    require(isinstance(admission, dict), "fit admission is not an object")
    required = {
        "n_prompts", "remaining_work_seconds", "measured_preflight_prompt_seconds",
        "historical_prompt_bound_seconds", "initial_prompt_bound_seconds",
        "required_seconds", "readout_reserve_seconds", "decision", "binding",
        "native_gate_manifest_sha256", "external_acceptance",
    }
    require(required <= set(admission), "fit admission record is incomplete")
    require(admission.get("n_prompts") == N_PROMPTS
            and admission.get("decision") == "admitted",
            "fit admission decision or prompt count changed")
    numeric = {
        key: admission.get(key)
        for key in ("remaining_work_seconds", "measured_preflight_prompt_seconds",
                    "historical_prompt_bound_seconds", "initial_prompt_bound_seconds",
                    "required_seconds", "readout_reserve_seconds")
    }
    for key, value in numeric.items():
        require(type(value) in (int, float) and not isinstance(value, bool)
                and math.isfinite(float(value)) and float(value) > 0,
                f"fit admission {key} is not finite")
    require(admission["historical_prompt_bound_seconds"] == HISTORICAL_PROMPT_BOUND
            and admission["readout_reserve_seconds"] == FIT_READOUT_RESERVE,
            "fit admission historical bound or reserve changed")
    require(admission["measured_preflight_prompt_seconds"] == measured_profile,
            "fit admission timing is not bound to the completed gate profile")
    expected_initial = FIT_SAFETY_MULTIPLIER * max(
        admission["measured_preflight_prompt_seconds"], HISTORICAL_PROMPT_BOUND)
    expected_required = N_PROMPTS * expected_initial + FIT_READOUT_RESERVE
    require(admission["initial_prompt_bound_seconds"] == expected_initial
            and admission["required_seconds"] == expected_required
            and admission["remaining_work_seconds"] >= admission["required_seconds"],
            "fit admission timing arithmetic changed")
    require(admission.get("binding") == manifest.get("binding"),
            "fit admission run binding changed")
    native_gate_manifest = _stage_manifest_sha256(manifest, contract, "development")
    require(admission.get("native_gate_manifest_sha256") == native_gate_manifest,
            "fit admission native gate manifest binding changed")
    acceptance = admission.get("external_acceptance")
    require(isinstance(acceptance, dict)
            and acceptance.get("kind") == "stage_science_authorization"
            and acceptance.get("binding") == manifest.get("binding")
            and acceptance.get("stage_manifest_sha256") == native_gate_manifest,
            "fit admission external acceptance changed")
    _valid_record(acceptance.get("acceptance_receipt_record"),
                  name="fit admission acceptance receipt")
    return {
        "n_prompts": N_PROMPTS,
        "required_seconds": admission["required_seconds"],
        "native_gate_manifest_sha256": native_gate_manifest,
        "decision": "admitted",
    }


def _legacy_contract_args(spec: dict[str, Any], legacy_paths: dict[str, Path]) -> SimpleNamespace:
    """Resolve the original deployment inputs from the frozen bundle.

    The receiver's ``run_spec.json`` is the new wrapper specification.  The
    historical Huginn validators must instead receive the original deployment
    specification and its adjacent combined/calibration records.
    """
    deployment = legacy_paths["run_refits"].parent
    # In the frozen archive the deployment is
    # ``<bundle>/repo/research/.../deployment``; in the source checkout it is
    # ``<repo>/research/.../deployment``.  Walk the deployment ancestry so the
    # lookup does not depend on either layout or on the receiver's cwd.
    bundle_roots = [*deployment.parents, Path.cwd(), Path.cwd().parent,
                    Path(__file__).resolve(), *Path(__file__).resolve().parents]
    # The external Ouro source is also present in the frozen bundle.  Select a
    # candidate by a recorded source byte, rather than by its pathname.
    expected_recurrent = _spec_record(spec, "ouro_project/src/ouro_jlens/recurrent.py")
    ouro_src = None
    seen: set[Path] = set()
    for bundle_root in bundle_roots:
        candidate = (Path(bundle_root) / "ouro_project/src").resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        recurrent = candidate / "ouro_jlens/recurrent.py"
        if recurrent.is_file():
            try:
                if record(recurrent) == expected_recurrent:
                    ouro_src = candidate
                    break
            except (OSError, ValueError):
                pass
    require(ouro_src is not None, "exact frozen Ouro source tree is unavailable")
    arguments = SimpleNamespace(
        run_spec=deployment / "run_spec.json",
        ouro_src=ouro_src,
        combined_contract=deployment / "combined_contract.json",
        calibration=deployment / "huginn_calibration.json",
    )
    for name in ("run_spec", "combined_contract", "calibration"):
        _regular(getattr(arguments, name))
    return arguments


def _legacy_source_paths(spec: dict[str, Any]) -> dict[str, Path]:
    """Find the exact old evaluator files in either source or frozen layout."""
    relative = {
        "run_refits": "repo/research/refit_round_2026-09-07/deployment/run_refits.py",
        "evaluate_refits": "repo/research/refit_round_2026-09-07/deployment/evaluate_refits.py",
        "run_huginn": "repo/research/refit_round_2026-09-07/deployment/run_huginn.py",
        "evaluate_huginn": "repo/research/refit_round_2026-09-07/deployment/evaluate_huginn.py",
    }
    source_records = spec.get("source_records")
    require(isinstance(source_records, dict), "Huginn source records are missing")
    module_paths: dict[str, Path] = {}
    here = Path(__file__).resolve()
    roots = [here.parents[1], *here.parents, Path.cwd()]
    for name, rel in relative.items():
        expected = source_records.get(rel)
        if expected is None:
            # The frozen records are keyed by bundle-relative paths.  Accept a
            # canonical exact suffix only when it is unique.
            matches = [value for key, value in source_records.items() if key.endswith(rel.removeprefix("repo/"))]
            require(len(matches) == 1, f"missing frozen source record: {rel}")
            expected = matches[0]
        _valid_record(expected, name=f"source {rel}")
        candidates = []
        for root in roots:
            candidates.extend((root / rel, root / rel.removeprefix("repo/")))
        candidates.extend([here.parent.parent / rel, here.parents[4] / rel.removeprefix("repo/")])
        found = None
        seen = set()
        for candidate in candidates:
            candidate = Path(candidate)
            if candidate in seen or not candidate.is_file():
                continue
            seen.add(candidate)
            try:
                if record(candidate) == expected:
                    found = candidate
                    break
            except (OSError, ValueError):
                continue
        require(found is not None, f"exact frozen legacy source is unavailable: {rel}")
        module_paths[name] = found
    deployment = module_paths["run_refits"].parent
    require(all(path.parent == deployment for path in module_paths.values()), "legacy evaluator files have different roots")
    return module_paths


def _load_legacy(spec: dict[str, Any]) -> tuple[
    types.ModuleType, types.ModuleType, types.ModuleType, types.ModuleType, dict[str, Path]
]:
    """Compile verified legacy modules under a private package namespace."""
    paths = _legacy_source_paths(spec)
    suffix = hashlib.sha256(canonical({key: str(value) for key, value in paths.items()})).hexdigest()[:16]
    package_name = f"_huginn_legacy_{suffix}"
    package = types.ModuleType(package_name)
    package.__path__ = [str(paths["run_refits"].parent)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    modules: dict[str, types.ModuleType] = {}
    old_flag = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        for short in ("run_refits", "evaluate_refits", "run_huginn", "evaluate_huginn"):
            name = f"{package_name}.{short}"
            module_spec = importlib.util.spec_from_file_location(name, paths[short])
            require(module_spec is not None and module_spec.loader is not None, f"cannot compile legacy {short}")
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[name] = module
            # Compile the bytes whose record was selected above.  Reading the
            # path again after hashing would allow a source replacement race to
            # change the code that is actually executed.
            expected = _spec_record(spec, f"/deployment/{short}.py")
            source = paths[short].read_bytes()
            require(record_bytes(source) == expected, f"legacy source changed before compiling: {short}")
            exec(compile(source, str(paths[short]), "exec"), module.__dict__)
            modules[short] = module
            setattr(package, short, module)
    except BaseException:
        for name in list(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)
        raise
    finally:
        sys.dont_write_bytecode = old_flag
    return (modules["evaluate_huginn"], modules["evaluate_refits"], modules["run_refits"],
            modules["run_huginn"], paths)


def _unload_legacy(paths: dict[str, Path]) -> None:
    # All callers pass the source-path mapping returned by _load_legacy.  The
    # package prefix is recovered from the uniquely scoped module path so an
    # unrelated validator invocation is left untouched.
    package_names = set()
    for module in ("run_refits", "evaluate_refits", "run_huginn", "evaluate_huginn"):
        path = paths.get(module)
        if path is None:
            continue
        for name, loaded in list(sys.modules.items()):
            if name.rsplit(".", 1)[-1] == module and getattr(loaded, "__file__", None) == str(path):
                package_names.add(name.rsplit(".", 1)[0])
    for package_name in package_names:
        for name in list(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)


def _independent_fp16_conversion(
    sums: dict[int, torch.Tensor],
    saved: dict[int, torch.Tensor],
    *,
    sources: tuple[int, ...] = SOURCES,
    width: int = WIDTH,
    samples: list[dict[str, Any]] | None = None,
) -> int:
    sample_positions = {
        (row, column)
        for row in (0, width // 2, width - 1)
        for column in (0, width // 2, width - 1)
    }
    count = 0
    for layer in sources:
        total = sums[layer]
        mean = saved[layer]
        _finite_tensor(total, (width, width), torch.float32, name=f"FP32 sum {layer}")
        _finite_tensor(mean, (width, width), torch.float16, name=f"FP16 mean {layer}")
        total_np = total.numpy()
        mean_np = mean.numpy()
        for start in range(0, width, 128):
            stop = min(width, start + 128)
            # Keep the denominator explicitly float32.  The cast and raw int16
            # comparison are independent of torch's implementation.
            expected = (total_np[start:stop] / np.float32(100)).astype(np.float16)
            observed = mean_np[start:stop]
            require(np.array_equal(observed.view(np.int16), expected.view(np.int16)),
                    f"FP16 N100 conversion differs at arm dense source {layer}, rows {start}:{stop}")
            if samples is not None:
                for row, column in sorted(sample_positions):
                    if start <= row < stop:
                        source_value = total_np[row, column]
                        expected_value = expected[row - start, column]
                        observed_value = observed[row - start, column]
                        samples.append({
                            "arm": "dense", "source": layer, "row": row, "column": column,
                            "fp32_sum_bits": int(np.asarray(source_value, dtype=np.float32).view(np.uint32)),
                            "numpy_fp16_bits": int(np.asarray(expected_value, dtype=np.float16).view(np.uint16)),
                            "saved_fp16_bits": int(np.asarray(observed_value, dtype=np.float16).view(np.uint16)),
                        })
            count += (stop - start) * width
    return count


def _validate_fit(root: Path, spec: dict[str, Any], paths: list[str],
                  *, manifest: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    require("fit/evidence.json" in paths and "fit/checkpoint.pt" in paths and "fit/lens.pt" in paths
            and "fit_admission.json" in paths,
            "fit stage is incomplete")
    admission_details = _validate_fit_admission(root, manifest, contract)
    evidence = read_json(root / "fit/evidence.json")
    require(isinstance(evidence, dict) and evidence.get("schema") == FIT_EVIDENCE_SCHEMA
            and evidence.get("category") == "newly_rerun_estimator" and evidence.get("n_prompts") == N_PROMPTS,
            "fit evidence schema or category changed")
    require(evidence.get("historical_final_bank_sha256") == spec.get("historical_bank_sha256") == HISTORICAL_BANK_SHA256,
            "historical fit comparison hash changed")
    decoded = _decode_evidence_files(evidence)
    owner_raw = decoded.get("OWNER.json")
    latest_raw = decoded.get("LATEST.json")
    complete_raw = decoded.get("COMPLETE.json")
    require(owner_raw is not None and latest_raw is not None and complete_raw is not None,
            "fit evidence omits a required pointer")
    owner = json.loads(owner_raw)
    identity = owner.get("identity") if isinstance(owner, dict) else None
    require(isinstance(owner, dict) and set(owner) == {"schema_version", "fit_identity_sha256", "identity"}
            and owner.get("schema_version") == 1 and owner.get("fit_identity_sha256") == digest(identity),
            "fit OWNER identity changed")
    _fit_identity_checks(identity, spec)
    identity_sha = owner["fit_identity_sha256"]
    latest = json.loads(latest_raw)
    complete = json.loads(complete_raw)
    checkpoint_rel, _ = _pointer_and_generation(latest, kind="checkpoint", identity_sha=identity_sha)
    final_rel, _ = _pointer_and_generation(complete, kind="final", identity_sha=identity_sha)
    expected_text = {
        "OWNER.json",
        "LATEST.json",
        "COMPLETE.json",
        f"{checkpoint_rel}/SEAL.json",
        f"{checkpoint_rel}/metadata.json",
        f"{final_rel}/SEAL.json",
        f"{final_rel}/metadata.json",
    }
    require(set(decoded) == expected_text, "fit evidence contains extra or missing sealed text files")

    def decoded_json(relative: str) -> Any:
        return json.loads(decoded[relative])

    for relative, kind, binary in (
        (f"{checkpoint_rel}/SEAL.json", "checkpoint", "state.pt"),
        (f"{final_rel}/SEAL.json", "final", "lens.pt"),
    ):
        seal = decoded_json(relative)
        require(isinstance(seal, dict)
                and set(seal) == {"schema_version", "kind", "fit_identity_sha256", "n_done", "files"}
                and seal.get("schema_version") == 1 and seal.get("kind") == kind
                and seal.get("fit_identity_sha256") == identity_sha and seal.get("n_done") == N_PROMPTS,
                f"{kind} generation seal changed")
        require(isinstance(seal.get("files"), dict) and set(seal["files"]) == {binary, "metadata.json"},
                f"{kind} generation seal members changed")
        for name, expected in seal["files"].items():
            relative_file = str(PurePosixPath(relative).parent / name)
            _valid_record(expected, name=f"fit {relative_file}")
            entry = decoded.get(relative_file)
            if entry is not None:
                # The binary is represented by a fixed payload file, while its
                # metadata remains embedded in evidence.
                if name == "metadata.json":
                    require(record_bytes(entry) == expected, f"fit {relative_file} seal differs")

    binaries = evidence.get("binaries")
    require(isinstance(binaries, dict) and set(binaries) == {"checkpoint.pt", "lens.pt"},
            "fit binary evidence members changed")
    binary_expected = {
        "checkpoint.pt": f"{checkpoint_rel}/state.pt",
        "lens.pt": f"{final_rel}/lens.pt",
    }
    for exported, original in binary_expected.items():
        entry = binaries[exported]
        require(isinstance(entry, dict) and set(entry) == {"original_relative_path", "record"}
                and entry.get("original_relative_path") == original,
                f"fit binary provenance changed: {exported}")
        expected = _valid_record(entry["record"], name=f"fit/{exported}")
        actual = _verify_record(root / "fit" / exported, expected, name=f"fit/{exported}")
        # export_fit writes the same binary record into the generation seal;
        # requiring equality binds the fixed payload path to that seal.
        seal_rel = f"{checkpoint_rel}/SEAL.json" if exported == "checkpoint.pt" else f"{final_rel}/SEAL.json"
        seal = decoded_json(seal_rel)
        require(seal["files"]["state.pt" if exported == "checkpoint.pt" else "lens.pt"] == actual,
                f"fit/{exported} differs from its embedded generation seal")

    legacy_huginn, legacy_ref, legacy_runner, legacy_huginn_runner, legacy_paths = _load_legacy(spec)
    scratch_parent = root.parent
    try:
        legacy_args = _legacy_contract_args(spec, legacy_paths)
        prepared = legacy_huginn_runner.contract(legacy_args)
        with tempfile.TemporaryDirectory(prefix=".huginn-fit-", dir=scratch_parent) as temporary:
            scratch = Path(temporary)
            _write_decoded(scratch, decoded)
            _link_fixed(root / "fit/checkpoint.pt", scratch / checkpoint_rel / "state.pt")
            _link_fixed(root / "fit/lens.pt", scratch / final_rel / "lens.pt")
            scratch_owner = read_json(scratch / "OWNER.json")
            legacy_huginn.validate_huginn_owner(
                scratch_owner,
                prepared["main"], prepared["combined"], prepared["manifest"], prepared["calibration"],
                combined_sha256=prepared["combined_sha256"],
                calibration_sha256=prepared["calibration_sha256"],
            )
            final_path, fit_record = legacy_ref._read_generation(torch, scratch, scratch_owner)
            init_validation = legacy_huginn._fit_initializations(final_path, prepared["calibration"])
            fit_record["initialization_validation"] = init_validation
            checkpoint_state = load_torch(scratch / checkpoint_rel / "state.pt")
            final_state = load_torch(scratch / final_rel / "lens.pt")
            require(isinstance(checkpoint_state, dict) and set(checkpoint_state) >= {"jacobian_sum"},
                    "legacy checkpoint loader returned no FP32 sums")
            require(isinstance(final_state, dict) and set(final_state) >= {"J"},
                    "legacy final loader returned no FP16 lens")
            sums = checkpoint_state["jacobian_sum"]
            saved = final_state["J"]
            require(isinstance(sums, dict) and isinstance(saved, dict)
                    and set(sums) == set(SOURCES) and set(saved) == set(SOURCES),
                    "fit bank source set changed")
            conversion_samples: list[dict[str, Any]] = []
            converted = _independent_fp16_conversion(sums, saved, samples=conversion_samples)
            fit_identity = {key: value for key, value in fit_record.items() if key != "fit_dir_at_evaluation"}
    finally:
        _unload_legacy(legacy_paths)

    actual_lens_record = record(root / "fit/lens.pt")
    historical_match = actual_lens_record["sha256"] == HISTORICAL_BANK_SHA256
    return {
        "fit_identity_sha256": identity_sha,
        "fit_record": fit_identity,
        "fp16_entries_compared": converted,
        "lens_sha256": actual_lens_record["sha256"],
        "historical_bank_sha256": HISTORICAL_BANK_SHA256,
        "historical_bank_hash_match": historical_match,
        "fp16_samples": conversion_samples,
        "initialization_validation": init_validation,
        "admission": admission_details,
        "legacy_loader": "evaluate_refits._read_generation",
    }


def record_bytes(raw: bytes) -> dict[str, Any]:
    return {"bytes": len(raw), "sha256": _sha256_bytes(raw)}


def _readout_identity_checks(root: Path, spec: dict[str, Any], fit: dict[str, Any]) -> dict[str, Any]:
    readout_root = root / "readouts"
    owner = read_json(readout_root / "OWNER.json")
    identity = owner.get("identity") if isinstance(owner, dict) else None
    require(isinstance(owner, dict) and owner.get("schema") == HUGINN_SCHEMA
            and owner.get("identity_sha256") == digest(identity), "Huginn readout owner identity changed")
    historical_spec_record = _valid_record(spec.get("historical_run_spec_record"),
                                           name="historical run specification")
    require(historical_spec_record == _spec_record(spec, "/deployment/run_spec.json"),
            "historical run specification is not the frozen deployment record")
    require(identity.get("kind") == "huginn_r8_n100_readout"
            and identity.get("run_spec_sha256") == historical_spec_record["sha256"]
            and identity.get("evaluation_base_seeds") == list(SEEDS)
            and identity.get("sections") == ["population", *(f"seeds/{seed}" for seed in SEEDS)]
            and identity.get("methods") == list(METHODS)
            and identity.get("learned_sources") == list(SOURCES)
            and identity.get("target_layer") == 33
            and identity.get("position") == -1
            and identity.get("evaluation_max_length") == 512,
            "Huginn readout identity geometry changed")
    fit_identity = identity.get("fit")
    require(isinstance(fit_identity, dict), "Huginn readout fit identity is missing")
    observed_fit = {key: value for key, value in fit_identity.items() if key != "fit_dir_at_evaluation"}
    expected_fit = {key: value for key, value in fit["fit_record"].items() if key != "fit_dir_at_evaluation"}
    require(observed_fit == expected_fit, "Huginn readout is bound to a different fit generation")
    require(fit_identity.get("fit_identity_sha256") == fit.get("fit_identity_sha256"),
            "Huginn readout fit digest differs")
    eligibility = read_json(readout_root / "population/eligibility.json")
    eligibility_record = record(readout_root / "population/eligibility.json")
    require(identity.get("eligibility_file") == eligibility_record
            and identity.get("eligibility_payload_sha256") == digest(eligibility),
            "Huginn readout eligibility binding differs")
    model = identity.get("model")
    require(isinstance(model, dict) and model.get("repo_id") == HUGINN_REPO_ID
            and model.get("revision") == HUGINN_REVISION,
            "Huginn readout model identity changed")
    manifest_record = _spec_record(spec, "/deployment/huginn_model_manifest.json")
    require(model.get("manifest_sha256") == manifest_record["sha256"], "Huginn model manifest binding differs")
    combined_record = _spec_record(spec, "/deployment/combined_contract.json")
    require(identity.get("combined_contract_sha256") == combined_record["sha256"],
            "Huginn combined contract binding differs")
    return identity


def _validate_readouts(root: Path, spec: dict[str, Any], paths: list[str], fit: dict[str, Any]) -> dict[str, Any]:
    expected = {f"readouts/{relative}" for relative in _expected_readout_paths(spec)}
    actual = {path for path in paths if path.startswith("readouts/")}
    require(actual == expected, "Huginn readout file membership changed")
    identity = _readout_identity_checks(root, spec, fit)
    legacy_huginn, legacy_ref, legacy_runner, legacy_huginn_runner, legacy_paths = _load_legacy(spec)
    try:
        checked = legacy_huginn.validate_output(root / "readouts")
    finally:
        _unload_legacy(legacy_paths)
    require(isinstance(checked, dict) and checked.get("files") == len(EXPECTED_READOUT_PATHS),
            "historical Huginn validator did not check all seventeen files")
    return {"files": checked["files"], "schema": HUGINN_SCHEMA,
            "seeds": list(SEEDS), "fit_identity_sha256": identity["fit"]["fit_identity_sha256"]}


def validate_outputs(*, root: Path, manifest: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    """Validate one complete development, fit, readout, or final stage."""
    root = Path(root)
    spec, paths = _validate_spec_and_manifest(root, manifest, contract)
    _validate_provenance(root, spec, manifest)
    gate_details = _validate_gate(root, spec)
    stage = manifest.get("stage_id") if manifest.get("kind") == "stage" else "final"
    details: dict[str, Any] = {"stage": stage, "native_gate": gate_details}
    fit_details = None
    if stage in ("fit", "readouts", "final"):
        fit_details = _validate_fit(root, spec, paths, manifest=manifest, contract=contract)
        details["fit"] = fit_details
    if stage in ("readouts", "final"):
        require(fit_details is not None, "readout stage lacks a validated fit")
        details["readouts"] = _validate_readouts(root, spec, paths, fit_details)
    return {
        "schema": SCHEMA,
        "status": "passed",
        "run_id": contract["run_id"],
        "manifest_sha256": digest(manifest),
        "contract_sha256": manifest["binding"]["output_contract_sha256"],
        "checked_files": sorted(manifest["files"]),
        "checks": {check: True for check in contract["required_checks"]},
        "details": details,
    }


def _self_test_source_spec(repo: Path) -> dict[str, Any]:
    """Build a source-record view of the checked-in historical closure."""
    old = repo / "research/refit_round_2026-09-07"
    deployment = old / "deployment"
    ouro_candidates = [
        Path("/home/moloch/ouro_project/src"),
        repo / "research/confirmation_2026-09-09/huginn_verification/resources/contract_probe/ouro_project/src",
    ]
    ouro = next((candidate for candidate in ouro_candidates if candidate.is_dir()), None)
    require(ouro is not None, "self-test Ouro source fixture is unavailable")
    files: dict[str, Path] = {}
    for name in ("run_refits.py", "evaluate_refits.py", "run_huginn.py",
                 "evaluate_huginn.py", "run_spec.json", "combined_contract.json",
                 "huginn_calibration.json", "huginn_model_manifest.json", "huginn_adapter.py"):
        files[f"repo/research/refit_round_2026-09-07/deployment/{name}"] = deployment / name
    files["repo/fit_estimators.py"] = old / "fit_estimators.py"
    for name in ("optimized_fitting.py", "cuda_graph_candidate.py", "saved_tensor_candidate.py"):
        files[f"repo/optimization/{name}"] = old / "optimization" / name
    for name in ("__init__.py", "fitting.py", "hooks.py", "hf.py", "lens.py", "protocol.py"):
        files[f"repo/jlens/{name}"] = repo / "jlens" / name
    for name in ("recurrent.py", "evidence.py", "bench.py"):
        files[f"ouro_project/src/ouro_jlens/{name}"] = ouro / "ouro_jlens" / name
    require(all(path.is_file() for path in files.values()),
            "self-test historical source fixture is unavailable")
    return {"schema": "huginn_verification_run.v1",
            "source_records": {key: record(path) for key, path in files.items()}}


def _expect_rejection(callable_: Any, checks: list[str], label: str) -> None:
    try:
        callable_()
    except (AssertionError, ValueError, RuntimeError, KeyError, TypeError):
        checks.append(label)
    else:
        raise AssertionError(f"self-test accepted invalid {label}")


def _self_test_producer_integration(repo: Path, checks: list[str]) -> None:
    """Exercise authentic gate/legacy/readout schemas without model loading."""
    spec = _self_test_source_spec(repo)
    gate_root = repo / "research/refit_round_2026-09-07/monitoring/attempt_06/20260909T115652Z/results/preflight_huginn"
    require((gate_root / "START.json").is_file() and (gate_root / "COMPLETE.json").is_file(),
            "self-test authentic gate fixture is unavailable")
    start = read_json(gate_root / "START.json")
    complete = read_json(gate_root / "COMPLETE.json")
    _validate_gate_runtime(start, complete, spec)
    _validate_benchmark_initialization(complete["benchmark_initialization"], 128)
    checks.append("authentic_benchmark_runtime_gate")

    changed_seed = copy.deepcopy(complete)
    changed_seed["benchmark_initialization"]["base_seed"] += 1
    _expect_rejection(lambda: _validate_benchmark_initialization(
        changed_seed["benchmark_initialization"], 128), checks,
        "changed_benchmark_seed_reject")
    changed_input = copy.deepcopy(complete)
    changed_input["benchmark_initialization"]["input_ids_sha256"] = "0" * 64
    _expect_rejection(lambda: _validate_benchmark_initialization(
        changed_input["benchmark_initialization"], 128), checks,
        "changed_benchmark_input_reject")
    changed_model_start = copy.deepcopy(start)
    changed_model_complete = copy.deepcopy(complete)
    changed_model_start["runtime"]["model"]["revision"] = "0" * 40
    changed_model_complete["runtime"]["model"]["revision"] = "0" * 40
    _expect_rejection(lambda: _validate_gate_runtime(
        changed_model_start, changed_model_complete, spec), checks,
        "changed_benchmark_model_reject")

    # Exercise the fit admission binding independently of the large fit
    # tensors.  The stage digest is reconstructed from the same manifest
    # records that the receiver receives at the fit stage.
    admission_binding = {"run_id": "self-test", "output_contract_sha256": "1" * 64}
    admission_contract = {"stages": {"development": ["run_spec.json"]}}
    admission_manifest = {
        "binding": admission_binding,
        "files": {"run_spec.json": {"bytes": 1, "sha256": "2" * 64}},
    }
    admission_stage_sha = _stage_manifest_sha256(
        admission_manifest, admission_contract, "development")
    measured = 10.0
    initial_bound = FIT_SAFETY_MULTIPLIER * HISTORICAL_PROMPT_BOUND
    required_seconds = N_PROMPTS * initial_bound + FIT_READOUT_RESERVE
    admission_root = Path(tempfile.mkdtemp(prefix="huginn-fit-admission-self-test-"))
    (admission_root / "gate").mkdir()
    (admission_root / "gate/huginn_r8.json").write_bytes(
        canonical({"prompt_with_checkpoint_seconds": measured}))
    admission = {
        "n_prompts": N_PROMPTS,
        "remaining_work_seconds": required_seconds + 1.0,
        "measured_preflight_prompt_seconds": measured,
        "historical_prompt_bound_seconds": HISTORICAL_PROMPT_BOUND,
        "initial_prompt_bound_seconds": initial_bound,
        "required_seconds": required_seconds,
        "readout_reserve_seconds": FIT_READOUT_RESERVE,
        "decision": "admitted",
        "binding": admission_binding,
        "native_gate_manifest_sha256": admission_stage_sha,
        "external_acceptance": {
            "kind": "stage_science_authorization",
            "binding": admission_binding,
            "stage_manifest_sha256": admission_stage_sha,
            "acceptance_receipt_record": {"bytes": 1, "sha256": "3" * 64},
        },
    }
    (admission_root / "fit_admission.json").write_bytes(canonical(admission))
    _validate_fit_admission(admission_root, admission_manifest, admission_contract)
    checks.append("fit_admission_binding_pass")
    changed_profile = {"prompt_with_checkpoint_seconds": measured + 1.0}
    (admission_root / "gate/huginn_r8.json").write_bytes(canonical(changed_profile))
    _expect_rejection(lambda: _validate_fit_admission(
        admission_root, admission_manifest, admission_contract), checks,
        "changed_fit_admission_profile_reject")
    (admission_root / "gate/huginn_r8.json").write_bytes(
        canonical({"prompt_with_checkpoint_seconds": measured}))
    changed_admission = copy.deepcopy(admission)
    changed_admission["required_seconds"] += 1.0
    (admission_root / "fit_admission.json").write_bytes(canonical(changed_admission))
    _expect_rejection(lambda: _validate_fit_admission(
        admission_root, admission_manifest, admission_contract), checks,
        "changed_fit_admission_arithmetic_reject")
    shutil.rmtree(admission_root)

    legacy_huginn, _legacy_ref, _legacy_runner, legacy_huginn_runner, legacy_paths = _load_legacy(spec)
    try:
        arguments = _legacy_contract_args(spec, legacy_paths)
        prepared = legacy_huginn_runner.contract(arguments)
        owner_path = repo / "research/refit_round_2026-09-07/cloud_leases/attempt_06/retrieved/huginn/fit_01/OWNER.json"
        require(owner_path.is_file(), "self-test producer OWNER fixture is unavailable")
        owner = read_json(owner_path)
        legacy_huginn.validate_huginn_owner(
            owner, prepared["main"], prepared["combined"], prepared["manifest"], prepared["calibration"],
            combined_sha256=prepared["combined_sha256"],
            calibration_sha256=prepared["calibration_sha256"],
        )
        checks.append("authentic_huginn_owner_pass")
        bad_calibration = copy.deepcopy(prepared["calibration"])
        bad_calibration["rows"][0]["base_seed"] += 1
        _expect_rejection(lambda: legacy_huginn.validate_huginn_owner(
            owner, prepared["main"], prepared["combined"], prepared["manifest"], bad_calibration,
            combined_sha256=prepared["combined_sha256"],
            calibration_sha256=prepared["calibration_sha256"],
        ), checks, "changed_frozen_calibration_reject")

        # _fit_initializations reads the producer's sealed metadata.  A small
        # temporary seal lets the test invoke that original checker while
        # retaining the real 100-row calibration and seed recipe.
        with tempfile.TemporaryDirectory(prefix="huginn-initialization-self-test-") as directory:
            final = Path(directory) / "final"
            final.mkdir()
            diagnostics = []
            for index, row in enumerate(prepared["calibration"]["rows"]):
                diagnostics.append({"index": index, "initialization": {
                    "base_seed": row["base_seed"], "namespace": "calibration",
                    "prompt_state_seed": row["prompt_state_seed"],
                    "input_ids_sha256": row["input_ids_sha256"],
                    "seed_recipe": SEED_RULE, "dtype": "torch.bfloat16",
                    "test_time_noise": 0,
                    "one_prompt_shape": [1, row["token_length"], WIDTH],
                }})
            metadata_path = final / "metadata.json"
            metadata_path.write_bytes(canonical({"diagnostics": diagnostics}))
            lens_path = final / "lens.pt"
            lens_path.write_bytes(b"self-test-lens")
            seal = {"files": {"metadata.json": record(metadata_path)}}
            (final / "SEAL.json").write_bytes(canonical(seal))
            legacy_huginn._fit_initializations(lens_path, prepared["calibration"])
            checks.append("authentic_fit_initializations_pass")
            changed_diagnostics = copy.deepcopy(diagnostics)
            changed_diagnostics[0]["initialization"]["prompt_state_seed"] += 1
            metadata_path.write_bytes(canonical({"diagnostics": changed_diagnostics}))
            # Reseal the changed metadata so this negative case reaches the
            # historical calibration/seed check rather than failing only at
            # the outer file-record comparison.
            seal["files"]["metadata.json"] = record(metadata_path)
            (final / "SEAL.json").write_bytes(canonical(seal))
            _expect_rejection(lambda: legacy_huginn._fit_initializations(
                lens_path, prepared["calibration"]), checks,
                "changed_frozen_seed_record_reject")
    finally:
        _unload_legacy(legacy_paths)

    readout_root = repo / "research/refit_round_2026-09-07/monitoring/attempt_06/huginn_handoff_20260909T153351Z_659a482c/results/huginn_evaluation"
    require((readout_root / "OWNER.json").is_file()
            and (readout_root / "population/eligibility.json").is_file(),
            "self-test historical readout OWNER fixture is unavailable")
    with tempfile.TemporaryDirectory(prefix="huginn-readout-identity-self-test-") as directory:
        root = Path(directory)
        (root / "readouts/population").mkdir(parents=True)
        (root / "readouts/OWNER.json").write_bytes((readout_root / "OWNER.json").read_bytes())
        (root / "readouts/population/eligibility.json").write_bytes(
            (readout_root / "population/eligibility.json").read_bytes())
        readout_owner = read_json(root / "readouts/OWNER.json")
        fit_identity = readout_owner["identity"]["fit"]
        readout_spec = copy.deepcopy(spec)
        deployment = repo / "research/refit_round_2026-09-07/deployment"
        readout_spec["historical_run_spec_record"] = record(deployment / "run_spec.json")
        fit = {"fit_identity_sha256": fit_identity["fit_identity_sha256"],
               "fit_record": copy.deepcopy(fit_identity)}
        _readout_identity_checks(root, readout_spec, fit)
        checks.append("historical_readout_owner_identity_pass")
        bad_spec = copy.deepcopy(readout_spec)
        bad_spec["historical_run_spec_record"]["sha256"] = "0" * 64
        _expect_rejection(lambda: _readout_identity_checks(root, bad_spec, fit), checks,
                          "changed_historical_run_spec_reject")


def _self_test() -> dict[str, Any]:
    """Run small CPU-only checks for raw bits, primal coupling, and evidence."""
    checks: list[str] = []
    with tempfile.TemporaryDirectory(prefix="huginn-validator-self-test-") as directory:
        root = Path(directory)
        maps = {"dense": {layer: torch.tensor([[1.0 + layer, -2.0], [3.5, 4.25]], dtype=torch.float32)
                           for layer in (0, 1)}}
        payload = {"maps": maps, "sources": [0, 1], "width": 2}
        bank, bit_hash = _validate_maps(payload, name="CPU fixture", sources=(0, 1), width=2)
        changed = {"maps": {"dense": {0: bank[0].clone(), 1: bank[1].clone()}}, "sources": [0, 1], "width": 2}
        changed["maps"]["dense"][1].view(torch.int32)[0, 0] ^= 1
        try:
            _bitwise_equal(bank[1], changed["maps"]["dense"][1], name="fixture mismatch")
        except ValueError:
            checks.append("matrix_raw_bits_reject")
        else:
            raise AssertionError("raw-bit mismatch was accepted")

        sums = {layer: torch.tensor([[100.0 + layer, -25.0], [3.0, 200.0]], dtype=torch.float32)
                for layer in (0, 1)}
        saved = {layer: (value / 100).half() for layer, value in sums.items()}
        require(_independent_fp16_conversion(sums, saved, sources=(0, 1), width=2) == 8,
                "CPU FP16 conversion fixture failed")
        checks.append("independent_numpy_fp16_all_entries")
        saved[1].view(torch.int16)[0, 0] ^= 1
        try:
            _independent_fp16_conversion(sums, saved, sources=(0, 1), width=2)
        except ValueError:
            checks.append("numpy_conversion_reject")
        else:
            raise AssertionError("invalid FP16 conversion was accepted")

        ids = torch.tensor([[65504, 4, 5]], dtype=torch.long).repeat(8, 1)
        initial = torch.zeros((8, 3, 2), dtype=torch.bfloat16)
        states = {layer: torch.full((8, 3, 2), layer + 1, dtype=torch.bfloat16) for layer in range(3)}
        logits = torch.arange(8 * 3 * 65536, dtype=torch.float32).reshape(8, 3, 65536)
        benchmark_recipe = {"version": 1, "base_seed": 2026090899,
                            "namespace": "benchmark", "input_ids": ids[0].tolist()}
        initialization = {
            "base_seed": 2026090899,
            "namespace": "benchmark",
            "prompt_state_seed": int.from_bytes(hashlib.sha256(canonical(benchmark_recipe)).digest()[:8], "big") % (2 ** 63),
            "input_ids_sha256": digest(ids[0].tolist()),
            "seed_recipe": SEED_RULE,
            "one_prompt_shape": [1, 3, 2],
            "generator_device": "cuda:0",
            "dtype": "torch.bfloat16",
            "test_time_noise": 0,
            "lane_coupling": "contiguous copies of the same one-prompt state",
        }
        primal = {"input_ids": ids, "initial_states": initial, "adapter_states": states,
                  "native_states": {k: v.clone() for k, v in states.items()},
                  "adapter_logits": logits, "native_logits": logits.clone(),
                  "initialization": initialization}
        _validate_primal(primal, width=2, vocab=65536, n_layers=3, sequence_length=3)
        checks.append("primal_all_cells_and_lanes")
        primal["native_states"][2].view(torch.int16)[0, 0] ^= 1
        try:
            _validate_primal(primal, width=2, vocab=65536, n_layers=3, sequence_length=3)
        except ValueError:
            checks.append("primal_raw_bits_reject")
        else:
            raise AssertionError("invalid primal state was accepted")

        malformed = {"files": {"x/../bad.json": {"record": {"bytes": 1, "sha256": "0" * 64}, "base64": "AA=="}}}
        try:
            _decode_evidence_files(malformed)
        except ValueError:
            checks.append("evidence_path_reject")
        else:
            raise AssertionError("unsafe evidence path was accepted")
        _self_test_producer_integration(Path(__file__).resolve().parents[4], checks)
    require(not torch.cuda.is_initialized(), "CPU self-test initialized CUDA")
    return {"status": "pass", "checks": checks, "cuda_initialized": torch.cuda.is_initialized(),
            "source_sha256": _sha256_bytes(Path(__file__).read_bytes())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.self_test:
        parser.error("this receiver module is called through validate_outputs; use --self-test for the local smoke")
    print(json.dumps(_self_test(), sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
