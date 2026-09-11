"""Portable, resumable runner for the five frozen Ouro N=100 refits.

``--plan`` uses only the standard library and never loads a model. Execution
requires the explicit local snapshot and Ouro source tree, verifies their
manifest before loading, and uses the accepted B=8 dense CUDA graph engine.
Every completed paragraph contributes once to an FP32 sum. A lens is published
only after all 100 paragraphs, as the FP16 equal-paragraph mean.

Each checkpoint/final attempt has its own directory. State, metadata, and seal
are durable before an atomic pointer commits the attempt. Resume follows only
that pointer, verifies bytes before deserialization, and never adopts an
incomplete or orphaned attempt. The reusable ``run_fit`` helper permits tiny
CPU matrices for orchestration tests; production CLI geometry is fixed.
Execution requires an absolute stop time. The runner stops between paragraphs;
the caller must separately end the Pod lease and its billing.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import signal
import stat
import sys
import tempfile
import time
import uuid


HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
REPO = HERE.parents[2]
REVISION = "1ed04250da1a9936042725d302e81c8fa2ab5abd"
SETTINGS = {
    "n_ut": 4, "n_physical": 48, "d_model": 2048,
    "target_layer": 191, "source_layers": list(range(191)),
    "mode": "dense", "dense_engine": "cuda_graph",
    "compress_saved_tensors": True, "dim_batch": 8,
    "max_seq_len": 128, "skip_first": 16,
    "accumulation_dtype": "float32", "saved_dtype": "float16",
    "checkpoint_every": 1,
}
# These are separate scientific fits, never aliases for the main replication.
# Omitting production_profile preserves the original main-fit artifact schema.
PROFILES = {
    "ouro_penultimate": {"d_model": 2048, "source_layers": list(range(190)),
                         "target_layer": 190, "bank_arms": None},
    "ouro_positions": {"d_model": 2048, "source_layers": list(range(191)),
                       "target_layer": 191, "bank_arms": ["sampled_sum", "diagonal"]},
    "huginn_r8": {"d_model": 5280, "source_layers": list(range(32)),
                  "target_layer": 33, "bank_arms": None},
}
_GENERATION = re.compile(r"cursor_([0-9]{6})_([0-9a-f]{32})\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _no_links(path):
    """Check lexical paths, without erasing symlinks through resolve()."""
    path = Path(os.path.abspath(path))
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"refusing linked output or evidence path: {current}")
    return path


def _fsync_dir(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir(path):
    path = _no_links(path)
    path.mkdir(parents=True, exist_ok=True)
    _no_links(path)
    return path


def _new_json(path, value):
    """An exclusive generation member; retain failed writes for inspection."""
    path = _no_links(path)
    with path.open("xb") as handle:
        handle.write(_canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_dir(path.parent)


def _atomic_json(path, value):
    # Generation payloads deliberately do not use an atomic helper that deletes
    # failed temporary state. A pointer itself is a small replacement document.
    path = _no_links(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    _new_json(temporary, value)
    os.replace(temporary, path)
    _fsync_dir(path.parent)


def _record(path):
    path = _no_links(path)
    if not path.is_file():
        raise ValueError(f"missing regular evidence file: {path}")
    return {"bytes": path.stat().st_size, "sha256": _file_hash(path)}


def _verify_record(path, record):
    if not isinstance(record, dict) or set(record) != {"bytes", "sha256"}:
        raise ValueError(f"invalid file record: {path}")
    _integer("record bytes", record["bytes"])
    if not isinstance(record["sha256"], str) or not _SHA256.fullmatch(record["sha256"]):
        raise ValueError(f"invalid SHA256: {path}")
    if _record(path) != record:
        raise ValueError(f"evidence bytes do not match their seal: {path}")


def _relative(path):
    if not isinstance(path, str):
        raise ValueError("manifest paths must be strings")
    parts = PurePosixPath(path)
    if parts.is_absolute() or not parts.parts or any(p in (".", "..") for p in parts.parts):
        raise ValueError(f"invalid relative manifest path: {path!r}")
    if parts.as_posix() != path or "\\" in path:
        raise ValueError(f"noncanonical relative manifest path: {path!r}")
    return parts


def _bank(torch, bank, sources, width, dtype, name):
    if not isinstance(bank, dict) or any(type(key) is not int for key in bank):
        raise ValueError(f"{name} must be a dictionary with integer source keys")
    if set(bank) != set(sources):
        raise ValueError(f"{name} source layers do not match the fixed fit")
    for layer in sources:
        tensor = bank[layer]
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                or tensor.device.type != "cpu" or tensor.dtype != dtype
                or tuple(tensor.shape) != (width, width) or tensor.requires_grad):
            raise ValueError(f"{name}[{layer}] has invalid shape, dtype, device, or gradient state")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name}[{layer}] is nonfinite")


def _banks(torch, bank, identity, dtype, name):
    """Validate one matrix bank, or explicitly named banks committed together."""
    arms = identity.get("bank_arms")
    if arms is None:
        _bank(torch, bank, identity["source_layers"], identity["d_model"], dtype, name)
        return
    if not isinstance(bank, dict) or set(bank) != set(arms):
        raise ValueError(f"{name} does not contain exactly the declared estimator arms")
    for arm in arms:
        _bank(torch, bank[arm], identity["source_layers"], identity["d_model"], dtype, f"{name}.{arm}")


def _map_banks(bank, identity, function):
    if identity.get("bank_arms") is None:
        return {layer: function(tensor) for layer, tensor in bank.items()}
    return {arm: {layer: function(tensor) for layer, tensor in bank[arm].items()}
            for arm in identity["bank_arms"]}


def _bank_pairs(left, right, identity):
    for arm in identity.get("bank_arms") or [None]:
        a, b = (left, right) if arm is None else (left[arm], right[arm])
        for layer in identity["source_layers"]:
            yield arm, layer, a[layer], b[layer]


def _utc_now():
    return datetime.now(timezone.utc)


def _deadline(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("deadline must be an absolute timezone-aware UTC timestamp")
    return value.astimezone(timezone.utc)


def _fit_identity(prompts, lengths, valid, fit_id, identity, width, sources, cpu_test,
                  production_profile=None):
    _integer("fit_id", fit_id, 1)
    if fit_id > 5:
        raise ValueError("fit_id must be in 1..5")
    _integer("d_model", width, 1)
    if (not isinstance(prompts, (list, tuple)) or len(prompts) != 100
            or any(not isinstance(prompt, str) or not prompt for prompt in prompts)
            or len(set(prompts)) != 100):
        raise ValueError("the fixed fit requires exactly 100 distinct nonempty paragraphs")
    if len(lengths) != 100 or len(valid) != 100:
        raise ValueError("token lengths and valid-position counts must each contain 100 entries")
    for length, count in zip(lengths, valid):
        _integer("token length", length, 2)
        _integer("valid-position count", count, 1)
        if count >= length:
            raise ValueError("valid-position count must be smaller than sequence length")
    if (not sources or any(type(layer) is not int or layer < 0 for layer in sources)
            or list(sources) != sorted(set(sources))):
        raise ValueError("source_layers must be sorted, unique nonnegative integers")
    if type(cpu_test) is not bool:
        raise ValueError("cpu_test must be boolean")
    if production_profile is not None and production_profile not in PROFILES:
        raise ValueError("unknown production fit profile")
    profile = PROFILES.get(production_profile)
    expected = profile or SETTINGS
    if not cpu_test and (width != expected["d_model"] or list(sources) != expected["source_layers"]):
        raise ValueError("production geometry differs from the fixed fit profile")
    if profile is not None and fit_id != 1:
        raise ValueError("bounded control and Huginn profiles require preselected fit_id=1")
    if not isinstance(identity, dict) or not identity:
        raise ValueError("a nonempty runtime identity is required")
    payload = {
        "schema_version": 1, "kind": "ouro_n100_fit", "fit_id": fit_id,
        "runtime": identity, "cpu_test": cpu_test, "n_prompts": 100,
        "d_model": width, "source_layers": list(sources),
        "ordered_prompts_sha256": _digest(list(prompts)),
        "prompt_sha256": [hashlib.sha256(p.encode("utf-8")).hexdigest() for p in prompts],
        "token_lengths": list(lengths), "n_valid": list(valid),
        "accumulation": "equal-paragraph FP32 sum, divided by 100 once",
        "saved_dtype": "float16", "checkpoint_every": 1,
    }
    if profile is not None:
        payload.update(kind=f"{production_profile}_n100_fit", production_profile=production_profile,
                       profile=profile, bank_arms=profile["bank_arms"])
    # A deep canonical copy prevents mutation of a caller-owned nested dict.
    return json.loads(_canonical(payload))


@contextmanager
def _owned_fit(output_dir, fit_id, identity):
    root = _mkdir(output_dir)
    path = root / f"fit_{fit_id:02d}"
    if path.exists() and not (path / "OWNER.json").is_file():
        if any(path.iterdir()):
            raise ValueError(f"existing fit directory lacks an owner identity: {path}")
    path = _mkdir(path)
    lock = _no_links(path / ".lock")
    with lock.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another process owns fit directory {path}") from error
        owner = {"schema_version": 1, "fit_identity_sha256": _digest(identity), "identity": identity}
        owner_path = _no_links(path / "OWNER.json")
        if owner_path.exists():
            if _json(owner_path) != owner:
                raise ValueError("fit identity changed; refusing resume or overwrite")
        else:
            unknown = [p.name for p in path.iterdir() if p.name != ".lock"]
            if unknown:
                raise ValueError(f"unowned entries in new fit directory: {unknown}")
            _new_json(owner_path, owner)
        try:
            yield path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _output_lock(output_dir):
    """Serialize production CLI invocations before GPU or model initialization."""
    root = _mkdir(output_dir)
    with _no_links(root / ".run_refits.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another runner owns output tree {root}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _bind_run_identity(output_dir, identity, *, kind="ouro_n100_run"):
    """Bind all five production fits to one runtime under ``_output_lock``."""
    if not isinstance(identity, dict) or not identity:
        raise ValueError("a nonempty shared runtime identity is required")
    identity = json.loads(_canonical(identity))
    root = _mkdir(output_dir)
    path = _no_links(root / "RUN_IDENTITY.json")
    if kind not in ("ouro_n100_run", "ouro_controls_n100_run", "huginn_r8_n100_run"):
        raise ValueError("unknown shared production run kind")
    record = {"schema_version": 1, "kind": kind,
              "identity_sha256": _digest(identity), "identity": identity}
    if path.exists():
        if _json(path) != record:
            raise ValueError("shared runtime identity changed across fits; refusing this output tree")
    else:
        if any(re.fullmatch(r"fit_[0-9]{2}", p.name) or p.name in PROFILES for p in root.iterdir()):
            raise ValueError("existing fit state lacks a shared production run identity")
        _new_json(path, record)
    return path


def _pointer_generation(fit_dir, pointer, kind, identity_sha):
    expected = {"schema_version", "kind", "generation", "seal", "fit_identity_sha256",
                "n_done", "next_idx", "observed_prompt_bound_seconds"}
    if not isinstance(pointer, dict) or set(pointer) != expected:
        raise ValueError("invalid committed generation pointer")
    if (pointer["schema_version"] != 1 or pointer["kind"] != kind
            or pointer["fit_identity_sha256"] != identity_sha):
        raise ValueError("committed pointer does not match this fit")
    cursor = _integer("checkpoint cursor", pointer["n_done"], 1)
    if cursor > 100 or pointer["next_idx"] != cursor or type(pointer["next_idx"]) is not int:
        raise ValueError("checkpoint count/cursor mismatch")
    parent = "checkpoints" if kind == "checkpoint" else "final"
    relative = _relative(pointer["generation"])
    if len(relative.parts) != 2 or relative.parts[0] != parent:
        raise ValueError("pointer does not reference an owned generation directory")
    match = _GENERATION.fullmatch(relative.parts[1])
    if not match or int(match.group(1)) != cursor:
        raise ValueError("generation name and cursor disagree")
    bound = pointer["observed_prompt_bound_seconds"]
    if isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(bound) or bound < 0:
        raise ValueError("invalid saved timing bound")
    path = _no_links(fit_dir / relative)
    _verify_record(path / "SEAL.json", pointer["seal"])
    seal = _json(path / "SEAL.json")
    if (set(seal) != {"schema_version", "kind", "fit_identity_sha256", "n_done", "files"}
            or seal["schema_version"] != 1 or seal["kind"] != kind
            or seal["fit_identity_sha256"] != identity_sha or seal["n_done"] != cursor):
        raise ValueError("generation seal identity or count mismatch")
    binary = "state.pt" if kind == "checkpoint" else "lens.pt"
    if not isinstance(seal["files"], dict) or set(seal["files"]) != {binary, "metadata.json"}:
        raise ValueError("generation seal has unexpected members")
    if set(p.name for p in path.iterdir()) != {binary, "metadata.json", "SEAL.json"}:
        raise ValueError("sealed generation has unrecorded entries")
    for name, record in seal["files"].items():
        _verify_record(path / name, record)
    return path, seal, _json(path / "metadata.json")


def _check_metadata(metadata, identity, cursor, kind):
    if not isinstance(metadata, dict):
        raise ValueError("invalid generation metadata")
    expected = {
        "schema_version": 1, "kind": kind, "fit_identity_sha256": _digest(identity),
        "identity": identity, "n_done": cursor, "next_idx": cursor,
        "completed_prompt_sha256": identity["prompt_sha256"][:cursor],
        "completed_prefix_sha256": _digest(identity["prompt_sha256"][:cursor]),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"generation metadata mismatch: {key}")
    diagnostics = metadata.get("diagnostics")
    if not isinstance(diagnostics, list) or len(diagnostics) != cursor:
        raise ValueError("diagnostic count does not match checkpoint cursor")
    for index, row in enumerate(diagnostics):
        for key, value in {
            "index": index, "prompt_sha256": identity["prompt_sha256"][index],
            "token_length": identity["token_lengths"][index], "n_valid": identity["n_valid"][index],
        }.items():
            if not isinstance(row, dict) or row.get(key) != value:
                raise ValueError(f"diagnostic prompt prefix mismatch at {index}: {key}")


def _read_checkpoint(torch, fit_dir, identity):
    pointer_path = _no_links(fit_dir / "LATEST.json")
    if not pointer_path.exists():
        return None
    pointer = _json(pointer_path)
    generation, seal, metadata = _pointer_generation(fit_dir, pointer, "checkpoint", _digest(identity))
    cursor = pointer["n_done"]
    _check_metadata(metadata, identity, cursor, "checkpoint")
    state = torch.load(generation / "state.pt", map_location="cpu", weights_only=True, mmap=True)
    expected = {
        "schema_version": 1, "kind": "checkpoint", "fit_identity_sha256": _digest(identity),
        "n_done": cursor, "next_idx": cursor, "d_model": identity["d_model"],
        "source_layers": identity["source_layers"],
        "completed_prefix_sha256": _digest(identity["prompt_sha256"][:cursor]),
    }
    if not isinstance(state, dict) or set(state) != {*expected, "jacobian_sum"}:
        raise ValueError("checkpoint payload schema mismatch")
    for key, value in expected.items():
        if state[key] != value:
            raise ValueError(f"checkpoint payload mismatch: {key}")
    _banks(torch, state["jacobian_sum"], identity, torch.float32, "FP32 sum")
    return {"sum": state["jacobian_sum"], "cursor": cursor, "diagnostics": metadata["diagnostics"],
            "pointer": pointer, "generation": generation, "seal": seal}


def _fault(hook, stage, kind, path, cursor):
    if hook is not None:
        hook(stage, {"kind": kind, "generation": str(path), "n_done": cursor, "next_idx": cursor})


def _space(path, payload_bytes):
    required = payload_bytes + 1024 ** 3
    if shutil.disk_usage(path).free < required:
        raise OSError(f"insufficient free disk for a sealed generation: need {required} bytes including reserve")


def _commit(torch, fit_dir, identity, sums, cursor, diagnostics, kind, *,
            checkpoint_pointer=None, started=None, observed_bound=0.0, fault_hook=None):
    identity_sha = _digest(identity)
    width, sources = identity["d_model"], identity["source_layers"]
    parent = _mkdir(fit_dir / ("checkpoints" if kind == "checkpoint" else "final"))
    _space(parent, len(sources) * width * width * (4 if kind == "checkpoint" else 2)
           * len(identity.get("bank_arms") or [None]))
    path = parent / f"cursor_{cursor:06d}_{uuid.uuid4().hex}"
    path.mkdir()
    _fsync_dir(parent)
    binary = "state.pt" if kind == "checkpoint" else "lens.pt"
    prefix = _digest(identity["prompt_sha256"][:cursor])
    if kind == "checkpoint":
        state = {
            "schema_version": 1, "kind": kind, "fit_identity_sha256": identity_sha,
            "n_done": cursor, "next_idx": cursor, "d_model": width,
            "source_layers": sources, "completed_prefix_sha256": prefix,
            "jacobian_sum": sums,
        }
        with (path / binary).open("xb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        if cursor != 100 or checkpoint_pointer is None:
            raise ValueError("a final lens requires the sealed N=100 FP32 checkpoint")
        means = _map_banks(sums, identity, lambda tensor: (tensor / 100).half())
        _banks(torch, means, identity, torch.float16, "FP16 saved mean")
        state = {"J": means, "n_prompts": 100, "d_model": width, "source_layers": sources}
        if identity.get("bank_arms") is not None:
            state.update(schema_version=1, kind="named_jacobian_lenses", bank_arms=identity["bank_arms"])
        with (path / binary).open("xb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        del state, means
    _fsync_dir(path)
    _fault(fault_hook, f"after_{kind}_state", kind, path, cursor)
    binary_record = _record(path / binary)
    metadata = {
        "schema_version": 1, "kind": kind, "fit_identity_sha256": identity_sha,
        "identity": identity, "n_done": cursor, "next_idx": cursor,
        "completed_prompt_sha256": identity["prompt_sha256"][:cursor],
        "completed_prefix_sha256": prefix, "diagnostics": diagnostics,
        "recorded_utc": _utc_now().isoformat(),
    }
    if kind == "final":
        metadata["fp32_checkpoint"] = checkpoint_pointer
        metadata["saved_dtype"] = "float16"
    _new_json(path / "metadata.json", metadata)
    _fault(fault_hook, f"after_{kind}_metadata", kind, path, cursor)
    seal = {"schema_version": 1, "kind": kind, "fit_identity_sha256": identity_sha,
            "n_done": cursor, "files": {binary: binary_record, "metadata.json": _record(path / "metadata.json")}}
    _new_json(path / "SEAL.json", seal)
    _fault(fault_hook, f"after_{kind}_seal", kind, path, cursor)
    # Includes compute, validation, binary write/fsync/hash, metadata and seal.
    # The tiny pointer write itself is covered by the caller's reserve margin.
    elapsed = time.monotonic() - started if started is not None else 0.0
    pointer = {
        "schema_version": 1, "kind": kind, "generation": path.relative_to(fit_dir).as_posix(),
        "seal": _record(path / "SEAL.json"), "fit_identity_sha256": identity_sha,
        "n_done": cursor, "next_idx": cursor,
        "observed_prompt_bound_seconds": max(observed_bound, elapsed),
    }
    _atomic_json(fit_dir / ("LATEST.json" if kind == "checkpoint" else "COMPLETE.json"), pointer)
    _fault(fault_hook, f"after_{kind}_pointer", kind, path, cursor)
    return pointer, path


def _read_final(torch, fit_dir, identity, checkpoint):
    pointer_path = _no_links(fit_dir / "COMPLETE.json")
    if not pointer_path.exists():
        return None
    if checkpoint is None or checkpoint["cursor"] != 100:
        raise ValueError("completed lens lacks its committed N=100 FP32 checkpoint")
    pointer = _json(pointer_path)
    path, seal, metadata = _pointer_generation(fit_dir, pointer, "final", _digest(identity))
    if pointer["n_done"] != 100:
        raise ValueError("final artifact is not N=100")
    _check_metadata(metadata, identity, 100, "final")
    if metadata.get("fp32_checkpoint") != checkpoint["pointer"] or metadata.get("saved_dtype") != "float16":
        raise ValueError("final artifact does not bind the current FP32 checkpoint")
    state = torch.load(path / "lens.pt", map_location="cpu", weights_only=True, mmap=True)
    keys = {"J", "n_prompts", "source_layers", "d_model"}
    if identity.get("bank_arms") is not None:
        keys |= {"schema_version", "kind", "bank_arms"}
        if (not isinstance(state, dict) or state.get("schema_version") != 1
                or state.get("kind") != "named_jacobian_lenses" or state.get("bank_arms") != identity["bank_arms"]):
            raise ValueError("final artifact does not declare the exact named estimator banks")
    if (not isinstance(state, dict) or set(state) != keys
            or state["n_prompts"] != 100 or state["d_model"] != identity["d_model"]
            or state["source_layers"] != identity["source_layers"]):
        raise ValueError("final JacobianLens payload has invalid geometry or prompt count")
    _banks(torch, state["J"], identity, torch.float16, "FP16 lens")
    for arm, layer, saved, summed in _bank_pairs(state["J"], checkpoint["sum"], identity):
        expected = (summed / 100).half()
        if not torch.equal(saved.view(torch.int16), expected.view(torch.int16)):
            raise ValueError(f"saved lens is not the FP16 N=100 mean at arm {arm}, source {layer}")
    return path / "lens.pt"


def _prune(fit_dir, identity, latest, *, complete=False):
    """Delete only fully validated, sealed generations of this owned fit."""
    parent = _no_links(fit_dir / "checkpoints")
    if not parent.exists():
        return
    # The current pointer is always retained. Keep the newest other directory
    # while active, without rehashing either multi-gigabyte state merely to
    # decide not to delete it. Every directory selected for deletion is then
    # fully verified below. Unknown/unsealed entries never enter this list.
    candidates = []
    for path in parent.iterdir():
        match = _GENERATION.fullmatch(path.name)
        if not match or path.is_symlink() or not path.is_dir() or not (path / "SEAL.json").is_file():
            continue
        candidates.append((int(match.group(1)), path.stat().st_mtime_ns, path))
    keep = {fit_dir / latest["generation"]}
    if not complete:
        other = sorted((row for row in candidates if row[2] not in keep), reverse=True)
        if other:
            keep.add(other[0][2])
    for cursor, _, path in candidates:
        if path in keep:
            continue
        try:
            pointer = {
                "schema_version": 1, "kind": "checkpoint", "generation": path.relative_to(fit_dir).as_posix(),
                "seal": _record(path / "SEAL.json"), "fit_identity_sha256": _digest(identity),
                "n_done": cursor, "next_idx": cursor, "observed_prompt_bound_seconds": 0.0,
            }
            _, _, metadata = _pointer_generation(fit_dir, pointer, "checkpoint", _digest(identity))
            _check_metadata(metadata, identity, cursor, "checkpoint")
        except (ValueError, OSError, TypeError, KeyError):
            # Unknown, corrupt, or incomplete attempts stay available for review.
            continue
        # _pointer_generation already checked the exact file membership and
        # every component for links. Avoid recursive deletion of unknowns.
        for name in ("state.pt", "metadata.json", "SEAL.json"):
            _no_links(path / name).unlink()
        path.rmdir()
    _fsync_dir(parent)


def run_fit(*, prompts, token_lengths, n_valid, fit_id, identity, output_dir,
            compute, d_model=2048, source_layers=tuple(range(191)),
            stop_requested=lambda: False, deadline_utc=None, reserve_seconds=600.0,
            initial_prompt_bound_seconds=600.0, cpu_test=False, fault_hook=None,
            production_profile=None):
    """Fit one frozen 100-paragraph set, or resume its committed prefix.

    ``compute(prompt, index, diagnostics)`` must return CPU FP32 source matrices,
    the actual encoded length, and actual eligible-position count. Its caller
    owns the model. Signals/deadlines are checked between complete paragraphs.
    ``output_dir/fit_XX`` belongs exclusively to this identity. A stop never
    publishes a partial lens. ``cpu_test=True`` only relaxes matrix geometry.
    """
    import torch

    sources = tuple(source_layers)
    bound_identity = _fit_identity(prompts, token_lengths, n_valid, fit_id, identity, d_model, sources, cpu_test,
                                   production_profile)
    deadline = _deadline(deadline_utc)
    for name, value in (("reserve_seconds", reserve_seconds), ("initial_prompt_bound_seconds", initial_prompt_bound_seconds)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")

    with _owned_fit(output_dir, fit_id, bound_identity) as fit_dir:
        checkpoint = _read_checkpoint(torch, fit_dir, bound_identity)
        final = _read_final(torch, fit_dir, bound_identity, checkpoint)
        if final is not None:
            _prune(fit_dir, bound_identity, checkpoint["pointer"], complete=True)
            return {"status": "complete", "n_done": 100, "next_idx": 100,
                    "checkpoint_path": str(checkpoint["generation"] / "state.pt"),
                    "lens_path": str(final), "completed_noop": True}
        cursor = checkpoint["cursor"] if checkpoint else 0
        sums = checkpoint["sum"] if checkpoint else None
        diagnostics = checkpoint["diagnostics"] if checkpoint else []
        observed = checkpoint["pointer"]["observed_prompt_bound_seconds"] if checkpoint else 0.0

        def should_stop(*, finalizing=False):
            if stop_requested():
                return True
            estimate = reserve_seconds
            if not finalizing:
                estimate += observed if observed > 0 else initial_prompt_bound_seconds
            return deadline is not None and (_utc_now().timestamp() + estimate >= deadline.timestamp())

        def stopped():
            return {"status": "stopped", "n_done": cursor, "next_idx": cursor,
                    "checkpoint_path": str(checkpoint["generation"] / "state.pt") if checkpoint else None,
                    "lens_path": None, "completed_noop": False}

        while cursor < 100:
            if should_stop():
                return stopped()
            started = time.monotonic()
            row = {}
            bank, length, count = compute(prompts[cursor], cursor, row)
            if type(length) is not int or length != token_lengths[cursor]:
                raise ValueError(f"actual token length changed for paragraph {cursor}")
            if type(count) is not int or count != n_valid[cursor]:
                raise ValueError(f"actual valid-position count changed for paragraph {cursor}")
            _banks(torch, bank, bound_identity, torch.float32, "per-prompt Jacobian")
            # Do not trust producer diagnostics to establish the sample identity.
            row.update(index=cursor, prompt_sha256=bound_identity["prompt_sha256"][cursor],
                       token_length=length, n_valid=count, compute_seconds=time.monotonic() - started)
            _canonical(row)
            with torch.no_grad():
                if sums is None:
                    sums = _map_banks(bank, bound_identity, lambda tensor: tensor.clone())
                else:
                    for arm, layer, summed, incoming in _bank_pairs(sums, bank, bound_identity):
                        summed.add_(incoming)
            _banks(torch, sums, bound_identity, torch.float32, "accumulated FP32 sum")
            del bank
            diagnostics.append(row)
            cursor += 1
            pointer, generation = _commit(
                torch, fit_dir, bound_identity, sums, cursor, diagnostics, "checkpoint",
                started=started, observed_bound=observed, fault_hook=fault_hook,
            )
            observed = max(pointer["observed_prompt_bound_seconds"], time.monotonic() - started)
            checkpoint = {"sum": sums, "cursor": cursor, "diagnostics": diagnostics,
                          "pointer": pointer, "generation": generation}
            _prune(fit_dir, bound_identity, pointer)
            observed = max(observed, time.monotonic() - started)
            if not cpu_test:
                print(json.dumps({"fit_id": fit_id, "n_done": cursor, "next_idx": cursor,
                                  "elapsed_seconds": time.monotonic() - started,
                                  "checkpoint": str(generation / "state.pt")}), flush=True)
        if should_stop(finalizing=True):
            return stopped()
        _commit(torch, fit_dir, bound_identity, sums, 100, diagnostics, "final",
                checkpoint_pointer=checkpoint["pointer"], observed_bound=observed, fault_hook=fault_hook)
        final = _read_final(torch, fit_dir, bound_identity, checkpoint)
        if final is None:
            raise RuntimeError("final pointer did not become visible after commit")
        _prune(fit_dir, bound_identity, checkpoint["pointer"], complete=True)
        return {"status": "complete", "n_done": 100, "next_idx": 100,
                "checkpoint_path": str(checkpoint["generation"] / "state.pt"),
                "lens_path": str(final), "completed_noop": False}


def _contract(spec_path, ouro_src, selected):
    """Read-only standard-library validation shared by plan and execution."""
    spec_path = _no_links(spec_path)
    spec = _json(spec_path)
    if (not isinstance(spec, dict) or type(spec.get("schema_version")) is not int
            or spec["schema_version"] != 1 or _canonical(spec.get("settings")) != _canonical(SETTINGS)):
        raise ValueError("run specification does not describe the frozen production settings")
    model = spec.get("model", {})
    if model.get("repo_id") != "ByteDance/Ouro-2.6B" or model.get("revision") != REVISION:
        raise ValueError("run specification model identity mismatch")
    model_manifest_path = _no_links(spec_path.parent / model["manifest_path"])
    if _file_hash(model_manifest_path) != model["manifest_sha256"]:
        raise ValueError("model manifest SHA256 mismatch")
    manifest = _json(model_manifest_path)
    if (manifest.get("schema_version") != 1 or manifest.get("repo_id") != model["repo_id"]
            or manifest.get("revision") != model["revision"]):
        raise ValueError("model manifest identity mismatch")
    environment = spec.get("environment", {})
    environment_path = _no_links(spec_path.parent / environment["path"])
    if _file_hash(environment_path) != environment["sha256"]:
        raise ValueError("environment manifest SHA256 mismatch")
    expected_environment = _json(environment_path)
    records = spec.get("source_files")
    if not isinstance(records, list) or not records:
        raise ValueError("the source manifest is required")
    roots = {"jlens": REPO, "ouro_src": Path(ouro_src)}
    actual_sources = []
    paths = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"root", "path", "sha256"}:
            raise ValueError("invalid source manifest record")
        if record["root"] not in roots:
            raise ValueError("unknown source root")
        relative = _relative(record["path"])
        key = (record["root"], relative.as_posix())
        if key in paths:
            raise ValueError("duplicate source manifest path")
        path = _no_links(roots[record["root"]] / relative)
        digest = _file_hash(path)
        if digest != record["sha256"]:
            raise ValueError(f"source SHA256 mismatch: {path}")
        paths[key] = path
        actual_sources.append(dict(record))
    required = {
        ("jlens", Path(__file__).relative_to(REPO).as_posix()),
        ("jlens", (ROUND / "fit_estimators.py").relative_to(REPO).as_posix()),
        *( ("jlens", (ROUND / "optimization" / filename).relative_to(REPO).as_posix())
           for filename in ("optimized_fitting.py", "cuda_graph_candidate.py", "saved_tensor_candidate.py") ),
        ("ouro_src", "ouro_jlens/recurrent.py"), ("ouro_src", "ouro_jlens/evidence.py"),
    }
    required.update(("jlens", p.relative_to(REPO).as_posix()) for p in (REPO / "jlens").rglob("*.py"))
    if not required <= set(paths):
        raise ValueError(f"source manifest omits executable files: {sorted(required - set(paths))}")
    fits = spec.get("fits")
    if (not isinstance(fits, list) or len(fits) != 5
            or [fit.get("fit_id") for fit in fits] != [1, 2, 3, 4, 5]):
        raise ValueError("the specification must contain the five ordered independent fits")
    prepared = []
    seen_prompts = set()
    for fit in fits:
        fit_id = fit["fit_id"]
        if fit.get("seed") != 2026090700 + fit_id or fit.get("n_prompts") != 100:
            raise ValueError("calibration seed or N differs from the frozen plan")
        prompt_path = _no_links(spec_path.parent / fit["prompts_path"])
        if _file_hash(prompt_path) != fit["sha256"]:
            raise ValueError(f"calibration file SHA256 mismatch: fit {fit_id}")
        prompts = _json(prompt_path)
        _fit_identity(prompts, fit["token_lengths"], fit["n_valid"], fit_id,
                      {"plan": True}, 2048, tuple(range(191)), False)
        if seen_prompts.intersection(prompts):
            raise ValueError("the five independent calibration sets must contain 500 distinct paragraphs")
        seen_prompts.update(prompts)
        for length, count in zip(fit["token_lengths"], fit["n_valid"]):
            if length > 128 or count != length - 17:
                raise ValueError("calibration token eligibility differs from skip16/final-token exclusion")
        if fit_id in selected:
            prepared.append({**fit, "prompts": prompts, "prompt_path": str(prompt_path)})
    return {"spec": spec, "spec_path": str(spec_path), "spec_sha256": _file_hash(spec_path),
            "manifest": manifest, "environment": expected_environment, "sources": actual_sources,
            "source_paths": paths, "fits": prepared}


def _snapshot(snapshot, manifest):
    """Hash all actual snapshot inputs; normal HF file symlinks are permitted."""
    snapshot = Path(os.path.abspath(snapshot))
    _no_links(snapshot)
    if not snapshot.is_dir():
        raise ValueError("snapshot must be an existing local directory")
    expected = {}
    for row in manifest["files"]:
        path = _relative(row["path"]).as_posix()
        if path in expected:
            raise ValueError("duplicate file in snapshot manifest")
        expected[path] = {"bytes": row["bytes"], "sha256": row["sha256"]}
    actual = {}
    for directory, directories, filenames in os.walk(snapshot, followlinks=False):
        for name in directories:
            if (Path(directory) / name).is_symlink():
                raise ValueError("snapshot has a linked directory")
        for name in filenames:
            path = Path(directory) / name
            if not path.is_file():
                raise ValueError(f"snapshot has a nonregular or dangling file: {path}")
            relative = path.relative_to(snapshot).as_posix()
            if relative not in expected:
                raise ValueError(f"snapshot has an unmanifested file: {relative}")
            actual[relative] = {"bytes": path.stat().st_size, "sha256": _file_hash(path)}
            if actual[relative] != expected[relative]:
                raise ValueError(f"snapshot file differs from the pinned manifest: {relative}")
    if actual != expected or sum(row["bytes"] for row in actual.values()) != manifest["total_bytes"]:
        raise ValueError("snapshot file membership or total byte count differs from the manifest")
    return actual


def _runtime_cache(output_dir):
    root = _mkdir(Path(output_dir) / "runtime_cache")
    for variable, name in {
        "HF_HOME": "huggingface", "HF_HUB_CACHE": "hub", "HF_MODULES_CACHE": "modules",
        "TORCH_HOME": "torch", "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "TRITON_CACHE_DIR": "triton", "XDG_CACHE_HOME": "xdg", "TMPDIR": "tmp",
    }.items():
        os.environ[variable] = str(_mkdir(root / name))
    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[variable] = "1"
    tempfile.tempdir = os.environ["TMPDIR"]
    sys.dont_write_bytecode = True


def _imports(contract, ouro_src):
    sys.path[:0] = [str(REPO), str(ROUND), str(Path(ouro_src).absolute())]
    modules = [importlib.import_module(name) for name in (
        "fit_estimators", "optimization.optimized_fitting", "optimization.cuda_graph_candidate",
        "optimization.saved_tensor_candidate", "ouro_jlens.recurrent", "ouro_jlens.evidence",
        "jlens", "jlens.fitting", "jlens.hooks", "jlens.hf", "jlens.lens", "jlens.protocol",
    )]
    expected = {str(path.absolute()): record["sha256"]
                for record in contract["sources"]
                for path in [contract["source_paths"][(record["root"], record["path"])]]}
    actual = []
    for module in modules:
        path = _no_links(module.__file__)
        digest = _file_hash(path)
        if str(path) not in expected or digest != expected[str(path)]:
            raise ValueError(f"imported module is not the hashed source: {module.__name__}: {path}")
        actual.append({"module": module.__name__, "path": str(path), "sha256": digest})
    return modules[0], modules[4], actual


def _environment(torch, expected):
    packages = {name: importlib.metadata.version(name) for name in expected["packages"]}
    if packages != expected["packages"] or torch.version.git_version != expected["torch_git"]:
        raise ValueError("runtime package versions or PyTorch commit differ from the validated environment")
    pinned_runtime = {"python_version": platform.python_version(),
                      "cuda_build_version": torch.version.cuda,
                      "cudnn_runtime_integer": torch.backends.cudnn.version()}
    for name, actual in pinned_runtime.items():
        if type(expected.get(name)) is not type(actual) or expected[name] != actual:
            raise ValueError(f"runtime {name} differs from the validated environment")
    # OS/Python build strings describe the actual portable environment. Exact
    # dependency versions/torch commit are required; the complete actual record
    # is also bound to checkpoints, so a resume cannot silently change it.
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "torch_git": torch.version.git_version, "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cpu_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "torch_build_config": torch.__config__.show(), **pinned_runtime}


def _precision(torch):
    def available(getter):
        try:
            return getter()
        except (AttributeError, RuntimeError) as error:
            return {"unavailable": type(error).__name__}

    matmul_attributes = (
        "allow_tf32", "fp32_precision", "allow_bf16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction_split_k", "allow_fp16_reduced_precision_reduction",
        "allow_fp16_reduced_precision_reduction_split_k", "allow_fp16_accumulation",
    )
    sdpa_attributes = (
        "flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled",
        "cudnn_sdp_enabled", "fp16_bf16_reduction_math_sdp_allowed",
    )
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul": {name: available(lambda name=name: getattr(torch.backends.cuda.matmul, name))
                   for name in matmul_attributes},
        "sdpa": {name: available(lambda name=name: getattr(torch.backends.cuda, name)())
                 for name in sdpa_attributes},
        "sdpa_scope": "enabled implementations; not a trace of the selected runtime kernel",
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_fp32_precision": available(lambda: torch.backends.cudnn.fp32_precision),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE",
            "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-spec", type=Path, default=HERE / "run_spec.json")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--ouro-src", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--fit-id", type=int, choices=range(1, 6))
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--stop-at-utc", type=str,
                        help="required for execution; absolute UTC deadline (caller separately ends the Pod lease)")
    parser.add_argument("--reserve-seconds", type=float, default=600.0)
    parser.add_argument("--initial-prompt-bound-seconds", type=float, default=600.0)
    args = parser.parse_args(argv)
    deadline = _deadline(args.stop_at_utc)
    if not args.plan and deadline is None:
        parser.error("--stop-at-utc is required for execution")
    for name in ("reserve_seconds", "initial_prompt_bound_seconds"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    selected = list(range(1, 6)) if args.all else [args.fit_id]
    contract = _contract(args.run_spec, args.ouro_src, selected)
    if args.plan:
        print(json.dumps({
            "status": "plan", "model_loaded": False, "gpu_initialized": False,
            "spec_sha256": contract["spec_sha256"], "settings": SETTINGS,
            "snapshot": str(args.snapshot.absolute()), "output_dir": str(args.output_dir.absolute()),
            "snapshot_content_verification": "required before model load in execution mode",
            "fits": [{key: value for key, value in fit.items() if key != "prompts"} for fit in contract["fits"]],
            "stop_at_utc": deadline.isoformat() if deadline else None,
        }, indent=2, allow_nan=False))
        return 0
    with _output_lock(args.output_dir):
        return _execute(args, contract, deadline)


def _load_ouro_runtime(args, contract):
    """Load and bind the pinned Ouro implementation for fitting or controls."""
    # Full 5.34 GB snapshot hashing happens before torch import/model load.
    snapshot_files = _snapshot(args.snapshot, contract["manifest"])
    _runtime_cache(args.output_dir)
    import torch
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    environment = _environment(torch, contract["environment"])
    if not torch.cuda.is_available():
        raise RuntimeError("the fixed production engine requires an available CUDA GPU")
    estimators, recurrent, imported_sources = _imports(contract, args.ouro_src)
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    gpu = {"index": device_index, "name": properties.name, "total_memory": properties.total_memory,
           "capability": [properties.major, properties.minor], "uuid": str(getattr(properties, "uuid", "unavailable")),
           "multiprocessor_count": properties.multi_processor_count}
    precision = _precision(torch)
    model = recurrent.load_ouro(path=args.snapshot, device=f"cuda:{device_index}", dtype=torch.bfloat16)
    if (model.n_ut, model.n_physical, model.n_layers, model.d_model) != (4, 48, 192, 2048):
        raise ValueError("loaded model geometry differs from the frozen fit")
    if any(module.training for module in model.hf_model.modules()):
        raise ValueError("loaded model must be entirely in evaluation mode")
    if any(parameter.requires_grad or parameter.dtype != torch.bfloat16
           or parameter.device != torch.device("cuda", device_index) for parameter in model.hf_model.parameters()):
        raise ValueError("loaded parameters must be frozen BF16 on the selected CUDA device")
    remote = []
    for cls in (type(model.hf_model), type(model.hf_model.config)):
        path = Path(inspect.getfile(cls))
        digest = _file_hash(path)
        filename = path.name
        if filename not in snapshot_files or digest != snapshot_files[filename]["sha256"]:
            raise ValueError(f"loaded remote implementation differs from the snapshot: {path}")
        remote.append({"class": f"{cls.__module__}.{cls.__name__}", "path": str(path), "sha256": digest})
    identity = {
        "run_spec_sha256": contract["spec_sha256"], "model": contract["spec"]["model"],
        "snapshot_path": str(args.snapshot.absolute()), "snapshot_files": snapshot_files,
        "source_files": contract["sources"], "imported_sources": imported_sources,
        "remote_implementations": remote, "environment": environment, "gpu": gpu,
        "precision": precision, "settings": SETTINGS,
        "attention_implementation": str(getattr(model.hf_model.config, "_attn_implementation", None)),
    }
    return model, estimators, identity


def _execute(args, contract, deadline):
    model, estimators, identity = _load_ouro_runtime(args, contract)
    _bind_run_identity(args.output_dir, identity)
    for fit in contract["fits"]:
        for index, prompt in enumerate(fit["prompts"]):
            length = int(model.encode(prompt, max_length=128).shape[1])
            if length != fit["token_lengths"][index] or length - 17 != fit["n_valid"][index]:
                raise ValueError(f"loaded tokenizer changes calibration fit {fit['fit_id']} paragraph {index}")

    stop = {"requested": False}
    old_handlers = {}
    def signal_stop(signum, frame):
        stop["requested"] = True
    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, signal_stop)
    try:
        for fit in contract["fits"]:
            def compute(prompt, index, diagnostics):
                maps, length, count = estimators.jacobians_for_prompt(
                    model, prompt, SETTINGS["source_layers"], target_layer=191,
                    dim_batch=8, max_seq_len=128, skip_first=16, mode="dense",
                    dense_engine="cuda_graph", compress_saved_tensors=True, diagnostics=diagnostics,
                )
                if set(maps) != {"dense"}:
                    raise ValueError("dense production engine returned unexpected estimator arms")
                return maps["dense"], length, count
            result = run_fit(
                prompts=fit["prompts"], token_lengths=fit["token_lengths"], n_valid=fit["n_valid"],
                fit_id=fit["fit_id"], identity={**identity, "calibration": {key: value for key, value in fit.items() if key != "prompts"}},
                output_dir=args.output_dir, compute=compute, stop_requested=lambda: stop["requested"],
                deadline_utc=deadline, reserve_seconds=args.reserve_seconds,
                initial_prompt_bound_seconds=args.initial_prompt_bound_seconds,
            )
            print(json.dumps({"fit_id": fit["fit_id"], **result}), flush=True)
            gc.collect()
            if result["status"] == "stopped":
                return 0
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
