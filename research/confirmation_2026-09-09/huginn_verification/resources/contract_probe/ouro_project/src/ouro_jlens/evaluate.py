"""Known-intermediate readout across (ut, layer) for Jacobian lens vs logit lens.

One forward per item caches the residual at the readout position (the token
preceding the target) for all 192 virtual locations. For every lens (one per
exit target) we then store per location: the rank of *every* intermediate name
of the item's task (min over single-token forms) — the item's own names give the
hit rate, the other names are matched controls for the position's prior — plus
top-1 token, KL to the actual exit logits, and, for the eventual-exit lens, the
cross-loop tensor from applying J fitted at (ut_i, L) to the state at (ut_j, L).
Output: one .npz + items.json under --out.

  python src/ouro_jlens/evaluate.py --lens 3=artifacts/jlens/lens/exit3.pt \
      [--lens 2=... --lens 1=... --lens 0=...] --out artifacts/jlens/eval/run1
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import jlens
from jlens.hooks import ActivationRecorder
from jlens.vis import _ranks_of

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ouro_jlens.evaldata import Item, load_items  # noqa: E402
from ouro_jlens.evidence import (  # noqa: E402
    SCHEMA_VERSION,
    atomic_savez,
    atomic_write_json,
    aggregate_sha256,
    file_record,
    sha256_json,
)
from ouro_jlens.fit_lens import (  # noqa: E402
    MODEL_FILE_NAMES,
    PROJECT_ROOT_KIND,
    TEST_BUNDLE_ROOT_KIND,
    IntegrityError,
    _context_record,
    _check_declared_root_kind,
    _record_for_path,
    _record_matches,
    _validation_context,
    _validate_lens_shape,
    model_snapshot_identity,
    reject_symlink_path,
    sidecar_path,
    validate_sidecar,
)
from ouro_jlens.recurrent import OURO_REVISION, load_ouro  # noqa: E402

MAX_INTER = 3
MAX_NAMES = 128
GREEDY_STEPS = 4
TASK_NAMES: dict[str, TaskNames] = {}

_ARTIFACT_LOGICAL_PREFIX = "artifacts/jlens/"
_SAFE_ARTIFACT_PARTS = {"", ".", ".."}
_WORKER_DEADLINE_RE = re.compile(r"^[0-9]{1,20}$")


def _worker_deadline_epoch() -> int | None:
    """Validate the optional paid-worker deadline binding from the environment."""

    raw = os.environ.get("JLENS_WORKER_DEADLINE_EPOCH")
    if raw is None:
        return None
    if _WORKER_DEADLINE_RE.fullmatch(raw) is None:
        raise IntegrityError("JLENS_WORKER_DEADLINE_EPOCH is malformed")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:  # pragma: no cover - regex guards this
        raise IntegrityError("JLENS_WORKER_DEADLINE_EPOCH is malformed") from exc
    if value <= 0:
        raise IntegrityError("JLENS_WORKER_DEADLINE_EPOCH must be positive")
    return value


def _check_worker_deadline(provenance: object) -> int | None:
    """Require an evaluation's optional deadline to match the current worker."""

    expected = _worker_deadline_epoch()
    declared = provenance.get("worker_deadline_epoch") if isinstance(provenance, dict) else None
    if declared is None:
        if expected is not None:
            raise IntegrityError("evaluation provenance has no worker deadline binding")
        return None
    if type(declared) is not int or declared <= 0:
        raise IntegrityError("evaluation provenance worker deadline is malformed")
    if expected is None:
        raise IntegrityError("evaluation provenance worker deadline cannot be checked")
    if declared != expected:
        raise IntegrityError("evaluation provenance worker deadline does not match the worker")
    return declared


def _artifact_root(root: str | Path) -> Path:
    """Validate and normalize the root of a synchronized artifact bundle."""

    candidate = Path(os.path.abspath(os.fspath(root)))
    reject_symlink_path(candidate)
    if not candidate.is_dir() or candidate.is_symlink():
        raise IntegrityError(f"artifact root is missing or linked: {candidate}")
    return candidate


def _artifact_suffix(logical: object) -> str | None:
    """Return the exact suffix for one ``artifacts/jlens`` logical path.

    Only canonical, relative POSIX spellings are eligible.  In particular,
    ``.``/``..`` components, absolute paths, and look-alike prefixes are not
    normalized into an artifact path.
    """

    if not isinstance(logical, str) or not logical.startswith(_ARTIFACT_LOGICAL_PREFIX):
        return None
    if logical != logical.replace("\\", "/"):
        return None
    parts = logical.split("/")
    if len(parts) < 3 or parts[:2] != ["artifacts", "jlens"]:
        return None
    if any(part in _SAFE_ARTIFACT_PARTS for part in parts):
        return None
    return "/".join(parts[2:])


def artifact_path(logical: object, artifact_root: str | Path, *, label: str = "artifact") -> Path:
    """Resolve one declared result-artifact path under *artifact_root*.

    This is intentionally narrow: callers must opt in for known result/lens
    records.  Source, model, and stimulus records continue through their
    normal trusted-root resolver.
    """

    suffix = _artifact_suffix(logical)
    if suffix is None:
        raise IntegrityError(f"{label} is not a canonical artifacts/jlens logical path")
    root = _artifact_root(artifact_root)
    candidate = Path(os.path.abspath(root / Path(suffix)))
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise IntegrityError(f"{label} escapes artifact root: {logical}") from exc
    reject_symlink_path(candidate)
    if not candidate.is_file() or candidate.is_symlink():
        raise IntegrityError(f"{label} is missing or linked: {candidate}")
    return candidate


def _record_matches_artifact(
    record: object,
    *,
    root: Path,
    label: str,
    artifact_root: str | Path | None = None,
    expected: Path | None = None,
) -> Path:
    """Resolve/hash-check a lens record, optionally against a synchronized root."""

    logical = record.get("path") if isinstance(record, dict) else None
    if artifact_root is None or not isinstance(logical, str) or not logical.startswith(_ARTIFACT_LOGICAL_PREFIX):
        candidate = _record_matches(record, root=root, label=label)
    else:
        record_kind = record.get("root_kind") if isinstance(record, dict) else None
        if record_kind not in {None, PROJECT_ROOT_KIND, TEST_BUNDLE_ROOT_KIND}:
            raise IntegrityError(f"{label} has an unsupported logical root kind: {record_kind!r}")
        candidate = artifact_path(record.get("path"), artifact_root, label=label)
        try:
            observed = file_record(candidate)
        except (OSError, ValueError) as exc:
            raise IntegrityError(f"{label} bytes are unavailable: {candidate}") from exc
        if observed["size"] != record.get("size") or observed["sha256"] != record.get("sha256"):
            raise IntegrityError(f"{label} hash/size mismatch: {record.get('path')}")
    if expected is not None and candidate != _lexical_path(expected):
        raise IntegrityError(f"{label} path does not match the synchronized artifact")
    return candidate


def _record_for_actual(record: dict[str, object], path: Path) -> dict[str, object]:
    """Preserve a declared logical path while refreshing only its byte fields."""

    actual = file_record(path)
    result: dict[str, object] = {
        "path": record["path"],
        "size": actual["size"],
        "sha256": actual["sha256"],
    }
    if "root_kind" in record:
        result["root_kind"] = record["root_kind"]
    return result


def _relocate_record(record: object, artifact_root: Path, *, label: str) -> object:
    """Convert only a result-artifact record to an external in-memory record."""

    logical = record.get("path") if isinstance(record, dict) else None
    if not isinstance(logical, str) or not logical.startswith(_ARTIFACT_LOGICAL_PREFIX):
        return record
    if _artifact_suffix(logical) is None:
        raise IntegrityError(f"{label} is not a canonical artifacts/jlens logical path")
    record_kind = record.get("root_kind") if isinstance(record, dict) else None
    if record_kind not in {None, PROJECT_ROOT_KIND, TEST_BUNDLE_ROOT_KIND}:
        raise IntegrityError(f"{label} has an unsupported logical root kind: {record_kind!r}")
    path = artifact_path(record.get("path"), artifact_root, label=label)
    observed = file_record(path)
    if observed["size"] != record.get("size") or observed["sha256"] != record.get("sha256"):
        raise IntegrityError(f"{label} hash/size mismatch: {record.get('path')}")
    return {
        "path": str(path),
        "root_kind": "external",
        "size": observed["size"],
        "sha256": observed["sha256"],
    }


def _relocated_sidecar_metadata(metadata: dict[str, object], path: Path, artifact_root: Path) -> dict[str, object]:
    """Adapt fit-lens output/shard records for one relocated lens in memory."""

    result = copy.deepcopy(metadata)
    original_output = result.get("output")
    result["output"] = _relocate_record(original_output, artifact_root, label=f"output for {path}")
    if isinstance(original_output, dict) and isinstance(result.get("output"), dict):
        if original_output.get("sha256") != metadata.get("output_sha256"):
            raise IntegrityError(f"output digest fields disagree for {path}")
        result["output_sha256"] = result["output"]["sha256"]

    original_records = result.get("shard_records")
    original_shards = result.get("shards")
    if isinstance(original_records, list) or isinstance(original_shards, list):
        if not isinstance(original_records, list) or not isinstance(original_shards, list):
            raise IntegrityError(f"merged sidecar shard seal is incomplete: {path}")
        if [record.get("path") for record in original_records if isinstance(record, dict)] != original_shards:
            raise IntegrityError(f"merged sidecar shard paths disagree: {path}")
        if metadata.get("shard_records_sha256") != sha256_json(original_records):
            raise IntegrityError(f"merged sidecar shard seal digest is invalid: {path}")
        # ``validate_evaluation_provenance`` can be reached from a legacy
        # caller already wrapped by ``relocated_evaluation_validation``.  In
        # that case the outer reader has already converted the sealed records
        # to absolute external paths.  Keep that conversion idempotent rather
        # than trying to interpret the absolute paths as logical producer
        # names a second time.
        if all(
            isinstance(record, dict)
            and isinstance(record.get("path"), str)
            and Path(record["path"]).is_absolute()
            and record.get("root_kind") == "external"
            for record in original_records
        ):
            for index, record in enumerate(original_records):
                binary_path = Path(record["path"])
                try:
                    binary_path.relative_to(artifact_root)
                except ValueError as exc:
                    raise IntegrityError(
                        f"shard[{index}] is outside the synchronized artifact root: {binary_path}"
                    ) from exc
                reject_symlink_path(binary_path)
                if not binary_path.is_file() or binary_path.is_symlink():
                    raise IntegrityError(f"shard[{index}] is missing or linked: {binary_path}")
                observed = file_record(binary_path)
                if (
                    observed["size"] != record.get("binary_size")
                    or observed["sha256"] != record.get("binary_sha256")
                    or observed["sha256"] != record.get("output_sha256")
                ):
                    raise IntegrityError(f"shard[{index}] hash/size mismatch: {binary_path}")
                sidecar_value = record.get("sidecar_path")
                sidecar_kind = record.get("sidecar_root_kind")
                if (
                    not isinstance(sidecar_value, str)
                    or not Path(sidecar_value).is_absolute()
                    or sidecar_kind != "external"
                ):
                    raise IntegrityError(f"shard sidecar[{index}] is not a relocated external record")
                sidecar_path_value = Path(sidecar_value)
                try:
                    sidecar_path_value.relative_to(artifact_root)
                except ValueError as exc:
                    raise IntegrityError(
                        f"shard sidecar[{index}] is outside the synchronized artifact root: {sidecar_path_value}"
                    ) from exc
                reject_symlink_path(sidecar_path_value)
                if not sidecar_path_value.is_file() or sidecar_path_value.is_symlink():
                    raise IntegrityError(
                        f"shard sidecar[{index}] is missing or linked: {sidecar_path_value}"
                    )
                sidecar_observed = file_record(sidecar_path_value)
                if (
                    sidecar_observed["size"] != record.get("sidecar_size", sidecar_observed["size"])
                    or sidecar_observed["sha256"] != record.get("sidecar_sha256")
                ):
                    raise IntegrityError(f"shard sidecar[{index}] hash/size mismatch: {sidecar_path_value}")
            return result
        relocated_records: list[object] = []
        relocated_shards: list[str] = []
        for index, record in enumerate(original_records):
            if not isinstance(record, dict):
                raise IntegrityError(f"merged sidecar shard record is malformed: {path}")
            changed = dict(record)
            binary_logical = record.get("path")
            if not isinstance(binary_logical, str) or not binary_logical.startswith(_ARTIFACT_LOGICAL_PREFIX):
                raise IntegrityError(f"shard[{index}] path is not a synchronized artifact: {binary_logical!r}")
            binary_kind = record.get("root_kind")
            if binary_kind not in {None, PROJECT_ROOT_KIND, TEST_BUNDLE_ROOT_KIND}:
                raise IntegrityError(f"shard[{index}] has an unsupported logical root kind: {binary_kind!r}")
            binary_path = artifact_path(binary_logical, artifact_root, label=f"shard[{index}] for {path}")
            binary_observed = file_record(binary_path)
            if (
                binary_observed["size"] != record.get("binary_size")
                or binary_observed["sha256"] != record.get("binary_sha256")
                or binary_observed["sha256"] != record.get("output_sha256")
            ):
                raise IntegrityError(f"shard[{index}] hash/size mismatch: {binary_logical}")
            changed.update({
                "path": str(binary_path),
                "root_kind": "external",
                "binary_size": binary_observed["size"],
                "binary_sha256": binary_observed["sha256"],
                "output_sha256": binary_observed["sha256"],
            })
            sidecar = record.get("sidecar_path")
            sidecar_kind = record.get("sidecar_root_kind")
            if sidecar_kind not in {None, PROJECT_ROOT_KIND, TEST_BUNDLE_ROOT_KIND, "external"}:
                raise IntegrityError(f"shard sidecar[{index}] has an unsupported root kind: {sidecar_kind!r}")
            if _artifact_suffix(sidecar) is not None:
                if sidecar_kind not in {None, PROJECT_ROOT_KIND, TEST_BUNDLE_ROOT_KIND}:
                    raise IntegrityError(f"shard sidecar[{index}] has an invalid logical root kind: {sidecar_kind!r}")
                sidecar_path_value = artifact_path(sidecar, artifact_root, label=f"shard sidecar[{index}] for {path}")
                sidecar_observed = file_record(sidecar_path_value)
                if (
                    sidecar_observed["size"] != record.get("sidecar_size", sidecar_observed["size"])
                    or sidecar_observed["sha256"] != record.get("sidecar_sha256")
                ):
                    raise IntegrityError(f"shard sidecar[{index}] hash/size mismatch: {sidecar}")
                changed["sidecar_path"] = str(sidecar_path_value)
                changed["sidecar_root_kind"] = "external"
                changed["sidecar_sha256"] = sidecar_observed["sha256"]
            relocated_records.append(changed)
            relocated_shards.append(str(binary_path))
        result["shard_records"] = relocated_records
        result["shards"] = relocated_shards
        result["shard_records_sha256"] = sha256_json(relocated_records)
    return result


@contextmanager
def relocated_lens_validation(artifact_root: str | Path):
    """Scope fit-lens relocation adaptation and restore all aliases reliably."""

    root = _artifact_root(artifact_root)
    import ouro_jlens.fit_lens as fit_lens_module

    original_reader = fit_lens_module._read_json
    original_validator = fit_lens_module.validate_sidecar
    try:
        import ouro_jlens.transport_report as transport_module
    except ImportError:  # pragma: no cover - transport is part of this package
        transport_module = None
    original_transport_validator = (
        getattr(transport_module, "validate_sidecar", None) if transport_module is not None else None
    )

    def read_json(path: Path) -> dict[str, object]:
        metadata = original_reader(path)
        absolute = Path(os.path.abspath(path))
        try:
            absolute.relative_to(root)
        except ValueError:
            return metadata
        return _relocated_sidecar_metadata(metadata, absolute, root)

    def validate(path: str | Path, *, kind: str = "fit", validation_root: str | Path | None = None):
        return original_validator(path, kind=kind, validation_root=validation_root)

    fit_lens_module._read_json = read_json
    fit_lens_module.validate_sidecar = validate
    if transport_module is not None:
        transport_module.validate_sidecar = validate
    try:
        yield root
    finally:
        fit_lens_module._read_json = original_reader
        fit_lens_module.validate_sidecar = original_validator
        if transport_module is not None:
            transport_module.validate_sidecar = original_transport_validator


def validate_lens_sidecar(
    path: str | Path,
    *,
    kind: str = "fit",
    validation_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> dict[str, object]:
    """Validate one lens sidecar, mapping only its synchronized result records."""

    if artifact_root is None:
        return validate_sidecar(path, kind=kind, validation_root=validation_root)
    with relocated_lens_validation(artifact_root):
        return validate_sidecar(path, kind=kind, validation_root=validation_root)


def validate_merged_lens_shards(
    merged_path: str | Path,
    shards: Iterable[str | Path],
    *,
    validation_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> dict[str, object]:
    """Validate a merged lens and its sealed shards under an optional bundle root."""

    import ouro_jlens.fit_lens as fit_lens_module

    if artifact_root is None:
        return fit_lens_module.validate_merged_shards(
            merged_path,
            shards,
            validation_root=validation_root,
        )
    with relocated_lens_validation(artifact_root):
        return fit_lens_module.validate_merged_shards(
            merged_path,
            shards,
            validation_root=validation_root,
        )


@contextmanager
def relocated_evaluation_validation(artifact_root: str | Path):
    """Make legacy evaluator callers use one scoped synchronized artifact root."""

    root = _artifact_root(artifact_root)
    original_validator = validate_evaluation_provenance

    def validate(eval_dir: str | Path, **kwargs: object) -> dict[str, object]:
        kwargs.setdefault("artifact_root", root)
        return original_validator(eval_dir, **kwargs)

    with relocated_lens_validation(root):
        globals()["validate_evaluation_provenance"] = validate
        try:
            yield root
        finally:
            globals()["validate_evaluation_provenance"] = original_validator


@torch.no_grad()
def greedy(m, ids: torch.Tensor, steps: int) -> str:
    out = ids
    for _ in range(steps):
        logits = m.hf_model(out, use_cache=False, exit_at_step=m.n_ut - 1).logits[0, -1]
        out = torch.cat([out, logits.argmax().view(1, 1)], dim=1)
    return m.tokenizer.decode(out[0, ids.shape[1]:])


def is_correct(continuation: str, target: str) -> bool:
    """Prefix match that must end at a token boundary, so "11" does not count as "1"."""
    c, t = continuation.strip().strip('"').lower(), target.strip().lower()
    if not c.startswith(t):
        return False
    rest = c[len(t):]
    return rest == "" or not rest[0].isalnum()


@torch.no_grad()
def cache_states(m, items: list[Item], position: int = -1) -> tuple[torch.Tensor, list[str]]:
    """Residuals at the readout position for all virtual locations: [n_items, 192, d]."""
    H = torch.empty(len(items), m.n_layers, m.d_model, dtype=torch.float32)
    continuations = []
    for i, item in enumerate(items):
        ids = torch.tensor([item.token_ids], device=m.input_device)
        with ActivationRecorder(m.layers, at=range(m.n_layers)) as rec:
            m.forward(ids)
        H[i] = torch.stack([rec.activations[v][0, position] for v in range(m.n_layers)]).float().cpu()
        continuations.append(greedy(m, ids, GREEDY_STEPS))
    return H, continuations


def kl(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) along the last dim, in nats."""
    lp, lq = F.log_softmax(logits_p.float(), -1), F.log_softmax(logits_q.float(), -1)
    return (lp.exp() * (lp - lq)).sum(-1)


class TaskNames:
    """All scorable intermediate names of a task with their single-token forms."""

    def __init__(self, items: list[Item], task: str) -> None:
        self.names: list[str] = []
        self.forms: dict[str, list[int]] = {}
        for it in items:
            if it.task != task:
                continue
            for name in it.scorable:
                if name not in self.forms:
                    self.names.append(name)
                    self.forms[name] = it.intermediate_tokens[name]
        flat = [t for n in self.names for t in self.forms[n]]
        self.flat_ids = torch.tensor(flat)
        self.slices = np.cumsum([0, *(len(self.forms[n]) for n in self.names)])

    def ranks(self, logits: torch.Tensor) -> np.ndarray:
        """[n_names, N] min-over-forms rank of every name of the task."""
        r = _ranks_of(logits, self.flat_ids.to(logits.device)).cpu().numpy()  # [N, n_forms]
        return np.stack([r[:, a:b].min(1) for a, b in zip(self.slices[:-1], self.slices[1:])])

    def own_index(self, item: Item) -> list[int]:
        return [self.names.index(n) if n in self.forms else -1 for n in item.intermediates[:MAX_INTER]] + [-1] * (MAX_INTER - len(item.intermediates[:MAX_INTER]))


def pad_names(rows: np.ndarray, n: int) -> np.ndarray:
    out = np.full((n, *rows.shape[1:]), -1, np.int32)
    out[: len(rows)] = rows
    return out


@torch.no_grad()
def readout_arrays(m, items, H, J, exit_logits: torch.Tensor, target_ut: int, eventual=None) -> tuple[dict, list]:
    """Per-location readout for one lens. J: [192, d, d] on GPU (identity at the target), or None
    for the vanilla logit lens. `eventual` holds the eventual-exit lens logits per item (fp16 CPU)
    so the local-vs-eventual monitor gap is measured directly."""
    n, V = len(items), m.n_layers
    res = {
        "rank": np.full((n, MAX_INTER, V), -1, np.int32),
        "allrank": np.full((n, MAX_NAMES, V), -1, np.int32),
        "top1": np.zeros((n, V), np.int64),
        "kl_to_final": np.zeros((n, V), np.float32),
        "kl_to_local": np.zeros((n, V), np.float32),
        "rank_of_final_top1": np.zeros((n, V), np.int32),
        "rank_of_local_top1": np.zeros((n, V), np.int32),
    }
    if eventual is not None:
        res["kl_to_eventual_readout"] = np.zeros((n, V), np.float32)
    kept = []
    for i, item in enumerate(items):
        h = H[i].to(m.input_device)
        transported = h if J is None else torch.einsum("vde,ve->vd", J, h)
        logits = m.unembed(transported).float()  # [192, vocab]
        final, local = exit_logits[i, m.n_ut - 1], exit_logits[i, target_ut]
        tn = TASK_NAMES[item.task]
        allrank = tn.ranks(logits)
        res["allrank"][i] = pad_names(allrank, MAX_NAMES)
        for k, j in enumerate(tn.own_index(item)):
            if j >= 0:
                res["rank"][i, k] = allrank[j]
        res["top1"][i] = logits.argmax(-1).cpu().numpy()
        res["kl_to_final"][i] = kl(logits, final).cpu().numpy()
        res["kl_to_local"][i] = kl(logits, local).cpu().numpy()
        res["rank_of_final_top1"][i] = _ranks_of(logits, final.argmax().view(1))[:, 0].cpu().numpy()
        res["rank_of_local_top1"][i] = _ranks_of(logits, local.argmax().view(1))[:, 0].cpu().numpy()
        if eventual is not None:
            res["kl_to_eventual_readout"][i] = kl(logits, eventual[i].to(logits.device)).cpu().numpy()
        kept.append(logits.half().cpu())
    return res, kept


@torch.no_grad()
def cross_loop(m, items, H, J: torch.Tensor) -> np.ndarray:
    """allrank[item, name, fit_ut, applied_ut, layer]: J at (fit_ut, L) applied to state (applied_ut, L)."""
    n, U, L = len(items), m.n_ut, m.n_physical
    Jr = J.view(U, L, m.d_model, m.d_model)
    out = np.full((n, MAX_NAMES, U, U, L), -1, np.int32)
    for i, item in enumerate(items):
        h = H[i].to(J.device).view(U, L, m.d_model)
        # per layer: J at (i, L) applied to the state at (j, L) -> [i, d, j]; stacked as [L, i, d, j]
        transported = torch.stack([Jr[:, l] @ h[:, l].T for l in range(L)])
        transported = transported.permute(1, 3, 0, 2).reshape(U * U * L, m.d_model)  # order (i, j, L)
        logits = m.unembed(transported).float()
        out[i] = pad_names(TASK_NAMES[item.task].ranks(logits), MAX_NAMES).reshape(MAX_NAMES, U, U, L)
    return out


def stacked_jacobians(m, lens: jlens.JacobianLens, target: int) -> torch.Tensor:
    """Build a complete virtual-layer map only from an exact contiguous lens."""

    if not isinstance(target, int) or target <= 0 or target >= m.n_layers:
        raise IntegrityError(f"invalid virtual target layer {target!r}")
    expected = list(range(target))
    if list(lens.source_layers) != expected:
        raise IntegrityError(
            f"lens source_layers must be exactly contiguous 0..{target - 1}; "
            f"got {lens.source_layers}"
        )
    if int(lens.d_model) != int(m.d_model):
        raise IntegrityError(
            f"lens d_model={lens.d_model} is incompatible with model d_model={m.d_model}"
        )
    J = torch.eye(m.d_model).repeat(m.n_layers, 1, 1)
    for v in lens.source_layers:
        matrix = lens.jacobians[v]
        if tuple(matrix.shape) != (m.d_model, m.d_model):
            raise IntegrityError(f"invalid Jacobian shape at virtual layer {v}: {tuple(matrix.shape)}")
        J[v] = matrix
    return J.to(m.input_device)


def _evaluation_source_manifest(
    *,
    validation_root: Path | None = None,
    validation_root_kind: str | None = None,
) -> dict[str, object]:
    if validation_root is None:
        validation_root, validation_root_kind = _validation_context(None, PROJECT_ROOT_KIND)
    elif validation_root_kind is None:
        validation_root, validation_root_kind = _validation_context(validation_root)
    paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("evaldata.py").resolve(),
        Path(__file__).with_name("evidence.py").resolve(),
        Path(__file__).with_name("fit_lens.py").resolve(),
        Path(__file__).with_name("recurrent.py").resolve(),
    ]
    records = [
        _record_for_path(
            path,
            root=validation_root,
            root_kind=validation_root_kind,
        )
        for path in paths
        if path.is_file()
    ]
    return {
        "files": records,
        "sha256": aggregate_sha256({record["path"]: record["sha256"] for record in records}),
    }


def load_lens_metadata(
    path: str | Path,
    *,
    validation_root: str | Path | None = None,
) -> dict[str, object]:
    """Load a lens only after verifying its sidecar and output hash."""
    metadata_path = sidecar_path(path)
    reject_symlink_path(path)
    reject_symlink_path(metadata_path)
    try:
        preview = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot read lens sidecar {metadata_path}: {exc}") from exc
    kind = preview.get("kind") if isinstance(preview, dict) else None
    if kind not in {"fit", "merged"}:
        raise IntegrityError(f"unsupported lens sidecar kind {kind!r}: {metadata_path}")
    if validation_root is None:
        return validate_sidecar(path, kind=kind)
    return validate_sidecar(path, kind=kind, validation_root=validation_root)


def _prompt_identity_fields(metadata: dict[str, object]) -> tuple[object, int, int, int, object, object, object]:
    prompt_slice = metadata.get("prompt_slice")
    if not isinstance(prompt_slice, dict):
        raise IntegrityError("lens metadata is missing prompt_slice")
    fields = (
        metadata.get("prompt_file_sha256"),
        prompt_slice.get("start"),
        prompt_slice.get("end"),
        prompt_slice.get("count"),
        prompt_slice.get("sha256"),
        metadata.get("prompt_provenance_sha256"),
        metadata.get("prompt_source_revision"),
    )
    if not isinstance(fields[1], int) or isinstance(fields[1], bool):
        raise IntegrityError("lens metadata has an invalid prompt start")
    if not isinstance(fields[2], int) or isinstance(fields[2], bool):
        raise IntegrityError("lens metadata has an invalid prompt end")
    if not isinstance(fields[3], int) or isinstance(fields[3], bool):
        raise IntegrityError("lens metadata has an invalid prompt count")
    return fields


def validate_lens_metadata(
    metadata: dict[str, object],
    *,
    target_ut: int,
    model: object,
    baseline: dict[str, object] | None = None,
    prompt_policy: str = "identical",
    validation_root: str | Path | None = None,
) -> None:
    """Check target, model bytes/shape, and an explicit cross-lens policy."""

    if prompt_policy not in {"identical", "nested"}:
        raise IntegrityError(f"unsupported cross-lens prompt policy: {prompt_policy!r}")

    target_virtual = int(model.exit_index(target_ut))
    if metadata.get("target_ut") != target_ut or metadata.get("target_virtual") != target_virtual:
        raise IntegrityError(
            f"lens metadata target mismatch: expected ut={target_ut}, virtual={target_virtual}"
        )
    if metadata.get("source_layers") != list(range(target_virtual)):
        raise IntegrityError("lens metadata source_layers are not exact contiguous sources")
    model_record = metadata.get("model")
    if not isinstance(model_record, dict):
        raise IntegrityError("lens metadata is missing model identity")
    current_revision = str(getattr(model, "model_revision", OURO_REVISION))
    if metadata.get("model_revision") != current_revision:
        raise IntegrityError(
            f"lens model revision {metadata.get('model_revision')!r} != loaded {current_revision!r}"
        )
    expected_shape = {
        "n_physical": int(model.n_physical),
        "n_ut": int(model.n_ut),
        "n_layers": int(model.n_layers),
        "d_model": int(model.d_model),
    }
    actual_shape = {key: model_record.get(key) for key in expected_shape}
    if actual_shape != expected_shape:
        raise IntegrityError(f"lens metadata model shape mismatch: {actual_shape} != {expected_shape}")
    if model_record.get("bytes_status") != "HASH_BOUND":
        raise IntegrityError(
            "new scientific evaluation requires HASH_BOUND model identity; "
            f"got {model_record.get('bytes_status')!r}"
        )
    current_model = model_snapshot_identity(
        model,
        validation_root=validation_root,
        validation_root_kind=(
            metadata.get("validation", {}).get("kind")
            if isinstance(metadata.get("validation"), dict)
            else None
        ),
    )
    if current_model.get("bytes_status") != "HASH_BOUND":
        raise IntegrityError("loaded model has no complete hash-bound snapshot")
    if metadata.get("model_snapshot_sha256") != current_model.get("aggregate_sha256"):
        raise IntegrityError("lens model snapshot digest does not match loaded model")
    if model_record.get("aggregate_sha256") != current_model.get("aggregate_sha256"):
        raise IntegrityError("lens model file manifest does not match loaded model")
    if model_record.get("files") != current_model.get("files"):
        raise IntegrityError("lens model file records do not match loaded model")
    identity_fields = (
        "model_revision",
        "model_snapshot_sha256",
        "jlens_commit",
        "prompt_file_sha256",
        "prompt_file_count",
        "prompt_provenance_sha256",
        "prompt_source_revision",
        "source_sha256",
        "generator_sha256",
        "max_seq_len",
        "skip_first",
    )
    if baseline is not None:
        for field in identity_fields:
            if metadata.get(field) != baseline.get(field):
                raise IntegrityError(f"lens metadata {field} mismatch between inputs")
        current_prompt = _prompt_identity_fields(metadata)
        baseline_prompt = _prompt_identity_fields(baseline)
        if current_prompt[0] != baseline_prompt[0]:
            raise IntegrityError("lens metadata prompt files differ between inputs")
        if prompt_policy == "identical":
            if current_prompt[1:] != baseline_prompt[1:]:
                raise IntegrityError("lens metadata prompt slices differ under identical policy")
        else:
            if current_prompt[1] != baseline_prompt[1]:
                raise IntegrityError("nested prompt policy requires a common prompt start")
            # Equal-start intervals are prefixes iff their ends are ordered;
            # rejecting an unrelated interval catches disjoint populations.
            if current_prompt[2] < current_prompt[1] or baseline_prompt[2] < baseline_prompt[1]:
                raise IntegrityError("nested prompt policy received invalid ranges")


def _evaluation_inputs(
    tasks: list[str],
    *,
    validation_root: Path | None = None,
    validation_root_kind: str | None = None,
) -> list[dict[str, object]]:
    """Record the exact evaluation stimulus bytes consumed by ``load_items``."""

    from ouro_jlens.evaldata import JLENS_DATA

    if validation_root is None:
        validation_root, validation_root_kind = _validation_context(None, PROJECT_ROOT_KIND)
    elif validation_root_kind is None:
        validation_root, validation_root_kind = _validation_context(validation_root)
    records = []
    for task in tasks:
        path = JLENS_DATA / f"lens-eval-{task}.json"
        records.append(
            _record_for_path(
                path,
                root=validation_root,
                root_kind=validation_root_kind,
            )
        )
    return records


def _parse_lens_specs(values: list[str]) -> list[tuple[int, Path]]:
    specs: list[tuple[int, Path]] = []
    seen: set[int] = set()
    for value in values:
        try:
            target_text, path_text = value.split("=", 1)
            target_ut = int(target_text)
        except (TypeError, ValueError) as exc:
            raise IntegrityError(f"lens must be specified as target_ut=path, got {value!r}") from exc
        if target_ut in seen:
            raise IntegrityError(f"duplicate lens target_ut={target_ut}")
        seen.add(target_ut)
        specs.append((target_ut, Path(path_text)))
    return sorted(specs, reverse=True)


def _records_equal(
    actual: object,
    expected: object,
    label: str,
) -> None:
    if actual != expected:
        raise IntegrityError(f"{label} provenance does not match current bytes")


def _output_record(path: Path, output_root: Path) -> dict[str, object]:
    return _record_for_path(
        path,
        root=output_root,
        root_kind=TEST_BUNDLE_ROOT_KIND,
    )


def _validate_provenance_records(
    records: object,
    *,
    root: Path,
    label: str,
    root_kind: str | None = None,
    allow_symlink: bool = False,
) -> list[Path]:
    if not isinstance(records, list) or not records:
        raise IntegrityError(f"{label} provenance is missing")
    resolved: list[Path] = []
    for index, record in enumerate(records):
        if root_kind is not None:
            _check_declared_root_kind(record, root_kind, f"{label}[{index}]")
        resolved.append(
            _record_matches(
                record,
                root=root,
                label=f"{label}[{index}]",
                allow_symlink=allow_symlink,
            )
        )
    return resolved


def validate_evaluation_provenance(
    eval_dir: str | Path,
    *,
    validation_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
    lens_paths: list[tuple[int, str | Path]] | None = None,
    model: object | None = None,
    expected_prompt_policy: str | None = None,
) -> dict[str, object]:
    """Validate a complete evaluation against current lenses, model, inputs, and code.

    Shell wrappers call this before deciding that an evaluation can be
    resumed.  It intentionally performs no model forward; a changed model
    snapshot is detected by rehashing the sealed snapshot records, while a
    fresh evaluation calls :func:`load_ouro` and repeats the identity check.
    """

    output_root = reject_symlink_path(eval_dir)
    normalized_artifact_root = _artifact_root(artifact_root) if artifact_root is not None else None
    if not output_root.is_dir() or output_root.is_symlink():
        raise IntegrityError(f"evaluation output root is missing or linked: {output_root}")
    provenance_path = output_root / "provenance.json"
    reject_symlink_path(provenance_path)
    if not provenance_path.is_file() or provenance_path.is_symlink():
        raise IntegrityError(f"evaluation provenance is missing or linked: {provenance_path}")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"invalid evaluation provenance {provenance_path}: {exc}") from exc
    if not isinstance(provenance, dict) or provenance.get("schema_version") != SCHEMA_VERSION:
        raise IntegrityError(f"unsupported evaluation provenance schema: {provenance_path}")
    _check_worker_deadline(provenance)

    declaration = provenance.get("validation")
    declared_kind = None
    if isinstance(declaration, dict):
        if declaration.get("path") != ".":
            raise IntegrityError("evaluation provenance has a non-relocatable validation root")
        declared_kind = declaration.get("kind") or declaration.get("root_kind")
    if declared_kind is None:
        raise IntegrityError("evaluation provenance has no validation root kind")
    if validation_root is None:
        if declared_kind == PROJECT_ROOT_KIND:
            input_root, input_root_kind = _validation_context(None, PROJECT_ROOT_KIND)
        elif declared_kind == TEST_BUNDLE_ROOT_KIND:
            input_root, input_root_kind = _validation_context(output_root.parent, TEST_BUNDLE_ROOT_KIND)
        else:
            raise IntegrityError(f"unsupported evaluation validation root kind: {declared_kind!r}")
    else:
        input_root, input_root_kind = _validation_context(validation_root, declared_kind)

    config = provenance.get("config")
    if not isinstance(config, dict):
        raise IntegrityError("evaluation provenance has no config")
    tasks = config.get("tasks")
    if not isinstance(tasks, list) or not tasks or not all(isinstance(task, str) for task in tasks):
        raise IntegrityError("evaluation provenance has invalid tasks")
    prompt_policy = config.get("prompt_policy", provenance.get("prompt_policy"))
    if prompt_policy not in {"identical", "nested"}:
        raise IntegrityError("evaluation provenance has no explicit prompt policy")
    if expected_prompt_policy is not None and prompt_policy != expected_prompt_policy:
        raise IntegrityError(
            "evaluation provenance prompt policy does not match the current command"
        )

    expected_source = _evaluation_source_manifest(
        validation_root=input_root,
        validation_root_kind=input_root_kind,
    )
    _records_equal(provenance.get("source_files"), expected_source["files"], "evaluation source")
    _records_equal(provenance.get("source_sha256"), expected_source["sha256"], "evaluation source")

    expected_stimuli = _evaluation_inputs(
        tasks,
        validation_root=input_root,
        validation_root_kind=input_root_kind,
    )
    inputs = provenance.get("inputs")
    if not isinstance(inputs, dict):
        raise IntegrityError("evaluation provenance has no inputs")
    _records_equal(inputs.get("evaluation_files"), expected_stimuli, "evaluation stimuli")
    _records_equal(
        inputs.get("evaluation_input_sha256"),
        sha256_json(expected_stimuli),
        "evaluation stimuli",
    )
    _validate_provenance_records(
        provenance.get("source_files"),
        root=input_root,
        label="evaluation source",
        root_kind=input_root_kind,
    )
    _validate_provenance_records(
        inputs.get("evaluation_files"),
        root=input_root,
        label="evaluation stimulus",
        root_kind=input_root_kind,
    )

    model_record = provenance.get("model")
    if not isinstance(model_record, dict) or model_record.get("bytes_status") != "HASH_BOUND":
        raise IntegrityError("evaluation model is not HASH_BOUND")
    shape_fields = ("n_physical", "n_ut", "n_layers", "d_model")
    if (
        not isinstance(model_record.get("revision"), str)
        or not all(
            isinstance(model_record.get(field), int)
            and not isinstance(model_record.get(field), bool)
            and model_record.get(field) > 0
            for field in shape_fields
        )
    ):
        raise IntegrityError("evaluation model identity has invalid revision or shape")
    model_files = model_record.get("files")
    _validate_provenance_records(
        model_files,
        root=input_root,
        label="evaluation model",
        root_kind=input_root_kind,
        allow_symlink=True,
    )
    if not isinstance(model_files, list) or len(model_files) == 0:
        raise IntegrityError("evaluation model file manifest is missing")
    if len({record.get("path") for record in model_files if isinstance(record, dict)}) != len(model_files):
        raise IntegrityError("evaluation model file manifest has duplicate or incomplete records")
    if {
        Path(record.get("path", "")).name
        for record in model_files
        if isinstance(record, dict)
    } < {Path(name).name for name in MODEL_FILE_NAMES}:
        raise IntegrityError("evaluation model file manifest has duplicate or incomplete records")
    model_aggregate = aggregate_sha256({record["path"]: record["sha256"] for record in model_files})
    if model_aggregate != model_record.get("aggregate_sha256"):
        raise IntegrityError("evaluation model aggregate digest mismatch")
    if provenance.get("model_snapshot_sha256") != model_record.get("aggregate_sha256"):
        raise IntegrityError("evaluation model snapshot digest mismatch")
    if model is not None:
        current_model = model_snapshot_identity(
            model,
            validation_root=input_root,
            validation_root_kind=input_root_kind,
        )
        for key in ("revision", "n_physical", "n_ut", "n_layers", "d_model", "bytes_status", "aggregate_sha256", "files"):
            if model_record.get(key) != current_model.get(key):
                raise IntegrityError(f"evaluation model {key} does not match loaded model")

    lens_records = provenance.get("lens_inputs")
    if not isinstance(lens_records, list) or not lens_records:
        raise IntegrityError("evaluation lens inputs are missing")
    expected_lens_paths: dict[int, Path] | None = None
    if lens_paths is not None:
        expected_lens_paths = {}
        for target, path in lens_paths:
            if not isinstance(target, int) or isinstance(target, bool) or target in expected_lens_paths:
                raise IntegrityError("requested evaluation lenses contain duplicate targets")
            expected_lens_paths[target] = _lexical_path(path)
    seen_targets: set[int] = set()
    prompt_identities: list[tuple[object, int, int, int, object]] = []
    for index, entry in enumerate(lens_records):
        if not isinstance(entry, dict) or not isinstance(entry.get("target_ut"), int):
            raise IntegrityError(f"evaluation lens input {index} is malformed")
        target_ut = entry["target_ut"]
        if target_ut in seen_targets:
            raise IntegrityError("evaluation lens inputs contain duplicate target_ut")
        seen_targets.add(target_ut)
        binary_record, sidecar_record = entry.get("binary"), entry.get("sidecar")
        _check_declared_root_kind(
            binary_record,
            input_root_kind,
            f"evaluation lens[{index}] binary",
        )
        _check_declared_root_kind(
            sidecar_record,
            input_root_kind,
            f"evaluation lens[{index}] sidecar",
        )
        expected_path = expected_lens_paths.get(target_ut) if expected_lens_paths is not None else None
        binary = _record_matches_artifact(
            binary_record,
            root=input_root,
            label=f"evaluation lens[{index}] binary",
            artifact_root=normalized_artifact_root,
            expected=expected_path,
        )
        sidecar = _record_matches_artifact(
            sidecar_record,
            root=input_root,
            label=f"evaluation lens[{index}] sidecar",
            artifact_root=normalized_artifact_root,
            expected=sidecar_path(expected_path) if expected_path is not None else None,
        )
        if sidecar != sidecar_path(binary):
            raise IntegrityError(f"evaluation lens[{index}] sidecar path mismatch")
        preview_path = sidecar_path(binary)
        try:
            preview = json.loads(preview_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IntegrityError(f"invalid evaluation lens sidecar {preview_path}: {exc}") from exc
        sidecar_kind = preview.get("kind") if isinstance(preview, dict) else None
        if sidecar_kind not in {"fit", "merged"}:
            raise IntegrityError(f"evaluation lens sidecar kind is invalid: {preview_path}")
        metadata = validate_lens_sidecar(
            binary,
            kind=sidecar_kind,
            validation_root=input_root,
            artifact_root=normalized_artifact_root,
        )
        if provenance.get("jlens_commit") != metadata.get("jlens_commit"):
            raise IntegrityError(f"evaluation lens[{index}] jlens commit identity mismatch")
        lens_model = metadata.get("model")
        if not isinstance(lens_model, dict) or lens_model.get("bytes_status") != "HASH_BOUND":
            raise IntegrityError(
                f"evaluation lens[{index}] is not bound to a HASH_BOUND model snapshot"
            )
        for key in (
            "revision",
            "n_physical",
            "n_ut",
            "n_layers",
            "d_model",
            "bytes_status",
            "aggregate_sha256",
            "files",
        ):
            if lens_model.get(key) != model_record.get(key):
                raise IntegrityError(f"evaluation lens[{index}] model identity mismatch: {key}")
        lens = jlens.JacobianLens.load(str(binary))
        _validate_lens_shape(lens, metadata, binary)
        expected_target_virtual = (
            target_ut * model_record["n_physical"] + model_record["n_physical"] - 1
        )
        if metadata.get("target_ut") != target_ut \
                or metadata.get("target_virtual") != expected_target_virtual \
                or metadata.get("source_layers") != list(range(expected_target_virtual)):
            raise IntegrityError(f"evaluation lens[{index}] target/shape identity mismatch")
        if not 0 <= target_ut < model_record["n_ut"]:
            raise IntegrityError(f"evaluation lens[{index}] target_ut is outside the model")
        prompt_identities.append(_prompt_identity_fields(metadata))
        if normalized_artifact_root is not None and _artifact_suffix(binary_record.get("path") if isinstance(binary_record, dict) else None) is not None:
            current_binary = _record_for_actual(binary_record, binary)
            current_sidecar = _record_for_actual(sidecar_record, sidecar)
        else:
            current_binary = _record_for_path(binary, root=input_root, root_kind=input_root_kind)
            current_sidecar = _record_for_path(sidecar, root=input_root, root_kind=input_root_kind)
        _records_equal(binary_record, current_binary, f"evaluation lens[{index}] binary")
        _records_equal(sidecar_record, current_sidecar, f"evaluation lens[{index}] sidecar")
        identity = entry.get("identity")
        if not isinstance(identity, dict):
            raise IntegrityError(f"evaluation lens[{index}] identity is missing")
        for key in (
            "target_ut",
            "target_virtual",
            "source_layers",
            "model_revision",
            "jlens_commit",
            "prompt_file_sha256",
            "prompt_slice_sha256",
            "prompt_slice",
            "n_requested",
            "n_fitted",
            "n_prompts",
            "prompt_provenance_sha256",
            "prompt_source_revision",
            "source_sha256",
            "generator_sha256",
        ):
            if identity.get(key) != metadata.get(key):
                raise IntegrityError(f"evaluation lens[{index}] identity mismatch: {key}")
        if expected_lens_paths is not None:
            if expected_path is None or expected_path != binary:
                raise IntegrityError(f"evaluation lens[{index}] does not match requested lens")

    expected_eventual = model_record.get("n_ut")
    if not isinstance(expected_eventual, int) or expected_eventual - 1 not in seen_targets:
        raise IntegrityError("evaluation lens inputs omit the eventual-exit lens")
    if expected_lens_paths is not None and seen_targets != set(expected_lens_paths):
        raise IntegrityError("evaluation lens inputs do not match requested lenses")
    first_prompt = prompt_identities[0]
    for current_prompt in prompt_identities[1:]:
        if current_prompt[0] != first_prompt[0]:
            raise IntegrityError("evaluation lens prompt files differ")
        if prompt_policy == "identical":
            if current_prompt[1:] != first_prompt[1:]:
                raise IntegrityError("evaluation lens prompt slices violate identical policy")
        elif current_prompt[1] != first_prompt[1]:
            raise IntegrityError("evaluation lens prompt slices violate nested policy")

    outputs = provenance.get("outputs")
    if not isinstance(outputs, dict):
        raise IntegrityError("evaluation outputs are missing")
    for key, filename in (("arrays", "arrays.npz"), ("items", "items.json"), ("task_names", "task_names.json")):
        record = outputs.get(key)
        target = output_root / filename
        _check_declared_root_kind(record, TEST_BUNDLE_ROOT_KIND, f"evaluation output {key}")
        reject_symlink_path(target)
        if not target.is_file() or target.is_symlink():
            raise IntegrityError(f"evaluation output is missing or linked: {target}")
        resolved = _record_matches(record, root=output_root, label=f"evaluation output {key}")
        if resolved != _lexical_path(target):
            raise IntegrityError(f"evaluation output {key} path mismatch")
        _records_equal(
            record,
            _output_record(target, output_root),
            f"evaluation output {key}",
        )
    try:
        with np.load(output_root / "arrays.npz", allow_pickle=False) as values:
            if not values.files:
                raise IntegrityError("evaluation arrays output is empty")
        item_payload = json.loads((output_root / "items.json").read_text(encoding="utf-8"))
        name_payload = json.loads((output_root / "task_names.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"evaluation output JSON/NPZ is invalid: {exc}") from exc
    if not isinstance(item_payload, list) or not isinstance(name_payload, dict):
        raise IntegrityError("evaluation output JSON has an invalid structure")
    if (
        type(provenance.get("item_count")) is not int
        or provenance.get("item_count") != len(item_payload)
    ):
        raise IntegrityError("evaluation item count does not match its output")
    if provenance.get("item_metadata_sha256") != sha256_json(item_payload):
        raise IntegrityError("evaluation item metadata digest does not match its output")
    correct_count = provenance.get("correct_count")
    if (
        not isinstance(correct_count, int)
        or isinstance(correct_count, bool)
        or correct_count != sum(
            item.get("correct") is True
            for item in item_payload
            if isinstance(item, dict)
        )
        or any(not isinstance(item, dict) or not isinstance(item.get("correct"), bool)
               for item in item_payload)
    ):
        raise IntegrityError("evaluation correct count does not match its output")
    return provenance


def _lexical_path(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lens", action="append", required=True, help="target_ut=path")
    p.add_argument("--tasks", nargs="+", default=["multihop", "order-ops"])
    p.add_argument(
        "--position",
        type=int,
        default=-1,
        help="readout token position (default: last prompt token)",
    )
    p.add_argument(
        "--prompt-policy",
        "--cross-lens-policy",
        "--cross-lens-prompt-policy",
        dest="prompt_policy",
        choices=("identical", "nested"),
        default="identical",
        help="relationship required between lens prompt populations (default: identical)",
    )
    p.add_argument(
        "--validation-root",
        help="explicit root for a relocated evaluation bundle",
    )
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    # Keep the output-record root absolute even when callers pass a relative
    # CLI path.  Otherwise _output_record classifies freshly written files as
    # external and the strict provenance validator rejects our own output.
    out = _lexical_path(args.out)
    reject_symlink_path(out)
    if out.exists() and (out.is_symlink() or not out.is_dir()):
        raise IntegrityError(f"evaluation output root is not a directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    reject_symlink_path(out)

    validation_root, validation_root_kind = _validation_context(
        args.validation_root,
        PROJECT_ROOT_KIND if args.validation_root is None else None,
        create=True,
    )

    m = load_ouro()
    specs = _parse_lens_specs(args.lens)
    if not specs or specs[0][0] != m.n_ut - 1:
        raise IntegrityError("the eventual-exit lens (target_ut = last) is required")

    # Verify all lens metadata before spending time on model forwards. Every
    # lens must describe this exact model and the same fit inputs/code; only
    # target_ut and its resulting source range may differ.
    lens_entries: list[tuple[int, Path, jlens.JacobianLens, dict[str, object]]] = []
    baseline: dict[str, object] | None = None
    for target_ut, path in specs:
        metadata = load_lens_metadata(path, validation_root=validation_root)
        validate_lens_metadata(
            metadata,
            target_ut=target_ut,
            model=m,
            baseline=baseline,
            prompt_policy=args.prompt_policy,
            validation_root=validation_root,
        )
        baseline = metadata if baseline is None else baseline
        lens = jlens.JacobianLens.load(str(path))
        _validate_lens_shape(lens, metadata, path)
        J_check = stacked_jacobians(m, lens, m.exit_index(target_ut))
        del J_check
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        lens_entries.append((target_ut, path, lens, metadata))

    items = load_items(m.tokenizer, args.tasks, encode=lambda s: m.encode(s)[0].tolist())
    TASK_NAMES.clear()
    for task in args.tasks:
        TASK_NAMES[task] = TaskNames(items, task)
        if len(TASK_NAMES[task].names) > MAX_NAMES:
            raise IntegrityError(
                f"task {task!r} has {len(TASK_NAMES[task].names)} names, exceeds MAX_NAMES={MAX_NAMES}"
            )
    H, continuations = cache_states(m, items, args.position)
    exit_logits = torch.stack(
        [
            m.unembed(H[:, m.exit_index(ut)].to(m.input_device)).float()
            for ut in range(m.n_ut)
        ],
        dim=1,
    )

    arrays: dict[str, np.ndarray] = {"exit_top1": exit_logits.argmax(-1).cpu().numpy()}
    eventual = None
    for target_ut, path, lens, _metadata in lens_entries:
        J = stacked_jacobians(m, lens, m.exit_index(target_ut))
        res, kept = readout_arrays(m, items, H, J, exit_logits, target_ut, eventual)
        arrays.update({f"jlens_exit{target_ut}_{key}": value for key, value in res.items()})
        if target_ut == m.n_ut - 1:
            eventual = kept
            arrays["xloop_allrank"] = cross_loop(m, items, H, J)
        del J
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"lens exit{target_ut}: {lens} from {path}", flush=True)
    ll, _ = readout_arrays(m, items, H, None, exit_logits, m.n_ut - 1, eventual)
    arrays.update({f"logitlens_{key}": value for key, value in ll.items()})

    # Every output is independently replace-written. The provenance document
    # records the actual byte hashes after the arrays/JSON files are complete.
    arrays_path = out / "arrays.npz"
    items_path = out / "items.json"
    names_path = out / "task_names.json"
    provenance_path = out / "provenance.json"
    atomic_savez(arrays_path, **arrays)
    item_metadata = [
        {
            "name": it.name,
            "task": it.task,
            "prompt": it.prompt,
            "target": it.target,
            "intermediates": it.intermediates[:MAX_INTER],
            "own_index": TASK_NAMES[it.task].own_index(it),
            "scorable": [
                bool(it.intermediate_tokens[k]) for k in it.intermediates[:MAX_INTER]
            ],
            "leaked": [bool(it.leaked[k]) for k in it.intermediates[:MAX_INTER]],
            "n_tokens": len(it.token_ids),
            "readout_token": m.tokenizer.decode([it.token_ids[args.position]]),
            "continuation": c,
            "correct": is_correct(c, it.target),
            "exit_top1": [
                m.tokenizer.decode([token]) for token in arrays["exit_top1"][i]
            ],
        }
        for i, (it, c) in enumerate(zip(items, continuations))
    ]
    atomic_write_json(items_path, item_metadata)
    atomic_write_json(
        names_path,
        {task: task_names.names for task, task_names in TASK_NAMES.items()},
    )
    input_records = _evaluation_inputs(
        args.tasks,
        validation_root=validation_root,
        validation_root_kind=validation_root_kind,
    )
    lens_records = []
    for target_ut, path, _lens, metadata in lens_entries:
        lens_records.append(
            {
                "target_ut": target_ut,
                "binary": _record_for_path(
                    path,
                    root=validation_root,
                    root_kind=validation_root_kind,
                ),
                "sidecar": _record_for_path(
                    sidecar_path(path),
                    root=validation_root,
                    root_kind=validation_root_kind,
                ),
                "identity": {
                    key: metadata.get(key)
                    for key in (
                        "target_ut",
                        "target_virtual",
                        "source_layers",
                        "model_revision",
                        "jlens_commit",
                        "prompt_file_sha256",
                        "prompt_slice_sha256",
                        "prompt_slice",
                        "n_requested",
                        "n_fitted",
                        "n_prompts",
                        "prompt_provenance_sha256",
                        "prompt_source_revision",
                        "source_sha256",
                        "generator_sha256",
                    )
                },
            }
        )
    source = _evaluation_source_manifest(
        validation_root=validation_root,
        validation_root_kind=validation_root_kind,
    )
    current_model = model_snapshot_identity(
        m,
        validation_root=validation_root,
        validation_root_kind=validation_root_kind,
    )
    worker_deadline_epoch = _worker_deadline_epoch()
    provenance: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation": _context_record(validation_root, validation_root_kind),
        "model": {
            **current_model,
        },
        "model_snapshot_sha256": current_model["aggregate_sha256"],
        "jlens_commit": lens_entries[0][3].get("jlens_commit"),
        "config": {
            "tasks": list(args.tasks),
            "position": args.position,
            "prompt_policy": args.prompt_policy,
            "max_intermediates": MAX_INTER,
            "max_names": MAX_NAMES,
            "greedy_steps": GREEDY_STEPS,
        },
        "inputs": {
            "evaluation_files": input_records,
            "evaluation_input_sha256": sha256_json(input_records),
        },
        "lens_inputs": lens_records,
        "source_files": source["files"],
        "source_sha256": source["sha256"],
        "outputs": {
            "arrays": _output_record(arrays_path, out),
            "items": _output_record(items_path, out),
            "task_names": _output_record(names_path, out),
        },
        "item_count": len(items),
        "correct_count": sum(bool(row["correct"]) for row in item_metadata),
        "item_metadata_sha256": sha256_json(item_metadata),
    }
    if worker_deadline_epoch is not None:
        provenance["worker_deadline_epoch"] = worker_deadline_epoch
    atomic_write_json(provenance_path, provenance)
    print(
        f"saved {out}: {len(items)} items, "
        f"model correct on target {provenance['correct_count']}/{len(items)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
