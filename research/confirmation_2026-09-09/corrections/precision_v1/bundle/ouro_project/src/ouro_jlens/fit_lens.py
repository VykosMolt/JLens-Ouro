"""Fit and merge recurrent Jacobian-lens shards with integrity sidecars.

The binary lens is useful only together with the inputs and code that made
it. Every completed ``.pt`` therefore has a JSON sidecar containing the
prompt slice, model/jlens identity, source hashes, configuration, and the
hash of the bytes on disk. A merge refuses to consume anything that cannot
be verified against that contract.

Examples::

    python src/ouro_jlens/fit_lens.py fit --target-ut 3 --start 0 --end 100 \
        --dim-batch 8 --out artifacts/jlens/lens/exit3/shard_000_100.pt
    python src/ouro_jlens/fit_lens.py merge --out lens.pt shard_*.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import jlens
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ouro_jlens.evidence import (  # noqa: E402
    SCHEMA_VERSION,
    aggregate_sha256,
    atomic_write_json,
    file_record,
    prompt_slice_sha256,
    sha256_file,
    sha256_json,
)
from ouro_jlens.recurrent import (  # noqa: E402
    OURO_REVISION,
    OURO_SNAPSHOT,
    PROJECT_ROOT,
    load_ouro,
    model_snapshot_files,
)

try:
    from jlens.fitting import SKIP_FIRST_N_POSITIONS
except (ImportError, AttributeError):  # pragma: no cover - only old jlens installs
    SKIP_FIRST_N_POSITIONS = 16

DEFAULT_PROMPTS = PROJECT_ROOT / "artifacts" / "jlens" / "data" / "wikitext_prompts_b08601e.json"
LOG = logging.getLogger(__name__)
PROJECT_ROOT_KIND = "project"
TEST_BUNDLE_ROOT_KIND = "test_bundle"
EXTERNAL_ROOT_KIND = "external"
DEPENDENCY_ROOT_KIND = "dependency"
DEPENDENCY_LOGICAL_ROOT = Path("dependency") / "jlens"
MODEL_FILE_NAMES = ("model.safetensors", "config.json", "modeling_ouro.py", "tokenizer.json")


class IntegrityError(ValueError):
    """Raised when a lens or sidecar cannot be tied to its declared inputs."""


def sidecar_path(path: str | os.PathLike[str]) -> Path:
    """Return the metadata path paired with a binary lens path."""

    return Path(path).with_suffix(".json")


def checkpoint_sidecar_path(path: str | os.PathLike[str]) -> Path:
    """Return the metadata path paired with a resumable ``.ckpt`` file."""

    return Path(f"{path}.json")


def _lexical_absolute(path: str | os.PathLike[str]) -> Path:
    """Return an absolute path without resolving symlinks.

    ``Path.resolve`` is deliberately not used for integrity decisions: it
    would follow a link planted by an attacker before we had a chance to
    reject it.
    """

    return Path(os.path.abspath(os.fspath(path)))


def reject_symlink_path(path: str | os.PathLike[str], *, include_leaf: bool = True) -> Path:
    """Reject a destination or any existing ancestor that is a symlink.

    Missing ancestors are allowed so callers can create a new artifact tree;
    they are checked again after creation.  The returned path is merely
    lexically absolute and remains safe to use with ``os.replace``.
    """

    absolute = _lexical_absolute(path)
    parts = absolute.parts
    current = Path(absolute.anchor)
    limit = len(parts) if include_leaf else max(1, len(parts) - 1)
    for part in parts[1:limit]:
        current /= part
        if current.is_symlink():
            raise IntegrityError(f"path traverses a symlink: {current}")
    return absolute


def _ensure_directory(path: Path) -> None:
    """Create *path* only after checking all pre-existing ancestors."""

    reject_symlink_path(path)
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir() or path.is_symlink():
        raise IntegrityError(f"validation root is not a regular directory: {path}")
    reject_symlink_path(path)


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validation_context(
    root: str | os.PathLike[str] | None,
    kind: str | None = None,
    *,
    create: bool = False,
) -> tuple[Path, str]:
    """Normalize a validation root and its explicit root kind."""

    if root is None:
        root_path, root_kind = PROJECT_ROOT, PROJECT_ROOT_KIND
    else:
        root_path = _lexical_absolute(root)
        root_kind = kind or (
            PROJECT_ROOT_KIND if root_path == _lexical_absolute(PROJECT_ROOT) else TEST_BUNDLE_ROOT_KIND
        )
    if root_kind not in {PROJECT_ROOT_KIND, TEST_BUNDLE_ROOT_KIND}:
        raise IntegrityError(f"unsupported validation root kind: {root_kind!r}")
    if root_kind == PROJECT_ROOT_KIND and root_path != _lexical_absolute(PROJECT_ROOT):
        raise IntegrityError("project validation root must be the current project root")
    if create:
        _ensure_directory(root_path)
    elif not root_path.is_dir() or root_path.is_symlink():
        raise IntegrityError(f"validation root is missing or linked: {root_path}")
    reject_symlink_path(root_path)
    return root_path, root_kind


def _context_record(root: Path, root_kind: str) -> dict[str, str]:
    """Return the relocatable root declaration stored in every sidecar."""

    # The physical producer path is intentionally absent.  A project root is
    # resolved to PROJECT_ROOT by default; a test bundle can supply an
    # explicit root when it is moved.
    return {"path": ".", "kind": root_kind}


def _record_for_path(
    path: str | os.PathLike[str],
    *,
    root: Path,
    root_kind: str,
    allow_symlink: bool = False,
    force_external: bool = False,
) -> dict[str, Any]:
    """Hash a file while storing a logical path when it belongs to *root*."""

    candidate = _lexical_absolute(path)
    reject_symlink_path(candidate, include_leaf=not allow_symlink)
    if not candidate.is_file():
        raise IntegrityError(f"required file is missing: {candidate}")
    if candidate.is_symlink() and not allow_symlink:
        raise IntegrityError(f"required file is linked: {candidate}")
    if _within(candidate, root) and not force_external:
        logical = candidate.relative_to(root).as_posix()
        record_kind = root_kind
    else:
        # Dependencies outside the validation root cannot be relocated.  Keep
        # their honest absolute identity rather than pretending they belong to
        # the moved bundle.
        logical = str(candidate)
        record_kind = EXTERNAL_ROOT_KIND
    actual = file_record(candidate)
    return {
        "path": logical,
        "root_kind": record_kind,
        "size": actual["size"],
        "sha256": actual["sha256"],
    }


def _jlens_package_root() -> Path:
    package_file = getattr(jlens, "__file__", None)
    if not package_file:
        raise IntegrityError("installed jlens package has no source path")
    root = Path(package_file).resolve().parent
    if not root.is_dir() or root.is_symlink():
        raise IntegrityError(f"installed jlens package root is missing or linked: {root}")
    reject_symlink_path(root)
    return root


def _jlens_source_paths() -> list[Path]:
    """Return every regular Python source file in the installed jlens package."""

    root = _jlens_package_root()
    paths = []
    for candidate in sorted(root.rglob("*.py"), key=lambda value: value.relative_to(root).as_posix()):
        reject_symlink_path(candidate)
        if candidate.is_file() and not candidate.is_symlink():
            paths.append(candidate)
    if not paths:
        raise IntegrityError(f"installed jlens package contains no Python sources: {root}")
    return paths


def _jlens_record(path: Path) -> dict[str, Any]:
    root = _jlens_package_root()
    candidate = _lexical_absolute(path)
    if not _within(candidate, root):
        raise IntegrityError(f"jlens source is outside its installed package: {candidate}")
    relative = candidate.relative_to(root).as_posix()
    actual = file_record(candidate)
    return {
        "path": f"{DEPENDENCY_LOGICAL_ROOT.as_posix()}/{relative}",
        "root_kind": DEPENDENCY_ROOT_KIND,
        "size": actual["size"],
        "sha256": actual["sha256"],
    }


def _record_path(
    record: Any,
    *,
    root: Path,
    label: str,
    allow_symlink: bool = False,
) -> Path:
    """Resolve and validate one logical/external file record."""

    if not isinstance(record, dict) or not isinstance(record.get("path"), str) or not record["path"]:
        raise IntegrityError(f"{label} record path is malformed")
    raw = Path(record["path"])
    record_kind = record.get("root_kind")
    if record_kind == DEPENDENCY_ROOT_KIND:
        if raw.is_absolute():
            raise IntegrityError(f"{label} dependency record path must be logical")
        try:
            relative = raw.relative_to(DEPENDENCY_LOGICAL_ROOT)
        except ValueError as exc:
            raise IntegrityError(f"{label} dependency record has an invalid logical path") from exc
        dependency_root = _jlens_package_root()
        candidate = _lexical_absolute(dependency_root / relative)
        if not _within(candidate, dependency_root):
            raise IntegrityError(f"{label} dependency record escapes installed package")
    elif record_kind == EXTERNAL_ROOT_KIND:
        if not raw.is_absolute():
            raise IntegrityError(f"{label} external record path must be absolute")
        candidate = _lexical_absolute(raw)
    elif raw.is_absolute():
        candidate = _lexical_absolute(raw)
    else:
        if record_kind not in {PROJECT_ROOT_KIND, TEST_BUNDLE_ROOT_KIND, None}:
            raise IntegrityError(f"{label} record has unsupported root kind: {record_kind!r}")
        candidate = _lexical_absolute(root / raw)
        if not _within(candidate, root):
            raise IntegrityError(f"{label} record escapes validation root: {raw}")
    reject_symlink_path(candidate, include_leaf=not allow_symlink)
    if not candidate.is_file():
        raise IntegrityError(f"{label} file is missing: {candidate}")
    if candidate.is_symlink() and not allow_symlink:
        raise IntegrityError(f"{label} file is linked: {candidate}")
    return candidate


def _record_matches(
    record: Any,
    *,
    root: Path,
    label: str,
    allow_symlink: bool = False,
) -> Path:
    candidate = _record_path(record, root=root, label=label, allow_symlink=allow_symlink)
    expected = {
        "path": record.get("path"),
        "root_kind": record.get("root_kind"),
        "size": record.get("size"),
        "sha256": record.get("sha256"),
    }
    if record.get("root_kind") == DEPENDENCY_ROOT_KIND:
        actual = _jlens_record(candidate)
    else:
        actual = _record_for_path(
            candidate,
            root=root,
            root_kind=(record.get("root_kind") if record.get("root_kind") in {
                PROJECT_ROOT_KIND, TEST_BUNDLE_ROOT_KIND
            } else PROJECT_ROOT_KIND),
            allow_symlink=allow_symlink,
        )
    # For old records without root_kind, tolerate the omitted field while
    # still requiring the path/size/digest to match.  New records always carry
    # it, so relocation retains an explicit root contract.
    if record.get("root_kind") is None:
        actual.pop("root_kind", None)
        expected.pop("root_kind", None)
    if actual != expected:
        raise IntegrityError(f"{label} hash/size/path mismatch: {record.get('path')}")
    return candidate


def _check_declared_root_kind(record: Any, root_kind: str, label: str) -> None:
    """Reject a logical record whose root kind contradicts its sidecar."""

    if not isinstance(record, dict):
        return
    record_kind = record.get("root_kind")
    if record_kind in {PROJECT_ROOT_KIND, TEST_BUNDLE_ROOT_KIND} and record_kind != root_kind:
        raise IntegrityError(
            f"{label} root kind {record_kind!r} disagrees with validation root {root_kind!r}"
        )


def _metadata_root(
    metadata: dict[str, Any],
    explicit_root: str | os.PathLike[str] | None = None,
) -> tuple[Path, str]:
    declaration = metadata.get("validation")
    declared_kind = None
    if isinstance(declaration, dict):
        declared_kind = declaration.get("kind") or declaration.get("root_kind")
    if declared_kind is None:
        declaration = metadata.get("validation_root")
        if isinstance(declaration, dict):
            declared_kind = declaration.get("kind") or declaration.get("root_kind")
        elif isinstance(metadata.get("validation_root_kind"), str):
            declared_kind = metadata["validation_root_kind"]
    if explicit_root is not None:
        return _validation_context(explicit_root, declared_kind)
    if declared_kind == PROJECT_ROOT_KIND or declared_kind is None:
        return _validation_context(None, PROJECT_ROOT_KIND)
    if declared_kind == TEST_BUNDLE_ROOT_KIND:
        # A sidecar in a relocated test bundle can be validated directly.  An
        # explicit root remains available for bundles whose root is not the
        # sidecar's parent.
        sidecar = metadata.get("_sidecar_path")
        if sidecar:
            return _validation_context(Path(sidecar).parent, TEST_BUNDLE_ROOT_KIND)
        raise IntegrityError("test-bundle sidecar requires an explicit validation root")
    raise IntegrityError(f"unsupported validation root kind: {declared_kind!r}")


def _validation_args_root(args: Any, prompt_path: Path, out: Path) -> tuple[Path, str]:
    explicit = getattr(args, "validation_root", None)
    explicit_kind = getattr(args, "validation_root_kind", None)
    if explicit is not None:
        return _validation_context(explicit, explicit_kind, create=True)
    project = _lexical_absolute(PROJECT_ROOT)
    prompt_abs, out_abs = _lexical_absolute(prompt_path), _lexical_absolute(out)
    if _within(prompt_abs, project) and _within(out_abs, project):
        return _validation_context(project, PROJECT_ROOT_KIND, create=True)
    if _within(prompt_abs, project) != _within(out_abs, project):
        raise IntegrityError(
            "prompt and output paths mix project and external roots; "
            "pass --validation-root explicitly"
        )
    # Lightweight/unit bundles commonly place their prompt and output files
    # together outside the project.  Treat that as an explicit test bundle,
    # while real commands with default paths retain the project root.
    common = Path(os.path.commonpath((str(prompt_abs.parent), str(out_abs.parent))))
    if common == Path(common.anchor):
        raise IntegrityError(
            "prompt and output paths have no useful common bundle root; "
            "pass --validation-root explicitly"
        )
    return _validation_context(common, TEST_BUNDLE_ROOT_KIND, create=True)


def model_snapshot_identity(
    model: Any,
    *,
    validation_root: str | os.PathLike[str] | Path | None = None,
    validation_root_kind: str | None = None,
) -> dict[str, Any]:
    """Return the loaded model's current revision, shape, and file identity.

    A revision-only model is deliberately represented as such.  It is useful
    for lightweight fitting tests and historical diagnostics, but evaluators
    require ``HASH_BOUND`` before producing new scientific evidence.
    """

    n_physical = getattr(model, "n_physical", None)
    n_ut = getattr(model, "n_ut", None)
    n_layers = getattr(model, "n_layers", None)
    d_model = getattr(model, "d_model", None)
    if not all(
        isinstance(v, int) and not isinstance(v, bool)
        for v in (n_physical, n_ut, n_layers, d_model)
    ):
        raise IntegrityError(
            "loaded model must expose integer n_physical, n_ut, n_layers, and d_model"
        )
    if min(n_physical, n_ut, n_layers, d_model) <= 0:
        raise IntegrityError("loaded model dimensions must all be positive")
    if validation_root is None and validation_root_kind is None:
        root, root_kind = _validation_context(None, PROJECT_ROOT_KIND)
    else:
        root, root_kind = _validation_context(validation_root, validation_root_kind)
    model_revision = str(getattr(model, "model_revision", OURO_REVISION))
    snapshot_path = getattr(model, "snapshot_path", None)
    if snapshot_path is None and model_revision == OURO_REVISION:
        snapshot_path = OURO_SNAPSHOT
    records: list[dict[str, Any]] = []
    if snapshot_path is not None:
        snapshot = Path(snapshot_path)
        try:
            snapshot_files = model_snapshot_files(snapshot)
        except (OSError, ValueError) as exc:
            raise IntegrityError(f"cannot enumerate loaded model snapshot: {snapshot}") from exc
        for candidate in snapshot_files:
            records.append(
                _record_for_path(
                    candidate,
                    root=root,
                    root_kind=root_kind,
                    allow_symlink=True,
                )
            )
    required_model_names = {Path(name).name for name in MODEL_FILE_NAMES}
    bytes_status = (
        "HASH_BOUND"
        if required_model_names <= {Path(record["path"]).name for record in records}
        else "REVISION_AND_SHAPE_ONLY"
    )
    if records:
        aggregate = aggregate_sha256({record["path"]: record["sha256"] for record in records})
    else:
        aggregate = aggregate_sha256(
            {
                "revision": model_revision,
                "shape": sha256_json(
                    {
                        "n_physical": n_physical,
                        "n_ut": n_ut,
                        "n_layers": n_layers,
                        "d_model": d_model,
                    }
                ),
            }
        )
    return {
        "revision": model_revision,
        "n_physical": n_physical,
        "n_ut": n_ut,
        "n_layers": n_layers,
        "d_model": d_model,
        "bytes_status": bytes_status,
        "files": records,
        "aggregate_sha256": aggregate,
    }


def jlens_commit() -> str:
    """Return the installed jlens checkout revision."""

    try:
        repo = Path(jlens.__file__).resolve().parents[1]
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError, IndexError):
        return "UNKNOWN"
    revision = result.stdout.strip()
    return revision or "UNKNOWN"


def _load_prompt_file(path: Path) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot read prompts file {path}: {exc}") from exc
    if isinstance(payload, dict):
        payload = payload.get("prompts")
    if not isinstance(payload, list) or not all(isinstance(prompt, str) for prompt in payload):
        raise IntegrityError(f"prompts file {path} must contain a JSON list of strings")
    return payload


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise IntegrityError(f"{name} must be an integer, got {value!r}")
    return value


def _validate_slice(start: Any, end: Any, n_prompts: int) -> tuple[int, int]:
    start, end = _integer(start, "start"), _integer(end, "end")
    if start < 0 or end <= start or end > n_prompts:
        raise IntegrityError(
            f"invalid prompt range [{start}, {end}) for {n_prompts} prompts; "
            "require 0 <= start < end <= len(prompts)"
        )
    return start, end


def _resolved_target(model: Any, target_ut: Any) -> tuple[int, int, list[int]]:
    target_ut = _integer(target_ut, "target_ut")
    n_ut = getattr(model, "n_ut", None)
    if not isinstance(n_ut, int) or isinstance(n_ut, bool) or n_ut <= 0:
        raise IntegrityError("loaded model does not expose a positive integer n_ut")
    if not 0 <= target_ut < n_ut:
        raise IntegrityError(f"target_ut={target_ut} out of range [0, {n_ut})")
    try:
        target_virtual = int(model.exit_index(target_ut))
    except (AssertionError, IndexError, TypeError, ValueError) as exc:
        raise IntegrityError(f"cannot resolve target_ut={target_ut}: {exc}") from exc
    n_layers = getattr(model, "n_layers", None)
    if not isinstance(n_layers, int) or target_virtual <= 0 or target_virtual >= n_layers:
        raise IntegrityError(
            f"resolved target virtual layer {target_virtual} is invalid for n_layers={n_layers}"
        )
    return target_ut, target_virtual, list(range(target_virtual))


def _path_if_exists(path: Path) -> Path:
    if not path.is_file():
        raise IntegrityError(f"required source file is missing: {path}")
    return path


def _source_identity(
    *,
    validation_root: Path | None = None,
    validation_root_kind: str | None = None,
) -> dict[str, Any]:
    """Hash the generator and implementation it delegates to.

    Files under the declared validation root are represented by logical
    paths.  The complete installed jlens Python source tree is represented
    under the logical ``dependency/jlens`` namespace; validation resolves
    that namespace against the current installed package while retaining
    exact byte hashes.
    """

    validation_root, validation_root_kind = _validation_context(
        validation_root, validation_root_kind
    )
    paths: list[Path] = [
        _path_if_exists(Path(__file__).resolve()),
        _path_if_exists(Path(__file__).with_name("recurrent.py").resolve()),
        _path_if_exists(Path(__file__).with_name("evidence.py").resolve()),
    ]
    records = []
    for path in paths:
        records.append(
            _record_for_path(
                path,
                root=validation_root,
                root_kind=validation_root_kind,
            )
        )
    # jlens is an installed execution dependency, not part of the relocatable
    # project bundle. Record the complete Python source tree under a logical
    # dependency namespace so a moved sidecar resolves it against the current
    # pinned installation while retaining exact byte hashes.
    records.extend(_jlens_record(path) for path in _jlens_source_paths())
    manifest = {
        "files": records,
        "sha256": aggregate_sha256({record["path"]: record["sha256"] for record in records}),
    }
    generator_record = records[0]
    return {
        "source_files": manifest["files"],
        "source_sha256": manifest["sha256"],
        "generator_file": generator_record,
        "generator_sha256": generator_record["sha256"],
    }


def _prompt_identity(
    prompt_path: Path,
    prompts: list[str],
    start: int,
    end: int,
    *,
    validation_root: Path | None = None,
    validation_root_kind: str | None = None,
) -> dict[str, Any]:
    validation_root, validation_root_kind = _validation_context(
        validation_root, validation_root_kind
    )
    record = _record_for_path(
        prompt_path,
        root=validation_root,
        root_kind=validation_root_kind,
    )
    selected = prompts[start:end]
    digest = prompt_slice_sha256(selected)
    identity: dict[str, Any] = {
        "prompt_file": record,
        "prompt_file_sha256": record["sha256"],
        "prompt_file_count": len(prompts),
        "prompt_slice": {
            "start": start,
            "end": end,
            "count": len(selected),
            "sha256": digest,
        },
        "prompt_slice_sha256": digest,
    }
    provenance_path = prompt_path.with_suffix(".provenance.json")
    if provenance_path.exists() or provenance_path.is_symlink():
        identity.update(
            _prompt_provenance_identity(
                prompt_path,
                prompts,
                provenance_path,
                validation_root=validation_root,
                validation_root_kind=validation_root_kind,
            )
        )
    elif prompt_path.name == DEFAULT_PROMPTS.name:
        raise IntegrityError(
            f"pinned prompt corpus is missing its provenance sidecar: {provenance_path}"
        )
    return identity


def _prompt_provenance_identity(
    prompt_path: Path,
    prompts: list[str],
    provenance_path: Path,
    *,
    validation_root: Path,
    validation_root_kind: str,
) -> dict[str, Any]:
    """Bind a prompt corpus to its revision-pinned fetch provenance.

    Custom prompt bundles may carry the same compact provenance shape. The
    canonical pinned WikiText corpus is additionally checked against the
    current fetch helper's declared revision and selection parameters.
    """

    provenance_record = _record_for_path(
        provenance_path,
        root=validation_root,
        root_kind=validation_root_kind,
    )
    try:
        document = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"prompt provenance is unreadable: {provenance_path}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != 1 \
            or document.get("status") != "FRESH_PINNED_DATASET_REVISION":
        raise IntegrityError(f"prompt provenance has an invalid schema/status: {provenance_path}")
    source = document.get("source")
    if not isinstance(source, dict):
        raise IntegrityError(f"prompt provenance has no source revision: {provenance_path}")
    revision = source.get("revision")
    if not isinstance(revision, str) or len(revision) != 40 \
            or any(character not in "0123456789abcdef" for character in revision):
        raise IntegrityError(f"prompt provenance has an invalid source revision: {provenance_path}")
    requested = source.get("requested_prompts")
    if not isinstance(requested, int) or isinstance(requested, bool) or requested != len(prompts):
        raise IntegrityError(f"prompt provenance prompt count does not match its corpus: {provenance_path}")
    minimum_characters = source.get("minimum_characters")
    if not isinstance(minimum_characters, int) or isinstance(minimum_characters, bool) \
            or minimum_characters <= 0:
        raise IntegrityError(f"prompt provenance has an invalid selection rule: {provenance_path}")
    if any(len(prompt.strip()) < minimum_characters for prompt in prompts):
        raise IntegrityError(f"prompt provenance selection rule does not match corpus: {provenance_path}")
    output = document.get("output")
    if not isinstance(output, dict) or output.get("path") != "wikitext_prompts" \
            or output.get("size") != prompt_path.stat().st_size \
            or output.get("sha256") != sha256_file(prompt_path):
        raise IntegrityError(f"prompt provenance output does not bind corpus bytes: {provenance_path}")
    # The canonical file is the only default scientific corpus. Keep its
    # pinned dataset identity explicit, rather than accepting a provenance
    # file that merely happens to contain a 40-character revision.
    if prompt_path.name == DEFAULT_PROMPTS.name:
        try:
            from ouro_jlens.fetch_wikitext import WIKITEXT_REVISION
        except ImportError as exc:  # pragma: no cover - package is in-tree
            raise IntegrityError("pinned prompt provenance helper is unavailable") from exc
        expected_source = {
            "dataset": "Salesforce/wikitext",
            "config": "wikitext-103-raw-v1",
            "split": "train",
            "revision": WIKITEXT_REVISION,
            "minimum_characters": 600,
            "requested_prompts": 1200,
        }
        if source != expected_source:
            raise IntegrityError(f"pinned prompt source identity changed: {provenance_path}")
        generator = document.get("generator")
        if not isinstance(generator, dict):
            raise IntegrityError(f"pinned prompt provenance has no generator identity: {provenance_path}")
        current_generator = _record_for_path(
            Path(__file__).with_name("fetch_wikitext.py").resolve(),
            root=PROJECT_ROOT,
            root_kind=PROJECT_ROOT_KIND,
        )
        # fetch_wikitext's standalone corpus provenance predates root_kind;
        # compare its logical path, size, and bytes without weakening the
        # current sidecar root declaration.
        current_generator.pop("root_kind", None)
        if generator != current_generator:
            raise IntegrityError(f"pinned prompt generator identity changed: {provenance_path}")
    return {
        "prompt_provenance": provenance_record,
        "prompt_provenance_sha256": provenance_record["sha256"],
        "prompt_source": copy.deepcopy(source),
        "prompt_source_revision": revision,
    }


def _base_metadata(
    model: Any,
    *,
    target_ut: int,
    target_virtual: int,
    source_layers: list[int],
    prompt_path: Path,
    prompts: list[str],
    start: int,
    end: int,
    dim_batch: int,
    max_seq_len: int,
    skip_first: int,
    checkpoint_every: int | None,
    validation_root: Path,
    validation_root_kind: str,
) -> dict[str, Any]:
    model_identity = model_snapshot_identity(
        model,
        validation_root=validation_root,
        validation_root_kind=validation_root_kind,
    )
    n_physical = model_identity["n_physical"]
    n_ut = model_identity["n_ut"]
    n_layers = model_identity["n_layers"]
    d_model = model_identity["d_model"]
    source = _source_identity(
        validation_root=validation_root,
        validation_root_kind=validation_root_kind,
    )
    prompts_identity = _prompt_identity(
        prompt_path,
        prompts,
        start,
        end,
        validation_root=validation_root,
        validation_root_kind=validation_root_kind,
    )
    model_revision = model_identity["revision"]
    model_files = model_identity["files"]
    model_bytes_status = model_identity["bytes_status"]
    model_snapshot_sha256 = model_identity["aggregate_sha256"]
    config = {
        "target_ut": target_ut,
        "target_virtual": target_virtual,
        "source_layers": source_layers,
        "n_physical": n_physical,
        "n_ut": n_ut,
        "n_layers": n_layers,
        "d_model": d_model,
        "dim_batch": dim_batch,
        "max_seq_len": max_seq_len,
        "skip_first": skip_first,
        "checkpoint_every": checkpoint_every,
        "virtual_index": "ut * n_physical + layer",
        "bos_prepended": True,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "fit",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **config,
        "config": config,
        "model_revision": model_revision,
        "model": {
            "revision": model_revision,
            "n_physical": n_physical,
            "n_ut": n_ut,
            "n_layers": n_layers,
            "d_model": d_model,
            "bytes_status": model_bytes_status,
            "files": model_files,
            "aggregate_sha256": model_snapshot_sha256,
        },
        "model_snapshot_sha256": model_snapshot_sha256,
        "jlens_commit": jlens_commit(),
        **prompts_identity,
        **source,
        "prompts": prompts_identity["prompt_file"]["path"],
        "validation": _context_record(validation_root, validation_root_kind),
        "start": start,
        "end": end,
        "n_requested": end - start,
        "seconds": None,
    }


# These fields define compatibility between independent shards. Runtime and
# output-only fields are deliberately absent.
IDENTITY_FIELDS = (
    "target_ut",
    "target_virtual",
    "source_layers",
    "n_physical",
    "n_ut",
    "n_layers",
    "d_model",
    "dim_batch",
    "max_seq_len",
    "skip_first",
    "checkpoint_every",
    "virtual_index",
    "bos_prepended",
    "model_revision",
    "model_snapshot_sha256",
    "jlens_commit",
    "prompt_file_sha256",
    "prompt_file_count",
    "prompt_provenance_sha256",
    "prompt_source_revision",
    "source_sha256",
    "generator_sha256",
)


def identity_projection(meta: dict[str, Any]) -> dict[str, Any]:
    """Return the fields that must agree for shards/lenses to be combined."""

    return {field: meta.get(field) for field in IDENTITY_FIELDS}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"invalid sidecar {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise IntegrityError(f"sidecar {path} must contain a JSON object")
    return payload


def _verify_binary(path: Path, meta: dict[str, Any], *, validation_root: Path) -> None:
    reject_symlink_path(path)
    if not path.is_file() or path.is_symlink():
        raise IntegrityError(f"binary output is missing or linked: {path}")
    output = meta.get("output")
    declared = meta.get("output_sha256")
    if not isinstance(output, dict) or not output.get("path"):
        raise IntegrityError(f"sidecar for {path} has no output file record")
    if output.get("sha256") != declared:
        raise IntegrityError(f"output digest fields disagree for {path}")
    if isinstance(output, dict):
        declared = output.get("sha256", declared)
    if not isinstance(declared, str) or len(declared) != 64:
        raise IntegrityError(f"sidecar for {path} has no output SHA-256")
    actual = sha256_file(path)
    if actual != declared:
        raise IntegrityError(
            f"output hash mismatch for {path}: expected {declared}, got {actual}"
        )
    if output.get("size") is not None:
        if int(output["size"]) != path.stat().st_size:
            raise IntegrityError(f"output size mismatch for {path}")
    output_path = _record_matches(output, root=validation_root, label=f"output for {path}")
    if output_path != _lexical_absolute(path):
        raise IntegrityError(f"output record path mismatch for {path}")


def _strict_count_identity(meta: dict[str, Any], path: Path) -> tuple[int, int, int, int, dict[str, Any]]:
    """Validate every duplicated range/count field before publication/use."""

    prompt_slice = meta.get("prompt_slice")
    if not isinstance(prompt_slice, dict):
        raise IntegrityError(f"invalid prompt slice record for {path}")
    fields: dict[str, int] = {}
    for name in ("start", "end", "n_requested", "n_fitted", "n_prompts"):
        value = meta.get(name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise IntegrityError(f"sidecar {path} has invalid {name}")
        fields[name] = value
    nested: dict[str, int] = {}
    for name in ("start", "end", "count"):
        value = prompt_slice.get(name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise IntegrityError(f"sidecar {path} has invalid prompt_slice.{name}")
        nested[name] = value
    start, end = fields["start"], fields["end"]
    expected_count = end - start
    if start < 0 or end <= start:
        raise IntegrityError(f"invalid prompt range [{start}, {end}) in {path}")
    if nested["start"] != start or nested["end"] != end:
        raise IntegrityError(f"prompt range fields disagree in {path}")
    if nested["count"] != expected_count:
        raise IntegrityError(f"prompt slice count/range mismatch in {path}")
    for name in ("n_requested", "n_fitted", "n_prompts"):
        if fields[name] != expected_count:
            raise IntegrityError(f"{name} does not equal prompt range count in {path}")
    if expected_count <= 0:
        raise IntegrityError(f"lens contains no requested prompts: {path}")
    return start, end, expected_count, fields["n_fitted"], prompt_slice


def validate_sidecar(
    path: str | os.PathLike[str],
    *,
    kind: str = "fit",
    validation_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Load and verify a completed lens and its sidecar."""

    binary = Path(path)
    metadata_path = sidecar_path(binary)
    reject_symlink_path(binary)
    reject_symlink_path(metadata_path)
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise IntegrityError(f"missing sidecar for {binary}: {metadata_path}")
    meta = _read_json(metadata_path)
    meta["_sidecar_path"] = metadata_path
    root, _root_kind = _metadata_root(meta, validation_root)
    meta.pop("_sidecar_path", None)
    declaration = meta.get("validation")
    if declaration is not None and (
        not isinstance(declaration, dict)
        or declaration.get("path") != "."
        or declaration.get("kind") not in {PROJECT_ROOT_KIND, TEST_BUNDLE_ROOT_KIND}
    ):
        raise IntegrityError(f"sidecar {metadata_path} has an invalid validation root declaration")
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise IntegrityError(f"unsupported sidecar schema in {metadata_path}")
    if meta.get("kind") != kind:
        raise IntegrityError(
            f"sidecar {metadata_path} has kind={meta.get('kind')!r}, expected {kind!r}"
        )
    required = (
        "source_layers",
        "target_ut",
        "target_virtual",
        "model",
        "model_revision",
        "model_snapshot_sha256",
        "jlens_commit",
        "prompt_file",
        "prompt_slice",
        "prompt_file_sha256",
        "prompt_slice_sha256",
        "start",
        "end",
        "n_requested",
        "n_fitted",
        "n_prompts",
        "validation",
        "source_files",
        "source_sha256",
        "generator_file",
        "generator_sha256",
        "output",
        "output_sha256",
    )
    missing = [field for field in required if field not in meta]
    if missing:
        raise IntegrityError(f"sidecar {metadata_path} is missing required fields: {missing}")
    if not isinstance(meta.get("source_layers"), list):
        raise IntegrityError(f"sidecar {metadata_path} has invalid source_layers")
    for field in (
        "target_virtual",
        "n_physical",
        "n_ut",
        "n_layers",
        "d_model",
        "prompt_file_count",
    ):
        value = meta.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise IntegrityError(f"sidecar {metadata_path} has invalid {field}")
    target_ut = meta.get("target_ut")
    if not isinstance(target_ut, int) or isinstance(target_ut, bool) or target_ut < 0:
        raise IntegrityError(f"sidecar {metadata_path} has invalid target_ut")
    if meta["target_virtual"] >= meta["n_layers"]:
        raise IntegrityError(f"sidecar {metadata_path} has an invalid target_virtual")
    if meta["target_ut"] >= meta["n_ut"]:
        raise IntegrityError(f"sidecar {metadata_path} has an invalid target_ut")
    expected_target_virtual = (
        meta["target_ut"] * meta["n_physical"] + meta["n_physical"] - 1
    )
    if meta["target_virtual"] != expected_target_virtual:
        raise IntegrityError(f"sidecar {metadata_path} target_virtual disagrees with target_ut")
    if meta["source_layers"] != list(range(meta["target_virtual"])):
        raise IntegrityError(f"sidecar {metadata_path} has non-contiguous source_layers")
    _strict_count_identity(meta, binary)
    prompt_slice = meta["prompt_slice"]
    if not isinstance(prompt_slice.get("sha256"), str) or not isinstance(
        meta.get("prompt_slice_sha256"), str
    ):
        raise IntegrityError(f"prompt slice digest fields are malformed in {metadata_path}")
    if prompt_slice["sha256"] != meta["prompt_slice_sha256"]:
        raise IntegrityError(f"prompt slice digest fields disagree in {metadata_path}")
    model_record = meta.get("model")
    if not isinstance(model_record, dict) or meta.get("model_revision") != model_record.get("revision"):
        raise IntegrityError(f"model identity mismatch in {metadata_path}")
    for field in ("n_physical", "n_ut", "n_layers", "d_model"):
        if model_record.get(field) != meta.get(field):
            raise IntegrityError(f"model shape field {field} disagrees in {metadata_path}")
    model_files = model_record.get("files", [])
    if not isinstance(model_files, list):
        raise IntegrityError(f"invalid model file manifest in {metadata_path}")
    if model_record.get("bytes_status") not in {"HASH_BOUND", "REVISION_AND_SHAPE_ONLY"}:
        raise IntegrityError(f"unsupported model bytes status in {metadata_path}")
    if model_record.get("bytes_status") == "HASH_BOUND" and not model_files:
        raise IntegrityError(f"hash-bound model has no file records in {metadata_path}")
    if model_record.get("bytes_status") == "HASH_BOUND" and not {
        Path(record.get("path", "")).name
        for record in model_files
        if isinstance(record, dict)
    } >= {Path(name).name for name in MODEL_FILE_NAMES}:
        raise IntegrityError(f"hash-bound model file manifest is incomplete in {metadata_path}")
    if model_files:
        model_digests: dict[str, str] = {}
        for record in model_files:
            if not isinstance(record, dict) or not record.get("path"):
                raise IntegrityError(f"invalid model file record in {metadata_path}")
            _check_declared_root_kind(
                record,
                _root_kind,
                f"model file {record.get('path')} in {metadata_path}",
            )
            _record_matches(
                record,
                root=root,
                label=f"model file {record.get('path')}",
                allow_symlink=True,
            )
            model_digests[record["path"]] = record["sha256"]
        if len(model_digests) != len(model_files):
            raise IntegrityError(f"hash-bound model has duplicate file records in {metadata_path}")
        if aggregate_sha256(model_digests) != model_record.get("aggregate_sha256"):
            raise IntegrityError(f"model aggregate digest mismatch in {metadata_path}")
    if meta.get("model_snapshot_sha256") != model_record.get("aggregate_sha256"):
        raise IntegrityError(f"model snapshot digest field mismatch in {metadata_path}")
    _check_declared_root_kind(meta.get("output"), _root_kind, f"output for {binary}")
    _verify_binary(binary, meta, validation_root=root)
    # Verify the prompt/source records still describe the bytes consumed by
    # this process. This catches a modified source file even when the output
    # itself was not modified.
    prompt_file = meta.get("prompt_file")
    if not isinstance(prompt_file, dict) or not prompt_file.get("path"):
        raise IntegrityError(f"invalid prompt file record for {binary}")
    if "prompts" in meta and meta.get("prompts") != prompt_file.get("path"):
        raise IntegrityError(f"prompt path fields disagree for {binary}")
    _check_declared_root_kind(prompt_file, _root_kind, f"prompt file for {binary}")
    prompt_path = _record_matches(prompt_file, root=root, label=f"prompt file for {binary}")
    prompt_record = _record_for_path(prompt_path, root=root, root_kind=_root_kind)
    if prompt_file.get("root_kind") is None:
        prompt_record.pop("root_kind", None)
    if prompt_record != prompt_file:
        raise IntegrityError(f"prompt file hash/size mismatch for {binary}")
    if meta.get("prompt_file_sha256") != prompt_record["sha256"]:
        raise IntegrityError(f"prompt file digest field mismatch for {binary}")
    prompts = _load_prompt_file(prompt_path)
    if meta.get("prompt_file_count") != len(prompts):
        raise IntegrityError(f"prompt file count mismatch for {binary}")
    paired_provenance = prompt_path.with_suffix(".provenance.json")
    stored_provenance = meta.get("prompt_provenance")
    if stored_provenance is not None:
        if not paired_provenance.is_file() or paired_provenance.is_symlink():
            raise IntegrityError(f"prompt provenance is missing or linked for {binary}")
        expected_provenance = _prompt_provenance_identity(
            prompt_path,
            prompts,
            paired_provenance,
            validation_root=root,
            validation_root_kind=_root_kind,
        )
        _check_declared_root_kind(
            stored_provenance,
            _root_kind,
            f"prompt provenance for {binary}",
        )
        if stored_provenance != expected_provenance["prompt_provenance"]:
            raise IntegrityError(f"prompt provenance record mismatch for {binary}")
        if meta.get("prompt_provenance_sha256") != expected_provenance["prompt_provenance_sha256"]:
            raise IntegrityError(f"prompt provenance digest field mismatch for {binary}")
        if meta.get("prompt_source") != expected_provenance["prompt_source"] \
                or meta.get("prompt_source_revision") != expected_provenance["prompt_source_revision"]:
            raise IntegrityError(f"prompt source revision fields disagree for {binary}")
        if stored_provenance.get("path") != _record_for_path(
            paired_provenance,
            root=root,
            root_kind=_root_kind,
        )["path"]:
            raise IntegrityError(f"prompt provenance path mismatch for {binary}")
    elif paired_provenance.exists() or paired_provenance.is_symlink():
        raise IntegrityError(f"prompt provenance record is missing for {binary}")
    elif prompt_path.name == DEFAULT_PROMPTS.name:
        raise IntegrityError(f"pinned prompt corpus provenance is missing for {binary}")
    start, end = prompt_slice.get("start"), prompt_slice.get("end")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or start < 0
        or end <= start
        or end > len(prompts)
        or prompt_slice.get("count") != end - start
    ):
        raise IntegrityError(f"invalid prompt slice bounds for {binary}")
    slice_digest = prompt_slice_sha256(prompts[start:end])
    if prompt_slice.get("sha256") != slice_digest or meta.get("prompt_slice_sha256") != slice_digest:
        raise IntegrityError(f"prompt slice hash mismatch for {binary}")
    source_records = meta.get("source_files", [])
    if not isinstance(source_records, list) or not source_records:
        raise IntegrityError(f"invalid source manifest in {metadata_path}")
    source_digests: dict[str, str] = {}
    for record in source_records:
        if not isinstance(record, dict) or not record.get("path"):
            raise IntegrityError(f"invalid source record in {metadata_path}")
        _check_declared_root_kind(record, _root_kind, f"source file {record.get('path')}")
        source_path = _record_matches(record, root=root, label=f"source file {record.get('path')}")
        source_digests[record["path"]] = record["sha256"]
    if len(source_digests) != len(source_records):
        raise IntegrityError(f"source manifest has duplicate records in {metadata_path}")
    if meta.get("source_sha256") != aggregate_sha256(source_digests):
        raise IntegrityError(f"source manifest digest mismatch for {binary}")
    generator = meta.get("generator_file")
    if (
        not isinstance(generator, dict)
        or meta.get("generator_sha256") != generator.get("sha256")
        or generator not in source_records
    ):
        raise IntegrityError(f"generator identity mismatch for {binary}")
    if kind == "merged":
        records = meta.get("shard_records")
        if not isinstance(records, list) or not records:
            raise IntegrityError(f"merged sidecar has no sealed shard_records: {metadata_path}")
        sealed = meta.get("shard_records_sha256")
        if not isinstance(sealed, str) or sealed != sha256_json(records):
            raise IntegrityError(f"merged shard_records seal mismatch: {metadata_path}")
        shard_paths = meta.get("shards")
        recorded_paths = [record.get("path") for record in records if isinstance(record, dict)]
        if not isinstance(shard_paths, list) or shard_paths != recorded_paths:
            raise IntegrityError(f"merged shard path fields disagree: {metadata_path}")
    return meta


def _atomic_lens_save(lens: Any, path: Path) -> None:
    destination = reject_symlink_path(path)
    _ensure_directory(destination.parent)
    # mkstemp creates an exclusive regular inode with an unpredictable name;
    # a pre-planted path (including a symlink) can therefore never be opened
    # by this save operation.
    fd, raw_temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    temporary = Path(raw_temporary)
    try:
        owned_stat = os.fstat(fd)
        owned_inode = (owned_stat.st_dev, owned_stat.st_ino)
    finally:
        os.close(fd)
    try:
        temporary_stat = os.lstat(temporary)
        if not stat.S_ISREG(temporary_stat.st_mode) or (
            temporary_stat.st_dev,
            temporary_stat.st_ino,
        ) != owned_inode:
            raise IntegrityError(f"temporary lens path is not a regular file: {temporary}")
        lens.save(str(temporary))
        temporary_stat = os.lstat(temporary)
        if not stat.S_ISREG(temporary_stat.st_mode) or (
            temporary_stat.st_dev,
            temporary_stat.st_ino,
        ) != owned_inode:
            raise IntegrityError(f"lens save replaced temporary path: {temporary}")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        # Reject a destination/ancestor link planted while the serializer was
        # running.  ``os.replace`` itself is atomic and never follows a leaf
        # symlink, but accepting one would violate the destination contract.
        reject_symlink_path(destination)
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            temporary_stat = os.lstat(temporary)
            if (
                temporary_stat.st_dev,
                temporary_stat.st_ino,
            ) == owned_inode:
                temporary.unlink()
        except OSError:
            pass
        raise


def _validate_fit_args(args: Any) -> tuple[Path, int, int, int, int, int]:
    prompt_path = Path(getattr(args, "prompts", DEFAULT_PROMPTS))
    start = _integer(getattr(args, "start", 0), "start")
    end = _integer(getattr(args, "end", 100), "end")
    dim_batch = _integer(getattr(args, "dim_batch", 8), "dim_batch")
    max_seq_len = _integer(getattr(args, "max_seq_len", 128), "max_seq_len")
    skip_first = _integer(getattr(args, "skip_first", SKIP_FIRST_N_POSITIONS), "skip_first")
    checkpoint_every = getattr(args, "checkpoint_every", None)
    if checkpoint_every is not None:
        checkpoint_every = _integer(checkpoint_every, "checkpoint_every")
        if checkpoint_every <= 0:
            raise IntegrityError(f"checkpoint_every must be > 0 or None, got {checkpoint_every}")
    if dim_batch <= 0:
        raise IntegrityError(f"dim_batch must be > 0, got {dim_batch}")
    if max_seq_len <= 1:
        raise IntegrityError(f"max_seq_len must be > 1, got {max_seq_len}")
    if skip_first < 0:
        raise IntegrityError(f"skip_first must be >= 0, got {skip_first}")
    return prompt_path, start, end, dim_batch, max_seq_len, skip_first


def _checkpoint_contract(meta: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(meta)
    result["kind"] = "fit_checkpoint"
    result.pop("output", None)
    result.pop("output_sha256", None)
    return result


def _prepare_checkpoint(checkpoint: Path, expected: dict[str, Any]) -> None:
    metadata_path = checkpoint_sidecar_path(checkpoint)
    if checkpoint.exists():
        if not metadata_path.is_file():
            raise IntegrityError(f"checkpoint exists without sidecar: {checkpoint}")
        found = _read_json(metadata_path)
        if found.get("schema_version") != SCHEMA_VERSION or found.get("kind") != "fit_checkpoint":
            raise IntegrityError(f"invalid checkpoint sidecar: {metadata_path}")
        if identity_projection(found) != identity_projection(expected):
            raise IntegrityError(f"checkpoint configuration/model/prompt mismatch: {checkpoint}")
        if found.get("start") != expected.get("start") or found.get("end") != expected.get("end"):
            raise IntegrityError(f"checkpoint prompt range mismatch: {checkpoint}")
        checkpoint_record = found.get("checkpoint")
        if checkpoint_record is None:
            raise IntegrityError(
                f"checkpoint was never sealed with a digest and cannot be resumed: {checkpoint}"
            )
        if not isinstance(checkpoint_record, dict):
            raise IntegrityError(f"invalid checkpoint record: {metadata_path}")
        actual = file_record(checkpoint)
        if actual != checkpoint_record:
            raise IntegrityError(f"checkpoint hash/size mismatch: {checkpoint}")
    elif metadata_path.exists():
        raise IntegrityError(f"stale checkpoint sidecar without checkpoint: {metadata_path}")
    else:
        atomic_write_json(metadata_path, _checkpoint_contract(expected))


def fit(args: Any) -> Any:
    """Fit one prompt range and atomically publish a verified lens."""

    out = Path(getattr(args, "out"))
    reject_symlink_path(out)
    reject_symlink_path(sidecar_path(out))
    prompt_path, start, end, dim_batch, max_seq_len, skip_first = _validate_fit_args(args)
    prompts = _load_prompt_file(prompt_path)
    start, end = _validate_slice(start, end, len(prompts))
    validation_root, validation_root_kind = _validation_args_root(args, prompt_path, out)

    # Validate target and construct the identity before running the expensive
    # fit. This also ensures that an existing output can be checked without
    # silently overwriting it.
    model = load_ouro()
    target_ut, target_virtual, source_layers = _resolved_target(model, getattr(args, "target_ut"))
    expected = _base_metadata(
        model,
        target_ut=target_ut,
        target_virtual=target_virtual,
        source_layers=source_layers,
        prompt_path=prompt_path,
        prompts=prompts,
        start=start,
        end=end,
        dim_batch=dim_batch,
        max_seq_len=max_seq_len,
        skip_first=skip_first,
        checkpoint_every=getattr(args, "checkpoint_every", None),
        validation_root=validation_root,
        validation_root_kind=validation_root_kind,
    )

    # A complete matching output is idempotent. Any partial or stale output
    # is rejected rather than replaced, because replacement could hide a
    # failed prior run from a reviewer.
    if out.exists():
        found = validate_sidecar(out, kind="fit", validation_root=validation_root)
        existing_lens = jlens.JacobianLens.load(str(out))
        _validate_lens_shape(existing_lens, found, out)
        if (
            identity_projection(found) != identity_projection(expected)
            or found.get("start") != start
            or found.get("end") != end
        ):
            raise IntegrityError(f"existing output has incompatible sidecar: {out}")
        LOG.info("verified existing lens %s; nothing to do", out)
        return None
    if sidecar_path(out).exists():
        raise IntegrityError(f"stale sidecar without output: {sidecar_path(out)}")

    checkpoint = Path(getattr(args, "checkpoint", None) or f"{out}.ckpt")
    _prepare_checkpoint(checkpoint, expected)
    t0 = time.perf_counter()
    try:
        lens = jlens.fit(
            model,
            prompts[start:end],
            source_layers=source_layers,
            target_layer=target_virtual,
            dim_batch=dim_batch,
            max_seq_len=max_seq_len,
            skip_first=skip_first,
            checkpoint_path=str(checkpoint),
            checkpoint_every=getattr(args, "checkpoint_every", None),
            resume=True,
        )
    except BaseException:
        # A checkpoint is resumable only when this wrapper observed a complete
        # checkpoint file and sealed its bytes.  An uncatchable process/pod
        # loss leaves the pre-run sidecar unsealed and the next run fails
        # closed instead of trusting a possibly partial file.
        if checkpoint.is_file():
            checkpoint_meta = _checkpoint_contract(expected)
            checkpoint_meta["checkpoint"] = file_record(checkpoint)
            checkpoint_meta["status"] = "INTERRUPTED_RESUMABLE"
            atomic_write_json(checkpoint_sidecar_path(checkpoint), checkpoint_meta)
        raise
    elapsed = round(time.perf_counter() - t0, 3)
    if list(getattr(lens, "source_layers", [])) != source_layers:
        raise IntegrityError("jlens.fit returned unexpected source_layers")
    if int(getattr(lens, "d_model", -1)) != int(expected["d_model"]):
        raise IntegrityError("jlens.fit returned unexpected d_model")

    raw_n_prompts = getattr(lens, "n_prompts", None)
    if not isinstance(raw_n_prompts, int) or isinstance(raw_n_prompts, bool):
        raise IntegrityError("jlens.fit returned an invalid n_prompts count")
    expected_count = end - start
    if raw_n_prompts <= 0 or raw_n_prompts != expected_count:
        raise IntegrityError(
            "jlens.fit returned an incomplete prompt count: "
            f"n_prompts={raw_n_prompts}, expected={expected_count}"
        )
    metadata = copy.deepcopy(expected)
    metadata.update(
        {
            "seconds": elapsed,
            "n_prompts": raw_n_prompts,
            "n_fitted": raw_n_prompts,
        }
    )
    _strict_count_identity(metadata, out)
    _validate_lens_shape(lens, metadata, out)
    _atomic_lens_save(lens, out)
    output = _record_for_path(
        out,
        root=validation_root,
        root_kind=validation_root_kind,
    )
    metadata["output"] = output
    metadata["output_sha256"] = output["sha256"]
    atomic_write_json(sidecar_path(out), metadata)
    if checkpoint.is_file():
        checkpoint_meta = _checkpoint_contract(expected)
        checkpoint_meta["checkpoint"] = file_record(checkpoint)
        atomic_write_json(checkpoint_sidecar_path(checkpoint), checkpoint_meta)
    LOG.info("saved %s (%d fitted prompts)", out, metadata["n_fitted"])
    return lens


def _range_from_meta(meta: dict[str, Any], path: Path) -> tuple[int, int]:
    start, end, expected_count, _n_fitted, _prompt_slice = _strict_count_identity(meta, path)
    if expected_count != end - start:
        raise IntegrityError(f"prompt count/range mismatch in {path}")
    return start, end


def _validate_lens_shape(lens: Any, meta: dict[str, Any], path: Path) -> None:
    expected_layers = meta.get("source_layers")
    if not isinstance(expected_layers, list) or list(getattr(lens, "source_layers", [])) != expected_layers:
        raise IntegrityError(f"lens source_layers disagree with sidecar: {path}")
    d_model = int(meta.get("d_model", -1))
    if int(getattr(lens, "d_model", -2)) != d_model:
        raise IntegrityError(f"lens d_model disagrees with sidecar: {path}")
    n_prompts = getattr(lens, "n_prompts", None)
    if not isinstance(n_prompts, int) or isinstance(n_prompts, bool) or n_prompts <= 0:
        raise IntegrityError(f"lens contains no valid fitted prompts: {path}")
    _start, _end, expected_count, n_fitted, _prompt_slice = _strict_count_identity(meta, path)
    if n_prompts != n_fitted or n_prompts != expected_count:
        raise IntegrityError(f"lens n_prompts disagrees with sidecar: {path}")
    for layer in expected_layers:
        matrix = getattr(lens, "jacobians", {}).get(layer)
        if matrix is None or tuple(matrix.shape) != (d_model, d_model):
            raise IntegrityError(f"invalid Jacobian shape at layer {layer} in {path}")
        try:
            finite = bool(torch.isfinite(matrix).all().item())
        except (TypeError, RuntimeError, AttributeError) as exc:
            raise IntegrityError(f"Jacobian at layer {layer} is not finite in {path}") from exc
        if not finite:
            raise IntegrityError(f"Jacobian at layer {layer} is not finite in {path}")


def _merged_prompt_identity(
    first: dict[str, Any],
    start: int,
    end: int,
    *,
    validation_root: Path,
) -> dict[str, Any]:
    prompt_file = first.get("prompt_file", {})
    if isinstance(prompt_file, dict) and prompt_file.get("path"):
        path = _record_path(prompt_file, root=validation_root, label="merged prompt file")
    else:
        path = Path(first.get("prompts", ""))
    result: dict[str, Any] = {"start": start, "end": end, "count": end - start}
    if path.is_file():
        prompts = _load_prompt_file(path)
        if end > len(prompts):
            raise IntegrityError(f"merged range [{start}, {end}) exceeds prompt file length")
        result["sha256"] = prompt_slice_sha256(prompts[start:end])
    else:
        # Keep merge usable after a verified artifact is relocated, while
        # retaining an identity that cannot be confused with a raw assertion.
        result["sha256"] = sha256_json(
            {
                "prompt_file_sha256": first.get("prompt_file_sha256"),
                "start": start,
                "end": end,
            }
        )
    return result


def _shard_record(
    path: Path,
    start: int,
    end: int,
    meta: dict[str, Any],
    *,
    validation_root: Path,
    validation_root_kind: str,
) -> dict[str, Any]:
    """Build the sealed, ordered identity stored by a merged lens."""

    binary = _record_for_path(
        path,
        root=validation_root,
        root_kind=validation_root_kind,
    )
    sidecar = sidecar_path(path)
    sidecar_record = _record_for_path(
        sidecar,
        root=validation_root,
        root_kind=validation_root_kind,
    )
    count = end - start
    output_sha256 = meta.get("output_sha256", meta.get("output", {}).get("sha256"))
    if output_sha256 != binary["sha256"]:
        raise IntegrityError(f"shard output digest disagrees with sidecar: {path}")
    return {
        "path": binary["path"],
        "root_kind": binary["root_kind"],
        "start": start,
        "end": end,
        "count": count,
        "binary_sha256": binary["sha256"],
        "binary_size": binary["size"],
        "sidecar_path": sidecar_record["path"],
        "sidecar_root_kind": sidecar_record["root_kind"],
        "sidecar_sha256": sidecar_record["sha256"],
        # Preserve the original compact key for consumers written against the
        # first sidecar schema.
        "output_sha256": binary["sha256"],
    }


def _expected_shard_records(
    entries: list[tuple[int, int, Path, dict[str, Any], Any]],
    *,
    validation_root: Path,
    validation_root_kind: str,
) -> list[dict[str, Any]]:
    return [
        _shard_record(
            path,
            start,
            end,
            meta,
            validation_root=validation_root,
            validation_root_kind=validation_root_kind,
        )
        for start, end, path, meta, _lens in entries
    ]


def validate_merged_shards(
    merged_path: str | os.PathLike[str],
    shards: Iterable[str | os.PathLike[str]],
    *,
    validation_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Require a merged sidecar's ordered shard seal to match current shards."""

    merged = Path(merged_path)
    raw_shards = [Path(value) for value in shards]
    if not raw_shards:
        raise IntegrityError("merged-shard validation requires at least one shard")
    found = validate_sidecar(merged, kind="merged", validation_root=validation_root)
    root_metadata = copy.copy(found)
    root_metadata["_sidecar_path"] = sidecar_path(merged)
    root, root_kind = _metadata_root(root_metadata, validation_root)
    merged_lens = jlens.JacobianLens.load(str(merged))
    _validate_lens_shape(merged_lens, found, merged)
    entries: list[tuple[int, int, Path, dict[str, Any], Any]] = []
    for shard in raw_shards:
        meta = validate_sidecar(shard, kind="fit", validation_root=root)
        lens = jlens.JacobianLens.load(str(shard))
        _validate_lens_shape(lens, meta, shard)
        start, end = _range_from_meta(meta, shard)
        entries.append((start, end, shard, meta, lens))
    entries.sort(key=lambda entry: (entry[0], entry[1], str(entry[2])))
    for previous, current in zip(entries, entries[1:]):
        if current[0] != previous[1]:
            relation = "overlap" if current[0] < previous[1] else "gap"
            raise IntegrityError(
                f"current shard ranges have a {relation}: "
                f"{previous[1]}..{current[0]}"
            )
    if entries:
        covered = entries[-1][1] - entries[0][0]
        summed = sum(entry[3]["n_prompts"] for entry in entries)
        if summed != covered:
            raise IntegrityError(
                f"current shard counts do not cover merged range: {summed} != {covered}"
            )
        if found.get("start") != entries[0][0] or found.get("end") != entries[-1][1]:
            raise IntegrityError("merged lens range does not match current shards")
    expected = _expected_shard_records(
        entries,
        validation_root=root,
        validation_root_kind=root_kind,
    )
    found_records = found.get("shard_records")
    if found_records != expected:
        raise IntegrityError(f"merged lens shard_records do not match current shards: {merged}")
    if found.get("shard_records_sha256") != sha256_json(expected):
        raise IntegrityError(f"merged lens shard_records seal does not match current shards: {merged}")
    return found


def merge(args: Any) -> Any:
    """Verify contiguous non-overlapping shards and publish their weighted mean."""

    raw_shards = [Path(p) for p in getattr(args, "shards", [])]
    if not raw_shards:
        raise IntegrityError("merge requires at least one shard")
    canonical = [p.resolve() for p in raw_shards]
    if len(set(canonical)) != len(canonical):
        raise IntegrityError("duplicate shard path")
    out = Path(getattr(args, "out"))
    reject_symlink_path(out)
    reject_symlink_path(sidecar_path(out))
    explicit_root = getattr(args, "validation_root", None)
    if out.resolve() in set(canonical):
        raise IntegrityError("merge output must not overwrite an input shard")

    entries: list[tuple[int, int, Path, dict[str, Any], Any]] = []
    for path in raw_shards:
        meta = validate_sidecar(path, kind="fit", validation_root=explicit_root)
        target_virtual = meta.get("target_virtual")
        if not isinstance(target_virtual, int) or target_virtual <= 0:
            raise IntegrityError(f"invalid target_virtual in {path}")
        if meta.get("source_layers") != list(range(target_virtual)):
            raise IntegrityError(f"source_layers are not exact contiguous range in {path}")
        lens = jlens.JacobianLens.load(str(path))
        _validate_lens_shape(lens, meta, path)
        start, end = _range_from_meta(meta, path)
        entries.append((start, end, path, meta, lens))

    entries.sort(key=lambda entry: (entry[0], entry[1], str(entry[2])))
    baseline = entries[0][3]
    for _, _, path, meta, _ in entries[1:]:
        if identity_projection(meta) != identity_projection(baseline):
            raise IntegrityError(f"shard configuration/model/prompt/source mismatch: {path}")
    for previous, current in zip(entries, entries[1:]):
        if current[0] < previous[1]:
            raise IntegrityError(f"overlapping shard ranges: {previous[2]} and {current[2]}")
        if current[0] > previous[1]:
            raise IntegrityError(f"gap in shard ranges: {previous[1]}..{current[0]}")

    start, end = entries[0][0], entries[-1][1]
    root_metadata = copy.copy(entries[0][3])
    root_metadata["_sidecar_path"] = sidecar_path(entries[0][2])
    validation_root, validation_root_kind = _metadata_root(root_metadata, explicit_root)
    summed_count = sum(entry[3]["n_prompts"] for entry in entries)
    if summed_count != end - start:
        raise IntegrityError(
            f"merged prompt count mismatch: shards sum to {summed_count}, "
            f"covered range is {end - start}"
        )
    if out.exists():
        if not sidecar_path(out).is_file():
            raise IntegrityError(f"merge output exists without sidecar: {out}")
        found = validate_sidecar(out, kind="merged", validation_root=validation_root)
        found_lens = jlens.JacobianLens.load(str(out))
        _validate_lens_shape(found_lens, found, out)
        if (
            identity_projection(found) != identity_projection(baseline)
            or found.get("start") != start
            or found.get("end") != end
        ):
            raise IntegrityError(f"existing merge output has incompatible sidecar: {out}")
        expected_records = _expected_shard_records(
            entries,
            validation_root=validation_root,
            validation_root_kind=validation_root_kind,
        )
        if found.get("shard_records") != expected_records:
            raise IntegrityError(f"existing merge output is not sealed to current shards: {out}")
        if found.get("shard_records_sha256") != sha256_json(expected_records):
            raise IntegrityError(f"existing merge output shard seal mismatch: {out}")
        LOG.info("verified existing merged lens %s; nothing to do", out)
        return None
    if sidecar_path(out).exists():
        raise IntegrityError(f"stale merge sidecar without output: {sidecar_path(out)}")

    merged = jlens.JacobianLens.merge([entry[4] for entry in entries])
    metadata = copy.deepcopy(baseline)
    metadata["kind"] = "merged"
    metadata["start"], metadata["end"] = start, end
    metadata["n_requested"] = end - start
    metadata["prompt_slice"] = _merged_prompt_identity(
        baseline,
        start,
        end,
        validation_root=validation_root,
    )
    metadata["prompt_slice_sha256"] = metadata["prompt_slice"]["sha256"]
    merged_count = getattr(merged, "n_prompts", None)
    if not isinstance(merged_count, int) or isinstance(merged_count, bool) or merged_count <= 0:
        raise IntegrityError("merged lens has an invalid n_prompts count")
    if merged_count != summed_count or merged_count != end - start:
        raise IntegrityError(
            f"merged lens prompt count mismatch: n_prompts={merged_count}, "
            f"shards={summed_count}, range={end - start}"
        )
    metadata["n_prompts"] = merged_count
    metadata["n_fitted"] = merged_count
    metadata["shards"] = [
        _record_for_path(
            entry[2],
            root=validation_root,
            root_kind=validation_root_kind,
        )["path"]
        for entry in entries
    ]
    metadata["shard_records"] = _expected_shard_records(
        entries,
        validation_root=validation_root,
        validation_root_kind=validation_root_kind,
    )
    metadata["shard_records_sha256"] = sha256_json(metadata["shard_records"])
    metadata["validation"] = _context_record(validation_root, validation_root_kind)
    _strict_count_identity(metadata, out)
    _atomic_lens_save(merged, out)
    output = _record_for_path(
        out,
        root=validation_root,
        root_kind=validation_root_kind,
    )
    metadata["output"] = output
    metadata["output_sha256"] = output["sha256"]
    atomic_write_json(sidecar_path(out), metadata)
    LOG.info("merged %d shards, %d fitted prompts -> %s", len(entries), merged.n_prompts, out)
    return merged


def main(argv: Iterable[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--target-ut", type=int, required=True)
    fit_parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    fit_parser.add_argument("--start", type=int, default=0)
    fit_parser.add_argument("--end", type=int, default=100)
    fit_parser.add_argument("--dim-batch", type=int, default=8)
    fit_parser.add_argument("--max-seq-len", type=int, default=128)
    fit_parser.add_argument("--skip-first", type=int, default=SKIP_FIRST_N_POSITIONS)
    fit_parser.add_argument("--checkpoint-every", type=int, default=None)
    fit_parser.add_argument("--checkpoint")
    fit_parser.add_argument("--validation-root")
    fit_parser.add_argument("--validation-root-kind", choices=(PROJECT_ROOT_KIND, TEST_BUNDLE_ROOT_KIND))
    fit_parser.add_argument("--out", required=True)
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--out", required=True)
    merge_parser.add_argument("--validation-root")
    merge_parser.add_argument("shards", nargs="+")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        fit(args) if args.cmd == "fit" else merge(args)
    except (IntegrityError, ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    main()
