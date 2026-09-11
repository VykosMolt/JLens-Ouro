"""Small, dependency-light helpers for durable JLens evidence.

The fitting and evaluation commands write large binary files and JSON
sidecars.  A successful process must leave either the complete file or no
file that can be mistaken for a complete result.  This module centralises the
few rules needed by those commands:

* all replacement writes happen in the destination directory and use
  ``os.replace``;
* hashes are SHA-256 over the bytes that are actually on disk;
* JSON digests use one canonical representation so a prompt slice has a
  stable identity across processes.

The module intentionally has no model or cloud dependencies.  It is also
used by lightweight tests and by report/probe code that only needs to record
file identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = 1


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of *data*."""

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# A short alias is useful at call sites which deal in records rather than
# arbitrary byte strings.  Keep the public name explicit as well.
hash_file = sha256_file


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON with deterministic separators, ordering, and UTF-8."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    """Hash the canonical JSON representation of *value*."""

    return sha256_bytes(canonical_json_bytes(value))


def prompt_slice_sha256(prompts: list[str] | tuple[str, ...]) -> str:
    """Hash an ordered prompt slice, including its boundaries implicitly."""

    return sha256_json(list(prompts))


def aggregate_sha256(values: Mapping[str, str] | list[str] | tuple[str, ...]) -> str:
    """Hash a deterministic collection of already-computed digests.

    Mappings are sorted by key; sequences retain their order.  This is used
    for source/generator manifests where the individual file records remain
    useful to a reviewer but a compact identity is convenient in a sidecar.
    """

    if isinstance(values, Mapping):
        payload: Any = {str(k): str(v) for k, v in sorted(values.items(), key=lambda item: str(item[0]))}
    else:
        payload = [str(v) for v in values]
    return sha256_json(payload)


def file_record(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return the byte identity of an existing file.

    ``path`` is preserved as supplied (rather than silently making an
    absolute path) so sidecars remain readable when a project is relocated.
    The file is opened and hashed before the record is returned; a missing or
    unreadable file therefore fails the producing command instead of creating
    an unverifiable sidecar.
    """

    candidate = Path(path)
    stat = candidate.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "sha256": sha256_file(candidate),
    }


def _absolute_path(path: Path) -> Path:
    """Make a lexical absolute path without resolving symlinks."""

    # ``Path.resolve`` would erase the very symlink components this module is
    # required to reject.  ``abspath`` only normalizes ``.``/``..`` and uses
    # the process cwd, so the lstat walk below still observes links.
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: Path, *, include_leaf: bool = True) -> None:
    """Reject every existing symlink in a lexical path.

    ``Path.is_dir`` and ``Path.exists`` follow links (and miss dangling
    links), so each component is inspected with ``lstat``.  Missing trailing
    components are allowed; they are created by ``_prepare_destination``.
    """

    absolute = _absolute_path(path)
    current = Path(absolute.anchor)
    parts = absolute.parts[1:]
    last = len(parts) - 1
    for index, part in enumerate(parts):
        current /= part
        if not include_leaf and index == last:
            break
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"refusing symlink path component: {current}")


def _prepare_destination(path: Path) -> Path:
    """Validate and create a destination parent without following links."""

    destination = _absolute_path(path)
    # Validate before mkdir so an existing linked parent can never be
    # silently accepted.  Validate again after mkdir to cover a directory
    # that was created between the first walk and mkdir.
    _reject_symlink_components(destination, include_leaf=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(destination.parent)
    _reject_symlink_components(destination)
    return destination


def _temporary_identity(fd: int, temporary: Path) -> tuple[int, int]:
    """Validate and return the device/inode owned by an open temp fd."""

    descriptor_stat = os.fstat(fd)
    path_stat = os.lstat(temporary)
    if not stat.S_ISREG(descriptor_stat.st_mode) or not stat.S_ISREG(
        path_stat.st_mode
    ):
        raise OSError(f"temporary evidence path is not a regular file: {temporary}")
    identity = (int(descriptor_stat.st_dev), int(descriptor_stat.st_ino))
    path_identity = (int(path_stat.st_dev), int(path_stat.st_ino))
    if identity != path_identity:
        raise OSError(f"temporary evidence path was replaced: {temporary}")
    return identity


def _unlink_owned_temp(
    fd: int, temporary: Path, identity: tuple[int, int] | None = None
) -> None:
    """Remove a temp pathname only while it still names our inode."""

    try:
        path_stat = os.lstat(temporary)
    except OSError:
        return
    if identity is None:
        try:
            descriptor_stat = os.fstat(fd)
        except OSError:
            return
        identity = (int(descriptor_stat.st_dev), int(descriptor_stat.st_ino))
    path_identity = (int(path_stat.st_dev), int(path_stat.st_ino))
    if path_identity == identity:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _close_fd(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass


def _temporary_path(path: Path, suffix: str = ".tmp") -> tuple[int, Path]:
    destination = _prepare_destination(path)
    fd, raw = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=suffix, dir=str(destination.parent)
    )
    temporary = Path(raw)
    try:
        _temporary_identity(fd, temporary)
    except BaseException:
        # Do not unlink an attacker-controlled replacement (especially a
        # symlink); only the inode returned by mkstemp is ours to remove.
        try:
            descriptor_stat = os.fstat(fd)
            identity = (int(descriptor_stat.st_dev), int(descriptor_stat.st_ino))
        except OSError:
            identity = None
        _unlink_owned_temp(fd, temporary, identity)
        _close_fd(fd)
        raise
    return fd, temporary


def _open_parent_dir(path: Path) -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags)


def _replace_complete(fd: int, temporary: Path, destination: Path) -> None:
    """Flush and atomically install the inode owned by *fd*.

    The temp fd remains open for the complete operation.  In particular, no
    serialization step is allowed to reopen ``temporary`` by pathname.  The
    source and destination names are passed relative to an opened parent
    directory, so a parent-path swap after validation cannot redirect the
    rename to another directory.
    """

    destination = _prepare_destination(destination)
    identity = _temporary_identity(fd, temporary)
    parent_fd = _open_parent_dir(destination.parent)
    try:
        # Re-check after obtaining the directory handle; a test or attacker
        # may have swapped the temporary pathname while the payload was being
        # serialized.  A symlink or different inode fails closed.
        if _temporary_identity(fd, temporary) != identity:
            raise OSError(f"temporary evidence path was replaced: {temporary}")
        os.fsync(fd)
        os.replace(
            temporary.name,
            destination.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.fsync(parent_fd)
    finally:
        _close_fd(parent_fd)


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    """Atomically write bytes to *path*.

    The temporary file is removed on every failure.  ``os.replace`` is used
    only after the complete payload has been written and flushed.
    """

    destination = Path(path)
    fd, temporary = _temporary_path(destination)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            _replace_complete(fd, temporary, _absolute_path(destination))
    except BaseException:
        _unlink_owned_temp(fd, temporary)
        raise
    finally:
        _close_fd(fd)


def atomic_write_text(
    path: str | os.PathLike[str], value: str, *, encoding: str = "utf-8"
) -> None:
    """Atomically write text to *path* using *encoding*."""

    atomic_write_bytes(path, value.encode(encoding))


def atomic_write_json(
    path: str | os.PathLike[str], payload: Any, *, indent: int | None = 1
) -> None:
    """Atomically write a JSON document encoded as UTF-8."""

    text = json.dumps(
        payload,
        indent=indent,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )
    atomic_write_bytes(path, (text + "\n").encode("utf-8"))


def atomic_savez(path: str | os.PathLike[str], **arrays: Any) -> None:
    """Atomically write a compressed NumPy ``.npz`` archive.

    NumPy appends ``.npz`` when given a filename without that suffix, so a
    named temporary file is opened explicitly to make the replacement path
    unambiguous.
    """

    import numpy as np

    destination = Path(path)
    fd, temporary = _temporary_path(destination, suffix=".npz")
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            _replace_complete(fd, temporary, _absolute_path(destination))
    except BaseException:
        _unlink_owned_temp(fd, temporary)
        raise
    finally:
        _close_fd(fd)


def atomic_torch_save(obj: Any, path: str | os.PathLike[str]) -> None:
    """Atomically serialize a PyTorch object without importing torch eagerly."""

    import torch

    destination = Path(path)
    fd, temporary = _temporary_path(destination, suffix=".pt")
    try:
        # torch.save accepts a seekable binary file object.  Keeping the
        # mkstemp descriptor open means a swapped temp pathname cannot divert
        # serialization into an attacker-selected file.
        with os.fdopen(fd, "w+b", closefd=False) as handle:
            torch.save(obj, handle)
            handle.flush()
            _replace_complete(fd, temporary, _absolute_path(destination))
    except BaseException:
        _unlink_owned_temp(fd, temporary)
        raise
    finally:
        _close_fd(fd)


def source_manifest(paths: list[str | os.PathLike[str]] | tuple[str | os.PathLike[str], ...]) -> dict[str, Any]:
    """Record source files and a stable aggregate identity."""

    records = [file_record(path) for path in paths]
    by_path = {record["path"]: record["sha256"] for record in records}
    return {
        "files": records,
        "sha256": aggregate_sha256(by_path),
    }


__all__ = [
    "SCHEMA_VERSION",
    "aggregate_sha256",
    "atomic_savez",
    "atomic_torch_save",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "canonical_json_bytes",
    "file_record",
    "hash_file",
    "prompt_slice_sha256",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "source_manifest",
]
