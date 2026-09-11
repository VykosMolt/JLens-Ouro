"""Shared paths and helpers for the acceptance tests. Run every test with python -B."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
DIAGNOSTIC = HERE.parent
ROUND = DIAGNOSTIC.parents[1]
BUNDLE = DIAGNOSTIC / 'bundle'
PARENT_BUNDLE = ROUND / 'corrections/precision_v1/bundle'
LEDGER = ROUND / 'cloud_leases'
SNAPSHOT = ROUND / 'resources/model_snapshot/1ed04250da1a9936042725d302e81c8fa2ab5abd'
EVIDENCE = DIAGNOSTIC / 'evidence'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def record(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 24), b''):
            digest.update(block)
    return {'bytes': Path(path).stat().st_size, 'sha256': digest.hexdigest()}


def tree(root):
    return {p.relative_to(root).as_posix(): record(p) for p in sorted(Path(root).rglob('*')) if not p.is_dir()}


def copy_bundle(destination):
    """Copy the bundle byte-identically, so test imports never write beside frozen files."""
    shutil.copytree(BUNDLE, destination)
    files = tree(BUNDLE)
    require(tree(destination) == files, 'bundle copy differs from the bundle')
    return files


def empty_directory(path):
    path = Path(path).resolve()
    require(path.is_dir() and not any(path.iterdir()), '--work must be an existing empty directory')
    return path


def save_evidence(name, value):
    EVIDENCE.mkdir(exist_ok=True)
    (EVIDENCE / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
