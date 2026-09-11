"""Publish fixed-path copies of the two sealed N100 estimator binaries."""
from __future__ import annotations
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from artifacts import canonical, json_save, record


def regular(root, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or any(part in ('', '.', '..') for part in rel.parts):
        raise ValueError('Invalid relative evidence path')
    path = root
    for part in rel.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('Linked evidence path')
    if not path.is_file():
        raise ValueError('Evidence is not a regular file')
    return path


def export_fit(root, destination):
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    text_paths = ['OWNER.json', 'LATEST.json', 'COMPLETE.json']
    pointers = {name: json.loads(regular(root, name).read_text()) for name in text_paths[1:]}
    binary_paths = {}
    owner = json.loads(regular(root, 'OWNER.json').read_text())
    if hashlib.sha256(canonical(owner['identity'])).hexdigest() != owner['fit_identity_sha256']:
        raise ValueError('Fit OWNER identity digest differs')
    for pointer_name, kind, binary, exported in (
        ('LATEST.json', 'checkpoint', 'state.pt', 'checkpoint.pt'),
        ('COMPLETE.json', 'final', 'lens.pt', 'lens.pt'),
    ):
        pointer = pointers[pointer_name]
        if (pointer['kind'] != kind or pointer['n_done'] != 100 or pointer['next_idx'] != 100
                or pointer['fit_identity_sha256'] != owner['fit_identity_sha256']):
            raise ValueError('Only a complete N100 generation can be published')
        generation = pointer['generation']
        seal_rel = generation + '/SEAL.json'
        seal_path = regular(root, seal_rel)
        if record(seal_path) != pointer['seal']:
            raise ValueError('Generation seal differs from its pointer')
        seal = json.loads(seal_path.read_text())
        if (seal['kind'] != kind or seal['n_done'] != 100
                or seal['fit_identity_sha256'] != pointer['fit_identity_sha256']
                or set(seal['files']) != {binary, 'metadata.json'}):
            raise ValueError('Unexpected generation seal')
        for name, expected in seal['files'].items():
            if record(regular(root, generation + '/' + name)) != expected:
                raise ValueError('Generation payload does not match its seal')
        original = regular(root, generation + '/' + binary)
        target = destination / exported
        os.link(original, target, follow_symlinks=False)
        with target.open('rb') as handle:
            os.fsync(handle.fileno())
        binary_paths[exported] = {'original_relative_path': generation + '/' + binary, 'record': record(target)}
        text_paths.extend([seal_rel, generation + '/metadata.json'])
    files = {}
    for relative in text_paths:
        path = regular(root, relative)
        raw = path.read_bytes()
        files[relative] = {'record': record(path), 'base64': base64.b64encode(raw).decode('ascii')}
    json_save(destination / 'evidence.json', {
        'schema': 'huginn_verification_fit_evidence.v1',
        'category': 'newly_rerun_estimator', 'n_prompts': 100,
        'files': files, 'binaries': binary_paths,
        'historical_final_bank_sha256': '7eddc849bca857a1406a901d8e4998b6bf90580b09f6ac6b326690ef378dc596',
        'original_fit_root_at_rerun': str(root),
    })


def export_checkpoint_evidence(root, destination):
    """Retain a sealed stopped-run checkpoint; never declare a complete fit."""
    root, destination = Path(root), Path(destination)
    owner_path, latest_path = regular(root, 'OWNER.json'), regular(root, 'LATEST.json')
    owner, pointer = json.loads(owner_path.read_text()), json.loads(latest_path.read_text())
    identity = owner['fit_identity_sha256']
    cursor = pointer['n_done']
    if (hashlib.sha256(canonical(owner['identity'])).hexdigest() != identity
            or pointer['kind'] != 'checkpoint' or type(cursor) is not int or not 0 <= cursor <= 100
            or pointer['next_idx'] != cursor or pointer['fit_identity_sha256'] != identity):
        raise ValueError('Invalid stopped-run checkpoint identity or cursor')
    generation = pointer['generation']
    seal_path = regular(root, generation + '/SEAL.json')
    if record(seal_path) != pointer['seal']:
        raise ValueError('Stopped-run checkpoint seal differs')
    seal = json.loads(seal_path.read_text())
    if (seal['kind'] != 'checkpoint' or seal['n_done'] != cursor
            or seal['fit_identity_sha256'] != identity or set(seal['files']) != {'state.pt', 'metadata.json'}):
        raise ValueError('Invalid stopped-run checkpoint seal')
    for name, expected in seal['files'].items():
        if record(regular(root, generation + '/' + name)) != expected:
            raise ValueError('Stopped-run checkpoint payload differs')
    destination.mkdir(parents=True, exist_ok=True)
    original = regular(root, generation + '/state.pt')
    target = destination / 'checkpoint.pt'
    if target.exists():
        if record(target) != record(original):
            raise ValueError('Existing exported checkpoint differs')
    else:
        os.link(original, target, follow_symlinks=False)
        with target.open('rb') as handle:
            os.fsync(handle.fileno())
    files = {}
    for relative in ('OWNER.json', 'LATEST.json', generation + '/SEAL.json', generation + '/metadata.json'):
        path = regular(root, relative)
        files[relative] = {'record': record(path), 'base64': base64.b64encode(path.read_bytes()).decode('ascii')}
    json_save(destination / 'evidence.json', {
        'schema': 'huginn_verification_checkpoint_evidence.v1', 'category': 'newly_rerun_estimator',
        'outcome': 'checkpoint_only', 'n_prompts_completed': cursor, 'n_prompts_required': 100,
        'complete_fit_accepted': False, 'files': files,
        'binaries': {'checkpoint.pt': {'original_relative_path': generation + '/state.pt', 'record': record(target)}},
        'original_fit_root_at_rerun': str(root),
    })
