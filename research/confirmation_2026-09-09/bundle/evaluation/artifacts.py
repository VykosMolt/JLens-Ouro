"""Atomic immutable payloads and manifest publication for this experiment."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()

def record(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return {'bytes': Path(path).stat().st_size, 'sha256': h.hexdigest()}

def atomic(path, write, *, replace=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            write(f); f.flush(); os.fsync(f.fileno())
        if replace:
            os.replace(tmp, path)
        else:
            os.link(tmp, path); os.unlink(tmp)
        d = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(d)
        finally: os.close(d)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def json_save(path, value, *, replace=False):
    atomic(path, lambda f: f.write(canonical(value)+b'\n'), replace=replace)

def torch_save(path, value):
    import torch
    atomic(path, lambda f: torch.save(value, f))

def npz_save(path, value):
    import numpy as np
    atomic(path, lambda f: np.savez_compressed(f, **value))

class Publisher:
    def __init__(self, work, binding, contract):
        self.work = Path(work); self.root = self.work/'results'
        self.binding, self.contract = binding, contract
        self.index = {'schema': 'confirmation_artifact_index.v1', 'binding': binding, 'manifests': []}
        if (self.work/'artifact_index.json').exists():
            raise ValueError('A new worker may not replace an old run')
    def status(self, phase, setup_complete=True, **details):
        json_save(self.work/'status.json', {'binding': self.binding, 'lease_name': self.binding['lease_name'],
                                          'setup_complete': setup_complete, 'phase': phase, **details}, replace=True)
    def publish(self, stage=None, outcome='complete'):
        paths = sorted(self.contract['files'] if stage is None else self.contract['stages'][stage])
        if outcome != 'complete': paths = [p for p in paths if (self.root/p).is_file()]
        manifest = {'schema': 'confirmation_artifact_manifest.v1', 'binding': self.binding,
                    'kind': 'final' if stage is None else 'stage', 'stage_id': stage, 'outcome': outcome,
                    'files': {p: record(self.root/p) for p in paths}}
        identifier = hashlib.sha256(canonical(manifest)).hexdigest()
        rel = f'manifests/{identifier}.json'
        json_save(self.work/rel, manifest)
        self.index['manifests'].append({'path': rel, 'record': record(self.work/rel)})
        json_save(self.work/'artifact_index.json', self.index, replace=True)
        return identifier
