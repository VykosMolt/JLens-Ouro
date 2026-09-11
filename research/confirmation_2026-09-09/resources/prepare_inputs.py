#!/usr/bin/env python3
"""Materialize the exact historical model snapshot and index verified banks."""
import hashlib
import json
from pathlib import Path
import shutil

ROUND = Path(__file__).resolve().parents[1]
OLD = ROUND.parent / 'refit_round_2026-09-07'
REV = '1ed04250da1a9936042725d302e81c8fa2ab5abd'
CACHES = [Path('/home/moloch/ouro_project/artifacts/hf_cache/hub/models--ByteDance--Ouro-2.6B/snapshots') / REV,
          Path('/home/moloch/.cache/huggingface/hub/models--ByteDance--Ouro-2.6B/snapshots') / REV]

def record(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}

def main():
    manifest = json.loads((OLD / 'deployment/model_manifest.json').read_text())
    out = ROUND / 'resources/model_snapshot' / REV
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for expected in manifest['files']:
        rel = expected['path']
        src = next((base / rel for base in CACHES if (base / rel).is_file()), None)
        if src is None:
            raise FileNotFoundError(rel)
        before = src.resolve().stat()
        actual = record(src)
        assert actual == {k: expected[k] for k in ('bytes', 'sha256')}, rel
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            temporary = dest.with_name(dest.name + '.partial')
            with src.open('rb') as reader, temporary.open('xb') as writer:
                shutil.copyfileobj(reader, writer, 8 * 1024 * 1024)
                writer.flush()
                import os
                os.fsync(writer.fileno())
            assert record(temporary) == actual
            temporary.rename(dest)
        else:
            assert not dest.is_symlink() and record(dest) == actual
        after = src.resolve().stat()
        assert (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        records.append({'path': rel, 'original': str(src), 'actual': actual})
        print('VERIFIED MODEL', rel, flush=True)
    ledger = json.loads((ROUND / 'artifacts/RECOVERY_LEDGER.json').read_text())
    banks = {}
    for key, row in ledger['binaries'].items():
        if '/final/' not in key or row['current_status'] not in ('complete_hash_verified', 'reconstructed_historical_bytes_verified'):
            continue
        if row['current_status'] == 'complete_hash_verified':
            path = Path(next(x['path'] for x in row['retained_copies'] if x['status'] == 'complete_hash_verified'))
        else:
            path = ROUND / 'artifacts/reconstructed/ouro/fit_02/lens.pt'
        assert record(path) == row['expected'], key
        banks[key] = {'path': str(path), 'record': row['expected'], 'category': row['current_status']}
        print('VERIFIED BANK', key, flush=True)
    result = {'schema': 'confirmation_materialized_inputs.v1', 'model': {'snapshot': str(out), 'manifest': record(OLD / 'deployment/model_manifest.json'), 'files': records}, 'banks': banks}
    dest = ROUND / 'resources/materialized_inputs.json'
    with dest.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write('\n')

if __name__ == '__main__':
    main()
