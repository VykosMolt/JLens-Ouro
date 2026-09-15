#!/usr/bin/env python3
"""Copy the surviving evidence for the central Ouro claims into one hash-verified bundle.

  python build_bundle.py core           banks, accepted payload, model snapshot, every file < 50 MiB of both rounds
  python build_bundle.py verification   this pass's records (run after they are written)

Destination is on the same physical disk as the originals: this is a second copy, not a backup.
Private keys and credentials are excluded by name.
"""
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

RESEARCH = Path('/home/moloch/ouro_project/jacobian-lens/research')
R = RESEARCH / 'confirmation_2026-09-09'
REFIT = RESEARCH / 'refit_round_2026-09-07'
V = RESEARCH / 'verification_2026-09-11'
OUT = V / 'preservation/bundle'
FINAL = R / ('cloud_leases/attempt_06/handoff/accepted/ouro_confirmation_20260911_fixed160_native1/'
             'ff7c0769b2b0dbe2e53b120daf7f0729434fe65e54c013288771f5052ffd4cef')
SNAPSHOT = Path('/home/moloch/ouro_project/artifacts/hf_cache/hub/models--ByteDance--Ouro-2.6B/snapshots/'
                '1ed04250da1a9936042725d302e81c8fa2ab5abd')
SMALL = 50 * 2**20
SECRET_PARTS = ('/ssh/', 'id_ed25519', 'known_hosts', '.env', 'api_key', 'apikey', 'token')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def secret(path):
    text = str(path).lower()
    return any(part in text for part in SECRET_PARTS)


def plan_core():
    spec = json.loads((FINAL / 'results/run_spec.json').read_text())
    retained = json.loads((V / 'replay/retained_bank_paths.json').read_text())
    entries = []
    for worker_path, source in sorted(retained.items()):
        record = next(e['record'] for e in spec['bank_records'].values() if e['worker_path'] == worker_path)
        entries.append((Path(source), worker_path, record))
    payload = json.loads((FINAL / 'MANIFEST.json').read_text())['files']
    for path in sorted(FINAL.rglob('*')):
        if path.is_file():
            rel = path.relative_to(FINAL).as_posix()
            entries.append((path, 'accepted_final_payload/' + rel, payload.get(rel.removeprefix('results/'))))
    manifest = json.loads((V.parent / 'confirmation_2026-09-09/corrections/native_check_v1/bundle/legacy/model_manifest.json').read_text())
    for row in manifest['files']:
        entries.append((SNAPSHOT / row['path'], 'model_snapshot/' + row['path'], {'bytes': row['bytes'], 'sha256': row['sha256']}))
    for root, label in ((R, 'confirmation_2026-09-09'), (REFIT, 'refit_round_2026-09-07')):
        for path in sorted(root.rglob('*')):
            if path.is_file() and not path.is_symlink() and path.stat().st_size < SMALL and not secret(path) and '__pycache__' not in path.parts:
                entries.append((path, f'{label}/' + path.relative_to(root).as_posix(), None))
    return entries


def plan_verification():
    skip = ('preservation/bundle/', 'replay/local_', 'replay/propagation_', 'gpu_shape_v1/inputs/', 'gpu_shape_v1/local_rtx5070ti/layouts/')
    entries = []
    for path in sorted(V.rglob('*')):
        rel = path.relative_to(V).as_posix()
        if path.is_file() and not rel.startswith(skip) and '__pycache__' not in path.parts and path.stat().st_size < SMALL:
            entries.append((path, 'verification_2026-09-11/' + rel, None))
    for run in ('local_cuda_rtx5070ti', 'local_cpu_fp64', 'local_cuda_regenerated_states'):
        for path in sorted((V / 'replay' / run).glob('*.json')):
            entries.append((path, f'verification_2026-09-11/replay/{run}/{path.name}', None))
    for path in sorted((V / 'replay').glob('propagation_*/*/*/analysis.json')):
        entries.append((path, 'verification_2026-09-11/' + path.relative_to(V).as_posix(), None))
    return entries


def main():
    part = sys.argv[1]
    entries = {'core': plan_core, 'verification': plan_verification}[part]()
    manifest_path = OUT.parent / f'BUNDLE_MANIFEST_{part}.json'
    if manifest_path.exists():
        raise SystemExit('manifest exists; bundles are not overwritten')
    rows, problems = [], []
    for source, rel, expected in entries:
        target = OUT / rel
        if target.exists():
            raise SystemExit(f'bundle file exists: {rel}')
        before = {'bytes': source.stat().st_size, 'sha256': sha256(source)}
        if expected is not None and before != {'bytes': expected['bytes'], 'sha256': expected['sha256']}:
            problems.append({'path': str(source), 'expected': expected, 'actual': before})
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        after = {'bytes': target.stat().st_size, 'sha256': sha256(target)}
        if after != before:
            problems.append({'path': rel, 'source': before, 'copy': after})
        rows.append({'bundle_path': rel, 'source_path': str(source), **after, 'expected_record_matched': expected is not None and before == {'bytes': expected['bytes'], 'sha256': expected['sha256']}})
    with manifest_path.open('x') as handle:
        json.dump({'schema': 'preservation_bundle_manifest.v1', 'part': part, 'created_utc': datetime.now(timezone.utc).isoformat(),
                   'destination': str(OUT), 'same_physical_disk_as_sources': True, 'files': len(rows),
                   'bytes': sum(r['bytes'] for r in rows), 'problems': problems, 'script_sha256': sha256(__file__), 'entries': rows},
                  handle, indent=1, sort_keys=True)
        handle.write('\n')
    print(json.dumps({'part': part, 'files': len(rows), 'bytes': sum(r['bytes'] for r in rows), 'problems': len(problems)}))
    if problems:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
