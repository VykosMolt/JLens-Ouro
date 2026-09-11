#!/usr/bin/env python3
"""Copy-only, versioned, hash-verified archive of surviving J-Lens evidence to the external SSD.

Order: preservation bundle first (verified against its own manifests), then every other root.
No move, mirror-delete or source modification. Secrets are excluded by exact path and flagged.
Outputs (also mirrored to the closeout dir): ARCHIVE_MANIFEST.json, DEDUP_TABLE.json, SECRETS_FLAGGED.json,
EXCLUSIONS.json, DEPENDENCIES.json, ACCEPTANCE_RECEIPT_partial.json.
"""
import hashlib, json, os, subprocess, sys, time, stat, shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

DEST = Path('/run/media/moloch/ARCH_BACKUP/JLENS_COLD_ARCHIVE_20260911')
TREE = DEST / 'tree'
LOCAL = Path('/home/moloch/jacobian-lens/research/closeout_2026-09-11/archive')
ROOTS = [
    '/home/moloch/jacobian-lens/research/verification_2026-09-11/preservation',   # bundle first
    '/home/moloch/jacobian-lens',
    '/home/moloch/ouro_project/artifacts/jlens',
    '/home/moloch/ouro_project/artifacts/hf_cache/hub/models--ByteDance--Ouro-2.6B',
    '/home/moloch/ouro_project/artifacts/hf_cache/hub/datasets--Salesforce--wikitext',
    '/home/moloch/ouro_project/src/ouro_jlens',
    '/home/moloch/ouro_project/docs/jlens',
    '/home/moloch/.cache/huggingface/hub/models--Vykos--ouro-jlens-results',
]
SECRET_FILES = {'/home/moloch/jacobian-lens/research/confirmation_2026-09-09/resources/ssh/id_ed25519'}
EXCLUDE_DIR_NAMES = {'__pycache__', '.pytest_cache', '.ruff_cache'}
EXCLUDE_PREFIXES = ['/home/moloch/jacobian-lens/research/closeout_2026-09-11']  # this round: added later as a manifested supplement
EXCLUDE_SUFFIXES = ('.pyc',)

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()

def excluded(p):
    s = str(p)
    if s in SECRET_FILES: return 'secret'
    if any(s == pre or s.startswith(pre + '/') for pre in EXCLUDE_PREFIXES): return 'this_round_supplement_later'
    if any(part in EXCLUDE_DIR_NAMES for part in p.parts): return 'bytecode_cache'
    if s.endswith(EXCLUDE_SUFFIXES): return 'bytecode_cache'
    return None

def walk(root):
    """Yield (path, kind) for regular files and symlinks under root (lstat; do not follow links)."""
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIR_NAMES and not excluded(Path(dirpath) / d)]
        for name in filenames:
            p = Path(dirpath) / name
            st = os.lstat(p)
            if stat.S_ISLNK(st.st_mode): yield p, 'symlink', st
            elif stat.S_ISREG(st.st_mode): yield p, 'file', st
            else: yield p, 'other', st

def inventory(roots):
    seen, entries, excl = set(), [], []
    for root in roots:
        for p, kind, st in walk(root):
            if p in seen: continue
            seen.add(p)
            reason = excluded(p)
            if reason:
                excl.append({'path': str(p), 'reason': reason, 'bytes': st.st_size, 'kind': kind}); continue
            e = {'source_path': str(p), 'kind': kind, 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns,
                 'inode': st.st_ino, 'nlink': st.st_nlink, 'dev': st.st_dev}
            if kind == 'symlink': e['link_target'] = os.readlink(p)
            entries.append(e)
    return entries, excl

def hash_all(entries, key):
    files = [e for e in entries if e['kind'] == 'file']
    with ThreadPoolExecutor(8) as ex:
        for e, h in zip(files, ex.map(lambda e: sha256(e[key]), files)):
            e['sha256' if key == 'source_path' else 'dest_sha256'] = h

def run(cmd, **kw):
    print('+', ' '.join(cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)

def rsync_root(root):
    # -R keeps the absolute path under TREE; -a preserves symlinks/perms/mtimes; -H preserves hard links.
    cmd = ['rsync', '-aHR', '--info=stats1', '--exclude=__pycache__', '--exclude=*.pyc', '--exclude=.pytest_cache', '--exclude=.ruff_cache']
    for s in SECRET_FILES: cmd.append('--exclude=' + s)
    for pre in EXCLUDE_PREFIXES: cmd.append('--exclude=' + pre)
    cmd += [root.rstrip('/'), str(TREE) + '/']
    r = run(cmd); print(r.stdout[-1200:], flush=True)

def destination_identity():
    out = {}
    out['findmnt'] = run(['findmnt', '-J', '-o', 'TARGET,SOURCE,FSTYPE,OPTIONS,UUID,LABEL', str(DEST.parent)]).stdout
    out['lsblk'] = run(['lsblk', '-J', '-o', 'NAME,SIZE,TYPE,FSTYPE,UUID,MOUNTPOINT,MODEL,SERIAL,TRAN,VENDOR', '/dev/sda']).stdout
    out['source_disk_lsblk'] = run(['lsblk', '-J', '-o', 'NAME,SIZE,TYPE,FSTYPE,UUID,MOUNTPOINT,MODEL,SERIAL,TRAN', '/dev/nvme0n1']).stdout
    out['df_dest'] = run(['df', '-B1', '--output=source,fstype,size,used,avail,target', str(DEST.parent)]).stdout
    out['df_source'] = run(['df', '-B1', '--output=source,fstype,size,used,avail,target', '/home/moloch']).stdout
    out['st_dev_dest'] = os.stat(DEST.parent).st_dev; out['st_dev_source'] = os.stat('/home/moloch').st_dev
    out['separate_physical_device'] = out['st_dev_dest'] != out['st_dev_source']
    return out

def git_records():
    g = DEST / 'git'; g.mkdir(parents=True, exist_ok=True)
    recs = {}
    for name, repo in (('jacobian-lens', '/home/moloch/jacobian-lens'), ('ouro_project', '/home/moloch/ouro_project')):
        head = run(['git', '-C', repo, 'rev-parse', 'HEAD']).stdout.strip()
        run(['git', '-C', repo, 'bundle', 'create', str(g / f'{name}.bundle'), '--all'])
        (g / f'{name}_status.txt').write_text(run(['git', '-C', repo, 'status', '--porcelain=v1', '--untracked-files=all']).stdout)
        (g / f'{name}_tracked_diff.patch').write_text(run(['git', '-C', repo, 'diff', 'HEAD']).stdout)
        (g / f'{name}_log.txt').write_text(run(['git', '-C', repo, 'log', '--format=%H %ci %s', '-n', '50']).stdout)
        recs[name] = {'repo': repo, 'head': head, 'bundle': str(g / f'{name}.bundle'),
                      'bundle_sha256': sha256(g / f'{name}.bundle'), 'bundle_verify': run(['git', 'bundle', 'verify', str(g / f'{name}.bundle')]).stdout.strip()[-400:]}
    return recs

def dependencies():
    py = '/home/moloch/ouro_project/venv/bin/python'
    dep = {'venv_python': py, 'python_version': run([py, '-c', 'import sys;print(sys.version)']).stdout.strip(),
           'pip_freeze': run([py, '-m', 'pip', 'freeze']).stdout.splitlines(),
           'venv_not_copied_reason': 'Re-creatable from research/confirmation_2026-09-09/corrections/native_check_v1/bundle/legacy/requirements.lock and deployment/ENVIRONMENT.md; 8.0 GB; the interpreter is /usr/bin/python3.14.',
           'nvidia_smi': subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'], text=True, capture_output=True).stdout,
           'external_public_models_not_copied': []}
    hub = Path('/home/moloch/ouro_project/artifacts/hf_cache/hub')
    for d in sorted(hub.iterdir()):
        if d.name in ('models--ByteDance--Ouro-2.6B', 'datasets--Salesforce--wikitext'): continue
        snaps = sorted(p.name for p in (d / 'snapshots').iterdir()) if (d / 'snapshots').exists() else []
        dep['external_public_models_not_copied'].append({'cache_dir': str(d), 'snapshots': snaps,
            'role': 'Huginn pilot model (evaluated on the pod)' if 'huginn' in d.name else 'application-era checkpoint comparison / unrelated project use',
            'bytes': sum(f.stat().st_size for f in d.rglob('*') if f.is_file() and not f.is_symlink())})
    dep['private_hub_repo_not_retrievable_here'] = {'repo_id': 'Vykos/ouro-jlens-results', 'reason': 'no Hugging Face credential on this machine (HF_TOKEN unset, no token file)',
        'recorded_remote_files': 'see PRESERVATION_LEDGER.json unresolved (19 files, 15 distinct contents) and ouro_project/artifacts/jlens/retrieved/*.receipts'}
    return dep

def main():
    t0 = time.time(); log = {'started_utc': datetime.now(timezone.utc).isoformat()}
    if DEST.exists() and any(DEST.iterdir()): raise SystemExit(f'destination exists and is not empty: {DEST}')
    DEST.mkdir(parents=True, exist_ok=True); TREE.mkdir(exist_ok=True)
    log['destination_identity_before'] = destination_identity()
    print('inventory + source hashing ...', flush=True)
    entries, excl = inventory(ROOTS)
    hash_all(entries, 'source_path')
    log['source_hash_seconds'] = time.time() - t0
    print(f'inventory: {len(entries)} entries, {sum(e["bytes"] for e in entries if e["kind"]=="file")/1e9:.2f} GB, excluded {len(excl)}', flush=True)
    # copy: bundle first, then the rest
    for root in ROOTS: rsync_root(root)
    log['git'] = git_records()
    (DEST / 'DEPENDENCIES.json').write_text(json.dumps(dependencies(), indent=1))
    run(['sync', '-f', str(DEST)])
    # destination verification: independent read + hash of every copied byte
    print('destination hashing ...', flush=True)
    for e in entries:
        e['archive_path'] = str(TREE / e['source_path'].lstrip('/'))
    hash_all(entries, 'archive_path')
    problems = []
    for e in entries:
        ap = Path(e['archive_path'])
        if e['kind'] == 'file':
            if not ap.is_file() or ap.is_symlink(): problems.append({'path': e['source_path'], 'problem': 'missing_or_link_at_destination'}); continue
            if e['dest_sha256'] != e['sha256'] or ap.stat().st_size != e['bytes']: problems.append({'path': e['source_path'], 'problem': 'hash_or_size_mismatch'})
            st = os.lstat(e['source_path'])
            if st.st_size != e['bytes'] or st.st_mtime_ns != e['mtime_ns']: problems.append({'path': e['source_path'], 'problem': 'source_changed_during_transfer'})
        elif e['kind'] == 'symlink':
            if not ap.is_symlink() or os.readlink(ap) != e['link_target']: problems.append({'path': e['source_path'], 'problem': 'symlink_not_preserved'})
            e['link_resolves_inside_archive'] = os.path.exists(ap)
        else:
            problems.append({'path': e['source_path'], 'problem': f'unsupported kind {e["kind"]}'})
    dedup = {}
    for e in entries:
        if e['kind'] == 'file': dedup.setdefault(e['sha256'], []).append(e['source_path'])
    dedup = {h: ps for h, ps in dedup.items() if len(ps) > 1}
    secrets = [{'path': p, 'bytes': os.stat(p).st_size, 'sha256': sha256(p), 'handling': 'excluded from the archive and from every package; protected handling by the owner'} for p in sorted(SECRET_FILES) if os.path.exists(p)]
    manifest = {'schema': 'jlens_cold_archive_manifest.v1', 'created_utc': datetime.now(timezone.utc).isoformat(), 'destination': str(DEST),
                'roots': ROOTS, 'entries': entries, 'files': sum(e['kind']=='file' for e in entries), 'symlinks': sum(e['kind']=='symlink' for e in entries),
                'bytes': sum(e['bytes'] for e in entries if e['kind']=='file'), 'problems': problems, 'script_sha256': sha256(__file__)}
    (DEST / 'ARCHIVE_MANIFEST.json').write_text(json.dumps(manifest, indent=1))
    (DEST / 'DEDUP_TABLE.json').write_text(json.dumps({'note': 'identical-content groups inside the archive; every original path was copied (no deduplication applied)', 'groups': dedup}, indent=1))
    (DEST / 'SECRETS_FLAGGED.json').write_text(json.dumps(secrets, indent=1))
    (DEST / 'EXCLUSIONS.json').write_text(json.dumps({'excluded': excl, 'rules': {'secret_files': sorted(SECRET_FILES), 'dir_names': sorted(EXCLUDE_DIR_NAMES), 'prefixes': EXCLUDE_PREFIXES, 'suffixes': list(EXCLUDE_SUFFIXES)}}, indent=1))
    run(['sync', '-f', str(DEST)])
    log['destination_identity_after'] = destination_identity()
    log.update({'files': manifest['files'], 'symlinks': manifest['symlinks'], 'bytes': manifest['bytes'], 'problems': len(problems),
                'manifest_sha256': sha256(DEST / 'ARCHIVE_MANIFEST.json'), 'dedup_groups': len(dedup), 'finished_utc': datetime.now(timezone.utc).isoformat(),
                'elapsed_seconds': time.time() - t0, 'status': 'copied_and_hash_verified' if not problems else 'PROBLEMS'})
    (DEST / 'ACCEPTANCE_RECEIPT_partial.json').write_text(json.dumps(log, indent=1))
    LOCAL.mkdir(parents=True, exist_ok=True)
    for name in ('ARCHIVE_MANIFEST.json', 'DEDUP_TABLE.json', 'SECRETS_FLAGGED.json', 'EXCLUSIONS.json', 'DEPENDENCIES.json', 'ACCEPTANCE_RECEIPT_partial.json'):
        shutil.copyfile(DEST / name, LOCAL / name)
    print(json.dumps({k: log[k] for k in ('files', 'symlinks', 'bytes', 'problems', 'manifest_sha256', 'status', 'elapsed_seconds')}, indent=1), flush=True)
    if problems: sys.exit(1)

if __name__ == '__main__':
    main()
