#!/usr/bin/env python3
"""User-approved deletion (2026-09-11) of the two pure-duplicate sets on the laptop:
  1. research/verification_2026-09-11/preservation/bundle/**  (same-disk copy of banks, payload, model snapshot, small files)
  2. research/confirmation_2026-09-09/resources/input_chunks/**/*.part  (64 MiB slices of the fit01/fit02 banks)
Per file: source exists, re-hashed == archive manifest source hash == recorded destination hash; the SSD copy exists and is
re-hashed now; only then os.remove. Records (MANIFEST.json, ledgers, scripts) are kept. Empty directories left under bundle/
are removed with rmdir only. A deletion manifest is written locally and copied to the SSD supplement."""
import json, os, hashlib, subprocess, sys, shutil
from datetime import datetime, timezone
from pathlib import Path
X = Path('/home/moloch/jacobian-lens/research/closeout_2026-09-11/archive'); DEST = Path('/run/media/moloch/ARCH_BACKUP/JLENS_COLD_ARCHIVE_20260911')
BUNDLE = '/home/moloch/jacobian-lens/research/verification_2026-09-11/preservation/bundle/'; CHUNKS = '/home/moloch/jacobian-lens/research/confirmation_2026-09-09/resources/input_chunks/'
UUID = '36df468b-3bf9-41a9-ace8-edccfb292434'
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()
# destination identity + manifest integrity
uuid = subprocess.run(['lsblk', '-no', 'UUID', subprocess.run(['findmnt', '-rno', 'SOURCE', str(DEST.parent)], capture_output=True, text=True).stdout.strip()], capture_output=True, text=True).stdout.strip()
receipt = json.load(open(X / 'ACCEPTANCE_RECEIPT.json'))
assert uuid == UUID, f'destination UUID {uuid} != {UUID}'
assert sha(DEST / 'ARCHIVE_MANIFEST.json') == receipt['manifest_sha256'], 'archive manifest hash changed'
man = json.load(open(DEST / 'ARCHIVE_MANIFEST.json'))
targets = [e for e in man['entries'] if e['kind'] == 'file' and (e['source_path'].startswith(BUNDLE) or (e['source_path'].startswith(CHUNKS) and e['source_path'].endswith('.part')))]
df0 = shutil.disk_usage('/home/moloch').free
deleted, skipped = [], []
for i, e in enumerate(targets):
    sp, ap = e['source_path'], e['archive_path']
    try:
        if not os.path.isfile(sp) or os.path.islink(sp): skipped.append({'path': sp, 'reason': 'missing or link'}); continue
        st = os.stat(sp)
        if st.st_size != e['bytes'] or sha(sp) != e['sha256'] or e['sha256'] != e.get('dest_sha256'): skipped.append({'path': sp, 'reason': 'source hash/size differs from archive manifest'}); continue
        if not os.path.isfile(ap) or os.path.getsize(ap) != e['bytes'] or sha(ap) != e['sha256']: skipped.append({'path': sp, 'reason': 'SSD copy missing or hash mismatch'}); continue
        os.remove(sp)
        deleted.append({'path': sp, 'bytes': e['bytes'], 'sha256': e['sha256'], 'archive_path': ap, 'deleted_utc': datetime.now(timezone.utc).isoformat()})
    except Exception as ex: skipped.append({'path': sp, 'reason': f'{type(ex).__name__}: {ex}'})
    if i % 500 == 0: print(f'{i}/{len(targets)} checked, {len(deleted)} deleted', flush=True)
# remove now-empty directories under bundle/ only (rmdir refuses non-empty)
removed_dirs = []
for dp, dn, fn in sorted(os.walk(BUNDLE, topdown=False)):
    try: os.rmdir(dp); removed_dirs.append(dp)
    except OSError: pass
df1 = shutil.disk_usage('/home/moloch').free
out = {'schema': 'jlens_user_approved_duplicate_deletion.v1', 'approved_by_user_utc': '2026-09-11 (chat instruction: purge the ~22 GB of duplicates)', 'executed_utc': datetime.now(timezone.utc).isoformat(),
       'destination_uuid': uuid, 'archive_manifest_sha256': receipt['manifest_sha256'], 'scope': [BUNDLE + '**', CHUNKS + '**/*.part'], 'kept_records': 'BUNDLE_MANIFEST_*.json, PRESERVATION_LEDGER.json, RESTORATION_CHECK.json, scripts, input_chunks MANIFEST.json files',
       'targets': len(targets), 'deleted_files': deleted, 'deleted_bytes': sum(d['bytes'] for d in deleted), 'skipped': skipped, 'removed_empty_dirs': removed_dirs, 'free_bytes_before': df0, 'free_bytes_after': df1,
       'copies_remaining': 'SSD archive (verified) plus, for every bundle file, the original at its source path on the laptop; the .part chunks remain only on the SSD (and are derivable from the retained banks)'}
(X / 'DELETION_MANIFEST_2026-09-11.json').write_text(json.dumps(out, indent=1))
sup = DEST / 'supplement_closeout_2026-09-11/closeout_2026-09-11/archive'; sup.mkdir(parents=True, exist_ok=True); shutil.copyfile(X / 'DELETION_MANIFEST_2026-09-11.json', sup / 'DELETION_MANIFEST_2026-09-11.json'); subprocess.run(['sync', '-f', str(DEST)])
print(json.dumps({'targets': len(targets), 'deleted': len(deleted), 'deleted_GB': round(out['deleted_bytes'] / 1e9, 2), 'skipped': len(skipped), 'freed_GB': round((df1 - df0) / 1e9, 2), 'removed_empty_dirs': len(removed_dirs)}))
if skipped: print('SKIPPED:', json.dumps(skipped[:10], indent=1))
