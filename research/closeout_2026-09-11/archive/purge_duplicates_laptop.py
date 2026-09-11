#!/usr/bin/env python3
"""User-approved deletion (2026-09-11, 'purge it from the device') of the two same-disk duplicate sets on the laptop.
SSD not attached at execution: verification is against the laptop originals instead (the SSD copies were hash-verified at
archive time; see ACCEPTANCE_RECEIPT.json). Per file:
  bundle copy -> must hash-match BOTH the archive manifest AND its original at the recorded source path (which stays);
  bank chunk  -> must hash-match the archive manifest AND be byte-identical to the slice of the retained bank at its offset.
Anything failing a check is skipped. Records (BUNDLE_MANIFEST_*.json, ledgers, scripts, chunk MANIFEST.json) are kept."""
import json, os, hashlib, shutil
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
X = Path('/home/moloch/jacobian-lens/research/closeout_2026-09-11/archive'); V = Path('/home/moloch/jacobian-lens/research/verification_2026-09-11')
BUNDLE = str(V / 'preservation/bundle') + '/'; CHUNKS = '/home/moloch/jacobian-lens/research/confirmation_2026-09-09/resources/input_chunks/'
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()
man = json.load(open(X / 'ARCHIVE_MANIFEST.json')); receipt = json.load(open(X / 'ACCEPTANCE_RECEIPT.json'))
arch = {e['source_path']: e for e in man['entries'] if e['kind'] == 'file'}
# bundle_path -> original source (from the preservation bundle manifests)
orig = {}
for part in ('core', 'verification'):
    for e in json.load(open(V / f'preservation/BUNDLE_MANIFEST_{part}.json'))['entries']: orig[str(V / 'preservation/bundle' / e['bundle_path'])] = e
banks = {'90f01f6afd83512feba4b1c9871052a9ee85de6cb7ab68bdb59d97c7c9d4b52f': '/home/moloch/jacobian-lens/research/refit_round_2026-09-07/cloud_leases/attempt_06/prefetch_v3_20260909T134030Z_649dff12/staging/ouro/fit_01/final/cursor_000100_e090a066bc3b4630891f195e7e7d1d29/lens.pt',
         '101f31db6aa56d97fbae7ecb3e2f241ef1805000484d0e22f5eba5fc0b9acde8': '/home/moloch/jacobian-lens/research/confirmation_2026-09-09/artifacts/reconstructed/ouro/fit_02/lens.pt',
         'dc6354df970b29b8312829259ac282c829562aaaf95e45bb43e1bbc5ced5e560': '/home/moloch/jacobian-lens/research/refit_round_2026-09-07/cloud_leases/attempt_06/retrieved/controls/ouro_penultimate/fit_01/final/cursor_000100_6d4d010756ab4a1cbca52509c897410c/lens.pt',
         'b8e8b7d2d3f40030799395dee9e8376fe1fa0bf09f278319d31a0dff4efe6775': '/home/moloch/jacobian-lens/research/refit_round_2026-09-07/cloud_leases/attempt_06/retrieved/controls/ouro_positions/fit_01/final/cursor_000100_435bacee56d94ff988ff069b59f2c987/lens.pt'}
bank_ok = {h: (os.path.isfile(p) and sha(p) == h) for h, p in banks.items()}
print('retained banks hash-verified:', bank_ok, flush=True)
targets = [e for e in man['entries'] if e['kind'] == 'file' and os.path.exists(e['source_path']) and (e['source_path'].startswith(BUNDLE) or (e['source_path'].startswith(CHUNKS) and e['source_path'].endswith('.part')))]
def check(e):
    sp = e['source_path']
    if not os.path.isfile(sp) or os.path.islink(sp): return sp, 'missing or link'
    if os.path.getsize(sp) != e['bytes'] or sha(sp) != e['sha256'] or e['sha256'] != e.get('dest_sha256'): return sp, 'hash/size differs from archive manifest'
    if sp.startswith(BUNDLE):
        o = orig.get(sp)
        if o is None: return sp, 'not in bundle manifests'
        if o['sha256'] != e['sha256']: return sp, 'bundle manifest hash differs'
        if o['source_path'] == sp or not os.path.isfile(o['source_path']) or sha(o['source_path']) != e['sha256']: return sp, f"original missing or changed: {o['source_path']}"
        return sp, None
    bank_hash = sp[len(CHUNKS):].split('/')[0]; off = int(Path(sp).stem)
    if not bank_ok.get(bank_hash): return sp, 'retained bank not verified'
    with open(banks[bank_hash], 'rb') as f:
        f.seek(off); piece = f.read(e['bytes'])
    if hashlib.sha256(piece).hexdigest() != e['sha256']: return sp, 'chunk is not the bank slice at its offset'
    return sp, None
with ThreadPoolExecutor(6) as ex: results = list(ex.map(check, targets))
df0 = shutil.disk_usage('/home/moloch').free; deleted, skipped = [], []
for e, (sp, problem) in zip(targets, results):
    if problem: skipped.append({'path': sp, 'reason': problem}); continue
    os.remove(sp); deleted.append({'path': sp, 'bytes': e['bytes'], 'sha256': e['sha256'], 'ssd_copy': e['archive_path'], 'laptop_original_or_derivation': orig[sp]['source_path'] if sp.startswith(BUNDLE) else banks[sp[len(CHUNKS):].split('/')[0]] + f' @ offset {int(Path(sp).stem)}', 'deleted_utc': datetime.now(timezone.utc).isoformat()})
removed_dirs = []
for dp, dn, fn in sorted(os.walk(BUNDLE, topdown=False)):
    try: os.rmdir(dp); removed_dirs.append(dp)
    except OSError: pass
df1 = shutil.disk_usage('/home/moloch').free
out = {'schema': 'jlens_user_approved_duplicate_deletion.v2', 'run': 2, 'approved_by_user': '2026-09-11 chat: "purge those 22gb, we dont need duplicates" / "purge it from the device not the ssd"', 'executed_utc': datetime.now(timezone.utc).isoformat(),
       'ssd_attached_at_execution': False, 'ssd_verification_reference': {'receipt_manifest_sha256': receipt['manifest_sha256'], 'note': 'every deleted file has a destination copy hash-verified at archive time (dest_sha256 in ARCHIVE_MANIFEST.json); SSD was unplugged at deletion time, so laptop-original / bank-slice verification was used instead'},
       'scope': [BUNDLE + '**', CHUNKS + '**/*.part'], 'kept_records': 'BUNDLE_MANIFEST_*.json, PRESERVATION_LEDGER.json, RESTORATION_CHECK.json, restore_check/, scripts, input_chunks */MANIFEST.json',
       'targets': len(targets), 'deleted_files': deleted, 'deleted_bytes': sum(d['bytes'] for d in deleted), 'skipped': skipped, 'removed_empty_dirs': removed_dirs, 'free_bytes_before': df0, 'free_bytes_after': df1,
       'copies_remaining': 'laptop original at source_path (bundle files) or the retained bank (chunks), plus the SSD archive copy'}
(X / 'DELETION_MANIFEST_2026-09-11.json').write_text(json.dumps(out, indent=1))
print(json.dumps({'targets': len(targets), 'deleted': len(deleted), 'deleted_GB': round(out['deleted_bytes'] / 1e9, 2), 'skipped': len(skipped), 'freed_GB': round((df1 - df0) / 1e9, 2), 'removed_empty_dirs': len(removed_dirs)}))
if skipped: print('SKIPPED (first 10):', json.dumps(skipped[:10], indent=1))
