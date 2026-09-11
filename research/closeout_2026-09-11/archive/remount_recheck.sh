#!/usr/bin/env bash
# Clean remount of the SSD via udisks (no force), then re-hash the archive's top-level records and a fixed sample of
# large payload files against ARCHIVE_MANIFEST.json. Writes REMOUNT_RECHECK.json next to this script.
set -euo pipefail
OUT=$(dirname "$0")/REMOUNT_RECHECK.json
DEV=/dev/sda1; MNT=/run/media/moloch/ARCH_BACKUP; ARCH=$MNT/JLENS_COLD_ARCHIVE_20260911
sync -f "$MNT"
if lsof +f -- "$MNT" >/dev/null 2>&1; then echo "mount busy; not forcing" ; BUSY=1; else BUSY=0; fi
if [ "$BUSY" = 0 ]; then
  udisksctl unmount -b "$DEV" >/tmp/claude-1000/remount_out.txt 2>&1 && udisksctl mount -b "$DEV" >>/tmp/claude-1000/remount_out.txt 2>&1
fi
NEWMNT=$(findmnt -rno TARGET "$DEV"); ARCH=$NEWMNT/JLENS_COLD_ARCHIVE_20260911
python3 - "$ARCH" "$OUT" "$BUSY" <<'PY'
import json, hashlib, sys, subprocess, os
from datetime import datetime, timezone
arch, out, busy = sys.argv[1], sys.argv[2], sys.argv[3]
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()
man = json.load(open(f'{arch}/ARCHIVE_MANIFEST.json')); part = json.load(open(f'{arch}/ACCEPTANCE_RECEIPT_partial.json'))
res = {'utc': datetime.now(timezone.utc).isoformat(), 'remounted': busy == '0', 'mount': arch, 'manifest_sha256_after_remount': sha(f'{arch}/ARCHIVE_MANIFEST.json'), 'manifest_sha256_in_receipt': part['manifest_sha256'],
       'blkid': subprocess.run(['lsblk', '-no', 'UUID,FSTYPE,MODEL,SERIAL', '/dev/sda1'], capture_output=True, text=True).stdout.strip()}
res['manifest_matches'] = res['manifest_sha256_after_remount'] == res['manifest_sha256_in_receipt']
# sample: the four confirmation banks, the model weights, the accepted payload readouts, 200 evenly spaced other files
entries = [e for e in man['entries'] if e['kind'] == 'file']
keys = [e for e in entries if e['source_path'].endswith('/lens.pt') or e['source_path'].endswith('model.safetensors') or '/readouts/' in e['source_path']]
step = max(1, len(entries) // 200); sample = keys + entries[::step]
mism = [e['source_path'] for e in sample if sha(e['archive_path']) != e['sha256']]
res.update({'sampled_files': len(sample), 'sampled_bytes': sum(e['bytes'] for e in sample), 'mismatched': mism})
json.dump(res, open(out, 'w'), indent=1); print(json.dumps({k: res[k] for k in ('remounted', 'manifest_matches', 'sampled_files', 'sampled_bytes', 'mismatched', 'blkid')}))
PY
