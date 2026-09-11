#!/usr/bin/env python3
"""DRY RUN ONLY. Build the proposed purge allowlist of bulky laptop copies that are verified in the SSD archive.
Nothing is deleted. Each row: exact source path, current size/hash (re-read now), matching archive path + verified hash,
restoration receipt reference, dependency checks, and physical bytes (unique inodes). Exclusions: shared caches used by
other projects (Ouro model weights, WikiText parquet), active manuscript/closeout sources, the handoff, receipts/indexes,
and every file below the size floor (small reports stay on the laptop)."""
import json, os, hashlib, csv, sys
from pathlib import Path
X = Path('/home/moloch/jacobian-lens/research/closeout_2026-09-11'); A = X / 'archive'
man = json.load(open(A / 'ARCHIVE_MANIFEST.json')); receipt = json.load(open(A / 'ACCEPTANCE_RECEIPT.json'))
FLOOR = 8 * 2**20
EXCLUDE_PREFIXES = ['/home/moloch/ouro_project/artifacts/hf_cache/hub/models--ByteDance--Ouro-2.6B', '/home/moloch/ouro_project/artifacts/hf_cache/hub/datasets--Salesforce--wikitext',
                    '/home/moloch/ouro_project/src/ouro_jlens', '/home/moloch/ouro_project/docs/jlens', '/home/moloch/jacobian-lens/research/closeout_2026-09-11', '/home/moloch/jacobian-lens/.git',
                    '/home/moloch/jacobian-lens/jlens', '/home/moloch/jacobian-lens/data', '/home/moloch/jacobian-lens/tests', '/home/moloch/jacobian-lens/assets']
reasons = {EXCLUDE_PREFIXES[0]: 'shared model cache used by ouro_project', EXCLUDE_PREFIXES[1]: 'shared dataset cache', EXCLUDE_PREFIXES[2]: 'active source code', EXCLUDE_PREFIXES[3]: 'active manuscript/application sources', EXCLUDE_PREFIXES[4]: 'this round (active, handoff, receipts)'}
def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()
by_source = {e['source_path']: e for e in man['entries'] if e['kind'] == 'file'}
rows, excluded, seen_inodes, physical = [], [], set(), 0
for sp, e in sorted(by_source.items()):
    if e['bytes'] < FLOOR: continue
    hit = next((pre for pre in EXCLUDE_PREFIXES if sp == pre or sp.startswith(pre + '/')), None)
    if hit: excluded.append({'path': sp, 'bytes': e['bytes'], 'reason': reasons.get(hit, 'excluded prefix')}); continue
    if not os.path.exists(sp): continue
    st = os.stat(sp); cur = sha256(sp)
    ok = cur == e['sha256'] == e.get('dest_sha256') and st.st_size == e['bytes'] and os.path.exists(e['archive_path'])
    key = (st.st_dev, st.st_ino); first = key not in seen_inodes; seen_inodes.add(key)
    if first and ok: physical += st.st_blocks * 512
    rows.append({'source_path': sp, 'bytes': e['bytes'], 'sha256_now': cur, 'sha256_at_archive': e['sha256'], 'archive_path': e['archive_path'], 'archive_sha256_verified': e.get('dest_sha256'),
                 'nlink': st.st_nlink, 'inode': st.st_ino, 'counts_physical_bytes': first, 'eligible_now': ok, 'restoration_receipt': str(A / 'ACCEPTANCE_RECEIPT.json') + ' (restoration_from_ssd)',
                 'dependency_check': 'referenced only by J-Lens rounds; not a shared cache; not an active source' })
elig = [r for r in rows if r['eligible_now']]
summary = {'schema': 'purge_allowlist_dryrun.v1', 'status': 'DRY RUN — nothing deleted; requires explicit informed approval of the single-copy risk and of this exact list',
           'size_floor_bytes': FLOOR, 'rows': len(rows), 'eligible_now': len(elig), 'not_eligible_now': len(rows) - len(elig), 'logical_bytes_eligible': sum(r['bytes'] for r in elig),
           'estimated_physical_bytes_reclaimed': physical, 'excluded_bulky_files': excluded, 'excluded_logical_bytes': sum(x['bytes'] for x in excluded),
           'copies_after_purge': 'ONE complete copy (the SSD archive) for every allowlisted file; the SSD would be a single-copy archive, not a redundant backup',
           'later_deletion_procedure': ['re-verify destination identity (UUID 36df468b-3bf9-41a9-ace8-edccfb292434, serial in receipt) and mount', 're-hash each source file and compare with sha256_at_archive; skip changed or missing files', 'confirm archive_path exists and its recorded dest_sha256 (optionally re-hash)', 'delete only listed regular files one by one; never a parent directory; no cache cleaner; no git clean', 'record a deletion manifest with path/size/hash/timestamp'],
           'archive_receipt': receipt.get('manifest_sha256'), 'destination': man['destination']}
(A / 'PURGE_ALLOWLIST_DRYRUN.json').write_text(json.dumps({'summary': summary, 'allowlist': rows}, indent=1))
with open(A / 'PURGE_ALLOWLIST_DRYRUN.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
md = [f"# Proposed purge allowlist (DRY RUN; nothing deleted)\n", f"- Files ≥ {FLOOR // 2**20} MiB under the archived roots, hash-verified on the SSD now: **{len(elig)} eligible** of {len(rows)} listed ({len(rows) - len(elig)} not eligible now).",
      f"- Logical bytes eligible: {summary['logical_bytes_eligible'] / 1e9:.2f} GB; estimated physical space reclaimed (unique inodes, allocated blocks): **{physical / 1e9:.2f} GB**.",
      f"- Excluded bulky files (shared caches, active sources): {len(excluded)} files, {summary['excluded_logical_bytes'] / 1e9:.2f} GB (see JSON).",
      "- After deletion the SSD would hold the **only** complete copy of these files (single-copy archive). Explicit informed approval of that risk and of this exact list is required; a later deletion must re-check every source hash and the destination identity and skip anything changed.",
      "- No recursive directory purge, no generic cache cleanup, no Git cleanup is proposed.\n", "| Source path | GB | eligible now | archive path |", "|---|---:|---|---|"]
for r in sorted(rows, key=lambda r: -r['bytes'])[:60]: md.append(f"| {r['source_path'].replace('/home/moloch/', '~/')} | {r['bytes'] / 1e9:.2f} | {r['eligible_now']} | {r['archive_path'].replace(man['destination'], '<ARCHIVE>')} |")
md.append(f"\n(60 largest shown; all {len(rows)} rows in PURGE_ALLOWLIST_DRYRUN.csv)")
(A / 'PURGE_ALLOWLIST_DRYRUN.md').write_text('\n'.join(md)); print(json.dumps({k: summary[k] for k in ('rows', 'eligible_now', 'logical_bytes_eligible', 'estimated_physical_bytes_reclaimed', 'excluded_logical_bytes')}))
