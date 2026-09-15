#!/usr/bin/env python3
"""Write README_ARCHIVE.md and ARCHIVE_INDEX.md on the SSD (and copies in the closeout dir) from the manifest, receipt
and restoration result; then add this round's closeout directory as a separately manifested supplement."""
import json, hashlib, os, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
DEST = Path('/run/media/moloch/ARCH_BACKUP/JLENS_COLD_ARCHIVE_20260911'); X = Path('/home/moloch/ouro_project/jacobian-lens/research/closeout_2026-09-11'); A = X / 'archive'
man = json.load(open(DEST / 'ARCHIVE_MANIFEST.json')); part = json.load(open(DEST / 'ACCEPTANCE_RECEIPT_partial.json'))
rest = json.load(open(A / 'RESTORATION_FROM_SSD.json')) if (A / 'RESTORATION_FROM_SSD.json').exists() else None
MODE = sys.argv[1] if len(sys.argv) > 1 else 'receipt'
def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()
# index by root
roots = {}
for e in man['entries']:
    r = next((rt for rt in man['roots'] if e['source_path'].startswith(rt.rstrip('/') + '/') or e['source_path'] == rt), 'other')
    d = roots.setdefault(r, {'files': 0, 'symlinks': 0, 'bytes': 0}); d['files' if e['kind'] == 'file' else 'symlinks'] += 1; d['bytes'] += e['bytes'] if e['kind'] == 'file' else 0
# supplement: copy the closeout dir (excluding the handoff zip's extracted duplicate? keep everything; exclude pages/ pngs and *_build.md scratch)
SUP = DEST / 'supplement_closeout_2026-09-11'
sup_entries, probs, sup_manifest_sha = [], [], None
if MODE == 'supplement':
    if SUP.exists(): shutil.rmtree(SUP)
    shutil.copytree(X, SUP / 'closeout_2026-09-11', ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'pages'))
    for p in sorted((SUP).rglob('*')):
        if p.is_file() and p.name != 'SUPPLEMENT_MANIFEST.json': sup_entries.append({'archive_path': str(p), 'source_path': str(X / p.relative_to(SUP / 'closeout_2026-09-11')), 'bytes': p.stat().st_size, 'sha256': sha256(p)})
    probs = [e for e in sup_entries if (not os.path.exists(e['source_path']) or sha256(e['source_path']) != e['sha256'])]
    (SUP / 'SUPPLEMENT_MANIFEST.json').write_text(json.dumps({'schema': 'jlens_cold_archive_supplement.v1', 'created_utc': datetime.now(timezone.utc).isoformat(), 'files': len(sup_entries), 'bytes': sum(e['bytes'] for e in sup_entries), 'problems': probs, 'entries': sup_entries}, indent=1))
    subprocess.run(['sync', '-f', str(DEST)], check=True)
    sup_manifest_sha = sha256(SUP / 'SUPPLEMENT_MANIFEST.json')
elif (SUP / 'SUPPLEMENT_MANIFEST.json').exists():
    sm = json.load(open(SUP / 'SUPPLEMENT_MANIFEST.json')); sup_entries = sm['entries']; probs = sm['problems']; sup_manifest_sha = sha256(SUP / 'SUPPLEMENT_MANIFEST.json')
# receipt
receipt = dict(part); receipt.update({'schema': 'jlens_cold_archive_acceptance_receipt.v1', 'archive_root': str(DEST), 'destination_identity_summary': {'device': '/dev/sda1 (SanDisk Extreme Portable SSD, USB)', 'filesystem': 'ext4', 'uuid': '36df468b-3bf9-41a9-ace8-edccfb292434', 'mountpoint': '/run/media/moloch/ARCH_BACKUP', 'label': 'ARCH_BACKUP', 'separate_from_source_disk': part['destination_identity_before']['separate_physical_device'], 'source_disk': 'nvme0n1p2 (WD PC SN8000S) UUID d06b03fe-5a17-4250-a6a2-acae19dfd890'},
    'full_manifest_sha256': part['manifest_sha256'], 'counts': {'files': part['files'], 'symlinks': part['symlinks'], 'bytes': part['bytes'], 'dedup_groups': part['dedup_groups']}, 'index_by_root': roots,
    'restoration_from_ssd': rest, 'supplement': ({'path': str(SUP), 'files': len(sup_entries), 'bytes': sum(e['bytes'] for e in sup_entries), 'manifest_sha256': sup_manifest_sha, 'problems': len(probs)} if sup_manifest_sha else 'not yet written'),
    'copy_status': 'SINGLE-COPY ARCHIVE once laptop copies are removed: this SSD is the only off-device copy; no second physical destination was available or authorized', 'excluded': 'see EXCLUSIONS.json and SECRETS_FLAGGED.json (SSH private key kept out)', 'symlink_note': 'two absolute symlinks (huginn_verification review fixtures linked_parent) point at laptop paths; their targets are archived under tree/ and they are not needed for restoration; HF cache symlinks are relative and resolve inside the archive',
    'unresolved_dependencies': ['venv (8 GB) not copied: requirements.lock + DEPENDENCIES.json pip freeze', 'public model weights other than Ouro-2.6B not copied (Huginn 15 GB, Ouro-Thinking, MiniCPM, DeBERTa): repo ids in DEPENDENCIES.json', 'private Hub repo Vykos/ouro-jlens-results: 19 files not retrievable here'], 'finalized_utc': datetime.now(timezone.utc).isoformat()})
(DEST / 'ACCEPTANCE_RECEIPT.json').write_text(json.dumps(receipt, indent=1)); shutil.copyfile(DEST / 'ACCEPTANCE_RECEIPT.json', A / 'ACCEPTANCE_RECEIPT.json')
idx = ['# JLENS_COLD_ARCHIVE_20260911 — index\n', f"Created {part['started_utc']} – {receipt['finalized_utc']}. Destination: SanDisk Extreme Portable SSD, ext4, UUID 36df468b-3bf9-41a9-ace8-edccfb292434. Manifest sha256 `{part['manifest_sha256']}`. {part['files']} files, {part['symlinks']} symlinks, {part['bytes']/1e9:.2f} GB, copy problems: {part['problems']}.\n", '| Root (original absolute path, preserved under tree/) | Files | Symlinks | GB |', '|---|---:|---:|---:|']
for r, d in roots.items(): idx.append(f"| {r} | {d['files']} | {d['symlinks']} | {d['bytes']/1e9:.2f} |")
idx += ['', '## Top-level files', '- `ARCHIVE_MANIFEST.json` — every file: source path, archive path, bytes, sha256 (source and destination), mtime, inode/nlink; symlink targets.', '- `ACCEPTANCE_RECEIPT.json` — destination identity, counts, manifest hash, isolated restoration result, supplement, risks.', '- `DEDUP_TABLE.json` — identical-content groups (informational; every path was copied).', '- `SECRETS_FLAGGED.json`, `EXCLUSIONS.json` — what was deliberately left out and why.', '- `DEPENDENCIES.json` — venv freeze, external public models not copied, private Hub repo status.', '- `git/` — `git bundle --all` of jacobian-lens and ouro_project, status, tracked diff, log.', '- `supplement_closeout_2026-09-11/` — this round\'s outputs (reviewer checks, local-exit inventory, methods, manuscript v2, handoff zip, ledgers, purge allowlist), separately manifested.', '', '## Where the key evidence is', '- Preservation bundle (banks, accepted payload, model snapshot, small files of both rounds): `tree/home/moloch/ouro_project/jacobian-lens/research/verification_2026-09-11/preservation/bundle/` with its `BUNDLE_MANIFEST_*.json`.', '- Confirmation round (frozen plan, benchmark, accepted payload with states, analysis): `tree/home/moloch/ouro_project/jacobian-lens/research/confirmation_2026-09-09/`.', '- Refit round (five fits, controls, Huginn pilot analyses, retained banks): `tree/home/moloch/ouro_project/jacobian-lens/research/refit_round_2026-09-07/`.', '- 7 Sep follow-up: `tree/home/moloch/ouro_project/jacobian-lens/research/followup_2026-09-07/`.', '- Final verification (replays, checker, metrics): `tree/home/moloch/ouro_project/jacobian-lens/research/verification_2026-09-11/`.', '- Application-era evidence (evaluations, lens sidecars, purge manifest): `tree/home/moloch/ouro_project/artifacts/jlens/`; application documents and the MATS submission (private): `tree/home/moloch/ouro_project/docs/jlens/`; custom code: `tree/home/moloch/ouro_project/src/ouro_jlens/`.', '- Model snapshot with real blobs: `tree/home/moloch/ouro_project/artifacts/hf_cache/hub/models--ByteDance--Ouro-2.6B/`; calibration corpus parquet: `.../datasets--Salesforce--wikitext/`.', '- Application-era final-target bank exit3 and four shards (only surviving copies): `tree/home/moloch/.cache/huggingface/hub/models--Vykos--ouro-jlens-results/blobs/`.']
(DEST / 'ARCHIVE_INDEX.md').write_text('\n'.join(idx)); shutil.copyfile(DEST / 'ARCHIVE_INDEX.md', A / 'ARCHIVE_INDEX.md')
readme = f"""# J-Lens cold archive (2026-09-11)

Copy-only archive of the surviving J-Lens/Ouro research evidence, written from the laptop (NVMe) to this external SSD on 2026-09-11. Nothing was moved or deleted at the source. Original absolute paths are preserved under `tree/`. See `ARCHIVE_INDEX.md` for contents and `ACCEPTANCE_RECEIPT.json` for verification.

**Status: {'hash-verified and restoration-tested' if rest and not rest.get('rehash_mismatched') and rest.get('analysis_rerun_equals_saved_json') else 'hash-verified; see receipt for restoration status'}.** If the laptop copies are later removed, this SSD is a **single-copy archive**, not a redundant backup.

Restoration test performed from this SSD alone (network and live laptop paths hidden): every bundle file re-hashed, frozen code archive matched its freeze, all four bank files loaded (five arms), the frozen analysis (primary + 20 secondaries) reproduced from archived readouts, and raw/fit01 rescoring from archived states and weights reproduced the saved ranks. Not repeated: the full numerical sensitivity suite, regeneration from prompts.

Requirements to use the archive: Python 3.14 with the pinned packages (`DEPENDENCIES.json`, `…/legacy/requirements.lock`), a CUDA GPU for rescoring, and the released `jlens` code (archived). Secrets were excluded (`SECRETS_FLAGGED.json`).
"""
(DEST / 'README_ARCHIVE.md').write_text(readme); shutil.copyfile(DEST / 'README_ARCHIVE.md', A / 'README_ARCHIVE.md')
subprocess.run(['sync', '-f', str(DEST)], check=True)
print(json.dumps({'supplement_files': len(sup_entries), 'supplement_bytes': sum(e['bytes'] for e in sup_entries), 'supplement_problems': len(probs), 'receipt_written': True}))
