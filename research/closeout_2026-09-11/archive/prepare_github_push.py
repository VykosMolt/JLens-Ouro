#!/usr/bin/env python3
"""Prepare the GitHub push set: write .gitignore rules that keep secrets, account records, same-disk duplicates,
and every file >= 50 MiB out of the repository; list what is excluded (with hashes from the SSD manifest) in
research/GITHUB_EXCLUSIONS.md; scan the remaining set for secret-like content."""
import os, re, json, subprocess, hashlib
from pathlib import Path
ROOT = Path('/home/moloch/jacobian-lens'); R = ROOT / 'research'
LIMIT = 50 * 2**20
RULES = ['', '# ---- research records: kept out of GitHub (all of it is in the SSD archive; see research/GITHUB_EXCLUSIONS.md) ----',
         'research/verification_2026-09-11/preservation/bundle/', 'research/verification_2026-09-11/preservation/restore_check/',
         'research/verification_2026-09-11/replay/local_*/**/*.npz', 'research/verification_2026-09-11/replay/local_*/*.npz', 'research/verification_2026-09-11/replay/propagation_*/**/*.npz', 'research/verification_2026-09-11/replay/propagation_*/**/*.pt',
         'research/verification_2026-09-11/gpu_shape_v1/**/*.pt', 'research/confirmation_2026-09-09/resources/input_chunks/', 'research/confirmation_2026-09-09/resources/transfer_probe/', 'research/confirmation_2026-09-09/resources/transfer_probe_download/',
         'research/refit_round_2026-09-07/optimization/*.npz', 'research/refit_round_2026-09-07/optimization/*.pt',
         '# secrets, keys, hosts, account observations', 'research/confirmation_2026-09-09/resources/ssh/', 'research/**/known_hosts', 'research/**/id_ed25519*', 'research/confirmation_2026-09-09/resources/account_*.json', 'research/confirmation_2026-09-09/resources/*account_observation*', 'research/confirmation_2026-09-09/resources/prior_debit.json', 'research/verification_2026-09-11/spending/observation_*.json', 'research/refit_round_2026-09-07/deployment/account_*.json',
         '# scratch and packaging', '*.zip', 'research/closeout_2026-09-11/manuscript/pages/', 'research/closeout_2026-09-11/manuscript/*.aux', 'research/closeout_2026-09-11/manuscript/*.log', 'research/closeout_2026-09-11/manuscript/*.out', 'research/closeout_2026-09-11/manuscript/*_build.md', 'research/closeout_2026-09-11/archive/ARCHIVE_MANIFEST.json.tmp']
gi = (ROOT / '.gitignore').read_text()
if 'GITHUB_EXCLUSIONS' not in gi: (ROOT / '.gitignore').write_text(gi.rstrip('\n') + '\n' + '\n'.join(RULES) + '\n')
# explicit list of remaining files >= LIMIT
def tracked_candidates():
    out = subprocess.run(['git', 'ls-files', '--others', '--exclude-standard', '-z', 'research'], cwd=ROOT, capture_output=True).stdout.decode().split('\0')
    return [p for p in out if p]
cands = tracked_candidates(); big = [p for p in cands if (ROOT / p).is_file() and not (ROOT / p).is_symlink() and (ROOT / p).stat().st_size >= LIMIT]
if big:
    with open(ROOT / '.gitignore', 'a') as f: f.write('# individual files >= 50 MiB (generated)\n' + '\n'.join(big) + '\n')
    cands = tracked_candidates()
sizes = {p: (ROOT / p).stat().st_size for p in cands if (ROOT / p).is_file() and not (ROOT / p).is_symlink()}
cands = [p for p in cands if not (ROOT / p).is_symlink()] + [p for p in cands if (ROOT / p).is_symlink()]
total = sum(sizes.values()); print(f'push set: {len(cands)} files, {total/1e9:.2f} GB')
print('largest 12 in push set:'); [print(f'  {s/1e6:8.1f} MB {p}') for p, s in sorted(sizes.items(), key=lambda x: -x[1])[:12]]
# exclusions list with hashes from the SSD manifest
man = json.load(open(R / 'closeout_2026-09-11/archive/ARCHIVE_MANIFEST.json')); by_src = {e['source_path']: e for e in man['entries'] if e['kind'] == 'file'}
allfiles = []
for dp, dn, fn in os.walk(R):
    dn[:] = [d for d in dn if d != '__pycache__']
    for f in fn:
        p = Path(dp) / f
        if p.is_symlink() or f.endswith('.pyc'): continue
        allfiles.append(p)
inset = set(cands); excl = [p for p in allfiles if str(p.relative_to(ROOT)) not in inset]
lines = ['# Files under research/ that are NOT in the GitHub repository\n', f'Generated {json.dumps(None)} from the working tree. Every listed file is in the external SSD archive `JLENS_COLD_ARCHIVE_20260911` (manifest sha256 {hashlib.sha256((R / "closeout_2026-09-11/archive/ARCHIVE_MANIFEST.json").read_bytes()).hexdigest()}); hashes below are from that manifest. Reasons: size ≥ 50 MiB (GitHub limit 100 MB, repository kept small), same-disk duplicates (preservation bundle), replay/benchmark binary arrays, secrets/keys/hosts, RunPod account observations, scratch.\n',
         f'Excluded: {len(excl)} files, {sum(p.stat().st_size for p in excl)/1e9:.2f} GB. Included: {len(cands)} files, {total/1e9:.2f} GB.\n', '| Path | Bytes | SHA-256 |', '|---|---:|---|']
for p in sorted(excl, key=lambda q: -q.stat().st_size):
    e = by_src.get(str(p)); lines.append(f"| {p.relative_to(ROOT)} | {p.stat().st_size} | {e['sha256'] if e else '(not in archive manifest: created after the archive copy, e.g. this round)'} |")
(R / 'GITHUB_EXCLUSIONS.md').write_text('\n'.join(lines))
# secret scan over the push set (text files only)
pat = re.compile(rb'(ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{28,}|rpa_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{30,}|BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY|AKIA[0-9A-Z]{16}|RUNPOD_API_KEY\s*[=:]\s*\S{10,})')
hits = []
for p in cands:
    fp = ROOT / p
    if fp.is_symlink() or not fp.is_file() or fp.stat().st_size > 20 * 2**20: continue
    data = fp.read_bytes()
    if b'\0' in data[:4096]: continue
    for m in pat.finditer(data): hits.append((p, m.group(0)[:12].decode('latin1') + '…'))
print('secret-pattern hits:', hits[:10] if hits else 'none')
print('excluded files:', len(excl), f'{sum(p.stat().st_size for p in excl)/1e9:.2f} GB')
