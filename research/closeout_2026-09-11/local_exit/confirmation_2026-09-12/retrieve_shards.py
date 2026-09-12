#!/usr/bin/env python3
"""Step 1 of the local-exit proposal: download the initial-study lens shards from the private Hub repository
and verify each against the sha256 recorded in LOCAL_EXIT_BANK_INVENTORY.json. Retrieval only; no GPU.
The token is read from the environment (HF_TOKEN) and never written anywhere."""
import hashlib, json, os, sys, time
from pathlib import Path
from huggingface_hub import hf_hub_download
REPO = 'Vykos/ouro-jlens-results'
DEST = Path('/home/moloch/ouro_project/artifacts/jlens/retrieved_2026-09-12')
OUT = Path(__file__).resolve().parent
inv = json.load(open(OUT.parent.parent / 'local_exit' / 'LOCAL_EXIT_BANK_INVENTORY.json'))
wanted = []
for b in inv['application_era_family']:
    for sh in b['shards']:
        if 'REMOTE_ONLY' in sh['status']:
            rp = sh['remote_path'] or ('jlens-b300-20260905-0359/artifacts/lens/n100/' + Path(sh['path']).name)
            wanted.append({'bank': b['bank'], 'remote_path': rp, 'bytes': sh['bytes'], 'sha256': sh['sha256'], 'prompt_range': sh['prompt_range']})
assert len(wanted) == 13, len(wanted)
def sha(p):
    m = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''): m.update(c)
    return m.hexdigest()
receipt = {'schema': 'local_exit_retrieval_receipt.v1', 'repo': REPO, 'started': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'files': []}
t0 = time.time()
for w in wanted:
    for rp in (w['remote_path'], w['remote_path'][:-3] + '.json'):
        t = time.time()
        local = Path(hf_hub_download(REPO, rp, local_dir=DEST, token=os.environ['HF_TOKEN']))
        rec = {'remote_path': rp, 'local_path': str(local), 'bytes': local.stat().st_size, 'sha256': sha(local), 'seconds': round(time.time() - t, 1)}
        if rp.endswith('.pt'):
            rec['expected_bytes'], rec['expected_sha256'] = w['bytes'], w['sha256']
            rec['verified'] = rec['bytes'] == w['bytes'] and rec['sha256'] == w['sha256']
        receipt['files'].append(rec)
        print(f"{rp.split('/')[-1]:<32} {rec['bytes']:>13,}  {rec.get('verified', 'sidecar')}  {rec['seconds']}s", flush=True)
receipt['finished'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()); receipt['total_seconds'] = round(time.time() - t0)
receipt['all_pt_verified'] = all(r['verified'] for r in receipt['files'] if r['remote_path'].endswith('.pt'))
json.dump(receipt, open(OUT / 'RETRIEVAL_RECEIPT.json', 'w'), indent=1)
print('ALL VERIFIED' if receipt['all_pt_verified'] else 'HASH MISMATCH', receipt['total_seconds'], 's')
sys.exit(0 if receipt['all_pt_verified'] else 1)
