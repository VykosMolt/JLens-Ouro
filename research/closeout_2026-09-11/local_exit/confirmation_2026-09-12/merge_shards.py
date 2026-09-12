#!/usr/bin/env python3
"""Merge the retrieved initial-study shards into one bank per exit with the frozen reference procedure
(jlens.JacobianLens.merge: n_prompts-weighted mean), and check the procedure against the surviving pod merge of exit3."""
import hashlib, json, os, sys, time
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1] / 'handoff/JLens_Ouro_Writer_Handoff_v2/08_code'
FC = Path('/tmp/claude-1000/-home-moloch-jacobian-lens/fcb94bb7-42d0-49fd-8993-7db96deb86a0/scratchpad/frozen_code'); FC.mkdir(parents=True, exist_ok=True)
for name, src in (('jlens', CODE / 'frozen_jlens'), ('ouro_jlens', CODE / 'frozen_ouro_jlens')):
    link = FC / name
    if not link.exists(): link.symlink_to(src)
sys.dont_write_bytecode = True; sys.path[:0] = [str(FC)]
import jlens
RET = Path('/home/moloch/ouro_project/artifacts/jlens/retrieved_2026-09-12/jlens-b300-20260905-0359/artifacts/lens/n100')
HF = Path.home() / '.cache/huggingface/hub/models--Vykos--ouro-jlens-results'
OUTB = Path('/home/moloch/ouro_project/artifacts/jlens/retrieved_2026-09-12/merged'); OUTB.mkdir(exist_ok=True)
inv = json.load(open(HERE.parent / 'LOCAL_EXIT_BANK_INVENTORY.json'))
receipt = json.load(open(HERE / 'RETRIEVAL_RECEIPT.json')); assert receipt['all_pt_verified']
def sha(p):
    m = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''): m.update(c)
    return m.hexdigest()
def blob_for(name):
    for p in HF.glob(f'snapshots/*/jlens-b300-20260905-0359/artifacts/lens/n100/{name}'):
        return p.resolve()
    return None
record = {'schema': 'local_exit_shard_merge.v1', 'procedure': 'jlens.JacobianLens.merge (frozen reference package): n_prompts-weighted mean of shard Jacobians, saved FP16', 'banks': {}}
for b in inv['application_era_family']:
    exit_name = b['bank'].split('/')[-1]
    shards = []
    for sh in sorted(b['shards'], key=lambda s: s['prompt_range'][0]):
        fn = Path(sh['path']).name
        p = RET / fn if (RET / fn).exists() else blob_for(fn)
        assert p and p.exists(), fn
        h = sha(p); assert h == sh['sha256'], (fn, h[:12], sh['sha256'][:12])
        shards.append((sh['prompt_range'], p, h))
    t = time.time()
    lenses = [jlens.JacobianLens.load(str(p)) for _, p, _ in shards]
    for (a, c), lens in zip([r for r, _, _ in shards], lenses): assert lens.n_prompts == c - a, (exit_name, a, c, lens.n_prompts)
    merged = jlens.JacobianLens.merge(lenses); out = OUTB / f'{exit_name}.pt'; merged.save(str(out)); del lenses
    record['banks'][exit_name] = {'target_virtual': b['target_virtual'], 'shards': [{'prompt_range': r, 'path': str(p), 'sha256': h} for r, p, h in shards],
                                  'n_prompts': merged.n_prompts, 'source_layers': [merged.source_layers[0], merged.source_layers[-1]], 'merged': {'path': str(out), 'bytes': out.stat().st_size, 'sha256': sha(out)},
                                  'purged_original_merge_sha256': b['merged_file']['sha256'], 'seconds': round(time.time() - t, 1)}
    print(exit_name, 'merged from', len(shards), 'shards;', merged.n_prompts, 'prompts;', record['banks'][exit_name]['merged']['sha256'][:12], f"{time.time()-t:.0f}s", flush=True)
    del merged
# equivalence check: my exit3 merge vs the surviving pod merge (d7c26297…)
pod = blob_for('exit3.pt'); assert pod and sha(pod).startswith('d7c26297')
A = torch.load(OUTB / 'exit3.pt', map_location='cpu', weights_only=True, mmap=True)['J']; B = torch.load(pod, map_location='cpu', weights_only=True, mmap=True)['J']
assert set(A) == set(B)
eq = tot = 0; maxdiff = 0.0
for v in sorted(A):
    a, b = A[v], B[v]; eq += int((a == b).sum()); tot += a.numel(); maxdiff = max(maxdiff, float((a.float() - b.float()).abs().max()))
record['exit3_equivalence_with_pod_merge'] = {'pod_merge_sha256': sha(pod), 'my_merge_sha256': record['banks']['exit3']['merged']['sha256'], 'byte_identical_files': sha(pod) == record['banks']['exit3']['merged']['sha256'],
                                              'fraction_entries_exactly_equal': eq / tot, 'max_abs_difference': maxdiff, 'fp16_eps_at_1': 2 ** -10}
print('exit3 vs pod merge: identical files', record['exit3_equivalence_with_pod_merge']['byte_identical_files'], '| equal entries', f'{eq/tot:.6f}', '| max |diff|', maxdiff)
json.dump(record, open(HERE / 'MERGE_RECORD.json', 'w'), indent=1)
