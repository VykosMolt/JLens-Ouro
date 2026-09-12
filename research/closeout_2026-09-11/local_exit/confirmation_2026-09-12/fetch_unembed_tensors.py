#!/usr/bin/env python3
"""The base Ouro-2.6B weights are no longer on this laptop (only the SSD archive has them). The readout needs two
tensors, lm_head.weight and model.norm.weight. Fetch exactly those byte ranges from the public Hub file
model.safetensors at the pinned revision, verify the header, and save them as a small safetensors file with a record."""
import hashlib, json, struct, sys
from pathlib import Path
import requests, torch
from huggingface_hub import HfApi, hf_hub_url
from safetensors.torch import save_file
REPO, REV = 'ByteDance/Ouro-2.6B', '1ed04250da1a9936042725d302e81c8fa2ab5abd'
OUT = Path('/home/moloch/ouro_project/artifacts/jlens/retrieved_2026-09-12/ouro26b_unembed_1ed04250.safetensors')
url = hf_hub_url(REPO, 'model.safetensors', revision=REV)
info = HfApi().model_info(REPO, revision=REV, files_metadata=True)
sib = [s for s in info.siblings if s.rfilename == 'model.safetensors'][0]
sess = requests.Session()
def rng(a, b):  # inclusive byte range
    r = sess.get(url, headers={'Range': f'bytes={a}-{b}'}, allow_redirects=True, timeout=120); r.raise_for_status(); assert r.status_code == 206, r.status_code; return r.content
n = struct.unpack('<Q', rng(0, 7))[0]
header = json.loads(rng(8, 8 + n - 1)); base = 8 + n
want = {'lm_head.weight', 'model.norm.weight'}
assert want <= set(header), sorted(k for k in header if 'head' in k or 'norm.weight' in k)[:5]
tensors, rec = {}, {}
DT = {'BF16': torch.bfloat16, 'F16': torch.float16, 'F32': torch.float32}
for k in sorted(want):
    m = header[k]; a, b = m['data_offsets']; raw = rng(base + a, base + b - 1); assert len(raw) == b - a
    t = torch.frombuffer(bytearray(raw), dtype=DT[m['dtype']]).reshape(m['shape']).clone()
    tensors[k] = t; rec[k] = {'dtype': m['dtype'], 'shape': m['shape'], 'byte_range_in_file': [base + a, base + b], 'sha256_of_bytes': hashlib.sha256(raw).hexdigest()}
    print(k, m['dtype'], m['shape'], len(raw), 'bytes')
OUT.parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, str(OUT), metadata={'source_repo': REPO, 'revision': REV, 'source_file': 'model.safetensors'})
record = {'schema': 'unembed_tensor_fetch.v1', 'source': {'repo': REPO, 'revision': REV, 'file': 'model.safetensors', 'size': sib.size, 'lfs_sha256': sib.lfs.sha256 if sib.lfs else None, 'header_bytes': n}, 'tensors': rec, 'output': {'path': str(OUT), 'bytes': OUT.stat().st_size, 'sha256': hashlib.sha256(OUT.read_bytes()).hexdigest()}}
json.dump(record, open(Path(__file__).resolve().parent / 'UNEMBED_TENSORS_RECORD.json', 'w'), indent=1)
print('saved', OUT, OUT.stat().st_size, 'bytes; source file lfs sha256', record['source']['lfs_sha256'])
