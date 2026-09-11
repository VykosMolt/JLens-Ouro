#!/usr/bin/env python3
"""Read-only historical audit; derived fit02 output lives beside this script.

Run `python recover_estimators.py inventory`, then CPU torch Python with `tensors`.
No producer code is imported. Partial files are hashed but never deserialized.
"""
import argparse
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
OLD = ROOT / 'research/refit_round_2026-09-07'
LEASE = OLD / 'cloud_leases/attempt_06'
MON = OLD / 'monitoring/attempt_06'
MANIFEST = MON / 'termination_incident/SURVIVING_ARTIFACTS.json'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def guard(path):
    path = Path(path)
    assert not any(p.is_symlink() for p in (path, *path.parents)), path
    s = path.stat()
    assert stat.S_ISREG(s.st_mode), path
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns, nlink=s.st_nlink)


def record(path):
    before = guard(path)
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    after = guard(path)
    assert before == after, f'file changed during hash: {path}'
    return {'bytes': before['bytes'], 'sha256': h.hexdigest(),
            'stat_before': before, 'stat_after': after, 'stat_guard_passed': True}


def same(rec, expected):
    return all(rec[k] == expected[k] for k in ('bytes', 'sha256'))


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('x') as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def receipt_root(fit):
    if fit.startswith('ouro/'):
        return sorted(MON.glob(f"{fit.split('/')[-1]}_*/COPY_VERIFIED.json"))[-1].parent
    name = fit.split('/')[1]
    return sorted(MON.glob(f'{name}_*/COPY_VERIFIED.json'))[-1].parent


def provenance(fit, known):
    base = receipt_root(fit)
    receipt = read(base / 'COPY_VERIFIED.json')
    checks = []
    for row in receipt['files']:
        rec = record(base / 'results' / row['path'])
        assert same(rec, row), row['path']
        checks.append(dict(path=str(base / 'results' / row['path']), record=rec,
                           expected={k: row[k] for k in ('bytes', 'sha256')}))
    fitdir = base / 'results' / fit
    owner = read(fitdir / 'OWNER.json')
    identity = owner['identity']; sha = digest(identity)
    assert set(owner) == {'schema_version', 'fit_identity_sha256', 'identity'}
    assert owner['schema_version'] == 1 and owner['fit_identity_sha256'] == sha
    assert identity['n_prompts'] == 100 and identity['d_model'] == 2048
    n_sources = 190 if 'penultimate' in fit else 191
    assert identity['source_layers'] == list(range(n_sources))
    assert len(identity['prompt_sha256']) == len(identity['token_lengths']) == len(identity['n_valid']) == 100
    pointers = {}; generations = {}
    for kind, pointer_name, binary in [('checkpoint', 'LATEST.json', 'state.pt'), ('final', 'COMPLETE.json', 'lens.pt')]:
        ptr = read(fitdir / pointer_name); pointers[kind] = ptr
        assert ptr['kind'] == kind and ptr['fit_identity_sha256'] == sha
        assert ptr['n_done'] == ptr['next_idx'] == 100
        assert Path(ptr['generation']).parts[0] == ('checkpoints' if kind == 'checkpoint' else 'final')
        gen = fitdir / ptr['generation']
        assert same(record(gen / 'SEAL.json'), ptr['seal'])
        seal = read(gen / 'SEAL.json'); meta = read(gen / 'metadata.json')
        assert set(seal) == {'schema_version', 'kind', 'fit_identity_sha256', 'n_done', 'files'}
        assert set(seal['files']) == {binary, 'metadata.json'}
        assert seal['fit_identity_sha256'] == sha and seal['n_done'] == 100 and seal['kind'] == kind
        assert same(record(gen / 'metadata.json'), seal['files']['metadata.json'])
        key = f'{fit}/{ptr["generation"]}/{binary}'
        assert seal['files'][binary] == known[key]
        expected = dict(schema_version=1, kind=kind, fit_identity_sha256=sha, identity=identity,
                        n_done=100, next_idx=100, completed_prompt_sha256=identity['prompt_sha256'],
                        completed_prefix_sha256=digest(identity['prompt_sha256']))
        assert all(meta[k] == v for k, v in expected.items())
        extra = {'diagnostics', 'recorded_utc'} | ({'fp32_checkpoint', 'saved_dtype'} if kind == 'final' else set())
        assert set(meta) == set(expected) | extra
        assert len(meta['diagnostics']) == 100
        for i, row in enumerate(meta['diagnostics']):
            for k, value in {'index': i, 'prompt_sha256': identity['prompt_sha256'][i],
                             'token_length': identity['token_lengths'][i], 'n_valid': identity['n_valid'][i]}.items():
                assert row[k] == value
        if kind == 'final':
            assert meta['fp32_checkpoint'] == pointers['checkpoint'] and meta['saved_dtype'] == 'float16'
        generations[kind] = dict(binary_key=key, metadata_path=str(gen / 'metadata.json'),
                                 seal_path=str(gen / 'SEAL.json'), expected_binary=seal['files'][binary])
    # Audit the exact conversion source bound into this fit, without importing it.
    sources = []
    for row in identity['runtime']['source_files']:
        if row['path'].endswith('/deployment/run_refits.py'):
            p = ROOT / row['path']; actual = record(p)
            assert actual['sha256'] == row['sha256']
            sources.append(dict(path=str(p), actual=actual, expected_sha256=row['sha256']))
    assert len(sources) == 1
    return dict(status='passed', receipt_path=str(base / 'COPY_VERIFIED.json'),
                receipt_record=record(base / 'COPY_VERIFIED.json'), receipt_files=checks,
                identity=identity, identity_sha256=sha, pointers=pointers,
                generations=generations, bound_conversion_source=sources)


def inventory():
    manifest = read(MANIFEST); known = manifest['known_estimator_binaries']
    assert len(known) == 16
    # Search every repository directory (including ignored and hidden staging files).
    discovered = []
    for current, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {'.git', '.venv', 'node_modules'} and not (Path(current) / d).is_symlink()]
        for name in files:
            p = Path(current) / name
            if OUT in p.parents:
                continue
            if p.suffix in {'.pt', '.pth', '.bin', '.npy', '.npz', '.partial', '.part', '.tmp'}:
                discovered.append(p)
    result = dict(schema='confirmation_artifact_inventory.v1', started_utc=now(),
                  search_root=str(ROOT), search_exclusions=['.git', '.venv', 'node_modules', 'symlink directories', str(OUT)],
                  manifest=dict(path=str(MANIFEST), record=record(MANIFEST)), binaries={}, provenance={})
    for key, expected in sorted(known.items()):
        copies = []
        for p in sorted(discovered):
            if p.as_posix().endswith('/' + key):
                actual = record(p)
                status = 'complete_hash_verified' if same(actual, expected) else ('truncated' if actual['bytes'] < expected['bytes'] else 'mismatch')
                copies.append(dict(path=str(p), actual=actual, status=status, deserialized=False))
                print(status, actual['bytes'], key, flush=True)
        result['binaries'][key] = dict(expected=expected, copies=copies,
                                       status='complete_hash_verified' if any(c['status'] == 'complete_hash_verified' for c in copies) else ('partial_only' if copies else 'absent'))
    for fit in ['ouro/fit_01', 'ouro/fit_02', 'ouro/fit_03', 'ouro/fit_04', 'ouro/fit_05',
                'controls/ouro_penultimate/fit_01', 'controls/ouro_positions/fit_01']:
        result['provenance'][fit] = provenance(fit, known)
    candidates = []
    for p in discovered:
        if 'huginn' in p.as_posix().lower():
            candidates.append(dict(path=str(p), stat=guard(p),
                                   role='incomplete FP32 sum' if p.name == 'state.pt' else 'saved evaluation/cache, not per-paragraph Jacobian contributions'))
    result['huginn_reconstruction'] = dict(status='not_reconstructable_from_retained_repository_files',
        candidates=candidates, all_repository_binary_candidates=[dict(path=str(p), bytes=p.stat().st_size) for p in sorted(discovered)],
        reason='Only a truncated N100 sum remains; no complete sum, final bank, earlier checkpoint, or per-paragraph Jacobian contributions were found. Evaluation rank arrays and activation caches cannot supply missing Jacobian entries.',
        partial_deserialized=False, scope='Repository search only; no provider access or claim about unsearched external storage.')
    result['counts'] = {status: sum(x['status'] == status for x in result['binaries'].values())
                        for status in ['complete_hash_verified', 'partial_only', 'absent']}
    result['finished_utc'] = now()
    atomic_json(OUT / 'inventory.json', result)
    print('INVENTORY', result['counts'], flush=True)


def tensor_audit():
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    inventory = read(OUT / 'inventory.json')
    result = dict(schema='confirmation_tensor_recovery.v1', started_utc=now(),
                  environment=dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                                   platform=platform.platform(), threads=2, interop_threads=1,
                                   device='cpu', weights_only=True, mmap=True),
                  inventory_record=record(OUT / 'inventory.json'), artifacts={}, pairs={}, samples=[])

    def original(key):
        copies = inventory['binaries'][key]['copies']
        p = Path(next(c['path'] for c in copies if c['status'] == 'complete_hash_verified'))
        prior = next(c['actual'] for c in copies if c['path'] == str(p))
        assert guard(p) == prior['stat_after'], f'changed since inventory: {p}'
        return p

    def load_validate(path, kind, identity, expected_sha):
        before = guard(path)
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        arms = identity.get('bank_arms'); layers = identity['source_layers']; width = identity['d_model']
        if kind == 'checkpoint':
            expected = dict(schema_version=1, kind='checkpoint', fit_identity_sha256=expected_sha,
                            n_done=100, next_idx=100, d_model=width, source_layers=layers,
                            completed_prefix_sha256=digest(identity['prompt_sha256']))
            key='jacobian_sum'; dtype=torch.float32
        else:
            expected=dict(n_prompts=100, d_model=width, source_layers=layers)
            if arms is not None:
                expected.update(schema_version=1, kind='named_jacobian_lenses', bank_arms=arms)
            key='J'; dtype=torch.float16
        assert type(payload) is dict and set(payload) == set(expected) | {key}
        assert all(payload[k] == v for k, v in expected.items())
        banks=payload[key]
        if arms is not None:
            assert type(banks) is dict and set(banks) == set(arms)
        rows=[]
        for arm in arms or [None]:
            bank=banks if arm is None else banks[arm]
            assert type(bank) is dict and set(bank)==set(layers) and all(type(k) is int for k in bank)
            for layer in layers:
                t=bank[layer]
                assert type(t) is torch.Tensor and t.device.type=='cpu' and t.layout==torch.strided
                assert t.dtype==dtype and tuple(t.shape)==(width,width) and not t.requires_grad
                assert bool(torch.isfinite(t).all()), (path,arm,layer)
                rows.append(dict(arm=arm, layer=layer, shape=list(t.shape), dtype=str(t.dtype),
                                 stride=list(t.stride()), finite=True, numel=t.numel()))
        assert guard(path)==before
        result['artifacts'][str(path)]=dict(status='passed', kind=kind, exact_schema=True,
              stat_before=before, stat_after=guard(path), layers=rows, tensor_count=len(rows),
              total_elements=sum(r['numel'] for r in rows))
        return payload

    for fit in ['ouro/fit_01', 'controls/ouro_penultimate/fit_01', 'controls/ouro_positions/fit_01', 'ouro/fit_02']:
        prov=inventory['provenance'][fit]; ident=prov['identity']; sha=prov['identity_sha256']
        ckey=prov['generations']['checkpoint']['binary_key']; fkey=prov['generations']['final']['binary_key']
        cp=original(ckey); state=load_validate(cp,'checkpoint',ident,sha)
        source=state['jacobian_sum']; recovered=fit=='ouro/fit_02'
        if recovered:
            dest=OUT/'reconstructed/ouro/fit_02'; dest.mkdir(parents=True,exist_ok=True)
            fp=dest/'lens.pt'
            assert not fp.exists(), 'refuse to overwrite previously reconstructed bank'
            # Independently expressed explicit FP32 division and FP16 conversion.
            mean={layer: torch.div(source[layer],100.0).to(dtype=torch.float16) for layer in source}
            payload={'J':mean,'n_prompts':100,'d_model':ident['d_model'],'source_layers':ident['source_layers']}
            temp=dest/'lens.pt.partial'
            with temp.open('xb') as handle:
                torch.save(payload,handle); handle.flush(); os.fsync(handle.fileno())
            os.replace(temp,fp)
            fd=os.open(dest,os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
            del payload,mean; gc.collect()
        else:
            fp=original(fkey)
        final=load_validate(fp,'final',ident,sha)
        rows=[]
        for arm in ident.get('bank_arms') or [None]:
            a=source if arm is None else source[arm]
            b=final['J'] if arm is None else final['J'][arm]
            for layer in ident['source_layers']:
                # Numpy is an independent CPU implementation of the producer's torch expression.
                import numpy as np
                src=a[layer].numpy(); expected=np.divide(src,np.float32(100),dtype=np.float32).astype(np.float16)
                got=b[layer].numpy(); differing=int(np.count_nonzero(expected.view(np.uint16)!=got.view(np.uint16)))
                assert differing==0,(fit,arm,layer,differing)
                rows.append(dict(arm=arm,layer=layer,compared_elements=int(got.size),differing_bits_elements=differing))
                if layer in [ident['source_layers'][0],ident['source_layers'][len(ident['source_layers'])//2],ident['source_layers'][-1]]:
                    for row,col in [(0,0),(0,1),(17,31),(ident['d_model']-1,ident['d_model']-1)]:
                        result['samples'].append(dict(fit=fit,arm=arm,layer=layer,row=row,column=col,
                          sum_fp32=float(src[row,col]), sum_uint32=int(src.view(np.uint32)[row,col]),
                          divided_fp32=float(np.float32(src[row,col]/np.float32(100))),
                          expected_fp16=float(expected[row,col]),expected_uint16=int(expected.view(np.uint16)[row,col]),
                          final_fp16=float(got[row,col]),final_uint16=int(got.view(np.uint16)[row,col])))
                del src,expected,got
        assert guard(cp)==result['artifacts'][str(cp)]['stat_before']
        assert guard(fp)==result['artifacts'][str(fp)]['stat_before']
        pair=dict(status='passed',checkpoint=str(cp),final=str(fp),reconstructed=recovered,
                  independent_comparison='NumPy FP32 divide by np.float32(100), astype(np.float16), compare all uint16 representations',
                  layers=rows,total_elements=sum(r['compared_elements'] for r in rows),differing_elements=0)
        if recovered:
            actual=record(fp); historical=inventory['binaries'][fkey]['expected']
            pair.update(reconstructed_record=actual,historical_final_record=historical,
                        historical_file_byte_identity=same(actual,historical),
                        numerical_identity_to_checkpoint='all 191 matrices / all entries independently bitwise verified',
                        numerical_identity_to_historical_final=('established by complete historical SHA256 match' if same(actual,historical) else 'not directly established: historical complete bank absent; exact sealed-checkpoint conversion established'))
            prefixes=[]
            for copy in inventory['binaries'][fkey]['copies']:
                if copy['status']!='truncated': continue
                p=Path(copy['path']); before=guard(p); h=hashlib.sha256(); remaining=before['bytes']
                with fp.open('rb') as f:
                    while remaining:
                        block=f.read(min(8*1024*1024,remaining)); assert block
                        h.update(block);remaining-=len(block)
                assert guard(p)==before==copy['actual']['stat_after']
                prefixes.append(dict(partial_path=str(p),bytes=before['bytes'],
                                     reconstructed_prefix_sha256=h.hexdigest(),partial_sha256=copy['actual']['sha256'],
                                     byte_prefix_equal=h.hexdigest()==copy['actual']['sha256']))
            pair['retained_partial_prefix_comparisons']=prefixes
        result['pairs'][fit]=pair
        print('VERIFIED',fit,pair['total_elements'],'reconstructed',recovered,flush=True)
        del state,source,final,a,b;gc.collect()
    # Preserve original partials and complete sources: recheck every hash-time stat guard.
    for entry in inventory['binaries'].values():
        for copy in entry['copies']:
            assert guard(copy['path'])==copy['actual']['stat_after']
    result['historical_file_stat_guards_after_all_work']='passed'
    result['finished_utc']=now();result['status']='passed_with_documented_missing_banks'
    atomic_json(OUT/'tensor_audit.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['inventory','tensors'])
    parser.add_argument('--output-dir', type=Path, help='Fresh directory for an independent reproduction; source roots remain bound to this script.')
    args=parser.parse_args()
    if args.output_dir is not None:
        OUT=args.output_dir.resolve()
        OUT.mkdir(parents=True,exist_ok=True)
    (inventory if args.stage=='inventory' else tensor_audit)()
