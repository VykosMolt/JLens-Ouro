"""Receiver-side numerical validation, independent of the GPU score producer.

This module is self-contained so the controller can pin its complete source.
The controller separately rehashes every exact-contract payload before/after.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def require(condition, message):
    if not condition: raise ValueError(message)

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def load(path):
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)

def finite(t, shape):
    require(type(t) is torch.Tensor and t.dtype == torch.float32 and tuple(t.shape) == tuple(shape)
            and t.device.type == 'cpu' and bool(torch.isfinite(t).all()), 'Invalid FP32 tensor')

def validate_outputs(*, root: Path, manifest: dict, contract: dict) -> dict:
    torch.set_num_threads(2)
    paths = sorted(manifest['files'])
    require(set(contract['required_checks']) == {'loadability','numerical'}, 'Unknown semantic checks')
    require(set(paths) <= set(contract['files']), 'Undeclared payload')
    spec = read(root/'run_spec.json')
    require(sha(root/'run_spec.json') == manifest['binding']['run_spec_sha256'], 'Run specification changed')
    require(spec['run_id'] == contract['run_id'] == manifest['binding']['run_id'], 'Run identity differs')
    require(sha(root/'benchmark.json') == spec['benchmark_record']['sha256'], 'Benchmark changed')
    require(sha(root/'population.json') == spec['population_record']['sha256'], 'Frozen population changed')
    population = read(root/'population.json')
    rows, forms = population['rows'], population['token_forms']['multihop']
    names = population['names']['multihop']
    n, count = len(rows), len(names)
    require(n == spec['planned_items'] and count == spec['planned_concepts'], 'Planned population differs')
    require(not population['cross_concept_token_collisions'], 'Controls contain overlapping own token aliases')
    require(all(len(r['token_ids']) == r['n_tokens'] and 0 < r['n_tokens'] <= 512 and r['readout_position'] == -1 for r in rows), 'Input geometry differs')
    for row in rows:
        own = row['own_index']
        for slot, yes in enumerate(row['eligible']):
            require(type(yes) is bool, 'Eligibility not boolean')
            if yes:
                require(row['control_indices'][slot] == [j for j in range(count) if j not in own], 'Control catalogue changed')
                require(not set(row['intermediate_tokens'][row['intermediates'][slot]]) & set(row['token_ids']), 'Eligible concept leaked')
    require(sha(root/'provenance.json') != '', 'Provenance unreadable')
    provenance = read(root/'provenance.json')
    require(provenance['binding'] == manifest['binding'], 'Actual deployed provenance differs')
    require(provenance['source_records'] == spec['source_records'], 'Executed source pins differ')
    require(provenance['model_record'] == spec['model_record'], 'Executed model differs')
    require(provenance['bank_records'] == spec['bank_records'], 'Executed bank differs')
    require(provenance['dtype'] == 'torch.bfloat16' and provenance['geometry'] == [4,48,192,2048], 'Runtime geometry/dtype differs')
    require(provenance['runtime']['precision'] == spec['original_precision'], 'Readout precision changed')
    details = {'validated_arms': [], 'native_state_extraction': False, 'transport_samples': 0, 'full_sort_samples': 0}
    if 'development/native.pt' in paths:
        native = load(root/'development/native.pt')
        m = len(native['item_names'])
        require(native['item_names'] == spec['development_item_names'], 'Development items differ')
        for key in ('virtual_states','physical_states','wrapper_states'): finite(native[key], (m,192,2048))
        for key in ('native_logits','unembedded_logits'): finite(native[key], (m,49152))
        require(torch.equal(native['virtual_states'],native['physical_states']) and torch.equal(native['virtual_states'],native['wrapper_states']), 'Independent physical/native/wrapper extraction disagrees')
        require(torch.equal(native['native_logits'],native['unembedded_logits']), 'Native final logits disagree with unembedding')
        details['native_state_extraction'] = True
    if 'common/cache.pt' in paths:
        cache = load(root/'common/cache.pt')
        require(set(cache) == {'H','exit_logits','continuations'}, 'Unexpected cache schema')
        finite(cache['H'], (n,192,2048)); finite(cache['exit_logits'], (n,4,49152))
        require(len(cache['continuations']) == n and all(type(x) is str for x in cache['continuations']), 'Continuation count differs')
    for arm, support in spec['arm_support'].items():
        rel = f'readouts/{arm}.npz'
        if rel not in paths: continue
        require('common/cache.pt' in paths, 'Readout stage lacks common cache')
        with np.load(root/rel, allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        columns = len(support)
        expected = {'rank','allrank','top1','kl_to_final','kl_to_local','rank_of_final_top1','rank_of_local_top1','top10_ids'}
        require(set(arrays) == expected, 'Readout schema changed')
        for key, a in arrays.items():
            shape = (n,128,columns) if key == 'allrank' else (n,3,columns) if key == 'rank' else (n,columns,10) if key == 'top10_ids' else (n,columns)
            dtype = np.float32 if key.startswith('kl_') else np.int64 if key == 'top1' else np.int32
            require(a.shape == shape and a.dtype == dtype and np.isfinite(a).all(), 'Readout shape, dtype or finite check failed')
            if not key.startswith('kl_'): require(np.all(a < 49152) and np.all(a >= (-1 if key in ('rank','allrank') else 0)), 'Invalid token/rank bounds')
        ranks, top = arrays['allrank'], arrays['top10_ids']
        require(np.all(ranks[:,:count] >= 0) and np.all(ranks[:,count:] == -1), 'Name padding/support invalid')
        require(np.all(np.diff(np.sort(top,axis=-1),axis=-1) > 0), 'Top-10 contains repeated token IDs')
        for i, row in enumerate(rows):
            for slot, j in enumerate(row['own_index']):
                require(np.array_equal(arrays['rank'][i,slot], ranks[i,j] if j >= 0 else np.full(columns,-1)), 'Own rank differs from name bank')
            for j, name in enumerate(names):
                hits = np.isin(top[i], forms[name]).any(axis=-1)
                require(np.array_equal(hits, ranks[i,j] < 10), 'Alias hit@10 not reproduced by saved top-10')
        sample_path = f'readouts/{arm}_samples.pt'
        require(sample_path in paths, 'Missing readout numerical samples')
        samples = load(root/sample_path)
        require([s['item_index'] for s in samples] == [0,1,2], 'Readout sample items changed')
        bank = None
        if arm != 'raw':
            entry = spec['bank_records'][arm]
            localpath = Path(entry['receiver_path'])
            require(localpath.is_file() and localpath.stat().st_size == entry['record']['bytes'] and sha(localpath) == entry['record']['sha256'], 'Outside-worker bank unavailable or hash differs')
            state = load(localpath)
            bank = state['J'] if arm not in ('sampled_sum','diagonal') else state['J'][arm]
        for sample in samples:
            i, vs = sample['item_index'], sample['virtual_indices']
            require(vs == [v for v in (0,47,95,143,170,176,181,189,190,191) if v in support], 'Sample locations changed')
            finite(sample['transported'], (len(vs),2048)); finite(sample['logits'], (len(vs),49152))
            order = sample['sorted_ids'].numpy()
            require(order.dtype == np.int32 and order.shape == (len(vs),49152), 'Full-sort sample geometry differs')
            require(np.array_equal(np.sort(order,axis=1),np.broadcast_to(np.arange(49152),order.shape)), 'Full-sort sample is not a vocabulary permutation')
            logits = sample['logits'].numpy()
            sorted_logits = np.take_along_axis(logits,order,axis=1)
            require(np.all(np.diff(sorted_logits,axis=1) <= 0), 'Saved full sort does not sort logits')
            inverse = np.argsort(order,axis=1)
            for k, v in enumerate(vs):
                column = support.index(v)
                require(np.array_equal(top[i,column],order[k,:10]), 'Top-10 differs from retained full sort')
                require(arrays['top1'][i,column] == np.argmax(logits[k]), 'Native argmax tie semantics differ')
                for j, name in enumerate(names):
                    require(ranks[i,j,column] == inverse[k,forms[name]].min(), 'Sampled full rank differs')
                h = cache['H'][i,v].numpy().astype(np.float64)
                expected_h = h if arm == 'raw' or v == 191 else bank[v].numpy().astype(np.float64) @ h
                observed = sample['transported'][k].numpy()
                require(np.allclose(observed,expected_h,rtol=2e-5,atol=2e-5), 'Independent FP64 bank-times-state sample disagrees')
                details['transport_samples'] += 1
                details['full_sort_samples'] += 1
        del bank, samples
        details['validated_arms'].append(arm)
    if manifest['kind'] == 'final':
        require(details['native_state_extraction'] and set(details['validated_arms']) == set(spec['arm_support']), 'Final acceptance lacks all required numerical validation')
    return {'schema': 'confirmation_validation.v1', 'status': 'passed', 'run_id': contract['run_id'],
            'manifest_sha256': hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest(),
            'contract_sha256': manifest['binding']['output_contract_sha256'], 'checked_files': paths,
            'checks': {check: True for check in contract['required_checks']}, 'details': details}
