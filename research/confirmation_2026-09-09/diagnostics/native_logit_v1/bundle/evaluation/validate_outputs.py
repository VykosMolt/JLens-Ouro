"""Receiver-side validation of the native-logit diagnostic outputs.

Self-contained so the controller can pin its source. The worker imports the
comparison definitions below, so both sides compute records identically.
Identity, schema, dtypes, shapes and finiteness are required, and every recorded
comparison must equal its recomputation from the saved tensors. Measured
mismatches are data and never fail validation.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import torch

WIDTH, VOCAB, STEPS = 2048, 49152, 4
GPU = 'NVIDIA GeForce RTX 5090'
NORM_LABELS = ('[2048]', '[1,2048]', '[1,1,2048]', '[S,2048]', '[1,S,2048]',
               '[148,2048]/first', '[148,2048]/last', '[160,2048]/first', '[160,2048]/last')
COMPOSITE_LABELS = ('[1,1,2048]', '[1,S,2048]', '[160,2048]/first', '[160,2048]/last')
M1_PAIRS = (('virtual_states', 'physical_states'), ('virtual_states', 'wrapper_states'),
            ('native_logits', 'unembedded_logits'))
RUN_KEYS = ('norm_input', 'norm_output', 'lm_head_input', 'lm_head_output', 'logits', 'virtual191')
BITS = {torch.bfloat16: torch.int16, torch.float32: torch.int32}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def run_shapes(s):
    bf16 = torch.bfloat16
    return dict(zip(RUN_KEYS, ((bf16, (STEPS, 1, s, WIDTH)), (bf16, (STEPS, 1, s, WIDTH)), (bf16, (1, s, WIDTH)),
                               (bf16, (1, s, VOCAB)), (bf16, (1, s, VOCAB)), (torch.float32, (WIDTH,)))))


def exit_step(run):
    """Steps whose last-token norm output equals the lm_head input row bitwise; the unique match, else the last step."""
    target = run['lm_head_input'][0, -1].view(torch.int16)
    matches = [step for step in range(STEPS) if torch.equal(run['norm_output'][step, 0, -1].view(torch.int16), target)]
    return matches, matches[0] if len(matches) == 1 else STEPS - 1


def compare(name, left, right):
    """torch.equal and the bitwise differing-element count decide; the other fields are descriptive."""
    require(left.dtype == right.dtype and left.shape == right.shape, 'Incomparable operands: ' + name)
    difference = (left.double() - right.double()).abs().flatten()
    result = {'name': name, 'dtype': str(left.dtype), 'shape': list(left.shape),
              'torch_equal': torch.equal(left, right),
              'differing_elements': int((left.view(BITS[left.dtype]) != right.view(BITS[right.dtype])).sum()),
              'max_abs_diff': float(difference.max()),
              # Elementwise float64 operations and fsum do not depend on reduction order.
              'mean_abs_diff': math.fsum(difference.tolist()) / difference.numel()}
    if left.shape == (VOCAB,):
        top = [torch.sort(t.double(), descending=True, stable=True).indices[:10].tolist() for t in (left, right)]
        result.update(top1_equal=top[0][0] == top[1][0], top10_set_equal=set(top[0]) == set(top[1]))
    return result


def comparisons(native, tensors):
    """Every comparison, in a fixed order, from the M1 tensors and the diagnostic tensors."""
    records, exit_steps = [], {}
    def add(name, left, right):
        records.append(compare(name, left, right))
    for left, right in M1_PAIRS:
        add(f'M1/all/{left}~{right}', native[left], native[right])
    for i, name in enumerate(tensors['item_names']):
        entry = tensors['items'][name]
        first, second = entry['runs']
        for left, right in M1_PAIRS:
            add(f'M1/{name}/{left}~{right}', native[left][i], native[right][i])
        (first_matches, step), (second_matches, _) = exit_step(first), exit_step(second)
        exit_steps[name] = {'M2_matching_steps': first_matches, 'M4_matching_steps': second_matches, 'exit_step': step}
        for label, run in (('M2', first), ('M4', second)):
            add(f'{label}/{name}/logits~lm_head_output', run['logits'], run['lm_head_output'])
            add(f'{label}/{name}/logits[0,-1]~M1_native_logits', run['logits'][0, -1].float(), native['native_logits'][i])
            add(f'{label}/{name}/virtual191~M1_virtual_states[191]', run['virtual191'], native['virtual_states'][i, 191])
        for key in RUN_KEYS:
            add(f'M4/{name}/{key}~M2', second[key], first[key])
        norm_row, head_row, logits_row = first['norm_output'][step, 0, -1], first['lm_head_output'][0, -1], first['logits'][0, -1]
        recomputed = [*((f'norm{label}', entry['norm'][label], norm_row) for label in NORM_LABELS),
                      *((f'lm_head{label}', entry['lm_head'][label], head_row) for label in NORM_LABELS),
                      *((f'lm_head_norm{label}', entry['lm_head_norm'][label], logits_row) for label in COMPOSITE_LABELS),
                      ('unembed_fp32_virtual191', entry['unembed_fp32_virtual191'], logits_row)]
        for label, rows, reference in recomputed:
            add(f'M3/{name}/{label}#1~native', rows[0], reference)
            add(f'M3/{name}/{label}#2~native', rows[1], reference)
            add(f'M3/{name}/{label}#1~#2', rows[0], rows[1])
    return {'schema': 'native_logit_comparisons.v1', 'exit_steps': exit_steps, 'comparisons': records}


def tensor(value, dtype, shape, name):
    require(type(value) is torch.Tensor and value.dtype == dtype and tuple(value.shape) == tuple(shape)
            and value.device.type == 'cpu' and bool(torch.isfinite(value).all()),
            'Invalid tensor shape, dtype or values: ' + name)


def check_payloads(native, tensors, names):
    require(isinstance(native, dict) and set(native) == {'item_names', *(key for pair in M1_PAIRS for key in pair)}
            and native['item_names'] == names, 'M1 schema differs')
    for key in ('virtual_states', 'physical_states', 'wrapper_states'):
        tensor(native[key], torch.float32, (len(names), 192, WIDTH), key)
    for key in ('native_logits', 'unembedded_logits'):
        tensor(native[key], torch.float32, (len(names), VOCAB), key)
    require(isinstance(tensors, dict) and set(tensors) == {'item_names', 'items'} and tensors['item_names'] == names
            and isinstance(tensors['items'], dict) and list(tensors['items']) == names, 'Diagnostic schema differs')
    for name in names:
        entry = tensors['items'][name]
        require(isinstance(entry, dict) and set(entry) == {'token_ids', 'runs', 'norm', 'lm_head', 'lm_head_norm',
                                                           'unembed_fp32_virtual191'}, 'Item schema differs: ' + name)
        ids = entry['token_ids']
        require(isinstance(ids, list) and ids and all(type(t) is int for t in ids), 'Invalid token IDs: ' + name)
        require(isinstance(entry['runs'], list) and len(entry['runs']) == 2, 'M2 and M4 runs differ: ' + name)
        for run in entry['runs']:
            require(isinstance(run, dict) and set(run) == set(RUN_KEYS), 'Run schema differs: ' + name)
            for key, (dtype, shape) in run_shapes(len(ids)).items():
                tensor(run[key], dtype, shape, f'{name}/{key}')
        for family, labels, width in (('norm', NORM_LABELS, WIDTH), ('lm_head', NORM_LABELS, VOCAB),
                                      ('lm_head_norm', COMPOSITE_LABELS, VOCAB)):
            require(isinstance(entry[family], dict) and set(entry[family]) == set(labels),
                    'Recomputation schema differs: ' + name)
            for label in labels:
                tensor(entry[family][label], torch.bfloat16, (2, width), f'{name}/{family}{label}')
        tensor(entry['unembed_fp32_virtual191'], torch.bfloat16, (2, VOCAB), name + '/unembed_fp32_virtual191')


def validate_outputs(*, root: Path, manifest: dict, contract: dict) -> dict:
    root = Path(root)
    paths = sorted(manifest['files'])
    binding = manifest['binding']
    require(manifest['kind'] == 'final' and manifest['stage_id'] is None and manifest['outcome'] == 'complete',
            'Only a complete final manifest is accepted')
    require(contract['stages'] == {} and set(contract['required_checks']) == {'loadability', 'numerical'},
            'Unexpected contract scope')
    require(paths == sorted(contract['files']), 'Manifest files differ from the contract')
    present = sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_symlink() or not p.is_dir())
    require(present == paths, 'Payload files differ from the manifest')
    require(all(record(root/p) == manifest['files'][p] for p in paths), 'Payload bytes differ from the manifest')
    spec = json.loads((root/'run_spec.json').read_text())
    require(record(root/'run_spec.json')['sha256'] == binding['run_spec_sha256'] == contract['run_spec_sha256'],
            'Run specification differs from the binding')
    require(spec['run_id'] == contract['run_id'] == binding['run_id'], 'Run identity differs')
    provenance = json.loads((root/'provenance.json').read_text())
    require(provenance['binding'] == binding, 'Provenance binding differs')
    require(provenance['source_records'] == spec['source_records']
            and provenance['actual_worker_source'] == spec['source_records']['evaluation/worker.py'], 'Executed sources differ')
    require(provenance['model_record'] == spec['model_record'], 'Executed model differs')
    require(provenance['dtype'] == 'torch.bfloat16' and provenance['geometry'] == [STEPS, 48, 192, WIDTH],
            'Runtime geometry or dtype differs')
    require(provenance['runtime']['precision'] == spec['original_precision'], 'Precision differs from the frozen original')
    require(provenance['runtime']['gpu']['name'] == GPU, 'GPU type differs')
    native = torch.load(root/'development/native.pt', map_location='cpu', weights_only=True, mmap=True)
    tensors = torch.load(root/'diagnostic/tensors.pt', map_location='cpu', weights_only=True, mmap=True)
    check_payloads(native, tensors, spec['development_item_names'])
    require(json.loads((root/'diagnostic/comparisons.json').read_text()) == comparisons(native, tensors),
            'Comparison record differs from its recomputation')
    return {'schema': 'confirmation_validation.v1', 'status': 'passed', 'run_id': contract['run_id'],
            'manifest_sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':'),
                                                         ensure_ascii=False, allow_nan=False).encode()).hexdigest(),
            'contract_sha256': binding['output_contract_sha256'], 'checked_files': paths,
            'checks': {check: True for check in contract['required_checks']}}
