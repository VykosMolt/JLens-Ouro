#!/usr/bin/env python3
"""Export the retained inputs of the scoring-shape GPU check; no bank leaves this machine.

  inputs/states.pt       retained H cast to BF16 (lossless: every value is BF16) and the saved continuations
  inputs/fit01_rows.pt   fit01 transported rows from the frozen transport, cast to BF16 as the head does first
  inputs/samples.pt      every arm's saved sample rows (FP32 transported) with their saved logits (BF16-exact)
  inputs/development.pt  first-run tensors of the three development items from the native-logit diagnostic
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'replay'))
import replay_scorer as rs  # noqa: E402

DIAGNOSTIC = rs.ROUND / ('cloud_leases/diagnostic_native_logit_v1/handoff/accepted/ouro_native_logit_diagnostic_v1/'
                         'f9751036b1fa746fc8a37287c755093cde92238ead58347f78869eeb949063e6')
OUT = HERE / 'inputs'


def save(name, value):
    path = OUT / name
    with path.open('xb') as handle:
        torch.save(value, handle)
    return {'bytes': path.stat().st_size, 'sha256': rs.sha256(path)}


def main():
    OUT.mkdir()
    manifest = json.loads((rs.FINAL / 'MANIFEST.json').read_text())['files']
    for rel in ('common/cache.pt', 'run_spec.json', *(f'readouts/{arm}_samples.pt' for arm in rs.ARMS)):
        rs.verified(rs.RESULTS / rel, manifest[rel])
    spec = json.loads((rs.RESULTS / 'run_spec.json').read_text())
    records = {}

    cache = torch.load(rs.RESULTS / 'common/cache.pt', map_location='cpu', weights_only=True)
    H = cache['H']
    if not torch.equal(H.to(torch.bfloat16).float(), H):
        raise ValueError('retained states are not BF16-valued')
    records['states.pt'] = save('states.pt', {'H': H.to(torch.bfloat16), 'continuations': cache['continuations']})

    entry = spec['bank_records']['fit01']
    bank_path = rs.BANK_FILES[entry['worker_path']]
    rs.verified(bank_path, entry['record'])
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(rs.BUNDLE / part) for part in ('repo', 'ouro_project/src', 'legacy')]
    import jlens
    from ouro_jlens import evaluate as evaluator
    import evaluate_refits as common
    torch.backends.cuda.matmul.allow_tf32 = False
    lens = jlens.JacobianLens(torch.load(bank_path, map_location='cpu', weights_only=True, mmap=True)['J'],
                              n_prompts=100, d_model=2048)
    plan = common.prepare_readout(rs.Stub('cuda', None), lens, target_layer=191, source_layers=list(range(191)),
                                  evaluator=evaluator)
    J = plan.jacobians[:192]
    del lens, plan
    saved = torch.load(rs.RESULTS / 'readouts/fit01_samples.pt', map_location='cpu', weights_only=True)
    rows = torch.empty(len(H), 192, 2048, dtype=torch.bfloat16)
    checked = 0
    for i in range(len(H)):
        transported = torch.einsum('vde,ve->vd', J, H[i].to('cuda')).cpu()
        for s in saved:
            if s['item_index'] == i:
                if not torch.equal(transported[s['virtual_indices']], s['transported']):
                    raise ValueError('local transport differs from the saved fit01 samples')
                checked += len(s['virtual_indices'])
        rows[i] = transported.to(torch.bfloat16)
    del J
    records['fit01_rows.pt'] = save('fit01_rows.pt', {'rows': rows, 'saved_sample_rows_matched': checked})

    samples = {}
    for arm in rs.ARMS:
        entries = torch.load(rs.RESULTS / f'readouts/{arm}_samples.pt', map_location='cpu', weights_only=True)
        for s in entries:
            if not torch.equal(s['logits'].to(torch.bfloat16).float(), s['logits']):
                raise ValueError('saved sample logits are not BF16-valued')
        samples[arm] = [{'item_index': s['item_index'], 'virtual_indices': s['virtual_indices'],
                         'transported': s['transported'], 'logits': s['logits'].to(torch.bfloat16)} for s in entries]
    records['samples.pt'] = save('samples.pt', samples)

    diagnostic_manifest = json.loads((DIAGNOSTIC / 'MANIFEST.json').read_text())['files']
    rs.verified(DIAGNOSTIC / 'results/diagnostic/tensors.pt', diagnostic_manifest['diagnostic/tensors.pt'])
    comparisons = json.loads((DIAGNOSTIC / 'results/diagnostic/comparisons.json').read_text())
    tensors = torch.load(DIAGNOSTIC / 'results/diagnostic/tensors.pt', map_location='cpu', weights_only=True)
    development = {}
    for name in tensors['item_names']:
        item, run = tensors['items'][name], tensors['items'][name]['runs'][0]
        development[name] = {'token_ids': item['token_ids'], 'exit_step': comparisons['exit_steps'][name]['exit_step'],
                             **{key: run[key] for key in ('norm_input', 'norm_output', 'lm_head_input', 'lm_head_output', 'logits')},
                             'retained_shapes': {family: item[family] for family in ('norm', 'lm_head', 'lm_head_norm')}}
    records['development.pt'] = save('development.pt', development)

    with (OUT / 'INPUTS.json').open('x') as handle:
        json.dump({'schema': 'shape_check_inputs.v1', 'script_sha256': rs.sha256(Path(__file__)), 'files': records,
                   'fit01_saved_sample_rows_matched': checked}, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(records, indent=1))


if __name__ == '__main__':
    main()
