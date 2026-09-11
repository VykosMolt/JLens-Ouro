#!/usr/bin/env python3
"""Restore and load the preservation bundle using bundle files only.

Rehashes every file, extracts the frozen code and checks it against its freeze, loads all four banks, rescores the
raw and fit01 arms (all 160 items, executed layout) and reruns the frozen analysis on the bundle's saved readouts.

  CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  PYTORCH_ALLOC_CONF=expandable_segments:True python -B restore_check.py
"""
import hashlib
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / 'bundle'
RESTORE = HERE / 'restore_check'
PAYLOAD = BUNDLE / 'accepted_final_payload/results'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


class Stub:
    n_ut, n_physical, n_layers, d_model = 4, 48, 192, 2048

    def __init__(self, unembed):
        self.input_device, self.unembed = torch.device('cuda'), unembed


def main():
    result = {'schema': 'preservation_restoration_check.v1', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'script_sha256': sha256(__file__), 'bundle': str(BUNDLE), 'same_physical_disk_as_sources': True}
    manifests = {part: json.loads(path.read_text()) for part in ('core', 'verification')
                 if (path := HERE / f'BUNDLE_MANIFEST_{part}.json').exists()}
    mismatched = [e['bundle_path'] for m in manifests.values() for e in m['entries']
                  if sha256(BUNDLE / e['bundle_path']) != e['sha256']]
    result['rehash'] = {part: {'files': m['files'], 'bytes': m['bytes'], 'manifest_problems': len(m['problems'])} for part, m in manifests.items()}
    result['rehash']['mismatched'] = mismatched

    freeze_dir = BUNDLE / 'confirmation_2026-09-09/corrections/native_check_v1'
    freeze = json.loads((freeze_dir / 'FREEZE.json').read_text())
    archive = freeze_dir / 'bundle.tar.gz'
    RESTORE.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(RESTORE, filter='data')
    code = RESTORE / 'bundle'
    members = {p.relative_to(code).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in sorted(code.rglob('*')) if p.is_file()}
    result['frozen_code'] = {'archive_matches_freeze': freeze['bundle'] == {'bytes': archive.stat().st_size, 'sha256': sha256(archive)},
                             'extracted_members_match_freeze': members == freeze['files'], 'members': len(members)}

    spec = json.loads((PAYLOAD / 'run_spec.json').read_text())
    result['banks'] = {}
    for arm, entry in spec['bank_records'].items():
        path = BUNDLE / entry['worker_path']
        state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        bank = state['J'][arm] if arm in ('sampled_sum', 'diagonal') else state['J']
        target = 190 if arm == 'penultimate' else 191
        result['banks'][arm] = {'file': entry['worker_path'], 'record_matches_run_spec': {'bytes': path.stat().st_size, 'sha256': sha256(path)} == entry['record'],
                                'keys_are_sources_0_to_target': sorted(bank) == list(range(target)),
                                'all_2048x2048_fp16': all(tuple(m.shape) == (2048, 2048) and m.dtype == torch.float16 for m in bank.values()),
                                'first_middle_last_finite': all(bool(torch.isfinite(bank[v]).all()) for v in (0, target // 2, target - 1))}

    sys.dont_write_bytecode = True
    sys.path[:0] = [str(code / part) for part in ('repo', 'ouro_project/src', 'legacy', 'evaluation', 'analysis')]
    import jlens
    from ouro_jlens import evaluate as evaluator, evaldata
    import evaluate_refits as common
    import readouts
    import analyze
    model = BUNDLE / 'model_snapshot'
    eps = json.loads((model / 'config.json').read_text())['rms_norm_eps']
    with safe_open(model / 'model.safetensors', 'pt') as tensors:
        weight, gain = tensors.get_tensor('lm_head.weight').cuda(), tensors.get_tensor('model.norm.weight').cuda()

    def unembed(residual):
        states = residual.to(torch.bfloat16).cuda().to(torch.float32)
        states = states * torch.rsqrt(states.pow(2).mean(-1, keepdim=True) + eps)
        return F.linear(gain * states.to(torch.bfloat16), weight)

    population = json.loads((PAYLOAD / 'population.json').read_text())
    items = [evaldata.Item(r['name'], 'multihop', r['prompt'], r['target'], r['intermediates'], r['token_ids'],
                           r['intermediate_tokens'], dict(zip(r['intermediates'], r['leaked']))) for r in population['rows']]
    cache = torch.load(PAYLOAD / 'common/cache.pt', map_location='cpu', weights_only=True)
    result['rescoring'] = {}
    with common.task_names(evaluator, items, tasks=('multihop',)), torch.no_grad():
        for arm in ('raw', 'fit01'):
            J = None
            if arm == 'fit01':
                lens = jlens.JacobianLens(torch.load(BUNDLE / 'banks/fit01.pt', map_location='cpu', weights_only=True, mmap=True)['J'],
                                          n_prompts=100, d_model=2048)
                J = common.prepare_readout(Stub(None), lens, target_layer=191, source_layers=list(range(191)), evaluator=evaluator).jacobians[:192]
                del lens
            outputs, _ = readouts.readout(evaluator, Stub(unembed), items, cache['H'], J, cache['exit_logits'].cuda(), 192)
            with np.load(PAYLOAD / f'readouts/{arm}.npz', allow_pickle=False) as saved:
                result['rescoring'][arm] = {key: bool(np.array_equal(outputs[key], saved[key])) for key in ('allrank', 'top10_ids', 'rank', 'top1')}
            del J, outputs
            torch.cuda.empty_cache()

    rerun, _ = analyze.analyze(PAYLOAD)
    saved = json.loads((BUNDLE / 'confirmation_2026-09-09/results/ouro_confirmation_20260911_fixed160_native1/analysis/analysis.json').read_text())
    result['analysis_rerun_equals_saved_json'] = json.loads(json.dumps(rerun, sort_keys=True)) == saved
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    with (HERE / 'RESTORATION_CHECK.json').open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps({k: result[k] for k in ('rehash', 'frozen_code', 'rescoring', 'analysis_rerun_equals_saved_json')}, indent=1))


if __name__ == '__main__':
    main()
