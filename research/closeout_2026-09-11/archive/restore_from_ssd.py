#!/usr/bin/env python3
"""Isolated restoration test of the SSD archive (run inside restore_isolated.sh: user+mount+net namespaces,
live project/cache paths bind-mounted empty, HF offline). Uses ONLY files under the archive copy of the
preservation bundle plus the venv interpreter. Loads all four banks (five arms), reruns the frozen analysis
(primary + 20 secondaries) from archived readouts, rescoring replay for raw and fit01 on the archived states."""
import hashlib, json, os, sys, tarfile, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from safetensors import safe_open
ARCH = Path('/run/media/moloch/ARCH_BACKUP/JLENS_COLD_ARCHIVE_20260911')
PRES = ARCH / 'tree/home/moloch/jacobian-lens/research/verification_2026-09-11/preservation'
BUNDLE = PRES / 'bundle'; PAYLOAD = BUNDLE / 'accepted_final_payload/results'
OUT = Path(sys.argv[1]); WORK = Path(sys.argv[2]); WORK.mkdir(parents=True, exist_ok=True)
def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()
res = {'schema': 'ssd_restoration_check.v1', 'started_utc': datetime.now(timezone.utc).isoformat(), 'archive': str(ARCH), 'bundle': str(BUNDLE)}
res['isolation'] = {'live_research_visible': os.path.exists('/home/moloch/jacobian-lens/research/verification_2026-09-11/preservation/bundle/banks/fit01.pt'),
                    'live_ouro_artifacts_visible': os.path.exists('/home/moloch/ouro_project/artifacts/hf_cache/hub/models--ByteDance--Ouro-2.6B/blobs'),
                    'live_hf_cache_visible': os.path.exists('/home/moloch/.cache/huggingface/hub/models--Vykos--ouro-jlens-results/blobs'),
                    'HF_HUB_OFFLINE': os.environ.get('HF_HUB_OFFLINE'), 'network': None}
try:
    import socket; s = socket.create_connection(('1.1.1.1', 53), timeout=2); s.close(); res['isolation']['network'] = 'REACHABLE'
except Exception as e: res['isolation']['network'] = f'unreachable ({type(e).__name__})'
# 1. bundle re-hash SKIPPED here: the full 52 GB destination hash verification already passed (ACCEPTANCE_RECEIPT_partial.json, 0 problems)
res['rehash_mismatched'] = 'skipped (destination hash verification of all 16,200 files already passed in build_archive.py)'
# 2. frozen code archive
fz = BUNDLE / 'confirmation_2026-09-09/corrections/native_check_v1'; freeze = json.loads((fz / 'FREEZE.json').read_text())
with tarfile.open(fz / 'bundle.tar.gz') as tar: tar.extractall(WORK, filter='data')
code = WORK / 'bundle'
members = {p.relative_to(code).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in sorted(code.rglob('*')) if p.is_file()}
res['frozen_code'] = {'archive_matches_freeze': freeze['bundle'] == {'bytes': (fz / 'bundle.tar.gz').stat().st_size, 'sha256': sha256(fz / 'bundle.tar.gz')}, 'members_match_freeze': members == freeze['files'], 'members': len(members)}
# 3. banks
spec = json.loads((PAYLOAD / 'run_spec.json').read_text()); res['banks'] = {}
for arm, entry in spec['bank_records'].items():
    p = BUNDLE / entry['worker_path']; st = torch.load(p, map_location='cpu', weights_only=True, mmap=True)
    bank = st['J'][arm] if arm in ('sampled_sum', 'diagonal') else st['J']; target = 190 if arm == 'penultimate' else 191
    res['banks'][arm] = {'file': entry['worker_path'], 'record_matches_run_spec': {'bytes': p.stat().st_size, 'sha256': sha256(p)} == entry['record'], 'keys_are_sources_0_to_target': sorted(bank) == list(range(target)),
                         'all_2048x2048_fp16': all(tuple(m.shape) == (2048, 2048) and m.dtype == torch.float16 for m in bank.values()), 'first_middle_last_finite': all(bool(torch.isfinite(bank[v]).all()) for v in (0, target // 2, target - 1))}
    del st, bank
# 4. frozen analysis rerun from archived readouts
sys.dont_write_bytecode = True; sys.path[:0] = [str(code / part) for part in ('repo', 'ouro_project/src', 'legacy', 'evaluation', 'analysis')]
import jlens, analyze, readouts
from ouro_jlens import evaluate as evaluator, evaldata
import evaluate_refits as common
rerun, _ = analyze.analyze(PAYLOAD)
saved = json.loads((BUNDLE / 'confirmation_2026-09-09/results/ouro_confirmation_20260911_fixed160_native1/analysis/analysis.json').read_text())
res['analysis_rerun_equals_saved_json'] = json.loads(json.dumps(rerun, sort_keys=True)) == saved
res['primary_from_archive'] = rerun['estimates'][0]; res['primary_interval_from_archive'] = rerun['family_percentile_95_intervals'][0]; res['secondary_from_archive'] = [(s['id'], s['estimate'], s['simultaneous_95_interval']) for s in rerun['secondary_family']]
# 5. bounded readout replay for raw and fit01 from archived states + archived weights
model = BUNDLE / 'model_snapshot'; eps = json.loads((model / 'config.json').read_text())['rms_norm_eps']
with safe_open(model / 'model.safetensors', 'pt') as t: weight, gain = t.get_tensor('lm_head.weight').cuda(), t.get_tensor('model.norm.weight').cuda()
def unembed(residual):
    s = residual.to(torch.bfloat16).cuda().to(torch.float32); s = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)
    return F.linear(gain * s.to(torch.bfloat16), weight)
class Stub:
    n_ut, n_physical, n_layers, d_model = 4, 48, 192, 2048
    def __init__(self, u): self.input_device, self.unembed = torch.device('cuda'), u
pop = json.loads((PAYLOAD / 'population.json').read_text())
items = [evaldata.Item(r['name'], 'multihop', r['prompt'], r['target'], r['intermediates'], r['token_ids'], r['intermediate_tokens'], dict(zip(r['intermediates'], r['leaked']))) for r in pop['rows']]
cache = torch.load(PAYLOAD / 'common/cache.pt', map_location='cpu', weights_only=True); res['rescoring'] = {}
with common.task_names(evaluator, items, tasks=('multihop',)), torch.no_grad():
    for arm in ('raw', 'fit01'):
        J = None
        if arm == 'fit01':
            lens = jlens.JacobianLens(torch.load(BUNDLE / 'banks/fit01.pt', map_location='cpu', weights_only=True, mmap=True)['J'], n_prompts=100, d_model=2048)
            J = common.prepare_readout(Stub(None), lens, target_layer=191, source_layers=list(range(191)), evaluator=evaluator).jacobians[:192]; del lens
        out, _ = readouts.readout(evaluator, Stub(unembed), items, cache['H'], J, cache['exit_logits'].cuda(), 192)
        with np.load(PAYLOAD / f'readouts/{arm}.npz', allow_pickle=False) as sv: res['rescoring'][arm] = {k: bool(np.array_equal(out[k], sv[k])) for k in ('allrank', 'top10_ids', 'rank', 'top1')}
        del J, out; torch.cuda.empty_cache()
res['replay_levels'] = {'level1_tables_from_saved_scores': 'executed (analysis rerun)', 'level2_rescoring_from_archived_banks_and_states': 'executed for raw and fit01 (all 160 items, executed layout)',
                       'level2_other_arms': 'NOT rerun here (fit02, penultimate, sampled_sum, diagonal loaded and shape/finite-checked only)', 'level3_regeneration_from_prompts': 'NOT run', 'numerical_sensitivity_suite': 'NOT repeated (see verification_2026-09-11)'}
res['gpu'] = torch.cuda.get_device_name(0); res['finished_utc'] = datetime.now(timezone.utc).isoformat()
OUT.write_text(json.dumps(res, indent=1)); print(json.dumps({k: res[k] for k in ('isolation', 'rehash_mismatched', 'frozen_code', 'analysis_rerun_equals_saved_json', 'primary_from_archive', 'rescoring')}, indent=1))
