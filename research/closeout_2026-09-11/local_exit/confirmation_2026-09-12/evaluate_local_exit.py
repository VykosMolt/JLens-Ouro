#!/usr/bin/env python3
"""Step 2 of the local-exit proposal, executed per ANALYSIS_PLAN.md (frozen before this ran):
own-exit (initial-study exit0/1/2) versus final-target (exit3) versus raw readouts on the accepted confirmation
states, passes 1-3, physical band 26-37, with the frozen scorer, frozen groups and frozen resampler.
Environment check first: the frozen scorer on these states must reproduce the accepted raw readouts bit for bit
(and fit01, when that bank is on disk)."""
import csv, hashlib, json, os, platform, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from safetensors import safe_open
HERE = Path(__file__).resolve().parent
CO = HERE.parents[1]; HAND = CO / 'handoff/JLens_Ouro_Writer_Handoff_v2'; CODE = HAND / '08_code'; PAY = HAND / '05_data/accepted_payload'
FC = Path('/tmp/claude-1000/-home-moloch-jacobian-lens/fcb94bb7-42d0-49fd-8993-7db96deb86a0/scratchpad/frozen_code'); FC.mkdir(parents=True, exist_ok=True)
for name, src in (('jlens', CODE / 'frozen_jlens'), ('ouro_jlens', CODE / 'frozen_ouro_jlens')):
    if not (FC / name).exists(): (FC / name).symlink_to(src)
sys.dont_write_bytecode = True; sys.path[:0] = [str(FC), str(CODE / 'frozen_evaluation'), str(CODE / 'frozen_analysis'), str(CODE / 'frozen_legacy')]
import jlens, analyze, readouts
from ouro_jlens import evaluate as evaluator, evaldata
import evaluate_refits as common

def sha(p):
    m = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''): m.update(c)
    return m.hexdigest()
PLAN = HERE / 'ANALYSIS_PLAN.md'; PLAN_SHA = sha(PLAN); assert PLAN_SHA.startswith('16cdcbd071e86449'), PLAN_SHA
CACHE = Path('/home/moloch/ouro_project/jacobian-lens/research/confirmation_2026-09-09/cloud_leases/attempt_06/handoff/accepted/ouro_confirmation_20260911_fixed160_native1/b355a0ea76244df587e8ea2865e40d125ba28eff09d1d8cc9a6f10e2c41aae9e/results/common/cache.pt')
UNEMBED = Path('/home/moloch/ouro_project/artifacts/jlens/retrieved_2026-09-12/ouro26b_unembed_1ed04250.safetensors')
MERGED = Path('/home/moloch/ouro_project/artifacts/jlens/retrieved_2026-09-12/merged')
CONFIG = Path.home() / '.cache/huggingface/hub/models--ByteDance--Ouro-2.6B/snapshots/1ed04250da1a9936042725d302e81c8fa2ab5abd/config.json'
FIT01 = os.environ.get('FIT01_BANK')  # optional: path to banks/fit01.pt (sha 90f01f6a…)
SEED_FAMILY = 2026091201
BAND = {p: list(range(48 * (p - 1) + 25, 48 * (p - 1) + 37)) for p in (1, 2, 3, 4)}
ARMS_NEW = {'exit0': 47, 'exit1': 95, 'exit2': 143, 'exit3': 191}
OWN = {1: 'exit0', 2: 'exit1', 3: 'exit2'}
(HERE / 'readouts').mkdir(exist_ok=True)
t0 = time.time()
merge_rec = json.load(open(HERE / 'MERGE_RECORD.json'))
assert merge_rec['exit3_equivalence_with_pod_merge']['byte_identical_files'] or merge_rec['exit3_equivalence_with_pod_merge']['max_abs_difference'] <= 2 ** -9, merge_rec['exit3_equivalence_with_pod_merge']

pop = json.load(open(PAY / 'population.json')); rows = pop['rows']; n = len(rows); names = pop['names']['multihop']
items = [evaldata.Item(r['name'], 'multihop', r['prompt'], r['target'], r['intermediates'], r['token_ids'], r['intermediate_tokens'], dict(zip(r['intermediates'], r['leaked']))) for r in rows]
own = np.array([r['own_index'][0] for r in rows]); ctrl = [r['control_indices'][0] for r in rows]; groups = [r['dependency_group_id'] for r in rows]
assert all(len(c) == 79 and own[i] not in c for i, c in enumerate(ctrl)) and all(r['eligible'] == [True] for r in rows)
cache = torch.load(CACHE, map_location='cpu', weights_only=True); H = cache['H']; exits = cache['exit_logits'].cuda()
assert tuple(H.shape) == (n, 192, 2048)
eps = json.load(open(CONFIG))['rms_norm_eps']
with safe_open(str(UNEMBED), 'pt') as t: weight, gain = t.get_tensor('lm_head.weight').cuda(), t.get_tensor('model.norm.weight').cuda()
def unembed(residual):
    s = residual.to(torch.bfloat16).cuda().to(torch.float32); s = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)
    return F.linear(gain * s.to(torch.bfloat16), weight)
class Stub:
    n_ut, n_physical, d_model = 4, 48, 2048
    def __init__(self, u, n_layers=192): self.input_device, self.unembed, self.n_layers = torch.device('cuda'), u, n_layers
saved = {arm: dict(np.load(PAY / f'readouts/{arm}.npz', allow_pickle=False)) for arm in ('raw', 'fit01')}
res = {'schema': 'local_exit_confirmation_population.v1', 'plan': {'path': str(PLAN.relative_to(CO)), 'sha256': PLAN_SHA}, 'seed_family': SEED_FAMILY,
       'inputs': {'cache_pt': {'path': str(CACHE), 'sha256': sha(CACHE)}, 'unembed_tensors': {'path': str(UNEMBED), 'sha256': sha(UNEMBED)}, 'config_rms_norm_eps': eps,
                  'merged_banks': {k: v['merged'] for k, v in merge_rec['banks'].items()}, 'population_sha256': sha(PAY / 'population.json')},
       'environment': {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'python': platform.python_version(), 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}, 'checks': {}}
KEYS = ('allrank', 'rank', 'top1', 'top10_ids')
with common.task_names(evaluator, items, tasks=('multihop',)), torch.no_grad():
    out, _ = readouts.readout(evaluator, Stub(unembed), items, H, None, exits, 192)
    res['checks']['raw_reproduced_bit_for_bit'] = {k: bool(np.array_equal(out[k], saved['raw'][k])) for k in KEYS}
    assert all(res['checks']['raw_reproduced_bit_for_bit'].values()), res['checks']['raw_reproduced_bit_for_bit']
    print('check: raw readouts reproduced bit for bit from these states with the fetched unembedding tensors', flush=True)
    if FIT01 and Path(FIT01).exists():
        lens = jlens.JacobianLens(torch.load(FIT01, map_location='cpu', weights_only=True, mmap=True)['J'], n_prompts=100, d_model=2048)
        J = common.prepare_readout(Stub(None), lens, target_layer=191, source_layers=list(range(191)), evaluator=evaluator).jacobians[:192]; del lens
        out, _ = readouts.readout(evaluator, Stub(unembed), items, H, J, exits, 192)
        res['checks']['fit01_reproduced_bit_for_bit'] = {k: bool(np.array_equal(out[k], saved['fit01'][k])) for k in KEYS}; del J; torch.cuda.empty_cache()
        print('check: fit01', res['checks']['fit01_reproduced_bit_for_bit'], flush=True)
    else:
        res['checks']['fit01_reproduced_bit_for_bit'] = 'not run: fit01 bank not on this machine (its readouts are reused from the accepted payload, which the restoration test reproduced bit for bit)'
    allrank = {'raw': saved['raw']['allrank'], 'fit01': saved['fit01']['allrank']}
    for arm, target in ARMS_NEW.items():
        t = time.time()
        lens = jlens.JacobianLens.load(str(MERGED / f'{arm}.pt'))
        prep = common.prepare_readout(Stub(None), lens, target_layer=target, source_layers=list(range(target)), evaluator=evaluator)
        J = prep.jacobians; cols = int(J.shape[0]); assert cols == (192 if target == 191 else target)
        out, _ = readouts.readout(evaluator, Stub(unembed, cols), items, H, J, exits, cols)
        np.savez_compressed(HERE / 'readouts' / f'{arm}.npz', **{k: out[k] for k in KEYS})
        allrank[arm] = out['allrank']; del J, lens, prep, out; torch.cuda.empty_cache()
        print(f'{arm}: readout of {cols} virtual layers x {n} items in {time.time()-t:.0f}s', flush=True)
del cache, H, exits, weight, gain; torch.cuda.empty_cache()

# ---------------- analysis (plan section "Quantities", "Inference", "Descriptive") ----------------
hits = {a: (v >= 0) & (v < 10) for a, v in allrank.items()}
def band_stats(arm, cols):
    h = hits[arm][:, :, cols]
    o = np.array([h[i, own[i]].mean() for i in range(n)]); c = np.array([h[i, ctrl[i]].mean() for i in range(n)])
    return o, c, o - c
saved_analysis = json.load(open(HAND / '01_reports/confirmation_2026-09-09/results/ouro_confirmation_20260911_fixed160_native1/analysis/analysis.json'))
anchor = (band_stats('fit01', BAND[4])[2] - band_stats('raw', BAND[4])[2]).mean()
assert abs(anchor - saved_analysis['estimates'][0]) < 1e-12, (anchor, saved_analysis['estimates'][0])
res['checks']['primary_anchor_reproduced'] = float(anchor)
per = {}
for p in (1, 2, 3):
    for arm in ('raw', 'exit3', 'fit01', OWN[p]): per[(p, arm)] = band_stats(arm, BAND[p])
for arm in ('raw', 'exit3', 'fit01'): per[(4, arm)] = band_stats(arm, BAND[4])
fam_names = [f'own_exit_minus_final_target_pass{p}' for p in (1, 2, 3)] + [f'own_exit_minus_raw_pass{p}' for p in (1, 2, 3)]
fam = np.stack([per[(p, OWN[p])][2] - per[(p, 'exit3')][2] for p in (1, 2, 3)] + [per[(p, OWN[p])][2] - per[(p, 'raw')][2] for p in (1, 2, 3)], axis=1)
boots = analyze.resample(fam, groups, seed=SEED_FAMILY); est = fam.mean(0); sd = boots.std(0, ddof=1)
q = float(np.quantile(np.abs((boots - est) / np.where(sd > 0, sd, 1)).max(axis=1), 0.95))
family = {}
for j, name in enumerate(fam_names):
    lo, hi = float(est[j] - q * sd[j]), float(est[j] + q * sd[j])
    family[name] = {'estimate': float(est[j]), 'simultaneous_95_maxt': [lo, hi], 'excludes_zero': bool(lo > 0 or hi < 0), 'bootstrap_sd': float(sd[j]),
                    'descriptive_group_percentile_95': [float(x) for x in np.quantile(boots[:, j], [0.025, 0.975])]}
res['inference'] = {'family': family, 'maxt_quantile_95': q, 'draws': analyze.BOOTSTRAPS, 'groups': len(set(groups)), 'method': 'centered bootstrap max-t over 6 contrasts, whole dependency groups resampled, seed ' + str(SEED_FAMILY)}
def desc(values):
    values = np.asarray(values, float); b = analyze.resample(values, groups, seed=SEED_FAMILY)
    return {'estimate': float(values.mean()), 'descriptive_group_percentile_95': [float(x) for x in np.quantile(b, [0.025, 0.975])]}
res['descriptive'] = {'components': {}, 'own_exit_minus_fit01': {}, 'pass4_final_target_bank_on_confirmation': {}, 'per_layer_excess_curves': {}}
for p in (1, 2, 3, 4):
    arms = ['raw', 'exit3', 'fit01'] + ([OWN[p]] if p < 4 else [])
    res['descriptive']['components'][f'pass{p}'] = {arm: {'intended': desc(per[(p, arm)][0]), 'control79': desc(per[(p, arm)][1]), 'excess': desc(per[(p, arm)][2])} for arm in arms}
    if p < 4: res['descriptive']['own_exit_minus_fit01'][f'pass{p}'] = desc(per[(p, OWN[p])][2] - per[(p, 'fit01')][2])
res['descriptive']['pass4_final_target_bank_on_confirmation'] = {'exit3_minus_raw': desc(per[(4, 'exit3')][2] - per[(4, 'raw')][2]), 'exit3_minus_fit01': desc(per[(4, 'exit3')][2] - per[(4, 'fit01')][2]),
                                                                 'fit01_minus_raw_ORIGINAL_PRIMARY': {'estimate': float(anchor), 'original_group_percentile_95': saved_analysis['family_percentile_95_intervals'][0]}}
def layer_curve(arm):
    hh = hits[arm]; V = hh.shape[2]
    o = np.stack([hh[i, own[i], :] for i in range(n)]); c = np.stack([hh[i, ctrl[i], :].mean(0) for i in range(n)])
    return {'virtual': list(range(V)), 'excess': (o - c).mean(0).tolist(), 'intended': o.mean(0).tolist(), 'control79': c.mean(0).tolist()}
for arm in allrank: res['descriptive']['per_layer_excess_curves'][arm] = layer_curve(arm)
res['environment']['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()); res['environment']['seconds'] = round(time.time() - t0)
json.dump(res, open(HERE / 'RESULTS.json', 'w'), indent=1)
with open(HERE / 'PER_ITEM.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['item', 'group', 'pass', 'own_exit_excess', 'final_target_exit3_excess', 'raw_excess', 'fit01_excess'])
    for p in (1, 2, 3):
        for i in range(n): w.writerow([rows[i]['name'], groups[i], p, per[(p, OWN[p])][2][i], per[(p, 'exit3')][2][i], per[(p, 'raw')][2][i], per[(p, 'fit01')][2][i]])
# report
L = ['# Own-exit versus final-target J-Lens on the confirmation population', '', f'Executed {res["environment"]["finished_utc"]} per `ANALYSIS_PLAN.md` (sha256 `{PLAN_SHA[:16]}…`), frozen before computation. Initial-study banks (12-article calibration prefix, dim_batch 32) applied with the frozen scorer to the accepted confirmation states; raw and fit01 readouts reused from the accepted payload. Percentage points of excess hit@10 (79 controls), physical layers 26–37.', '',
     '## Checks', '', f'- Raw readouts reproduced bit for bit from these states with the fetched unembedding tensors: {res["checks"]["raw_reproduced_bit_for_bit"]}', f'- fit01 check: {res["checks"]["fit01_reproduced_bit_for_bit"]}', f'- Primary anchor (fit01 − raw, pass 4) recomputed: {anchor:+.6f} (saved {saved_analysis["estimates"][0]:+.6f})',
     f'- exit3 merged from shards vs pod merge: identical files {merge_rec["exit3_equivalence_with_pod_merge"]["byte_identical_files"]}, max |diff| {merge_rec["exit3_equivalence_with_pod_merge"]["max_abs_difference"]}, fraction equal {merge_rec["exit3_equivalence_with_pod_merge"]["fraction_entries_exactly_equal"]:.6f}', '',
     f'## Inference: the six-contrast family (simultaneous 95% max-t, whole-group bootstrap, seed {SEED_FAMILY}, quantile {q:.4f})', '', '| Contrast | Estimate (pts) | Simultaneous 95% | Excludes zero |', '|---|---:|---|---|']
for name, v in family.items(): L.append(f"| {name.replace('_', ' ')} | {100*v['estimate']:+.2f} | [{100*v['simultaneous_95_maxt'][0]:+.2f}, {100*v['simultaneous_95_maxt'][1]:+.2f}] | {'yes' if v['excludes_zero'] else 'no'} |")
L += ['', '## Components by pass (descriptive; group-percentile 95% intervals in RESULTS.json)', '', '| Pass | Arm | Intended | Control (79) | Excess |', '|---:|---|---:|---:|---:|']
for p in (1, 2, 3, 4):
    for arm, v in res['descriptive']['components'][f'pass{p}'].items():
        L.append(f"| {p} | {arm} | {100*v['intended']['estimate']:.2f}% | {100*v['control79']['estimate']:.2f}% | {100*v['excess']['estimate']:+.2f} |")
L += ['', '## Descriptive contrasts', '']
for p in (1, 2, 3):
    v = res['descriptive']['own_exit_minus_fit01'][f'pass{p}']; L.append(f"- own exit − fit01, pass {p}: {100*v['estimate']:+.2f} [{100*v['descriptive_group_percentile_95'][0]:+.2f}, {100*v['descriptive_group_percentile_95'][1]:+.2f}] (different calibration; does not isolate the target)")
for k, v in res['descriptive']['pass4_final_target_bank_on_confirmation'].items():
    iv = v.get('descriptive_group_percentile_95') or v.get('original_group_percentile_95'); L.append(f"- pass 4, {k.replace('_', ' ')}: {100*v['estimate']:+.2f} [{100*iv[0]:+.2f}, {100*iv[1]:+.2f}]")
L += ['', 'Labeling per plan: post-confirmation analysis on the already-used confirmation population; estimator family differs from the confirmation fit01 family; own-exit − exit3 isolates the target within the initial-study family; nothing here alters the primary endpoint.']
(HERE / 'REPORT.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L[10:18])); print('done in', res['environment']['seconds'], 's')
