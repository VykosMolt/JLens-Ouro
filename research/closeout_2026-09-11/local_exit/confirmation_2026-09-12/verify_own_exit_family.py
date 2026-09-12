#!/usr/bin/env python3
"""Independent NumPy-only check of the six-contrast own-exit family (Section 7, Table 6) from the stored rank
arrays. Reimplements the excess score, the whole-group bootstrap and the max-t interval without importing the
project's scoring or analysis code, and compares with RESULTS.json to 1e-12.
    python3 verify_own_exit_family.py            # from this directory, supplement or repository layout
Requires only numpy."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
for cand in (HERE.parents[1] / 'data' / 'accepted_payload', HERE.parents[1] / 'handoff' / 'JLens_Ouro_Writer_Handoff_v2' / '05_data' / 'accepted_payload'):
    if (cand / 'population.json').exists(): PAY = cand; break
else: sys.exit('accepted payload not found')
res = json.load(open(HERE / 'RESULTS.json')); seed = res['seed_family']; draws = res['inference']['draws']
pop = json.load(open(PAY / 'population.json')); rows = pop['rows']; n = len(rows)
own = np.array([r['own_index'][0] for r in rows]); ctrl = [np.array(r['control_indices'][0]) for r in rows]; groups = [r['dependency_group_id'] for r in rows]
assert all(len(c) == 79 for c in ctrl)
def allrank(arm):
    p = HERE / 'readouts' / f'{arm}.npz' if arm != 'raw' else PAY / 'readouts' / 'raw.npz'
    return np.load(p, allow_pickle=False)['allrank']
def excess(arm, cols):
    h = (lambda a: (a >= 0) & (a < 10))(allrank(arm))[:, :, cols]
    return np.array([h[i, own[i]].mean() - h[i, ctrl[i]].mean() for i in range(n)])
band = {p: list(range(48 * (p - 1) + 25, 48 * (p - 1) + 37)) for p in (1, 2, 3)}
OWN = {1: 'exit0', 2: 'exit1', 3: 'exit2'}
cols = [excess(OWN[p], band[p]) - excess('exit3', band[p]) for p in (1, 2, 3)] + [excess(OWN[p], band[p]) - excess('raw', band[p]) for p in (1, 2, 3)]
names = [f'own_exit_minus_final_target_pass{p}' for p in (1, 2, 3)] + [f'own_exit_minus_raw_pass{p}' for p in (1, 2, 3)]
V = np.stack(cols, axis=1).astype(np.float64)
# whole-dependency-group bootstrap, groups in order of first appearance, ratio-of-sums estimator, blocks of 500 draws
labels = list(dict.fromkeys(groups)); idx = [np.flatnonzero(np.asarray(groups) == g) for g in labels]
sums = np.stack([V[i].sum(0) for i in idx]); sizes = np.array([len(i) for i in idx])
rng = np.random.default_rng(seed); boots = np.empty((draws, V.shape[1]))
for start in range(0, draws, 500):
    d = rng.integers(len(labels), size=(min(500, draws - start), len(labels)))
    boots[start:start + len(d)] = sums[d].sum(1) / sizes[d].sum(1)[:, None]
est = V.mean(0); sd = boots.std(0, ddof=1); q = float(np.quantile(np.abs((boots - est) / sd).max(1), 0.95))
ok = abs(q - res['inference']['maxt_quantile_95']) < 1e-12
print(f'max-t quantile: {q:.12f} (saved {res["inference"]["maxt_quantile_95"]:.12f})')
for j, name in enumerate(names):
    saved = res['inference']['family'][name]; lo, hi = est[j] - q * sd[j], est[j] + q * sd[j]
    same = abs(est[j] - saved['estimate']) < 1e-12 and abs(lo - saved['simultaneous_95_maxt'][0]) < 1e-12 and abs(hi - saved['simultaneous_95_maxt'][1]) < 1e-12
    ok &= same
    print(f"{name:<36} {100*est[j]:+7.2f} [{100*lo:+7.2f}, {100*hi:+7.2f}]  {'matches RESULTS.json' if same else 'MISMATCH'}")
print('six-contrast family reproduced exactly' if ok else 'FAILED'); sys.exit(0 if ok else 1)
