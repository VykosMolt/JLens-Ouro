#!/usr/bin/env python3
"""Export machine-readable plot arrays from retained results (no new scoring rules):
 1. confirmation per-layer own/control/excess curves for six arms + fit01-raw paired excess per layer with descriptive group-bootstrap 95% bands
 2. refit five-fit discovery curves (per fit, mean, crossed fit/item pointwise 95%) for multihop, from paired_items.npz and main_cells.csv
 3. secondary-contrast forest data from analysis.json; 4. same-band pass table from reviewer RESULTS.json"""
import json, csv, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
C = Path('/home/moloch/ouro_project/jacobian-lens/research/confirmation_2026-09-09')
A = C / 'cloud_leases/attempt_06/handoff/accepted/ouro_confirmation_20260911_fixed160_native1/ff7c0769b2b0dbe2e53b120daf7f0729434fe65e54c013288771f5052ffd4cef/results'
R = Path('/home/moloch/ouro_project/jacobian-lens/research/refit_round_2026-09-07/analysis')
sys.dont_write_bytecode = True; sys.path[:0] = [str(C / 'evaluation'), str(C / 'analysis')]
from measurement import ARMS, SUPPORT, score
import analyze
pop = json.load(open(A / 'population.json')); rows = pop['rows']; groups = [r['dependency_group_id'] for r in rows]
saved = json.load(open(C / 'results/ouro_confirmation_20260911_fixed160_native1/analysis/analysis.json'))
scores = {}
for arm in ('raw', *ARMS):
    with np.load(A / f'readouts/{arm}.npz', allow_pickle=False) as a: scores[arm] = score(a['allrank'], rows, pop['names'], SUPPORT[arm])
# 1. curves
with open(HERE / 'confirmation_layer_curves.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['virtual', 'pass', 'physical', 'in_primary_band'] + [f'{arm}_{k}' for arm in ('raw', *ARMS) for k in ('own', 'control', 'excess')])
    for v in range(192):
        w.writerow([v, v // 48 + 1, v % 48 + 1, int(169 <= v <= 180)] + [(f"{scores[arm][k][:, v].mean():.6f}" if v < len(SUPPORT[arm]) else '') for arm in ('raw', *ARMS) for k in ('own', 'control', 'excess')])
    for arm in ('raw', *ARMS):
        for k in ('own', 'control', 'excess'):
            assert np.allclose(scores[arm][k].mean(axis=0), saved['descriptive_curves'][arm][k])
diff = scores['fit01']['excess'] - scores['raw']['excess']  # [160,192]
b = analyze.resample(diff, groups, seed=2026091106)  # [20000,192] descriptive pointwise
lo, hi = np.quantile(b, 0.025, axis=0), np.quantile(b, 0.975, axis=0)
diff2 = scores['fit02']['excess'] - scores['raw']['excess']; b2 = analyze.resample(diff2, groups, seed=2026091106)
with open(HERE / 'confirmation_paired_excess_by_layer.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['virtual', 'pass', 'physical', 'in_primary_band', 'fit01_minus_raw', 'group_boot_2.5_descriptive', 'group_boot_97.5_descriptive', 'fit02_minus_raw', 'fit02_group_boot_2.5', 'fit02_group_boot_97.5', 'fit01_intended_minus_raw_intended', 'fit01_control_minus_raw_control'])
    for v in range(192):
        w.writerow([v, v // 48 + 1, v % 48 + 1, int(169 <= v <= 180), f'{diff[:, v].mean():.6f}', f'{lo[v]:.6f}', f'{hi[v]:.6f}', f'{diff2[:, v].mean():.6f}', f'{np.quantile(b2[:, v], 0.025):.6f}', f'{np.quantile(b2[:, v], 0.975):.6f}',
                    f'{(scores["fit01"]["own"][:, v] - scores["raw"]["own"][:, v]).mean():.6f}', f'{(scores["fit01"]["control"][:, v] - scores["raw"]["control"][:, v]).mean():.6f}'])
# 2. refit five-fit curves
rep = json.load(open(R / 'main_run01/report.json')); idx = rep['populations']['multihop']['item_indices']; assert len(idx) == 90
z = np.load(R / 'main_run01/paired_items.npz'); d = z['main_paired_delta_layer']  # (5,148,192)
assert z['main_raw_eligible'][idx].all()
cells = {}
with open(R / 'main_run01/main_cells.csv') as f:
    for r in csv.DictReader(f):
        if r['task'] == 'multihop': cells[int(r['virtual_index'])] = r
with open(HERE / 'discovery_five_fit_curves.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['virtual', 'pass', 'physical', 'known_identity', 'fit1', 'fit2', 'fit3', 'fit4', 'fit5', 'fit_mean', 'fit_sd', 'crossed_fit_item_pointwise_low', 'crossed_fit_item_pointwise_high', 'crossed_fit_item_simultaneous_low', 'crossed_fit_item_simultaneous_high', 'in_selected_band_26_37', 'in_final_third_33_48'])
    for v in range(192):
        per = d[:, idx, v].mean(axis=1); c = cells[v]
        assert abs(per.mean() - float(c['mean'])) < 1e-9, (v, per.mean(), c['mean'])
        w.writerow([v, v // 48 + 1, v % 48 + 1, c['known_identity']] + [f'{x:.6f}' for x in per] + [f'{per.mean():.6f}', c['fit_sd'], c['crossed_fit_item_pointwise_low'], c['crossed_fit_item_pointwise_high'], c['crossed_fit_item_simultaneous_low'], c['crossed_fit_item_simultaneous_high'], int(v // 48 == 3 and 25 <= v % 48 <= 36), int(v // 48 == 3 and v % 48 >= 32)])
# 3. forest data
with open(HERE / 'confirmation_secondary_contrasts.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['id', 'left', 'right', 'metric', 'region_first', 'region_last', 'estimate', 'simultaneous_low', 'simultaneous_high'])
    for s in saved['secondary_family']: w.writerow([s['id'], s['left'], s['right'], s['metric'], s['region'][0], s['region'][-1], f"{s['estimate']:.6f}", f"{s['simultaneous_95_interval'][0]:.6f}", f"{s['simultaneous_95_interval'][1]:.6f}"])
with open(HERE / 'confirmation_primary_components.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['component', 'estimate', 'group_percentile_2.5', 'group_percentile_97.5'])
    for k, e, iv in zip(saved['component_order'], saved['estimates'], saved['family_percentile_95_intervals']): w.writerow([k, f'{e:.6f}', f'{iv[0]:.6f}', f'{iv[1]:.6f}'])
# 4. same-band + within-domain from reviewer results
rv = json.load(open(HERE.parents[1] / 'reviewer_checks/RESULTS.json'))
with open(HERE / 'same_band_by_pass.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['pass', 'fit01_intended', 'fit01_control', 'raw_intended', 'raw_control', 'paired_excess_diff', 'interval_low', 'interval_high', 'interval_kind'])
    for p in rv['C_same_band']['passes']:
        iv = p['paired_excess_diff'].get('simultaneous_95_maxt') or p['paired_excess_diff'].get('ORIGINAL_PRIMARY_group_percentile_95')
        w.writerow([p['pass'], f"{p['fit01_intended']['estimate']:.6f}", f"{p['fit01_control79']['estimate']:.6f}", f"{p['raw_intended']['estimate']:.6f}", f"{p['raw_control79']['estimate']:.6f}", f"{p['paired_excess_diff']['estimate']:.6f}", f'{iv[0]:.6f}', f'{iv[1]:.6f}', 'N1 simultaneous max-t (post-confirmation)' if p['pass'] < 4 else 'original primary group percentile'])
print('figure data written:', sorted(p.name for p in HERE.glob('*.csv')))
