#!/usr/bin/env python3
"""Prespecified paired family resampling of a single frozen primary endpoint."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation'))
from measurement import ARMS, SUPPORT, PRIMARY_VIRTUAL, score, fixed_mean, paired, secondary_specs, any_layer

BOOTSTRAPS = 20000
SEED = 2026090901

def resample(values, groups, seed=SEED):
    """Resample complete groups; ratio preserves the equal-item estimand."""
    values = np.asarray(values, np.float64)
    if values.ndim == 1:
        values = values[:, None]
    labels = list(dict.fromkeys(groups))
    indices = [np.flatnonzero(np.asarray(groups) == label) for label in labels]
    sums = np.stack([values[idx].sum(axis=0) for idx in indices])
    sizes = np.asarray([len(idx) for idx in indices])
    rng = np.random.default_rng(seed)
    result = np.empty((BOOTSTRAPS, values.shape[1]), np.float64)
    for start in range(0, BOOTSTRAPS, 500):
        draw = rng.integers(len(labels), size=(min(500, BOOTSTRAPS - start), len(labels)))
        result[start:start+len(draw)] = sums[draw].sum(axis=1) / sizes[draw].sum(axis=1)[:, None]
    return result

def interval(samples):
    return np.quantile(samples, [0.025, 0.975], axis=0).T.tolist()

def analyze(root):
    population = json.loads((root / 'population.json').read_text())
    rows, names = population['rows'], population['names']
    arrays, scores = {}, {}
    for arm in ('raw', *ARMS):
        with np.load(root / f'readouts/{arm}.npz', allow_pickle=False) as a:
            arrays[arm] = a['allrank'].copy()
        scores[arm] = score(arrays[arm], rows, names, SUPPORT[arm])
    mask = scores['raw']['eligible']
    assert all(np.array_equal(mask, scores[arm]['eligible']) for arm in ARMS)
    groups = [r['dependency_group_id'] for r, yes in zip(rows, mask) if yes]
    components = np.stack([fixed_mean(scores[arm][field], SUPPORT[arm], PRIMARY_VIRTUAL)[mask]
                           for arm, field in [('fit01','own'),('fit01','control'),('raw','own'),('raw','control')]], axis=1)
    primary = paired(scores['fit01'], scores['raw'], SUPPORT['fit01'], SUPPORT['raw'], PRIMARY_VIRTUAL)[mask]
    allvalues = np.column_stack([primary, components, components[:,0]-components[:,2], components[:,1]-components[:,3]])
    boots = resample(allvalues, groups)
    itemboots = resample(primary, [str(i) for i in range(len(primary))], seed=SEED+1)
    unique = list(dict.fromkeys(groups))
    heterogeneity = []
    group_array = np.asarray(groups)
    for group in unique:
        chosen = group_array == group
        heterogeneity.append({'group': group, 'items': int(chosen.sum()), 'primary_mean': float(primary[chosen].mean()),
                              'without_group': float(primary[~chosen].mean()),
                              'intended_difference': float((components[:,0]-components[:,2])[chosen].mean()),
                              'control_difference': float((components[:,1]-components[:,3])[chosen].mean())})
    secondary = []
    secvalues = []
    for spec in secondary_specs():
        left, right, region = spec['left'], spec['right'], spec['region']
        if spec['metric'] == 'fixed_mean':
            value = paired(scores[left], scores[right], SUPPORT[left], SUPPORT[right], region)
        else:
            value = any_layer(arrays[left], rows, SUPPORT[left], region) - any_layer(arrays[right], rows, SUPPORT[right], region)
        secvalues.append(value[mask])
        secondary.append(spec)
    secvalues = np.stack(secvalues, axis=1)
    secboots = resample(secvalues, groups, seed=SEED+2)
    estimates, std = secvalues.mean(axis=0), secboots.std(axis=0, ddof=1)
    safe = np.where(std > 0, std, 1)
    max_t = np.abs((secboots-estimates) / safe).max(axis=1)
    q = np.quantile(max_t, 0.95)
    for j, spec in enumerate(secondary):
        spec.update(estimate=float(estimates[j]), simultaneous_95_interval=[float(estimates[j]-q*std[j]), float(estimates[j]+q*std[j])])
    curves = {arm: {field: scores[arm][field][mask].mean(axis=0).tolist() for field in ('own','control','excess')} for arm in ('raw',*ARMS)}
    output = {'schema': 'confirmation_analysis.v1', 'eligible_items': int(mask.sum()), 'groups': len(unique),
              'component_order': ['primary_excess_difference','fit01_intended','fit01_control','raw_intended','raw_control','intended_difference','control_difference'],
              'estimates': allvalues.mean(axis=0).tolist(), 'family_percentile_95_intervals': interval(boots),
              'primary_item_resampling_95_interval': interval(itemboots)[0], 'group_heterogeneity': heterogeneity,
              'leave_one_group_range': [min(x['without_group'] for x in heterogeneity), max(x['without_group'] for x in heterogeneity)],
              'secondary_family': secondary, 'secondary_max_t_95_quantile': float(q), 'descriptive_curves': curves,
              'bootstrap': {'replicates': BOOTSTRAPS, 'seed': SEED, 'resampling': 'whole dependency groups with replacement; sum divided by sampled item count', 'secondary_policy': '20 contrasts; centered bootstrap max-t, one shared 95% interval family'}}
    return output, {'primary': primary, 'components': components, 'secondary': secvalues, 'family_bootstrap': boots, 'secondary_bootstrap': secboots}

def main():
    p = argparse.ArgumentParser(); p.add_argument('--results', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); args.out.mkdir(exist_ok=False)
    result, arrays = analyze(args.results)
    (args.out/'analysis.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    np.savez_compressed(args.out/'item_statistics_and_resamples.npz', **arrays)

if __name__ == '__main__':
    main()
