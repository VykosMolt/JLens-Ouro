"""Frozen item-weighted hit@10 measurement; no model fitting or selection."""
from __future__ import annotations
import numpy as np

# Reports number physical layers 1..48; source tensors use 0..47.
PRIMARY_VIRTUAL = list(range(3 * 48 + 25, 3 * 48 + 37))
ARMS = ('fit01', 'fit02', 'penultimate', 'sampled_sum', 'diagonal')
SUPPORT = {arm: list(range(190 if arm == 'penultimate' else 192 if arm in ('fit01', 'fit02') else 191)) for arm in ARMS}
SUPPORT['raw'] = list(range(192))

def score(allrank, rows, names, virtual_indices):
    """Average control names per eligible label, then labels per item."""
    n, v = len(rows), len(virtual_indices)
    if allrank.shape != (n, 128, v) or allrank.dtype != np.int32:
        raise ValueError('rank bank has wrong geometry or dtype')
    own = np.zeros((n, v), np.float64)
    control = np.zeros_like(own)
    eligible = np.zeros(n, np.bool_)
    for i, row in enumerate(rows):
        count = len(names[row['task']])
        if np.any(allrank[i, :count] < 0) or np.any(allrank[i, :count] >= 49152) or np.any(allrank[i, count:] != -1):
            raise ValueError('name rank support/padding invalid')
        slots = [s for s, yes in enumerate(row['eligible']) if yes]
        eligible[i] = bool(slots)
        for s in slots:
            j = row['own_index'][s]
            controls = row['control_indices'][s]
            expected = [k for k in range(count) if k not in row['own_index']]
            if not 0 <= j < count or controls != expected or not controls:
                raise ValueError('multihop own/control weighting changed')
            own[i] += (allrank[i, j] < 10) / len(slots)
            control[i] += (allrank[i, controls] < 10).mean(axis=0) / len(slots)
    return {'eligible': eligible, 'own': own, 'control': control, 'excess': own - control}

def fixed_mean(values, virtual_indices, region):
    return values[:, [virtual_indices.index(v) for v in region]].mean(axis=1)

def paired(left, right, left_indices, right_indices, region):
    return fixed_mean(left['excess'], left_indices, region) - fixed_mean(right['excess'], right_indices, region)

def secondary_specs():
    """20 fixed contrasts; simultaneous bootstrap max-t interval family."""
    out = []
    for loop in range(3):
        region = list(range(loop * 48, (loop + 1) * 48))
        for metric in ('fixed_mean', 'any_layer'):
            out.append({'id': f'early_loop{loop+1}_{metric}', 'left': 'fit01', 'right': 'raw', 'region': region, 'metric': metric})
    out.extend([
        {'id': 'fit01_final_third', 'left': 'fit01', 'right': 'raw', 'region': list(range(176, 192)), 'metric': 'fixed_mean'},
        {'id': 'fit01_layer32', 'left': 'fit01', 'right': 'raw', 'region': [175], 'metric': 'fixed_mean'},
        {'id': 'fit02_local', 'left': 'fit02', 'right': 'raw', 'region': PRIMARY_VIRTUAL, 'metric': 'fixed_mean'},
        {'id': 'fit01_minus_fit02_local', 'left': 'fit01', 'right': 'fit02', 'region': PRIMARY_VIRTUAL, 'metric': 'fixed_mean'},
    ])
    for arm in ('penultimate', 'sampled_sum', 'diagonal'):
        for label, region in [('local', PRIMARY_VIRTUAL), ('final_third', list(range(176, 190 if arm == 'penultimate' else 191)))]:
            out.append({'id': f'{arm}_{label}', 'left': arm, 'right': 'raw', 'region': region, 'metric': 'fixed_mean'})
    for left, right in [('fit01', 'penultimate'), ('sampled_sum', 'diagonal')]:
        for label, region in [('local', PRIMARY_VIRTUAL), ('final_third', list(range(176, 190 if right == 'penultimate' else 191)))]:
            out.append({'id': f'{left}_minus_{right}_{label}', 'left': left, 'right': right, 'region': region, 'metric': 'fixed_mean'})
    assert len(out) == 20
    return out

def any_layer(allrank, rows, virtual_indices, region):
    columns = [virtual_indices.index(v) for v in region]
    result = np.zeros(len(rows), np.float64)
    for i, row in enumerate(rows):
        slots = [s for s, yes in enumerate(row['eligible']) if yes]
        for s in slots:
            ranks = allrank[i][:, columns]
            own = ranks[row['own_index'][s]]
            controls = ranks[row['control_indices'][s]]
            result[i] += (np.any((own >= 0) & (own < 10)) - np.any((controls >= 0) & (controls < 10), axis=1).mean()) / len(slots)
    return result
