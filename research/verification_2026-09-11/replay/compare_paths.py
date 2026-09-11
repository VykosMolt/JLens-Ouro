#!/usr/bin/env python3
"""Compare replayed scorer layouts with the saved outputs and propagate each through the frozen analysis.

  python compare_paths.py --run NAME=REPLAY_DIR [--run ...] --roots DIR --out FILE

Each propagation root (population.json, allrank-only readouts, analysis.json) can be rechecked by the
independent endpoint checker. The frozen analysis reuses its seeds, so interval changes are numerical only.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np

RESEARCH = Path('/home/moloch/jacobian-lens/research')
ROUND = RESEARCH / 'confirmation_2026-09-09'
BUNDLE = ROUND / 'corrections/native_check_v1/bundle'
RESULTS = ROUND / ('cloud_leases/attempt_06/handoff/accepted/ouro_confirmation_20260911_fixed160_native1/'
                   'ff7c0769b2b0dbe2e53b120daf7f0729434fe65e54c013288771f5052ffd4cef/results')
SAVED_ANALYSIS = ROUND / 'results/ouro_confirmation_20260911_fixed160_native1/analysis/analysis.json'
ARMS = ('raw', 'fit01', 'fit02', 'penultimate', 'sampled_sum', 'diagonal')
BAND = list(range(169, 181))
THRESHOLDS = (0.0, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2)
DELTAS = (1 / 32, 1 / 8)  # largest per-logit packing differences: 5090 development diagnostic, local single-row

sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location('frozen_analyze', BUNDLE / 'analysis/analyze.py')
analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze)

population = json.loads((RESULTS / 'population.json').read_text())
ROWS = population['rows']
N_NAMES = len(population['names']['multihop'])
if any(sum(r['eligible']) != 1 for r in ROWS):
    raise ValueError('comparison assumes one eligible label per item')
OWN = np.array([r['own_index'][r['eligible'].index(True)] for r in ROWS])
INTENDED = np.zeros((len(ROWS), 128), bool)
INTENDED[np.arange(len(ROWS)), OWN] = True
CONTROL = np.zeros_like(INTENDED)
CONTROL[:, :N_NAMES] = True
CONTROL[np.arange(len(ROWS)), OWN] = False
for i, r in enumerate(ROWS):
    if np.flatnonzero(CONTROL[i]).tolist() != r['control_indices'][r['eligible'].index(True)]:
        raise ValueError('control masks differ from the population')
KINDS = (('intended', INTENDED), ('control', CONTROL))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def regions(columns):
    return {'all': list(range(columns)), 'band': BAND, 'loops_1_3': list(range(144)), 'loop_4': list(range(144, columns))}


def hits(rank):
    return (rank >= 0) & (rank < 10)


def hit_changes(saved, path, columns):
    before, after = hits(saved), hits(path)
    out = {}
    for label, cols in regions(columns).items():
        b, a = before[:, :, cols], after[:, :, cols]
        out[label] = {kind: {'cells': int(mask.sum()) * len(cols), 'saved_hits': int((b & mask[:, :, None]).sum()),
                             'gained': int((a & ~b & mask[:, :, None]).sum()), 'lost': int((b & ~a & mask[:, :, None]).sum())}
                      for kind, mask in KINDS}
    return out


def top10_changes(saved, path):
    changed = np.array([[len(set(p) - set(s)) for p, s in zip(pi, si)] for pi, si in zip(path, saved)])
    return {'rows': int(changed.size), 'rows_with_membership_change': int((changed > 0).sum()),
            'ids_replaced': int(changed.sum()), 'rows_with_order_change': int((path != saved).any(-1).sum())}


def ties(arrays, columns):
    best, worst, rank = arrays['best'], arrays['worst'], arrays['allrank']
    tied = (best >= 0) & (best < 10) & (worst >= 10)
    valid = best >= 0
    out = {'frozen_rank_within_tie_bounds': bool(((rank[valid] >= best[valid]) & (rank[valid] <= worst[valid])).all())}
    for label, cols in (('all', list(range(columns))), ('band', BAND)):
        out[label] = {kind: {'tie_determined_cells': int((tied[:, :, cols] & mask[:, :, None]).sum()),
                             'counted_as_hit': int((tied[:, :, cols] & hits(rank)[:, :, cols] & mask[:, :, None]).sum())}
                      for kind, mask in KINDS}
    return out


def boundary_margin(arrays):
    """Best-alias logit minus the 11th logit for hits (>= 0), minus the 10th logit for misses (<= 0)."""
    hit = hits(arrays['allrank'])
    tenth, eleventh = arrays['boundary'][:, None, :, 0], arrays['boundary'][:, None, :, 1]
    return np.where(hit, arrays['value'] - eleventh, arrays['value'] - tenth), hit


def margins(arrays):
    margin, hit = boundary_margin(arrays)
    valid = arrays['allrank'] >= 0
    out = {'hit_margins_nonnegative': bool((margin[hit] >= 0).all()), 'miss_margins_nonpositive': bool((margin[valid & ~hit] <= 0).all())}
    for kind, mask in KINDS:
        cells = np.broadcast_to(mask[:, :, None], (len(mask), 128, len(BAND)))
        m, h = margin[:, :, BAND][cells], hit[:, :, BAND][cells]
        out[kind] = {'band_cells': int(m.size), 'band_hits': int(h.sum()),
                     'hits_within': {f'{t:g}': int((h & (m <= t)).sum()) for t in THRESHOLDS},
                     'misses_within': {f'{t:g}': int((~h & (m >= -t)).sum()) for t in THRESHOLDS},
                     'abs_margin_quantiles_0_1_5_50': [float(q) for q in np.quantile(np.abs(m), [0, 0.01, 0.05, 0.5])]}
    return out


def propagate(root, allranks):
    (root / 'readouts').mkdir(parents=True)
    shutil.copyfile(RESULTS / 'population.json', root / 'population.json')
    for arm in ARMS:
        np.savez_compressed(root / 'readouts' / f'{arm}.npz', allrank=allranks[arm])
    result, _ = analyze.analyze(root)
    write_json(root / 'analysis.json', result)
    return result


def endpoint(result, saved):
    order = result['component_order']
    estimate, base = dict(zip(order, result['estimates'])), dict(zip(order, saved['estimates']))
    family, base_family = dict(zip(order, result['family_percentile_95_intervals'])), dict(zip(order, saved['family_percentile_95_intervals']))
    secondary = []
    for a, b in zip(result['secondary_family'], saved['secondary_family']):
        if a['id'] != b['id']:
            raise ValueError('secondary order differs')
        excludes, saved_excludes = [(lo > 0 or hi < 0) for lo, hi in (a['simultaneous_95_interval'], b['simultaneous_95_interval'])]
        secondary.append({'id': a['id'], 'estimate': a['estimate'], 'change': a['estimate'] - b['estimate'],
                          'interval': a['simultaneous_95_interval'], 'excludes_zero': excludes, 'saved_excludes_zero': saved_excludes,
                          'sign_changed': bool(np.sign(a['estimate']) != np.sign(b['estimate']))})
    return {'estimates': estimate, 'changes': {k: estimate[k] - base[k] for k in order},
            'family_intervals': family, 'interval_endpoint_changes': {k: [family[k][j] - base_family[k][j] for j in (0, 1)] for k in order},
            'primary_item_resampling_interval': result['primary_item_resampling_95_interval'],
            'leave_one_group_range': result['leave_one_group_range'], 'secondary_max_t_quantile': result['secondary_max_t_95_quantile'],
            'secondary': secondary,
            'secondary_zero_exclusion_changed': [s['id'] for s in secondary if s['excludes_zero'] != s['saved_excludes_zero']],
            'secondary_sign_changed': [s['id'] for s in secondary if s['sign_changed']]}


def tie_rule_ranks(arrays, variant):
    """Alternative tie orders over identical logits; the primary variants are adversarial bounds, not estimates."""
    out = {}
    for arm in ARMS:
        a = arrays[arm]
        if variant == 'favor_every_name':
            out[arm] = a['best']
        elif variant == 'against_every_name':
            out[arm] = a['worst']
        elif arm in ('fit01', 'raw'):
            up = (variant == 'maximize_primary') == (arm == 'fit01')
            out[arm] = np.where(INTENDED[:, :, None], a['best'] if up else a['worst'], a['worst'] if up else a['best'])
        else:
            out[arm] = a['allrank']
    return out


def perturbation_ranks(arrays, delta, maximize):
    """Every logit moved by at most delta: flip each primary-arm cell within 2*delta of the top-10 boundary adversarially."""
    out = {arm: arrays[arm]['allrank'] for arm in ARMS}
    for arm in ('fit01', 'raw'):
        margin, hit = boundary_margin(arrays[arm])
        rank = arrays[arm]['allrank'].copy()
        can_gain = (rank >= 0) & ~hit & (margin >= -2 * delta)
        can_lose = hit & (margin <= 2 * delta)
        favor_intended = maximize == (arm == 'fit01')
        rank[np.where(INTENDED[:, :, None], can_gain if favor_intended else False, False if favor_intended else can_gain)] = 0
        rank[np.where(INTENDED[:, :, None], False if favor_intended else can_lose, can_lose if favor_intended else False)] = 10
        out[arm] = rank
    return out


def load(directory, arm, keys):
    with np.load(directory / f'{arm}.npz', allow_pickle=False) as data:
        return {key: data[key] for key in keys if key in data.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', action='append', required=True, help='NAME=REPLAY_DIR')
    parser.add_argument('--roots', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.roots.mkdir(parents=True, exist_ok=False)
    saved_analysis = json.loads(SAVED_ANALYSIS.read_text())
    saved = {arm: load(RESULTS / 'readouts', arm, ('allrank', 'top10_ids')) for arm in ARMS}
    report = {'schema': 'scorer_sensitivity.v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'script_sha256': sha256(__file__), 'saved_analysis': {'path': str(SAVED_ANALYSIS), 'sha256': sha256(SAVED_ANALYSIS)},
              'population_sha256': sha256(RESULTS / 'population.json'), 'band_virtual': BAND,
              'saved_endpoint': endpoint(saved_analysis, saved_analysis), 'runs': {}, 'summary': []}
    fp64 = {}
    for item in args.run:
        name, directory = item.split('=', 1)
        directory = Path(directory)
        run = json.loads((directory / 'run.json').read_text())
        entry = {'run_json_sha256': sha256(directory / 'run.json'), 'device': run['device'], 'runtime': run['runtime'],
                 'precision_matches_original': run['precision_matches_original'], 'items': run['items'],
                 'frozen_readout_vs_saved': {arm: {key: v['equal'] for key, v in d.get('frozen_readout_vs_saved', {}).items()}
                                             for arm, d in run['arms_detail'].items()},
                 'frozen_samples_vs_saved': {arm: all(s['virtual_indices'] and all(s[k]['equal'] for k in ('transported', 'logits', 'sorted_ids'))
                                                      for s in d.get('frozen_samples_vs_saved', [])) for arm, d in run['arms_detail'].items()},
                 'transport_vs_saved_samples': {arm: all(c['equal'] for c in d.get('transport_vs_saved_samples', []))
                                                for arm, d in run['arms_detail'].items()},
                 'col160_vs_saved_exit_logits': run['arms_detail'].get('raw', {}).get('col160_vs_saved_exit_logits'),
                 'layouts': {}}
        if run['items'] != len(ROWS) or sorted(run['arms']) != sorted(ARMS):
            raise ValueError(f'{name}: propagation needs all items and arms')
        for layout in run['layouts']:
            arrays = {arm: load(directory / layout, arm, ('allrank', 'best', 'worst', 'top10', 'value', 'boundary')) for arm in ARMS}
            per_arm = {}
            for arm in ARMS:
                columns = arrays[arm]['allrank'].shape[2]
                detail = run['arms_detail'][arm]['layouts'][layout]
                per_arm[arm] = {'logits_vs_executed': detail['logits_vs_executed'],
                                'allrank_cells_differing_from_saved': int((arrays[arm]['allrank'] != saved[arm]['allrank']).sum()),
                                'top10_vs_saved': top10_changes(saved[arm]['top10_ids'], arrays[arm]['top10']),
                                'hit_changes_vs_saved': hit_changes(saved[arm]['allrank'], arrays[arm]['allrank'], columns),
                                'ties': ties(arrays[arm], columns)}
                if 'value' in arrays[arm]:
                    per_arm[arm]['boundary_margins_band'] = margins(arrays[arm])
            result = propagate(args.roots / name / layout, {arm: arrays[arm]['allrank'] for arm in ARMS})
            layout_entry = {'per_arm': per_arm, 'endpoint': endpoint(result, saved_analysis)}
            if layout == 'executed':
                layout_entry['tie_rule_bounds'] = {}
                for variant in ('favor_every_name', 'against_every_name', 'maximize_primary', 'minimize_primary'):
                    tie_result = propagate(args.roots / name / f'ties_{variant}', tie_rule_ranks(arrays, variant))
                    layout_entry['tie_rule_bounds'][variant] = endpoint(tie_result, saved_analysis)
                layout_entry['perturbation_bounds'] = {}
                for delta in DELTAS:
                    for maximize in (True, False):
                        label = f"delta_{delta:g}_{'max' if maximize else 'min'}"
                        bound = propagate(args.roots / name / f'perturb_{label}', perturbation_ranks(arrays, delta, maximize))
                        layout_entry['perturbation_bounds'][label] = {k: endpoint(bound, saved_analysis)[k] for k in ('estimates', 'changes')}
            if layout == 'fp64':
                fp64[name] = {arm: arrays[arm]['allrank'] for arm in ARMS}
            entry['layouts'][layout] = layout_entry
            e = layout_entry['endpoint']
            band = {arm: per_arm[arm]['hit_changes_vs_saved']['band'] for arm in ('fit01', 'raw')}
            report['summary'].append({
                'run': name, 'layout': layout, 'primary': e['estimates']['primary_excess_difference'],
                'primary_change': e['changes']['primary_excess_difference'], 'primary_interval': e['family_intervals']['primary_excess_difference'],
                'intended_difference_change': e['changes']['intended_difference'], 'control_difference_change': e['changes']['control_difference'],
                'band_hit_changes': {arm: {kind: [band[arm][kind]['gained'], band[arm][kind]['lost']] for kind in ('intended', 'control')} for arm in band},
                'allrank_cells_differing': sum(per_arm[arm]['allrank_cells_differing_from_saved'] for arm in ARMS),
                'secondary_zero_exclusion_changed': e['secondary_zero_exclusion_changed'], 'secondary_sign_changed': e['secondary_sign_changed']})
            del arrays
        report['runs'][name] = entry
    if len(fp64) == 2:
        (a, ra), (b, rb) = fp64.items()
        report['fp64_cross_device'] = {'runs': [a, b], 'allrank_cells_differing': {arm: int((ra[arm] != rb[arm]).sum()) for arm in ARMS},
                                       'hit_cells_differing': {arm: int((hits(ra[arm]) != hits(rb[arm])).sum()) for arm in ARMS}}
    write_json(args.out, report)


if __name__ == '__main__':
    main()
