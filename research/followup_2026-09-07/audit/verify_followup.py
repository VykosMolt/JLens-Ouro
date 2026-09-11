"""Independent executable check of follow-up layer and uncertainty outputs.

No production analysis helpers are imported. Source ranks are rescored from
scratch. Bootstrap arrays use direct item/cluster indexing, rather than the
production weighted-matrix implementation. Only the audit report is written.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / 'results'
SOURCE = Path('/home/moloch/ouro_project/artifacts/jlens/retrieved/jlens-b300-20260905-0359/eval/n100_exit3')
OPS = {'addition', 'subtraction', 'multiplication', 'division', 'mod', 'squared'}
WORDS = {word: str(i) for i, word in enumerate('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split())}
WORDS['third'] = '3'
METHODS = ('jlens_exit3', 'logitlens')
N_BOOT = 20_000
CHECKS = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected, name, tolerance=3e-12):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    err = float(np.max(abs(actual - expected))) if actual.size else 0.0
    assert err < tolerance, (name, err)
    CHECKS.append((name, err))


def clean_slots(item):
    return [k for k, label in enumerate(item['intermediates'])
            if item['scorable'][k] and not item['leaked'][k]
            and (item['task'] == 'multihop' or label not in OPS)]


def independently_score(items, names, arrays, task):
    indices = [i for i, item in enumerate(items) if item['task'] == task and clean_slots(item)]
    result = {}
    for method in METHODS:
        rows = {key: [] for key in ('hit', 'control', 'excess', 'any_hit', 'any_control', 'any_excess')}
        for i in indices:
            item = items[i]
            rank = arrays[method + '_allrank'][i, :len(names[task]), :]
            hit_by_name = rank < 10
            any_by_name = np.max(hit_by_name.reshape(len(names[task]), 4, 48), axis=2)
            own_names = {j for j in item['own_index'] if j >= 0}
            own_slots = [item['own_index'][slot] for slot in clean_slots(item)]
            controls_by_slot = [[k for k, name in enumerate(names[task])
                                 if k not in own_names and (name in OPS) == (names[task][own] in OPS)]
                                for own in own_slots]
            own = np.mean(hit_by_name[own_slots], axis=0).reshape(4, 48)
            control = np.stack([np.mean(hit_by_name[cs], axis=0) for cs in controls_by_slot]).mean(axis=0).reshape(4, 48)
            any_own = any_by_name[own_slots].mean(axis=0)
            any_control = np.stack([any_by_name[cs].mean(axis=0) for cs in controls_by_slot]).mean(axis=0)
            for key, value in [('hit', own), ('control', control), ('excess', own-control),
                               ('any_hit', any_own), ('any_control', any_control), ('any_excess', any_own-any_control)]:
                rows[key].append(value)
        result[method] = {key: np.stack(value) for key, value in rows.items()}
    return np.asarray(indices), result


def independent_components(items, indices):
    concepts = []
    for i in indices:
        labels = [items[i]['intermediates'][slot].lower() for slot in clean_slots(items[i])]
        concepts.append({WORDS.get(label, str(int(label)) if label.isdigit() else label) for label in labels})
    # Graph search, independent of the production union-find implementation.
    adjacency = [{j for j, other in enumerate(concepts) if labels & other} for labels in concepts]
    unused, components = set(range(len(indices))), []
    while unused:
        frontier, component = {min(unused)}, set()
        while frontier:
            vertex = frontier.pop()
            if vertex in component:
                continue
            component.add(vertex)
            frontier.update(adjacency[vertex] - component)
        components.append(sorted(component))
        unused -= component
    return components


def direct_draws(values, seed):
    selected = np.random.default_rng(seed).integers(0, len(values), (N_BOOT, len(values)))
    draws = np.empty((N_BOOT, *values.shape[1:]))
    for start in range(0, N_BOOT, 200):
        draws[start:start+200] = values[selected[start:start+200]].mean(axis=1)
    return draws, selected


def direct_cluster_draws(values, clusters, seed):
    labels = sorted(set(map(int, clusters)))
    groups = [np.where(clusters == label)[0] for label in labels]
    group_sums = np.stack([values[group].sum(axis=0) for group in groups])
    sizes = np.array([len(group) for group in groups])
    selected = np.random.default_rng(seed).integers(0, len(groups), (N_BOOT, len(groups)))
    draws = np.empty((N_BOOT, *values.shape[1:]))
    for start in range(0, N_BOOT, 200):
        choice = selected[start:start+200]
        numerator = group_sums[choice].sum(axis=1)
        denominator = sizes[choice].sum(axis=1).reshape((-1,) + (1,)*(values.ndim-1))
        draws[start:start+200] = numerator / denominator
    # Materialize a few variable-length cluster resamples to test that the
    # ratio denominator retains equal item weight, including repeated groups.
    for b in (0, 1, 2, 19, 103):
        explicit_indices = np.concatenate([groups[group] for group in selected[b]])
        close(draws[b], values[explicit_indices].mean(axis=0), f'cluster_materialized_{seed}_{b}')
    return draws


def ci(draws):
    return np.quantile(draws, [0.025, 0.975], axis=0).transpose(*range(1, draws.ndim), 0)


def conditional_cluster_draws(values, keep, clusters):
    groups = [np.where(clusters == label)[0] for label in sorted(set(map(int,clusters)))]
    choice = np.random.default_rng(19).integers(0,len(groups),(N_BOOT,len(groups)))
    sums = np.stack([values[group[keep[group]]].sum(axis=0) for group in groups])
    sizes = np.array([keep[group].sum() for group in groups])
    denominator = sizes[choice].sum(axis=1)
    assert denominator.min() > 0
    output = np.empty((N_BOOT,*values.shape[1:]))
    for start in range(0,N_BOOT,200):
        total = sums[choice[start:start+200]].sum(axis=1)
        output[start:start+200] = total / denominator[start:start+200].reshape((-1,)+(1,)*(values.ndim-1))
    return output


def verify_correctness(computed):
    path = RESULTS/'correctness.json'
    if not path.exists():
        return ['Correctness outputs were not present for this verification run.']
    data = json.loads(path.read_text())
    independent = json.loads((HERE/'continuation_audit.json').read_text())
    counts = []
    for criterion,field in [('historical','historical_pass'),('strict_numeric','strict_numeric_pass'),('numeric_equivalence','numeric_equivalence_pass')]:
        label = np.array([row[field] for row in independent],bool)
        assert data['all_148_counts'][criterion] == {'passing':int(label.sum()),'failing':int((~label).sum())}
        counts.append((criterion,int(label.sum())))
        for task,(indices,scores,clusters) in computed.items():
            mask = label[indices]
            record = data['criteria'][criterion][task]
            assert record['n_passing'] == int(mask.sum())
            assert record['n_failing'] == int((~mask).sum())
            all_d = scores['jlens_exit3']['any_excess']-scores['logitlens']['any_excess']
            pair_draws, pair_cluster_draws, method_draws = {}, {}, {}
            for stratum,keep,seed in [('passing',mask,0),('failing',~mask,1)]:
                st = record['strata'][stratum]
                assert st['n_items'] == int(keep.sum())
                for method in METHODS:
                    for key,values in scores[method].items():
                        close(values[keep].mean(axis=0),st['methods'][method][key]['mean'],f'correctness_{criterion}_{task}_{stratum}_{method}_{key}_mean')
                    draws,_ = direct_draws(scores[method]['any_excess'][keep],seed)
                    close(ci(draws),st['methods'][method]['any_excess']['ci95'],f'correctness_{criterion}_{task}_{stratum}_{method}_ci')
                    method_draws[(stratum,method)] = draws
                for key in ('excess','any_excess'):
                    difference = scores['jlens_exit3'][key]-scores['logitlens'][key]
                    draws,_ = direct_draws(difference[keep],seed)
                    clustered = conditional_cluster_draws(difference,keep,clusters)
                    close(difference[keep].mean(axis=0),st[key+'_difference']['mean'],f'correctness_{criterion}_{task}_{stratum}_{key}_difference')
                    close(ci(draws),st[key+'_difference']['ci95'],f'correctness_{criterion}_{task}_{stratum}_{key}_ci')
                    close(ci(clustered),st[key+'_difference_cluster']['ci95'],f'correctness_{criterion}_{task}_{stratum}_{key}_cluster_ci')
                    if key == 'any_excess':
                        pair_draws[stratum],pair_cluster_draws[stratum] = draws,clustered
            interaction = record['interaction_passing_minus_failing']
            close(all_d[mask].mean(axis=0)-all_d[~mask].mean(axis=0),interaction['mean'],f'correctness_{criterion}_{task}_interaction_mean')
            close(ci(pair_draws['passing']-pair_draws['failing']),interaction['item_ci95'],f'correctness_{criterion}_{task}_interaction_item_ci')
            close(ci(pair_cluster_draws['passing']-pair_cluster_draws['failing']),interaction['cluster_ci95'],f'correctness_{criterion}_{task}_interaction_cluster_ci')
            for method in METHODS:
                values = scores[method]['any_excess']
                rr = record['readability_passing_minus_failing'][method]
                close(values[mask].mean(axis=0)-values[~mask].mean(axis=0),rr['mean'],f'correctness_{criterion}_{task}_{method}_readability_mean')
                close(ci(method_draws['passing',method]-method_draws['failing',method]),rr['ci95'],f'correctness_{criterion}_{task}_{method}_readability_ci')
    return [
        'Correctness verification also passes. Independent prior continuation annotations reproduce all three rules: historical 72/148, strict numerical 70/148, narrow numerical equivalence 71/148. Every stratum/method mean, all 2,304 stratum layerwise paired intervals (three criteria × two tasks × two strata × 192 cells), corresponding cluster intervals, any-layer effects, and passing-minus-failing interactions reproduce.',
        'The cluster interaction correctly resamples concepts jointly across outcome strata; it does not independently draw a shared concept once for passing items and again for failing items. Each stratum is normalized by its own resampled item count. No draw loses an entire stratum in the retained 20,000 replicates. These outcome comparisons remain observational and their intervals are pointwise; task/template composition can explain differences.',
    ]


def main():
    paths = [RESULTS / name for name in ('layerwise.csv', 'layerwise.json', 'item_scores.npz', 'depth_bands.csv')]
    if (RESULTS/'correctness.json').exists():
        paths.append(RESULTS/'correctness.json')
    starting_hashes = {str(path): sha(path) for path in paths}
    items = json.loads((SOURCE / 'items.json').read_text())
    names = json.loads((SOURCE / 'task_names.json').read_text())
    arrays = dict(np.load(SOURCE / 'arrays.npz', allow_pickle=False))
    saved = dict(np.load(RESULTS / 'item_scores.npz', allow_pickle=False))
    published = json.loads((RESULTS / 'layerwise.json').read_text())
    with (RESULTS / 'layerwise.csv').open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 384
    assert published['bootstrap']['layer_task_seeds'] == [0, 1]
    assert published['bootstrap']['historical_seed'] == 0
    assert published['bootstrap']['draws'] == N_BOOT
    # Fixed sample of cells is selected before inspecting their outcomes.
    chosen = sorted(np.random.default_rng(11092026).choice(384, size=10, replace=False).tolist())
    maxima, cluster_maxima = np.zeros(N_BOOT), np.zeros(N_BOOT)
    summaries, all_delta, selected_cell_records, computed = {}, {}, [], {}
    for seed, task in enumerate(('multihop', 'order-ops')):
        indices, scores = independently_score(items, names, arrays, task)
        close(indices, saved[task + '_indices'], task + '_indices', tolerance=0.5)
        n = len(indices)
        root_task = published['tasks'][task]
        assert n == root_task['n_items']
        assert sum(len(clean_slots(items[i])) for i in indices) == root_task['n_slots']
        components = independent_components(items, indices)
        clusters = saved[task + '_concept_clusters']
        assert len(components) == root_task['n_concept_clusters']
        for component in components:
            label = clusters[component[0]]
            assert set(np.where(clusters == label)[0]) == set(component)
        computed[task] = (indices,scores,clusters)
        for method, fields in scores.items():
            for key, values in fields.items():
                close(values, saved[task+'_'+method+'_'+key], task+'_'+method+'_'+key)
                close(values.mean(axis=0), root_task['methods'][method][key]['mean'], task+'_'+method+'_'+key+'_mean')
        delta = scores['jlens_exit3']['excess'] - scores['logitlens']['excess']
        all_delta[task] = delta
        draws, choices = direct_draws(delta, seed)
        cluster_draws = direct_cluster_draws(delta, clusters, seed)
        mean = delta.mean(axis=0)
        draw_ci, cluster_ci = ci(draws), ci(cluster_draws)
        close(mean, root_task['difference']['mean'], task+'_difference_mean')
        close(draw_ci, root_task['difference']['ci95'], task+'_all_192_pointwise_cis')
        close(cluster_ci, root_task['difference_cluster']['ci95'], task+'_all_192_cluster_cis')
        maxima = np.maximum(maxima, np.max(abs(draws-mean), axis=(1,2)))
        cluster_maxima = np.maximum(cluster_maxima, np.max(abs(cluster_draws-mean), axis=(1,2)))
        # Identity cell is truly zero for every item; both pointwise procedures
        # return [0,0]. A uniform simultaneous envelope can be conservative here.
        close(delta[:,3,47], np.zeros(n), task+'_identity_cell_zero')
        close(draw_ci[3,47], np.zeros(2), task+'_identity_pointwise_zero')
        close(cluster_ci[3,47], np.zeros(2), task+'_identity_cluster_zero')
        historical = scores['jlens_exit3']['any_excess'] - scores['logitlens']['any_excess']
        historical_draws, _ = direct_draws(historical, 0)
        close(historical.mean(axis=0), root_task['any_layer_difference']['mean'], task+'_any_layer_mean')
        close(ci(historical_draws), root_task['any_layer_difference']['ci95'], task+'_any_layer_ci')
        # Regression via least-squares design, rather than the production
        # centered-covariance expression. Depth spans a complete stack, 0..1.
        design = np.column_stack([np.linspace(0,1,48), np.ones(48)])
        slopes = np.linalg.lstsq(design, delta.reshape(n*4,48).T, rcond=None)[0][0].reshape(n,4)
        slope_draws, _ = direct_draws(slopes, seed)
        cluster_slope_draws = direct_cluster_draws(slopes, clusters, seed)
        close(slopes.mean(axis=0), root_task['depth_slope']['mean'], task+'_depth_slope_mean')
        close(ci(slope_draws), root_task['depth_slope']['ci95'], task+'_depth_slope_ci')
        close(ci(cluster_slope_draws), root_task['depth_slope_cluster']['ci95'], task+'_depth_slope_cluster_ci')
        for start, stop in ((0,16),(16,32),(32,48)):
            label = f'{start+1}-{stop}'
            band = delta[:,:,start:stop].mean(axis=2)
            record = root_task['depth_bands'][label]
            close(band.mean(axis=0), record['difference']['mean'], task+'_band_'+label+'_mean')
            close(ci(draws[:,:,start:stop].mean(axis=2)), record['difference']['ci95'], task+'_band_'+label+'_ci')
            close(ci(cluster_draws[:,:,start:stop].mean(axis=2)), record['difference_cluster']['ci95'], task+'_band_'+label+'_cluster_ci')
        for cell in chosen:
            if cell // 192 != seed:
                continue
            u, layer = divmod(cell % 192, 48)
            row = csv_rows[cell]
            assert row['task'] == task and int(row['loop']) == u+1 and int(row['physical_layer']) == layer+1
            for method in METHODS:
                for key in ('hit', 'control', 'excess'):
                    values = scores[method][key][:,u,layer]
                    endpoints = np.quantile(values[choices].mean(axis=1), [0.025,0.975])
                    close(endpoints, [float(row[method+'_'+key+'_ci_low']), float(row[method+'_'+key+'_ci_high'])],
                          f'selected_{cell}_{method}_{key}_ci')
            selected_cell_records.append((task,u+1,layer+1,float(mean[u,layer]),draw_ci[u,layer].tolist()))
        for flat, row in enumerate(csv_rows[seed*192:(seed+1)*192]):
            u, layer = divmod(flat,48)
            for method in METHODS:
                for key in ('hit','control','excess'):
                    close(float(row[method+'_'+key]), scores[method][key][:,u,layer].mean(), f'csv_{task}_{flat}_{method}_{key}')
            close(float(row['difference']), mean[u,layer], f'csv_{task}_{flat}_difference')
            close([float(row['ci_low']),float(row['ci_high'])], draw_ci[u,layer], f'csv_{task}_{flat}_ci')
            close([float(row['cluster_ci_low']),float(row['cluster_ci_high'])], cluster_ci[u,layer], f'csv_{task}_{flat}_cluster_ci')
        summaries[task] = {'n': n, 'clusters': len(components)}
    radius = float(np.quantile(maxima,.95))
    cluster_radius = float(np.quantile(cluster_maxima,.95))
    close(radius,published['bootstrap']['simultaneous_absolute_radius'],'simultaneous_radius')
    close(cluster_radius,published['bootstrap']['cluster_simultaneous_absolute_radius'],'cluster_simultaneous_radius')
    for task, delta in all_delta.items():
        mean = delta.mean(axis=0)
        for key, rr in (('difference',radius),('difference_cluster',cluster_radius)):
            close(np.stack([mean-rr,mean+rr],axis=2),published['tasks'][task][key]['simultaneous_ci95_384'],task+'_'+key+'_simultaneous_band')
    correctness_notes = verify_correctness(computed)
    for path in paths:
        assert sha(path) == starting_hashes[str(path)], f'Output changed during verification: {path}'
    lines = [
        '# Independent verification of follow-up layers — 7 September 2026', '',
        'PASS. No production scoring or bootstrap helper was imported. Source ranks were rescored independently; bootstrap means used direct resampled indexing rather than production matrix weights.', '',
        f'Checked {len(CHECKS):,} numerical comparisons; maximum absolute discrepancy {max(error for _,error in CHECKS):.3g}.', '',
        '| Check | Result |', '|---|---|',
        '| Every per-item score, both methods, all 192 locations and any-layer summaries | Pass |',
        '| All 384 paired item-bootstrap intervals | Pass |',
        '| All 384 concept-cluster intervals | Pass |',
        '| Sixty method/metric intervals at ten randomly selected cells | Pass |',
        '| Every depth-band mean and item/cluster interval | Pass |',
        '| Eight OLS slopes and item/cluster intervals | Pass |',
        '| Historical any-layer statistic and seed-0 uncertainty | Pass |',
        '| Concept-cluster membership: independent graph search versus union-find | Pass |',
        '| Cluster ratio weights versus materialized variable-length resamples | Pass |',
        '| Final endpoint zero per item and degenerate pointwise intervals | Pass |',
        '| Output files remained unchanged during verification | Pass |', '',
        f'The maximum centered absolute bootstrap error across both tasks and all four loops/48 layers gives radius **{radius:.15f}** for paired item resampling and **{cluster_radius:.15f}** for concept-cluster resampling. Independent task seeds 0 and 1 are recorded; their errors are joined by bootstrap replicate to calibrate the 384-cell family. Empirical coverage of the bootstrap errors by these radii is {np.mean(maxima<=radius):.6f} and {np.mean(cluster_maxima<=cluster_radius):.6f}.', '',
        'These are approximate bootstrap simultaneous bands, not exact finite-sample coverage guarantees. They are unstudentized: every cell receives the same absolute radius, which can be conservative for low-variance cells. The final identity cell has exactly zero paired effect and [0,0] pointwise intervals; its displayed simultaneous envelope is conservative, not evidence of uncertainty about identity. Pointwise method ribbons and fixed-block/slope intervals do not receive the 384-cell simultaneous correction.', '',
        'Concept clusters preserve equal item weight: draw the original number of clusters with replacement, include every item whenever its cluster is drawn, and divide by the number of sampled items. This differs appropriately from taking an unweighted mean of cluster means. Membership was independently checked before aligning cluster order to the saved labels for exact seeded replay. There are 64 multihop and 14 arithmetic clusters. Clustering is a sensitivity assumption; it does not establish independence between templates.', '',
        'The main statistic performs oracle any-layer scoring with the maximum before control averaging. The new fixed-layer statistic performs no layer selection. Both methods use the same item draws, and subgroup/fit uncertainty is not smuggled into these intervals. All curves still condition on one fitted J-Lens.', '',
        'Ten verification cells were chosen with seed 11092026 before outcomes were read:', '',
        '| Task | Loop | Physical layer | Paired difference | Pointwise 95% interval |', '|---|---:|---:|---:|---|',
    ]
    for task,u,l,value,endpoints in selected_cell_records:
        lines.append(f'| {task} | {u} | {l} | {value:+.6f} | [{endpoints[0]:+.6f}, {endpoints[1]:+.6f}] |')
    lines += ['', 'Verified output SHA256:', '']
    lines += [f'- `{Path(path).name}`: `{value}`' for path,value in starting_hashes.items()]
    for paragraph in correctness_notes:
        lines.extend(['',paragraph])
    (HERE/'FOLLOWUP_VERIFICATION.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:26]))


if __name__ == '__main__':
    main()
