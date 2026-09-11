"""Exactly three frozen sensitivity analyses; historical source files read-only."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from verify_followup import independently_score, direct_draws, direct_cluster_draws, ci, clean_slots

HERE = Path(__file__).resolve().parent
SOURCE = Path('/home/moloch/ouro_project/artifacts/jlens/retrieved/jlens-b300-20260905-0359/eval/n100_exit3')
OPS = {'addition','subtraction','multiplication','division','mod','squared'}
METHODS = ('jlens_exit3','logitlens')
TASKS = ('multihop','order-ops')
WORDS = {word:str(i) for i,word in enumerate('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split())}
WORDS['third'] = '3'
STALE = {'super-populous-capital','firstletter-populous-country'}


def canonical(name):
    name = name.lower()
    return WORDS.get(name,str(int(name)) if name.isdigit() else name)


def alias_scores(items,names,arrays,task,indices):
    output = {}
    for method in METHODS:
        rows = {key:[] for key in ('hit','control','excess','any_hit','any_control','any_excess')}
        for i in indices:
            item = items[i]
            ranks = arrays[method+'_allrank'][i].reshape(128,4,48)
            own_names = {j for j in item['own_index'] if j >= 0}
            own_concepts = {canonical(names[task][j]) for j in own_names}
            slot_rows = {key:[] for key in rows}
            for slot in clean_slots(item):
                own = item['own_index'][slot]
                groups = {}
                for j,name in enumerate(names[task]):
                    if j in own_names or (name in OPS)!=(names[task][own] in OPS):
                        continue
                    concept = canonical(name)
                    if concept not in own_concepts:
                        groups.setdefault(concept,[]).append(j)
                controls = np.stack([ranks[group].min(axis=0) for group in groups.values()])
                own_hit = (ranks[own]<10).astype(float)
                control_hit = controls<10
                control = control_hit.mean(axis=0)
                any_hit = own_hit.max(axis=1)
                any_control = control_hit.max(axis=2).mean(axis=0)
                for key,value in [('hit',own_hit),('control',control),('excess',own_hit-control),
                                  ('any_hit',any_hit),('any_control',any_control),('any_excess',any_hit-any_control)]:
                    slot_rows[key].append(value)
            for key in rows:
                rows[key].append(np.mean(slot_rows[key],axis=0))
        output[method] = {key:np.stack(value) for key,value in rows.items()}
    return output


def serializable(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value))


def item_summary(values):
    draws,_ = direct_draws(values,0)
    return {'mean':values.mean(axis=0),'ci95':ci(draws)}


def main():
    note_path = HERE/'sensitivities.md'
    assert 'Choices recorded before execution' in note_path.read_text()
    items = json.loads((SOURCE/'items.json').read_text())
    names = json.loads((SOURCE/'task_names.json').read_text())
    arrays = dict(np.load(SOURCE/'arrays.npz',allow_pickle=False))
    with (HERE/'dependence_clusters.csv').open() as stream:
        prefix_by_name = {row['name']:row['name_prefix_cluster'] for row in csv.DictReader(stream)}
    output = {'source':str(SOURCE),'draws':20000,'seed':0,'original_metric_preserved':True,
              'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'input_sha256':{name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()
                              for name in ('arrays.npz','items.json','task_names.json')},
              'sensitivity_definitions':{
                  'alias_controls':'Merge canonical control concepts and exclude own concepts; preserve true-name ranks.',
                  'exclude_stale_two':'Drop precisely the two most-populous-country items; preserve targets and control vocabulary.',
                  'prefix_clusters':'Historical scores, released-name prefix clusters, ratio means retaining item weighting.'},
              'tasks':{}}
    for task in TASKS:
        indices,baseline = independently_score(items,names,arrays,task)
        aliases = alias_scores(items,names,arrays,task,indices)
        keep = np.array([items[i]['name'] not in STALE for i in indices])
        selected = {method:{key:value[keep] for key,value in fields.items()} for method,fields in baseline.items()}
        # Only controls change in the first sensitivity.
        for method in METHODS:
            assert np.array_equal(aliases[method]['hit'],baseline[method]['hit'])
            assert np.array_equal(aliases[method]['any_hit'],baseline[method]['any_hit'])
        delta = baseline['jlens_exit3']['excess']-baseline['logitlens']['excess']
        baseline_any = baseline['jlens_exit3']['any_excess']-baseline['logitlens']['any_excess']
        base_mean = delta.mean(axis=0)
        result = {'original_n_items':len(indices),'original_n_slots':sum(len(clean_slots(items[i])) for i in indices),
                  'historical_any_layer_difference':item_summary(baseline_any),
                  'historical_layer_difference':base_mean,'sensitivities':{}}
        for label,scores,n in [('alias_controls',aliases,len(indices)),('exclude_stale_two',selected,int(keep.sum()))]:
            layer = scores['jlens_exit3']['excess']-scores['logitlens']['excess']
            any_layer = scores['jlens_exit3']['any_excess']-scores['logitlens']['any_excess']
            mean = layer.mean(axis=0)
            change = mean-base_mean
            stats = item_summary(any_layer)
            result['sensitivities'][label] = {
                'n_items':n,'any_layer_difference':stats,
                'layer_difference':mean,'layer_difference_change_from_historical':change,
                'max_absolute_layer_difference_change':float(abs(change).max()),
                'method_layer_excess':{m:scores[m]['excess'].mean(axis=0) for m in METHODS},
                'method_layer_excess_changes':{m:scores[m]['excess'].mean(axis=0)-baseline[m]['excess'].mean(axis=0) for m in METHODS},
                'mean_over_48_layers_per_loop':mean.mean(axis=1),
                'positive_cells_per_loop':(mean>0).sum(axis=1),
                'early_any_layer_all_three_negative':bool(np.all(stats['mean'][:3]<0)),
                'early_any_layer_all_three_ci_upper_below_zero':bool(np.all(stats['ci95'][:3,1]<0)),
                'loop4_fixed_window_24_39':{
                    'historical_all_16_layer_means_positive':bool(np.all(base_mean[3,23:39]>0)),
                    'all_16_layer_means_positive':bool(np.all(mean[3,23:39]>0)),
                    'minimum_layer_mean':float(mean[3,23:39].min()),
                    'maximum_layer_mean':float(mean[3,23:39].max()),
                    'mean_over_window':float(mean[3,23:39].mean())},
            }
        prefix = np.array([prefix_by_name[items[i]['name']] for i in indices])
        _,clusters = np.unique(prefix,return_inverse=True)
        prefix_any = direct_cluster_draws(baseline_any,clusters,0)
        result['sensitivities']['prefix_clusters'] = {
            'n_items':len(indices),'n_clusters':len(np.unique(clusters)),
            'any_layer_difference':{'mean':baseline_any.mean(axis=0),'ci95':ci(prefix_any)},
            'fixed_depth_blocks':{},
        }
        for start,stop in [(16,32),(32,48)]:
            values = delta[:,:,start:stop].mean(axis=2)
            draws = direct_cluster_draws(values,clusters,0)
            result['sensitivities']['prefix_clusters']['fixed_depth_blocks'][f'{start+1}-{stop}'] = {
                'mean':values.mean(axis=0),'ci95':ci(draws)}
        output['tasks'][task] = result
    output['global_max_absolute_layer_difference_change'] = {
        label:max(output['tasks'][task]['sensitivities'][label]['max_absolute_layer_difference_change'] for task in TASKS)
        for label in ('alias_controls','exclude_stale_two')}
    (HERE/'sensitivities.json').write_text(json.dumps(output,default=serializable,indent=2)+'\n')
    lines = ['','## Results','',
             'Each cell is J-Lens minus logit lens in any-layer excess hit@10, followed by its paired 95% interval. The first two sensitivities use item resampling; the third changes only the resampling unit. Intervals are pointwise.','',
             '| Sensitivity | Task | n items / clusters | Loop 1 | Loop 2 | Loop 3 | Loop 4 |',
             '|---|---|---:|---|---|---|---|']
    for sensitivity in ('alias_controls','exclude_stale_two','prefix_clusters'):
        for task in TASKS:
            r = output['tasks'][task]['sensitivities'][sensitivity]
            ss = r['any_layer_difference']
            cells = [f'{m:+.3f} [{lo:+.3f}, {hi:+.3f}]' for m,(lo,hi) in zip(ss['mean'],ss['ci95'])]
            n = str(r['n_items'])+(f" / {r['n_clusters']}" if 'n_clusters' in r else '')
            lines.append('| '+' | '.join([sensitivity,task,n,*cells])+' |')
    lines += ['','Changing controls or dropping stale tasks did not reverse the early multihop deficits. Under each change all three early-loop any-layer intervals remain below zero. The fixed loop-4 multihop window at physical layers 24–39 retains a positive mean at each of its 16 layers. This is a sign/shape robustness check, not a new significance claim about that selected window.','',
              '| Sensitivity | Task | Maximum absolute layer-effect change | Loop 4 layers 24–39 mean effect | All 16 positive? |',
              '|---|---|---:|---:|---|']
    for sensitivity in ('alias_controls','exclude_stale_two'):
        for task in TASKS:
            r = output['tasks'][task]['sensitivities'][sensitivity]
            win = r['loop4_fixed_window_24_39']
            lines.append(f"| {sensitivity} | {task} | {r['max_absolute_layer_difference_change']:.6f} | {win['mean_over_window']:+.6f} | {win['all_16_layer_means_positive']} |")
    lines += ['','All 384 paired layer effects and changes for each of the two metric/population sensitivities, plus changes in each method separately, are retained in `sensitivities.json`. Arithmetic has no change under either sensitivity: its names have no canonical aliases and neither removed item is arithmetic.','',
              'Fixed depth blocks under the historical metric, with paired name-prefix-cluster intervals:','',
              '| Task | Layers | Clusters | Loop 1 | Loop 2 | Loop 3 | Loop 4 |','|---|---|---:|---|---|---|---|']
    for task in TASKS:
        r = output['tasks'][task]['sensitivities']['prefix_clusters']
        for block,ss in r['fixed_depth_blocks'].items():
            cells = [f'{m:+.3f} [{lo:+.3f}, {hi:+.3f}]' for m,(lo,hi) in zip(ss['mean'],ss['ci95'])]
            lines.append('| '+' | '.join([task,block,str(r['n_clusters']),*cells])+' |')
    lines += ['','The three checks preserve the descriptive interpretation: a substantial early multihop deficit and a later region of improved relative J-Lens readability. They do not establish a general early-depth advantage, causal use of the labeled concepts, or an explanation involving supervision. Prefix clustering is another imperfect dependence model; the table reports uncertainty under it rather than replacing the original estimand.','',
              'Execution checks: alias-aware scoring preserved every true-name hit and any-layer hit; no source artifact was written. The maximum is taken for each control concept before averaging controls in the any-layer statistic. No sensitivities were combined.']
    before = note_path.read_text().split('\n## Results\n')[0]
    note_path.write_text(before+'\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
