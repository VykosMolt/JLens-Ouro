"""Independent arithmetic-probe audit. Does not import the original evaluator.

Run using ouro_project/venv/bin/python. Saved-rank analysis runs by default;
--refit additionally performs the prospectively specified bounded CPU checks.
All labels, prompt order, folds, and statistics are reconstructed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import warnings

import numpy as np

ROOT = Path('/home/moloch/ouro_project')
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'artifacts/jlens/probe/cv_all648'
CACHE = ROOT / 'artifacts/jlens/probe/n80_v2/gpu_cache.npz'
SEED = 20260907
DRAWS = 20000


def record(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return {'path': str(path), 'sha256': h.hexdigest(), 'size': path.stat().st_size}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def design():
    prompts = np.array([(a, b, c) for a in range(1, 10)
                        for b in range(1, 10) for c in range(2, 10)])
    y = prompts[:, :2].sum(1)
    pairs = sorted(set(tuple(int(v) for v in sorted(x[:2])) for x in prompts))
    pid = np.array([pairs.index(tuple(sorted(x[:2]))) for x in prompts])
    assign = np.zeros(45, int)
    assign[np.random.default_rng(0).permutation(45)] = np.arange(45) % 5
    fold = assign[pid]
    eligible = np.array([label in y[fold != f] for label, f in zip(y, fold)])
    rng = np.random.default_rng(1)
    vals = []
    for f in range(5):
        train_pairs = np.flatnonzero(assign != f)
        vals.append(train_pairs[rng.permutation(len(train_pairs))[:8]])
    return prompts, y, pairs, pid, fold, eligible, vals


def select(ranks, fold, eligible, selection=None, mask=None):
    hit = ranks.reshape(648, 4, 48) == 0
    choices = np.zeros((5, 4), int)
    scores = np.zeros((648, 4), float)
    selected_rows = eligible if mask is None else eligible & mask
    for f in range(5):
        if selection is None:
            choices[f] = hit[(fold != f) & selected_rows].mean(0).argmax(1)
        else:
            choices[f] = selection[f].reshape(4, 48).argmax(1)
        for loop in range(4):
            scores[fold == f, loop] = hit[fold == f, loop, choices[f, loop]]
    return scores, choices


def ci(values):
    return np.percentile(values, [2.5, 97.5], axis=0).T.tolist()


def rank_analysis():
    a = dict(np.load(SOURCE / 'arrays.npz', allow_pickle=False))
    prompts, y, pairs, pid, folds, eligible, vals = design()
    assert np.array_equal(a['folds'], folds)
    assert np.array_equal(a['labels'], y)
    assert eligible.sum() == 576
    assert len(set(pid[eligible])) == 39
    old = json.loads((SOURCE / 'summary.json').read_text())
    des = json.loads((SOURCE / 'design.json').read_text())
    assert record(SOURCE / 'arrays.npz')['sha256'] == old['input']['sha256']
    assert record(CACHE)['sha256'] == des['inputs']['gpu_cache']['sha256']
    with np.load(SOURCE / 'lens_all648.npz') as l:
        for name in l.files:
            assert np.array_equal(l[name], a[name]), name
    for f in range(5):
        observed = sorted([list(pairs[g]) for g in vals[f]])
        assert observed == des['design']['validation_pairs_by_fold'][f]
    clusters = np.unique(pid[eligible])
    counts = np.array([sum(pid[eligible] == g) for g in clusters])
    cf = np.array([folds[pid == g][0] for g in clusters])
    rng = np.random.default_rng(SEED)
    draws = []
    while len(draws) < DRAWS:
        d = np.bincount(rng.choice(39, 39, replace=True), minlength=39)
        if len(set(cf[d > 0])) == 5:
            draws.append(d)
    weights = np.array(draws)
    denom = weights @ counts
    names = {'probe': 'probe_rank', 'logit_lens': 'll_cand', 'j_lens': 'jl_cand',
             'logit_lens_one_form': 'll_one', 'j_lens_one_form': 'jl_one'}
    fixed_boot, reselect_boot, points, choices, score_rows = {}, {}, {}, {}, {}
    result = {'draws': DRAWS, 'seed': SEED, 'population': {
        'prompts': 648, 'eligible_prompts': int(eligible.sum()),
        'clusters': 45, 'eligible_clusters': 39,
        'excluded_pairs': [list(pairs[g]) for g in np.unique(pid[~eligible])],
        'fold_test_prompts': [int(sum(folds == f)) for f in range(5)],
        'fold_training_prompts': [int(sum(folds != f)) for f in range(5)],
        'fold_eligible_test_prompts': [int(sum((folds == f) & eligible)) for f in range(5)],
        'inner_train_prompts': [int(sum((folds != f) & ~np.isin(pid, vals[f]))) for f in range(5)],
        'inner_validation_prompts': [int(sum(np.isin(pid, vals[f]))) for f in range(5)],
        'inner_validation_absent_label_prompts': [
            int(sum(~np.isin(y[np.isin(pid, vals[f])],
                            y[(folds != f) & ~np.isin(pid, vals[f])]))) for f in range(5)],
    }, 'readouts': {}, 'contrasts': {}}
    majority_predictions = np.zeros(648,int); majority_choices = []
    for f in range(5):
        values, frequencies = np.unique(y[folds != f],return_counts=True)
        label = int(values[frequencies.argmax()])
        majority_choices.append(label); majority_predictions[folds == f] = label
    result['baselines'] = {
        'constant_sum10_eligible_accuracy':float(np.mean(y[eligible]==10)),
        'training_fold_majority_smallest_label_tie_breaking_choices':majority_choices,
        'training_fold_majority_eligible_accuracy':float(np.mean(majority_predictions[eligible]==y[eligible])),
        'fold_label_counts':[{str(label):int(sum(y[folds==f]==label)) for label in range(2,19)} for f in range(5)],
    }
    for name, key in names.items():
        selection = a['selection_accuracy'] if name == 'probe' else None
        score, chosen = select(a[key], folds, eligible, selection)
        score_rows[name], choices[name] = score, chosen
        point = score[eligible].mean(0)
        points[name] = point
        sums = np.array([score[pid == g].sum(0) for g in clusters])
        fixed_boot[name] = (weights @ sums) / denom[:, None]
        # Independently reproduce the published asymmetry: frozen supervised
        # choices, but resample the fixed lenses' layer-selection population.
        if selection is None:
            hits = (a[key] == 0).reshape(648, 4, 48)
            hs = np.array([hits[pid == g].sum(0) for g in clusters])
            numer = np.zeros((DRAWS, 4))
            for f in range(5):
                tr = cf != f
                te = ~tr
                train_score = (weights[:, tr] @ hs[tr].reshape(sum(tr), -1)).reshape(DRAWS, 4, 48)
                layer = train_score.argmax(2)
                test_score = (weights[:, te] @ hs[te].reshape(sum(te), -1)).reshape(DRAWS, 4, 48)
                numer += np.take_along_axis(test_score, layer[:, :, None], axis=2)[:, :, 0]
            reselect_boot[name] = numer / denom[:, None]
        else:
            reselect_boot[name] = fixed_boot[name]
        result['readouts'][name] = {'per_loop': point.tolist(),
            'ci95_fixed_choices': ci(fixed_boot[name]),
            'ci95_published_selection_policy': ci(reselect_boot[name]),
            'selected_physical_layer_one_based_per_fold': (chosen + 1).tolist(),
            'fold_loop1_accuracy': [float(score[(folds == f) & eligible, 0].mean()) for f in range(5)],
            'all648_same_selected_layers': score.mean(0).tolist()}
    for left, right in [('probe', 'logit_lens'), ('j_lens', 'logit_lens'),
                        ('j_lens', 'probe'), ('logit_lens_one_form', 'logit_lens'),
                        ('j_lens_one_form', 'j_lens')]:
        result['contrasts'][left + '_minus_' + right] = {
            'per_loop': (points[left] - points[right]).tolist(),
            'ci95_fixed_choices': ci(fixed_boot[left] - fixed_boot[right]),
            'ci95_published_selection_policy': ci(reselect_boot[left] - reselect_boot[right])}
    for name, old_name in [('probe','supervised_probe'), ('logit_lens','logit_lens'),
                           ('j_lens','eventual_exit_jacobian_lens')]:
        assert np.allclose(points[name], old['readouts'][old_name]['per_loop'], atol=5e-7)
    # Prospectively specified selection-budget check: choose each fixed lens's
    # layer on the exact same inner-validation pairs and trainable classes as
    # the probe. The probe's impossible inner-validation labels contribute zero
    # at every location, so removing them cannot change its original argmax.
    matched_boot = {'probe': fixed_boot['probe']}
    matched_points = {'probe': points['probe']}
    result['matched_inner_selection'] = {'readouts': {}, 'contrasts': {}}
    for name,key in names.items():
        if name == 'probe': continue
        matched = np.zeros((648,4),float); selected = np.zeros((5,4),int)
        hit = (a[key] == 0).reshape(648,4,48)
        for f in range(5):
            va = np.isin(pid,vals[f]); tr = (folds != f) & ~va
            selectable = va & np.isin(y,np.unique(y[tr]))
            selected[f] = hit[selectable].mean(0).argmax(1)
            for loop in range(4):
                matched[folds==f,loop] = hit[folds==f,loop,selected[f,loop]]
        sums = np.array([matched[pid==g].sum(0) for g in clusters])
        matched_boot[name] = weights@sums/denom[:,None]
        matched_points[name] = matched[eligible].mean(0)
        result['matched_inner_selection']['readouts'][name] = {
            'per_loop':matched_points[name].tolist(), 'ci95':ci(matched_boot[name]),
            'selected_layers_one_based':(selected+1).tolist()}
    for left,right in [('probe','logit_lens'),('probe','logit_lens_one_form'),('j_lens','logit_lens')]:
        result['matched_inner_selection']['contrasts'][left+'_minus_'+right] = {
            'per_loop':(matched_points[left]-matched_points[right]).tolist(),
            'ci95':ci(matched_boot[left]-matched_boot[right])}
    result['selected_C'] = [[float(a['chosen_C'][f, loop*48 + choices['probe'][f, loop]])
                            for loop in range(4)] for f in range(5)]
    result['C_counts_all_locations'] = {str(c): int(sum(a['chosen_C'].ravel() == c))
                                       for c in np.unique(a['chosen_C'])}
    result['selection_optimism_diagnostic'] = {}
    for name, key in names.items():
        acc = (a[key][eligible] == 0).mean(0).reshape(4, 48)
        result['selection_optimism_diagnostic'][name] = {
            'test_oracle_best_fixed_layer_per_loop': acc.max(1).tolist(),
            'test_oracle_layer_one_based': (acc.argmax(1) + 1).tolist()}
    # Keep the published selections when describing success/failure subgroups.
    result['correctness_descriptive'] = {'correct_prompts': int(a['correct'][eligible].sum())}
    for name, scores in score_rows.items():
        result['correctness_descriptive'][name] = {
            'passing': scores[eligible & a['correct']].mean(0).tolist(),
            'failing': scores[eligible & ~a['correct']].mean(0).tolist()}
    result['inputs'] = [record(SOURCE/'arrays.npz'), record(SOURCE/'design.json'),
                        record(SOURCE/'lens_all648.npz'), record(CACHE)]
    result['source_manifest_current_match'] = {}
    for source in des['source_files']:
        if source['path'].startswith('src/'):
            result['source_manifest_current_match'][source['path']] = (
                record(ROOT/source['path'])['sha256'] == source['sha256'])
    save_json(OUT/'saved_rank_audit.json', result)
    rows = []
    for loop in range(4):
        for layer in range(48):
            row = {'loop': loop+1, 'physical_layer': layer+1}
            for name, key in names.items():
                row[name] = float(np.mean(a[key][eligible, loop*48+layer] == 0))
            rows.append(row)
    with (OUT/'probe_layerwise.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/jlens-probe-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
    for loop, ax in enumerate(axes):
        for name, label, color in [('probe', 'Supervised probe', '#58508d'),
                                  ('logit_lens','Logit lens','#007f86'),
                                  ('j_lens','J-Lens (older fit)','#e45c3b')]:
            ys = [row[name] for row in rows if row['loop'] == loop+1]
            ax.plot(range(1,49), ys, label=label, color=color)
        ax.set_title(f'Loop {loop+1}'); ax.set_xlabel('Physical layer'); ax.set_ylim(0,1)
    axes[0].set_ylabel('17-candidate top-1 accuracy'); axes[0].legend(fontsize=8)
    fig.suptitle('576 eligible arithmetic prompts; fixed-layer curves are descriptive')
    fig.tight_layout(); fig.savefig(OUT/'probe_layerwise.png', dpi=180); plt.close(fig)
    np.savez_compressed(OUT/'audit_pairing.npz', weights=weights, clusters=clusters,
                        counts=counts, pid=pid, eligible=eligible,
                        **{name: arr for name, arr in score_rows.items()})
    print(json.dumps({'population': result['population'], 'contrasts': result['contrasts'],
                      'selected_C': result['selected_C']}, indent=2), flush=True)
    return a, (prompts, y, pairs, pid, folds, eligible, vals), choices


def bounded_refits(a, d, choices):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    prompts, y, pairs, pid, folds, eligible, vals = d
    with np.load(CACHE) as c:
        H = c['H']
    assert H.shape == (648,192,2048) and H.dtype == np.float16
    assert np.isfinite(H).all()
    result = {'prospective_design': {
        'target': '17 nominal sum classes a+b; same outer unordered-pair folds',
        'locations': 'five original inner-selected loop-1 physical layers, fixed',
        'learning_curve': '18 versus 36 pairs; 3 deterministic class-covering subsets per fold',
        'subset_seeds': [20260907, 20260908, 20260909],
        'regularization': 'same mean-loss L2 strength: C_half=C_full*n_full/n_half',
        'C_check': 'one added C=10 chosen only if existing inner-validation accuracy improves',
        'optimization_check': 'same C/full data with tol=1e-6 versus 1e-4; max_iter=4000 versus2000',
        'uncertainty': '20,000 paired eligible-pair cluster resamples; fixed selections and subsets',
    }, 'folds': []}
    predicted = {name: np.full(648, -1, int) for name in ('original','tight','C_extended')}
    predicted['half'] = np.full((648,3), -1, int)
    diagnostic = []
    for f in range(5):
        location = int(choices['probe'][f,0]); C = float(a['chosen_C'][f,location])
        X = H[:,location].astype(np.float32)
        train = folds != f; test = ~train; inner_val = np.isin(pid, vals[f]); inner = train & ~inner_val
        def fit(mask, c, tol=1e-4, max_iter=2000):
            scaler = StandardScaler().fit(X[mask]); xs = scaler.transform(X)
            with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always', ConvergenceWarning)
                clf = LogisticRegression(C=c, max_iter=max_iter, solver='lbfgs',tol=tol).fit(xs[mask], y[mask])
            pred = clf.predict(xs)
            diag = {'C':c, 'training_rows':int(mask.sum()), 'n_iter':int(clf.n_iter_[0]),
                    'train_accuracy':float(np.mean(pred[mask] == y[mask])),
                    'warnings':[str(w.message) for w in ws],
                    'coefficient_l2':float(np.linalg.norm(clf.coef_))}
            return clf, xs, pred, diag
        clf, xs, pred, diag = fit(train,C)
        predicted['original'][test] = pred[test]
        prob = clf.predict_proba(xs[test]); true = np.zeros(sum(test))
        for j, label in enumerate(y[test]):
            col = np.flatnonzero(clf.classes_ == label)
            if len(col): true[j] = prob[j,col[0]]
        ranks = (prob > true[:,None]).sum(1)
        assert np.array_equal(ranks, a['probe_rank'][test,location]), (f, 'rank mismatch')
        _, _, tight, tight_diag = fit(train,C,tol=1e-6,max_iter=4000)
        predicted['tight'][test] = tight[test]
        _, _, alt_val, alt_diag = fit(inner,10.)
        alt_acc = float(np.mean(alt_val[inner_val] == y[inner_val]))
        original_val_acc = float(a['selection_accuracy'][f,location])
        new_C = 10. if alt_acc > original_val_acc else C
        if new_C == C:
            extended, extended_diag = pred, diag
        else:
            _, _, extended, extended_diag = fit(train,new_C)
        predicted['C_extended'][test] = extended[test]
        subsets = []
        train_ids = np.unique(pid[train]); pair_sums = np.array([sum(pairs[g]) for g in train_ids])
        for rep in range(3):
            rng = np.random.default_rng(SEED+rep+f*100)
            chosen = [int(rng.choice(train_ids[pair_sums == label])) for label in np.unique(pair_sums)]
            remaining = np.array([g for g in train_ids if g not in chosen])
            chosen.extend(rng.choice(remaining,18-len(chosen),replace=False).tolist())
            half = np.isin(pid, chosen)
            assert len(np.unique(pid[half])) ==18 and np.array_equal(np.unique(y[half]),np.unique(y[train]))
            half_C = C * sum(train)/sum(half)
            _, _, half_pred, half_diag = fit(half,half_C)
            predicted['half'][test,rep] = half_pred[test]
            subsets.append({'pairs':[list(pairs[g]) for g in sorted(chosen)],
                            'training_diagnostic':half_diag})
        result['folds'].append({'fold':f,'physical_layer_one_based':location+1,
            'original':diag,'tight':tight_diag,'C10_inner':alt_diag,
            'C10_inner_accuracy':alt_acc,'original_inner_accuracy':original_val_acc,
            'extended_C_selected':new_C,'extended':extended_diag, 'subsets':subsets})
        print(f'fold {f} exact ranks reproduced; C={C}; train accuracy={diag["train_accuracy"]:.3f}; '
              f'tol check={sum(pred[test]!=tight[test])} changed predictions; C10 validation='
              f'{alt_acc:.3f} versus {original_val_acc:.3f}',flush=True)
    for p in predicted.values(): assert np.all(p>=0)
    b = dict(np.load(OUT/'audit_pairing.npz'))
    hits = {name:(pred==y[:,None]).mean(1) if name=='half' else (pred==y).astype(float)
            for name,pred in predicted.items()}
    boot = {}; result['performance'] = {}
    for name, score in hits.items():
        sums = np.array([score[pid == g].sum() for g in b['clusters']])
        boot[name] = b['weights']@sums/(b['weights']@b['counts'])
        result['performance'][name] = {'point':float(score[eligible].mean()),
            'ci95':ci(boot[name]),
            'per_fold':[float(score[(folds==f)&eligible].mean()) for f in range(5)]}
    result['contrasts'] = {}
    for left,right in [('original','half'),('tight','original'),('C_extended','original')]:
        result['contrasts'][left+'_minus_'+right] = {
            'point':float((hits[left]-hits[right])[eligible].mean()), 'ci95':ci(boot[left]-boot[right])}
    np.savez_compressed(OUT/'bounded_refit_predictions.npz', **predicted, y=y, eligible=eligible,
                        folds=folds, pid=pid)
    save_json(OUT/'bounded_refit_results.json',result)
    rng=np.random.default_rng(SEED+22)
    selected=rng.choice(np.flatnonzero(eligible),8,replace=False)
    for i in selected:
        a0,b0,c0=prompts[i]; f=folds[i]
        diagnostic.append({'prompt':f'({a0} + {b0}) * {c0} = ', 'intermediate':int(y[i]),
             'target':int(y[i]*c0), 'model_passes_saved_check':bool(a['correct'][i]),
             'probe_prediction':int(predicted['original'][i]),
             'probe_true_rank':int(a['probe_rank'][i,choices['probe'][f,0]]),
             'logit_lens_true_rank':int(a['ll_cand'][i,choices['logit_lens'][f,0]]),
             'J_lens_true_rank':int(a['jl_cand'][i,choices['j_lens'][f,0]])})
    save_json(OUT/'random_examples.json',diagnostic)
    print(json.dumps({'performance':result['performance'],'contrasts':result['contrasts']},indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--refit',action='store_true'); args=parser.parse_args()
    a,d,choices=rank_analysis()
    if args.refit: bounded_refits(a,d,choices)
