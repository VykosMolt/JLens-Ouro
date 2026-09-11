#!/usr/bin/env python3
"""Reviewer checks A–D on the accepted confirmation payload. Read-only on inputs; writes into this directory."""
import json, sys, csv, collections, re, hashlib
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
C = Path('/home/moloch/jacobian-lens/research/confirmation_2026-09-09')
A = C / 'cloud_leases/attempt_06/handoff/accepted/ouro_confirmation_20260911_fixed160_native1/ff7c0769b2b0dbe2e53b120daf7f0729434fe65e54c013288771f5052ffd4cef/results'
SAVED = C / 'results/ouro_confirmation_20260911_fixed160_native1/analysis/analysis.json'
sys.dont_write_bytecode = True
sys.path[:0] = [str(C / 'evaluation'), str(C / 'analysis')]
from measurement import PRIMARY_VIRTUAL, score, fixed_mean, secondary_specs
import analyze
assert analyze.BOOTSTRAPS == 20000
SEED_FAMILY, SEED_ITEM, SEED_SENS = 2026091101, 2026091102, 2026091103

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

pop = json.load(open(A / 'population.json')); rows = pop['rows']; names = pop['names']['multihop']; forms = pop['token_forms']['multihop']
bench = json.load(open(A / 'benchmark.json')); saved = json.load(open(SAVED))
allrank = {arm: np.load(A / f'readouts/{arm}.npz', allow_pickle=False)['allrank'] for arm in ('fit01', 'raw', 'fit02')}
hits = {arm: (allrank[arm] >= 0) & (allrank[arm] < 10) for arm in allrank}
n = len(rows); groups = [r['dependency_group_id'] for r in rows]
own = np.array([r['own_index'][0] for r in rows]); ctrl79 = [r['control_indices'][0] for r in rows]
assert all(len(c) == 79 and own[i] not in c for i, c in enumerate(ctrl79)) and all(r['eligible'] == [True] for r in rows)
domain_of = {c['intermediate']: c['domain'] for c in bench['concepts']}; assert len(domain_of) == 80
name_index = {nm: k for k, nm in enumerate(names)}
item_domain = [domain_of[names[own[i]]] for i in range(n)]
ctrl7 = []
for i in range(n):
    same = [name_index[nm] for nm, d in domain_of.items() if d == item_domain[i] and name_index[nm] != own[i]]
    assert len(same) == 7 and set(same) <= set(ctrl79[i]), (i, same)
    ctrl7.append(sorted(same))
# mapping check
bands = {p: list(range(48 * (p - 1) + 25, 48 * (p - 1) + 37)) for p in (1, 2, 3, 4)}
assert bands[4] == PRIMARY_VIRTUAL == list(range(169, 181))
mapping = [{'pass': p, 'physical_one_based': list(range(26, 38)), 'virtual_zero_based': bands[p]} for p in bands]
assert allrank['fit01'].shape == allrank['raw'].shape == (160, 128, 192)

def band_stats(arm, cols, controls):
    h = hits[arm][:, :, cols]  # [160,128,12]
    o = np.array([h[i, own[i]].mean() for i in range(n)])
    c = np.array([h[i, controls[i]].mean() for i in range(n)])
    return o, c, o - c

# ---------- reproduce frozen primary to anchor ----------
sc = {arm: score(allrank[arm], rows, pop['names'], list(range(192))) for arm in ('fit01', 'raw')}
primary_item = fixed_mean(sc['fit01']['excess'], list(range(192)), PRIMARY_VIRTUAL) - fixed_mean(sc['raw']['excess'], list(range(192)), PRIMARY_VIRTUAL)
assert abs(primary_item.mean() - saved['estimates'][0]) < 1e-12, (primary_item.mean(), saved['estimates'][0])

# ---------- A + C: per-item quantities ----------
per_item = {}
for p, cols in bands.items():
    for arm in ('fit01', 'raw', 'fit02'):
        o, c79, e79 = band_stats(arm, cols, ctrl79)
        _, c7, e7 = band_stats(arm, cols, ctrl7)
        per_item[(p, arm)] = {'own': o, 'c79': c79, 'e79': e79, 'c7': c7, 'e7': e7}
assert np.allclose(per_item[(4, 'fit01')]['e79'] - per_item[(4, 'raw')]['e79'], primary_item)
d = lambda p, k, a='fit01', b='raw': per_item[(p, a)][k] - per_item[(p, b)][k]
family_cols = {'A_within_domain_excess_diff_pass4': d(4, 'e7'), 'C1_sameband_excess_diff_pass1': d(1, 'e79'),
               'C2_sameband_excess_diff_pass2': d(2, 'e79'), 'C3_sameband_excess_diff_pass3': d(3, 'e79')}
fam = np.column_stack(list(family_cols.values()))
boots = analyze.resample(fam, groups, seed=SEED_FAMILY)
est, sd = fam.mean(axis=0), boots.std(axis=0, ddof=1)
q = float(np.quantile(np.abs((boots - est) / np.where(sd > 0, sd, 1)).max(axis=1), 0.95))
family = {}
for j, k in enumerate(family_cols):
    family[k] = {'estimate': float(est[j]), 'simultaneous_95_maxt': [float(est[j] - q * sd[j]), float(est[j] + q * sd[j])],
                 'descriptive_group_percentile_95': [float(x) for x in np.quantile(boots[:, j], [0.025, 0.975])], 'bootstrap_sd': float(sd[j])}
# descriptive components with group percentile intervals (one shared draw set, seed SEED_ITEM for item-level)
def desc(values, seed=SEED_FAMILY):
    values = np.asarray(values); b = analyze.resample(values, groups, seed=seed)
    return {'estimate': float(values.mean()), 'descriptive_group_percentile_95': [float(x) for x in np.quantile(b, [0.025, 0.975])]}
def desc_item(values, seed=SEED_ITEM):
    values = np.asarray(values); b = analyze.resample(values, [str(i) for i in range(n)], seed=seed)
    return [float(x) for x in np.quantile(b, [0.025, 0.975])]

A_out = {'definition': 'controls = the 7 other concept names in the item\'s benchmark domain (subset of the frozen 79); pass-4 band 169–180; fit01 vs raw', 'per_method': {}, 'paired': {}, 'per_domain': [], 'bound_check': {}}
for arm in ('fit01', 'raw'):
    A_out['per_method'][arm] = {k: desc(per_item[(4, arm)][kk]) for k, kk in (('intended_recovery', 'own'), ('all79_control_recovery', 'c79'), ('within_domain_control_recovery', 'c7'), ('excess_all79', 'e79'), ('excess_within_domain', 'e7'))}
A_out['paired'] = {'excess_all79_diff_ORIGINAL_PRIMARY': {'estimate': float(d(4, 'e79').mean()), 'original_group_percentile_95': saved['family_percentile_95_intervals'][0]},
                   'excess_within_domain_diff': family['A_within_domain_excess_diff_pass4'] | {'item_resampling_descriptive_95': desc_item(d(4, 'e7'))},
                   'intended_diff': desc(d(4, 'own')), 'all79_control_diff': desc(d(4, 'c79')), 'within_domain_control_diff': desc(d(4, 'c7'))}
lb_item = d(4, 'own') - (79 / 7) * per_item[(4, 'fit01')]['c79']
A_out['bound_check'] = {'inequality': 'excess7_diff >= intended_diff - (79/7)*fit01_all79_control_recovery',
    'aggregate_lower_bound_exact': float(lb_item.mean()), 'aggregate_observed_excess7_diff': float(d(4, 'e7').mean()),
    'aggregate_holds': bool(d(4, 'e7').mean() >= lb_item.mean() - 1e-12),
    'per_item_holds_all': bool(np.all(d(4, 'e7') >= lb_item - 1e-12)), 'per_item_violations': int(np.sum(d(4, 'e7') < lb_item - 1e-12)),
    'published_rounded_values_bound': 0.23490 - (79 / 7) * 0.00344, 'note': 'lower bound on the observed point estimate; not an interval or a substitute for the estimate'}
for dom in sorted(set(item_domain)):
    idx = [i for i in range(n) if item_domain[i] == dom]
    A_out['per_domain'].append({'domain': dom, 'items': len(idx), 'groups': sorted({groups[i] for i in idx}), 'names': sorted({names[own[i]] for i in idx}),
        'fit01_intended': float(per_item[(4, 'fit01')]['own'][idx].mean()), 'raw_intended': float(per_item[(4, 'raw')]['own'][idx].mean()),
        'fit01_c79': float(per_item[(4, 'fit01')]['c79'][idx].mean()), 'fit01_c7': float(per_item[(4, 'fit01')]['c7'][idx].mean()),
        'raw_c79': float(per_item[(4, 'raw')]['c79'][idx].mean()), 'raw_c7': float(per_item[(4, 'raw')]['c7'][idx].mean()),
        'excess79_diff': float(d(4, 'e79')[idx].mean()), 'excess7_diff': float(d(4, 'e7')[idx].mean())})

# ---------- C ----------
sec = {s['id']: s for s in saved['secondary_family']}
C_out = {'mapping': mapping, 'formula': 'virtual = 48*(pass-1) + (physical-1); zero-based virtual, one-based pass/physical', 'support': {'fit01': 192, 'raw': 192, 'unsupported_tail_in_any_band': False},
         'original_20_contain_same_band_early_contrast': False, 'passes': []}
for p in (1, 2, 3, 4):
    row = {'pass': p, 'virtual': bands[p]}
    for arm in ('fit01', 'raw'):
        row[f'{arm}_intended'] = desc(per_item[(p, arm)]['own']); row[f'{arm}_control79'] = desc(per_item[(p, arm)]['c79']); row[f'{arm}_excess'] = desc(per_item[(p, arm)]['e79'])
    row['paired_excess_diff'] = family[f'C{p}_sameband_excess_diff_pass{p}'] if p < 4 else {'estimate': float(d(4, 'e79').mean()), 'ORIGINAL_PRIMARY_group_percentile_95': saved['family_percentile_95_intervals'][0], 'status': 'original prospective primary; not re-inferred'}
    row['paired_intended_diff'] = desc(d(p, 'own')); row['paired_control79_diff'] = desc(d(p, 'c79'))
    row['fit02_minus_raw_excess_descriptive'] = desc(d(p, 'e79', 'fit02', 'raw'))
    if p < 4:
        row['existing_all_layer_fixed_mean_ORIGINAL'] = {k: sec[f'early_loop{p}_fixed_mean'][k] for k in ('estimate', 'simultaneous_95_interval')}
        row['existing_any_layer_ORIGINAL'] = {k: sec[f'early_loop{p}_any_layer'][k] for k in ('estimate', 'simultaneous_95_interval')}
    C_out['passes'].append(row)

# ---------- B: overlap audit ----------
import transformers
snap = C / '../verification_2026-09-11/preservation/bundle/model_snapshot'
tok = transformers.AutoTokenizer.from_pretrained(str(snap.resolve()), trust_remote_code=True, local_files_only=True)
bos = tok.bos_token_id
sys.path.insert(0, str(C / 'corrections/native_check_v1/bundle/ouro_project/src')); sys.path.insert(0, str(C / 'corrections/native_check_v1/bundle/repo'))
from ouro_jlens.evaldata import surface_forms
ledger, ans_rows = [], []
for i, r in enumerate(rows):
    ids = r['token_ids']; assert ids[0] == bos and r['readout_position'] == -1 and r['n_tokens'] == len(ids)
    dec = [tok.decode([t]) for t in ids]; text = ''.join(dec)
    assert text.startswith(r['prompt'].rstrip()) or r['prompt'].startswith(text.replace(tok.decode([bos]), '')), (i, text[:60])
    cand = [(names[own[i]], 'intended', r['intermediate_tokens'][names[own[i]]])] + [(names[k], 'control', forms[names[k]]) for k in ctrl79[i]]
    prompt_l = r['prompt'].lower()
    for nm, role, tids in cand:
        for t in tids:
            for pos in [k for k, x in enumerate(ids) if x == t]:
                ledger.append({'item': r['name'], 'role': role, 'name': nm, 'match_kind': 'exact_token_id', 'token_id': t, 'token': tok.decode([t]), 'position': pos,
                               'visible_to_scored_state': True, 'context': ''.join(dec[max(1, pos - 5):pos + 6])})
        for f in surface_forms(nm):
            fl = f.lower()
            for m in re.finditer(re.escape(fl), prompt_l):
                s, e = m.start(), m.end()
                wb = (s == 0 or not prompt_l[s - 1].isalnum()) and (e == len(prompt_l) or not prompt_l[e].isalnum())
                ledger.append({'item': r['name'], 'role': role, 'name': nm, 'match_kind': 'substring_word' if wb else 'substring_inside_word', 'form': f, 'position': f'char {s}-{e}',
                               'visible_to_scored_state': True, 'context': r['prompt'][max(0, s - 30):e + 30]})
    tgt = r['target']; tl = tgt.lower()
    tgt_ids = set(); 
    for v in (tgt, ' ' + tgt): tgt_ids.update(tok(v, add_special_tokens=False).input_ids)
    coinc = [nm for nm in names if nm.lower() == tl or nm.lower() in tl.split() or any(x in tgt_ids for x in forms[nm])]
    ans_rows.append({'item': r['name'], 'intermediate': names[own[i]], 'target': tgt, 'target_equals_own_intermediate': tl == names[own[i]].lower(),
                     'names_coinciding_with_target_or_its_tokens': coinc})
exact = [l for l in ledger if l['match_kind'] == 'exact_token_id']
B_out = {'items_audited': n, 'visible_input': 'exactly token_ids (BOS + prompt tokens up to the common-prefix boundary); one item per forward; no wrapper, demonstrations, chat template, prefilled response or padding; readout at position −1 so all tokens are causally visible',
         'exact_intended_overlaps': sum(l['role'] == 'intended' for l in exact), 'exact_control_overlaps': sum(l['role'] == 'control' for l in exact),
         'items_with_exact_control_overlap': sorted({l['item'] for l in exact if l['role'] == 'control'}),
         'substring_word_matches': sum(l['match_kind'] == 'substring_word' for l in ledger), 'substring_inside_word_matches': sum(l['match_kind'] == 'substring_inside_word' for l in ledger),
         'substring_intended': sum(l['role'] == 'intended' and l['match_kind'].startswith('substring') for l in ledger),
         'targets_equal_own_intermediate': sum(a['target_equals_own_intermediate'] for a in ans_rows),
         'items_whose_target_coincides_with_some_concept_name': [a for a in ans_rows if a['names_coinciding_with_target_or_its_tokens']],
         'semantic_clue_review': 'not automatable; substring/word matches listed in OVERLAP_LEDGER.csv for manual review',
         'caveat': 'zero exact token overlap does not establish absence of semantic information or pretraining contamination'}
# sensitivities if exact control overlap exists
if B_out['exact_control_overlaps']:
    bad = collections.defaultdict(set)
    for l in exact:
        if l['role'] == 'control': bad[l['item']].add(name_index[l['name']])
    ctrl_filt = [[k for k in ctrl79[i] if k not in bad.get(rows[i]['name'], set())] for i in range(n)]
    o1, c1, e1 = band_stats('fit01', bands[4], ctrl_filt); o2, c2, e2 = band_stats('raw', bands[4], ctrl_filt)
    s1 = e1 - e2; keep = np.array([rows[i]['name'] not in bad for i in range(n)])
    b1 = analyze.resample(s1, groups, seed=SEED_SENS); b2 = analyze.resample(primary_item[keep], [g for g, k in zip(groups, keep) if k], seed=SEED_SENS)
    B_out['sensitivities'] = {'S1_control_filtering': {'rule': 'drop exact-token-overlapping controls for the affected items, both methods; denominators recorded', 'affected_items': len(bad),
            'denominators': sorted(collections.Counter(len(c) for c in ctrl_filt).items()), 'estimate': float(s1.mean()), 'descriptive_group_percentile_95': [float(x) for x in np.quantile(b1, [0.025, 0.975])]},
        'S2_clean_subset': {'rule': 'drop affected items entirely, both methods', 'items_kept': int(keep.sum()), 'groups_kept': len(set(g for g, k in zip(groups, keep) if k)),
            'estimate': float(primary_item[keep].mean()), 'descriptive_group_percentile_95': [float(x) for x in np.quantile(b2, [0.025, 0.975])]}}
else:
    B_out['sensitivities'] = 'not required: no exact scored-token overlap for any control or intended label'

# ---------- D ----------
edges = bench['dependency_graph']['edges']; comps = bench['dependency_graph']['components']
by_name = {it['name']: it for it in bench['items']}
D_rows = []
for c in comps:
    kinds = collections.Counter(k for e in edges if e['left'] in c['items'] and e['right'] in c['items'] for k in {r['kind'] for r in e['reasons']})
    D_rows.append({'group': c['dependency_group_id'], 'size': c['size'], 'items': c['items'], 'intermediates': sorted({by_name[x]['intermediates'][0] for x in c['items']}),
                   'domains': sorted({domain_of[by_name[x]['intermediates'][0]] for x in c['items']}), 'edge_kind_counts': dict(kinds),
                   'saved_primary_mean': next(h['primary_mean'] for h in saved['group_heterogeneity'] if h['group'] == c['dependency_group_id']),
                   'saved_without_group': next(h['without_group'] for h in saved['group_heterogeneity'] if h['group'] == c['dependency_group_id'])})
# dg006 structure: sub-clusters using identity+template edges only, then bridging collision edges
dg6 = next(c for c in comps if c['dependency_group_id'] == 'dg006')['items']
def components(nodes, es):
    adj = {x: set() for x in nodes}
    for e in es: adj[e['left']].add(e['right']); adj[e['right']].add(e['left'])
    seen, out = set(), []
    for x in sorted(nodes):
        if x in seen: continue
        stack, comp = [x], set()
        while stack:
            y = stack.pop()
            if y in comp: continue
            comp.add(y); stack.extend(adj[y] - comp)
        seen |= comp; out.append(sorted(comp))
    return out
strong_kinds = {'same_intermediate_identity', 'shared_underlying_fact', 'concrete_entity_substitution_template'}
in6 = [e for e in edges if e['left'] in dg6 and e['right'] in dg6]
strong = [e for e in in6 if {r['kind'] for r in e['reasons']} & strong_kinds]
sub = components(dg6, strong)
sub_of = {x: k for k, comp in enumerate(sub) for x in comp}
bridges = [{'left': e['left'], 'right': e['right'], 'left_intermediate': by_name[e['left']]['intermediates'][0], 'right_intermediate': by_name[e['right']]['intermediates'][0],
            'left_target': by_name[e['left']]['target'], 'right_target': by_name[e['right']]['target'], 'reasons': e['reasons']}
           for e in in6 if sub_of[e['left']] != sub_of[e['right']]]
tmpl = collections.defaultdict(set)
for e in strong:
    for r in e['reasons']:
        if r['kind'] == 'concrete_entity_substitution_template': tmpl[str(r.get('value', r.get('template', r)))].update([e['left'], e['right']])
D_out = {'rule': bench['dependency_design']['grouping_policy'], 'edge_count': len(edges), 'group_count': len(comps), 'group_sizes': sorted((c['size'] for c in comps), reverse=True),
         'effective_group_count': saved['eligible_items'] ** 2 / sum(c['size'] ** 2 for c in comps), 'retained_from_analysis': {'primary_item_resampling_95_interval': saved['primary_item_resampling_95_interval'],
         'leave_one_group_range': saved['leave_one_group_range'], 'group_bootstrap': saved['bootstrap']}, 'groups': D_rows,
         'dg006': {'size': len(dg6), 'subclusters_by_identity_fact_template_edges': [{'items': s, 'intermediates': sorted({by_name[x]['intermediates'][0] for x in s})} for s in sub],
                   'bridging_collision_edges': bridges, 'template_families_inside': {k: sorted(v) for k, v in tmpl.items()}}}

# ---------- write outputs ----------
out = {'schema': 'closeout_reviewer_checks.v1', 'plan_sha256': sha(HERE / 'ANALYSIS_PLAN.md'), 'inputs': {'population_sha256': sha(A / 'population.json'), 'benchmark_sha256': sha(A / 'benchmark.json'),
       'fit01_npz_sha256': sha(A / 'readouts/fit01.npz'), 'raw_npz_sha256': sha(A / 'readouts/raw.npz'), 'fit02_npz_sha256': sha(A / 'readouts/fit02.npz'), 'saved_analysis_sha256': sha(SAVED)},
       'anchor_primary_reproduced': float(primary_item.mean()), 'family_N1': {'policy': 'centered bootstrap max-t, 4 contrasts, 20000 whole-group draws', 'seed': SEED_FAMILY, 'max_t_95_quantile': q, 'contrasts': family},
       'A_within_domain': A_out, 'B_overlap': B_out, 'C_same_band': C_out, 'D_groups': D_out}
(HERE / 'RESULTS.json').write_text(json.dumps(out, indent=1))
with open(HERE / 'OVERLAP_LEDGER.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['item', 'role', 'name', 'match_kind', 'token_id', 'token', 'form', 'position', 'visible_to_scored_state', 'context']); w.writeheader()
    for l in ledger: w.writerow({k: l.get(k, '') for k in w.fieldnames})
with open(HERE / 'ANSWER_FORM_COINCIDENCE.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(ans_rows[0].keys())); w.writeheader(); [w.writerow(a) for a in ans_rows]
with open(HERE / 'PER_ITEM.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['item', 'group', 'domain', 'intermediate'] + [f'{k}_pass{p}_{arm}' for p in (1, 2, 3, 4) for arm in ('fit01', 'raw') for k in ('own', 'c79', 'c7', 'e79', 'e7')])
    for i in range(n): w.writerow([rows[i]['name'], groups[i], item_domain[i], names[own[i]]] + [f'{per_item[(p, arm)][k][i]:.10f}' for p in (1, 2, 3, 4) for arm in ('fit01', 'raw') for k in ('own', 'c79', 'c7', 'e79', 'e7')])
with open(HERE / 'GROUP_MEMBERSHIP.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['group', 'size', 'item', 'intermediate', 'domain', 'target', 'prompt'])
    for c in comps:
        for x in c['items']: it = by_name[x]; w.writerow([c['dependency_group_id'], c['size'], x, it['intermediates'][0], domain_of[it['intermediates'][0]], it['target'], it['prompt']])
print(json.dumps({'anchor': out['anchor_primary_reproduced'], 'family': {k: (round(v['estimate'], 5), [round(x, 5) for x in v['simultaneous_95_maxt']]) for k, v in family.items()}, 'q': q,
                  'bound': A_out['bound_check'], 'B': {k: B_out[k] for k in ('exact_intended_overlaps', 'exact_control_overlaps', 'substring_word_matches', 'substring_inside_word_matches', 'substring_intended', 'targets_equal_own_intermediate')},
                  'B_sens': B_out['sensitivities'] if isinstance(B_out['sensitivities'], str) else {k: (v['estimate'], v['descriptive_group_percentile_95']) for k, v in B_out['sensitivities'].items()},
                  'dg006_subclusters': len(sub), 'dg006_bridges': len(bridges)}, indent=1, default=str))
