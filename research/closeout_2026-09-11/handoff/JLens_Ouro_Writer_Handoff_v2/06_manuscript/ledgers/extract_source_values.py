#!/usr/bin/env python3
"""Collect every machine-readable source value the manuscripts cite, with path and key, into SOURCE_VALUES.json."""
import json, csv
from pathlib import Path
RS = Path('/home/moloch/ouro_project/jacobian-lens/research'); HERE = Path(__file__).resolve().parent
C = RS / 'confirmation_2026-09-09'; V = RS / 'verification_2026-09-11'; R = RS / 'refit_round_2026-09-07'; F = RS / 'followup_2026-09-07'; X = RS / 'closeout_2026-09-11'
vals = {}
def put(key, value, path, jkey, note=''): vals[key] = {'value': value, 'source_path': str(path), 'source_key': jkey, 'note': note}
an = json.load(open(C / 'results/ouro_confirmation_20260911_fixed160_native1/analysis/analysis.json')); ap = 'results/.../analysis/analysis.json'
for k, e, iv in zip(an['component_order'], an['estimates'], an['family_percentile_95_intervals']): put(f'conf.{k}', e, ap, f'estimates[{k}]'); put(f'conf.{k}.interval', iv, ap, f'family_percentile_95_intervals[{k}]')
put('conf.item_resampling_interval', an['primary_item_resampling_95_interval'], ap, 'primary_item_resampling_95_interval'); put('conf.leave_one_group_range', an['leave_one_group_range'], ap, 'leave_one_group_range')
put('conf.groups', an['groups'], ap, 'groups'); put('conf.eligible_items', an['eligible_items'], ap, 'eligible_items'); put('conf.secondary_max_t', an['secondary_max_t_95_quantile'], ap, 'secondary_max_t_95_quantile')
put('conf.bootstrap', an['bootstrap'], ap, 'bootstrap'); sizes = sorted((h['items'] for h in an['group_heterogeneity']), reverse=True); put('conf.group_sizes', sizes, ap, 'group_heterogeneity[*].items'); put('conf.effective_groups', 160 ** 2 / sum(s * s for s in sizes), ap, 'derived 160^2/sum n_g^2')
for s in an['secondary_family']: put(f'conf.sec.{s["id"]}', s['estimate'], ap, f'secondary_family[{s["id"]}].estimate'); put(f'conf.sec.{s["id"]}.interval', s['simultaneous_95_interval'], ap, f'secondary_family[{s["id"]}].simultaneous_95_interval')
m = json.load(open(V / 'metrics/VERIFICATION_METRICS.json')); mp = 'verification_2026-09-11/metrics/VERIFICATION_METRICS.json'
for path in ('executed_replay_local_gpu', 'fp64_local_gpu', 'fp64_cpu', 'fp32_tf32_off_local_gpu', 'regenerated_states_local_gpu', 'single_row_local_gpu', 'rows190_local_gpu', 'rows191_local_gpu', 'rows192_local_gpu', 'col160_local_gpu'):
    ep = m['paths'][path]['endpoint']; put(f'ver.{path}.primary', ep['estimates']['primary_excess_difference'] if 'estimates' in ep else None, mp, f'paths.{path}.endpoint.estimates.primary_excess_difference')
    put(f'ver.{path}.primary_change', ep['changes']['primary_excess_difference'], mp, f'paths.{path}.endpoint.changes.primary_excess_difference')
    if 'family_intervals' in ep: put(f'ver.{path}.primary_interval', ep['family_intervals']['primary_excess_difference'], mp, f'paths.{path}.endpoint.family_intervals.primary_excess_difference')
    for kk in ('fit01_intended', 'raw_intended', 'intended_difference', 'control_difference'):
        if 'estimates' in ep: put(f'ver.{path}.{kk}', ep['estimates'][kk], mp, f'paths.{path}.endpoint.estimates.{kk}')
    for s in m['paths'][path].get('secondary', []): put(f'ver.{path}.sec.{s["id"]}', {'estimate': s['estimate'], 'interval': s['interval'], 'excludes_zero': s['excludes_zero']}, mp, f'paths.{path}.secondary[{s["id"]}]')
for path in ('executed_replay_local_gpu', 'regenerated_states_local_gpu'):
    put(f'ver.{path}.tie_rule_bounds', m['paths'][path]['tie_rule_bounds'], mp, f'paths.{path}.tie_rule_bounds'); put(f'ver.{path}.perturbation_bounds', m['paths'][path]['perturbation_bounds'], mp, f'paths.{path}.perturbation_bounds')
na = m['native_output_A_local_gpu']; put('ver.native.regenerated_bit_identical_items', na['regenerated_states_bit_identical_to_retained_items'], mp, 'native_output_A_local_gpu.regenerated_states_bit_identical_to_retained_items'); put('ver.native.continuations_equal_items', na['continuations_equal_items'], mp, 'native_output_A_local_gpu.continuations_equal_items'); put('ver.native.variants', na['variants'], mp, 'native_output_A_local_gpu.variants')
put('ver.checker', {k: m['independent_checker'][k] for k in ('self_test_cases', 'self_test_passed', 'saved_outputs')}, mp, 'independent_checker'); put('ver.checker.replay_roots', len(m['independent_checker']['replay_roots']), mp, 'independent_checker.replay_roots')
sr = m['paths']['single_row_local_gpu']['per_arm']; put('ver.single_row.per_arm_allrank_cells_differing', {a: sr[a]['allrank_cells_differing_from_saved'] for a in sr}, mp, 'paths.single_row_local_gpu.per_arm.*.allrank_cells_differing_from_saved')
fv = (V / 'FINAL_VERIFICATION.md').read_text(); put('ver.protected_root_files', 7937, 'FINAL_VERIFICATION.md', 'prose: 7,937 files'); put('ver.purged_files', 43, 'FINAL_VERIFICATION.md/ARTIFACT_AUDIT.md', 'prose: 43 purged files (33 distinct contents)'); put('ver.unresolved_files', 19, 'ARTIFACT_AUDIT.md', 'prose'); put('ver.bundle', {'files': 7380, 'bytes': 14968501364}, 'preservation/BUNDLE_MANIFEST_core.json', 'files/bytes')
led = json.load(open(V / 'preservation/PRESERVATION_LEDGER.json')); put('ver.unique_and_lost_count', len(led['unique_and_lost']), 'preservation/PRESERVATION_LEDGER.json', 'len(unique_and_lost)'); put('ver.distinct_lost', led['distinct_contents']['class_c_distinct_sha256'], 'PRESERVATION_LEDGER.json', 'distinct_contents.class_c_distinct_sha256'); put('ver.unresolved_count', len(led['unresolved']), 'PRESERVATION_LEDGER.json', 'len(unresolved)')
# refit
rep = json.load(open(R / 'analysis/main_run01/report.json')); mh = rep['P1']['tasks']['multihop']['metrics']; rp = 'refit_round_2026-09-07/analysis/main_run01/report.json'
mdefs = rep['P1']['metric_definitions']; assert len(mdefs) == len(mh['mean']) == 28
for i, md in enumerate(mdefs):
    k = md['id']
    put(f'refit.multihop.{k}', {'mean': mh['mean'][i], 'fit_sd': mh['fit_sd'][i], 'fit_min': mh['fit_min'][i], 'fit_max': mh['fit_max'][i], 'per_fit': [mh['per_fit'][f][i] for f in range(5)], 'positive_fits': mh['positive_fits'][i], 'negative_fits': mh['negative_fits'][i]}, rp, f'P1.tasks.multihop.metrics[*][{i}] (id {k})')
    put(f'refit.multihop.{k}.intervals', {fam: {lvl: mh['intervals'][fam][lvl][i] for lvl in mh['intervals'][fam]} for fam in mh['intervals']}, rp, f'P1.tasks.multihop.metrics.intervals[*][*][{i}]')
put('refit.multihop.metric_names', [md['id'] for md in mdefs], rp, 'P1.metric_definitions[*].id'); put('refit.loop4_peak', rep['P1']['tasks']['multihop']['loop4_learned_peak_of_fit_mean'], rp, 'P1.tasks.multihop.loop4_learned_peak_of_fit_mean'); put('refit.loop4_peaks_per_fit', [(p['fit_id'], p['tied_one_based_physical_layers'], p['maximum']) for p in rep['P1']['tasks']['multihop']['loop4_learned_peaks_per_fit']], rp, 'P1.tasks.multihop.loop4_learned_peaks_per_fit')
put('refit.populations', {t: {k: rep['populations'][t][k] for k in ('component_count',)} | {'items': len(rep['populations'][t]['item_indices'])} for t in rep['populations']}, rp, 'populations'); put('refit.families', rep['P1']['families'], rp, 'P1.families')
o2 = json.load(open(R / 'analysis/ouro_run01/report.json'))['P2']; put('refit.P2.keys', list(o2.keys()), 'refit .../ouro_run01/report.json', 'P2'); put('refit.P2.multihop', o2['tasks']['multihop'] if 'tasks' in o2 else None, 'refit .../ouro_run01/report.json', 'P2.tasks.multihop', 'large; consult file')
hm = list(csv.DictReader(open(R / 'analysis/huginn_run01/metrics.csv'))); put('huginn.metrics', hm, 'refit .../huginn_run01/metrics.csv', 'rows'); hr = json.load(open(R / 'analysis/huginn_run01/report.json')); put('huginn.populations', {t: {k: v for k, v in hr['populations'][t].items() if k in ('component_count', 'eligible_items', 'eligible_slots', 'n_items', 'n_slots')} for t in hr['populations']}, 'huginn_run01/report.json', 'populations'); put('huginn.counts', hr.get('counts'), 'huginn_run01/report.json', 'counts')
# followup
ea = json.load(open(F / 'results/exit_agreement.json')); put('fu.exit_agreement_-1', {'actual_exit_agreement_mean': ea['-1']['actual_exit_agreement']['mean'], 'ci95': ea['-1']['actual_exit_agreement']['ci95'], 'n_items': ea['-1']['n_items']}, 'followup .../results/exit_agreement.json', "['-1'].actual_exit_agreement")
for pos in ('-1', '-2'):
    for loop in ('1', '2', '3', '4'):
        lw = ea[pos]['loops'][loop]['windows']
        for win in lw: put(f'fu.exit.{pos}.loop{loop}.{win}', {'labels': ea[pos]['loops'][loop]['matrix_labels'], 'mean': lw[win]['top1_agreement_matrix']['mean'], 'ci95': lw[win]['top1_agreement_matrix']['ci95']}, 'followup exit_agreement.json', f"['{pos}'].loops.{loop}.windows.{win}.top1_agreement_matrix")
        for k in ea[pos]['loops'][loop]:
            if k not in ('windows', 'matrix_labels'): put(f'fu.exit.{pos}.loop{loop}.{k}', ea[pos]['loops'][loop][k], 'followup exit_agreement.json', f"['{pos}'].loops.{loop}.{k}")
co = json.load(open(F / 'results/correctness.json')); put('fu.correctness_counts', co['all_148_counts'], 'followup results/correctness.json', 'all_148_counts')
for crit in co['criteria']:
    for task in co['criteria'][crit]:
        t = co['criteria'][crit][task]; put(f'fu.correctness.{crit}.{task}.n', {'passing': t['n_passing'], 'failing': t['n_failing']}, 'followup correctness.json', f'criteria.{crit}.{task}')
        for k in t:
            if k not in ('strata', 'n_passing', 'n_failing'): put(f'fu.correctness.{crit}.{task}.{k}', t[k], 'followup correctness.json', f'criteria.{crit}.{task}.{k}', 'large' if isinstance(t[k], (dict, list)) and len(json.dumps(t[k])) > 2000 else '')
pr = json.load(open(F / 'probe/bounded_refit_results.json')); put('fu.probe.performance', pr['performance'], 'followup probe/bounded_refit_results.json', 'performance'); put('fu.probe.contrasts', pr['contrasts'], 'followup probe/bounded_refit_results.json', 'contrasts'); put('fu.probe.design', pr['prospective_design'], 'followup probe/bounded_refit_results.json', 'prospective_design')
lw = json.load(open(F / 'results/layerwise.json')); put('fu.layerwise.bootstrap', lw['bootstrap'], 'followup results/layerwise.json', 'bootstrap')
for t in lw['tasks']:
    tk = lw['tasks'][t]; put(f'fu.layerwise.{t}.n', {k: tk[k] for k in ('n_items', 'n_slots', 'n_concept_clusters')}, 'followup layerwise.json', f'tasks.{t}')
    for k in tk:
        if k not in ('item_indices',) and not isinstance(tk[k], list): put(f'fu.layerwise.{t}.{k}', tk[k], 'followup layerwise.json', f'tasks.{t}.{k}', 'large' if isinstance(tk[k], dict) and len(json.dumps(tk[k])) > 3000 else '')
ir = json.load(open(F / 'audit/independent_reproduction.json')); put('fu.independent_reproduction', ir, 'followup audit/independent_reproduction.json', '*', 'large' if len(json.dumps(ir)) > 3000 else '')
# closeout
rv = json.load(open(X / 'reviewer_checks/RESULTS.json')); put('rev.family_N1', rv['family_N1'], 'closeout reviewer_checks/RESULTS.json', 'family_N1'); put('rev.A', rv['A_within_domain'], 'closeout RESULTS.json', 'A_within_domain'); put('rev.B', {k: v for k, v in rv['B_overlap'].items() if k != 'items_whose_target_coincides_with_some_concept_name'}, 'closeout RESULTS.json', 'B_overlap'); put('rev.C', rv['C_same_band'], 'closeout RESULTS.json', 'C_same_band'); put('rev.D.effective', rv['D_groups']['effective_group_count'], 'closeout RESULTS.json', 'D_groups.effective_group_count')
le = json.load(open(X / 'local_exit/DISCOVERY_POPULATION_FIXED_BAND.json')); put('localexit.discovery', le['passes'], 'closeout local_exit/DISCOVERY_POPULATION_FIXED_BAND.json', 'passes')
(HERE / 'SOURCE_VALUES.json').write_text(json.dumps(vals, indent=1, default=str))
print('source values:', len(vals)); print('refit metric names:', vals['refit.multihop.metric_names']['value'])
print('probe performance keys:', list(pr['performance'].keys())[:20]); print('probe contrasts keys:', list(pr['contrasts'].keys())[:20])
print('fu exit -1 loop3 windows:', list(ea['-1']['loops']['3']['windows'].keys()), 'other keys:', [k for k in ea['-1']['loops']['3'] if k not in ('windows','matrix_labels')])
print('fu layerwise multihop keys:', [k for k in lw['tasks']['multihop'] if k != 'item_indices'][:30])
print('correctness multihop hist keys:', [k for k in co['criteria']['historical']['multihop'] if k!='strata'])
