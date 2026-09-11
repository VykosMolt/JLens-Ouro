#!/usr/bin/env python3
"""Source-to-manuscript ledger: every empirical number/interval/count in the first and second drafts, mapped to its record.
Rows are declared by hand (quoted text → source key in SOURCE_VALUES.json or a record path); numeric agreement is checked
automatically where a machine-readable source exists (percentage points = 100 × proportion)."""
import json, csv, re
from pathlib import Path
HERE = Path(__file__).resolve().parent; SV = json.load(open(HERE / 'SOURCE_VALUES.json'))
def val(key, sub=None):
    v = SV[key]['value']
    if sub is not None:
        for s in sub: v = v[s]
    return v
rows = []
def row(version, location, quoted, source, key=None, sub=None, scale=100, kind='number', status=None, note=''):
    """quoted: string with the number(s) as printed; key/sub: SOURCE_VALUES lookup; scale: multiply source by this before comparing."""
    src_val, match = None, None
    if key is not None:
        try:
            src_val = val(key, sub)
            nums = [float(x) for x in re.findall(r'[-+]?\d+(?:\.\d+)?', quoted.replace('−', '-').replace('–', ' '))]
            svs = [float(x) * scale for x in (src_val if isinstance(src_val, (list, tuple)) else [src_val])]
            if len(nums) == len(svs):
                tol = [0.5 * 10 ** (-(len(n.split('.')[1]) if '.' in n else 0)) + 1e-9 for n in re.findall(r'\d+(?:\.\d+)?', quoted)]
                match = all(abs(a - b) <= t for a, b, t in zip(nums, svs, tol))
            else: match = None
        except Exception as e: src_val, match = f'lookup error: {e}', None
    rows.append({'version': version, 'location': location, 'quoted': quoted, 'source': source if key is None else f"{SV[key]['source_path']} :: {SV[key]['source_key']}" + (f" [{'.'.join(map(str, sub))}]" if sub else ''),
                 'source_value': json.dumps(src_val, default=str)[:160] if src_val is not None else '', 'status': status or ('exact (rounded)' if match else ('MISMATCH' if match is False else 'record/prose')), 'kind': kind, 'note': note})
for V, LOC in (('v1', 'first draft'), ('v2', 'second draft')):
    both = lambda loc, *a, **k: row(V, loc, *a, **k)
    # Abstract / §4.1 primary
    both('Abstract/§4.1', '23.19', 'analysis.json', 'conf.primary_excess_difference'); both('Abstract/§4.1', '16.29–32.90', 'analysis.json', 'conf.primary_excess_difference.interval')
    both('Abstract/§4.1', '37.92', 'analysis.json', 'conf.fit01_intended'); both('Abstract/§4.1', '14.43', 'analysis.json', 'conf.raw_intended')
    both('§4.1 Table 1', '30.44–46.56', 'analysis.json', 'conf.fit01_intended.interval'); both('§4.1 Table 1', '10.45–17.87', 'analysis.json', 'conf.raw_intended.interval')
    both('§4.1 Table 1', '0.34', 'analysis.json', 'conf.fit01_control'); both('§4.1 Table 1', '0.18–0.51', 'analysis.json', 'conf.fit01_control.interval')
    both('§4.1 Table 1', '0.04', 'analysis.json', 'conf.raw_control'); both('§4.1 Table 1', '0.02–0.08', 'analysis.json', 'conf.raw_control.interval')
    both('§4.1', '23.49', 'analysis.json', 'conf.intended_difference'); both('§4.1', '17.97–28.71', 'analysis.json', 'conf.item_resampling_interval'); both('§4.1/§3.2', '19.73 and 24.88' if V == 'v1' else '19.73 and 24.87', 'analysis.json', 'conf.leave_one_group_range', note='v1 double-rounded the report value 0.24875; exact 0.248748' if V == 'v1' else '')
    both('§4.1', '22.56', 'analysis.json', 'conf.sec.fit02_local'); both('§4.1', '9.52–35.59', 'analysis.json', 'conf.sec.fit02_local.interval'); both('§4.1', '0.63', 'analysis.json', 'conf.sec.fit01_minus_fit02_local'); both('§4.1', '−0.73 to +1.99', 'analysis.json', 'conf.sec.fit01_minus_fit02_local.interval')
    for p in (1, 2, 3):
        both('§4.2 Table 2', {1: '−8.83 [−13.64, −4.01]', 2: '−14.86 [−19.54, −10.18]', 3: '−15.59 [−22.40, −8.77]'}[p].split(' [')[0], 'analysis.json', f'conf.sec.early_loop{p}_fixed_mean')
        both('§4.2 Table 2', {1: '−13.64, −4.01', 2: '−19.54, −10.18', 3: '−22.40, −8.77'}[p], 'analysis.json', f'conf.sec.early_loop{p}_fixed_mean.interval')
        both('§4.2 Table 2', {1: '−44.41', 2: '−57.37', 3: '−61.08'}[p], 'analysis.json', f'conf.sec.early_loop{p}_any_layer'); both('§4.2 Table 2', {1: ('−59.03, −29.79' if V == 'v1' else '−59.03, −29.78'), 2: '−72.01, −42.72', 3: '−79.59, −42.58'}[p], 'analysis.json', f'conf.sec.early_loop{p}_any_layer.interval', note=('v1 double-rounded −0.29785; exact −0.297847' if (p == 1 and V == 'v1') else ''))
    for cid, q, iv in (('fit01_minus_penultimate_local', '+14.38', '+7.70, +21.07'), ('sampled_sum_minus_diagonal_local', '+15.77', '+5.80, +25.75'), ('penultimate_local', '+8.80', '+0.76, +16.85'), ('sampled_sum_local', '+21.72', '+9.16, +34.29'), ('diagonal_local', '+5.95', '−0.34, +12.24')):
        both('§4.3 Table 4', q, 'analysis.json', f'conf.sec.{cid}'); both('§4.3 Table 4', iv, 'analysis.json', f'conf.sec.{cid}.interval')
    both('§3.2', '28 dependency groups', 'analysis.json', 'conf.groups', scale=1); both('§3.2', '42, 24, 16, 10, 10, 6', 'analysis.json', 'conf.group_sizes', sub=None, scale=1, status='record/prose', note='first six of the sorted sizes; remainder four 4s and eighteen 2s')
    both('§3.2', '8.63', 'derived', 'conf.effective_groups', scale=1); both('§3.2', '20,000', 'analysis.json bootstrap', 'conf.bootstrap', sub=['replicates'], scale=1); both('§3.2', '2026090901', 'analysis.json bootstrap', 'conf.bootstrap', sub=['seed'], scale=1)
    both('§3.2', '160 two-hop questions, 80 intermediate concepts', 'RUN_SPECIFICATION.json population.items/names', status='record/prose'); both('§3.2', '96 … 45 … 19', 'benchmark.json design.relation_balance.counts', status='record/prose')
    both('§3.2', 'Three component facts overlap known calibration text', 'final_lexical_novelty.json / benchmark_review.md (items 076, 130, 151)', status='record/prose')
    # §5 verification
    both('§5 Table', '+22.93', 'VERIFICATION_METRICS fp64_local_gpu', 'ver.fp64_local_gpu.primary'); both('§5 Table', '+16.17, +32.52', 'VERIFICATION_METRICS', 'ver.fp64_local_gpu.primary_interval'); both('§5 Table', '−0.26', 'VERIFICATION_METRICS', 'ver.fp64_local_gpu.primary_change')
    both('§5 Table', '+23.61', 'VERIFICATION_METRICS regenerated', 'ver.regenerated_states_local_gpu.primary'); both('§5 Table', '+16.52, +33.77', 'VERIFICATION_METRICS', 'ver.regenerated_states_local_gpu.primary_interval'); both('§5 Table', '+0.42', 'VERIFICATION_METRICS', 'ver.regenerated_states_local_gpu.primary_change')
    both('§5', '22.60 and 23.35', 'VERIFICATION_METRICS tie_rule_bounds (minimize/maximize primary)', 'ver.executed_replay_local_gpu.tie_rule_bounds', status='record/prose', note='minimize_primary 0.22604, maximize_primary 0.23352')
    both('§5', '37 of 160', 'VERIFICATION_METRICS', 'ver.native.regenerated_bit_identical_items', scale=1); both('§5', '154', 'VERIFICATION_METRICS', 'ver.native.continuations_equal_items', scale=1)
    both('§5', '13 planted errors', 'VERIFICATION_METRICS independent_checker (14 fixture cases incl. unperturbed)', 'ver.checker', status='record/prose'); both('§5', '9×10⁻¹⁴ (CPU vs GPU FP64)', 'REPORT_confirmation_verified.md', status='record/prose')
    both('§5', '160, 190, 191 and 192 rows', 'VERIFICATION_METRICS layout paths (changes 0)', 'ver.rows190_local_gpu.primary_change', scale=1, status='record/prose')
    both('§6/§7 Reproducibility', '43 … 33 distinct', 'PRESERVATION_LEDGER.json', 'ver.unique_and_lost_count', scale=1, status='record/prose'); both('Appendix B', '−0.34 to +12.24 / +0.04 (regenerated)', 'VERIFICATION_METRICS regenerated secondary diagonal_local', 'ver.regenerated_states_local_gpu.sec.diagonal_local', status='record/prose', note='regenerated lower bound 0.00037 → +0.04 points')
    # §3.1 discovery
    both('§3.1', '18.72', 'refit report P1 loop4_historical_local_mean.mean', 'refit.multihop.loop4_historical_local_mean', sub=['mean']); both('§3.1', '0.39', 'refit report fit_sd', 'refit.multihop.loop4_historical_local_mean', sub=['fit_sd'])
    both('§3.1', '90 eligible multihop items … 51 arithmetic', 'refit report populations', 'refit.populations', status='record/prose'); both('§3.1', 'every fit peaked at physical layer 32', 'refit report loop4_learned_peaks_per_fit', 'refit.loop4_peaks_per_fit', status='record/prose')
    # Appendix A
    both('App. A', '85.1%', 'exit_agreement.json actual_exit_agreement[2][3]', 'fu.exit_agreement_-1', sub=['actual_exit_agreement_mean', 2, 3]); both('App. A', '79.1–90.5%', 'exit_agreement.json ci95[2][3]', 'fu.exit_agreement_-1', sub=['ci95', 2, 3])
    both('App. A', '1.45%', 'exit_agreement.json loop3 window 41-47 local-vs-final lens', 'fu.exit.-1.loop3.41-47', sub=['mean', 0, 1]); both('App. A', '0.58–2.51%', 'exit_agreement.json', 'fu.exit.-1.loop3.41-47', sub=['ci95', 0, 1])
    both('App. A', '96.6%', 'exit_agreement.json position -2 actual exit agreement [2][3]', 'fu.exit_agreement_-1', status='record/prose', note='position −2 record: fu.exit.-2.* (checked separately below)'); both('App. A', '0.39%', 'exit_agreement.json position -2 loop3 41-47 lens agreement', 'fu.exit.-2.loop3.41-47', sub=['mean', 0, 1])
    both('App. A', '72 passes out of 148 to 70 … 71', 'correctness.json all_148_counts', 'fu.correctness_counts', status='record/prose'); both('App. A', '59.4% … 76.4% … 35.6% … 17.9% … 36 pairs', 'probe/bounded_refit_results.json + probe_audit.md', 'fu.probe.performance', status='record/prose')
    both('App. A', '88 multihop … 51 arithmetic', 'huginn_run01 report populations', 'huginn.populations', status='record/prose'); both('App. A', 'negative under all six prespecified summaries', 'huginn_run01 metrics.csv', 'huginn.metrics', status='record/prose')
    both('§2.1', 'revision 1ed04250…', 'RUN_SPECIFICATION.json model.revision', status='record/prose'); both('§2.1', '48 blocks, four passes, 2,048, 49,152', 'RUN_SPECIFICATION.json model.geometry/vocab_size', status='record/prose')
# v2-only rows
v2 = lambda loc, *a, **k: row('v2', loc, *a, **k)
v2('Abstract/§4.2 Table 3', '−6.5 / −6.50', 'closeout RESULTS.json family_N1 C1', 'rev.family_N1', sub=['contrasts', 'C1_sameband_excess_diff_pass1', 'estimate']); v2('§4.2 Table 3', '−11.91, −1.08', 'RESULTS.json', 'rev.family_N1', sub=['contrasts', 'C1_sameband_excess_diff_pass1', 'simultaneous_95_maxt'])
v2('Abstract/§4.2 Table 3', '−8.6 / −8.64', 'RESULTS.json', 'rev.family_N1', sub=['contrasts', 'C2_sameband_excess_diff_pass2', 'estimate']); v2('§4.2 Table 3', '−12.43, −4.85', 'RESULTS.json', 'rev.family_N1', sub=['contrasts', 'C2_sameband_excess_diff_pass2', 'simultaneous_95_maxt'])
v2('Abstract/§4.2 Table 3', '−11.1 / −11.14', 'RESULTS.json', 'rev.family_N1', sub=['contrasts', 'C3_sameband_excess_diff_pass3', 'estimate']); v2('§4.2 Table 3', '−16.76, −5.52', 'RESULTS.json', 'rev.family_N1', sub=['contrasts', 'C3_sameband_excess_diff_pass3', 'simultaneous_95_maxt'])
v2('Abstract/§4.4', '20.8 / 20.80', 'RESULTS.json A', 'rev.family_N1', sub=['contrasts', 'A_within_domain_excess_diff_pass4', 'estimate']); v2('Abstract/§4.4', '9.1–32.5 / 9.13–32.46', 'RESULTS.json', 'rev.family_N1', sub=['contrasts', 'A_within_domain_excess_diff_pass4', 'simultaneous_95_maxt'])
v2('§4.4', '3.07% … 0.34%', 'RESULTS.json A per_method fit01', 'rev.A', sub=['per_method', 'fit01', 'within_domain_control_recovery', 'estimate'], status='record/prose', note='3.07% within-domain; 0.34% all-79')
v2('§4.4', '0.38% … 0.04%', 'RESULTS.json A per_method raw', 'rev.A', sub=['per_method', 'raw', 'within_domain_control_recovery', 'estimate'], status='record/prose'); v2('§4.4', '19.61', 'RESULTS.json bound_check.aggregate_lower_bound_exact', 'rev.A', sub=['bound_check', 'aggregate_lower_bound_exact'])
v2('§4.4', '+51 … +40 … about zero … −1 (+5 with 79 controls)', 'RESULTS.json A per_domain excess7_diff (countries 0.5104, environment 0.4033, si_units −0.0052, astronomy −0.0082; excess79_diff astronomy 0.051)', 'rev.A', status='record/prose'); v2('§4.4', '6 control tokens … six items', 'RESULTS.json B', 'rev.B', sub=['exact_control_overlaps'], scale=1)
v2('§4.4', '23.19 (16.25–32.72)', 'RESULTS.json B S1', 'rev.B', sub=['sensitivities', 'S1_control_filtering', 'estimate']); v2('§4.4', '22.90 (15.92–33.15)', 'RESULTS.json B S2', 'rev.B', sub=['sensitivities', 'S2_clean_subset', 'estimate'])
v2('§4.2 Table 3', '0.00% / 6.46%', 'RESULTS.json C pass 1 intended', 'rev.C', sub=['passes', 0, 'raw_intended', 'estimate'], status='record/prose', note='fit01 0.0000, raw 0.06458'); v2('§4.2 Table 3', '4.22% / 12.71%', 'RESULTS.json C pass 2', 'rev.C', sub=['passes', 1, 'raw_intended', 'estimate'], status='record/prose'); v2('§4.2 Table 3', '2.76% / 13.91%', 'RESULTS.json C pass 3', 'rev.C', sub=['passes', 2, 'raw_intended', 'estimate'], status='record/prose')
v2('§3.1', '10.96–26.99', 'refit report loop4_historical_local_mean intervals crossed_fit_item pointwise', 'refit.multihop.loop4_historical_local_mean.intervals', status='record/prose', note='REPORT.md: [0.1096, 0.2699]'); v2('§3.1', '−0.86 to 38.29', 'refit REPORT.md crossed fit/item simultaneous', status='record/prose'); v2('§3.1', '−7.49 to 44.92', 'refit REPORT.md crossed fit/component simultaneous', status='record/prose')
v2('§3.1', '6.13 (SD 0.21)', 'refit report loop4_final_third_mean', 'refit.multihop.loop4_final_third_mean', sub=['mean']); v2('§3.1', '18.91 own-name minus 0.19 control', 'refit REPORT.md own/control decomposition', status='record/prose')
v2('§6', '15.4%, 16.2%, 25.2% / 2.1, 4.3, 10.1 / 0.3, 0.4, 2.1', 'DISCOVERY_POPULATION_FIXED_BAND.json passes[*].{local_target,raw,final_target}.intended', 'localexit.discovery', status='record/prose')
for p, k, q, qi in ((0, 'local_minus_final', '+14.6', '9.1–20.9'), (1, 'local_minus_final', '+16.9', '11.5–22.8'), (2, 'local_minus_final', '+23.0', '16.0–30.6'), (0, 'local_minus_raw', '+13.0', '8.1–19.0'), (1, 'local_minus_raw', '+11.3', '6.6–16.6'), (2, 'local_minus_raw', '+14.9', '7.9–22.4')):
    v2('§6', q, 'DISCOVERY_POPULATION_FIXED_BAND.json', 'localexit.discovery', sub=[p, k, 'excess_diff']); v2('§6', qi, 'DISCOVERY_POPULATION_FIXED_BAND.json cluster interval', 'localexit.discovery', sub=[p, k, 'concept_cluster_bootstrap_95_descriptive'])
v2('§6', '162 blocks and three pass-boundary normalizations', 'DISCOVERY_POPULATION_FIXED_BAND.json passes[0].blocks_source_to_final_target_range (155–166 for layers 26–37; layer 30 → 161 blocks)', status='DISCREPANCY-CHECK', note='layer 30 of pass 1 = virtual 29; blocks to virtual 191 = 191−29 = 162 ✓')
v2('§6', '11 GB … eight GPU-hours', 'LOCAL_EXIT_STATUS.md proposal (13 files 11.2 GB; (47+95+143)/191×100×192.3 s ≈ 8.0 h)', status='record/prose')
v2('§2.2', '131 of 160 … 13–35 tokens … token 20 … 16293', 'population.json boundary_checks; SCORING_PATH_REVIEW §1a; EXAMPLES.md', status='record/prose'); v2('§2.1', 'seeds 2026090701–05 … 0–3 … RTX 5090', 'calibration_plan.json; refit LEASE.json gpu_type_id', status='record/prose')
v2('App. B', '2.639', 'RESULTS.json family_N1.max_t_95_quantile', 'rev.family_N1', sub=['max_t_95_quantile'], scale=1)
# checks of v1 statements against records where v1 lacked precision (documentation of discrepancies)
row('v1', '§2.1', 'position controls use "the same sampled target positions and all 2,048 derivative directions per paragraph"', 'combined_contract.json controls.positions (one uniform q per paragraph, seed 2026090801)', status='imprecise (corrected in v2)')
row('v1', '§3.1', '"Both positive band estimates had simultaneous intervals containing zero"', 'refit REPORT.md (56-metric family)', status='exact (prose), values added in v2')
row('v1', 'Abstract', '"change the primary estimate by at most 0.42 points"', 'VERIFICATION_METRICS regenerated change 0.00419', 'ver.regenerated_states_local_gpu.primary_change', status='exact (rounded)')
row('v1', 'App. A', 'Huginn coda/training statements cited only to internal [S2]', 'literature.md → Geiping et al. §3.3, Lu et al. §3.2', status='citation added in v2')
out = HERE / 'SOURCE_TO_MANUSCRIPT_LEDGER.csv'
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
md = ['# Source-to-manuscript ledger (v1 and v2)\n', 'Status codes: exact (rounded) = machine-checked against the record at the printed precision; record/prose = traced to a record without an automatic numeric comparison; MISMATCH = printed value disagrees with the record; imprecise/citation = documentation discrepancy corrected in v2.\n', '| Version | Location | Quoted | Source | Source value | Status |', '|---|---|---|---|---|---|']
for r in rows: md.append(f"| {r['version']} | {r['location']} | {r['quoted']} | {r['source'][:90]} | {r['source_value'][:60]} | {r['status']} |")
(HERE / 'SOURCE_TO_MANUSCRIPT_LEDGER.md').write_text('\n'.join(md))
from collections import Counter
print(Counter(r['status'] for r in rows)); print('MISMATCHES:'); [print(' ', r['version'], r['location'], r['quoted'], '->', r['source_value']) for r in rows if r['status'] == 'MISMATCH']
