#!/usr/bin/env python3
"""Collect this pass's before/after endpoint metrics, hit changes, numerical diagnostics and coverage into one file."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

V = Path(__file__).resolve().parents[1]
OUT = V / 'metrics/VERIFICATION_METRICS.json'
ARMS = ('raw', 'fit01', 'fit02', 'penultimate', 'sampled_sum', 'diagonal')
PATHS = {  # label: (sensitivity file, run, layout, evidence)
    'executed_replay_local_gpu': ('sensitivity_local_cuda.json', 'local_cuda_rtx5070ti', 'executed', 'local RTX 5070 Ti, retained 5090 states'),
    'single_row_local_gpu': ('sensitivity_local_cuda.json', 'local_cuda_rtx5070ti', 'row1', 'local RTX 5070 Ti, retained 5090 states'),
    'rows190_local_gpu': ('sensitivity_local_cuda.json', 'local_cuda_rtx5070ti', 'rows190', 'local RTX 5070 Ti, retained 5090 states'),
    'rows191_local_gpu': ('sensitivity_local_cuda.json', 'local_cuda_rtx5070ti', 'rows191', 'local RTX 5070 Ti, retained 5090 states'),
    'rows192_local_gpu': ('sensitivity_local_cuda.json', 'local_cuda_rtx5070ti', 'rows192', 'local RTX 5070 Ti, retained 5090 states'),
    'col160_local_gpu': ('sensitivity_local_cuda.json', 'local_cuda_rtx5070ti', 'col160', 'local RTX 5070 Ti, retained 5090 states'),
    'fp32_tf32_off_local_gpu': ('sensitivity_local_cuda.json', 'local_cuda_rtx5070ti', 'fp32', 'local RTX 5070 Ti, retained 5090 states'),
    'fp64_head_local_gpu': ('sensitivity_local_cuda.json', 'local_cuda_rtx5070ti', 'fp64_head', 'local RTX 5070 Ti, retained 5090 states'),
    'fp64_local_gpu': ('sensitivity_local_cuda.json', 'local_cuda_rtx5070ti', 'fp64', 'local RTX 5070 Ti, retained 5090 states'),
    'fp64_cpu': ('sensitivity_local_cpu_fp64.json', 'local_cpu_fp64', 'fp64', 'CPU, retained 5090 states'),
    'regenerated_states_local_gpu': ('sensitivity_local_regenerated.json', 'local_cuda_regenerated_states', 'executed',
                                     'local RTX 5070 Ti, states regenerated from the frozen prompts on that GPU (level 3)'),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    sources, files = {}, {}
    for name, _, _, _ in PATHS.values():
        files[name] = json.loads((V / 'replay' / name).read_text())
        sources['replay/' + name] = sha256(V / 'replay' / name)
    saved = files['sensitivity_local_cuda.json']['saved_endpoint']
    paths = {}
    for label, (name, run, layout, evidence) in PATHS.items():
        entry = files[name]['runs'][run]
        detail = entry['layouts'][layout]
        e = detail['endpoint']
        paths[label] = {
            'evidence': evidence, 'coverage': {'items': entry['items'], 'arms': list(ARMS), 'layout': layout},
            'precision_matches_original': entry['precision_matches_original'],
            'endpoint': {k: e[k] for k in ('estimates', 'changes', 'family_intervals', 'interval_endpoint_changes',
                                           'primary_item_resampling_interval', 'leave_one_group_range', 'secondary_max_t_quantile',
                                           'secondary_zero_exclusion_changed', 'secondary_sign_changed')},
            'secondary': [{k: s[k] for k in ('id', 'estimate', 'change', 'interval', 'excludes_zero', 'saved_excludes_zero')} for s in e['secondary']],
            'per_arm': {arm: {'logits_vs_executed': detail['per_arm'][arm]['logits_vs_executed'],
                              'allrank_cells_differing_from_saved': detail['per_arm'][arm]['allrank_cells_differing_from_saved'],
                              'top10_vs_saved': detail['per_arm'][arm]['top10_vs_saved'],
                              'hit_changes_vs_saved': detail['per_arm'][arm]['hit_changes_vs_saved'],
                              'ties': detail['per_arm'][arm]['ties'],
                              **({'boundary_margins_band': detail['per_arm'][arm]['boundary_margins_band']}
                                 if 'boundary_margins_band' in detail['per_arm'][arm] else {})} for arm in ARMS}}
        for key in ('tie_rule_bounds', 'perturbation_bounds'):
            if key in detail:
                paths[label][key] = {variant: {'primary': v['estimates']['primary_excess_difference'],
                                               'primary_change': v['changes']['primary_excess_difference']}
                                     for variant, v in detail[key].items()}
    local = files['sensitivity_local_cuda.json']['runs']['local_cuda_rtx5070ti']
    shape = json.loads((V / 'gpu_shape_v1/local_rtx5070ti/run.json').read_text())
    sources['gpu_shape_v1/local_rtx5070ti/run.json'] = sha256(V / 'gpu_shape_v1/local_rtx5070ti/run.json')
    variants = {}
    for record in shape['native']:
        for label, v in record['variants'].items():
            agg = variants.setdefault(label, {'items': 0, 'exactly_equal_items': 0, 'max_differing_logits': 0, 'max_abs': 0.0,
                                              'items_with_top10_change': 0, 'intended_hit_changes': 0, 'control_hit_changes': 0})
            agg['items'] += 1
            agg['exactly_equal_items'] += v['equal']
            agg['max_differing_logits'] = max(agg['max_differing_logits'], v['differing'])
            agg['max_abs'] = max(agg['max_abs'], v['max_abs'])
            agg['items_with_top10_change'] += v['top10_ids_replaced'] > 0
            agg['intended_hit_changes'] += v['hit10_changed_intended']
            agg['control_hit_changes'] += v['hit10_changed_controls']
    development = {}
    for name, entries in shape['development'].items():
        for label, v in entries.items():
            if isinstance(v, dict):
                development.setdefault(label, {'items': 0, 'equal_native_both_repeats': 0, 'max_differing': 0, 'retained_5090_output_reproduced': None})
                d = development[label]
                d['items'] += 1
                d['equal_native_both_repeats'] += all(x['equal'] for x in v['vs_native'])
                d['max_differing'] = max(d['max_differing'], *(x['differing'] for x in v['vs_native']))
                if 'vs_retained_diagnostic' in v:
                    ok = all(x['equal'] for x in v['vs_retained_diagnostic'])
                    d['retained_5090_output_reproduced'] = ok if d['retained_5090_output_reproduced'] is None else d['retained_5090_output_reproduced'] and ok
    checks = {}
    for path in sorted((V / 'checker/replay_checks').glob('*.json')):
        checks[path.stem] = json.loads(path.read_text()).get('status')
    saved_check = json.loads((V / 'checker/saved_outputs.json').read_text())
    self_test = json.loads((V / 'checker/self_test.json').read_text())
    value = {
        'schema': 'verification_metrics.v1', 'created_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha256(__file__),
        'label': 'post-confirmation verification; not prospectively registered', 'sources': sources,
        'accepted_endpoint': saved, 'paths': paths,
        'executed_path_reproduction_local_gpu': {
            'frozen_readout_arrays_equal_saved': local['frozen_readout_vs_saved'], 'saved_samples_equal': local['frozen_samples_vs_saved'],
            'transport_equals_saved_samples': local['transport_vs_saved_samples'], 'exit_logits_equal_saved': local['col160_vs_saved_exit_logits']},
        'fp64_cross_device': {arm: {'allrank_cells_differing': 0} for arm in ARMS} if False else None,
        'native_output_A_local_gpu': {'items': len(shape['native']), 'variants': variants,
                                      'regenerated_states_bit_identical_to_retained_items': sum(r['regenerated_vs_retained_states']['equal'] for r in shape['native']),
                                      'regenerated_states_max_abs_difference': max(r['regenerated_vs_retained_states']['max_abs'] for r in shape['native']),
                                      'continuations_equal_items': sum(r['continuation_equal'] for r in shape['native']),
                                      'captured_final_state_equals_native_head_input_items': sum(r['captured_final_state_equals_native_head_path_input'] for r in shape['native'])},
        'saved_sample_rows_local_gpu': {arm: {label: sum(s[label]['equal'] for s in entries) for label in entries[0] if isinstance(entries[0][label], dict)}
                                        for arm, entries in shape['samples'].items()},
        'development_diagnostic_local_gpu': development,
        'layout_logits_local_gpu_primary_arms': shape['layouts'],
        'independent_checker': {'saved_outputs': saved_check['status'], 'self_test_passed': self_test['passed'],
                                'self_test_cases': len(self_test['cases']), 'replay_roots': checks},
    }
    value.pop('fp64_cross_device')
    with OUT.open('x') as handle:
        json.dump(value, handle, indent=1, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(OUT)


if __name__ == '__main__':
    main()
