#!/usr/bin/env python3
"""A6: the bundle holds no bank, population, benchmark or confirmation prompt, and its shared files equal precision_v1.
Usage: python -B tests/audit_bundle.py
"""
import json

from common import BUNDLE, PARENT_BUNDLE, ROUND, record, require, save_evidence, tree

AUTHORED = {'evaluation/validate_outputs.py', 'evaluation/worker.py', 'frozen/output_contract.json', 'frozen/run_spec.json'}
INHERITED = ('development_item_names', 'model_identity', 'model_record', 'original_precision')


def main():
    files = tree(BUNDLE)
    parent = json.loads((ROUND / 'corrections/precision_v1/FREEZE.json').read_text())['files']
    spec = json.loads((BUNDLE / 'frozen/run_spec.json').read_text())
    parent_spec = json.loads((PARENT_BUNDLE / 'frozen/run_spec.json').read_text())
    contract = json.loads((BUNDLE / 'frozen/output_contract.json').read_text())
    shared = sorted(set(files) - AUTHORED)
    require(AUTHORED <= set(files), 'an authored file is missing')
    require(all(files[name] == parent.get(name) == record(PARENT_BUNDLE / name) for name in shared),
            'a shared file differs from the precision_v1 freeze')
    pins = json.loads((ROUND / 'cloud_leases/attempt_05/LEASE.json').read_text())['controller_sources']
    monitor = ROUND / 'cloud_leases/attempt_05/monitor'
    require(all(files['controller/' + name] == pins[name] == record(monitor / name) for name in pins),
            'controller differs from the attempt05 pins')
    require(not [n for n in files if n.rsplit('/', 1)[-1] in ('population.json', 'benchmark.json', 'PROSPECTIVE_PLAN.md')
                 or n.endswith(('.pt', '.pth', '.npz', '.safetensors', '.bin'))], 'input, bank or tensor file present')
    require(spec['bank_records'] == {} and contract['stages'] == {}, 'banks or stages declared')
    require(not {'population_record', 'benchmark_record', 'arm_support', 'planned_items'} & set(spec),
            'confirmation inputs referenced by the run spec')
    require(spec['source_records'] == {n: files[n] for n in files if not n.startswith('frozen/')}, 'source records differ')
    require(all(spec[key] == parent_spec[key] for key in INHERITED), 'inherited run-spec fields differ')
    population = json.loads((PARENT_BUNDLE / 'frozen/population.json').read_text())
    benchmark = json.loads((PARENT_BUNDLE / 'frozen/benchmark.json').read_text())
    prompts = {row['prompt'] for row in population['rows']} | {item['prompt'] for item in benchmark['items']}
    # Raw UTF-8 and both JSON escapings of every confirmation prompt.
    needles = {form for text in prompts for form in (text.encode(), json.dumps(text)[1:-1].encode(),
                                                     json.dumps(text, ensure_ascii=False)[1:-1].encode())}
    found = sorted(name for name in files if any(needle in (BUNDLE / name).read_bytes() for needle in needles))
    require(not found, 'confirmation prompt text found in: ' + ', '.join(found))
    historical = {row['name'] for row in json.loads((BUNDLE / 'repo/data/evaluations/lens-eval-multihop.json').read_text())['items']}
    require(set(spec['development_item_names']) <= historical - {row['name'] for row in population['rows']},
            'development items are not historical old items')
    save_evidence('A6_bundle_audit.json', {
        'schema': 'native_logit_bundle_audit.v1', 'status': 'passed', 'bundle_records': files,
        'authored_files': sorted(AUTHORED), 'shared_files_identical_to_precision_v1': len(shared),
        'controller_identical_to_attempt05_pins_and_monitor': sorted(pins),
        'bank_records': spec['bank_records'], 'stages': contract['stages'],
        'confirmation_prompts_searched': len(prompts), 'search_forms': len(needles), 'prompt_matches': found,
        'inherited_run_spec_fields_identical': list(INHERITED)})
    print(json.dumps({'status': 'passed', 'files': len(files), 'prompts_searched': len(prompts)}))


if __name__ == '__main__':
    main()
