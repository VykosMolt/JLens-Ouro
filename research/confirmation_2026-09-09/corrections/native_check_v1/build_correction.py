"""Freeze the native-check correction: the development check unembeds the whole final-step sequence; nothing else changes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
ROUND = HERE.parents[1]
BUNDLE = HERE / 'bundle'
PARENT = ROUND / 'corrections/precision_v1'
DIAGNOSTIC = (ROUND / 'cloud_leases/diagnostic_native_logit_v1/handoff/accepted/ouro_native_logit_diagnostic_v1/'
              'f9751036b1fa746fc8a37287c755093cde92238ead58347f78869eeb949063e6')
RUN_ID = 'ouro_confirmation_20260911_fixed160_native1'
CHANGED = 'evaluation/readouts.py'
# Sized from measured work: ~200 s of scoring and ~6 min of setup, plus the 8 GB bank upload (1 h 50 min to 3 h 28 min so far).
BUDGETS = {'setup_budget_seconds': 12800, 'compute_budget_seconds': 1200, 'preservation_reserve_seconds': 900}


def read(path):
    return json.loads(path.read_text())


def record(path):
    assert path.is_file() and not path.is_symlink()
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def write(path, value, *, replace=False):
    with path.open('w' if replace else 'x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


parent = read(PARENT / 'FREEZE.json')
assert not (HERE / 'FREEZE.json').exists() and not (HERE / 'bundle.tar.gz').exists()
actual = {p.relative_to(BUNDLE).as_posix(): record(p) for p in sorted(BUNDLE.rglob('*')) if p.is_file()}
assert set(actual) == set(parent['files'])
assert [name for name, expected in parent['files'].items() if actual[name] != expected] == [CHANGED]
assert read(DIAGNOSTIC / 'RECEIPT.json')['status'] == 'passed'

correction = {
    'schema': 'confirmation_native_check_correction.v1',
    'parent_run_id': parent['run_id'],
    'parent_freeze': record(PARENT / 'FREEZE.json'),
    'original_readouts': parent['files'][CHANGED],
    'corrected_readouts': actual[CHANGED],
    'diagnostic_receipt': record(DIAGNOSTIC / 'RECEIPT.json'),
    'diagnostic_comparisons': record(DIAGNOSTIC / 'results/diagnostic/comparisons.json'),
    'diagnostic_reanalysis': record(ROUND / 'reviews/diagnostic_native_logit_v1_reanalysis.json'),
    'new_confirmation_results_exposed': False,
    'scope': 'The native development check unembeds the whole final-step sequence [1,S,2048], the shape the native head uses; '
             'on the deployed GPU a single row gives different BF16 logits. Exact equality and every other gate, input, '
             'estimator, scoring and analysis rule are unchanged.',
}
write(BUNDLE / 'frozen/native_check_correction.json', correction)

spec_path = BUNDLE / 'frozen/run_spec.json'
spec = read(spec_path)
assert CHANGED in spec['source_records']
spec['run_id'] = RUN_ID
spec['frozen_utc'] = datetime.now(timezone.utc).isoformat()
spec['source_records'][CHANGED] = actual[CHANGED]
spec['native_check_correction_record'] = record(BUNDLE / 'frozen/native_check_correction.json')
write(spec_path, spec, replace=True)

contract_path = BUNDLE / 'frozen/output_contract.json'
contract = read(contract_path)
contract['run_id'] = RUN_ID
contract['run_spec_sha256'] = record(spec_path)['sha256']
write(contract_path, contract, replace=True)

config = read(PARENT / 'run_config.json')
config.update(BUDGETS, run_id=RUN_ID, run_spec_path=str(spec_path), output_contract_path=str(contract_path))
config['semantic_verifier']['path'] = str(BUNDLE / 'evaluation/validate_outputs.py')
assert config['semantic_verifier']['record'] == record(BUNDLE / 'evaluation/validate_outputs.py')
write(HERE / 'run_config.json', config)

with tarfile.open(HERE / 'bundle.tar.gz', 'x:gz') as archive:
    for path in sorted(BUNDLE.rglob('*')):
        if path.is_file():
            archive.add(path, arcname='bundle/' + path.relative_to(BUNDLE).as_posix(), recursive=False)
write(HERE / 'FREEZE.json', {
    'schema': 'confirmation_prospective_freeze.v1', 'run_id': RUN_ID, 'frozen_utc': spec['frozen_utc'],
    'new_confirmation_results_exposed': False, 'parent_freeze': record(PARENT / 'FREEZE.json'),
    'builder_record': record(Path(__file__)), 'bundle': record(HERE / 'bundle.tar.gz'),
    'controller_run_config': record(HERE / 'run_config.json'),
    'files': {p.relative_to(BUNDLE).as_posix(): record(p) for p in sorted(BUNDLE.rglob('*')) if p.is_file()},
})
print(json.dumps({'status': 'frozen', 'run_id': RUN_ID, 'receipt': record(HERE / 'FREEZE.json')}))
