"""Freeze the reviewed precision correction without modifying the original freeze."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
ROUND = HERE.parents[1]
BUNDLE = HERE / 'bundle'


def read(path):
    return json.loads(path.read_text())


def record(path):
    assert path.is_file() and not path.is_symlink()
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def write(path, value, *, replace=False):
    with path.open('w' if replace else 'x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


parent = read(ROUND / 'FREEZE.json')
candidate = read(HERE / 'CANDIDATE.json')
review = read(HERE / 'IMPLEMENTATION_REVIEW.json')
assert candidate['parent_freeze'] == record(ROUND / 'FREEZE.json')
assert review['status'] == 'passed' and review['new_confirmation_results_exposed'] is False
assert review['parent_freeze_record'] == candidate['parent_freeze']
assert review['corrected_worker_record'] == candidate['corrected_worker']
assert record(BUNDLE / 'evaluation/worker.py') == candidate['corrected_worker']
assert record(ROUND / 'resources/PRECISION_PREFLIGHT_DIAGNOSIS.json') == candidate['diagnosis']
assert not (HERE / 'FREEZE.json').exists() and not (HERE / 'bundle.tar.gz').exists()
actual = {p.relative_to(BUNDLE).as_posix(): record(p)
          for p in sorted(BUNDLE.rglob('*')) if p.is_file()}
assert set(actual) == set(parent['files'])
for name, expected in parent['files'].items():
    assert record(ROUND / 'bundle' / name) == expected
    if name != 'evaluation/worker.py':
        assert actual[name] == expected, name
for directory in ('evaluation', 'analysis'):
    for path in sorted((BUNDLE / directory).glob('*.py')):
        assert review['reviewed_sources'][str(path)] == record(path)
failure = read(ROUND / 'resources/ATTEMPT03_PREFLIGHT_FAILURE_PRESERVED.json')
assert failure['remote']['worker_quiescent'] and not failure['remote']['development_ack_present']
assert failure['failed_final']['outcome'] == 'failed' and failure['failed_final']['manifest']['files'] == {}
lease = read(ROUND / 'cloud_leases/attempt_03/LEASE.json')
assert lease['status'] == 'terminated' and not lease.get('stage_acceptances')

correction = {
    'schema': 'confirmation_precision_correction.v1',
    'parent_run_id': parent['run_id'], 'parent_freeze': candidate['parent_freeze'],
    'original_worker': candidate['original_worker'], 'corrected_worker': candidate['corrected_worker'],
    'diagnosis': read(ROUND / 'resources/PRECISION_PREFLIGHT_DIAGNOSIS.json'),
    'diagnosis_record': candidate['diagnosis'],
    'failed_attempt_preservation': record(ROUND / 'resources/ATTEMPT03_PREFLIGHT_FAILURE_PRESERVED.json'),
    'failed_terminal_manifest': failure['failed_final']['manifest'],
    'implementation_review': record(HERE / 'IMPLEMENTATION_REVIEW.json'),
    'new_confirmation_results_exposed': False,
    'scope': 'Remove redundant legacy TF32 setters; retain exact original precision equality and every scientific gate. '
             'Benchmark, population, plan, estimator records, scoring, analysis, native checks and numerical validator remain unchanged.',
}
write(BUNDLE / 'frozen/precision_correction.json', correction)
review_path = BUNDLE / 'frozen/implementation_review.json'
review_path.write_bytes((HERE / 'IMPLEMENTATION_REVIEW.json').read_bytes())
spec_path = BUNDLE / 'frozen/run_spec.json'
spec = read(spec_path)
spec['run_id'] = 'ouro_confirmation_20260910_fixed160_precision1'
spec['frozen_utc'] = datetime.now(timezone.utc).isoformat()
spec['implementation_review_record'] = record(review_path)
spec['source_records']['evaluation/worker.py'] = record(BUNDLE / 'evaluation/worker.py')
spec['precision_correction_record'] = record(BUNDLE / 'frozen/precision_correction.json')
write(spec_path, spec, replace=True)
contract_path = BUNDLE / 'frozen/output_contract.json'
contract = read(contract_path)
contract['run_id'] = spec['run_id']
contract['run_spec_sha256'] = record(spec_path)['sha256']
write(contract_path, contract, replace=True)
config = read(ROUND / 'controller/run_config_retry_02.json')
config.update(run_id=spec['run_id'], run_spec_path=str(spec_path), output_contract_path=str(contract_path))
config['semantic_verifier']['path'] = str(BUNDLE / 'evaluation/validate_outputs.py')
assert config['semantic_verifier']['record'] == record(BUNDLE / 'evaluation/validate_outputs.py')
write(HERE / 'run_config.json', config)
with tarfile.open(HERE / 'bundle.tar.gz', 'x:gz') as archive:
    for path in sorted(BUNDLE.rglob('*')):
        if path.is_file():
            archive.add(path, arcname='bundle/' + path.relative_to(BUNDLE).as_posix(), recursive=False)
receipt = {
    'schema': 'confirmation_prospective_freeze.v1', 'run_id': spec['run_id'],
    'frozen_utc': spec['frozen_utc'], 'new_confirmation_results_exposed': False,
    'parent_freeze': candidate['parent_freeze'], 'builder_record': record(Path(__file__)),
    'bundle': record(HERE / 'bundle.tar.gz'), 'controller_run_config': record(HERE / 'run_config.json'),
    'files': {p.relative_to(BUNDLE).as_posix(): record(p)
              for p in sorted(BUNDLE.rglob('*')) if p.is_file()},
}
write(HERE / 'FREEZE.json', receipt)
print(json.dumps({'status': 'frozen', 'run_id': spec['run_id'], 'receipt': record(HERE / 'FREEZE.json')}))
