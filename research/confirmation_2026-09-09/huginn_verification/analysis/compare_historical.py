#!/usr/bin/env python3
"""Compare accepted rerun readouts with authenticated history, without inference."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import numpy as np

ROUND = Path(__file__).resolve().parents[2]
REPO = ROUND.parents[1]
OLD = REPO / 'research/refit_round_2026-09-07'
HISTORY = OLD / 'monitoring/attempt_06/huginn_handoff_20260909T153351Z_659a482c/results/huginn_evaluation'
SEEDS = (2026090803, 2026090804)
HISTORICAL_BANK_SHA = '7eddc849bca857a1406a901d8e4998b6bf90580b09f6ac6b326690ef378dc596'
SUPPLEMENT_SHA = 'fa243c6f58ac622f4b1b5746ae85fa5b19d84892421353ff025e664f5aa05010'


def read(path):
    return json.loads(Path(path).read_text())


def guard(path):
    path = Path(path).absolute()
    parents = []
    for parent in reversed(path.parents):
        value = parent.lstat()
        if not stat.S_ISDIR(value.st_mode):
            raise ValueError('Linked or non-directory comparison parent: ' + str(parent))
        parents.append((value.st_dev, value.st_ino))
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError('Expected a regular file: ' + str(path))
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns, tuple(parents))


def record(path):
    before = guard(path)
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != before[:5]:
            raise ValueError('Opened input differs from checked path')
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
        after = os.fstat(stream.fileno())
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != before[:5]:
            raise ValueError('Opened input changed during hash')
    if guard(path) != before:
        raise ValueError('Input changed during hash: ' + str(path))
    return {'bytes': before[2], 'sha256': digest.hexdigest()}


def compare_array(old, new):
    old, new = np.asarray(old), np.asarray(new)
    if old.shape != new.shape or old.dtype != new.dtype or old.dtype.kind not in 'biuf':
        raise ValueError('Array geometry or numeric dtype differs')
    if not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError('Nonfinite comparison input')
    # Compare representations as well as numeric values, including signed zero.
    a, b = np.ascontiguousarray(old).reshape(-1), np.ascontiguousarray(new).reshape(-1)
    bits = a.view(np.dtype('u' + str(a.dtype.itemsize))) != b.view(np.dtype('u' + str(b.dtype.itemsize)))
    changed = np.flatnonzero(bits)
    max_absolute = 0.0
    sum_squared = 0.0
    for start in range(0, a.size, 1_000_000):
        difference = b[start:start+1_000_000].astype(np.float64) - a[start:start+1_000_000].astype(np.float64)
        max_absolute = max(max_absolute, float(np.abs(difference).max(initial=0)))
        sum_squared += float(np.dot(difference, difference))
    return {'shape': list(old.shape), 'dtype': str(old.dtype), 'entries': int(old.size),
            'bitwise_different_entries': int(len(changed)), 'numeric_different_entries': int(np.count_nonzero(a != b)),
            'max_absolute_difference': max_absolute, 'rms_difference': float(np.sqrt(sum_squared / max(1, a.size))),
            'first_differences': [{'index': [int(x) for x in np.unravel_index(int(i), old.shape)],
                                   'historical': a[i].item(), 'rerun': b[i].item()} for i in changed[:16]]}


def compare_json(old, new, path=''):
    if isinstance(old, dict) and isinstance(new, dict) and set(old) == set(new):
        return [row for key in sorted(old) for row in compare_json(old[key], new[key], path + '/' + key)]
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        return [row for i, (a, b) in enumerate(zip(old, new)) for row in compare_json(a, b, path + '/' + str(i))]
    if type(old) is type(new) and old == new:
        return []
    return [{'path': path, 'historical': old, 'rerun': new}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--accepted', type=Path, required=True, help='Directory containing RECEIPT.json and results/')
    parser.add_argument('--expected-receipt-sha256', required=True, help='Hash of the independently inspected actual external receipt')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    accepted = args.accepted.absolute()
    observed, guards = {}, {}

    def check(path, expected):
        path = Path(path)
        actual = record(path)
        if actual != {key: expected[key] for key in ('bytes', 'sha256')}:
            raise ValueError('Comparison input differs from its accepted record: ' + str(path))
        observed[str(path)] = actual
        guards[str(path)] = guard(path)
        return path

    def raw(path):
        path = Path(path)
        expected = observed[str(path)]
        if expected['bytes'] > 256 * 1024**2:
            raise ValueError('Comparison deserialization input exceeds its bound')
        before = guard(path)
        with path.open('rb') as handle:
            opened = os.fstat(handle.fileno())
            if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != before[:5]:
                raise ValueError('Deserialization opened a different file')
            content = handle.read(256 * 1024**2 + 1)
            after = os.fstat(handle.fileno())
            if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != before[:5]:
                raise ValueError('Deserialization file changed during read')
        if (guard(path) != before or {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()} != expected):
            raise ValueError('Deserialized bytes differ from their accepted record')
        return content

    def data(path):
        return json.loads(raw(path))

    receipt_record = record(accepted / 'RECEIPT.json')
    if receipt_record['sha256'] != args.expected_receipt_sha256:
        raise ValueError('External receipt differs from the independently inspected pin')
    check(accepted / 'RECEIPT.json', receipt_record)
    receipt = data(accepted / 'RECEIPT.json')
    check(accepted / 'MANIFEST.json', receipt['manifest_record'])
    check(accepted / 'VALIDATION.json', receipt['validation_record'])
    manifest = data(accepted / 'MANIFEST.json')
    validation = data(accepted / 'VALIDATION.json')
    manifest_id = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':'),
                                          ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if (receipt.get('status') != 'passed' or receipt.get('kind') != 'final'
            or manifest.get('outcome') != 'complete' or manifest.get('kind') != 'final'
            or receipt.get('binding') != manifest.get('binding')
            or receipt.get('manifest_record') != record(accepted / 'MANIFEST.json')
            or receipt.get('validation_record') != record(accepted / 'VALIDATION.json')
            or receipt.get('files') != manifest.get('files')
            or receipt.get('manifest_sha256') != manifest_id
            or validation.get('status') != 'passed'
            or validation.get('run_id') != receipt['binding']['run_id']
            or validation.get('contract_sha256') != receipt['binding']['output_contract_sha256']
            or validation.get('manifest_sha256') != manifest_id
            or sorted(validation.get('checked_files', [])) != sorted(receipt['files'])
            or any(validation.get('checks', {}).get(key) is not True for key in ('loadability', 'numerical'))):
        raise ValueError('Expected one complete externally accepted final chain')
    supplement_path = ROUND / 'artifacts/RETAINED_INPUTS_SUPPLEMENT.json'
    supplement_record = record(supplement_path)
    if supplement_record['sha256'] != SUPPLEMENT_SHA:
        raise ValueError('Historical supplement differs from its fixed trusted pin')
    check(supplement_path, supplement_record)
    historical_records = data(supplement_path)['all_consumed_file_records']

    def pair(relative):
        historical = HISTORY / relative
        fresh = accepted / 'results/readouts' / relative
        expected = historical_records.get(str(historical))
        if expected is None:
            complete_path = HISTORY / 'COMPLETE.json'
            check(complete_path, historical_records[str(complete_path)])
            complete = data(complete_path)
            section = Path(relative).parent.as_posix()
            seal_record = next(row['seal'] for row in complete['sections'] if row['section'] == section)
            seal_path = check(HISTORY / section / 'SEAL.json', seal_record)
            seal = data(seal_path)
            if seal['identity_sha256'] != complete['identity_sha256']:
                raise ValueError('Historical section identity differs')
            expected = seal['files'][Path(relative).name]
        return (check(historical, expected),
                check(fresh, receipt['files']['readouts/' + relative]))

    result = {'schema': 'huginn_historical_rerun_comparison.v1',
              'category': 'newly_rerun_estimator', 'utc': datetime.now(timezone.utc).isoformat(),
              'acceptance_receipt': record(accepted / 'RECEIPT.json'), 'binding': receipt['binding'],
              'historical_supplement': record(supplement_path), 'seeds': {}}
    lens = check(accepted / 'results/fit/lens.pt', receipt['files']['fit/lens.pt'])
    result['bank'] = {'rerun': observed[str(lens)], 'historical_sha256': HISTORICAL_BANK_SHA,
                      'whole_file_matches_history': observed[str(lens)]['sha256'] == HISTORICAL_BANK_SHA,
                      'historical_binary_available_for_matrix_comparison': False}
    for seed in SEEDS:
        prefix = f'seeds/{seed}/'
        rows = {}
        a, b = pair(prefix + 'arrays.npz')
        with np.load(io.BytesIO(raw(a)), allow_pickle=False) as old, np.load(io.BytesIO(raw(b)), allow_pickle=False) as new:
            if set(old.files) != set(new.files):
                raise ValueError('Saved rank/argmax array fields differ')
            rows['rank_and_argmax_arrays'] = {key: compare_array(old[key], new[key]) for key in old.files}
        a, b = pair(prefix + 'cache.pt')
        old = torch.load(io.BytesIO(raw(a)), map_location='cpu', weights_only=True)
        new = torch.load(io.BytesIO(raw(b)), map_location='cpu', weights_only=True)
        if set(old) != {'H', 'target_states', 'native_logits'} or set(new) != set(old):
            raise ValueError('Saved state/logit fields differ')
        rows['cached_tensors'] = {key: compare_array(old[key].numpy(), new[key].numpy()) for key in old}
        del old, new
        for filename in ('initializations.json', 'summaries.json'):
            a, b = pair(prefix + filename)
            differences = compare_json(data(a), data(b))
            rows[filename] = {'exact_value_equality': not differences, 'different_fields': len(differences),
                              'differences': differences}
        result['seeds'][str(seed)] = rows
    if any(guard(path) != expected for path, expected in guards.items()):
        raise ValueError('An input changed during comparison')
    result['input_records'] = observed
    result['source_record'] = record(Path(__file__))
    result['interpretation'] = ('Differences are reported, not treated as permission to tune or repeat a run. '
                                'A matching rerun does not recover the lost original checkpoint.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as output:
        json.dump(result, output, indent=2, sort_keys=True, allow_nan=False)
        output.write('\n')
    print(json.dumps({'report': str(args.out), 'bank_matches_history': result['bank']['whole_file_matches_history']}))


if __name__ == '__main__':
    main()
