#!/usr/bin/env python3
"""Build, archive, freeze and verify the native-logit diagnostic bundle.

  build.py bundle   copy shared precision_v1 files; write run spec, output contract and run config
  build.py archive  create bundle.tar.gz from the bundle directory
  build.py freeze   write FREEZE.json binding every frozen file and the acceptance evidence
  build.py verify   check FREEZE.json against the directory and the archive

Run with python -B so no bytecode is written into the bundle.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

HERE = Path(__file__).resolve().parent
ROUND = HERE.parents[1]
PARENT = ROUND / 'corrections/precision_v1'
BUNDLE = HERE / 'bundle'
ARCHIVE = HERE / 'bundle.tar.gz'
RECEIPT = HERE / 'FREEZE.json'
RUN_ID = 'ouro_native_logit_diagnostic_v1'
JOB_CAP_USD = 1.5
LEASE_ROOT = ROUND / 'cloud_leases/diagnostic_native_logit_v1'
SHARED = ('controller/', 'legacy/', 'ouro_project/', 'repo/', 'evaluation/artifacts.py', 'evaluation/bootstrap.py',
          'evaluation/download_model.py', 'evaluation/launch.py', 'evaluation/readouts.py')
AUTHORED = ('evaluation/validate_outputs.py', 'evaluation/worker.py')
GENERATED = ('frozen/output_contract.json', 'frozen/run_spec.json')
OUTSIDE = ('CONTRACT.md', 'RUNBOOK.md', 'build.py', 'observe_account.py', 'run_config.json')
ROLES = {'run_spec.json': 'frozen diagnostic run specification',
         'provenance.json': 'runtime and provenance record: precision dictionary, versions, GPU identity',
         'development/native.pt': 'M1: unchanged readouts.native_development tensors',
         'diagnostic/tensors.pt': 'M2 and M4 captured tensors; M3 recomputed target rows; produced dtypes',
         'diagnostic/comparisons.json': 'exact comparison record, recomputed by the verifier'}


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def record(path):
    require(path.is_file() and not path.is_symlink(), 'not a regular file: ' + str(path))
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def read(path):
    return json.loads(path.read_text())


def write(path, value, *, exclusive=False):
    with path.open('x' if exclusive else 'w') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def tree(root):
    return {p.relative_to(root).as_posix(): record(p) for p in sorted(root.rglob('*')) if not p.is_dir()}


def shared_names():
    """Shared files, checked against the precision_v1 freeze receipt."""
    parent = read(PARENT / 'FREEZE.json')['files']
    names = sorted(name for name in parent if name.startswith(SHARED))
    for name in names:
        require(record(PARENT / 'bundle' / name) == parent[name], 'precision_v1 differs from its freeze: ' + name)
    return names, parent


def checked_bundle():
    names, parent = shared_names()
    files = tree(BUNDLE)
    require(set(files) == {*names, *AUTHORED, *GENERATED}, 'bundle file set differs')
    require(all(files[name] == parent[name] for name in names), 'a shared file differs from precision_v1')
    return files


def build_bundle():
    names, parent = shared_names()
    for name in names:
        target = BUNDLE / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PARENT / 'bundle' / name, target)
    sources = {name: value for name, value in tree(BUNDLE).items() if not name.startswith('frozen/')}
    require(set(sources) == {*names, *AUTHORED} and all(sources[n] == parent[n] for n in names),
            'bundle sources differ from the shared and authored set')
    parent_spec = read(PARENT / 'bundle/frozen/run_spec.json')
    spec_path, contract_path = BUNDLE / 'frozen/run_spec.json', BUNDLE / 'frozen/output_contract.json'
    spec_path.parent.mkdir(exist_ok=True)
    write(spec_path, {'schema': 'native_logit_diagnostic_run.v1', 'run_id': RUN_ID,
                      'contract_record': record(HERE / 'CONTRACT.md'), 'parent_freeze_record': record(PARENT / 'FREEZE.json'),
                      'bank_records': {}, 'source_records': sources,
                      **{key: parent_spec[key] for key in ('development_item_names', 'model_identity',
                                                           'model_record', 'original_precision')}})
    write(contract_path, {'schema': 'confirmation_output_contract.v1', 'run_id': RUN_ID,
                          'run_spec_sha256': record(spec_path)['sha256'],
                          'files': {name: {'role': role} for name, role in ROLES.items()},
                          'stages': {}, 'required_checks': ['loadability', 'numerical']})
    verifier = BUNDLE / 'evaluation/validate_outputs.py'
    write(HERE / 'run_config.json', {
        'schema': 'confirmation_run_config.v1', 'run_id': RUN_ID,
        'run_spec_path': str(spec_path), 'output_contract_path': str(contract_path),
        'semantic_verifier': {'path': str(verifier), 'record': record(verifier), 'callable': 'validate_outputs'},
        'setup_budget_seconds': 2400, 'compute_budget_seconds': 1200,
        'preservation_reserve_seconds': 900, 'transfer_timeout_seconds': 600})
    checked_bundle()
    print(json.dumps({'status': 'bundle_built', 'files': len(tree(BUNDLE))}))


def archive_members():
    with tarfile.open(ARCHIVE, 'r:gz') as archive:
        members = archive.getmembers()
        require(all(m.isfile() and m.name.startswith('bundle/') for m in members), 'archive has a non-file member')
        files = {m.name.removeprefix('bundle/'): {'bytes': m.size,
                                                 'sha256': hashlib.sha256(archive.extractfile(m).read()).hexdigest()}
                 for m in members}
    require(len(files) == len(members), 'archive has duplicate members')
    return files


def build_archive():
    files = checked_bundle()
    with tarfile.open(ARCHIVE, 'x:gz') as archive:
        for name in files:
            archive.add(BUNDLE / name, arcname='bundle/' + name, recursive=False)
    require(archive_members() == files, 'archive members differ from the bundle directory')
    print(json.dumps({'status': 'archived', 'bundle': record(ARCHIVE)}))


def evidence():
    return {p.relative_to(HERE).as_posix(): record(p)
            for directory in ('tests', 'evidence') for p in sorted((HERE / directory).glob('*')) if not p.is_dir()}


def freeze():
    files = checked_bundle()
    require(archive_members() == files, 'archive members differ from the bundle directory')
    for name in evidence():
        if name.endswith('.json'):
            tested = read(HERE / name).get('bundle_records')
            require(tested in (None, files), 'evidence was produced against another bundle: ' + name)
    write(RECEIPT, {'schema': 'native_logit_diagnostic_freeze.v1', 'run_id': RUN_ID,
                    'frozen_utc': datetime.now(timezone.utc).isoformat(), 'job_cap_usd': JOB_CAP_USD,
                    'lease_root': str(LEASE_ROOT), 'parent_freeze': record(PARENT / 'FREEZE.json'),
                    'bundle': record(ARCHIVE), 'archive_members_match_directory': True, 'files': files,
                    'outside_bundle': {name: record(HERE / name) for name in OUTSIDE}, 'evidence': evidence()},
          exclusive=True)
    print(json.dumps({'status': 'frozen', 'receipt': record(RECEIPT)}))


def verify():
    receipt = read(RECEIPT)
    files = checked_bundle()
    require(receipt['files'] == files and archive_members() == files, 'bundle or archive differs from the receipt')
    require(receipt['bundle'] == record(ARCHIVE) and receipt['parent_freeze'] == record(PARENT / 'FREEZE.json'),
            'archive or parent freeze differs from the receipt')
    require(receipt['outside_bundle'] == {name: record(HERE / name) for name in OUTSIDE}, 'a frozen file differs')
    require(receipt['evidence'] == evidence(), 'evidence differs from the receipt')
    print(json.dumps({'status': 'verified', 'receipt': record(RECEIPT)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('bundle', 'archive', 'freeze', 'verify'))
    {'bundle': build_bundle, 'archive': build_archive, 'freeze': freeze, 'verify': verify}[parser.parse_args().command]()
