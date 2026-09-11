#!/usr/bin/env python3
"""Prepare the immutable H debit only after actual primary preservation/shutdown.

This helper is offline. Its snapshot must be a fresh account/pod observation;
actual create separately repeats live account/ownership/funding checks.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'controller'))
import lease


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary-lease-root', type=Path, required=True)
    parser.add_argument('--account-snapshot', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=HERE / 'prior_debit.json')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Immutable phase debit already exists: ' + str(args.out))
    with lease.locked(lease.PRIMARY_LEDGER, '.lease_creation.lock'), lease.locked(lease.LEDGER, '.lease_creation.lock'):
        value, proof = lease.prepare_phase_debit(args.primary_lease_root, args.account_snapshot)
        lease.io._new_json(args.out, value)
        proof_path = args.out.with_name(args.out.stem + '_preparation_proof.json')
        lease.io._new_json(proof_path, {'schema': 'huginn_debit_preparation_proof.v1', 'status': 'passed',
                                       'debit': {'path': str(args.out.resolve()), **lease.handoff.checked_record(args.out)},
                                       'helper': {'path': str(Path(__file__).resolve()), **lease.handoff.checked_record(__file__)},
                                       'primary_full_rehash': proof, 'no_cloud_calls': True})
    print(json.dumps({'status': 'prepared', 'debit': str(args.out), 'proof': str(proof_path),
                      'prior_spend_upper_usd': value['prior_spend_upper_usd'],
                      'remaining_authorized_upper_usd': value['remaining_authorized_upper_usd'],
                      'huginn_job_incremental_cap_usd': value['huginn_job_incremental_cap_usd']}))


if __name__ == '__main__':
    main()
