#!/usr/bin/env python3
"""A2: admission by the unchanged controller's offline plan against the current ledger.

The controller runs from a byte-identical temporary copy with -B. `plan` only
reads; the ledger tree is hashed before and after to show it. The balance is the
last recorded read-only observation; create re-reads the live balance.
Usage: python -B tests/admission_plan.py --work EMPTY_DIRECTORY
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from common import DIAGNOSTIC, LEDGER, ROUND, empty_directory, record, require, save_evidence, tree

JOB_CAP_USD = 1.5
BALANCE_SOURCE = ROUND / 'resources/attempt05_final_account_observation.json'
PRIOR_DEBIT = ROUND / 'resources/prior_debit.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    work = empty_directory(parser.parse_args().work)
    pins = json.loads((LEDGER / 'attempt_05/LEASE.json').read_text())['controller_sources']
    for name in pins:
        shutil.copyfile(DIAGNOSTIC / 'bundle/controller' / name, work / name)
        require(record(work / name) == pins[name], 'controller copy differs from the attempt05 pin: ' + name)
    balance = json.loads(BALANCE_SOURCE.read_text())['observations']['account']['value']['clientBalance']
    debit = json.loads(PRIOR_DEBIT.read_text())
    command = [sys.executable, '-B', str(work / 'lease.py'), 'plan', '--bundle', str(DIAGNOSTIC / 'bundle.tar.gz'),
               '--balance', repr(balance), '--run-config', str(DIAGNOSTIC / 'run_config.json'),
               '--prior-debit', str(PRIOR_DEBIT), '--job-cap-usd', repr(JOB_CAP_USD)]
    before = tree(LEDGER)
    plan = json.loads(subprocess.run(command, capture_output=True, text=True, check=True, cwd=work).stdout)
    require(tree(LEDGER) == before, 'plan changed the ledger')
    seconds = math.ceil(plan['setup_budget_seconds'] + plan['compute_budget_seconds'] + plan['preservation_reserve_seconds'] + 600)
    bound = seconds / 3600 * plan['all_in_rate'] + 0.15
    remaining = 25.0 - plan['prior_spend_upper_usd']
    minimum_balance = JOB_CAP_USD + debit['minimum_balance_to_preserve_usd']
    require(JOB_CAP_USD <= 1.5 and bound <= min(remaining, JOB_CAP_USD) and balance >= minimum_balance, 'admission arithmetic fails')
    save_evidence('A2_admission_plan.json', {
        'schema': 'native_logit_admission_plan.v1', 'status': 'admitted', 'command': command,
        'controller_records': pins, 'bundle_record': record(DIAGNOSTIC / 'bundle.tar.gz'),
        'run_config_record': record(DIAGNOSTIC / 'run_config.json'), 'prior_debit_record': record(PRIOR_DEBIT),
        'balance_source': {'path': str(BALANCE_SOURCE), **record(BALANCE_SOURCE)}, 'balance_usd': balance,
        'job_cap_usd': JOB_CAP_USD, 'prior_spend_upper_usd': plan['prior_spend_upper_usd'], 'remaining_authorized_usd': remaining,
        'admission_seconds': seconds, 'admission_bound_usd': bound, 'minimum_balance_usd': minimum_balance,
        'ledger_unchanged': True, 'ledger_entries_hashed': len(before), 'plan': plan})
    print(json.dumps({'status': 'admitted', 'admission_bound_usd': bound, 'remaining_authorized_usd': remaining}))


if __name__ == '__main__':
    main()
