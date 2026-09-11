#!/usr/bin/env python3
"""Read-only account and pod observation through the pinned controller client.

No mutation, upload or worker contact; credentials are never printed. The read
calls are those of resources/observe_attempt05_final_account.py.
Usage: python -B observe_account.py --out NEW.json [--pod-id POD_ID]
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / 'bundle/controller'))
import runpod_api as api  # noqa: E402


def now():
    return datetime.now(timezone.utc).isoformat()


def observe(call):
    try:
        value = {'status': 'observed', 'value': call()}
    except api.APIError as error:
        value = {'status': 'error', 'error': str(error), 'http_status': error.status}
    return {**value, 'observed_utc': now()}


def pod_lookup(pod_id):
    try:
        return {'lookup': 'listed', 'pod': api.pod_v2(pod_id)}
    except api.APIError as error:
        if error.status == 404:
            return {'lookup': 'not_found_http_404'}
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pod-id')
    args = parser.parse_args()
    calls = {'account': lambda: api.gql('{ myself { id clientBalance currentSpendPerHr } }')['myself'], 'pods': api.my_pods}
    if args.pod_id:
        calls.update(pod=lambda: pod_lookup(args.pod_id), billing=lambda: api.billing_pods(pod_id=args.pod_id, last_n=7))
    result = {'schema': 'native_logit_account_observation.v1', 'started_utc': now(), 'pod_id': args.pod_id,
              'scope': 'Read-only account, pod-list and optional single-pod and pod-billing observation.',
              'observations': {name: observe(call) for name, call in calls.items()}}
    result['finished_utc'] = now()
    with args.out.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(result['observations'], indent=2))


if __name__ == '__main__':
    main()
