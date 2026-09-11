"""Read-only final account/pod observation after attempt05 termination; never prints or copies credentials."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LEASE_DIR = HERE.parent / 'cloud_leases/attempt_05'
LEASE_PATH = LEASE_DIR / 'LEASE.json'
CLIENT = LEASE_DIR / 'monitor/runpod_api.py'
RECORD = HERE / 'attempt05_final_account_observation.json'


def record(path):
    raw = Path(path).read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


lease_record = record(LEASE_PATH)
lease = json.loads(LEASE_PATH.read_text())
if lease.get('status') != 'terminated':
    raise SystemExit('attempt05 lease is not recorded as terminated')
pod_id = lease['pod_id']
client_record = record(CLIENT)
pinned = lease['controller_sources']['runpod_api.py']
if (client_record['bytes'], client_record['sha256']) != (pinned['bytes'], pinned['sha256']):
    raise SystemExit('attempt05 monitor runpod_api.py differs from its pinned record')

sys.dont_write_bytecode = True
sys.path.insert(0, str(CLIENT.parent))
import runpod_api as api  # noqa: E402

if Path(api.__file__).resolve() != CLIENT.resolve():
    raise SystemExit('imported an unexpected runpod_api module')


def pod_lookup():
    try:
        return {'lookup': 'listed', 'pod': api.pod_v2(pod_id)}
    except api.APIError as error:
        if error.status == 404:
            return {'lookup': 'not_found_http_404'}
        raise


out = {
    'schema': 'confirmation_attempt05_final_account_observation.v1',
    'started_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'Read-only account, pod-list, single-pod and pod-billing observation after attempt05 termination. '
             'No allocation, mutation, upload or worker contact.',
    'lease': {**lease_record, 'status': lease['status'], 'pod_id': pod_id,
              'terminated_verified_utc': lease['terminated_verified_utc'],
              'lease_spend_upper_usd': lease['lease_spend_upper_usd'],
              'combined_spend_upper_usd': lease['combined_spend_upper_usd']},
    'client_source': client_record,
    'observations': {},
}
for name, call in [
    ('account', lambda: api.gql('{ myself { id clientBalance currentSpendPerHr } }')['myself']),
    ('pods', api.my_pods),
    ('attempt05_pod', pod_lookup),
    ('attempt05_billing', lambda: api.billing_pods(pod_id=pod_id, last_n=7)),
]:
    try:
        value = {'status': 'observed', 'value': call()}
    except api.APIError as error:
        value = {'status': 'error', 'error': str(error), 'http_status': error.status}
    value['observed_utc'] = datetime.now(timezone.utc).isoformat()
    out['observations'][name] = value
out['finished_utc'] = datetime.now(timezone.utc).isoformat()

with RECORD.open('x') as handle:
    json.dump(out, handle, indent=2, sort_keys=True)
    handle.write('\n')
print(json.dumps({'record': record(RECORD), 'observations': out['observations']}, indent=2))
