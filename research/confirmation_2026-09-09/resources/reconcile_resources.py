"""Read-only account reconciliation; never prints or copies credentials."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / 'research/refit_round_2026-09-07'
sys.path.insert(0, str(OLD / 'deployment'))
import runpod_api as api

out = {
    'schema': 'confirmation_resource_observation.v1',
    'started_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'Read-only live account, pod and billing observations. No allocation or mutation.',
    'previous_cap_usd': 25.0,
    'previous_controller_bound_usd': 13.851844033145376,
    'observations': {},
}
for name, call in [
    ('account', lambda: api.gql('{ myself { id clientBalance currentSpendPerHr } }')['myself']),
    ('pods', api.my_pods),
    ('billing', lambda: api.billing(last_n=7)),
    ('attempt06_billing', lambda: api.billing_pods(pod_id='vzwx0cj43g2uw5', last_n=7)),
    ('attempt03_billing', lambda: api.billing_pods(pod_id='1zdsbzba4m637l', last_n=7)),
    ('attempt04_billing', lambda: api.billing_pods(pod_id='3mnl58tg6nhk8c', last_n=7)),
]:
    try:
        value = {'status': 'observed', 'value': call()}
    except api.APIError as error:
        value = {'status': 'error', 'error': str(error)}
    value['observed_utc'] = datetime.now(timezone.utc).isoformat()
    out['observations'][name] = value
out['finished_utc'] = datetime.now(timezone.utc).isoformat()
out['client_source_sha256'] = hashlib.sha256(Path(api.__file__).read_bytes()).hexdigest()
path = Path(__file__).parent / ('account_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
with path.open('x') as handle:
    json.dump(out, handle, indent=2, sort_keys=True)
    handle.write('\n')
print(json.dumps({'record': str(path), 'observations': out['observations']}, indent=2))
