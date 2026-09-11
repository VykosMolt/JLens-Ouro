"""Record attempt05 spending/shutdown reconciliation from existing local records; no provider or worker calls."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
OUT = HERE / 'ATTEMPT05_SPENDING_AND_SHUTDOWN.json'
CAP = 25.0
PRIMARY_ATTEMPT_CAP = 3.77
HUGINN_ADMISSION = 6.0


def record(path):
    raw = Path(path).read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def require(condition, message):
    if not condition:
        raise SystemExit(message)


prior_path = HERE / 'ATTEMPT04_SPENDING_AND_SHUTDOWN.json'
prior = json.loads(prior_path.read_text())
attempts = []
for entry in prior['new_round_attempts']:
    current = record(entry['record']['path'])
    require((current['bytes'], current['sha256']) == (entry['record']['bytes'], entry['record']['sha256']),
            f"attempt{entry['attempt']} lease changed since the attempt04 reconciliation")
    attempts.append(entry)

lease_path = ROUND / 'cloud_leases/attempt_05/LEASE.json'
lease = json.loads(lease_path.read_text())
require(lease['status'] == 'terminated' and lease['absence_confirmations'] >= 3, 'attempt05 termination is not verified')
require(lease['stage_acceptances'] == {}, 'attempt05 unexpectedly records stage acceptances')
require(lease['prior_spend_upper_usd'] == prior['combined_spend_upper_usd'], 'attempt05 prior bound differs from the attempt04 reconciliation')
attempts.append({'attempt': 5, 'pod_id': lease['pod_id'], 'status': lease['status'],
                 'lease_spend_upper_usd': lease['lease_spend_upper_usd'], 'record': record(lease_path)})
historical = prior['historical_spend_upper_usd']
combined = lease['combined_spend_upper_usd']
require(abs(historical + sum(a['lease_spend_upper_usd'] for a in attempts) - combined) < 1e-9,
        'historical plus attempt bounds differ from the attempt05 combined bound')
remaining = CAP - combined

observation_path = HERE / 'attempt05_final_account_observation.json'
observation = json.loads(observation_path.read_text())
obs = observation['observations']
require(all(value['status'] == 'observed' for value in obs.values()), 'final observation is incomplete')
require(observation['lease']['sha256'] == attempts[-1]['record']['sha256'], 'final observation bound a different lease state')
require(obs['pods']['value'] == [] and obs['account']['value']['currentSpendPerHr'] == 0, 'final observation shows active resources')
require(obs['attempt05_pod']['value'] == {'lookup': 'not_found_http_404'}, 'attempt05 pod is still visible')
billing = obs['attempt05_billing']['value']
require(billing['metadata']['query']['podId'] == lease['pod_id'] and billing['metadata']['uniquePodCount'] == 1,
        'billing observation is not specific to the attempt05 pod')
precreate_path = HERE / 'attempt05_precreate_account_observation.json'
precreate = json.loads(precreate_path.read_text())

staging = 'cloud_leases/attempt_05/handoff/staging/ouro_confirmation_20260910_fixed160_precision1'
evidence = {name: record(ROUND / rel) for name, rel in {
    'native_failure_preserved': 'resources/ATTEMPT05_NATIVE_FAILURE_PRESERVED.json',
    'native_failure_forensics': 'reviews/attempt05_native_failure_forensics.json',
    'failed_final_manifest': f"{staging}/{lease['computation_finished']['manifest_sha256']}/MANIFEST.json",
    'setup_deadline_amendment': 'cloud_leases/attempt_05/SETUP_DEADLINE_AMENDMENT.json',
    'setup_deadline_amendment_review': 'reviews/attempt05_deadline_amendment_review.json',
    'root_teardown_execution': 'cloud_leases/attempt_05/ROOT_TEARDOWN_EXECUTION.json',
    'final_observation_client': 'resources/observe_attempt05_final_account.py',
}.items()}

out = {
    'schema': 'confirmation_attempt05_spending_shutdown.v1',
    'recorded_utc': datetime.now(timezone.utc).isoformat(),
    'settled_invoice': False,
    'scientific_status': 'prospective_confirmation_inconclusive_not_evaluated',
    'scientific_status_reason': 'The frozen exact native-logit equality gate failed on historical development items before '
                                'development acceptance; no new-item state was cached or scored.',
    'new_round_attempts': attempts,
    'historical_spend_upper_usd': historical,
    'combined_spend_upper_usd': combined,
    'remaining_authorized_usd': remaining,
    'combined_cap_usd': CAP,
    'attempt05': {
        'lease_name': lease['name'],
        'pod_id': lease['pod_id'],
        'gpu_type_id': lease['gpu_type_id'],
        'all_in_rate_usd_per_hour': lease['all_in_rate'],
        'billing_start_utc': lease['billing_start_utc'],
        'deployment_response_utc': lease['deployment_response_utc'],
        'terminated_verified_utc': lease['terminated_verified_utc'],
        'termination_class': lease['termination_class'],
        'termination_reason': lease['termination_reason'],
        'absence_confirmations': lease['absence_confirmations'],
        'termination_authorization': lease['termination_authorization'],
        'job_incremental_cap_usd': lease['job_incremental_cap_usd'],
        'lease_spend_upper_usd': lease['lease_spend_upper_usd'],
        'failed_manifest_sha256': lease['computation_finished']['manifest_sha256'],
        'worker_error': lease['last_worker_status']['error'],
        'stage_acceptances': lease['stage_acceptances'],
    },
    'final_observation': {
        'record': record(observation_path),
        'account_observed_utc': obs['account']['observed_utc'],
        'account': obs['account']['value'],
        'pods': obs['pods']['value'],
        'attempt05_pod': obs['attempt05_pod']['value'],
        'attempt05_provider_billing_totals': billing['metadata']['totals'],
        'precreate_record': record(precreate_path),
        'precreate_balance_usd': precreate['account']['clientBalance'],
        'account_balance_decrease_since_precreate_usd':
            round(precreate['account']['clientBalance'] - obs['account']['value']['clientBalance'], 10),
        'note': 'Provider billing and account balance are observations, not settled invoices; '
                'whole-account balance changes are not assigned to this round.',
    },
    'admission': {
        'primary_attempt_cap_usd': PRIMARY_ATTEMPT_CAP,
        'primary_attempt_admissible': remaining >= PRIMARY_ATTEMPT_CAP,
        'primary_attempt_shortfall_usd': PRIMARY_ATTEMPT_CAP - remaining,
        'optional_huginn_admission_usd': HUGINN_ADMISSION,
        'optional_huginn_admissible': remaining >= HUGINN_ADMISSION,
        'optional_huginn_shortfall_usd': HUGINN_ADMISSION - remaining,
    },
    'prior_reconciliation': record(prior_path),
    'evidence': evidence,
}
with OUT.open('x') as handle:
    json.dump(out, handle, indent=2, sort_keys=True)
    handle.write('\n')
print(json.dumps({'record': record(OUT), 'combined_spend_upper_usd': combined,
                  'remaining_authorized_usd': remaining, 'admission': out['admission']}, indent=2))
