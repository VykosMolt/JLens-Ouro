#!/usr/bin/env python3
"""Write PRESERVATION_LEDGER.json: purge reconciliation (from the independent audit), recovery attempts, bundle and restoration."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

V = Path(__file__).resolve().parents[1]
AUDIT = V / 'reviews/artifact_audit/ARTIFACT_AUDIT.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    audit = json.loads(AUDIT.read_text())
    core = json.loads((V / 'preservation/BUNDLE_MANIFEST_core.json').read_text())
    restoration = json.loads((V / 'preservation/RESTORATION_CHECK.json').read_text())
    keep = ('path', 'bytes', 'sha256', 'reason')
    ledger = {
        'schema': 'preservation_ledger.v1', 'created_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha256(__file__),
        'sources': {'artifact_audit': {'path': str(AUDIT), 'sha256': sha256(AUDIT)},
                    'purge_manifests': [str(Path(root) / 'PURGE_MANIFEST_2026-09-11.json') for root in audit['scope_roots']]},
        'class_labels': audit['class_labels'], 'classification_rules': audit['classification_rules'],
        'purge_reconciliation_by_root': audit['summary_by_root'], 'distinct_contents': audit['distinct_content_summary'],
        'unique_and_lost': [{k: f[k] for k in keep} for f in audit['files'] if f['class'] == 'c'],
        'unresolved': [{**{k: f[k] for k in keep}, 'remote_record': f.get('remote_record')} for f in audit['files'] if f['class'] == 'd'],
        'recovery_attempts_this_pass': [
            {'target': 'the 19 unresolved files recorded in private Hub repository Vykos/ouro-jlens-results',
             'outcome': 'not attempted: no Hugging Face credential on this machine (huggingface_hub.get_token() returned None)',
             'effect': 'classification unchanged'},
            {'target': 'unique-and-lost files', 'outcome': 'no intact copy or retained derivation exists; nothing recoverable locally',
             'effect': 'classification unchanged'}],
        'surviving_banks_and_arms': audit['bank_to_arm_mapping'], 'bank_load_check_this_pass': restoration['banks'],
        'evidence_levels_by_claim_audit': audit['evidence_levels_by_claim'],
        'historical_huginn_pilot': audit['huginn_historical_pilot'],
        'bundle': {'destination': core['destination'], 'core_manifest': {'path': str(V / 'preservation/BUNDLE_MANIFEST_core.json'),
                                                                        'sha256': sha256(V / 'preservation/BUNDLE_MANIFEST_core.json')},
                   'files': core['files'], 'bytes': core['bytes'], 'copy_problems': core['problems']},
        'restoration_check': {'path': str(V / 'preservation/RESTORATION_CHECK.json'), 'sha256': sha256(V / 'preservation/RESTORATION_CHECK.json'),
                              **{k: restoration[k] for k in ('rehash', 'frozen_code', 'rescoring', 'analysis_rerun_equals_saved_json')}},
        'single_copy_risk': ('Every surviving copy, including this bundle, the Hugging Face cache and the originals, is on one physical NVMe disk. '
                             'No second durable destination is available or authorized, so the bundle protects against accidental deletion, '
                             'not device loss. The model weights are also retrievable by pinned revision from the Hugging Face Hub.'),
        'storage': audit['storage'],
    }
    with (V / 'preservation/PRESERVATION_LEDGER.json').open('x') as handle:
        json.dump(ledger, handle, indent=1, sort_keys=True)
        handle.write('\n')
    print({'lost': len(ledger['unique_and_lost']), 'unresolved': len(ledger['unresolved']), 'bundle_files': core['files']})


if __name__ == '__main__':
    main()
