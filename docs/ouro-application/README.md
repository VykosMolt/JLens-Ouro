# Ouro JLens status

This directory is the write-up and operational record for a local
observational study of prompt-averaged Jacobian-lens readout in the fixed
Ouro-2.6B snapshot. It is not a completed mechanism study. A B300 run is a
separate evidence-collection campaign; it is not permission to start a pod.

## Current conclusion

The canonical report currently has status
`LOCAL_OBSERVATIONAL_ARITHMETIC_REPRODUCIBLE_LINEAGE_UNFROZEN`. The retained
arrays reproduce the local arithmetic table, including a relative deficit for
the final-exit average Jacobian lens on multihop loops 1–3 and arithmetic loop
1. The historical evaluation directories have no complete current
`evaluate.py` provenance record, so the fit size is not authenticated and no
80-prompt/current-generation claim is made. This is descriptive local
arithmetic, not a completed mechanism result.

The canonical report records the local instrumentation validation at
`SUPPORTED_CURRENT_VALIDATION`. The retained probe arrays and Base, Thinking,
and RLTT checkpoint measurements are earlier-source/local evidence; they are
not re-certified as current-source artifacts for this launch. They are outside
the new B300 paid-verifier contract and outside the application claim. No probe
or checkpoint regeneration is launch-critical. Mechanism,
estimator-independence, 1000-prompt robustness, matched-model generalisation,
and B300 validation/fit remain unresolved until the appropriate evidence is
actually produced.

## Current paid-run boundary

The code-enforced hard limits are a $5.00 minimum account reserve, $30.00
maximum spend, and a 12,600-second provider TTL. The worker deadline is fixed
900 seconds earlier than the provider deadline so it has a bounded shutdown
window. The paid defaults use these exact limits and reject a usable worker
window shorter than 10,800 seconds; the fitting-only measured floor is already
7,227 seconds before validation, evaluation, upload, and teardown.

Before a lease, the results repository must pass a write-read-list canary.
The HF credential is read from a permission-checked 0600 file and passed via a
protected, explicit token channel; it is not taken from ambient environment
state, command-line arguments, receipts, or durable lease state.

No pod may be created or started without explicit user signoff. Passing local
gates establishes readiness only.

## Authority

- `artifacts/jlens/final/analysis.json`: current machine-readable result and claim statuses.
- `artifacts/jlens/final/verification.json`: currently absent; the historical
  verifier result is quarantined because its required transport input cannot
  pass the current provenance gate.
- `artifacts/jlens/MANIFEST.json`: an older/incomplete manifest is present;
  it is not current paid-run authority. Custody remains blocked until every
  required current-generation artifact exists on a committed tree.
- `artifacts/jlens/final/fit_size.json` and `transport.json`: currently absent;
  the pre-hardening bytes are retained only in the historical quarantine.
- `artifacts/jlens/probe/cv_all648/summary.json`: retained earlier-source/local
  arithmetic-probe report (not current-source proof for the new claim).
- The retained probe and checkpoint files are earlier-source/local evidence,
  not current-source proof for a new B300 claim.
- `HANDOFF.md`: operational state and safe next action.
- `METHODS.md`: populations and estimands.
- `REPRODUCE.md`: validation and regeneration commands.
- `INCIDENT_2026-09-04.md`: paid-run postmortem.

The pre-repair Opus documents and peer notes are preserved byte-for-byte under `history/2026-09-04-opus/`. They are historical evidence, not current authority.
