# JLens reproduction and verification

Run commands from the repository root. None of the commands in this document
creates a cloud resource.

## Fast mechanical suite

```bash
venv/bin/python -m pytest -q \
  utilities/tests/unit/test_ouro_jlens.py \
  utilities/tests/unit/test_jlens_rental.py \
  utilities/tests/unit/test_jlens_integrity.py \
  utilities/tests/unit/test_jlens_probe_chain.py \
  utilities/tests/unit/test_jlens_reports.py \
  utilities/tests/unit/test_peer3_jlens_audit.py \
  utilities/tests/unit/test_jlens_secondary_reports.py \
  utilities/tests/unit/test_jlens_atomic_validation.py \
  utilities/tests/unit/test_jlens_pod_safety.py \
  utilities/tests/unit/test_jlens_paid_relocation.py \
  utilities/tests/unit/test_jlens_publish_hardening.py \
  utilities/tests/unit/test_jlens_shell_hardening.py \
  utilities/tests/unit/test_jlens_recovery_command.py \
  utilities/tests/unit/test_jlens_token_channel.py

venv/bin/python -O -m pytest -q \
  utilities/tests/unit/test_ouro_jlens.py \
  utilities/tests/unit/test_jlens_rental.py \
  utilities/tests/unit/test_jlens_integrity.py \
  utilities/tests/unit/test_jlens_probe_chain.py \
  utilities/tests/unit/test_jlens_reports.py \
  utilities/tests/unit/test_peer3_jlens_audit.py \
  utilities/tests/unit/test_jlens_secondary_reports.py \
  utilities/tests/unit/test_jlens_atomic_validation.py \
  utilities/tests/unit/test_jlens_pod_safety.py \
  utilities/tests/unit/test_jlens_paid_relocation.py \
  utilities/tests/unit/test_jlens_publish_hardening.py \
  utilities/tests/unit/test_jlens_shell_hardening.py \
  utilities/tests/unit/test_jlens_recovery_command.py \
  utilities/tests/unit/test_jlens_token_channel.py

venv/bin/python -m pytest -q utilities/tests/unit/test_peer1_index_map.py
bash -n src/ouro_jlens/*.sh
venv/bin/python -m py_compile src/ouro_jlens/*.py
```

The last pytest file is CUDA/model-backed; it is not a CPU test.

## Current-source instrumentation validation

```bash
venv/bin/python src/ouro_jlens/validate.py --out artifacts/jlens/validation
```

Acceptance requires `numerical_pass: true`, human-loop sources
`[16,64,112,160]`, and a zero exit status. Bit-exact status is reported
separately.

## Probe and checkpoint evidence

The retained arithmetic probe and Base/Thinking/RLTT checkpoint files are
earlier-source/local evidence. They are not re-certified as current-source
artifacts for the B300 campaign, are excluded from the paid verifier and
application claim, and do not need to be regenerated for launch. No command in
this document asks for that rerun.

## Canonical reports and blocked historical products

```bash
venv/bin/python src/ouro_jlens/fitsize_report.py  # expected FAILED_INCOMPLETE

venv/bin/python src/ouro_jlens/transport_report.py \
  --lens 8=artifacts/jlens/lens/exit3/exit3_p0-8.pt \
  --lens 32=artifacts/jlens/lens/exit3/exit3_merged.pt \
  --lens 56=artifacts/jlens/lens/exit3/exit3_n56.pt \
  --lens 80=artifacts/jlens/lens/exit3/exit3_n80.pt \
  --out artifacts/jlens/final/transport.json  # expected provenance rejection

venv/bin/python src/ouro_jlens/report.py \
  --eval artifacts/jlens/eval/fitsize_n80 \
  --local-eval artifacts/jlens/eval/round1_exit3x32 \
  --validation artifacts/jlens/validation/milestones.json \
  --probe artifacts/jlens/probe/cv_all648/summary.json \
  --checkpoints artifacts/jlens/checkpoints/exit_divergence.json

venv/bin/python src/ouro_jlens/verify_artifacts.py  # blocked: no current transport
venv/bin/python src/ouro_jlens/manifest.py build    # blocked until all required outputs exist
venv/bin/python src/ouro_jlens/manifest.py verify  # no current manifest yet
```

The report invocation above is retained for local write-up inspection only.
Its probe/checkpoint inputs are earlier-source/local evidence and it does not
authorize or validate a B300 launch.

The verifier and custody commands must fail closed in the present state; their
pre-hardening outputs were quarantined. Once current paid inputs exist,
acceptance requires a `PASS` verifier and zero manifest mismatches. The paid
verifier intentionally does not require the retained probe or checkpoint
files. A manifest never upgrades the `UNFROZEN`, `INCONCLUSIVE`,
`NOT_RETAINED`, or `NOT_ESTABLISHED` scientific states.

## Historical custody

```bash
(cd docs/jlens/history/2026-09-04-opus && sha256sum -c SHA256SUMS)
```

The recovered pre-repair probe cache and score arrays are preserved under
`artifacts/jlens/probe/history/recovered-2026-09-04/`; current results do not
depend on those copies.

## Rental dry run

```bash
DRY_RUN=1 bash src/ouro_jlens/stage_upload.sh --dry-run
venv/bin/python src/ouro_jlens/pod.py create --dry-run --run-id offline-check
```

These are offline configuration checks. The stage command also refuses
uncommitted JLens source, tests, or documentation and verifies the resulting
archive twice. Do not remove `--dry-run` as part of reproduction. A real pod
must not be created without explicit user signoff after the independent
verification, review, and paid-readiness gates finish.

The hard code-enforced paid limits are a `$5.00` minimum account reserve,
`$30.00` maximum spend, `12,600` seconds of provider TTL, and a worker
deadline `900` seconds earlier than that provider deadline. Those are also the
paid defaults, yielding an 11,700-second worker window; configurations below
the declared 10,800-second end-to-end worker floor are rejected. Before leasing,
the results repository must pass a write-read-list canary. HF authentication
must use the protected non-ambient token channel from a permission-checked
0600 file; never put the token in a command argument or durable state.

The no-cost full-tensor recovery path is:

```bash
venv/bin/python src/ouro_jlens/pod.py recover --state <state-file>
```

Recovery is offline custody, not authorization to reuse the artifact namespace
for another paid attempt. The paid controller rejects distinct attempt and
artifact-lineage IDs until every attempt-variant output is independently
namespaced.

The normal successful controller path first retrieves the roughly 180 MB
evaluation/write-up bundle and validates it against the complete remote
receipt inventory. It does not block the application on the approximately
27.98 GB of duplicated lens tensors. `recover` retrieves those tensors later
and performs the independent local paid-verifier replay. A failed or uncertain
run still retrieves every available recovery shard immediately.

For an authorized paid run, remote process exit zero is only provisional. The
controller must also verify exact-pod termination, converge and validate the
complete remote receipt union, retrieve and rehash the application bundle,
match the schema-2 expected inventory and current attempt heartbeat, and
locally interpret the paid verifier/validation acceptance fields. Any missing
required file, stale listing, unknown receipt kind, unverified termination, or
custody write failure leaves a nonzero result or an explicit partial snapshot.
