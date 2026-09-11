# Native-logit diagnostic v1: paid-run runbook

Root runs this only after A7 (independent verification and adversarial review) has passed. Steps 1 to 10 run in order. Any failure goes to the matching branch in [Failure branches](#failure-branches). Never retry a paid failure without new user approval.

Never run `parallel_upload.py` or `launch.py --phase accept-development`, and never point a `lease.py` command at `attempt_01` to `attempt_05`.

## Shell setup

```sh
set -o pipefail
R=/home/moloch/jacobian-lens/research/confirmation_2026-09-09
D=$R/diagnostics/native_logit_v1
PY=/home/moloch/ouro_project/venv/bin/python   # torch-capable; create gives the watcher and verifier this interpreter
CTL=$D/bundle/controller                        # byte-identical to the six attempt05 pins
LEASE=$R/cloud_leases/diagnostic_native_logit_v1
RUN=$D/run
KEY=$R/resources/ssh/id_ed25519
export PYTHONDONTWRITEBYTECODE=1
field() { $PY -B -c 'import json,sys; s=json.load(open(sys.argv[1])); print(json.dumps({k: s.get(k) for k in sys.argv[2:]}, indent=1))' $LEASE/LEASE.json "$@"; }
```

## 0. Offline preflight

```sh
mkdir $RUN
$PY -B $D/build.py verify
$PY -B $D/tests/inventory.py write $RUN/inventory_before.json
```

`verify` must print `"status": "verified"`.

## 1. Fresh read-only observation

```sh
$PY -B $D/observe_account.py --out $RUN/observation_precreate.json
```

Continue only if all of the following hold:
- the account `id` is `user_3I33u3QLWgO7POHKINt6GfLeTUw`;
- `pods` is empty;
- `currentSpendPerHr` is 0;
- `clientBalance` is at least 4.0286755391453752, which is the $1.50 cap plus the 2.5286755391453752 floor.

Otherwise go to F1.

## 2. User-approval checkpoint

Root sends the user:
- the `FREEZE.json` SHA-256;
- one community RTX 5090 at up to $0.69/h, which is $0.7064384/h all-in;
- a job cap of $1.50 within the unchanged $25 cap, against a prior bound of $21.645567963766442;
- an admission bound of $1.150788 for 5,100 s;
- export of `bundle.tar.gz` to the provider-assigned pod address; the archive holds no bank, population, benchmark or confirmation prompt;
- the three historical development items only;
- no retry after a paid failure.

Root asks for explicit approval of both creation and that export, then records the user's exact words:

```sh
$PY -B -c 'import json,sys; from datetime import datetime,timezone; open(sys.argv[1],"x").write(json.dumps({"schema":"native_logit_user_approval.v1","recorded_utc":datetime.now(timezone.utc).isoformat(),"user_words":sys.argv[2],"freeze_sha256":sys.argv[3],"scope":"Create one community RTX 5090 lease for ouro_native_logit_diagnostic_v1, job cap $1.50 within the $25 cap, and export bundle.tar.gz to its provider-assigned address."},indent=2)+"\n")' \
  $RUN/USER_APPROVAL.json "<exact user words>" "$(sha256sum $D/FREEZE.json | cut -d' ' -f1)"
```

Denied, conditional or missing approval: go to F2.

## 3. Plan

```sh
BALANCE=<clientBalance from step 1>
$PY -B $CTL/lease.py plan --bundle $D/bundle.tar.gz --balance $BALANCE --run-config $D/run_config.json \
  --prior-debit $R/resources/prior_debit.json --job-cap-usd 1.50 > $RUN/admission_plan.json
```

A nonzero exit means admission failed: go to F3.

## 4. Create within the notice window

Tell the user creation is starting, then run both commands at once. `create` requires the notice to be at most 300 s old, both at start and again after watcher readiness, which can take up to 180 s. Start within 60 s of `NOTICE`.

```sh
NOTICE=$(date -u +%Y-%m-%dT%H:%M:%SZ); echo $NOTICE > $RUN/notice_utc.txt
$PY -B $CTL/lease.py create --bundle $D/bundle.tar.gz --root $LEASE --notice-utc $NOTICE \
  --public-key $KEY.pub --ssh-identity $KEY --run-config $D/run_config.json \
  --prior-debit $R/resources/prior_debit.json --job-cap-usd 1.50 2>&1 | tee $RUN/create.log
```

Success prints `pod_id` and `work_deadline_utc`. The detached watcher is a systemd user unit named after the lease. Any error: go to F4.

## 5. Bundle upload

Wait until the watcher has bound SSH:

```sh
field status pod_id ssh setup_deadline_utc halt_requested
```

Once `ssh` is set:

```sh
HOST=<ssh.host>; PORT=<ssh.port>; POD=<pod_id>
SSH="ssh -i $KEY -p $PORT -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$LEASE/known_hosts root@$HOST"
```

The unchanged `launch.py --phase start` in step 6 performs the upload itself:
- it checks `provider_lease_name` over SSH;
- it rsyncs `bundle.tar.gz` and `worker_config.json`;
- it extracts the bundle;
- because `bank_records` is `{}`, it uploads no bank.

## 6. Launch

```sh
$PY -B $D/bundle/evaluation/launch.py --lease $LEASE --phase start 2>&1 | tee -a $RUN/launch.log
```

Success ends with `"status": "launched"`. Bootstrap then installs the pinned packages, downloads the pinned model and receives `BANKS_READY` with `bank_records: {}`. The worker then runs M0 to M4.

If launch fails:
- **`$LEASE/launch/BOOTSTRAP_STARTED.json` exists:** rerun the same command. It skips the upload and republishes `BANKS_READY`.
- **It does not exist:** run `$SSH test -e /workspace/jlens/bundle`.
  - Exit 0: the bundle was extracted without a recorded start, and the unchanged launcher cannot resume. Go to F5.
  - Otherwise, for example sshd is not yet up after apt: rerun the command while more than 60 s of setup remains, then go to F5.

## 7. Watch and sync

The watcher polls every 15 s. It latches setup completion, starts `sync` and records the final manifest.

```sh
systemctl --user is-active $(field name | $PY -B -c 'import json,sys; print(json.load(sys.stdin)["name"])')
field last_worker_status setup_completion computation_finished retrieval_in_progress retrieval_accepted \
  retrieval_last_error watch_last_error_type observed_combined_spend_upper_usd
```

Expected worker phases, in order:
- the setup phases `venv`, `packages`, `model`, `awaiting_verified_bank_uploads` and `setup_complete`;
- `load_model`;
- `native_logit_measurements`;
- `computation_finished`, with outcome `complete`.

Deadlines count from `billing_start_utc`:

| Deadline | Offset |
|---|---|
| Setup | +2,400 s |
| Work | +3,600 s |
| Spending watch | +4,500 s |
| Provider | +5,100 s |

A manual retrieval is always safe:

```sh
$PY -B $CTL/lease.py sync --root $LEASE
```

## 8. Acceptance and automatic termination

For a complete final manifest, `sync` does three things:
- copies the five files to `$LEASE/handoff/accepted/ouro_native_logit_diagnostic_v1/<manifest sha256>/results/`;
- runs the pinned verifier;
- records `retrieval_accepted`.

The watcher then deletes the pod with class `accepted_artifacts`. Confirm:

```sh
field computation_finished retrieval_accepted termination_class termination_authorization
```

The required values are:
- `computation_finished.outcome` is `complete`;
- `retrieval_accepted` is present;
- `termination_class` is `accepted_artifacts`.

## 9. Absence confirmation

```sh
field status absence_confirmations terminated_verified_utc lease_spend_upper_usd combined_spend_upper_usd
```

The required values are:
- `status` is `terminated`;
- `absence_confirmations` is 3;
- `lease_spend_upper_usd` is at most 1.50.

The watcher unit must be inactive. Otherwise go to F10.

## 10. Final read-only observation

```sh
$PY -B $D/observe_account.py --pod-id $POD --out $RUN/observation_final.json
$PY -B $D/tests/inventory.py write $RUN/inventory_after.json
$PY -B $D/tests/inventory.py compare $RUN/inventory_before.json $RUN/inventory_after.json \
  --allow-added cloud_leases/diagnostic_native_logit_v1/
```

The required results are:
- `pods` is empty;
- `currentSpendPerHr` is 0;
- the pod lookup returns `not_found_http_404`;
- `compare` prints `"status": "unchanged"`. Otherwise go to F11.

Then complete the contract's post-run acceptance, including the independent CPU reanalysis of the accepted payloads.

## Failure branches

Every branch ends with a report to the user.

| Branch | Condition | Action |
|---|---|---|
| F1 | Observation shows another account, a pod, hourly spend or too low a balance | Stop before creation. |
| F2 | Approval denied, conditional or missing | Stop before creation. |
| F3 | `plan` rejects | Admission failed. Stop before creation. |
| F4 | `create` raises | Never rerun `create`. See below. |
| F5 | Launch cannot complete: SSH is never reachable, or the bundle is extracted without `BOOTSTRAP_STARTED.json` | P if SSH works, then T. |
| F6 | Setup fails: worker phase `setup_failed` or `setup_stopped`, or the setup deadline passes without `setup_completion` | The watcher writes STOP but does not delete. P, then T. |
| F7 | Final outcome `failed` or `stopped`; this includes every M0 failure (sources, snapshot, imports, environment, precision) | `sync` stages the manifest under `handoff/staging/` and never accepts it. P, then T. |
| F8 | A complete final is rejected by the verifier (`retrieval_last_error`), or transfer keeps failing | Rerun `sync`. Change no verifier, frozen or controller file. If unresolved, P then T. Without root action, the watch-deadline emergency deletes at +4,500 s. |
| F9 | Budget emergency: watch deadline, spend at $1.35, or balance at the preserve level plus $0.15 | The watcher deletes automatically with class `budget_emergency`. Run steps 9 and 10. |
| F10 | `termination_unresolved_utc` is set, or absence is unconfirmed | Rerun T with the same document. The provider TTL (`terminateAfter`, +5,100 s) is the backstop; it has never been observed to fire. |
| F11 | `inventory.py compare` reports a removed, changed or unexpected entry | Invariant violation. Stop. |

F4 has two cases:
- **`status` is `create_rejected`** (provider supply constraint): the GPU type is unavailable. The watcher confirms that no pod exists and marks the lease terminated. Run steps 9 and 10.
- **Any other error**, such as `create_uncertain` or a price, geometry or binding error: the watcher reconciles by name.
  - No pod listed: it records absence.
  - A pod listed: nothing deletes it before the watch deadline, so run T now.

P preserves remote records before teardown; it only reads from the pod:

```sh
mkdir -p $RUN/remote_preserved
rsync -rt -e "ssh -i $KEY -p $PORT -o BatchMode=yes -o ConnectTimeout=15 -o UserKnownHostsFile=$LEASE/known_hosts" \
  root@$HOST:/workspace/jlens/{status.json,bootstrap_stdout.log,setup_logs,artifact_index.json,manifests} $RUN/remote_preserved/
$PY -B $CTL/lease.py sync --root $LEASE
```

Missing remote files are expected on early failures. Record an unreachable pod instead of retrying.

T is explicit root teardown. It needs a bound `pod_id`; without one, rely on the watcher's absence reconciliation.

```sh
SCOPE='<branch and reason, e.g. F7: worker final failed; staged evidence preserved>'
$PY -B -c 'import json,sys; sys.path.insert(0,sys.argv[1]); import artifact_handoff as h; s=json.load(open(sys.argv[2])); open(sys.argv[3],"x").write(json.dumps({"schema":"confirmation_explicit_teardown.v1","binding":h.binding(s),"scope":sys.argv[4],"authorized_by":"root"},indent=2)+"\n")' \
  $CTL $LEASE/LEASE.json $RUN/ROOT_TEARDOWN.json "$SCOPE"
$PY -B $CTL/lease.py terminate --root $LEASE --authorization $RUN/ROOT_TEARDOWN.json
```

After T, run steps 9 and 10.
