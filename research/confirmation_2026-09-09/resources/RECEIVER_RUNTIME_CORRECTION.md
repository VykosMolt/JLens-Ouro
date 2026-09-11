# Receiver runtime correction

Attempt 03's active spending watchdog was started with `/usr/bin/python`, which
lacks PyTorch. Its unchanged `ensure_sync` inherits that interpreter. The
supplementary `receiver_watch.py` instead invokes the same attempt's pinned
`monitor/lease.py sync` with `/home/moloch/ouro_project/venv/bin/python`.

The existing spending watchdog remains active. The receiver acquires the existing
`.sync.lock` inside `sync`; accepted receipts retain the original numerical and
hash checks. The system-Python watchdog can verify those receipts and perform its
existing teardown without importing PyTorch. No controller, worker, bank, contract,
lease, budget, or deadline is changed by this correction.

The wrapper requires a root directly below this round's `cloud_leases` and an
explicit digest of the existing handoff binding. Every cycle checks that binding,
stable lease/input/deadline fields, and all six monitor source hashes. It compiles
the pinned `transport.py` bytes directly and uses its `run_bounded` process-group
cleanup for sync calls, including timeouts and service shutdown. Collection errors
are retried after 15 seconds; each invocation is bounded to 1,200 seconds. The
wrapper exits after observing `terminated` for the same verified lease. It never
creates or deletes a cloud resource, acknowledges development, or starts any SSH
operation except the unchanged collector. The original watchdog remains solely
responsible for spending limits and deadlines.

Run it as a separate transient user service with `Restart=on-failure`,
`RestartSec=15`, `KillMode=control-group`, and `TimeoutStopSec=30`. Do not restart or
replace the existing watcher. The wrapper's journal records invocation status and
process-group quiescence. A successful invocation alone is not artifact acceptance;
acceptance still requires the original receipt and validation gates.

Launch command for the currently verified attempt 03 binding (not executed while
preparing this correction):

```bash
systemd-run --user \
  --unit=jlens-confirm-58682ddbc5194900912577d31cd000fe-receiver --collect \
  --property=Restart=on-failure --property=RestartSec=15 \
  --property=KillMode=control-group --property=TimeoutStopSec=30 \
  /home/moloch/ouro_project/venv/bin/python -B \
  /home/moloch/jacobian-lens/research/confirmation_2026-09-09/resources/receiver_watch.py \
  --lease /home/moloch/jacobian-lens/research/confirmation_2026-09-09/cloud_leases/attempt_03 \
  --binding-sha256 c0fd230546c57ffb28f02a26664eab91e0ff0ba87427ff3eb25ff862dac2d895
```

Validation: explicit `py_compile` into a temporary directory passed. Offline
fixtures using the actual pinned monitor bytes verified the terminal exit and
rejection of an invalid root, changed binding, changed deadline, and changed
monitor source. All fixture paths forbade subprocess creation. No cloud operation
or live sync was performed, and the service was not launched by the implementer.
