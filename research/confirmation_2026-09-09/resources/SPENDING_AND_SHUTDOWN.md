# Spending and worker shutdown

All seven new-round workers are provider-terminated: six confirmation attempts and the native-logit diagnostic. The confirmation run's [lease](../cloud_leases/attempt_06/LEASE.json) records verified termination at 2026-09-11T01:06:59Z; the controller deleted the pod automatically after accepting its results. A [final read-only observation](../corrections/native_check_v1/run/observation_final.json) lists no pods and zero hourly spending, and the pod lookup returned HTTP 404. No Huginn worker was launched.

| Scope | Conservative spending bound |
|---|---:|
| Historical round | $13.851844 |
| Primary attempt01 | $0.225884 |
| Primary attempt02 | $0.840481 |
| Primary attempt03 | $1.467093 |
| Primary attempt04 | $2.701677 |
| Primary attempt05 | $2.558589 |
| Native-logit diagnostic | $0.080620 |
| Confirmation run (attempt06) | $1.469640 |
| Combined | **$23.195828** |
| Existing combined ceiling | $25.00 |
| Remaining authorization | $1.804172 |

These are conservative bounds, not settled invoices, and account balance is separate from remaining authorization. For the confirmation run, the whole-account balance fell $1.433279 between its pre-creation and final observations, below the lease bound. Its pod billing record ($0.786277) probably lags: the GPU charge equals about 1.11 hours at $0.69/h, against 2.08 lease hours. The attempt05 pod billing record ($1.988154) lagged the same way. The [attempt05 record](ATTEMPT05_SPENDING_AND_SHUTDOWN.json) binds the first five leases and both of its observations.

Root explicitly authorized attempt05's teardown after its failure evidence was [preserved](ATTEMPT05_NATIVE_FAILURE_PRESERVED.json).

The remaining $1.804172 cannot admit the optional $6 Huginn run, whose frozen archive was also purged.
