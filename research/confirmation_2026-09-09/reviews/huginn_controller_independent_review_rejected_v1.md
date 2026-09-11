# Huginn controller review: rejected pending retry-accounting correction

The supplied 51 cases, including the CPU preservation smoke test, pass independently. The exact candidate lease is 76,058 bytes, SHA-256 `5f600870bbf91d1d117b8e62fbc4f82df513394210486feda97862cc33f2c15e`. All primary controller and original debit pins remain unchanged. Five other runtime files and 45 existing lease functions match the primary baseline.

Three additional temporary-fixture checks expose a blocking gap in H `prior_debit`:

1. A terminated H row with two billable hours and a zero charge is accepted, omitting a minimum $1.4128767123 from the carried debit.
2. A row claiming termination while its create remains uncertain and its provider deadline is an hour away is accepted, even though the existing `absence_can_finish` predicate returns false.

3. After a valid prior H charge of $1, a new $5.98989041096 envelope is accepted, exceeding the root-clarified aggregate H ceiling of $6.

The H retry path needs the corresponding absence, elapsed/rate charge and prior/combined debit consistency checks already applied to primary prerequisites. The implementer and root acknowledged these findings. The candidate remains unaccepted until the focused correction passes all three regressions and the existing suite under new exact source pins.

The full primary acceptance rehash is correctly bracketed by identical payload inventories, and the immediate pre-mutation check binds the same inventory plus exact source, lease, receipt and validation records. The isolated process compiles pinned primary bytes under their original module locations. Those successful checks do not resolve the retry gap.

This review covers controller/resource admission. The H scientific gate and full-N100 timing check, actual output-transfer capacity, bundle and final receiver remain separate integration gates. The subsequent bound 64 MiB reverse eight-stream probe measured 3.545 MB/s with all part hashes verified. The actual bounded parallel preservation transport and full-size timing for the approximately 8.42 GB gate and 5.64 GB final handoffs still require a separate frozen integration before H admission.

No real primary or H record was changed, no provider action was performed, and no new primary method outcome was inspected. The JSON companions preserve the source records, complete independent 51-case proof and all three additional failure observations. No admitted H debit exists or is authorized by this review.
