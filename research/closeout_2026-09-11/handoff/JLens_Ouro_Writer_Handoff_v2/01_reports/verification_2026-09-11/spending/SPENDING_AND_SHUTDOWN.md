# Spending and shutdown: final verification pass

No paid computation ran in this pass, and no provider resource was created, changed or deleted.

| Record | Value |
|---|---|
| Conservative combined bound before the pass | $23.195828: historical debit $13.851844 plus seven new-round lease bounds ([prior debit](../../confirmation_2026-09-09/resources/prior_debit.json), `cloud_leases/*/LEASE.json`) |
| Spent in this pass | $0 |
| Bound after the pass | $23.195828 of $25; $1.804172 remains under that accounting |
| [Start observation](observation_start.json), 08:24:03Z | balance $4.3725561096, no pods, zero hourly spend |
| [End observation](observation_end.json), 13:34:23Z | balance $4.3725561096, no pods, zero hourly spend |

These bounds are not invoices, and the account balance is not spending authorization.

**Prepared but not run: an RTX 5090 scoring-shape run.**
- **Inputs:** `../gpu_shape_v1/inputs`, 288 MB, with no bank.
- **What it would measure:** native output for the 160 items, and the 160-, 190-, 191-, 192-row and single-row layouts, on the confirmation GPU type.
- **Cost:** the unchanged controller would admit it at $1.150788. That covers 2,400 s setup, 1,200 s compute, 900 s preservation and 600 s termination at $0.706438/h, plus $0.15, within the remaining $1.804172.
- **Why it was not run:** the saved RTX 5090 outputs already give the numerical question a measured answer (see [FINAL_VERIFICATION.md](../FINAL_VERIFICATION.md)). The upload would also need destination-specific export approval for a new pod.
- **Status of the pod worker:** `gpu_shape_v1/bundle/evaluation/worker.py` has not been executed or frozen. Its measurement module ran locally through `run_local.py`.
