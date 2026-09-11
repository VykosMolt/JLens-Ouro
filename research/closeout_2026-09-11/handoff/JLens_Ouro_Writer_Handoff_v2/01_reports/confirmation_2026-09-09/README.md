# Ouro new-item confirmation and artifact recovery

Opened 9 September 2026. Prior scientific record: [report](../refit_round_2026-09-07/REPORT.md), [claims](../refit_round_2026-09-07/CLAIMS.md), [incident](../refit_round_2026-09-07/monitoring/attempt_06/termination_incident/attempt06_termination_incident_review.json).

**Status: the prospective confirmation is complete. On 160 new questions, the preselected Ouro fit01 J-Lens exceeded the raw logit lens in the frozen loop-4 band by +0.232 excess hit@10 (95% dependency-group bootstrap interval +0.163 to +0.329), driven by higher intended-concept recovery.** Benchmark construction and annotation were blind to method outcomes, and the estimator, endpoint and analysis are those frozen before the run. Two separately versioned corrections were frozen before any new-item outcome and change no scientific input. precision_v1 removed redundant precision setters. native_check_v1 made the development check unembed the whole sequence, after a bounded diagnostic traced attempt05's failure to single-row lm_head arithmetic on this GPU. The independent numerical reconstruction matched all 231 compared quantities. All workers are terminated. The append-only [run log](RUN_LOG.md) preserves the sequence.

Current evidence:

- [Report](REPORT.md) and [claim table](CLAIMS.md).
- [Confirmation analysis](results/ouro_confirmation_20260911_fixed160_native1/analysis/analysis.json), [primary figure](results/ouro_confirmation_20260911_fixed160_native1/figures/primary_and_components.png) and [independent numerical check](results/ouro_confirmation_20260911_fixed160_native1/independent_numerical_check_native1.json).
- [Prospective plan](PROSPECTIVE_PLAN.md), [freeze receipt](FREEZE.json), [precision-correction freeze](corrections/precision_v1/FREEZE.json) and [native-check freeze](corrections/native_check_v1/FREEZE.json).
- [Native-logit diagnostic contract](diagnostics/native_logit_v1/CONTRACT.md), [independent reanalysis](reviews/diagnostic_native_logit_v1_reanalysis.md) and [attempt05 forensics](reviews/attempt05_native_failure_forensics.md).
- [Artifact recovery findings](artifacts/FINDINGS.md), [sixteen-binary ledger](artifacts/RECOVERY_LEDGER.json), and [upstream/downstream supplement](artifacts/RETAINED_INPUTS_FINDINGS.md).
- [Independent benchmark review](reviews/benchmark_review.md) and [precision/dependence assessment](reviews/precision_design_review.md).
- [Controller review](reviews/controller_independent_review.md) and [actual primary retry deployment](controller/DEPLOYED_PRIMARY_RETRY_02.json).
- [Spending and shutdown status](resources/SPENDING_AND_SHUTDOWN.md): combined conservative bound $23.195828 of $25; the final observation lists no pods.
- Storage purge manifests for [this round](PURGE_MANIFEST_2026-09-11.json), the [refit round](../refit_round_2026-09-07/PURGE_MANIFEST_2026-09-11.json) and `/home/moloch/ouro_project/artifacts/jlens/PURGE_MANIFEST_2026-09-11.json`: every file of at least 50 MiB in those directories was deleted except one verified copy of each of the four confirmation banks.
- [Optional Huginn verification plan](huginn_verification/VERIFICATION_PLAN.md): not run; the remaining budget cannot admit it and the purge deleted its frozen archive.

The strongest design limitation is substantial grouping (28 dependency groups, effective count 8.6) and a relation mixture that differs from discovery; the frozen plan states this before outcomes.
