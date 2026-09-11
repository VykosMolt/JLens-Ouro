# Ouro J-Lens follow-up, 7 September 2026

[Research report](REPORT.md) · [Claim table](CLAIMS.md) · [Bugs and concerns](BUGS_AND_CONCERNS.md) · [Note for Neel](NEEL_NOTE.md)

J-Lens's early multihop deficit survives the full layer comparison and correctness stratification. A favorable final-pass region remains tentative. Actual-exit comparisons show that lens disagreement greatly exceeds changes in the model's predictions. The probe's loop-1 deficit survives bounded checks, with a loop-3 qualification. Huginn's literature and feasibility were checked; no new model comparison was run.

| Plot | PNG | PDF |
|---|---|---|
| J-Lens and logit lens at every layer and loop | [View](results/layerwise_performance.png) | [Export](results/layerwise_performance.pdf) |
| Paired difference with pointwise/simultaneous intervals | [View](results/layerwise_difference.png) | [Export](results/layerwise_difference.pdf) |
| Lens agreements against actual exits and final output | [View](results/exit_agreement_matrix.png) | [Export](results/exit_agreement_matrix.pdf) |
| Actual exit-to-exit agreement at both positions | [View](results/actual_exit_agreement.png) | [Export](results/actual_exit_agreement.pdf) |
| Passing/failing performance | [View](results/correctness_stratified.png) | [Export](results/correctness_stratified.pdf) |
| Passing/failing paired differences by layer | [View](results/correctness_layerwise_difference.png) | [Export](results/correctness_layerwise_difference.pdf) |

[Reproduction guide](REPRODUCE.md) · [Chronological log](RESEARCH_LOG.md) · [Literature](literature.md) · [Huginn feasibility](huginn_feasibility.md) · [Artifact inventory](MANIFEST.json)

The report links complete CSV/JSON tables and independent checks. The submitted application and original artifacts were not modified.
