# Manuscript change log: first draft (11 Sep 2026, sha256 c13da59b…) → second draft v2 (11 Sep 2026)

Material scientific changes (new evidence, all labeled post-confirmation or descriptive):
1. Added the same-band pass comparison (Table 3, Fig. 3): layers 26–37 show deficits of −6.50/−8.64/−11.14 points in passes 1–3 with a four-contrast simultaneous family; the abstract and conclusion now lead with the pass-dependent pattern at fixed depth.
2. Added the within-domain control result (+20.80 [9.13, 32.46]) and the finding that the lens recovers same-domain names about nine times more often than arbitrary controls; domain heterogeneity reported.
3. Added the exhaustive tokenized-input overlap audit: 0 intended overlaps, 6 control overlaps in 6 items, sensitivities 23.19 / 22.90.
4. New Section 6 on local-exit targets: bank inventory, precise blocker (banks remote-only), retrieval-first proposal, and the descriptive discovery-population result (local target beats final target and raw in layers 26–37 of passes 1–3). Labeled post hoc, descriptive, different population.
5. Interpretation (§7) now names target dependence as a contributing factor on the discovery population while keeping supervision, architecture and calibration entangled.

Documentation changes:
6. §2.1 and Appendix C give the executed fitting recipe (estimator, positions, precision, seeds, engine, GPU) and the control recipes from the frozen records; §2.2 gives the exact prompt/tokenization/readout rule with a real ID-selected example; §3.2 explains the dependency grouping and the 42-item group.
7. Five real figures from retained arrays replace the tables-only draft; discovery-selected and frozen regions are marked; fit variation and item/group uncertainty are distinguished; pointwise vs simultaneous stated in captions.
8. Discovery intervals added (crossed pointwise 10.96–26.99; simultaneous −0.86–38.29; component −7.49–44.92).
9. Removed the internal [S1–S4] drafting markers and the "Internal source records" bibliography; every number is now traced in `ledgers/SOURCE_TO_MANUSCRIPT_LEDGER.csv`. Added references [3] and [4] for the Huginn statements in Appendix A (these were previously unreferenced).
10. Artifact-availability paragraph updated with the actual loss counts and the external single-copy archive status; no repository release is claimed.
11. "FP64 reference computation" retained; "source of the measured gain" used for the intended/control decomposition; Huginn's limited status retained; contribution/AI statement added as Appendix D (draft for Jan).

Corrections of v1 statements:
12. v1 §3.1 said the discovery band's "simultaneous intervals" contained zero without values; v2 gives the values and which families. v1 §3.1 "between-fit standard deviation of 0.39 points" confirmed (fit SD 0.00394).
13. v1 §2.1 described the position controls as using "the same sampled target positions and all 2,048 derivative directions per paragraph"; v2 states the actual rule (one uniformly drawn valid position q per paragraph, seed 2026090801) and that sampled-sum uses no source-count divisor.
14. v1 said fits ran with an unspecified "new GPU/runtime configuration"; v2 names the RTX 5090 pod from the lease record (the budget note had proposed an RTX 4090).
15. v1 Appendix A Huginn paragraph cited only [S2]; v2 cites the primary Huginn papers for the training/coda claims.
16. Two double-rounding slips in v1 corrected: leave-one-group-out upper end 24.88 → 24.87 (0.248748), and the pass-1 any-layer lower bound −29.79 → −29.78 (−0.297847); both came from re-rounding the 5-decimal report values.
No historical record was edited; the first draft is preserved unchanged in the handoff.
