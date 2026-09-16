# J-Lens in Ouro

The Jacobian lens implementation in this repository is Anthropic's reference code, released under Apache-2.0, and their original README is preserved unchanged in [`README-upstream.md`](README-upstream.md). My contribution is the Ouro study in [`research/`](research/).

Apart from `research/`, `docs/ouro-application/`, this README and some added `.gitignore` entries, every file is as Anthropic released it; the `main` branch holds their initial release for comparison.

## The study

I compare the J-Lens with the raw logit lens across the four recurrent passes of frozen Ouro-2.6B, which applies the same 48 blocks in each pass. The checkpoint is the base model (`ByteDance/Ouro-2.6B`, revision `1ed04250da1a9936042725d302e81c8fa2ab5abd`), not the Thinking or RLTT checkpoint.

The task is two-hop questions whose intermediate concept is implied but never named. The score is excess hit@10: whether the intended concept ranks in the top ten tokens of the full vocabulary at a given pass and layer, minus the same hit rate for control concepts. Layerwise discovery on earlier questions found a positive J-Lens region in pass 4, and five independent calibration fits reproduced it. I then froze one estimator (fit01) and the band before testing on new questions.

## Results

- Pass 4, physical layers 26–37, with the estimator and band frozen before testing 160 new two-hop questions with 80 unseen intermediate concepts: +23.19 points excess hit@10, 95% dependency-group bootstrap interval 16.29–32.90. Intended-concept recovery is 37.92% against 14.43% for the raw lens.
- The same band in passes 1–3, with the final-target lens: −6.50 / −8.64 / −11.14 points.
- Lenses targeting each pass's own exit beat the final-target lens by +22.60 / +26.50 / +34.17 points and the raw lens by +16.41 / +18.32 / +23.18 points. All six simultaneous 95% intervals exclude zero.
- The early deficit belongs to the estimator's target, not to the passes.

Only the first result is prospective. The passes 1–3 and own-exit comparisons are post-confirmation analyses on the same states, each with its plan fixed before computation. The own-exit lenses come from the initial study and are compared with the final-target lens of their own family.

This is a localized, estimator- and target-specific improvement in concept recovery. It does not show that the J-Lens is better in general, that answers improve, or that the model uses the recovered content.

## Limitations

- **No tuned-lens baseline.** This is the first limitation. The comparison covers two training-free readouts and no learned one, so it does not show whether a data-trained map recovers the same intermediates.
- **Strong dependence on the estimator.** Changing the derivative target or the position aggregation substantially reduces the late-pass advantage.
- **The Huginn pilot was negative.** J-Lens minus raw was negative under all six prespecified summaries, and the pilot's fitted estimator was lost.
- **The freeze was local.** The estimator, band, scoring and analysis were fixed and timestamped before any new-question outcome, but this was not a public preregistration.
- **Few effective dependency groups.** The 160 questions form 28 dependency groups of unequal size; the effective group count is about 8.6.

## Paper

"J-Lens in Ouro: A Confirmed Late-Pass Band and a Target-Dependent Early-Pass Deficit" (arXiv listing pending). The current build is [`research/closeout_2026-09-11/manuscript/JLens_Ouro_Third_Draft.pdf`](research/closeout_2026-09-11/manuscript/JLens_Ouro_Third_Draft.pdf).

## What is in `research/`

Each directory is one round of work, kept as it was recorded; corrections go into later directories.

| Directory | Contents | Start at |
|---|---|---|
| [`followup_2026-09-07/`](research/followup_2026-09-07/) | Discovery analysis on the application-era outputs: layerwise J-Lens and logit-lens curves in all four passes, exit agreement, correctness strata, probe audit | `REPORT.md` |
| [`refit_round_2026-09-07/`](research/refit_round_2026-09-07/) | Five independent calibration fits, derivative-target and position controls, the Huginn pilot, and the retrieval incident in which nine of sixteen estimator files were not recovered in full | `REPORT.md` |
| [`confirmation_2026-09-09/`](research/confirmation_2026-09-09/) | The frozen confirmation: prospective plan, 160-question benchmark and dependency audit, freeze receipts, evaluation and analysis code, accepted readouts, reviews, spending | `README.md` |
| [`verification_2026-09-11/`](research/verification_2026-09-11/) | Post-confirmation numerical checks (batch packing, FP32/FP64 precision, regenerated states, tie-order and logit-perturbation bounds), the repaired report and the claim-change log | `FINAL_VERIFICATION.md` |
| [`closeout_2026-09-11/`](research/closeout_2026-09-11/) | Reviewer checks (within-domain controls, input-overlap audit, same band in every pass, dependency groups), the own-exit comparison in `local_exit/confirmation_2026-09-12/`, methods completion, a source-to-manuscript number ledger, manuscript sources and builds, the writer handoff, archive receipts | `PLAN.md` |

[`research/GITHUB_EXCLUSIONS.md`](research/GITHUB_EXCLUSIONS.md) lists the 8,423 files (37.52 GB) left out of GitHub, each with its SHA-256: large lens banks, activation caches, replay arrays and model weights, same-disk duplicates, and secrets. [`docs/ouro-application/`](docs/ouro-application/) holds the earlier application-stage write-up and records the study started from.

## Checking the numbers

Both scripts need only NumPy and run in seconds:

```bash
python research/closeout_2026-09-11/local_exit/confirmation_2026-09-12/verify_own_exit_family.py
python research/closeout_2026-09-11/handoff/JLens_Ouro_Writer_Handoff_v2/verify_handoff.py
```

The first recomputes the six own-exit contrasts and their simultaneous intervals from the stored rank arrays. The second rebuilds the primary result and the twenty secondary contrasts with the frozen analysis code. In a clone it also reports one missing file, `08_code/frozen_jlens/data/slice_vis.html`, which the upstream `.gitignore` keeps out of the repository, and exits non-zero for that reason alone.

## License

Anthropic's code and data are under the Apache License 2.0 ([`LICENSE`](LICENSE)). Code under `research/` is released under the same license unless a file states otherwise.
