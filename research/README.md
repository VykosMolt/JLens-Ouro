# Research records: J-Lens versus the raw logit lens in Ouro-2.6B

This directory holds the complete small-file record of the post-application J-Lens/Ouro study (September 2026), added on top of the unmodified Anthropic reference implementation in the repository root (`jlens/`, Apache-2.0). The research records are Jan Kirin's; code under `research/` is released under the same Apache-2.0 licence unless a file states otherwise.

| Directory | Round | Start here |
|---|---|---|
| `followup_2026-09-07/` | Post-submission discovery analysis on the retained application-era outputs (layerwise curves, exit agreement, correctness strata, probe audit) | `REPORT.md`, `CLAIMS.md` |
| `refit_round_2026-09-07/` | Five independent N100 calibration fits, target and position controls, Huginn pilot, retrieval incident | `REPORT.md`, `CLAIMS.md`, `analysis/` |
| `confirmation_2026-09-09/` | Frozen 160-item / 80-concept prospective confirmation (plan, benchmark, freeze receipts, accepted payload readouts, analysis) | `README.md`, `PROSPECTIVE_PLAN.md`, `REPORT.md` |
| `verification_2026-09-11/` | Post-confirmation numerical verification, repaired report, claim-change log, preservation ledger | `FINAL_VERIFICATION.md`, `report/REPORT_confirmation_verified.md` |
| `closeout_2026-09-11/` | Off-device archive receipt, reviewer checks A–D, local-exit inventory and blocker, methods completion, manuscript v2 with figures, writer handoff, numerical source ledger, purge dry run | `PLAN.md`, `handoff/JLens_Ouro_Writer_Handoff_v2/README_FIRST.md` |

**Headline result** (frozen before outcomes, 160 new questions): in pass 4, physical layers 26–37, the frozen J-Lens fit01 exceeds the raw logit lens by +0.23188 excess hit@10 (95% dependency-group bootstrap interval [+0.16289, +0.32898]); the same band is negative in passes 1–3. See the manuscript in `closeout_2026-09-11/manuscript/`.

**What is not here.** Files ≥ 50 MiB (lens banks, activation-state caches, replay arrays, model weights), same-disk duplicate bundles, SSH keys and cloud-account observations are excluded; `GITHUB_EXCLUSIONS.md` lists every excluded file with its SHA-256. All of them are in the owner's external archive `JLENS_COLD_ARCHIVE_20260911` (receipt in `closeout_2026-09-11/archive/ACCEPTANCE_RECEIPT.json`). The four confirmation banks (SHA-256 in `verification_2026-09-11/RUN_SPECIFICATION.json`) are available from the author on request; there is no public release of them yet.

**Reproducing the tables.** `closeout_2026-09-11/handoff/JLens_Ouro_Writer_Handoff_v2/verify_handoff.py` rebuilds the primary and the twenty secondary contrasts from the included readout arrays with the frozen code (numpy only). Model-level replay needs the pinned environment (`confirmation_2026-09-09/corrections/native_check_v1/bundle/legacy/requirements.lock`) and a CUDA GPU.

Historical directories are immutable records; corrections are versioned in later directories rather than edited in place.
