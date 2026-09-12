# TMLR version of draft three

Built by `tmlr/build_tmlr.sh` from `JLens_Ouro_Third_Draft.md` with the official TMLR style files (`tmlr.sty`, `tmlr.bst`, Apache-2.0, from github.com/JmlrOrg/tmlr-style-file; licence in `LICENSE-tmlr-style`). pdfLaTeX + BibTeX, 15 pages.

## Outputs

- `JLens_Ouro_TMLR_submission.pdf` — double-blind version (`\usepackage{tmlr}`): "Anonymous authors", running head "Under review as submission to TMLR"; the availability sentence in Section 7.1 points to the supplementary material instead of the repository. This is the file to upload on OpenReview (TMLR takes the PDF, not the sources).
- `JLens_Ouro_TMLR_supplement.zip` — anonymous supplementary material for reviewers (84 MB; TMLR allows 100 MB, PDF or ZIP). Built by `make_supplement.py` from the writer handoff: frozen code, compact readouts, benchmark, freeze and plan records, reports of every round, reviewer checks, numerical verification, local-exit inventory and arrays, methods completion, source-to-manuscript ledger, figure data and figures, plus `README_SUPPLEMENT.md`, `MANIFEST.json` and `verify_supplement.py` (rehashes the package and rebuilds the primary and 20 secondary contrasts from the included readouts; needs numpy). Machine paths were rewritten and person/account/repository identifiers replaced by bracketed placeholders; the builder refuses to package if any identifying string survives, and it extracts the finished zip, runs the verifier and rescans. Operational records (run logs, budgets, spending, purge manifests, cloud leases, incident reviews, archive receipts) are not included.
- `JLens_Ouro_TMLR_preprint.pdf` — named version (`\usepackage[preprint]{tmlr}`), same content with the repository link. Fill in `\email` and `\addr` in `main_preprint.tex`; they are `TODO` placeholders.
- Sources: `main_submission.tex`, `main_preprint.tex`, `body_*.tex` (generated), `references.bib` (shared with the kirin build), `figures_v3/*.pdf` (identical to the kirin build).

## What changed relative to the kirin draft

Content is the same text, paragraph for paragraph. Only the presentation differs:

- Kirin boxes became run-in paragraphs: "Central claim.", "Result 1: …", "Audit note: …", "Descriptive observation: …". The "How to read this paper" box became an "Organization." paragraph at the end of Section 1.
- Sections are numbered by LaTeX; the manual numbers in the markdown were stripped. Verified: every hard-coded "Section N", "Table N", "Figure N" and "Appendix X" in the text matches the number LaTeX assigned.
- Tables and figures are floats with captions (tables above, figures below); the appendix claim-status table is set in place. Floats are kept within their sections.
- Citations are natbib author–year through `tmlr.bst`. The `note` fields of the four references (section pointers) print at the end of each entry, as in your original reference list; delete them from `references.bib` if you prefer bare entries.
- Added, per the TMLR template: an unnumbered "Broader Impact Statement" before the references, composed only of sentences already in Sections 1 and 7. References come before the appendices, as in the template.
- The AI-assistance statement is an unnumbered footnote on the first page (both variants), shortened from Appendix D of the kirin draft; Appendix D itself is dropped in the TMLR version and the "Organization" paragraph no longer lists it. TMLR's template also offers unnumbered "Author Contributions" and "Acknowledgments" sections before the references for the accepted version.

## Before submitting

- Anonymity: the submission PDF contains no author name, repository name or link, and no mention of the application context (checked by text extraction in `build_tmlr.sh`).
- The paper is written in the first person singular, which is fine for TMLR but is a mild de-anonymization signal.
- Camera-ready: switch to `\usepackage[accepted]{tmlr}` and set `\month`, `\year` and `\openreview` as in the template.
- TMLR has no page limit. Check the current author guidelines on OpenReview for any disclosure they require about AI assistance; Appendix D already discloses it.

## Revision after the pre-submission review (12 September)

- **Related work** is now Section 2 (both TMLR variants, the kirin draft and the arXiv package); every later section moved up by one and all in-text cross-references were renumbered and re-verified against the .aux. Six references were added after checking each against its arXiv or LessWrong page: nostalgebraist (2020, logit lens), Belrose et al. (2023, tuned lens), Blayney et al. (2026, cyclic fixed points in looped models, arXiv:2604.11791), Popescu et al. (2026, halting gates and trajectory readouts, arXiv:2607.20519), Korbak et al. (2025, chain-of-thought monitorability), and Kirin (2026, OPI, arXiv:2607.18553) cited in the third person. The anonymity check exempts that third-person self-citation and nothing else.
- **Tuned lens**: not fitted. Section 8 now opens its limitations with a "No tuned lens" paragraph giving the justification (training-free comparison against the head the model was trained with; the J-Lens authors' report that the tuned lens skips ahead to the output) and naming the missing arm explicitly. Fitting tuned lenses on the frozen band (12 layers x 4 passes) would close the question; that is compute you would have to authorize.
- **Supplement scrub**: `make_supplement.py` already rewrote `/home/<user>` paths and replaced the author name, handle, mentor, program and repository names; it aborts on any surviving hit and rescans the extracted zip. Independent grep of the extracted tree for the name, handle, email, `moloch`, `MATS`, `Neel`: zero files. `/home/` occurs only as the rewritten `/home/user`. `github.com` occurs twice, both the public `anthropic-experimental/agentic-misalignment` URL inside the J-Lens reference data. No git metadata is included.
- **Local-exit confirmation**: done on 12 September with the author's credential (plan frozen first). All six own-exit contrasts exclude zero on the confirmation population; Section 7 is now Result 4 with Table 6 and a new Figure 5, the claim-status row is Established, and the supplement carries the plan, receipts, merge record, readouts and results (`local_exit/confirmation_2026-09-12/`). The supplement is 96.6 MB; two regenerable files (the saved bootstrap resamples and the follow-up correctness JSON) were dropped to stay under 100 MB.
- Gurnee et al. reference: `month = jul`, prints "July 2026".
- The arXiv Appendix D and the TMLR footnote name the same activities (review, editing, bug fixes, verification, archiving; implementation, analysis, figures and drafting of the follow-up experiments).
- On the OpenReview form, consider requesting the Reproducibility Certification.

## Second review pass (12 September, after the own-exit result)

- Two factual corrections applied: the Section 7 interpretation no longer claims that pass-4 agreement between the families bounds calibration effects in earlier passes (the within-family comparison is the control; the pass-4 agreement is a descriptive observation), and the related-work and limits sentences now describe the tuned lens as a learned per-layer affine map applied before the fixed unembedding rather than a replacement of the head.
- `local_exit/confirmation_2026-09-12/verify_own_exit_family.py`: NumPy-only reimplementation of the six-contrast family (excess score, whole-group bootstrap in the same draw structure, max-t interval); reproduces every estimate and interval endpoint of Table 6 to 1e-12. Shipped in the supplement, listed in its README, and run on the extracted zip by `make_supplement.py`.
- Main text runs past twelve pages (references begin on page 13 of 17). TMLR permits it but applies its longer review timetable to submissions over twelve main-text pages; shortening, if wanted, should move operational detail out rather than compress the layout.
