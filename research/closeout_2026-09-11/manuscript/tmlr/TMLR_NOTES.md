# TMLR version of draft three

Built by `tmlr/build_tmlr.sh` from `JLens_Ouro_Third_Draft.md` with the official TMLR style files (`tmlr.sty`, `tmlr.bst`, Apache-2.0, from github.com/JmlrOrg/tmlr-style-file; licence in `LICENSE-tmlr-style`). pdfLaTeX + BibTeX, 15 pages.

## Outputs

- `JLens_Ouro_TMLR_submission.pdf` — double-blind version (`\usepackage{tmlr}`): "Anonymous authors", running head "Under review as submission to TMLR", repository URL replaced by "[URL withheld for double-blind review]". This is the file to upload on OpenReview (TMLR takes the PDF, not the sources).
- `JLens_Ouro_TMLR_preprint.pdf` — named version (`\usepackage[preprint]{tmlr}`), same content with the repository link. Fill in `\email` and `\addr` in `main_preprint.tex`; they are `TODO` placeholders.
- Sources: `main_submission.tex`, `main_preprint.tex`, `body_*.tex` (generated), `references.bib` (shared with the kirin build), `figures_v3/*.pdf` (identical to the kirin build).

## What changed relative to the kirin draft

Content is the same text, paragraph for paragraph. Only the presentation differs:

- Kirin boxes became run-in paragraphs: "Central claim.", "Result 1: …", "Audit note: …", "Descriptive observation: …". The "How to read this paper" box became an "Organization." paragraph at the end of Section 1.
- Sections are numbered by LaTeX; the manual numbers in the markdown were stripped. Verified: every hard-coded "Section N", "Table N", "Figure N" and "Appendix X" in the text matches the number LaTeX assigned.
- Tables and figures are floats with captions (tables above, figures below); the appendix claim-status table is set in place. Floats are kept within their sections.
- Citations are natbib author–year through `tmlr.bst`. The `note` fields of the four references (section pointers) print at the end of each entry, as in your original reference list; delete them from `references.bib` if you prefer bare entries.
- Added, per the TMLR template: an unnumbered "Broader Impact Statement" before the references, composed only of sentences already in Sections 1 and 7. References come before the appendices, as in the template.
- Appendix D (contribution and AI-assistance statement) is unchanged. TMLR's template also offers unnumbered "Author Contributions" and "Acknowledgments" sections before the references for the accepted version.

## Before submitting

- Anonymity: the submission PDF contains no author name, repository name or link (checked by text extraction). Appendix D still says "the MATS application", which does not name anyone but narrows the author pool; your call.
- The paper is written in the first person singular, which is fine for TMLR but is a mild de-anonymization signal.
- Camera-ready: switch to `\usepackage[accepted]{tmlr}` and set `\month`, `\year` and `\openreview` as in the template.
- TMLR has no page limit. Check the current author guidelines on OpenReview for any disclosure they require about AI assistance; Appendix D already discloses it.
