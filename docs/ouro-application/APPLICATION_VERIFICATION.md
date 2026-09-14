# Rewrite verification — 5 September 2026

This is an editing record, separate from the application. The original Markdown, form answers, DOCX and PDF are preserved in `history/2026-09-05-pre-rewrite/`.

## Research evidence

- Headline N=100 effects and every paired confidence interval were rederived from `artifacts/jlens/retrieved/jlens-b300-20260905-0359/eval/n100_exit3/{arrays.npz,items.json,task_names.json}` using the analysis utilities. The 20,000-draw item bootstrap with seed 0 reproduces the reported intervals. Arithmetic loop 2 favours J-Lens; the older blanket negative claim was removed.
- Lens-to-lens KL and agreement were recomputed from `artifacts/jlens/eval/b300_local_allexits_strict/arrays.npz` and `artifacts/jlens/eval/b300_local_allexits_pos-2_strict/arrays.npz`. KL averages all 148 items and all 48 layers; agreement uses the last eight layers. Position −2's maximum is 0.0050676, hence “around 0.5% or less.” These are local evaluations of B300-fitted lenses, not retained pod all-exit evaluations.
- Figure 1 uses these rederived values, rather than differences between rounded summary means. Eight pointwise intervals are explicitly unadjusted.
- The four `artifacts/jlens/lens/n100/exit{0,1,2,3}.json` metadata files each record 100 fitting prompts.
- Local 18/18 bit-exact comparisons are retained in `artifacts/jlens/validation/milestones_20260904T210354.666558Z.json`. The B300 pass is described in the watch board, but its underlying validation report was not located; the rewrite claims the local checks only.
- Probe figures are supported by `artifacts/jlens/probe/cv_all648/summary.json` and `artifacts/jlens/final/analysis.json`. The separate task and incomplete older fitting provenance are stated.
- Random examples retain the sample in `APPLICATION_EXAMPLES.md`, explicitly labelled as the older local evaluation. The underlying source is `artifacts/jlens/eval/round1_exit3x32/`.
- Rental details are supported by `INCIDENT_2026-09-04.md`, `watch/BOARD.md` and the nine retained run-state files. Approximately 29 minutes versus roughly 40 seconds is per fitting prompt, not total campaign speed.

## Background and correspondence

Reviewed the specified Rui/Ridger and Jonathan Williams correspondence, plus the September 3 outreaches to Tomasz Korbak, Micah Carroll and Buck Shlegeris. Those outreaches inform the opening motivation and voice; their scientific claims were not imported as new J-Lens results.

- The local erratum source `/home/moloch/Documents/Research/ouro/kirin2026_erratum/paper.tex` supports 95.2% → 63.9% (strict result 0.6392). Jan's August 17 correction to Ridger also states the correction: [correspondence](https://mail.google.com/mail/#all/1a0102cff53d70ff).
- Per Jan’s subsequent instruction, Williams is not named in the application or form answers. The research reference is Goran Đambić. His May 4 correspondence contains detailed feedback on model coverage, data splits and repeated random-seed runs: [thread](https://mail.google.com/mail/#all/19d6d9970f3b6427). OPI’s local `latex-v3/paper1.tex:46` credits him with the early-layer readability experiment. His academic rank and department-head role are now verified against the university’s official [faculty profile](https://www.algebra.hr/sveuciliste/en/lecturers-and-associates/lecturer/?id=25) and [Senate listing](https://www.algebra.hr/sveuciliste/o-nama/senat/), checked September 5, 2026: Associate Professor, PhD; Pročelnik Sveučilišnog odjela za programsko inženjerstvo (Head of the University Department of Software Engineering). His ICLR-paper assistance and detailed in-person discussions are supplied directly by Jan.
- Xingwei Qu's August 6 reply explicitly confirms the DLCM reporting error: displayed average 42.24 and improvement +1.01, versus reported 43.92 and +2.69. He said they would correct it. The rewrite does not claim a correction has already been published: [confirmation](https://mail.google.com/mail/#all/19fd7fcdeaa5faff).
- `/home/moloch/Documents/Research/One-Concept-Multiple-Geometries/paper/main.tex` supports the initial false positive and spelling control.
- The 15 active hours remains the applicant's estimate from the original draft, not independently verified time accounting. Sleep and personal reactions likewise remain personal narration.

The rewrite discloses AI assistance with editing. No correspondence was sent or modified.

## LaTeX revision

The current application PDF is compiled with XeLaTeX using Jan’s saved `paper_style` package, copied into this folder for rebuilding. The prose was revised toward the papers’ direct first-person explanations; the original overnight jokes remain. Revision strings, run IDs, GPU model identifiers and internal file references were removed from the reader-facing application and forms. This editing record and the historical originals remain separate from the submission. The Markdown, LaTeX and secondary Word copy carry the revised text.

## Final form and format check

The linked Application Format tab was retrieved directly from Google Docs on September 5. The compiled summary is one page including two graph panels and has 235 prose words before the final whitespace clarification (well below 600); five randomly selected raw examples follow immediately. The remaining body is a condensed retrospective chronological record drawn from saved logs. All substantive screenshot fields have drafted answers. Jan clarified that agents were mainly used for run supervision and launch debugging, that he checked all their output, and that Fable also reviewed code; the disclosure now reflects that. Earlier draft wording about not manually recomputing statistics is superseded by the direct clarification, without asserting that he recomputed everything by hand.

The actual Google Doc URL/access remains to be supplied. The guide also asks the applicant to write the final summary/form in their own voice; the saved answers are drafts for Jan’s final wording. [Application format requirements](https://docs.google.com/document/d/1p-ggQV3vVWIQuCccXEl1fD0thJOgXimlbBpGk6FI32I/edit?tab=t.sa4u6llpnkph).

## Final narrative structure

Jan subsequently supplied the research/writing guidance and explicitly chose its narrative structure over chronology. The final PDF therefore leads with two findings, followed by raw examples, baseline comparison, the exit/position experiment, concrete checks and limitations, and a short rental anecdote. It does not use the intermediate dated-log layout. The summary remains 235 words before captions; all screenshot questions have drafted responses, and the importable Word copy matches the revised content.

## Repository correction and humour

Jan clarified that the experiment code is not in the previously linked repository. Removed that project link from the current application, title block and optional form field; no replacement code URL was supplied. Added two dry jokes to the rental anecdote without changing the experimental claims.
