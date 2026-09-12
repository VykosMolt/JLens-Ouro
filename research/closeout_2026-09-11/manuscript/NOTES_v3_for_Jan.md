# Notes on draft three (my thoughts, not part of the paper)

## On the science

1. **The strongest new fact is the same-band reversal, and the paper should be built around it.** Pass 4 layers 26–37 give +23 points; the identical layers give −6.5, −8.6 and −11.1 in passes 1–3, all excluding zero. That is a cleaner statement than "early deficit, late gain" because it removes the layer-averaging objection. The abstract and Result 2 already lead with it; the title could too ("the same layers reverse sign across passes").

2. **The domain-level component is a real caveat, not a footnote.** J-Lens recovers same-domain names nine times more often than arbitrary controls. The within-domain contrast survives (+20.8), but astronomy goes to −1 within domain and SI units are zero under both methods. Per-domain heterogeneity spans −1 to +51 points over 16-item groups. A reader will ask whether "concept recovery" is partly "category recovery"; the paper says so, but a per-domain figure (bars with descriptive intervals) would make it honest at a glance. I did not add one because domains are not independence units and I did not want to invite over-reading; your call.

3. **The local-exit observation is the most interesting lead and the least defensible number in the paper.** On the discovery set the own-exit lens beats the final-target lens by 15–23 points in the same band where the final-target lens recovers almost nothing. If it holds on the confirmation states it changes the interpretation of the early deficit from "early passes are unreadable by J-Lens" to "the final-pass target is the wrong target for early passes". It costs no fitting: 11 GB from your private Hub repo and ten minutes on the laptop GPU. I would do this before any submission; the paper currently has to carry it as a diagnostic.

4. **Diagonal − raw should stay borderline**, and Result 3 should not be read as "position aggregation matters" in general: it is one control fit on fit01's paragraphs with no fit-variance estimate.

5. **The Huginn appendix is honest but weak.** With the estimator lost and two seeds of one fit, it cannot support either direction. If a venue pushes back, dropping it entirely would lose little; keeping it documents that the prediction was made and not confirmed.

## On presentation

6. Figures 1–5 are in the kirin system with the gentle-red accent. Source Sans Pro is installed system-wide now (TeX Live OTFs linked into `~/.local/share/fonts`), so figure text and body text share the typeface.
7. Figure 2 is dense (eight panels). It earns its place because the bottom row shows that the raw lens has its own late peaks in every pass; consider cropping to passes 1 and 4 if space is tight.
8. Table 6 uses your status vocabulary. The "Bounded screen" label for the within-domain result is my judgement; you may prefer Established with the caveat in the basis column.
9. Appendix D is your statement of 12 September, with the sentence on follow-up contributions added at the end. The interim fact sheet `CONTRIBUTION_AI_ASSISTANCE.md` was removed from the repository and the handoff at your request; draft two (`JLens_Ouro_Second_Draft.md`, repository and handoff) still carries the old draft statement in its Appendix D as the dated record.
10. Numbers are unchanged from v2 (ledger applies). Two v1 double-rounding slips were corrected in v2 and remain corrected here.

## Reconciliation with your edited draft (12 September)

Your `~/Downloads/JLens_Ouro_Third_Draft.md` (saved 01:05) was ahead of the repository copy the kirin build was made from. Ported paragraph for paragraph, with the kirin boxes and captions kept: the title ("Final-Target J-Lens in Ouro"), the abstract, "under the final-target estimator" in Section 1 and the restored fourth paragraph there, "accepted single-token forms" in Section 2.2, the Newton/"N" alias disclosure in Section 3.2, the rewritten within-domain paragraph in Section 4.4 (sensitivity to the control set, itemwise inequality vs. aggregate bound), the local-exit inventory wording (remote availability not verified here), your longer Section 6 interpretation with the exit-state hypothesis, the Section 7 first paragraph, the reproducibility paragraph (43 lost / 19 unresolved split), the conclusion, the Records paragraph in Appendix B, and Appendix D. The Table 6 row on within-domain controls was reworded to match Section 4.4; the scope box in Section 1 now says the raw lens beats *this final-target estimator* in the early passes. Figure 5: the negative band means in panel (d) had their labels drawn on top of the error bars; labels now sit below negative intervals and the axes have more headroom.

## Layout rules (12 September, second pass)

Pandoc emits every table as a `longtable`, which breaks across pages; the Lua filter now rewrites each table as a plain `tabular` and wraps table-or-figure plus its `\kirinfigcaption` in one `minipage`, so neither a table nor a figure is ever separated from its caption or split. The kirin boxes are set `unbreakable` in the wrapper (your `boxes.tex` is untouched), and `\kirinpart` asks for ten lines of room so a part heading cannot sit alone at a page foot. Cost: some pages end early (5 and 6 in this build) because the next block did not fit whole.

## Related work and renumbering (12 September, after the pre-submission review)

Section 2 is now "Related work" (readouts of intermediate states; looped and recurrent-depth models; prior readouts on Ouro, citing OPI in the third person; why read latent passes). Sections 2-8 became 3-9 and every in-text "Section N" was updated; the TMLR build verifies heading numbers against LaTeX, and I listed every in-text reference by hand after the change. The limits section opens with a "No tuned lens" paragraph. All six new references were checked against their source pages before being added.

## Own-exit result on the confirmation population (12 September)

Section 7 now reports Result 4: with the initial-study banks applied to the accepted confirmation states, the own-exit lens beats the final-target bank of the same family by 22.6, 26.5 and 34.2 points and the raw lens by 16.4, 18.3 and 23.2 points in the band, passes 1–3, all six simultaneous intervals excluding zero (plan frozen before the banks were applied). The same family's final-target bank reproduces the pass-4 gain (+22.39 vs fit01's +23.19; exit3 − fit01 −0.80 [−1.71, +0.30]), which is the best evidence that the calibration difference between the families is small. Table 6 and a new Figure 5 carry it; the discovery-population precursor moved to Appendix A; the claim-status row is Established; abstract and conclusion updated. The retrieved shards and merged banks (15 GB) are on the laptop only, in `~/ouro_project/artifacts/jlens/retrieved_2026-09-12/`; they should go to the SSD archive next time it is attached. Things I did not do: the penultimate-target-in-early-pass control the hypothesis predicts (no such bank exists), and any tuned lens.
