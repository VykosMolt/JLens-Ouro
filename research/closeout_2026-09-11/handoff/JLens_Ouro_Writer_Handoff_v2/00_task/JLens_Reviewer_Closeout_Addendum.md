# J-Lens reviewer addendum to the closeout task

Apply this alongside `JLens_Closeout_Prompt.md`. Preserve its storage safeguards, spending limits, source-grounding requirements, and prohibition on deleting source evidence. Do not restart completed numerical verification.

The reviewer raised one useful explanatory experiment and several narrower measurement and manuscript checks. Finish off-device preservation first. Then do the checks possible from retained data. The original prospective result remains unchanged; new analyses must be identified as post-confirmation reviewer follow-up.

## 1. Secure the evidence before doing more work

Copy the full required evidence to the verified external SSD using the original closeout procedure. Verify hashes and restoration without relying on the working directories. Do not delay this copy while assembling a publication-quality report. Add subsequent outputs to a versioned supplement rather than modifying the frozen archive.

Specifically inventory application-era local-exit J-Lens banks, their calibration inputs/configurations, the matching final-exit banks, and relevant saved states/readouts. These may be needed for the experiment below. Their earlier existence is not proof they survived. Search already-authorized archive locations non-destructively and record exact availability.

Do not delete anything in this task. An external SSD left as the only copy after a later purge is a sole archive, not redundant preservation.

## 2. Run three bounded checks on the retained confirmation data

### A. Within-domain negative controls

The accepted endpoint uses the other 79 intermediate names as equally weighted controls. Verify the reported ten-domain/eight-concept structure against the actual benchmark, then use the other seven names in each item's domain as a secondary control set.

Do not change the full-vocabulary top-10 definition, accepted token forms, intended labels, population, item weights, fitted lens, or loop-4 layers 26–37. This is a control-subset reaggregation of existing readouts, not ranking among only eight candidate names. Inspect whether each domain actually groups names of the same semantic type; do not claim semantic matching solely because a field is named `domain`.

Report intended recovery, original control recovery, within-domain control recovery, and the paired excess difference for both methods. Use paired dependency-group resampling with the frozen grouping. Include per-domain estimates and sample sizes. Keep the original primary endpoint and its inference intact. This new contrast was not among the original prespecified twenty secondaries.

Independent arithmetic check: if the seven controls are a subset of the original 79 for every item, with unchanged eligibility and weights, their aggregate J-Lens hit rate is at most 79/7 times the all-control rate. Using the rounded published values, the within-domain paired excess estimate is therefore bounded below by

    0.23490 - (79/7) * 0.00344 ≈ 0.1961.

This is a bound on the observed point estimate, not a confidence interval or evidence about new control names. Derive it again from the exact arrays. If the computed within-domain estimate violates it, resolve the population, weighting, alias, or implementation mismatch before interpreting the result. Do not use the bound as a substitute for computing the actual estimate and uncertainty.

### B. Explicit tokenized-input overlap audit

For all 160 items, compare every scored accepted token ID for the intended label and controls with the actual non-padding model input visible at the readout location. Include any wrapper, demonstrations, and prefilled response prefix that the frozen extractor actually used. Do not audit a human-readable question while ignoring additional model-visible text.

Save an item-level record of matches, token positions, accepted form, intended/control role, and the visible text around the match. Distinguish actual token-ID matches from substring matches and incidental or ambiguous word senses. A zero exact-token overlap does not prove absence of semantic clues or pretraining overlap.

Do not write “none occurs in any prompt” unless the exhaustive check supports that exact claim. Report intended-label overlap and control overlap separately. Where overlap exists, preserve the full-population endpoint, report the affected subset, and add clearly labeled leak-clean sensitivity analyses. Removing a contaminated control must use the same rule for both methods and record the new denominator. Do not discard items or aliases selectively to improve an effect.

### C. Same physical band in every pass

Recompute the fixed-layer average for physical layers 26–37 separately in passes 1, 2, 3, and 4, using the same estimator specification, items, aliases, controls, and grouping. These are not the already-reported early means over all 48 layers.

Compare the original twenty contrast definitions before assigning status. If the same-band early contrasts were not prespecified, label them exploratory/post-confirmation. Give paired effect sizes and justified uncertainty, with an explicit multiplicity policy for any new family of inferential claims. Do not replace an original endpoint or move the band after seeing the curves.

Return a compact same-band table alongside the existing all-layer and any-layer summaries. Do not infer same-band deficits from an all-layer average.

## 3. Document the dependency groups

Recover the actual grouping rule, construction code, and memberships from the frozen records. Explain in ordinary language what joins items: intermediate concept, shared fact, relation template, connected components of overlapping dependencies, or whatever was actually used.

Identify why the largest group contains 42 items. Include a membership/example table sufficient to make that decision inspectable. Retain the existing item-resampling sensitivity and leave-one-group-out range. The weight-based effective group count is not a literal number of independent observations and does not certify the adequacy of the grouping.

Do not invent a rationale for old groups or regroup until an interval becomes favorable. Any proposed alternative grouping must have an outcome-independent rationale and be labeled a new sensitivity analysis.

## 4. Local-exit targets: the single conditional scientific extension

Hypothesis: the early-pass deficit may partly reflect the distance and transformations between the source state and the main bank's final-pass target, rather than a property unique to early-pass representations.

If usable local-exit banks and comparable final-exit banks survived, evaluate concept recovery from the same early-pass source states using:

- Raw logit lens.
- A J-Lens targeting the current pass's own exit.
- A J-Lens targeting the final pass's exit.

Use the same physical band, layers 26–37, in passes 1–3. Report local-minus-raw and local-minus-final contrasts, separately by pass. Retain full curves as descriptive output; no best-layer search.

For an interpretable target comparison, verify matched calibration texts/count, source and target position treatment, derivative budget, gradient conventions, normalization, unembedding, model revision, and scoring. An old application bank fitted under a different recipe versus the newer final bank does not isolate target horizon. A mismatched comparison can be reported as exploratory, with the mismatch explicit, but cannot settle the hypothesis.

Check actual source-to-target block counts and virtual indexing, excluding identity endpoints from performance evidence. A layer-48 local-exit equality is a useful implementation check, not a positive scientific result. Where recipes are identical, pass-4 local and final targets should coincide.

Do not reconstruct a local-exit bank by inverting or factoring a context-averaged final-target bank: averages of Jacobian products generally do not factor into products of average Jacobians.

If the requisite banks or provenance are missing, return the precise missing artifacts and a bounded matched-fit proposal, including time, storage, and incremental cost. No new paid run or budget increase is authorized by this addendum. Do not reopen Huginn fitting or turn this into a target-by-layer-by-model sweep.

Freeze the extension's contrasts and recipe before its first outcomes, while acknowledging that its hypothesis and reused evaluation set are post-confirmation. A positive local-target result would support target dependence and be consistent with the transport explanation. It would not, by itself, prove that averaging destroyed information. A null would weaken the simple rescue prediction, not prove the shared-head-supervision explanation.

For interpretation, account for two existing qualifications: the penultimate target shortens the main path by one block but reduces the observed late gain; and pass counts are not comparable distances across different-sized recurrent cores. Do not describe the historical Huginn pilot as identifying a cause or ruling out all supervision-related explanations.

## 5. Reconcile and revise the manuscript

Create a source-to-manuscript ledger covering every empirical number, interval, count, date, sample definition, table cell, and numerical claim in the abstract, main text, appendices, captions, and notes. Read intact frozen reports and machine-readable outputs. Damaged terminal pastes are not the authoritative transcription source. Mark discrepancies rather than silently harmonizing conflicting versions.

Use reproducible scripts for tables and figures wherever practical. Export the actual five-fit layer curves and confirmation curves from retained data. Keep discovery and confirmation distinct, and distinguish fit variation from item/group uncertainty. Do not reconstruct curves by interpolating a few reported peaks or averages. If original figure data are unavailable, report that limitation instead of drawing an invented figure.

Move drafting-source tags such as [S1–S4] and the internal provenance bibliography out of the publication prose into the source ledger/supplement. Keep real literature citations and a meaningful artifact-availability statement. Do not remove reproducibility limitations with the drafting scaffolding.

The existing abstract already states the main positive and negative findings. Tighten it to lead with the observed pass-dependent pattern rather than rebuilding it as a new causal story. State the nearby-exit hypothesis as unresolved unless the new comparison supports a narrower update.

Prepare an accurate AI-assistance/contribution statement from available logs, covering implementation, analysis/checking, literature help, and drafting as applicable. Distinguish author decisions and verified human review from agent work. Mark attribution requiring Jan's confirmation; do not claim Jan personally rechecked every number or wrote all code manually. Do not invent tool/model versions or assume a venue policy before a venue is selected.

## Deliverable and stopping condition

Extend `JLens_Ouro_Writer_Handoff.zip` with:
- The within-domain control result and arithmetic check.
- The exhaustive input-overlap ledger and sensitivity results, if needed.
- The same-band cross-pass table and its analysis status.
- Group definitions, memberships, and the largest-group explanation.
- Local-exit bank inventory and either the bounded comparison or a precise blocker/proposal.
- Intact reports, a complete numerical source ledger, real plot data/assets, and revised manuscript sources.
- The off-device preservation receipt and updated supplement manifest.

Keep the original accepted endpoint and every historical file untouched. Report what changes the scientific interpretation and what only improves documentation. Stop when these specific reviewer issues are answered or concretely blocked; do not restart general verification or add unrelated research.

No submission edits, publication, outreach, or source deletion in this task.
