# Independent scientific wording review

Reviewed `REPORT.md` SHA-256 `843fa92b84866c77cd83fa517bbb4e6f726749580b6219813574499c098d1a7f` against the full-paper reading record, relevant original source passages, result JSON, `audit/FOLLOWUP_VERIFICATION.md`, `audit/AUDIT.md`, and `probe/probe_audit.md`. This was a claims review with targeted value checks, not a repeat of the independent numerical verification. No report or analysis file was edited.

**Main conclusions pass.** The fixed-layer losses in multihop loops 1–3 support the conclusion that the any-layer deficit is not solely a late-layer selection effect. The loop-4 positive window is explicitly qualified by its location, selection-adjusted band sensitivity, and conditional uncertainty; it is not presented as a general early-depth advantage. The report correctly separates intermediate recovery from next-token prediction, attributes the verbalizability objective to Anthropic, limits Ouro supervision to pass boundaries, and leaves its causal role untested. Huginn is a prospective prediction with no new comparison. The passing/failing analysis remains observational, and the probe's loop-3 win and inconclusive data-size sensitivity are retained.

**One wording correction is needed.** In next-test item 1 (reviewed line 70), replace “the documented target/position mismatch.” Sonnet's default penultimate target differs from this final-target fit, but the released estimator's summed present/future-position reduction follows the paper's pseudocode. The paper also explicitly studies target and position variants. Suggested replacement: “the target-layer difference and sensitivity to readout position and position reduction.” If the intended mismatch is instead calibration positions versus evaluation positions, name that explicitly and do not imply an established estimator error. Evidence: [original methodological details](https://transformer-circuits.pub/2026/workspace/index.html#app-method-details), local `sources/jlens.txt:2180–2277`, and `jlens/fitting.py:8–16`.

**Short clarifications recommended before finalizing:**

- At reviewed line 40, replace “genuine revision” with “changes in native next-token predictions.” The retained exit tokens establish prediction changes; they do not establish revision of an internal reasoning plan.
- In the probe paragraph, add that its intervals are pointwise and conditional on the fitted classifiers and selected layers. Resampling operand pairs does not include retraining or reselection uncertainty; the probe audit states this explicitly.
- Introduce the Huginn prior with “On the published arithmetic probe” and change “Before any Huginn run” to “Before any new Huginn J-Lens run.” These specify the evidence's task scope and distinguish this follow-up from the pre-existing Huginn inference/profile work.

No additional experiment is required to resolve these wording points. No numerical correction or reversal of the report's main interpretation was identified.
