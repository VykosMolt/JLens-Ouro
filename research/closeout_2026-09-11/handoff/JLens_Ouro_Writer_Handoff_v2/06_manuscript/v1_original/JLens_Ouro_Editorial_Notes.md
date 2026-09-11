# Editorial notes for the first Ouro J-Lens manuscript

11 September 2026

## What this draft is based on

The manuscript is written in Jan's research-author voice from four supplied run reports. It is not an independent rerun of the experiments by the drafting assistant. The original J-Lens article and Ouro paper were consulted for the readout definition, original target configuration, and architectural context. Project-specific measurements come from the supplied reports.

The latest verification report takes precedence where it corrects earlier statements. In particular, the draft uses “FP64 reference computation,” not “exact FP64 arithmetic”; treats diagonal-minus-raw as borderline; and keeps the later preservation audit separate from earlier assurances that application artifacts had been preserved.

The manuscript has six pages of main text and two pages of appendices and references. The narrow result is the primary story. Historical exit comparisons, arithmetic probes, and Huginn are kept in Appendix A rather than merged into the prospective test.

## Method details to insert from the frozen records before submission

1. **Exact extraction position and prompt format.** The supplied reports do not fully specify the prompt wrapper, selected token index, padding/masking conventions, or whether the same extraction-position rule applies to all confirmation items. The draft therefore says “recorded readout position” rather than inventing a rule. Copy these details from the actual frozen extractor and include at least one real benchmark example. Do not use a made-up example as though it came from the 160-item test.

2. **Exact fitting recipe and calibration inputs.** The reports establish 100 paragraphs per fit and describe the target/position comparisons, but do not provide a complete, untruncated fitting specification. Insert the calibration distribution and identifiers, sequence processing, position sampling, averaging/normalization, derivative accumulation, attention-gradient treatment, and checkpoint-to-bank conversion from the retained code and records. Do not borrow defaults from the Anthropic implementation to fill unverified gaps. State which details are recoverable despite the original checkpoint losses.

3. **Benchmark construction and dependency grouping.** Supply the actual item/alias file, ten domain names, factual sources, annotation instructions, grouping rule, and group memberships. The aggregate group sizes and bootstrap settings are already in the draft. The three calibration-text overlaps should remain disclosed. The confirmation set should not be called pretraining-uncontaminated or difficulty-matched.

4. **Complete provenance and availability.** Replace abbreviated model and bank hashes with complete verified identifiers, give a stable code/environment snapshot and all 20 secondary contrast definitions, and provide a real artifact-access statement. No public repository or release destination has been verified for this manuscript. A local freeze is not automatically a public preregistration. Do not claim an off-device backup until one exists and has been checked.

5. **Figures and authorship statement.** No complete layerwise arrays or original plot assets are attached here, so this draft uses reported tables rather than invented curves. Add the actual fixed-layer plots and benchmark examples from retained data. Before submission, complete an accurate contribution/AI-assistance statement and check the target venue's current requirements; no venue policy was assumed in drafting.

These are documentation and presentation gaps in the supplied drafting materials, not a request to rerun the research.

## Evidence boundaries retained

The primary interval is the prespecified dependency-group percentile interval for one fixed estimator and one fixed band. The displayed secondary intervals come from the original 20-contrast simultaneous family. Component intervals are descriptive. Discovery and confirmation are not pooled, calibration fits are not counted as new questions, and similarly performing fits are not called formally equivalent.

A higher concept-recovery rate is not higher answer accuracy. The derivative-target and position controls establish estimator sensitivity, not Ouro's causal reasoning mechanism. The historical Huginn pilot was run and failed to support the relative prediction, but its missing estimator prevents full verification; it is not silently discarded or promoted.

The preservation statements distinguish exact table reconstruction, rescoring from banks and states, forward replay, and complete estimator fitting. The last of these is not available for the original fits. This limit is in the main manuscript, not hidden in these notes.

## Source map

- **S1:** `Pasted text(20260907-182136).txt` — post-submission discovery/diagnostic report.
- **S2:** `Pasted text(2).txt` — independent calibration fits, estimator controls, Huginn pilot, and watchdog incident.
- **S3:** `Pasted text(3).txt` — prospective fixed-160 confirmation, primary components, and secondary table.
- **S4:** `Pasted text(4).txt` — final numerical verification, repairs, and preservation inventory.

The supplied terminal pastes contain truncated prose and damaged table borders. Numbers and statements were included only where legible in the provided record or explicitly supplied by another report. The original repaired Markdown report and raw experiment directory are not mounted in the drafting environment. Missing clauses, full hashes, and omitted method settings were not reconstructed from guesswork.

The other mounted evaluator documents and `paper.pdf` concern older OPI/RPE work, including superseded findings. They were not treated as sources for the J-Lens results.
