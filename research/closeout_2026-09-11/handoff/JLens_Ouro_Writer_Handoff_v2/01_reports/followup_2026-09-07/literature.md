# Literature check, 7 September 2026

This is a post-submission audit. It records evidence needed to interpret the follow-up; it does not change what was done during the application task.

The J-Lens and Ouro papers were read in full, including their appendices. The original Huginn probing paper was read in full. The Huginn architecture paper's architecture, training objective, evaluations, and early-exit sections were inspected directly. Full HTML and paragraph-preserving text snapshots are in `sources/`; they are reading records, not material to reproduce in the report. Section links below point to the primary sources.

## What the J-Lens comparison tests

The paper motivates J-Lens as recovering information in “earlier layers where the logit lens produces uninterpretable readouts.” Its map averages derivatives from source states to present and future target positions over a corpus; decoding applies the native normalization and unembedding after that map. [Introduction and Methods](https://transformer-circuits.pub/2026/workspace/index.html#methods-jlens).

The quantitative intermediate-recovery benchmark itself accepts a hit at **any layer**, then summarizes pass@k across k. Best-layer aggregation therefore follows the paper's readout protocol, although it can obscure the depth profile. The same appendix reports J-Lens as the worst of its three lenses at predicting actual next-token outputs through most depths. Output fidelity and intermediate recovery are distinct evaluations. [Quantitative comparisons, Figures 52–56](https://transformer-circuits.pub/2026/workspace/index.html#app-quant).

The default Sonnet recipe targets the **penultimate** residual; including the final block sometimes adds noisy artifacts. The main definition and pseudocode also describe a final-layer target, so this is a documented variant rather than a uniquely specified default. Position effects are summed before averaging source positions, then prompts. Typical fitting uses 1,000 sequences of 128 tokens. [Method details, Figures 57–60 and pseudocode](https://transformer-circuits.pub/2026/workspace/index.html#app-method-details).

## Ouro supervision and exits

Ouro states that “at each recurrent step” it produces an LM-head output. A shared transformer stack is repeated; Eq. 2 defines next-token cross-entropy after each pass, and Eq. 4 weights those losses by learned exit probabilities with entropy regularization. This supervises **pass boundaries**, not every physical block. [Sections 3.1–3.3, Eqs. 1–4](https://arxiv.org/html/2510.25741v2#S3.SS1).

The gate is separate from the token readout. Stage II freezes the language model and learns gate decisions from detached improvements in task loss. [Section 3.4](https://arxiv.org/html/2510.25741v2#S3.SS4). The 2.6B model uses 48 physical layers, width 2,048, and four passes in its later training stages. [Sections 4.1 and 4.3.1](https://arxiv.org/html/2510.25741v2#S4.SS1).

The paper's QQP analysis compares actual pass predictions: step 2 versus step 4 agrees on 361/1,000 items; step 2 versus step 3 on 551/1,000. It explicitly identifies this as an observational mediation proxy. These findings establish revision on that setup, not causal faithfulness or revision in our lens outputs. [Section 7.2, Figure 9](https://arxiv.org/html/2510.25741v2#S7.SS2).

Inference: pass supervision plausibly strengthens Ouro's native output basis. Its causal contribution to our lens gap remains untested.

## What is actually known about Huginn readability

The original probing source is Lu et al., *Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer* (arXiv:2507.02199v1). Its raw logit lens applies RMSNorm and the unembedding; its coda lens adds two transformer blocks, with normalization before and after. [Section 2.2, Eqs. 2–3](https://arxiv.org/html/2507.02199v1#S2.SS2).

On the first 100 arithmetic questions, with 16 recurrent passes, raw logit-lens ranks oscillate by block. Blocks R1–R3 frequently decode signed-integer prefixes; R4 almost never does. Coda readouts reverse the pattern: R4 approaches 100% numeric prefixes, R1–R2 approach zero. The paper concludes that “lens applicability must be assessed on a per-layer basis.” [Section 3.2, Figures 2–3](https://arxiv.org/html/2507.02199v1#S3.SS2).

This verifies a **block-specific failure of direct unembedding**, not generally unreadable Huginn states, and not lower concept recovery than Ouro on matched data. Numeric-prefix frequency is also weaker evidence than recovering the designated intermediate. Their intermediate analysis selects 67 correctly answered, single-token cases from 2,000 questions and uses R3 for logit lens, R4 for coda. [Section 3.3, Figure 4](https://arxiv.org/html/2507.02199v1#S3.SS3).

## Huginn's training differs, but its coda is trained across depths

The original model has two prelude layers, four recurrent layers, and two coda layers; hidden width is 5,280. It concatenates the prelude output with the recurrent state through an adapter at each pass, starting from a random state. The coda decodes the completed recurrent state. [Geiping et al., Sections 3.1–3.2](https://arxiv.org/html/2502.05171v2#S3.SS1).

Training samples a depth from a heavy-tailed distribution, applies next-token loss after the coda at that depth, and backpropagates through the final eight recurrent passes. This is not Ouro's simultaneous weighted loss at every pass boundary. However, the same coda learns to decode states at many sampled depths; saying Huginn has no intermediate supervision without that qualification would mislead. [Section 3.3](https://arxiv.org/html/2502.05171v2#S3.SS3).

The model supports early exits by comparing successive output distributions, without adding exit heads. Those distributions include the coda. Raw unembedding of R4 is therefore not the native early-exit prediction. [Section 6.1](https://arxiv.org/html/2502.05171v2#S6.SS1).

## Implications fixed before a new Huginn result

Prediction: if J-Lens recovers useful latent content that the output head cannot directly decode, its paired advantage over raw logit lens on the same concept-recovery metric should be larger in Huginn than in Ouro. A failed prediction remains a failed prediction.

Keep all recurrent blocks and predeclared depths in the comparison. Include coda readouts as a separate baseline and actual model exits as ground truth. For a native exit comparison, propagate the complete sequence state through the coda; that module includes attention. Record the random initial state or seed so paired readouts refer to the same trajectory. Differences in tokenizer, task correctness, training corpus, width, recurrence, and coda architecture prevent attributing a cross-model gap specifically to supervision.

## Methodological concerns encountered

- Any-layer coverage is not held-out layer selection.
- J-Lens: target ambiguity and output-fidelity mismatch, described above.
- Ouro: observational revision cannot establish causality. Its losses do not constrain each trajectory to improve monotonically; frozen-LM gate fitting cannot guarantee this.
- Huginn: block, readout, task, and correctness-selection restrictions prevent the broad readability claim.
- Cross-model comparisons cannot isolate a training mechanism without a controlled ablation.

## Chronological record

1. Located primary papers and downloaded full text. The guessed Huginn architecture v3 URL returned 404; arXiv lists v2 as current, which was used. No third-party summary is used as evidence.
2. Read J-Lens fully, including appendices; re-read any truncated tool output.
3. Read Ouro fully, including appendices; checked the cited equations directly.
4. Read the Huginn probing paper fully and checked the original architecture/training source.
5. Recorded the Huginn prediction above before inspecting any new Huginn result. No model runs were performed for this literature audit.
