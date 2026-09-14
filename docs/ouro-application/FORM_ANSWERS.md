# Form answers

## ~16 hour research task — Google Doc link

[Add the Google Doc link after importing the write-up and setting General access to “Anyone with the link — Viewer”.]

## (Optional) Link to any other relevant outputs (code, colab, etc)

Operational Proto-Introspection in Looped Language Models
https://arxiv.org/abs/2607.18553

Relational Preference Encoding
https://arxiv.org/abs/2604.09870

## The first 1–3 pages of the attached doc are an executive summary

[Check: the write-up has a one-page executive summary, including the figure. Confirm the pagination after importing into Google Docs.]

## The document permissions are set so that anyone with the link can see my doc

[Set Google Docs sharing to “Anyone with the link — Viewer”, then check.]

## What question did you try to answer?

How well does J-Lens read known intermediate variables from Ouro's hidden states across its four recurrent passes, compared with the plain logit lens? I also asked whether lenses fitted to an earlier loop's exit and to the final exit agree more as the model keeps looping.

## Why is this question interesting / why did you choose it?

I've been working with Ouro's hidden states for a while. My earlier work found useful information before the model answers, but reading a useful signal and understanding what the model is doing are different things. J-Lens seemed like a good way to investigate that more directly. A looped model also gives a useful comparison: the same physical layers are visited several times, so I can ask how the readout changes as the computation continues. I wanted to see whether the more elaborate lens actually adds anything over the cheap baseline.

## What conclusions have you reached about this research problem?

On 90 multihop items, the plain logit lens reads the labelled intermediate better at each of the first three loops. With 100 fitting prompts, final-exit J-Lens minus logit lens is −0.356, −0.479, −0.407 and −0.022 in excess hit@10. The first three paired 95% intervals are below zero; the fourth is unresolved. Arithmetic is mixed: J-Lens loses at loop 1 but wins at loop 2.

The local-versus-final lens KL gap does not steadily shrink at the last prompt token, but does one token earlier. Their top-1 agreement remains low at both positions. So the early multihop deficit is the result I am most comfortable with; a general claim about convergence, or why the deficit happens, would go beyond what I have.

## Technical setup: What are the key things you try to quantify in this study and how do you define and measure them? Give the key technical details: what models you use, datasets, prompts, the metrics used.

Ouro-2.6B runs 48 shared layers four times. I recorded hidden states at all 192 loop/layer locations and fitted Jacobian lenses on Wikitext, using up to 100 prompts per exit. The main comparison is with the logit lens on the released multihop and order-of-operations tasks. After excluding leaked or unscorable intermediates, there are 90 multihop items and 51 numeric arithmetic items.

At the last prompt token, I check whether each intermediate is in the top ten at any layer within a loop. I subtract the average of the identical test over matched control names, average multiple intermediates within each item, and compare the two lenses with 20,000 paired item-bootstrap draws. The eight 95% intervals are unadjusted. Local-versus-final lenses are compared with KL across all 148 prompts and 48 layers, plus top-1 agreement over the last eight layers. I also tested one token earlier, increasing fit size, and a supervised probe on a separate arithmetic task.

## What is the strongest evidence you found against these hypotheses?

I expected J-Lens to improve intermediate readout. The strongest counterexample is multihop loop 2: J-Lens minus logit lens is −0.479, with a paired 95% interval of [−0.579, −0.377]. Increasing the fit from 8 to 100 prompts did not recover the early multihop readout.

I also expected local and final-exit lenses to agree more with depth. At the last prompt token their KL gaps are 4.77, 4.10 and 6.30 nats, which is not a steady decline. But the one-token-earlier control pushes back against the opposite generalisation: there KL does decline. Arithmetic loop 2 also pushes back against a blanket claim that J-Lens is worse; its advantage is +0.140, interval [+0.020, +0.264].

## What are the biggest limitations to your results? Could you have addressed them?

One model and one estimator. Ouro's training through a shared output head might favour the logit lens, but I have no matched ablation to test that. Readable intermediate tokens do not show that the model causally uses them. The fits are small and nested, so they do not estimate variation between independent fits, and the intervals are unadjusted. The separate probe comparison uses an older lens with incomplete fitting records.

I could address several of these with a tuned-lens baseline, independent repeat fits and matched changes to the Jacobian estimator. The model-training explanation needs a larger ablation. I didn't complete those within this project, so I have kept the conclusion about what these readouts do on this model.

## How did you use LLMs in this research task and write-up? Which LLMs? How exactly did you make sure they weren't just giving you slop?

I used Claude Opus and Codex mainly to oversee the remote run and debug the GPU launches. There were nine attempts, so that took a fair amount of work. I checked their code changes, raw prompts and results, plots, and reported numbers against the saved outputs. I also had Fable check the code. Agents helped with additional audits, and Codex helped check the write-up against the saved results and edit it, including the LaTeX version.

I prioritised the recording and scoring code because an error there would affect everything downstream, then checked the comparisons, plots and written claims. The checks covered whether recording changes the model computation, whether controls get the same treatment as true intermediates, whether related examples leak across folds, and whether the answer matcher accepts false positives. The local recorder passes all 18 equality checks. Review still found mistakes: an inflated control statistic, mirrored-pair leakage and an unsupported Jacobian-norm interpretation. Those were corrected or retracted. The headline effects and confidence intervals were also rederived from the saved arrays during the write-up.

A basic recorder error would surprise me more than another analysis mistake, given the exact comparisons and the bugs already caught. The explanation involving shared-head training is much less secure; it is still a hypothesis.

## What, if any, prior experience do you have with mechanistic interpretability?

In Operational Proto-Introspection, I studied whether hidden states predict success before a looped model answers, how that signal moves across recurrent passes, and the limits of steering the frozen model. I ran this on a laptop GPU. In Relational Preference Encoding, I built hidden-state preference readouts; a later audit cut my headline result from 95.2% to 63.9%, and I issued an erratum. In One Concept, Multiple Geometries, spelling controls killed an apparent circle-of-fifths result. These projects gave me practice building readouts, testing what they actually measure, and correcting claims when the controls disagree.

## Why are you interested in Neel's stream specifically?

I want to get better at turning an interesting readout into a small experiment that tells me what it actually means. I've spent a lot of time exploring looped models independently, and my tendency is to let a question grow into a much larger project. The emphasis in your application on clear claims, strong baselines and checking the result against the evidence is useful pressure for me. I'd like direct feedback on which questions are worth pursuing and which experiments would really distinguish the explanations, especially around monitoring hidden computation.

## What is the likelihood you will join Neel Nanda's training program (Sept 28 - Oct 30) if accepted?

100%, absolutely.

## (Optional) Is there anything else important I should know about your application or project not covered above?

I estimate about 15 active hours on this project, excluding rental setup, waiting for fits and the form. The write-up happened during the final overnight run, which was not the plan.

For a research reference: Goran Đambić, Associate Professor and Head of the University Department of Software Engineering at Algebra Bernays University. He helped with the paper I submitted to ICLR, and we've discussed my work in detail. In OPI I credit him with suggesting the early-layer readability experiment.
