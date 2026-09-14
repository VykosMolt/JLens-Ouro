# Form answers

Project in one paragraph.
I tested J-Lens against the logit lens on Ouro-2.6B, a transformer that runs 48 layers four times. On 90 multihop items, the final-exit J-Lens loses at loops 1–3 (paired excess-hit@10 differences −0.356, −0.479, −0.407). Arithmetic is mixed: J-Lens wins at loop 2. Lenses fitted to local and final exits also disagree strongly; whether their KL gap shrinks depends on the token position. The 100-prompt B300 fits reproduce the early multihop deficit. I expected J-Lens to do better. The cheap baseline had other plans.

Most surprising number.
At multihop loop 2, J-Lens minus logit lens is −0.479, with an unadjusted paired 95% interval of [−0.579, −0.377].

Biggest limitation.
One model and one estimator. Shared-head supervision might explain the result, but I have no matched ablation. The fits are small and nested, and readable intermediate tokens do not establish causal use.

Evidence I can do research.
I wrote Operational Proto-Introspection in Looped Language Models using a laptop GPU. I also issued an erratum to my earlier Ouro paper after an audit reduced its headline result from 95.2% to 63.9%. In One Concept, Multiple Geometries, spelling controls killed my first, apparently convincing result. I have some practice being disappointed by my own experiments. I also caught DLCM Table 2’s averaging error (+1.01 improvement, not +2.69); Xingwei Qu confirmed it by email and said they would correct it.

Research reference.
Associate Professor Goran Đambić, PhD, Head of the University Department of Software Engineering at Algebra Bernays University. He helped with the paper I submitted to ICLR, and we have discussed my work in detail. In OPI, I credit him with suggesting the experiment on early-layer readability across recurrent passes.

Tools.
Claude Code and Codex for implementation, analysis, review and editing, including this application rewrite. Three agent audits are retained; they caught real errors. I did not manually recompute every statistic.
