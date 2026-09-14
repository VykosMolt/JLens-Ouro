# Form answers (drafts; keep short and concrete)

Project in one paragraph.
I tested Anthropic's Jacobian lens on Ouro-2.6B, a looped transformer that
runs 48 layers four times, against the logit lens and a supervised probe on
the released multihop and arithmetic intermediates. The logit lens beats the
Jacobian lens at every loop before the last on multihop (paired difference
-0.36, -0.48, -0.41 at loops 1 to 3), and a lens that reads the final exit
sees almost none of what an earlier loop is doing (KL 4.1 to 6.3 nats,
last-eight-layer top-1 agreement at most 0.033). At the primary readout
position that gap does not steadily shrink; one token earlier KL does, but
top-1 agreement remains at most 0.005.

Most surprising number.
Multihop Jacobian-minus-logit excess of -0.483 at loop 2, interval
[-0.583, -0.382]. I honestly expected the Jacobian lens to win everywhere.

Biggest limitation.
One model, one 100-prompt fit per exit, eight unadjusted intervals, and the
mechanism (shared-head training at every loop) is a hypothesis I have not
tested against a matched model.

Evidence I can do research.
1. Operational Proto-Introspection in Looped Language Models
   (arXiv 2607.18553), done on one laptop GPU, plus a self-issued erratum on
   my earlier paper (2604.09870) after a project-wide audit cut its headline
   number from 95.2% to 0.639.
2. I found a reporting error in Table 2 of DLCM (2512.24617): the displayed
   scores average to +1.01, not the reported +2.69. The authors confirmed
   and are correcting it.
3. One Concept, Multiple Geometries (September 2026), where the first
   result was a convincing false positive that the controls killed.

Tools.
Claude Code (Fable) for the recorder, estimator integration and analysis;
Codex for the rental controller rebuild; three independent reviewer agents.
The headline numbers were checked by independent reviewers against the
analysis code. I did not manually recompute the statistics.
