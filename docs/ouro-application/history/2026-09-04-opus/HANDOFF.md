# Handoff — Jacobian lenses across recurrent depth in Ouro-2.6B

Written 2026-09-04 after the rented-GPU run failed. Read this first; it supersedes
the launch instructions in `HANDOFF_B300.md`, which are still correct about the
science but must not be followed until the incident items below are closed.

## 1. Where the project actually stands

The core project is **complete and audited on local hardware**. Every item of the
original scope was delivered: the recurrent recorder, all five validation
milestones, the eventual-exit lens, the full local-exit lens family, the
known-intermediate evaluation against both baselines, the cross-loop analysis,
the local-versus-eventual comparison, and the write-up.

The result did not need the rented run to exist. What the rented run was for was
raising the fitting set from 80 prompts to 100 and then toward 1000, which one of
the audits showed is a real test rather than a formality.

Read `RESULTS.md` (455 lines) for the findings and `RESEARCH_LOG.md` (502 lines)
for how they were arrived at, including every claim that was retracted.

### The headline, in one paragraph

A Jacobian lens targeting Ouro's **final** recurrent exit reads essentially
nothing out of a first-loop state, on both tasks and every metric, even though
the latent intermediate is present there and is read out by a vanilla logit lens,
by a Jacobian lens targeting the **local** exit, and by a supervised probe. On
multihop the deficit persists through every loop before the last. The failure is
not an artifact of magnitude, of fit size, or of Anthropic's estimator, each of
which was attacked separately. The mechanism is that individual prompts transport
a loop-1 state as strongly as a loop-4 one (mean single-prompt Jacobian norm 0.708
against 0.715) but do not agree on a direction, so the averaged map the lens
depends on is left with almost nothing to carry.

## 2. What is verified, and by whom

Three independent Opus auditors ran against a frozen claim set. Their full notes
are preserved in `docs/jlens/peer_audits/` (1735 lines) alongside the claim set
they were given. Summary of what changed:

| claim | outcome |
|---|---|
| instrumentation, index map, exit equality | HOLDS, confirmed causally by an independent perturbation test |
| the main result (eventual-exit lens blind at loop 1) | HOLDS, survived attacks on magnitude, position mismatch, and estimator reduction |
| "J-lens never beats the logit lens" | WEAKENED to "never *significantly* beats"; asserted without an interval originally |
| fit size does not matter | WEAKENED; hit@10 was floored, ranks improve with data in 6 of 8 cells, only multihop loop 1 is genuinely flat |
| cross-loop horizon effect | HOLDS on multihop and survives Bonferroni; the state effect I predicted does not; arithmetic is noise |
| transport-norm ratio | WEAKENED; the two-point estimator was unstable and silently clipped, replaced with raw norms |
| supervised probe comparison | REWRITTEN twice; the probe is data-limited after all |
| whitening rescues loop 3 | REFUTED before it entered the paper |

Two claims I wrote were wrong and were caught by peers: that the transport-norm
estimates agreed across fits, and that the probe was not data-limited. One claim
a peer made was wrong and was caught by another peer. The log records each.

## 3. What is on disk

```
src/ouro_jlens/          22 modules and scripts; recurrent.py is the core
docs/jlens/RESULTS.md    the write-up
docs/jlens/RESEARCH_LOG.md   decisions, retractions, commands
docs/jlens/peer_audits/  the three auditors' notes plus the frozen claim set
utilities/tests/unit/    20 passing CPU tests across three files
artifacts/jlens/lens/    15 GB: exit3 at 8/24/32/56/80 prompts, exit0/1/2 at 24
artifacts/jlens/eval/    7 evaluation directories, both readout positions
artifacts/jlens/probe/   n80_v2 is the current one; the others predate a leak fix
artifacts/jlens/checkpoints/exit_divergence.json   base vs Thinking vs RLTT
```

Nothing is committed to git. Nothing was ever pushed to a remote.

## 4. Open questions, in priority order

1. **Does the loop-1 deficit close with a large fit?** The only cell flat at both
   the threshold and the rank level is multihop loop 1. A 1000-prompt eventual-exit
   lens is the test. This is what the rented run was for.
2. **Is the estimator's position reduction implicated?** An A/B at 3 prompts said
   no, but the diagonal arm had 3 position samples against the summed arm's 93,
   and at that sample size the summed estimator itself loses up to 60 percent of
   its readout. A 2×2 design that gets both reductions from one backward pass is
   described in `peer_audits/peer3_NOTES.md` and costs no extra GPU.
3. **Post-training and the monitor gap.** Divergence of the first-loop exit from
   the final output falls monotonically, base to Thinking to RLTT: on arithmetic
   4.24, 2.07, 1.78 nats. But top-1 agreement does not improve at all, so the
   drop may be distributional sharpness rather than alignment. Entropy and a
   bounded divergence were added to `checkpoints.py` but that rerun never
   completed. Run it; it is four minutes per checkpoint on the local card.
4. **Huginn** is the clean control for the mechanism claim, being recurrent with
   no per-step head. 15 GB, too big for the local 12 GB card.

## 5. The rented-run incident, and what must change before renting again

A B300 was rented at 01:33 on 2026-09-04 at $7.89/hr. All five validation
milestones passed **bit-exactly** on that hardware, which was worth learning. The
benchmark gave 29 seconds per prompt at batch 32, seven times the local card, and
showed the job is compute-bound at that batch, so co-residency of several
checkpoints would not have given proportional speedup.

Then the run was lost. $52.74 spent, nothing retrieved. Causes, in order of
importance:

1. **Nobody watched it.** The driving session went idle for eight hours after
   confirming the first shard was running. No scheduled wake-up, no monitor.
2. **The kill switch did not survive disconnection.** A detached 4.5-hour timer
   was armed on the pod and confirmed *running* two seconds later. It was never
   confirmed to survive the SSH session closing, which is the only failure mode it
   existed for. It did not.
3. **No retrieval step.** The fit was driven by an inline script with no upload.
   Results existed only on ephemeral container disk. The eventual-exit lens at 100
   prompts almost certainly completed around 02:25 and was destroyed with the pod.
4. **No balance floor.** Nothing noticed the account draining to zero.

Before any future rental, all four must be closed:

- retrieval first: copy each shard back the moment it merges, so the pod never
  holds anything unique;
- a balance poll that terminates the pod on breach of a floor;
- a kill switch proven by disconnecting and then re-checking it is alive;
- a scheduled self-wake so an idle session cannot leave a billing machine alone.

The run itself is cheap: about 90 minutes and $12 for all four exit targets at
100 prompts. The waste was entirely in the 5.8 idle hours, roughly $46.

## 6. Next steps if you want to finish this

Local, free, and worth doing first:

```bash
venv/bin/python src/ouro_jlens/checkpoints.py --out artifacts/jlens/checkpoints
```
That settles open question 3 with the scale-free measures already added.

Rented, once the four incident items are closed:

```bash
venv/bin/python src/ouro_jlens/pod.py create        # guards: balance floor, duplicate pod
# then, on the pod, in order, with retrieval after each stage:
bash src/ouro_jlens/setup_b300.sh
python src/ouro_jlens/validate.py                   # hard gate, exits non-zero on failure
python src/ouro_jlens/bench.py 128 8 32 64
bash src/ouro_jlens/run_b300.sh 100 32
venv/bin/python src/ouro_jlens/pod.py destroy all   # panic button
```

The supervised probe stays local; use `probe_cv.py`, not `probe.py`.

## 7. If you submit this as it stands

It is a complete, self-contained negative-plus-mechanism result on 80-prompt
lenses, with an unusually thorough audit trail and an honest record of six
retracted claims. The main caveat to state up front is fit size: the paper's own
lenses use 1000 prompts and ours use 80, and one of our own findings says that
matters everywhere except the cell the headline rests on.
