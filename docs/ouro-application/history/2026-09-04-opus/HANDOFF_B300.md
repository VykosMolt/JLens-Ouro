# B300 handoff

Everything below is scripted and resumable. Nothing here needs the local GPU.

## What the rented run adds

The local lenses use 8 to 80 fitting prompts for the eventual exit and 24 for
each local exit. Anthropic use 1000 and call ~100 usable. The rented run fits
all four exit targets at 100 prompts over all 192 virtual source locations, so
every number in `RESULTS.md` gets a properly sized replacement, and the
fit-size sweep gets its final point.

## Cost and time

| | B300 288 GB | B200 180 GB | H200 141 GB |
|---|---|---|---|
| price per hour, community | 6.94 | 5.98 | 3.59 |
| core plan | 2 to 2.5 h | similar | slower, similar total |

Top up to about $50 for the core plan with margin. Stock showed "low" for B300
and B200 when checked; H200 runs the same script with a smaller `dim_batch`.

## Launch

On the pod, with `HF_TOKEN` in the environment and this repo's `src/ouro_jlens`
plus `artifacts/jlens/data/wikitext_prompts.json` present:

```bash
bash src/ouro_jlens/setup_b300.sh          # deps, jlens @ 581d398, Ouro @ 1ed04250
bash src/ouro_jlens/run_b300.sh 100 32     # validation, bench, all four exits, evaluations
```

To move the code without a git remote, run `src/ouro_jlens/stage_upload.sh`
locally to push a tarball to a private HF repo; `src/ouro_jlens/pod_entry.sh`
pulls it, runs the job and uploads results.

`run_b300.sh` runs `validate.py` first. **If any milestone fails, stop.** The
five milestones are bit-exact locally, so a failure means the environment
differs, not that the science changed. Milestone 5 halves its `dim_batch` and
retries if the card is short of memory, so it cannot abort the run spuriously.

Ask the pod for at least 64 GB of host RAM and 100 GB of disk. Both figures are
measured, not estimated: peak resident memory is 10.1 GB for a four-shard merge
and 17.0 GB for the four-lens evaluation. A 32 GB host would die on the
evaluation. Shard checkpoints are ~3 GB each and are deleted once their shard
completes; at 100 prompts the lenses total 20 GB, of which 16 GB is per-shard
files the merged lenses already contain and which the results upload excludes. It then runs `bench.py`, which prints
real memory and per-pass timing; if peak memory is far below the card, raise
`dim_batch` (fewer backward sweeps, same total FLOPs) and restart. Shards are
skipped once their `.pt` exists, so restarting costs nothing.

## After it finishes

Download `artifacts/jlens/lens/n100/exit*.pt` and run the supervised probe
locally. Its sklearn phase is about ten minutes on one core and would only waste
paid GPU time, so it is deliberately not in the pod script:

```bash
python src/ouro_jlens/probe_cv.py --lens artifacts/jlens/lens/n100/exit3.pt \
    --out artifacts/jlens/probe/n100_cv
```

Use `probe_cv.py`, not `probe.py`. The single-split probe leaks through mirrored
operand pairs and its output is what section 8 no longer reports; `analyze.py
--probe` also expects the single-split layout and will not read a `probe_cv`
directory.

Then rerun the transport-coherence estimate on two disjoint shards, which is
what section 6 of `RESULTS.md` rests on:

```bash
python src/ouro_jlens/lens_convergence.py <shard_A.pt> <shard_B.pt>
```

## What to check in the results

1. `validate.py` milestones all pass. Non-negotiable.
2. Section 5 of `RESULTS.md`: does the eventual-exit lens still read near zero
   at loops 1 to 3 while the local-exit lens does not? This is the main result.
3. Section 7: does the fit loop still dominate the state loop in the cross-loop
   variance share? If the state loop grows at 100 prompts, the horizon story
   needs revising.
4. Section 4: does the Jacobian lens still fail to beat the logit lens? This is
   the most fit-sensitive claim in the document.
5. The fit-size sweep figure should extend cleanly from the local 8 to 80 points.

## Files

| path | purpose |
|---|---|
| `src/ouro_jlens/recurrent.py` | LoopTap recorder, Ouro lens model |
| `src/ouro_jlens/validate.py` | the five milestones |
| `src/ouro_jlens/fit_lens.py` | shard fitting and merging |
| `src/ouro_jlens/evaluate.py` | readout, cross-loop, local vs eventual |
| `src/ouro_jlens/analyze.py` | metrics, contrasts, figures |
| `src/ouro_jlens/probe.py` | supervised reference |
| `src/ouro_jlens/lens_convergence.py` | transport coherence |
| `src/ouro_jlens/fitsize.sh` | fit-size sweep |
| `docs/jlens/RESULTS.md` | write-up |
| `docs/jlens/RESEARCH_LOG.md` | decisions, corrections, commands |
| `utilities/tests/unit/test_ouro_jlens.py` | CPU tests, 6 passing |
