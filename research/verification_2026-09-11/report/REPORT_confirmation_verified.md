# Ouro new-item confirmation: verified report

Written 11 September 2026 from the [original report](../../confirmation_2026-09-09/REPORT.md), which is unchanged, and from the machine-readable records linked below. **The numerical verification in this version is post-confirmation and was not prospectively registered.** It changes no input, estimator, score, analysis rule or recorded result. Paths below are relative to this file unless absolute.

**The prospective confirmation supports the preselected loop-4 advantage on the frozen 160-question population, and the numerical verification leaves it intact.** Ouro fit01 exceeded the raw logit lens by **+0.23188** excess hit@10 across loop 4, physical layers 26–37, with a 95% dependency-group bootstrap interval of **[+0.16289, +0.32898]**. Replaying the scorer from the retained RTX 5090 states and banks reproduces every saved rank array exactly. Exact FP64 arithmetic over the same stored tensors gives +0.22928 [+0.16170, +0.32517]. States regenerated from the prompts on a different GPU give +0.23608 [+0.16521, +0.33769].

1. **Did the fixed loop-4 advantage generalize?** On this curated population and this fit, yes: +0.232 [+0.163, +0.329], against a discovery endpoint of +0.189. The two are not pooled.
2. **Source of the measured gain.** Higher intended-concept recovery: fit01 0.37917 against raw 0.14427, a difference of +0.23490 [+0.16615, +0.33160]. Control hits rose slightly (+0.00301 [+0.00146, +0.00442]), so the excess does not come from fewer control hits.
3. **Did the early-loop deficit generalize?** Yes. All six early-loop contrasts are negative (−0.088 to −0.611), and their simultaneous 95% intervals exclude zero on every numerical path tested.
4. **Did the untested scorer shapes change the result?**
   - **Packing:** on a GPU of the same architecture running the confirmation's software, 160-, 190-, 191- and 192-row head calls give bit-identical logits. Single-row calls change deep ranks but no endpoint quantity.
   - **Precision:** against exact FP64 arithmetic, the executed BF16 path raised the primary by 0.00260.
   - **Deployed GPU:** parity on an RTX 5090 was not re-measured for layouts the run did not execute.
5. **What is supported or unresolved?**
   - **Supported:** transfer of fit01's fixed-band advantage and of the early-loop deficit to this curated mixture.
   - **Borderline:** "diagonal minus raw includes zero" does not hold on every path (see Secondary contrasts).
   - **Not established:** absence from pretraining; the model's use of the annotated intermediates; difficulty matched to discovery; generality beyond this population and fit.
   - **Unsupported:** historical Huginn remains unsupported.

## Scope

- Concept recovery is not answer accuracy or demonstrated causal use.
- Calibration-fit similarity is not formal equivalence: fit01 minus fit02 is +0.006 [−0.007, +0.020].
- The 28 unequal dependency groups do not provide 160 independent observations; the effective group count is 8.63.
- New evaluation concepts are not necessarily unseen pretraining facts.
- This is transfer under relation shift, not a difficulty-matched replication.
- Target and position sensitivity concerns the estimator; it does not establish the model's reasoning mechanism.
- Three component facts overlap known calibration text; none is a reused complete two-hop question or discovery intermediate identity.
- No general J-Lens superiority or monitoring-superiority claim follows.

## Confirmation result (primary endpoint)

Frozen [analysis](../../confirmation_2026-09-09/results/ouro_confirmation_20260911_fixed160_native1/analysis/analysis.json) of the accepted final payload. The primary excess difference is the single inferential endpoint; component intervals are descriptive.

| Quantity | Estimate | 95% dependency-group interval |
|---|---:|---|
| Primary: fit01 minus raw excess hit@10 | +0.23188 | [+0.16289, +0.32898] |
| fit01 intended recovery | 0.37917 | [0.30444, 0.46563] |
| fit01 control recovery | 0.00344 | [0.00182, 0.00507] |
| Raw intended recovery | 0.14427 | [0.10455, 0.17870] |
| Raw control recovery | 0.00043 | [0.00018, 0.00075] |
| Intended difference | +0.23490 | [+0.16615, +0.33160] |
| Control difference | +0.00301 | [+0.00146, +0.00442] |

All 160 questions were eligible, across 28 dependency groups. Two sensitivity checks:
- **Item resampling** gives [+0.17971, +0.28706].
- **Leave-one-group-out** estimates range from **+0.19730 to +0.24875**.

The interval uses 20,000 resamples of whole dependency groups (seed 2026090901), retaining equal item weights.

## Numerical verification of the scoring path

**How the scorer runs.**
- **Transport:** for each item it applies the bank in FP32, one `einsum` over all scored columns.
- **Unembedding:** it casts the rows to BF16 and applies the final RMSNorm (computed in FP32, cast back) and the BF16 lm_head in one [C, 2048] call.
- **Row counts per call:** 192 for raw, fit01 and fit02; 190 for penultimate; 191 for sampled_sum and diagonal.
- **Ranking:** a descending argsort over the full 49,152-token vocabulary. A name's rank is the minimum over its alias forms, and hit@10 means rank < 10.
- **Ties:** in the saved sorts, tied logits are ordered by ascending token ID.
- **Stored indices:** zero-based, with virtual = 48 × (loop − 1) + (physical − 1). The primary band is virtual 169–180.

**Before this pass, no one had recomputed logits from the transported states,** and the development diagnostic never measured the 190–192-row layouts ([scoring-path review](../reviews/scoring_path/SCORING_PATH_REVIEW.md)).

**Evidence.** The frozen transport, readout and rank code was replayed from the retained 5090 states and the four hash-verified banks. The replay ran on a local RTX 5070 Ti Laptop GPU with the confirmation's torch build and an identical precision dictionary, over all 160 items, six arms and every scored column:
- the replay of the executed layout reproduced every saved array bit for bit;
- that includes ranks, top-10 IDs, top-1, KL diagnostics, sampled logits and exit logits;
- a second GPU of the same architecture is independent evidence, not the deployed runtime.

Each path below changes only packing or arithmetic ([metrics](../metrics/VERIFICATION_METRICS.json)).

| Path | Primary | Change | 95% group interval | fit01 intended / raw intended | Band hit changes (gained/lost) |
|---|---:|---:|---|---|---|
| Accepted, RTX 5090 | +0.23188 | — | [+0.16289, +0.32898] | 0.37917 / 0.14427 | — |
| Executed layout replayed | +0.23188 | 0 | identical | identical | none |
| 160-, 190-, 191-, 192-row packing | +0.23188 | 0 | identical | identical | none (logits bit-identical) |
| Single-row head calls | +0.23188 | 0 | identical | identical | none |
| FP32 with TF32 off; FP64 head; full FP64 | +0.22928 | −0.00260 | [+0.16170, +0.32517] | 0.37812 / 0.14583 | fit01 intended 0/2, controls 8/7; raw intended 3/0, controls 1/0 |
| States regenerated from prompts (level 3) | +0.23608 | +0.00419 | [+0.16521, +0.33769] | 0.38437 / 0.14531 | fit01 intended 13/3, controls 9/10; raw intended 3/1, controls 3/0 |

**Single-row calls.**
- They change 1.48–1.62 million of about 1.5 billion logits per arm, with a maximum difference of 0.125 (0.25 for raw).
- One control hit changed, in the penultimate arm and outside every endpoint region.
- Top-10 membership changed in 2–11 rows per arm.

**FP32 and FP64 paths.**
- They change every row: RMS 0.006–0.007, with a maximum difference of 0.29.
- FP64 on the CPU gives the same ranks and hits as FP64 on the GPU; value differences are at most 9 × 10⁻¹⁴.
- Under FP64, control differences move by less than 0.00001.

**Ties and boundary margins.**
- Some band hits for fit01 and raw are decided by tie order alone:

  | Arm | Intended cells (counted as hits) | Control cells (counted as hits) |
  |---|---:|---:|
  | fit01 | 9 (6) | 26 (9) |
  | raw | 5 (0) | 3 (1) |

- Resolving every tie adversarially gives a primary between **+0.22604 and +0.23352**. If every logit could move by up to 1/32 or 1/8 in the adversarial direction, the primary could lie between +0.22013 and +0.24408, or between +0.18302 and +0.27263. These are worst-case bounds, not estimates.
- Of fit01's 728 intended hit cells in the band, 24 lie within 1/8 of the top-10 boundary.

**Independent checker.** Written from a specification without producer code, it rebuilt the saved analysis with a maximum difference of 0.0. In isolated fixtures, the unperturbed copy passed and all 13 deliberate indexing, weighting, control, padding and seed perturbations were detected. It also reproduced the analysis of all 27 replayed and bounded paths ([checker](../checker/README.md)).

**Native output (A).** This was run on the local GPU, with states captured in the same run.
- **Exact matches:** native final logits equal unembedding of the captured final state exactly for all 160 items in three layouts: the scorer's [192, 2048] call, the [160, 2048] exit call and the [1, S, 2048] sequence.
- **Single-row call:** one [1, 2048] call differs in up to 50 logits, with no top-10 or hit change.
- **Regenerated states:** these match the retained 5090 states bit for bit for 37 of 160 items (maximum difference 1.25 elsewhere), and 154 of 160 greedy continuations match.
- **Not measured:** native output for these 160 items on an RTX 5090.

## Secondary contrasts

The twenty secondary contrasts share simultaneous 95% intervals (max-t quantile 3.2255 accepted, 3.2091 FP64, 3.2143 regenerated). Neither a favorable secondary nor a relocated peak can replace the primary endpoint.

| Contrast | Accepted | FP64 | Regenerated states |
|---|---|---|---|
| fit01 − raw, loop 1 fixed-layer mean | −0.08827 [−0.13642, −0.04013] | −0.08893 [−0.13664, −0.04122] | −0.08801 [−0.13525, −0.04078] |
| fit01 − raw, loop 1 any layer | −0.44407 [−0.59029, −0.29785] | −0.44422 [−0.58950, −0.29895] | −0.43758 [−0.58359, −0.29157] |
| fit01 − raw, loop 2 fixed-layer mean | −0.14858 [−0.19541, −0.10175] | −0.14820 [−0.19402, −0.10237] | −0.14858 [−0.19565, −0.10152] |
| fit01 − raw, loop 2 any layer | −0.57366 [−0.72008, −0.42723] | −0.57381 [−0.71942, −0.42821] | −0.57358 [−0.71956, −0.42759] |
| fit01 − raw, loop 3 fixed-layer mean | −0.15587 [−0.22404, −0.08770] | −0.15560 [−0.22372, −0.08747] | −0.15574 [−0.22409, −0.08739] |
| fit01 − raw, loop 3 any layer | −0.61084 [−0.79591, −0.42577] | −0.61108 [−0.79531, −0.42684] | −0.61100 [−0.79495, −0.42704] |
| fit01 − raw, final third (176–191) | +0.11543 [+0.04673, +0.18413] | +0.11735 [+0.04924, +0.18546] | +0.11817 [+0.05015, +0.18620] |
| fit01 − raw, physical layer 32 (virtual 175) | +0.24684 [+0.10717, +0.38650] | +0.24691 [+0.10800, +0.38583] | +0.26566 [+0.09980, +0.43153] |
| fit02 − raw, primary band | +0.22558 [+0.09523, +0.35593] | +0.22296 [+0.09684, +0.34907] | +0.23184 [+0.09722, +0.36645] |
| fit01 − fit02, primary band | +0.00630 [−0.00727, +0.01988] | +0.00632 [−0.00835, +0.02099] | +0.00424 [−0.00887, +0.01734] |
| Penultimate − raw, primary band | +0.08804 [+0.00759, +0.16849] | +0.08755 [+0.00795, +0.16714] | +0.08910 [+0.00716, +0.17103] |
| Penultimate − raw, matched final third | −0.04872 [−0.08742, −0.01002] | −0.04693 [−0.08620, −0.00766] | −0.04915 [−0.08896, −0.00935] |
| Sampled-sum − raw, primary band | +0.21724 [+0.09155, +0.34293] | +0.21408 [+0.09060, +0.33757] | +0.22094 [+0.09034, +0.35154] |
| Sampled-sum − raw, matched final third | +0.11439 [+0.03956, +0.18923] | +0.11648 [+0.04184, +0.19111] | +0.11401 [+0.04071, +0.18732] |
| Diagonal − raw, primary band | +0.05952 [−0.00340, +0.12244] | +0.05797 [−0.00400, +0.11994] | +0.05951 [+0.00037, +0.11866] |
| Diagonal − raw, matched final third | +0.00171 [−0.04664, +0.05005] | +0.00210 [−0.04588, +0.05009] | +0.00089 [−0.04830, +0.05008] |
| fit01 − penultimate, primary band | +0.14384 [+0.07699, +0.21069] | +0.14173 [+0.07637, +0.20709] | +0.14698 [+0.07789, +0.21607] |
| fit01 − penultimate, matched final third | +0.16917 [+0.09716, +0.24118] | +0.16958 [+0.09885, +0.24031] | +0.17275 [+0.09926, +0.24623] |
| Sampled-sum − diagonal, primary band | +0.15772 [+0.05796, +0.25748] | +0.15611 [+0.05743, +0.25479] | +0.16143 [+0.05674, +0.26611] |
| Sampled-sum − diagonal, matched final third | +0.11268 [+0.04515, +0.18022] | +0.11437 [+0.04610, +0.18264] | +0.11312 [+0.04367, +0.18258] |

Packing alternatives leave every secondary unchanged.

**Early-loop contrasts.** Every sign and every zero-exclusion status of the six early-loop contrasts holds on every path.

**Estimator controls.**
- **Stable across paths:**
  - fit01 exceeds the penultimate-target control;
  - sampled-sum aggregation exceeds diagonal-only;
  - penultimate minus raw stays positive, with a lower bound of +0.007 to +0.008.
- **Borderline:** diagonal minus raw in the primary band has a lower bound within 0.004 of zero on every path, and it excludes zero with regenerated states. State it as borderline rather than as including zero.

## Frozen new-item test

The following are fixed by the [freeze receipt](../../confirmation_2026-09-09/FREEZE.json):
- the [prospective plan](../../confirmation_2026-09-09/PROSPECTIVE_PLAN.md);
- the [benchmark](../../confirmation_2026-09-09/benchmark/final_candidates.json);
- the eligibility and control population;
- the analysis and executable sources.

The primary estimator is the original verified Ouro fit01, with its original target, position reduction, normalization, aliases, readout token and full-sort rank semantics. fit02 and the surviving target and position controls are fixed secondary comparisons. There is no refitting, selection by new score, ensemble or sample extension.

**Population.**
- **Items:** 160 eligible questions, two for each of 80 intermediate identities absent from the discovery benchmark.
- **Domains:** 10 domains of eight concepts each: arts and music, astronomy, biology, chemistry, countries, environment and geology, historical people, landforms, SI units and technology.
- **Labels and controls:** every item has one scored label and 79 equally weighted controls, namely all other catalogue names.
- **Review:** an independent reviewer checked both factual hops, novelty and control hygiene before outcomes. No absence from base-model pretraining is claimed.

**Relation mixture.** The reviewed annotation of the requested relation, at typed-predicate granularity, gives:

| Relation status | Items | Definition |
|---|---:|---|
| Not represented | 96 | the requested relation is not found at this granularity in the 93 discovery prompts |
| Familiar requested family | 45 | includes a broad geographic, orbit, count or season relation applied to a changed entity class |
| Familiar fact relation recombined | 19 | a known component, location, release or etymology relation, used in another hop or with a changed requested slot |

These are annotations, not 160 independent relation families ([benchmark record](../../confirmation_2026-09-09/cloud_leases/attempt_06/handoff/accepted/ouro_confirmation_20260911_fixed160_native1/ff7c0769b2b0dbe2e53b120daf7f0729434fe65e54c013288771f5052ffd4cef/results/benchmark.json), `design.relation_balance`).

**Dependency groups.** The 28 groups have sizes 42, 24, 16, 10, 10, 6, four groups of 4 and eighteen groups of 2. The weighted effective group count is 160² / Σ size² = 8.63. Conservative planning scenarios did not achieve the initial precision target of about ±0.10.

The old five-fit +0.1872 estimate describes calibration stability on the discovery population. The original fit01 discovery endpoint is +0.18907; neither is pooled with the confirmation.

## Freezes and deviations

| Record | Frozen (UTC) | SHA-256 | What it fixed or changed |
|---|---|---|---|
| [Original freeze](../../confirmation_2026-09-09/FREEZE.json) | 9 Sep 22:40:58 | `8148c2ab0bb62e96…` | Plan, benchmark, population, estimator, scoring, analysis and sources |
| [precision_v1](../../confirmation_2026-09-09/corrections/precision_v1/FREEZE.json) | 10 Sep 10:44:52 | `8f05445eae5505d9…` | Removed two redundant TF32 precision setters from `evaluation/worker.py` after attempt03's strict precision check rejected them. The runtime still requires the full recorded precision dictionary before readouts. |
| [native_check_v1](../../confirmation_2026-09-09/corrections/native_check_v1/FREEZE.json) | 10 Sep 22:56:18 | `fc74b0c244e64a76…` | The development check unembeds the step-3 final-norm input at the head's input shape [1, S, 2048], instead of one residual vector; exact equality is unchanged. It also changed the run ID, the frozen time and the run-spec and output-contract provenance records. |

**Deviations and post-hoc changes, as recorded:**
- **native_check_v1 run budgets:** setup went from 12,300 to 12,800 s, compute from 3,600 to 1,200 s and preservation from 1,800 to 900 s. These are non-scientific lease settings, but the correction record's scope does not list them.
- **native_check_v1 review:** the change was not independently reviewed before the run. The post-run review in this pass found it confined to the stated verification behavior and provenance ([review](../reviews/native_check_and_pin/REVIEW.md), pass with findings).
- **Corrections were frozen before outcomes,** but relative to the original freeze, `worker.py` differs because of precision_v1.
- **Attempt05 setup deadline:** after the user's instruction "Extend the deadline", root extended only attempt05's setup deadline, by 15 minutes; independent review passed.
- **Checker pin:** the frozen independent checker stopped at its pin of the original run spec. `reviews/primary_numerical_check_native1.py` adds a pin for the corrected spec. It was written after the run's outcomes had been computed.
  - It then matched all 231 compared quantities (maximum difference 8.9 × 10⁻¹⁶).
  - It reads the run ID, which differs; every scientific field it reads is identical to the original freeze.
  - This pass's separate checker reproduces the endpoint without it.
- **This verification pass** is post-confirmation and was not prospectively registered.

## Native development gate failure (attempt05)

**The gate.**
- On three historical development items, the worker saves residual states from physical-block hooks, virtual taps and the wrapper forward, plus native final logits and `unembed` of the final residual.
- It requires `torch.equal` for all three comparisons.
- Attempt05's saved development tensors ([forensics](../../confirmation_2026-09-09/reviews/attempt05_native_failure_forensics.md)) give:

| Comparison | Shape | Differing elements | Maximum absolute difference |
|---|---|---:|---:|
| Virtual vs physical-hook states | [3, 192, 2048] | 0 of 1,179,648 | 0 |
| Virtual vs wrapper states | [3, 192, 2048] | 0 of 1,179,648 | 0 |
| Native vs unembedded logits | [3, 49152] | 112 of 147,456 | 0.03125 |

**The logit differences.**
- The mean absolute logit difference is 6.05 × 10⁻⁶, and 106 of the 112 differences are one BF16 code step.
- Top-100 token order is identical for every item. Deeper orderings differ, the earliest at zero-based rank 270 (mars-color).

## Native-logit diagnostic

A bounded diagnostic reran the development items on the same GPU type and frozen runtime, under analysis rules fixed before any outcome ([contract](../../confirmation_2026-09-09/diagnostics/native_logit_v1/CONTRACT.md), [rules](../../confirmation_2026-09-09/diagnostics/native_logit_v1/run/PRE_RUN_RULES.md)).
- **How the native forward reaches the head:** it normalizes each recurrent step's state, gathers each token's exit-step state and applies lm_head to the whole sequence. For every item, the last token's state came from step 3 (zero-based).
- **Retained recomputations:** counts below are from [comparisons.json](../../confirmation_2026-09-09/cloud_leases/diagnostic_native_logit_v1/handoff/accepted/ouro_native_logit_diagnostic_v1/f9751036b1fa746fc8a37287c755093cde92238ead58347f78869eeb949063e6/results/diagnostic/comparisons.json), for carnival-ocean / amazon-language / mars-color. Every recomputation agreed with its repeat.
- **Local reproduction:** this pass applied the same operations to the diagnostic's retained inputs on the local GPU.

| Recomputation | RTX 5090: differing elements vs native | Local GPU, from retained inputs |
|---|---|---|
| Final norm at [2048], [1,2048], [1,1,2048], [S,2048], [1,S,2048], [148,2048], [160,2048] | 0 / 0 / 0 | 5090 outputs reproduced exactly |
| lm_head on one row: [2048], [1,2048], [1,1,2048] | 39 / 28 / 45 | 5090 outputs reproduced exactly, including the differences |
| lm_head on several rows: [S,2048], [1,S,2048], [148,2048], [160,2048] | 0 / 0 / 0 | 5090 outputs reproduced exactly |
| Norm then lm_head at [1,1,2048] | 39 / 28 / 45 | 5090 outputs reproduced exactly |
| Norm then lm_head at [1,S,2048], [160,2048] | 0 / 0 / 0 | 5090 outputs reproduced exactly |
| Attempt05's check: unembed of the single final residual | 39 / 28 / 45 | not rerun |
| lm_head, and norm then lm_head, at [190,2048], [191,2048], [192,2048], target row first or last | not measured | 0 / 0 / 0 |

**What the diagnostic showed.**
- At [148, 2048] and larger, the target row was placed first or last among the item's own positions, cycled.
- Single-row lm_head is the localized source of attempt05's mismatch; the final norm is exact at every tested shape.
- The underlying kernel difference is not identified.
- On the 5090, the conclusions cover the three items and the measured shapes.

## Recovery and preservation

**Recovery before the purge.** The [artifact ledger](../../confirmation_2026-09-09/artifacts/RECOVERY_LEDGER.json) records seven of sixteen historical estimator binaries surviving intact. Ouro fit02's bank was reconstructed from its intact checkpoint and matches the historical SHA-256 `101f31db6aa56d97fbae7ecb3e2f241ef1805000484d0e22f5eba5fc0b9acde8`. Independent NumPy conversion checked 4,001,366,016 FP16 entries across five estimator arms with zero bit differences. This checks the recorded checkpoint-to-bank conversion, not the scientific validity of every readout interpretation.

**The purge.** The storage purge deleted 189 files (98,410,290,077 bytes) from the application evidence, the refit round and this round. Its manifests record path, size and SHA-256 for each file. The independent [artifact audit](../reviews/artifact_audit/ARTIFACT_AUDIT.md) reconciles them ([ledger](../preservation/PRESERVATION_LEDGER.json)):

| Class | Files | Examples |
|---|---:|---|
| Verified duplicate | 126 | re-materialized 64 MiB upload chunks, the Ouro model weights, the retrieved fit01 lens |
| Reconstructible from verified inputs | 1 | the partial fit02 lens (a prefix of the reconstructed bank) |
| Unique and lost | 43 (33 distinct contents) | all four Ouro FP32 N100 checkpoints, the truncated Huginn checkpoint and both seed caches, every copy of the historical Ouro common cache, optimization binaries, application-era lens files and probe caches |
| Unresolved | 19 | application-era files recorded in the private Hub repository `Vykos/ouro-jlens-results`; no credential on this machine, so not checked |

**Surviving banks.** Four bank files serve five bank arms and six evaluated arms; each hashes to its recorded value and loads:

| Bank file | Arms | Transport |
|---|---|---|
| fit01 | fit01 | target 191, identity row 191 |
| fit02 (reconstructed) | fit02 | target 191, identity row 191 |
| penultimate | penultimate | target 190 |
| positions | sampled_sum and diagonal | `J['sampled_sum']`, `J['diagonal']`, target 191 |

**Bundle.** A preservation bundle holds the banks, the accepted payload, the model snapshot, every file under 50 MiB of both rounds, and this pass's records:
- **Size:** 7,380 core files (14.97 GB).
- **Restoration check** ([result](../preservation/RESTORATION_CHECK.json)), from bundle files alone:
  - every file rehashed;
  - the frozen code archive matched its freeze;
  - all banks loaded;
  - rescoring raw and fit01 reproduced the saved ranks exactly;
  - the frozen analysis rerun equalled the saved analysis.
- **Single-copy risk:** the bundle is on the same physical disk as every other copy, so it protects against accidental deletion, not device loss.

**Evidence levels for the central claims.**
1. **Saved scores to tables:** available.
2. **Rescoring from verified banks and retained states:** executed for all six arms, exact on the local GPU.
3. **Replay from the frozen prompts:** executed on the local GPU. Its states are not bit-identical to the 5090's, and the endpoint moved +0.00419.
4. **Fitting reproduction:** unavailable for the original estimators, because every FP32 checkpoint was purged; a new fit would be a new estimator.

## Run timeline and paths

| UTC | Event | Pod | Conservative bound |
|---|---|---|---:|
| 9 Sep 22:40:58 | Original freeze | — | — |
| 9 Sep 22:41:49 – 23:01:00 | Attempt01 cancelled during input transfer | `xugytsithv80ib` | $0.225884 |
| 9 Sep 23:20:01 – 10 Sep 00:31:24 | Attempt02: transfer interrupted; export approval did not arrive; cancelled | `jbuyae4289kcje` | $0.840481 |
| 10 Sep 08:36:33 – 10:41:09 | Attempt03: precision preflight failed after setup | `8d5naxkec5lnaf` | $1.467093 |
| 10 Sep 10:44:52 | precision_v1 frozen | — | — |
| 10 Sep 10:47:31 – 14:36:58 | Attempt04: setup admission expired (14:12:31) before destination-specific export approval | `bg2gfkxq5fjvzc` | $2.701677 |
| 10 Sep 14:55:01 – 18:32:19 | Attempt05: development native check failed; 19 remote files preserved before teardown | `i7zq14rj0f0ief` | $2.558589 |
| 10 Sep 21:50:00 – 21:56:50 | Native-logit diagnostic | `mq25wyg0nf9z36` | $0.080620 |
| 10 Sep before 22:07:48, and 22:17:17 | Storage purges: application evidence and refit round, then this round | — | — |
| 10 Sep 22:56:18 | native_check_v1 frozen | — | — |
| 10 Sep 23:02:10 – 11 Sep 01:06:59 | Attempt06, the confirmation run | `gj9dwkkqtb3e14` | $1.469640 |

**Attempt06 in detail.**
- **Upload:** billing started at 23:02:10. The eight-stream upload helper began at 23:03:34, and the four bank files were verified on the worker at 23:48:00, 00:10:57, 00:34:27 and 00:56:26. BANKS_READY was published at 00:56:51.
- **Development check:** setup completion was observed at 00:59:00. The corrected development check passed all three exact equalities, and after the controller accepted that stage, new-item scoring was authorized at about 01:00.
- **Scoring:** the 160 questions were cached and six arms scored by 01:04:09.
- **Retrieval and shutdown:** the controller retrieved, verified and accepted seven stages and the complete final manifest, then deleted the pod. Termination was verified at 01:06:59.

**Paths.**
- **Accepted payload:** `confirmation_2026-09-09/cloud_leases/attempt_06/handoff/accepted/ouro_confirmation_20260911_fixed160_native1/ff7c0769b2b0dbe2e53b120daf7f0729434fe65e54c013288771f5052ffd4cef/`
- **Analysis and figures:** `confirmation_2026-09-09/results/ouro_confirmation_20260911_fixed160_native1/`
- **Banks:** `verification_2026-09-11/replay/retained_bank_paths.json`
- **Run specification:** `verification_2026-09-11/RUN_SPECIFICATION.json`

## Spending and shutdown

**Attempt06 transfer.** Attempt06 transferred four bank files totalling 8,002,994,696 bytes (8.00 GB; 7.45 GiB). The transfer took 1 h 52 min 52 s, from helper start at 23:03:34 to the fourth verification at 00:56:26; that averages about 1.18 MB/s over eight streams. The earlier "2–3.5 h" was a pre-run estimate, not a measurement.

**Attempt06 lease.** The lease lasted 2 h 4 min 49 s, which at $0.706438/h gives its $1.469640 bound. Its pod billing record ($0.786277) probably lags: its GPU charge equals about 1.11 hours at $0.69/h. The whole-account balance fell $1.433279 between the pre-creation and final observations.

**Combined bound.** The combined conservative bound is **$23.195828 against $25**, leaving $1.804172. These are bounds, not invoices.

**Workers and this pass.** All seven workers have verified provider shutdown. This verification pass spent nothing: the account showed $4.3725561096, no pods and zero hourly spend at 08:24Z and at 13:34Z ([record](../spending/SPENDING_AND_SHUTDOWN.md)).

**Huginn rerun.** The optional full Huginn verification rerun remains resource-blocked: its frozen $6 admission exceeds the remaining authorization. The purge also deleted its frozen archive's cache, so it can no longer run as frozen.

## Historical Huginn pilot

The historical Huginn pilot's evaluation completed, and J-Lens minus raw was negative in all six fixed summaries in both evaluation seeds. The relative prediction is unsupported, not verified.

**Missing estimator.** Its final bank was never retrieved, and the 1,745,780,736-byte partial checkpoint, of 3,568,444,419 bytes, was never deserialized. The purge then deleted that checkpoint and both seed caches.

**Unverified conversion.** The bank's conversion relation rests only on the producer's assertion. Only saved scores remain (level 1), and any new fit would be a rerun, not the original estimator.
