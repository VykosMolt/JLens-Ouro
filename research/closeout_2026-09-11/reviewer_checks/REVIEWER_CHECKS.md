# Reviewer checks on retained confirmation data (post-confirmation)

Executed 2026-09-11 on the accepted payload of `ouro_confirmation_20260911_fixed160_native1` (population sha256 `d7b4efa7…`, benchmark `735816c5…`, readouts `fit01.npz` / `raw.npz` / `fit02.npz`). Plan frozen before computation: [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md). Machine-readable results: [RESULTS.json](RESULTS.json), [PER_ITEM.csv](PER_ITEM.csv), [OVERLAP_LEDGER.csv](OVERLAP_LEDGER.csv), [ANSWER_FORM_COINCIDENCE.csv](ANSWER_FORM_COINCIDENCE.csv), [GROUP_MEMBERSHIP.csv](GROUP_MEMBERSHIP.csv). Code: [run_checks.py](run_checks.py), reusing the frozen `measurement.py`/`analyze.py` unchanged.

These are reviewer follow-up analyses on already-used data. The original primary endpoint (**+0.23188 [+0.16289, +0.32898]**, reproduced exactly here as an anchor: 0.23188291139240508) and the original 20 secondary contrasts are unchanged. The new inferential family N1 has exactly four contrasts (A, C1, C2, C3) with one centered bootstrap max-t 95% interval family (20,000 whole-group draws of the frozen 28 dependency groups, seed 2026091101; max-t quantile 2.6386). Everything else is descriptive; descriptive percentile intervals are labeled as such.

## A. Within-domain controls (pass 4, layers 26–37)

**Structure verified.** `benchmark.json` assigns the 80 concepts to 10 domains of exactly 8 (arts_music, astronomy, biology, chemistry, countries, environment_geology, historical_people, landforms, si_units, technology). For every item the 7 same-domain names are a subset of its frozen 79 controls (asserted). Ranking, labels, accepted forms, item weights, banks and fixed-layer averaging are unchanged; this is a control-subset reaggregation, not a top-10 among eight candidates.

**Is the grouping semantic matching?** No. Four domains hold one semantic type (chemistry: 8 elements; countries: 8 countries; si_units: 8 units; historical_people: 8 people). The others mix types: astronomy (5 planets/dwarf planets + 3 moons), landforms (mountain, desert, two ranges, three rivers, an island), biology (two nucleic acids, three proteins, a cell type, an organelle, a pigment), technology (OS, language, spacecraft, telescope, website, encoding standard, network standard, board), environment_geology (gas layer, two rocks, four weather/geological phenomena), arts_music (five composers, two painters, one author). This is therefore a **within-domain control**, not a same-type control.

| Quantity, pass-4 band | fit01 | raw |
|---|---:|---:|
| Intended recovery | 0.37917 [0.30426, 0.46630] | 0.14427 [0.10417, 0.17958] |
| All-79 control recovery | 0.00344 [0.00180, 0.00505] | 0.00043 [0.00018, 0.00076] |
| Within-domain (7) control recovery | 0.03073 [0.01133, 0.04787] | 0.00379 [0.00106, 0.00709] |
| Excess, all-79 controls | 0.37573 | 0.14384 |
| Excess, within-domain controls | 0.34844 | 0.14048 |

Brackets: descriptive group-percentile 95%.

| Paired difference (fit01 − raw) | Estimate | Interval | Status |
|---|---:|---|---|
| Excess, all-79 controls | +0.23188 | [+0.16289, +0.32898] | original prospective primary |
| **Excess, within-domain controls** | **+0.20796** | **[+0.09130, +0.32463]** simultaneous (N1) | post-confirmation |
| Intended difference | +0.23490 | [+0.16667, +0.33218] descriptive | |
| All-79 control difference | +0.00301 | [+0.00144, +0.00443] descriptive | |
| Within-domain control difference | +0.02693 | [+0.00921, +0.04182] descriptive | |

Item-resampling descriptive 95% for the within-domain difference: [+0.16143, +0.25397].

**Arithmetic check (exact arrays).** Lower bound = intended difference − (79/7)·fit01 all-79 control recovery = 0.23490 − 11.2857·0.00344 = **0.19606**; observed within-domain difference **0.20796 ≥ 0.19606**. The inequality holds for every one of the 160 items (0 violations). The published rounded values give 0.19608. The bound is a bound on the point estimate, not an interval.

**What it means.** J-Lens recovers same-domain names about nine times more often than arbitrary controls (0.031 vs 0.003), so part of what the lens reads is domain or type information; subtracting within-domain controls removes 0.024 of the 0.232 gain. The paired advantage remains positive with a simultaneous interval excluding zero.

**Per-domain descriptive means (16 items each; group counts in the JSON).**

| Domain | fit01 intended | raw intended | fit01 within-domain control | Excess diff, 79 controls | Excess diff, within-domain |
|---|---:|---:|---:|---:|---:|
| countries | 0.635 | 0.089 | 0.037 | +0.543 | +0.510 |
| environment_geology | 0.682 | 0.255 | 0.024 | +0.425 | +0.403 |
| chemistry | 0.589 | 0.250 | 0.086 | +0.331 | +0.254 |
| arts_music | 0.323 | 0.062 | 0.045 | +0.257 | +0.222 |
| biology | 0.323 | 0.078 | 0.004 | +0.244 | +0.241 |
| historical_people | 0.464 | 0.224 | 0.001 | +0.239 | +0.239 |
| landforms | 0.229 | 0.068 | 0.025 | +0.158 | +0.145 |
| technology | 0.370 | 0.292 | 0.000 | +0.078 | +0.078 |
| astronomy | 0.161 | 0.104 | 0.086 | +0.051 | −0.008 |
| si_units | 0.016 | 0.021 | 0.000 | −0.006 | −0.005 |

The gain is heterogeneous: near zero for SI units (neither method recovers unit names in this band), small for astronomy (where the lens recovers other planets and moons at 0.086) and technology, large for countries and environment/geology. These are descriptive; domains are not independence units (countries is one dependency group; chemistry and si_units sit inside dg006).

## B. Tokenized-input overlap audit

**Visible input.** The worker runs one forward per item on exactly `token_ids` (BOS + prompt tokens up to the common-prefix boundary with the target); there is no wrapper, demonstration, chat template, prefilled response or padding (worker.py:114–129, evaluate.cache_states). The readout position is −1, so every token is causally visible to the scored state. Token lengths: 13–35 including BOS; 131 items drop the prompt's trailing-space token at the boundary, 29 drop none. Decoded token sequences were checked against the stored prompts.

**Exact scored-token overlap.** Intended labels: **0 of 160** (matches the frozen `leaked` flags). Controls: **6 exact matches in 6 items**, all controls, none intended:

| Item | Intermediate | Control whose scored token appears | Token | Context |
|---|---|---|---|---|
| confirmation-005 | Neptune | Voyager | ` Voyager` | "…was visited by the **Voyager** 2 spacecraft in" |
| confirmation-006 | Neptune | Voyager | ` Voyager` | "The planet where **Voyager** 2 observed the Great Dark Spot" |
| confirmation-016 | Io | Voyager | ` Voyager` | "…eruption was detected in **Voyager** 1 images in 1979" |
| confirmation-054 | Andes | Chile | ` Chile` | "…the border between **Chile** and Argentina" |
| confirmation-056 | Nile | Egypt | ` Egypt` | "…north-flowing river through **Egypt** has two major branches" |
| confirmation-098 | Beethoven | Io | `io` | "the opera Fidel**io**" (the accepted lowercase form `io` is its own token inside the word) |

The first five are genuine lexical mentions of another catalogue concept; the sixth is an exact token-ID overlap with an incidental word sense. Per the plan it is not exempted.

**Substring matches** (case-insensitive, decoded prompt): 12 word-boundary matches (the five mentions above in both capitalizations, plus "newton" in confirmation-132 "one newton per square metre", where ` newton` is not a single scored token, so it is a lexical mention without scored-token overlap) and 103 inside-word matches (84 for `Io`, 12 for `mole`, 3 `RNA`, 2 `Andes`, 2 `Egypt`), which are incidental. Intended-label substring matches: 2, both `io` inside "eruption" (confirmation-016), inside-word only. Semantic clues cannot be audited mechanically; the ledger lists every lexical hit for manual review.

**Sensitivities (pass-4 primary, both methods, frozen groups, 20,000 draws, seed 2026091103; descriptive):**

| Sensitivity | Rule | Denominators | Estimate | Descriptive group 95% |
|---|---|---|---|---:|
| S1 control filtering | drop the overlapping control(s) for the 6 affected items | 78 controls for 6 items, 79 for 154 | +0.23189 | [+0.16250, +0.32725] |
| S2 clean subset | drop the 6 affected items | 154 items, 28 groups | +0.22899 | [+0.15918, +0.33151] |

The full-population result stands; the overlaps concern controls that the lens recovers rarely.

**Intermediates versus final-answer forms.** No item's target equals its own intermediate (0/160), and no target string or target token coincides with any of the 80 concept names or their scored forms (0 items). Answer tokens are not visible at the readout position in any case.

**Not established by this audit:** absence of semantic information about the intermediate in the prompt (the prompt is designed to imply it) or absence of the facts from pretraining.

## C. The same physical band (26–37) in all four passes

Mapping verified: virtual = 48·(pass−1) + (physical−1); pass 4 band = virtual 169–180 = `PRIMARY_VIRTUAL`. fit01 and raw both carry 192 scored columns, so no unsupported tail enters any band. The original 20 secondaries contain no same-band early contrast (they use full-loop fixed means and any-layer scores), so passes 1–3 below are exploratory/post-confirmation.

| Pass | fit01 intended | fit01 control | raw intended | raw control | fit01 − raw excess | Interval | Existing all-layer mean (orig.) | Existing any-layer (orig.) |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | 0.0000 | 0.00079 | 0.0646 | 0.00042 | **−0.0650** | [−0.1191, −0.0108] N1 | −0.0883 | −0.4441 |
| 2 | 0.0422 | 0.00200 | 0.1271 | 0.00047 | **−0.0864** | [−0.1243, −0.0485] N1 | −0.1486 | −0.5737 |
| 3 | 0.0276 | 0.00044 | 0.1391 | 0.00047 | **−0.1114** | [−0.1676, −0.0552] N1 | −0.1559 | −0.6108 |
| 4 | 0.3792 | 0.00344 | 0.1443 | 0.00043 | **+0.2319** | [+0.1629, +0.3290] original primary | — | — |

In the very band where pass 4 shows the gain, passes 1–3 show a deficit whose simultaneous intervals exclude zero; fit01 recovers no intended label at all in the pass-1 band. fit02 − raw (descriptive): −0.0567, −0.0860, −0.1114, +0.2256. The pattern is therefore pass-dependent at fixed physical depth, not an artifact of averaging over all 48 layers.

## D. Dependency groups

**Rule.** Connected components over 294 declared edges of four classes: same intermediate identity (and its semantic aliases), shared underlying factual proposition, concrete entity-substitution template (same typed clue→entity→requested-slot construction), and final-answer or alias/code collision after casefold normalization (`benchmark.json.dependency_graph`, audit sha256 `5640316b…`, `finalize_dependencies.py`). Broad domain, generic schema, source page and broad relation taxonomy are not edges.

**Groups.** 28 components; sizes 42, 24, 16, 10, 10, 6, 4, 4, 4, 4, and eighteen of 2; effective count 160²/2968 = 8.63 (a weight-based summary, not a count of independent observations). Full membership with prompts: [GROUP_MEMBERSHIP.csv](GROUP_MEMBERSHIP.csv). Groups of size ≥ 4:

| Group | Size | Domains | Intermediates | Main edge kinds | Group primary mean | Estimate without group |
|---|---:|---|---|---|---:|---:|
| dg006 | 42 | chemistry, si_units, landforms, biology, environment | 8 elements, 8 SI units, Alps, Andes, Borneo, nucleus, ozone | 89 template, 21 identity, 10 shared fact, 5 answer collisions, 4 W code collisions | 0.184 | 0.2487 |
| dg016 | 24 | arts_music, historical_people | 12 people | 57 template (person→birthplace etc.), 12 identity | 0.203 | 0.2370 |
| dg007 | 16 | countries | 8 countries | 43 template (country→currency/capital), 8 identity | 0.543 | 0.1973 |
| dg003 | 10 | astronomy, environment | Neptune, Titan, Europa, Io, permafrost | 5 identity, 3 answer collisions (methane), 2 template | 0.080 | 0.2420 |
| dg013 | 10 | biology, technology | hemoglobin, chlorophyll, Linux, Python, Unicode | 5 identity, 4 answer collisions (1991, 4), 3 shared fact, 2 template | 0.158 | 0.2368 |
| dg010 | 6 | landforms | Nile, Danube, Mekong | 3 identity, 3 template | 0.051 | 0.2389 |
| dg002, dg008, dg011, dg024 | 4 each | mixed | see CSV | identity + one collision/template each | 0.313, 0.184, 0.041, 0.516 | |

**Why dg006 has 42 items.** Using only identity, shared-fact and template edges, dg006 splits into five sub-clusters: the 16 chemistry items (element→atomic-number and element→symbol templates), the 16 SI-unit items (unit→symbol and unit→quantity templates), 6 landform items (Alps, Andes, Borneo via range→continent and region→highest-summit templates), the 2 nucleus items and the 2 ozone items. Eight collision edges join them: tungsten's symbol answer "W" (confirmation-027) collides with Watt's symbol answer "W" (confirmation-136) and Watt's annotated alias "W" (also linking 028 and 135); helium's answer "2" (017) collides with nucleus's "2 membranes" (076); lithium's answer "3" (019) collides with Borneo's "3 countries" (063) and ozone's "3 atoms" (146); and 063–146 collide with each other. So the largest group is two template families bridged by one surface-code collision, plus three small clusters attached by small-integer answers. Item weights were unchanged; no alternative grouping was constructed.

**Retained from the frozen analysis.** Item-resampling interval [+0.17971, +0.28706]; leave-one-group-out range [+0.19730, +0.24875] (the low end removes dg007, the high end removes dg006); 20,000 draws, seed 2026090901. The grouping captures declared identity, fact, template and answer dependencies; it does not certify the absence of other dependence (shared templates across domains, shared source pages), and the effective count is not a literal sample size.
