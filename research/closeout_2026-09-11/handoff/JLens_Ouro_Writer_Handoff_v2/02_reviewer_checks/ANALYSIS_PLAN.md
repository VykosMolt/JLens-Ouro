# Post-confirmation reviewer analyses on retained data: plan (frozen before computation)

Date: 2026-09-11. These are reviewer follow-up analyses on the already-used 160-item confirmation data. They are not a new prospective confirmation. The original primary endpoint (+0.23188 [+0.16289, +0.32898]) and the original 20 secondary contrasts are reported unchanged and are not part of the new family.

## Data (all from the accepted payload, hash-bound by RUN_SPECIFICATION.json)

- `results/population.json`: 160 rows, 80 names, `own_index`, `control_indices` (79 per item), `dependency_group_id` (28 groups), `token_ids`, `intermediate_tokens`, `token_forms`.
- `results/benchmark.json`: `concepts[*].domain` (10 domains × 8 concepts), `dependency_graph.edges` (294) and `components` (28).
- `results/readouts/{fit01,raw}.npz` `allrank` [160,128,192] int32 (zero-based full-vocabulary rank, min over accepted forms; −1 padding beyond the 80 names). fit02/penultimate/sampled_sum/diagonal are used only descriptively where stated.
- Frozen code reused unchanged: `evaluation/measurement.py` (`score`, `fixed_mean`, `PRIMARY_VIRTUAL`), `analysis/analyze.py` (`resample`: whole-group resampling, sum/selected-item-count ratio, 20,000 draws).

## Fixed definitions

- Hit: rank < 10. Fixed-layer mean over the 12 virtual columns of a band. Item weight equal. Groups: frozen `dependency_group_id`. Estimand: item-weighted mean of the paired per-item difference.
- Band for pass p (1-based): virtual 48(p−1)+25 … 48(p−1)+36 (physical 26–37). Pass 4 = PRIMARY_VIRTUAL 169–180.
- Method pair: fit01 (frozen main estimator, 192 columns, identity row 191) versus raw (192 columns). Both support every band column; no unsupported tail enters any band.

## New inferential family N1 (exactly four contrasts; centered bootstrap max-t 95%, 20,000 group draws, seed 2026091101)

1. A: within-domain paired excess difference, pass 4 band: [own − mean(7 same-domain controls)]_fit01 − [same]_raw.
2. C1: same-band paired excess difference (all-79 controls), pass 1.
3. C2: same-band paired excess difference, pass 2.
4. C3: same-band paired excess difference, pass 3.

Everything else (components, per-domain means, within-domain control recoveries, pass-4 same-band which is the original primary, sensitivities) is descriptive: percentile intervals from the same group bootstrap are shown as descriptive only. No other contrasts will be added after inspection.

## A. Within-domain controls

Controls for item i are the 7 other concept names in i's `domain` (from `benchmark.json`), which must be a subset of its frozen 79 controls (asserted). Full-vocabulary ranking, intended labels, accepted forms, item weights and fixed-layer averaging are unchanged. The domain grouping is inspected for semantic homogeneity in `RESULTS.md` and is called a "within-domain control", not semantic matching. Reported per method: intended recovery, all-79 control recovery, within-domain control recovery, paired excess differences; per-domain item counts and descriptive means.

Arithmetic bound (checked from exact arrays, per item and in aggregate): excess7_diff ≥ intended_diff − (79/7)·c79_fit01, because c7_raw ≥ 0 and 7·c7_fit01 ≤ 79·c79_fit01. A violation would indicate an indexing, alias, denominator or weighting mismatch to be resolved, never forced.

## B. Tokenized-input overlap

For every item the model input is exactly `token_ids` (worker.py:114–118 and evaluate.cache_states: one item per forward, no wrapper, demonstrations, chat template, prefilled response or padding; BOS is `token_ids[0]`; readout position −1 so every token is causally visible to the scored state). For every scored token ID of the intended label and of all 79 controls, record exact ID matches against `token_ids` with position, decoded token and ±5-token context. Separately record case-insensitive substring matches of every accepted surface form against the decoded prompt (word-boundary vs inside-word). Semantic clues are listed for manual review only. Also record whether the item's target (final answer) coincides with any concept name or its scored forms (answer tokens are not visible at readout; this is a construction check).

If any exact control overlap exists: (S1) control-filtering sensitivity: drop the overlapping controls for that item for both methods, record new denominators; (S2) clean-subset sensitivity: drop affected items for both methods. Both use the frozen groups, 20,000 draws, seed 2026091103, and are descriptive sensitivities of the pass-4 primary. Zero exact overlap does not establish absence of semantic information or pretraining contamination.

## C. Same physical band in all four passes

Verified mapping: virtual = 48(pass−1) + (physical−1). For passes 1–4 report fit01 and raw intended/control/excess band means, the paired difference, group-bootstrap descriptive percentile intervals and the N1 simultaneous intervals for passes 1–3. Pass 4 is the original primary and is reported with its original interval. The original 20 contrasts contain no same-band early contrast (they use full-loop fixed means and any-layer), so C1–C3 are exploratory/post-confirmation. The existing all-layer and any-layer summaries are shown alongside for comparison, not inference.

## D. Dependency groups

Recovered from `benchmark.json.dependency_graph` (audit sha256 5640316b…): rule = connected components over declared edges of four classes (same intermediate identity/aliases, shared underlying fact, concrete entity-substitution template, final-answer or alias/code collision). Output: full membership table (group, size, items, intermediates, domains, edge-kind counts), and for dg006 the sub-clusters joined by template edges and the exact collision edges bridging them. Retained from analysis.json: item-resampling interval, leave-one-group-out range, per-group means, sizes, effective count 8.63 (not a literal count of independent observations).
