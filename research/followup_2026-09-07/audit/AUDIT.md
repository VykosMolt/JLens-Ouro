# Independent metric and artifact audit — 7 September 2026

The N=100 headline survives independent arithmetic. All eight differences and
paired 95% intervals reproduce to the reported rounding. The correctness labels
contain two further false positives: continuations `3.8` and `3.75` pass target
`3`. That changes a stricter numerical criterion from 72/148 to 70/148. Accepting
the unambiguous numeral `20` for the word target `twenty` gives 71/148 under a
separate sensitivity. None of these numbers is complete semantic accuracy.

This audit did not edit the submitted application or historical artifacts. It
did not perform the new per-layer comparison, which is a separate analysis.

## What was read and checked

`docs/jlens/APPLICATION_DRAFT.md` was read in full, followed by `RESULTS.md`,
`APPLICATION_VERIFICATION.md`, `B300_RESULTS_2026-09-05.md`, the current decision
ledger, the relevant historical chronology, peer audit passages, and the rental
board entries concerning evaluation failures. The relevant implementation is
`/home/moloch/ouro_project/src/ouro_jlens/{analyze,evaluate,evaldata,recurrent}.py`,
with upstream `jlens/vis.py` and `jlens/hf.py` for ranking and unembedding.

`audit_existing.py` computes the headline directly from NumPy arrays and JSON;
it imports no study scoring/report functions. It also checks item identity,
rank-slot bookkeeping, padding, exit identities, and output hashes.
`audit_controls.py` uses the original token-form construction to inspect the
actual candidate token sets; it imports no scoring/report helpers.

## Metric and denominator

The primary source is
`artifacts/jlens/retrieved/jlens-b300-20260905-0359/eval/n100_exit3/`.
There are 93 raw multihop and 55 raw arithmetic items. Eligibility is evaluated
per intermediate slot: single-token scorable, not leaked into the prompt, and
for arithmetic a numeric rather than operation label. This yields 90 multihop
items with 100 slots and 51 arithmetic items with 51 slots: 141 items, not 142.
The application explicitly identifies its five examples as an earlier sample
from 142 eligible items; that historical number is not the main denominator.

At each loop, each true name receives its own event “any of 48 layers has
zero-based vocabulary rank < 10.” Every control name receives exactly the same
event before controls are averaged. Slot scores are then averaged within the
item, and items receive equal weight. Controls exclude all the item's own
scorable names and match operation versus non-operation within the task.
Source: `analyze.py:208`, `:242`, `:266`, `:278`.

This is an oracle any-layer estimand, not a held-out rule choosing one layer
for future examples. The main comparison performs no learned layer selection;
each label on each item can succeed at its own layer. That is acceptable as a
descriptive “somewhere in the loop” question but does not answer where a lens
helps. The separate `selected_location_ci95_conditional` in `analyze.py` chooses
the location on these same observations; it should not be read as selection-
adjusted inference. It is not the application headline.

| Population | Loop 1 J−LL | Loop 2 | Loop 3 | Loop 4 |
|---|---:|---:|---:|---:|
| Multihop, n=90 | −0.355761 | −0.478945 | −0.406690 | −0.021559 |
| Paired 95% interval | [−0.462672, −0.248954] | [−0.579320, −0.376991] | [−0.510436, −0.303571] | [−0.096131, +0.055584] |
| Arithmetic, n=51 | −0.254902 | +0.140271 | +0.088989 | −0.021116 |
| Paired 95% interval | [−0.361991, −0.138763] | [+0.019608, +0.263952] | [−0.045249, +0.226244] | [−0.072398, +0.024133] |

These are 20,000 paired item-bootstrap draws with seed 0, unadjusted across the
eight comparisons. They condition on the retained fit and curated item set.
`independent_reproduction.json` contains unrounded hit, control, excess, and
effect values. The old iron example is not the N=100 value: N=100 gives minimum
loop-1 zero-based rank 131 (ordinary rank 132) for J-Lens and 0 (rank 1) for LL.
The application identifies the 979 example as the earlier evaluation.

## Correctness is a task criterion

All 148 raw continuations were inspected. `continuation_audit.{json,csv}` keeps
the original criterion and annotations for every item. `evaluate.py:468` and
`analyze.py:41` accept a target prefix whenever the next character is not
alphanumeric. This repairs `11` versus `1` but permits a decimal point.

| Criterion | All 148 passing | Eligible multihop passing/failing | Eligible arithmetic passing/failing |
|---|---:|---:|---:|
| Historical boundary rule | 72 | 36 / 54 | 31 / 20 |
| Reject attached decimal/fraction/exponent after integer target | 70 | 36 / 54 | 29 / 22 |
| Additionally accept numeral for exact number-word target | 71 | 36 / 54 | 30 / 21 |

The two decimal false positives are `div-sub-left` and
`nested4-add-mult-add-div`. `word-parens` outputs `20.` for `twenty`, explaining
the extra numerical-equivalence pass. Other cases were preserved: `daytime`
versus `day` is a plausible semantic equivalent but deliberately fails the
historical boundary rule; `written as "V` contains the expected Roman numeral
after a prefix; `the North Pole` is related to `north`. The four-token generation
budget also limits adjudication of delayed answers (`evaluate.py:460`). A
passing/failing label is more defensible than “ultimately correct/incorrect.”

Two released fact tasks still assume that China is the most populous country:
`super-populous-capital` and `firstletter-populous-country`. UN DESA dated the
India/China crossover to April 2023. This is a separate target-validity concern;
the audit does not silently relabel the task. [UN DESA Policy Brief 153](https://desapublications.un.org/policy-briefs/un-desa-policy-brief-no-153-india-overtakes-china-worlds-most-populous-country).

## Controls and dependence

Exact token-form equivalences in the multihop name set are `four=4`, `3=three`,
`5=five`, `6=six`; `third` also shares the digit-3 token. Four rhyme items have
an equivalent own label left among their controls: `rhyme-door-doubled`,
`rhyme-tree-squared`, `rhyme-hive-plusone`, and `rhyme-fix-halved`. Thus a control
can literally be the intended concept under another name. This is conservative
for that item's excess in a single method, but its effect on the paired method
difference need not have a fixed sign. No arithmetic control token-form overlap
was found. `control_forms.json` records exact token IDs and affected items.

In 43/90 multihop and 39/51 arithmetic items, a control shares token forms with
the final target. This is compatible with measuring intermediate specificity
over a same-task background, but controls are not guaranteed unrelated
alternatives. “Matched task-name controls” is accurate; matching does not cover
frequency, semantic type, or number of surface forms.

No exact duplicate prompt strings or duplicate item names occur. Several items
share templates, concepts, and few-shot demonstrations, so 90 or 51 IID prompt
draws is an assumption, not a design fact. Two prespecified-from-metadata
sensitivity partitions are saved in `dependence_clusters.csv`: same canonical
intermediate concept (64 multihop / 14 arithmetic clusters), and released name
prefix before the first hyphen (36 / 15 clusters). Either misses some forms of
dependence and neither is a uniquely correct population model. They are
sensitivity checks, not a license to generalize to all multihop/arithmetic tasks.

## Actual exits and stored fields

The strict local all-exit source is
`artifacts/jlens/eval/b300_local_allexits_strict/`. It saves `exit_top1[148,4]`,
and each lens's `top1`, `kl_to_final`, `kl_to_local`, `rank_of_final_top1`, and
`rank_of_local_top1`, each `[148,192]`. Main ranks are `[148,128,192]`, with -1
padding after each task's true name count. Recurrent positions are ordered
`loop*48 + physical_layer`, both zero-based.

Actual exits come from unembedding each recorded loop's last physical layer
(`evaluate.py:1282`), an operation covered by the existing recorder exit-equality
checks. The logit-lens prediction at all four endpoints exactly equals the saved
exit token for all 148 items. The final J-Lens endpoint equals final exit, also
for every item.

The word `local` is easy to misuse: in `readout_arrays` it means `target_ut`
(`evaluate.py:551`), not the recurrent pass of the source state. Both the final
J-Lens and LL are passed `target_ut=3`, so their `*_to_local` fields duplicate
their `*_to_final` fields everywhere. These cannot supply LL-to-loop-k KL or
final-J-to-loop-k ranks. Broadcast the actual `exit_top1[:,k]` for true top-1
comparisons; local-J-to-own-exit ranks and KL are present.

The local lens has identity matrices at and after its target (`evaluate.py:600`).
Only its source prefix and target are meaningful local-exit applications; later
identity tails must not be interpreted as successful local-lens predictions.
All three local identity tails equal LL in the saved token arrays. Its own
endpoint agreement is definitional and should be separated from preceding
layers. Full logits, hidden states, and top-k token lists were not persisted in
these evaluation outputs. General top-k set overlaps require a new readout run;
the retained rank of the true model token already supplies useful top-k recall.

## Source identity and numerical differences

The arrays, items, and task-name file hashes match all corresponding provenance
output records in both primary bundles. The local strict bundle matches all
five current source hashes. The pod bundle's `evaluate.py` hash differs from the
current working tree but matches `git show HEAD:src/ouro_jlens/evaluate.py`
exactly (`50a905bdd51fcfe81d3871ba3ee58a1dae4f4bcc9f803497e9ef7ca0823923bd`).
The working-tree difference is absolute output-root handling and CUDA
`empty_cache()` after the preflight Jacobian allocation; the readout function
is unchanged. The other four pod source hashes match current files. This is a
recoverable historical source version, not an unknown generator. Lens identities
record one N=100 final lens for the pod and four N=100 lenses for the local run.

The pod and local evaluations are not numerically identical: 8/148 continuations,
8/592 actual exit predictions, 635/28,416 LL top-1 locations, and 580/28,416 final
J-Lens top-1 locations differ. The original passing/failing labels do not differ.
The array field differences are saved in `independent_reproduction.json`.
Each run must be compared to its own exit predictions. A claim of byte-identical
replication across devices would be false.

## Chronological audit record

1. Read the submitted write-up and located the correct project and N=100 bundles.
2. Read the metric, recorder, task construction, and historical bug records.
3. Checked the raw array schema; identified identity tails and target-relative
   meaning of `local` before new exit comparisons.
4. Inspected all raw continuations; found the two decimal false positives and
   kept exact criterion, stricter criterion, and numerical equivalence separate.
5. Reproduced all eight headline effects and intervals without report helpers;
   checked denominators, control aggregation order, padding, and endpoint identities.
6. Compared pod and local records, resolved the source hash difference to the
   retained Git version, and measured token/continuation differences.
7. Inspected token-form overlap and froze two dependence sensitivity partitions
   without using per-layer effects. Verified the stale populous-country premise
   against an original UN source. No historical output was overwritten.
