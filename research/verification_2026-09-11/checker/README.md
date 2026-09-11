# Independent endpoint checker: Ouro J-Lens confirmation (160 items)
Built from the delegated specification, PROSPECTIVE_PLAN.md and the data only; no producer or replay code was opened.
Uses stdlib + numpy; run with /home/moloch/ouro_project/venv/bin/python. It never overwrites `--out`.

## Commands
```
PY=/home/moloch/ouro_project/venv/bin/python; A=<accepted .../results>; R=<results/ouro_confirmation_20260911_fixed160_native1/analysis>
$PY endpoint_checker.py --population $A/population.json --readouts $A/readouts --expected-analysis $R/analysis.json --out saved_outputs.json
$PY endpoint_checker.py --population $A/population.json --readouts replay/<run>/<layout> --expected-analysis $R/analysis.json --out <new file>
$PY endpoint_checker.py --self-test --out self_test.json
```
Exit 0: all input checks and comparisons pass. 1: input rejected or a comparison failed. 2: `--out` already exists.

## What is checked
- Inputs: 80 names; 160 rows; `allrank` int32 [160,128,C], C = 192 (raw, fit01, fit02), 190 (penultimate), 191 (sampled_sum, diagonal); name ranks in [0,49152); padding exactly -1; eligible own_index in [0,80); controls = all names not in own_index, nonempty.
- Rebuilt from `allrank` only: 7 item values and estimates; group-bootstrap percentile intervals (seed 2026090901); item-resampling interval (2026090902); group heterogeneity and leave-one-group range; 20 secondary contrasts with max-t simultaneous intervals (2026090903); descriptive curves.
- Comparison with analysis.json: ids, regions, metrics and integers exactly; numbers within 1e-12 absolute. Each quantity group records its maximum absolute difference.
- Self-test: `fixtures/base` holds allrank-only copies, population.json and the saved analysis, all hashed in MANIFEST.json. The unperturbed copy must pass. F1-F10 are applied in memory and must be detected or rejected; single-rank flips must match analytic deltas.

## What is not checked
- The rank arrays themselves: forward pass, bank readout, token aliases, readout position.
- Other npz keys; eligibility, annotation or dependency-graph correctness; analysis.json `schema`, `bootstrap` and `component_order`; item_statistics_and_resamples.npz.
- That population rows and array axis 0 describe the same items: `allrank` carries no item key.
- Zero-SD secondary columns: the plan specifies zero-width intervals, but the spec replaces SD 0 with 1 (width 2q). The checker follows the spec; the accepted data has no zero-SD column.

## Specification/data disagreements
- `own_index` has 3 entries per row (trailing -1), while `eligible` and `control_indices` have 1. Slots follow `eligible`; trailing entries must be -1.
- F2: no item has fit01 intended rank 9 at virtual 100, so item 4 (rank 2) is flipped to 10. No candidate can change early_loop3_any_layer (each has at least 14 hits in 96..143).
