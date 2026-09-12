# Own-exit versus final-target J-Lens on the confirmation population

Executed 2026-09-12T17:34:12Z per `ANALYSIS_PLAN.md` (sha256 `16cdcbd071e86449…`), frozen before computation. Initial-study banks (12-article calibration prefix, dim_batch 32) applied with the frozen scorer to the accepted confirmation states; raw and fit01 readouts reused from the accepted payload. Percentage points of excess hit@10 (79 controls), physical layers 26–37.

## Checks

- Raw readouts reproduced bit for bit from these states with the fetched unembedding tensors: {'allrank': True, 'rank': True, 'top1': True, 'top10_ids': True}
- fit01 check: {'allrank': True, 'rank': True, 'top1': True, 'top10_ids': True}
- Primary anchor (fit01 − raw, pass 4) recomputed: +0.231883 (saved +0.231883)
- exit3 merged from shards vs pod merge: identical files False, max |diff| 0.0, fraction equal 1.000000

## Inference: the six-contrast family (simultaneous 95% max-t, whole-group bootstrap, seed 2026091201, quantile 2.6991)

| Contrast | Estimate (pts) | Simultaneous 95% | Excludes zero |
|---|---:|---|---|
| own exit minus final target pass1 | +22.60 | [+11.61, +33.60] | yes |
| own exit minus final target pass2 | +26.50 | [+18.37, +34.63] | yes |
| own exit minus final target pass3 | +34.17 | [+23.39, +44.94] | yes |
| own exit minus raw pass1 | +16.41 | [+8.92, +23.91] | yes |
| own exit minus raw pass2 | +18.32 | [+10.53, +26.11] | yes |
| own exit minus raw pass3 | +23.18 | [+11.49, +34.87] | yes |

## Components by pass (descriptive; group-percentile 95% intervals in RESULTS.json)

| Pass | Arm | Intended | Control (79) | Excess |
|---:|---|---:|---:|---:|
| 1 | raw | 6.46% | 0.04% | +6.42 |
| 1 | exit3 | 0.57% | 0.35% | +0.23 |
| 1 | fit01 | 0.00% | 0.08% | -0.08 |
| 1 | exit0 | 23.12% | 0.29% | +22.83 |
| 2 | raw | 12.71% | 0.05% | +12.66 |
| 2 | exit3 | 4.69% | 0.20% | +4.48 |
| 2 | fit01 | 4.22% | 0.20% | +4.02 |
| 2 | exit1 | 31.30% | 0.32% | +30.98 |
| 3 | raw | 13.91% | 0.05% | +13.86 |
| 3 | exit3 | 2.92% | 0.05% | +2.87 |
| 3 | fit01 | 2.76% | 0.04% | +2.72 |
| 3 | exit2 | 37.40% | 0.36% | +37.04 |
| 4 | raw | 14.43% | 0.04% | +14.38 |
| 4 | exit3 | 37.14% | 0.36% | +36.77 |
| 4 | fit01 | 37.92% | 0.34% | +37.57 |

## Descriptive contrasts

- own exit − fit01, pass 1: +22.91 [+16.72, +32.78] (different calibration; does not isolate the target)
- own exit − fit01, pass 2: +26.97 [+21.83, +34.33] (different calibration; does not isolate the target)
- own exit − fit01, pass 3: +34.32 [+26.88, +43.54] (different calibration; does not isolate the target)
- pass 4, exit3 minus raw: +22.39 [+16.07, +31.68]
- pass 4, exit3 minus fit01: -0.80 [-1.71, +0.30]
- pass 4, fit01 minus raw ORIGINAL PRIMARY: +23.19 [+16.29, +32.90]

Labeling per plan: post-confirmation analysis on the already-used confirmation population; estimator family differs from the confirmation fit01 family; own-exit − exit3 isolates the target within the initial-study family; nothing here alters the primary endpoint.
