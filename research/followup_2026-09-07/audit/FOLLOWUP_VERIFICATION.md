# Independent verification of follow-up layers — 7 September 2026

PASS. No production scoring or bootstrap helper was imported. Source ranks were rescored independently; bootstrap means used direct resampled indexing rather than production matrix weights.

Checked 3,914 numerical comparisons; maximum absolute discrepancy 5.55e-16.

| Check | Result |
|---|---|
| Every per-item score, both methods, all 192 locations and any-layer summaries | Pass |
| All 384 paired item-bootstrap intervals | Pass |
| All 384 concept-cluster intervals | Pass |
| Sixty method/metric intervals at ten randomly selected cells | Pass |
| Every depth-band mean and item/cluster interval | Pass |
| Eight OLS slopes and item/cluster intervals | Pass |
| Historical any-layer statistic and seed-0 uncertainty | Pass |
| Concept-cluster membership: independent graph search versus union-find | Pass |
| Cluster ratio weights versus materialized variable-length resamples | Pass |
| Final endpoint zero per item and degenerate pointwise intervals | Pass |
| Output files remained unchanged during verification | Pass |

The maximum centered absolute bootstrap error across both tasks and all four loops/48 layers gives radius **0.230769230769231** for paired item resampling and **0.333036473330591** for concept-cluster resampling. Independent task seeds 0 and 1 are recorded; their errors are joined by bootstrap replicate to calibrate the 384-cell family. Empirical coverage of the bootstrap errors by these radii is 0.950000 and 0.950000.

These are approximate bootstrap simultaneous bands, not exact finite-sample coverage guarantees. They are unstudentized: every cell receives the same absolute radius, which can be conservative for low-variance cells. The final identity cell has exactly zero paired effect and [0,0] pointwise intervals; its displayed simultaneous envelope is conservative, not evidence of uncertainty about identity. Pointwise method ribbons and fixed-block/slope intervals do not receive the 384-cell simultaneous correction.

Concept clusters preserve equal item weight: draw the original number of clusters with replacement, include every item whenever its cluster is drawn, and divide by the number of sampled items. This differs appropriately from taking an unweighted mean of cluster means. Membership was independently checked before aligning cluster order to the saved labels for exact seeded replay. There are 64 multihop and 14 arithmetic clusters. Clustering is a sensitivity assumption; it does not establish independence between templates.

The main statistic performs oracle any-layer scoring with the maximum before control averaging. The new fixed-layer statistic performs no layer selection. Both methods use the same item draws, and subgroup/fit uncertainty is not smuggled into these intervals. All curves still condition on one fitted J-Lens.

Ten verification cells were chosen with seed 11092026 before outcomes were read:

| Task | Loop | Physical layer | Paired difference | Pointwise 95% interval |
|---|---:|---:|---:|---|
| multihop | 1 | 43 | -0.184491 | [-0.272104, -0.105863] |
| multihop | 2 | 8 | -0.013351 | [-0.036864, -0.000462] |
| multihop | 2 | 41 | -0.241484 | [-0.332653, -0.156130] |
| multihop | 3 | 11 | +0.008691 | [-0.016591, +0.044923] |
| multihop | 3 | 38 | -0.088396 | [-0.154741, -0.032174] |
| multihop | 4 | 32 | +0.241058 | [+0.142185, +0.341221] |
| order-ops | 1 | 31 | -0.466063 | [-0.610860, -0.321267] |
| order-ops | 2 | 30 | +0.126697 | [-0.013575, +0.268477] |
| order-ops | 3 | 6 | +0.067873 | [-0.028658, +0.171946] |
| order-ops | 4 | 41 | -0.125189 | [-0.235294, -0.027149] |

Verified output SHA256:

- `layerwise.csv`: `f50901dbb1360e489c4fe98b031511ff327f27b7c1a265d6b2bc3738b9711dc0`
- `layerwise.json`: `fc7b988306560a41e814b866dadbcda6c4fb54d56018b329de888eb8a154ccf1`
- `item_scores.npz`: `ccdbb8cfee0d41f9b48cca75994ec997bfbebf4ac977f194f24906c600d476d5`
- `depth_bands.csv`: `55249949277ec1b357203afc5f1a8b169091f35564535bc8362adc08131f4d5e`
- `correctness.json`: `a8b21700191882c14a4834e23a0fe80e179931378f27ffe935e806fcd0108e3e`

Correctness verification also passes. Independent prior continuation annotations reproduce all three rules: historical 72/148, strict numerical 70/148, narrow numerical equivalence 71/148. Every stratum/method mean, all 2,304 stratum layerwise paired intervals (three criteria × two tasks × two strata × 192 cells), corresponding cluster intervals, any-layer effects, and passing-minus-failing interactions reproduce.

The cluster interaction correctly resamples concepts jointly across outcome strata; it does not independently draw a shared concept once for passing items and again for failing items. Each stratum is normalized by its own resampled item count. No draw loses an entire stratum in the retained 20,000 replicates. These outcome comparisons remain observational and their intervals are pointwise; task/template composition can explain differences.
