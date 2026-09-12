# Source-to-manuscript ledger (v1 and v2)

Status codes: exact (rounded) = machine-checked against the record at the printed precision; record/prose = traced to a record without an automatic numeric comparison; MISMATCH = printed value disagrees with the record; imprecise/citation = documentation discrepancy corrected in v2.

| Version | Location | Quoted | Source | Source value | Status |
|---|---|---|---|---|---|
| v1 | Abstract/§4.1 | 23.19 | results/.../analysis/analysis.json :: estimates[primary_excess_difference] | 0.23188291139240502 | exact (rounded) |
| v1 | Abstract/§4.1 | 16.29–32.90 | results/.../analysis/analysis.json :: family_percentile_95_intervals[primary_excess_differ | [0.1628899320083425, 0.3289769593495046] | exact (rounded) |
| v1 | Abstract/§4.1 | 37.92 | results/.../analysis/analysis.json :: estimates[fit01_intended] | 0.3791666666666666 | exact (rounded) |
| v1 | Abstract/§4.1 | 14.43 | results/.../analysis/analysis.json :: estimates[raw_intended] | 0.14427083333333332 | exact (rounded) |
| v1 | §4.1 Table 1 | 30.44–46.56 | results/.../analysis/analysis.json :: family_percentile_95_intervals[fit01_intended] | [0.3044442204301075, 0.465625] | exact (rounded) |
| v1 | §4.1 Table 1 | 10.45–17.87 | results/.../analysis/analysis.json :: family_percentile_95_intervals[raw_intended] | [0.10454545454545454, 0.1787037037037037] | exact (rounded) |
| v1 | §4.1 Table 1 | 0.34 | results/.../analysis/analysis.json :: estimates[fit01_control] | 0.0034414556962025313 | exact (rounded) |
| v1 | §4.1 Table 1 | 0.18–0.51 | results/.../analysis/analysis.json :: family_percentile_95_intervals[fit01_control] | [0.0018166315049226443, 0.005065116602777693] | exact (rounded) |
| v1 | §4.1 Table 1 | 0.04 | results/.../analysis/analysis.json :: estimates[raw_control] | 0.0004285337552742617 | exact (rounded) |
| v1 | §4.1 Table 1 | 0.02–0.08 | results/.../analysis/analysis.json :: family_percentile_95_intervals[raw_control] | [0.0001849416406378432, 0.0007549433275419872] | exact (rounded) |
| v1 | §4.1 | 23.49 | results/.../analysis/analysis.json :: estimates[intended_difference] | 0.23489583333333322 | exact (rounded) |
| v1 | §4.1 | 17.97–28.71 | results/.../analysis/analysis.json :: primary_item_resampling_95_interval | [0.17971271756329119, 0.2870582805907173] | exact (rounded) |
| v1 | §4.1/§3.2 | 19.73 and 24.88 | results/.../analysis/analysis.json :: leave_one_group_range | [0.19730133614627288, 0.24874848029750418] | MISMATCH |
| v1 | §4.1 | 22.56 | results/.../analysis/analysis.json :: secondary_family[fit02_local].estimate | 0.22558016877637127 | exact (rounded) |
| v1 | §4.1 | 9.52–35.59 | results/.../analysis/analysis.json :: secondary_family[fit02_local].simultaneous_95_interv | [0.09523375118988045, 0.3559265863628621] | exact (rounded) |
| v1 | §4.1 | 0.63 | results/.../analysis/analysis.json :: secondary_family[fit01_minus_fit02_local].estimate | 0.006302742616033756 | exact (rounded) |
| v1 | §4.1 | −0.73 to +1.99 | results/.../analysis/analysis.json :: secondary_family[fit01_minus_fit02_local].simultaneo | [-0.007270138454696015, 0.019875623686763527] | exact (rounded) |
| v1 | §4.2 Table 2 | −8.83 | results/.../analysis/analysis.json :: secondary_family[early_loop1_fixed_mean].estimate | -0.08827465717299579 | exact (rounded) |
| v1 | §4.2 Table 2 | −13.64, −4.01 | results/.../analysis/analysis.json :: secondary_family[early_loop1_fixed_mean].simultaneou | [-0.13641974092762596, -0.040129573418365615] | exact (rounded) |
| v1 | §4.2 Table 2 | −44.41 | results/.../analysis/analysis.json :: secondary_family[early_loop1_any_layer].estimate | -0.4440664556962024 | exact (rounded) |
| v1 | §4.2 Table 2 | −59.03, −29.79 | results/.../analysis/analysis.json :: secondary_family[early_loop1_any_layer].simultaneous | [-0.5902854897612019, -0.2978474216312029] | MISMATCH |
| v1 | §4.2 Table 2 | −14.86 | results/.../analysis/analysis.json :: secondary_family[early_loop2_fixed_mean].estimate | -0.14858089398734178 | exact (rounded) |
| v1 | §4.2 Table 2 | −19.54, −10.18 | results/.../analysis/analysis.json :: secondary_family[early_loop2_fixed_mean].simultaneou | [-0.19540695469825603, -0.10175483327642752] | exact (rounded) |
| v1 | §4.2 Table 2 | −57.37 | results/.../analysis/analysis.json :: secondary_family[early_loop2_any_layer].estimate | -0.5736550632911395 | exact (rounded) |
| v1 | §4.2 Table 2 | −72.01, −42.72 | results/.../analysis/analysis.json :: secondary_family[early_loop2_any_layer].simultaneous | [-0.7200754085098435, -0.4272347180724354] | exact (rounded) |
| v1 | §4.2 Table 2 | −15.59 | results/.../analysis/analysis.json :: secondary_family[early_loop3_fixed_mean].estimate | -0.15586926424050637 | exact (rounded) |
| v1 | §4.2 Table 2 | −22.40, −8.77 | results/.../analysis/analysis.json :: secondary_family[early_loop3_fixed_mean].simultaneou | [-0.2240409794100318, -0.08769754907098096] | exact (rounded) |
| v1 | §4.2 Table 2 | −61.08 | results/.../analysis/analysis.json :: secondary_family[early_loop3_any_layer].estimate | -0.6108386075949369 | exact (rounded) |
| v1 | §4.2 Table 2 | −79.59, −42.58 | results/.../analysis/analysis.json :: secondary_family[early_loop3_any_layer].simultaneous | [-0.7959090115574546, -0.4257682036324192] | exact (rounded) |
| v1 | §4.3 Table 4 | +14.38 | results/.../analysis/analysis.json :: secondary_family[fit01_minus_penultimate_local].esti | 0.1438422995780591 | exact (rounded) |
| v1 | §4.3 Table 4 | +7.70, +21.07 | results/.../analysis/analysis.json :: secondary_family[fit01_minus_penultimate_local].simu | [0.07699378313022422, 0.210690816025894] | exact (rounded) |
| v1 | §4.3 Table 4 | +15.77 | results/.../analysis/analysis.json :: secondary_family[sampled_sum_minus_diagonal_local].e | 0.15772020042194096 | exact (rounded) |
| v1 | §4.3 Table 4 | +5.80, +25.75 | results/.../analysis/analysis.json :: secondary_family[sampled_sum_minus_diagonal_local].s | [0.05796365904793578, 0.25747674179594615] | exact (rounded) |
| v1 | §4.3 Table 4 | +8.80 | results/.../analysis/analysis.json :: secondary_family[penultimate_local].estimate | 0.08804061181434597 | exact (rounded) |
| v1 | §4.3 Table 4 | +0.76, +16.85 | results/.../analysis/analysis.json :: secondary_family[penultimate_local].simultaneous_95_ | [0.007594657812076772, 0.16848656581661517] | exact (rounded) |
| v1 | §4.3 Table 4 | +21.72 | results/.../analysis/analysis.json :: secondary_family[sampled_sum_local].estimate | 0.2172402426160338 | exact (rounded) |
| v1 | §4.3 Table 4 | +9.16, +34.29 | results/.../analysis/analysis.json :: secondary_family[sampled_sum_local].simultaneous_95_ | [0.09155303182978272, 0.3429274534022849] | exact (rounded) |
| v1 | §4.3 Table 4 | +5.95 | results/.../analysis/analysis.json :: secondary_family[diagonal_local].estimate | 0.05952004219409286 | exact (rounded) |
| v1 | §4.3 Table 4 | −0.34, +12.24 | results/.../analysis/analysis.json :: secondary_family[diagonal_local].simultaneous_95_int | [-0.003395438727201716, 0.12243552311538744] | exact (rounded) |
| v1 | §3.2 | 28 dependency groups | results/.../analysis/analysis.json :: groups | 28 | exact (rounded) |
| v1 | §3.2 | 42, 24, 16, 10, 10, 6 | results/.../analysis/analysis.json :: group_heterogeneity[*].items | [42, 24, 16, 10, 10, 6, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2,  | record/prose |
| v1 | §3.2 | 8.63 | results/.../analysis/analysis.json :: derived 160^2/sum n_g^2 | 8.625336927223719 | exact (rounded) |
| v1 | §3.2 | 20,000 | results/.../analysis/analysis.json :: bootstrap [replicates] | 20000 | record/prose |
| v1 | §3.2 | 2026090901 | results/.../analysis/analysis.json :: bootstrap [seed] | 2026090901 | exact (rounded) |
| v1 | §3.2 | 160 two-hop questions, 80 intermediate concepts | RUN_SPECIFICATION.json population.items/names |  | record/prose |
| v1 | §3.2 | 96 … 45 … 19 | benchmark.json design.relation_balance.counts |  | record/prose |
| v1 | §3.2 | Three component facts overlap known calibration text | final_lexical_novelty.json / benchmark_review.md (items 076, 130, 151) |  | record/prose |
| v1 | §5 Table | +22.93 | verification_2026-09-11/metrics/VERIFICATION_METRICS.json :: paths.fp64_local_gpu.endpoint | 0.22927874472573836 | exact (rounded) |
| v1 | §5 Table | +16.17, +32.52 | verification_2026-09-11/metrics/VERIFICATION_METRICS.json :: paths.fp64_local_gpu.endpoint | [0.16170389501712013, 0.32517161366683084] | exact (rounded) |
| v1 | §5 Table | −0.26 | verification_2026-09-11/metrics/VERIFICATION_METRICS.json :: paths.fp64_local_gpu.endpoint | -0.0026041666666666574 | exact (rounded) |
| v1 | §5 Table | +23.61 | verification_2026-09-11/metrics/VERIFICATION_METRICS.json :: paths.regenerated_states_loca | 0.23607594936708862 | exact (rounded) |
| v1 | §5 Table | +16.52, +33.77 | verification_2026-09-11/metrics/VERIFICATION_METRICS.json :: paths.regenerated_states_loca | [0.16520561020913593, 0.33769448126153034] | exact (rounded) |
| v1 | §5 Table | +0.42 | verification_2026-09-11/metrics/VERIFICATION_METRICS.json :: paths.regenerated_states_loca | 0.004193037974683594 | exact (rounded) |
| v1 | §5 | 22.60 and 23.35 | verification_2026-09-11/metrics/VERIFICATION_METRICS.json :: paths.executed_replay_local_g | "lookup error: float() argument must be a string or a real n | record/prose |
| v1 | §5 | 37 of 160 | verification_2026-09-11/metrics/VERIFICATION_METRICS.json :: native_output_A_local_gpu.reg | 37 | record/prose |
| v1 | §5 | 154 | verification_2026-09-11/metrics/VERIFICATION_METRICS.json :: native_output_A_local_gpu.con | 154 | exact (rounded) |
| v1 | §5 | 13 planted errors | verification_2026-09-11/metrics/VERIFICATION_METRICS.json :: independent_checker | "lookup error: float() argument must be a string or a real n | record/prose |
| v1 | §5 | 9×10⁻¹⁴ (CPU vs GPU FP64) | REPORT_confirmation_verified.md |  | record/prose |
| v1 | §5 | 160, 190, 191 and 192 rows | verification_2026-09-11/metrics/VERIFICATION_METRICS.json :: paths.rows190_local_gpu.endpo | 0.0 | record/prose |
| v1 | §6/§7 Reproducibility | 43 … 33 distinct | preservation/PRESERVATION_LEDGER.json :: len(unique_and_lost) | 43 | record/prose |
| v1 | Appendix B | −0.34 to +12.24 / +0.04 (regenerated) | verification_2026-09-11/metrics/VERIFICATION_METRICS.json :: paths.regenerated_states_loca | "lookup error: float() argument must be a string or a real n | record/prose |
| v1 | §3.1 | 18.72 | refit_round_2026-09-07/analysis/main_run01/report.json :: P1.tasks.multihop.metrics[*][18] | 0.18717072612687255 | exact (rounded) |
| v1 | §3.1 | 0.39 | refit_round_2026-09-07/analysis/main_run01/report.json :: P1.tasks.multihop.metrics[*][18] | 0.003938554888786443 | exact (rounded) |
| v1 | §3.1 | 90 eligible multihop items … 51 arithmetic | refit_round_2026-09-07/analysis/main_run01/report.json :: populations | "lookup error: float() argument must be a string or a real n | record/prose |
| v1 | §3.1 | every fit peaked at physical layer 32 | refit_round_2026-09-07/analysis/main_run01/report.json :: P1.tasks.multihop.loop4_learned_ | "lookup error: float() argument must be a string or a real n | record/prose |
| v1 | App. A | 85.1% | followup .../results/exit_agreement.json :: ['-1'].actual_exit_agreement [actual_exit_agre | 0.8513513513513513 | exact (rounded) |
| v1 | App. A | 79.1–90.5% | followup .../results/exit_agreement.json :: ['-1'].actual_exit_agreement [ci95.2.3] | [0.7905405405405409, 0.9054054054054059] | exact (rounded) |
| v1 | App. A | 1.45% | followup exit_agreement.json :: ['-1'].loops.3.windows.41-47.top1_agreement_matrix [mean.0 | 0.014478764478764476 | exact (rounded) |
| v1 | App. A | 0.58–2.51% | followup exit_agreement.json :: ['-1'].loops.3.windows.41-47.top1_agreement_matrix [ci95.0 | [0.005791505791505791, 0.025096525096525095] | exact (rounded) |
| v1 | App. A | 96.6% | followup .../results/exit_agreement.json :: ['-1'].actual_exit_agreement | "lookup error: float() argument must be a string or a real n | record/prose |
| v1 | App. A | 0.39% | followup exit_agreement.json :: ['-2'].loops.3.windows.41-47.top1_agreement_matrix [mean.0 | 0.0038610038610038607 | exact (rounded) |
| v1 | App. A | 72 passes out of 148 to 70 … 71 | followup results/correctness.json :: all_148_counts | "lookup error: float() argument must be a string or a real n | record/prose |
| v1 | App. A | 59.4% … 76.4% … 35.6% … 17.9% … 36 pairs | followup probe/bounded_refit_results.json :: performance | "lookup error: float() argument must be a string or a real n | record/prose |
| v1 | App. A | 88 multihop … 51 arithmetic | huginn_run01/report.json :: populations | "lookup error: float() argument must be a string or a real n | record/prose |
| v1 | App. A | negative under all six prespecified summaries | refit .../huginn_run01/metrics.csv :: rows | "lookup error: float() argument must be a string or a real n | record/prose |
| v1 | §2.1 | revision 1ed04250… | RUN_SPECIFICATION.json model.revision |  | record/prose |
| v1 | §2.1 | 48 blocks, four passes, 2,048, 49,152 | RUN_SPECIFICATION.json model.geometry/vocab_size |  | record/prose |
| v2 | Abstract/§4.1 | 23.19 | results/.../analysis/analysis.json :: estimates[primary_excess_difference] | 0.23188291139240502 | exact (rounded) |
| v2 | Abstract/§4.1 | 16.29–32.90 | results/.../analysis/analysis.json :: family_percentile_95_intervals[primary_excess_differ | [0.1628899320083425, 0.3289769593495046] | exact (rounded) |
| v2 | Abstract/§4.1 | 37.92 | results/.../analysis/analysis.json :: estimates[fit01_intended] | 0.3791666666666666 | exact (rounded) |
| v2 | Abstract/§4.1 | 14.43 | results/.../analysis/analysis.json :: estimates[raw_intended] | 0.14427083333333332 | exact (rounded) |
| v2 | §4.1 Table 1 | 30.44–46.56 | results/.../analysis/analysis.json :: family_percentile_95_intervals[fit01_intended] | [0.3044442204301075, 0.465625] | exact (rounded) |
| v2 | §4.1 Table 1 | 10.45–17.87 | results/.../analysis/analysis.json :: family_percentile_95_intervals[raw_intended] | [0.10454545454545454, 0.1787037037037037] | exact (rounded) |
| v2 | §4.1 Table 1 | 0.34 | results/.../analysis/analysis.json :: estimates[fit01_control] | 0.0034414556962025313 | exact (rounded) |
| v2 | §4.1 Table 1 | 0.18–0.51 | results/.../analysis/analysis.json :: family_percentile_95_intervals[fit01_control] | [0.0018166315049226443, 0.005065116602777693] | exact (rounded) |
| v2 | §4.1 Table 1 | 0.04 | results/.../analysis/analysis.json :: estimates[raw_control] | 0.0004285337552742617 | exact (rounded) |
| v2 | §4.1 Table 1 | 0.02–0.08 | results/.../analysis/analysis.json :: family_percentile_95_intervals[raw_control] | [0.0001849416406378432, 0.0007549433275419872] | exact (rounded) |
| v2 | §4.1 | 23.49 | results/.../analysis/analysis.json :: estimates[intended_difference] | 0.23489583333333322 | exact (rounded) |
| v2 | §4.1 | 17.97–28.71 | results/.../analysis/analysis.json :: primary_item_resampling_95_interval | [0.17971271756329119, 0.2870582805907173] | exact (rounded) |
| v2 | §4.1/§3.2 | 19.73 and 24.87 | results/.../analysis/analysis.json :: leave_one_group_range | [0.19730133614627288, 0.24874848029750418] | exact (rounded) |
| v2 | §4.1 | 22.56 | results/.../analysis/analysis.json :: secondary_family[fit02_local].estimate | 0.22558016877637127 | exact (rounded) |
| v2 | §4.1 | 9.52–35.59 | results/.../analysis/analysis.json :: secondary_family[fit02_local].simultaneous_95_interv | [0.09523375118988045, 0.3559265863628621] | exact (rounded) |
| v2 | §4.1 | 0.63 | results/.../analysis/analysis.json :: secondary_family[fit01_minus_fit02_local].estimate | 0.006302742616033756 | exact (rounded) |
| v2 | §4.1 | −0.73 to +1.99 | results/.../analysis/analysis.json :: secondary_family[fit01_minus_fit02_local].simultaneo | [-0.007270138454696015, 0.019875623686763527] | exact (rounded) |
| v2 | §4.2 Table 2 | −8.83 | results/.../analysis/analysis.json :: secondary_family[early_loop1_fixed_mean].estimate | -0.08827465717299579 | exact (rounded) |
| v2 | §4.2 Table 2 | −13.64, −4.01 | results/.../analysis/analysis.json :: secondary_family[early_loop1_fixed_mean].simultaneou | [-0.13641974092762596, -0.040129573418365615] | exact (rounded) |
| v2 | §4.2 Table 2 | −44.41 | results/.../analysis/analysis.json :: secondary_family[early_loop1_any_layer].estimate | -0.4440664556962024 | exact (rounded) |
| v2 | §4.2 Table 2 | −59.03, −29.78 | results/.../analysis/analysis.json :: secondary_family[early_loop1_any_layer].simultaneous | [-0.5902854897612019, -0.2978474216312029] | exact (rounded) |
| v2 | §4.2 Table 2 | −14.86 | results/.../analysis/analysis.json :: secondary_family[early_loop2_fixed_mean].estimate | -0.14858089398734178 | exact (rounded) |
| v2 | §4.2 Table 2 | −19.54, −10.18 | results/.../analysis/analysis.json :: secondary_family[early_loop2_fixed_mean].simultaneou | [-0.19540695469825603, -0.10175483327642752] | exact (rounded) |
| v2 | §4.2 Table 2 | −57.37 | results/.../analysis/analysis.json :: secondary_family[early_loop2_any_layer].estimate | -0.5736550632911395 | exact (rounded) |
| v2 | §4.2 Table 2 | −72.01, −42.72 | results/.../analysis/analysis.json :: secondary_family[early_loop2_any_layer].simultaneous | [-0.7200754085098435, -0.4272347180724354] | exact (rounded) |
| v2 | §4.2 Table 2 | −15.59 | results/.../analysis/analysis.json :: secondary_family[early_loop3_fixed_mean].estimate | -0.15586926424050637 | exact (rounded) |
| v2 | §4.2 Table 2 | −22.40, −8.77 | results/.../analysis/analysis.json :: secondary_family[early_loop3_fixed_mean].simultaneou | [-0.2240409794100318, -0.08769754907098096] | exact (rounded) |
| v2 | §4.2 Table 2 | −61.08 | results/.../analysis/analysis.json :: secondary_family[early_loop3_any_layer].estimate | -0.6108386075949369 | exact (rounded) |
| v2 | §4.2 Table 2 | −79.59, −42.58 | results/.../analysis/analysis.json :: secondary_family[early_loop3_any_layer].simultaneous | [-0.7959090115574546, -0.4257682036324192] | exact (rounded) |
| v2 | §4.3 Table 4 | +14.38 | results/.../analysis/analysis.json :: secondary_family[fit01_minus_penultimate_local].esti | 0.1438422995780591 | exact (rounded) |
| v2 | §4.3 Table 4 | +7.70, +21.07 | results/.../analysis/analysis.json :: secondary_family[fit01_minus_penultimate_local].simu | [0.07699378313022422, 0.210690816025894] | exact (rounded) |
| v2 | §4.3 Table 4 | +15.77 | results/.../analysis/analysis.json :: secondary_family[sampled_sum_minus_diagonal_local].e | 0.15772020042194096 | exact (rounded) |
| v2 | §4.3 Table 4 | +5.80, +25.75 | results/.../analysis/analysis.json :: secondary_family[sampled_sum_minus_diagonal_local].s | [0.05796365904793578, 0.25747674179594615] | exact (rounded) |
| v2 | §4.3 Table 4 | +8.80 | results/.../analysis/analysis.json :: secondary_family[penultimate_local].estimate | 0.08804061181434597 | exact (rounded) |
| v2 | §4.3 Table 4 | +0.76, +16.85 | results/.../analysis/analysis.json :: secondary_family[penultimate_local].simultaneous_95_ | [0.007594657812076772, 0.16848656581661517] | exact (rounded) |
| v2 | §4.3 Table 4 | +21.72 | results/.../analysis/analysis.json :: secondary_family[sampled_sum_local].estimate | 0.2172402426160338 | exact (rounded) |
| v2 | §4.3 Table 4 | +9.16, +34.29 | results/.../analysis/analysis.json :: secondary_family[sampled_sum_local].simultaneous_95_ | [0.09155303182978272, 0.3429274534022849] | exact (rounded) |
| v2 | §4.3 Table 4 | +5.95 | results/.../analysis/analysis.json :: secondary_family[diagonal_local].estimate | 0.05952004219409286 | exact (rounded) |
| v2 | §4.3 Table 4 | −0.34, +12.24 | results/.../analysis/analysis.json :: secondary_family[diagonal_local].simultaneous_95_int | [-0.003395438727201716, 0.12243552311538744] | exact (rounded) |
| v2 | §3.2 | 28 dependency groups | results/.../analysis/analysis.json :: groups | 28 | exact (rounded) |
| v2 | §3.2 | 42, 24, 16, 10, 10, 6 | results/.../analysis/analysis.json :: group_heterogeneity[*].items | [42, 24, 16, 10, 10, 6, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2,  | record/prose |
| v2 | §3.2 | 8.63 | results/.../analysis/analysis.json :: derived 160^2/sum n_g^2 | 8.625336927223719 | exact (rounded) |
| v2 | §3.2 | 20,000 | results/.../analysis/analysis.json :: bootstrap [replicates] | 20000 | record/prose |
| v2 | §3.2 | 2026090901 | results/.../analysis/analysis.json :: bootstrap [seed] | 2026090901 | exact (rounded) |
| v2 | §3.2 | 160 two-hop questions, 80 intermediate concepts | RUN_SPECIFICATION.json population.items/names |  | record/prose |
| v2 | §3.2 | 96 … 45 … 19 | benchmark.json design.relation_balance.counts |  | record/prose |
| v2 | §3.2 | Three component facts overlap known calibration text | final_lexical_novelty.json / benchmark_review.md (items 076, 130, 151) |  | record/prose |
| v2 | §5 Table | +22.93 | verification_2026-09-11/metrics/VERIFICATION_METRICS.json :: paths.fp64_local_gpu.endpoint | 0.22927874472573836 | exact (rounded) |
| v2 | §5 Table | +16.17, +32.52 | verification_2026-09-11/metrics/VERIFICATION_METRICS.json :: paths.fp64_local_gpu.endpoint | [0.16170389501712013, 0.32517161366683084] | exact (rounded) |
| v2 | §5 Table | −0.26 | verification_2026-09-11/metrics/VERIFICATION_METRICS.json :: paths.fp64_local_gpu.endpoint | -0.0026041666666666574 | exact (rounded) |
| v2 | §5 Table | +23.61 | verification_2026-09-11/metrics/VERIFICATION_METRICS.json :: paths.regenerated_states_loca | 0.23607594936708862 | exact (rounded) |
| v2 | §5 Table | +16.52, +33.77 | verification_2026-09-11/metrics/VERIFICATION_METRICS.json :: paths.regenerated_states_loca | [0.16520561020913593, 0.33769448126153034] | exact (rounded) |
| v2 | §5 Table | +0.42 | verification_2026-09-11/metrics/VERIFICATION_METRICS.json :: paths.regenerated_states_loca | 0.004193037974683594 | exact (rounded) |
| v2 | §5 | 22.60 and 23.35 | verification_2026-09-11/metrics/VERIFICATION_METRICS.json :: paths.executed_replay_local_g | "lookup error: float() argument must be a string or a real n | record/prose |
| v2 | §5 | 37 of 160 | verification_2026-09-11/metrics/VERIFICATION_METRICS.json :: native_output_A_local_gpu.reg | 37 | record/prose |
| v2 | §5 | 154 | verification_2026-09-11/metrics/VERIFICATION_METRICS.json :: native_output_A_local_gpu.con | 154 | exact (rounded) |
| v2 | §5 | 13 planted errors | verification_2026-09-11/metrics/VERIFICATION_METRICS.json :: independent_checker | "lookup error: float() argument must be a string or a real n | record/prose |
| v2 | §5 | 9×10⁻¹⁴ (CPU vs GPU FP64) | REPORT_confirmation_verified.md |  | record/prose |
| v2 | §5 | 160, 190, 191 and 192 rows | verification_2026-09-11/metrics/VERIFICATION_METRICS.json :: paths.rows190_local_gpu.endpo | 0.0 | record/prose |
| v2 | §6/§7 Reproducibility | 43 … 33 distinct | preservation/PRESERVATION_LEDGER.json :: len(unique_and_lost) | 43 | record/prose |
| v2 | Appendix B | −0.34 to +12.24 / +0.04 (regenerated) | verification_2026-09-11/metrics/VERIFICATION_METRICS.json :: paths.regenerated_states_loca | "lookup error: float() argument must be a string or a real n | record/prose |
| v2 | §3.1 | 18.72 | refit_round_2026-09-07/analysis/main_run01/report.json :: P1.tasks.multihop.metrics[*][18] | 0.18717072612687255 | exact (rounded) |
| v2 | §3.1 | 0.39 | refit_round_2026-09-07/analysis/main_run01/report.json :: P1.tasks.multihop.metrics[*][18] | 0.003938554888786443 | exact (rounded) |
| v2 | §3.1 | 90 eligible multihop items … 51 arithmetic | refit_round_2026-09-07/analysis/main_run01/report.json :: populations | "lookup error: float() argument must be a string or a real n | record/prose |
| v2 | §3.1 | every fit peaked at physical layer 32 | refit_round_2026-09-07/analysis/main_run01/report.json :: P1.tasks.multihop.loop4_learned_ | "lookup error: float() argument must be a string or a real n | record/prose |
| v2 | App. A | 85.1% | followup .../results/exit_agreement.json :: ['-1'].actual_exit_agreement [actual_exit_agre | 0.8513513513513513 | exact (rounded) |
| v2 | App. A | 79.1–90.5% | followup .../results/exit_agreement.json :: ['-1'].actual_exit_agreement [ci95.2.3] | [0.7905405405405409, 0.9054054054054059] | exact (rounded) |
| v2 | App. A | 1.45% | followup exit_agreement.json :: ['-1'].loops.3.windows.41-47.top1_agreement_matrix [mean.0 | 0.014478764478764476 | exact (rounded) |
| v2 | App. A | 0.58–2.51% | followup exit_agreement.json :: ['-1'].loops.3.windows.41-47.top1_agreement_matrix [ci95.0 | [0.005791505791505791, 0.025096525096525095] | exact (rounded) |
| v2 | App. A | 96.6% | followup .../results/exit_agreement.json :: ['-1'].actual_exit_agreement | "lookup error: float() argument must be a string or a real n | record/prose |
| v2 | App. A | 0.39% | followup exit_agreement.json :: ['-2'].loops.3.windows.41-47.top1_agreement_matrix [mean.0 | 0.0038610038610038607 | exact (rounded) |
| v2 | App. A | 72 passes out of 148 to 70 … 71 | followup results/correctness.json :: all_148_counts | "lookup error: float() argument must be a string or a real n | record/prose |
| v2 | App. A | 59.4% … 76.4% … 35.6% … 17.9% … 36 pairs | followup probe/bounded_refit_results.json :: performance | "lookup error: float() argument must be a string or a real n | record/prose |
| v2 | App. A | 88 multihop … 51 arithmetic | huginn_run01/report.json :: populations | "lookup error: float() argument must be a string or a real n | record/prose |
| v2 | App. A | negative under all six prespecified summaries | refit .../huginn_run01/metrics.csv :: rows | "lookup error: float() argument must be a string or a real n | record/prose |
| v2 | §2.1 | revision 1ed04250… | RUN_SPECIFICATION.json model.revision |  | record/prose |
| v2 | §2.1 | 48 blocks, four passes, 2,048, 49,152 | RUN_SPECIFICATION.json model.geometry/vocab_size |  | record/prose |
| v2 | Abstract/§4.2 Table 3 | −6.5 / −6.50 | closeout reviewer_checks/RESULTS.json :: family_N1 [contrasts.C1_sameband_excess_diff_pass | -0.06495253164556963 | record/prose |
| v2 | §4.2 Table 3 | −11.91, −1.08 | closeout reviewer_checks/RESULTS.json :: family_N1 [contrasts.C1_sameband_excess_diff_pass | [-0.11914387978749716, -0.010761183503642115] | exact (rounded) |
| v2 | Abstract/§4.2 Table 3 | −8.6 / −8.64 | closeout reviewer_checks/RESULTS.json :: family_N1 [contrasts.C2_sameband_excess_diff_pass | -0.08642536919831224 | record/prose |
| v2 | §4.2 Table 3 | −12.43, −4.85 | closeout reviewer_checks/RESULTS.json :: family_N1 [contrasts.C2_sameband_excess_diff_pass | [-0.12431223211303809, -0.0485385062835864] | exact (rounded) |
| v2 | Abstract/§4.2 Table 3 | −11.1 / −11.14 | closeout reviewer_checks/RESULTS.json :: family_N1 [contrasts.C3_sameband_excess_diff_pass | -0.11141877637130806 | record/prose |
| v2 | §4.2 Table 3 | −16.76, −5.52 | closeout reviewer_checks/RESULTS.json :: family_N1 [contrasts.C3_sameband_excess_diff_pass | [-0.16763256699210505, -0.05520498575051108] | exact (rounded) |
| v2 | Abstract/§4.4 | 20.8 / 20.80 | closeout reviewer_checks/RESULTS.json :: family_N1 [contrasts.A_within_domain_excess_diff_ | 0.20796130952380953 | record/prose |
| v2 | Abstract/§4.4 | 9.1–32.5 / 9.13–32.46 | closeout reviewer_checks/RESULTS.json :: family_N1 [contrasts.A_within_domain_excess_diff_ | [0.09129698871641183, 0.3246256303312072] | record/prose |
| v2 | §4.4 | 3.07% … 0.34% | closeout RESULTS.json :: A_within_domain [per_method.fit01.within_domain_control_recovery. | 0.03072916666666666 | record/prose |
| v2 | §4.4 | 0.38% … 0.04% | closeout RESULTS.json :: A_within_domain [per_method.raw.within_domain_control_recovery.es | 0.0037946428571428567 | record/prose |
| v2 | §4.4 | 19.61 | closeout RESULTS.json :: A_within_domain [bound_check.aggregate_lower_bound_exact] | 0.19605654761904762 | exact (rounded) |
| v2 | §4.4 | +51 … +40 … about zero … −1 (+5 with 79 controls) | closeout RESULTS.json :: A_within_domain | "lookup error: float() argument must be a string or a real n | record/prose |
| v2 | §4.4 | 6 control tokens … six items | closeout RESULTS.json :: B_overlap [exact_control_overlaps] | 6 | exact (rounded) |
| v2 | §4.4 | 23.19 (16.25–32.72) | closeout RESULTS.json :: B_overlap [sensitivities.S1_control_filtering.estimate] | 0.23188992683652493 | record/prose |
| v2 | §4.4 | 22.90 (15.92–33.15) | closeout RESULTS.json :: B_overlap [sensitivities.S2_clean_subset.estimate] | 0.22898514987122587 | record/prose |
| v2 | §4.2 Table 3 | 0.00% / 6.46% | closeout RESULTS.json :: C_same_band [passes.0.raw_intended.estimate] | 0.06458333333333333 | record/prose |
| v2 | §4.2 Table 3 | 4.22% / 12.71% | closeout RESULTS.json :: C_same_band [passes.1.raw_intended.estimate] | 0.12708333333333333 | record/prose |
| v2 | §4.2 Table 3 | 2.76% / 13.91% | closeout RESULTS.json :: C_same_band [passes.2.raw_intended.estimate] | 0.1390625 | record/prose |
| v2 | §3.1 | 10.96–26.99 | refit_round_2026-09-07/analysis/main_run01/report.json :: P1.tasks.multihop.metrics.interv | "lookup error: float() argument must be a string or a real n | record/prose |
| v2 | §3.1 | −0.86 to 38.29 | refit REPORT.md crossed fit/item simultaneous |  | record/prose |
| v2 | §3.1 | −7.49 to 44.92 | refit REPORT.md crossed fit/component simultaneous |  | record/prose |
| v2 | §3.1 | 6.13 (SD 0.21) | refit_round_2026-09-07/analysis/main_run01/report.json :: P1.tasks.multihop.metrics[*][17] | 0.0613136664644948 | record/prose |
| v2 | §3.1 | 18.91 own-name minus 0.19 control | refit REPORT.md own/control decomposition |  | record/prose |
| v2 | §6 | 15.4%, 16.2%, 25.2% / 2.1, 4.3, 10.1 / 0.3, 0.4, 2.1 | closeout local_exit/DISCOVERY_POPULATION_FIXED_BAND.json :: passes | "lookup error: float() argument must be a string or a real n | record/prose |
| v2 | §6 | +14.6 | closeout local_exit/DISCOVERY_POPULATION_FIXED_BAND.json :: passes [0.local_minus_final.ex | 0.14572933440687064 | exact (rounded) |
| v2 | §6 | 9.1–20.9 | closeout local_exit/DISCOVERY_POPULATION_FIXED_BAND.json :: passes [0.local_minus_final.co | [0.09088155370843991, 0.20936431697524552] | exact (rounded) |
| v2 | §6 | +16.9 | closeout local_exit/DISCOVERY_POPULATION_FIXED_BAND.json :: passes [1.local_minus_final.ex | 0.1685486646289251 | exact (rounded) |
| v2 | §6 | 11.5–22.8 | closeout local_exit/DISCOVERY_POPULATION_FIXED_BAND.json :: passes [1.local_minus_final.co | [0.11497637817220568, 0.22823205500497273] | exact (rounded) |
| v2 | §6 | +23.0 | closeout local_exit/DISCOVERY_POPULATION_FIXED_BAND.json :: passes [2.local_minus_final.ex | 0.2300641636470485 | exact (rounded) |
| v2 | §6 | 16.0–30.6 | closeout local_exit/DISCOVERY_POPULATION_FIXED_BAND.json :: passes [2.local_minus_final.co | [0.1601311419097825, 0.3063063558631991] | exact (rounded) |
| v2 | §6 | +13.0 | closeout local_exit/DISCOVERY_POPULATION_FIXED_BAND.json :: passes [0.local_minus_raw.exce | 0.1301014906286979 | exact (rounded) |
| v2 | §6 | 8.1–19.0 | closeout local_exit/DISCOVERY_POPULATION_FIXED_BAND.json :: passes [0.local_minus_raw.conc | [0.08098000989991197, 0.19019554867512808] | exact (rounded) |
| v2 | §6 | +11.3 | closeout local_exit/DISCOVERY_POPULATION_FIXED_BAND.json :: passes [1.local_minus_raw.exce | 0.11345785989065753 | exact (rounded) |
| v2 | §6 | 6.6–16.6 | closeout local_exit/DISCOVERY_POPULATION_FIXED_BAND.json :: passes [1.local_minus_raw.conc | [0.06601991294180325, 0.16554117145987002] | exact (rounded) |
| v2 | §6 | +14.9 | closeout local_exit/DISCOVERY_POPULATION_FIXED_BAND.json :: passes [2.local_minus_raw.exce | 0.14893851819334697 | exact (rounded) |
| v2 | §6 | 7.9–22.4 | closeout local_exit/DISCOVERY_POPULATION_FIXED_BAND.json :: passes [2.local_minus_raw.conc | [0.07910663163006994, 0.2244045154003551] | exact (rounded) |
| v2 | §6 | 162 blocks and three pass-boundary normalizations | DISCOVERY_POPULATION_FIXED_BAND.json passes[0].blocks_source_to_final_target_range (155–16 |  | DISCREPANCY-CHECK |
| v2 | §6 | 11 GB … eight GPU-hours | LOCAL_EXIT_STATUS.md proposal (13 files 11.2 GB; (47+95+143)/191×100×192.3 s ≈ 8.0 h) |  | record/prose |
| v2 | §2.2 | 131 of 160 … 13–35 tokens … token 20 … 16293 | population.json boundary_checks; SCORING_PATH_REVIEW §1a; EXAMPLES.md |  | record/prose |
| v2 | §2.1 | seeds 2026090701–05 … 0–3 … RTX 5090 | calibration_plan.json; refit LEASE.json gpu_type_id |  | record/prose |
| v2 | App. B | 2.639 | closeout reviewer_checks/RESULTS.json :: family_N1 [max_t_95_quantile] | 2.6386394248749863 | exact (rounded) |
| v1 | §2.1 | position controls use "the same sampled target positions and all 2,048 derivative directions per paragraph" | combined_contract.json controls.positions (one uniform q per paragraph, seed 2026090801) |  | imprecise (corrected in v2) |
| v1 | §3.1 | "Both positive band estimates had simultaneous intervals containing zero" | refit REPORT.md (56-metric family) |  | exact (prose), values added in v2 |
| v1 | Abstract | "change the primary estimate by at most 0.42 points" | verification_2026-09-11/metrics/VERIFICATION_METRICS.json :: paths.regenerated_states_loca | 0.004193037974683594 | exact (rounded) |
| v1 | App. A | Huginn coda/training statements cited only to internal [S2] | literature.md → Geiping et al. §3.3, Lu et al. §3.2 |  | citation added in v2 |

## Added 2026-09-12: own-exit comparison on the confirmation population (Section 7, Table 6, Figure 5, abstract, conclusion, Appendix B)

Source: `research/closeout_2026-09-11/local_exit/confirmation_2026-09-12/RESULTS.json` (plan `ANALYSIS_PLAN.md`, sha256 `16cdcbd0…`). Values in percentage points unless marked %.

| Manuscript value | JSON path | Exact value |
|---|---|---|
| own exit minus final target pass1: +22.60 [+11.61, +33.60] | inference.family.own_exit_minus_final_target_pass1 | 0.226048 [0.116146, 0.335951] |
| own exit minus final target pass2: +26.50 [+18.37, +34.63] | inference.family.own_exit_minus_final_target_pass2 | 0.264999 [0.183735, 0.346262] |
| own exit minus final target pass3: +34.17 [+23.39, +44.94] | inference.family.own_exit_minus_final_target_pass3 | 0.341693 [0.233937, 0.449449] |
| own exit minus raw pass1: +16.41 [+8.92, +23.91] | inference.family.own_exit_minus_raw_pass1 | 0.164142 [0.089201, 0.239083] |
| own exit minus raw pass2: +18.32 [+10.53, +26.11] | inference.family.own_exit_minus_raw_pass2 | 0.183228 [0.105324, 0.261131] |
| own exit minus raw pass3: +23.18 [+11.49, +34.87] | inference.family.own_exit_minus_raw_pass3 | 0.231791 [0.114891, 0.348690] |
| max-t quantile 2.699 | inference.maxt_quantile_95 | 2.699112 |
| pass 1 raw intended 6.46%, control 0.04% | descriptive.components.pass1.raw | 0.064583, 0.000422 |
| pass 1 exit3 intended 0.57%, control 0.35% | descriptive.components.pass1.exit3 | 0.005729, 0.003474 |
| pass 1 fit01 intended 0.00%, control 0.08% | descriptive.components.pass1.fit01 | 0.000000, 0.000791 |
| pass 1 exit0 intended 23.12%, control 0.29% | descriptive.components.pass1.exit0 | 0.231250, 0.002947 |
| pass 2 raw intended 12.71%, control 0.05% | descriptive.components.pass2.raw | 0.127083, 0.000468 |
| pass 2 exit3 intended 4.69%, control 0.20% | descriptive.components.pass2.exit3 | 0.046875, 0.002031 |
| pass 2 fit01 intended 4.22%, control 0.20% | descriptive.components.pass2.fit01 | 0.042188, 0.001998 |
| pass 2 exit1 intended 31.30%, control 0.32% | descriptive.components.pass2.exit1 | 0.313021, 0.003178 |
| pass 3 raw intended 13.91%, control 0.05% | descriptive.components.pass3.raw | 0.139063, 0.000475 |
| pass 3 exit3 intended 2.92%, control 0.05% | descriptive.components.pass3.exit3 | 0.029167, 0.000481 |
| pass 3 fit01 intended 2.76%, control 0.04% | descriptive.components.pass3.fit01 | 0.027604, 0.000435 |
| pass 3 exit2 intended 37.40%, control 0.36% | descriptive.components.pass3.exit2 | 0.373958, 0.003580 |
| pass 4 exit3 − raw +22.39 [+16.07, +31.68] | descriptive.pass4_final_target_bank_on_confirmation.exit3_minus_raw | 0.223866 |
| pass 4 exit3 − fit01 -0.80 [-1.71, +0.30] | descriptive.pass4_final_target_bank_on_confirmation.exit3_minus_fit01 | -0.008017 |
