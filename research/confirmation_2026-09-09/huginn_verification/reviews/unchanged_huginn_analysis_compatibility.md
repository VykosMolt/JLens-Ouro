# Unchanged historical six-contrast analysis compatibility

The original analyzer is compatible in design with a newly fitted H estimator. It does **not** require the lost historical H fit identity or bank SHA. It reconstructs and validates the actual embedded N100 owner against the original scientific contract, then requires the exact preceding Ouro/control owner and completion records. No outcome analysis was run for this inspection.

Use the original complete five-fit Ouro and paired-control evaluations below. Their prerequisite identities match those carried by the historical H owner and those selected for the new H bundle. The new 160-item primary experiment is not an input to this historical common-population analysis. The H input is the inner sealed `results/readouts` directory, with all 17 original files; the outer H verification run specification is not the analyzer’s `--run-spec`.

Run only after the new H generation has actual external acceptance, substituting its absolute path and a fresh output name outside every accepted/sealed input tree:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home/moloch/ouro_project/venv/bin/python \
/home/moloch/jacobian-lens/research/refit_round_2026-09-07/analysis/analyze_huginn.py analyze \
  --main /home/moloch/jacobian-lens/research/refit_round_2026-09-07/monitoring/attempt_06/ouro_handoff_20260909T110503Z/results/ouro_evaluation \
  --controls /home/moloch/jacobian-lens/research/refit_round_2026-09-07/monitoring/attempt_06/ouro_handoff_20260909T110503Z/results/controls_evaluation \
  --huginn ACCEPTED_GENERATION/results/readouts \
  --output /home/moloch/jacobian-lens/research/confirmation_2026-09-09/huginn_verification/analysis/accepted_rerun_six_contrast_RUN_ID \
  --ouro-src /home/moloch/ouro_project/src \
  --run-spec /home/moloch/jacobian-lens/research/refit_round_2026-09-07/deployment/run_spec.json \
  --combined-contract /home/moloch/jacobian-lens/research/refit_round_2026-09-07/deployment/combined_contract.json \
  --huginn-manifest /home/moloch/jacobian-lens/research/refit_round_2026-09-07/deployment/huginn_model_manifest.json \
  --huginn-calibration /home/moloch/jacobian-lens/research/refit_round_2026-09-07/deployment/huginn_calibration.json \
  --eligibility /home/moloch/jacobian-lens/research/refit_round_2026-09-07/deployment/huginn_eligibility.json
```

Then use the same unchanged script’s `validate OUTPUT_DIRECTORY` command. Keep its export unchanged and link its OWNER/COMPLETE records to the actual new H receipt in adjacent provenance marked `newly_rerun_estimator`. Matching historical bytes would not recover the lost checkpoint.

The existing plan stays fixed: 88 multihop and 51 arithmetic eligible items; all-learned mean, seven-cell mean and any-of-seven; both H seeds; 10,000 paired item/component draws with seed 2026090808; the original simultaneous six-contrast rule. No new endpoint or favorable selection is introduced.

The source check verified the original 42-file scientific closure and all five calibration files. Exact analyzer, contract, helper, argument-document and prerequisite records are in `unchanged_huginn_analysis_compatibility.json`. The canonical host analysis directory is used because the 42-file worker closure does not include the old analysis scripts.

If eventual new inputs fail an existing identity/runtime/population/prerequisite gate, preserve that failure instead of weakening it. The separately reviewed direct historical comparison already covers state, rank, initialization and summary discrepancies; no alternative statistical plan is required.

Key source pins:

- `research/refit_round_2026-09-07/analysis/analyze_huginn.py`: 67882 bytes, SHA256 `f7e65c275c83fc6aa3d6e40e54205ad50f0f07441699768d43351e69cd7793bd`.
- `research/refit_round_2026-09-07/analysis/HUGINN_COMPARISON_CONTRACT.md`: 16353 bytes, SHA256 `de409521cd0f14be05e3c384e683e4b645655670035f50654183bbb8734f0cb8`.
- `research/refit_round_2026-09-07/analysis/analyze_refits.py`: 65490 bytes, SHA256 `297ec665f6f64889bdbe319dde34858a02e9cd3162437d9e8f1fc5cba506df6b`.
- `research/refit_round_2026-09-07/analysis/CONTRACT.md`: 5477 bytes, SHA256 `277e73a6bf10d92b828e3da82106584f3e47d281e96401bd00f1a5bf9c2e5879`.
