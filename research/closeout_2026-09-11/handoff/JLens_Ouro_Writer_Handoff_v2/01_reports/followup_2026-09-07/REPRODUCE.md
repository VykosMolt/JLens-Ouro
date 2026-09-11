# Reproducing the follow-up

All work here was performed on 7 September 2026 after the submitted application. Original source artifacts under `/home/moloch/ouro_project` are read-only inputs. This folder is the new deliverable in the shared writable checkout.

The analysis uses the existing `/home/moloch/ouro_project/venv/bin/python` environment: NumPy 2.4.4 and Matplotlib 3.10.9, with scikit-learn for the bounded probe refits. No GPU is needed for the completed analyses. No model inference or Jacobian fitting was performed in this follow-up. Scripts use explicit local source paths near their tops; adjust these constants if moving the input bundles to another machine.

Run from `/home/moloch/jacobian-lens`. Thread limits keep the small matrix calculations from oversubscribing the CPU:

```bash
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MPLCONFIGDIR=/tmp/jlens-followup-mpl
export PYTHONDONTWRITEBYTECODE=1
FOLLOWUP_PY=/home/moloch/ouro_project/venv/bin/python
FOLLOWUP_DIR=research/followup_2026-09-07

"$FOLLOWUP_PY" "$FOLLOWUP_DIR/analyze_followup.py" --part all
"$FOLLOWUP_PY" "$FOLLOWUP_DIR/sample_examples.py"
"$FOLLOWUP_PY" "$FOLLOWUP_DIR/audit/sensitivities.py"
"$FOLLOWUP_PY" "$FOLLOWUP_DIR/probe/audit.py" --refit
```

The first script also accepts `--part layers`, `exits` or `correctness` and `--output PATH`. It was executed in that priority order during the follow-up. The result files contain the historical score as the primary comparison, with explicitly named sensitivity results; regeneration does not overwrite the original application artifacts. Plot PDFs include export metadata, so byte hashes can change on regeneration without a scientific change.

The main input bundle is `artifacts/jlens/retrieved/jlens-b300-20260905-0359/eval/n100_exit3`. Exit diagnostics use `eval/b300_local_allexits_strict` and `eval/b300_local_allexits_pos-2_strict`, each with its own native outputs. The probe uses `probe/cv_all648` and its `probe/n80_v2/gpu_cache.npz` feature cache. Exact input hashes and package versions are in [inputs.json](results/inputs.json) and [the probe manifest](probe/artifact_manifest.json). [The source manifest](sources/manifest.json) identifies downloaded reading copies, which are ignored by Git rather than bundled for republication.

Independent checks, which do not import the main scoring/statistical helpers:

```bash
"$FOLLOWUP_PY" "$FOLLOWUP_DIR/audit/audit_existing.py"
"$FOLLOWUP_PY" "$FOLLOWUP_DIR/audit/verify_followup.py"
"$FOLLOWUP_PY" "$FOLLOWUP_DIR/probe/verify_main_followup.py" > /tmp/jlens-followup-independent-verification.json
```

These were run successfully. The first audit independently reproduces the eight submitted effects and their intervals. The layer verifier independently reconstructs item scores, all 384 layer contrasts and intervals, depth summaries, simultaneous bands and all correctness strata. The exit verifier checks both positions, every retained window, target identities, missing-field handling, and paired subgroup/exit statistics. [Layer verification](audit/FOLLOWUP_VERIFICATION.md) and [exit/correctness verification](probe/VERIFY_MAIN_FOLLOWUP.md) state their numerical checks and limitations. The probe's five selected classifiers also exactly reproduce the saved ranks before any new variant is scored.

All main bootstrap calculations use 20,000 draws. Historical effects use seed 0. The two tasks' layer bands use independent seeds 0 and 1, with their maximum centered errors joined across the 384-cell family. Exits use seed 20260907; correctness uses independently sampled passing/failing items and a shared concept-cluster draw for both strata. Cluster means preserve item weighting; they are not unweighted averages of cluster means. All intervals condition on retained fits. Method ribbons, fixed depth blocks, slopes and subgroup interactions remain pointwise. The simultaneous bands are approximate and unstudentized.

The main code writes fixed-layer hit, control, excess and paired differences to [CSV](results/layerwise.csv) and [JSON](results/layerwise.json). [item_scores.npz](results/item_scores.npz) retains per-item values. JSON keeps unavailable exit ranks/KL as `null`; CSV leaves them empty. The `actual_token_*` columns concern rank of the model's actual argmax token, not full top-k set overlap. Source ranks are zero-based; the human-readable random examples convert to ordinary one-based ranks.

The source and generated-output inventory is [MANIFEST.json](MANIFEST.json). Literature and Huginn feasibility were independently reviewed; the lead reconciled the review and clarified the distinction between a documented target difference and an untested position-related explanation. [Review](REVIEW.md).
