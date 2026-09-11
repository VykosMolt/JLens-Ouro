# Final verification pass: plan and progress

This is the post-confirmation verification of `ouro_confirmation_20260911_fixed160_native1`; it was not prospectively registered. Originals, freezes, results, incident records and the MATS submission stay unchanged, and nothing was deleted or overwritten. The spending cap is unchanged: $23.195828 of $25 bound, $0 spent in this pass.

## Decisions
- **Readout replay:** uses the retained 5090 states and the four hash-verified banks. Regenerated states are a separate, labeled level-3 check; on the local RTX 5070 Ti they are not bit-identical to the 5090 states.
- **Local evidence:** the local GPU (same torch build, precision dictionary and compute capability 12.0) and the CPU are independent evidence, not deployed-runtime claims.
- **RTX 5090 run: prepared, not run.** The saved 5090 outputs already give a measured answer: the FP64 reference is device-independent and moves the primary by −0.00260. A new pod upload would need destination-specific export approval; admission would be $1.150788.
- **Bounds:** tie and perturbation bounds are adversarial bounds, not estimates.
- **Superseded metrics:** `metrics/superseded/VERIFICATION_METRICS_checker_keys_unread.json` is a first metrics build that read the checker's self-test fields incorrectly (both were null). It was replaced by a rebuild, not edited.

## Status
- [x] Verification directory, protected inventory, spending observations (start and end).
- [x] Independent reviews: scoring path, native_check_v1 and checker pin, artifact audit; reconciled in FINAL_VERIFICATION.md.
- [x] Run specification.
- [x] Readout replay: 9 layouts × 6 arms × 160 items on the local GPU, plus FP64 on CPU. Propagated through the frozen analysis.
- [x] Independent checker: saved outputs, 14 fixture cases, and 27 propagation roots.
- [x] Local scoring-shape check: native output for 160 items, saved sample rows, development shapes, regenerated states. Level-3 replay from the regenerated states.
- [x] Preservation: core bundle (7,380 files, 14.97 GB), restoration and loading check, ledger.
- [x] Repaired report, claim-change log, FINAL_VERIFICATION.md, metrics, reproduction commands, spending record.
- [x] Protected roots unchanged (7,937 files).
- [x] Verification part of the bundle: 198 files, 89.6 MB, copy hashes match. The bundled copy of this file predates this line.
