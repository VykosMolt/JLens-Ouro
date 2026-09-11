# Final verification: Ouro J-Lens prospective confirmation

This verifies run `ouro_confirmation_20260911_fixed160_native1` after confirmation; the verification was not prospectively registered. No input, estimator, score, analysis rule or original record changed. The three protected roots match their start-of-pass inventory (7,937 files; bytes, modification time and SHA-256), and nothing was spent.

**Outcome: numerical differences with quantified endpoint changes; the substantive conclusion is intact. One secondary statement needs rewording.**

| Path | Evidence | Primary | Change | 95% group interval |
|---|---|---:|---:|---|
| Accepted result | RTX 5090, executed | +0.23188 | — | [+0.16289, +0.32898] |
| Executed layout replayed | Local GPU, retained 5090 states and banks | +0.23188 | 0 | identical |
| 160-, 190-, 191-, 192-row head calls | same | +0.23188 | 0 | identical; logits bit-identical |
| Single-row head calls | same | +0.23188 | 0 | identical |
| FP32 (TF32 off), FP64 head, full FP64 | Local GPU; FP64 also on CPU, with identical ranks | +0.22928 | −0.00260 | [+0.16170, +0.32517] |
| States regenerated from the prompts | Local GPU (level 3) | +0.23608 | +0.00419 | [+0.16521, +0.33769] |
| Any tie order over identical logits | Adversarial bound | +0.22604 to +0.23352 | | |
| Every logit moved by ≤1/32 or ≤1/8 | Adversarial bounds | +0.22013 to +0.24408; +0.18302 to +0.27263 | | |

The local GPU is an RTX 5070 Ti Laptop (compute capability 12.0) running the confirmation's torch build and precision dictionary. It reproduced every saved 5090 array bit for bit, but it is independent evidence, not the deployed runtime. The full tables are in the [repaired report](report/REPORT_confirmation_verified.md); every per-path number is in the [metrics](metrics/VERIFICATION_METRICS.json).

## Answers

1. **Did the scorer shapes change the primary result, and by how much?**
   - **Packing:** no measurable change. On the local GPU, 160-, 190-, 191- and 192-row calls give the same logits as the executed layout. Single-row calls change deep ranks but no endpoint quantity.
   - **Precision:** against exact FP64 arithmetic over the same stored tensors, the executed BF16 path raised the primary by 0.00260. Under FP64, fit01 intended recovery is 0.00104 lower and raw intended recovery 0.00156 higher; two fit01 and three raw intended band hits change.
   - **Ties:** tie order can move the primary by at most −0.00584 to +0.00164.
   - **Not measured:** parity on an RTX 5090 for layouts the run did not execute.
2. **Do the early deficit and estimator-control conclusions remain?**
   - **Early deficit:** yes. All six early-loop contrasts stay negative, with simultaneous intervals excluding zero, on every path.
   - **Estimator controls:** fit01 − penultimate, sampled-sum − diagonal, fit02 − raw and sampled-sum − raw keep their signs and zero-exclusion. Penultimate − raw stays positive, with a lower bound of +0.007 to +0.008.
   - **Exception:** diagonal − raw in the primary band has a lower bound of −0.00340 (accepted), −0.00400 (FP64) and +0.00037 (regenerated states). Report it as borderline, not as including zero.
3. **What can be reproduced from inputs, and what only from saved outputs?**
   - **Tables from saved scores:** reproduced exactly, both by the frozen analysis and by an independent checker without producer code. The checker's unperturbed fixture passed, and it detected all 13 deliberate perturbations.
   - **Rescoring from verified banks and retained states:** exact for all six arms and 160 items on the local GPU, and repeated from the preservation bundle.
   - **Replay from the frozen prompts:** runs, but is not bit-identical off the 5090. 37 of 160 items regenerate identically, and the endpoint moves by +0.00419.
   - **Fitting:** impossible for the original estimators, because every FP32 checkpoint was purged.
   - **Saved outputs only:** the historical Huginn pilot, old-item calibration stability, the checkpoint-to-bank conversion audit, and the 5090-side diagnostic measurements.
4. **What remains lost or unverified?**
   - **Lost:** 43 purged files (33 distinct contents). These include all four Ouro FP32 checkpoints, the truncated Huginn checkpoint and both seed caches, every copy of the historical Ouro common cache, optimization binaries, and application-era lens files and probe caches.
   - **Unresolved:** 19 application-era files recorded in the private Hub repository `Vykos/ouro-jlens-results`. This machine has no credential to check them.
   - **Unverified on an RTX 5090:** native output for the 160 items, the non-executed layouts, and bit-level state regeneration.
   - **Single-device storage:** every copy, including the new 14.97 GB bundle, is on one NVMe disk.
5. **Is there a concrete verification blocker before writing up the narrow Ouro result?**
   - **Blocker:** none.
   - **Before writing up:** reword diagonal − raw, keep the scope limits in the repaired report, and disclose the single-disk storage.
   - **Optional closing check:** the RTX 5090 scoring-shape check is prepared (288 MB of inputs, no banks). It would need destination-specific export approval, and the unchanged controller would admit it at $1.150788, within the $1.804172 remaining. No conclusion above depends on it.

## Reviews, reconciled

- **[Scoring path](reviews/scoring_path/SCORING_PATH_REVIEW.md):**
  - No deviation from the frozen score definition.
  - Its three open gaps are now measured: logits never recomputed, untested 190/191-row calls, ties decided at the boundary.
  - Its statement that the model weights were absent locally was wrong: the pinned snapshot is in the local Hugging Face cache and hash-verified.
- **[native_check_v1 and checker pin](reviews/native_check_and_pin/REVIEW.md):** pass with findings. The changes were confined to the stated verification behavior and provenance. The report now records:
  - the unlisted budget change;
  - the TF32 setter removal inherited from precision_v1;
  - the post-outcome pin;
  - "every scientific field".
- **[Artifact audit](reviews/artifact_audit/ARTIFACT_AUDIT.md):** the basis of the [preservation ledger](preservation/PRESERVATION_LEDGER.json). The four banks map to the six arms; Hub recovery was not possible.

## Records

- [RUN_SPECIFICATION.json](RUN_SPECIFICATION.json): model revision, bank hashes, tokenizer, population, controls, score, environment and indexing (virtual 169–180 = loop 4, physical 26–37).
- [Metrics](metrics/VERIFICATION_METRICS.json): before and after values per path, hit changes, logit differences, ties, margins, native output, development shapes, coverage and checker results.
- [Repaired report](report/REPORT_confirmation_verified.md) and [claim-change log](report/CLAIM_CHANGES.md).
- Preservation:
  - [ledger](preservation/PRESERVATION_LEDGER.json);
  - bundle manifests ([core](preservation/BUNDLE_MANIFEST_core.json), [verification](preservation/BUNDLE_MANIFEST_verification.json));
  - [restoration check](preservation/RESTORATION_CHECK.json).
- [Reproduction commands](REPRODUCE.md) and the [independent checker](checker/README.md).
- [Spending and shutdown](spending/SPENDING_AND_SHUTDOWN.md).
- [Protected-root comparison](inventory/protected_after_comparison.json).
  - Its git check for `/home/moloch/ouro_project` reports a difference only because the starting record kept just the last 2,000 characters of `git status`.
  - The current status ends with exactly those characters, and no file in that repository changed during the pass.
