# Supplement: actual inputs and saved evaluation integrity

The upstream and downstream checks passed. This supplement preserves the earlier [RECOVERY_LEDGER.json](RECOVERY_LEDGER.json) unchanged.

- **Eight fit identities:** actual local calibration JSON hashes match the sealed runtime. Every ordered UTF-8 paragraph hash and the canonical ordered-text digest match the fit identity. The actual task input JSONs, run specification and other bound input files checked by this supplement match their recorded hashes. Huginn's retained tokenization/initial-state recipe exactly matches its runtime record, including all 100 declared token lengths and valid-position counts. This supplement does not retokenize calibration text.
- **Four completed evaluation trees:** original OWNER→COMPLETE→section-seal chains pass. Every evaluated fit identity, final-lens checksum, final pointer and retained checkpoint pointer matches the corresponding original fit records. This includes all main fits, both control profiles and Huginn. Binary availability remains a separate condition for missing estimators.
- **23 retained evaluation binaries:** 19 NPZ archives and four CPU tensor caches pass full present-file hashes and stat guards, loadability, exact expected fields, dtypes, shapes and finite-value checks. NPZ loading uses `allow_pickle=False`; caches use `weights_only=True, mmap=True`. Main/controls rank validation checks full vocabulary bounds, supported columns, `-1` padding, own-slot references and identical token aliases. Huginn checks its joint-name support, rank padding, own-slot references and coda/native endpoint. Main identity and native-exit endpoints also agree.
- **Saved-cache consistency:** cached native logits' argmax equals the already saved native token arrays. The main final-target logits match the final native-exit logits bitwise. Four paired control rank exports exactly equal the declared slices of the saved main/raw arrays. The saved score archive has its exact 97 fields, expected eligibility mask/support and consistent saved contrast arithmetic. No rank rescoring or new model outcomes were generated.
- **Duplicate preservation:** the seven numerical files in the older main-only handoff exactly match the corresponding files in the later Ouro handoff by full-file SHA256. All consumed-file stat guards still match.

The recovery categories remain **7 original complete binaries + 1 exact reconstruction + 8 unresolved binaries**. The exact reconstructed Ouro fit02 SHA256 remains `101f31db6aa56d97fbae7ecb3e2f241ef1805000484d0e22f5eba5fc0b9acde8`.

[RETAINED_INPUTS_SUPPLEMENT.json](RETAINED_INPUTS_SUPPLEMENT.json) records the actual and expected file hashes, every array/tensor's dtype and shape, upstream text and downstream bank bindings, prior-ledger hash, and hashes of the recovery and validation scripts. [audit_retained_inputs.py](audit_retained_inputs.py) is the executable reproduction. It reuses only the historical portable seal validator, rank validators and CPU cache loader; it does not invoke model loading, fitting, inference, evaluation or bootstrap analysis.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/moloch/ouro_project/venv/bin/python research/confirmation_2026-09-09/artifacts/audit_retained_inputs.py
```
