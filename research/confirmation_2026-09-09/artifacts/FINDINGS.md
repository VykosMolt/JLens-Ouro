# Estimator recovery and numerical verification

The missing Ouro fit02 final bank was reconstructed from its authenticated complete N100 FP32 checkpoint. The new file is **byte-identical to the historical sealed final file**, not merely numerically equivalent:

- File: [reconstructed/ouro/fit_02/lens.pt](reconstructed/ouro/fit_02/lens.pt)
- Size: **1,602,276,737 bytes**
- SHA256: `101f31db6aa56d97fbae7ecb3e2f241ef1805000484d0e22f5eba5fc0b9acde8`
- All **801,112,064** FP16 entries independently match the checkpoint's FP32 sum divided by 100, rounded to FP16.
- Its first **179,527,232** bytes also match the retained partial final file. The partial was preserved unchanged.

There are now **8 complete binaries out of the 16 historically recorded binaries**: seven originally retained, plus the exact fit02 reconstruction. The complete usable estimators are Ouro main fit01 and fit02, the penultimate control, and both sampled-sum and diagonal position controls. The position controls share one final file.

| Estimator | FP32 N100 checkpoint | FP16 final bank | Full conversion comparison |
|---|---|---|---|
| Ouro main fit01 | Authenticated | Authenticated | 191 matrices; all bits equal |
| Ouro main fit02 | Authenticated | Reconstructed, historical SHA256 equal | 191 matrices; all bits equal |
| Ouro main fits03–05 | Absent | Absent | Unavailable |
| Ouro penultimate fit01 | Authenticated | Authenticated | 190 matrices; all bits equal |
| Ouro positions fit01 | Authenticated, both arms | Authenticated, both arms | 382 matrices; all bits equal |
| Huginn fit01 | Truncated | Absent | Unavailable |

Full-file SHA256 checks with device/inode/size/mtime/ctime guards covered every retained physical copy of a known binary, including four partial copies and the extra complete fit01 final copy. All eight original complete physical files and the reconstructed file were loaded on CPU with `weights_only=True, mmap=True`. Exact top-level schemas, declared integer source keys, arm names, geometry, dtypes, CPU storage, gradient state and finite values passed. The FP32 checkpoints and FP16 finals have 2,048×2,048 matrices; declared support is 0–190 for main/positions and 0–189 for penultimate.

The independent conversion comparison uses NumPy FP32 division by `np.float32(100)` followed by FP16 conversion and compares the full `uint16` representations, including signed zero. It imports no producer code. Across all five available estimator arms it compared **4,001,366,016 entries**, with **zero differing entries**. The ledger retains 60 deterministic numerical samples with FP32 and FP16 values and integer bit representations.

The audit also authenticated the original small-file receipt chains for all five main fits and both control profiles, including OWNER identity digests, LATEST/COMPLETE pointers, seals, metadata hashes, completed paragraph prefixes, all 100 diagnostic row identities, and the exact conversion source bound into each fit. Receipt integrity does not replace absent binary verification for fits03–05.

Huginn's only retained September N100 checkpoint contains **1,745,780,736 of 3,568,444,419 bytes**; **1,822,663,683 bytes are missing**. Its full present-file SHA256 remains `099a99fd3b1aba3ac06112411ec06d5aac5ce423f245c3cd64835150bc890473`. It was never deserialized. The retained Huginn OWNER, pointers, checkpoint seal and metadata authenticate against the accepted evaluation owner and prior supplementary review. All 100 diagnostic row identities were checked. The final bank, final seal and final metadata remain absent.

The repository search included ignored/hidden staging files and retained source-archive member lists. Supplemental searches covered `/tmp` and `/home/moloch/ouro_project`, excluding environments, Git metadata, node modules and symlink directories. The older July Huginn feature/probe files and model snapshot links are not September N100 Jacobian contributions. No complete Huginn checkpoint, earlier September checkpoint or per-paragraph derivative banks were found. Saved rank outcomes and activation caches cannot reconstruct the missing Jacobian matrices. This conclusion is bounded to the searched local storage.

No historical file was rewritten. All recorded source/partial stat guards still matched after the work. No new model readout, inference, fit, GPU job, cloud action or score analysis was performed. This recovery does not establish complete retrieval of the original experiment.

## Evidence and reproduction

- [RECOVERY_LEDGER.json](RECOVERY_LEDGER.json): current classification of all 16 binaries, availability, hashes and script/evidence records.
- [inventory.json](inventory.json): full physical-copy hashes/stat guards and original receipt/seal chains.
- [tensor_audit.json](tensor_audit.json): exact per-layer validation, full bitwise comparisons and numerical samples.
- [huginn_retention.json](huginn_retention.json): surviving Huginn provenance, missing-byte accounting and candidate search.
- [recover_estimators.py](recover_estimators.py), [audit_huginn_retention.py](audit_huginn_retention.py), [finalize_ledger.py](finalize_ledger.py): executable audit/recovery source.

The original run used the existing Python 3.14.7 environment with Torch `2.12.0.dev20260407+cu128` and NumPy 2.4.4, limited to two CPU threads and one interop thread. CUDA was not used. `torch.save` wrote to an exclusive temporary file, fsynced it, renamed it atomically into the new directory and fsynced that directory.

For an independent repetition, use a **fresh output directory** (the script refuses to overwrite an existing reconstructed final):

```bash
PYTHONDONTWRITEBYTECODE=1 python research/confirmation_2026-09-09/artifacts/recover_estimators.py inventory --output-dir /tmp/jlens-recovery-reproduction
PYTHONDONTWRITEBYTECODE=1 /home/moloch/ouro_project/venv/bin/python research/confirmation_2026-09-09/artifacts/recover_estimators.py tensors --output-dir /tmp/jlens-recovery-reproduction
```

Both stages stream/hash or load bounded CPU matrix data from the original files. The tensor stage writes a new approximately 1.6 GB final file. A different Torch serialization version could produce different archive bytes; the independent all-entry numerical check remains the criterion for checkpoint conversion, and the output ledger separately reports historical full-file identity.
