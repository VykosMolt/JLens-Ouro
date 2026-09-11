# Historical comparison review

Passed for the exact source recorded in `historical_comparison_independent_review.json`.

The review verified the fixed historical supplement → COMPLETE → seed-seal records, actual cached tensors/ranks, initializations/summaries, and the new receipt-to-file mapping. The caller must provide the independently inspected actual receipt SHA; the tool checks the canonical manifest and full successful validation chain. Every parsed input is tied to its checked bytes and descriptor.

Fifteen checks passed. A synthetic hardlinked self-pair compared 26 real historical arrays/tensors totaling 74,724,904 numeric entries, plus both seeds' initialization/summary JSON, with zero differences. Wrong receipt pins, failed validation/checks, wrong manifest identity and incomplete checked-file lists were rejected. Parent swaps during hashing and all three JSON/NPZ/Torch consumers were rejected. Signed zeros are correctly distinguished bitwise.

One generic helper limitation is outside the accepted data scope: adjacent uint64 values above 2**53 can have understated absolute/RMS difference after float64 conversion. Mismatch counts remain exact. The actual accepted rank/token integers are small and tensors are FP32; root accepted this bounded scope.

The fixture is synthetic, uses only hardlinks to historical payloads, and has no actual external acceptance or new H outcomes. The missing historical bank remains missing. The tool reports differences and does not authorize another fit or select a result.
