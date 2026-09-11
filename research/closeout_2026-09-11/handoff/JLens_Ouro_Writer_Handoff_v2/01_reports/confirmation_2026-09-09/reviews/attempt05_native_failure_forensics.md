# Attempt 05 native gate forensics

The gate failure is confined to logits. All three extraction paths produce exactly equal, bit-identical finite residual states, each shaped [3, 192, 2048] in FP32 on CPU. Native and recomputed logits have shape [3, 49152], FP32 on CPU. Of 147,456 logit elements, 112 differ (39, 28 and 45 across the three historical development items). Maximum absolute difference is 0.03125; mean absolute difference is 6.051051766715116e-06. All values are finite and exactly representable in BF16.

The BF16 bit-code distance histogram for differing logits is {'1': 106, '2': 3, '3': 1, '4': 1, '6': 1}. Sign-bit differences: 0. Per-item top-1 and top-10 set agreement: [('carnival-ocean', True, True), ('amazon-language', True, True), ('mars-color', True, True)]. Detailed bit patterns, counts, statistics and source hashes are retained in the JSON.

The full recorded precision dictionary matches the frozen historical precision. Native code applies its final normalization and lm_head to full sequence tensors; the development recomputation applies unembed to a single residual vector. Sparse BF16 rounding differences caused by this shape change are the leading explanation. The saved evidence cannot isolate normalization from the projection kernel or prove that mechanism. There is no observed recurrent-index or wrapper-state mismatch. The checker correctly rejected actually different logits; this is not a serialization or torch.equal false alarm.

All five retrieved files were rehashed against the latched failed terminal manifest. These files are staging evidence, not accepted development results. The strict gate remains failed. This review used CPU-only loading and did not run a model, inspect confirmation outputs, change the worker or relax any criterion.
