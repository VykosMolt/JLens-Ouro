# Huginn pilot budget — 8 September 2026

**Propose an additional $10 for Huginn, bringing the proposed Ouro-plus-Huginn budget to $20.** This covers one N100 Huginn pilot at eight recurrent passes, using the same $0.34/hour RTX 4090 Community offer as Ouro if its validated retained graph fits. It is an allocation pending a complete Huginn prompt benchmark, not a measured or guaranteed completion price. No rental or spending has been authorized by the budget question.

The pilot includes all four recurrent block locations per pass (32 source cells), the final coda-block residual target, all 5,280 output directions, and 100 calibration paragraphs with the established T128/position policy. It includes raw logit-lens, trained-coda and J-Lens evaluation on the frozen comparable concept population, with paired initialization and the predetermined state-seed sensitivity. Matching calibration size means N100: the earlier feasibility document's N10 example cannot be substituted to lower this quote. Use the calibration texts paired with a preselected Ouro fit and establish tokenizer eligibility before inspecting results. One Huginn fit does not estimate Huginn fit-to-fit variance. This quote does not add five Huginn refits or Ouro target/position controls. Huginn remains after the Ouro adjudication, as requested.

The [pinned architecture inspection](../followup_2026-09-07/huginn_feasibility.md) gives width 5,280 and MLP width 17,920, versus Ouro's 2,048 and 5,632. Huginn executes 36 transformer blocks at R8, including prelude and coda, and has a 2d-to-d adapter at each recurrence. It needs 2.578 times as many output derivative directions. The width alone is therefore not a runtime multiplier.

A transparent arithmetic scenario counts a block's linear work as `8*d*d + 6*d*mlp_width` FLOPs per token, and a recurrence adapter as `4*d*d`. Multiplying full executed linear work by the output-direction count gives:

```text
ratio = ((36 * F_Huginn_block + 8 * F_Huginn_adapter)
         / (192 * F_Ouro_block)) * (5280 / 2048)
      = 3.8363
```

Applying that proxy to the [measured local Ouro B8 prompt time](optimization/RESULTS.md) of 192.3013439 seconds gives:

| Component | Conditional calculation | Cost |
| --- | --- | ---: |
| One N100 Huginn fit | 100 × 192.3013439 × 3.8363 seconds = 20.49 hours, at $0.34/hour | $6.97 |
| Setup, validation, evaluation and retrieval allowance | Two additional GPU-hours | $0.68 |
| Temporary storage | 50 GB for 22.49 hours, at $0.10/GB/month with a 30-day month | $0.16 |
| Arithmetic scenario total | Assumes comparable effective throughput | **$7.80** |
| Proposed Huginn allocation | Allows some margin over this scenario | **$10** |

This is a work-based scenario, not a Huginn timing measurement or an upper bound. Full executed blocks overcount operations before the earliest recurrent source in repeated backward; attention, normalization, kernel efficiency, batch choice and coda evaluation cost are not captured accurately by this proxy. The 4090 itself also remains unbenchmarked. The two-hour operational allowance is an allowance, not a measurement. Do not label the $7.80 scenario an expected bill with known confidence.

The [public GPU catalogue](gpu_catalogue_after_optimization.json) was refreshed today at 07:49 UTC and quotes 24 GB VRAM, 50 GB host RAM and 20 vCPUs for the $0.34/hour offer. BF16 Huginn weights previously occupied about 6.8 GiB locally. One R8 bank of 32 dense FP32 Jacobians occupies 3.323 GiB of host memory; two banks need 6.647 GiB before transient copies and other allocations. These sizes support trying the 24 GB card, but inference memory and tensor sizes do not prove its retained backward graph fits. Replicated random states and compression must pass Huginn-specific validation; Ouro optimization results cannot stand in for that validation. Do not silently switch to a more expensive GPU or reduce recurrence/calibration scope if the benchmark fails.

The storage rate and billing policy were rechecked in [RunPod's Pod pricing documentation](https://docs.runpod.io/pods/pricing). Complete adapter checks locally where feasible; measure a complete rented-GPU prompt before committing to N100 and project the remaining fit plus evaluation against the allocated amount. Preserve the existing prediction exactly: “If J-Lens recovers content outside the native output basis, its advantage over raw logit lens should be larger in Huginn than in Ouro.” No new Huginn result exists.
