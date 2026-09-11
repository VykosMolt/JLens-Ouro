# Attempt05 closure documentation review v2

**Verdict: PASS_WITH_FINDINGS.** This re-verifies the repairs to the v1 findings and adds a line-by-line necessity and writing review under the user's standard. All v1 major and minor findings are resolved or acceptably handled, and nothing now implies gate passage, a proven cause or existing scores. Two minor wording inaccuracies and some editorial nits remain. The recording time and full evidence are in `attempt05_closure_documentation_review_v2.json`.

## Integrity

- **RUN_LOG.md:** 31,613 bytes, sha256 `3916ce60…`. The first 30,578 bytes match `b74658ef…` and the first 24,554 bytes still match `f75f750a…`. The correction entry (line 165) is write-time stamped and leaves the earlier entries unchanged.
- **Reverted text:** the REPORT "Frozen new-item test" paragraph and the CLAIMS closing sentence are byte-identical to the pre-edit originals.
- **Changed files:** since v1, only the four docs, RUN_LOG and the v1 review changed. All bound records, freezes, bundles, native.pt and the manifest re-hash unchanged.
- **Links:** all 65 relative links resolve, including `#continuing-the-confirmation`.
- **Authorization search:** a broader search still finds no attempt05 authorization record, so REPORT's statement is accurate.

## v1 findings

- **Resolved:** F01–F12, F14–F16 and F19–F21.
- **F10:** the log wording matches the output contract's development stage, so no change was needed.
- **F17 and F18:** acknowledged for future records.
- **F13:** still a nit. The stated rationale is wrong: `3.354432036233558` does not parse to the recorded `3.3544320362335576` (one double apart), though the difference is immaterial.

## Remaining findings

| ID | Severity | Location | Problem | Fix |
|---|---|---|---|---|
| N01 | minor | REPORT.md:47 | "root extended only the setup deadline by 15 minutes under independent review" | Write: 'root extended only the setup deadline by 15 minutes; independent review then passed.' |
| N02 | minor | resources/SPENDING_AND_SHUTDOWN.md:17 | "The attempt05 pod billing record ($1.988154) probably lags, since it covers about 2.81 of 3.62 hours." | Write: 'The attempt05 pod billing record ($1.988154) probably lags: its GPU charge equals about 2.81 hours at $0.69/h, against 3.62 lease hours.' |
| F13 | nit | RUN_LOG.md:162 | "Remaining authorization under the $25 cap is $3.354432036233558." Left unchanged on the grounds that it is the same value in a different format. | No change is required. If the log is corrected again, quote 3.3544320362335576 or the rounded $3.354432. |
| N03 | nit | RUN_LOG.md:167 | Correction (1): "the controller required only a user-notification timestamp (14:54:31Z)". | If a further correction is appended, say the notice timestamp equals the pre-creation observation time and is not evidence of user contact or approval. Otherwise no change. |
| N04 | nit | REPORT.md:39 | "The criterion is exact equality, so attempt05 is not accepted after the fact. It was the first attempt in this round to reach this gate." | Delete both sentences. |
| N05 | nit | REPORT.md:49 | "leaving $3.354432, which is below the frozen $3.77 attempt cap" and "See [spending and shutdown](...) and the [run log](...)." | Drop the 'which is below…' clause and the 'See…' sentence. |
| N06 | nit | REPORT.md:37 | "Deeper orderings differ, the earliest at zero-based rank 270" (CLAIMS.md:20 likewise). | Write 'the earliest at zero-based rank 270 (mars-color)'. |
| N07 | nit | README.md:5 | "Status: recovery concluded with 8 of 16 historical binaries available; the prospective confirmation is inconclusive because it was not evaluated, and is suspended." | Write: '**Status:** recovery concluded (8 of 16 historical binaries available). The prospective confirmation is inconclusive: it was not evaluated and is suspended.' |
| N08 | nit | CLAIMS.md:3 | "Status: **the prospective confirmation is inconclusive: it was not evaluated and is suspended.**" | Write: 'Status: **inconclusive — the prospective confirmation was not evaluated and is suspended.**' |
| N09 | nit | README.md:15 | "... and no pods remain." | Write 'and the final observation lists no pods'. |
| N10 | nit | REPORT.md:6 | "The frozen endpoint, the item-weighted mean excess hit@10 difference over loop 4, physical layers 26–37, was never computed." | Write: 'Unresolved: the frozen endpoint (item-weighted mean excess hit@10, fit01 minus raw, loop 4 physical layers 26–37) was never computed.' |
| N11 | nit | resources/SPENDING_AND_SHUTDOWN.md:19 | "Attempt05 failed the frozen exact native-logit gate before any new-item scoring. Its evidence is preserved, and root explicitly authorized its teardown." | Write: 'Root explicitly authorized attempt05's teardown after its failure evidence was [preserved](...).' On line 3, write 'lease records verified termination'. |
| N12 | nit | CLAIMS.md:22 | "Deployed hashes, verified shutdown and preservation of failure evidence are demonstrated." | Restore the [retry deployment](controller/DEPLOYED_PRIMARY_RETRY_02.json) link. |

## Not performed

- Provider, network, SSH or GPU checks (not permitted).
- Authorization content outside local records (not available).
- Repeating the rank recomputation (review v1 values reused).
