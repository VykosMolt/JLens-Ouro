# Attempt05 closure documentation review

**Verdict: PASS_WITH_FINDINGS.** The rewritten README, REPORT, CLAIMS and SPENDING_AND_SHUTDOWN, the three appended RUN_LOG entries and the two new JSON records are substantively accurate against primary evidence. There is one major completeness gap, several minor issues and a set of nits. Nothing states or implies that the native gate passed, that the cause is proven, or that any score, interval or plot exists. This review does not relax or reinterpret the failed gate. The recording time is in `attempt05_closure_documentation_review.json`.

## Independent verification

- **native.pt on CPU.** State comparisons match exactly: 0 of 1,179,648 elements differ for virtual vs physical and for virtual vs wrapper. Logits fail `torch.equal`:
  - 112 of 147,456 elements differ, split 39/28/45 by item.
  - Maximum absolute difference 0.03125; mean 6.051051766715116e-06.
  - All values finite and BF16-exact; no sign differences.
  - BF16 code-distance histogram {1:106, 2:3, 3:1, 4:1, 6:1}.
  - Top-1, top-10 and top-100 order agree for every item.
  - Everything matches the forensics review.
- **Additional rank result.** Under a stable full-vocabulary sort, ranks change for 1,322, 1,391 and 1,374 tokens per item. The smallest affected rank is 270.
- **Manifest.** All five staged files match MANIFEST.json, and the canonical manifest hash is `144c9a93…`. The staged files are byte-identical to the preserved remote copies, and all 19 preserved entries decode to their recorded hashes.
- **Freezes and records.**
  - The original FREEZE.json (`8148c2ab…`) and bundle match their bindings.
  - The precision_v1 archive and directory match its FREEZE.json, and the archive matches the lease and bootstrap records.
  - The attempt01–04 leases, prior-round baseline files and every evidence record bound in ATTEMPT05_SPENDING_AND_SHUTDOWN.json all re-hash correctly.
  - The six monitor copies match their lease pins.
- **RUN_LOG.** The first 24,554 bytes are byte-identical to the pre-edit copy. The unchanged REPORT and CLAIMS sections really are unchanged.
- **Arithmetic.** The spending sums, remaining $3.3544320362335576, shortfalls, $2.5522609725 balance decrease and table rounding are all correct.
- **Links.** All 64 relative links resolve, and no stale phrases remain.
- **Scripts.** The observation script is read-only: it uses the read key only, hash-checks the client before import, never prints credentials and creates its output exclusively. The record script makes no network calls.

## Findings

| ID | Severity | Location | Problem | Fix |
|---|---|---|---|---|
| F01 | major | RUN_LOG.md:152 (REPORT.md:49) | Attempt05's creation and export are recorded without any authorization basis. The preceding entry left approval pending, and no local attempt05 authorization record exists. | Cite the approval record, or state that none was found locally. |
| F02 | minor | RUN_LOG.md:162 | The 18:36:20Z "no watcher/receiver/uploader/unit/timer/cron" check has no preserved record. | Preserve and cite the output, or label it an unrecorded observation. |
| F03 | minor | RUN_LOG.md:157 | "Root authorized teardown at 18:31:38Z" comes only from the file mtime; the record has no timestamp. | Say so. |
| F04 | minor | README.md:18, REPORT.md:57, RUN_LOG.md:162 | "Diagnose the shape-dependent computation" presupposes an unproven mechanism. | Name it as the leading hypothesis only. |
| F05 | minor | SPENDING_AND_SHUTDOWN.md:21 | "No further paid work fits the existing ceiling" overstates the evidence. Only the $3.77 and $6 admissions are shown not to fit $3.354432. | Restrict the statement to those two. |
| F06 | minor | CLAIMS.md:12 and 16, README.md:5, REPORT.md:3, RUN_LOG.md:160 | "Inconclusive" is the plan's term for an evaluated result, and the RUN_LOG heading says "closed". | Use "not evaluated (no outcome)", state that the one-time evaluation has not occurred, and call the confirmation suspended. |
| F07 | minor | REPORT.md:49 | "At the user's explicit instruction, only the setup deadline was extended, 18:20:01Z→18:35:01Z." The quote "Extend the deadline" names neither the field nor the amount. | Attribute the choice of field and +15 min to root, under review. |
| F08 | minor | README.md:5 | "Recovery complete", yet 8 of 16 binaries remain unrecovered. | "Recovery work concluded (8 of 16 available)." |
| F09 | minor | REPORT.md:39 (CLAIMS.md:20) | Only top-1/top-10 agreement is reported; full-vocabulary ranks do change below the top 100. | Add the deeper rank changes. |
| F10 | nit | RUN_LOG.md:157, REPORT.md:29 | "Five development-stage files": four are run-level metadata in a failed final manifest. | Describe them accurately. |
| F11 | nit | REPORT.md:3 | The gate is described by its logit check only; it has three equality checks. | Mention all three. |
| F12 | nit | RUN_LOG.md:152 | "Every scientific file … unchanged" overstates the review, which compared lease records. | Narrow the wording. |
| F13 | nit | RUN_LOG.md:162 | Remaining $3.354432036233558 differs in its last digits from the JSON's 3.3544320362335576. | Use the JSON value. |
| F14 | nit | RUN_LOG.md:160 | Entry 3 is partly retrospective but unlabeled, and "Root rechecked" is unrecorded. The substance was re-verified here. | Label it and reference a record. |
| F15 | nit | REPORT.md:49 | Eight rsync children got SIGKILL, and the initiator is not recorded. The docs correctly attribute no intent. | Add "initiator not recorded". |
| F16 | nit | SPENDING_AND_SHUTDOWN.md:17 | Pod billing probably lags (about 2.81 GPU-h vs 3.62 h), and the bound exceeds the balance decrease by only $0.006328. | Note both. |
| F17 | nit | record_attempt05_spending_shutdown.py:60 | Missing checks: manifest canonical hash, precreate/lease consistency, caps from the lease, forensics/preservation status. scientific_status is hard-coded. | Add these in future records. |
| F18 | nit | observe_attempt05_final_account.py:30 | An existing `__pycache__` .pyc could bypass the source hash check. It was verified equivalent here, so there was no impact. | Use a fresh pycache prefix. |
| F19 | nit | CLAIMS.md:17 | "Reproduces the frozen runtime precision" is really a settings-record match. | Rename it. |
| F20 | nit | REPORT.md:57 | A revised criterion would be chosen after seeing attempt05's discrepancy. | Require justification independent of the observed magnitudes. |
| F21 | nit | CLAIMS.md:22 | "Retrieved and hash-matched" is true locally, but the controller handoff stayed pending (PENDING.json, phase verifying_source). | Say so. |

## Not performed

- Provider-side confirmation of pods, billing and balance (no network allowed).
- Proof that the observation ran exactly once.
- The 18:36:20Z local process check (no record exists).
- User-authorization content beyond local records.
- Re-derivation of the amended lease bytes.
- Any test of the mismatch mechanism (would need a GPU or model execution).
