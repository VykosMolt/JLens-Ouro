# Attempt 05 setup deadline amendment review

**Passed for the actual user-authorized administrative amendment.** The recorded
authorization reference was checked against the user's request and agrees between
the intent and receipt. No existing file or resource was changed by this review.

Decoding the preserved original lease bytes reproduces their recorded 6,962-byte
SHA-256 `9f24d9a8f66f2625367fe95cdfdb2c062e3243f05d77e6072e50ebace88176e2`.
Changing only `setup_deadline_utc` and applying the controller's exact canonical
serialization reproduces the receipt's 6,962-byte post-amendment SHA-256
`89b024a20e1d2386f06b952dbbf498523936a03ff6c67a6ff7ccaad8b221ab26`.

The sole change was `18:20:01Z` to `18:35:01Z`, applied at `18:13:54Z` on
2026-09-10. The following limits remain identical in the original, reconstructed
amended, and inspected live state:

| Limit | Unchanged value |
| --- | --- |
| Work deadline | 19:20:01Z |
| Spending-watch deadline | 19:50:01Z |
| Provider deadline | 20:00:01Z |
| Attempt cap | $3.77 |
| Combined cap | $25.00 |

Identity, source and bundle hashes, scientific specification, config, verifier,
prior debits, billing origin, rate, and preservation floor also remain unchanged.
The live state's additional differences are five routine watcher observation
fields. All six active and monitor controller files still match their original
pins; heartbeat `18:16:36Z` confirms watcher activity after the amendment.

The helper holds the existing `.state.lock` across preservation, amendment, and
receipt publication and uses atomic replacement plus directory fsync. The uploader
recorded all 51 process groups quiescent at `18:12:50Z`; its subsequent invocation
records the new setup deadline. The local launch directory was empty when inspected.

This is an explicit exception to `lease.change`'s prohibition on extending
deadlines. The unmodified watcher rereads the amended state and continues enforcing
the unchanged work/spending limits. Receipt bindings exclude this deadline; source,
precision and native acceptance gates remain intact. No successful scientific
outcome is asserted.

The helper is suitable only as the audited one-use operation: it does not itself
establish uploader quiescence, inspect remote launch state, or explicitly reject
the state's halt/stop-reason fields. The preserved actual state has
`halt_requested=false` and no computation stop reason or stop request. These limits
and the complete evidence records are documented in
`attempt05_deadline_amendment_review.json`.
