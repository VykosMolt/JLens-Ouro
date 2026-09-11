# Independent H controller review

Status: **rejected**. The three original retry accounting findings are fixed in `lease.py` SHA-256 `a03bc0a2e517860d7cf65ed971a5da999935218766e6625fb10450d159344dfb` (78,354 bytes), and independent reproductions now reject them.

A fourth direct probe still fails: after the function parses a valid $1 H lease row, atomically replacing that row with a valid $2 charge makes the function return the old $1 charge together with a hash of the new $2 row. The final record recheck succeeds because hashing starts after parsing. The parsed values and recorded evidence must derive from the same read bytes, followed by unchanged-record and complete-ledger-inventory checks.

The observed 53-case suite run passed, but the receiver source changed during that run. It cannot certify the final runtime closure. A fresh independent rerun is required once the receiver and controller are frozen.

Exact rejected v1 review and proof bytes are preserved as separate artifacts. The current JSON and proof record the fixed original findings, the new failing regression and its exact lease source pin. No cloud, SSH or new confirmation results were accessed. H admission remains blocked pending this fix and separate scientific/preservation integration acceptance.
