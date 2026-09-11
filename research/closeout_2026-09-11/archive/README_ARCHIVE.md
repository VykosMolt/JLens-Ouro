# J-Lens cold archive (2026-09-11)

Copy-only archive of the surviving J-Lens/Ouro research evidence, written from the laptop (NVMe) to this external SSD on 2026-09-11. Nothing was moved or deleted at the source. Original absolute paths are preserved under `tree/`. See `ARCHIVE_INDEX.md` for contents and `ACCEPTANCE_RECEIPT.json` for verification.

**Status: hash-verified; see receipt for restoration status.** If the laptop copies are later removed, this SSD is a **single-copy archive**, not a redundant backup.

Restoration test performed from this SSD alone (network and live laptop paths hidden): every bundle file re-hashed, frozen code archive matched its freeze, all four bank files loaded (five arms), the frozen analysis (primary + 20 secondaries) reproduced from archived readouts, and raw/fit01 rescoring from archived states and weights reproduced the saved ranks. Not repeated: the full numerical sensitivity suite, regeneration from prompts.

Requirements to use the archive: Python 3.14 with the pinned packages (`DEPENDENCIES.json`, `…/legacy/requirements.lock`), a CUDA GPU for rescoring, and the released `jlens` code (archived). Secrets were excluded (`SECRETS_FLAGGED.json`).
