# JLENS_COLD_ARCHIVE_20260911 — index

Created 2026-09-11T19:37:23.081404+00:00 – 2026-09-11T20:44:09.141680+00:00. Destination: SanDisk Extreme Portable SSD, ext4, UUID 36df468b-3bf9-41a9-ace8-edccfb292434. Manifest sha256 `77a13e9f6873b804578a52af386f50dd41fef06cad12d6d1fa06305f4a6d11ec`. 16200 files, 23 symlinks, 52.15 GB, copy problems: 0.

| Root (original absolute path, preserved under tree/) | Files | Symlinks | GB |
|---|---:|---:|---:|
| /home/moloch/jacobian-lens/research/verification_2026-09-11/preservation | 7644 | 0 | 15.06 |
| /home/moloch/jacobian-lens | 8077 | 0 | 23.12 |
| /home/moloch/ouro_project/artifacts/jlens | 354 | 0 | 0.46 |
| /home/moloch/ouro_project/artifacts/hf_cache/hub/models--ByteDance--Ouro-2.6B | 13 | 13 | 5.34 |
| /home/moloch/ouro_project/artifacts/hf_cache/hub/datasets--Salesforce--wikitext | 6 | 2 | 0.16 |
| /home/moloch/ouro_project/src/ouro_jlens | 33 | 0 | 0.00 |
| /home/moloch/ouro_project/docs/jlens | 54 | 0 | 0.00 |
| /home/moloch/.cache/huggingface/hub/models--Vykos--ouro-jlens-results | 19 | 8 | 8.01 |

## Top-level files
- `ARCHIVE_MANIFEST.json` — every file: source path, archive path, bytes, sha256 (source and destination), mtime, inode/nlink; symlink targets.
- `ACCEPTANCE_RECEIPT.json` — destination identity, counts, manifest hash, isolated restoration result, supplement, risks.
- `DEDUP_TABLE.json` — identical-content groups (informational; every path was copied).
- `SECRETS_FLAGGED.json`, `EXCLUSIONS.json` — what was deliberately left out and why.
- `DEPENDENCIES.json` — venv freeze, external public models not copied, private Hub repo status.
- `git/` — `git bundle --all` of jacobian-lens and ouro_project, status, tracked diff, log.
- `supplement_closeout_2026-09-11/` — this round's outputs (reviewer checks, local-exit inventory, methods, manuscript v2, handoff zip, ledgers, purge allowlist), separately manifested.

## Where the key evidence is
- Preservation bundle (banks, accepted payload, model snapshot, small files of both rounds): `tree/home/moloch/jacobian-lens/research/verification_2026-09-11/preservation/bundle/` with its `BUNDLE_MANIFEST_*.json`.
- Confirmation round (frozen plan, benchmark, accepted payload with states, analysis): `tree/home/moloch/jacobian-lens/research/confirmation_2026-09-09/`.
- Refit round (five fits, controls, Huginn pilot analyses, retained banks): `tree/home/moloch/jacobian-lens/research/refit_round_2026-09-07/`.
- 7 Sep follow-up: `tree/home/moloch/jacobian-lens/research/followup_2026-09-07/`.
- Final verification (replays, checker, metrics): `tree/home/moloch/jacobian-lens/research/verification_2026-09-11/`.
- Application-era evidence (evaluations, lens sidecars, purge manifest): `tree/home/moloch/ouro_project/artifacts/jlens/`; application documents and the MATS submission (private): `tree/home/moloch/ouro_project/docs/jlens/`; custom code: `tree/home/moloch/ouro_project/src/ouro_jlens/`.
- Model snapshot with real blobs: `tree/home/moloch/ouro_project/artifacts/hf_cache/hub/models--ByteDance--Ouro-2.6B/`; calibration corpus parquet: `.../datasets--Salesforce--wikitext/`.
- Application-era final-target bank exit3 and four shards (only surviving copies): `tree/home/moloch/.cache/huggingface/hub/models--Vykos--ouro-jlens-results/blobs/`.