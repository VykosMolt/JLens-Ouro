# Local-exit targets: inventory, blocker, proposal, and what the retained data already show

> **Update 2026-09-12.** Steps 1 and 2 of Section 3 were executed at the author's direction with the author's Hub credential: the 13 shards were retrieved and hash-verified, merged with the frozen procedure (the exit3 merge equals the pod merge entry for entry), and applied to the accepted confirmation states under the plan frozen in `confirmation_2026-09-12/ANALYSIS_PLAN.md`. All six contrasts exclude zero: own exit − final target +22.60 / +26.50 / +34.17 points and own exit − raw +16.41 / +18.32 / +23.18 points in passes 1–3 (simultaneous 95%). Results: `confirmation_2026-09-12/REPORT.md`, `RESULTS.json`. The blocker described below is resolved; the text below is kept as the dated record.

Status 2026-09-11. Files: [LOCAL_EXIT_BANK_INVENTORY.json](LOCAL_EXIT_BANK_INVENTORY.json), [DISCOVERY_POPULATION_FIXED_BAND.json](DISCOVERY_POPULATION_FIXED_BAND.json), [discovery_population_curves.csv](discovery_population_curves.csv), [script](local_exit_inventory_and_discovery_summary.py). No lens was fitted and no paid resource was used.

## 1. Bank inventory

The only local-exit (current-pass-exit) J-Lens banks ever fitted for Ouro are the application-era N100 family on the B300 pod (2026-09-05): `n100/exit0` (target virtual 47, pass 1 exit), `exit1` (95), `exit2` (143) and `exit3` (191, the final exit). Their JSON sidecars survive and record one shared recipe that differs only in the target: 100 calibration paragraphs (prompt slice 0–100 of `wikitext_prompts_b08601e.json`, sha256 `972bd023…`, 12 Wikipedia articles), T128 with BOS, `skip_first=16`, `dim_batch=32`, stock reduction (cotangent summed over valid target positions, mean over valid source positions, count-weighted shard merge), FP32 accumulation, FP16 storage, `fit_lens.py` sha256 `007517db…`, jlens commit `581d3986`, model revision `1ed04250…`.

| Bank | Target (virtual) | Sources | Merged file | Shards |
|---|---:|---:|---|---|
| exit0 | 47 | 0–46 | purged; no local copy (sha `3693469e…`) | 4 × 25 prompts: remote only (`Vykos/ouro-jlens-results`, receipts with sha256 recorded) |
| exit1 | 95 | 0–94 | purged; no local copy (`12c1a434…`) | 4 × 25: remote only |
| exit2 | 143 | 0–142 | purged; no local copy (`a5e6e1e7…`) | 4 × 25: remote only |
| exit3 | 191 | 0–190 | local merge purged (`d3bc7c42…`); pod merge of the same shards survives in the HF cache (`d7c26297…`, 1,602,279,480 bytes) | shards 0–8, 8–32, 32–56, 56–80 survive in the HF cache; shard 80–100 remote only |

Provenance of the surviving exit3 file is complete (sidecar, receipts, hash). Whether its FP16 tensors equal the purged local merge bit for bit is untested; both are merges of the same five shards.

Also surviving: the application-era readouts of **all four banks and the raw lens on the same 148 discovery items** (`ouro_project/artifacts/jlens/eval/b300_local_allexits_strict/arrays.npz`, `allrank` [148,128,192] per lens; evaluated locally on the RTX 5070 Ti on 2026-09-05 with the pod-fitted banks; sha256 in the inventory JSON). No local-exit readout of the 160 confirmation items exists, because the local banks were never applied to them.

The confirmation-era fit01 family (fit01–fit05, penultimate, positions) is final-target only. Its recipe differs from the application-era family in calibration texts (random 100-article samples vs the 12-article prefix), `dim_batch` (8 with CUDA-graph replay vs 32), GPU/runtime and date, so an application-era local bank versus the confirmation fit01 bank would **not** isolate target choice.

## 2. Blocker for the requested comparison

The requested comparison (raw vs local-target vs final-target J-Lens on the same stored early-pass confirmation states, pass 1–3 band 26–37, 160 items, 79 controls) needs the exit0/1/2 banks. They exist only as 12 shards in the private Hugging Face repository `Vykos/ouro-jlens-results` (9.6 GB), and this machine has no credential (no `HF_TOKEN`, no token file). The required states (`common/cache.pt`, [160,192,2048] FP32, hash-verified) and the matched final-target bank (exit3, `d7c26297…`) are on disk. **No comparison was improvised.**

## 3. Proposal (bounded; no fitting unless retrieval fails)

**Step 1, retrieval only (no GPU cost).** With Jan's Hub credential, download the 12 shards and `exit3_shard_0080_0100.pt` (13 files, 11.2 GB) and verify each against the recorded sha256 in `PRESERVATION_LEDGER.json.unresolved` / the retrieval receipts. Merge each exit's four shards with the frozen `fit_lens.py merge` (count-weighted mean, deterministic); the merged files may not be byte-identical to the purged local merges, so compare tensors to the surviving exit3 pod merge as an equivalence check of the procedure. Time: download-bound (about 1 h at 3 MB/s); storage 11.2 GB plus 3.6 GB merged.

**Step 2, evaluation on the local GPU (no paid resource).** Apply raw, exit-p-local and exit3 banks to the retained confirmation states for passes 1–3 with the frozen scorer (`readouts.readout`, 79 controls, same aliases), the fixed band 26–37, the frozen groups and the frozen `analyze.resample`. Family (to freeze before running): local−final and local−raw excess differences for passes 1–3 (six contrasts, one max-t family), intended/control components descriptive, full curves descriptive. Pass-4 local = final by construction (identity check only). Runtime on the RTX 5070 Ti: about 6 arms × 160 items, minutes. The result would be labeled a **separate, application-era estimator comparison**, matched within its own family, post-confirmation, on the confirmation population. If the overlap audit's clean-subset sensitivity is wanted, run it separately.

**Step 3 only if retrieval fails: matched local-target fits in the fit01 family.** Fit three banks targeting virtual 47, 95 and 143 with fit01's exact calibration list (`calibration_fit_01.json`, seed 2026090701) and the refit recipe (`run_refits.py` SETTINGS: dense, `dim_batch=8`, CUDA-graph replay, T128, `skip_first=16`, FP32 accumulation, FP16 bank). Cost anchor from the measured B8 profile (192.3 s per paragraph for 191 sources on the laptop; cost scales with the number of source layers): (47+95+143)/191 × 100 × 192.3 s ≈ 8.0 GPU-hours at laptop speed, i.e. about $3–4 on an RTX 4090 community pod at the quoted $0.34/h plus setup and retrieval, or about 8 h on the local RTX 5070 Ti at no cost; storage 2.3 GB FP16 banks + 4.6 GB FP32 checkpoints. This would be a **new estimator family** (fit01-matched), not the application-era banks, and requires explicit authorization; none is granted by this round.

## 4. What the retained application-era readouts already show (descriptive, discovery population)

Because all four application-era banks share one recipe except the target, their retained readouts on the discovery items are a matched local/final/raw comparison on a **different population** (the 90 eligible multihop discovery items, 100 slots, 69 same-catalogue controls per slot; the same population on which the 26–37 band was selected for the final-target lens). Fixed band 26–37, same scoring rule (rank < 10, min over accepted forms, controls averaged within slot, slots within item). Descriptive percentile intervals: 20,000 item draws and 20,000 shared-concept-cluster draws (64 clusters). Not the confirmation population; not prospectively frozen; local GPU evaluation (numerically not identical to the pod: loop-4 any-layer excess 0.461 here vs 0.472 pod-side).

| Pass | Blocks to local / final target | raw excess | final-target excess | local-target excess | local − final | local − raw |
|---:|---|---:|---:|---:|---|---|
| 1 | 11–22 / 155–166 (3 loop norms) | +0.016 | +0.000 | +0.146 | **+0.146** [+0.090, +0.207] item; [+0.091, +0.209] cluster | **+0.130** [+0.079, +0.186]; [+0.081, +0.190] |
| 2 | 11–22 / 107–118 (2) | +0.036 | −0.019 | +0.150 | **+0.169** [+0.110, +0.232]; [+0.115, +0.228] | **+0.114** [+0.064, +0.168]; [+0.066, +0.166] |
| 3 | 11–22 / 59–70 (1) | +0.092 | +0.011 | +0.240 | **+0.230** [+0.154, +0.310]; [+0.160, +0.306] | **+0.149** [+0.078, +0.224]; [+0.079, +0.224] |

Intended recovery (raw / final / local): 0.021 / 0.003 / 0.154 (pass 1); 0.043 / 0.004 / 0.162 (pass 2); 0.101 / 0.021 / 0.252 (pass 3). Control recovery stays below 0.023 everywhere; the differences come from intended recovery. Pass-4 identity checks pass (the local and final bank coincide; raw equals the final bank at virtual 191).

**Interpretation, within the frozen rules.** On the discovery population the local-target lens beats both the final-target lens and the raw lens in the fixed band in every early pass: a target-dependent early-pass advantage, consistent with a transport explanation. It is not proof that averaging destroyed information, it is on the population used to select the band, it uses the older calibration prefix, and it has not been tested on the confirmation population. The final-target application-era lens recovers almost nothing early (0.003–0.021), matching the confirmation fit01 early-band pattern (0.000–0.042). Existing qualifications remain: shortening the final target by one block (penultimate) reduced the late gain, so distance alone is not established; and none of this speaks to Huginn or to a supervision explanation.
