#!/usr/bin/env python3
"""Local-exit banks: survival inventory, provenance match, and a descriptive fixed-band summary from retained
application-era readouts on the DISCOVERY population (148 items, 90 eligible multihop). No fitting, no model forward.
The requested confirmation-population comparison is blocked (local banks remote-only); see BLOCKER_AND_PROPOSAL.md."""
import json, os, csv, hashlib, collections
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
O = Path('/home/moloch/ouro_project/artifacts/jlens')
HUB = Path('/home/moloch/.cache/huggingface/hub/models--Vykos--ouro-jlens-results')
LED = json.load(open('/home/moloch/ouro_project/jacobian-lens/research/verification_2026-09-11/preservation/PRESERVATION_LEDGER.json'))
PURGE = json.load(open(O / 'PURGE_MANIFEST_2026-09-11.json'))
E = O / 'eval/b300_local_allexits_strict'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

# ---------- 1. bank inventory ----------
blobs = {b.name: b.stat().st_size for b in (HUB / 'blobs').iterdir()}
snap_links = {}
for s in (HUB / 'snapshots').iterdir():
    for p in s.rglob('*'):
        if p.is_symlink(): snap_links[str(p.relative_to(HUB / 'snapshots'))] = os.path.basename(os.readlink(p))
purged = {f['sha256']: f for f in PURGE['deleted_files']}
unresolved = {u['sha256']: u for u in LED['unresolved']}; lost = {u['sha256']: u for u in LED['unique_and_lost']}
inv = []
for ut in range(4):
    meta = json.load(open(O / f'lens/n100/exit{ut}.json'))
    cfg = meta['config']
    merged_sha = meta['output_sha256']
    status = 'LOCAL_HF_CACHE_BLOB' if merged_sha in blobs else ('LOST_no_copy' if merged_sha in lost else ('REMOTE_ONLY_unverified' if merged_sha in unresolved else 'unknown'))
    shards = []
    for rec in meta['shard_records']:
        h = rec['binary_sha256']
        st = 'LOCAL_HF_CACHE_BLOB' if h in blobs else ('REMOTE_ONLY_unverified(private Hub, receipt recorded)' if h in unresolved else ('LOST_no_copy' if h in lost else 'unknown'))
        rr = unresolved.get(h, {}).get('remote_record') or {}
        shards.append({'path': rec['path'], 'bytes': rec['binary_size'], 'sha256': h, 'prompt_range': [rec.get('start', rec.get('count')), rec.get('end')], 'status': st, 'remote_path': rr.get('remote_path')})
    # the pod-merged exit3 (d7c26297) is a second merge of the same shards
    alt = None
    if ut == 3:
        for h, f in purged.items():
            if f['path'].endswith('retrieved/jlens-b300-20260905-0359/lens/n100/exit3.pt'):
                alt = {'path': f['path'], 'bytes': f['bytes'], 'sha256': h, 'status': 'LOCAL_HF_CACHE_BLOB' if h in blobs else 'unknown', 'note': 'pod-side merge of the same five shards; different bytes from the local merge (fp16 merge order), tensor equivalence untested'}
    inv.append({'bank': f'n100/exit{ut}', 'role': 'local exit of pass %d' % (ut + 1) if ut < 3 else 'final exit (pass 4) = final-target bank of the application era',
                'target_ut_zero_based': cfg['target_ut'], 'target_virtual': cfg['target_virtual'], 'source_virtual_layers': [cfg['source_layers'][0], cfg['source_layers'][-1]], 'n_sources': len(cfg['source_layers']),
                'recipe': {'n_prompts': meta['n_fitted'], 'prompt_slice': meta['prompt_slice'], 'prompt_file_sha256': meta['prompt_file_sha256'], 'prompt_source': meta['prompt_source'],
                           'max_seq_len': cfg['max_seq_len'], 'skip_first': cfg['skip_first'], 'dim_batch': cfg['dim_batch'], 'bos_prepended': cfg['bos_prepended'],
                           'generator_sha256': meta['generator_sha256'], 'jlens_commit': meta['jlens_commit'], 'model_revision': meta['model_revision'], 'model_snapshot_sha256': meta['model_snapshot_sha256'],
                           'saved_dtype': 'float16 (jlens.lens.JacobianLens.save default)', 'reduction': 'stock: cotangent at every valid target position (sum), mean over valid source positions 16..len-2, mean over prompts (count-weighted shard merge)',
                           'created_at': meta['created_at'], 'fit_seconds': meta['seconds']},
                'merged_file': {'path': meta['output']['path'], 'bytes': meta['output']['size'], 'sha256': merged_sha, 'status': status}, 'alternative_merge': alt, 'shards': shards})
# matched-recipe check across the four banks
keys = ['n_prompts', 'prompt_slice', 'prompt_file_sha256', 'max_seq_len', 'skip_first', 'dim_batch', 'bos_prepended', 'generator_sha256', 'jlens_commit', 'model_revision', 'model_snapshot_sha256']
matched = all(inv[0]['recipe'][k] == inv[u]['recipe'][k] for u in range(1, 4) for k in keys)
confirmation_recipe_differences = ['calibration texts: first 100 wikitext paragraphs (12 articles) vs fit01 100 random paragraphs (100 articles, seed 2026090701)',
    'dim_batch 32 (B300 pod) vs 8 with CUDA-graph replay + saved-tensor compression (fit01)', 'GPU/runtime: B300 pod 2026-09-05 vs RTX 4090-class community pod 2026-09-09 runtime (torch 2.12 nightly)',
    'target: exit3 = virtual 191 (same as fit01); local banks target 47/95/143', 'stored dtype fp16 both; FP32 accumulation both']
inventory = {'schema': 'local_exit_bank_inventory.v1', 'application_era_family': inv, 'all_four_banks_share_one_recipe_except_target': matched, 'recipe_keys_compared': keys,
             'differences_from_confirmation_fit01_family': confirmation_recipe_differences,
             'retained_readouts_of_all_four_banks': {'path': str(E / 'arrays.npz'), 'sha256': sha(E / 'arrays.npz'), 'population': 'discovery set, 148 items (93 multihop, 55 order-ops), evaluated locally on the RTX 5070 Ti with the pod-fitted N100 banks on 2026-09-05', 'provenance': str(E / 'provenance.json')},
             'hf_cache_blobs_present': blobs, 'snapshot_links': snap_links,
             'credential_status': 'no Hugging Face token on this machine; the private repo Vykos/ouro-jlens-results cannot be listed or downloaded here'}
(HERE / 'LOCAL_EXIT_BANK_INVENTORY.json').write_text(json.dumps(inventory, indent=1))

# ---------- 2. descriptive fixed-band summary on the discovery population ----------
z = np.load(E / 'arrays.npz'); items = json.load(open(E / 'items.json')); tn = json.load(open(E / 'task_names.json'))
names = tn['multihop']; n_names = len(names)
elig = []
for i, it in enumerate(items):
    if it['task'] != 'multihop': continue
    slots = [s for s, (sc, l) in enumerate(zip(it['scorable'], it['leaked'])) if sc and not l]
    if slots: elig.append((i, slots))
assert len(elig) == 90 and sum(len(s) for _, s in elig) == 100
arrays = {'raw': z['logitlens_allrank'], 'final': z['jlens_exit3_allrank'], 'local1': z['jlens_exit0_allrank'], 'local2': z['jlens_exit1_allrank'], 'local3': z['jlens_exit2_allrank']}
support = {'raw': 192, 'final': 192, 'local1': 47, 'local2': 95, 'local3': 143}  # learned sources 0..target-1; identity at target for final (191)
def item_scores(a, cols):
    own, ctrl = np.zeros(len(elig)), np.zeros(len(elig))
    for k, (i, slots) in enumerate(elig):
        it = items[i]; h = (a[i][:, cols] >= 0) & (a[i][:, cols] < 10)
        owns = [o for o in it['own_index'] if o >= 0]
        for s in slots:
            j = it['own_index'][s]; c = [q for q in range(n_names) if q not in owns]
            own[k] += h[j].mean() / len(slots); ctrl[k] += h[c].mean() / len(slots)
    return own, ctrl, own - ctrl
clusters = {}
with open('/home/moloch/ouro_project/jacobian-lens/research/followup_2026-09-07/audit/dependence_clusters.csv') as f:
    for r in csv.DictReader(f): clusters[int(r['index'])] = r['concept_cluster']
groups = [clusters[i] for i, _ in elig]
rng_item = np.random.default_rng(2026091104); rng_grp = np.random.default_rng(2026091105)
def boot(values, labels, rng, draws=20000):
    values = np.asarray(values); labs = list(dict.fromkeys(labels)); idx = [np.flatnonzero(np.asarray(labels) == l) for l in labs]
    sums = np.array([values[i].sum() for i in idx]); sizes = np.array([len(i) for i in idx])
    out = np.empty(draws)
    for s in range(0, draws, 1000):
        dr = rng.integers(len(labs), size=(min(1000, draws - s), len(labs))); out[s:s + len(dr)] = sums[dr].sum(1) / sizes[dr].sum(1)
    return [float(x) for x in np.quantile(out, [0.025, 0.975])]
passes = []
for p in (1, 2, 3):
    cols = list(range(48 * (p - 1) + 25, 48 * (p - 1) + 37)); assert max(cols) < support[f'local{p}']
    sc = {m: item_scores(arrays[m], cols) for m in ('raw', 'final', f'local{p}')}
    loc = sc[f'local{p}']
    row = {'pass': p, 'virtual': cols, 'physical': list(range(26, 38)), 'blocks_source_to_local_target_range': [48 - 37, 48 - 26], 'blocks_source_to_final_target_range': [48 * (4 - p) + 48 - 37, 48 * (4 - p) + 48 - 26],
           'loop_boundary_norms_between_source_and_final_target': 4 - p, 'loop_boundary_norms_between_source_and_local_target': 0}
    for m, lab in (('raw', 'raw'), ('final', 'final_target'), (f'local{p}', 'local_target')):
        row[lab] = {'intended': float(sc[m][0].mean()), 'control': float(sc[m][1].mean()), 'excess': float(sc[m][2].mean())}
    for lab, a, b in (('local_minus_final', loc, sc['final']), ('local_minus_raw', loc, sc['raw']), ('final_minus_raw', sc['final'], sc['raw'])):
        d = a[2] - b[2]
        row[lab] = {'excess_diff': float(d.mean()), 'intended_diff': float((a[0] - b[0]).mean()), 'control_diff': float((a[1] - b[1]).mean()),
                    'item_bootstrap_95_descriptive': boot(d, [str(k) for k in range(len(d))], rng_item), 'concept_cluster_bootstrap_95_descriptive': boot(d, groups, rng_grp)}
    passes.append(row)
# pass 4 identity check: local target == final target (same bank); and identity endpoint agreement raw==final at virtual 191
same = bool(np.array_equal(arrays['final'][:, :, 191], arrays['raw'][:, :, 191]))
# per-layer descriptive curves (excess) for passes 1-3 and all methods
curves = {}
for m, a in arrays.items():
    ex = []
    for v in range(192):
        if v >= support[m] and m.startswith('local'): ex.append(None); continue
        ex.append(float(item_scores(a, [v])[2].mean()))
    curves[m] = ex
summary = {'schema': 'local_exit_discovery_population_fixed_band.v1', 'status': 'DESCRIPTIVE, post hoc, DISCOVERY population (not the confirmation population); application-era recipe (12-article calibration prefix, B300 fits); saved outputs only (level 1); local evaluation on RTX 5070 Ti, which differs numerically from the pod-side n100_exit3 evaluation (e.g. loop-4 any-layer excess 0.461 here vs 0.472 pod-side)',
           'population': {'items': 148, 'eligible_multihop_items': len(elig), 'eligible_slots': 100, 'controls': 'other multihop catalogue names (69 of 70) per slot, historical rule', 'concept_clusters_for_sensitivity': len(set(groups)), 'cluster_source': 'followup_2026-09-07/audit/dependence_clusters.csv concept_cluster'},
           'comparison_family_frozen_before_outputs': ['local−final excess diff, passes 1–3', 'local−raw excess diff, passes 1–3'], 'inference_policy': 'descriptive only: percentile 95% item bootstrap and concept-cluster bootstrap (20000 draws, seeds 2026091104/2026091105); no simultaneous inference is claimed for this post hoc, different-population summary',
           'passes': passes, 'pass4_identity_checks': {'local_and_final_bank_identical_at_pass4': True, 'raw_equals_final_bank_at_identity_virtual_191': same},
           'descriptive_layer_curves_excess': curves, 'interpretation_rules': ['local improves over final but loses to raw: partial recovery, not rescued superiority', 'local beats both: target-dependent early-pass advantage consistent with transport, not proof averaging destroyed information', 'no clear local improvement: rescue unsupported at this precision']}
(HERE / 'DISCOVERY_POPULATION_FIXED_BAND.json').write_text(json.dumps(summary, indent=1))
with open(HERE / 'discovery_population_curves.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['virtual', 'pass', 'physical', *curves]); 
    for v in range(192): w.writerow([v, v // 48 + 1, v % 48 + 1, *[('' if curves[m][v] is None else f'{curves[m][v]:.6f}') for m in curves]])
for row in passes:
    print(f"pass {row['pass']}: raw {row['raw']['excess']:+.4f} final {row['final_target']['excess']:+.4f} local {row['local_target']['excess']:+.4f} | local-final {row['local_minus_final']['excess_diff']:+.4f} {row['local_minus_final']['item_bootstrap_95_descriptive']} cl {row['local_minus_final']['concept_cluster_bootstrap_95_descriptive']} | local-raw {row['local_minus_raw']['excess_diff']:+.4f} {row['local_minus_raw']['item_bootstrap_95_descriptive']} | intended: raw {row['raw']['intended']:.3f} final {row['final_target']['intended']:.3f} local {row['local_target']['intended']:.3f}")
print('matched recipe:', matched, '| identity check raw==final@191:', same)
for b in inv: print(b['bank'], b['merged_file']['status'], [s['status'][:22] for s in b['shards']])
