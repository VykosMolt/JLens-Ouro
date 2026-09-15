#!/usr/bin/env python3
"""Frozen P4 preparation, bounded extraction and CPU analysis. No work on import."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

HERE = Path(__file__).resolve().parent
CONTRACT_SHA = 'b92411f0a4551870cc3a3a7be6d4d5fa168a3eec97b124f6634292eb211c6b8b'
OURO = Path('/home/moloch/ouro_project')
REPO = Path('/home/moloch/ouro_project/jacobian-lens')
SNAPSHOT = OURO / 'artifacts/hf_cache/hub/models--ByteDance--Ouro-2.6B/snapshots/1ed04250da1a9936042725d302e81c8fa2ab5abd'
LAYERS = [25, 26, 29, 31, 35, 39, 42]
PROBE_LAYERS = [39, 31, 29, 42, 26]
RAW_LAYERS = [31, 25, 31, 35, 31]
CS = [0.1, 1., 1., 1., 1.]
LABELS = list(range(2, 19))
INPUTS = {
    'old_cache': (str(OURO / 'artifacts/jlens/probe/n80_v2/gpu_cache.npz'), '4cb03e201dd27b21fd296a9a2b4d98f3254612d1c08c34deecfdd590b081aad6'),
    'provenance': (str(OURO / 'artifacts/jlens/probe/n80_v2/gpu_cache.provenance.json'), '9b7a4733bdf9094440e05a62aa8d8d6fc9c1325a3de7a6426b1836c14467c16e'),
    'cv': (str(OURO / 'artifacts/jlens/probe/cv_all648/arrays.npz'), '1ff6157d3bddda1b3fcf782f2ceb98749bc0c5f5d4c1200d172b4b8d04d0c274'),
    'design': (str(OURO / 'artifacts/jlens/probe/cv_all648/design.json'), '28a80d90145f98cdeeb2c80e800e85526e919d696471e1334f73ae73eb6d26ac'),
    'pairing': (str(REPO / 'research/followup_2026-09-07/probe/audit_pairing.npz'), 'e34cdde8a695a30f6a516aaa73bb6f4c8a2ed004fbd80f8605eff1a4c3a593e7'),
    'selection': (str(REPO / 'research/followup_2026-09-07/probe/saved_rank_audit.json'), '1ccaba197a91706b84c5425c4eb9574ef000634ac0519b1a9589f5df137630f1'),
    'historical_predictions': (str(REPO / 'research/followup_2026-09-07/probe/bounded_refit_predictions.npz'), 'f5a50cbcae0328ef0cda9892a366a5f3d0593409fac53599b84550d58ae495d0'),
    'historical_audit': (str(REPO / 'research/followup_2026-09-07/probe/bounded_refit_results.json'), '8aac73e657009e6847b003156896c2f9b90b5bbed175b4b260fd7e493a10fe81'),
}
SOURCE_PINS = {
    str(OURO / 'src/ouro_jlens/probe.py'): '15914aa97704afbe60ba7f335c3ac16a501d57679476c78d96cbc6809cf020d3',
    str(OURO / 'src/ouro_jlens/probe_cv.py'): '370500ae9749a062651d6e6e4e9756e9d20e19346288174118724b5ebff1870f',
    str(OURO / 'src/ouro_jlens/recurrent.py'): '4477cde085e9e5965c33ebe737ca6b84148a0bd19562c1fa9e6077168e5a96ed',
    str(OURO / 'src/ouro_jlens/evaldata.py'): '2772ddb2cbaa2a7c1fce2a23db14b36fbf9c266aabbd0425cf2c566ed01e6c74',
    str(REPO / 'jlens/hooks.py'): 'c781d6944fd23396d3fc65a04db1f1db807f6f12cd5912cdbd2fb67eb3508081',
    str(REPO / 'jlens/hf.py'): '228cf078e4586a7b7f61a6f5064403b8960de337afd19256efa56f04d53e3222',
    str(REPO / 'research/followup_2026-09-07/probe/audit.py'): '7b49d87389fe7adb63182569ceff27069b33c0faa52d71260864415dbe61c94c',
    str(OURO / 'venv/lib/python3.14/site-packages/sklearn/preprocessing/_data.py'): '1255040a0eb9c99f58775d55cf34f675cf61539f7c0a0dd0d6a36cb1f0da1e15',
    str(OURO / 'venv/lib/python3.14/site-packages/sklearn/linear_model/_logistic.py'): '981f4e11c9a6fc2b02cd9322f2926849143e4a8bf1819e3a4159a9fce615a00c',
}
GPU_PINS = {'python': '3.14.7', 'torch': '2.12.0.dev20260407+cu128', 'transformers': '4.54.1', 'numpy': '2.4.4'}
CPU_PINS = {'numpy': '2.4.4', 'scikit-learn': '1.7.2', 'scipy': '1.18.0', 'threadpoolctl': '3.6.0'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n').encode()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    p = Path(path)
    return {'path': str(p.absolute()), 'sha256': digest(p), 'size': p.stat().st_size}


def check_identity(path, record):
    p = Path(path)
    require(p.is_file(), f'missing input: {p}')
    if 'size' in record:
        require(p.stat().st_size == record['size'], f'size changed: {p}')
    require(digest(p) == record['sha256'], f'SHA-256 changed: {p}')


def write_json(path, value):
    # New output directories and exclusive file creation prevent replacement.
    with Path(path).open('xb') as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


def read_json(path):
    return json.loads(Path(path).read_text())


def fresh_dir(path):
    p = Path(path).absolute()
    p.mkdir(parents=True, exist_ok=False)
    return p


def load_npz(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def runtime_versions(names):
    return {'python': platform.python_version(), **{n: importlib.metadata.version(n) for n in names}}


def rows():
    operands = [(a, b, c) for a in range(1, 10) for b in range(1, 10) for c in range(2, 10)]
    added = sorted((a, b) for a in range(1, 18) for b in range(a, 18) if a + b <= 18 and b >= 10)
    operands += [(x, y, c) for a, b in added for x, y in ((a, b), (b, a)) for c in range(2, 10)]
    return [{'row': i, 'a': a, 'b': b, 'c': c, 'prompt': f'({a} + {b}) * {c} = ',
             'label': a + b, 'target': str((a + b) * c), 'population': 'old' if i < 648 else 'new'}
            for i, (a, b, c) in enumerate(operands)]


def design_arrays():
    import numpy as np
    population = rows()
    pairs = [(a, b) for a in range(1, 10) for b in range(a, 10)]
    assignment = np.empty(45, dtype=np.int64)
    assignment[np.random.default_rng(0).permutation(45)] = np.arange(45) % 5
    pid = np.array([pairs.index(tuple(sorted((r['a'], r['b'])))) for r in population[:648]])
    folds = assignment[pid]
    labels = np.array([r['label'] for r in population])
    eligible = np.array([y in labels[:648][folds != f] for y, f in zip(labels[:648], folds)], dtype=bool)
    return population, pairs, pid, folds, labels, eligible


def fold_training(labels, folds, f, augmented):
    import numpy as np
    old = np.flatnonzero(folds != f)
    classes, old_counts = np.unique(labels[old], return_counts=True)
    added = np.flatnonzero(np.isin(labels[648:], classes)) + 648
    selected = np.concatenate([old, added]) if augmented else old
    weights = np.empty(len(selected), dtype=np.float64)
    counts = []
    for label, n in zip(classes, old_counts):
        mask = labels[selected] == label
        N = int(mask.sum())
        weights[mask] = int(n) / N
        counts.append({'class': int(label), 'original_count': int(n), 'included_count': N,
                       'weight': int(n) / N, 'actual_weight_sum': float(weights[mask].sum())})
    require(np.isfinite(weights).all() and (weights > 0).all(), 'invalid training weights')
    return selected, weights, counts


def validate_population(cv, pairing, selection, design, historical):
    import numpy as np
    population, pairs, pid, folds, labels, eligible = design_arrays()
    old_pairs = set(pairs)
    new_pairs = {tuple(sorted((r['a'], r['b']))) for r in population[648:]}
    require(len(population) == len({r['prompt'] for r in population}) == 1224, 'prompt enumeration')
    require(len(new_pairs) == 36 and not new_pairs & old_pairs, 'pair leakage')
    require([sum(a + b == s for a, b in new_pairs) for s in range(11, 19)] == list(range(1, 9)), 'new class counts')
    for key, expected in [('folds', folds), ('labels', labels[:648])]:
        require(np.array_equal(cv[key], expected), f'CV {key} does not match independent reconstruction')
    require(design['seed'] == 0 and design['design']['fold_assignment_unit'] == 'unordered_operand_pair', 'split ancestry')
    require(design['inputs']['gpu_cache']['sha256'] == INPUTS['old_cache'][1], 'CV cache ancestry')
    for name, expected in [('pid', pid), ('eligible', eligible)]:
        require(np.array_equal(pairing[name], expected), f'saved {name} changed')
    require(pairing['eligible'].dtype == np.bool_, 'eligible must be Boolean')
    require(int(eligible.sum()) == 576 and int((eligible & (labels[:648] >= 11)).sum()) == 264, 'eligible population')
    excluded = [list(p) for i, p in enumerate(pairs) if not eligible[pid == i].any()]
    require(excluded == [[1, 1], [1, 2], [1, 3], [2, 2], [8, 9], [9, 9]], 'excluded pair set')
    clusters, counts, W = (pairing[n] for n in ('clusters', 'counts', 'weights'))
    require(clusters.shape == counts.shape == (39,) and W.shape == (20000, 39), 'saved resample shapes')
    require(all(np.issubdtype(a.dtype, np.integer) for a in [pid, clusters, counts, W]), 'resample integer types')
    require(np.array_equal(np.sort(clusters), np.unique(pid[eligible])), 'saved cluster set/order identity')
    expected_counts = np.array([(pid == g).sum() for g in clusters])
    require(np.array_equal(counts, expected_counts) and set(counts) <= {8, 16} and counts.sum() == 576, 'saved counts')
    require((W >= 0).all() and (W.sum(axis=1) == 39).all(), 'resample multiplicities')
    cluster_folds = np.array([folds[pid == g][0] for g in clusters])
    require(all((W[:, cluster_folds == f].sum(axis=1) > 0).all() for f in range(5)), 'draw lost an outer fold')
    checks = []
    absent_expected = [[4], [], [3, 17, 18], [], [2]]
    for f in range(5):
        old, one, old_counts = fold_training(labels, folds, f, False)
        aug, w, class_counts = fold_training(labels, folds, f, True)
        classes = set(labels[old])
        require(sorted(set(LABELS) - classes) == absent_expected[f], 'original class support')
        require(len(old) == [520, 512, 528, 512, 520][f] and len(aug) == [1096, 1088, 864, 1088, 1096][f], 'training row counts')
        require(int(((folds == f) & eligible).sum()) == [104, 136, 80, 136, 120][f], 'test count')
        require(set(aug) & set(np.flatnonzero(folds == f)) == set(), 'train/test row leakage')
        train_pairs = {tuple(sorted((population[i]['a'], population[i]['b']))) for i in aug}
        test_pairs = {pairs[g] for g in pid[folds == f]}
        require(not train_pairs & test_pairs and len(train_pairs) == [72, 72, 57, 72, 72][f], 'train/test unordered-pair leakage')
        require(np.allclose(w.sum(), len(old), rtol=0, atol=1e-10), 'total weight drift')
        require(all(abs(c['actual_weight_sum'] - c['original_count']) < 1e-10 for c in class_counts), 'class total weight drift')
        require(cv['chosen_C'][f, PROBE_LAYERS[f] - 1] == CS[f], 'frozen selected C')
        for name, layer in [('probe', PROBE_LAYERS[f]), ('logit_lens', RAW_LAYERS[f])]:
            require(selection['readouts'][name]['selected_physical_layer_one_based_per_fold'][f][0] == layer, 'frozen layer mismatch')
        require(int(np.argmax(cv['selection_accuracy'][f, :48])) + 1 == PROBE_LAYERS[f], 'original inner-selection ancestry')
        checks.append({'fold': f, 'old_rows': len(old), 'added_rows': len(aug) - len(old), 'augmented_rows': len(aug),
                       'old_pairs': 36, 'augmented_pairs': len(train_pairs), 'absent_classes': absent_expected[f],
                       'eligible_test_rows': int(((folds == f) & eligible).sum()), 'classes': class_counts,
                       'actual_total_weight': float(w.sum()), 'C': CS[f], 'probe_layer': PROBE_LAYERS[f], 'raw_layer': RAW_LAYERS[f]})
    for key, expected in [('y', labels[:648]), ('folds', folds), ('pid', pid), ('eligible', eligible)]:
        require(np.array_equal(historical[key], expected), f'historical CPU {key} mismatch')
    for f in range(5):
        ix = folds == f
        require(np.array_equal(pairing['probe'][ix, 0], (cv['probe_rank'][ix, PROBE_LAYERS[f] - 1] == 0).astype(float)), 'archived original selected probe scores')
        require(np.array_equal(pairing['logit_lens'][ix, 0], (cv['ll_cand'][ix, RAW_LAYERS[f] - 1] == 0).astype(float)), 'archived raw selected scores')
    return checks


def canonical_source_path(relative):
    if relative.startswith('src/'):
        return OURO / relative
    require(relative.startswith('dependency/'), 'unknown provenance source path')
    return REPO / relative.removeprefix('dependency/')


def prepare(args):
    # Preparation hashes archives but never reads historical H or fits a classifier.
    check_identity(HERE / 'CONTRACT.md', {'sha256': CONTRACT_SHA})
    bound_inputs = {}
    for name, (path, expected) in INPUTS.items():
        check_identity(path, {'sha256': expected})
        bound_inputs[name] = identity(path)
    for path, expected in SOURCE_PINS.items():
        check_identity(path, {'sha256': expected})
    provenance = read_json(INPUTS['provenance'][0])
    require(len(provenance['model_files']) == 13, 'full 13-file model inventory')
    source_records = []
    unused_source_differences = []
    for item in provenance['source_files']:
        p = canonical_source_path(item['path'])
        # evaluate.py is archived provenance, not imported or consumed by P4.
        # The current project may have changed unrelated historical entrypoints.
        if item['path'].startswith('dependency/') or str(p) in SOURCE_PINS:
            check_identity(p, item)
        elif p.is_file() and digest(p) != item['sha256']:
            unused_source_differences.append({'canonical': str(p), 'archived': item, 'current': identity(p),
                                              'reason': 'Historical provenance only; not imported or consumed by P4.'})
        source_records.append({'canonical': str(p), **item})
    # The empty Ouro package initializer also participates in module loading.
    source_records.append({'canonical': str(OURO / 'src/ouro_jlens/__init__.py'),
                           **identity(OURO / 'src/ouro_jlens/__init__.py')})
    checks = validate_population(load_npz(INPUTS['cv'][0]), load_npz(INPUTS['pairing'][0]),
                                 read_json(INPUTS['selection'][0]), read_json(INPUTS['design'][0]),
                                 load_npz(INPUTS['historical_predictions'][0]))
    out = fresh_dir(args.output)
    population = rows()
    write_json(out / 'prompts.json', population)
    model_manifest = {'canonical_snapshot': str(SNAPSHOT), 'provenance': bound_inputs['provenance'],
                      'files': provenance['model_files']}
    write_json(out / 'model_manifest.json', model_manifest)
    shutil.copyfile(INPUTS['provenance'][0], out / 'historical_provenance.json')
    plan = {'schema': 'p4-plan-v1', 'created_utc': utc(), 'contract_sha256': CONTRACT_SHA,
            'implementation': identity(__file__), 'inputs': bound_inputs,
            'source_pins': SOURCE_PINS, 'provenance_sources': source_records,
            'unused_historical_source_differences': unused_source_differences,
            'historical_provenance': provenance, 'model_manifest': identity(out / 'model_manifest.json'),
            'prompts': identity(out / 'prompts.json'), 'prompt_canonical_sha256': hashlib.sha256(canonical(population)).hexdigest(),
            'layers': LAYERS, 'probe_layers': PROBE_LAYERS, 'raw_layers': RAW_LAYERS, 'C': CS,
            'folds': checks, 'gpu_runtime_pins': GPU_PINS, 'cpu_runtime_pins': CPU_PINS,
            'feature_shape': [1224, 7, 2048], 'raw_shape': [648, 7, 17], 'gpu_limit_seconds': 900,
            'status': 'PREPARED_NO_SCIENTIFIC_EXECUTION'}
    # Recheck consumed inputs before the plan is sealed.
    for entry in bound_inputs.values():
        check_identity(entry['path'], entry)
    write_json(out / 'plan.json', plan)
    print(json.dumps({'plan': identity(out / 'plan.json'), 'prompts': 1224, 'fold_checks': len(checks)}))


def load_plan(path):
    p = Path(path)
    plan = read_json(p)
    require(plan['schema'] == 'p4-plan-v1' and plan['contract_sha256'] == CONTRACT_SHA, 'plan contract/schema')
    check_identity(HERE / 'CONTRACT.md', {'sha256': CONTRACT_SHA})
    check_identity(__file__, plan['implementation'])
    require(plan['layers'] == LAYERS and plan['probe_layers'] == PROBE_LAYERS and plan['raw_layers'] == RAW_LAYERS and plan['C'] == CS, 'plan fixed locations')
    require(plan['feature_shape'] == [1224, 7, 2048] and plan['raw_shape'] == [648, 7, 17] and plan['gpu_limit_seconds'] == 900, 'plan fixed geometry/time')
    require(plan['gpu_runtime_pins'] == GPU_PINS and plan['cpu_runtime_pins'] == CPU_PINS, 'plan runtime pins')
    for key, (canonical_path, sha) in INPUTS.items():
        require(plan['inputs'][key]['path'] == canonical_path and plan['inputs'][key]['sha256'] == sha, 'canonical historical identity')
    require(plan['source_pins'] == SOURCE_PINS, 'plan source pins')
    # Metadata is staged next to the plan; canonical old paths are never opened remotely.
    for filename, key in [('prompts.json', 'prompts'), ('model_manifest.json', 'model_manifest')]:
        check_identity(p.parent / filename, plan[key])
    require(read_json(p.parent / 'prompts.json') == rows(), 'prompt list/order changed')
    require(plan['prompt_canonical_sha256'] == hashlib.sha256(canonical(rows())).hexdigest(), 'prompt canonical digest')
    model_manifest = read_json(p.parent / 'model_manifest.json')
    check_identity(p.parent / 'historical_provenance.json', plan['inputs']['provenance'])
    require(read_json(p.parent / 'historical_provenance.json') == plan['historical_provenance'], 'historical provenance metadata changed')
    expected_sources = [{'canonical': str(canonical_source_path(s['path'])), **s}
                        for s in plan['historical_provenance']['source_files']]
    require(plan['provenance_sources'][:-1] == expected_sources, 'full source provenance mapping changed')
    require(plan['provenance_sources'][-1]['canonical'] == str(OURO / 'src/ouro_jlens/__init__.py'), 'package initializer binding missing')
    require(model_manifest['canonical_snapshot'] == str(SNAPSHOT), 'canonical snapshot')
    require(model_manifest['files'] == plan['historical_provenance']['model_files'] and len(model_manifest['files']) == 13, 'complete model inventory ancestry')
    require(model_manifest['provenance'] == plan['inputs']['provenance'], 'model provenance binding')
    return plan


class Deadline:
    def __init__(self, receipt):
        self.receipt_path = Path(receipt)
        self.receipt_identity = identity(receipt)
        self.receipt = read_json(receipt)
        self.start = float(self.receipt['start_monotonic'])
        self.end = float(self.receipt['deadline_monotonic'])
        now = time.monotonic()
        require(self.receipt['limit_seconds'] == 900 and 0 < self.end - self.start <= 900, 'deadline limit must be <=900 seconds')
        require(self.start <= now < self.end, 'deadline must have started on this host before initialization')
        self.events = []
        self.previous_handler = signal.signal(signal.SIGALRM, self._expired)
        signal.setitimer(signal.ITIMER_REAL, self.end - now)
        self.check('process_entered')

    def _expired(self, *_):
        raise TimeoutError('P4 900-second total GPU phase expired')

    def check(self, phase):
        now = time.monotonic()
        if now >= self.end:
            self._expired()
        self.events.append({'phase': phase, 'utc': utc(), 'monotonic': now, 'elapsed_seconds': now - self.start})

    def close(self):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.previous_handler)


def mapped_sources(plan, ouro_src, jlens_root):
    records = []
    for source in plan['provenance_sources']:
        c = source['canonical']
        if c in SOURCE_PINS:
            require(source['sha256'] == SOURCE_PINS[c], 'staged source differs from contract pin')
        if c.startswith(str(OURO / 'src') + '/'):
            # Only these Ouro modules are imported; no legacy probe/CV dependency.
            rel = Path(c).relative_to(OURO / 'src')
            if rel.as_posix() not in ['ouro_jlens/__init__.py', 'ouro_jlens/recurrent.py', 'ouro_jlens/evaldata.py']:
                continue
            path = Path(ouro_src) / rel
        else:
            path = Path(jlens_root) / Path(c).relative_to(REPO)
        check_identity(path, source)
        records.append({'canonical': c, **identity(path)})
    return records


def verify_snapshot(snapshot, manifest):
    root = Path(snapshot)
    require(root.is_dir() and not root.is_symlink(), 'snapshot root missing or symlink')
    files = []
    for p in sorted(root.rglob('*')):
        if p.is_symlink() and not p.is_file():
            raise ValueError(f'snapshot directory/dangling symlink: {p}')
        if p.is_file():
            files.append(p)
    expected = {r['path'].removeprefix('model_snapshot/'): r for r in manifest['files']}
    require({p.relative_to(root).as_posix() for p in files} == set(expected), 'snapshot file inventory differs')
    records = []
    for p in files:
        rel = p.relative_to(root).as_posix()
        require('..' not in Path(rel).parts, 'snapshot path traversal')
        current = p.parent
        while current != root:
            require(not current.is_symlink(), 'snapshot linked directory')
            current = current.parent
        check_identity(p, expected[rel])
        records.append({'relative': rel, **identity(p)})
    return records


def validate_encoded(m, population):
    encoded = []
    for row in population:
        text = row['prompt']
        untruncated = m.tokenizer(text, truncation=False).input_ids
        expected = [m.tokenizer.bos_token_id, *untruncated]
        require(len(untruncated) <= 511, 'prompt would be truncated')
        ids = m.encode(text, max_length=512)
        require(ids.ndim == 2 and tuple(ids.shape) == (1, len(expected)), 'encoded batch/length mismatch')
        actual = ids[0].tolist()
        require(actual == expected, 'Ouro explicit BOS/tokenization differs')
        encoded.append({'row': row['row'], 'input_ids': actual, 'length': len(actual), 'readout_index': len(actual) - 1})
    return encoded


def tokenizer_metadata(value):
    """Preserve AddedToken fields without opaque repr strings or JSON failures."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): tokenizer_metadata(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [tokenizer_metadata(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if type(value).__name__ == 'AddedToken':
        return {'type': 'AddedToken', **{k: getattr(value, k) for k in
                ['content', 'single_word', 'lstrip', 'rstrip', 'normalized', 'special']}}
    raise TypeError(f'unsupported tokenizer metadata type: {type(value).__name__}')


def extract_features(m, population, encoded, recorder, clock, progress=None):
    import numpy as np
    import torch
    require((m.n_ut, m.n_physical, m.n_layers, m.d_model) == (4, 48, 192, 2048), 'Ouro extraction geometry')
    H = np.empty((len(population), 7, 2048), dtype=np.float16)
    at = [v - 1 for v in LAYERS]
    with torch.no_grad():
        for i, row in enumerate(population):
            clock.check(f'feature_{i}_start')
            ids = m.encode(row['prompt'], max_length=512)
            require(ids[0].tolist() == encoded[i]['input_ids'], 'encoding changed after freeze')
            with recorder(m.layers, at=at) as rec:
                m.forward(ids)
            require(set(rec.activations) == set(at), 'missing residual tap')
            require(all(tuple(rec.activations[v].shape) == (1, len(encoded[i]['input_ids']), 2048) for v in at), 'post-block residual shape')
            # Conversion is at the same location and precision as the source cache.
            H[i] = torch.stack([rec.activations[v][0, -1].half().cpu() for v in at]).numpy()
            require(np.isfinite(H[i]).all(), f'nonfinite feature row {i}')
            if progress and (i + 1) % 32 == 0:
                progress(i + 1)
    return H


def raw_scores(m, H, aliases, clock):
    import numpy as np
    import torch
    result = np.empty((648, 7, 17), dtype=np.float32)
    with torch.no_grad():
        for i in range(648):
            clock.check(f'raw_{i}_start')
            # Fixed batch of seven, starting from the saved FP16 feature matrix.
            logits = m.unembed(torch.from_numpy(H[i].copy()).to(m.input_device).float()).float()
            require(logits.ndim == 2 and logits.shape[0] == 7, 'native head batch must be seven')
            scores = torch.stack([logits[:, a['token_ids']].max(-1).values for a in aliases], dim=1)
            result[i] = scores.cpu().numpy()
            require(np.isfinite(result[i]).all(), f'nonfinite raw row {i}')
    return result


def strict_ranks(scores, labels):
    import numpy as np
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    require(scores.shape[-1] == 17 and scores.shape[0] == len(labels) and np.isfinite(scores).all(), 'invalid candidate scores')
    require(np.isin(labels, LABELS).all(), 'unknown nominal class')
    true = np.take_along_axis(scores, (labels - 2).reshape((len(labels),) + (1,) * (scores.ndim - 1)), axis=-1)
    return (scores > true).sum(axis=-1).astype(np.int32)


def seal(out, kind, bindings, extra):
    outputs = {p.name: identity(p) for p in sorted(out.iterdir()) if p.is_file()}
    require('COMPLETE.json' not in outputs and 'manifest.json' not in outputs and 'FAILURE.json' not in outputs, 'cannot reseal an output')
    manifest = {'schema': 'p4-seal-v1', 'kind': kind, 'status': 'COMPLETE', 'bindings': bindings,
                'outputs': outputs, **extra}
    write_json(out / 'manifest.json', manifest)
    write_json(out / 'COMPLETE.json', {'status': 'COMPLETE', 'manifest': identity(out / 'manifest.json'), 'sealed_utc': utc()})


def validate_seal(out, kind, plan_path):
    out = Path(out)
    require((out / 'COMPLETE.json').is_file() and not (out / 'FAILURE.json').exists(), 'incomplete output cannot be consumed')
    complete = read_json(out / 'COMPLETE.json')
    require(complete['status'] == 'COMPLETE', 'invalid completion status')
    check_identity(out / 'manifest.json', complete['manifest'])
    manifest = read_json(out / 'manifest.json')
    require(manifest['schema'] == 'p4-seal-v1' and manifest['status'] == 'COMPLETE' and manifest['kind'] == kind, 'seal kind/status')
    require(manifest['bindings']['contract_sha256'] == CONTRACT_SHA, 'cache contract binding')
    check_identity(plan_path, manifest['bindings']['plan'])
    check_identity(__file__, manifest['bindings']['implementation'])
    require(set(p.name for p in out.iterdir()) == set(manifest['outputs']) | {'manifest.json', 'COMPLETE.json'}, 'extra/missing output entries')
    for name, record in manifest['outputs'].items():
        require(Path(name).name == name and not (out / name).is_symlink(), 'invalid sealed output path')
        check_identity(out / name, record)
    if kind == 'gpu-cache':
        require(0 < manifest['gpu_elapsed_seconds'] < 900, 'GPU deadline not satisfied')
    return manifest


def failure(out, error, clock=None):
    # A process killed externally may leave only RUNNING.json; that is incomplete.
    (out / 'COMPLETE.json').unlink(missing_ok=True)
    if not (out / 'FAILURE.json').exists():
        write_json(out / 'FAILURE.json', {'status': 'INCOMPLETE', 'error_type': type(error).__name__,
                   'error': str(error), 'utc': utc(), 'events': clock.events if clock else []})


def validate_supervisor(receipt_path, cache, plan_path, deadline_path=None):
    """A child COMPLETE alone cannot establish success before the external ceiling."""
    receipt = read_json(receipt_path)
    require(receipt['schema'] == 'p4-supervisor-v1' and receipt['status'] == 'COMPLETE', 'supervisor did not complete')
    require(receipt['child_exit_code'] == 0 and receipt['timed_out'] is False, 'child failed or timed out')
    elapsed = receipt['end_monotonic'] - receipt['start_monotonic']
    require(0 < elapsed <= 900 and abs(elapsed - receipt['elapsed_seconds']) < 1e-9, 'external whole-phase deadline exceeded')
    require(receipt['end_monotonic'] <= receipt['deadline_monotonic'] and receipt['deadline_monotonic'] - receipt['start_monotonic'] <= 900, 'external deadline exceeded')
    check_identity(plan_path, receipt['plan'])
    check_identity(__file__, receipt['implementation'])
    check_identity(Path(cache) / 'COMPLETE.json', receipt['cache_complete'])
    check_identity(Path(cache) / 'manifest.json', receipt['cache_manifest'])
    manifest = validate_seal(cache, 'gpu-cache', plan_path)
    require(receipt['deadline'] == manifest['bindings']['deadline'], 'supervisor/child deadline receipt differs')
    timing_document = read_json(Path(cache) / 'timing.json')
    timing = timing_document['receipt']
    require(timing['start_monotonic'] == receipt['start_monotonic'] and timing['deadline_monotonic'] == receipt['deadline_monotonic'], 'supervisor/child timer differs')
    require(hashlib.sha256(canonical(timing)).hexdigest() == receipt['deadline']['sha256'], 'embedded deadline bytes differ')
    require(manifest['gpu_elapsed_seconds'] <= elapsed, 'supervisor ended before child elapsed checkpoint')
    require(timing_document['events'] == manifest['events'] and bool(manifest['events']), 'child timing event binding differs')
    previous = receipt['start_monotonic']
    for event in manifest['events']:
        require(previous <= event['monotonic'] <= receipt['end_monotonic'], 'supervisor end does not cover ordered child events')
        require(abs(event['elapsed_seconds'] - (event['monotonic'] - receipt['start_monotonic'])) < 1e-9, 'child event elapsed mismatch')
        previous = event['monotonic']
    if deadline_path is not None:
        check_identity(deadline_path, receipt['deadline'])
    return receipt


def supervise_extract(args):
    """Standard-library process-group deadline, starting before child initialization."""
    require(not Path(args.output).exists(), 'GPU output directory already exists')
    require(not Path(args.deadline).exists() and not Path(args.receipt).exists(), 'deadline/receipt must be new paths')
    require(Path(args.receipt).absolute().parent != Path(args.output).absolute(), 'supervisor receipt must be outside cache')
    plan_id = identity(args.plan)
    started_utc, start = utc(), time.monotonic()
    deadline = {'limit_seconds': 900, 'start_monotonic': start, 'deadline_monotonic': start + 900,
                'started_utc': started_utc, 'supervisor_pid': os.getpid()}
    write_json(args.deadline, deadline)
    command = [args.python, '-B', str(Path(__file__).absolute()), 'extract']
    for name in ['snapshot', 'ouro_src', 'jlens_root', 'plan', 'output', 'deadline']:
        command += ['--' + name.replace('_', '-'), str(getattr(args, name))]
    child = None
    timed_out = False
    code = None
    error = None
    def terminated(*_):
        raise InterruptedError('outer supervisor termination requested')
    previous_handlers = {sig: signal.signal(sig, terminated) for sig in [signal.SIGTERM, signal.SIGHUP]}
    try:
        child = subprocess.Popen(command, start_new_session=True)
        try:
            code = child.wait(timeout=max(0, start + 900 - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGKILL)
            code = child.wait()
        require(not timed_out and code == 0, f'extraction child exit={code}, timeout={timed_out}')
        validate_seal(args.output, 'gpu-cache', args.plan)
        # These hashes and all child cleanup are inside the measured phase.
        complete_id = identity(Path(args.output) / 'COMPLETE.json')
        manifest_id = identity(Path(args.output) / 'manifest.json')
        check_identity(args.plan, plan_id)
        end = time.monotonic()
        require(end <= start + 900, 'deadline expired during supervisor seal verification')
    except BaseException as exc:
        error = exc
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            code = child.wait()
        end = time.monotonic()
    finally:
        for sig, previous in previous_handlers.items():
            signal.signal(sig, previous)
    receipt = {'schema': 'p4-supervisor-v1', 'status': 'COMPLETE' if error is None else 'INCOMPLETE',
               'started_utc': started_utc, 'finished_utc': utc(), 'start_monotonic': start,
               'end_monotonic': end, 'deadline_monotonic': start + 900, 'elapsed_seconds': end - start,
               'child_exit_code': code, 'timed_out': timed_out, 'command': command,
               'plan': plan_id, 'deadline': identity(args.deadline), 'implementation': identity(__file__)}
    if error is None:
        receipt.update(cache_complete=complete_id, cache_manifest=manifest_id)
    else:
        receipt['error'] = f'{type(error).__name__}: {error}'
    write_json(args.receipt, receipt)
    if error is not None:
        raise error
    print(json.dumps({'status': 'COMPLETE', 'receipt': identity(args.receipt), 'total_gpu_phase_seconds': end - start}))


def extract(args):
    # The externally created receipt starts the clock before Python/GPU imports.
    clock = Deadline(args.deadline)
    out = None
    m = None
    torch = None
    try:
        out = fresh_dir(args.output)
        write_json(out / 'RUNNING.json', {'status': 'RUNNING_NOT_COMPLETE', 'utc': utc(), 'deadline': clock.receipt})
        plan = load_plan(args.plan)
        bindings = {'contract_sha256': CONTRACT_SHA, 'plan': identity(args.plan), 'implementation': identity(__file__),
                    'deadline': clock.receipt_identity, 'historical_inputs_metadata_only': plan['inputs']}
        sources = mapped_sources(plan, args.ouro_src, args.jlens_root)
        manifest_path = Path(args.plan).parent / 'model_manifest.json'
        model_records = verify_snapshot(args.snapshot, read_json(manifest_path))
        clock.check('identities_checked_before_import')
        # Existing packages only; never import legacy probe/CV/audit on GPU.
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        sys.path[:0] = [str(Path(args.ouro_src).absolute()), str(Path(args.jlens_root).absolute())]
        import numpy as np
        import torch as torch_module
        torch = torch_module
        import ouro_jlens.recurrent as recurrent
        import ouro_jlens.evaldata as evaldata
        import jlens.hooks as hooks
        import jlens.hf as hf
        versions = runtime_versions(['torch', 'transformers', 'numpy', 'safetensors', 'huggingface-hub'])
        require(all(versions[k] == v for k, v in GPU_PINS.items()), 'GPU runtime differs from accepted pins')
        for module, relative, base in [(recurrent, 'ouro_jlens/recurrent.py', args.ouro_src), (evaldata, 'ouro_jlens/evaldata.py', args.ouro_src),
                                       (hooks, 'jlens/hooks.py', args.jlens_root), (hf, 'jlens/hf.py', args.jlens_root)]:
            require(Path(module.__file__).resolve() == (Path(base) / relative).resolve(), 'imported source path shadowed')
        require(not any(n == 'sklearn' or n.startswith('sklearn.') for n in sys.modules), 'GPU imported sklearn')
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'expected one available GPU')
        clock.check('model_load_start')
        m = recurrent.load_ouro(path=args.snapshot, device='cuda', dtype=torch.bfloat16)
        torch.cuda.synchronize()
        clock.check('model_load_end')
        require((m.n_ut, m.n_physical, m.n_layers, m.d_model) == (4, 48, 192, 2048), 'loaded model geometry')
        require(not m.hf_model.training and all(not p.requires_grad for p in m.hf_model.parameters()), 'model must be eval/frozen')
        require(all(p.dtype == torch.bfloat16 for p in m.hf_model.parameters() if p.is_floating_point()), 'native BF16 parameters required')
        loaded_code = []
        for obj in [type(m.hf_model), type(m.hf_model.config), type(m.tokenizer), type(m), type(m)._text_module if hasattr(type(m), '_text_module') else type(m.hf_model.model)]:
            path = inspect.getsourcefile(obj)
            require(path is not None, 'loaded implementation source unavailable')
            loaded_code.append(identity(path))
        for filename in ['modeling_ouro.py', 'configuration_ouro.py']:
            copies = [r for r in loaded_code if Path(r['path']).name == filename]
            require(copies and all(r['sha256'] == digest(Path(args.snapshot) / filename) for r in copies), 'loaded remote-code identity differs')
        population = read_json(Path(args.plan).parent / 'prompts.json')
        encoded = validate_encoded(m, population)
        aliases = [{'class': s, 'surface_forms': evaldata.surface_forms(str(s)),
                    'token_ids': evaldata.single_token_ids(m.tokenizer, evaldata.surface_forms(str(s)))} for s in LABELS]
        require(all(a['token_ids'] for a in aliases), 'class has no single-token alias')
        write_json(out / 'encoding.json', encoded)
        write_json(out / 'aliases.json', aliases)
        write_json(out / 'prompts.json', population)
        props = torch.cuda.get_device_properties(0)
        runtime = {'versions': versions, 'interpreter': sys.executable, 'sources': sources, 'loaded_code': loaded_code,
                   'model_files': model_records, 'tokenizer_class': str(type(m.tokenizer)),
                   'tokenizer_init_kwargs': tokenizer_metadata(m.tokenizer.init_kwargs), 'gpu_name': props.name, 'gpu_uuid': str(getattr(props, 'uuid', '')),
                   'gpu_capability': list(torch.cuda.get_device_capability()), 'gpu_memory_bytes': props.total_memory,
                   'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip(),
                   'torch_cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                   'environment': {key: os.environ.get(key) for key in [
                       'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'CUBLAS_WORKSPACE_CONFIG',
                       'PYTORCH_ALLOC_CONF', 'HF_HOME', 'HF_HUB_DISABLE_TELEMETRY', 'TOKENIZERS_PARALLELISM',
                       'PYTHONDONTWRITEBYTECODE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']},
                   'attention_implementation': m.hf_model.config._attn_implementation,
                   'precision': {'parameters': 'bfloat16', 'native_batch': 1, 'readout_batch': 7, 'cache': 'float16',
                                 'matmul_precision': torch.get_float32_matmul_precision(), 'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                                 'tf32_cudnn': torch.backends.cudnn.allow_tf32, 'autocast': torch.is_autocast_enabled(),
                                 'flash_sdp': torch.backends.cuda.flash_sdp_enabled(), 'mem_efficient_sdp': torch.backends.cuda.mem_efficient_sdp_enabled(),
                                 'math_sdp': torch.backends.cuda.math_sdp_enabled()},
                   'geometry': [4, 48, 2048], 'use_cache': False, 'readout': 'last complete encoded literal prompt token'}
        write_json(out / 'runtime.json', runtime)
        clock.check('metadata_frozen')
        H = extract_features(m, population, encoded, hooks.ActivationRecorder, clock,
                             lambda n: print(json.dumps({'phase': 'features', 'rows': n, 'elapsed': time.monotonic() - clock.start}), flush=True))
        clock.check('features_complete')
        # Serialize first and score the reloaded bytes, making common cache ancestry explicit.
        np.save(out / 'features.npy', H, allow_pickle=False)
        del H
        H = np.load(out / 'features.npy', allow_pickle=False, mmap_mode='r')
        raw = raw_scores(m, H, aliases, clock)
        np.save(out / 'raw_scores.npy', raw, allow_pickle=False)
        np.save(out / 'raw_ranks.npy', strict_ranks(raw, np.array([r['label'] for r in population[:648]])), allow_pickle=False)
        clock.check('scoring_serialized')
        # Release all model/GPU references before the final identities/hashes and seal.
        del m
        m = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        clock.check('gpu_cleanup_complete')
        for entry in sources + loaded_code:
            check_identity(entry['path'], entry)
        for entry in model_records:
            check_identity(entry['path'], entry)
        load_plan(args.plan)
        check_identity(args.plan, bindings['plan'])
        check_identity(args.deadline, clock.receipt_identity)
        require(H.shape == (1224, 7, 2048) and H.dtype == np.float16 and np.isfinite(H).all(), 'finite complete feature cache')
        require(raw.shape == (648, 7, 17) and raw.dtype == np.float32 and np.isfinite(raw).all(), 'finite complete raw cache')
        clock.check('inputs_rechecked_before_seal')
        write_json(out / 'timing.json', {'receipt': clock.receipt, 'events': clock.events})
        # seal hashes every output, including serialization, within the external deadline.
        seal(out, 'gpu-cache', bindings, {'gpu_elapsed_seconds': time.monotonic() - clock.start,
                                        'events': clock.events, 'runtime': runtime, 'rows': 1224})
        clock.check('seal_written')
        # This receipt is outside the sealed directory so it cannot perturb its inventory.
        print(json.dumps({'status': 'COMPLETE', 'output': str(out), 'elapsed_including_seal_seconds': time.monotonic() - clock.start}), flush=True)
    except BaseException as error:
        if out is not None:
            failure(out, error, clock)
        raise
    finally:
        if m is not None:
            del m
            gc.collect()
            if torch is not None:
                torch.cuda.empty_cache()
        clock.close()


def expand_probabilities(probabilities, classes):
    import numpy as np
    classes = np.asarray(classes)
    require(np.isin(classes, LABELS).all() and len(np.unique(classes)) == len(classes), 'classifier classes')
    result = np.zeros((len(probabilities), 17), dtype=np.float64)
    result[:, classes.astype(int) - 2] = probabilities
    require(np.isfinite(result).all() and (result >= 0).all() and np.allclose(result.sum(1), 1), 'invalid probabilities')
    return result


def fit_fixed(Xtrain, ytrain, Xtest, weights, C):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits, threadpool_info
    require(Xtrain.dtype == Xtest.dtype == np.float32, 'fit inputs must be promoted to float32')
    require(np.isfinite(Xtrain).all() and np.isfinite(Xtest).all(), 'nonfinite fit inputs')
    settings = dict(C=C, solver='lbfgs', penalty='l2', tol=1e-4, max_iter=2000,
                    fit_intercept=True, class_weight=None, warm_start=False)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter('always')
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        scaler.fit(Xtrain, sample_weight=weights)
        scaled_train = scaler.transform(Xtrain)
        scaled_test = scaler.transform(Xtest)
        clf = LogisticRegression(**settings)
        clf.fit(scaled_train, ytrain, sample_weight=weights)
        proba = expand_probabilities(clf.predict_proba(scaled_test), clf.classes_)
        prediction = clf.predict(scaled_test)
        pools = threadpool_info()
    messages = [{'category': type(w.message).__name__, 'message': str(w.message)} for w in caught]
    require(not any(issubclass(w.category, ConvergenceWarning) for w in caught), f'nonconvergence: {messages}')
    require(int(clf.n_iter_.max()) < 2000, 'iteration limit reached; experiment incomplete')
    require(all(p['num_threads'] == 1 for p in pools), 'native CPU thread limit not held')
    retained = {'coef': clf.coef_, 'intercept': clf.intercept_, 'classes': clf.classes_, 'scaler_mean': scaler.mean_,
                'scaler_var': scaler.var_, 'scaler_scale': scaler.scale_, 'scaler_n_samples_seen': np.asarray(scaler.n_samples_seen_),
                'n_iter': clf.n_iter_, 'probabilities': proba, 'prediction': prediction}
    require(all(np.isfinite(a).all() for a in retained.values()), 'nonfinite fitted output')
    diagnostics = {'classifier_settings': clf.get_params(deep=False), 'scaler_settings': scaler.get_params(deep=False),
                   'iterations': clf.n_iter_.tolist(), 'warnings': messages, 'threadpools_during_fit': pools,
                   'training_rows': len(ytrain), 'training_weight_sum_float64': float(weights.sum()),
                   'mean_loss_l2_strength_float64': 1. / (C * float(weights.sum())),
                   'solver_sample_weight_dtype': 'float64', 'scaled_training_dtype': str(scaled_train.dtype),
                   'scaler_effective_float32_weight_sum_float64': float(weights.astype(np.float32).sum(dtype=np.float64))}
    return retained, diagnostics


def paired_summary(values, pairing):
    """Prompt-weighted point and fixed saved pair-multiplicity bootstrap."""
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    eligible, pid, clusters, W, counts = (pairing[k] for k in ['eligible', 'pid', 'clusters', 'weights', 'counts'])
    require(values.shape == eligible.shape and np.isfinite(values).all(), 'invalid summary values')
    require(all(eligible[pid == g].all() for g in clusters), 'cluster includes excluded rows')
    cluster_sums = np.array([values[pid == g].sum() for g in clusters], dtype=np.float64)
    denominator = W @ counts
    require((denominator > 0).all(), 'invalid bootstrap denominator')
    draws = (W @ cluster_sums) / denominator
    return {'point': float(values[eligible].mean()),
            'ci95': np.percentile(draws, [2.5, 97.5], method='linear').tolist()}, draws, cluster_sums, denominator


def feature_diagnostics(old, fresh):
    import numpy as np
    result = []
    for layer in PROBE_LAYERS:
        archived = old[:, layer - 1]
        current = fresh[:648, LAYERS.index(layer)]
        # Promote before subtraction; exact bits distinguish signed zero as well.
        a, b = archived.astype(np.float32), current.astype(np.float32)
        diff = np.abs(b - a)
        nz = np.abs(a) > 0
        rel = diff[nz] / np.abs(a[nz])
        denom = float(np.linalg.norm(a.astype(np.float64)))
        result.append({'physical_layer': layer, 'bitwise_equal': bool(np.array_equal(archived.view(np.uint16), current.view(np.uint16))),
                       'equal_bit_fraction': float((archived.view(np.uint16) == current.view(np.uint16)).mean()),
                       'absolute_mean': float(diff.mean(dtype=np.float64)), 'absolute_max': float(diff.max()),
                       'absolute_rms': float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))),
                       'relative_l2': float(np.linalg.norm((b - a).astype(np.float64)) / denom) if denom else None,
                       'relative_elementwise_nonzero_reference_mean': float(rel.mean(dtype=np.float64)) if len(rel) else None,
                       'relative_elementwise_nonzero_reference_max': float(rel.max()) if len(rel) else None,
                       'zero_reference_coordinates': int((~nz).sum()),
                       'zero_reference_nonzero_fresh_coordinates': int(((~nz) & (b != 0)).sum())})
    return result


def validate_cache(out, plan_path):
    import numpy as np
    manifest = validate_seal(out, 'gpu-cache', plan_path)
    out = Path(out)
    H = np.load(out / 'features.npy', allow_pickle=False, mmap_mode='r')
    raw = np.load(out / 'raw_scores.npy', allow_pickle=False)
    ranks = np.load(out / 'raw_ranks.npy', allow_pickle=False)
    require(H.shape == (1224, 7, 2048) and H.dtype == np.float16 and np.isfinite(H).all(), 'incomplete/nonfinite feature tensor')
    require(raw.shape == (648, 7, 17) and raw.dtype == np.float32 and np.isfinite(raw).all(), 'incomplete/nonfinite raw scores')
    labels = np.array([r['label'] for r in rows()[:648]])
    require(ranks.dtype == np.int32 and ranks.shape == (648, 7) and np.array_equal(ranks, strict_ranks(raw, labels)), 'raw saved ranks disagree')
    require(read_json(out / 'prompts.json') == rows(), 'cache population changed')
    encoded = read_json(out / 'encoding.json')
    require(len(encoded) == 1224, 'incomplete encoding list')
    for i, record in enumerate(encoded):
        ids = record['input_ids']
        require(record['row'] == i and 1 < len(ids) <= 512 and all(type(x) is int and x >= 0 for x in ids), 'encoding rows/token IDs')
        require(record['length'] == len(ids) and record['readout_index'] == len(ids) - 1, 'readout index/length')
    aliases = read_json(out / 'aliases.json')
    require([a['class'] for a in aliases] == LABELS and all(a['token_ids'] for a in aliases), 'alias classes')
    runtime = read_json(out / 'runtime.json')
    require(runtime['geometry'] == [4, 48, 2048] and runtime['use_cache'] is False, 'cache model geometry/cache mode')
    require(runtime['precision']['parameters'] == 'bfloat16' and runtime['precision']['native_batch'] == 1 and runtime['precision']['readout_batch'] == 7,
            'cache precision/batch semantics')
    require(all(runtime['versions'][k] == v for k, v in GPU_PINS.items()), 'cache runtime pins')
    return H, raw, ranks, manifest


def analyze(args):
    import numpy as np
    plan = load_plan(args.plan)
    out = fresh_dir(args.output)
    start = time.monotonic()
    try:
        write_json(out / 'RUNNING.json', {'status': 'RUNNING_NOT_COMPLETE', 'utc': utc()})
        require(Path(sys.executable).absolute() == OURO / 'venv/bin/python', 'use the existing pinned Ouro CPU interpreter')
        versions = runtime_versions(list(CPU_PINS))
        require(all(versions[k] == v for k, v in CPU_PINS.items()), 'CPU runtime pins changed')
        import sklearn.preprocessing._data as scale_source
        import sklearn.linear_model._logistic as logistic_source
        from threadpoolctl import threadpool_info
        for module in [scale_source, logistic_source]:
            source = str(Path(inspect.getsourcefile(module)).absolute())
            require(source in SOURCE_PINS, 'unexpected installed weighted API source path')
            check_identity(source, {'sha256': SOURCE_PINS[source]})
        for entry in plan['inputs'].values():
            check_identity(entry['path'], entry)
        for path, sha in SOURCE_PINS.items():
            check_identity(path, {'sha256': sha})
        cv, pairing, historical = [load_npz(plan['inputs'][name]['path']) for name in ['cv', 'pairing', 'historical_predictions']]
        selection, design = [read_json(plan['inputs'][name]['path']) for name in ['selection', 'design']]
        checks = validate_population(cv, pairing, selection, design, historical)
        require(checks == plan['folds'], 'prepared fold/weight plan differs')
        supervisor_id = identity(args.receipt)
        validate_supervisor(args.receipt, args.cache, args.plan)
        fresh, raw, raw_rank, cache_manifest = validate_cache(args.cache, args.plan)
        old = load_npz(plan['inputs']['old_cache']['path'])['H']
        require(old.shape == (648, 192, 2048) and old.dtype == np.float16 and np.isfinite(old).all(), 'historical diagnostic cache invalid')
        _, _, pid, folds, labels, eligible = design_arrays()
        arrays = {'labels': labels[:648], 'folds': folds, 'eligible': eligible, 'pid': pid,
                  'archived_cpu_prediction': historical['original'], 'raw_scores': raw, 'raw_rank_all_locations': raw_rank}
        fits = []
        for arm in ['fresh_baseline', 'augmented', 'historical_diagnostic']:
            proba = np.zeros((648, 17), dtype=np.float64)
            prediction = np.zeros(648, dtype=np.int64)
            for f in range(5):
                selected, weights, class_counts = fold_training(labels, folds, f, arm == 'augmented')
                test = np.flatnonzero(folds == f)
                layer = PROBE_LAYERS[f]
                source = old[:, layer - 1] if arm == 'historical_diagnostic' else fresh[:, LAYERS.index(layer)]
                # Only this arm's training rows enter the scaler. No fitting on test or new unsupported classes.
                retained, diagnostic = fit_fixed(source[selected].astype(np.float32), labels[selected],
                                                  source[test].astype(np.float32), weights, CS[f])
                require(set(retained['classes']) == set(labels[selected]), 'fitted class support changed')
                proba[test] = retained['probabilities']
                prediction[test] = retained['prediction']
                prefix = f'{arm}_fold{f}'
                np.savez(out / f'{prefix}.npz', **retained, training_rows=selected, test_rows=test, sample_weights=weights)
                fits.append({'arm': arm, 'fold': f, 'physical_layer': layer, 'class_weights': class_counts, **diagnostic})
                print(json.dumps({'arm': arm, 'fold': f, 'iterations': diagnostic['iterations'], 'elapsed': time.monotonic() - start}), flush=True)
            arrays[f'{arm}_scores'] = proba
            arrays[f'{arm}_rank'] = strict_ranks(proba, labels[:648])
            arrays[f'{arm}_prediction'] = prediction
        require(len(fits) == 15, 'all fifteen fixed fits required')
        arrays['fresh_raw_rank'] = np.array([raw_rank[i, LAYERS.index(RAW_LAYERS[f])] for i, f in enumerate(folds)], dtype=np.int32)
        arrays['archived_original_rank'] = np.array([cv['probe_rank'][i, PROBE_LAYERS[f] - 1] for i, f in enumerate(folds)], dtype=np.int32)
        hits = {arm: (arrays[f'{arm}_rank'] == 0).astype(np.float64)
                for arm in ['fresh_baseline', 'augmented', 'historical_diagnostic', 'fresh_raw', 'archived_original']}
        summaries, draws = {}, {}
        for method, hit in hits.items():
            summary, bootstrap, cluster_sums, denominator = paired_summary(hit, pairing)
            summary['per_fold'] = [float(hit[(folds == f) & eligible].mean()) for f in range(5)]
            summaries[method] = summary
            draws[method] = bootstrap
            draws[f'{method}_cluster_sums'] = cluster_sums
            arrays[f'{method}_hit'] = hit.astype(bool)
        contrasts = {}
        for left, right in [('augmented', 'fresh_baseline'), ('fresh_baseline', 'fresh_raw'), ('augmented', 'fresh_raw'),
                            ('historical_diagnostic', 'archived_original'), ('fresh_baseline', 'historical_diagnostic')]:
            name = f'{left}_minus_{right}'
            summary, bootstrap, cluster_sums, denominator = paired_summary(hits[left] - hits[right], pairing)
            contrasts[name] = summary
            draws[name] = bootstrap
            draws[f'{name}_cluster_sums'] = cluster_sums
        draws['denominator'] = denominator
        # Preserve the actual saved bytes' arrays, not replacement draws.
        draws.update({k: pairing[k] for k in ['weights', 'counts', 'clusters']})
        np.savez(out / 'per_item.npz', **arrays)
        np.savez(out / 'paired_resamples.npz', **draws)
        diagnostics = {'features': feature_diagnostics(old, fresh),
                       'historical_diagnostic_vs_archived_original': {
                           'eligible_rank_agreement': float((arrays['historical_diagnostic_rank'][eligible] == arrays['archived_original_rank'][eligible]).mean()),
                           'eligible_prediction_agreement_with_archived_cpu': float((arrays['historical_diagnostic_prediction'][eligible] == historical['original'][eligible]).mean())},
                       'fresh_baseline_vs_historical_diagnostic': {
                           'eligible_rank_agreement': float((arrays['fresh_baseline_rank'][eligible] == arrays['historical_diagnostic_rank'][eligible]).mean()),
                           'eligible_prediction_agreement': float((arrays['fresh_baseline_prediction'][eligible] == arrays['historical_diagnostic_prediction'][eligible]).mean())},
                       'archived_j_lens_context_only': {'eligible_accuracy': float(pairing['j_lens'][eligible, 0].mean()), 'map': 'historical N80'}}
        runtime = {'versions': versions, 'interpreter': sys.executable, 'platform': platform.platform(),
                   'threadpools_outside_fit': threadpool_info(), 'threads_during_all_scaling_and_solver_operations': 1,
                   'weighted_api_sources': [identity(inspect.getsourcefile(m)) for m in [scale_source, logistic_source]],
                   'implementation': identity(__file__)}
        result = {'status': 'COMPLETE', 'primary': contrasts['augmented_minus_fresh_baseline'], 'accuracies': summaries,
                  'contrasts': contrasts, 'fits': fits, 'diagnostics': diagnostics, 'population': {'eligible_prompts': 576, 'eligible_pairs': 39, 'upper_sum_eligible_prompts': 264},
                  'uncertainty': '20,000 saved paired cluster multiplicities; prompt-weighted denominator; linear percentiles; conditional on fixed training data, models and locations',
                  'interpretation_limits': [
                      'The augmentation adds only upper-sum operand pairs and shifts operand/token distributions.',
                      'The confidence interval does not refit classifiers or quantify augmentation-family uncertainty.',
                      'An interval containing zero does not establish equivalence or a closed gap.',
                      'Historical N80 J-Lens is context only; there is no fresh J-Lens comparator.',
                      'Feature and historical CPU differences are diagnostics, with no numerical acceptance tolerance.',
                      'A positive primary contrast supports this augmentation on the fixed test set; a null or negative result does not establish the original probe was not data limited.'],
                  'cpu_elapsed_seconds_before_seal': time.monotonic() - start}
        write_json(out / 'results.json', result)
        write_json(out / 'runtime.json', runtime)
        write_json(out / 'fold_plan.json', checks)
        # Revalidate every consumed artifact, source, and complete GPU seal after fitting.
        for record in plan['inputs'].values():
            check_identity(record['path'], record)
        for path, sha in SOURCE_PINS.items():
            check_identity(path, {'sha256': sha})
        load_plan(args.plan)
        validate_seal(args.cache, 'gpu-cache', args.plan)
        check_identity(args.receipt, supervisor_id)
        validate_supervisor(args.receipt, args.cache, args.plan)
        bindings = {'contract_sha256': CONTRACT_SHA, 'plan': identity(args.plan), 'implementation': identity(__file__),
                    'historical_inputs': plan['inputs'], 'source_pins': SOURCE_PINS,
                    'supervisor_receipt': supervisor_id,
                    'gpu_manifest': identity(Path(args.cache) / 'manifest.json'), 'gpu_complete': identity(Path(args.cache) / 'COMPLETE.json'),
                    'gpu_outputs': cache_manifest['outputs']}
        seal(out, 'cpu-analysis', bindings, {'fits_completed': 15, 'cpu_elapsed_seconds_before_seal': time.monotonic() - start})
        print(json.dumps({'status': 'COMPLETE', 'output': str(out), 'primary': result['primary']}))
    except BaseException as error:
        failure(out, error)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare', help='bind local fixed inputs and save deterministic metadata; no fits/model loads')
    p.add_argument('--output', required=True)
    p.set_defaults(func=prepare)
    p = commands.add_parser('extract', help='actual GPU execution; only after root confirms P1/P2 interpretation and budget')
    for name in ['snapshot', 'ouro-src', 'jlens-root', 'plan', 'output', 'deadline']:
        p.add_argument('--' + name, required=True)
    p.set_defaults(func=extract)
    p = commands.add_parser('supervise-extract', help='external process-group timeout plus receipt; starts complete GPU-phase timer')
    for name in ['python', 'snapshot', 'ouro-src', 'jlens-root', 'plan', 'output', 'deadline', 'receipt']:
        p.add_argument('--' + name, required=True)
    p.set_defaults(func=supervise_extract)
    p = commands.add_parser('analyze', help='actual 15 CPU fits, only from complete sealed GPU cache')
    for name in ['plan', 'cache', 'output', 'receipt']:
        p.add_argument('--' + name, required=True)
    p.set_defaults(func=analyze)
    p = commands.add_parser('validate', help='validate plan and optionally complete cache without fitting')
    p.add_argument('--plan', required=True)
    p.add_argument('--cache')
    p.add_argument('--receipt')
    def validate_command(a):
        load_plan(a.plan)
        if a.cache:
            require(a.receipt is not None, '--cache requires --receipt')
            validate_supervisor(a.receipt, a.cache, a.plan)
            validate_cache(a.cache, a.plan)
        print('VALID')
    p.set_defaults(func=validate_command)
    p = commands.add_parser('self-test', help='local CPU synthetic fixtures only')
    p.add_argument('--proof', required=True)
    p.set_defaults(func=self_test)
    args = parser.parse_args()
    args.func(args)


def self_test(args):
    """Meaningful small CPU fixtures; never load a checkpoint or real feature cache."""
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from types import SimpleNamespace
    sys.path[:0] = [str(OURO / 'src'), str(REPO)]
    from ouro_jlens.recurrent import LoopTap, OuroLensModel
    from jlens.hooks import ActivationRecorder
    from jlens.hf import HFLensModel
    from ouro_jlens.evaldata import surface_forms, single_token_ids
    checks = []
    def check(name, condition):
        require(condition, f'self-test failed: {name}')
        checks.append(name)
    def rejects(name, function):
        try:
            function()
        except (ValueError, FileNotFoundError, TimeoutError):
            checks.append(name)
        else:
            raise AssertionError(f'self-test failed to reject: {name}')
    check_identity(HERE / 'CONTRACT.md', {'sha256': CONTRACT_SHA})
    exercised_sources = [OURO / 'src/ouro_jlens/recurrent.py', OURO / 'src/ouro_jlens/evaldata.py',
                         REPO / 'jlens/hooks.py', REPO / 'jlens/hf.py',
                         OURO / 'venv/lib/python3.14/site-packages/sklearn/preprocessing/_data.py',
                         OURO / 'venv/lib/python3.14/site-packages/sklearn/linear_model/_logistic.py']
    for source in exercised_sources:
        check_identity(source, {'sha256': SOURCE_PINS[str(source)]})
    check('frozen_contract_and_exercised_source_pins', True)
    population, pairs, pid, folds, labels, eligible = design_arrays()
    check('1224_literal_prompts_old_enumeration_trailing_space', len(population) == 1224 and population[0]['prompt'] == '(1 + 1) * 2 = ' and population[647]['prompt'] == '(9 + 9) * 9 = ' and all(r['prompt'].endswith(' = ') for r in population))
    check('576_new_rows_36_unordered_pairs_no_reversal_leakage', len(population[648:]) == 576 and len({tuple(sorted((r['a'], r['b']))) for r in population[648:]}) == 36 and all(max(r['a'], r['b']) >= 10 for r in population[648:]))
    check('eligible576_39_pairs_upper264', eligible.sum() == 576 and len(set(pid[eligible])) == 39 and (eligible & (labels[:648] >= 11)).sum() == 264)
    for f in range(5):
        ix, w, counts = fold_training(labels, folds, f, True)
        old, baseline_w, _ = fold_training(labels, folds, f, False)
        check(f'fold{f}_row_count_weight_objective', len(ix) == [1096, 1088, 864, 1088, 1096][f] and np.all(baseline_w == 1) and np.isclose(w.sum(), len(old), rtol=0, atol=1e-10) and all(abs(c['actual_weight_sum'] - c['original_count']) < 1e-10 for c in counts))
        check(f'fold{f}_classes_test_mask_pair_leakage', set(labels[ix]) == set(labels[old]) and not set(ix) & set(np.flatnonzero(folds == f)) and ((folds == f) & eligible).sum() == [104, 136, 80, 136, 120][f])
        for c in counts:
            check(f'fold{f}_class{c["class"]}_weights_both_old_and_new', np.all(w[labels[ix] == c['class']] == c['original_count'] / c['included_count']))
    tied = np.zeros((3, 17)); tied[0, :2] = .5; tied[1, 0] = .6; tied[1, 1] = .4
    check('strict_top1_ties_count_as_success', strict_ranks(tied, np.array([3, 3, 18])).tolist() == [0, 1, 0])
    expanded = expand_probabilities(np.array([[.5, .5], [.1, .9]]), [2, 18])
    check('missing_classes_zero_probability_fixed17', np.all(expanded[:, 1:16] == 0) and strict_ranks(expanded, np.array([18, 3])).tolist() == [0, 2])
    rejects('nonfinite_score_refused', lambda: strict_ranks(np.full((1, 17), np.nan), [2]))
    # Unequal 8/16 prompt cluster counts make a wrong equal-pair denominator visible.
    fixture = {'eligible': np.ones(24, bool), 'pid': np.array([10] * 8 + [20] * 16), 'clusters': np.array([20, 10]),
               'counts': np.array([16, 8]), 'weights': np.array([[2, 0], [1, 1], [0, 2]], np.int64)}
    hit = np.array([1.] * 8 + [0.] * 16)
    summary, bootstrap, A, D = paired_summary(hit, fixture)
    check('unequal_pair_denominator_saved_column_order', summary['point'] == 1/3 and A.tolist() == [0, 8] and D.tolist() == [32, 24, 16] and np.array_equal(bootstrap, [0, 1/3, 1]))
    check('linear_percentile_rule', np.array_equal(summary['ci95'], np.percentile([0, 1/3, 1], [2.5, 97.5], method='linear')))
    other = 1 - hit
    _, diff, _, _ = paired_summary(hit - other, fixture)
    _, other_bs, _, _ = paired_summary(other, fixture)
    check('paired_shared_draws_and_denominator', np.allclose(diff, bootstrap - other_bs))
    # Weighted duplicate observations preserve both scaler and penalized objective.
    rng = np.random.default_rng(121)
    X = rng.normal(size=(36, 6)).astype(np.float32)
    y = np.tile([2, 7, 18], 12)
    Xt = rng.normal(size=(9, 6)).astype(np.float32)
    first, first_d = fit_fixed(X, y, Xt, np.ones(36), 1.)
    repeated, repeated_d = fit_fixed(np.repeat(X, 2, axis=0), np.repeat(y, 2), Xt, np.full(72, .5), 1.)
    check('weighted_scaler_duplicate_invariance', np.allclose(first['scaler_mean'], repeated['scaler_mean'], atol=1e-12) and np.allclose(first['scaler_var'], repeated['scaler_var'], atol=1e-12))
    check('weighted_solver_objective_duplicate_invariance', np.allclose(first['coef'], repeated['coef'], atol=1e-6) and np.allclose(first['probabilities'], repeated['probabilities'], atol=1e-6) and first_d['mean_loss_l2_strength_float64'] == repeated_d['mean_loss_l2_strength_float64'])
    check('one_native_thread_scaler_and_solver', all(p['num_threads'] == 1 for d in [first_d, repeated_d] for p in d['threadpools_during_fit']))
    rejects('nonfinite_fit_input_refused', lambda: fit_fixed(np.full((3, 2), np.nan, np.float32), np.array([2, 3, 4]), np.ones((1, 2), np.float32), np.ones(3), 1.))
    from unittest.mock import patch
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    original_fit = LogisticRegression.fit
    def warning_fit(self, *a, **kw):
        fitted = original_fit(self, *a, **kw)
        warnings.warn('synthetic convergence failure', ConvergenceWarning)
        return fitted
    with patch.object(LogisticRegression, 'fit', warning_fit):
        rejects('convergence_warning_makes_fit_incomplete', lambda: fit_fixed(X, y, Xt, np.ones(36), 1.))
    # Exercise the real encode method, LoopTap and ActivationRecorder with a CPU model.
    class Tokenizer:
        bos_token_id = 99
        def __call__(self, text, truncation=False, max_length=None, **kwargs):
            ids = [ord(c) for c in text]
            if truncation:
                ids = ids[:max_length]
            return SimpleNamespace(input_ids=ids)
        def encode(self, text, add_special_tokens=False):
            return [int(text.strip())] if text.strip().isdigit() else [42, 43]
    class Block(torch.nn.Module):
        def __init__(self, number):
            super().__init__(); self.number = number
        def forward(self, x, current_ut):
            return x + self.number + current_ut * 1000
    class Fake:
        n_ut, n_physical, n_layers, d_model = 4, 48, 192, 2048
        input_device = torch.device('cpu')
        tokenizer = Tokenizer()
        encode = OuroLensModel.encode
        def __init__(self):
            self.blocks = [Block(n + 1) for n in range(48)]
            self.layers = [LoopTap(b, ut) for ut in range(4) for b in self.blocks]
            self.calls = 0
        def forward(self, ids):
            self.calls += 1
            x = torch.arange(ids.shape[1], dtype=torch.float32)[None, :, None].expand(1, -1, 2048).clone()
            for ut in range(4):
                for block in self.blocks:
                    x = block(x, current_ut=ut)
            return x
    class NoDeadline:
        def check(self, phase):
            pass
    fake = Fake(); selected_rows = population[:2]
    encoded = validate_encoded(fake, selected_rows)
    h = extract_features(fake, selected_rows, encoded, ActivationRecorder, NoDeadline())
    expected = np.array([layer * (layer + 1) / 2 + encoded[0]['length'] - 1 for layer in LAYERS], dtype=np.float16)
    check('native_b1_explicit_bos_complete_trailing_space', encoded[0]['input_ids'][0] == 99 and encoded[0]['input_ids'][-1] == ord(' ') and encoded[0]['readout_index'] == len(encoded[0]['input_ids']) - 1)
    check('loop1_postblock_seven_axis_last_token_no_later_overwrite', h.shape == (2, 7, 2048) and h.dtype == np.float16 and np.array_equal(h[0, :, 0], expected) and fake.calls == 2)
    check('hooks_removed_after_extraction', all(not b._forward_hooks for b in fake.blocks))
    fake.n_ut = 3
    rejects('wrong_recurrence_geometry_refused', lambda: extract_features(fake, selected_rows, encoded, ActivationRecorder, NoDeadline()))
    fake.n_ut = 4
    rejects('truncated_prompt_refused', lambda: validate_encoded(fake, [{'row': 0, 'prompt': 'x' * 512}]))
    class AliasTokenizer(Tokenizer):
        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=self.encode(text))
    aliases = [{'class': s, 'surface_forms': surface_forms(str(s)), 'token_ids': single_token_ids(AliasTokenizer(), surface_forms(str(s)))} for s in LABELS]
    check('existing_surface_forms_single_token_helpers', all(a['token_ids'] == [a['class']] for a in aliases))
    class FakeNorm(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.seen = []
        def forward(self, x):
            self.seen.append((tuple(x.shape), x.dtype))
            return x
    class FakeHead(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.empty(1, dtype=torch.bfloat16)
        def forward(self, x):
            return x[:, :1] + torch.arange(64, dtype=torch.bfloat16)[None, :]
    class FakeRaw:
        input_device = torch.device('cpu')
        _final_norm = FakeNorm()
        _lm_head = FakeHead()
        _logit_softcap = None
        unembed = HFLensModel.unembed
    raw_model = FakeRaw()
    synthetic_h = np.zeros((648, 7, 2048), dtype=np.float16)
    synthetic_h[:, :, 0] = np.arange(7, dtype=np.float16)[None, :]
    raw_aliases = [{'token_ids': [s, s + 20]} for s in LABELS]
    synthetic_raw = raw_scores(raw_model, synthetic_h, raw_aliases, NoDeadline())
    check('raw_native_unembed_fixed_batch7_bf16_norm_head', len(raw_model._final_norm.seen) == 648 and all(shape == (7, 2048) and dtype == torch.bfloat16 for shape, dtype in raw_model._final_norm.seen))
    check('raw17class_max_over_all_aliases_from_fp16', synthetic_raw.dtype == np.float32 and np.array_equal(synthetic_raw[0], np.arange(7)[:, None] + np.arange(22, 39)[None, :]) and np.array_equal(synthetic_raw[-1], synthetic_raw[0]))
    from tokenizers import AddedToken
    metadata = tokenizer_metadata({'token': AddedToken('<x>', lstrip=True, special=True), 'nested': [1, False]})
    check('added_token_metadata_explicit_serializable', metadata['token']['content'] == '<x>' and metadata['token']['lstrip'] is True and json.loads(canonical(metadata)) == metadata)
    with tempfile.TemporaryDirectory(prefix='p4-synthetic-') as tmp:
        base = Path(tmp)
        source = base / 'source.py'; source.write_text('original')
        original = identity(source); source.write_text('tampered')
        rejects('source_tamper_refused', lambda: check_identity(source, original))
        out = base / 'cache'; out.mkdir()
        plan = base / 'plan.json'; write_json(plan, {'fixture': True})
        rejects('incomplete_cache_refused', lambda: validate_seal(out, 'gpu-cache', plan))
        np.save(out / 'fixture.npy', np.array([1.], np.float16), allow_pickle=False)
        bindings = {'contract_sha256': CONTRACT_SHA, 'plan': identity(plan), 'implementation': identity(__file__)}
        seal(out, 'gpu-cache', bindings, {'gpu_elapsed_seconds': 1})
        validate_seal(out, 'gpu-cache', plan)
        check('complete_sealed_fixture_accepted', True)
        # No scientific draw generation: repeat the saved synthetic multiplicities
        # verbatim and compare to an explicit denominator computation at 20,000 rows.
        many = dict(fixture, weights=np.tile(fixture['weights'], (6667, 1))[:20000])
        _, many_bs, _, many_d = paired_summary(hit, many)
        check('20000_retained_multiplicity_rows_exact_replay', len(many_bs) == 20000 and np.array_equal(many_bs, np.tile(bootstrap, 6667)[:20000]) and np.array_equal(many_d, np.tile(D, 6667)[:20000]))
        (out / 'fixture.npy').write_bytes(b'tampered')
        rejects('sealed_cache_tamper_refused', lambda: validate_seal(out, 'gpu-cache', plan))
        receipt = base / 'expired.json'; now = time.monotonic()
        write_json(receipt, {'start_monotonic': now - 901, 'deadline_monotonic': now - 1, 'limit_seconds': 900})
        rejects('expired_deadline_refused_before_import', lambda: Deadline(receipt))
        for name, state in [('killed_child_with_complete_refused', {'child_exit_code': -9, 'timed_out': True}),
                            ('late_child_with_complete_refused', {'child_exit_code': 0, 'timed_out': False})]:
            r = base / (name + '.json')
            write_json(r, {'schema': 'p4-supervisor-v1', 'status': 'COMPLETE',
                          'start_monotonic': now - 901, 'end_monotonic': now, 'elapsed_seconds': 901,
                          'deadline_monotonic': now - 1, **state})
            rejects(name, lambda r=r: validate_supervisor(r, out, plan))
        rejects('missing_supervisor_receipt_refused', lambda: validate_supervisor(base / 'missing.json', out, plan))
        def supervised_fixture(name, child_elapsed=750.):
            cache_dir = base / name; cache_dir.mkdir()
            timer = {'start_monotonic': 100., 'deadline_monotonic': 1000., 'limit_seconds': 900}
            timer_path = base / (name + '-deadline.json'); write_json(timer_path, timer)
            events = [{'phase': 'synthetic_cleanup_complete', 'monotonic': 850., 'elapsed_seconds': 750.}]
            write_json(cache_dir / 'timing.json', {'receipt': timer, 'events': events})
            seal(cache_dir, 'gpu-cache', {**bindings, 'deadline': identity(timer_path)},
                 {'gpu_elapsed_seconds': child_elapsed, 'events': events})
            receipt = {'schema': 'p4-supervisor-v1', 'status': 'COMPLETE', 'child_exit_code': 0, 'timed_out': False,
                       'start_monotonic': 100., 'end_monotonic': 851., 'deadline_monotonic': 1000., 'elapsed_seconds': 751.,
                       'plan': identity(plan), 'implementation': identity(__file__), 'deadline': identity(timer_path),
                       'cache_manifest': identity(cache_dir / 'manifest.json'), 'cache_complete': identity(cache_dir / 'COMPLETE.json')}
            return cache_dir, receipt
        supervised, good_receipt = supervised_fixture('supervised')
        good_path = base / 'good-supervisor.json'; write_json(good_path, good_receipt)
        validate_supervisor(good_path, supervised, plan)
        check('consistent_external_complete_receipt_accepted', True)
        shortened = {**good_receipt, 'end_monotonic': 101., 'elapsed_seconds': 1.}
        short_path = base / 'short-supervisor.json'; write_json(short_path, shortened)
        rejects('external_elapsed_must_cover_child_checkpoint', lambda: validate_supervisor(short_path, supervised, plan))
        late_event_cache, late_event_receipt = supervised_fixture('late-event', child_elapsed=.5)
        late_event_path = base / 'late-event-supervisor.json'
        write_json(late_event_path, {**late_event_receipt, 'end_monotonic': 101., 'elapsed_seconds': 1.})
        rejects('external_end_must_cover_child_events', lambda: validate_supervisor(late_event_path, late_event_cache, plan))
        # Execute real cancellation cleanup with a CPU-only sleeping child.
        # The shim ignores extraction arguments and never imports GPU libraries.
        for sig in [signal.SIGTERM, signal.SIGHUP]:
            trial = base / sig.name; trial.mkdir()
            child_pid_path = trial / 'child.pid'
            shim = trial / 'synthetic-python'
            shim.write_text(f'#!{sys.executable}\nimport os,time\nfrom pathlib import Path\nPath({str(child_pid_path)!r}).write_text(str(os.getpid()))\ntime.sleep(30)\n')
            shim.chmod(0o700)
            command = [sys.executable, '-B', str(Path(__file__).absolute()), 'supervise-extract',
                       '--python', str(shim), '--snapshot', str(trial / 'unused'),
                       '--ouro-src', str(trial / 'unused'), '--jlens-root', str(trial / 'unused'),
                       '--plan', str(plan), '--output', str(trial / 'cache'),
                       '--deadline', str(trial / 'deadline.json'), '--receipt', str(trial / 'receipt.json')]
            parent = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            child_pid = None
            try:
                until = time.monotonic() + 5
                while not child_pid_path.exists() and time.monotonic() < until and parent.poll() is None:
                    time.sleep(.01)
                require(child_pid_path.exists(), 'synthetic supervisor child failed to start')
                child_pid = int(child_pid_path.read_text())
                os.kill(parent.pid, sig)
                parent.communicate(timeout=5)
                try:
                    os.killpg(child_pid, 0)
                except ProcessLookupError:
                    gone = True
                else:
                    gone = False
                cancelled = read_json(trial / 'receipt.json')
                check(f'{sig.name}_kills_real_synthetic_child_group', gone and parent.returncode != 0 and cancelled['status'] == 'INCOMPLETE' and cancelled['child_exit_code'] == -signal.SIGKILL)
            finally:
                if parent.poll() is None:
                    os.killpg(parent.pid, signal.SIGKILL)
                    parent.wait()
                if child_pid is not None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(child_pid, signal.SIGKILL)
    proof = {'status': 'PASS_SYNTHETIC_CPU_ONLY', 'utc': utc(), 'contract': identity(HERE / 'CONTRACT.md'),
             'implementation': identity(__file__), 'runtime': runtime_versions(['numpy', 'scikit-learn', 'scipy', 'threadpoolctl', 'torch']),
             'interpreter': sys.executable, 'checks_count': len(checks), 'checks': checks,
             'real_features_read': False, 'real_classifier_fits': 0, 'synthetic_classifier_fits': 3,
             'model_loads': 0, 'cuda_calls': 0, 'cloud_actions': 0,
             'sources': [identity(source) for source in exercised_sources],
             'limitations': ['Synthetic CPU tests do not establish GPU timing, native model execution, or scientific outcomes.',
                             'Real historical identity validation and saved resample validation belong to prepare/analyze; no replacement scientific resamples are generated.']}
    write_json(args.proof, proof)
    print(json.dumps({'status': proof['status'], 'checks': len(checks), 'proof': identity(args.proof)}))


if __name__ == '__main__':
    main()
