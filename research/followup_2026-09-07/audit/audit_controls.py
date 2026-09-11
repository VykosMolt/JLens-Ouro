"""Inspect task-name token overlap and freeze dependence labels from metadata.

Uses the original token-form construction (no scoring/report helpers). Run via
/home/moloch/ouro_project/venv/bin/python. This does not recompute readout effects.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from audit_existing import OUT, PROJECT, SOURCE, OPERATIONS, WORDS

sys.path.insert(0, str(PROJECT / 'src'))
from ouro_jlens.evaldata import single_token_ids, surface_forms  # noqa: E402

names = json.loads((SOURCE / 'task_names.json').read_text())
items = json.loads((OUT / 'continuation_audit.json').read_text())
tokenizer = AutoTokenizer.from_pretrained(
    str(PROJECT / 'artifacts/hf_cache/hub/models--ByteDance--Ouro-2.6B/snapshots/1ed04250da1a9936042725d302e81c8fa2ab5abd'),
    trust_remote_code=True, local_files_only=True,
)
forms = {task: {n: set(single_token_ids(tokenizer, surface_forms(n))) for n in ns}
         for task, ns in names.items()}
own_overlap, target_overlap, clusters = [], [], []
for item in items:
    if not item['eligible_slots']:
        continue
    task, own = item['task'], set(item['intermediates'])
    controls = [name for name in names[task] if name not in own and name not in OPERATIONS]
    target_ids = set(single_token_ids(tokenizer, surface_forms(item['target'])))
    overlap = [name for name in controls if target_ids & forms[task][name]]
    if overlap:
        target_overlap.append({'index': item['index'], 'name': item['name'], 'task': task,
                               'target': item['target'], 'controls_overlapping_target': overlap})
    for slot in item['eligible_slots']:
        name = item['intermediates'][slot]
        overlap = [control for control in controls if forms[task][name] & forms[task][control]]
        if overlap:
            own_overlap.append({'index': item['index'], 'name': item['name'], 'own': name,
                                'controls_overlapping_own': overlap})
    # These are TWO separate sensitivity schemes, not a claim that either one
    # identifies the true independent sampling units of this curated dataset.
    concepts = [item['intermediates'][slot] for slot in item['eligible_slots']]
    canonical = sorted(set(str(3 if name == 'third' else WORDS.get(name, name)) for name in concepts))
    clusters.append({'index': item['index'], 'name': item['name'], 'task': task,
                     'concept_cluster': task + ':' + '|'.join(canonical),
                     'name_prefix_cluster': task + ':' + item['name'].split('-')[0]})
result = {'token_forms': {task: {name: sorted(ids) for name, ids in fs.items()} for task, fs in forms.items()},
          'own_control_overlap': own_overlap, 'target_control_overlap': target_overlap,
          'target_overlap_counts': dict(Counter(row['task'] for row in target_overlap)),
          'cluster_counts': {task: {col: len({row[col] for row in clusters if row['task'] == task})
                                    for col in ('concept_cluster', 'name_prefix_cluster')}
                             for task in names}}
(OUT / 'control_forms.json').write_text(json.dumps(result, indent=2) + '\n')
with (OUT / 'dependence_clusters.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(clusters[0]))
    writer.writeheader()
    writer.writerows(clusters)
print(json.dumps({key: result[key] for key in ('own_control_overlap', 'target_overlap_counts', 'cluster_counts')}, indent=2))
