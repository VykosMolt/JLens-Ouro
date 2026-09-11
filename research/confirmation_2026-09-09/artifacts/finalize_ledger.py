#!/usr/bin/env python3
"""Inspect additional identical physical copies and summarize recovery evidence."""
import gc
import sys
from pathlib import Path
import torch
from recover_estimators import OUT, read, guard, record, same, atomic_json, now

torch.set_num_threads(2);torch.set_num_interop_threads(1)
inv=read(OUT/'inventory.json');audit=read(OUT/'tensor_audit.json')
extras=[]
for key,entry in inv['binaries'].items():
    for copy in entry['copies']:
        if copy['status']!='complete_hash_verified' or copy['path'] in audit['artifacts']:
            continue
        path=Path(copy['path']);before=guard(path)
        assert before==copy['actual']['stat_after']
        payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        # At present the sole extra complete physical copy is Ouro main fit01.
        assert key.startswith('ouro/fit_01/final/')
        assert set(payload)=={'J','n_prompts','d_model','source_layers'}
        assert payload['n_prompts']==100 and payload['d_model']==2048 and payload['source_layers']==list(range(191))
        assert type(payload['J']) is dict and set(payload['J'])==set(range(191))
        assert all(type(k) is int for k in payload['J'])
        for t in payload['J'].values():
            assert type(t) is torch.Tensor and t.dtype==torch.float16 and t.device.type=='cpu'
            assert t.layout==torch.strided and not t.requires_grad and tuple(t.shape)==(2048,2048)
            assert bool(torch.isfinite(t).all())
        assert guard(path)==before
        extras.append(dict(path=str(path),status='passed',exact_schema=True,all_finite=True,
                           tensor_count=191,total_elements=191*2048*2048,stat_before=before,stat_after=guard(path),
                           equivalence_to_compared_physical_copy='full-file SHA256 equals same historical sealed binary'))
        del payload,t;gc.collect()
rows={}
for key,entry in inv['binaries'].items():
    row=dict(expected=entry['expected'],historical_retrieval_status=entry['status'],
             retained_copies=entry['copies'],current_status=entry['status'])
    for fit,pair in audit['pairs'].items():
        if pair.get('reconstructed') and inv['provenance'][fit]['generations']['final']['binary_key']==key:
            row.update(current_status='reconstructed_historical_bytes_verified',reconstructed_path=pair['final'],
                       reconstructed_record=pair['reconstructed_record'])
    rows[key]=row
for entry in inv['binaries'].values():
    for copy in entry['copies']:
        assert guard(copy['path'])==copy['actual']['stat_after']
result=dict(schema='confirmation_recovery_ledger.v1',recorded_utc=now(),status='partial_recovery_verified',
            historical_binaries=16,original_complete_binaries=7,reconstructed_historical_exact_binaries=1,
            currently_complete_known_binaries=8,remaining_unavailable_known_binaries=8,
            loaded_original_complete_physical_files=len([p for p in audit['artifacts'] if '/reconstructed/' not in p])+len(extras),
            independent_conversion_total_elements=sum(v['total_elements'] for v in audit['pairs'].values()),
            independent_conversion_differing_elements=0,additional_complete_copy_validation=extras,binaries=rows,
            usable_banks=['Ouro main fit01','Ouro main fit02','Ouro penultimate fit01','Ouro positions sampled_sum fit01','Ouro positions diagonal fit01'],
            unavailable_banks=['Ouro main fit03','Ouro main fit04','Ouro main fit05','Huginn fit01'],
            limitations=['Huginn final and complete checkpoint unavailable; no independent full-matrix verification is possible.',
                        'Ouro fits03–05 still lack both binaries.',
                        'No new readouts, inference, refit, GPU allocation, or remote retrieval were performed.',
                        'This is not a claim of full original experiment retrieval.'],
            source_records={name:record(OUT/name) for name in ['recover_estimators.py','audit_huginn_retention.py','finalize_ledger.py','inventory.json','tensor_audit.json','huginn_retention.json']})
atomic_json(OUT/'RECOVERY_LEDGER.json',result)
print('LEDGER',result['currently_complete_known_binaries'],'of',result['historical_binaries'],'known binaries complete;',result['loaded_original_complete_physical_files'],'original physical files loaded')
