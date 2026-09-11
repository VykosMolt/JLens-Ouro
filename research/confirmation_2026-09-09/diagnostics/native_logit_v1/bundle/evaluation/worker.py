#!/usr/bin/env python3
"""Native-logit diagnostic: the precision_v1 runtime gate (M0), then measurements M1-M4.

Measured mismatches are data: once M0 passes, a run that finishes publishes a
complete final manifest. Any exception publishes failed; a stop publishes stopped.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import signal
import sys
import time

from artifacts import Publisher, atomic, json_save, record, torch_save

class StopRequested(Exception): pass


def forward_captures(model, ids):
    """M2: the M1 native call and activation recorder, plus read-only final-norm and lm_head hooks."""
    import torch
    from jlens.hooks import ActivationRecorder
    seen = {'norm_input': [], 'norm_output': [], 'lm_head_input': [], 'lm_head_output': []}
    def reader(prefix):
        def hook(module, args, output):
            seen[prefix + '_input'].append(args[0].detach())
            seen[prefix + '_output'].append(output.detach())
        return hook
    handles = [model.hf_model.model.norm.register_forward_hook(reader('norm')),
               model.hf_model.lm_head.register_forward_hook(reader('lm_head'))]
    try:
        with ActivationRecorder(model.layers, at=range(192)) as rec:
            output = model.hf_model(input_ids=ids, use_cache=False)
    finally:
        for handle in handles: handle.remove()
    if [len(tensors) for tensors in seen.values()] != [model.n_ut, model.n_ut, 1, 1]:
        raise ValueError('Final norm or lm_head call count differs from the native forward')
    # Copies leave the device only after the forward has returned.
    return {'norm_input': torch.stack([t.cpu() for t in seen['norm_input']]),
            'norm_output': torch.stack([t.cpu() for t in seen['norm_output']]),
            'lm_head_input': seen['lm_head_input'][0].cpu(), 'lm_head_output': seen['lm_head_output'][0].cpu(),
            'logits': output.logits.detach().cpu(), 'virtual191': rec.activations[191][0, -1].float().cpu()}


def shaped_inputs(target, positions, stacked):
    """Defined shapes containing `target`, the last row of `positions`: label -> (input, target row index)."""
    import torch
    cases = {'[2048]': (target, -1), '[1,2048]': (target[None], -1), '[1,1,2048]': (target[None, None], -1),
             '[S,2048]': (positions, -1), '[1,S,2048]': (positions[None], -1)}
    for n in stacked:
        # The other rows are the item's own positions, cycled.
        fill = positions[torch.arange(n - 1) % len(positions)]
        cases[f'[{n},2048]/first'] = (torch.cat([target[None], fill]), 0)
        cases[f'[{n},2048]/last'] = (torch.cat([fill, target[None]]), -1)
    return cases


def recompute(model, run, step):
    """M3: target rows of every defined recomputation, the whole sequence performed twice."""
    import torch
    from validate_outputs import COMPOSITE_LABELS, NORM_LABELS
    norm, head = model.hf_model.model.norm, model.hf_model.lm_head
    norm_cases = shaped_inputs(run['norm_input'][step, 0, -1], run['norm_input'][step, 0], (148, 160))
    head_cases = shaped_inputs(run['lm_head_input'][0, -1], run['lm_head_input'][0], (148, 160))
    plan = (('norm', norm, norm_cases, NORM_LABELS), ('lm_head', head, head_cases, NORM_LABELS),
            ('lm_head_norm', lambda t: head(norm(t)), norm_cases, COMPOSITE_LABELS))
    rows = {family: {label: [] for label in labels} for family, _, _, labels in plan}
    unembedded = []
    for _ in range(2):
        for family, function, cases, labels in plan:
            for label in labels:
                tensor, index = cases[label]
                # Every input is a fresh contiguous tensor on the model device.
                output = function(tensor.to(model.input_device, copy=True))
                rows[family][label].append(output.reshape(-1, output.shape[-1])[index].cpu())
        # The attempt05 path: readouts.native_development unembeds the FP32 virtual[191] row.
        unembedded.append(model.unembed(run['virtual191'].to(model.input_device)).cpu())
    result = {family: {label: torch.stack(r) for label, r in labels.items()} for family, labels in rows.items()}
    result['unembed_fp32_virtual191'] = torch.stack(unembedded)
    return result


def measure(model, items, check_stop):
    """M1 unchanged over all items; then, for each item in turn, M2, M3 and M4."""
    import torch
    from readouts import native_development
    from validate_outputs import exit_step
    native = native_development(model, items)
    tensors = {'item_names': [item.name for item in items], 'items': {}}
    with torch.no_grad():
        for item in items:
            check_stop()
            ids = torch.tensor([item.token_ids], device=model.input_device)
            first = forward_captures(model, ids)
            recomputed = recompute(model, first, exit_step(first)[1])
            second = forward_captures(model, ids)
            tensors['items'][item.name] = {'token_ids': list(item.token_ids), 'runs': [first, second], **recomputed}
    return native, tensors


def save_outputs(root, native, tensors):
    from validate_outputs import comparisons
    torch_save(root/'development/native.pt', native)
    torch_save(root/'diagnostic/tensors.pt', tensors)
    json_save(root/'diagnostic/comparisons.json', comparisons(native, tensors))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(); config = json.loads(args.config.read_text())
    work, bundle = Path(config['work']), Path(config['bundle'])
    specpath = bundle/'frozen/run_spec.json'; spec = json.loads(specpath.read_text())
    contract = json.loads((bundle/'frozen/output_contract.json').read_text())
    binding = config['binding']
    if record(specpath)['sha256'] != binding['run_spec_sha256'] or record(bundle/'frozen/output_contract.json')['sha256'] != binding['output_contract_sha256']:
        raise ValueError('Launch binding differs from frozen specification')
    publisher = Publisher(work,binding,contract)
    deadline = datetime.fromisoformat(config['work_deadline_utc'].replace('Z','+00:00')).timestamp()
    stopped = False
    def stop(signum,frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    def check_stop():
        if stopped or (work/'STOP').exists() or time.time() >= deadline:
            raise StopRequested('Early stop/deadline')
    def verify_sources():
        for rel, expected in spec['source_records'].items():
            if record(bundle/rel) != expected: raise ValueError(f'Changed source: {rel}')
    def stage_status(phase):
        check_stop(); publisher.status(phase,updated_utc=datetime.now(timezone.utc).isoformat())
    def old_items(evaluator,model):
        allitems = evaluator.load_items(model.tokenizer,('multihop',),encode=lambda t:model.encode(t)[0].tolist())
        chosen = [item for name in spec['development_item_names'] for item in allitems if item.name == name]
        if [i.name for i in chosen] != spec['development_item_names']: raise ValueError('Development items changed')
        return chosen
    try:
        # M0: the precision_v1 worker's source, snapshot, imported-module and precision checks.
        verify_sources()
        sys.dont_write_bytecode=True
        sys.path[:0] = [str(bundle/'repo'),str(bundle/'ouro_project/src'),str(bundle/'legacy')]
        import torch
        from ouro_jlens import evaluate as evaluator, recurrent
        import evaluate_refits as common
        import run_refits as legacy
        torch.set_num_threads(8); torch.set_num_interop_threads(1)
        env = json.loads((bundle/'legacy/environment.json').read_text())
        environment = legacy._environment(torch, env)
        model_manifest = json.loads((bundle/'legacy/model_manifest.json').read_text())
        snapshot = Path(config['snapshot'])
        model_files = legacy._snapshot(snapshot,model_manifest)
        imported = []
        for module in ('ouro_jlens.evaluate','ouro_jlens.evaldata','ouro_jlens.recurrent','jlens','jlens.hf','jlens.hooks','jlens.lens','jlens.vis'):
            actual = Path(importlib.import_module(module).__file__).resolve()
            relative = actual.relative_to(bundle).as_posix()
            if relative not in spec['source_records'] or record(actual) != spec['source_records'][relative]: raise ValueError('Unpinned imported source')
            imported.append({'module':module,'path':relative,'record':record(actual)})
        stage_status('load_model')
        model, runtime = common._evaluation_runtime(torch,{'environment':env,'spec':{'model':spec['model_identity']}},model_files,snapshot,recurrent,imported)
        if runtime['precision'] != spec['original_precision']:
            raise ValueError('Readout precision switches differ from the original evaluation')
        atomic(publisher.root/'run_spec.json', lambda f, raw=specpath.read_bytes(): f.write(raw))
        json_save(publisher.root/'provenance.json',{'binding':binding,'source_records':spec['source_records'],
                  'model_record':spec['model_record'],'dtype':'torch.bfloat16',
                  'geometry':[model.n_ut,model.n_physical,model.n_layers,model.d_model], 'runtime':runtime,
                  'actual_worker_source':record(Path(__file__)), 'config_record':record(args.config)})
        stage_status('native_logit_measurements')
        native, tensors = measure(model, old_items(evaluator,model), check_stop)
        save_outputs(publisher.root, native, tensors)
        verify_sources()
        publisher.publish()
        publisher.status('computation_finished',outcome='complete')
    except BaseException as error:
        outcome = 'stopped' if isinstance(error,StopRequested) else 'failed'
        try:
            publisher.publish(outcome=outcome)
            publisher.status('computation_finished',outcome=outcome,error_type=type(error).__name__,error=str(error))
        finally:
            raise

if __name__=='__main__': main()
