#!/usr/bin/env python3
"""Bounded frozen-estimator evaluation with external development acceptance."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import time

from artifacts import Publisher, record, json_save, torch_save, npz_save

class StopRequested(Exception): pass

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
            raise StopRequested('Early stop/deadline; preserve committed stages')
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
        verify_sources()
        sys.dont_write_bytecode=True
        sys.path[:0] = [str(bundle/'repo'),str(bundle/'ouro_project/src'),str(bundle/'legacy')]
        import torch
        from ouro_jlens import evaluate as evaluator, recurrent, evaldata
        import evaluate_refits as common
        import run_refits as legacy
        from readouts import readout, native_development
        from measurement import ARMS, SUPPORT
        torch.set_num_threads(8); torch.set_num_interop_threads(1)
        # Retain the pinned runtime defaults; the full precision check below
        # rejects any difference from the original evaluation before readouts.
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
        actual_banks = {}
        for arm,entry in spec['bank_records'].items():
            check_stop()
            if record(work/entry['worker_path']) != entry['record']: raise ValueError(f'Input bank mismatch: {arm}')
            actual_banks[arm] = entry
        for rel in ('run_spec.json','benchmark.json','population.json'):
            raw=(bundle/'frozen'/rel).read_bytes()
            from artifacts import atomic
            atomic(publisher.root/rel, lambda f, raw=raw:f.write(raw))
        json_save(publisher.root/'provenance.json',{'binding':binding,'source_records':spec['source_records'],
                  'model_record':spec['model_record'],'bank_records':actual_banks,'dtype':'torch.bfloat16',
                  'geometry':[model.n_ut,model.n_physical,model.n_layers,model.d_model], 'runtime':runtime,
                  'actual_worker_source':record(Path(__file__)), 'config_record':record(args.config)})
        stage_status('development_native_checks')
        devitems = old_items(evaluator,model)
        native = native_development(model,devitems)
        torch_save(publisher.root/'development/native.pt',native)
        # Preserve actual evidence even if the development comparison fails.
        if (not torch.equal(native['virtual_states'],native['physical_states'])
                or not torch.equal(native['virtual_states'],native['wrapper_states'])
                or not torch.equal(native['native_logits'],native['unembedded_logits'])):
            raise ValueError('Saved native development evidence fails state/logit equality')
        del native
        # A real tensor save/retrieve/load check precedes all new model states.
        devmanifest = publisher.publish('development')
        publisher.status('awaiting_external_development_acceptance',development_manifest_sha256=devmanifest)
        ackpath = work/'DEVELOPMENT_ACCEPTED.json'
        while not ackpath.exists():
            check_stop(); time.sleep(2)
        ack = json.loads(ackpath.read_text())
        if ack.get('kind') != 'stage_science_authorization' or ack.get('binding') != binding or ack.get('stage_manifest_sha256') != devmanifest or not ack.get('acceptance_receipt_record'):
            raise ValueError('Invalid external development authorization')
        stage_status('confirmation_state_cache')
        population = json.loads((publisher.root/'population.json').read_text())
        rows = population['rows']
        items = []
        for row in rows:
            ids,dropped = evaldata.readout_context(lambda t:model.encode(t)[0].tolist(),row['prompt'],row['target'])
            forms = {name:evaldata.single_token_ids(model.tokenizer,evaldata.surface_forms(name)) for name in row['intermediates']}
            leaked = {name:any(t in ids for t in tokens) for name,tokens in forms.items()}
            if ids != row['token_ids'] or forms != row['intermediate_tokens'] or list(leaked.values()) != row['leaked'] or dropped>1: raise ValueError('Worker tokenization differs from frozen population')
            items.append(evaldata.Item(row['name'],'multihop',row['prompt'],row['target'],row['intermediates'],ids,forms,leaked))
        with common.task_names(evaluator,items,tasks=('multihop',)) as registry, torch.no_grad():
            names = {task:tn.names for task,tn in registry.items()}
            if names != population['names']: raise ValueError('Worker name catalogue differs')
            state_chunks,continuations = [],[]
            for i,item in enumerate(items):
                stage_status(f'confirmation_cache_{i+1}_of_{len(items)}')
                chunk,text = evaluator.cache_states(model,[item],position=-1)
                state_chunks.append(chunk); continuations.extend(text)
            states = torch.cat(state_chunks); del state_chunks
            exits = torch.stack([model.unembed(states[:,model.exit_index(ut)].to(model.input_device)).float() for ut in range(4)],dim=1)
            torch_save(publisher.root/'common/cache.pt',{'H':states,'exit_logits':exits.cpu(),'continuations':continuations})
            for arm in ('raw',*ARMS):
                stage_status(f'readout_{arm}')
                columns = len(SUPPORT[arm]); J=None
                if arm != 'raw':
                    entry = spec['bank_records'][arm]
                    state = torch.load(work/entry['worker_path'],map_location='cpu',weights_only=True,mmap=True)
                    bank = state['J'][arm] if arm in ('sampled_sum','diagonal') else state['J']
                    lens = importlib.import_module('jlens').JacobianLens(bank,n_prompts=100,d_model=2048)
                    target = 190 if arm=='penultimate' else 191
                    plan = common.prepare_readout(model,lens,target_layer=target,source_layers=list(range(target)),evaluator=evaluator)
                    J = plan.jacobians[:columns]
                    del lens,bank,state
                # Keep the frozen original scorer; a per-item wrapper bounds RAM
                # and provides stop boundaries without changing batch geometry.
                outputs,samples = readout(evaluator,model,items,states,J,exits,columns,check_stop=check_stop)
                npz_save(publisher.root/f'readouts/{arm}.npz',outputs)
                torch_save(publisher.root/f'readouts/{arm}_samples.pt',samples)
                publisher.publish(arm)
                del outputs,samples,J
                if arm!='raw': del plan
                gc.collect(); torch.cuda.empty_cache()
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
