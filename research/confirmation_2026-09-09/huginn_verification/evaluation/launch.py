#!/usr/bin/env python3
"""Upload frozen inputs and acknowledge locally accepted development evidence.

This command never creates or deletes a provider resource; the reviewed lease
controller owns those actions. All SSH commands target its exact bound worker.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
import time
from artifacts import json_save, record

ROUND=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROUND/'controller'))
import lease
import artifact_handoff as handoff
from transport import run_bounded


def main():
    p=argparse.ArgumentParser();p.add_argument('--lease',type=Path,required=True)
    p.add_argument('--phase',choices=('start','accept-development'),required=True)
    args=p.parse_args();root=args.lease.resolve();state=json.loads((root/'LEASE.json').read_text())
    binding=handoff.binding(state);handoff.verify_local_sources(state)
    if state['status']=='terminated' or state.get('halt_requested'):raise ValueError('Worker is no longer admitted')
    if not state.get('ssh'):raise ValueError('Controller has not bound an SSH endpoint')
    commands=lease.ssh_options(root,state);host='root@'+state['ssh']['host']
    def remote(command,timeout=60):
        return run_bounded([*commands,host,command],timeout=timeout,operation='confirmation_remote').stdout
    if remote('cat /workspace/jlens/provider_lease_name').decode().strip()!=state['name']:
        raise ValueError('SSH endpoint does not identify the owned lease')
    local=root/'launch';local.mkdir(exist_ok=True)
    def upload(path,dest,timeout):
        run_bounded(['rsync','--partial','--partial-dir=.rsync-partial','--protect-args','-t','-e',shlex.join(commands),str(path),host+':'+dest],
                    timeout=timeout,operation='confirmation_upload')
    if args.phase=='start':
        specpath=Path(state['run_spec_path']);spec=json.loads(specpath.read_text())
        config={'work':'/workspace/jlens','bundle':'/workspace/jlens/bundle','snapshots':{'ouro':'/workspace/jlens/models/ouro','huginn':'/workspace/jlens/models/huginn'},
                'binding':binding,'work_deadline_utc':state['work_deadline_utc'],'setup_deadline_utc':state['setup_deadline_utc']}
        path=local/'worker_config.json'
        if path.exists():
            if json.loads(path.read_text())!=config:raise ValueError('Changed local launch configuration')
        else:json_save(path,config)
        remaining=lambda:max(1,int(lease.epoch(state['setup_deadline_utc'])-time.time()))
        if remaining()<=300:raise ValueError('Insufficient admitted setup time')
        handoff.verify_record(Path(state['bundle']),state['bundle_record'])
        started=local/'BOOTSTRAP_STARTED.json'
        if not started.exists():
            upload(Path(state['bundle']),'/workspace/jlens/bundle.tar.gz',min(1800,remaining()-300))
            upload(path,'/workspace/jlens/worker_config.json',min(60,remaining()))
            remote('test ! -e /workspace/jlens/bundle && tar -xzf /workspace/jlens/bundle.tar.gz -C /workspace/jlens')
            # Bootstrap stdout lives outside contracted immutable payloads.
            pid=remote('nohup python /workspace/jlens/bundle/evaluation/bootstrap.py --config /workspace/jlens/worker_config.json > /workspace/jlens/bootstrap_stdout.log 2>&1 < /dev/null & echo $!').decode().strip()
            if not pid.isdigit():raise ValueError('Bootstrap PID was not returned')
            json_save(started,{'binding':binding,'worker_config_record':record(path),'bootstrap_pid':int(pid),
                               'started_utc':datetime.now(timezone.utc).isoformat(),'bundle_record':state['bundle_record'],
                               'deployed_controller_sources':state['controller_sources'],'worker_source':spec['source_records']['evaluation/worker.py']})
        elif json.loads(started.read_text())['binding']!=binding:raise ValueError('Stale bootstrap record')
        print(json.dumps({'status':'launched','binding':binding,'deployment_record':str(started)}))
    else:
        candidates=[v['manifest'] for v in state.get('known_manifests',{}).values()
                    if v['manifest'].get('kind')=='stage' and v['manifest'].get('stage_id')=='development']
        if len(candidates)!=1:raise ValueError('Exact development manifest is not available')
        manifest=candidates[0];pointer=handoff.acceptance_pointer(root,state,manifest,rehash=True)
        if state['stage_acceptances'].get(handoff.digest(manifest))!=pointer:raise ValueError('Development receipt is not durably accepted')
        ack={'kind':'stage_science_authorization','binding':binding,'stage_manifest_sha256':handoff.digest(manifest),
             'acceptance_receipt_record':pointer['record']}
        path=local/'DEVELOPMENT_ACCEPTED.json'
        if path.exists():
            if json.loads(path.read_text())!=ack:raise ValueError('Existing science authorization differs')
        else:json_save(path,ack)
        upload(path,'/workspace/jlens/DEVELOPMENT_ACCEPTED.json.upload',60)
        remote('mv /workspace/jlens/DEVELOPMENT_ACCEPTED.json.upload /workspace/jlens/DEVELOPMENT_ACCEPTED.json')
        print(json.dumps({'status':'development_accepted_and_huginn_n100_authorized','manifest_sha256':handoff.digest(manifest),'local_receipt':pointer}))

if __name__=='__main__':main()
