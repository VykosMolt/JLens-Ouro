#!/usr/bin/env python3
"""Install the original pinned runtime, then launch only the frozen evaluator."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from artifacts import json_save, record

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    args=p.parse_args(); config=json.loads(args.config.read_text())
    work,bundle=Path(config['work']),Path(config['bundle'])
    binding=config['binding']; spec=json.loads((bundle/'frozen/run_spec.json').read_text())
    if record(bundle/'frozen/run_spec.json')['sha256']!=binding['run_spec_sha256']:raise ValueError('Wrong setup specification')
    for rel,expected in spec['source_records'].items():
        if record(bundle/rel)!=expected:raise ValueError(f'Wrong setup source: {rel}')
    deadline=datetime.fromisoformat(config['setup_deadline_utc'].replace('Z','+00:00')).timestamp()
    env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'0',
         'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','CUBLAS_WORKSPACE_CONFIG':':4096:8',
         'HF_HOME':str(work/'runtime_cache/huggingface'),'PYTORCH_ALLOC_CONF':'expandable_segments:True',
         'HF_HUB_DISABLE_TELEMETRY':'1','TOKENIZERS_PARALLELISM':'false'}
    if any(k in env for k in ('RUNPOD_API_KEY','HF_TOKEN','HUGGING_FACE_HUB_TOKEN')):raise ValueError('Unexpected worker credential')
    stop=False
    def handler(signum,frame):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
    def status(phase, **details):
        json_save(work/'status.json',{'binding':binding,'lease_name':binding['lease_name'],'setup_complete':False,
                                    'phase':phase,'updated_utc':datetime.now(timezone.utc).isoformat(),**details},replace=True)
    def check():
        if stop or (work/'STOP').exists() or time.time()>=deadline:raise InterruptedError('Setup deadline or early stop')
    def run(label, command):
        check();status(label)
        log=work/'setup_logs'/f'{label}.log';log.parent.mkdir(exist_ok=True)
        with log.open('xb') as output:
            child=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            try:
                while child.poll() is None:
                    check();time.sleep(2)
                if child.returncode:raise RuntimeError(f'{label} exited {child.returncode}')
            finally:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
                # Reap any same-group descendant even when its parent exited.
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
    try:
        python=work/'venv/bin/python'
        run('venv',[sys.executable,'-m','venv',str(work/'venv')])
        run('packages',[str(python),'-m','pip','install','--no-deps','--no-cache-dir','--require-hashes','--only-binary=:all:','-r',str(bundle/'repo/research/refit_round_2026-09-07/deployment/requirements.lock')])
        deployment=bundle/'repo/research/refit_round_2026-09-07/deployment'
        for model_name, manifest in (('huginn','huginn_model_manifest.json'),('ouro','model_manifest.json')):
            run('model_'+model_name,[str(python),str(bundle/'evaluation/download_model.py'),'--manifest',str(deployment/manifest),'--out',str(work/'models'/model_name)])
        json_save(work/'status.json',{'binding':binding,'lease_name':binding['lease_name'],'setup_complete':True,'phase':'setup_complete'},replace=True)
        os.execve(str(python),[str(python),str(bundle/'evaluation/worker.py'),'--config',str(args.config)],env)
    except BaseException as error:
        status('setup_stopped' if isinstance(error,InterruptedError) else 'setup_failed',error_type=type(error).__name__,error=str(error))
        raise

if __name__=='__main__':main()
