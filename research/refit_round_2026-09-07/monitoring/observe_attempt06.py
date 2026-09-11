from pathlib import Path
import sys,subprocess,shlex
p=Path('research/refit_round_2026-09-07/deployment').resolve();sys.path.insert(0,str(p));import lease
root=lease.LEDGER/'attempt_06';state=lease.io._json(root/'LEASE.json')
print('local_watch', {k:state.get(k) for k in ['status','watch_ready_utc','watch_heartbeat_utc','last_poll_utc','watch_last_error_type','observed_balance','observed_combined_spend_upper_usd']})
if state['status']=='terminated': raise SystemExit(0)
script="""from pathlib import Path
import json,re,subprocess
root=Path('/workspace/jlens'); s=json.loads((root/'status.json').read_text());print('worker',json.dumps(s))
phase=s.get('phase',''); names=['worker.log']
if re.fullmatch('[a-zA-Z0-9_]+',phase): names.append('results/logs/'+phase+'.log')
for name in names:
 p=root/name
 if p.exists():
  with p.open('rb') as f: f.seek(max(0,p.stat().st_size-5000));data=f.read().decode(errors='replace')
  lines=[x for x in data.splitlines() if x.strip()]
  print(name,p.stat().st_size,'\\n'.join(lines[-3:]))
for stage in ['preflight_ouro','preflight_huginn']:
 p=root/'results'/stage
 if p.exists(): print(stage,sorted(x.name for x in p.glob('*.json')))
print('gpu',subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip())
"""
result=subprocess.run([*lease.ssh_options(root,state),'root@'+state['ssh']['host'],'python -c '+shlex.quote(script)],text=True,capture_output=True,timeout=45,check=True)
print(result.stdout);print(result.stderr)
