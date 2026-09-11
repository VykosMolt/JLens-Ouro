from pathlib import Path
import sys,subprocess,shlex,json,hashlib,base64,datetime
p=Path('research/refit_round_2026-09-07/deployment').resolve();sys.path.insert(0,str(p));import lease
root=lease.LEDGER/'attempt_06';s=lease.io._json(root/'LEASE.json')
script="""from pathlib import Path
import json,hashlib,base64
r=Path('/workspace/jlens/results');paths=sorted((r/'preflight_ouro').glob('*.json'))+sorted((r/'preflight_huginn').glob('*.json'))
paths += [r/n for n in ['initial_budget_projection.json','huginn_budget_projection.json'] if (r/n).is_file()]
rows=[]
for p in paths:
 assert p.is_file() and not p.is_symlink() and p.stat().st_size<16000000
 b=p.read_bytes();rows.append({'path':str(p.relative_to(r)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'data':base64.b64encode(b).decode()})
print(json.dumps(rows))
"""
q=subprocess.run([*lease.ssh_options(root,s),'root@'+s['ssh']['host'],'python -c '+shlex.quote(script)],capture_output=True,text=True,check=True,timeout=45)
rows=json.loads(q.stdout);dest=p.parent/'monitoring/attempt_06'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');dest.mkdir(parents=True,exist_ok=False)
for row in rows:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 b=base64.b64decode(row.pop('data'),validate=True);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
 target=dest/'results'/rel;target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('xb') as f: f.write(b)
lease.io._new_json(dest/'COPY_VERIFIED.json',{'status':'passed','lease_name':s['name'],'pod_id':s['pod_id'],'files':rows,'scope':'Interim immutable preflight JSON and budget records only; not final experiment retrieval.'})
print(json.dumps({'directory':str(dest),'files':rows},indent=2))
