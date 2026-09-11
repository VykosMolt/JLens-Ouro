"""Local-only checks of V2 failure precedence and bounded retry state."""
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('prefetch_v2_independent','/tmp/prefetch_completed_ouro06_v2.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks=[]
with tempfile.TemporaryDirectory(prefix='prefetch-v2-independent-') as d:
 root=Path(d)
 for operation,code in [('snapshot',1),('transfer',23),('transfer',20)]:
  tag=operation+'_'+str(code)
  go=root/(tag+'_go')
  script='from pathlib import Path;import sys,time;p=Path(sys.argv[1]);\nwhile not p.exists():time.sleep(.001)\nsys.exit(int(sys.argv[2]))'
  def guard():
   go.write_text('exit');time.sleep(.08)
   raise m.TransportFailure('status','simulated status SSH timeout')
  error=None
  try:m.run_polled([sys.executable,'-B','-c',script,str(go),str(code)],root/(tag+'.out'),root/(tag+'.err'),guard,2,operation=operation)
  except Exception as e:error=e
  proof=json.loads((root/(tag+'.err.process.json')).read_text())
  assert isinstance(error,subprocess.CalledProcessError) and error.returncode==code and proof['before_cleanup_returncode']==code and proof['process_group_quiescent'] is True
  checks.append(operation+' pre-cleanup nontransport exit '+str(code)+' wins over guard transport timeout')
 def timeout_guard():raise m.TransportFailure('status','simulated status timeout')
 error=None
 try:m.run_polled([sys.executable,'-B','-c','import time;time.sleep(60)'],root/'cancel.out',root/'cancel.err',timeout_guard,2,operation='transfer')
 except Exception as e:error=e
 proof=json.loads((root/'cancel.err.process.json').read_text())
 assert isinstance(error,m.TransportFailure) and proof['before_cleanup_returncode'] is None and proof['process_group_quiescent'] is True
 checks.append('cancellation-generated signal preserves genuine transport retry')
 def bad_guard():raise ValueError('integrity failure')
 error=None
 try:m.run_polled([sys.executable,'-B','-c','import time;time.sleep(60)'],root/'integrity.out',root/'integrity.err',bad_guard,2,operation='transfer')
 except Exception as e:error=e
 assert isinstance(error,ValueError) and not isinstance(error,m.TransportFailure)
 checks.append('local integrity failure is never transport retryable')

 for operation,code,retry in [('snapshot',1,False),('transfer',20,True)]:
  go=root/('during_'+operation+'_go')
  script='from pathlib import Path;import sys,time;p=Path(sys.argv[1]);\nwhile not p.exists():time.sleep(.001)\nsys.exit(int(sys.argv[2]))'
  def finish_during_cleanup(process):
   go.write_text('exit');process.wait(timeout=3)
  error=None
  with patch.object(m,'stop_process',finish_during_cleanup):
   try:m.run_polled([sys.executable,'-B','-c',script,str(go),str(code)],root/('during_'+operation+'.out'),root/('during_'+operation+'.err'),timeout_guard,2,operation=operation)
   except Exception as e:error=e
  proof=json.loads((root/('during_'+operation+'.err.process.json')).read_text())
  assert proof['before_cleanup_returncode'] is None and proof['returncode']==code
  assert isinstance(error,m.TransportFailure) is retry
  if not retry:assert isinstance(error,subprocess.CalledProcessError) and error.returncode==code
  checks.append('during-cleanup semantic exit remains terminal' if not retry else 'rsync cancellation exit20 remains transport retryable')

 class Fake(m.Prefetch):
  def __init__(self,folder,errors):
   folder.mkdir();super().__init__({'binaries':{},'metadata':{},'consumed':{}},folder);self.counter=1;self.errors=errors;self.calls=[]
  def attempt_file(self,relative,expected):
   self.calls.append(self.file_deadline)
   if self.errors:
    error=self.errors.pop(0)
    if error:raise error
   return {'path':relative}
 runner=Fake(root/'retry5',[m.TransportFailure('status','t') for _ in range(4)])
 result=runner.run_file('x',{'bytes':1,'sha256':'x'})
 assert result=={'path':'x'} and len(runner.calls)==5 and len(set(runner.calls))==1
 checks.append('five attempts retain one original absolute deadline')
 runner=Fake(root/'exhausted',[m.TransportFailure('status','t') for _ in range(6)])
 error=None
 try:runner.run_file('x',{'bytes':1,'sha256':'x'})
 except Exception as e:error=e
 assert isinstance(error,m.TransportFailure) and len(runner.calls)==5
 checks.append('sixth attempt is forbidden')
 runner=Fake(root/'semantic',[subprocess.CalledProcessError(1,['synthetic'])])
 error=None
 try:runner.run_file('x',{'bytes':1,'sha256':'x'})
 except Exception as e:error=e
 assert isinstance(error,subprocess.CalledProcessError) and len(runner.calls)==1
 checks.append('semantic failure never starts second attempt')
 class Expired(Fake):
  def attempt_file(self,relative,expected):
   self.calls.append(self.file_deadline);self.file_deadline=time.monotonic()-1
   raise m.TransportFailure('status','t')
 runner=Expired(root/'deadline',[])
 error=None
 try:runner.run_file('x',{'bytes':1,'sha256':'x'})
 except Exception as e:error=e
 assert isinstance(error,m.TransportFailure) and len(runner.calls)==1
 checks.append('expired original deadline forbids transport retry')
print(json.dumps({'status':'passed','checks':checks,'count':len(checks),'helper_sha256':hashlib.sha256(Path('/tmp/prefetch_completed_ouro06_v2.py').read_bytes()).hexdigest()}))
