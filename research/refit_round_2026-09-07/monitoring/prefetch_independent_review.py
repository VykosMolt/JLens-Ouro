"""Focused local-only regressions for prefetch review; never invokes remote methods."""
import fcntl
import hashlib
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('prefetch_independent','/tmp/prefetch_completed_ouro06.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks=[]
body=b'accepted-binary'
expected={'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()}
mtime=100000000000

def file(path,content=body,stamp=mtime):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content);os.utime(path,ns=(stamp,stamp));return path

def reject(action):
    try:action()
    except (ValueError,m.StopPrefetch,FileExistsError):return
    raise AssertionError('expected rejection')

with tempfile.TemporaryDirectory(prefix='prefetch-independent-') as directory:
    root=Path(directory)
    def install(stage,target,recheck=lambda:None):return m.install_verified(stage,target,expected,mtime,root,recheck)
    stage=file(root/'stage1');target=root/'retrieved'/'one.pt'
    assert install(stage,target)=='installed' and target.read_bytes()==body and target.stat().st_mtime_ns==mtime and not stage.exists()
    checks.append('new publication preserves exact bytes and mtime')
    stage=file(root/'stage2');assert install(stage,target)=='already_present' and stage.exists()
    checks.append('matching existing file accepted without overwrite')
    stage=file(root/'stage3');badtime=file(root/'retrieved'/'badtime.pt',stamp=mtime//2)
    reject(lambda:install(stage,badtime));assert badtime.stat().st_mtime_ns==mtime//2 and badtime.read_bytes()==body
    checks.append('existing wrong mtime rejected without rewriting')
    stage=file(root/'stage4');badbytes=file(root/'retrieved'/'badbytes.pt',b'wrong')
    reject(lambda:install(stage,badbytes));assert badbytes.read_bytes()==b'wrong'
    checks.append('existing wrong bytes rejected without overwriting')
    stage=file(root/'stage5');locked=root/'retrieved'/'locked.pt'
    with (root/'.sync.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reject(lambda:install(stage,locked));assert not locked.exists() and stage.exists()
    checks.append('final collector lock prevents publication')
    stage=file(root/'stage6');collision=root/'retrieved'/'collision.pt';real_link=os.link
    def collision_link(source,destination,**kwargs):
        Path(destination).write_bytes(b'concurrent-writer');return real_link(source,destination,**kwargs)
    with patch.object(m.os,'link',collision_link):reject(lambda:install(stage,collision))
    assert collision.read_bytes()==b'concurrent-writer' and stage.exists()
    checks.append('atomic link collision never overwrites concurrent target')
    stage=file(root/'stage7');changed=root/'retrieved'/'changed.pt'
    reject(lambda:install(stage,changed,lambda:stage.write_bytes(b'changed')));assert not changed.exists()
    checks.append('stage change during final recheck rejected')
    stage=file(root/'stage8');stopped=root/'retrieved'/'stopped.pt'
    def stop():raise m.StopPrefetch('fit ended')
    reject(lambda:install(stage,stopped,stop));assert not stopped.exists()
    checks.append('phase stop before publication leaves target absent')

    for label,leader_exits in [('exited leader',True),('live leader',False)]:
        childfile=root/(label.replace(' ','_')+'.pid')
        script='import subprocess,sys,time;from pathlib import Path;c=subprocess.Popen([sys.executable,"-c","import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)"]);Path(sys.argv[1]).write_text(str(c.pid));'+('' if leader_exits else 'time.sleep(60)')
        leader=subprocess.Popen([sys.executable,'-B','-c',script,str(childfile)],start_new_session=True)
        deadline=time.monotonic()+5
        while not childfile.exists() and time.monotonic()<deadline:time.sleep(.01)
        assert childfile.exists()
        child=int(childfile.read_text())
        if leader_exits:leader.wait(timeout=5)
        time.sleep(.05)
        try:
            start=time.monotonic();m.stop_process(leader);elapsed=time.monotonic()-start
            assert elapsed<12
            statfile=Path('/proc')/str(child)/'stat'
            if statfile.exists():assert statfile.read_text().split()[2]=='Z',('active surviving descendant',label)
            assert leader.poll() is not None
            checks.append('cancellation cleans group with '+label)
        finally:
            try:os.killpg(leader.pid,signal.SIGKILL)
            except ProcessLookupError:pass

print({'status':'passed','count':len(checks),'checks':checks,'source_sha256':hashlib.sha256(Path('/tmp/prefetch_completed_ouro06.py').read_bytes()).hexdigest()})
