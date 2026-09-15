from pathlib import Path
import json,torch,numpy as np
R=Path('/home/moloch/ouro_project/jacobian-lens/research/confirmation_2026-09-09')
paths=json.loads((R/'resources/materialized_inputs.json').read_text())['banks']
cache=R.parent/'refit_round_2026-09-07/monitoring/attempt_06/ouro_handoff_20260909T110503Z/results/ouro_evaluation/common/cache.pt'
C=torch.load(cache,map_location='cpu',mmap=True,weights_only=True);print('cache keys',list(C),flush=True)
H=C['H']; results=[]
for name,e in paths.items():
 if not name.startswith('ouro/fit_01/'):continue
 state=torch.load(e['path'],map_location='cpu',mmap=True,weights_only=True)
 for v in [0,47,95,143,170,176,181,189,190]:
  J=state['J'][v].float();h=H[:3,v];out=torch.einsum('de,ie->id',J,h).numpy();exact=J.numpy().astype(np.float64)@h.numpy().astype(np.float64).T;exact=exact.T
  passed=np.allclose(out,exact,rtol=2e-5,atol=2e-5)
  results.append(dict(v=v,passed=passed,max_abs=float(np.abs(out-exact).max()),bad=int((np.abs(out-exact)>(2e-5+2e-5*np.abs(exact))).sum())))
print(json.dumps(results));assert all(x['passed'] for x in results)
Path('/tmp/confirmation_transport_probe.json').write_text(json.dumps({'status':'passed','historical_only':True,'results':results},indent=2))
