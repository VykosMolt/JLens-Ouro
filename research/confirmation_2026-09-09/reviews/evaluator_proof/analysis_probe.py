from pathlib import Path
import sys,tempfile,json
import numpy as np
R=Path('/home/moloch/jacobian-lens/research/confirmation_2026-09-09');sys.path[:0]=[str(R/'analysis'),str(R/'evaluation')]
from analyze import analyze
from measurement import SUPPORT,PRIMARY_VIRTUAL
from plot import main
P=Path(tempfile.mkdtemp(prefix='confirmation-analysis-proof-'));(P/'readouts').mkdir()
rows=[dict(task='multihop',eligible=[True],own_index=[0,-1,-1],control_indices=[[1]],dependency_group_id=g) for g in ['A','A','B']]
(P/'population.json').write_text(json.dumps({'rows':rows,'names':{'multihop':['A','B']}}))
for arm,support in SUPPORT.items():
 a=np.full((3,128,len(support)),-1,np.int32);a[:,0]=1;a[:,1]=100
 if arm=='fit01':a[0,0,PRIMARY_VIRTUAL]=100
 np.savez(P/f'readouts/{arm}.npz',allrank=a)
result,arrays=analyze(P)
assert np.array_equal(arrays['primary'],[-1.,0.,0.])
assert np.allclose(result['estimates'],[-1/3,2/3,0,1,0,-1/3,0])
assert len(result['secondary_family'])==20 and result['groups']==2
(P/'analysis.json').write_text(json.dumps(result,allow_nan=False))
sys.argv=['plot','--analysis',str(P/'analysis.json'),'--out',str(P/'figures')];main()
assert len(list((P/'figures').iterdir()))==6
proof={'status':'passed','new_confirmation_results_exposed':False,'synthetic_primary':result['estimates'][0],'primary_item_values':arrays['primary'].tolist(),'groups':2,'secondary_contrasts':20,'static_figures':6,'root':str(P)}
(P/'PROOF.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
