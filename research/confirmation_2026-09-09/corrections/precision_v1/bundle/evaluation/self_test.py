#!/usr/bin/env python3
"""Pre-unblinding scorer/alias tests using historical data and synthetic ties."""
import json
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
ROUND = Path(__file__).resolve().parents[1]
OLD = ROUND.parent/'refit_round_2026-09-07'
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROUND.parents[1]), '/home/moloch/ouro_project/src', str(OLD/'deployment')]
import evaluate_controls as legacy
from ouro_jlens import evaluate
from ouro_jlens.evaldata import Item
from measurement import score, paired, PRIMARY_VIRTUAL, secondary_specs
from readouts import readout

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROUND/'evaluation/SELF_TEST.json')
    args=parser.parse_args()
    torch.set_num_threads(2)
    root = OLD/'monitoring/attempt_06/ouro_handoff_20260909T110503Z/results/ouro_evaluation'
    rows = json.loads((root/'common/items.json').read_text())
    names = json.loads((root/'common/task_names.json').read_text())
    idx = [i for i,r in enumerate(rows) if r['task']=='multihop']
    rows = [rows[i] for i in idx]
    results = {}
    for label, path, prefix in [('raw',root/'common/arrays.npz','logitlens_'), ('fit01',root/'fits/fit_01/arrays.npz','jlens_exit3_')]:
        with np.load(path,allow_pickle=False) as a: ranks = a[prefix+'allrank'][idx]
        actual = score(ranks,rows,names,list(range(192)))
        reference = legacy.score_rank_arrays(ranks,rows,names,list(range(192)),bands={'primary':PRIMARY_VIRTUAL})
        for ours,old in [('eligible','eligible'),('own','own_layer'),('control','control_layer'),('excess','delta_layer')]:
            assert np.array_equal(actual[ours],reference[old]), (label,ours)
        results[label] = actual
    discovery = paired(results['fit01'],results['raw'],list(range(192)),list(range(192)),PRIMARY_VIRTUAL)[results['raw']['eligible']].mean()
    assert abs(discovery-0.18906941504940958) < 1e-10, discovery
    # All scores tie. This catches accidental torch.topk use and altered alias
    # aggregation, while keeping the original producer's argmax vs sort rules.
    class Fake:
        input_device = torch.device('cpu'); n_ut=4; n_layers=192
        def unembed(self,x): return torch.zeros((len(x),49152),dtype=torch.float32)
    items = [Item('development-ties','multihop','fixture','answer',['A'],[1],{'A':[1,2]}, {'A':False}),
             Item('development-control','multihop','fixture','answer',['B'],[3],{'B':[3,4]}, {'B':False})]
    old = dict(evaluate.TASK_NAMES)
    try:
        evaluate.TASK_NAMES.clear(); evaluate.TASK_NAMES['multihop'] = evaluate.TaskNames(items,'multihop')
        arrays,samples = readout(evaluate,Fake(),items,torch.zeros(2,192,4),None,torch.zeros(2,4,49152),192)
        direct,kept = evaluate.readout_arrays(Fake(),items,torch.zeros(2,192,4),None,torch.zeros(2,4,49152),3)
        assert all(np.array_equal(arrays[k],v) for k,v in direct.items())
        assert len(samples)==2 and np.all(arrays['top1']==0)
        del kept
    finally:
        evaluate.TASK_NAMES.clear(); evaluate.TASK_NAMES.update(old)
    assert len(secondary_specs())==20
    sys.path.insert(0,str(ROUND/'analysis'))
    from analyze import resample, BOOTSTRAPS
    values=np.asarray([[1.,2.],[1.,2.],[1.,2.],[-1.,2.]])
    boot=resample(values,['a','a','a','b'],seed=991)
    draws=np.random.default_rng(991).integers(2,size=(BOOTSTRAPS,2))
    explicit=np.asarray([np.concatenate([values[:3] if j==0 else values[3:] for j in draw]).mean(axis=0) for draw in draws])
    assert np.array_equal(boot,explicit)
    assert np.all(boot[:,1]==2.)
    evidence = {'schema':'confirmation_measurement_self_test.v1','status':'passed','historical_multihop_items':len(rows),
                'historical_eligible_items':int(results['raw']['eligible'].sum()),'fit01_discovery_primary':float(discovery),
                'legacy_fixed_layer_scores':'bitwise equal for own, control, excess, eligibility on every historical multihop layer',
                'synthetic_tied_logits':'captured argsort top-10 agrees with original ranks; original arrays equal including argmax; two aliases per name',
                'unequal_group_bootstrap':'every draw equals explicit whole-group item-weighted reconstruction; constant paired component preserved',
                'new_confirmation_results_exposed':False}
    from artifacts import record
    evidence['source_records']={str(path):record(path) for path in [Path(__file__),ROUND/'evaluation/measurement.py',ROUND/'evaluation/readouts.py',ROUND/'analysis/analyze.py']}
    dest = args.out
    with dest.open('x') as f: json.dump(evidence,f,indent=2); f.write('\n')
    print(json.dumps(evidence))

if __name__=='__main__': main()
