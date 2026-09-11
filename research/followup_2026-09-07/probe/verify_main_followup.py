"""Independent checks of main follow-up exits/correctness; no root helpers imported."""
from pathlib import Path
import csv
import hashlib
import json
import re

import numpy as np

HERE = Path(__file__).resolve().parent
FOLLOW = HERE.parent
RESULTS = FOLLOW / 'results'
ART = Path('/home/moloch/ouro_project/artifacts/jlens')
MAIN = ART / 'retrieved/jlens-b300-20260905-0359/eval/n100_exit3'
RUNS = {'-1': ART/'eval/b300_local_allexits_strict',
        '-2': ART/'eval/b300_local_allexits_pos-2_strict'}
B = 20000
OP = {'addition', 'subtraction', 'multiplication', 'division', 'mod', 'squared'}
NUM = {s:n for n,s in enumerate('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split())}
COUNTER = {'means':0, 'intervals':0, 'structural':0}
MAXERR = {}


def read(path): return json.loads(Path(path).read_text())


def check(label, got, expected, category='means', tol=1e-10):
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape, (label, got.shape, expected.shape)
    err = float(np.max(np.abs(got-expected),initial=0))
    MAXERR[category] = max(MAXERR.get(category,0),err)
    assert err <= tol, (label, err, got, expected)
    COUNTER[category] += int(got.size)


def iid_counts(n, seed):
    rows = np.random.default_rng(seed).integers(n,size=(B,n))
    return np.stack([np.bincount(row,minlength=n) for row in rows])


def interval(draws): return np.quantile(draws,[.025,.975],axis=0).T


def stats(x, counts):
    x=np.asarray(x)
    flat=x.reshape(len(x),-1)
    draws=counts@flat/counts.sum(1)[:,None]
    return x.mean(0), interval(draws).reshape(*x.shape[1:],2), draws


def load(run): return read(run/'items.json'),dict(np.load(run/'arrays.npz'))


def loop3_final_contrasts(a,counts):
    ix=np.arange(136,143);truth=a['exit_top1'][:,-1,None]
    result={}
    for name,j,l in [
        ('top1',(a['jlens_exit3_top1'][:,ix]==truth).mean(1),(a['logitlens_top1'][:,ix]==truth).mean(1)),
        ('actual_final_token_hit10',(a['jlens_exit3_rank_of_final_top1'][:,ix]<10).mean(1),
         (a['logitlens_rank_of_final_top1'][:,ix]<10).mean(1))]:
        point,ci,_=stats(j-l,counts)
        result[name]={'final_J':float(j.mean()),'logit_lens':float(l.mean()),
                      'paired_difference':float(point),'paired_ci95':ci.tolist(),
                      'n_items':148,'loop':3,'physical_layers':[41,47],
                      'draws':B,'seed':20260907}
    return result


def eligible(item):
    return [j for j,(name,scorable,leaked) in enumerate(zip(item['intermediates'],item['scorable'],item['leaked']))
            if scorable and not leaked and (item['task']=='multihop' or name not in OP)]


def canonical(name):
    s=name.lower()
    return str(NUM[s] if s in NUM else 3 if s=='third' else int(s) if s.isdigit() else s)


def components(items):
    concepts=[{canonical(it['intermediates'][s]) for s in eligible(it)} for it in items]
    remaining=set(range(len(items))); groups=[]
    while remaining:
        g={min(remaining)}; labels=set(concepts[min(g)])
        while True:
            additions={i for i in remaining-g if concepts[i]&labels}
            if not additions:break
            g |= additions
            for i in additions: labels |= concepts[i]
        groups.append(g);remaining -= g
    return groups


def independently_score(items,names,a,task):
    idx=[i for i,it in enumerate(items) if it['task']==task and eligible(it)]
    scored={}
    for prefix in ('jlens_exit3','logitlens'):
        parts={key:[] for key in ('hit','control','excess','any_hit','any_control','any_excess')}
        for i in idx:
            it=items[i]; own=set(j for j in it['own_index'] if j>=0)
            slot_parts={key:[] for key in parts}
            for slot in eligible(it):
                term=it['intermediates'][slot]
                controls=[j for j,name in enumerate(names[task]) if j not in own and ((name in OP)==(term in OP))]
                true=(a[prefix+'_rank'][i,slot].reshape(4,48)<10)
                false=(a[prefix+'_allrank'][i,controls].reshape(len(controls),4,48)<10)
                values={'hit':true.astype(float),'control':false.mean(0),
                        'any_hit':np.max(true,axis=1).astype(float),
                        'any_control':np.mean(np.max(false,axis=2),axis=0)}
                values['excess']=values['hit']-values['control']
                values['any_excess']=values['any_hit']-values['any_control']
                for k,v in values.items():slot_parts[k].append(v)
            for k,v in slot_parts.items():parts[k].append(np.mean(v,axis=0))
        scored[prefix]={k:np.stack(v) for k,v in parts.items()}
    return np.array(idx),scored


def classify(it):
    answer=it['continuation'].strip().strip('"').lower()
    target=it['target'].strip().lower()
    boundary=lambda text:answer[:len(text)]==text and (len(answer)==len(text) or not answer[len(text)].isalnum())
    historical=boundary(target)
    numeric=bool(re.fullmatch('[+-]?[0-9]+',target))
    remainder=answer[len(target):]
    decimal_extension=bool(re.match(r'[.,/][0-9]|[eE][+-]?[0-9]',remainder))
    strict=historical and not(numeric and decimal_extension)
    equiv=strict
    if target in NUM:
        spelling=str(NUM[target]);rest=answer[len(spelling):]
        equiv |= boundary(spelling) and not bool(re.match(r'[.,/][0-9]|[eE][+-]?[0-9]',rest))
    return bool(historical),bool(strict),bool(equiv)


def verify_correctness():
    items,a=load(MAIN);names=read(MAIN/'task_names.json')
    reported=read(RESULTS/'correctness.json'); layer=read(RESULTS/'layerwise.json')
    criteria=('historical','strict_numeric','numeric_equivalence')
    masks=np.array([classify(it) for it in items])
    check('overall passing',masks.sum(0),[72,70,71],'structural')
    changed=[(it['name'],it['target'],it['continuation'],list(map(bool,masks[i])))
             for i,it in enumerate(items) if len(set(masks[i]))>1]
    assert [it['correct'] for it in items]==masks[:,0].tolist()
    interactions={}; score_cache={}
    for task in ('multihop','order-ops'):
        idx,scores=independently_score(items,names,a,task);score_cache[task]=(idx,scores)
        assert len(idx)==(90 if task=='multihop' else 51)
        # Verify the entire partition independently as connected components;
        # only then reuse its arbitrary published component numbering so the
        # identical RNG seed addresses corresponding components.
        own_groups={frozenset(g) for g in components([items[i] for i in idx])}
        published_labels=np.array(layer['tasks'][task]['concept_clusters'])
        root_groups={frozenset(np.flatnonzero(published_labels==g)) for g in set(published_labels)}
        assert own_groups==root_groups
        group_ids=np.unique(published_labels)
        group_members=[published_labels==g for g in group_ids]
        cc=iid_counts(len(group_ids),19)
        for c,criterion in enumerate(criteria):
            mask=masks[idx,c];r=reported['criteria'][criterion][task]
            check('strata counts', [sum(mask),sum(~mask)],[r['n_passing'],r['n_failing']],'structural')
            iid_draws={}; cluster_draws={}; point={}
            for label,keep,seed in [('passing',mask,0),('failing',~mask,1)]:
                counts=iid_counts(sum(keep),seed);rr=r['strata'][label]
                for method,parts in scores.items():
                    for key,v in parts.items():
                        check(f'{criterion}/{task}/{label}/{method}/{key}',v[keep].mean(0),rr['methods'][method][key]['mean'])
                        if key.startswith('any'):
                            mean,ci,draw=stats(v[keep],counts)
                            check('method interval',ci,rr['methods'][method][key]['ci95'],'intervals')
                for key in ('excess','any_excess'):
                    delta=scores['jlens_exit3'][key]-scores['logitlens'][key]
                    check('paired difference mean',delta[keep].mean(0),rr[key+'_difference']['mean'])
                    if key=='any_excess':
                        mean,ci,draw=stats(delta[keep],counts)
                        check('paired iid CI',ci,rr[key+'_difference']['ci95'],'intervals')
                        iid_draws[label]=draw;point[label]=mean
                        gs=np.stack([delta[g&keep].sum(0) for g in group_members])
                        gn=np.array([sum(g&keep) for g in group_members])
                        assert (cc@gn>0).all()
                        cd=cc@gs/(cc@gn)[:,None]
                        check('cluster paired CI',interval(cd),rr[key+'_difference_cluster']['ci95'],'intervals')
                        cluster_draws[label]=cd
            interaction=point['passing']-point['failing']
            rr=r['interaction_passing_minus_failing']
            check('interaction mean',interaction,rr['mean'])
            check('interaction iid CI',interval(iid_draws['passing']-iid_draws['failing']),rr['item_ci95'],'intervals')
            check('interaction cluster CI',interval(cluster_draws['passing']-cluster_draws['failing']),rr['cluster_ci95'],'intervals')
            interactions[criterion+'/'+task]=rr
    # Check the CSV against independently reconstructed item-level quantities.
    for row in csv.DictReader((RESULTS/'correctness.csv').open()):
        idx,scores=score_cache[row['task']];mask=masks[idx,criteria.index(row['criterion'])]
        keep=mask if row['stratum']=='passing' else ~mask;u=int(row['loop'])-1
        for method in scores:
            for key in ('any_hit','any_control','any_excess'):
                check('CSV stratum mean',scores[method][key][keep,u].mean(),float(row[method+'_'+key]))
        delta=scores['jlens_exit3']['any_excess']-scores['logitlens']['any_excess']
        check('CSV paired difference',delta[keep,u].mean(),float(row['difference']))
    return changed,interactions


def verify_exits():
    main_items,pod=load(MAIN);reported=read(RESULTS/'exit_agreement.json')
    matrices={(r['position'],int(r['loop']),r['layers'],r['estimator_a'],r['estimator_b']):r
              for r in csv.DictReader((RESULTS/'exit_agreement_matrix.csv').open())}
    diagnostics={(r['position'],int(r['loop']),r['layers'],r['lens'],r['target']):r
                 for r in csv.DictReader((RESULTS/'exit_diagnostics.csv').open())}
    labels=['Local J-Lens','Final J-Lens','Logit lens','Actual loop exit','Actual final exit']
    windows={'1-47':np.arange(47),'41-47':np.arange(40,47),'41-48':np.arange(40,48),'48':np.array([47])}
    summaries={}; count=iid_counts(148,20260907)
    for position,run in RUNS.items():
        items,a=load(run);assert len(items)==148
        assert [it['name'] for it in items]==[it['name'] for it in main_items]
        truth=a['exit_top1'];assert truth.shape==(148,4)
        assert np.array_equal(truth,a['logitlens_top1'][:,[47,95,143,191]])
        for prefix in ('jlens_exit3','logitlens'):
            for metric in ('kl_to_','rank_of_'):
                end='top1' if metric=='rank_of_' else ''
                local=prefix+'_'+metric+'local'+('_'+end if end else '')
                final=prefix+'_'+metric+'final'+('_'+end if end else '')
                assert np.array_equal(a[local],a[final]),'target_ut interpretation changed'
        actual=(truth[:,:,None]==truth[:,None,:]).astype(float)
        mean,ci,_=stats(actual,count)
        check('actual exit matrix',mean,reported[position]['actual_exit_agreement']['mean'])
        check('actual exit matrix CI',ci,reported[position]['actual_exit_agreement']['ci95'],'intervals')
        per_item_matrices=[]; loopKL=[]
        for u in range(4):
            ll=a['logitlens_top1'][:,u*48:(u+1)*48]
            jl=a['jlens_exit3_top1'][:,u*48:(u+1)*48]
            local=a[f'jlens_exit{u}_top1'][:,u*48:(u+1)*48]
            assert np.array_equal(local[:,-1],truth[:,u])
            assert np.all(a[f'jlens_exit{u}_kl_to_local'][:,u*48+47]==0)
            top=np.stack([local,jl,ll,np.repeat(truth[:,u,None],48,axis=1),np.repeat(truth[:,-1,None],48,axis=1)],axis=2)
            same=top[:,:,:,None]==top[:,:,None,:]
            for window,ix in windows.items():
                per_item=same[:,ix].mean(1);mean,ci,_=stats(per_item,count)
                rr=reported[position]['loops'][str(u+1)]['windows'][window]['top1_agreement_matrix']
                check('lens agreement matrix',mean,rr['mean'])
                check('lens agreement matrix interval',ci,rr['ci95'],'intervals')
                for j,name in enumerate(labels):
                    for k,other in enumerate(labels):
                        row=matrices[position,u+1,window,name,other]
                        check('matrix CSV',mean[j,k],float(row['agreement']))
                for j,prefix in enumerate([f'jlens_exit{u}','jlens_exit3','logitlens']):
                    for target,ref in [('local','actual_current_exit'),('final','actual_final_exit')]:
                        row=diagnostics[position,u+1,window,labels[j],ref]
                        retained=not(target=='local' and j>0 and u<3)
                        assert (row['rank_kl_retained']=='True')==retained
                        check('diagnostic top1',mean[j,3 if target=='local' else 4],float(row['top1_agreement']))
                        if not retained:
                            assert all(row[k]=='' for k in ('actual_token_hit10','actual_token_mrr','kl_lens_to_actual'))
                            continue
                        ranks=a[prefix+'_rank_of_'+target+'_top1'][:,u*48+ix]
                        kl=a[prefix+'_kl_to_'+target][:,u*48+ix]
                        values={'actual_token_hit10':(ranks<10).mean(1),
                                'actual_token_hit100':(ranks<100).mean(1),
                                'actual_token_mrr':(1/(ranks+1)).mean(1),
                                'actual_token_mean_log10_rank':np.log10(ranks+1).mean(1),
                                'kl_lens_to_actual':kl.mean(1)}
                        for metric,v in values.items():
                            check('exit diagnostic '+metric,v.mean(),float(row[metric]),tol=2e-5)
                            _,metric_ci,_=stats(v,count)
                            check('exit diagnostic interval',metric_ci,[float(row[metric+'_ci_low']),float(row[metric+'_ci_high'])],'intervals',tol=2e-5)
            if u<3:
                per_item_matrices.append(same[:,40:47].mean(1))
                kl=a[f'jlens_exit{u}_kl_to_eventual_readout'][:,u*48:(u+1)*48].mean(1)
                mean,ci,_=stats(kl,count);rr=reported[position]['loops'][str(u+1)]['kl_local_lens_to_final_lens_all48']
                check('local final lens KL',mean,rr['mean'],tol=2e-5)
                check('local final lens KL CI',ci,rr['ci95'],'intervals',tol=2e-5)
                loopKL.append(float(mean))
            kl=a['logitlens_kl_to_final'][:,u*48+47]
            mean,ci,_=stats(kl,count);rr=reported[position]['loops'][str(u+1)]['actual_exit_to_final_kl']
            check('actual exit final KL',mean,rr['mean'],tol=2e-5)
            check('actual exit final KL CI',ci,rr['ci95'],'intervals',tol=2e-5)
        avg=np.stack(per_item_matrices,1).mean(1);point,ci,_=stats(avg,count)
        summaries[position]={'actual_exit_matrix':actual.mean(0).tolist(),
            'local_to_final_lens_KL_all48_first3':loopKL,
            'first3loops_layers41to47_matrix':point.tolist(),
            'first3loops_layers41to47_matrix_CI':ci.tolist()}
        if position=='-1':
            summaries[position]['pod_local_exit_mismatches']=np.sum(pod['exit_top1']!=truth,axis=0).tolist()
            summaries[position]['loop3_layers41to47_paired_final_token_contrasts']=loop3_final_contrasts(a,count)
    return summaries


def main():
    exits=verify_exits();changed,interactions=verify_correctness()
    payload={'checked_scalar_values':COUNTER,'maximum_absolute_error':MAXERR,
             'changed_correctness_items':changed,'exit_summaries':exits,'correctness_interactions':interactions,
             'root_generator_sha256':hashlib.sha256((FOLLOW/'analyze_followup.py').read_bytes()).hexdigest()}
    print(json.dumps(payload,indent=2))


if __name__=='__main__': main()
