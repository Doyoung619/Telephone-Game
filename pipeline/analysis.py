"""Leakage-safe Telephone Game scheduling, reports and analysis.

Editor inference is deliberately separate. All figures contain measured scores;
search scaling is explicitly labelled synthetic. Prefix state fitting measures
first-loss survival, never treats its optimization certificate as an S guarantee.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

REFERENCE = Path(__file__).resolve().parents[1] / 'reference'
sys.path.insert(0, str(REFERENCE))
import schedulers as sc

ORDERS = [tuple(x) for x in sc.all_orders(4)]
ROLES = 'ABCD'
VERSION = 'telephone_analysis_v1.2_reporting_and_static_ablation_guards'
BOOTSTRAPS = 10000
TIE_TOLERANCE = 1e-6
# Predeclared deterministic tie break by increasing inference/fit complexity.
TIE_ORDER = ['A1', 'A2', 'A3', 'A6', 'A7', 'A4', 'A5']


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def jsonable(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {str(k): jsonable(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [jsonable(v) for v in x]
    return x


def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def write_csv(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:
            w.writerow({k:json.dumps(jsonable(v),ensure_ascii=False) if isinstance(v,(list,dict,tuple,np.ndarray)) else v for k,v in r.items()})


def write_jsonl(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(jsonable(r),ensure_ascii=False,allow_nan=False)+'\n' for r in rows))


def digest(obj):
    return hashlib.sha256(json.dumps(jsonable(obj),sort_keys=True).encode()).hexdigest()


def order_text(order): return ''.join(ROLES[int(a)] for a in order)


def _prefix(row):
    p=row.get('prefix',row.get('permutation',row.get('chosen_order',[])))
    if isinstance(p,str): return tuple(ROLES.index(a) for a in p if a in ROLES)
    return tuple(int(a) for a in p)


def _partition(record):
    split=record.get('split','')
    if split in ('calibration_fit','calibration_safety'): return split
    if split=='calibration': return 'calibration_'+record.get('calibration_partition','fit')
    return split


def _metric(row, name):
    v=row.get(name)
    return float(v) if v is not None else None


def _aggregate_rows(rows):
    """Average true decoding replicates per source, without multiplying N."""
    if len(rows)==1: return dict(rows[0])
    result=dict(rows[0])
    for k in ('F','Q','S','strict_success','omission_rate','altered_rate','elapsed_seconds'):
        v=[_metric(r,k) for r in rows if r.get(k) is not None]
        if v: result[k]=float(np.mean(v))
    result['replicates_averaged']=len(rows)
    return result


def _row_index(rows):
    index=defaultdict(list)
    for r in rows: index[(r['source_id'],_prefix(r))].append(r)
    return {k:_aggregate_rows(v) for k,v in index.items()}


def _bank(records,index,metric='S',require_complete=True):
    chosen=[]; values=[]
    for record in records:
        rr=[index.get((record['source_id'],p)) for p in ORDERS]
        if any(r is None or r.get(metric) is None for r in rr):
            if require_complete: raise ValueError(f"Incomplete 24-order {metric} bank: {record['source_id']}")
            continue
        y=np.array([float(r[metric]) for r in rr])
        if not np.all(np.isfinite(y)) or np.min(y)<0 or np.max(y)>1: raise ValueError('Scores outside [0,1]')
        chosen.append(record); values.append(y)
    if not values: raise ValueError(f'No complete {metric} bank')
    return chosen,np.array(values)


def _states(row):
    if 'fact_states' in row: return list(row['fact_states'])
    return [s.get('state','LOST') for s in row.get('fact_statuses',[])]


def _source_length(record):
    return int(record.get('metadata',{}).get('source_words',len(record['source_text'].split())))


def cluster_bootstrap(values, groups, n=BOOTSTRAPS, seed=2026):
    """CI for source-weighted mean, resampling independent source groups."""
    values=np.asarray(values,dtype=float)
    if values.ndim==1: values=values[:,None]
    if len(values)==0: return None
    names=list(dict.fromkeys(groups)); sizes=[]; totals=[]
    for group in names:
        block=values[np.array([g==group for g in groups])]
        sizes.append(len(block));totals.append(block.sum(axis=0))
    sizes=np.array(sizes);totals=np.array(totals)
    rng=np.random.default_rng(seed);out=[]
    for start in range(0,n,500):
        picks=rng.integers(0,len(names),size=(min(500,n-start),len(names)))
        out.append(totals[picks].sum(axis=1)/sizes[picks].sum(axis=1)[:,None])
    samples=np.concatenate(out)
    return np.quantile(samples,[.025,.975],axis=0)


def fit_transfer(records,index):
    """Empirical 2-live-state model, LOST absorbing by FIRST-loss definition.

    Actual restored facts remain recorded in the evaluator; only this model uses
    first-loss survival. No pseudo-observations are added. An unobserved column
    uses identity and is explicitly recorded as unidentifiable.
    """
    counts=np.zeros((4,2,2));denom=np.zeros((4,2));init=np.zeros(2)
    unobserved=[];restored=0;observed_edges=0
    for rec in records:
        sid=rec['source_id'];root=index.get((sid,()))
        if root is None or not _states(root): raise ValueError(f'Missing origin fact states: {sid}')
        root_states=_states(root)
        for state in root_states:
            if state in ('standalone','embedded'): init[0 if state=='standalone' else 1]+=1
        for depth in range(1,5):
            for prefix in itertools.permutations(range(4),depth):
                child=index.get((sid,prefix));parent=index.get((sid,prefix[:-1]))
                if child is None or parent is None: raise ValueError(f'Missing calibration prefix {sid}:{prefix}')
                cs=_states(child);ps=_states(parent)
                if len(cs)!=len(root_states) or len(ps)!=len(cs): raise ValueError('Fact-axis mismatch')
                prev_alive=np.ones(len(cs),dtype=bool)
                for d in range(depth):
                    prev_alive &= np.array([s!='LOST' for s in _states(index[(sid,prefix[:d])])])
                role=prefix[-1]
                for i,(old,new) in enumerate(zip(ps,cs)):
                    if old=='LOST' and new!='LOST': restored+=1
                    if not prev_alive[i] or old=='LOST': continue
                    a=0 if old=='standalone' else 1;denom[role,a]+=1
                    if new!='LOST': counts[role,0 if new=='standalone' else 1,a]+=1
                    observed_edges+=1
    if init.sum()==0: raise ValueError('No recoverable origin facts')
    initial=init/init.sum();mat=np.zeros_like(counts)
    for a in range(4):
        for state in range(2):
            if denom[a,state]: mat[a,:,state]=counts[a,:,state]/denom[a,state]
            else:
                mat[a,state,state]=1
                unobserved.append({'role':ROLES[a],'state':['standalone','embedded'][state],'fallback':'identity_unidentified_column'})
    return {'matrices':mat,'initial':initial,'counts':counts,'denominators':denom,
            'observed_live_edges':observed_edges,'restored_edge_events':restored,
            'unobserved_columns':unobserved,'objective':'first_loss_fact_survival_surrogate_not_joint_S'}


def fit_candidates(records,score_rows):
    if any(_partition(r) not in ('calibration_fit','calibration_safety') for r in records):
        raise ValueError('Fitting accepts calibration records only')
    index=_row_index(score_rows)
    fit_records=[r for r in records if _partition(r)=='calibration_fit']
    safety_records=[r for r in records if _partition(r)=='calibration_safety']
    fit_records,y=_bank(fit_records,index)
    _,safety=_bank(safety_records,index)
    lengths=np.array([_source_length(r) for r in fit_records]);safe_lengths=np.array([_source_length(r) for r in safety_records])
    registry={a:{'id':a,'fit_partition':'calibration_fit','objective':'S'} for a in [f'A{i}' for i in range(1,8)]}
    models={};fit_start=time.perf_counter()
    def store(algorithm,result,params,start):
        models[algorithm]={'algorithm':algorithm,'order':list(result.order),'proxy_value':result.value,
                           'proxy_upper':result.upper,'search_gap':result.gap,'expanded':result.expanded,
                           'fit_seconds':time.perf_counter()-start,'parameters':jsonable(params)}
    t=time.perf_counter();store('A1',sc.erm(y,ORDERS),{},t)
    t=time.perf_counter();b,w=sc.fit_pairwise(y,ORDERS,ridge=.001)
    store('A2',sc.pairwise_dp(w,b),{'intercept':b,'pair_coefficients':w,'ridge':.001},t)
    t=time.perf_counter();b,start,end,v=sc.fit_adjacent(y,ORDERS,ridge=.001)
    store('A3',sc.adjacent_dp(start,end,v,b),{'intercept':b,'start':start,'end':end,'adjacency':v,'ridge':.001},t)
    t=time.perf_counter();transfer=fit_transfer(fit_records,index)
    transfer_fit=time.perf_counter()-t
    for algorithm in ['A4','A5']:
        t=time.perf_counter()
        result=(sc.transfer_beam(transfer['matrices'],transfer['initial'],width=4) if algorithm=='A4'
                else sc.transfer_branch_bound(transfer['matrices'],transfer['initial'],max_nodes=100000))
        store(algorithm,result,{**transfer,**({'beam_width':4} if algorithm=='A4' else {'max_nodes':100000})},t)
        models[algorithm]['shared_transition_fit_seconds']=transfer_fit
    t=time.perf_counter();scenarios={name:y[mask] for name,mask in [('short',lengths<=120),('long',lengths>120)] if mask.any()}
    robust=sc.robust_order(scenarios,ORDERS)
    models['A6']={'algorithm':'A6','order':list(robust['order']),'parameters':jsonable(robust),
                  'fit_seconds':time.perf_counter()-t,'search_gap':None,'expanded':len(ORDERS),
                  'missing_preregistered_scenarios':sorted(set(['short','long'])-set(scenarios))}
    t=time.perf_counter();context=sc.contextual_safe(y,lengths,safety,safe_lengths,ORDERS,
                                                   minimum_leaf=10 if len(fit_records)>=24 else 3)
    models['A7']={'algorithm':'A7','order':list(ORDERS[context['baseline_index']]),
                  'parameters':jsonable(context),'fit_seconds':time.perf_counter()-t,'expanded':len(ORDERS),'search_gap':None}
    # Public-only heuristic: fit-average one-step F losses, preserve fixed order.
    f0=np.mean([index[(r['source_id'],())]['F'] for r in fit_records])
    one_step=np.array([np.mean([index[(r['source_id'],(a,))]['F'] for r in fit_records]) for a in range(4)])
    baseline={'best_fixed':models['A1']['order'],'intuitive':[2,0,1,3],
              'least_loss_first':np.argsort(f0-one_step,kind='stable').tolist(),
              'one_step_F':one_step.tolist(),'origin_F':float(f0)}
    return {'models':models,'baseline_orders':baseline,'fit_source_ids':[r['source_id'] for r in fit_records],
            'safety_source_ids':[r['source_id'] for r in safety_records],'fit_seconds':time.perf_counter()-fit_start,
            'registry':registry,'version':VERSION}


def predict_order(model,record):
    if model['algorithm']=='A7' and 'orders' in model.get('parameters',{}):
        p=model['parameters'];leaf=0 if _source_length(record)<=p['threshold'] else 1
        return tuple(p['orders'][p['leaf_indices'][leaf]])
    return tuple(model['order'])


def comparison_orders(frozen,record):
    selected=frozen.get('model',frozen['models'][frozen['algorithm']])
    rnd=int(hashlib.sha256(('2026:'+record['source_id']).encode()).hexdigest(),16)%len(ORDERS)
    return {'selected':predict_order(selected,record),'best_fixed':tuple(frozen['baseline_orders']['best_fixed']),
            'intuitive':tuple(frozen['baseline_orders']['intuitive']),
            'least_loss_first':tuple(frozen['baseline_orders']['least_loss_first']),
            'random_once':ORDERS[rnd]}


def _plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.dpi':200,'figure.dpi':130,'pdf.fonttype':42})
    return plt


def _save_figure(fig,review,name):
    review=Path(review);review.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'): fig.savefig(review/f'{name}.{ext}',bbox_inches='tight')
    _plots().close(fig)


def selection_analysis(fitted,records,score_rows,out_dir,review_dir,metadata=None,freeze=True):
    if any(_partition(r)!='selection' for r in records): raise ValueError('Selection report accepts selection only')
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    index=_row_index(score_rows);records,y=_bank(records,index)
    groups=[r.get('group_id',r['source_id']) for r in records]
    source_rows=[];report=[];predictions=[];score_by_method={}
    for algorithm,model in fitted['models'].items():
        rows=[];times=[]
        for rec in records:
            t=time.perf_counter();order=predict_order(model,rec);times.append(time.perf_counter()-t)
            row=index[(rec['source_id'],order)];rows.append(row)
            source_rows.append({'source_id':rec['source_id'],'group_id':rec.get('group_id',rec['source_id']),
                                'algorithm':algorithm,'chosen_order':order_text(order),
                                **{k:row.get(k) for k in ('F','Q','S','strict_success','omission_rate','altered_rate')}})
            predictions.append({'source_id':rec['source_id'],'algorithm':algorithm,'chosen_order':list(order),
                                'input_contract':'source_text_and_public_requirements_only',
                                'policy_hash':digest(model),'observed_gold_access':False})
        scores=np.array([r['S'] for r in rows]);score_by_method[algorithm]=scores
        ci=cluster_bootstrap(scores,groups)
        a1=np.array([index[(r['source_id'],tuple(fitted['models']['A1']['order']))]['S'] for r in records])
        diffci=cluster_bootstrap(scores-a1,groups)
        report.append({'algorithm':algorithm,'N_sources':len(rows),'N_groups':len(set(groups)),
                       **{k:float(np.mean([float(r.get(k,0) or 0) for r in rows])) for k in ('F','Q','S','strict_success','omission_rate','altered_rate')},
                       'S_CI_low':float(ci[0,0]),'S_CI_high':float(ci[1,0]),
                       'S_delta_vs_A1':float(np.mean(scores-a1)),
                       'delta_CI_low':float(diffci[0,0]),'delta_CI_high':float(diffci[1,0]),
                       'chosen_order':order_text(model['order']) if algorithm!='A7' else 'length_policy' if model['parameters']['activated'] else order_text(model['order']),
                       'fit_seconds':model['fit_seconds'],'scheduling_seconds_per_source':float(np.mean(times)),
                       'editor_calls_per_deployed_source':4,'planner_editor_calls':0,
                       'search_nodes':model.get('expanded'),'surrogate_search_gap':model.get('search_gap')})
    maximum=max(r['S'] for r in report)
    winner=next(a for a in TIE_ORDER if next(r['S'] for r in report if r['algorithm']==a)>=maximum-TIE_TOLERANCE)
    ranking=sorted(report,key=lambda r:(-r['S'],TIE_ORDER.index(r['algorithm'])))
    for i,r in enumerate(ranking,1):r['rank']=i
    write_csv(out/'selection_report.csv',ranking);write_csv(out/'selection_per_source.csv',source_rows)
    write_jsonl(out/'predicted_orders.jsonl',predictions)
    write_json(out/'candidate_registry.json',{'version':VERSION,'tie_tolerance':TIE_TOLERANCE,'tie_order':TIE_ORDER,
               'objective':'source_mean_S','registry':fitted['registry'],'fit_source_ids':fitted['fit_source_ids'],
               'safety_source_ids':fitted['safety_source_ids']})
    frozen={**fitted,'status':'frozen' if freeze else 'selection_complete','algorithm':winner,
            'model':fitted['models'][winner],'selection_rule':'maximum_selection_mean_S_then_predeclared_cost_tie_break',
            'selection_source_ids':[r['source_id'] for r in records],
            'selection_group_ids':groups,'selection_scores':ranking,
            'metadata':metadata or {},'data_manifest_hash':digest(records),
            'policy_hash':digest(fitted['models'][winner]),
            'authorization':'User requested automatic strongest-algorithm selection before remaining experiments',
            'test_used_for_selection':False,'frozen_at_unix':time.time()}
    write_json(out/'selected_method.json',frozen)
    lines=['# Algorithm selection',f'Frozen winner: **{winner}**, maximum held-out selection source-mean S.',
           f'Fit {len(fitted["fit_source_ids"])} independent sources; safety {len(fitted["safety_source_ids"])}; selection {len(records)}.',
           'Selection ranks are point estimates. Paired source-group bootstrap intervals reflect measurement sampling uncertainty, not proof of superiority.',
           'All seven candidates share the same inference bank. A4/A5 optimize fitted first-loss survival; their search gap does not certify joint S.',
           'A7 activates only when the predeclared independent safety lower bound is positive.',
           '|Algorithm|Order|F|Q|S|95% CI S|Paired S−A1|','|---|---|---:|---:|---:|---|---:|']
    for r in ranking: lines.append(f"|{r['algorithm']}|{r['chosen_order']}|{r['F']:.3f}|{r['Q']:.3f}|{r['S']:.3f}|[{r['S_CI_low']:.3f}, {r['S_CI_high']:.3f}]|{r['S_delta_vs_A1']:+.3f}|")
    (out/'algorithm_selection_report.md').write_text('\n\n'.join(lines)+'\n')
    plt=_plots();fig,axes=plt.subplots(1,2,figsize=(11,4))
    xs=np.arange(len(report));means=[r['S'] for r in report]
    axes[0].bar(xs,means,color=['#15658A' if r['algorithm']==winner else '#79A8B8' for r in report])
    axes[0].errorbar(xs,means,yerr=np.maximum(0,np.array([[r['S']-r['S_CI_low'] for r in report],
                                          [r['S_CI_high']-r['S'] for r in report]])),fmt='none',color='#253944',capsize=3)
    axes[0].set(xticks=xs,xticklabels=[r['algorithm'] for r in report],ylabel='Selection mean joint score S',ylim=(0,1.02),title=f'Independent selection (n={len(records)}), frozen {winner}')
    mu=y.mean(axis=0);sort=np.argsort(mu)
    axes[1].plot(np.arange(24),mu[sort],'o-',color='#536F31',markersize=4)
    axes[1].set(xticks=np.arange(24),xticklabels=[order_text(ORDERS[j]) for j in sort],ylabel='Selection mean S',ylim=(0,1.02),title='All 24 fixed orders (exploratory ranking)')
    axes[1].tick_params(axis='x',labelrotation=90,labelsize=7);fig.tight_layout()
    _save_figure(fig,review_dir,'01_algorithm_tournament')
    plt=_plots();fig,ax=plt.subplots(figsize=(7,3.6));diff=[r['S_delta_vs_A1'] for r in report]
    ax.axhline(0,color='#555',lw=.8)
    ax.errorbar(xs,diff,yerr=np.maximum(0,np.array([[r['S_delta_vs_A1']-r['delta_CI_low'] for r in report],
                                     [r['delta_CI_high']-r['S_delta_vs_A1'] for r in report]])),fmt='o',capsize=4,color='#15658A')
    ax.set(xticks=xs,xticklabels=[r['algorithm'] for r in report],ylabel='Paired selection S − A1',title='Source-group bootstrap 95% intervals')
    fig.tight_layout();_save_figure(fig,review_dir,'02_selection_paired_differences')
    return frozen


def fit_and_select(records,score_rows,out_dir,review_dir,metadata=None):
    calibration=[r for r in records if _partition(r) in ('calibration_fit','calibration_safety')]
    selection=[r for r in records if _partition(r)=='selection']
    # Avoid passing any test score to the fitting implementation, even if caller
    # accidentally supplied its rows in a combined file.
    fit_ids={r['source_id'] for r in calibration};sel_ids={r['source_id'] for r in selection}
    fitted=fit_candidates(calibration,[r for r in score_rows if r['source_id'] in fit_ids])
    return selection_analysis(fitted,selection,[r for r in score_rows if r['source_id'] in sel_ids],out_dir,review_dir,metadata)


def final_analysis(frozen,records,score_rows,out_dir,review_dir,label='main'):
    if frozen.get('status')!='frozen': raise ValueError('Final evaluation requires frozen selection')
    if any(_partition(r) not in ('test','diagnostic') for r in records): raise ValueError('Final analysis accepts locked test/diagnostic only')
    if set(r['source_id'] for r in records)&set(frozen.get('fit_source_ids',[])+frozen.get('safety_source_ids',[])+frozen.get('selection_source_ids',[])):
        raise ValueError('Test IDs overlap fit/safety/selection')
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);index=_row_index(score_rows)
    rows=[];comparisons=[];summaries=[]
    for rec in records:
        sid=rec['source_id'];orders=comparison_orders(frozen,rec)
        for method,order in orders.items():
            observed=index.get((sid,order))
            if observed is None: continue
            rows.append({'source_id':sid,'group_id':rec.get('group_id',sid),'method':method,
                         'dataset':rec.get('dataset','synthetic'),'profile':label,'chosen_order':order_text(order),
                         'selected_algorithm':frozen['algorithm'],
                         **{k:observed.get(k) for k in ('F','Q','S','strict_success','omission_rate','altered_rate','unsupported_claims')},
                         'evaluation_status':observed.get('evaluation_status','measured'),
                         'editor_seconds':observed.get('elapsed_seconds'),'text_path':observed.get('text_path')})
        origin=index.get((sid,()))
        if origin:
            rows.append({'source_id':sid,'group_id':rec.get('group_id',sid),'method':'copy_source_diagnostic',
                         'dataset':rec.get('dataset','synthetic'),'profile':label,'chosen_order':'none',
                         **{k:origin.get(k) for k in ('F','Q','S','strict_success','omission_rate','altered_rate')},
                         'evaluation_status':'measured_unequal_edit_budget'})
        allrows=[index.get((sid,p)) for p in ORDERS]
        if all(r is not None for r in allrows):
            for method,aggregate in [('uniform_random_exact',np.mean),('posthoc_measured_oracle_reference',np.max)]:
                rows.append({'source_id':sid,'group_id':rec.get('group_id',sid),'method':method,
                             'dataset':rec.get('dataset','synthetic'),'profile':label,'chosen_order':'reference',
                             **{k:float(aggregate([float(r.get(k,0) or 0) for r in allrows])) for k in ('F','Q','S','strict_success')},
                             'evaluation_status':'diagnostic_all_24_orders_reference'})
    datasets=sorted(set(r['dataset'] for r in rows))
    for dataset in datasets:
        drows=[r for r in rows if r['dataset']==dataset]
        for method in dict.fromkeys(r['method'] for r in drows):
            mrows=[r for r in drows if r['method']==method];groups=[r['group_id'] for r in mrows]
            summary={'dataset':dataset,'profile':label,'method':method,'N_sources':len(mrows),'N_groups':len(set(groups))}
            for metric in ('F','Q','S','strict_success','omission_rate','altered_rate'):
                good=[r for r in mrows if r.get(metric) is not None]
                if not good: continue
                vals=[r[metric] for r in good];ci=cluster_bootstrap(vals,[r['group_id'] for r in good])
                # A nonparametric bootstrap cannot identify a population tail
                # when every independent binary observation is zero or one.
                if metric=='strict_success' and len(set(r['group_id'] for r in good))==len(good):
                    if all(float(v)==0 for v in vals):
                        ci=np.array([[0.],[1-.025**(1/len(vals))]])
                        summary['strict_success_CI_method']='two_sided_95_exact_binomial_boundary'
                    elif all(float(v)==1 for v in vals):
                        ci=np.array([[.025**(1/len(vals))],[1.]])
                        summary['strict_success_CI_method']='two_sided_95_exact_binomial_boundary'
                summary.update({metric:float(np.mean(vals)),metric+'_CI_low':float(ci[0,0]),metric+'_CI_high':float(ci[1,0])})
            summaries.append(summary)
        lookup={(r['source_id'],r['method']):r for r in drows}
        for baseline in ('best_fixed','random_once','intuitive','least_loss_first','uniform_random_exact'):
            pairs=[(r,lookup[(r['source_id'],baseline)]) for r in drows if r['method']=='selected' and (r['source_id'],baseline) in lookup]
            if not pairs:continue
            for metric in ('F','Q','S'):
                good=[(a,b) for a,b in pairs if a.get(metric) is not None and b.get(metric) is not None]
                if not good:continue
                values=[a[metric]-b[metric] for a,b in good];ci=cluster_bootstrap(values,[a['group_id'] for a,b in good])
                comparisons.append({'dataset':dataset,'profile':label,'comparison':'selected_minus_'+baseline,
                                    'metric':metric,'N_sources':len(good),'mean_difference':float(np.mean(values)),
                                    'CI_low':float(ci[0,0]),'CI_high':float(ci[1,0]),
                                    'bootstrap_unit':'source_group','bootstrap_replicates':BOOTSTRAPS})
    write_csv(out/'per_source.csv',rows);write_csv(out/'summary.csv',summaries);write_csv(out/'paired_comparisons.csv',comparisons)
    write_json(out/'scope.json',{'status':'complete_for_observed_orders','requested_sources':len(records),
               'measured_selected_sources':sum(r['method']=='selected' for r in rows),
               'random_baseline':'one uniform permutation sampled independently per source; not exact uniform expectation unless all24 available',
               'oracle_reference':'not run unless full24 test bank exists','profile':label,'frozen_policy_hash':frozen['policy_hash']})
    if summaries:
        plt=_plots();fig,axes=plt.subplots(1,len(datasets),figsize=(max(6,5*len(datasets)),4),squeeze=False)
        for dataset,ax in zip(datasets,axes[0]):
            relevant=[r for r in summaries if r['dataset']==dataset and not r['method'].startswith(('copy','posthoc'))]
            x=np.arange(len(relevant));vals=[r.get('S',0) for r in relevant]
            ax.bar(x,vals,color=['#15658A' if r['method']=='selected' else '#8AAFB6' for r in relevant])
            ax.errorbar(x,vals,yerr=np.maximum(0,np.array([[r.get('S',0)-r.get('S_CI_low',0) for r in relevant],
                                                        [r.get('S_CI_high',0)-r.get('S',0) for r in relevant]])),fmt='none',color='#253944',capsize=3)
            ax.set(xticks=x,xticklabels=[r['method'].replace('_',' ') for r in relevant],ylim=(0,1.02),ylabel='Held-out joint score S',title=f'{dataset}: {label}')
            ax.tick_params(axis='x',labelrotation=32,labelsize=8)
        fig.tight_layout();_save_figure(fig,review_dir,'03_final_performance_'+label)
    # Deterministic examples include both tails; do not alter quantitative rows.
    selected=[r for r in rows if r['method']=='selected'];selected.sort(key=lambda r:(r.get('S',0),r['source_id']))
    write_json(out/'representative_examples.json',{'selection_rule':'three lowest and three highest measured selected-method S; exploratory illustrations',
               'lowest':selected[:3],'highest':list(reversed(selected[-3:]))})
    report=['# Frozen-method final evaluation',f"Frozen algorithm: {frozen['algorithm']}; profile: {label}.",
            f'{len(records)} requested test sources. Report counts reflect observed rows only.',
            'Random-once uses one presampled uniform permutation per source. Its across-source estimate is an unbiased randomized-policy comparison, not an exact 24-order mean.',
            'Confidence intervals resample whole source groups 10,000 times. Test data never change the frozen method.',
            '|Dataset|Method|N|F|Q|S|95% CI S|','|---|---|---:|---:|---:|---:|---|']
    for r in summaries:
        report.append(f"|{r['dataset']}|{r['method']}|{r['N_sources']}|{r.get('F',0):.3f}|{r.get('Q',0):.3f}|{r.get('S',0):.3f}|[{r.get('S_CI_low',0):.3f}, {r.get('S_CI_high',0):.3f}]|")
    (out/'final_report.md').write_text('\n\n'.join(report)+'\n')
    return {'summary':summaries,'paired_comparisons':comparisons,'per_source':rows}


def lifetime_analysis(rows,out_dir,review_dir,records=None):
    """Summarize measured t0..16 trajectories; incomplete sources are explicit."""
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    by_source=defaultdict(dict);record_map={r['source_id']:r for r in (records or [])}
    for row in rows:
        method=row.get('condition',row.get('method','selected'))
        hop=int(row.get('hop',len(_prefix(row))))
        by_source[(method,row['source_id'])][hop]=row
    complete={k:v for k,v in by_source.items() if set(range(17)).issubset(v)}
    incomplete=[{'method':m,'source_id':s,'observed_hops':sorted(v)} for (m,s),v in by_source.items() if (m,s) not in complete]
    curves=[];summaries=[];events=[];restorations=[];source_metrics=[]
    matrices=defaultdict(list);groups=defaultdict(list)
    for (method,sid),trajectory in complete.items():
        states=[_states(trajectory[t]) for t in range(17)]
        if not states[0] or any(len(s)!=len(states[0]) for s in states): raise ValueError(f'Invalid lifetime fact axis {sid}')
        c=np.array([[s!='LOST' for s in hop] for hop in states],dtype=float)
        if not np.all(c[0]==1): raise ValueError(f'Origin audit F0 != 1: {sid}')
        u=np.cumprod(c,axis=0);f=c.mean(axis=1);survival=u.mean(axis=1)
        matrices[method].append(np.stack([f,survival],axis=1));groups[method].append(trajectory[0].get('group_id',record_map.get(sid,{}).get('group_id',sid)))
        types=[s.get('type','unlabelled') for s in trajectory[0].get('fact_statuses',[])]
        for i in range(c.shape[1]):
            losses=np.flatnonzero(c[1:,i]==0);first=int(losses[0]+1) if len(losses) else None
            events.append({'method':method,'source_id':sid,'fact_index':i,'fact_type':types[i] if len(types)>i else 'unlabelled',
                           'first_loss_hop':first,'right_censored':first is None,'censor_hop':16})
            for t in range(1,17):
                if c[t,i]>c[t-1,i]: restorations.append({'method':method,'source_id':sid,'fact_index':i,'restored_hop':t,
                                                        'prior_evidence':trajectory[t-1].get('fact_statuses',[]),'restored_evidence':trajectory[t].get('fact_statuses',[])})
        source_metrics.append({'method':method,'source_id':sid,'normalized_F_AUC':float(np.trapezoid(f)/16) if hasattr(np,'trapezoid') else float(np.trapz(f)/16),
                               'first_loss_RMST_hops':float(survival[:16].sum()),'F16':float(f[-1]),'U16':float(survival[-1]),
                               'restored_transitions':int(np.sum(c[1:]>c[:-1])),
                               'total_loss_transitions':int(np.sum(c[1:]<c[:-1]))})
    for method,values in matrices.items():
        arr=np.array(values);mean=arr.mean(axis=0);ci=cluster_bootstrap(arr.reshape(len(arr),-1),groups[method]).reshape(2,17,2)
        def half(curve):
            below=np.flatnonzero(curve<=.5*curve[0]);return int(below[0]) if len(below) else '>16'
        for t in range(17):
            curves.append({'method':method,'hop':t,'N_sources':len(arr),'F':float(mean[t,0]),'F_low':float(ci[0,t,0]),'F_high':float(ci[1,t,0]),
                           'U':float(mean[t,1]),'U_low':float(ci[0,t,1]),'U_high':float(ci[1,t,1]),'CI_scope':'pointwise_95_percent'})
        # Bootstrap half-life: censoring is retained rather than replacing >16.
        rng=np.random.default_rng(2026);sample_halves=[[],[]];censored=[0,0]
        groupnames=list(dict.fromkeys(groups[method]));blocks=[arr[np.array([g==group for g in groups[method]])] for group in groupnames]
        for _ in range(BOOTSTRAPS):
            picks=rng.integers(0,len(blocks),len(blocks));resampled=np.concatenate([blocks[j] for j in picks]).mean(axis=0)
            for k in (0,1):
                h=half(resampled[:,k])
                if h=='>16':censored[k]+=1
                else:sample_halves[k].append(h)
        sm=[r for r in source_metrics if r['method']==method]
        summaries.append({'method':method,'N_sources':len(arr),'F_half_life_hops':half(mean[:,0]),'first_loss_half_life_hops':half(mean[:,1]),
                          'F16':float(mean[-1,0]),'U16':float(mean[-1,1]),
                          'normalized_F_AUC':float(np.mean([r['normalized_F_AUC'] for r in sm])),
                          'first_loss_RMST_hops':float(np.mean([r['first_loss_RMST_hops'] for r in sm])),
                          'F_half_life_bootstrap_censored_fraction':censored[0]/BOOTSTRAPS,
                          'U_half_life_bootstrap_censored_fraction':censored[1]/BOOTSTRAPS,
                          'F_half_life_uncensored_conditional_CI':np.quantile(sample_halves[0],[.025,.975]).tolist() if sample_halves[0] else None,
                          'U_half_life_uncensored_conditional_CI':np.quantile(sample_halves[1],[.025,.975]).tolist() if sample_halves[1] else None})
    write_jsonl(out/'trajectories.jsonl',rows);write_csv(out/'fact_events.csv',events);write_csv(out/'curves.csv',curves)
    write_csv(out/'summary.csv',summaries);write_csv(out/'per_source.csv',source_metrics);write_jsonl(out/'restoration_events.jsonl',restorations)
    write_json(out/'scope.json',{'complete_trajectories':len(complete),'incomplete_trajectories':incomplete,
               'maximum_hops':16,'half_life_censoring':'right-censored >16; never imputed as 16 or17'})
    if curves:
        plt=_plots();fig,axes=plt.subplots(1,2,figsize=(11,4));colors=plt.get_cmap('tab10')
        for i,method in enumerate(matrices):
            cc=[r for r in curves if r['method']==method];x=[r['hop'] for r in cc]
            for ax,metric,title in zip(axes,['F','U'],['Current recoverability','First-loss survival (absorbing)']):
                ax.plot(x,[r[metric] for r in cc],label=method.replace('_',' '),color=colors(i%10))
                if i<4:ax.fill_between(x,[r[metric+'_low'] for r in cc],[r[metric+'_high'] for r in cc],alpha=.10,color=colors(i%10))
                ax.set(xlabel='Editing hops (4 hops = one cycle)',ylabel='Mean fraction of original facts',ylim=(0,1.03),title=title,xticks=[0,1,2,4,8,12,16])
        axes[0].legend(fontsize=7,ncol=2);fig.tight_layout();_save_figure(fig,review_dir,'04_lifetime_recoverability_and_survival')
        type_rows=[]
        for method in matrices:
            facts=[r for r in events if r['method']==method]
            for typ in sorted(set(r['fact_type'] for r in facts)):
                subset=[r for r in facts if r['fact_type']==typ]
                for t in range(17):type_rows.append({'method':method,'fact_type':typ,'hop':t,'U':float(np.mean([r['first_loss_hop'] is None or r['first_loss_hop']>t for r in subset])),'N_facts':len(subset)})
        write_csv(out/'fact_type_survival.csv',type_rows)
        if any(r['fact_type']!='unlabelled' for r in type_rows):
            plt=_plots();fig,ax=plt.subplots(figsize=(7,4))
            chosen='selected' if 'selected' in matrices else next(iter(matrices))
            for typ in sorted(set(r['fact_type'] for r in type_rows)):
                tt=[r for r in type_rows if r['method']==chosen and r['fact_type']==typ]
                ax.plot([r['hop'] for r in tt],[r['U'] for r in tt],label=typ)
            ax.set(xlabel='Editing hops',ylabel='First-loss survival',ylim=(0,1.03),title=f'Fact-type survival: {chosen}');ax.legend(fontsize=8)
            fig.tight_layout();_save_figure(fig,review_dir,'05_fact_type_survival')
    report=['# Lifetime analysis',f'Complete measured trajectories: {len(complete)}. Incomplete: {len(incomplete)}.',
            'F is present recoverability; U removes a fact at its first loss and is monotone by construction. Restorations in F are preserved in the records.',
            'Bootstrap bands are pointwise 95% source-group intervals. Half-lives beyond 16 hops remain right-censored; conditional uncensored intervals are labelled explicitly.',
            'First-loss restricted mean lifetime is sum U(t), t=0..15, in hops. F AUC is trapezoidal area/16. These are fact-retention measures, not mutual information or SDPI coefficients.',
            '|Method|N|F16|U16|F half-life|First-loss half-life|RMST hops|','|---|---:|---:|---:|---|---|---:|']
    for r in summaries:report.append(f"|{r['method']}|{r['N_sources']}|{r['F16']:.3f}|{r['U16']:.3f}|{r['F_half_life_hops']}|{r['first_loss_half_life_hops']}|{r['first_loss_RMST_hops']:.2f}|")
    (out/'lifetime_report.md').write_text('\n\n'.join(report)+'\n')
    return {'curves':curves,'summary':summaries,'incomplete':incomplete}


def _nested_fit_records(records,n):
    """Deterministic round-robin domain/length stratification, fixed before scores."""
    strata=defaultdict(list)
    for r in records:
        strata[(r.get('dataset','synthetic'),_source_length(r)<=120)].append(r)
    for k in strata:strata[k].sort(key=lambda r:hashlib.sha256(('nested11:'+r['source_id']).encode()).hexdigest())
    ordered=[]
    for i in range(max(map(len,strata.values()))):
        for key in sorted(strata):
            if i<len(strata[key]):ordered.append(strata[key][i])
    return ordered[:n]


def prepare_ablation_models(frozen,calibration_records,score_rows,out_dir):
    """Freeze a small chosen-method ablation grid before observing test scores."""
    if frozen.get('status')!='frozen': raise ValueError('Ablations require frozen winner')
    if any(_partition(r) not in ('calibration_fit','calibration_safety') for r in calibration_records):raise ValueError('Calibration only')
    fit=[r for r in calibration_records if _partition(r)=='calibration_fit'];safety=[r for r in calibration_records if _partition(r)=='calibration_safety']
    index=_row_index(score_rows);_,y=_bank(fit,index);algorithm=frozen['algorithm']
    model=frozen['models'][algorithm];models={'default':model};registry=[]
    def add(name,order,parameters=None,**extra):
        models[name]={'algorithm':algorithm,'order':list(order),'parameters':parameters or {},**extra}
    for size in (6,12,24):
        if size>len(fit):continue
        nested=_nested_fit_records(fit,size);ids={r['source_id'] for r in nested+safety}
        ff=fit_candidates(nested+safety,[r for r in score_rows if r['source_id'] in ids])
        models[f'calibration_fit_n{size}']=ff['models'][algorithm]
        models[f'calibration_fit_n{size}']['nested_fit_source_ids']=[r['source_id'] for r in nested]
    if algorithm=='A1':
        rng=np.random.default_rng(2026);sample=sorted(rng.choice(len(ORDERS),size=6,replace=False).tolist())
        chosen=sample[int(np.argmax(y[:,sample].mean(axis=0)))];add('six_random_orders',ORDERS[chosen],{'sampled_order_indices':sample})
    elif algorithm=='A2':
        for ridge in (.0001,.01):
            b,w=sc.fit_pairwise(y,ORDERS,ridge);add(f'ridge_{ridge}',sc.pairwise_dp(w,b).order,{'ridge':ridge})
        b,w=sc.fit_pairwise(y,ORDERS,.001);add('pair_coefficients_off',ORDERS[0],{'all_coefficients_zero':True})
        limited=w.copy()
        for a in range(4):
            for c in range(4):
                if (min(a,c),max(a,c)) not in [(0,1),(1,2),(2,3)]:limited[a,c]=0
        add('three_predeclared_pairs_only',sc.pairwise_dp(limited,b).order,{'pairs':[[0,1],[1,2],[2,3]]})
    elif algorithm=='A3':
        b,start,end,v=sc.fit_adjacent(y,ORDERS,.001)
        add('adjacency_off',sc.adjacent_dp(start,end,np.zeros_like(v),b).order)
        add('start_end_off',sc.adjacent_dp(np.zeros(4),np.zeros(4),v,b).order)
        add('start_off',sc.adjacent_dp(np.zeros(4),end,v,b).order)
        add('end_off',sc.adjacent_dp(start,np.zeros(4),v,b).order)
    elif algorithm in ('A4','A5'):
        p=np.array(model['parameters']['matrices']);initial=np.array(model['parameters']['initial'])
        for width in (1,8,32):
            result=sc.transfer_beam(p,initial,width=width)
            add(f'beam_width_{width}',result.order,{'beam_width':width,'proxy_value':result.value,'search_gap':result.gap,'expanded':result.expanded})
        # Collapse standalone/embedded into one survival coordinate: scalar
        # contractions commute, explicitly removing all order effects.
        scalar=np.array([float(np.ones(2)@p[a]@initial) for a in range(4)])[:,None,None]
        result=sc.transfer_branch_bound(scalar,np.ones(1),max_nodes=100000)
        add('one_live_state_commutative',result.order,{'role_survival_scalars':scalar[:,0,0].tolist(),'proxy_value':result.value})
        if algorithm=='A5':
            for budget in (0,100,1000):
                result=sc.transfer_branch_bound(p,initial,max_nodes=budget)
                add(f'node_budget_{budget}',result.order,{'max_nodes':budget,'search_gap':result.gap,'expanded':result.expanded})
            values=[sc.transfer_score(p,initial,order) for order in ORDERS];j=int(np.argmax(values))
            add('pruning_off_exhaustive',ORDERS[j],{'permutations_evaluated':24,'proxy_value':values[j],'search_gap':0})
    elif algorithm=='A6':
        add('mean_objective',sc.erm(y,ORDERS).order)
        lengths=np.array([_source_length(r) for r in fit])
        for name,mask in [('omit_long',lengths<=120),('omit_short',lengths>120)]:
            if mask.any():add(name,sc.erm(y[mask],ORDERS).order)
    elif algorithm=='A7':
        p=dict(model['parameters']);baseline=ORDERS[p['baseline_index']]
        add('global_only',baseline)
        unsafe=dict(p);unsafe['leaf_indices']=p['proposed_leaf_indices'];unsafe['activated']=True
        add('safety_fallback_off_EXPLORATORY',baseline,unsafe,unsafe_exploratory=True)
        add('length_feature_off',baseline)
    for name,mm in models.items():
        registry.append({'name':name,'algorithm':algorithm,'model_hash':digest(mm),
                         'parameters':mm.get('parameters',{}),'nested_fit_source_ids':mm.get('nested_fit_source_ids'),
                         'unsafe_exploratory':bool(mm.get('unsafe_exploratory'))})
    out=Path(out_dir);write_json(out/'ablation_registry.json',{'status':'frozen_before_test_analysis','algorithm':algorithm,
         'default_policy_hash':frozen['policy_hash'],'variants':registry,
         'common_generation_conditions':{'compression_fractions':[.6,.75,.9],'bullet_condition':['one_to_eight','exactly_three'],
         'source_subset':'fixed predeclared test IDs; witness feasibility checked before generation'},
         'frozen_at_unix':time.time()})
    write_json(out/'ablation_models.json',models)
    return models


def ablation_orders(frozen,record,models=None):
    if models is None:models={'default':frozen.get('model',frozen['models'][frozen['algorithm']])}
    return {name:predict_order(model,record) for name,model in models.items()}


def ablation_analysis(frozen,records,score_rows,models,out_dir,review_dir):
    out=Path(out_dir);rows=[];summary=[]
    record_ids={r['source_id'] for r in records}
    grouped=defaultdict(list)
    for observed in score_rows:
        if observed['source_id'] not in record_ids or int(observed.get('hop',len(_prefix(observed))))!=4:continue
        name=observed.get('condition',observed.get('method','default'))
        grouped[(observed['source_id'],name)].append(observed)
    # Compression and format changes have distinct condition/variant keys even
    # when their role prefix is identical; never average across interventions.
    for (sid,name),observations in grouped.items():
        observed=_aggregate_rows(observations)
        rows.append({'source_id':sid,'group_id':observed.get('group_id',sid),'variant':name,
                     'order':order_text(_prefix(observed)),
                     **{k:observed.get(k) for k in ('F','Q','S','strict_success')},
                     'unsafe_exploratory':bool(models.get(name,{}).get('unsafe_exploratory'))})
    names=list(dict.fromkeys(list(models)+[r['variant'] for r in rows]))
    for name in names:
        rr=[r for r in rows if r['variant']==name]
        if not rr:continue
        ci=cluster_bootstrap([r['S'] for r in rr],[r['group_id'] for r in rr])
        baseline={r['source_id']:r['S'] for r in rows if r['variant']=='default'}
        diffs=[r['S']-baseline[r['source_id']] for r in rr if r['source_id'] in baseline]
        summary.append({'variant':name,'N_sources':len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in ('F','Q','S')},
                        'S_CI_low':float(ci[0,0]),'S_CI_high':float(ci[1,0]),'S_minus_default':float(np.mean(diffs)) if diffs else None,
                        'unsafe_exploratory':bool(models.get(name,{}).get('unsafe_exploratory'))})
    write_csv(out/'ablation_per_source.csv',rows);write_csv(out/'ablation_summary.csv',summary)
    if summary:
        plt=_plots();fig,ax=plt.subplots(figsize=(8,4));x=np.arange(len(summary));means=[r['S'] for r in summary]
        ax.bar(x,means,color=['#15658A' if r['variant']=='default' else '#8AAFB6' for r in summary])
        ax.errorbar(x,means,yerr=np.maximum(0,np.array([[r['S']-r['S_CI_low'] for r in summary],[r['S_CI_high']-r['S'] for r in summary]])),fmt='none',capsize=3,color='#253944')
        ax.set(xticks=x,xticklabels=[r['variant'].replace('_',' ') for r in summary],ylabel='Held-out joint score S',ylim=(0,1.02),title=f'Predeclared {frozen["algorithm"]} ablations')
        ax.tick_params(axis='x',labelrotation=32,labelsize=8);fig.tight_layout();_save_figure(fig,review_dir,'07_chosen_method_ablations')
    return summary


def theory_diagnostics(frozen,cal_records,cal_rows,heldout_records,heldout_rows,out_dir,review_dir):
    """Evaluate observed prediction errors without manufacturing missing banks."""
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);cal_index=_row_index(cal_rows);index=_row_index(heldout_rows)
    residuals=[];certificates=[];failures=[];summary=[]
    fit=[r for r in cal_records if _partition(r)=='calibration_fit'];_,fit_y=_bank(fit,cal_index)
    selected=frozen['models'][frozen['algorithm']]
    if selected['algorithm']!='A7':
        certificate=sc.finite_bank_certificate(fit_y,ORDERS,selected['order'])
        certificates.append({'kind':'finite_static_policy_class_population_bound','partition':'calibration_fit',
                             'uses_measured_joint_S':True,**certificate,
                             'vacuous_lower':certificate['lower']==0,'notes':'Uniform over all24 fixed permutations; not contextual policy or pointwise guarantee'})
    for algorithm in ('A2','A3','A4','A5'):
        model=frozen['models'][algorithm];p=model['parameters'];pred=[]
        for order in ORDERS:
            if algorithm=='A2':
                w=np.array(p['pair_coefficients']);v=float(p['intercept']+sum(w[a,b] for i,a in enumerate(order) for b in order[i+1:]))
            elif algorithm=='A3':
                v=float(p['intercept']+p['start'][order[0]]+p['end'][order[-1]]+sum(p['adjacency'][a][b] for a,b in zip(order,order[1:])))
            else:v=sc.transfer_score(p['matrices'],p['initial'],order)
            pred.append(v)
        fit_metric='S' if algorithm in ('A2','A3') else 'first_loss_survival'
        for rec in heldout_records:
            sid=rec['source_id']
            for j,order in enumerate(ORDERS):
                row=index.get((sid,order))
                if row is None:continue
                if algorithm in ('A2','A3'):observed=row['S']
                else:
                    sequence=[index.get((sid,order[:depth])) for depth in range(5)]
                    if any(r is None or not _states(r) for r in sequence):continue
                    m=len(_states(sequence[0]));alive=np.ones(m,dtype=bool)
                    for rr in sequence:alive &= np.array([state!='LOST' for state in _states(rr)])
                    observed=float(alive.mean())
                residuals.append({'algorithm':algorithm,'source_id':sid,'group_id':rec.get('group_id',sid),
                                  'partition':_partition(rec),'order':order_text(order),'target':fit_metric,
                                  'predicted':pred[j],'observed':observed,'residual':observed-pred[j],
                                  'absolute_error':abs(observed-pred[j])})
        rr=[r for r in residuals if r['algorithm']==algorithm]
        if rr:
            error=np.array([r['residual'] for r in rr]);summary.append({'algorithm':algorithm,'N_observed_order_rows':len(rr),
                  'N_sources':len(set(r['source_id'] for r in rr)),'target':fit_metric,
                  'MAE':float(np.mean(abs(error))),'RMSE':float(np.sqrt(np.mean(error**2))),
                  'max_observed_error':float(np.max(abs(error))),'search_gap':model.get('search_gap'),
                  'metric_warning':'Observed maximum is not a uniform population error bound'})
        if algorithm in ('A2','A3'):
            cert=sc.certify_surrogate(fit_y,ORDERS,pred,model['order'])
            certificates.append({'algorithm':algorithm,'kind':'measured_S_surrogate_plus_finite_policy_bound',
                                 'partition':'calibration_fit',**cert,'vacuous_regret':cert['regret_upper']>=1})
        else:
            proxy_opt=max(pred);certificates.append({'algorithm':algorithm,'kind':'fitted_first_loss_surrogate_search_only',
                  'incumbent':model['proxy_value'],'upper':model['proxy_upper'],'search_gap':model['search_gap'],
                  'enumerated_proxy_optimum':proxy_opt,'certificate_covers_proxy_optimum':bool(model['proxy_value']<=proxy_opt+1e-9 and model['proxy_upper']+1e-9>=proxy_opt),
                  'certifies_actual_joint_S':False})
            failures.extend([{'algorithm':algorithm,'kind':'unobserved_transition_column',**r} for r in p.get('unobserved_columns',[])])
    worst=sorted(residuals,key=lambda r:-r['absolute_error'])[:3]
    write_csv(out/'heldout_residuals.csv',residuals);write_csv(out/'prediction_error_summary.csv',summary)
    write_jsonl(out/'certificates.jsonl',certificates);write_jsonl(out/'assumption_failures.jsonl',failures)
    write_json(out/'largest_prediction_failures.json',worst)
    if summary:
        plt=_plots();fig,axes=plt.subplots(1,2,figsize=(10,4))
        for algorithm in ('A2','A3','A4','A5'):
            rr=[r for r in residuals if r['algorithm']==algorithm]
            if not rr:continue
            ax=axes[0 if algorithm in ('A2','A3') else 1]
            ax.scatter([r['predicted'] for r in rr],[r['observed'] for r in rr],s=8,alpha=.25,label=algorithm)
        for ax,target in zip(axes,['Joint score S','First-loss survival proxy target']):
            ax.plot([0,1],[0,1],'--',color='#555',lw=.8);ax.set(xlabel='Calibration-model prediction',ylabel='Held-out observation',title=target);ax.legend()
        fig.tight_layout();_save_figure(fig,review_dir,'06_heldout_surrogate_diagnostics')
    (out/'theory_diagnostics_report.md').write_text('\n\n'.join(['# Theory diagnostics',
        'Transfer matrices describe first-loss survival in standalone/embedded states. Their fitted search certificates do not imply guarantees for measured joint S.',
        'Nonlinear language context, restored facts, unobserved transition columns and finite evaluator accuracy can violate the state approximation. These records are retained.',
        'Held-out errors use observed order rows only. This is a descriptive error audit, not a uniform matrix law or population supremum.',
        'Finite-policy confidence radii apply to independent source records and the fixed static permutation class. Any clipping/vacuity is recorded in certificates.jsonl.',
        'SDPI coefficient eta is not identified. Fact-retention ratios are not mutual-information contraction coefficients.',
        json.dumps(summary,indent=2)])+'\n')
    return {'summary':summary,'certificates':certificates,'largest_failures':worst}


def synthetic_search_diagnostics(out_dir,review_dir):
    """Real measured mathematical checks and search timings, not LLM outputs."""
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(2026);checks=[];scaling=[]
    for n in (4,5,6,7):
        orders=list(itertools.permutations(range(n)));w=rng.normal(size=(n,n));start=rng.normal(size=n);end=rng.normal(size=n);adj=rng.normal(size=(n,n))
        pair=sc.pairwise_dp(w);truth=max(sum(w[a,b] for i,a in enumerate(p) for b in p[i+1:]) for p in orders)
        checks.append({'case':'pairwise_DP_exact','n':n,'estimate':pair.value,'exact':truth,'passed':abs(pair.value-truth)<1e-9})
        result=sc.adjacent_dp(start,end,adj);truth=max(start[p[0]]+end[p[-1]]+sum(adj[a,b] for a,b in zip(p,p[1:])) for p in orders)
        checks.append({'case':'adjacent_DP_exact','n':n,'estimate':result.value,'exact':truth,'passed':abs(result.value-truth)<1e-9})
        p=rng.uniform(size=(n,2,2));p=p/(p.sum(axis=1,keepdims=True)+.3);initial=np.array([.6,.4])
        truth=max(sc.transfer_score(p,initial,order) for order in orders)
        for name,result in [('beam4',sc.transfer_beam(p,initial,width=4)),('branch_bound',sc.transfer_branch_bound(p,initial,max_nodes=100000))]:
            checks.append({'case':name+'_admissibility','n':n,'estimate':result.value,'upper':result.upper,'exact':truth,
                           'passed':bool(result.value<=truth+1e-10 and result.upper>=truth-1e-10),'search_gap':result.gap})
    # Identical scalar contraction products cannot explain an order preference.
    scalar=np.array([[[.9]],[[.8]],[[.7]],[[.6]]]);values=[sc.transfer_score(scalar,[1],p) for p in ORDERS]
    checks.append({'case':'scalar_contraction_order_invariant','n':4,'estimate':max(values)-min(values),'passed':max(values)-min(values)<1e-12})
    scalar[2]=0;values=[sc.transfer_score(scalar,[1],p) for p in ORDERS]
    checks.append({'case':'all_information_deletion_no_positive_lower','n':4,'estimate':max(values),'passed':max(values)==0})
    # Exercise safety activation and fallback using frozen fake numeric records.
    fake=np.zeros((200,24));fake[:,0]=.1;fake[:100,1]=1;fake[100:,2]=1;length=np.array([100]*100+[140]*100)
    pos=sc.contextual_safe(fake,length,fake,length,ORDERS,minimum_leaf=10)
    negative=fake.copy();negative[:,0]=1;negative[:,1:]=0
    neg=sc.contextual_safe(fake,length,negative,length,ORDERS,minimum_leaf=10)
    checks.extend([{'case':'A7_positive_independent_safety','passed':bool(pos['activated'])},
                   {'case':'A7_negative_safety_fallback','passed':not neg['activated']}])
    for n in (4,6,8,10):
        p=rng.uniform(size=(n,2,2));p=p/(p.sum(axis=1,keepdims=True)+.3);initial=np.array([.6,.4])
        for name,fn in [('beam4',lambda:sc.transfer_beam(p,initial,width=4)),
                        ('branch_bound_10000',lambda:sc.transfer_branch_bound(p,initial,max_nodes=10000)),
                        ('pairwise_DP',lambda:sc.pairwise_dp(rng.normal(size=(n,n)))),
                        ('adjacent_DP',lambda:sc.adjacent_dp(rng.normal(size=n),rng.normal(size=n),rng.normal(size=(n,n))))]:
            t=time.perf_counter();result=fn();elapsed=time.perf_counter()-t
            scaling.append({'profile':'synthetic_search_not_LLM','algorithm':name,'roles':n,'wall_seconds':elapsed,
                            'search_gap':result.gap,'expanded_nodes':result.expanded,'live_states':2 if 'DP' not in name else None})
    write_csv(out/'toy_checks.csv',checks);write_csv(out/'search_scaling.csv',scaling)
    if not all(r['passed'] for r in checks):raise AssertionError('Mathematical toy check failed')
    plt=_plots();fig,axes=plt.subplots(1,2,figsize=(10,4))
    for name in dict.fromkeys(r['algorithm'] for r in scaling):
        rr=[r for r in scaling if r['algorithm']==name]
        axes[0].plot([r['roles'] for r in rr],[r['wall_seconds'] for r in rr],'o-',label=name)
        axes[1].plot([r['roles'] for r in rr],[r['expanded_nodes'] for r in rr],'o-',label=name)
    axes[0].set(xlabel='Synthetic roles',ylabel='Measured CPU seconds',yscale='log',title='Synthetic search runtime (no LLM calls)')
    axes[1].set(xlabel='Synthetic roles',ylabel='Expanded states/nodes',yscale='log',title='Different search state spaces');axes[0].legend(fontsize=8)
    fig.tight_layout();_save_figure(fig,review_dir,'08_synthetic_search_scaling')
    write_json(out/'synthetic_search_scope.json',{'profile':'controlled_synthetic_mathematical_models_not_natural_language',
               'all_checks_passed':True,'checks':len(checks),'SDPI_eta_for_actual_LLM':'not_identified'})
    return {'checks':checks,'scaling':scaling}


def runtime_report(state_rows,out_dir,review_dir,metadata=None):
    """Count actual inference sequences rather than attributing shared cache to A4."""
    groups=defaultdict(list)
    for row in state_rows:groups[row.get('stage',row.get('phase','unspecified'))].append(row)
    report=[]
    for stage,rows in groups.items():
        edits=[r for r in rows if int(r.get('hop',len(_prefix(r))))>0]
        # Cached responses retain the original response token counts/latency in
        # their bodies. Their logical editing steps must not inflate generation.
        inferred=[r for r in edits if not r.get('cache_hit',False) and int(r.get('output_tokens',0) or 0)>0]
        timing=(metadata or {}).get('timings',{}).get(stage,{})
        report.append({'stage':stage,'logical_editor_steps':len(edits),
                       'editor_input_sequences':int(timing.get('input_sequences',len(inferred))),
                       'stored_document_states':len(rows),
                       'scored_document_states':sum('generation_only' not in str(r.get('evaluation_status','')) for r in rows),
                       'cache_hits':int(timing.get('cache_hits',sum(bool(r.get('cache_hit')) for r in edits))),
                       'input_tokens':int(timing.get('input_tokens',sum(int(r.get('input_tokens',0) or 0) for r in inferred))),
                       'output_tokens':int(timing.get('output_tokens',sum(int(r.get('output_tokens',0) or 0) for r in inferred))),
                       'reported_inference_seconds':float(timing.get('inference_seconds',sum(float(r.get('elapsed_seconds',0) or 0) for r in inferred))),
                       'stage_wall_seconds':timing.get('wall_seconds'),
                       'model_generate_batch_invocations':timing.get('batch_invocations','not_identified_from_rows'),
                       'fact_evaluator_model_sequences':0 if (metadata or {}).get('evaluator_kind','rule_based')=='rule_based' else 'not_identified',
                       'rule_fact_decisions':sum(len(_states(r)) for r in rows),
                       'counter_basis':'runner_phase_counters' if timing else 'noncached_generated_state_rows'})
    out=Path(out_dir);write_csv(out/'runtime_breakdown.csv',report);write_json(out/'runtime_metadata.json',metadata or {})
    (out/'cost_ablation_report.md').write_text('\n\n'.join(['# Cost and ablation scope',
       'All algorithms share generated calibration/selection states. A full four-role prefix bank costs64 unique edits/source, compared with96 edits for24 uncached chains. This shared saving is not an algorithm-specific inference advantage.',
       'A deployed chosen-order policy makes four editor calls/source. CPU fitting and schedule selection are measured separately from GPU inference.',
       'Summed per-sequence latency is not batched wall time. Generate invocation counts, cold start, download, and peak GPU memory must come from runner metadata.',
       json.dumps(report,indent=2)])+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['select','final','lifetime','synthetic','theory','cost'])
    parser.add_argument('--records');parser.add_argument('--scores');parser.add_argument('--calibration-records');parser.add_argument('--calibration-scores')
    parser.add_argument('--frozen');parser.add_argument('--out',required=True);parser.add_argument('--review',required=True);parser.add_argument('--label',default='main')
    args=parser.parse_args();records=read_jsonl(args.records) if args.records else [];scores=read_jsonl(args.scores) if args.scores else []
    frozen=json.loads(Path(args.frozen).read_text()) if args.frozen else None
    if args.command=='select':result=fit_and_select(records,scores,args.out,args.review)
    elif args.command=='final':result=final_analysis(frozen,records,scores,args.out,args.review,args.label)
    elif args.command=='lifetime':result=lifetime_analysis(scores,args.out,args.review,records)
    elif args.command=='synthetic':result=synthetic_search_diagnostics(args.out,args.review)
    elif args.command=='theory':result=theory_diagnostics(frozen,read_jsonl(args.calibration_records),read_jsonl(args.calibration_scores),records,scores,args.out,args.review)
    else:result=runtime_report(scores,args.out,args.review)
    print(json.dumps({'status':'complete','command':args.command,'out':args.out,'version':VERSION}))


if __name__=='__main__':main()
