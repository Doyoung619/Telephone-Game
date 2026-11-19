"""Ordered, resumable local experiments. Run via supervise.py/start.sh."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

from engine import ROOT, Journal, BatchedEditor, atomic_json, digest
from data_eval import read_jsonl, evaluate, write_jsonl

REVIEW = ROOT / 'artifacts' / 'Review'
ROLES = 'ABCD'
ORDERS = list(itertools.permutations(range(4)))


class Study:
    def __init__(self, run, batch_size=32, mock=False, data_dir=None):
        self.run = Path(run); self.run.mkdir(parents=True, exist_ok=True)
        self.journal = Journal(self.run/'journal.sqlite')
        self.editor = BatchedEditor(self.journal, batch_size, mock=mock)
        self.mock = mock
        self.batch_size = batch_size
        self.start = time.time()
        self.phase = 'initialization'
        self.phase_start = time.time()
        self.phase_total = 0
        self.phase_done = 0
        self.timings = json.loads((self.run/'timings.json').read_text()) if (self.run/'timings.json').exists() else {}
        self.previous_status = json.loads((self.run/'status.json').read_text()) if (self.run/'status.json').exists() else {}
        self.data_dir=Path(data_dir) if data_dir is not None else ROOT/'data/paper'
        self.metadata = {**self.editor.metadata,
            'data_manifest':json.loads((self.data_dir/'manifest.json').read_text()),
            'runner_hash':digest(Path(__file__).read_text()),
            'evaluator_hash':digest((ROOT/'pipeline/data_eval.py').read_text()),
            'engine_hash':digest((ROOT/'pipeline/engine.py').read_text()),
            'analysis_hash':digest((ROOT/'pipeline/analysis.py').read_text()),
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'no_paid_inference':True, 'bootstrap_unit':'source_group', 'bootstrap_replicates':10000}
        previous=self.run/'environment.json'
        if previous.exists():
            old=json.loads(previous.read_text())
            for k in ('resolved_revision','evaluator_hash','engine_hash'):
                if old.get(k)!=self.metadata.get(k):
                    raise RuntimeError(f'Frozen {k} changed; use a new run directory instead of mixing results')
        atomic_json(previous, self.metadata)

    def status(self, **extra):
        wall = time.time()-self.phase_start
        rate = self.phase_done/wall if self.phase_done and wall else None
        obj=dict(status='running', pid=os.getpid(), phase=self.phase, phase_done=self.phase_done,
            phase_total=self.phase_total, phase_elapsed_seconds=wall,
            phase_remaining_seconds=(self.phase_total-self.phase_done)/rate if rate else None,
            updated_utc=datetime.now(timezone.utc).isoformat(), counters=self.editor.counters,
            timings=self.timings, **extra)
        atomic_json(self.run/'status.json',obj)

    @contextmanager
    def stage(self, name, total=0):
        previous = dict(self.timings.get(name, {}))
        if not previous and self.previous_status.get('phase') == name:
            previous = dict(wall_seconds=self.previous_status.get('phase_elapsed_seconds', 0),
                            state_count=self.previous_status.get('phase_done', 0))
            for key in self.editor.counters:
                accounted = sum(value.get(key, 0) for value in self.timings.values())
                previous[key] = max(0, self.editor.counters[key] - accounted)
        self.phase=name;self.phase_start=time.time();self.phase_total=total;self.phase_done=0
        before=dict(self.editor.counters)
        self.status()
        print(json.dumps({'event':'stage_start','stage':name,'total_states':total}),flush=True)
        try:
            yield
        except BaseException as e:
            self.status(status_detail='stage_failed',error=str(e))
            raise
        self.timings[name]=dict(wall_seconds=previous.get('wall_seconds',0)+time.time()-self.phase_start,
            state_count=max(previous.get('state_count',0),self.phase_done),
            **{k:previous.get(k,0)+self.editor.counters[k]-before[k] for k in before},
            completed_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(self.run/'timings.json',self.timings)
        self.status()
        print(json.dumps({'event':'stage_complete','stage':name,**self.timings[name]}),flush=True)
        self.progress()

    def progress(self):
        if self.mock:return
        script=ROOT/'pipeline/progress.py'
        if script.exists():
            try:
                subprocess.run([sys.executable,str(script),'--run',str(self.run),'--review',str(REVIEW)],
                               check=False,timeout=45, stdout=subprocess.DEVNULL)
            except (subprocess.TimeoutExpired, OSError):pass

    def state_id(self, phase, record, prefix, condition='bank', variant='main'):
        return digest(dict(phase=phase,source=record['source_id'],source_hash=digest(record['source_text']),
            prefix=prefix,condition=condition,variant=variant,requirements=record['requirements'],
            revision=self.editor.metadata['resolved_revision'],evaluator=self.metadata['evaluator_hash']))

    def score(self, text, record, gold, failed=False):
        result=evaluate(text,record,gold)
        if failed:
            result.update(F=0.,Q=0. if result.get('Q') is not None else None,
                          S=0. if result.get('S') is not None else None,strict_success=False,
                          evaluation_status='generation_failure_conservative_zero')
            result['fact_states']=['LOST']*len(gold.get('gold_facts',[]))
            for item in result.get('fact_statuses',[]):item.update(status='missing',state='LOST')
        return result

    def source_state(self, phase, record, gold, condition='bank', variant='main'):
        key=self.state_id(phase,record,[],condition,variant)
        old=self.journal.get(key)
        if old:return old
        state=dict(state_id=key, phase=phase, source_id=record['source_id'],group_id=record.get('group_id',record['source_id']),
            dataset=record['dataset'],split=record['split'],calibration_partition=record.get('calibration_partition'),
            prefix=[],hop=0,step=0,method=condition,condition=condition,variant=variant,text=record['source_text'],
            parent_state_id=None,input_tokens=0,output_tokens=0,elapsed_seconds=0.,status='ok',failure=False,
            model_profile='qwen15b',model_revision=self.editor.metadata['resolved_revision'],replicate_id=0,
            **self.score(record['source_text'],record,gold))
        self.journal.save(state);self.journal.commit()
        return state

    def execute_jobs(self, jobs):
        """Write every completed batch before moving to the next batch."""
        results={}
        for offset in range(0,len(jobs),self.batch_size):
            chunk=jobs[offset:offset+self.batch_size]
            fresh=[]
            for job in chunk:
                key=self.state_id(job['phase'],job['record'],job['prefix'],job['condition'],job.get('variant','main'))
                job['state_id']=key
                prior=self.journal.get(key)
                if prior:
                    results[key]=prior;self.phase_done+=1
                else:fresh.append(job)
            valid=[j for j in fresh if not j['parent'].get('failure')]
            generated=self.editor.edit_tasks([dict(text=j['parent']['text'],role=j['role'],
                max_words=j['record']['requirements']['max_words'],variant=j.get('variant','main')) for j in valid])
            output_by_id={j['state_id']:o for j,o in zip(valid,generated)}
            for job in fresh:
                p=job['record'];gold=job['gold'];parent=job['parent']
                output=output_by_id.get(job['state_id'],dict(text='',input_tokens=0,output_tokens=0,seconds=0.,
                    hit_token_cap=False,status='ancestor_generation_failure',cache_hit=False))
                failed=output['status']!='ok'
                state=dict(state_id=job['state_id'],phase=job['phase'],source_id=p['source_id'],
                    group_id=p.get('group_id',p['source_id']),dataset=p['dataset'],split=p['split'],
                    calibration_partition=p.get('calibration_partition'),prefix=job['prefix'],
                    permutation=job['prefix'] if len(job['prefix'])==4 else None,
                    hop=len(job['prefix']),step=len(job['prefix']),role=job['role'],
                    parent_state_id=parent['state_id'],method=job['condition'],condition=job['condition'],
                    variant=job.get('variant','main'),model_profile='qwen15b',
                    model_revision=self.editor.metadata['resolved_revision'],replicate_id=0,
                    **output, elapsed_seconds=output.get('seconds',0.), failure=failed,
                    **self.score(output['text'],p,gold,failed))
                self.journal.save(state);results[state['state_id']]=state;self.phase_done+=1
            self.journal.commit();self.status()
            print(json.dumps({'event':'progress','stage':self.phase,'done':self.phase_done,
                              'total':self.phase_total,'inferred':self.editor.counters['input_sequences']}),flush=True)
        return results

    def bank(self, records, golds, phase):
        nodes={}
        for record in records:
            nodes[(record['source_id'],())]=self.source_state(phase,record,golds[record['source_id']])
        for depth in range(1,5):
            jobs=[]
            for record in records:
                sid=record['source_id']
                for prefix in itertools.permutations(range(4),depth):
                    jobs.append(dict(record=record,gold=golds[sid],phase=phase,prefix=list(prefix),
                        parent=nodes[(sid,prefix[:-1])],role=ROLES[prefix[-1]],condition='bank'))
            result=self.execute_jobs(jobs)
            for job in jobs:nodes[(job['record']['source_id'],tuple(job['prefix']))]=result[job['state_id']]
        rows=self.journal.rows(phase)
        write_jsonl(self.run/f'{phase}_scores.jsonl',rows)
        return rows

    def chains(self, records, golds, policies, phase, hops=4, variant='main'):
        """policies maps source_id to condition -> list of 4 or 16 role indices."""
        nodes={}
        for record in records:
            for method in policies[record['source_id']]:
                nodes[(record['source_id'],method)]=self.source_state(phase,record,golds[record['source_id']],method,variant)
        for hop in range(1,hops+1):
            jobs=[]
            for record in records:
                sid=record['source_id']
                for method, sequence in policies[sid].items():
                    sequence=list(sequence)
                    sequence=(sequence*((hops+len(sequence)-1)//len(sequence)))[:hops]
                    role=sequence[hop-1]
                    jobs.append(dict(record=record,gold=golds[sid],phase=phase,prefix=sequence[:hop],
                        parent=nodes[(sid,method)],role='R' if role==4 else ROLES[role],condition=method,variant=variant))
            result=self.execute_jobs(jobs)
            for job in jobs:nodes[(job['record']['source_id'],job['condition'])]=result[job['state_id']]
        rows=self.journal.rows(phase)
        write_jsonl(self.run/f'{phase}_scores.jsonl',rows)
        return rows


def smoke_records(records):
    wanted={'calibration_fit':4,'calibration_safety':2,'selection':6}
    return [r for split,n in wanted.items() for r in [x for x in records if x['split']==split][:n]]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(ROOT/'runs/main'))
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--mock-check',action='store_true')
    p.add_argument('--smoke-only',action='store_true');args=p.parse_args()
    if args.mock_check:
        import tempfile
        from data_eval import prepare_data
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp)/'data';prepare_data(data,profile='paper',seed=11)
            records=read_jsonl(data/'public_documents.jsonl')
            golds={r['source_id']:r for r in read_jsonl(data/'private_gold.jsonl')}
            study=Study(Path(tmp)/'run',mock=True,data_dir=data)
            rec=records[:1]
            with study.stage('mock_bank',64):rows=study.bank(rec,golds,'mock_bank')
            assert len(rows)==65 and len([r for r in rows if len(r['prefix'])==4])==24
            seqs=study.editor.counters['input_sequences']
            with study.stage('mock_bank',64):study.bank(rec,golds,'mock_bank')
            assert study.editor.counters['input_sequences']==seqs
            prompt=study.editor.prompt(rec[0]['source_text'],'B',rec[0]['requirements']['max_words'])
            assert 'gold_facts' not in prompt and 'evaluation_queries' not in prompt
            assert str(rec[0]['requirements']['max_words']) in prompt
        atomic_json(ROOT/'runs/infrastructure_checks.json',dict(status='passed',
            checks=['64_prefixes_24_permutations','resume_zero_model_calls','private_data_prompt_whitelist','fixed_absolute_cap'],
            scope='mock_functionality_only_not_natural_language_results'))
        print('Infrastructure checks passed');return
    records=read_jsonl(ROOT/'data/paper/public_documents.jsonl')
    golds={r['source_id']:r for r in read_jsonl(ROOT/'data/paper/private_gold.jsonl')}
    import analysis
    study=Study(args.run,args.batch_size)
    plan={'basis':'Initial planning ranges only; current-run observations determine measured progress',
          'stages':[
            {'phase':'P0_evaluator_audit','planned_states':320,'minutes_low':0,'minutes_high':1},
            {'phase':'P1_algorithm_tournament','planned_states':7680,'minutes_low':10,'minutes_high':20},
            {'phase':'E1_final_performance','planned_states':1600,'minutes_low':2,'minutes_high':5},
            {'phase':'E2_lifetime','planned_states':12800,'minutes_low':8,'minutes_high':20},
            {'phase':'E3_theory_diagnostics','planned_states':0,'minutes_low':0,'minutes_high':1},
            {'phase':'E4_ablation_and_cost','planned_states':4000,'minutes_low':3,'minutes_high':10},
            {'phase':'E1_external_transfer','planned_states':2880,'minutes_low':5,'minutes_high':15}]}
    atomic_json(study.run/'timing_plan.json',plan)
    audit=json.loads((ROOT/'data/paper/audit/evaluator_report.json').read_text())
    with study.stage('P0_evaluator_audit',audit['cases']):
        manifest=json.loads((ROOT/'data/paper/manifest.json').read_text())
        if not manifest.get('audit_passed') or not audit.get('passed'):raise RuntimeError('Evaluator audit must pass before experiments')
        study.phase_done=audit['cases']
    tiny=smoke_records(records)
    with study.stage('P1_smoke',len(tiny)*64):
        smoke_rows=study.bank(tiny,golds,'P1_smoke')
    with study.stage('P1_smoke_analysis'):
        analysis.fit_and_select(tiny,smoke_rows,study.run/'smoke',study.run/'smoke_figures',metadata={**study.metadata,'provisional':True})
    if args.smoke_only:study.status(status_detail='smoke_complete');return
    train=[r for r in records if r['split']!='test']
    with study.stage('P1_algorithm_tournament',len(train)*64):
        bank_rows=study.bank(train,golds,'P1_algorithm_tournament')
    with study.stage('P1_selection_freeze'):
        path=study.run/'selection/selected_method.json'
        frozen=json.loads(path.read_text()) if path.exists() else analysis.fit_and_select(train,bank_rows,study.run/'selection',REVIEW,metadata=study.metadata)
        if frozen['status']!='frozen' or frozen.get('test_used_for_selection'):raise RuntimeError('Test access denied: missing valid frozen method')
    test=[r for r in records if r['split']=='test']
    policies={r['source_id']:analysis.comparison_orders(frozen,r) for r in test}
    with study.stage('E1_final_performance',sum(len(p)*4 for p in policies.values())):
        final_rows=study.chains(test,golds,policies,'E1_final_performance')
    with study.stage('E1_final_analysis'):
        analysis.final_analysis(frozen,test,final_rows,study.run/'final',REVIEW)
    lifetime={}
    for r in test:
        sid=r['source_id'];base=policies[sid]
        rng=random.Random(int(digest('lifetime2026:'+sid),16))
        lifetime[sid]={k:list(base[k])*4 for k in ('selected','best_fixed','intuitive','random_once')}
        lifetime[sid]['random_per_cycle']=[a for _ in range(4) for a in rng.sample(range(4),4)]
        lifetime[sid]['faithful_relay']=[4]*16
        for a in range(4):lifetime[sid]['repeat_'+ROLES[a]]=[a]*16
    with study.stage('E2_lifetime',sum(len(p)*16 for p in lifetime.values())):
        lifetime_rows=study.chains(test,golds,lifetime,'E2_lifetime',hops=16)
    with study.stage('E2_lifetime_analysis'):
        analysis.lifetime_analysis(lifetime_rows,study.run/'lifetime',REVIEW,records=test)
    with study.stage('E3_theory_diagnostics'):
        analysis.theory_diagnostics(frozen,[r for r in train if r['split'].startswith('calibration')],bank_rows,test,final_rows,study.run/'theory',REVIEW)
    run_ablations(study,analysis,train,bank_rows,test,golds,frozen)
    run_external(study,analysis,frozen)
    with study.stage('E4_cost_and_search_scaling'):
        analysis.runtime_report(study.journal.rows(),study.run/'cost',REVIEW,metadata={**study.metadata,'counters':study.editor.counters,'timings':study.timings})
        analysis.synthetic_search_diagnostics(study.run/'cost',REVIEW)
    study.status(status_detail='all_complete')
    status=json.loads((study.run/'status.json').read_text());status['status']='complete'
    atomic_json(study.run/'status.json',status);study.progress()
    print('All registered experiments completed',flush=True)


def run_ablations(study,analysis,train,bank_rows,test,golds,frozen):
    """The registry is frozen before any ablation test output is scored."""
    small=test[:24]
    registry=dict(frozen_policy_hash=frozen['policy_hash'],source_ids=[r['source_id'] for r in small],
        calibration_sizes=[6,12,24],compression_fractions=[.6,.75,.9],format=['one_to_eight','exactly_three'],
        registered_before_test=True)
    atomic_json(study.run/'ablation_registry.json',registry)
    all_rows=[]
    with study.stage('E4_algorithm_ablations'):
        models=analysis.prepare_ablation_models(frozen,[r for r in train if r['split'].startswith('calibration')],bank_rows,study.run/'ablation')
        policies={r['source_id']:analysis.ablation_orders(frozen,r,models) for r in small}
        study.phase_total=sum(len(x)*4 for x in policies.values())
        all_rows.extend(study.chains(small,golds,policies,'E4_algorithm_ablations'))
    exclusions=[]
    for fraction in (.6,.9):
        changed=[]
        for r in small:
            x=deepcopy(r);x['requirements']['max_words']=int(fraction*len(r['source_text'].split()))
            if len(golds[r['source_id']]['feasibility_witness'].split())>x['requirements']['max_words']:
                exclusions.append(dict(source_id=r['source_id'],fraction=fraction,status='infeasible_private_witness'));continue
            changed.append(x)
        phase=f'E4_compression_{fraction}'
        with study.stage(phase,len(changed)*4):
            policies={r['source_id']:{f'compression_{fraction}':analysis.comparison_orders(frozen,r)['selected']} for r in changed}
            all_rows.extend(study.chains(changed,golds,policies,phase))
    hard=deepcopy(small)
    for r in hard:r['requirements'].update(bullet_min=3,bullet_max=3,min_bullets=3,max_bullets=3)
    with study.stage('E4_three_bullets',len(hard)*4):
        policies={r['source_id']:{'exactly_three_bullets':analysis.comparison_orders(frozen,r)['selected']} for r in hard}
        all_rows.extend(study.chains(hard,golds,policies,'E4_three_bullets',variant='three_bullets'))
    atomic_json(study.run/'ablation_infeasible.json',exclusions)
    analysis.ablation_analysis(frozen,small,all_rows,models,study.run/'ablation',REVIEW)


def run_external(study,analysis,frozen):
    import external_data
    external=ROOT/'data/external'
    # External outcome scores never enter the frozen main policy or select it again.
    records=read_jsonl(external/'public_documents.jsonl')
    golds={r['source_id']:r for r in read_jsonl(external/'private_gold.jsonl')}
    summaries=[]
    for dataset in sorted({r['dataset'] for r in records}):
        test=[r for r in records if r['dataset']==dataset and r['split']=='test']
        policies={r['source_id']:{'zero_shot_'+k:v for k,v in analysis.comparison_orders(frozen,r).items()
                                 if k in ('selected','best_fixed','intuitive','random_once')} for r in test}
        phase='E1_external_'+dataset
        with study.stage(phase,sum(len(x)*4 for x in policies.values())):
            rows=study.chains(test,golds,policies,phase)
        result=external_data.analyze_outputs(rows,test,golds,study.run/'external'/dataset)
        summaries.append(result)
    atomic_json(study.run/'external/summary.json',summaries)


if __name__=='__main__':
    main()
