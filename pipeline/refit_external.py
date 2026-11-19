"""Post-main preregistered external A1 refitting, without algorithm reselection.

The external objective is each dataset's F-only QA/coverage measurement. It is
separate from the frozen Synthetic-6 S objective. Importing this module does not
load a GPU model. Call run_refit only after the original main run has completed.
"""
from __future__ import annotations
import gc
import itertools
import json
import math
from pathlib import Path
import time
from datetime import datetime, timezone

ORDERS=tuple(itertools.permutations(range(4)))


def fit_a1(terminal_scores,fit_source_ids):
    """Pure fitting logic; accepts only calibration rows and no test records."""
    requested=set(fit_source_ids);bank={}
    for row in terminal_scores:
        sid=row['source_id']
        if sid not in requested:raise ValueError('Non-fit source reached external A1 fitting')
        prefix=tuple(row['prefix'])
        if prefix not in ORDERS:raise ValueError('Expected four-role terminal permutation')
        bank[(sid,prefix)]=row.get('F')
    for sid in requested:
        if any((sid,order) not in bank for order in ORDERS):raise ValueError('Incomplete calibration terminal bank: '+sid)
    eligible=sorted(sid for sid in requested if all(bank[(sid,o)] is not None and math.isfinite(bank[(sid,o)]) for o in ORDERS))
    excluded=sorted(requested-set(eligible))
    if not eligible:raise ValueError('No consistently scorable calibration source; external A1 refit is undefined')
    report=[{'order':list(order),'order_label':''.join('ABCD'[x] for x in order),
             'mean_F':sum(bank[(sid,order)] for sid in eligible)/len(eligible),'n_fit':len(eligible)} for order in ORDERS]
    # ORDERS is lexicographic; max returns its first encountered exact tie.
    winner=max(report,key=lambda x:x['mean_F'])
    return {'algorithm':'A1','order':winner['order'],'order_label':winner['order_label'],
            'calibration_mean_F':winner['mean_F'],'eligible_fit_source_ids':eligible,'excluded_fit_source_ids':excluded,
            'order_scores':report,'tie_break':'exact mean-F tie then lexicographic role indices',
            'eligibility':'same sources with non-null finite F for all24 permutations'}


def _load_gold(path,allowed):
    allowed=set(allowed);result={}
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():continue
            item=json.loads(line)
            if item['source_id'] in allowed:result[item['source_id']]=item
    if set(result)!=allowed:raise ValueError('Missing requested private source records')
    return result


def _utc():return datetime.now(timezone.utc).isoformat()


def _run_refit_locked(run_dir):
    from engine import ROOT, atomic_json, digest
    from data_eval import read_jsonl,write_jsonl
    import external_data
    run=Path(run_dir).resolve();dest=run/'external';dest.mkdir(parents=True,exist_ok=True)
    completion=dest/'refit_completion.json'
    if completion.exists():
        prior=json.loads(completion.read_text())
        if prior.get('status')=='complete':return prior
    prereg_path=dest/'refit_preregistration.json'
    if not prereg_path.exists():raise RuntimeError('Missing external-refit preregistration; do not infer a protocol after test')
    prereg=json.loads(prereg_path.read_text())
    mainpolicy_path=run/'selection/selected_method.json';mainpolicy=json.loads(mainpolicy_path.read_text())
    if mainpolicy.get('status')!='frozen' or mainpolicy.get('algorithm')!='A1':raise RuntimeError('Extension supports only the already frozen A1 winner')
    if prereg.get('main_algorithm')!='A1' or prereg.get('algorithm_reselection') is not False:raise RuntimeError('Preregistered algorithm must be fixed A1')
    oldstatus=json.loads((run/'status.json').read_text())
    backup=dest/'main_status_before_refit.json'
    if not backup.exists():
        if oldstatus.get('status')!='complete':raise RuntimeError('Main experiments must finish before loading the refit editor')
        atomic_json(backup,oldstatus)
    elif oldstatus.get('status')!='complete' and not completion.exists():
        raise RuntimeError('Main is active and no resumable refit marker exists')
    started=time.perf_counter();start_counters=dict(oldstatus.get('counters',{}));study=None
    original_environment=(run/'environment.json').read_bytes()
    for path in [dest/'summary.json', *dest.glob('*/summary.json')]:
        saved=path.with_name('primary_zero_shot_summary.json')
        if path.exists() and not saved.exists():saved.write_bytes(path.read_bytes())
    atomic_json(completion,{'status':'running','started_utc':_utc(),'preregistration_sha256':digest(prereg_path.read_text())})
    atomic_json(run/'status.json',{**oldstatus,'status':'running','phase':'E1_refit_initialization',
        'status_detail':'external_refit_extension','updated_utc':_utc()})
    try:
        from run_all import Study
        class GenerationOnlyStudy(Study):
            def score(self,text,record,gold,failed=False):
                # All authoritative calibration measurements happen ONLY below,
                # on the 24 terminal outputs. Intermediate prefixes stay unscored.
                return {'F':None,'Q':None,'S':None,'strict_success':False,'fact_statuses':[],
                    'fact_states':[],'requirement_pass':{},'unsupported_claims':None,
                    'evaluation_status':'external_refit_generation_only; authoritative_terminal_scores_separate'}
        study=GenerationOnlyStudy(run,batch_size=int(oldstatus.get('batch_size',32)))
        atomic_json(dest/'refit_environment.json',{**study.metadata,'refit_module_sha256':digest(Path(__file__).read_text()),
            'external_evaluator_sha256':digest(Path(external_data.__file__).read_text()),'main_environment_preserved':True})
        (run/'environment.json').write_bytes(original_environment)
        for key in study.editor.counters:
            study.editor.counters[key]=max(study.editor.counters[key],start_counters.get(key,0))
        start_counters=dict(study.editor.counters)
        records=read_jsonl(ROOT/'data/external/public_documents.jsonl')
        private_path=ROOT/'data/external/private_gold.jsonl';results=[];fits={}
        datasets=sorted({r['dataset'] for r in records})
        for dataset in datasets:
            fit=[r for r in records if r['dataset']==dataset and r['split']=='calibration_fit']
            fit_ids=[r['source_id'] for r in fit]
            if fit_ids!=prereg['fit_source_ids'].get(dataset):raise RuntimeError('Calibration IDs differ from preregistration: '+dataset)
            goldfit=_load_gold(private_path,fit_ids);byid={r['source_id']:r for r in fit}
            data_dest=dest/dataset;data_dest.mkdir(exist_ok=True)
            phase='E1_refit_'+dataset+'_calibration'
            with study.stage(phase,len(fit)*64):bankrows=study.bank(fit,goldfit,phase)
            terminal=[r for r in bankrows if len(r.get('prefix',[]))==4]
            cachefile=data_dest/'refit_calibration_score_cache.jsonl'
            cache={r['evaluation_key']:r for r in read_jsonl(cachefile)} if cachefile.exists() else {}
            scored=[];qa_before=external_data._QA.sequences if external_data._QA is not None else 0
            calls_before=external_data._QA.calls if external_data._QA is not None else 0
            with study.stage('E1_refit_'+dataset+'_authoritative_scoring',len(terminal)):
                for row in terminal:
                    sid=row['source_id'];key=digest({'source_id':sid,'text':row['text'],
                        'external_evaluator':digest(Path(external_data.__file__).read_text())})
                    if key not in cache:
                        cache[key]={'evaluation_key':key,**external_data.evaluate(row['text'],byid[sid],goldfit[sid])}
                        write_jsonl(cachefile,list(cache.values()))
                    score=cache[key]
                    f=score.get('F')
                    if row.get('failure') and f is not None:f=0.
                    scored.append({'state_id':row['state_id'],'source_id':sid,'split':'calibration_fit',
                        'prefix':row['prefix'],'F':f,'Q':None,'S':None,'measurement_scope':score.get('measurement_scope'),
                        'evaluation_status':score.get('evaluation_status'),'generation_failure':bool(row.get('failure')),
                        'score_cache_key':key})
                    study.phase_done+=1
                    if study.phase_done%16==0:study.status()
                write_jsonl(data_dest/'refit_calibration_terminal_scores.jsonl',scored)
            frozen_path=data_dest/'refit_selected_method.json'
            fit_result=fit_a1(scored,fit_ids)
            frozen={**fit_result,'status':'frozen','dataset':dataset,'objective':'external_F_retention_only',
                'fit_partition':'calibration_fit','fit_source_ids':fit_ids,'fit_group_ids':[r['group_id'] for r in fit],
                'calibration_safety_used':False,'selection_used':False,'test_used_for_fit':False,
                'algorithm_reselection':False,'main_frozen_algorithm':'A1','main_policy_sha256':digest(mainpolicy_path.read_text()),
                'preregistration_sha256':digest(prereg_path.read_text()),'refit_module_sha256':digest(Path(__file__).read_text()),
                'external_evaluator_sha256':digest(Path(external_data.__file__).read_text()),
                'calibration_scores_sha256':digest((data_dest/'refit_calibration_terminal_scores.jsonl').read_text()),
                'QA_input_sequences':(external_data._QA.sequences-qa_before) if external_data._QA is not None else 0,
                'QA_batch_invocations':(external_data._QA.calls-calls_before) if external_data._QA is not None else 0,
                'frozen_utc':_utc(),'scope':prereg['limitations']}
            if frozen_path.exists():
                oldfrozen=json.loads(frozen_path.read_text())
                if oldfrozen['order']!=frozen['order'] or oldfrozen['calibration_scores_sha256']!=frozen['calibration_scores_sha256']:
                    raise RuntimeError('Resumed calibration changed an already frozen external policy')
                frozen=oldfrozen
            else:atomic_json(frozen_path,frozen)
            fits[dataset]=frozen
            # Test gold is loaded only AFTER the calibration-only freeze above.
            test=[r for r in records if r['dataset']==dataset and r['split']=='test']
            if set(fit_ids)&{r['source_id'] for r in test}:raise RuntimeError('Fit/test source overlap')
            goldtest=_load_gold(private_path,[r['source_id'] for r in test])
            policies={r['source_id']:{'within_dataset_refit':frozen['order']} for r in test}
            test_phase='E1_refit_'+dataset+'_test'
            with study.stage(test_phase,len(test)*4):refitrows=study.chains(test,goldtest,policies,test_phase)
            oldrows=study.journal.rows('E1_external_'+dataset)
            with study.stage('E1_refit_'+dataset+'_test_analysis',len(test)):
                result=external_data.analyze_outputs(oldrows+refitrows,test,goldtest,data_dest)
                result.update(selection='Preregistered fixed A1, external calibration_fit F-only refit frozen before test; zero-shot comparison retained.',
                    within_dataset_refit=frozen,main_algorithm_reselected=False)
                atomic_json(data_dest/'summary.json',result);study.phase_done=len(test)
            results.append(result)
        atomic_json(dest/'summary.json',results)
        completed={'status':'complete','completed_utc':_utc(),'wall_seconds':time.perf_counter()-started,
            'algorithm':'A1','algorithm_reselection':False,'preregistration_sha256':digest(prereg_path.read_text()),
            'objective':'dataset-specific external F; separate from frozen synthetic S','datasets':fits,
            'counters_total':dict(study.editor.counters),
            'extension_counter_delta':{k:study.editor.counters[k]-start_counters.get(k,0) for k in study.editor.counters},
            'main_environment_preserved':True,'summary_path':str(dest/'summary.json')}
        study.status(status_detail='all_complete_including_external_refit')
        finalstatus=json.loads((run/'status.json').read_text());finalstatus.update(status='complete',external_refit_complete=True)
        atomic_json(run/'status.json',finalstatus);study.progress()
    except BaseException as error:
        atomic_json(completion,{'status':'failed','failed_utc':_utc(),'error':repr(error),
            'preregistration_sha256':digest(prereg_path.read_text()),'resumable':True})
        import traceback
        failure=json.loads((run/'status.json').read_text())
        failure.update(status='failed',status_detail='external_refit_failed',error=repr(error),traceback=traceback.format_exc(),updated_utc=_utc())
        atomic_json(run/'status.json',failure)
        raise
    finally:
        (run/'environment.json').write_bytes(original_environment)
        if study is not None:
            study.journal.commit();study.journal.db.close()
            if hasattr(study.editor,'model'):del study.editor.model
            study=None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():torch.cuda.empty_cache()
        except ImportError:pass
    # Final success marker is published only after timings, environment, status,
    # journal and GPU cleanup have completed. The wrapper still owns the lock.
    atomic_json(completion,completed)
    return completed


def run_refit(run_dir):
    import fcntl
    run=Path(run_dir).resolve();run.mkdir(parents=True,exist_ok=True)
    with (run/'supervisor.lock').open('a') as lock:
        # Primary status may reach complete a moment before its supervisor exits.
        # Waiting for the same lock prevents concurrent GPU ownership/resume.
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:return _run_refit_locked(run)
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);a=p.parse_args()
    print(json.dumps(run_refit(a.run),indent=2))
