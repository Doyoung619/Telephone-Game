"""Rebuild current reports from persisted observations, with no editor inference."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

from engine import ROOT, Journal, atomic_json, digest


def finalize(run):
    import analysis
    from external_summary_plot import plot_external_transfer
    run=Path(run);review=ROOT/'artifacts/Review'
    environment=json.loads((run/'environment.json').read_text())
    timings=json.loads((run/'timings.json').read_text())
    status=json.loads((run/'status.json').read_text())
    if status.get('status')!='complete':
        raise RuntimeError('Finalization requires completed registered experiment stages')
    rows=Journal(run/'journal.sqlite').rows()
    frozen=json.loads((run/'selection/selected_method.json').read_text())
    test_records=[r for r in analysis.read_jsonl(ROOT/'data/paper/public_documents.jsonl') if r['split']=='test']
    analysis.final_analysis(frozen,test_records,[r for r in rows if r['phase']=='E1_final_performance'],run/'final',review)
    analysis.runtime_report(rows,run/'cost',review,metadata={**environment,'timings':timings,
                                                          'counters':status.get('counters',{})})
    plot_external_transfer(run/'external',review)
    # The combined cross-dataset figure is the primary external review figure.
    extra=run/'external/individual_figures';extra.mkdir(parents=True,exist_ok=True)
    for path in review.glob('07_external_*'):
        shutil.move(str(path),str(extra/path.name))
    code=run/'provenance/pipeline';code.mkdir(parents=True,exist_ok=True)
    for path in (ROOT/'pipeline').glob('*.py'):shutil.copy2(path,code/path.name)
    frozen_hash=digest((run/'selection/selected_method.json').read_text())
    manifest=dict(status='complete',completed_utc=status.get('updated_utc'),
        algorithm=frozen['algorithm'],selected_method_sha256=frozen_hash,
        order=frozen['model'].get('order'),criterion='heldout-selection mean controlled-evaluator S',
        limitations=['Controlled evaluator has observed paraphrase/tense errors; ranking is not ground-truth semantic superiority',
                     'External metrics are separately scoped; WebNLG relation coverage is limited',
                     'Alternative editor model profiles are not run',
                     'Test random baseline is one prespecified uniform draw per source; full24 random expectation/oracle not run'],
        analysis_source_sha256=digest((ROOT/'pipeline/analysis.py').read_text()),
        stages=timings,figures=sorted(p.name for p in review.glob('*.png')),
        environment=environment)
    from collections import Counter
    failures=Counter((r['phase'],r.get('condition'),r['status']) for r in rows if r.get('failure'))
    failure_rows=[dict(phase=p,condition=c,status=s,count=n) for (p,c,s),n in failures.items()]
    analysis.write_csv(run/'cost/generation_failures.csv',failure_rows)
    manifest['generation_failures']=failure_rows
    if (run/'ablation_infeasible.json').exists():
        manifest['ablation_feasibility_exclusions']=json.loads((run/'ablation_infeasible.json').read_text())
    public={r['source_id']:r for r in analysis.read_jsonl(ROOT/'data/paper/public_documents.jsonl')}
    selected=[r for r in rows if r['phase']=='E1_final_performance' and r.get('condition')=='selected' and r['hop']==4]
    selected.sort(key=lambda r:(r['S'],r['source_id']))
    examples=[]
    for label,subset in [('lowest_score',selected[:3]),('highest_score',list(reversed(selected[-3:])) )]:
        for r in subset:
            examples.append(dict(case=label,source_id=r['source_id'],source_text=public[r['source_id']]['source_text'],
                edited_text=r['text'],S=r['S'],F=r['F'],Q=r['Q'],fact_statuses=r['fact_statuses'],
                selection_rule='Three lowest/highest frozen-evaluator S; illustrative, not manually relabeled truth'))
    atomic_json(run/'final/qualitative_text_pairs.json',examples)
    lines=['Telephone Game experiment completion',
           'Frozen algorithm: '+frozen['algorithm'],
           'Criterion: maximum independent selection mean S from the frozen controlled relation evaluator.',
           'Statistical tie/uncertainty: consult selection_report.csv; a rank is not proof of superiority.',
           '', 'Measured wall times:']
    for name,v in timings.items():lines.append(f'{name}: {v["wall_seconds"]/60:.2f} min')
    lines+=['','Scope limitations:']+manifest['limitations']
    (run/'COMPLETED.txt').write_text('\n'.join(lines)+'\n')
    make_cost_figure(rows,timings,run,review)
    make_contact_sheet(review)
    manifest['figures']=sorted(p.name for p in review.glob('*.png'))
    manifest['external_evaluation']={}
    for path in (run/'external').glob('*/summary.json'):
        item=json.loads(path.read_text())
        manifest['external_evaluation'][item['dataset']]={k:item.get(k) for k in ['evaluation_wall_seconds','QA_batch_invocations','QA_input_sequences','QA_revision']}
    evaluator_cost=[]
    for path in (run/'external').glob('*/primary_zero_shot_summary.json'):
        obj=json.loads(path.read_text())
        evaluator_cost.append(dict(dataset=obj['dataset'],phase='zero_shot_external_rescoring',
            wall_seconds=obj.get('evaluation_wall_seconds'),QA_input_sequences=obj.get('QA_input_sequences'),
            QA_batch_invocations=obj.get('QA_batch_invocations')))
    for path in (run/'external').glob('*/summary.json'):
        obj=json.loads(path.read_text())
        evaluator_cost.append(dict(dataset=obj['dataset'],phase='refit_test_external_rescoring',
            wall_seconds=obj.get('evaluation_wall_seconds'),QA_input_sequences=obj.get('QA_input_sequences'),
            QA_batch_invocations=obj.get('QA_batch_invocations')))
    for path in (run/'external').glob('*/refit_selected_method.json'):
        obj=json.loads(path.read_text())
        stage='E1_refit_'+obj['dataset']+'_authoritative_scoring'
        evaluator_cost.append(dict(dataset=obj['dataset'],phase='refit_calibration_external_scoring',
            wall_seconds=timings.get(stage,{}).get('wall_seconds'),QA_input_sequences=obj.get('QA_input_sequences'),
            QA_batch_invocations=obj.get('QA_batch_invocations')))
    analysis.write_csv(run/'cost/external_evaluator_cost.csv',evaluator_cost)
    atomic_json(run/'completion_manifest.json',manifest)
    return manifest


def make_cost_figure(rows,timings,run,review):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['7-algorithm comparison','Final performance','16-hop lifetime','Theory diagnostics',
            'Ablation + search cost','External zero-shot','External refit']
    def group(phase):
        if phase.startswith('P1_'):return 0
        if phase.startswith('E1_final'):return 1
        if phase.startswith('E2_'):return 2
        if phase.startswith('E3_'):return 3
        if phase.startswith('E4_'):return 4
        if phase.startswith('E1_external_'):return 5
        if phase.startswith('E1_refit_'):return 6
        return None
    seconds=np.zeros(7);inferred=np.zeros(7);logical=np.zeros(7)
    for phase,item in timings.items():
        i=group(phase)
        if i is not None:
            seconds[i]+=item.get('wall_seconds',0);inferred[i]+=item.get('input_sequences',0)
    for path in (run/'external').glob('*/primary_zero_shot_summary.json'):
        seconds[5]+=json.loads(path.read_text()).get('evaluation_wall_seconds',0)
    for row in rows:
        i=group(row['phase'])
        if i is not None and row['hop']>0:logical[i]+=1
    fig,axes=plt.subplots(1,2,figsize=(12,4.8));y=np.arange(7)
    axes[0].barh(y,seconds/60,color='#257d91')
    for i,v in enumerate(seconds/60):axes[0].text(v+.03,i,f'{v:.2f}',va='center',fontsize=8)
    axes[0].set(yticks=y,yticklabels=labels,xlabel='Measured stage minutes',title='Completed experiment wall time')
    axes[0].invert_yaxis();axes[0].set_xlim(0,max(seconds/60)*1.18)
    axes[1].barh(y-.17,logical,height=.32,label='Logical editor steps',color='#bfd0d6')
    axes[1].barh(y+.17,inferred,height=.32,label='Actual editor input sequences',color='#257d91')
    axes[1].set(yticks=y,yticklabels=labels,xlabel='Editing transitions / actual model inputs',title='Shared-prefix and text-cache reuse')
    axes[1].invert_yaxis();axes[1].legend(fontsize=8)
    fig.text(.5,.01,'Stage wall times include scoring where instrumented; setup and between-stage overhead are excluded. QA inference is recorded separately.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1))
    for ext in ('png','pdf'):fig.savefig(review/f'11_measured_experiment_cost.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def make_contact_sheet(review):
    from PIL import Image, ImageDraw, ImageOps
    paths=[p for p in sorted(Path(review).glob('*.png')) if p.name!='00_REVIEW_OVERVIEW.png']
    if not paths:return
    width=1600;cellw=800;cellh=490
    sheet=Image.new('RGB',(width,((len(paths)+1)//2)*cellh),'#f0f3f6')
    draw=ImageDraw.Draw(sheet)
    for i,path in enumerate(paths):
        pic=Image.open(path).convert('RGB');pic.thumbnail((cellw-28,cellh-48))
        x=(i%2)*cellw+(cellw-pic.width)//2;y=(i//2)*cellh+35
        sheet.paste(pic,(x,y));draw.text(((i%2)*cellw+14,(i//2)*cellh+10),path.stem,fill='#183346')
    sheet.save(Path(review)/'00_REVIEW_OVERVIEW.png')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(ROOT/'runs/main'));p.add_argument('--wait',action='store_true')
    p.add_argument('--skip-external-refit',action='store_true',help='Report the completed main study without running the legacy A1 external refit.')
    a=p.parse_args()
    import fcntl,os
    run_path=Path(a.run);run_path.mkdir(parents=True,exist_ok=True)
    final_lock=(run_path/'finalizer.lock').open('a')
    try:fcntl.flock(final_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Another finalizer is already active')
    (run_path/'finalizer.pid').write_text(str(os.getpid()))
    if a.wait:
        failed_snapshots=0
        for _ in range(1440):
            path=Path(a.run)/'status.json'
            status=json.loads(path.read_text()) if path.exists() else {}
            if status.get('status')=='complete':break
            if (run_path/'external/refit_completion.json').exists() and (run_path/'external/main_status_before_refit.json').exists():break
            failed_snapshots=failed_snapshots+1 if status.get('status')=='failed' else 0
            if failed_snapshots>=3:raise SystemExit('Experiment failed; finalization awaits successful resume')
            time.sleep(30)
        else:raise SystemExit('Finalization timeout')
    refit_path=Path(a.run)/'external/refit_completion.json'
    if not a.skip_external_refit and (not refit_path.exists() or json.loads(refit_path.read_text()).get('status')!='complete'):
        from refit_external import run_refit
        run_refit(a.run)
    result=finalize(a.run)
    if a.skip_external_refit:
        result['external_refit_requested']=False
        result['limitations'].append('External refit was explicitly skipped by the reporting command.')
        atomic_json(Path(a.run)/'completion_manifest.json',result)
    print(json.dumps({'status':result['status'],'algorithm':result['algorithm'],'figures':result['figures']}))
