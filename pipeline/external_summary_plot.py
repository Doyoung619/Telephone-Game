"""Plot observed external-transfer reports, keeping unlike metrics separate.

Reads external_data.analyze_outputs summary JSONs. Does not invoke an evaluator,
fit a method, alter results, or fill unmeasured datasets with predicted scores.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def plot_external_transfer(root,review):
    root=Path(root);review=Path(review);reports=[]
    for path in sorted(root.glob('*/summary.json')):
        obj=json.loads(path.read_text())
        if isinstance(obj,dict) and 'methods' in obj and 'dataset' in obj:reports.append(obj)
    if not reports:return {'status':'no_measured_external_reports','figures_written':False}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,len(reports),figsize=(max(6,4.7*len(reports)),7),squeeze=False)
    table=[]
    for col,report in enumerate(reports):
        ds=report['dataset'];metric='QA exact match' if 'squad' in ds.lower() else 'Relation/alias coverage proxy'
        methods=report['methods'];names=[m for m in methods if m!='source']
        valid=[name for name in ['source']+names if methods[name].get('mean') is not None]
        labels=[m.replace('zero_shot_','').replace('_',' ') for m in valid]
        values=np.array([methods[m]['mean'] for m in valid]);cis=np.array([methods[m].get('ci95',[methods[m]['mean']]*2) for m in valid])
        ax=axes[0,col];colors=['#A8B2B8' if m=='source' else '#15658A' if m.endswith('selected') else '#80A6B0' for m in valid]
        if valid:
            ax.bar(np.arange(len(valid)),values,color=colors)
            ax.errorbar(np.arange(len(valid)),values,yerr=np.maximum(0,np.stack([values-cis[:,0],cis[:,1]-values])),fmt='none',color='#253944',capsize=3)
        ax.set(xticks=np.arange(len(valid)),xticklabels=labels,ylabel=metric,ylim=(0,1.03),title=ds+' — frozen order transfer')
        ax.tick_params(axis='x',labelrotation=30,labelsize=8)
        for name in valid:
            st=methods[name];table.append({'dataset':ds,'method':name,'metric':metric,'N_sources':st.get('n'),
                                           'mean':st['mean'],'CI_low':st.get('ci95',[None,None])[0],'CI_high':st.get('ci95',[None,None])[1],
                                           'change_from_source':st.get('change_from_source',{}).get('mean'),
                                           'measurement_scope':report.get('measurement_scope')})
        paired=[m for m in names if methods[m].get('change_from_source',{}).get('mean') is not None]
        ax=axes[1,col];ax.axhline(0,color='#555',lw=.8)
        if paired:
            values=np.array([methods[m]['change_from_source']['mean'] for m in paired]);cis=np.array([methods[m]['change_from_source']['ci95'] for m in paired])
            ax.errorbar(np.arange(len(paired)),values,yerr=np.maximum(0,np.stack([values-cis[:,0],cis[:,1]-values])),fmt='o',color='#15658A',capsize=4)
        ax.set(xticks=np.arange(len(paired)),xticklabels=[m.replace('zero_shot_','').replace('_',' ') for m in paired],
               ylabel='Paired metric change from original source',title='Source-group bootstrap 95% intervals')
        ax.tick_params(axis='x',labelrotation=30,labelsize=8)
    fig.text(.5,.005,'External raw-condition measures: Q/S not reported. Coverage proxies and QA exact match are separate measurements; no pooled score.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,1));review.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(review/f'09_external_transfer_comparison.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    with (root/'cross_dataset_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]) if table else ['dataset']);writer.writeheader();writer.writerows(table)
    result={'status':'complete','datasets':[r['dataset'] for r in reports],'metrics_pooled':False,'figures_written':True,
            'source_files':[str(p) for p in sorted(root.glob('*/summary.json'))]}
    (root/'cross_dataset_plot_scope.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--review',required=True)
    args=parser.parse_args();print(json.dumps(plot_external_transfer(args.root,args.review)))
