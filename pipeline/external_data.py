"""Public source-normalization and explicitly scoped external evaluators.

E2E/WebNLG metrics are conservative relation/alias coverage proxies. SQuAD uses
an independent local extractive QA model; gold is never supplied to that model.
All raw-domain Q/S values are null and never pooled with Synthetic-6.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, os, random, re, string, time
from pathlib import Path
from data_eval import read_jsonl, write_jsonl, sha, norm, word_count

ROOT=Path(__file__).resolve().parents[1]
QA_MODEL='deepset/minilm-uncased-squad2'
QA_REVISION='934656cdda79824eabf503ed56e15c01ddbdbe3f'
_QA=None

def entity(s):return re.sub(r'[^a-z0-9£]+',' ',str(s).replace('_',' ').lower()).strip()
def contains(text,value):return bool(re.search(r'(?<!\w)'+re.escape(entity(value))+r'(?!\w)',entity(text)))

def e2e_evidence(text,fact):
    slot=fact['relation'];v=fact['value'];t=entity(text);found=[]
    aliases=fact.get('accepted_answers',[v])
    if slot=='name':return [text] if contains(text,v) else []
    # Require the restaurant binding somewhere plus the slot evidence in current text.
    if fact['subject'] and not contains(text,fact['subject']):return []
    if slot=='familyfriendly':
        neg=bool(re.search(r'not (?:family|child|children|kid)|no (?:children|kids)|adults only|not suitable for',t))
        pos=bool(re.search(r'family friendly|child friendly|children friendly|kid friendly|suitable for (?:families|children)|welcomes children',t))
        return [text] if (v.lower()=='yes' and pos and not neg) or (v.lower()=='no' and neg) else []
    for alias in aliases:
        if not contains(text,alias):continue
        cue={'food':r'food|serves?|cuisine|dishes|restaurant','pricerange':r'price|cost|cheap|expensive|moderate|£|pounds',
             'customer rating':r'rat(?:ing|ed)|stars?|customer','customerrating':r'rat(?:ing|ed)|stars?|customer',
             'area':r'in |located|area|riverside|city centre|city center','near':r'near|close|beside|next to',
             'eattype':r'pub|restaurant|coffee shop'}
        if re.search(cue.get(slot,r'.'),t):found.append(text)
    return found

RELATIONS={
 'league':['play in','plays in','league'], 'country':['in','country'], 'nationality':['nationality','national','citizen','american','british'],
 'birthplace':['born in','birthplace'], 'deathplace':['died in','death place'], 'capital':['capital'],
 'location':['located in','in','location'], 'city':['city','in'], 'leader':['leader','led by','mayor'],
 'leadername':['leader','led by','mayor'], 'author':['written by','author','wrote'],
 'architect':['designed by','architect'], 'genre':['genre','music'], 'owner':['owned by','owner'],
 'manager':['managed by','manager'], 'established':['established','founded'], 'foundationplace':['founded in'],
 'team':['plays for','team'], 'club':['plays for','club'], 'alma mater':['studied at','alma mater'],
 'district':['district'], 'region':['region'], 'populationtotal':['population'], 'occupation':['works as','occupation'],
 'language':['language','spoken'], 'officiallanguage':['official language','language'], 'product':['produces','product'],
 'manufacturer':['manufactured by','manufacturer','made by'], 'birthdate':['born','birth'], 'deathdate':['died','death'],
}
def web_evidence(text,fact):
    rel=fact['relation'].lower();aliases=RELATIONS.get(rel,[])
    if not aliases:return []
    t=entity(text);s=entity(fact['subject']);o=entity(fact['value'])
    if not s or not o or not contains(t,s) or not contains(t,o):return []
    # A transparent relation-keyword coverage proxy, no claim of full entailment.
    return [text] if any(contains(t,a) for a in aliases) else []

def prepare_external(outdir,rawdir=None,seed=11):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True);rawdir=Path(rawdir or ROOT/'data/external_raw')
    manifests={}
    for ds in ['e2e','webnlg','squad2']:
        groups=read_jsonl(rawdir/f'{ds}_train_seed{seed}_groups.jsonl')
        raw=read_jsonl(rawdir/f'{ds}_train_seed{seed}_raw_rows.jsonl')
        bygroup=collections.defaultdict(list)
        for r in raw:bygroup[r['group_id']].append(r)
        eligible=[]
        for g in groups:
            if ds=='squad2':
                answerable=[r for r in bygroup[g['group_id']] if r['raw']['answers']['text']]
                if not answerable or word_count(g['candidate_source_text'])>160:continue
            eligible.append(g)
        random.Random(seed).shuffle(eligible)
        if len(eligible)<80:raise ValueError(f'{ds}: only {len(eligible)} eligible groups')
        pubs=[];privs=[]
        for i,g in enumerate(eligible[:80]):
            rows=bygroup[g['group_id']];rr=next(r for r in rows if r['original_row_index']==g['source_row_index'])['raw']
            text=g['candidate_source_text'];sid=f'{ds}_s{seed}_{i:04d}';split='calibration_fit' if i<16 else 'calibration_safety' if i<20 else 'selection' if i<40 else 'test'
            facts=[];queries=[];uncovered=[]
            if ds=='e2e':
                pairs=re.findall(r'([^,\[\]]+)\[([^\]]*)\]',rr['meaning_representation']);mr={k.strip().lower():v for k,v in pairs};subject=mr.get('name','')
                for j,(key,val) in enumerate(mr.items()):
                    aliases=[val]
                    if val=='city centre':aliases+=['city center','city-center','city-centre']
                    if val=='English':aliases+=['British']
                    if val=='high':aliases+=['5 out of 5','five out of five','highly rated','excellent']
                    if val=='low':aliases+=['1 out of 5','one out of five','poor']
                    if val=='moderate':aliases+=['moderately priced','mid-priced','average']
                    fact={'fact_id':f'e2e_{j}','type':'external_slot','subject':subject,'relation':key,'object':val,'value':val,'accepted_answers':aliases}
                    if e2e_evidence(text,fact):facts.append(fact)
                    else:uncovered.append(fact)
            elif ds=='webnlg':
                for j,triple in enumerate(rr['input']):
                    s,p,o=[x.strip().strip('"') for x in triple.split('|')]
                    fact={'fact_id':f'webnlg_{j}','type':'external_triple','subject':s,'relation':p,'object':o,'value':o,'accepted_answers':[o.replace('_',' ')]}
                    if web_evidence(text,fact):facts.append(fact)
                    else:uncovered.append(fact)
            else:
                for row in rows:
                    q=row['raw'];answers=q['answers']['text']
                    if not answers:continue
                    fid=q['id'];queries.append({'fact_id':fid,'question':q['question'],'type':'extractive_qa'})
                    facts.append({'fact_id':fid,'type':'extractive_qa','question':q['question'],'value':answers[0],'accepted_answers':answers,'answer_starts':q['answers']['answer_start']})
            pub={'source_id':sid,'group_id':g['group_id'],'dataset':ds+'_raw','split':split,
                'calibration_partition':'fit' if i<16 else 'safety' if i<20 else None,'source_text':text,
                'requirements':{'max_words':math.floor(.75*word_count(text)),'target_language':'en','bullet_min':1,'bullet_max':8,'active_requirements':['compress','structure']+(['simplify'] if re.search(r'\b(?:utilize|commence|prior to)\b',norm(text)) else [])+(['proofread'] if re.search(r'\bteh\b|\bthe\s+the\b|\bdetails\s+is\b',norm(text)) else [])},
                'evaluation_queries':queries,'metadata':{'source_words':word_count(text),'original_source_id':rr.get('gem_id',rr.get('id')),
                    'original_row_index':g['source_row_index'],'raw_condition':True,'source_group_rows':len(rows),'editor_context_limit_words':160,
                    'active_requirement_count':2+int(bool(re.search(r'\b(?:utilize|commence|prior to)\b',norm(text))))+int(bool(re.search(r'\bteh\b|\bthe\s+the\b|\bdetails\s+is\b',norm(text)))),'initial_Q':0.0,'objective_scope':'F_retention_only; external QA/proxy distinct from synthetic'}}
            private={'source_id':sid,'gold_facts':facts,'source_uncovered_facts':uncovered,'source_coverage_status':'audited_keyword_coverage' if ds!='squad2' else 'answerable_gold; local_QA_baseline_pending',
                'witness_target':None,'reference_answers':{f['fact_id']:f['accepted_answers'] for f in facts}}
            pubs.append(pub);privs.append(private)
        dest=out/ds;dest.mkdir(exist_ok=True);write_jsonl(dest/'public_documents.jsonl',pubs);write_jsonl(dest/'private_gold.jsonl',privs)
        original=json.loads((rawdir/f'{ds}_train_seed{seed}_manifest.json').read_text())
        m={'dataset':ds,'documents':80,'splits':dict(collections.Counter(p['split'] for p in pubs)),
            'source_rows_sha256':original['rows_sha256'],'dataset_revision':original['resolved_revision'],'repo_id':original['repo_id'],'license':original['license'],
            'documents_with_audited_facts':sum(bool(g['gold_facts']) for g in privs),'audited_facts':sum(len(g['gold_facts']) for g in privs),
            'uncovered_facts':sum(len(g['source_uncovered_facts']) for g in privs),'public_sha256':sha((dest/'public_documents.jsonl').read_text()),
            'private_sha256':sha((dest/'private_gold.jsonl').read_text()),'evaluator_scope':'extractive_QA_EM_F1' if ds=='squad2' else 'relation_keyword_alias_coverage_proxy',
            'zero_shot_frozen_synthetic_order':True,'external_Q_and_S':None,'source_group_disjoint':True}
        (dest/'manifest.json').write_text(json.dumps(m,indent=2));manifests[ds]=m
    write_jsonl(out/'public_documents.jsonl',[r for ds in ['e2e','webnlg','squad2'] for r in read_jsonl(out/ds/'public_documents.jsonl')])
    write_jsonl(out/'private_gold.jsonl',[r for ds in ['e2e','webnlg','squad2'] for r in read_jsonl(out/ds/'private_gold.jsonl')])
    (out/'manifest.json').write_text(json.dumps(manifests,indent=2));return manifests

def normalize_answer(s):
    s=s.lower();s=''.join(c for c in s if c not in string.punctuation);s=re.sub(r'\b(a|an|the)\b',' ',s);return ' '.join(s.split())
def em_f1(pred,answers):
    p=normalize_answer(pred);bestem=0.;bestf=0.
    for answer in answers:
        g=normalize_answer(answer);bestem=max(bestem,float(p==g));pt=p.split();gt=g.split()
        common=sum((collections.Counter(pt)&collections.Counter(gt)).values())
        f=(2*common/(len(pt)+len(gt))) if pt and gt else float(pt==gt)
        bestf=max(bestf,f)
    return bestem,bestf

class LocalExtractiveQA:
    def __init__(self,model=QA_MODEL):
        import torch
        from transformers import AutoTokenizer,AutoModelForQuestionAnswering
        torch.set_num_threads(4)
        self.revision=QA_REVISION;self.model_id=model
        try:
            self.tokenizer=AutoTokenizer.from_pretrained(model,revision=self.revision,local_files_only=True)
            self.model=AutoModelForQuestionAnswering.from_pretrained(model,revision=self.revision,local_files_only=True).eval().to('cpu')
        except OSError:
            if os.environ.get('HF_HUB_OFFLINE') == '1':
                raise
            self.tokenizer=AutoTokenizer.from_pretrained(model,revision=self.revision)
            self.model=AutoModelForQuestionAnswering.from_pretrained(model,revision=self.revision).eval().to('cpu')
        self.calls=0;self.sequences=0
    def extract(self,text,questions,batch_size=16):
        import torch
        answers=[]
        for off in range(0,len(questions),batch_size):
            qs=questions[off:off+batch_size]
            enc=self.tokenizer(qs,[text]*len(qs),return_tensors='pt',padding=True,truncation=False,return_offsets_mapping=True)
            if enc['input_ids'].shape[1]>512:raise ValueError('QA input exceeds 512 tokens; no silent truncation')
            offsets=enc.pop('offset_mapping');seqs=[enc.sequence_ids(i) for i in range(len(qs))]
            with torch.inference_mode():output=self.model(**enc)
            self.calls+=1;self.sequences+=len(qs)
            for i in range(len(qs)):
                starts=output.start_logits[i];ends=output.end_logits[i]
                valid=[j for j,x in enumerate(seqs[i]) if x==1 and offsets[i,j,1]>offsets[i,j,0]]
                null=float(starts[0]+ends[0]);best=(-1e30,None,None)
                for a in sorted(valid,key=lambda j:float(starts[j]),reverse=True)[:12]:
                    for b in sorted(valid,key=lambda j:float(ends[j]),reverse=True)[:12]:
                        if b<a or b-a>30:continue
                        score=float(starts[a]+ends[b])
                        if score>best[0]:best=(score,a,b)
                if best[1] is None or best[0]<=null:answers.append({'predicted_value':'','evidence':[],'no_answer':True,'score_margin':best[0]-null})
                else:
                    a=int(offsets[i,best[1],0]);b=int(offsets[i,best[2],1]);span=text[a:b]
                    answers.append({'predicted_value':span,'evidence':[span],'no_answer':False,'score_margin':best[0]-null})
        return answers

def get_qa():
    global _QA
    if _QA is None:_QA=LocalExtractiveQA()
    return _QA

def evaluate_external(text,public,private):
    ds=public['dataset'];facts=private.get('gold_facts',[]);fs=[];extra={}
    if ds=='squad2_raw':
        qa=get_qa();answers=qa.extract(text,[f['question'] for f in facts])
        for fact,pred in zip(facts,answers):
            em,f1=em_f1(pred['predicted_value'],fact['accepted_answers'])
            fs.append({'fact_id':fact['fact_id'],'status':'preserved' if em else 'missing' if pred['no_answer'] else 'altered',
                **pred,'EM':em,'token_F1':f1,'state':'embedded' if em else 'LOST','type':'extractive_qa'})
        f=sum(x['EM'] for x in fs)/len(fs) if fs else None
        extra={'QA_EM':f,'QA_token_F1':sum(x['token_F1'] for x in fs)/len(fs) if fs else None,
            'QA_no_answer_rate':sum(x['no_answer'] for x in fs)/len(fs) if fs else None,
            'QA_incorrect_nonempty_rate':sum(not x['no_answer'] and not x['EM'] for x in fs)/len(fs) if fs else None,
            'QA_model':qa.model_id,'QA_revision':qa.revision,
            'measurement_scope':'local_extractive_QA_exact_answer_retention; correctness depends on QA model'}
    else:
        func=e2e_evidence if ds=='e2e_raw' else web_evidence
        for fact in facts:
            evidence=func(text,fact)
            fs.append({'fact_id':fact['fact_id'],'status':'preserved' if evidence else 'missing','predicted_value':fact['value'] if evidence else None,
                'evidence':evidence,'state':'embedded' if evidence else 'LOST','type':fact['type']})
        f=sum(x['status']=='preserved' for x in fs)/len(fs) if fs else None
        extra={'measurement_scope':'audited_source_relation_keyword_alias_coverage_proxy; not semantic entailment'}
    return {'evaluator_version':'external_controlled_v1','F':f,'Q':None,'S':None,'strict_success':False,
        'fact_statuses':fs,'fact_states':[x['state'] for x in fs],'omission_rate':sum(x['status']=='missing' for x in fs)/len(fs) if fs else None,
        'altered_rate':sum(x['status']=='altered' for x in fs)/len(fs) if fs else None,'requirement_pass':{},'unsupported_claims':None,
        'evaluation_status':'ok_QA' if ds=='squad2_raw' else 'proxy_only' if facts else 'unscorable_no_audited_facts',
        'word_count':word_count(text),**extra}

# Same spelling as synthetic API for callers.
evaluate=evaluate_external

def audit_qa(outdir,data_dir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True);qa=get_qa()
    pubs=read_jsonl(Path(data_dir)/'squad2/public_documents.jsonl');privs={g['source_id']:g for g in read_jsonl(Path(data_dir)/'squad2/private_gold.jsonl')}
    # Baseline source EM/F1; deterministic audit uses calibration only, never test tuning.
    baseline=[]
    for pub in pubs:
        if not pub['split'].startswith('calibration'):continue
        score=evaluate_external(pub['source_text'],pub,privs[pub['source_id']]);baseline.append({'source_id':pub['source_id'],**score})
    absent=qa.extract('No details about the requested topic are stated here.',[q['question'] for p in pubs[:8] for q in p['evaluation_queries']])
    write_jsonl(out/'squad_source_QA_audit.jsonl',baseline)
    report={'QA_model':qa.model_id,'revision':qa.revision,'calibration_documents':len(baseline),'question_count':sum(len(x['fact_statuses']) for x in baseline),
        'source_macro_EM':sum(x['QA_EM'] for x in baseline)/len(baseline),'source_macro_token_F1':sum(x['QA_token_F1'] for x in baseline)/len(baseline),
        'no_answer_audit_n':len(absent),'no_answer_false_positive_rate':sum(not x['no_answer'] for x in absent)/len(absent),
        'scope':'Calibration baseline QA accuracy; scores retain model error. No threshold tuning on test; null-score threshold fixed at 0.',
        'external_metrics_never_pooled_with_synthetic':True}
    (out/'squad_QA_report.json').write_text(json.dumps(report,indent=2));return report

def analyze_outputs(rows,test_records,golds_dict,out_dir,write_review=True):
    """Rescore source+4hop outputs using dataset-appropriate frozen measurement."""
    started=time.perf_counter()
    qa_calls_before=_QA.calls if _QA is not None else 0
    qa_sequences_before=_QA.sequences if _QA is not None else 0
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    publics={r['source_id']:r for r in test_records}
    dataset=test_records[0]['dataset'] if test_records else 'unknown'
    evaluated=[];source_scores={};cached={};text_score_cache={}
    cachepath=out/'external_evaluations.jsonl'
    if cachepath.exists():cached={x['evaluation_key']:x for x in read_jsonl(cachepath)}
    for p in test_records:
        key=sha('external_v1:'+p['source_id']+':source:'+p['source_text'])
        sc=cached.get(key)
        if sc is None:sc={'evaluation_key':key,'source_id':p['source_id'],'condition':'source','hop':0,**evaluate_external(p['source_text'],p,golds_dict[p['source_id']])}
        source_scores[p['source_id']]=sc;evaluated.append(sc)
    for row in rows:
        hop=row.get('hop',row.get('step',len(row.get('prefix',[]))))
        if hop!=4:continue
        sid=row['source_id'];condition=row.get('condition',row.get('method','selected'))
        key=sha('external_v1:'+row.get('state_id',sid+condition)+':'+row['text'])
        sc=cached.get(key)
        if sc is None:
            text_key=(sid,sha(row['text']))
            if text_key not in text_score_cache:
                text_score_cache[text_key]=evaluate_external(row['text'],publics[sid],golds_dict[sid])
            # Copy before applying path-specific inference-failure accounting.
            score={**text_score_cache[text_key],'fact_statuses':[dict(x) for x in text_score_cache[text_key]['fact_statuses']]}
            if row.get('failure'):
                score.update(F=0.0 if score.get('F') is not None else None,evaluation_status='generation_failure_conservative_zero')
                for metric in ('QA_EM','QA_token_F1'):
                    if metric in score and score[metric] is not None:score[metric]=0.
                for f in score.get('fact_statuses',[]):f.update(status='missing',state='LOST')
            sc={'evaluation_key':key,'source_id':sid,'condition':condition,'hop':hop,'state_id':row.get('state_id'),**score}
        evaluated.append(sc)
        # Preserve completed CPU QA work if interrupted.
        write_jsonl(cachepath,evaluated)
    methods=sorted({r['condition'] for r in evaluated if r['hop']==4});summary={}
    rng=np.random.default_rng(11)
    def aggregate(values):
        a=np.asarray(values,dtype=float)
        if len(a)==0:return {'n':0,'mean':None,'ci95':[None,None]}
        boot=a[rng.integers(0,len(a),(5000,len(a)))].mean(axis=1)
        return {'n':len(a),'mean':float(a.mean()),'ci95':[float(x) for x in np.percentile(boot,[2.5,97.5])]}
    summary['source']=aggregate([r['F'] for r in source_scores.values() if r['F'] is not None])
    for method in methods:
        subset=[r for r in evaluated if r['condition']==method and r['hop']==4]
        valid=[r for r in subset if r['F'] is not None]
        m=aggregate([r['F'] for r in valid]);m['total_documents']=len(subset);m['unscorable_documents']=len(subset)-len(valid)
        paired=[r['F']-source_scores[r['source_id']]['F'] for r in valid if source_scores[r['source_id']]['F'] is not None]
        m['change_from_source']=aggregate(paired)
        if dataset=='squad2_raw':
            m['QA_token_F1']=aggregate([r['QA_token_F1'] for r in valid]);m['QA_no_answer_rate']=aggregate([r['QA_no_answer_rate'] for r in valid])
        summary[method]=m
    result={'dataset':dataset,'test_documents':len(test_records),'methods':summary,
        'measurement_scope':evaluated[0].get('measurement_scope') if evaluated else None,
        'selection':'Frozen synthetic winner transferred zero-shot; external test did not choose method.',
        'statistics':'5000 source-group bootstrap replicates; source change paired within document',
        'external_Q':None,'external_S':None,'not_pooled_with_synthetic':True,
        'QA_revision':get_qa().revision if dataset=='squad2_raw' else None,
        'evaluation_wall_seconds':time.perf_counter()-started,
        'QA_batch_invocations':(_QA.calls-qa_calls_before) if dataset=='squad2_raw' else 0,
        'QA_input_sequences':(_QA.sequences-qa_sequences_before) if dataset=='squad2_raw' else 0,
        'evaluator_module_sha256':sha(Path(__file__).read_text())}
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    fig,ax=plt.subplots(figsize=(8.5,4.5));labels=['source']+methods
    vals=[summary[m]['mean'] for m in labels];cis=[summary[m]['ci95'] for m in labels]
    y=[v if v is not None else 0 for v in vals]
    lower=[max(0,v-c[0]) if v is not None else 0 for v,c in zip(vals,cis)];upper=[max(0,c[1]-v) if v is not None else 0 for v,c in zip(vals,cis)]
    ax.bar(range(len(labels)),y,yerr=[lower,upper],capsize=4,color=['#8596a8']+['#275d92','#58a190','#b78949','#8c649c'][:len(methods)])
    ax.set_xticks(range(len(labels)),labels,rotation=12);ax.set_ylim(0,1.05)
    ax.set_ylabel('QA exact match' if dataset=='squad2_raw' else 'Relation/alias coverage proxy')
    ax.set_title(dataset+f' — frozen-order transfer, {len(test_records)} test source groups')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.text(.5,.015,'95% source-group bootstrap CI. Raw external metric; separate from Synthetic-6.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,1));fig.savefig(out/'external_transfer.png',dpi=180)
    if write_review:
        review=ROOT/'artifacts/Review';review.mkdir(exist_ok=True)
        fig.savefig(review/('07_external_'+dataset+'.png'),dpi=180)
    plt.close(fig)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--rawdir');p.add_argument('--audit-qa',action='store_true')
    a=p.parse_args();print(json.dumps(prepare_external(a.out,a.rawdir),indent=2))
    if a.audit_qa:print(json.dumps(audit_qa(Path(a.out)/'audit',a.out),indent=2))
