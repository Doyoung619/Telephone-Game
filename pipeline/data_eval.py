"""Frozen rule-based Telephone Game datasets and evidence-bearing evaluator.

This evaluator is a conservative controlled-English scorer, not a general semantic
oracle. External structured results are explicitly separated from Synthetic-6.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, random, re, unicodedata
from collections import Counter
from pathlib import Path

VERSION = 'controlled_relation_rules_v2_calibration_frozen'
ROLES = ['A', 'B', 'C', 'D']
PROFILES = {'paper': (64,16,40,80), 'pilot':(19,5,24,48),'smoke':(4,2,6,0)}
SIMPLIFICATIONS = {'utilize':'use','commence':'begin','prior to':'before'}
NAMES = ['Mira','Jina','Samir','Leona','Owen','Arun','Nadia','Felix','Tara','Iris','Ravi','Lena']
DAYS = ['Monday','Tuesday','Wednesday','Thursday','Friday']

def word_count(text): return len(text.split())
def sha(x): return hashlib.sha256(x.encode()).hexdigest()
def write_jsonl(path,rows):
    Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
def read_jsonl(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def norm(text):
    s=' '.join(unicodedata.normalize('NFKC',text).lower().replace('’',"'").split())
    for a,b in {"isn't":'is not',"aren't":'are not',"wasn't":'was not',"weren't":'were not',"hasn't":'has not',"haven't":'have not',"doesn't":'does not',"don't":'do not',"can't":'cannot',"won't":'will not'}.items():s=s.replace(a,b)
    return s

def number_words(n):
    small=['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen']
    if n<20:return small[n]
    return ['','','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety'][n//10]+(' '+small[n%10] if n%10 else '')
NUMBER_FORMS=sorted([(number_words(i),str(i)) for i in range(100)],key=lambda p:-len(p[0]))
def normalize_numbers(text):
    text=norm(text).replace('-',' ')
    for w,n in NUMBER_FORMS:text=re.sub(r'\b'+re.escape(w)+r'\b',n,text)
    return text

def sentences(text):
    # Evidence is always an exact span in current text, including bullet text.
    return [m.group(0).strip() for m in re.finditer(r'[^\n.!?]+(?:[.!?]|$)',text) if m.group(0).strip(' -*\t')]

def make_record(i,seed=11):
    rng=random.Random(seed*100000+i)
    name=NAMES[i%len(NAMES)]; project=f'P{1000+i*13+seed}'
    count=rng.randint(21,79); days=rng.randint(2,9); day=DAYS[i%5]
    reader=rng.choice(['external reviewer','independent auditor','outside assessor'])
    variant=i%4
    lead=[f'{name} is the person who leads Project {project}.',f'Project {project} is led by {name}.',f'{name} serves as the leader of Project {project}.',f'The leader of Project {project} is {name}.'][variant]
    nums=[f'The test involved {count} units and continued for {days} days.',f'A total of {count} units were tested over {days} days.',f'Testing lasted {days} days and included {count} units.',f'There were {count} units in the test, which lasted {days} days.'][variant]
    source=(f'{lead} {nums} The launch is not approved. The team may commence trials only if the sensor test passes. '
            f'A reduction in delays is possible, but it is not certain. The {reader} raised the cost concern. '
            f'The review is scheduled for {day}. Prior to reading the note, readers should distinguish a possible outcome from a confirmed outcome. '
            'The team will utilize this note for its review. Teh note repeats the the same planning information. '
            'These details is provided for discussion. The wording is expanded here to record a plan rather than a final outcome.')
    if i%2:source+=' The report repeats its caution that a scheduled review alone does not constitute launch approval.'
    witness=(f'- {name} leads Project {project}; {count} units were tested over {days} days.\n'
             '- The launch is not approved. Trials may begin only if the sensor test passes.\n'
             f'- Reduced delays are possible, not certain. The {reader} raised the cost concern. The review is on {day}.\n'
             '- Before reading the note, distinguish possible and confirmed outcomes. Use this note for the review. '
             'The note repeats the same planning information. These details are provided for discussion.')
    cap=math.floor(.75*word_count(source)); assert 80<=word_count(source)<=160,(i,word_count(source))
    assert word_count(witness)<=cap,(word_count(witness),cap)
    sid=f'synthetic6_s{seed}_{i:04d}'
    # Fixed full-paper split FIRST. Smaller profiles are subsets of these IDs.
    if i<64:split='calibration_fit'
    elif i<80:split='calibration_safety'
    elif i<120:split='selection'
    else:split='test'
    queries=[('leader','entity','Who leads the project?',name),('count','number','How many units were tested?',str(count)),
             ('days','number','How many days did testing last?',str(days)),('approval','negation','What is the launch approval status?','not approved'),
             ('condition','condition','Under what condition may trials begin?','sensor test passes'),
             ('certainty','uncertainty','How certain is a reduction in delays?','possible not certain'),
             ('speaker','attribution','Who raised the cost concern?',reader),('review_day','entity','On which day is the review scheduled?',day),
             ('note_use','entity','For what purpose will the team use the note?','review'),
             ('repeats','entity','What information does the note repeat?','planning information'),
             ('discussion','entity','For what purpose are the details provided?','discussion')]
    facts=[]
    for fid,typ,q,val in queries:
        facts.append({'fact_id':fid,'type':typ,'question':q,'value':val,'accepted_answers':[val],
                      'subject':('Project '+project if fid=='leader' else fid), 'relation':fid,'object':val,
                      'polarity':'negative' if fid=='approval' else 'positive',
                      'modality':'possible' if fid=='certainty' else None,
                      'condition':'sensor test passes' if fid=='condition' else None,
                      'attribution':reader if fid=='speaker' else None,'project':project})
    public={'source_id':sid,'group_id':sid,'dataset':'synthetic6','split':split,
            'calibration_partition':('fit' if split=='calibration_fit' else 'safety' if split=='calibration_safety' else None),
            'source_text':source,'requirements':{'max_words':cap,'target_language':'en','bullet_min':1,'bullet_max':8,
                'min_bullets':1,'max_bullets':8,'active_requirements':['simplify','compress','proofread','structure']},
            'evaluation_queries':[{'fact_id':f,'type':t,'question':q} for f,t,q,v in queries],
            'metadata':{'source_words':word_count(source),'template_family':f'controlled_project_v{variant}',
                'fixture_only':False,'seed':seed,'scenario':'short' if word_count(source)<120 else 'long',
                'domain':'project_review','source_index':i}}
    private={'source_id':sid,'gold_facts':facts,'reference_answers':{f:v for f,t,q,v in queries},
             'witness_target':witness,'feasibility_witness':witness,'source_coverage_status':'all_generated_facts_audited',
             'injected_errors':['Teh','the the','details is'],'simplifications':SIMPLIFICATIONS}
    return public,private

def _capture(text, patterns):
    out=[]
    for span in sentences(text):
        s=normalize_numbers(span)
        for pat in patterns:
            for m in re.finditer(pat,s,re.I):
                value=m.groupdict().get('value') or m.group(0)
                out.append((norm(value).strip(' ,:;.-'),span))
    # Pattern overlap does not create contradictory observations.
    return list(dict.fromkeys(out))

def extract_fact(text,fact):
    fid=fact['fact_id']; value=norm(str(fact['value'])); candidates=[]
    p=r'(?:project\s+)?p\d+'
    name=r'[a-z][a-z\-]+'
    if fid=='leader':
        candidates=_capture(text,[rf'(?P<value>{name})(?:\s+is\s+the\s+person\s+who)?\s+(?:leads|heads|runs)\s+(?:project\s+)?(?P<project>p\d+)',
            rf'(?P<value>{name})\s+(?:serves as|is)\s+(?:the\s+)?leader\s+of\s+(?:project\s+)?(?P<project>p\d+)',
            rf'(?:project\s+)?(?P<project>p\d+)\s+(?:is\s+)?(?:led|headed|run)\s+by\s+(?P<value>{name})',
            rf'(?:the\s+)?leader\s+of\s+(?:project\s+)?(?P<project>p\d+)\s+is\s+(?P<value>{name})'])
        # Ensure correct project binding; mention of a different project is altered.
        candidates=[(v if norm(fact.get('project','')) in norm(e) else 'WRONG_PROJECT:'+v,e) for v,e in candidates]
    elif fid=='count':
        candidates=_capture(text,[r'(?P<value>\d+)\s+(?:test\s+)?units?\b',r'units?\s*(?:tested|included|involved)?\s*[:=]\s*(?P<value>\d+)'])
    elif fid=='days':
        candidates=_capture(text,[r'(?:for|over|lasted|lasts|lasting|duration(?:\s+of)?)\s+(?P<value>\d+)\s+days?',r'(?P<value>\d+)\s+days?\s+(?:of\s+)?(?:testing|test|duration)',r'(?P<value>\d+)\s*day\s+(?:test|trial)'])
    elif fid=='approval':
        for span in sentences(text):
            s=norm(span)
            if 'launch' not in s or not re.search(r'approv|authoriz',s):continue
            if re.search(r'(?:does not|doesn.t)\s+(?:constitute|mean|imply|grant)\s+(?:launch\s+)?approval',s):continue
            negative=bool(re.search(r'not\s+(?:yet\s+|been\s+)?(?:approved|authorized)|unapproved|approval\s+(?:has\s+)?not|no\s+(?:launch\s+)?approval|launch.{0,18}(?:lacks|without)\s+approval|approval\s+(?:is\s+)?(?:pending|denied)',s))
            candidates.append(('not approved' if negative else 'approved',span))
    elif fid=='condition':
        for span in sentences(text):
            s=norm(span)
            if not re.search(r'trials?|testing',s) or not re.search(r'sensor|condition',s):continue
            good=bool(re.search(r'(?:only\s+if|only\s+when|only\s+after|provided\s+(?:that\s+)?|conditional\s+on|contingent\s+on|requires?|subject\s+to)',s)) and bool(re.search(r'sensor\s+tests?\s+(?:passes|pass|succeeds)|(?:passing|successful)\s+(?:the\s+)?sensor\s+test',s))
            candidates.append(('sensor test passes' if good else 'condition altered',span))
    elif fid=='certainty':
        for span in sentences(text):
            s=norm(span)
            if not re.search(r'(?:delay|delays)',s) or not re.search(r'reduc|lower|decreas|fewer|improv',s):continue
            uncertain=bool(re.search(r'possible|may\b|might\b|could\b|not\s+(?:certain|guaranteed)|uncertain|potential',s))
            certain=bool(re.search(r'(?<!not )\bcertain\b|guaranteed|definite|\bwill\b',s))
            # "not certain" and "not guaranteed" are negative modal evidence.
            certain=bool(re.search(r'(?<!not )\bcertain\b|(?<!not )\bguaranteed\b|definite|\bwill\b',s))
            candidates.append(('possible not certain' if uncertain and not certain else 'certain',span))
    elif fid=='speaker':
        candidates=_capture(text,[r'(?:the\s+)?(?P<value>(?:external reviewer|independent auditor|outside assessor|team|manager|internal reviewer))\s+(?:raised|voiced|expressed|reported|noted|highlighted|mentioned)\s+(?:the\s+)?(?:a\s+)?(?:cost\s+concerns?|concerns?\s+about\s+costs?)',
            r'(?:the\s+)?cost\s+concerns?\s+(?:was\s+)?(?:raised|voiced|expressed|reported)\s+by\s+(?:(?:the|an?)\s+)?(?P<value>external reviewer|independent auditor|outside assessor|team|manager|internal reviewer)'])
    elif fid=='review_day':
        candidates=_capture(text,[r'review\s+(?:is\s+)?(?:scheduled\s+|planned\s+|set\s+)?(?:for|on)\s+(?P<value>monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
            r'(?P<value>monday|tuesday|wednesday|thursday|friday|saturday|sunday).{0,10}review'])
    elif fid=='note_use':
        for span in sentences(text):
            s=norm(span)
            if re.search(r'\bnote\b',s) and re.search(r'\b(?:use|uses|utilize|utilizes|using)\b',s):
                candidates.append(('review' if re.search(r'\breview\b',s) else 'purpose unspecified',span))
    elif fid=='repeats':
        for span in sentences(text):
            s=norm(span)
            if (re.search(r'\bnote\b',s) or ('note' in norm(text) and re.search(r'\bit\b',s))) and re.search(r'repeat|reiterat|restat',s):
                candidates.append(('planning information' if re.search(r'plann(?:ing|ed)\s+(?:information|info|details)|same\s+plan',s) else 'different content',span))
    elif fid=='discussion':
        for span in sentences(text):
            s=norm(span)
            if re.search(r'details|information',s) and re.search(r'provid|present|suppli|given|for\s+discussion|discuss',s):
                if re.search(r'discussion|discussed',s):candidates.append(('discussion',span))
    else:return {'fact_id':fid,'status':'evaluator_error','predicted_value':None,'evidence':[],'state':'LOST','type':fact.get('type')}
    good=[e for v,e in candidates if v==value]; bad=[e for v,e in candidates if v!=value]
    if good and bad:status='ambiguous'
    elif good:status='preserved'
    elif bad:status='altered'
    else:status='missing'
    return {'fact_id':fid,'type':fact.get('type'),'status':status,'predicted_value':list(dict.fromkeys(v for v,e in candidates)) or None,
            'evidence':list(dict.fromkeys(e for v,e in candidates)),'state':'embedded' if status=='preserved' else 'LOST'}

def evaluate(text,public_record,private_record):
    if public_record.get('dataset')!='synthetic6':return evaluate_external(text,public_record,private_record)
    fs=[extract_fact(text,f) for f in private_record['gold_facts']]
    preserved=[f for f in fs if f['status']=='preserved']
    # Standalone vs embedded is sentence-local observed representation, not fact type.
    for f in preserved:
        shared=any(set(f['evidence']) & set(g['evidence']) for g in preserved if g is not f)
        f['state']='embedded' if shared else 'standalone'
    by={f['fact_id']:f['status']=='preserved' for f in fs}; s=norm(text)
    # Content deletion cannot pass a simplification/correction condition.
    simple=(not any(re.search(r'\b'+re.escape(k)+r'\b',s) for k in SIMPLIFICATIONS)
            and by.get('condition',False) and by.get('note_use',False))
    proof=(not re.search(r'\bteh\b|\bthe\s+the\b|\bdetails\s+is\b',s)
           and by.get('repeats',False) and by.get('discussion',False))
    bullets=[line for line in text.splitlines() if re.match(r'^\s*(?:[-*•]|\d+[.)])\s+\S',line)]
    req=public_record['requirements']; p={'simplify':simple,'compress':word_count(text)<=req['max_words'],
        'proofread':proof,'structure':req.get('bullet_min',1)<=len(bullets)<=req.get('bullet_max',8)}
    active=req.get('active_requirements',list(p)); q=sum(bool(p[k]) for k in active)/len(active) if active else None
    f=len(preserved)/len(fs) if fs else 0.; score=2*f*q/(f+q) if q is not None and f+q else 0. if q is not None else None
    return {'evaluator_version':VERSION,'F':f,'Q':q,'S':score,'strict_success':f==1 and q==1,'fact_statuses':fs,
            'fact_states':[x['state'] for x in fs],'omission_rate':sum(x['status']=='missing' for x in fs)/len(fs),
            'altered_rate':sum(x['status'] in ('altered','ambiguous') for x in fs)/len(fs),
            'requirement_pass':p,'unsupported_claims':None,'evaluation_status':'ok','word_count':word_count(text),
            'measurement_scope':'controlled_relation_rules; paraphrase misses possible'}

def evaluate_external(text,public,private):
    # Exact relation evidence coverage, separate from synthetic's slot accuracy.
    fs=[]
    for fact in private.get('gold_facts',[]):
        expected=norm(fact['value']); aliases=fact.get('accepted_answers',[fact['value']]); evidence=[]
        subject=norm(fact.get('subject',''))
        for span in sentences(text):
            s=norm(span)
            if any(norm(a) in s for a in aliases) and (not subject or subject in s):evidence.append(span)
        fs.append({'fact_id':fact['fact_id'],'type':fact.get('type','external_slot'),'status':'preserved' if evidence else 'missing',
                   'predicted_value':fact['value'] if evidence else None,'evidence':evidence,'state':'embedded' if evidence else 'LOST'})
    f=sum(x['status']=='preserved' for x in fs)/len(fs) if fs else 0.
    return {'evaluator_version':VERSION,'F':f,'Q':None,'S':None,'strict_success':False,'fact_statuses':fs,
            'fact_states':[x['state'] for x in fs],'omission_rate':1-f,'altered_rate':None,'requirement_pass':{},
            'unsupported_claims':None,'evaluation_status':'proxy_only','word_count':word_count(text),
            'measurement_scope':'external subject-value cooccurrence proxy, not validated semantic entailment'}

def audit_evaluator(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True); rows=[]
    for i in range(8):
        pub,gold=make_record(i)
        witness=gold['witness_target']; by={f['fact_id']:f for f in gold['gold_facts']}
        def add(label,fid,txt,expected):
            got=extract_fact(txt,by[fid]); rows.append({'case':label,'index':i,'fact_id':fid,'expected':expected,'predicted':got['status'],
                'correct':got['status']==expected,'evidence_valid':all(e in txt for e in got['evidence']),'text':txt})
        add('number_change','count',witness.replace(by['count']['value']+' units',str(int(by['count']['value'])+1)+' units'),'altered')
        add('entity_change','leader',witness.replace(by['leader']['value'],'Vera'),'altered')
        add('negation_flip','approval',witness.replace('not approved','approved'),'altered')
        add('condition_deletion','condition',witness.replace(' only if the sensor test passes',''),'missing')
        add('uncertainty_strengthening','certainty',witness.replace('possible, not certain','certain'),'altered')
        add('attribution_swap','speaker',witness.replace(by['speaker']['value'],'internal reviewer'),'altered')
        for fid in by:add('preserved_original',fid,pub['source_text'],'preserved')
        for fid in by:add('preserved_witness',fid,witness,'preserved')
        add('spelled_number','count',witness.replace(by['count']['value']+' units',number_words(int(by['count']['value']))+' units'),'preserved')
        add('paraphrase','leader',f"Project {by['leader']['project']} is led by {by['leader']['value']}.",'preserved')
        add('missing_fact','count','The group discussed the project.','missing')
        add('wrong_binding','count',f"The budget was {by['count']['value']} dollars; the group discussed {by['days']['value']} days.",'missing')
        add('contradiction','approval','The launch is approved. The launch is not approved.','ambiguous')
        add('paraphrase_contraction','approval',"The launch isn't approved yet.",'preserved')
        add('paraphrase_perfect_negation','approval','The launch has not been approved yet.','preserved')
        add('paraphrase_condition','condition','Trials can begin only after passing the sensor test.','preserved')
        add('paraphrase_review_set','review_day',f"The review is set for {by['review_day']['value']}.",'preserved')
        add('paraphrase_speaker','speaker',f"An {by['speaker']['value']} mentioned concerns about costs.",'preserved')
        add('paraphrase_repeats','repeats','The note repeats the planned info.','preserved')
        add('paraphrase_discussion','discussion','These details are discussed.','preserved')
    with (out/'evaluator_audit.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    expected_pres=sum(r['expected']=='preserved' for r in rows);pred_pres=sum(r['predicted']=='preserved' for r in rows)
    tp=sum(r['predicted']==r['expected']=='preserved' for r in rows)
    groups={k:{'n':sum(r['case']==k for r in rows),'accuracy':sum(r['correct'] for r in rows if r['case']==k)/sum(r['case']==k for r in rows)} for k in sorted({r['case'] for r in rows})}
    report={'version':VERSION,'cases':len(rows),'preserved_precision':tp/pred_pres if pred_pres else 0,
            'preserved_recall':tp/expected_pres,'evidence_error_rate':sum(not r['evidence_valid'] for r in rows)/len(rows),
            'no_answer_false_positive_rate':sum(r['predicted']=='preserved' for r in rows if r['expected']=='missing')/sum(r['expected']=='missing' for r in rows),
            'by_type':groups,'evaluator_sha256':sha(Path(__file__).read_text()),
            'scope':'Known-template counterfactual audit; does NOT validate unrestricted LLM paraphrases or external data.',
            'confusion':dict(Counter(r['expected']+'->'+r['predicted'] for r in rows))}
    critical=['number_change','entity_change','negation_flip','condition_deletion','uncertainty_strengthening','attribution_swap']
    report['passed']=report['preserved_precision']>=.9 and report['preserved_recall']>=.9 and all(groups[t]['accuracy']>=.8 for t in critical)
    (out/'evaluator_report.json').write_text(json.dumps(report,indent=2))
    (out/'evaluator_report.md').write_text(f"Evaluator {VERSION}: controlled relation rules.\n\nAudit: {len(rows)} generated counterfactual/preservation cases; precision={report['preserved_precision']:.3f}, recall={report['preserved_recall']:.3f}, passed={report['passed']}.\n\nThis audit tests controlled English only. Novel model paraphrases may be falsely marked missing; source and witness checks do not establish general semantic accuracy. External cooccurrence is a proxy and is never pooled with Synthetic-6. Missing/altered ambiguity and unsupported new claims remain limitations.\n")
    return report

def prepare_data(out_dir,profile='paper',seed=11):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    full=[make_record(i,seed) for i in range(200)]
    nfit,nsafe,nsel,ntest=PROFILES[profile]
    ids=list(range(nfit))+list(range(64,64+nsafe))+list(range(80,80+nsel))+list(range(120,120+ntest))
    pairs=[full[i] for i in ids];public=[p for p,g in pairs];private=[g for p,g in pairs]
    for p,g in pairs:
        s=evaluate(p['source_text'],p,g);w=evaluate(g['witness_target'],p,g)
        if s['F']!=1 or w['F']!=1 or w['Q']!=1:raise ValueError(('source/witness audit',p['source_id'],s,w))
    write_jsonl(out/'public_documents.jsonl',public);write_jsonl(out/'private_gold.jsonl',private)
    splits={split:[p['source_id'] for p in public if p['split']==split] for split in ['calibration_fit','calibration_safety','selection','test']}
    (out/'splits.json').write_text(json.dumps(splits,indent=2))
    audit=audit_evaluator(out/'audit')
    manifest={'profile':profile,'seed':seed,'documents':len(public),'splits':{k:len(v) for k,v in splits.items()},
              'public_sha256':sha((out/'public_documents.jsonl').read_text()),'private_sha256':sha((out/'private_gold.jsonl').read_text()),
              'evaluator_sha256':sha(Path(__file__).read_text()),'all_source_F0':1,'all_witness_F':1,'all_witness_Q':1,
              'nested_in_paper_ids':True,'absolute_word_cap':'floor(0.75*source_words); frozen before any edit','audit_passed':audit['passed']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--profile',choices=PROFILES,default='paper');parser.add_argument('--seed',type=int,default=11)
    args=parser.parse_args();print(json.dumps(prepare_data(args.out,args.profile,args.seed),indent=2))
