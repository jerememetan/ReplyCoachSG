"""Generate/review fictional paired training conversations using local Ollama."""
import argparse,copy,json,random,re
from collections import Counter
from hashlib import sha256
from pathlib import Path

from transformers import AutoTokenizer
from critic_preferences import save
from judge_replies import judge_settings,prepare_judge,read_env,request_json
from review_synthetic_pairs import review_pair,VERSION as REVIEW_VERSION
from replycoach.generate.style_control import mt5_input
from replycoach.generate.synthetic_v5 import fingerprint,load_synthetic
from replycoach.io import ROOT,read_jsonl

GROUPS=('roles_requests','direct_answers','changes_constraints','refusals_counteroffers','context_references','missing_new_details')
TOPICS=('group assignment','lunch stall','study space','club registration','commute','borrowed charger','class notes','printing','sports booking','meeting pickup','shared groceries','rehearsal','project slides','lost property','volunteer event','campus cafe','library books','movie outing','tutorial coordination','dorm kitchen','society poster','reading group','shared umbrella','costume pickup','quiz preparation','campus concert','parcel collection','equipment return','weekend plans','room booking','photo sharing','fundraising booth','orientation event','presentation practice','document editing','food order','station exit','workshop materials','lab partner','club practice')
REGISTERS=('casual_english','short_text_english','light_singlish','casual_english','short_text_english','light_singlish','casual_english','code_switched_english_mandarin')
WRITER_VERSION='volume-paired-v1'
WRITER_RUBRIC='''Write diverse FICTIONAL Singapore-university peer-chat training examples.
For every requested item, create a short history, one incoming PEER message, and
the USER's next reply expressed in STANDARD and LOCAL styles with the SAME meaning.
STANDARD: natural conversational standard English, not stiff customer-service prose.
LOCAL: natural restrained Singapore English, using syntax/vocabulary where useful;
no particle stuffing, caricature or mandatory lah. Output remains English-led.
Preserve actor, recipient, time, location, quantities, negation, uncertainty and
commitment in BOTH replies. Track who does what; never put the peer's situation
into the user's mouth. Respect explicit facts, constraints and the latest plan.
Plausible new personal progress, preferences, room/deadline/approval details are
allowed in selectable drafts. Do not default to uncertainty or clarification just
because a fact was not given. Useful clarifications for an ambiguous referent,
helpful offers, refusals and counterproposals are also valid.
Make the replies answer the incoming message instead of just mentioning its topic.
History turns may include consecutive user/peer messages; incoming is always peer.
Follow each requested speech-act group, topic, input register and history length.
Keep messages like real short chats, not formal question templates. Most histories
and messages should be short enough for a 128-token encoder input; target replies
should normally have 3-16 words, at most 24. Vary reply openings; do not repeatedly
start with Okay, Sure, No worries or Can. Avoid generic non-answers and copying.
Invent different situations, not the same template with different names/numbers.
No real private messages, phone numbers, emails, URLs or actual official policies.
Use ordinary casual topics rather than medical/legal/high-stakes advice.
Return only the requested JSON object. Do not add IDs, review flags or provenance.
'''
TURN={'type':'object','additionalProperties':False,'properties':{'speaker':{'type':'string','enum':['peer','user','other']},'text':{'type':'string','minLength':1,'maxLength':180}},'required':['speaker','text']}
ITEM={'type':'object','additionalProperties':False,'properties':{'session_memory':{'type':'array','maxItems':6,'items':TURN},'message':{'type':'string','minLength':1,'maxLength':180},'replies':{'type':'object','additionalProperties':False,'properties':{'standard':{'type':'string','minLength':1,'maxLength':180},'local':{'type':'string','minLength':1,'maxLength':180}},'required':['standard','local']}},'required':['session_memory','message','replies']}


def schema_for(count):
    return {'type':'object','additionalProperties':False,'properties':{'items':{'type':'array','minItems':count,'maxItems':count,'items':ITEM}},'required':['items']}


def structural_case(raw):
    if not isinstance(raw,dict) or set(raw)!=set(ITEM['required']):raise ValueError('Unexpected candidate fields')
    turns=raw['session_memory']
    if not isinstance(turns,list) or len(turns)>6:raise ValueError('Invalid history')
    for t in turns:
        if not isinstance(t,dict) or set(t)!={'speaker','text'} or t['speaker'] not in ('peer','user','other') or not isinstance(t['text'],str) or not t['text'].strip() or len(t['text'])>180:raise ValueError('Invalid history turn')
    if not isinstance(raw['message'],str) or not raw['message'].strip() or len(raw['message'])>180:raise ValueError('Invalid incoming message')
    replies=raw['replies']
    if not isinstance(replies,dict) or set(replies)!={'standard','local'}:raise ValueError('Two style replies required')
    if any(not isinstance(t,str) or not t.strip() or len(t)>180 or len(t.split())>24 for t in replies.values()):raise ValueError('Reply length/shape')
    if fingerprint([],replies['standard'])==fingerprint([],replies['local']):raise ValueError('Identical style wording')
    text=' '.join([*[t['text'] for t in turns],raw['message'],*replies.values()])
    if re.search(r'https?://|\b\S+@\S+\b|(?<!\d)\d{8,}(?!\d)',text):raise ValueError('Contact/link-like text')
    return raw


def lexical_terms(text):
    words=fingerprint([],text).split()
    return set(words)|{' '.join(pair) for pair in zip(words,words[1:])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',type=int,required=True);p.add_argument('--max-candidates',type=int,required=True)
    p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--seed-reviewed',type=Path);p.add_argument('--protected-synthetic',type=Path)
    p.add_argument('--env-file',type=Path,default=ROOT/'.env')
    args=p.parse_args()
    if not 1<=args.target<=args.max_candidates:raise ValueError('Invalid target/budget')
    if args.out_dir.exists():raise FileExistsError('Choose a fresh output directory')
    args.out_dir.mkdir(parents=True)
    protected=[]
    for split in ('train','validation'):
        for c in read_jsonl(ROOT/f'data/dev/{split}.jsonl'):
            protected.append({'item_id':c['id'],'context':[t['text'] for t in c['session_memory']],'message':c['message']['text']})
    older=args.protected_synthetic or ROOT/'data/synthetic/paired60_v1_reviewed.jsonl'
    previous=read_jsonl(older) if older.exists() else []
    config=judge_settings(read_env(args.env_file));review_settings=prepare_judge(config)
    writer_settings={**review_settings,'options':{**review_settings['options'],'temperature':0.8,'top_p':0.9,'num_ctx':8192,'num_predict':4096}}
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    accepted=read_jsonl(args.seed_reviewed) if args.seed_reviewed else []
    if accepted:
        load_synthetic(args.seed_reviewed,protected)
        if any(c['provenance'].get('prompt_version')!=WRITER_VERSION or c['provenance'].get('model_digest')!=review_settings['model_digest'] for c in accepted):raise ValueError('Seed corpus generator/model differs')
    groups=Counter(c['scenario_group'] for c in accepted)
    contexts=[' '.join([*c['context'],c['message']]) for c in protected]
    contexts += [' '.join([*[t['text'] for t in c['session_memory']],c['message']]) for c in [*previous,*accepted]]
    index=[lexical_terms(text) for text in contexts]
    seen_keys={fingerprint(c['context'],c['message']) for c in protected}
    seen_messages={fingerprint([],c['message']) for c in protected}
    for c in [*previous,*accepted]:
        seen_keys.add(fingerprint([t['text'] for t in c['session_memory']],c['message']));seen_messages.add(fingerprint([],c['message']))
    counts=Counter();serial=0;batch=(max(c['provenance']['seed'] for c in accepted)-425+1) if accepted else 0
    rng=random.Random(425+batch*7919);consecutive_errors=0
    report=dict(purpose='fictional_synthetic_volume_generation_and_review',status='running',target=args.target,
        max_candidates=args.max_candidates,accepted=len(accepted),new_candidates=0,counts={},group_counts=dict(groups),
        generator_settings=writer_settings,review_settings=review_settings,writer_version=WRITER_VERSION,writer_rubric=WRITER_RUBRIC,
        review_version=REVIEW_VERSION,near_duplicate_method='word_unigram_bigram_jaccard',near_duplicate_threshold=0.75,token_limits={'source':128,'target':48},
        protected_hashes={split:sha256((ROOT/f'data/dev/{split}.jsonl').read_bytes()).hexdigest() for split in ('train','validation')},
        seed_reviewed_sha256=sha256(args.seed_reviewed.read_bytes()).hexdigest() if args.seed_reviewed else None,
        limitation='Writer and reviewer use the SAME frozen local model in separate fresh calls; not independent human/model verification. Lexical duplicate filters do not prove semantic independence. No test content supplied or read.')
    save(args.out_dir/'manifest.json',report)
    def append(name,obj):
        with (args.out_dir/name).open('a') as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n')
    try:
        while len(accepted)<args.target and serial<args.max_candidates:
            count=min(4,args.max_candidates-serial);requests=[];tentative=groups.copy()
            for j in range(count):
                group=min(GROUPS,key=lambda g:(tentative[g],GROUPS.index(g)))
                tentative[group]+=1
                requests.append(dict(speech_act_group=group,topic=rng.choice(TOPICS),input_register=REGISTERS[(batch*4+j)%len(REGISTERS)],history_turns=rng.choice([0,1,2,2,3,3,4,5,6])))
            options={**writer_settings['options'],'seed':425+batch}
            response=request_json(config['base'],'/api/chat',{'model':writer_settings['judge_model'],'stream':False,'think':False,
                'format':schema_for(count),'options':options,'messages':[{'role':'system','content':WRITER_RUBRIC},{'role':'user','content':json.dumps({'batch':batch,'requests':requests})}]})
            append('generation_raw.jsonl',{'batch':batch,'requests':requests,'options':options,'response':response})
            batch+=1
            try:
                if response.get('done') is not True or response.get('done_reason')=='length':raise ValueError('Incomplete generation')
                data=json.loads(response['message']['content'])
                if not isinstance(data,dict) or set(data)!={'items'} or not isinstance(data['items'],list) or len(data['items'])!=count:raise ValueError('Wrong generation batch shape')
                consecutive_errors=0
            except Exception as exc:
                counts['generation_error']+=1;serial+=count;consecutive_errors+=1
                append('rejections.jsonl',{'batch':batch-1,'reason':str(exc)})
                if consecutive_errors>=5:raise ValueError('Five consecutive malformed generations')
                continue
            for j,(raw,request) in enumerate(zip(data['items'],requests)):
                serial+=1
                if not isinstance(raw,dict):
                    counts['structural_or_duplicate_rejected']+=1
                    append('rejections.jsonl',{'batch':batch-1,'reason':'Candidate is not an object','raw':raw})
                    continue
                c={'id':f'SYN-volume500-v1-{args.out_dir.name}-{batch-1:03}-{j:02}',**raw,
                   'scenario_group':request['speech_act_group'],'requested_input_register':request['input_register'],
                   'provenance':{'generator':writer_settings['judge_model'],'model_digest':writer_settings['model_digest'],
                       'prompt_version':WRITER_VERSION,'batch_id':f'{args.out_dir.name}-{batch-1:03}','seed':options['seed']},
                   'review':{'reviewer':'','meaning_preserved':False,'speaker_correct':False,'natural_local':False,'facts_supported':False}}
                append('candidates.jsonl',c)
                try:
                    structural_case(raw)
                    speakers=[t['speaker'] for t in c['session_memory']];context=[t['text'] for t in c['session_memory']]
                    key=fingerprint(context,c['message']);msg=fingerprint([],c['message'])
                    if key in seen_keys or msg in seen_messages:raise ValueError('Exact duplicate/protected message')
                    row=lexical_terms(' '.join([*context,c['message']]))
                    score=max((len(row&other)/len(row|other) for other in index if row|other),default=0.0)
                    if score>=0.75:raise ValueError(f'Lexical near duplicate {score:.3f}')
                    if any(len(tokenizer(mt5_input(style,c['message'],context,True,speakers))['input_ids'])>128 or len(tokenizer(c['replies'][style.lower()])['input_ids'])>48 for style in ('STANDARD','LOCAL')):raise ValueError('Would truncate model input or target')
                except (ValueError,KeyError,TypeError) as exc:
                    counts['structural_or_duplicate_rejected']+=1;append('rejections.jsonl',{'id':c['id'],'reason':str(exc)});continue
                review=review_pair(c,config,review_settings)
                append('reviews.jsonl',{'id':c['id'],'review':review})
                if not review['approved']:
                    counts['review_'+review['verdict']]+=1;continue
                c['review']={'reviewer':review_settings['judge_model']+' separate fresh call; SAME model weights as generator; not human',
                    'meaning_preserved':True,'speaker_correct':True,'natural_local':True,'facts_supported':True,
                    'scope':'Fictional selectable draft compatibility, not real-world fact verification.',
                    'model_digest':review_settings['model_digest'],'prompt_version':REVIEW_VERSION,'raw_review':review}
                accepted.append(c);groups[c['scenario_group']]+=1;counts['approved']+=1
                seen_keys.add(key);seen_messages.add(msg);index.append(row)
                print(json.dumps({'accepted':len(accepted),'target':args.target,'new_candidates':serial,'group':c['scenario_group']}),flush=True)
                if len(accepted)>=args.target:break
            report.update(accepted=len(accepted),new_candidates=serial,counts=dict(counts),group_counts=dict(groups))
            save(args.out_dir/'manifest.json',report)
            from replycoach.io import write_jsonl_atomic
            write_jsonl_atomic(args.out_dir/'reviewed_partial.jsonl',accepted)
        selected=accepted[:args.target]
        from replycoach.io import write_jsonl_atomic
        write_jsonl_atomic(args.out_dir/'reviewed.jsonl',selected)
        rows=load_synthetic(args.out_dir/'reviewed.jsonl',protected)
        assert len(rows)==2*len(selected)
        report.update(status='complete' if len(selected)==args.target else 'insufficient_approved',accepted=len(selected),
            reviewed_sha256=sha256((args.out_dir/'reviewed.jsonl').read_bytes()).hexdigest(),new_candidates=serial,counts=dict(counts),group_counts=dict(Counter(c['scenario_group'] for c in selected)))
        save(args.out_dir/'manifest.json',report);print(json.dumps({k:report[k] for k in ('status','accepted','new_candidates','counts','group_counts')}),flush=True)
        if report['status']!='complete':raise RuntimeError('Approved target not reached; no training permitted')
    except Exception:
        if report['status']=='running':report['status']='failed'
        report.update(accepted=len(accepted),new_candidates=serial,counts=dict(counts));save(args.out_dir/'manifest.json',report);raise


if __name__=='__main__':main()
