"""Run: .venv/bin/streamlit run scripts/model_ab_test.py"""
from pathlib import Path
import hashlib,json,random,secrets,sys

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'data/human_review/model_ab_v1'
SOURCES={
 'default':ROOT/'artifacts/spk1_ul025_s425.json',
 'full_epoch1':ROOT/'artifacts/standard_val_v1/full_lr3e4.json',
 'full_epoch2':ROOT/'artifacts/standard_val_v1/full_lr1e4.json',
 'lora_epoch4':ROOT/'artifacts/standard_val_v1/lora_lr3e4.json',
}

def prepare(directory=DIRECTORY,sources=SOURCES,items_path=ROOT/'data/dev/validation.jsonl'):
    packet_path=directory/'packet.json'
    if packet_path.exists():return json.loads(packet_path.read_text())
    items=[json.loads(l) for l in items_path.read_text().splitlines() if l.strip()]
    ids={i['id'] for i in items}
    if not items or len(ids)!=len(items):raise ValueError('Unique nonempty review cases required')
    outputs={}
    for name,path in sources.items():
        report=json.loads(path.read_text())
        rows=report.get('development_generations',report.get('validation_outputs',[]))
        outputs[name]={r['item_id']:r['generated_standard'] for r in rows}
        if len(rows)!=len(ids) or set(outputs[name])!=ids:raise ValueError('Incomplete or duplicate model outputs')
    rng=random.Random(secrets.randbits(64));questions=[];mapping={}
    for candidate in sources:
        if candidate=='default':continue
        for item in items:
            order=['default',candidate];rng.shuffle(order)
            qid=f'{candidate}:{item["id"]}'
            questions.append(dict(id=qid,comparison=candidate,item_id=item['id'],conversation=item['session_memory'],message=item['message']['text'],A=outputs[order[0]][item['id']],B=outputs[order[1]][item['id']]))
            mapping[qid]=dict(zip(('A','B'),order))
    packet=dict(questions=questions,scope='STANDARD replies on reused held-out development conversations')
    directory.mkdir(parents=True,exist_ok=True)
    packet_path.write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n')
    (directory/'key.json').write_text(json.dumps(dict(mapping=mapping,source_sha256={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in sources.items()},packet_sha256=hashlib.sha256(packet_path.read_bytes()).hexdigest()),indent=2)+'\n')
    return packet

def save_vote(directory,qid,choice,notes):
    from filelock import FileLock
    if choice not in ('A','B','Tie','Both bad'):raise ValueError('Select a review choice')
    with FileLock(str(directory/'ratings.lock')):
        path=directory/'ratings.json'
        votes=json.loads(path.read_text()) if path.exists() else {}
        votes[qid]=dict(choice=choice,notes=notes)
        temporary=directory/'ratings.tmp';temporary.write_text(json.dumps(votes,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)

def main():
    import streamlit as st
    st.set_page_config(page_title='ReplyCoach blind A/B',layout='wide')
    packet=prepare();questions=packet['questions']
    st.title('Which reply would you send?')
    st.caption(packet.get('scope','STANDARD replies')+' • A/B positions shuffled per conversation. Judge relevance, speaker roles and supported facts; multiple replies can be valid.')
    labels=packet.get('comparison_labels',{'full_epoch1':'Comparison 1','full_epoch2':'Comparison 2','lora_epoch4':'Comparison 3'})
    comparison=st.sidebar.selectbox('Comparison',list(labels),format_func=labels.get)
    qs=[q for q in questions if q['comparison']==comparison]
    path=DIRECTORY/'ratings.json';votes=json.loads(path.read_text()) if path.exists() else {}
    st.sidebar.write(f"Saved: {sum(q['id'] in votes for q in qs)} / {len(qs)}")
    position_key=f'position_{comparison}'
    if position_key not in st.session_state:st.session_state[position_key]=next((i for i,q in enumerate(qs) if q['id'] not in votes),0)
    index=st.session_state[position_key];q=qs[index];qid=q['id'];previous=votes.get(qid,{})
    left,mid,right=st.columns([1,3,1])
    if left.button('Previous',disabled=index==0):st.session_state[position_key]-=1;st.rerun()
    mid.write(f"Conversation {index+1} / {len(qs)} · {q['item_id']}")
    if right.button('Next',disabled=index==len(qs)-1):st.session_state[position_key]+=1;st.rerun()
    for turn in q['conversation']:
        with st.chat_message('user' if turn['speaker']=='user' else 'assistant'):
            st.caption('You' if turn['speaker']=='user' else 'Friend');st.write(turn['text'])
    st.info('Incoming message: '+q['message'])
    a,b=st.columns(2)
    a.subheader('A');a.write(q['A']);b.subheader('B');b.write(q['B'])
    choices=['A','B','Tie','Both bad']
    with st.form(qid):
        choice=st.radio('Your choice',choices,index=choices.index(previous['choice']) if previous else None,horizontal=True)
        notes=st.text_area('Notes (optional)',value=previous.get('notes',''))
        save=st.form_submit_button('Save and next')
    if save:
        if choice is None:st.warning('Choose A, B, Tie or Both bad before saving.')
        else:
            save_vote(DIRECTORY,qid,choice,notes)
            st.session_state[position_key]=min(index+1,len(qs)-1);st.rerun()
    st.sidebar.caption('Progress saves locally and survives restarts. References are hidden so valid alternative replies can win.')
    if st.sidebar.button('Reveal identities and results'):
        key=json.loads((DIRECTORY/'key.json').read_text())
        st.write('Model identities',packet.get('identities',{'default':'Current default: spk1_ul025_s425', 'full_epoch1':'Full3e-4 best epoch1','full_epoch2':'Full1e-4 best epoch2','lora_epoch4':'LoRA3e-4 best epoch4'}))
        counts={k:0 for k in ['default',comparison,'Tie','Both bad']}
        for row in qs:
            vote=votes.get(row['id'])
            if vote:
                c=vote['choice'];counts[key['mapping'][row['id']][c] if c in ('A','B') else c]+=1
        st.write(counts);st.caption('Counts cover saved reviews only, not unreviewed cases. Revealing identities may influence subsequent ratings.')
        st.write('Current pair identities',key['mapping'][qid])

if __name__=='__main__':
    if '--prepare' in sys.argv:
        print('Prepared',len(prepare()['questions']),'A/B cases.')
    else:main()
