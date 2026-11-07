"""Blind review of whole STANDARD/LOCAL pairs on fresh conversations."""
import hashlib,json,os,random,secrets
from pathlib import Path
from filelock import FileLock
ROOT=Path(__file__).resolve().parents[1]
DEFAULT_FOLDER=ROOT/'data/human_review/hybrid_pair_ab_v1'
ISSUES=('STANDARD misses context or facts','LOCAL changes STANDARD meaning','LOCAL sounds unnatural','Unsure about this pair')
CHOICES=('A','B','Tie','Both bad')


def prepare_packet(folder=DEFAULT_FOLDER,report_path=ROOT/'artifacts/hybrid_pair_ab_v1/paired_outputs.json'):
    packet_path=folder/'packet.json';key_path=folder/'key.json'
    if packet_path.exists():
        if hashlib.sha256(packet_path.read_bytes()).hexdigest()!=json.loads(key_path.read_text())['packet_sha256']:
            raise ValueError('Frozen review packet changed')
        return json.loads(packet_path.read_text())
    if key_path.exists():raise FileExistsError('Existing key without packet; preserve evidence')
    cases=json.loads((folder/'inputs.json').read_text());report=json.loads(report_path.read_text())
    if report['status']!='complete':raise ValueError('Both model outputs must be complete')
    ids={c['item_id'] for c in cases};outputs={r['item_id']:r for r in report['outputs']}
    if len(ids)!=len(cases) or len(outputs)!=len(report['outputs']) or set(outputs)!=ids:raise ValueError('Incomplete or duplicate paired outputs')
    if report['inputs']!=cases:raise ValueError('Inference inputs differ from review cases')
    rng=random.Random(secrets.randbits(64));questions=[];mapping={}
    for c in cases:
        order=['hybrid','earlier_paired'];rng.shuffle(order);options={}
        for letter,arm in zip(('A','B'),order):
            row=outputs[c['item_id']][arm]
            if not row['standard'].strip() or not row['local'].strip() or row['local_conditioned_on']!=row['standard']:
                raise ValueError('Missing output or wrong LOCAL anchor')
            options[letter]={k:row[k] for k in ('standard','local')}
        questions.append(dict(id=c['item_id'],conversation=[dict(speaker=s,text=t) for s,t in zip(c['speakers'],c['context'])],message=c['message'],options=options))
        mapping[c['item_id']]=dict(zip(('A','B'),order))
    packet=dict(scope='Fresh conversational inputs, paired human preference review. A/B shuffled independently per case; references and identities hidden.',questions=questions)
    packet_path.write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n')
    key_path.write_text(json.dumps(dict(mapping=mapping,packet_sha256=hashlib.sha256(packet_path.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest()),indent=2)+'\n')
    return packet


def save_vote(folder,qid,choice,issues,corrected_standard,corrected_local,notes):
    if choice not in CHOICES:raise ValueError('Explicit preference required')
    packet=json.loads((folder/'packet.json').read_text())
    if qid not in {q['id'] for q in packet['questions']}:raise ValueError('Unknown review case')
    if set(issues)!= {'A','B'} or any(i not in ISSUES for flags in issues.values() for i in flags):raise ValueError('Unknown issue')
    with FileLock(str(folder/'ratings.lock')):
        path=folder/'ratings.json';votes=json.loads(path.read_text()) if path.exists() else {}
        votes[qid]=dict(choice=choice,issues=issues,corrected_standard=corrected_standard,corrected_local=corrected_local,notes=notes,corrections_scope='Explicit human draft suggestions, not automatically accepted training targets',unmarked_issues='Not rated; absence of a flag is not a pass')
        temporary=folder/'ratings.tmp';temporary.write_text(json.dumps(votes,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def main():
    import streamlit as st
    folder=Path(os.environ.get('REPLYCOACH_PAIR_REVIEW_DIR',DEFAULT_FOLDER))
    st.set_page_config(page_title='ReplyCoach paired A/B',layout='wide')
    packet=prepare_packet(folder);questions=packet['questions']
    path=folder/'ratings.json';votes=json.loads(path.read_text()) if path.exists() else {}
    st.title('Which reply pair would you keep?')
    st.caption('Judge STANDARD relevance, LOCAL meaning preservation, and natural Singlish. Plausible replies, refusals and alternatives can all be valid. Model identities stay hidden.')
    st.sidebar.write(f'Saved: {len(votes)} / {len(questions)}')
    st.sidebar.caption('Progress saves locally. Optional problem flags and corrections help decide what to train next; no feedback is sent to training automatically.')
    if len(votes)==len(questions):st.success('All reviews are saved. Tell me when you are ready for the results.')
    if 'paired_position' not in st.session_state:st.session_state.paired_position=next((i for i,q in enumerate(questions) if q['id'] not in votes),0)
    index=st.session_state.paired_position;q=questions[index];previous=votes.get(q['id'],{})
    a,b,c=st.columns([1,3,1])
    if a.button('Previous',disabled=index==0):st.session_state.paired_position-=1;st.rerun()
    b.write(f'Conversation {index+1} / {len(questions)}')
    if c.button('Next',disabled=index==len(questions)-1):st.session_state.paired_position+=1;st.rerun()
    for turn in q['conversation']:
        st.write(('You: ' if turn['speaker']=='user' else 'Friend: ')+turn['text'])
    st.info('Incoming message: '+q['message'])
    left,right=st.columns(2)
    for column,letter in ((left,'A'),(right,'B')):
        column.subheader(letter);column.caption('STANDARD');column.write(q['options'][letter]['standard'])
        column.caption('LOCAL');column.write(q['options'][letter]['local'])
    with st.form(q['id']):
        choice=st.radio('Which pair is better overall?',CHOICES,index=CHOICES.index(previous['choice']) if previous else None,horizontal=True)
        with st.expander('Optional: flag problems or write a better pair'):
            flags_a=st.multiselect('Problems in A',ISSUES,default=previous.get('issues',{}).get('A',[]))
            flags_b=st.multiselect('Problems in B',ISSUES,default=previous.get('issues',{}).get('B',[]))
            corrected_standard=st.text_area('Your better STANDARD reply (optional)',value=previous.get('corrected_standard',''))
            corrected_local=st.text_area('Your better LOCAL reply with the same meaning (optional)',value=previous.get('corrected_local',''))
            notes=st.text_area('Notes (optional)',value=previous.get('notes',''))
        save=st.form_submit_button('Save and next')
    if save:
        if choice is None:st.warning('Choose A, B, Tie or Both bad before saving.')
        else:
            save_vote(folder,q['id'],choice,{'A':flags_a,'B':flags_b},corrected_standard,corrected_local,notes)
            st.session_state.paired_position=min(index+1,len(questions)-1);st.rerun()
    # No identity reveal while human reviews are incomplete.
    if len(votes)==len(questions) and st.sidebar.button('Reveal model identities and results'):
        key=json.loads((folder/'key.json').read_text());counts={'hybrid':0,'earlier_paired':0,'Tie':0,'Both bad':0}
        for qid,vote in votes.items():
            choice=vote['choice'];counts[key['mapping'][qid][choice] if choice in ('A','B') else choice]+=1
        st.write('Hybrid: epoch1 STANDARD model followed by the earlier LOCAL rewriter. Earlier paired: the earlier model generates both styles sequentially.')
        st.write(counts)

if __name__=='__main__':main()
