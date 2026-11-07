"""Review proposed RL rewards: streamlit run scripts/rl_reward_review.py."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare_rl_reward_review_v1 import prepare,FOLDER


def save_review(folder,qid,verdict,notes):
    from filelock import FileLock
    if verdict not in ('acceptable','unacceptable','uncertain'):raise ValueError('Explicit verdict required')
    with FileLock(str(folder/'human_reviews.lock')):
        path=folder/'human_reviews.json';saved=json.loads(path.read_text()) if path.exists() else {}
        saved[qid]=dict(verdict=verdict,reward={'acceptable':1,'unacceptable':-1,'uncertain':None}[verdict],notes=notes)
        tmp=folder/'human_reviews.tmp';tmp.write_text(json.dumps(saved,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def main():
    import streamlit as st
    st.set_page_config(page_title='RL reward policy review',layout='wide')
    packet=prepare();questions=packet['questions']
    path=FOLDER/'human_reviews.json';saved=json.loads(path.read_text()) if path.exists() else {}
    st.title('Would this reply deserve this reward?')
    st.caption('24 examples • Judge the reply as the next message from YOU. These are proposed labels, not predictions from a working reward model.')
    st.info('Acceptable = +1; Unacceptable = −1; Uncertain = no reward. A natural refusal or alternative can be acceptable. Reject explicit contradictions and reversed ownership. You can change any proposed label.')
    if 'reward_position' not in st.session_state:st.session_state.reward_position=next((i for i,q in enumerate(questions) if q['id'] not in saved),0)
    index=st.session_state.reward_position;q=questions[index];previous=saved.get(q['id'],{})
    st.sidebar.write(f'{len(saved)} / {len(questions)} saved')
    a,b,c=st.columns([1,3,1])
    if a.button('Previous',disabled=index==0):st.session_state.reward_position-=1;st.rerun()
    b.write(f"Example {index+1} / {len(questions)} · {q['category']}")
    if c.button('Next',disabled=index==len(questions)-1):st.session_state.reward_position+=1;st.rerun()
    st.subheader('Conversation')
    for speaker,text in zip(q['speakers'],q['context']):st.write(('You: ' if speaker=='user' else 'Friend: ')+text)
    st.write('Friend: '+q['message'])
    st.subheader('Proposed reply from you');st.write(q['draft'])
    st.write('**Proposed scoring:** '+q['proposed_verdict']+' · reward '+str(q['proposed_reward'] if q['proposed_reward'] is not None else 'none'))
    st.caption(q['reason'])
    options=['Accept proposed score','Change to acceptable (+1)','Change to unacceptable (−1)','Change to uncertain (no reward)']
    with st.form(q['id']):
        choice=st.radio('Your decision',options,index=None)
        if previous:st.caption('Previously saved: '+previous['verdict'])
        notes=st.text_area('Notes or a better reply (optional)',value=previous.get('notes',''))
        submit=st.form_submit_button('Save and next')
    if submit:
        if choice is None:st.warning('Choose a decision before saving.')
        else:
            verdict=q['proposed_verdict'] if choice==options[0] else ['acceptable','unacceptable','uncertain'][options.index(choice)-1]
            save_review(FOLDER,q['id'],verdict,notes)
            if index<len(questions)-1:st.session_state.reward_position+=1;st.rerun()
            else:st.success('Saved. All examples are complete if the progress counter shows 24/24.')
    st.sidebar.caption('Saved locally. Development and held-out scorer examples are separate. RL training has not started.')

if __name__=='__main__':main()
