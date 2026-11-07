"""Recorded assistant curation for recognizable Singlish and a fixed standard meaning."""
import copy,json
from pathlib import Path
from hashlib import sha256
ROOT=Path(__file__).resolve().parents[1]
LOCAL={
'ITEM-A-003':"Checked already, I free!",
'ITEM-A-005':"Yellow button must press first, then press print.",
'ITEM-A-006':"Can, meet for dinner first.",
'ITEM-A-008':"I jio the others also already.",
'ITEM-A-009':"Faster leh, first class don't be late.",
'ITEM-A-014':"Before Sunday I finish, don't worry.",
'ITEM-A-015':"Can, I reaching there soon!",
'ITEM-A-020':"Okay, I reaching soon.",
'ITEM-A-021':"Cannot, seat already got people chope.",
'ITEM-A-023':"Never mind lah, long day today.",
'ITEM-A-025':"Okay, next time on time ah.",
'ITEM-A-029':"Can, I order from noodle stall for you.",
'ITEM-A-031':"Change to indoor plan lah.",
'ITEM-A-036':"Never mind, happens one.",
'ITEM-A-039':"Can, see you there!",
'ITEM-A-042':"Can! You all watching what movie ah?",
'ITEM-A-045':"Faster leh, prof might start soon.",
'ITEM-A-047':"See you already, coming over now.",
'ITEM-A-048':"Never mind, slowly come.",
'ITEM-A-056':"Confirm going lah.",
'ITEM-A-058':"Okay lah, understand.",
'ITEM-A-060':"Ya lor, ask them also.",
'ITEM-B-005':"Seven better for me, can.",
'ITEM-B-006':"Paiseh, today cannot. Need finish work first.",
'ITEM-B-007':"I coming, reach soon.",
'ITEM-B-016':"Ya, slides upload to course page already.",
'ITEM-B-017':"Thought you sending me. You need my recording also ah?",
'ITEM-B-018':"Can, I send you the link.",
'ITEM-B-020':"Never mind, I try find the notes somewhere else.",
'ITEM-B-021':"Paiseh, now cannot send. Later I send you.",
'ITEM-B-022':"By Friday evening must submit.",
'ITEM-B-024':"Can, after class we discuss.",
'ITEM-B-028':"Never mind, ready then send me.",
'ITEM-B-033':"Thanks ah, I go check new room.",
'ITEM-B-037':"Ya, campus gate we meet.",
'ITEM-B-039':"Can, I come over find you.",
'ITEM-B-041':"Can, we walk back together lah.",
'ITEM-B-044':"Eight tonight lah.",
'ITEM-B-045':"Ya, still on lah.",
'ITEM-B-046':"Thanks for jio, tonight cannot ah.",
'ITEM-B-053':"Can, confirm time first then I book.",
'ITEM-B-055':"Paiseh, today cannot join. Need finish assignment.",
'ITEM-B-058':"Can try the second floor printer.",
'ITEM-B-059':"You said bringing yours, I need bring mine also ah?",
'ITEM-B-060':"Found mine liao, can borrow when we meet.",
'ITEM-C-003':"I anything, both also can.",
'ITEM-C-004':"Seven start lah.",
'ITEM-C-005':"Ya, Tuesday we no class.",
'ITEM-C-006':"Okay lor, maybe I skip then.",
'ITEM-C-007':"Haven't eaten leh, I was waiting for you.",
'ITEM-C-009':"Can lah, I come over then we share table.",
'ITEM-C-010':"Got two seats on third floor liao.",
'ITEM-C-012':"Can, eleven also can.",
'ITEM-C-015':"Got the slides, can send you.",
'ITEM-C-021':"Haven't reach, about five more minutes.",
'ITEM-C-022':"Okay, few more minutes I reach.",
'ITEM-C-025':"Can, what kind of iced coffee you want?",
'ITEM-C-026':"Can, after class meet you.",
'ITEM-C-027':"Never mind lah, rest first. Next time we go.",
'ITEM-C-029':"Can, library we find somewhere quiet.",
'ITEM-C-030':"Ya, two o'clock I reach.",
'ITEM-C-031':"Okay lah, go back rest first.",
'ITEM-C-033':"Can, after class compare notes.",
'ITEM-C-035':"Swee, thanks for telling me.",
'ITEM-C-036':"Can, I send you the photo.",
'ITEM-C-039':"Never mind, rehearsal finish then we update you.",
'ITEM-C-040':"Swee, final version I check later.",
'ITEM-C-043':"Thanks ah, I go level four now.",
'ITEM-C-047':"Can, after class we meet.",
'ITEM-C-049':"Ya, yesterday that bus stop.",
'ITEM-C-050':"Never mind, slowly come over.",
'ITEM-C-052':"Can, I send you the link.",
'ITEM-C-054':"Thanks ah, I come over soon.",
'ITEM-C-057':"Can, I get one for you.",
'ITEM-C-059':"Ya, I waiting near entrance.",
'ITEM-C-060':"Never mind, I wait at court for you.",
'ITEM-D-002':"Aiyoh, I didn't know leh.",
'ITEM-D-004':"Never mind lah.",
'ITEM-D-005':"Don't want lah, I cannot take spicy.",
'ITEM-D-012':"Actually ah, that part I didn't catch clearly.",
'ITEM-D-014':"Maybe we take some time read up those topics lah.",
'ITEM-D-015':"Paiseh, I thought needed one.",
'ITEM-D-017':"Ya, confirm.",
'ITEM-D-018':"Tomorrow I got something on. Day after you free?",
'ITEM-D-019':"Paiseh, I forgot about that.",
'ITEM-D-021':"Just need a bit more time only.",
'ITEM-D-022':"I also almost done liao. First question you know the answer anot?",
'ITEM-D-023':"Ya, got changes, so we adding more user stories.",
'ITEM-D-024':"Ya, conclusion need change a bit, summarise our work better.",
'ITEM-D-031':"Ya, makeup class I going.",
'ITEM-D-043':"Can, I buy your ticket. Reaching then tell me.",
'ITEM-D-044':"Can, how much ah?",
'ITEM-D-047':"Ya, later still playing badminton.",
'ITEM-D-048':"Can, three rackets I bring.",
'ITEM-D-051':"Oh no, I go see got other courts free anot.",
'ITEM-D-052':"Okay, later I book.",
'ITEM-E-003':"Paiseh, you want come play with us anot?",
'ITEM-E-006':"Aiyah, next time lor.",
'ITEM-E-008':"Tonight want do together anot?",
'ITEM-E-014':"Ya, I know liao.",
'ITEM-E-017':"Can, see you later ah.",
'ITEM-E-018':"Can, tomorrow night we do together.",
'ITEM-E-019':"Swee lah!",
'ITEM-E-020':"Noted, thanks ah prof!",
'ITEM-E-022':"Paiseh, still need it leh. Next week then can return.",
'ITEM-E-023':"Swee, thanks ah!",
'ITEM-E-029':"Found seat for you liao, don't worry.",
'ITEM-E-031':"Okay lor, change to another day.",
'ITEM-E-034':"Tickets book liao.",
'ITEM-E-037':"Ya, I return one, don't worry.",
'ITEM-E-043':"Wah, okay lor.",
'ITEM-E-045':"Never mind lah.",
'ITEM-E-047':"No leh, got third instead.",
'ITEM-E-049':"Okay lah, never mind. All good.",
'ITEM-E-052':"Never mind, next week then meet.",
'ITEM-E-056':"Okay lor, you go first.",
'ITEM-E-060':"Can, two o'clock then eat.",
}

def main():
    source=ROOT/'artifacts/shared_intent_v1/local_candidates.json'
    anchor_path=ROOT/'data/distillation/shared_intent_anchors_v1.json'
    anchors=json.loads(anchor_path.read_text());raw=json.loads(source.read_text())
    if raw['status'] not in ('generated_pending_audit','incomplete_pending_repair'):raise ValueError('Generation not complete')
    if raw['anchors_sha256']!=sha256(anchor_path.read_bytes()).hexdigest():raise ValueError('Anchors changed')
    by={c['item_id']:c for c in raw['items']};items=[];ledger=[]
    for a in anchors['items']:
        old=by.get(a['item_id']);c=copy.deepcopy(old) if old else {k:a[k] for k in ('item_id','context','speakers','message')}
        old_local=old['replies']['local'] if old else None
        local=LOCAL.get(a['item_id'],old_local)
        if not local:raise ValueError('Unrepaired generation failure: '+a['item_id'])
        assert all(c[k]==a[k] for k in ('context','speakers','message'))
        if old:assert c['replies']['standard']==a['standard']
        c['replies']={'standard':a['standard'],'local':local}
        c['review']=dict(approved=True,reviewer='Codex assistant full input/pair/style inspection; not human gold',
            meaning_preserved=True,speaker_correct=True,natural_local=True,context_compatible=True)
        items.append(c);ledger.append(dict(item_id=a['item_id'],standard_edited=a['standard_edited'],
            original_standard=a['original_standard'],standard=a['standard'],generated_local=old_local,
            final_local=local,local_edited=old_local!=local))
    assert len(items)==190 and set(LOCAL)<={c['item_id'] for c in items}
    out=ROOT/'data/distillation/shared_intent_v1_reviewed.json'
    if out.exists():raise FileExistsError('Do not overwrite a frozen reviewed corpus')
    report={**raw,'status':'audited','items':items,'audit':{'reviewer':'Codex assistant; not independent human gold',
        'source_sha256':sha256(source.read_bytes()).hexdigest(),'standard_edits':anchors['standard_edits'],
        'local_edits':sum(c['local_edited'] for c in ledger),'repaired_generation_failures':sum(c['generated_local'] is None for c in ledger),
        'policy':'Recognizable natural Singlish, approximately same meaning, coherent selectable drafts. Minor politeness differences allowed. Particle counts do not establish quality.'}}
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (source.parent/'target_audit.json').write_text(json.dumps({'audit':report['audit'],'reviewed_sha256':sha256(out.read_bytes()).hexdigest(),'items':ledger},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report['audit'],indent=2))

if __name__=='__main__':main()
