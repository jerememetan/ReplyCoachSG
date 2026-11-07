"""Recorded manual audit of translation candidates; native LOCAL stays verbatim."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];FOLDER=ROOT/'data/public/native_rewrite_v2'
# Decisions made by contextual/linguistic assistant inspection, not human gold.
KEEP_TRAIN={1,2,6,8,9,10,11,12,13,14,17,18,20,21,22,23,27,28,31,32,33,35,36,38,41,42,43,45,46,47,48,49,50,52,55,57,59,60,61,62,64,66,67,68,69,71,72,75,77,78,80,81,82,83,85,87,88,92,93,94,95,96,98,101,103,105,107,109,110,112,113,115,116,117,118,120}
KEEP_VALIDATION={1,2,3,4,5,6,7,8,10,12,13,15,16,17,18,19,20,21,22,24,25,26,27,30}
STANDARD={
'NATIVE-PUB-train-006':"You take taxis, so you don't know the difference, haha. Adults no longer get concession fares?",
'NATIVE-PUB-train-022':'Over four weeks, maybe more, depending on your preparation.',
'NATIVE-PUB-train-027':'Oh my goodness... what do we do?',
'NATIVE-PUB-train-031':'Is the balloon filled with helium? How are you going to carry it? Haha.',
'NATIVE-PUB-train-036':"Hahaha, that's so cute.",
'NATIVE-PUB-train-047':'I only saw one for yesterday.',
'NATIVE-PUB-train-048':"Where is dinner? Actually, I'll join you all for dinner. Sorry, I'll need to leave after that.",
'NATIVE-PUB-train-055':"Yes, ask your friends when you're free.",
'NATIVE-PUB-train-064':"Yes. Why aren't you bringing your controller, anyway?",
'NATIVE-PUB-train-066':"I like the one where my face is really big. Crop your own faces into it, haha.",
'NATIVE-PUB-train-071':"I don't talk to her anymore, haha.",
'NATIVE-PUB-train-080':"I'm already at the staff room.",
'NATIVE-PUB-train-083':'Is anything replacing that, then? As safe as can be. 🌚',
'NATIVE-PUB-train-085':'What time?',
'NATIVE-PUB-train-088':'Hey, everyone! Do you all want to watch The Lion King together? I have15% off!',
'NATIVE-PUB-train-094':"Just pretend you don't know and stay quiet. It's so easy.",
'NATIVE-PUB-train-095':'So, is that okay?',
'NATIVE-PUB-train-103':"Hey, we're all going to the same class! A reunion!",
'NATIVE-PUB-train-105':"No, I've got chemistry. Where were you planning to go after school?",
'NATIVE-PUB-train-107':'Hahaha, girls too?',
'NATIVE-PUB-train-110':'Sorry, my arm really hurts. Can I be excused from army duties for a week?',
'NATIVE-PUB-train-112':"Haha, the sunset is nice, isn't it? You're good at this! 💋",
'NATIVE-PUB-train-120':'Just enter the ballot for the good ones, whatever fits your timetable.',
'NATIVE-PUB-validation-001':"I'm laughing too. Oh, come on.",
'NATIVE-PUB-validation-003':"Oh, haha, there's a lot of drama about this. I can tell you all next time. :))))",
'NATIVE-PUB-validation-013':"Just tell us who; we're not asking you to confess. :p",
'NATIVE-PUB-validation-018':"Haha, okay, see you! I'll wait for you at the gantry there—the Guardian one.",
'NATIVE-PUB-validation-024':"Oh, that's rough. Okay, let's eat after you finish. All the best for your paper tomorrow!",
'NATIVE-PUB-validation-026':"It doesn't seem to be working.",
'NATIVE-PUB-validation-030':"That's so early! Wait, are you using up your leave? What do you do in the army?",
}


def audit(folder=FOLDER):
    path=folder/'teacher_normalizations.json';data=json.loads(path.read_text());assert data['status']=='complete' and len(data['items'])==150
    out=folder/'public_repaired.json'
    if out.exists():raise FileExistsError('Preserve audit decisions')
    kept=[];ledger=[]
    for row in data['items']:
        n=int(row['item_id'].rsplit('-',1)[1]);valid=n in (KEEP_TRAIN if row['split']=='train' else KEEP_VALIDATION)
        if row['standard_candidate'] is None:valid=False
        standard=STANDARD.get(row['item_id'],row['standard_candidate'])
        ledger.append(dict(item_id=row['item_id'],eligible=valid,standard_corrected=row['item_id'] in STANDARD,local_unchanged=True,reason='Assistant audited faithful normalization; not human gold' if valid else 'Quarantined: opaque referents/names, merged or malformed fragments, ambiguous translation or inappropriate conversational wording; original data retained'))
        if valid:
            kept.append(dict(item_id=row['item_id'],split=row['split'],group_id=row['group_id'],context=row['context'],speakers=row['speakers'],message=row['message'],local=row['local'],standard=standard,provenance='Unmodified native CoSEM LOCAL; automatic STANDARD candidate individually assistant-audited/corrected, not human gold'))
    assert set(STANDARD)<={r['item_id'] for r in kept}
    out.write_text(json.dumps(kept,ensure_ascii=False,indent=2)+'\n')
    (folder/'public_audit_ledger.json').write_text(json.dumps(dict(candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),kept=len(kept),excluded=150-len(kept),standard_corrections=len(STANDARD),items=ledger),indent=2)+'\n')
    print('Kept',len(kept),'native utterances; quarantined',150-len(kept),'English corrections',len(STANDARD))

if __name__=='__main__':audit()
