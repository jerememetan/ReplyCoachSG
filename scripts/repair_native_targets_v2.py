"""Faithful STANDARD normalizations of original LOCAL; never edit original LOCAL."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
# Reviewed against original LOCAL and conversation, not generated from a Singlish template.
STANDARD={
'ITEM-A-005':'Just press the yellow button first.',
'ITEM-A-008':'Of course I invited them. You can count on me.',
'ITEM-A-011':"Okay, I'll go reserve a seat first.",
'ITEM-A-014':"I will, don't worry.",
'ITEM-A-020':"Okay, I'm arriving now.",
'ITEM-A-021':"You can't; the seat has already been reserved.",
'ITEM-A-022':'Sure, hurry up.',
'ITEM-A-029':"Okay, I'll order noodles for you.",
'ITEM-A-031':"Let's meet indoors instead.",
'ITEM-A-039':"Sure, I'll meet all of you there.",
'ITEM-A-040':'Use mine for now.',
'ITEM-A-041':'Could you help print it for me too?',
'ITEM-A-042':'Sure, which movie?',
'ITEM-A-045':'Hurry, the professor is about to start.',
'ITEM-A-047':"Oh, I'm coming now.",
'ITEM-A-056':"Yes, I've confirmed it.",
'ITEM-A-057':"You can count on me. I've checked the information.",
'ITEM-B-005':'Yes, seven works for me.',
'ITEM-B-007':"Okay, I'm on my way.",
'ITEM-B-023':"Sure, I'll check it later.",
'ITEM-B-037':"Yes, we're still meeting at the gate.",
'ITEM-B-039':"Okay, I'm coming now.",
'ITEM-B-040':'Okay, that works.',
'ITEM-B-041':'Yes, sure.',
'ITEM-B-044':"It's at eight tonight.",
'ITEM-B-045':"Yes, it's still on for Saturday.",
'ITEM-B-047':'Sure, Friday then.',
'ITEM-B-049':"Sure, send me the details and I'll handle it.",
'ITEM-B-053':"Sure, I'll book it later.",
'ITEM-C-002':'What is there to eat there?',
'ITEM-C-003':'Either is fine with me.',
'ITEM-C-009':"That's fine; a small table is okay too. I'm coming over now.",
'ITEM-C-010':"Yes, on the third floor. I've already saved you a seat.",
'ITEM-C-011':"Yes, I'll reserve a seat for you first.",
'ITEM-C-012':'Yes, eleven is better for me too.',
'ITEM-C-015':"Yes, I have them. I'll send them to you in a bit.",
'ITEM-C-019':'Yes, 5:30 is fine. Come over after class.',
'ITEM-C-021':'Not yet, maybe five more minutes.',
'ITEM-C-022':"Okay, I'm coming over and will arrive soon.",
'ITEM-C-026':"Yes, it's still on. We'll eat after class.",
'ITEM-C-029':"Sure, let's find a quiet place in the library later.",
'ITEM-C-030':'Yes, two still works. See you there.',
'ITEM-C-035':"Great, you've found them. Thanks.",
'ITEM-C-036':"Sure, I'll send you the photo now.",
'ITEM-C-040':"Great, the references are done. I'll check later.",
'ITEM-C-043':"Okay, I'm coming to level four now.",
'ITEM-C-047':"Sure, let's take the MRT together later.",
'ITEM-C-050':"It's okay, come over when you find the right exit.",
'ITEM-C-051':"Sure, the MRT is easier. I'll meet you there.",
'ITEM-C-052':"Sure, I'll send you the link now.",
'ITEM-C-059':"Yes, I've arrived. I'm waiting at the entrance.",
'ITEM-C-060':"It's okay, I'll wait here. Come when you can.",
'ITEM-D-002':'Oh, oops.',
'ITEM-D-004':'Oh, come on.',
'ITEM-D-005':"No, I don't want it. I can't eat spicy food.",
'ITEM-D-008':'Would Saturday or Sunday work?',
'ITEM-D-011':'I think we still need to work on the slides.',
'ITEM-D-012':"I don't know.",
'ITEM-D-014':"We didn't study hard enough. We should spend more time studying.",
'ITEM-D-015':'My mistake. I thought it was okay.',
'ITEM-D-017':'Yes.',
'ITEM-D-022':"I'm almost finished. What's your answer to the first question?",
'ITEM-D-023':"Yes, we have to. They've changed so much.",
'ITEM-D-025':"No, I don't have time. Sorry about that.",
'ITEM-D-028':'Which row would you like to sit in?',
'ITEM-D-031':"Yes, sure. Let's go.",
'ITEM-D-032':'Okay, do we need to email the professor?',
'ITEM-D-034':"Yes, I've just left.",
'ITEM-D-043':"Okay, let me know when you're arriving.",
'ITEM-D-047':'Yes, of course.',
'ITEM-D-048':"Okay, I'll bring three.",
'ITEM-D-049':'Okay, I need to run and exercise a little.',
'ITEM-E-011':'Can I pay you later?',
'ITEM-E-019':"Okay, that's great!",
'ITEM-E-029':"Don't worry, I've already reserved a seat for you.",
'ITEM-E-033':"I'm in!",
'ITEM-E-043':"That's a shame. Okay.",
'ITEM-E-044':'All right, fine.',
'ITEM-E-048':"Yes, it's definitely legitimate.",
'ITEM-E-054':'Sure.',
}
QUARANTINE={
'ITEM-A-009':'Malformed double negation in LOCAL makes intended meaning unreliable.',
'ITEM-A-025':'Steady a bit is underspecified here; do not silently rewrite it as punctuality.',
'ITEM-A-055':'Ambiguous triple-check-with-you statement instead of a clear confirmation.',
'ITEM-B-004':'LOCAL asks peer to buy water in response to peer asking user to buy a drink; role conflict.',
'ITEM-E-008':'Asks whether peer finished immediately after peer explicitly said not started.',
'ITEM-E-022':'Prior lending/returning roles are internally inconsistent; not a reliable ownership target.',
'ITEM-E-024':'Incomplete LOCAL wording; do not fabricate a repair or treat as faithful supervised target.',
}


def prepare(folder=ROOT/'data/public/native_rewrite_v2'):
    folder.mkdir(parents=True,exist_ok=True)
    out=folder/'original_repaired.json'
    if out.exists():raise FileExistsError('Preserve frozen repair ledger')
    path=ROOT/'data/dev/train.jsonl';rows=[json.loads(l) for l in path.read_text().splitlines()]
    assert set(STANDARD)|set(QUARANTINE)<={r['id'] for r in rows}
    output=[];ledger=[]
    for r in rows:
        id=r['id'];reason=QUARANTINE.get(id)
        standard=STANDARD.get(id,r['replies']['standard'])
        ledger.append(dict(item_id=id,eligible=reason is None,reason=reason or 'Assistant checked LOCAL/context and faithful STANDARD normalization; not human gold',local_unchanged=True,standard_changed=standard!=r['replies']['standard'],original_standard=r['replies']['standard'],normalized_standard=standard))
        if reason:continue
        output.append(dict(item_id=id,split='validation' if id.startswith('ITEM-E-') else 'train',group_id='original-author:'+id.split('-')[1],context=[t['text'] for t in r['session_memory']],speakers=[t['speaker'] for t in r['session_memory']],message=r['message']['text'],standard=standard,local=r['replies']['local'],provenance='Original team-authored LOCAL preserved verbatim; STANDARD assistant-normalized/audited, not new human gold'))
    out.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    (folder/'original_repair_ledger.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),kept=len(output),quarantined=len(QUARANTINE),items=ledger,split_scope='Hold out original authorE for rewriter validation. Original epoch1 STANDARD generator has seen original190; this is not unseen end-to-end evaluation.'),ensure_ascii=False,indent=2)+'\n')
    print('Preserved',len(output),'original LOCAL targets; quarantined',len(QUARANTINE))

if __name__=='__main__':prepare()
