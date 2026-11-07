"""Keep authored standard intentions, with recorded corrections to problematic targets."""
import json
from hashlib import sha256
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EDITS={
 'ITEM-A-014':"Okay, I'll finish it before Sunday. Don't worry.",
 'ITEM-B-004':"Sure, what drink would you like me to buy?",
 'ITEM-B-017':"I thought you were sending it to me. Do you need my recording too?",
 'ITEM-B-059':"You said you'd bring yours. Should I bring mine too?",
 'ITEM-B-060':"I found my adapter; you can borrow it when we meet.",
 'ITEM-C-030':"Yes, I'll be there at two.",
 'ITEM-C-044':"Yes, the lecturer is inside already, but class hasn't started yet.",
 'ITEM-D-012':"Actually, I didn't catch that part clearly.",
 'ITEM-D-043':"Okay, I'll buy your ticket. Let me know when you're nearby.",
 'ITEM-D-048':"Yes, I'll bring three rackets.",
 'ITEM-E-002':"Sorry, something came up and I can't make it. Can someone take my place?",
 'ITEM-E-008':"Would you like to work on it together tonight?",
 'ITEM-E-022':"Sorry, I still need it. I can only return it next week.",
 'ITEM-E-024':"No, you need to earn it. I'm not giving it away for free.",
}

def main():
    source=ROOT/'data/dev/train.jsonl';items=[json.loads(l) for l in source.read_text().splitlines()]
    out=ROOT/'data/distillation/shared_intent_anchors_v1.json'
    if out.exists():raise FileExistsError('Choose a fresh anchor version')
    rows=[]
    for c in items:
        rows.append(dict(item_id=c['id'],context=[t['text'] for t in c['session_memory']],speakers=[t['speaker'] for t in c['session_memory']],
            message=c['message']['text'],standard=EDITS.get(c['id'],c['replies']['standard']),
            original_standard=c['replies']['standard'],standard_edited=c['id'] in EDITS))
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(schema_version=1,source_train_sha256=sha256(source.read_bytes()).hexdigest(),
        purpose='One fixed standard intention per original training input; rewrite only its register',
        reviewer='Codex assistant reviewed all 190 input/standard targets; not new human gold',
        standard_edits=len(EDITS),items=rows),ensure_ascii=False,indent=2)+'\n')
    print('Prepared',len(rows),'anchors;',len(EDITS),'recorded standard-target revisions.')

if __name__=='__main__':main()
