"""Recorded assistant editorial audit of the first frozen 190-input teacher batch."""
import copy,json
from hashlib import sha256
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
# Explicit editorial decisions; original generated candidates are preserved.
EDITS={
 'ITEM-A-001':("Yes, count me in for dinner.","Steady, count me in for dinner."),
 'ITEM-A-006':("What time should we meet downstairs after eating?","Eat first then meet downstairs what time?"),
 'ITEM-A-025':("No worries, just get here safely. I'm still waiting.","No worries, reach safely. I still waiting here."),
 'ITEM-A-027':("Yes, I'm taking the MRT. Want to travel together?","Yes, taking MRT. Want go together?"),
 'ITEM-A-031':("Let's wait inside until the rain slows down.","Wait inside first, let rain slow down."),
 'ITEM-B-017':("I thought you were sending it to me. Do you need mine too?","Thought you sending to me. Need my recording also?"),
 'ITEM-B-059':("You said you'd bring yours. Should I bring mine too?","You said bring yours, need me bring mine also?"),
 'ITEM-B-060':("I found mine; you can borrow it when we meet.","Found mine already, you can borrow when we meet."),
 'ITEM-C-004':("It starts at nine in the morning.","Start nine in the morning."),
 'ITEM-C-027':("No worries, get some rest. Let me know if you want to go another time.","No problem, go rest first. Want go next time then tell me."),
 'ITEM-C-054':("Thanks for letting me know. I'll head over soon.","Noted, thanks. I'll go over soon."),
 'ITEM-D-004':("No worries, just let me know when you're free next time.","No problem, next time you free then tell me."),
 'ITEM-D-011':("I think we still need to finish the research and final presentation.","I think still need to do research and final presentation."),
 'ITEM-D-015':("I was just trying to help with the slides.","Just want to help with slides only."),
 'ITEM-D-035':("No worries, text me when you're close.","No problem, text me when you're nearby."),
 'ITEM-D-045':("I think only a few of us from the team are going.","I think only a few from our team going."),
 'ITEM-D-047':("Yes, we're still playing badminton tonight.","Still playing badminton tonight, confirm."),
 'ITEM-D-052':("Sure, I'll remember to book the court for tomorrow.","Can, tomorrow's court I'll remember to book."),
 'ITEM-D-053':("Oh no, would you like me to check the printer?","Alamak, want me help check the printer?"),
 'ITEM-E-012':("No worries, I'll look for another outlet.","No problem, I'll go find another plug."),
 'ITEM-E-020':("Tuesday works. Let me know when the notes are ready.","Tuesday can. Notes ready then tell me."),
 'ITEM-E-022':("I can return the charger to you today.","Can, I return the charger to you today."),
 'ITEM-E-024':("Sure, I'll send you the notes my bot generated.","Can, I'll send you notes my bot made."),
 'ITEM-E-037':("I'll return your notes by tonight.","Can, I'll give back your notes by tonight."),
 'ITEM-E-042':("I've already secured seats for us.","I already chope seats for us."),
 'ITEM-E-043':("No worries, maybe we can go another time.","No problem, maybe next time then go."),
 'ITEM-E-044':("Six works for me. See you then.","Six can. See you then."),
 'ITEM-E-054':("Sure, I'll bring two spare rackets.","Can, I'll bring two extra rackets."),
 'ITEM-E-056':("No worries, go ahead. I'll see you another time.","No problem, you go first. Next time then meet."),
}

def main():
    source=ROOT/'artifacts/mt5_distill_v1/teacher_candidates.json'
    out=ROOT/'data/distillation/teacher190_v1_reviewed.json'
    if out.exists():raise FileExistsError('Do not overwrite a frozen reviewed corpus')
    raw=json.loads(source.read_text())
    if raw['status']!='generated_pending_audit' or raw['failures'] or len(raw['items'])!=190:raise ValueError('Complete generation required')
    revised=copy.deepcopy(raw);ledger=[]
    for c in revised['items']:
        before=copy.deepcopy(c['replies']);new=EDITS.get(c['item_id'])
        if new:c['replies']=dict(zip(('standard','local'),new))
        c['review']=dict(approved=True,reviewer='Codex assistant inspection of every source and pair; not human gold',
            meaning_preserved=True,speaker_correct=True,natural_local=True,context_compatible=True)
        ledger.append(dict(item_id=c['item_id'],decision='edit' if new else 'keep',before=before,after=c['replies']))
    revised['status']='audited'
    revised['audit']=dict(reviewer='Codex assistant; not an independent human review',source_sha256=sha256(source.read_bytes()).hexdigest(),
        inspected=190,edited=len(EDITS),policy='Plausible new facts and valid refusals allowed. Pair meaning and roles checked. Some ambiguous input exchanges receive useful clarifications.',
        limitation='Assistant-curated teacher targets; gains cannot be attributed to raw teacher output alone. No unreliable reward-judge auto-approval.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(revised,ensure_ascii=False,indent=2)+'\n')
    audit={'source_sha256':revised['audit']['source_sha256'],'reviewed_sha256':sha256(out.read_bytes()).hexdigest(),**revised['audit'],'items':ledger}
    (source.parent/'teacher_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='items'},indent=2))

if __name__=='__main__':main()
