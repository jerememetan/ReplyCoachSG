"""Freeze proposed reward-policy examples; no learned scorer or RL update."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'data/reward_diagnostics/targeted_rl_v1'


def scenarios():
    result=[]
    for split,offset in [('review',0),('development',2),('heldout',4)]:
        for j in range(2):
            k=offset+j
            earliest=14+k;request=earliest-1
            result.append(dict(id=f'TIME-{k}',split=split,category='time',context=[f'I cannot leave before {earliest}:00.'],speakers=['user'],message=f'Can we meet at {request}:00?',facts=dict(earliest_departure=earliest,requested_meeting=request),drafts=[
                (f'I cannot make {request}:00. Could we meet after {earliest}:00?','acceptable','Explicitly respects the departure constraint and proposes a later time.'),
                (f'Sure, see you at {request}:00.','unacceptable','Accepts a meeting before the user can leave.'),
                ('Okay, see you then.','unacceptable','Unqualified agreement accepts the requested impossible time.'),
                ('Let me check and get back to you.','uncertain','Defers the decision without contradicting the constraint; whether this is sufficient deserves user review.')]))
            days=[('Monday','Tuesday'),('Wednesday','Thursday'),('Friday','Saturday'),('Sunday','Monday'),('Tuesday','Wednesday'),('Thursday','Friday')]
            busy,free=days[k]
            result.append(dict(id=f'AVAIL-{k}',split=split,category='availability',context=[f'I am not available on {busy}, but I am free on {free}.'],speakers=['user'],message=f'Shall we meet on {busy}?',facts=dict(unavailable=busy,available=free),drafts=[
                (f'I cannot meet on {busy}. Would {free} work instead?','acceptable','Declines the unavailable day and offers the explicitly available one.'),
                (f'Yes, {busy} works for me.','unacceptable','Contradicts explicit unavailability.'),
                ('Sorry, I cannot make it.','acceptable','A valid refusal; proposing an alternative is optional.'),
                ('Maybe, I will let you know.','uncertain','Does not commit, but fails to clearly preserve stated unavailability.')]))
            objects=['calculator','umbrella','headphones','textbook','charger','camera'];obj=objects[k]
            if k%2==0:
                context=f'I lent you my {obj} this morning.';message=f'Can I keep your {obj} until tomorrow?'
                good=f'Yes, you can keep my {obj} until tomorrow.';bad=f'I can keep your {obj} until tomorrow.'
                refusal=f'Sorry, I need my {obj} back today.'
                roles=dict(owner='user',borrower='peer')
            else:
                context=f'I borrowed your {obj} this morning.';message=f'Could you return my {obj} this evening?'
                good=f'Yes, I will return your {obj} this evening.';bad=f'Please return my {obj} to me this evening.'
                refusal=f'I cannot return your {obj} this evening. Could I bring it tomorrow instead?'
                roles=dict(owner='peer',borrower='user')
            result.append(dict(id=f'OWNER-{k}',split=split,category='ownership',context=[context],speakers=['user'],message=message,facts=dict(object=obj,**roles),drafts=[
                (good,'acceptable','Preserves who owns and who holds the object.'),
                (bad,'unacceptable','Reverses ownership or the direction of return.'),
                (refusal,'acceptable','A refusal or counterproposal is allowed; ownership and roles remain correct.'),
                ('Okay, I will sort it out.','uncertain','No explicit reversal, but the intended action is underspecified.')]))
    return result


def prepare(output=FOLDER):
    if output.exists():return json.loads((output/'review_packet.json').read_text())
    groups=scenarios();rows=[]
    for g in groups:
        for n,(draft,verdict,reason) in enumerate(g['drafts'],1):
            rows.append(dict(id=f"{g['id']}-{n}",scenario_id=g['id'],split=g['split'],category=g['category'],context=g['context'],speakers=g['speakers'],message=g['message'],draft=draft,proposed_verdict=verdict,proposed_reward={'acceptable':1,'unacceptable':-1,'uncertain':None}[verdict],reason=reason,provenance='Assistant-authored proposed policy label; not a scorer prediction or human-approved gold.'))
    output.mkdir(parents=True)
    for split in ('review','development','heldout'):
        (output/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows if r['split']==split))
    packet=dict(purpose='Human review of proposed reward policy; no automatic scoring capability established',questions=[r for r in rows if r['split']=='review'])
    (output/'review_packet.json').write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n')
    manifest=dict(status='awaiting_user_policy_review',scenario_groups=18,review_drafts=24,development_drafts=24,heldout_drafts=24,human_approved=0,rl_updates=0,scorer_implemented=False,limitation='Template family shared across partitions; disjoint scenario instances do not establish broad language generalization. Heldout labels assistant-authored, not human gold. Neither packet nor labels constitute an automated reward scorer.',files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*') if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return packet

if __name__=='__main__':print('Prepared',len(prepare()['questions']),'policy review drafts.')
