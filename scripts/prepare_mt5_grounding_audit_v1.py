"""Freeze inference-only checkpoint audit; never edits training data."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'data/evaluation/mt5_grounding_audit_v1'


def main():
    FOLDER.mkdir(parents=True,exist_ok=True)
    previous=json.loads((ROOT/'data/training_expansion/mt5_act_follow_v1/evaluation.json').read_text())
    inputs=[{k:r[k] for k in ['item_id','conversation_id','context','speakers','message']} for r in previous[::2]]
    probes=[
        ('budget','My breakfast budget is seven dollars.','My breakfast budget is twenty-five dollars.','We can eat near campus.','Would you like the sixteen-dollar breakfast platter?'),
        ('duration','I must leave the workshop by 5:10.','I must leave the workshop by 5:50.','The review takes forty minutes.','Shall we start the review at five?'),
        ('availability','I can only attend the online briefing.','I can attend either the online or in-person briefing.','There is an in-person briefing tomorrow.','Will you come to the in-person briefing?'),
        ('completion','I have already paid the registration fee.','I have not paid the registration fee yet.','Thanks for checking the registration.','Have you paid the fee yet?'),
        ('knowledge','I checked the timetable: the pool closes at six.','I have not checked the swimming pool timetable.','The holiday timetable is the relevant one.','Does the pool close at six?'),
        ('ownership','You still have my tripod.','I still have your tripod.','It is safely in my bag.' ,'What should we do about returning the tripod?'),
    ]
    facts=[]
    for name,first,second,peer,message in probes:
        for variant,context in [('A',first),('B',second)]:
            # Holder remains coherent when the direction of lending changes.
            peer_turn=('It is safely in my bag.' if variant=='A' else 'Thanks for keeping it safe.') if name=='ownership' else peer
            key=f'GROUND-{name}-{variant}'
            inputs.append(dict(item_id=key,conversation_id=f'GROUND-{name}',context=[context,peer_turn],speakers=['user','peer'],message=message))
            facts.append(dict(item_id=key,changed_fact=context))
    # These inputs are inspected development probes, never locked-test records.
    inp=FOLDER/'inputs.json';inp.write_text(json.dumps(inputs,indent=2)+'\n')
    (FOLDER/'probe_notes.json').write_text(json.dumps(dict(scope='Inference-only development probes. A different response is not required if both are valid; inspect factual support, not exact target matching.',facts=facts,ownership_limit='Ownership probe necessarily changes peer holder acknowledgment too, to keep both scenarios coherent; it is not a single-token intervention.'),indent=2)+'\n')
    files=[*sorted(ROOT.glob('src/**/*.py')),ROOT/'scripts/run_mt5_grounding_audit_v1.py',ROOT/'scripts/run_mt5_act_follow_v1.py',ROOT/'scripts/run_mt5_reply_act_v1.py',ROOT/'scripts/prepare_mt5_grounding_audit_v1.py',ROOT/'scripts/v5_mt5_grounding_audit_v1.sbatch',inp]
    recipe=dict(experiment='mt5_grounding_audit_v1',training=False,default_changed=False,cases=len(inputs),
        stages=['earlier_dialogue_base','selected_standard','paired_direct','paired_gated_predicted','paired_gated_disabled'],
        purpose='Locate inherited vs acquired contextual failures using matched inference, fact-change probes, and training-example inspection.',
        limitations=['All probes are assistant-authored inspected development material; no unbiased test or human accuracy claim.','Base versus selected training checkpoint compares the combined training trajectory, not a causal isolated data effect.','Factual variation need not change a valid reply; refuse or counterpropose is allowed if consistent.','Ownership probe alters acknowledgment as well as ownership to preserve coherent conversation.','English generator audit only; existing LOCAL failures analyzed separately.'],
        files_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    rec=ROOT/'experiments/mt5_grounding_audit_v1_recipe.json';rec.write_text(json.dumps(recipe,indent=2)+'\n')
    with tarfile.open('/tmp/replycoach_mt5_grounding_audit_v1.tar.gz','w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(ROOT)))
        tar.add(rec,arcname='recipe.json')
    print('Frozen',len(inputs),'inference-only inputs and',len(files),'source files')

if __name__=='__main__':main()
