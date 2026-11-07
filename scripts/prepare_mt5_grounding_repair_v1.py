"""Stage matched STANDARD-only target-version controls; no training submission."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    audit=json.loads((ROOT/'experiments/mt5_training_lineage_audit_v1.json').read_text())
    proposals={r['item_id']:r for r in audit['proposals']}
    original=[json.loads(l) for l in (ROOT/'data/dev/train.jsonl').read_text().splitlines()]
    paired=json.loads((ROOT/'data/training_expansion/mt5_act_follow_v1/train.json').read_text())
    out=ROOT/'data/training_expansion/mt5_grounding_repair_v1';out.mkdir(exist_ok=True,parents=True)
    control=[];repair=[]
    for r in original:
        row=dict(item_id=r['id'],context=[t['text'] for t in r['session_memory']],speakers=[t['speaker'] for t in r['session_memory']],message=r['message']['text'],response=r['replies']['standard'],style='STANDARD',origin='original training target')
        control.append(row)
        if row['item_id'] in proposals:
            p=proposals[row['item_id']]
            assert row['response']==p['old_target']
            repair.append({**row,'response':p['proposed_standard'],'origin':'assistant target repair/clarification; not independently human reviewed'})
        else:repair.append(dict(row))
    for r in paired:
        row={k:r[k] for k in ['item_id','context','speakers','message','response']}
        row.update(style='STANDARD',origin='unchanged assistant paired training target; legitimate alternative replies share contexts')
        control.append(row);repair.append(dict(row))
    assert len(control)==len(repair)==238
    changed=[]
    for a,b in zip(control,repair,strict=True):
        assert all(a[k]==b[k] for k in ['item_id','context','speakers','message','style'])
        if a['response']!=b['response']:changed.append(a['item_id'])
    assert set(changed)==set(proposals)
    paths=[]
    for name,data in [('control',control),('repair',repair)]:
        p=out/f'{name}.jsonl';p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data));paths.append(p)
    manifest=dict(status='prepared_not_trained',rows_per_arm=238,changed_target_ids=changed,
        scope='Matched STANDARD-only target version comparison. Exactly5 original reply targets differ; same238 conversations/variants. Historical originals, all validation and evaluation records, and LOCAL targets untouched.',
        provenance='Five corrections/clarifications are assistant proposals based on explicit conversations, not independently human-reviewed gold. Other targets are retained, not certified correct.',
        interpretation='A targeted consistency repair, not a demonstrated solution to all grounding errors. Starting-checkpoint audit shows failures predate these5 examples.',
        future_comparison='Use identical architecture, starting checkpoint, optimizer, classifier policy, training budget and heldout automatic-usefulness evaluation for both arms. Do not compare a changed-objective repair run directly against an old auxiliary-loss baseline.',
        new_training_submitted=False,sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Prepared matched238-row controls; exactly5 changed STANDARD targets; no original/LOCAL/validation edits.')

if __name__=='__main__':main()
