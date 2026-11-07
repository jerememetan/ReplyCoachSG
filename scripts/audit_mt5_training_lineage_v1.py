"""Compare actual training corpora without editing historical files or labels."""
import hashlib,json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT/'data/dev/train.jsonl'
EXPANDED=ROOT/'data/training_expansion/blanket600_v1/expanded.jsonl'
NOTES={
 'ITEM-A-014':'Deadline wording changes by Sunday to before Sunday, matching the incoming message more precisely.',
 'ITEM-B-004':'Original responds to peer purchase request by asking peer to buy for user. Revised target addresses the incoming purchase request. Direction mismatch is worth correcting.',
 'ITEM-B-017':'Earlier peer offers recording, then asks if user is sending it. Revised target clarifies inconsistent directions. Original may be a reciprocal promise; do not call it definitively false.',
 'ITEM-B-059':'Peer says they will bring charger, then asks user to bring one. Revised target clarifies possible shared responsibility. Both can be plausible choices; source is ambiguous.',
 'ITEM-B-060':'Conversation has an unclear lending direction. Expanded target newly claims adapter found; neither version should be accepted as verified factual gold automatically.',
 'ITEM-C-030':'Primarily wording/time-format normalization.',
 'ITEM-C-044':'Primarily grammar/abbreviation normalization.',
 'ITEM-D-012':'Rephrases admission of not understanding after previously saying yes. Not a clear factual repair.',
 'ITEM-D-043':'Revised reply explicitly commits to requested ticket purchase; original only requests arrival update.',
 'ITEM-D-048':'Revised reply names rackets instead of implicit three. Mostly explicitness.',
 'ITEM-E-002':'Grammar and readability repair.',
 'ITEM-E-008':'Original asks if peer finished immediately after peer says they have not started. Revised target offers help instead of ignoring that state.',
 'ITEM-E-022':'Wording repair of charger-return refusal; essentially same act.',
 'ITEM-E-024':'Wording/tone change; no demonstrated grounding repair.',
}
PROPOSALS={
 'ITEM-B-004':('Sure, what drink would you like me to buy?','Respond to the peer purchase request instead of reversing the buyer.'),
 'ITEM-B-017':('I thought you were sending the recording to me. Do you need one from me as well?','Clarify the inconsistent sending directions without assuming an unmentioned recording exists.'),
 'ITEM-B-059':('You said you would bring your charger. Do you need me to bring one too?','Clarify who is bringing a charger rather than silently changing responsibility.'),
 'ITEM-B-060':('I was still looking for my adapter. Are you asking to borrow mine?','Clarify the lending direction without asserting that the adapter was found.'),
 'ITEM-E-008':('Would you like to work on it together tonight?','Acknowledge not-started state by offering a next action rather than asking if it is finished.'),
}


def main():
    originals=[json.loads(l) for l in ORIGINAL.read_text().splitlines()]
    expanded=[json.loads(l) for l in EXPANDED.read_text().splitlines()]
    old={r['id']:r for r in originals}
    changes=[]
    for r in expanded:
        if r['id'] in old and r['replies']['standard']!=old[r['id']]['replies']['standard']:
            changes.append(dict(item_id=r['id'],context=r['session_memory'],message=r['message']['text'],
                original_target=old[r['id']]['replies']['standard'],expanded_target=r['replies']['standard'],assistant_observation=NOTES[r['id']]))
    assert len(changes)==14 and len(originals)==190 and len(expanded)==790
    exemplars=[r for r in expanded if r['id'] in ['SYN-HUMAN-EXP1-035','SYN-HUMAN-EXP1-021','SYN-HUMAN-EXP1-004']]
    assert len(exemplars)==3
    report=dict(status='complete',scope='Read-only corpus lineage audit; not an exhaustive semantic audit of all790 targets.',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ORIGINAL,EXPANDED]},
        original_rows=190,expanded_rows=790,changed_original_targets=changes,
        generation_sources=dict(Counter(r.get('provenance',{}).get('generator','original') for r in expanded)),
        supportive_training_examples=exemplars,
        findings=['Recent reply-act pilots replay original190 targets, while their initialization checkpoint was previously trained on an expanded corpus with14 original STANDARD targets revised. This is a real target-version change, not proof it caused the observed failures.',
                  'Representative budget, shift-end and borrower-return targets already teach correct constraint handling. A few bad targets cannot explain all generalization errors.',
                  'Some original/revised directions are ambiguous; do not blanket-copy all revised targets as human-approved gold.',
                  'The600 additions include500 Gemma-generated and100 assistant-generated examples. Authorization is not independent human quality validation.'],
        proposed_change='Use one explicit, versioned target policy for original conversations across decoder and act-label training, starting with five targeted repairs/clarifications. Do not edit original or historical corpora. First compare earlier checkpoint audit outputs before training.',
        proposals=[dict(item_id=key,old_target=old[key]['replies']['standard'],proposed_standard=value[0],reason=value[1],approval_status='assistant proposal, not independently human reviewed',applied=False) for key,value in PROPOSALS.items()],
        historical_files_changed=False,new_training_submitted=False)
    p=ROOT/'experiments/mt5_training_lineage_audit_v1.json';p.write_text(json.dumps(report,indent=2)+'\n')
    print('Recorded14 target-version differences,3 supportive examples,5 unapplied target proposals.')

if __name__=='__main__':main()
