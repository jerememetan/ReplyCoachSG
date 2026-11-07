"""Freeze conservative paired-meaning eligibility; never edit team targets."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/training_expansion/sealion_rewrite_v1'
EXCLUDE={
'ITEM-A-009':'Ambiguous double negation in LOCAL.',
'ITEM-A-014':'Sunday deadline dropped.', 'ITEM-A-025':'On-time request becomes vague steady request.',
'ITEM-A-031':'Indoor activity plan becomes indoor meeting.', 'ITEM-A-045':'Might-start uncertainty becomes imminent start.',
'ITEM-A-047':'English observation of seeing peer omitted.', 'ITEM-B-004':'Prior source-role audit found buyer/receiver ambiguity.',
'ITEM-B-005':'Works better preference reduced to acceptance.', 'ITEM-B-017':'Known inconsistent sending roles in original conversation.',
'ITEM-B-023':'After finishing own work condition omitted.', 'ITEM-B-039':'LOCAL adds immediate departure.',
'ITEM-B-040':'Six-thirty detail omitted.', 'ITEM-B-041':'Walking-together action omitted.',
'ITEM-B-045':'LOCAL adds Saturday explicitly.', 'ITEM-B-053':'Once time confirmed condition omitted.',
'ITEM-B-059':'Prior source-role audit flagged ambiguous bring/return roles.', 'ITEM-B-060':'Known adapter ownership/source ambiguity.',
'ITEM-C-009':'Sharing table becomes small-table acceptance plus immediate arrival.',
'ITEM-C-010':'Finding two seats becomes having reserved one for peer.', 'ITEM-C-012':'Acceptance becomes preference for eleven.',
'ITEM-C-019':'LOCAL adds after-class arrival instruction.', 'ITEM-C-026':'Meeting becomes an explicit meal plan.',
'ITEM-C-035':'LOCAL adds found-object status.', 'ITEM-C-036':'LOCAL adds immediate sending.',
'ITEM-C-040':'Checking final version becomes references-settled statement.', 'ITEM-C-047':'After-class meeting becomes taking MRT together.',
'ITEM-C-050':'LOCAL adds right-exit condition.', 'ITEM-C-051':'LOCAL adds comparative MRT convenience claim.',
'ITEM-C-052':'LOCAL adds immediate sending.', 'ITEM-D-002':'Did-not-know statement omitted.',
'ITEM-D-004':'English acceptance versus LOCAL exasperation.', 'ITEM-D-011':'Certain remaining-work statement becomes tentative.',
'ITEM-D-014':'Study suggestion gains blame for inadequate previous study.',
'ITEM-D-015':'Thought necessary versus thought acceptable are different reasons.',
'ITEM-D-023':'User-story action and slight change detail not retained.',
'ITEM-D-049':'Tomorrow availability becomes need to exercise.',
'ITEM-E-008':'Prior grounding audit flagged ambiguous/not-started conversation.',
'ITEM-E-011':'Payment commitment becomes request for permission.',
'ITEM-E-024':'Missing work verb and substantial unneeded profanity difference.',
'ITEM-E-029':'Found seat becomes reserved seat.', 'ITEM-E-044':'Six-pm detail omitted.',
'ITEM-E-054':'Acknowledgement/reassurance reduced to generic agreement.',
'ITEM-B-029':'Different room numbers SR3-3 versus SR3-2.',
'ITEM-C-028':'LOCAL adds instruction to eat first.', 'ITEM-C-041':'LOCAL adds notification request.',
'ITEM-C-045':'LOCAL adds reserving seats.', 'ITEM-C-055':'LOCAL adds eating first.',
'ITEM-C-058':'Coming slowly versus arriving later.',
'ITEM-D-009':'Completion fact omitted and ease-of-question judgement added.',
'ITEM-D-010':'Unable to help versus rhetorical claim of not knowing how.',
'SYN-volume500-v1-full-044-00':'LOCAL adds instruction to others to use laptop.',
'SYN-volume500-v1-full-062-01':'Heading there at5:30 versus arriving by5:30.',
'SYN-volume500-v1-full-062-02':'English commits to booking while LOCAL only agrees to attend.'}
def rows(path):return [json.loads(l) for l in (ROOT/path).read_text().splitlines()]
def main():
    source='data/training_expansion/generated600_repair_v1/combined_train.jsonl';train=rows(source);val=rows('data/dev/validation.jsonl')
    seen={r['id'] for r in train+val}
    if set(EXCLUDE)-seen:raise ValueError('Unknown exclusion IDs')
    audit=[];counts={}
    for split,records in [('train',train),('validation',val)]:
        included=[]
        for r in records:
            reason=EXCLUDE.get(r['id']);eligible=reason is None
            audit.append(dict(id=r['id'],split=split,eligible=eligible,reason=reason or ('Original paired target comparison: compatible core actors, decision, facts and commitments; minor stylistic/implicit-context compression allowed.' if r['id'].startswith('ITEM-') else 'Reuse full600 assistant paired-target editorial review; final equivalence pass retained compatible core meaning.'),reviewer='Codex assistant',human_gold=False))
            if eligible:included.append(r['id'])
        counts[split]=dict(total=len(records),included=len(included),excluded=len(records)-len(included))
    OUT.mkdir(exist_ok=True)
    (OUT/'pair_audit.json').write_text(json.dumps(dict(source=source,source_sha256=hashlib.sha256((ROOT/source).read_bytes()).hexdigest(),scope='Conservative assistant eligibility audit, not human gold or automatic semantic guarantee. Exclusion is experiment-only; original team/source records unchanged.600 generated pairs reuse prior complete editorial review and final paired-equivalence pass.190 original training and50 validation pairs inspected in this audit.',counts=counts,items=audit),indent=2)+'\n')
    print(json.dumps(counts))
if __name__=='__main__':main()
