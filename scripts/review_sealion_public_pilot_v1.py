"""Record bounded assistant sample inspection; quarantine six ambiguous/noisy pairs."""
import hashlib,json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'data/public/sealion_dialogue_v1'
QUARANTINE={
 'PUBLIC-001bc3ee4e7f7d851d2f':'Reply changes to rain/location from an unrelated latest message; unclear adjacency.',
 'PUBLIC-0027a128e19edb3fcf18':'Reply mixes a weather acknowledgment with a separate gaming topic.',
 'PUBLIC-0044694e0b228c00493d':'Greeting/name-like response without enough recoverable response meaning.',
 'PUBLIC-00700e57e13ba9826505':'Borrow request followed by a venue answer; the actual response intent is unclear.',
 'PUBLIC-01243c99eaf2a2dfb046':'Response ends with a historical date-like export fragment.',
 'PUBLIC-015f73143c11ff727316':'Unclear punchline/reference response; poor fit for generic reply adaptation.',
}


def main():
    manifest=json.loads((FOLDER/'manifest.json').read_text())
    train=[json.loads(l) for l in (FOLDER/'train.jsonl').read_text().splitlines()]
    found={r['item_id'] for r in train}&set(QUARANTINE)
    if found!=set(QUARANTINE):raise ValueError('Quarantine IDs missing or already applied')
    excluded=[{**r,'quarantine_reason':QUARANTINE[r['item_id']]} for r in train if r['item_id'] in QUARANTINE]
    kept=[r for r in train if r['item_id'] not in QUARANTINE]
    (FOLDER/'quarantine_private.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2)+'\n')
    (FOLDER/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in kept))
    for corpus,meta in manifest['corpora'].items():
        rows=[r for r in kept if r['corpus']==corpus]
        meta.update(selected_train=len(rows),train_groups=len({r['group_id'] for r in rows}),train_strata=dict(Counter(r['stratum'] for r in rows)))
    manifest.update(status='ready_exploratory',train_rows=len(kept),sample_inspection=dict(rows_inspected=40,source_counts={'cosem':20,'dailydialog':20},quarantined=6,reviewer='assistant',independent_human_review=False,interpretation='Small qualitative inspection, not certification of the whole dataset or a measured corpus correctness rate. Native wording is retained; only exclusions, no replacement replies.'),quarantine_ids=QUARANTINE)
    for p in FOLDER.glob('*.json*'):
        if p.name=='manifest.json':continue
        manifest['files_sha256'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    (FOLDER/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Public sample inspected;6 rows quarantined; training rows',len(kept))

if __name__=='__main__':main()
