"""All usable filtered public pairs, without the earlier4000/source sample caps."""
import hashlib,json
from collections import Counter
from pathlib import Path
from replycoach.generate.sealion_public_v1 import assign_groups,clean,pair_key,normal
from replycoach.generate.sealion_data import encode_reply
from replycoach.generate.sealion_v1 import checkpoint_tokenizer
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/public/sealion_full_v1';SOURCE=Path('/tmp/replycoach_public_dialogue_sources')
def rows(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if OUT.exists():raise FileExistsError('Use a fresh public version')
    OUT.mkdir(parents=True);tok=checkpoint_tokenizer(Path('/tmp/sealion3b_source'))
    forbidden={pair_key(dict(message=r['message']['text'],response=t)) for p in ['data/training_expansion/generated600_repair_v1/combined_train.jsonl','data/dev/validation.jsonl'] for r in rows(ROOT/p) for t in r['replies'].values()}
    quarantine={r['item_id'] for r in json.loads((ROOT/'data/public/sealion_dialogue_v1/quarantine_private.json').read_text())}
    oldval=rows(ROOT/'data/public/sealion_dialogue_v1/validation.jsonl');data={};stats={};source_hashes={}
    names={'cosem':('cosem_clean_train_pairs.jsonl','cosem_validation_pairs.jsonl'),'dailydialog':('dailydialog_train_pairs.jsonl','dailydialog_validation_pairs.jsonl')}
    for corpus,(tr,va) in names.items():
        print('Filtering',corpus,flush=True);rawtrain=assign_groups(rows(SOURCE/tr),corpus);rawval=assign_groups(rows(SOURCE/va),corpus);heldgroups={r['group_id'] for r in rawval}
        heldpairs=set();counts=Counter();selected=[];seen=set()
        for r in rawval:
            row,why=clean(r,corpus)
            if row is not None:heldpairs.add(pair_key(row))
        for r in rawtrain:
            if r['group_id'] in heldgroups:counts['heldout_group']+=1;continue
            row,why=clean(r,corpus)
            if why:counts[why]+=1;continue
            if pair_key(row) in heldpairs:counts['heldout_pair']+=1;continue
            if pair_key(row) in forbidden:counts['project_pair']+=1;continue
            if row['item_id'] in quarantine:counts['prior_quarantine']+=1;continue
            key=json.dumps([row['corpus'],row['context'],row['speakers'],row['message'],row['response']],ensure_ascii=False)
            if key in seen:counts['exact_contextual_duplicate']+=1;continue
            try:encode_reply(row,tok,384)
            except ValueError:counts['token_budget']+=1;continue
            seen.add(key);selected.append(row)
        data[corpus]=selected;stats[corpus]=dict(raw_train=len(rawtrain),raw_validation=len(rawval),included=len(selected),excluded=dict(counts),training_groups=len({r['group_id'] for r in selected}));source_hashes.update({tr:sha(SOURCE/tr),va:sha(SOURCE/va)})
    train=sorted(data['cosem']+data['dailydialog'],key=lambda r:r['item_id'])
    assert not {r['group_id'] for r in train}&{r['group_id'] for r in oldval}
    assert not {pair_key(r) for r in train}&{pair_key(r) for r in oldval}
    (OUT/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in train));(OUT/'validation.jsonl').write_bytes((ROOT/'data/public/sealion_dialogue_v1/validation.jsonl').read_bytes())
    pilot=[r for corpus in names for r in data[corpus][:20]];(OUT/'pilot_private.json').write_text(json.dumps(pilot,ensure_ascii=False,indent=2)+'\n')
    m=dict(status='prepared_pending_sample_inspection',train_rows=len(train),validation_rows=len(oldval),corpora=stats,source_sha256=source_hashes,files_sha256={p:sha(OUT/p) for p in ['train.jsonl','validation.jsonl','pilot_private.json']},scope='All unique usable contextual pairs from available historically-cleaned CoSEM and DailyDialog source exports. No4000 sample limits, per-chat caps, response caps or marker quota. Not the complete unfiltered published CoSEM corpus.',policy='Same conservative original filters and DIALOGUE tag; whole group/pair holdouts and project pairs excluded; six previous quarantine IDs excluded. Original public wording retained, DailyDialog punctuation detokenized.400 validation pairs identical to pilot.',limitations=['Pattern filters are not independent semantic quality or full de-identification guarantees.','CoSEM source was historically mT5-relatedness filtered before this experiment.','Public source validation groups are not trained; product locked test unread; raw public text stays ignored/private.'])
    (OUT/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(dict(train=len(train),corpora=stats),indent=2))
if __name__=='__main__':main()
