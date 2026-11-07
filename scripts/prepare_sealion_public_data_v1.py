"""Create bounded group-separated public subsets; raw text stays in ignored data."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
from replycoach.generate.sealion_public_v1 import assign_groups,clean,sample,pair_key
from replycoach.generate.sealion_data import encode_reply

ROOT=Path(__file__).resolve().parents[1]


def read_jsonl(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sources',type=Path,required=True);parser.add_argument('--tokenizer',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(exist_ok=True,parents=True)
    from replycoach.generate.sealion_v1 import checkpoint_tokenizer
    tok=checkpoint_tokenizer(args.tokenizer)
    forbidden=set()
    for split in ['train','validation']:
        for r in read_jsonl(ROOT/f'data/dev/{split}.jsonl'):
            for target in r['replies'].values():forbidden.add(pair_key({'message':r['message']['text'],'response':target}))
    paths={'cosem':('cosem_clean_train_pairs.jsonl','cosem_validation_pairs.jsonl'),'dailydialog':('dailydialog_train_pairs.jsonl','dailydialog_validation_pairs.jsonl')}
    all_train=[];all_val=[];corpora={};source_hashes={}
    for corpus,(trname,vaname) in paths.items():
        train_path,val_path=args.sources/trname,args.sources/vaname
        source_hashes[trname]=digest(train_path);source_hashes[vaname]=digest(val_path)
        raw_train,raw_val=assign_groups(read_jsonl(train_path),corpus),assign_groups(read_jsonl(val_path),corpus)
        held_groups={r['group_id'] for r in raw_val}
        excluded_groups=sum(r['group_id'] in held_groups for r in raw_train)
        raw_train=[r for r in raw_train if r['group_id'] not in held_groups]
        filters={};cleaned={}
        for name,rows in [('train',raw_train),('validation',raw_val)]:
            counts=Counter();eligible=[]
            for r in rows:
                row,reason=clean(r,corpus)
                if reason:counts[reason]+=1;continue
                if pair_key(row) in forbidden:counts['exact_project_message_target']+=1;continue
                try:encode_reply(row,tok,384)
                except ValueError:counts['token_budget']+=1;continue
                eligible.append(row)
            filters[name]=dict(counts);cleaned[name]=eligible
        held_pairs={pair_key(r) for r in cleaned['validation']}
        before=len(cleaned['train']);cleaned['train']=[r for r in cleaned['train'] if pair_key(r) not in held_pairs]
        tr=sample(cleaned['train'],4000,group_cap=200 if corpus=='cosem' else 8,response_cap=3,stratified=corpus=='cosem')
        va=sample(cleaned['validation'],200,group_cap=40 if corpus=='cosem' else 8,response_cap=3,seed=426)
        if len(tr)!=4000 or len(va)!=200:raise ValueError(f'{corpus}: only{len(tr)} training/{len(va)}validation rows after caps; revise explicit sample plan before training')
        assert not {r['group_id'] for r in tr}&{r['group_id'] for r in va}
        assert not {pair_key(r) for r in tr}&{pair_key(r) for r in va}
        all_train.extend(tr);all_val.extend(va)
        corpora[corpus]=dict(raw_train_after_group_exclusion=len(raw_train),raw_validation=len(raw_val),heldout_group_rows_excluded=excluded_groups,filters=filters,eligible={k:len(v) for k,v in cleaned.items()},heldout_pair_rows_excluded=before-len(cleaned['train']),selected_train=len(tr),selected_validation=len(va),train_groups=len({r['group_id'] for r in tr}),validation_groups=len({r['group_id'] for r in va}),train_strata=dict(Counter(r['stratum'] for r in tr)),validation_strata=dict(Counter(r['stratum'] for r in va)))
    # Balanced source interleaving is shuffled by the trainer per seed afterward.
    all_train.sort(key=lambda r:r['item_id']);all_val.sort(key=lambda r:r['item_id'])
    for name,rows in [('train',all_train),('validation',all_val)]:
        (out/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    pilot=[]
    for corpus in paths:
        candidates=[r for r in all_train if r['corpus']==corpus]
        pilot.extend(candidates[:20])
    (out/'pilot_private.json').write_text(json.dumps(pilot,ensure_ascii=False,indent=2)+'\n')
    manifest=dict(status='prepared_pending_sample_inspection',train_rows=8000,validation_rows=400,corpora=corpora,source_sha256=source_hashes,files_sha256={n:digest(out/n) for n in ['train.jsonl','validation.jsonl','pilot_private.json']},
        provenance='Original public replies only, no generation/rewrites. DailyDialog punctuation detokenized; CoSEM wording preserved. Input CoSEM clean set was historically selected by mT5 likelihood-relatedness, not human or LLM-chat gold.',
        grouping='CoSEM original chat IDs hashed; DailyDialog dialogues reconstructed from contiguous original pair export with strict continuity; repeated whole dialogues removed from train if present in validation.',
        policy='Both sources use training-only DIALOGUE tag, avoiding automatic corpus=style labeling. Two-party visible context required, not a guarantee that entire original CoSEM chats had only two participants. Train response repetition<=3, perchat caps; marker/shortform strata provide coverage, not style gold or model-selection reward.',
        limitations=['Pattern privacy/export checks are not proof of full de-identification. Keep raw/processed public text out of Git and do not quote private examples in user reports.','Filtering is not independent human semantic review; original pair extraction can contain topic shifts/truncated turns. Inspect a small sample before training.','Sources/public validation groups remain separate. No project evaluation or locked test enters public training.'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(train=8000,validation=400,corpora=corpora),indent=2))

if __name__=='__main__':main()
