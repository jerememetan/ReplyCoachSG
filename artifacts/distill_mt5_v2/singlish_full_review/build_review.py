"""Verify reviewed English anchors and bind explicit Singlish review decisions."""
import collections
import csv
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'src'))
from replycoach.generate.distill_mt5_v2 import common_subset,digest,input_only,reviewed,student_rows
from replycoach.generate.distill_mt5_v2_matching import match_reviewed_arms

HERE=Path(__file__).resolve().parent
RESULTS=ROOT/'results'

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def main():
    records=list(csv.DictReader((HERE/'decisions.tsv').open(),delimiter='\t'))
    inputs=read(RESULTS/'data/train_inputs.json')
    recipe=read(RESULTS/'recipe.json')
    assert len(records)==len(inputs)==768
    assert [int(r['index']) for r in records]==list(range(768))
    assert len({r['item_id'] for r in inputs})==768
    progress=read(HERE/'progress.json')
    english_summary=read(ROOT/'english_full_review/summary.json')
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    rows_by_arm={}; decisions_by_arm={}; paired={}; counts={}; evidence=[]; pilot_differences=[]
    for arm in ['sealion','qwen']:
        erawpath=RESULTS/f'outputs/{arm}/english_full.json'
        ereviewpath=RESULTS/f'reviews/{arm}/english_full.json'
        assert sha(erawpath)==english_summary['teachers'][arm]['raw_file_sha256']
        assert sha(ereviewpath)==english_summary['teachers'][arm]['review_file_sha256']
        english_raw=read(erawpath)['outputs']; ereview=read(ereviewpath)
        approved_english=reviewed(english_raw,ereview['decisions'],'english')
        rawpath=RESULTS/f'outputs/{arm}/singlish_full.json'
        report=read(rawpath); raw=report['outputs']
        assert rawpath.read_bytes()==(HERE/f'snapshots/{arm}.json').read_bytes()
        assert sha(rawpath)==progress['files'][f'outputs/{arm}/singlish_full.json']['sha256']
        expected_inputs=[r for r in inputs if r['item_id'] in approved_english]
        assert report['status']=='complete' and report['completed']==report['requested']==len(approved_english)==len(raw)
        assert [input_only(r) for r in raw]==expected_inputs
        assert report['identity']['input_sha256']==digest(expected_inputs)
        assert report['identity']['anchors_sha256']==digest(approved_english)
        assert report['identity']['recipe_sha256']==sha(RESULTS/'recipe.json')
        assert report['identity']['checkpoint']==recipe['teachers'][arm]['singlish']
        assert len({r['item_id'] for r in raw})==len(raw)
        raw_by_id={r['item_id']:r for r in raw}
        decisions=[]
        for record,source in zip(records,inputs):
            key=source['item_id']; choice=record[arm]
            assert (choice=='skip')==(key not in approved_english),(arm,key,choice)
            if choice=='skip':continue
            row=raw_by_id[key]
            assert row['standard_anchor']==approved_english[key]['target']
            reason=record[arm+'_reason']
            if reason=='=':reason=record['sealion_reason']
            assert reason.strip() and reason!='='
            decisions.append(dict(item_id=key,raw_sha256=digest(row),decision=choice,reason=reason))
        review=dict(reviewer='Assistant exploratory meaning/style review',blind=False,independent_human_gold=False,
            scope='full',stage='singlish',completed_at_utc=now,raw_file_sha256=sha(rawpath),
            english_review_file_sha256=sha(ereviewpath),english_generation_file_sha256=sha(erawpath),
            recipe_sha256=sha(RESULTS/'recipe.json'),protocol=recipe['review']['singlish'],
            interpretation='Compare every rewrite with its own exact accepted English anchor and supplied conversation. Keep roles, ownership, polarity, factual status, conditions, times, numbers, and material commitments. Context can resolve references and ordinary ellipsis. Reject material additions, lost conditions, changed roles or states, and unnatural/broken local wording. Neutral conversational English can be natural locally; particles or spelling changes alone do not establish fidelity or naturalness.',
            coverage=f'All {len(raw)} generated Singlish rewrites read with their English anchors and complete supplied contexts/speaker roles.',
            source_labels_consulted_during_review=False,targets_edited=0,decisions=decisions)
        approved_singlish=reviewed(raw,decisions,'singlish')
        assert len(decisions)==len({d['item_id'] for d in decisions})==len(raw)
        assert all(not d['edited'] for d in approved_singlish.values())
        reviewpath=RESULTS/f'reviews/{arm}/singlish_full.json'
        write(reviewpath,review)
        paired[arm]=student_rows(inputs,approved_english,approved_singlish)
        c=collections.Counter(d['decision'] for d in decisions)
        statuses=collections.Counter(r['singlish']['status'] for r in raw)
        counts[arm]=dict(generated=len(raw),reviewed=len(decisions),accepted=c['accept'],rejected=c['reject'],edited=0,
            generation_status_counts=dict(statuses),raw_file_sha256=sha(rawpath),review_file_sha256=sha(reviewpath),
            exact_english_copies=sum(r['singlish']['reply'].strip()==r['standard_anchor'].strip() for r in raw),
            accepted_exact_english_copies=sum(raw_by_id[k]['singlish']['reply'].strip()==raw_by_id[k]['standard_anchor'].strip() for k in approved_singlish),
            rejected_ids=[d['item_id'] for d in decisions if d['decision']=='reject'])
        counts[arm]['termination_counts']=dict(collections.Counter(str(r['singlish']['termination']) for r in raw))
        rows_by_arm[arm]=raw_by_id;decisions_by_arm[arm]={d['item_id']:d for d in decisions}
        evidence.extend([erawpath,ereviewpath,rawpath,reviewpath])
        old=read(RESULTS/f'reviews/{arm}/singlish_pilot.json')['decisions']
        old_raw={r['item_id']:r for r in read(RESULTS/f'outputs/{arm}/singlish_pilot.json')['outputs']}
        for prior in old:
            if prior['item_id'] not in decisions_by_arm[arm]:continue
            current=decisions_by_arm[arm][prior['item_id']]
            if current['decision']!=prior['decision']:
                key=prior['item_id'];pilot_differences.append(dict(teacher=arm,item_id=key,
                    pilot_decision=prior['decision'],full_decision=current['decision'],
                    pilot_reason=prior['reason'],full_reason=current['reason'],
                    english_anchor_same=old_raw[key]['standard_anchor']==raw_by_id[key]['standard_anchor'],
                    raw_reply_same=old_raw[key]['singlish']['reply']==raw_by_id[key]['singlish']['reply']))
    originalpath=ROOT/'stage/data/train_rewrite.json'
    assert sha(originalpath)==recipe['payload_manifest']['data/train_rewrite.json']
    original=read(originalpath)
    common=match_reviewed_arms(original,paired,inputs)
    common_ids=[r['item_id'] for r in common['original']]
    both=set(r['item_id'] for r in paired['sealion'])&set(r['item_id'] for r in paired['qwen'])
    summary=dict(status='full_singlish_review_complete',completed_at_utc=now,
        generated_and_reviewed_rewrites=sum(c['generated'] for c in counts.values()),teachers=counts,
        original_eligible=len(original),english_common_original_eligible=english_summary['common_english_with_original_eligibility'],
        common_paired_accepted_before_original_eligibility=len(both),common_paired_original_eligible=len(common_ids),
        common_paired_ids=common_ids,targets_edited=0,blind=False,independent_human_gold=False,
        limitation='Assistant exploratory screening of training targets, not an independent model accuracy estimate. Naturalness and borderline fidelity decisions are subjective. Exact-copy counts are descriptive, not semantic or style scores.',
        pilot_decision_differences=pilot_differences,pilot_reviews_unchanged=True,
        full_student_training_submitted=False,next_required='Freeze the reviewed common paired subset; run the full-data student fitting preflight before submitting matched student training.',
        verification=dict(complete_unique_decision_coverage=True,raw_file_download_hashes=True,
            exact_accepted_english_anchors=True,original_english_reviews_unchanged=True,
            generation_identity_and_checkpoint_verified=True,exact_conversation_inputs=True,zero_edited_targets=True),
        files={str(p.relative_to(ROOT)):sha(p) for p in evidence+[HERE/'decisions.tsv',HERE/'build_review.py',HERE/'borderline_second_pass.json',originalpath]})
    write(HERE/'summary.json',summary)
    fields=['index','item_id','context','message']+[a+'_'+f for a in ['sealion','qwen'] for f in ['english_anchor','singlish_reply','decision','reason']]
    with (HERE/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for i,row in enumerate(inputs):
            value=dict(index=i,item_id=row['item_id'],context=json.dumps([dict(speaker=p,text=t) for p,t in zip(row['speakers'],row['context'])],ensure_ascii=False),message=row['message'])
            for arm in ['sealion','qwen']:
                raw=rows_by_arm[arm].get(row['item_id']); decision=decisions_by_arm[arm].get(row['item_id'])
                value.update({arm+'_english_anchor':raw['standard_anchor'] if raw else '',arm+'_singlish_reply':raw['singlish']['reply'] if raw else '',arm+'_decision':decision['decision'] if decision else 'not_generated_english_rejected',arm+'_reason':decision['reason'] if decision else 'English rejected; no rewrite requested.'})
            writer.writerow(value)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['files','common_paired_ids','teachers']},indent=2))
    print(json.dumps({a:{k:v for k,v in c.items() if k!='rejected_ids'} for a,c in counts.items()},indent=2))

if __name__=='__main__':main()
