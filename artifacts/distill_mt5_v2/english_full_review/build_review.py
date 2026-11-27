"""Bind explicit assistant decisions to preserved full English teacher outputs."""
import collections
import csv
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / 'src'))
from replycoach.generate.distill_mt5_v2 import digest, input_only, reviewed

HERE = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    records = list(csv.DictReader((HERE / 'decisions.tsv').open(), delimiter='\t'))
    assert len(records) == 768
    assert [int(r['index']) for r in records] == list(range(768))
    inputs = read(RESULTS / 'data/train_inputs.json')
    recipe = read(RESULTS / 'recipe.json')
    assert len({r['item_id'] for r in inputs}) == len(inputs) == 768
    verified = read(RESULTS / 'english_full_download_verification.json')
    for name, spec in verified.items():
        assert sha(RESULTS / name) == spec['sha256'], name
    assert (ROOT / 'stage/data/train_inputs.json').read_bytes() == (RESULTS / 'data/train_inputs.json').read_bytes()
    accepted = {}
    details = {}
    pilot_differences = []
    raw_reports = {}
    reports = {}
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for arm in ('sealion', 'qwen'):
        path = RESULTS / f'outputs/{arm}/english_full.json'
        raw_report = read(path)
        raw = raw_report['outputs']
        assert raw_report['status'] == 'complete'
        assert raw_report['completed'] == raw_report['requested'] == len(raw) == 768
        assert [input_only(r) for r in raw] == inputs
        assert raw_report['identity']['recipe_sha256'] == sha(RESULTS / 'recipe.json')
        assert raw_report['identity']['checkpoint'] == recipe['teachers'][arm]['english']
        assert all(r['english']['status'] == 'complete' and r['english']['reply'].strip() for r in raw)
        decisions = []
        for record, row in zip(records, raw):
            reason = record[arm + '_reason']
            if reason == '=':
                reason = record['sealion_reason']
            assert reason and reason != '='
            decisions.append(dict(item_id=row['item_id'], raw_sha256=digest(row),
                                  decision=record[arm], reason=reason))
        report = dict(reviewer='Assistant exploratory contextual review', blind=False,
                      independent_human_gold=False, scope='full', stage='english',
                      completed_at_utc=now, raw_file_sha256=sha(path),
                      recipe_sha256=sha(RESULTS / 'recipe.json'),
                      protocol=recipe['review']['english'],
                      interpretation='Reasonable new choices, offers, personal status answers, and clarifications may pass where no explicit constraint is violated. Minor colloquial wording alone is not a rejection. Unsupported external specifics, contradictions, role reversals, missed requests, and incoherent responses are excluded. Borderline judgments are explained per item.',
                      source_labels_consulted=False, targets_edited=0,
                      coverage='All 768 saved English outputs read in full with their complete supplied conversation and speaker roles.',
                      decisions=decisions)
        accepted[arm] = reviewed(raw, decisions, 'english')
        assert len(decisions) == len({d['item_id'] for d in decisions}) == 768
        assert all(not a['edited'] for a in accepted[arm].values())
        write(RESULTS / f'reviews/{arm}/english_full.json', report)
        reports[arm] = report
        raw_reports[arm] = raw
        counts = collections.Counter(d['decision'] for d in decisions)
        details[arm] = dict(reviewed=768, accepted=counts['accept'], rejected=counts['reject'],
                            edited=0, raw_file_sha256=sha(path),
                            review_file_sha256=sha(RESULTS / f'reviews/{arm}/english_full.json'),
                            rejected_ids=[d['item_id'] for d in decisions if d['decision'] == 'reject'])
        old = read(RESULTS / f'reviews/{arm}/english_pilot.json')
        pilot_raw = {r['item_id']:r for r in read(RESULTS / f'outputs/{arm}/english_pilot.json')['outputs']}
        by_id = {d['item_id']:d for d in decisions}
        full_raw = {r['item_id']:r for r in raw}
        for prior in old['decisions']:
            current = by_id[prior['item_id']]
            if prior['decision'] != current['decision']:
                pilot_differences.append(dict(teacher=arm, item_id=prior['item_id'],
                    pilot_decision=prior['decision'], full_decision=current['decision'],
                    pilot_reason=prior['reason'], full_reason=current['reason'],
                    raw_reply_same=pilot_raw[prior['item_id']]['english']['reply']==full_raw[prior['item_id']]['english']['reply']))
    both = set(accepted['sealion']) & set(accepted['qwen'])
    original = read(ROOT / 'stage/data/train_rewrite.json')
    original_ids = {r['item_id'] for r in original}
    assert len(original_ids) == len(original) == 766
    summary = dict(status='full_english_review_complete', completed_at_utc=now,
                   reviewed_responses=1536, source_conversations=768,
                   teachers=details, common_english_accepted=len(both),
                   common_english_with_original_eligibility=len(both & original_ids),
                   common_english_ids=[r['item_id'] for r in inputs if r['item_id'] in both],
                   original_eligible=766, targets_edited=0,
                   independent_human_gold=False, blind=False,
                   limitation='Assistant exploratory screening of training targets, not model accuracy or an independent quality ranking. Borderline judgments remain subjective. No final paired dataset exists until Singlish review passes.',
                   pilot_decision_revisions=pilot_differences,
                   pilot_reviews_unchanged=True,
                   next_required='Generate Singlish from each teacher\'s own accepted English; review all rewrites, then freeze the common paired subset.',
                   new_generation_or_training_jobs_submitted=False,
                   verification=dict(complete_unique_decision_coverage=True,
                                     exact_input_identity=True, exact_raw_row_hashes=True,
                                     original_recipe_and_checkpoint_identity=True,
                                     remote_download_file_hashes=True,
                                     accepted_targets_match_unedited_raw=True),
                   files={str(p.relative_to(ROOT)):sha(p) for p in [HERE/'decisions.tsv',HERE/'build_review.py',
                         RESULTS/'reviews/sealion/english_full.json',RESULTS/'reviews/qwen/english_full.json',
                         RESULTS/'outputs/sealion/english_full.json',RESULTS/'outputs/qwen/english_full.json']})
    write(HERE / 'summary.json', summary)
    fields=['index','item_id','context','message','sealion_reply','sealion_decision','sealion_reason',
            'qwen_reply','qwen_decision','qwen_reason']
    with (HERE / 'comparison.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(inputs):
            record=dict(index=i,item_id=row['item_id'],
                context=json.dumps([dict(speaker=p,text=t) for p,t in zip(row['speakers'],row['context'])],ensure_ascii=False),
                message=row['message'])
            for arm in ('sealion','qwen'):
                record[arm+'_reply']=raw_reports[arm][i]['english']['reply']
                record[arm+'_decision']=reports[arm]['decisions'][i]['decision']
                record[arm+'_reason']=reports[arm]['decisions'][i]['reason']
            writer.writerow(record)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['common_english_ids','teachers','files']},indent=2))
    print(json.dumps({a:{k:v for k,v in d.items() if k!='rejected_ids'} for a,d in details.items()},indent=2))


if __name__=='__main__':
    main()
