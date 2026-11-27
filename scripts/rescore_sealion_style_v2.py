"""CPU-only guard correction and penalty trade-offs on saved model candidates."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import sealion_style_guard_v2 as revised
import sealion_style_v1 as original


def main():
    assert Path.home().name=='hm.teoh.2024'
    # Regression checks for the actual false rejections, plus meaning violations.
    compatible=[('I have not started making the slides yet.',"Haven't start making the slides leh."),
                ("That's too expensive for me. Could we find something cheaper?",'That one too expensive for me leh. Find something cheaper can?')]
    for anchor,candidate in compatible:assert revised.guard(anchor,candidate)['passed'],(anchor,candidate)
    incompatible=[('I have not booked room 4A.', 'I booked room 4A already.'),
                  ('I will return your notes to you.','You will return my notes to me.'),
                  ('My bottle is purple.','My bottle blue.'),
                  ('I am outside room 4A.','I outside room 4B.'),
                  ('Please send $19 to Mei before 18:25.','Send $90 to Mei after 18:25.')]
    assert all(not revised.guard(anchor,candidate)['passed'] for anchor,candidate in incompatible)
    report=json.loads((ROOT/'outputs/algorithms/results.json').read_text())
    assert report['status']=='complete' and report['source_bundle_weights_unchanged']
    lexicon=json.loads((ROOT/'outputs/algorithms/style_lexicon.json').read_text())
    folder=ROOT/'review/guard_v2';folder.mkdir(parents=True,exist_ok=True)
    output=[];sweeps=[]
    for source in report['outputs']:
        row=dict(item_id=source['item_id'],scope=source['scope'],english=source['standard_anchor'],
            baseline=source['methods']['beam4']['text'],methods={},original_guard=original.guard(source['standard_anchor'],source['methods']['beam4']['text']),
            revised_guard=revised.guard(source['standard_anchor'],source['methods']['beam4']['text']))
        for copy_weight in [0.,.25,.5,1.,2.]:
            for style_weight in [0.,.2]:
                valid,_=revised.rank(row['english'],source['candidate_pool'],lexicon,copy_weight,style_weight)
                chosen=valid[0] if valid else dict(text=row['english'],provenance='english_fallback_no_guard_pass')
                row['methods'][f'copy{copy_weight:g}_style{style_weight:g}']=chosen
        output.append(row)
    for key in output[0]['methods']:
        rows=[row for row in output if row['scope']=='reused_development']
        sweeps.append(dict(method=key,cases=len(rows),
            exact_copies=sum(original.canonical(row['english'])==original.canonical(row['methods'][key]['text']) for row in rows),
            unnecessary_copies=sum(original.copy_features(row['english'],row['methods'][key]['text'])['unnecessary_copy'] for row in rows),
            mean_style_heuristic=sum(original.style_score(row['methods'][key]['text'],lexicon) for row in rows)/len(rows),
            caution='Heuristic score and guard passage do not establish naturalness or correctness.'))
    summary=dict(status='complete',model_inference_performed=False,training_performed=False,
        source_job=357998,regression_checks_passed=len(compatible)+len(incompatible),
        original_baseline_copies=sum(original.canonical(row['english'])==original.canonical(row['baseline']) for row in output if row['scope']=='reused_development'),
        guard_correction='Normalize contractions before surface checks so Haven\'t and That\'s are not mistaken for names. Frozen v1 training data/objective unchanged.',
        baseline_rejections_removed_by_normalization=sum(not row['original_guard']['passed'] and row['revised_guard']['passed'] for row in output),
        sweeps=sweeps,scope='Exploratory rescoring of reused development candidates; no deployment selection.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (folder/'results.json').write_text(json.dumps(dict(summary=summary,outputs=output),indent=2,ensure_ascii=False)+'\n')
    (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (folder/'comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        fields=['item_id','scope','english','baseline']+list(output[0]['methods'])
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in output:writer.writerow({key:row[key] for key in fields[:4]}|{key:value['text'] for key,value in row['methods'].items()})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
