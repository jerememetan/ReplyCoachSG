import importlib.util,json
from pathlib import Path
import pytest
pytest.importorskip('transformers');pytest.importorskip('peft')
spec=importlib.util.spec_from_file_location('review_fit',Path(__file__).resolve().parents[1]/'scripts/review_fit_probe_v1.py')
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)

def test_exact_match_is_not_the_only_accepted_reply_and_missing_cases_fail(tmp_path):
    data=tmp_path/'data';data.mkdir();rows={}
    for split,file in [('train','train.jsonl'),('paraphrase','paraphrases.jsonl')]:
        rows[split]=[dict(item_id=f'{split}{i}',context=[],speakers=[],message='Could I borrow it?',response='Yes, you can borrow it.') for i in range(24)]
        (data/file).write_text(''.join(json.dumps(c)+'\n' for c in rows[split]))
    for name in review.VARIANTS:
        outputs={split:[dict(item_id=c['item_id'],generated_standard='Yes, you can borrow it.' if name=='base' else 'Sure, you may use it.',exact_target_match=name=='base') for c in items] for split,items in rows.items()}
        r=dict(status='complete',optimizer_steps=400,train_items=24,base_files_sha256={'same':'bytes'},train_generations=outputs['train'] if name=='base' else {'400':outputs['train']},paraphrase_generations=outputs['paraphrase'])
        (tmp_path/f'{name}.json').write_text(json.dumps(r))
    out=tmp_path/'review';review.prepare(tmp_path,data,out)
    p=json.loads((out/'packet.json').read_text());r=json.loads((out/'ratings.json').read_text());assert len(r['ratings'])==48
    done={(x['n'],x['option']) for x in r['ratings']}
    for q in p['questions']:
        for o in q['options']:
            if (q['n'],o) not in done:r['ratings'].append(dict(n=q['n'],option=o,adequacy='pass',reason='Valid alternative.'))
    (out/'ratings.json').write_text(json.dumps(r));review.score(out)
    summary=json.loads((out/'summary.json').read_text());assert all(summary['training_gate_pass'].values())
    assert all(c['pass']==24 for group in summary['counts'].values() for c in group.values())
    r['ratings'].pop();(out/'ratings.json').write_text(json.dumps(r))
    with pytest.raises(AssertionError):review.score(out)
