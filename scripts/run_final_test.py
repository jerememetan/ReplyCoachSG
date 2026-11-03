"""Frozen one-time held-out evaluation. Resume identical runs; never tune on results."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from replycoach.io import ROOT, read_jsonl
from replycoach.final_test import validate_test_rows, make_packet
from replycoach.judging import write_judging_packet


def digest(path):
    h=sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if any(os.environ.get(k) for k in ['REPLYCOACH_V4_CHECKPOINT','V5_CHECKPOINT',
                                      'REPLYCOACH_QWEN_BASE','REPLYCOACH_QWEN_DD_ADAPTER']):
        raise ValueError('Unset model-path overrides to use the exact frozen demo checkpoints')
    out=args.out.resolve()
    if not out.is_relative_to(ROOT/'experiments/judging'):
        raise ValueError('final output must be in the ignored experiments/judging directory')
    spec=importlib.util.spec_from_file_location('local_judge',ROOT/'scripts/judge_replies.py')
    judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
    config=judge.judge_settings(judge.read_env(ROOT/'.env'))
    frozen_judge=judge.prepare_judge(config)
    previous=json.loads((ROOT/'experiments/judging/four_systems_20261026_01/main_packet/ratings/run.json').read_text(encoding='utf-8'))
    assert all(frozen_judge[k]==previous[k] for k in frozen_judge), 'Judge changed since validation'
    assert judge.load_personas(ROOT/'data/evaluation/personas.csv')==previous['personas']
    assert (ROOT/'data/evaluation/judge_rubric.md').read_text(encoding='utf-8')==previous['rubric']
    model_manifest=json.loads((ROOT/'artifacts/reranker_qwen/team_bundle_20261030/manifest.json').read_text(encoding='utf-8'))
    for entry in model_manifest['files']:
        assert digest(ROOT/entry['path'])==entry['sha256'], 'A frozen model file changed'
    files=[ROOT/p for p in ['data/dev/train.jsonl','data/dev/validation.jsonl','data/test/test.jsonl',
        'data/split/manifest.json','data/evaluation/personas.csv','data/evaluation/judge_rubric.md',
        'scripts/judge_replies.py','scripts/run_final_test.py']]
    files+=sorted((ROOT/'src/replycoach').rglob('*.py'))
    freeze=dict(record_type='held_out_test_freeze',evaluation_split='test',memory='M2',
        sources=['BM25','MiniLM','V4','V5'],selector='V6 Qwen + DailyDialog',
        files_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in files},
        models=model_manifest['files'],judge=frozen_judge,personas=previous['personas'],
        test_query_count=60,expected_candidates=480,expected_ratings=4800,
        v6_rating_policy='Reuse source candidate rating, lowest anonymous ID for identical text',
        no_test_tuning=True)
    if args.resume:
        assert json.loads((out/'freeze.json').read_text(encoding='utf-8'))==freeze, 'Frozen inputs/settings changed'
    else:
        assert not out.exists(), 'Use --resume for this run; do not start another test experiment'
        out.mkdir(parents=True)
        (out/'freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
    assignment=json.loads((ROOT/'data/split/manifest.json').read_text(encoding='utf-8'))['assignment']
    expected={k for k,v in assignment.items() if v=='test'}
    assert len(expected)==60
    dev=read_jsonl(ROOT/'data/dev/train.jsonl')+read_jsonl(ROOT/'data/dev/validation.jsonl')
    items=validate_test_rows(read_jsonl(ROOT/'data/test/test.jsonl'),expected,dev)
    journal=out/'outputs.jsonl'
    outputs={r['item_id']:r['output'] for r in read_jsonl(journal)} if journal.exists() else {}
    assert set(outputs)<=expected
    if len(outputs)<60:
        import torch
        from replycoach import demo_resources
        from replycoach.all_models import run_all_models
        from replycoach.reply_bank import load_reply_bank
        torch.set_num_threads(2)
        bank=load_reply_bank(ROOT/'data/dev/train.jsonl')
        for index,item in enumerate(items,1):
            if item.id in outputs: continue
            output=run_all_models(item.message.text,item.memory_texts(),'M2',
                speakers=[t.speaker for t in item.session_memory],log=False,
                bm25_factory=lambda:demo_resources.get_bm25(bank.serialized),
                minilm_factory=lambda:demo_resources.get_minilm(bank.serialized),
                v4_factory=lambda:demo_resources.get_generator('v4'),
                v5_factory=lambda:demo_resources.get_generator('v5'),
                scorer_factory=lambda:demo_resources.get_shared_ranker('qwen_dailydialog'))
            if not output.complete: raise RuntimeError('Incomplete model comparison: '+str(output.source_errors)+str(output.ranking_error))
            outputs[item.id]=output.to_dict()
            with journal.open('a',encoding='utf-8') as f:
                f.write(json.dumps({'item_id':item.id,'evaluation_split':'test','output':outputs[item.id]},ensure_ascii=False)+'\n')
                f.flush()
            print(json.dumps({'event':'final_export','completed':len(outputs),'total':60}),flush=True)
        demo_resources.get_shared_ranker.clear();demo_resources.get_generator.clear()
        demo_resources.get_minilm.clear();demo_resources.get_minilm_model.clear()
        import gc;gc.collect()
    public,private,selections=make_packet(items,outputs)
    packet=out/'packet'
    if not packet.exists():
        write_judging_packet(packet,public,private)
        (out/'v6_selections.json').write_text(json.dumps(selections,indent=2),encoding='utf-8')
    else:
        assert read_jsonl(packet/'judge_inputs.jsonl')==public
        assert json.loads((out/'v6_selections.json').read_text(encoding='utf-8'))==selections
    # Every frozen file is checked again before the first judge call.
    assert all(digest(ROOT/name)==value for name,value in freeze['files_sha256'].items())
    print(json.dumps({'event':'final_judging','expected_ratings':4800}),flush=True)
    cmd=[sys.executable,str(ROOT/'scripts/judge_replies.py'),str(packet)]
    if (packet/'ratings/run.json').exists(): cmd.append('--resume')
    subprocess.check_call(cmd,cwd=ROOT)
    latest={}
    for r in read_jsonl(packet/'ratings/ratings.jsonl'): latest[(r['candidate_id'],r['persona_id'])]=r
    assert len(latest)==4800 and all(r['status']=='ok' for r in latest.values())
    mapping={r['candidate_id']:r for r in read_jsonl(packet/'PRIVATE_mapping.jsonl')}
    groups=defaultdict(list)
    for (cid,_),r in latest.items(): groups[(mapping[cid]['system'],mapping[cid]['style'])].append(r['rating'])
    for s in selections:
        for persona in freeze['personas']:
            groups[('V6',s['style'])].append(latest[(s['candidate_id'],persona['id'])]['rating'])
    summary=[dict(system=system,style=style,ratings=len(rows),
        **{key:sum(float(r[key]) for r in rows)/len(rows) for key in [*judge.SCORES,'would_send']})
        for (system,style),rows in sorted(groups.items())]
    result=dict(completed=True,evaluation_split='test',conversations=60,source_ratings=4800,
        summary=summary,finished_utc=datetime.now(timezone.utc).isoformat(),
        limitation='Correlated AI-persona ratings; not independent human judgments. No tuning from test results.')
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'event':'final_test_complete','conversations':60,'ratings':4800}),flush=True)


if __name__=='__main__': main()
