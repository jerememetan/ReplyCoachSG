"""Freeze existing eligible rewrite pairs and development-only style controls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'artifacts/sealion_style_v1'
SPEC = importlib.util.spec_from_file_location('style_tools',ROOT/'src/replycoach/generate/sealion_style_v1.py')
STYLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STYLE)
TEAM_SHA='03485c7d6b510711fbdee5f3eac0044e334d9d7a25bf3c5c86665b0646e8eee7'


def read(path):return json.loads((ROOT/path).read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def records(path):return [json.loads(line) for line in (ROOT/path).read_text().splitlines() if line.strip()]
def write(name,value):(DEST/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def prepare():
    DEST.mkdir(exist_ok=True)
    assert digest(ROOT/'data/dev/train.jsonl')==TEAM_SHA
    original=(ROOT/'data/dev/train.jsonl').read_bytes()
    source=ROOT/'data/training_expansion/generated600_repair_v1/combined_train.jsonl'
    assert source.read_bytes().startswith(original)
    audit=read('data/training_expansion/sealion_rewrite_v1/pair_audit.json')
    splits={}
    for split,path in [('train','data/training_expansion/generated600_repair_v1/combined_train.jsonl'),
                       ('validation','data/dev/validation.jsonl')]:
        allowed={row['id'] for row in audit['items'] if row['split']==split and row['eligible']}
        pairs=[dict(item_id=row['id'],context=[t['text'] for t in row['session_memory']],
            speakers=[t['speaker'] for t in row['session_memory']],message=row['message']['text'],
            standard_anchor=row['replies']['standard'],response=row['replies']['local'],
            original_provenance=row.get('origin','unknown')) for row in records(path) if row['id'] in allowed]
        assert len(pairs)==(745 if split=='train' else 42)
        splits[split]=pairs
        write(split+'.json',pairs)
        auxiliary=[row for row in pairs if STYLE.auxiliary_eligible(row)]
        assert auxiliary
        write('copy_'+split+'.json',auxiliary)
    assert not {r['item_id'] for r in splits['train']} & {r['item_id'] for r in splits['validation']}
    baseline=read('artifacts/sealion_contrast_v1/repair_results.json')
    assert baseline['stages']['english']['selected_step']==0
    assert baseline['stages']['rewrite']['selected_step']==25
    english={r['item_id']:r['output'] for r in baseline['outputs']['english']['beam4_ng3']}
    local={r['item_id']:r for r in baseline['outputs']['rewrite']['beam4_ng3']['own_english']}
    evaluation=sum((read(path) for path in ['data/evaluation/sealion_full_rewrite_v1/inputs.json',
        'data/evaluation/restored_pipeline_v1/inputs.json','data/evaluation/context_pairs_v1/inputs.json']),[])
    assert len(evaluation)==len({r['item_id'] for r in evaluation})==126
    cases=[]
    for row in evaluation:
        ident=row['item_id'];anchor=english[ident]['reply']
        assert local[ident]['standard_anchor']==anchor
        cases.append(dict(item_id=ident,context=row['context'],speakers=row['speakers'],message=row['message'],
            standard_anchor=anchor,baseline_local=local[ident]['output']['reply'],
            diagnostic_anchor=row.get('diagnostic_anchor'),scope='reused_development',
            anchor_provenance='Frozen repair25 English generator; may itself be contextually incorrect.'))
    assert not {r['item_id'] for r in cases} & {r['item_id'] for r in splits['train']}
    write('evaluation.json',cases)
    controls=[
        ('Okay.','I can meet at the library.','Shall we meet there?'),
        ('Purple.','My bottle is purple.','What colour is your bottle?'),
        ('Room 4A.','I am outside room 4A.','Which room?'),
        ('No.','I have not paid the fee yet.','Have you paid?'),
        ('Yes.','I am free on Thursday.','Are you free on Thursday?'),
        ('I cannot attend on Thursday.','I cannot attend on Thursday.','Can you attend?'),
        ('Please send $19 to Mei before 18:25.','Mei needs $19 before 18:25.','What should I send?'),
        ('You should return my cable to me.','You borrowed my cable.','Who should return it?'),
        ('I am waiting outside room 4A.','I am outside room 4A.','Where are you waiting?'),
        ('I have not submitted the form yet.','I have not submitted the form.','Have you submitted it?')]
    write('controls.json',[dict(item_id=f'STYLE-CONTROL-{i:03d}',context=[context],speakers=['user'],
        message=message,standard_anchor=anchor,baseline_local=None,scope='assistant_authored_diagnostic',
        anchor_provenance='Assistant supplied diagnostic English; not a generated English reply or human gold.')
        for i,(anchor,context,message) in enumerate(controls,1)])
    write('data_manifest.json',dict(train=len(splits['train']),validation=len(splits['validation']),
        copy_train=len(read_json(DEST/'copy_train.json')),copy_validation=len(read_json(DEST/'copy_validation.json')),
        main_evaluation=126,controls=len(controls),original_team_sha256=TEAM_SHA,
        source_sha256=digest(source),new_target_replies_created=0,
        auxiliary_definition='Existing LOCAL preferred; identical English draft is style-negative only for long, non-local, non-cosmetic, heuristic-compatible pairs.',
        provenance='Existing approved targets and assistant pair audit; no new human gold or test-set claims.',
        limitations='Surface guard is conservative and incomplete; training-only phrase scores do not establish naturalness.'))


def read_json(path):return json.loads(path.read_text())


if __name__=='__main__':
    prepare()
    print((DEST/'data_manifest.json').read_text())
