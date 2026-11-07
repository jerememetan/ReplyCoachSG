"""Execute frozen matched comparison; one GPU per experiment job."""
import hashlib,json,os,sys
from pathlib import Path
from transformers import AutoTokenizer
from replycoach.generate.finetune_v5 import finetune,item_rows,context_pairs,load_finetuned,validation_outputs
from replycoach.generate.rewrite_v5 import rewrite_rows
from replycoach.generate.style_control import mt5_input
import torch
ROOT=Path.cwd();BASE=os.environ['FOCUS_BASE'];DATA=ROOT/'data/synthetic/standard_focus_v1'
replication_recipe=json.loads(Path('replication_recipe.json').read_text())
for f,digest in replication_recipe['files_sha256'].items():
    assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest,f
recipe=json.loads(Path('recipe.json').read_text())
for f,digest in recipe['files_sha256'].items():
    assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest,f
original=item_rows(ROOT/'data/dev/train.jsonl');validation=item_rows(ROOT/'data/dev/validation.jsonl')
protected={r['message'].lower().strip() for r in original+validation}
adds=item_rows(DATA/'additions.jsonl')
assert not protected & {r['message'].lower().strip() for r in adds}
assert len(context_pairs(DATA/'targeted.jsonl'))==len(context_pairs(DATA/'control.jsonl'))==180
assert len(adds)==160
for name in ('control','targeted'):
    rows=item_rows(DATA/f'{name}.jsonl')
    assert len({r['item_id'] for r in rows})== (190 if name=='control' else 270)
    assert not {r['item_id'] for r in rows}&{r['item_id'] for r in validation}
    tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    for row in rewrite_rows(rows):
        source=mt5_input(row['style'],row['message'],row['context'],True,row['speakers'],standard_reply=row.get('standard_reply'))
        assert len(tokenizer(source)['input_ids'])<=192,(row['item_id'],'source')
        assert len(tokenizer(row['response'])['input_ids'])<=48,(row['item_id'],'target')
print('Source/data hashes, role labels, row counts, original-only context supervision and token budgets verified.',flush=True)
seed=int(sys.argv[1])
assert seed in (1,2)
arms=['control','capacity']
for arm in arms:
    name=f'{arm}_s{seed}';args={}
    if arm=='capacity':args['lora_targets']=('q','k','v','o','wi_0','wi_1','wo')
    if arm=='standard_heavy':args['standard_repeats']=2
    report=finetune(BASE,train_path=DATA/('control.jsonl' if arm=='control' else 'targeted.jsonl'),output_dir=ROOT/'artifacts',output_name=name,
        seed=seed,speaker_labels=True,use_context_pairs=True,rewrite_local=True,epochs=30,patience=100,max_steps=784,
        learning_rate=3e-4,batch_size=16,max_source=192,max_target=48,save_final=True,**args)
    assert report['optimizer_steps']==784
    assert report['train_pairs']=={'control':560,'targeted':720,'capacity':720,'standard_heavy':990}[arm]
    # Frozen fresh diagnostics are generated only after training; neither labels nor references enter prompts.
    model=load_finetuned(ROOT/'artifacts'/name/'final',device='cuda')
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'artifacts'/name/'final',local_files_only=True)
    diag=item_rows(ROOT/'data/reward_diagnostics/standard_focus_v1.jsonl')
    outputs=validation_outputs(model,tokenizer,diag,192,48,'cuda',speaker_labels=True,rewrite_local=True)
    (ROOT/'artifacts'/f'{name}_diagnostic.json').write_text(json.dumps(dict(arm=arm,checkpoint='final step 784',outputs=outputs),ensure_ascii=False,indent=2)+'\n')
    del model,tokenizer
    torch.cuda.empty_cache()
