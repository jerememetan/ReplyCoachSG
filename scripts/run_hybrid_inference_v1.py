"""Evaluate existing two-checkpoint inference composition; no training."""
import hashlib,json,os,tarfile
from pathlib import Path
from replycoach.generate.hybrid_v5 import HybridV5Generator
recipe=json.loads(Path('recipe.json').read_text())
for f,digest in recipe['files_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest,f
base=Path(os.environ['PROJECT_ROOT'])/'artifacts/mt5_base_after_spans_spk/best'
h=hashlib.sha256()
with (base/'model.safetensors').open('rb') as f:
    for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
assert h.hexdigest()==recipe['rewrite_base_sha256'],'Portable adapter must use identical base weights'
portable=Path('checkpoints/local_rewrite')
if portable.exists():raise FileExistsError('Fresh inference snapshot required')
portable.mkdir(parents=True)
with tarfile.open('portable_rewriter.tar.gz','r:gz') as t:t.extractall(portable,filter='data')
config_path=portable/'adapter_config.json';original=config_path.read_text();config=json.loads(original)
config['base_model_name_or_path']=str(base)
config_path.write_text(json.dumps(config,indent=2)+'\n')
Path('artifacts').mkdir(exist_ok=True)
Path('artifacts/adapter_relocation.json').write_text(json.dumps(dict(original_adapter_config=json.loads(original),relocated_adapter_config=config,base_model_sha256=h.hexdigest()),indent=2)+'\n')
generator=HybridV5Generator(Path(os.environ['ACTOR_BASE']),portable.resolve(),device='cuda')
rows=json.loads(Path('data/hybrid_inference_v1/inputs.json').read_text());assert len(rows)==53
controls=json.loads(Path('data/hybrid_inference_v1/controls.json').read_text())
report=dict(status='running',standard_checkpoint=str(generator.standard.checkpoint),local_checkpoint=str(generator.local.checkpoint),rows=rows,outputs=[],limitation='Inference-only composition of existing checkpoints.50 reused development cases plus3 user examples; no independent test, training or automatic promotion.')
output=Path('artifacts/hybrid_report.json')
def save():output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
try:
    for start in range(0,len(rows),4):
        paired=generator.generate_rows(rows[start:start+4])
        for pair in paired:
            assert pair['local_conditioned_on']==pair['standard']
            pair['same_standard_as_standalone']=pair['standard']==controls['epoch1_standard'][pair['item_id']]
        report['outputs']+=paired;save();print('Paired',len(report['outputs']),'/53',flush=True)
    report['app_candidate_outputs']=[]
    for row in rows[-3:]:
        candidates=generator.generate(row['message'],row['context'])
        assert candidates[1].meta['local_conditioned_on']==candidates[0].candidate_text
        report['app_candidate_outputs'].append(dict(item_id=row['item_id'],candidates=[c.to_dict() for c in candidates]))
    report['same_standard_as_standalone']=sum(r['same_standard_as_standalone'] for r in report['outputs'])
    report['status']='complete';save()
except Exception:
    report['status']='failed';save();raise
