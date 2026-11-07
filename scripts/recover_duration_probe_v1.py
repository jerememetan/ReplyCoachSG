"""Audit final/step784 tensors, then regenerate only the missing inference artifact."""
import gc,hashlib,json,os
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from replycoach.generate.finetune_v5 import item_rows,load_finetuned,validation_outputs
from replycoach.generate.train_v5 import encode_rows,validation_loss
ROOT=Path.cwd();arm='control';run=ROOT/'artifacts/control_s425'
recipe=json.loads(Path('recipe.json').read_text())
for f,digest in recipe['files_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest,f
left=load_file(str(run/'step_784/adapter_model.safetensors'));right=load_file(str(run/'final/adapter_model.safetensors'))
assert set(left)==set(right)
unequal=[k for k in left if not torch.equal(left[k],right[k])]
metadata={name:json.loads((run/'step_784'/name).read_text())==json.loads((run/'final'/name).read_text()) for name in ['v5_input.json','adapter_config.json','tokenizer_config.json']}
audit=dict(optimizer_steps=json.loads((ROOT/'artifacts/control_s425.json').read_text())['optimizer_steps'],step_and_final_tensors_equal=not unequal,unequal_tensors=unequal,metadata_equal=metadata,reason='Original job completed784 training updates but rejected nonidentical generated replies on reload. This recovery does inference only from the predeclared step784 weights; no labels/settings/thresholds changed.')
(ROOT/'artifacts/control_reload_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
assert not unequal and all(metadata.values()),'Checkpoint contents differ; do not proceed with inference recovery'
del left,right;gc.collect();torch.cuda.empty_cache()
path=run/'step_784';model=load_finetuned(path,device='cuda').eval();tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
fresh=item_rows(ROOT/'data/reward_diagnostics/duration_probe_v1.jsonl');old=item_rows(ROOT/'data/dev/validation.jsonl')
loss=validation_loss(model,encode_rows([r for r in old if r['style']=='STANDARD'],tokenizer,192,48,speaker_labels=True),32,tokenizer.pad_token_id,'cuda')
outputs=validation_outputs(model,tokenizer,fresh,192,48,'cuda',speaker_labels=True,rewrite_local=True)
secondary=validation_outputs(model,tokenizer,old,192,48,'cuda',speaker_labels=True,rewrite_local=True)
prior={r['item_id']:r for r in json.loads((ROOT/'artifacts/control_s425.json').read_text())['final_validation_outputs']}
differences=[dict(item_id=r['item_id'],original={k:prior[r['item_id']][k] for k in ['generated_standard','generated_local']},reloaded={k:r[k] for k in ['generated_standard','generated_local']}) for r in secondary if any(r[k]!=prior[r['item_id']][k] for k in ['generated_standard','generated_local'])]
audit.update(reloaded_generation_differences= differences,identical_secondary_pairs=50-len(differences),checkpoint_weights_verified=True)
(ROOT/'artifacts/control_reload_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
out=ROOT/'artifacts/control_step784.json';assert not out.exists()
out.write_text(json.dumps(dict(arm=arm,seed=425,optimizer_step=784,checkpoint=str(path),original_reference_standard_loss=loss,fresh_source_sha256=recipe['files_sha256']['data/reward_diagnostics/duration_probe_v1.jsonl'],outputs=outputs,secondary_validation_outputs=secondary,recovery='Inference only; exact step784/final tensor identity verified; generated text differences retained in audit.',limitation='Assistant-authored development probe, not human gold; generated reply differences on identical weights limit bitwise reproducibility.'),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(audit,indent=2),flush=True)
