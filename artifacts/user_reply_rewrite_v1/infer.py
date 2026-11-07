import gc,json,os
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from replycoach.generate.train_v5 import generate_replies
rows=json.loads(Path('inputs.json').read_text());project=Path(os.environ['PROJECT_ROOT'])
models=[('sequential_rewrite_s425',project/'experiments/shared_intent_v1/artifacts/rewrite_s425/final')]
results=[]
for name,checkpoint in models:
    adapter=checkpoint/'adapter_config.json'
    if adapter.exists():
        base=Path(json.loads(adapter.read_text())['base_model_name_or_path'])
        if not base.is_absolute():base=project/base
        model=AutoModelForSeq2SeqLM.from_pretrained(base,local_files_only=True).to('cuda')
        model=PeftModel.from_pretrained(model,checkpoint)
    else:model=AutoModelForSeq2SeqLM.from_pretrained(checkpoint,local_files_only=True).to('cuda')
    model.eval();tok=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
    meta=json.loads((checkpoint/'v5_input.json').read_text()) if (checkpoint/'v5_input.json').exists() else {}
    assert meta.get('rewrite_local') and not meta.get('joint')
    from replycoach.generate.rewrite_v5 import generate_pair
    generated=generate_pair(model,tok,rows,meta.get('max_source',192),meta.get('max_target',48),'cuda',dict(num_beams=4,no_repeat_ngram_size=3),speaker_labels=meta.get('speaker_labels',False))
    results.append(dict(model=name,checkpoint=str(checkpoint),input_format=meta,outputs=[dict(item_id=r['item_id'],standard=generated['STANDARD'][i],local=generated['LOCAL'][i],local_anchor_standard=generated['STANDARD'][i]) for i,r in enumerate(rows)]))
    Path('outputs.json').write_text(json.dumps(dict(status='running',results=results),ensure_ascii=False,indent=2)+'\n')
    del model,tok;gc.collect();torch.cuda.empty_cache()
Path('outputs.json').write_text(json.dumps(dict(status='complete',results=results),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(results,ensure_ascii=False),flush=True)
