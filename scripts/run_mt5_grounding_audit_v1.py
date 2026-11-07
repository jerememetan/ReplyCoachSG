"""Compare historical mT5 checkpoints without weight updates or LOCAL confounds."""
import gc,hashlib,json,time
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from replycoach.generate.act_follow_v1 import ActFollowMT5,conversation_source
from replycoach.generate.v5_provider import DECODING
from run_mt5_act_follow_v1 import render
from run_mt5_reply_act_v1 import dump

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts'
HOME=Path('/common/home/users/h/hm.teoh.2024')
BASE=HOME/'ReplyCoachSG_focus_v1/artifacts/mt5_base_after_spans_spk/best'
STANDARD=HOME/'ReplyCoachSG_standard_val_v1/artifacts/full_lr3e4/best'
PAIRED=HOME/'ReplyCoachSG_mt5_act_follow_v1/artifacts'


def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


@torch.no_grad()
def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'results.json').exists():raise FileExistsError('Fresh report required')
    recipe=json.loads((ROOT/'recipe.json').read_text())
    for p,h in recipe['files_sha256'].items():
        if file_hash(ROOT/p)!=h:raise ValueError('Snapshot mismatch: '+p)
    rows=json.loads((ROOT/'data/evaluation/mt5_grounding_audit_v1/inputs.json').read_text())
    old_report=json.loads((PAIRED/'results.json').read_text());patterns=old_report['supported_act_patterns']
    report=dict(status='running',recipe=recipe,checkpoints={},outputs={})
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    for name,path in [('earlier_dialogue_base',BASE),('selected_standard',STANDARD),('paired_direct',PAIRED/'direct_control/best.pt'),('paired_gated',PAIRED/'gated_paired/best.pt')]:
        started=time.monotonic();torch.manual_seed(425)
        if path.is_dir():
            model=AutoModelForSeq2SeqLM.from_pretrained(path).cuda().eval();tok=AutoTokenizer.from_pretrained(path)
            report['checkpoints'][name]=dict(path=str(path),weights_sha256=file_hash(path/'model.safetensors'),input_format=json.loads((path/'v5_input.json').read_text()))
            outputs=[]
            for row in rows:
                source=conversation_source(row);ids=tok(source,return_tensors='pt')['input_ids'].cuda()
                if ids.shape[-1]>192:raise ValueError('Source budget exceeded')
                tokens=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=48,**DECODING)
                reply=tok.decode(tokens[0],skip_special_tokens=True)
                outputs.append(dict(item_id=row['item_id'],standard=reply,status='complete' if reply.strip() else 'failed_empty'))
            report['outputs'][name]=outputs
        else:
            mode='direct' if name=='paired_direct' else 'gated'
            model=ActFollowMT5(AutoModelForSeq2SeqLM.from_pretrained(STANDARD),mode).cuda().eval()
            model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True));tok=AutoTokenizer.from_pretrained(STANDARD)
            report['checkpoints'][name]=dict(path=str(path),weights_sha256=file_hash(path),initialization_path=str(STANDARD),mode=mode)
            if mode=='direct':report['outputs'][name]=render(model,tok,rows,'predicted',patterns)
            else:
                report['outputs']['paired_gated_predicted']=render(model,tok,rows,'predicted',patterns)
                report['outputs']['paired_gated_disabled']=render(model,tok,rows,'disabled',patterns)
        report['checkpoints'][name]['seconds']=time.monotonic()-started
        dump(OUT/'results.json',report);print('Audited',name,flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    report['status']='complete';dump(OUT/'results.json',report)
    print('Completed inference-only checkpoint audit',flush=True)

if __name__=='__main__':main()
