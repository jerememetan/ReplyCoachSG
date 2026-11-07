"""CPU preflight for every frozen data row; no truncation or sample removal."""
import argparse,json,hashlib
from collections import Counter
from pathlib import Path
from replycoach.generate.mt5_pretrained_compat import original_tokenizer
from replycoach.generate.mt5_matched_v1 import encode,task_rows,paired_rows,source
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def rows(p):return [json.loads(l) for l in (ROOT/p).read_text().splitlines() if l.strip()]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tokenizer',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 tok=original_tokenizer(a.tokenizer)
 import sentencepiece as spm
 sp=spm.SentencePieceProcessor(model_file=str(Path(a.tokenizer)/'spiece.model'));checked=0
 audit=read('data/training_expansion/sealion_rewrite_v1/pair_audit.json');report={}
 train=rows('data/training_expansion/generated600_repair_v1/combined_train.jsonl');valid=rows('data/dev/validation.jsonl')
 assert len(train)==790 and len(valid)==50
 assert (ROOT/'data/training_expansion/generated600_repair_v1/combined_train.jsonl').read_bytes().startswith((ROOT/'data/dev/train.jsonl').read_bytes())
 assert not {r['id'] for r in train}&{r['id'] for r in valid}
 groups={}
 for name in ['sealion_dialogue_v1','sealion_full_v1']:
  tr=rows('data/public/'+name+'/train.jsonl');va=rows('data/public/'+name+'/validation.jsonl')
  assert not {r['group_id'] for r in tr}&{r['group_id'] for r in va}
  groups[name+'/train']=tr;groups[name+'/validation']=va
 assert len(groups['sealion_full_v1/train'])==117634 and len(groups['sealion_dialogue_v1/train'])==7994
 report['small_full_shared_training_ids']=len({r['item_id'] for r in groups['sealion_dialogue_v1/train']}&{r['item_id'] for r in groups['sealion_full_v1/train']})
 assert (ROOT/'data/public/sealion_dialogue_v1/validation.jsonl').read_bytes()==(ROOT/'data/public/sealion_full_v1/validation.jsonl').read_bytes()
 groups['task/train']=task_rows(train);groups['task/validation']=task_rows(valid)
 for split,records,count in [('train',train,745),('validation',valid,42)]:
  ids={r['id'] for r in audit['items'] if r['split']==split and r['eligible']};assert len(ids)==count;groups['rewrite/'+split]=paired_rows(records,ids)
 for split,count in [('train',86),('validation',18),('challenge',18)]:
  groups['fidelity/'+split]=read('data/training_expansion/sealion_fidelity_v1/'+split+'.json');assert len(groups['fidelity/'+split])==count
 for name,data in groups.items():
  maximum=[0,0]
  for r in data:
   x,y=encode(r,tok,'rewrite' if 'standard_anchor' in r else 'direct',max_source=1536,max_target=384)
   text=source(r,'LOCAL_REWRITE',r['standard_anchor']) if 'standard_anchor' in r else source(r,r.get('style','DIALOGUE'))
   if x!=sp.encode(text)+[sp.eos_id()] or y!=sp.encode(r['response'])+[sp.eos_id()]:raise ValueError('Original SentencePiece IDs differ: '+r['item_id'])
   checked+=2
   maximum=[max(maximum[0],len(x)),max(maximum[1],len(y))]
  report[name]=dict(count=len(data),max_source_tokens=maximum[0],max_target_tokens=maximum[1],corpora=dict(Counter(r.get('corpus','task') for r in data)))
  print(name,report[name],flush=True)
 ev=read('data/evaluation/sealion_rewrite_v1/inputs.json');assert len(ev)==82
 report['evaluation']=dict(count=82,diagnostic_anchors=sum('diagnostic_anchor' in r for r in ev),max_source_tokens=max(len(tok(source(r,s,anchor))['input_ids']) for r in ev for s,anchor in [('STANDARD',None),('LOCAL',None),('LOCAL_REWRITE',r['actual_anchor'])]))
 report['protected_team_sha256']=hashlib.sha256((ROOT/'data/dev/train.jsonl').read_bytes()).hexdigest();report['status']='pass';report['source_limit']=1536;report['target_limit']=384;report['original_sentencepiece_id_checks']=checked;report['exclusions']=0;report['truncations']=0
 Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
