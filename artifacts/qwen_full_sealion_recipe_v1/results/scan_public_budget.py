import collections,json
from pathlib import Path
from transformers import AutoTokenizer
from replycoach.generate.sealion_data import source
r=json.loads(Path('recipe.json').read_text());tok=AutoTokenizer.from_pretrained(r['models']['qwen']['path'],local_files_only=True)
out={'splits':{}}
for split,entry in r['public']['sources'].items():
 rows=[json.loads(x) for x in Path(entry['path']).read_text().splitlines() if x.strip()];counts=collections.Counter();longest=[]
 for row in rows:
  n=len(tok.encode(source(row,'DIALOGUE'),add_special_tokens=False))+len(tok.encode(row['response'],add_special_tokens=False))+1
  counts[n]+=1
  if n>384:longest.append({'item_id':row['item_id'],'tokens':n})
 out['splits'][split]={'rows':len(rows),'max_tokens':max(counts),'over384':sum(v for k,v in counts.items() if k>384),'over512':sum(v for k,v in counts.items() if k>512),'longest':sorted(longest,key=lambda x:x['tokens'],reverse=True)[:10]}
Path('public_token_budget_scan.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
