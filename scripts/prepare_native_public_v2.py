"""Select existing CoSEM text for translation candidates; never regenerate LOCAL."""
import hashlib,json,random,re,os
from pathlib import Path
ROOT=Path.cwd();out=ROOT/'data/public/native_rewrite_v2';out.mkdir(parents=True,exist_ok=False)
source=Path(os.environ['PROJECT_ROOT'])/'data/public'
norm=lambda t:' '.join(re.findall(r"[a-z0-9']+",t.casefold()))
original=[json.loads(l) for l in Path('data/dev/train.jsonl').read_text().splitlines()]
used={norm(r['replies']['local']) for r in original};chatsets={};selected=[]
markers=re.compile(r'\b(?:lah|leh|lor|ah|hor|sia|paiseh|swee|makan|chope|onz|eh|aiyo|alr|liao|cmi|tmr|abit|yall|u|ur)\b',re.I)
private=re.compile(r'https?://|www\.|@|\d{5,}|\b(?:fuck\w*|shit\w*|bitch\w*|knn|ccb|cb|lanjiao)\b',re.I)
for split,name,count,cap in [('train','cosem_train_pairs.jsonl',120,2),('validation','cosem_validation_pairs.jsonl',30,8)]:
    path=source/name;rows=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    random.Random(425).shuffle(rows);rows.sort(key=lambda r:not bool(markers.search(r['response'])))
    chats={};starts={};chosen=[]
    for r in rows:
        text=r['response'];words=text.split();normalized=norm(text);start=' '.join(normalized.split()[:2])
        if not 3<=len(words)<=18 or normalized in used or private.search(text) or '\ufffd' in text or re.search(r'[\u3400-\u9fff]',text):continue
        chat=r['chat']
        if chats.get(chat,0)>=cap or starts.get(start,0)>=3:continue
        used.add(normalized);chats[chat]=chats.get(chat,0)+1;starts[start]=starts.get(start,0)+1
        chosen.append(dict(item_id=f'NATIVE-PUB-{split}-{len(chosen)+1:03}',split=split,context=r['context'],speakers=r.get('speakers'),message=r['message'],local=text,group_id='cosem:'+chat,provenance='Unmodified original CoSEM utterance from existing licensed project data; automatically filtered, translation candidate not human gold'))
        if len(chosen)==count:break
    assert len(chosen)==count,(split,len(chosen));selected+=chosen;chatsets[split]=set(chats)
assert not chatsets['train']&chatsets['validation']
(out/'selected.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n')
(out/'selection_manifest.json').write_text(json.dumps(dict(count=len(selected),train_chats=len(chatsets['train']),validation_chats=len(chatsets['validation']),source_files_sha256={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ('cosem_train_pairs.jsonl','cosem_validation_pairs.jsonl')},sampling='Deterministic seed425; prioritize marked colloquial texts, diverse openings, group caps; not representative corpus sample or automatic naturalness certification'),indent=2)+'\n')
print('Selected150 unmodified native LOCAL utterances',flush=True)
