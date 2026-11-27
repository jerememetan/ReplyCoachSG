"""Native-target rewrite ablations, preserving one fixed STANDARD intention."""
import re
from replycoach.generate.style_control import mt5_input,REWRITE_STYLE

MODES=('full_context','anchor_only','exemplar')

def normalized(text):return ' '.join(re.findall(r"[a-z0-9']+",text.casefold()))

def act(text):
    if '?' in text:return 'question'
    if re.search(r"\b(?:no|not|cannot|can't|don't|won't|sorry)\b",text,re.I):return 'negative'
    return 'other'


class StyleExamples:
    """Retrieval is fit on training STANDARD only; exemplar text is training LOCAL."""
    def __init__(self,train):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.rows=list(train)
        if not self.rows or any(r['split']!='train' for r in self.rows):raise ValueError('Style pool must contain training rows only')
        self.vectorizer=TfidfVectorizer(ngram_range=(1,2),lowercase=True)
        self.matrix=self.vectorizer.fit_transform([r['standard'] for r in train])

    def choose(self,row):
        query=row['standard'];scores=(self.matrix@self.vectorizer.transform([query]).T).toarray().ravel()
        eligible=[]
        for i,r in enumerate(self.rows):
            if r['group_id']==row['group_id'] or normalized(r['standard'])==normalized(query):continue
            # Prefer short, fact-light language examples. No test/validation targets enter retrieval.
            if len(r['local'].split())>18 or re.search(r'\d|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',r['local'],re.I):continue
            eligible.append(i)
        if not eligible:raise ValueError('No group-disjoint training style exemplar available')
        same=[i for i in eligible if act(self.rows[i]['standard'])==act(query)]
        candidates=same or eligible
        chosen=max(candidates,key=lambda i:(float(scores[i]),-i))
        return self.rows[chosen]


def source(row,mode,retriever=None):
    if mode not in MODES:raise ValueError('Unknown LOCAL ablation')
    standard=row['standard']
    if not isinstance(standard,str) or not standard.strip():raise ValueError('Nonempty actual STANDARD anchor required')
    if mode=='full_context':
        return mt5_input(REWRITE_STYLE,row['message'],row['context'],True,row['speakers'],standard_reply=standard),None
    if mode=='anchor_only':return f'{REWRITE_STYLE} Standard draft: {standard} | Singlish rewrite:',None
    if retriever is None:raise ValueError('Training-only style retriever required')
    example=retriever.choose(row)
    return f"{REWRITE_STYLE} Style example (wording only): {example['local']} | Standard draft: {standard} | Singlish rewrite:",example['item_id']


def encode(rows,tokenizer,mode,retriever,max_source=384,max_target=64):
    result=[]
    for row in rows:
        text,_=source(row,mode,retriever);src=tokenizer(text)['input_ids'];target=tokenizer(row['local'])['input_ids']
        if len(src)>max_source or len(target)>max_target:raise ValueError('Native rewrite budget exceeded; no silent truncation')
        result.append((src,target))
    return result


def generate(model,tokenizer,rows,mode,retriever,device,batch_size=4,max_source=384,max_target=64):
    import torch
    from replycoach.generate.train_v5 import collate
    model.eval();outputs=[]
    with torch.no_grad():
        for start in range(0,len(rows),batch_size):
            batch=rows[start:start+batch_size];texts=[source(r,mode,retriever) for r in batch]
            encoded=[]
            for text,_ in texts:
                ids=tokenizer(text)['input_ids']
                if len(ids)>max_source:raise ValueError('Generated-anchor source exceeds budget')
                encoded.append((ids,[tokenizer.eos_token_id]))
            ids,mask,_=collate(encoded,tokenizer.pad_token_id)
            generated=model.generate(input_ids=ids.to(device),attention_mask=mask.to(device),max_new_tokens=max_target,num_beams=4,no_repeat_ngram_size=3)
            for row,text,(_,example) in zip(batch,tokenizer.batch_decode(generated,skip_special_tokens=True),texts):
                outputs.append(dict(item_id=row['item_id'],standard=row['standard'],local=text.strip(),local_conditioned_on=row['standard'],style_example_id=example))
    return outputs
