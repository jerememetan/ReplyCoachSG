"""Assemble audited native targets and group-preserving anchor variants."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];FOLDER=ROOT/'data/public/native_rewrite_v2'

def normalize(text):return ' '.join(re.findall(r"[a-z0-9']+",text.casefold()))

def variants(text):
    forms={"I'm":"I am","I'll":"I will","I've":"I have","can't":"cannot","don't":"do not","won't":"will not","isn't":"is not","hasn't":"has not","haven't":"have not","we're":"we are","We're":"We are","we'll":"we will","We'll":"We will","you're":"you are","You're":"You are","let's":"let us","Let's":"Let us"}
    # Avoid ambiguous 'he's', 'she's', 'they'd', etc. All variants stay in their source group.
    expanded=text
    for old,new in forms.items():expanded=re.sub(r'(?<!\w)'+re.escape(old)+r'(?!\w)',new,expanded)
    return [text] if expanded==text else [text,expanded]

def finalize(folder=FOLDER):
    if (folder/'train.jsonl').exists():raise FileExistsError('Preserve frozen training snapshot')
    original=json.loads((folder/'original_repaired.json').read_text());public=json.loads((folder/'public_repaired.json').read_text())
    allrows=original+public
    for r in allrows:r['standard_variants']=variants(r['standard'])
    train=[r for r in allrows if r['split']=='train'];val=[r for r in allrows if r['split']=='validation']
    assert not {r['group_id'] for r in train}&{r['group_id'] for r in val}
    assert not {normalize(r['local']) for r in train}&{normalize(r['local']) for r in val}
    assert len({r['item_id'] for r in allrows})==len(allrows)
    for name,rows in [('train',train),('validation',val)]:
        (folder/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    manifest=dict(train=len(train),validation=len(val),native_local_total=len(allrows),original_local_kept=len(original),public_local_kept=len(public),local_generated=0,standard_variants='Only unambiguous contraction expansions; sampled per root item, not counted as independent conversations',split='Original authorE held out; CoSEM public validation chats separate from training chats. This is rewriter-component validation, not pristine end-to-end evaluation of fixed epoch1 generator.',originals_intact=True,provenance='Original LOCAL verbatim; all STANDARD normalizations assistant reviewed, not human gold',files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.json*') if p.name!='manifest.json'})
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k!='files_sha256'},indent=2))

if __name__=='__main__':finalize()
