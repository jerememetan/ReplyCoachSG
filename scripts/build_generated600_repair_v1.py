"""Build auditable corrected generated data while preserving original team bytes."""
import copy,csv,hashlib,json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'data/training_expansion/generated600_repair_v1'

def sha(data):return hashlib.sha256(data).hexdigest()
def jsonl(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
def signature(r):return json.dumps([r['session_memory'],r['message']['text']],sort_keys=True,ensure_ascii=False).casefold()
def apply_patch(row,patch):
    result=copy.deepcopy(row)
    allowed={'id','reason','standard','local','context','context_speakers','message_text'}
    if set(patch)-allowed or row['id']!=patch.get('id'):raise ValueError('Invalid patch identity/fields')
    if not row['id'].startswith('SYN-'):raise ValueError('Team-authored rows cannot be patched')
    if 'context' in patch and 'context_speakers' in patch:raise ValueError('Conflicting context changes')
    if 'context' in patch:result['session_memory']=copy.deepcopy(patch['context'])
    if 'context_speakers' in patch:
        if len(result['session_memory'])!=len(patch['context_speakers']):raise ValueError('Role count mismatch')
        for turn,speaker in zip(result['session_memory'],patch['context_speakers'],strict=True):turn['speaker']=speaker
    if 'message_text' in patch:result['message']['text']=patch['message_text']
    for style in ['standard','local']:
        if style in patch:result['replies'][style]=patch[style]
    if any(t['speaker'] not in ['peer','user','other'] or not t['text'].strip() for t in result['session_memory']):raise ValueError('Invalid context')
    if not result['message']['text'].strip() or any(not result['replies'][s].strip() for s in ['standard','local']):raise ValueError('Empty message/target')
    if not patch.get('reason','').strip():raise ValueError('Editorial reason required')
    return result

def build():
    review=json.loads((FOLDER/'review_record.json').read_text());patches=json.loads((FOLDER/'patches.json').read_text())
    original_path=ROOT/review['original_team_source'];source_path=ROOT/review['expanded_source'];original_raw=original_path.read_bytes();source_raw=source_path.read_bytes()
    if sha(original_raw)!=review['team_sha256'] or sha(source_raw)!=review['expanded_sha256']:raise ValueError('Frozen source changed')
    original=jsonl(original_path);source=jsonl(source_path);validation=jsonl(ROOT/'data/dev/validation.jsonl');team_ids={r['id'] for r in original}
    generated=[r for r in source if r['id'] not in team_ids]
    if len(original)!=190 or len(generated)!=600 or len({r['id'] for r in generated})!=600:raise ValueError('Incorrect source counts')
    if review['reviewed_ids']!=[r['id'] for r in generated]:raise ValueError('Incomplete/mismatched per-example review')
    if any(not k.isdigit() or not 1<=int(k)<=600 for k in patches):raise ValueError('Invalid patch index')
    corrected=[];ledger=[];counts=Counter()
    for n,row in enumerate(generated,1):
        patch=patches.get(str(n));updated=apply_patch(row,patch) if patch else copy.deepcopy(row)
        changes={field:{'before':row[field],'after':updated[field]} for field in ['session_memory','message','replies'] if row[field]!=updated[field]}
        if patch and not changes:raise ValueError('No-op patch: '+row['id'])
        counts['revised' if changes else 'unchanged']+=1
        counts['context_or_message_repaired']+=any(k in changes for k in ['session_memory','message'])
        for s in ['standard','local']:counts[s+'_revised']+=row['replies'][s]!=updated['replies'][s]
        reason=patch['reason'] if patch else 'Inspected full context, role direction, constraints, paired meaning and wording; retained existing targets.'
        audit=dict(id=row['id'],reviewer='Codex assistant',status='assistant_corrected' if changes else 'assistant_reviewed_unchanged',reason=reason,changes=changes,human_per_item_rating=False,original_row_sha256=sha(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()))
        updated['assistant_revision']=dict(version='generated600_repair_v1',reviewer='Codex assistant',human_per_item_rating=False,status=audit['status'],source_row_sha256=audit['original_row_sha256'])
        corrected.append(updated);ledger.append(audit)
    combined=[*original,*corrected]
    if len({r['id'] for r in combined})!=790 or {r['id'] for r in combined}&{r['id'] for r in validation}:raise ValueError('ID duplication/validation overlap')
    if {signature(r) for r in combined}&{signature(r) for r in validation}:raise ValueError('Exact validation input overlap')
    def serialize(rows):return ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode()
    generated_raw=serialize(corrected)
    if not original_raw.endswith(b'\n'):raise ValueError('Team source must end in newline for byte-preserving concat')
    combined_raw=original_raw+generated_raw
    assert combined_raw[:len(original_raw)]==original_raw
    (FOLDER/'generated_corrected.jsonl').write_bytes(generated_raw);(FOLDER/'combined_train.jsonl').write_bytes(combined_raw)
    (FOLDER/'audit.jsonl').write_bytes(serialize(ledger))
    with (FOLDER/'changes.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['id','status','reason','original_conversation','corrected_conversation','original_incoming','corrected_incoming','original_standard','corrected_standard','original_local','corrected_local'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for before,after,audit in zip(generated,corrected,ledger,strict=True):
            if not audit['changes']:continue
            fmt=lambda r:'\n'.join(t['speaker']+': '+t['text'] for t in r['session_memory'])
            w.writerow(dict(id=before['id'],status=audit['status'],reason=audit['reason'],original_conversation=fmt(before),corrected_conversation=fmt(after),original_incoming=before['message']['text'],corrected_incoming=after['message']['text'],original_standard=before['replies']['standard'],corrected_standard=after['replies']['standard'],original_local=before['replies']['local'],corrected_local=after['replies']['local']))
    files=['generated_corrected.jsonl','combined_train.jsonl','audit.jsonl','changes.csv','patches.json','review_record.json']
    manifest=dict(version='generated600_repair_v1',generated_reviewed=600,team_records_untouched=190,combined_conversations=790,paired_targets=1580,counts=dict(counts),team_original_bytes_preserved=True,validation_untouched=True,locked_test_read=False,training_started=False,default_changed=False,review_type=review['review_type'],limitations=review['limitations'],sources=dict(team_sha256=sha(original_raw),expanded_sha256=sha(source_raw),validation_sha256=sha((ROOT/'data/dev/validation.jsonl').read_bytes())),builder_sha256=sha(Path(__file__).read_bytes()),files_sha256={p:sha((FOLDER/p).read_bytes()) for p in files})
    (FOLDER/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    assert original_path.read_bytes()==original_raw and source_path.read_bytes()==source_raw
    print(json.dumps(dict(counts=counts,team_untouched=190,total=790),indent=2))
    return manifest
if __name__=='__main__':build()
