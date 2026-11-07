"""Build expansion inputs under explicit blanket authorization, without inventing item reviews."""
import copy,hashlib,json,re
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/training_expansion/blanket600_v1'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
def msg(x):return x['message']['text'] if isinstance(x['message'],dict) else x['message']
def normalize(x):return re.sub(r'\W+',' ',x.casefold()).strip()
def main():
 if OUT.exists():raise FileExistsError(OUT)
 first_path=ROOT/'data/human_review/expansion_v1/approved_batch01/approved.jsonl'
 first={r['id']:r for r in read(first_path)};assert len(first)==10
 paths=[ROOT/'data/human_review/expansion_v1/fresh_candidates.jsonl',ROOT/'data/synthetic/volume500_v1_reviewed.jsonl']
 candidates=[r for p in paths for r in read(p)];assert len(candidates)==600 and len({r['id'] for r in candidates})==600
 base_path=ROOT/'data/synthetic/standard_focus_v1/control.jsonl';base=read(base_path);assert len(base)==190
 protected_paths=[ROOT/'data/dev/validation.jsonl',ROOT/'data/reward_diagnostics/standard_focus_v1.jsonl',ROOT/'data/reward_diagnostics/duration_probe_v1.jsonl']
 protected={normalize(msg(x)) for p in protected_paths for x in read(p)}|{normalize(msg(x)) for x in base}
 assert not protected&{normalize(msg(x)) for x in candidates}
 assert len({normalize(msg(x)) for x in candidates})==600
 expanded=copy.deepcopy(base);ledger=[]
 for candidate in candidates:
  item=copy.deepcopy(candidate);reviewed=first.get(item['id']);replies=(reviewed or item)['replies']
  assert all(isinstance(replies[k],str) and replies[k].strip() for k in ('standard','local'))
  approval=dict(mode='individual_review' if reviewed else 'blanket_training_authorization',individually_reviewed=bool(reviewed),user_instruction='yes all 10 looks good' if reviewed else 'how about just accept everything, and go train',scope='Training authorized by project owner; no per-item quality judgment inferred for blanket-authorized rows.')
  ledger.append(dict(id=item['id'],approval=approval,source_provenance=item['provenance'],previous_ai_review=item.get('review')))
  expanded.append(dict(id=item['id'],split='dev',session_memory=item['session_memory'],message=dict(text=msg(item)),replies=replies,training_authorization=approval,provenance=item['provenance']))
 assert len(expanded)==790 and len({r['id'] for r in expanded})==790
 OUT.mkdir(parents=True)
 for name,rows in [('control',base),('expanded',expanded)]:
  (OUT/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 (OUT/'approval_ledger.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n')
 (OUT/'manifest.json').write_text(json.dumps(dict(original_conversations=190,added_conversations=600,total_conversations=790,individually_reviewed_additions=10,blanket_authorized_additions=590,user_instruction='how about just accept everything, and go train',source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [base_path,*paths,first_path]},training_files_sha256={p.name:sha(p) for p in OUT.glob('*.jsonl')},scenario_counts=dict(Counter(x.get('scenario_group','') for x in candidates)),dpo_preferences_created=0,locked_test_read=False),indent=2)+'\n')
 print('Prepared190-control versus790-expanded conversations:10 individually reviewed +590 blanket-authorized additions;0 invented DPO preferences.')
if __name__=='__main__':main()
