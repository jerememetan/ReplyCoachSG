"""Development-only matched comparison; assistant issue labels, never human ratings."""
import csv,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ISSUES={1:'Off-topic meeting suggestion in response to relief after submitting report.',2:'Changes supportive eye-appointment conversation into own scheduling conflict.',3:'Asked which bread; answers fried eggs without addressing bread choice.',5:'Swaps borrower/lender role: peer offers to return user notes; user offers to return them.',6:'Claims completed proofreading without prior evidence of completion.',7:'Contradicts stated inability to play by saying arrival in minutes.',9:'Asks for notification of overthinking when peer has just reported it.',12:'Should identify blue sign; instead requests to see it.',13:'Incoherent location: behind your stand for leaving that same stand.',14:'Treats distress about unequal workload as a good idea.',16:'Accepts tonight despite explicitly being busy tonight.',17:'Requests finding announcement earlier, an impossible backward-time request.',18:'Responds to gratitude by saying glad peer did not notice; mismatched meaning.'}
def main():
 anchors=json.loads((ROOT/'data/public/native_rewrite_v2/fixed_anchors.json').read_text())['items'];anchors={r['item_id']:r for r in anchors if r['split']=='development'}
 folder=ROOT/'experiments/native_rewrite_v2_results';systems={}
 for mode in ('full_context','anchor_only','exemplar'):
  systems[mode]=[r for r in json.loads((folder/f'{mode}.json').read_text())['fresh_outputs'] if r['item_id'] in anchors]
 for c in json.loads((folder/'incumbent_development_controls.json').read_text()):systems[f"incumbent_{c['checkpoint']}_beam{c['num_beams']}"]=c['outputs']
 out=ROOT/'experiments/rewrite_diagnostic_v3';out.mkdir(exist_ok=True)
 with (out/'matched_development.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['item_id','context','incoming','standard','assistant_standard_issue']+list(systems));writer.writeheader()
  for key,r in anchors.items():
   writer.writerow(dict(item_id=key,context=json.dumps(r['context']),incoming=r['message'],standard=r['standard'],assistant_standard_issue=ISSUES.get(int(key.rsplit('-',1)[1]),''),**{name:next(x['local'] for x in rows if x['item_id']==key) for name,rows in systems.items()}))
 norm=lambda s:re.sub(r'[^\w]+',' ',s.lower()).strip()
 summary={name:dict(n=len(rows),can_openings=sum(bool(re.match(r'^can\b',r['local'],re.I)) for r in rows),normalized_copies=sum(norm(r['standard'])==norm(r['local']) for r in rows)) for name,rows in systems.items()}
 (out/'audit.json').write_text(json.dumps(dict(scope='Same 20 development contexts and fixed STANDARD anchors across all systems',assistant_flagged_standard_cases=ISSUES,limitations=['Assistant issue flags, not human preference labels or calibrated quality ratings.','Unflagged cases are not certified passes.','Can openings include grammatical uses; not a naturalness metric.','Do not compare rates with previous different 30-case review.','Development cases are not untouched test evidence.'],surface_diagnostics=summary),indent=2)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
