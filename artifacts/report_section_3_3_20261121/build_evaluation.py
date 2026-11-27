"""Reproduce the report tables from explicit assistant judgments, not automated scores."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
packet = json.loads((OUT / 'matched50_packet.json').read_text())

# Each code is English adequacy / LOCAL adequacy / meaning / LOCAL style.
# p=pass, f=fail, u=uncertain. These are retrospective, non-blind assistant labels.
# They are deliberately kept separate from the historical shared-intent study.
labels = {
'A-010': ('ffpf', 'pppp', 'ffpf', 'mT5 responds to an unspecified object; SEA acknowledges the difficult lesson; Qwen echoes the incoming speaker-labelled message.'),
'A-017': ('pppf', 'pppp', 'pppp', 'All confirm seven; mT5 remains Standard English. Qwen adds a plausible meeting location.'),
'A-024': ('ffpp', 'pppp', 'pppp', 'mT5 asks the explicitly uninformed peer for printer instructions. The larger models offer to print; no explicit inability to print is established for the user.'),
'A-030': ('pppp', 'pppp', 'pppp', 'All propose another booking and retain the decision.'),
'A-033': ('pppp', 'pppp', 'pppp', 'All answer the opening-hours question. mT5 supplies a selectable factual draft; unknown personal facts alone are not an error under the project policy.'),
'A-035': ('pppf', 'pppp', 'pppf', 'All acknowledge the reminder. SEA uses dabao; the other LOCAL replies remain standard.'),
'A-038': ('pppp', 'ppup', 'pppf', 'mT5 gives a plausible own-arrival update. SEA LOCAL says ten more minutes rather than explicitly promising to wait, so equivalence is uncertain. Qwen preserves waiting but remains standard.'),
'A-044': ('ffpp', 'pppp', 'ffpf', 'mT5 apologizes for an unspecified missed event. SEA uses blur in a local sense; Qwen uses transitive blur it, which does not sensibly answer the caution.'),
'A-046': ('pppp', 'pppp', 'pppp', 'All can function as acceptance of the proposed meeting; mT5 is vague about time but does not explicitly contradict two.'),
'A-054': ('pppp', 'pppp', 'pppf', 'mT5 can plausibly defer going home while accepting going now. SEA and Qwen explicitly agree to go now; Qwen LOCAL stays standard.'),
'B-002': ('pppp', 'pppp', 'pppp', 'All provide a plausible preference or next step. Qwen changes a suggestion to a confirmation question without changing the proposed venue.'),
'B-009': ('pppp', 'pppp', 'pppp', 'mT5 answers yes and gives a possible waiting plan; no explicit current location forbids it. SEA answers space directly; Qwen offers to check.'),
'B-029': ('ffuf', 'pppp', 'pppp', 'mT5 gives an awkward floor answer and malformed LOCAL; their equivalence is uncertain. Both larger models sensibly check the timetable.'),
'B-034': ('ffpp', 'ffpp', 'pppp', 'mT5 substitutes a free-time request for a class update. SEA reverses who should provide the update. Qwen offers the requested summary.'),
'B-042': ('ffpp', 'ppfp', 'pppf', 'mT5 gives an unrelated apology. SEA changes asking the peer to report arrival into its own arrival update. Qwen preserves its own previously stated arrival.'),
'B-043': ('uupp', 'pppf', 'pppp', 'mT5 asks the organizer what time they want to come, leaving direction ambiguous. SEA accepts in generic texting; Qwen makes a permissible refusal.'),
'B-048': ('pppp', 'pppf', 'pppf', 'All acknowledge the postponement. Arriving earlier is not explicitly prohibited. SEA and Qwen LOCAL remain standard.'),
'B-050': ('pppp', 'pppp', 'pppp', 'All accept a run; shared activity remains implied in the short LOCAL replies.'),
'B-056': ('pppp', 'pppp', 'pufp', 'mT5 acknowledges availability and SEA confirms seven. Qwen English can be a valid refusal, but LOCAL adds a booking-as-obstacle interpretation and an extra reason.'),
'B-057': ('fffp', 'pppp', 'pppp', 'mT5 reverses lending into borrowing and then changes ownership. Both larger models correctly lend the user-owned charger.'),
'C-001': ('pppp', 'pppp', 'pppp', 'All accept the LKS meeting. mT5 merges the place and time typographically, but the intended meeting remains recoverable.'),
'C-017': ('puff', 'pppp', 'pppf', 'mT5 English can supply a newly completed status, but LOCAL introduces a relative future completion plan and sounds repetitive. Both larger models retain unfinished status.'),
'C-018': ('uupp', 'pppp', 'pppp', 'mT5 repeats the merge plan without clearly addressing its own slide reminder. SEA offers to add it; Qwen offers to upload its section after dinner, which is not prohibited for the user.'),
'C-023': ('pppp', 'pppf', 'pppp', 'All give plausible invitation responses. Uncertainty about attending is allowed despite previously having no plans. SEA LOCAL is neutral English.'),
'C-028': ('ffpp', 'ffpf', 'uupf', 'mT5 offers to take the peer to the cafe they already occupy. SEA waits for a peer who is already there. Qwen offers more food after the peer ordered, which is contextually uncertain.'),
'C-038': ('pppp', 'pppp', 'pppp', 'All offer to check the passage, preserving the request.'),
'C-041': ('pppf', 'pppf', 'pppf', 'All acknowledge a delayed upload. Generic nvm or omitted punctuation alone is insufficient evidence of recognizable Singlish.'),
'C-045': ('ufpf', 'pppp', 'pppp', 'mT5 does not clearly confirm arriving early and repeats for the seats in LOCAL. Both larger models confirm the plan.'),
'C-055': ('pppp', 'pppp', 'pppp', 'All accept the two oclock rescheduling.'),
'C-058': ('uupp', 'pppp', 'pppf', 'mT5 waits for the peer rather than clearly for court availability, leaving relevance uncertain. SEA and Qwen wait; Qwen wording is standard.'),
'D-007': ('pppp', 'pppf', 'uupf', 'mT5 wine is a permissible selectable preference despite being unusual. SEA defers choosing in neutral English. Qwen green tea latte may repeat the declined matcha choice, so adequacy is uncertain.'),
'D-009': ('pppp', 'pppp', 'pppp', 'mT5 offers to do the identified question later; the larger models offer help. These are plausible next replies.'),
'D-010': ('pppf', 'pppp', 'pppp', 'All respond to knowing the question or offer to check notes; mT5 LOCAL stays standard.'),
'D-013': ('ffpp', 'pppp', 'pppp', 'mT5 reverses who supplies material. Both larger models offer help and ask for the specific difficulty.'),
'D-029': ('ffpp', 'pppp', 'pppp', 'mT5 gives an unrelated see-you-later response. Both larger models acknowledge the bidding reminder.'),
'D-033': ('pppp', 'ffpp', 'pppp', 'mT5 and Qwen give location/arrival information. SEA asks what the peer is doing instead of answering where it is.'),
'D-037': ('fffp', 'pppp', 'pppf', 'mT5 offers to take the already-arrived peer there, then changes the action again in LOCAL. Both larger models give plausible own-arrival updates.'),
'D-038': ('pppp', 'pppp', 'pffp', 'mT5 and SEA preserve waiting for the peer. Qwen LOCAL reverses the instruction into wait for me.'),
'D-046': ('ffpp', 'pppf', 'pppf', 'mT5 promises the sold-out ticket without acknowledging unavailability. SEA asks about an alternative and Qwen sympathizes; both LOCAL replies remain neutral English.'),
'D-050': ('pppp', 'pppf', 'pppf', 'All appropriately respond to illness. SEA and Qwen LOCAL remain standard.'),
'E-005': ('uupp', 'pppp', 'pppf', 'mT5 changes which person joins the other, though both are heading to the same cafe, so adequacy is uncertain. SEA and Qwen acknowledge the peer joining.'),
'E-007': ('ffpp', 'ffpf', 'ffpf', 'All miss the where-to-meet question in na li. Qwen additionally interprets it as the location name Na Li.'),
'E-009': ('ffpp', 'pufp', 'pppp', 'mT5 misses the charger confirmation. SEA LOCAL can sound like asking for confirmation rather than giving it; English supplies availability. Qwen preserves ownership and after-three timing.'),
'E-015': ('pppf', 'ppfp', 'pppf', 'All give possible quiz status. SEA changes not completed into not started, a stronger claim absent from its English; Qwen copies neutral English.'),
'E-025': ('pppp', 'ffpp', 'pppp', 'mT5 and Qwen give their own Thursday availability. SEA tells the apparent lecturer to check with the lecturer rather than answering the availability request.'),
'E-027': ('pppp', 'uupf', 'pppp', 'mT5 and Qwen permit a change. SEA promises an availability update without clearly answering whether the booking can change; adequacy is uncertain.'),
'E-028': ('ffpp', 'ffpf', 'pppf', 'mT5 accepts after an explicit cannot-make-it refusal. SEA treats busyness as illness. Qwen correctly accepts the refusal in neutral English.'),
'E-032': ('pppp', 'pppf', 'pppf', 'mT5 can acknowledge refusal then ask when class starts; a new question is allowed. SEA accepts refusal; Qwen retains the original venue. Latter two remain neutral English.'),
'E-040': ('pupf', 'uupp', 'pppp', 'mT5 English offers a meeting but LOCAL repeats library. SEA substitutes lunch without clearly linking it to work. Qwen makes a permissible refusal with an offer to help.'),
'E-046': ('pppp', 'pppp', 'pffp', 'mT5 and SEA approve five. Qwen English permits the peer to release it, but LOCAL makes the user the releaser.'),
}
assert len(labels) == len(packet) == 50
axes = ['english_adequacy', 'local_adequacy', 'meaning_preservation', 'local_style']
lookup = dict(p='pass', f='fail', u='uncertain')
regional_content = {'A-010':'kiasu', 'A-030':'sian', 'A-035':'dabao', 'A-044':'blur like sotong', 'A-046':'chiong', 'C-023':'jio', 'E-007':'onz; na li', 'E-028':'shag; cmi; dui bu qi'}
pragmatic = {'A-017','A-024','A-038','A-054','C-018','C-028','C-041','D-009','E-005','E-009','E-025','E-027','E-032','E-040','E-046'}
syntax = {'A-033','B-002','B-009','B-034','B-043','B-048','B-050','C-017'}
reviews = []
audit = []
original = {r['id']:r for r in (json.loads(l) for l in (ROOT/'data/dev/validation.jsonl').read_text().splitlines() if l.strip())}
for row in packet:
    suffix = row['item_id'].removeprefix('ITEM-')
    codes = labels[suffix]
    review = {'item_id':row['item_id'], 'rationale':codes[3], 'models':{}}
    for name, code in zip(['mt5','sealion','qwen'], codes[:3]):
        assert len(code)==4 and set(code)<=set(lookup)
        review['models'][name] = {k:lookup[v] for k,v in zip(axes,code)}
        review['models'][name]['all_four'] = all(v=='p' for v in code)
    reviews.append(review)
    item = original[row['item_id']]
    category = 'regional_lexical_content' if suffix in regional_content else 'regional_pragmatic_marker' if suffix in pragmatic else 'colloquial_syntax' if suffix in syntax else 'general_english_or_texting'
    audit.append({'item_id':row['item_id'], 'message':row['message'], 'category':category, 'regional_content_expression':regional_content.get(suffix,''), 'annotated_local_terms':item['message'].get('local_terms',[]), 'annotated_language_mix':item['message'].get('language_mix',[]), 'campus_abbreviation':suffix=='C-001'})

summary={'scope':'Retrospective non-blind assistant review on 50 reused development conversations; descriptive pass counts, not human accuracy. This compares saved system recipes, not isolated model architectures. Uncertain is not a pass. No new training or inference.', 'protocol_sha256':hashlib.sha256((OUT/'review_protocol.json').read_bytes()).hexdigest(), 'packet_sha256':hashlib.sha256((OUT/'matched50_packet.json').read_bytes()).hexdigest(), 'models':{}, 'language_metadata':{}, 'manual_incoming_language_categories':dict(Counter(r['category'] for r in audit)), 'manual_language_limitations':'Categories are assistant judgments about incoming messages, not a validated regional-language benchmark or a causal measure of model understanding. History and outputs can contain additional regional features. Pragmatic markers and colloquial syntax still require local competence.'}
for name in ['mt5','sealion','qwen']:
    summary['models'][name] = {k:dict(Counter(r['models'][name][k] for r in reviews)) for k in axes}
    summary['models'][name]['all_four_pass'] = sum(r['models'][name]['all_four'] for r in reviews)
    summary['models'][name]['both_styles_adequate'] = sum(r['models'][name]['english_adequacy']=='pass' and r['models'][name]['local_adequacy']=='pass' for r in reviews)
    for category, ids in [('regional_lexical_content', set(regional_content)),('other_incoming',set(labels)-set(regional_content))]:
        group=[r for r in reviews if r['item_id'].removeprefix('ITEM-') in ids]
        summary['models'][name][category]={'n':len(group),'english_adequacy_pass':sum(r['models'][name]['english_adequacy']=='pass' for r in group),'both_styles_adequate':sum(r['models'][name]['english_adequacy']=='pass' and r['models'][name]['local_adequacy']=='pass' for r in group)}
for split in ['train','validation']:
    rows=[json.loads(l) for l in (ROOT/f'data/dev/{split}.jsonl').read_text().splitlines() if l.strip()]
    summary['language_metadata'][split]={'n':len(rows),'local_terms_nonempty':sum(bool(r['message'].get('local_terms')) for r in rows),'singlish_tag':sum('singlish' in r['message'].get('language_mix',[]) for r in rows),'pinyin_tag':sum('pinyin' in r['message'].get('language_mix',[]) for r in rows),'zh_tag':sum('zh' in r['message'].get('language_mix',[]) for r in rows),'pinyin_or_zh_tag':sum(bool(set(r['message'].get('language_mix',[]))&{'pinyin','zh'}) for r in rows),'english_only_tag':sum(r['message'].get('language_mix')==['en'] for r in rows),'history_turn_counts':dict(Counter(len(r['session_memory']) for r in rows))}
for name,data in [('matched50_ratings.json',reviews),('incoming_language_audit50.json',audit),('evaluation_summary.json',summary)]:
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
with (OUT/'matched50_review.csv').open('w',newline='') as f:
    fields=['item_id','context','message','language_category','model','english','local',*axes,'all_four','rationale']
    writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
    for row,review,language in zip(packet,reviews,audit):
        for model,text in row['models'].items():
            writer.writerow({'item_id':row['item_id'],'context':json.dumps(list(zip(row['speakers'],row['context'])),ensure_ascii=False),'message':row['message'],'language_category':language['category'],'model':model,**text,**review['models'][model],'rationale':review['rationale']})
print(json.dumps(summary,ensure_ascii=False,indent=2))
