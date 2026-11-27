"""Apply the section reviewed in chat; retain the original DOCX package and peers' work."""
from pathlib import Path
import ast
import hashlib
import json
import os
import shutil
import sys
import zipfile

sys.path.insert(0,'/tmp/replycoach_report_20261121_lib')
from lxml import etree as E

OUT=Path(__file__).resolve().parent
SOURCE=Path('/home/spikeypear/Downloads/Copy of Final Report.docx')
BACKUP=SOURCE.with_name('Copy of Final Report - before approved revision.docx')
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS={'w':W}

# Reuse only the pure formatting helpers, without executing the earlier editor.
tree=ast.parse((OUT/'shorten_section33.py').read_text())
helpers=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in {'w','paragraph','make_table'}]
exec(compile(ast.Module(body=helpers,type_ignores=[]),'approved_format_helpers','exec'))

def text(e):return ''.join(e.xpath('.//w:t/text()',namespaces=NS))
def canon(e):return E.tostring(e,method='c14n')
def digest(b):return hashlib.sha256(b).hexdigest()

blocks=[]
def p(s):blocks.append({'type':'paragraph','text':s})
def h(s):blocks.append({'type':'heading','text':s})
def label(s):blocks.append({'type':'label','text':s})
def note(s):blocks.append({'type':'note','text':s})
def figure(s):blocks.append({'type':'figure_placeholder','text':s})
def example(s):blocks.append({'type':'example','text':s})
def table(caption,headers,rows):blocks.append({'type':'table','caption':caption,'headers':headers,'rows':rows})

p('Although Luong Attention improved the RNN’s reply generation, contextual errors remained. We therefore explored pretrained Transformer models, beginning with mT5-base (Vaswani et al., 2017; Xue et al., 2021). Our aim was to generate a relevant Standard English reply and a natural Singlish counterpart that expressed the same meaning.')
h('3.3.1 Initial Training Plan')
p('We initially planned to adapt mT5-base through two main stages: language and conversation adaptation, followed by project-specific reply training. In the first stage, the full model learned from public data to become more familiar with informal language and conversational replies. This included span-corruption training, followed by next-reply training. The span phase was recorded separately as Stage 0 in the implementation.')
p('In the second stage, we adapted the model using our 190 training conversations, each containing Standard English and Singlish reference replies. We used LoRA to update a small proportion of the model’s parameters (Hu et al., 2021). One shared adapter learned both styles, with the input prefix determining which style to generate.')
table('Table 3.3a. Datasets and settings for the initial mT5 training phases.',
      ['Training phase','Dataset used','Main settings'],[
      ['Stage 1a: Language adaptation','CoSEM and NUS SMS Corpus: 96,295 text chunks for span-corruption training.','2 epochs; learning rate 3 × 10⁻⁴; batch size 32.'],
      ['Stage 1b: Conversation adaptation','DailyDialog: 76,052 reply pairs tagged STANDARD; CoSEM: 80,000 reply pairs tagged LOCAL.','Full-model fine-tuning; 5 epochs; learning rate 5 × 10⁻⁴; batch size 32.'],
      ['Stage 2: Project-specific adaptation','Our team-authored dataset: 190 conversations, providing 380 English/Singlish reply targets and 180 additional reply examples extracted from their histories.','LoRA rank 16; learning rate 10⁻³; batch size 16; early stopping.']])
note('Stages 1a and 1b together formed the language and conversation adaptation stage.')
label('Evaluation and observed limitations')
p('We evaluated development checkpoints using the same 50 validation conversations. Validation loss helped select checkpoints within each training setup, while inspection of generated replies assessed whether they answered the incoming message, sounded natural, and preserved meaning between English and Singlish. Loss alone was insufficient: a model could better predict the reference wording while still generating an unsuitable reply.')
p('The initial model produced fluent short responses, but its contextual understanding and style consistency were weaker than expected. Some replies missed the latest question or confused speaker roles, such as who should lend or borrow an item. Independently generating the English and Singlish replies also allowed them to express different decisions.')
p('Several factors may have contributed to these limitations. First, the 190 project conversations provided limited coverage of different situations and conversational constraints. Second, the public-dialogue supervision was imperfect: assigning STANDARD and LOCAL labels according to dataset source did not teach the model to express the same response in both styles. A small audit also found that 13 of 40 sampled CoSEM pairs did not directly answer their associated messages, indicating noise in the training material.')
p('These observations motivated a change from independently generating both styles to generating an English reply first and then rewriting that reply into Singlish.')

h('3.3.2 Adaptations')
p('To reduce differences in meaning between the two styles, we changed the generation process. Instead of generating English and Singlish replies independently, the model first produced an English reply, then rewrote that exact draft into Singlish. Both tasks used the same mT5 adapter with different task prefixes. We also aligned the paired training targets using assistant-reviewed rewrites.')
figure('Figure 3.3a placeholder — Independent English/Singlish generation compared with the revised English-first pipeline.')
p('The revised approach improved meaning consistency in development checks. Across three training seeds, the average number of meaning-consistent pairs increased from 20.3 to 46.0 out of 50, while contextual appropriateness remained approximately unchanged.')
table('Table 3.3b. Development comparison of independent generation and English-first rewriting.',
      ['Criterion','Independent generation','English-first rewriting'],[
      ['Contextually appropriate reply','33.0/50','33.3/50'],
      ['English–Singlish meaning preserved','20.3/50','46.0/50'],
      ['Natural, recognizable Singlish','23.7/50','36.7/50']])
note('Values are mean pass counts across three seeds on the same 50 validation conversations. They come from exploratory assistant reviews, not independent human ratings or the later judge evaluation.')
p('These results showed that preserving meaning did not necessarily produce a correct response. For example, when the peer asked “can I borrow your charger?”, one English-first model generated:')
example('English: “Sure, I’ll borrow it.”')
example('Singlish: “Can, I borrow my charger.”')
p('Both replies retained the borrowing error instead of offering to lend the charger. We therefore explored further changes to improve contextual understanding.')
table('Table 3.3c. Further mT5 experiments: design, results and examples.',
      ['Experiment','What we tested and why','Results and examples'],[
      ['Training-data expansion','Expanded 190 to 790 conversations with 600 synthetic additions to cover more situations. Control and expanded models received the same number of optimizer updates.','English adequacy remained 36/50, while recognizable Singlish fell from 38/50 to 25/50. Exact English–Singlish copies increased from 3 to 19 pairs, suggesting weaker style differentiation.'],
      ['Full fine-tuning versus LoRA','Trained on 24 English examples to check whether restricted adaptation prevented the model from learning the task. We then tested reworded versions of the same scenarios.','Full fine-tuning reproduced 24/24 training targets and produced adequate replies for 18/24 paraphrases, compared with 15/24 for the stronger tested LoRA setting. Greater capacity helped, but transfer remained imperfect.'],
      ['Targeted examples and broader LoRA coverage','Added 80 conversations targeting ownership, refusals and timing. We also extended LoRA beyond query/value attention projections to more attention and feed-forward components.','The combined intervention improved targeted diagnostics, but average English adequacy on the original validation set was 36.7/50, compared with 37.7/50 for the control. Improvements did not transfer consistently across cases.'],
      ['Alternative rewriting setups','Tested separate English/Singlish adapters and different rewrite inputs: the English draft alone, the draft with conversation history, and an additional style example. The aim was to separate content generation from style conversion.','Copying and meaning changes remained. One full-context rewrite changed “Friday afternoon works for me” into “Friday afternoon cannot.” Providing more context did not guarantee faithful rewriting.'],
      ['Preference learning','Added 73 reviewed preference pairs through Direct Preference Optimisation (DPO), encouraging suitable replies over unsuitable alternatives while retaining supervised training.','Reply-quality changes were mixed across three seeds. In one case, a reply accepting a proposed 2 pm meeting changed into an unframed 3 pm arrival, illustrating that preference training could also introduce errors.'],
      ['Reply-act conditioning and factual supervision','Tested acceptance/refusal conditioning, auxiliary fact extraction, and explicit context notes before reply generation. These aimed to make the intended action and relevant constraints clearer.','Some targeted cases improved, but constraints could still be ignored. Even with a supplied refusal act, the model could accept an $18 lunch despite an $8 budget. Incorrect extracted facts could also affect the final reply.']])
note('Quality counts are from recorded exploratory assistant reviews. Each experiment used its own comparison setup; these are not the later Gemma judge scores or a ranking across table rows.')
label('Decoding strategy')
p('We also compared greedy decoding with beam search. Greedy decoding follows the highest-probability token at each step, while beam search retains several possible continuations before selecting a completed reply. Beam search produced cleaner and more complete replies in reviewed examples, so we adopted four-beam decoding with repeated three-grams blocked. The mT5 reference allowed up to 48 generated tokens, while the later SEA-LION pipeline allowed 64.')
p('This improved the decoding configuration, but speaker-role and timing errors still occurred. Those errors required better contextual understanding as well as better selection of output sequences.')

h('3.3.3 Larger Models')
p('As the mT5 adaptations did not consistently resolve contextual errors, we explored larger pretrained models. We first investigated SEA-LION-v1-3B, a decoder-only model pretrained on languages used in Southeast Asia (AI Singapore, n.d.). Its regional-language exposure made it a relevant candidate for Singaporean conversations.')
p('SEA-LION underwent public-dialogue adaptation using 117,634 reply pairs, comprising 53,691 CoSEM and 63,943 DailyDialog examples. Project-specific training then used 790 conversations, including the original 190 and 600 corrected synthetic additions. We retained the English-first pipeline, with separate adapters for English generation and Singlish rewriting. The selected configuration used the four-beam decoding strategy described above; further training settings are summarized in Section 3.5.')
p('SEA-LION produced more coherent and contextually useful replies in the examples reviewed. However, fluent wording did not eliminate errors involving ownership, availability or exact times, and some local replies remained close to Standard English.')
p('We subsequently explored Qwen2.5-3B as a similarly sized alternative (Qwen Team, n.d.). Later experiments used an audited set of 768 conversations and the same English-first design. A follow-up Qwen run also used the full public-dialogue adaptation set, bringing its training recipe closer to the later SEA-LION experiments.')
table('Table 3.3d. Qualitative development observations for the three model backbones.',
      ['Model','Reason for testing','Development observations'],[
      ['mT5-base','Smaller multilingual encoder–decoder baseline, approximately 582 million parameters.','English-first rewriting improved paired meaning, but contextual and speaker-role errors remained.'],
      ['SEA-LION-v1-3B','Larger model with Southeast Asian language exposure.','More coherent replies in reviewed examples and useful local phrasing, although factual and rewriting errors persisted.'],
      ['Qwen2.5-3B','Similarly sized alternative to examine whether another multilingual backbone could handle the task.','Produced coherent English and some faithful rewrites, but local outputs sometimes remained Standard English or changed the intended meaning.']])
figure('Figure 3.3b placeholder — The same conversation with mT5, SEA-LION and Qwen outputs side by side. Highlight a specific difference in relevance, speaker roles or meaning preservation.')
p('The Qwen experiments also prompted us to examine the dataset’s language coverage. Thirty of the 50 validation messages listed local terms, but only two carried a pinyin tag. The dataset therefore primarily tested English-based conversations with light Singlish, providing limited evidence about extended code-switching. Regional interpretation still mattered: in one example, Qwen treated “na li”, meaning “where,” as the place name “Na Li.”')
label('Evaluation and generator choice')
p('Validation loss was used to select checkpoints within each training setup, rather than rank different model families. A fair comparison should assess generated replies on the same conversations using the same judge, rubric and criteria: appropriateness, naturalness and style fit, together with a separate English–Singlish meaning check. The observations above are qualitative; a matched judge evaluation of these model variants remains necessary.')
p('SEA-LION was selected as the V5 generator because it offered a useful balance of contextual coherence and local phrasing. The existing shared-system evaluation using Gemma 12B is reported in Section 4. Qwen’s separate use for reranking and wording refinement is described in Section 3.4.')

raw=SOURCE.read_bytes()
with zipfile.ZipFile(SOURCE) as z:
    infos=z.infolist();parts={i.filename:z.read(i.filename) for i in infos}
root=E.fromstring(parts['word/document.xml']);body=root.find('w:body',NS);nodes=list(body)
s=next(i for i,e in enumerate(nodes) if text(e).strip()=='3.3 Transformer reply generation (V5)')
e=next(i for i,x in enumerate(nodes) if text(x).strip().startswith('3.4 Reranking,'))
before=[canon(x) for x in nodes[:s+1]];after=[canon(x) for x in nodes[e:]]
new=[]
for b in blocks:
    kind=b['type']
    if kind=='table':
        new.append(paragraph(b['caption'],kind='caption',keep=True,size=20))
        t=make_table(b)
        # Long experiment tables can continue at row boundaries; repeat headers.
        if sum(len(v.split()) for row in b['rows'] for v in row)>220:
            for tr in list(t.findall('w:tr',NS))[1:]:
                for keep in tr.findall('.//w:keepNext',NS):keep.getparent().remove(keep)
        new.extend([t,paragraph('',size=20)])
    elif kind=='label':new.append(paragraph(b['text'],keep=True,bold=True))
    elif kind=='note':new.append(paragraph(b['text'],kind='caption',size=19))
    else:
        n=paragraph(b['text'],kind=kind)
        if kind=='example':E.SubElement(n.find('w:pPr',NS),w('ind')).set(w('left'),'240')
        new.append(n)
for n in nodes[s+1:e]:body.remove(n)
for i,n in enumerate(new,s+1):body.insert(i,n)
assert [canon(x) for x in list(body)[:s+1]]==before
assert [canon(x) for x in list(body)[s+1+len(new):]]==after
assert sum(x.tag==w('tbl') for x in new)==4
assert sum(b['type']=='figure_placeholder' for b in blocks)==2
assert not any('3.3.4 Distillation' in text(x) for x in new)
if BACKUP.exists():assert BACKUP.read_bytes()==raw,'Existing backup differs.'
else:shutil.copy2(SOURCE,BACKUP)
assert SOURCE.read_bytes()==raw,'Source changed concurrently.'
temp=SOURCE.with_name('.Copy of Final Report.approved-section33.tmp.docx')
xml=E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
with zipfile.ZipFile(temp,'w') as z:
    for i in infos:z.writestr(i,xml if i.filename=='word/document.xml' else parts[i.filename])
with zipfile.ZipFile(temp) as z:
    assert z.testzip() is None
    for name,data in parts.items():
        if name!='word/document.xml':assert z.read(name)==data,name
os.replace(temp,SOURCE)
(OUT/'approved_section33_content.json').write_text(json.dumps(blocks,ensure_ascii=False,indent=2)+'\n')
receipt={'document':str(SOURCE),'backup':str(BACKUP),'original_sha256':digest(raw),'updated_sha256':digest(SOURCE.read_bytes()),'section_word_count':len(' '.join(text(x) for x in new).split()),'native_tables':4,'figure_placeholders':2,'outside_section_unchanged':True,'other_package_parts_byte_identical':True,'google_document_id':'1TLPDJWHtaHo8UtL7ZkTaive5Y2gE3o_dGGBrO5Sn8og','google_access':'403 PERMISSION_DENIED; no Google document changed','accepted_revisions':['datasets and settings by training phase','early evaluation and possible causes of limitations','three-seed paired-style results with assistant-review caveat','expanded experiment descriptions, values and examples','adopted beam4 decoding; mT5 48 / SEA-LION 64 output tokens','larger models, regional-language coverage and common-judge evaluation','two figure placeholders; no distillation section']}
(OUT/'approved_section33_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
