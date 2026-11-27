"""Shorten section 3.3 without changing other sections or package parts."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
import zipfile

sys.path.insert(0, '/tmp/replycoach_report_20261121_lib')
from lxml import etree as E

OUT=Path(__file__).resolve().parent
SOURCE=Path('/home/spikeypear/Downloads/Copy of Final Report.docx')
BACKUP=SOURCE.with_name('Copy of Final Report - before shortening section 3.3.docx')
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS={'w':W}
def w(s):return '{'+W+'}'+s
def text(e):return ''.join(e.xpath('.//w:t/text()',namespaces=NS))
def canon(e):return E.tostring(e,method='c14n')
def digest(s):return hashlib.sha256(s).hexdigest()

blocks=[]
def p(s):blocks.append({'type':'paragraph','text':s})
def h(s):blocks.append({'type':'heading','text':s})
def figure(s):blocks.append({'type':'figure_placeholder','text':s})
def table(caption,headers,rows):blocks.append({'type':'table','caption':caption,'headers':headers,'rows':rows})

p('Following the RNN experiments, we explored pretrained Transformers to improve reply generation (Vaswani et al., 2017). V5 aimed to produce a relevant Standard English reply and a natural Singlish version with the same meaning. We began with mT5-base before exploring larger models.')
h('3.3.1 Initial Training Plan')
p('We selected mT5-base, a multilingual encoder–decoder model with approximately 582 million parameters (Xue et al., 2021). Training had two main stages: language and conversation adaptation, followed by project-specific reply training. An initial span-corruption phase, recorded as Stage 0 in the code, formed part of language adaptation.')
table('Table 3.3a. Main settings for the initial mT5 approach.',
      ['Stage','Training approach','Key settings'],[
      ['1. Language and conversation adaptation','Full-model training on CoSEM/NUS SMS text, followed by DailyDialog/CoSEM reply pairs.','Span phase: 2 epochs, LR 3 × 10⁻⁴. Dialogue phase: 5 epochs, LR 5 × 10⁻⁴. Batch size 32.'],
      ['2. Project-specific adaptation','190 conversations; STANDARD and LOCAL prefixes controlled reply style. One shared LoRA adapter learned both styles.','LoRA rank 16, alpha 32, dropout 0.1; LR 10⁻³; batch size 16; early stopping.']])
p('LoRA reduced the number of parameters updated during task training (Hu et al., 2021). However, independently generating the two styles often changed the reply’s meaning. For example, one version could agree while the other refused, or confuse who should lend an item.')
h('3.3.2 Adaptations')
p('We changed the pipeline to generate an English reply first, then rewrite that exact draft into Singlish. Both tasks used the same mT5 adapter with different task prefixes. This improved consistency between the two outputs, although an incorrect English reply could still lead to an incorrect Singlish reply.')
figure('Figure 3.3a placeholder — English-first pipeline: conversation → English draft → Singlish rewrite. Show the same intended meaning across both outputs.')
p('We also tested several ways to improve contextual understanding. These produced useful changes in individual cases, but no consistent overall improvement over the stronger mT5 reference.')
table('Table 3.3b. Summary of further mT5 experiments.',
      ['Changes tested','Main finding'],[
      ['More public and synthetic training data','Additional examples did not consistently resolve speaker-role or timing errors.'],
      ['Longer training, full fine-tuning and broader LoRA coverage','Better training fit did not reliably translate into better replies.'],
      ['Preference learning, reply-act conditioning and factual supervision','Some targeted gains, but contextual constraints were still missed.'],
      ['Separate adapters and alternative rewrite inputs','Style copying and meaning changes remained.']])
h('3.3.3 Larger Models')
p('We next explored SEA-LION-v1-3B, a decoder-only model pretrained on Southeast Asian languages (AI Singapore, n.d.). Its replies were generally more coherent and useful in the conversations reviewed. We retained the English-first approach, using separate English and Singlish adapters. Public-dialogue adaptation was followed by project-specific LoRA training; the final settings are summarized in Section 3.5.')
p('We preferred four-beam search over greedy decoding because it produced cleaner, more complete replies in reviewed examples. The selected pipeline blocked repeated three-grams and allowed up to 64 generated tokens per reply.')
p('Qwen2.5-3B was also explored as a similarly sized alternative (Qwen Team, n.d.). It produced coherent replies, suggesting that general conversational ability was important alongside regional-language exposure. Our validation inputs were mainly English-based chats with light Singlish: 30/50 listed local terms, while only 2/50 carried a pinyin tag. This gives limited evidence about extended code-switching, rather than showing that regional understanding is unimportant.')
figure('Figure 3.3b placeholder — One shared conversation with mT5, SEA-LION and Qwen replies side by side. Highlight relevance and English–Singlish meaning consistency.')
p('Validation loss was used to select checkpoints within a training setup, rather than rank different model families. A fair model comparison should use the same conversations, judge settings and criteria: appropriateness, naturalness and style fit, with a separate check of English–Singlish meaning. The exploratory observations here are qualitative; the model variants still require matched re-evaluation.')
p('SEA-LION was selected as the V5 generator for its balance of coherent replies and local phrasing. The shared system evaluation uses Gemma 12B as the judge and is reported in Section 4. Qwen’s separate role in reranking and local-wording refinement is described in Section 3.4.')

raw=SOURCE.read_bytes()
with zipfile.ZipFile(SOURCE) as z:
    infos=z.infolist();parts={i.filename:z.read(i.filename) for i in infos}
root=E.fromstring(parts['word/document.xml']);body=root.find('w:body',NS);nodes=list(body)
start=next(i for i,e in enumerate(nodes) if text(e).strip()=='3.3 Transformer reply generation (V5)')
end=next(i for i,e in enumerate(nodes) if text(e).strip().startswith('3.4 Reranking,'))
before=[canon(e) for e in nodes[:start+1]];after=[canon(e) for e in nodes[end:]]
old_words=len(' '.join(text(e) for e in nodes[start:end]).split())

def paragraph(s,kind='paragraph',keep=False,size=22,bold=False):
    e=E.Element(w('p'));pp=E.SubElement(e,w('pPr'))
    if kind=='heading':
        E.SubElement(pp,w('pStyle')).set(w('val'),'Heading3');keep=True;bold=True
    if keep:E.SubElement(pp,w('keepNext'))
    E.SubElement(pp,w('widowControl'))
    E.SubElement(pp,w('spacing'),{w('before'):'160' if kind=='heading' else '0',w('after'):'100',w('line'):'240',w('lineRule'):'auto'})
    if kind=='figure_placeholder':
        E.SubElement(pp,w('shd'),{w('val'):'clear',w('fill'):'F2F2F2'})
        E.SubElement(pp,w('ind'),{w('left'):'100',w('right'):'100'})
    r=E.SubElement(e,w('r'));rp=E.SubElement(r,w('rPr'))
    if bold:E.SubElement(rp,w('b'))
    if kind in ['figure_placeholder','caption']:E.SubElement(rp,w('i'))
    if kind=='figure_placeholder':E.SubElement(rp,w('color')).set(w('val'),'666666');size=20
    E.SubElement(rp,w('sz')).set(w('val'),str(size));E.SubElement(rp,w('szCs')).set(w('val'),str(size))
    t=E.SubElement(r,w('t'));t.text=s;t.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
    return e

def make_table(b):
    count=len(b['headers']);ratios=[.22,.38,.40] if count==3 else [.44,.56]
    widths=[round(9360*x) for x in ratios];widths[-1]=9360-sum(widths[:-1])
    e=E.Element(w('tbl'));pr=E.SubElement(e,w('tblPr'))
    E.SubElement(pr,w('tblW'),{w('w'):'9360',w('type'):'dxa'})
    borders=E.SubElement(pr,w('tblBorders'))
    for side in ['top','left','bottom','right','insideH','insideV']:E.SubElement(borders,w(side),{w('val'):'single',w('sz'):'5',w('color'):'B7B7B7'})
    E.SubElement(pr,w('tblLayout')).set(w('type'),'fixed')
    margins=E.SubElement(pr,w('tblCellMar'))
    for side,v in [('top',70),('bottom',70),('left',90),('right',90)]:E.SubElement(margins,w(side),{w('w'):str(v),w('type'):'dxa'})
    grid=E.SubElement(e,w('tblGrid'))
    for width in widths:E.SubElement(grid,w('gridCol')).set(w('w'),str(width))
    rows=[b['headers'],*b['rows']]
    for ri,row in enumerate(rows):
        tr=E.SubElement(e,w('tr'));tp=E.SubElement(tr,w('trPr'));E.SubElement(tp,w('cantSplit'))
        if ri==0:E.SubElement(tp,w('tblHeader'))
        for width,s in zip(widths,row):
            c=E.SubElement(tr,w('tc'));cp=E.SubElement(c,w('tcPr'));E.SubElement(cp,w('tcW'),{w('w'):str(width),w('type'):'dxa'})
            if ri==0:E.SubElement(cp,w('shd'),{w('fill'):'E7E6E6',w('val'):'clear'})
            c.append(paragraph(s,keep=ri<len(rows)-1,size=20,bold=ri==0))
    return e

new=[]
for b in blocks:
    if b['type']=='table':
        new.append(paragraph(b['caption'],kind='caption',keep=True,size=20));new.append(make_table(b));new.append(paragraph(''))
    else:new.append(paragraph(b['text'],kind=b['type']))
for e in nodes[start+1:end]:body.remove(e)
for i,e in enumerate(new,start+1):body.insert(i,e)
assert [canon(e) for e in list(body)[:start+1]]==before
assert [canon(e) for e in list(body)[start+1+len(new):]]==after
new_text=' '.join(text(e) for e in new)
assert '30/50 listed local terms' in new_text
assert 'New retrospective assistant review' not in new_text
assert 'Gemma 12B' in new_text and 'four-beam search' in new_text
assert sum(b['type']=='figure_placeholder' for b in blocks)==2
assert len(new_text.split())<850

if BACKUP.exists():assert BACKUP.read_bytes()==raw,'Existing backup differs.'
else:shutil.copy2(SOURCE,BACKUP)
assert SOURCE.read_bytes()==raw,'Source changed during the edit.'
temp=SOURCE.with_name('.Copy of Final Report.short-section33.tmp.docx')
xml=E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
with zipfile.ZipFile(temp,'w') as z:
    for i in infos:z.writestr(i,xml if i.filename=='word/document.xml' else parts[i.filename])
with zipfile.ZipFile(temp) as z:
    assert z.testzip() is None
    for name,data in parts.items():
        if name!='word/document.xml':assert z.read(name)==data
os.replace(temp,SOURCE)
new_words=len(new_text.split())+len(text(nodes[start]).split())
receipt={'document':str(SOURCE),'backup':str(BACKUP),'original_sha256':digest(raw),'updated_sha256':digest(SOURCE.read_bytes()),'old_section_word_count':old_words,'new_section_word_count':new_words,'reduction_percent':round(100*(old_words-new_words)/old_words,1),'native_tables_in_section':2,'figure_placeholders':2,'outside_section_unchanged':True,'other_zip_parts_byte_identical':True,'provisional_assistant_score_table_removed':True,'loss_distinguished_from_common_quality_metrics':True,'decoding':'Selected beam4/ngram3/64; cleaner reviewed examples, no universal superiority claim','judge':'Gemma 12B per existing shared report; Qwen reranking is separate'}
(OUT/'shortened_section33_content.json').write_text(json.dumps(blocks,ensure_ascii=False,indent=2)+'\n')
(OUT/'shortened_section33_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
