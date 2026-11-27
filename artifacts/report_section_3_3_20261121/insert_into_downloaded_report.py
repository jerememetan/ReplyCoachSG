"""Replace only section 3.3 and add its citations, preserving the source DOCX package."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import sys
import zipfile

sys.path.insert(0, '/tmp/replycoach_report_20261121_lib')
from lxml import etree as E

OUT = Path(__file__).resolve().parent
SOURCE = Path('/home/spikeypear/Downloads/Copy of Final Report.docx')
BACKUP = SOURCE.with_name('Copy of Final Report - before section 3.3.docx')
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W}
def w(name): return '{'+W+'}'+name
def text(e): return ''.join(e.xpath('.//w:t/text()', namespaces=NS))
def canonical(e): return E.tostring(e, method='c14n')
def sha(data): return hashlib.sha256(data).hexdigest()

original_bytes = SOURCE.read_bytes()
with zipfile.ZipFile(SOURCE) as z:
    infos = z.infolist()
    parts = {i.filename:z.read(i.filename) for i in infos}
root = E.fromstring(parts['word/document.xml'])
body = root.find('w:body', NS)
original_nodes = list(body)
start = next(i for i,e in enumerate(body) if text(e).strip()=='3.3 Transformer reply generation (V5)')
end = next(i for i,e in enumerate(body) if text(e).strip().startswith('3.4 Reranking,'))
assert start < end
assert any('3.3.4 Distillation' in text(e) for e in original_nodes[start:end])
before = [canonical(e) for e in original_nodes[:start+1]]
after = [canonical(e) for e in original_nodes[end:]]

blocks = json.loads((OUT/'report_content.json').read_text())
reference_start = next(i for i,b in enumerate(blocks) if b.get('text')=='References and project evidence')
blocks = blocks[1:reference_start]

# The supplied complete report contains later final-evaluation results and uses
# the repair25 model. Keep the exploratory C/Qwen comparison distinct from it.
replacements = {
    'Public adaptation used 117,634 reply pairs':
    'For the SEA-LION pipeline used in the shared comparison, public adaptation used 117,634 reply pairs: 53,691 CoSEM and 63,943 DailyDialog examples. The frozen backbone was adapted through rank-8 LoRA, with alpha 16 and dropout 0.05. Public training used one full pass at a learning rate of 5 × 10⁻⁵ and an effective batch size of 16. Project tuning then used 790 conversations, comprising the original 190 and 600 corrected generated additions, with a learning rate of 10⁻⁴ and an effective batch size of eight. The English checkpoint was selected at epoch two. An anchored rewrite stage used 745 eligible pairs at a learning rate of 3 × 10⁻⁵; its selected epoch-one adapter was followed by a contrastive repair branch, which retained the original English checkpoint and selected local repair step 25. These settings correspond to the SEA-LION pipeline summarized in Section 3.5.',
    'Qwen2.5-3B was then investigated':
    'Later exploratory runs used an audited subset of 768 conversations and 766 eligible rewrite pairs. The SEA-LION C recipe used English/rewrite peak learning rates of 5 × 10⁻⁵ and 10⁻⁵ respectively, eight epochs per stage, and 5% linear warmup followed by cosine decay. Qwen2.5-3B was investigated as a similarly sized multilingual causal language model with approximately 3.09 billion parameters (Qwen Team, n.d.). An initial fresh-adapter comparison trained SEA-LION and Qwen for four epochs per task without project public adaptation. A subsequent Qwen run reproduced the full 117,634-example public pass and the audited C task/rewrite recipe, selecting epoch two for both English generation and rewriting. These were later research comparisons; they did not replace the SEA-LION repair-step-25 pipeline used in the shared system evaluation.',
    'A separate retrospective study compared saved outputs':
    'A separate retrospective development study compared saved outputs on all 50 original validation conversations. The mT5 reference was the historical shared-intent seed-1 checkpoint at update 784; the larger-model references were the later audited SEA-LION C pipeline and the full-public-recipe Qwen pipeline. Their incoming messages, histories and speaker labels were verified to match. The mT5 reference used the reviewed 190-conversation task dataset, whereas the later larger-model pipelines used the audited expanded dataset. This compares complete trained systems rather than isolating model size or architecture, and it is separate from the 60-conversation evaluation of the final systems in Section 4.',
    'Taken together, the evidence supports carrying SEA-LION':
    'The development studies supported choosing SEA-LION as the V5 generator because it offered a useful balance of coherent replies and recognizable Singlish. Qwen remained a strong exploratory alternative for English content, while mT5 provided a smaller reference whose cross-style consistency improved with English-first rewriting. The later SEA-LION C/Qwen review does not establish a definitive ranking between those two research variants. The final system comparison and the decision to use SEA-LION directly for Standard English, with a separate local-wording refinement stage, are reported in Sections 3.4 and 4.'
}
seen = set()
for b in blocks:
    if b['type']=='paragraph':
        for prefix,replacement in replacements.items():
            if b['text'].startswith(prefix):
                b['text']=replacement; seen.add(prefix); break
assert seen==set(replacements)

def citations(s):
    s=s.replace('[1]', '(Vaswani et al., 2017)').replace('[2]', '(Xue et al., 2021)')
    s=s.replace('[3]', '(Hu et al., 2021)').replace('[4]', '(AI Singapore, n.d.)')
    s=s.replace('[5]', '(Qwen Team, n.d.)')
    s=re.sub(r'\s*\[P\d+(?:[–-]P?\d+)?\]', '', s)
    s=s.replace('counts below are exploratory review outcomes', 'counts below are exploratory assistant-review outcomes')
    return s

def paragraph(s, *, style=None, size=22, bold=False, italic=False, keep=False, before=0, after=120):
    p=E.Element(w('p')); pp=E.SubElement(p,w('pPr'))
    if style:
        E.SubElement(pp,w('pStyle')).set(w('val'),style)
    if keep: E.SubElement(pp,w('keepNext'))
    E.SubElement(pp,w('widowControl'))
    E.SubElement(pp,w('spacing'),{w('before'):str(before),w('after'):str(after),w('line'):'240',w('lineRule'):'auto'})
    r=E.SubElement(p,w('r'));rp=E.SubElement(r,w('rPr'))
    if bold:E.SubElement(rp,w('b'))
    if italic:E.SubElement(rp,w('i'))
    E.SubElement(rp,w('sz')).set(w('val'),str(size))
    E.SubElement(rp,w('szCs')).set(w('val'),str(size))
    t=E.SubElement(r,w('t'));t.set('{http://www.w3.org/XML/1998/namespace}space','preserve');t.text=s
    return p

def make_table(block):
    count=len(block['headers']); rows=[block['headers'],*block['rows']]
    ratios=([.27,.146,.146,.146,.146,.146] if count==6 else [.32,.68] if count==2 else [.40,.30,.30] if block['caption'].startswith('Table 3.3b') else [.23,.25,.52])
    widths=[round(9360*r) for r in ratios];widths[-1]=9360-sum(widths[:-1])
    table=E.Element(w('tbl'));pr=E.SubElement(table,w('tblPr'))
    E.SubElement(pr,w('tblW'),{w('w'):'9360',w('type'):'dxa'})
    borders=E.SubElement(pr,w('tblBorders'))
    for side in ['top','left','bottom','right','insideH','insideV']:
        E.SubElement(borders,w(side),{w('val'):'single',w('sz'):'5',w('color'):'000000'})
    E.SubElement(pr,w('tblLayout')).set(w('type'),'fixed')
    margins=E.SubElement(pr,w('tblCellMar'))
    for side,v in [('top',65),('bottom',65),('left',80),('right',80)]:
        E.SubElement(margins,w(side),{w('w'):str(v),w('type'):'dxa'})
    grid=E.SubElement(table,w('tblGrid'))
    for width in widths:E.SubElement(grid,w('gridCol')).set(w('w'),str(width))
    for ri,row in enumerate(rows):
        tr=E.SubElement(table,w('tr'));trpr=E.SubElement(tr,w('trPr'))
        E.SubElement(trpr,w('cantSplit'))
        if ri==0:E.SubElement(trpr,w('tblHeader'))
        for width,s in zip(widths,row):
            cell=E.SubElement(tr,w('tc'));cp=E.SubElement(cell,w('tcPr'))
            E.SubElement(cp,w('tcW'),{w('w'):str(width),w('type'):'dxa'})
            if ri==0:E.SubElement(cp,w('shd'),{w('fill'):'CCCCCC',w('val'):'clear'})
            E.SubElement(cp,w('vAlign')).set(w('val'),'top')
            cell.append(paragraph(citations(str(s)),size=19 if count==6 else 20,bold=ri==0,keep=(len(rows)<=7 and ri<len(rows)-1),after=40))
    return table

new=[]
for b in blocks:
    if b['type']=='heading':
        new.append(paragraph(citations(b['text']),style='Heading3',bold=True,keep=True,before=180,after=100))
    elif b['type']=='paragraph':new.append(paragraph(citations(b['text'])))
    else:
        new.append(paragraph(citations(b['caption']),italic=True,keep=True,size=20,before=80,after=80))
        new.append(make_table(b))
        new.append(paragraph('',after=80))

for e in original_nodes[start+1:end]:body.remove(e)
for i,e in enumerate(new,start+1):body.insert(i,e)

references = [
    'AI Singapore. (n.d.). SEA-LION-v1-3B [Model card]. Hugging Face. https://huggingface.co/aisingapore/SEA-LION-v1-3B',
    'Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-rank adaptation of large language models [Preprint]. arXiv. https://arxiv.org/abs/2106.09685',
    'Qwen Team. (n.d.). Qwen2.5-3B [Model card]. Hugging Face. https://huggingface.co/Qwen/Qwen2.5-3B',
    'Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need [Preprint]. arXiv. https://arxiv.org/abs/1706.03762',
    'Xue, L., Constant, N., Roberts, A., Kale, M., Al-Rfou, R., Siddhant, A., Barua, A., & Raffel, C. (2021). mT5: A massively multilingual pre-trained text-to-text transformer [Preprint]. arXiv. https://arxiv.org/abs/2010.11934',
]
ref_heading=next(e for e in body if text(e).strip()=='References')
existing_references=list(body)[body.index(ref_heading)+1:]
added_refs=[]
for s in references:
    p=paragraph(s,before=220,after=220)
    following=next((e for e in existing_references if e.tag==w('p') and text(e).strip().casefold()>s.casefold()),None)
    if following is not None:body.insert(body.index(following),p)
    else:body.insert(body.index(body.find('w:sectPr',NS)),p)
    added_refs.append(p)

# Preserve every pre-existing element outside section 3.3, including references,
# figures, bookmarks, teammate tables, and the final-evaluation narrative.
assert [canonical(e) for e in list(body)[:start+1]]==before
suffix=list(body)[body.index(original_nodes[end]):]
assert [canonical(e) for e in suffix if e not in added_refs]==after
assert sum(e.tag==w('tbl') for e in body)==11
assert '3.3.4 Distillation' not in ''.join(text(e) for e in new)
assert not re.search(r'\[P\d+', ''.join(text(e) for e in new))
assert not any('Lead: Hao Ming' in text(e) for e in new)

if BACKUP.exists():
    assert BACKUP.read_bytes()==original_bytes, 'Existing backup differs; refusing to overwrite it.'
else:shutil.copy2(SOURCE,BACKUP)
assert SOURCE.read_bytes()==original_bytes, 'The source was changed concurrently.'
new_xml=E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
TEMP=SOURCE.with_name('.Copy of Final Report.section33.tmp.docx')
with zipfile.ZipFile(TEMP,'w') as z:
    for info in infos:z.writestr(info,new_xml if info.filename=='word/document.xml' else parts[info.filename])
with zipfile.ZipFile(TEMP) as z:
    assert z.testzip() is None
    for name,data in parts.items():
        if name!='word/document.xml':assert z.read(name)==data,name
    E.fromstring(z.read('word/document.xml'))
os.replace(TEMP,SOURCE)
receipt={'source':str(SOURCE),'backup':str(BACKUP),'original_sha256':sha(original_bytes),'updated_sha256':sha(SOURCE.read_bytes()),'replaced_section':'3.3 only','native_tables_added':5,'bibliographic_entries_added':len(added_refs),'all_original_elements_outside_section_3_3_preserved':True,'all_other_zip_parts_byte_identical':True,'original_figures_preserved':True,'distillation_placeholder_removed':True,'existing_author_date_citation_style_matched':True,'later_report_evaluation_context_incorporated':True,'exploratory_C_Qwen_variants_distinguished_from_final_repair25':True,'new_content':blocks}
(OUT/'downloaded_report_edit_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='new_content'},ensure_ascii=False,indent=2))
