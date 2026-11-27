"""Update only section 3.3, preserving peers' XML and every other DOCX part."""
from pathlib import Path
import ast
import hashlib
import json
import os
import shutil
import sys
import zipfile

sys.path.insert(0, '/tmp/replycoach_report_20261121_lib')
from lxml import etree as E

OUT = Path(__file__).resolve().parent
UPDATE = OUT / 'distillation_update'
SOURCE = Path('/home/spikeypear/Downloads/Copy of Final Report.docx')
BACKUP = SOURCE.with_name('Copy of Final Report - before distillation results update.docx')
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W}

tree = ast.parse((OUT / 'shorten_section33.py').read_text())
helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'w', 'paragraph'}]
exec(compile(ast.Module(body=helpers, type_ignores=[]), 'format_helpers', 'exec'))


def text(node):
    return ''.join(node.xpath('.//w:t/text()', namespaces=NS))


def canon(node):
    return E.tostring(node, method='c14n')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def table(block):
    columns = len(block['headers'])
    widths = {3: [2060, 3550, 3750], 5: [2480, 1720, 1720, 1720, 1720]}[columns]
    element = E.Element(w('tbl'))
    props = E.SubElement(element, w('tblPr'))
    E.SubElement(props, w('tblW'), {w('w'): '9360', w('type'): 'dxa'})
    E.SubElement(props, w('tblLayout')).set(w('type'), 'fixed')
    borders = E.SubElement(props, w('tblBorders'))
    for side in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        E.SubElement(borders, w(side), {w('val'): 'single', w('sz'): '5', w('color'): 'B7B7B7'})
    margins = E.SubElement(props, w('tblCellMar'))
    for side, value in [('top', 70), ('bottom', 70), ('left', 90), ('right', 90)]:
        E.SubElement(margins, w(side), {w('w'): str(value), w('type'): 'dxa'})
    grid = E.SubElement(element, w('tblGrid'))
    for width in widths:
        E.SubElement(grid, w('gridCol')).set(w('w'), str(width))
    for i, row in enumerate([block['headers'], *block['rows']]):
        assert len(row) == columns
        tr = E.SubElement(element, w('tr'))
        tp = E.SubElement(tr, w('trPr'))
        E.SubElement(tp, w('cantSplit'))
        if i == 0:
            E.SubElement(tp, w('tblHeader'))
        for width, value in zip(widths, row):
            cell = E.SubElement(tr, w('tc'))
            cp = E.SubElement(cell, w('tcPr'))
            E.SubElement(cp, w('tcW'), {w('w'): str(width), w('type'): 'dxa'})
            if i == 0:
                E.SubElement(cp, w('shd'), {w('fill'): 'E7E6E6', w('val'): 'clear'})
            cell.append(paragraph(value, keep=i == 0, size=19 if columns == 5 else 20, bold=i == 0))
    return element


def main():
    blocks = json.loads((OUT / 'approved_section33_content.json').read_text())
    changed = []
    for b in blocks:
        changed.append(b)
        if b.get('text', '').startswith('We initially planned to adapt mT5-base'):
            changed.append({'type': 'paragraph', 'text': 'This was a transfer-learning strategy for a low-data task. mT5 already supplied multilingual representations (Xue et al., 2021); CoSEM and NUS SMS adaptation exposed it to informal local text, while CoSEM/DailyDialog reply training taught conversational response generation before learning from only 190 project conversations. Skipping this intermediate adaptation would leave the small project set to teach both the reply task and local style, increasing the risk of memorisation and weak generalisation. This is the design rationale: we did not isolate removal of this stage in a controlled ablation, so failure without it is not established.'})
        if b.get('text', '').startswith('We subsequently explored Qwen2.5-3B'):
            b['text'] = 'We subsequently explored Qwen2.5-3B as a similarly sized alternative (Qwen Team, n.d.). Later SEA-LION and Qwen experiments used the same audited 768 task conversations and 766 eligible English–Singlish rewrite pairs. Qwen also received the full public-dialogue pass. This brought the recipes closer, although differences between the pretrained models remained.'
        if b.get('text', '').startswith('The Qwen experiments also prompted'):
            changed.extend([
                {'type': 'label', 'text': 'Dataset audit and weak Singlish differentiation'},
                {'type': 'paragraph', 'text': 'We audited the actual training files, reviewed 40 sampled rewrite pairs, and checked both teachers on the same 50 validation conversations. The targets contained useful local syntax, such as “i check got notes anot then update you”, but also generic casual English. The issue was therefore uneven style supervision and unreliable rewriting, rather than a complete absence of Singlish.'},
                {'type': 'table', 'caption': 'Table 3.3e. Evidence relevant to weak Singlish style.',
                 'headers': ['Evidence', 'Finding', 'Implication'], 'rows': [
                     ['Project rewrite targets', '591/766 pairs (77.2%) were synthetic. The sampled targets often used short acknowledgements, ellipsis and article omission.', 'Limited native style variety may encourage narrow patterns. These forms can be natural Singlish; particles are not compulsory.'],
                     ['Teacher outputs on the same 50 cases', 'SEA-LION copied its English wording in 7/50 cases; Qwen in 19/50, ignoring case and punctuation. None of the 766 reference pairs were normalized copies.', 'The models sometimes failed to apply the learned style distinction at inference. Copying cannot be explained solely by duplicate reference labels.'],
                     ['Retained distillation targets', 'English/LOCAL normalized copies occurred in 45/571 SEA-LION pairs (7.9%) and 40/571 Qwen pairs (7.0%).', 'Students inherit some weakly differentiated teacher targets; distillation does not automatically strengthen local style.']
                 ]},
                {'type': 'note', 'text': 'Copying is a style diagnostic, not a semantic-error rate: some short replies work in both registers. The 40-pair review was an assistant inspection, not a human benchmark. Surface particle counts cannot capture all Singlish syntax.'},
                {'type': 'paragraph', 'text': 'Public reply training provided broader conversation exposure but did not supply equivalent English–Singlish versions of every reply. Regional pretraining alone therefore did not guarantee controlled rewriting. More varied, meaning-aligned local targets and explicit style checks are plausible improvements, but this audit does not prove a single cause. The validation inputs also offered limited code-switching coverage: 30/50 listed local terms, while only 2/50 carried a pinyin tag.'}
            ])
            changed.remove(b)
        if b.get('text', '').startswith('Validation loss was used to select checkpoints within each training setup, rather than rank different model families.'):
            b['text'] = 'Validation loss selected checkpoints within a training setup; it was not used to rank model families. SEA-LION and Qwen were compared qualitatively on matched conversations, assessing relevance, naturalness, local style and paired meaning. The numerical judge results in Section 3.3.4 concern mT5 distillation variants only; the teachers were not scored in that run.'
        if b.get('text', '').startswith('SEA-LION was selected as the V5 generator'):
            b['text'] = 'SEA-LION was retained as the V5 generator for its useful balance of contextual coherence and local phrasing, while the smaller-model experiments continued. This was an engineering choice rather than a demonstrated win on every criterion. The shared-system evaluation is reported in Section 4; Qwen’s separate reranking role is described in Section 3.4.'
    changed.extend([
        {'type': 'heading', 'text': '3.3.4 Distillation'},
        {'type': 'paragraph', 'text': 'We next tested whether the larger teachers could improve the smaller generator. The reported run continued the existing mT5 A model: its backbone had already undergone Singlish span adaptation and public-dialogue training, and its trained shared task adapter was retained. All four arms started from that same checkpoint; earlier fresh-adapter experiments are excluded from this comparison.'},
        {'type': 'paragraph', 'text': 'The teachers supplied English replies and Singlish rewrites on the training inputs. After assistant review and matching the accepted cases, 571 paired conversations remained. For each teacher, we compared cross-entropy training on its text with the same text objective plus aligned soft distillation (KD), which transferred additional teacher-distribution information across different vocabularies. KD also introduced training-only projection/alignment components, so this was not a pure KL-only ablation.'},
        {'type': 'paragraph', 'text': 'Each arm completed 12 epochs (1,716 updates), continuing one rank-16 query/value LoRA adapter for both tasks while freezing the 582-million-parameter backbone. The learning rate was 3 × 10⁻⁵ with 5% warmup and cosine decay. Checkpoints were selected using original-reference validation loss. Inference retained English-first rewriting, four beams, repeated-three-gram blocking and a 48-token output limit.'},
        {'type': 'label', 'text': 'Matched judge evaluation'},
        {'type': 'paragraph', 'text': 'We assessed the unchanged model and four selected variants on the same 50 validation conversations, in both styles, using the group’s Gemma 4 12B judge, unchanged rubric and ten simulated personas (temperature 0, seed 425). Model identities were hidden. Identical replies within the same conversation/style shared ratings, giving 3,390 unique requests mapped to 5,000 ratings. All requests completed successfully. Persona ratings were averaged within each conversation before averaging across conversations.'},
        {'type': 'table', 'caption': 'Table 3.3f. Matched judge scores for adapted mT5 (English / Singlish).',
         'headers': ['mT5 variant', 'Appropriateness /5', 'Naturalness /5', 'Style fit /5', 'Would send (%)'], 'rows': []},
        {'type': 'note', 'text': 'Each cell reports English / Singlish. Text = teacher-text cross-entropy; +KD = text plus aligned soft distillation. Selected epochs: SEA-LION text 3, SEA-LION +KD 1, Qwen text 2, Qwen +KD 1. Personas are simulated by one judge, not ten independent human participants.'},
        {'type': 'paragraph', 'text': 'SEA-LION text training had the highest observed appropriateness and willingness-to-send means. Relative to unchanged mT5, appropriateness increased by 0.25 for English and 0.12 for Singlish. However, conversation-level paired 95% bootstrap intervals included zero (−0.13 to 0.66 and −0.37 to 0.59). Singlish style fit fell from 4.16 to 3.90. Adding soft KD showed no consistent advantage, so the results support further investigation of teacher-text training rather than a conclusive replacement.'},
        {'type': 'paragraph', 'text': 'Sampled teacher–student comparisons exposed remaining gaps. When asked to change a printer booking, Qwen replied “Sure, go ahead and change your booking”, while its text-trained student replied “sure, i’ll try to use it later”, changing whose action was discussed. The students handled simple time confirmations better, but role reversals and missed requests persisted. These examples are qualitative; they do not quantify the teacher–student performance gap.'},
        {'type': 'paragraph', 'text': 'These are repeatedly used validation cases, not untouched test results. The judge scored each draft separately and therefore could miss English–Singlish reversals; for example, a KD model’s English reply said the quiz was finished while its Singlish reply said it was not. English style-fit scores also saturated at 5/5. A separate meaning-consistency check and human style review remain necessary.'}
    ])
    scores = json.loads((OUT.parents[1] / 'experiments/mt5_distill_judge_v1_results.json').read_text())
    lookup = {(r['model'], r['style']): r for r in scores['summary']}
    result_table = next(b for b in changed if b.get('caption', '').startswith('Table 3.3f'))
    for arm, label in [('unchanged', 'Unchanged mT5 A'), ('sealion_ce', 'SEA-LION text'), ('sealion_kd', 'SEA-LION text +KD'), ('qwen_ce', 'Qwen text'), ('qwen_kd', 'Qwen text +KD')]:
        en, sg = lookup[arm, 'STANDARD'], lookup[arm, 'LOCAL']
        result_table['rows'].append([label, *[f'{en[key]:.2f} / {sg[key]:.2f}' for key in ['appropriateness', 'naturalness', 'style_fit']], f"{en['would_send_pct']:.1f} / {sg['would_send_pct']:.1f}"])
    raw = SOURCE.read_bytes()
    with zipfile.ZipFile(SOURCE) as z:
        infos = z.infolist()
        parts = {i.filename: z.read(i.filename) for i in infos}
    root = E.fromstring(parts['word/document.xml'])
    body = root.find('w:body', NS)
    nodes = list(body)
    start = next(i for i, n in enumerate(nodes) if text(n).strip() == '3.3 Transformer reply generation (V5)')
    end = next(i for i, n in enumerate(nodes) if text(n).strip().startswith('3.4 Reranking,'))
    before, after = [canon(n) for n in nodes[:start + 1]], [canon(n) for n in nodes[end:]]
    new = []
    for b in changed:
        kind = b['type']
        if kind == 'table':
            new.extend([paragraph(b['caption'], kind='caption', keep=True, size=20), table(b), paragraph('', size=20)])
        elif kind == 'label':
            new.append(paragraph(b['text'], keep=True, bold=True))
        elif kind == 'note':
            new.append(paragraph(b['text'], kind='caption', size=19))
        else:
            new.append(paragraph(b['text'], kind=kind))
    for n in nodes[start + 1:end]:
        body.remove(n)
    for i, n in enumerate(new, start + 1):
        body.insert(i, n)
    assert [canon(n) for n in list(body)[:start + 1]] == before
    assert [canon(n) for n in list(body)[start + 1 + len(new):]] == after
    assert sum(b['type'] == 'table' for b in changed) == 6
    assert sum(b['type'] == 'figure_placeholder' for b in changed) == 2
    words = len(' '.join(text(n) for n in new).split())
    assert words < 2800, words
    if BACKUP.exists():
        assert BACKUP.read_bytes() == raw, 'Do not overwrite a different backup or reapply silently.'
    else:
        shutil.copy2(SOURCE, BACKUP)
    assert SOURCE.read_bytes() == raw, 'Source changed during editing.'
    temp = SOURCE.with_name('.Copy of Final Report.distillation-update.tmp.docx')
    xml = E.tostring(root, encoding='UTF-8', xml_declaration=True, standalone=True)
    with zipfile.ZipFile(temp, 'w') as z:
        for info in infos:
            z.writestr(info, xml if info.filename == 'word/document.xml' else parts[info.filename])
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None
        assert all(z.read(name) == value for name, value in parts.items() if name != 'word/document.xml')
    os.replace(temp, SOURCE)
    (UPDATE / 'updated_section33_content.json').write_text(json.dumps(changed, ensure_ascii=False, indent=2) + '\n')
    receipt = {'document': str(SOURCE), 'backup': str(BACKUP), 'original_sha256': sha(raw), 'updated_sha256': sha(SOURCE.read_bytes()),
               'section_word_count': words, 'native_tables': 6, 'figure_placeholders': 2,
               'outside_section_unchanged': True, 'other_package_parts_byte_identical': True,
               'distillation_source': 'experiments/mt5_warm_distill_v1_results.json',
               'judge_source': 'experiments/mt5_distill_judge_v1_results.json',
               'dataset_audit': str(UPDATE / 'dataset_style_audit.json'),
               'theory_reference': 'https://aclanthology.org/2021.naacl-main.41/'}
    (UPDATE / 'edit_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
