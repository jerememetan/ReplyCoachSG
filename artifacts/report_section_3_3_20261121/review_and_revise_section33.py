"""Record an editorial review, then revise section 3.3 without touching peers' work."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import zipfile

sys.path.insert(0, '/tmp/replycoach_report_20261121_lib')
from lxml import etree as E
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
OUT = BASE / 'editorial_review'
SOURCE = Path('/home/spikeypear/Downloads/Copy of Final Report.docx')
BACKUP = SOURCE.with_name('Copy of Final Report - before editorial review.docx')
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W}
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PR = 'http://schemas.openxmlformats.org/package/2006/relationships'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def text(node):
    return ''.join(node.xpath('.//w:t/text()', namespaces=NS))


def canon(node):
    return E.tostring(node, method='c14n')


def make_figure():
    im = Image.new('RGB', (1800, 570), 'white')
    draw = ImageDraw.Draw(im)
    fontpath = '/usr/share/fonts/truetype/dejavu/'
    font = ImageFont.truetype(fontpath + 'DejaVuSans.ttf', 40)
    small = ImageFont.truetype(fontpath + 'DejaVuSans.ttf', 32)
    bold = ImageFont.truetype(fontpath + 'DejaVuSans-Bold.ttf', 38)
    def box(rect, words, fill='#EFF4F8'):
        draw.rounded_rectangle(rect, radius=14, fill=fill, outline='#7892A7', width=3)
        draw.multiline_text(((rect[0]+rect[2])/2, (rect[1]+rect[3])/2), words, font=font, fill='#162A3C', anchor='mm', align='center', spacing=8)
    def arrow(points):
        draw.line(points, fill='#47677F', width=5)
        x, y = points[-1]
        draw.polygon([(x, y), (x-19, y-11), (x-19, y+11)], fill='#47677F')
    draw.text((20, 15), 'Initial approach: two replies generated independently', font=bold, fill='#162A3C')
    box((30, 105, 510, 225), 'Conversation +\nincoming message')
    box((1110, 85, 1740, 165), 'English reply')
    box((1110, 190, 1740, 270), 'Singlish reply')
    arrow([(510,165), (800,165), (800,125), (1100,125)])
    arrow([(800,165), (800,230), (1100,230)])
    draw.text((20, 315), 'Revised approach: rewrite the generated English draft', font=bold, fill='#162A3C')
    box((30, 390, 510, 510), 'Conversation +\nincoming message')
    box((660, 390, 1140, 510), 'English draft')
    box((1290, 390, 1770, 510), 'Singlish rewrite', '#EAF3EA')
    arrow([(510,450), (650,450)])
    arrow([(1140,450), (1280,450)])
    draw.text((1298, 530), 'Goal: preserve meaning', font=small, fill='#355D35')
    path = OUT / 'english_first_pipeline.png'
    im.save(path)
    return path


def record_review(raw, previous):
    findings = [
        ('High', 'Comparison boundaries', 'The report alternates between early mT5, later SEA-LION/Qwen teachers and the deployed V5 model without identifying them consistently.', 'Are the SEA-LION results in Sections 3.3 and 4 from the same checkpoint and the same conversations?', 'Distinguish the historical V5 configuration (Section 3.5) from later 768-case teachers; distinguish 50-case development studies from the 60-case system evaluation.', 'Clarified in the larger-model and distillation sections; no peer results were edited.'),
        ('High', 'Attribution of the early improvement', 'The 20.3 to 46.0 meaning-preservation comparison changed both target alignment and generation order.', 'Did English-first rewriting alone cause the gain?', 'Name both changes in the prose and table headers; do not present this as an isolated pipeline ablation.', 'Table 3.3b now identifies both conditions explicitly.'),
        ('High', 'Changing dataset sizes', 'The sequence 190, 790, 768, 766 and 571 appears without a complete explanation of what was removed or counted.', 'Did the dataset change, and are conversations being confused with reply targets?', 'Explain 190+600=790, 22 source exclusions, two extra rewrite exclusions and the common reviewed teacher subset of 571.', 'Added a compact provenance chain; checked against the frozen recipes.'),
        ('High', 'Evaluation definitions', 'Contextual appropriateness, adequacy, recognizable Singlish and willingness to send are used without short operational definitions.', 'What would count as a pass or a high score?', 'Define relevance/constraints, paired meaning and local syntax; explain 1–5 judge scales and binary would-send ratings.', 'Definitions added; uncertainty is explicitly not a pass in the exploratory counts.'),
        ('High', 'Generator choice', 'The conclusion gives an engineering preference but does not connect it clearly to the existing system evaluation.', 'Why retain SEA-LION when Qwen also appears coherent and mT5 is smaller?', 'Connect the historical V5 choice to Section 4 while identifying later teacher/distillation work as follow-up research.', 'The revised ending retains SEA-LION and separates generator Qwen from the Qwen-based reranker.'),
        ('High', 'Dataset audit interpretation', '“The issue was therefore uneven style supervision” sounds more causal than the descriptive audit supports.', 'Has the dataset been shown to cause weak Singlish?', 'Use possible-contributor wording; distinguish target variety, inference copying and inherited teacher copies. Preserve the warning that particles and copy counts are not quality scores.', 'Causal wording softened and audit evidence retained with clear scope.'),
        ('Medium', 'Stage names and unexplained terms', 'Two main stages are followed by Stage 1a, Stage 1b and an implementation Stage 0; span corruption is unexplained.', 'How many stages are there, and what does each one learn?', 'Keep two conceptual stages with two public-data phases; remove internal numbering and briefly explain masked-span reconstruction.', 'Simplified stage explanation; corpus roles made explicit.'),
        ('Medium', 'Transfer-learning rationale', 'The rationale is useful but repeats the initial training description and ends with a lengthy defensive qualification.', 'What extra learning does public adaptation provide before the small project set?', 'Condense into the progression multilingual pretraining → informal-language/reply adaptation → project-specific task; retain one sentence that necessity was not isolated experimentally.', 'Condensed without claiming proven failure when omitted.'),
        ('Medium', 'Repeated pipeline explanations', 'Independent versus English-first generation is explained at the end of 3.3.1, again at the start of 3.3.2 and again around the figure.', 'What information is new here?', 'Explain the change once and show the process visually.', 'Removed repeated transitions and replaced the placeholder with a real pipeline diagram.'),
        ('Medium', 'Repeated backbone summary', 'Table 3.3d repeats the larger-model paragraphs without adding measured evidence or concrete outputs.', 'What does the table show that the prose does not?', 'Replace it with saved replies on one shared conversation, labelled as illustrative.', 'Added exact mT5/SEA-LION/Qwen replies for the sold-out-ticket case.'),
        ('Medium', 'Experiment-table density', 'Six long design descriptions restate the expected purpose, followed by verbose descriptions of mixed outcomes.', 'Which change mattered, what happened, and why was it insufficient?', 'Keep every reviewed experimental family, but compress each row to the intervention and its decision-relevant result.', 'All six families, representative numbers and failure examples retained in shorter rows.'),
        ('Medium', 'Uncontextualised 13/40 public-pair audit', 'The CoSEM audit count is given without sampling or rating details and can be read as a corpus-wide noise rate.', 'Does this estimate the quality of the public corpus used here?', 'Remove this optional count from the main narrative; retain the stronger explanation that source labels do not give parallel style equivalents.', 'Removed the unsupported prevalence implication rather than inventing missing audit details.'),
        ('Medium', 'Decoding detail', 'The paragraph teaches generic greedy/beam-search mechanics and repeats the persistence of role/timing errors.', 'Which decoding rule was actually adopted and what did it improve?', 'Keep four beams, repeated-three-gram blocking, token limits and the bounded qualitative reason for adoption.', 'Shortened while retaining the adopted decoding settings.'),
        ('Medium', 'Distillation terminology', '“Aligned soft distillation”, “different vocabularies”, “projection/alignment components” and “KL-only ablation” appear in quick succession.', 'How does the soft method differ from learning teacher-written replies?', 'Explain text targets versus teacher probability information, why learned alignment is needed and that KD adds both an objective and alignment modules.', 'Simplified the explanation; preserved the confound.'),
        ('Medium', 'Judge implementation bookkeeping', 'The paragraph spends space on 3,390 requests, deduplication and successful execution before explaining the metrics.', 'Does the number of API calls establish stronger evidence?', 'Keep matched inputs, blinded identities, persona correlation and averaging; leave operational request counts in the results artifacts.', 'Removed call-level bookkeeping from the school report; 5,000 rating cells and all scores remain.'),
        ('Medium', 'Repeated qualifications', '“Not independent human ratings”, “reused development” and “not a conclusive replacement” recur throughout neighboring paragraphs.', 'Can the limitation be stated once without weakening the conclusion?', 'Use one shared note for early exploratory counts and one distinct final paragraph for the automated study.', 'Consolidated qualifications while retaining their different meanings.'),
        ('Medium', 'Naturalness and style ceilings', 'Near-5 naturalness scores and uniformly perfect English style scores can look like uniformly excellent replies.', 'How can a model score highly but still reverse roles?', 'Explain that fluency is distinct from contextual correctness; a saturated style metric cannot distinguish systems.', 'Added this interpretation to the evaluation limits.'),
        ('Medium', 'Teacher–student gap', 'One printer example illustrates lost intent, but the text could be mistaken for a numerical claim that teachers outperformed students.', 'Were the teachers scored in the same run?', 'Explicitly state they were not; retain the example as an illustration, not an aggregate ranking.', 'Clarified; a matched teacher evaluation remains a research requirement.'),
        ('Low', 'Figure placeholders', 'Two placeholders remain in an otherwise completed section.', 'Are these figures still missing?', 'Supply the explanatory pipeline figure and use a concrete output table instead of the redundant second placeholder.', 'One embedded diagram and one real output table replace both placeholders.'),
        ('Low', 'Citation and methods boundaries', 'CoSEM, DailyDialog, NUS SMS, LoRA and multiple model names arrive quickly; full hyperparameters compete with explanation.', 'Which details are needed to understand this section?', 'Explain corpus/model roles briefly, retain the requested training table and distillation settings, and use Section 3.5 for the historical deployment configuration.', 'Scope clarified without changing peers’ methods or bibliography.'),
    ]
    review = {
        'review_type': 'Fresh-reader editorial simulation by the same assistant; not an independent blinded reviewer, a human grade or a new model evaluation.',
        'source_document_sha256': sha(raw),
        'before_word_count': len(' '.join(b.get('text', b.get('caption', '')) + ' ' + ' '.join(' '.join(r) for r in b.get('rows', [])) + ' ' + ' '.join(b.get('headers', [])) for b in previous).split()),
        'overall_verdict': 'The section has a coherent research story and useful evidence. Its main editorial weaknesses are unclear experiment boundaries, duplicated explanations and dense tables. Several claims also need tighter separation between observed results and plausible explanations.',
        'strengths': [
            'Clear progression from mT5 adaptation to meaning-preserving rewriting, larger teachers and compression.',
            'Concrete datasets, settings, values and failure examples make the work substantive.',
            'Appropriateness, local style and paired meaning are treated as separate objectives.',
            'The latest distillation results use matched inputs and metrics and report uncertainty.',
            'The account is candid about persistent errors and does not claim that every experiment succeeded.'
        ],
        'reader_questions_to_resolve': [f[3] for f in findings[:8]],
        'findings': [dict(priority=p, topic=t, issue=i, reader_question=q, recommended_change=c, action_taken=a) for p,t,i,q,c,a in findings],
        'research_gaps_that_editing_cannot_resolve': [
            'No clean ablation removes only the original public-adaptation stage, so its necessity and isolated effect remain unmeasured.',
            'The two teachers have not received the same matched judge scoring as the five students; a numerical teacher–student quality gap cannot be reported.',
            'The dataset audit is descriptive; it does not establish the causal contribution of synthetic proportion, target style, decoding or regional pretraining.',
            'The current judge schema does not measure English–Singlish consistency jointly, and no new blinded human local-style study has been run.',
            'The repeated 50-conversation validation results do not establish untouched test performance or broad statistical reliability.',
            'Varying teacher models and adding KD alignment components do not isolate regional pretraining or a single distillation loss term.'
        ],
        'scope': 'Only section3.3 revised; other authors’ prose, reported metrics and references preserved.'
    }
    OUT.mkdir(exist_ok=True)
    (OUT / 'full_editorial_review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n')
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = sec.bottom_margin = Inches(.7)
    sec.left_margin = sec.right_margin = Inches(.8)
    doc.styles['Normal'].font.name = 'Calibri'
    doc.styles['Normal'].font.size = Pt(11)
    doc.add_heading('Section 3.3: editorial review and revisions', 0)
    doc.add_paragraph(review['review_type'])
    doc.add_heading('Overall assessment', 1)
    doc.add_paragraph(review['overall_verdict'])
    doc.add_heading('What already works', 1)
    for s in review['strengths']:
        doc.add_paragraph(s, style='List Bullet')
    doc.add_heading('Detailed findings and actions', 1)
    for i, f in enumerate(review['findings'], 1):
        doc.add_heading(f"{i}. {f['topic']} ({f['priority'].lower()} priority)", 2)
        for label, key in [('Issue', 'issue'), ('Reader’s question', 'reader_question'), ('Recommended change', 'recommended_change'), ('Revision', 'action_taken')]:
            p = doc.add_paragraph()
            p.add_run(label + ': ').bold = True
            p.add_run(f[key])
    doc.add_heading('Research gaps requiring further evidence', 1)
    for s in review['research_gaps_that_editing_cannot_resolve']:
        doc.add_paragraph(s, style='List Bullet')
    doc.add_paragraph('The review recommendations above are editorial judgments. No missing experiment, human assessment or teacher score has been fabricated to close a research gap.')
    doc.save(OUT / 'Section 3.3 - Editorial Review.docx')
    return review


def revised_blocks():
    blocks = []
    def p(t): blocks.append({'type': 'paragraph', 'text': t})
    def h(t): blocks.append({'type': 'heading', 'text': t})
    def label(t): blocks.append({'type': 'label', 'text': t})
    def note(t): blocks.append({'type': 'note', 'text': t})
    def table(c, headers, rows): blocks.append({'type': 'table', 'caption': c, 'headers': headers, 'rows': rows})
    p('Following the RNN experiments, we explored pretrained Transformers to improve contextual reply generation (Vaswani et al., 2017). The goal was to produce a relevant Standard English reply and a natural Singlish counterpart with the same meaning. We began with mT5-base, an approximately 582-million-parameter multilingual encoder–decoder model (Xue et al., 2021).')
    h('3.3.1 Initial Training Plan')
    p('Training had two main stages. First, full-model adaptation used informal text and public reply pairs. Reconstructing masked text spans exposed mT5 to local language, while next-reply training taught conversational response generation. Second, one shared LoRA adapter learned from 190 project conversations, with STANDARD and LOCAL input prefixes selecting the reply style (Hu et al., 2021).')
    p('This transfer-learning strategy used broader data before specialising on the small project set: CoSEM supplied Singapore English messages, NUS SMS supplied informal text, and DailyDialog supplied English dialogues. Without intermediate adaptation, more task and style learning would depend on the 190 conversations, with a greater risk of memorisation. This was the rationale; we did not measure the isolated effect of removing this stage.')
    table('Table 3.3a. Initial mT5 training phases, datasets and settings.', ['Training phase', 'Dataset used', 'Main settings'], [
        ['Stage 1a: Language adaptation', 'CoSEM and NUS SMS: 96,295 text chunks for masked-span reconstruction.', 'Full-model training; 2 epochs; learning rate 3 × 10⁻⁴; batch 32.'],
        ['Stage 1b: Conversation adaptation', '76,052 DailyDialog reply pairs tagged STANDARD; 80,000 CoSEM pairs tagged LOCAL.', 'Full-model training; 5 epochs; learning rate 5 × 10⁻⁴; batch 32.'],
        ['Stage 2: Project adaptation', '190 team-authored conversations: 380 English/Singlish targets plus 180 history-derived reply examples.', 'Shared LoRA rank 16; learning rate 10⁻³; batch 16; early stopping.']
    ])
    p('We used 50 validation conversations throughout development. Checkpoints were selected by validation loss within each setup; generated replies were also inspected for contextual appropriateness, paired meaning and local style. Appropriateness required answering the latest message without contradicting roles or constraints. Meaning preservation required consistent decisions, participants and timing; local style required natural Singaporean phrasing rather than fluency alone.')
    p('The initial model often missed questions, reversed speaker roles or made different decisions in the two styles. The small project set limited coverage, while labelling entire public corpora STANDARD or LOCAL did not teach equivalent expressions of the same response.')
    h('3.3.2 Adaptations')
    p('We aligned the paired targets using assistant-reviewed rewrites and changed to an English-first pipeline: generate the English reply, then rewrite that exact draft into Singlish using the same adapter with a different task prefix. The comparison below evaluates these changes together, rather than isolating generation order.')
    blocks.append({'type': 'figure', 'caption': 'Figure 3.3a. Independent generation and English-first rewriting.', 'image': 'english_first_pipeline.png'})
    table('Table 3.3b. Early development comparison: mean pass counts out of 50.', ['Criterion', 'Original targets; independent replies', 'Aligned targets; English-first rewriting'], [
        ['Contextually appropriate reply', '33.0', '33.3'],
        ['English–Singlish meaning preserved', '20.3', '46.0'],
        ['Natural, recognisable Singlish', '23.7', '36.7']
    ])
    note('Tables 3.3b–c report exploratory assistant reviews, not independent human ratings. Table 3.3b averages three training seeds on the same 50 conversations; uncertain judgments were not passes. Table 3.3c experiments used their own comparison setups.')
    p('Paired meaning improved, but contextual errors remained. For “can I borrow your charger?”, one model generated English “Sure, I’ll borrow it” and Singlish “Can, I borrow my charger”. Rewriting preserved the incorrect borrowing direction. We therefore investigated the content generator as well as the style conversion.')
    table('Table 3.3c. Further mT5 experiments and observed limitations.', ['Experiment', 'Change tested and purpose', 'Result and limitation'], [
        ['Training-data expansion', 'Added 600 synthetic conversations to the original 190; matched optimizer updates between control and expanded models.', 'English adequacy stayed at 36/50; recognisable Singlish fell from 38/50 to 25/50. Exact English–Singlish copies rose from 3 to 19.'],
        ['Full fine-tuning versus LoRA', 'Used 24 English examples and paraphrased scenarios to test whether restricted adaptation limited learning.', 'Full tuning reproduced 24/24 training targets, but handled only 18/24 paraphrases, versus 15/24 for the stronger tested LoRA setting.'],
        ['Targeted data and broader LoRA', 'Added 80 role/timing/refusal examples and extended LoRA to more attention and feed-forward components.', 'Targeted cases improved, but mean English adequacy was 36.7/50 versus 37.7/50 for the control. The combined changes did not transfer consistently.'],
        ['Alternative rewriting setups', 'Tested separate adapters and rewrite inputs containing the draft alone, conversation history or a style example.', 'Copying and meaning changes persisted; one full-context rewrite changed “Friday afternoon works for me” into “Friday afternoon cannot”.'],
        ['Preference learning', 'Added 73 reviewed preference pairs through Direct Preference Optimisation alongside supervised training.', 'Changes were mixed across three seeds; one acceptance of a 2 pm meeting became an unexplained 3 pm arrival.'],
        ['Reply acts and factual supervision', 'Supplied acceptance/refusal information, auxiliary fact extraction or explicit context notes before generation.', 'Some cases improved, but supplied constraints could still be ignored: a refusal-conditioned model accepted an $18 lunch despite an $8 budget.']
    ])
    label('Decoding strategy')
    p('We adopted four-beam search with repeated three-grams blocked because it produced cleaner, more complete replies than greedy decoding in reviewed examples. The mT5 pipeline allowed 48 generated tokens and the later SEA-LION/Qwen pipelines 64. These decoding choices did not resolve the remaining role and timing errors.')
    h('3.3.3 Larger Models')
    p('We next explored SEA-LION-v1-3B, a larger decoder-only model with Southeast Asian language exposure (AI Singapore, n.d.). Public adaptation used 117,634 reply pairs: 53,691 CoSEM and 63,943 DailyDialog examples. Project training initially used 790 conversations (190 original plus 600 synthetic), followed by separate English-generation and Singlish-rewrite adapters.')
    p('Later experiments excluded 22 source cases with unresolved concerns, leaving 768 task conversations. Two further pairs were excluded from rewriting, leaving 766. Qwen2.5-3B used the same audited data, English-first design and full public-data pass (Qwen Team, n.d.). These were follow-up generator experiments; Qwen’s reranking role in Section 3.4 is a separate component.')
    p('Both larger models produced useful contextual replies, although neither eliminated factual or style errors. Table 3.3d illustrates a shared validation case; it is an example rather than an aggregate ranking.')
    sample = json.loads((ROOT / 'artifacts/mt5_distill_judge_v1/teacher_comparison/same_random10_teacher_student.json').read_text())['samples'][0]
    assert sample['item_id'] == 'ITEM-D-046'
    originals = json.loads((ROOT / 'artifacts/mt5_distill_judge_v1/sample_reviews/random10_20261121T121638Z.json').read_text())['samples'][0]
    table('Table 3.3d. Replies to “the cinema ticket is sold out bruh” after “eh help” / “what”.', ['Model configuration', 'English reply', 'Singlish reply'], [
        ['mT5 A baseline', originals['unchanged_english'], originals['unchanged_singlish']],
        ['Later SEA-LION teacher', sample['outputs']['sealion_teacher']['english'], sample['outputs']['sealion_teacher']['singlish']],
        ['Later Qwen teacher', sample['outputs']['qwen_teacher']['english'], sample['outputs']['qwen_teacher']['singlish']]
    ])
    label('Dataset audit and local style')
    p('To investigate weak Singlish differentiation, we inspected the actual training files and a stratified sample of 40 rewrite pairs (20 original and 20 synthetic). Some targets used clear local syntax, such as “i check got notes anot then update you”; others resembled generic casual English. The audit suggests possible limitations in style variety and inference control, rather than establishing one cause.')
    table('Table 3.3e. Descriptive evidence about Singlish differentiation.', ['Evidence', 'Finding', 'Interpretation'], [
        ['Reference targets', '591/766 rewrite pairs (77.2%) were synthetic. No reference pairs were identical after ignoring case and punctuation.', 'Synthetic targets dominated; their native-style variety was not independently validated. Duplicate labels cannot fully explain output copying.'],
        ['Teacher outputs: same 50 validation cases', 'SEA-LION copied English in 7/50 cases; Qwen in 19/50, ignoring case and punctuation.', 'Learned style differences were not reliably applied at inference. Some short replies are nevertheless valid in both styles.'],
        ['Distillation targets: common 571 cases', 'Normalized English/Singlish copies: SEA-LION 45/571 (7.9%); Qwen 40/571 (7.0%).', 'Students also receive some weakly differentiated teacher targets.']
    ])
    p('Singlish can be expressed through syntax and context-appropriate ellipsis; particles are neither compulsory nor sufficient. Public reply training also did not provide parallel style equivalents for every response. More varied, meaning-aligned targets are a plausible next step. Regional exposure cannot be isolated here: the validation inputs were mainly English-based, with 30/50 listing local terms and only 2/50 carrying a pinyin tag.')
    label('Generator choice')
    p('The historical V5 SEA-LION configuration remained the deployed generator; its settings are in Section 3.5 and its separate 60-conversation system comparison is in Section 4. The later 768-case SEA-LION/Qwen configurations served as distillation teachers. Their qualitative observations and the 50-case student results below should not be read as scores from the deployed-system evaluation.')
    h('3.3.4 Distillation')
    p('We tested whether the larger teachers could improve mT5 A: the existing model with a backbone already adapted on local text and public dialogue, plus its trained shared task adapter. All four training arms continued that same checkpoint. Earlier fresh-adapter experiments are outside this comparison.')
    p('From the 768 training conversations, we retained the common 571 cases with eligible original pairs and accepted English/Singlish outputs from both teachers after assistant review. For each teacher, one arm learned its reply text through cross-entropy; another added soft knowledge distillation (KD), using teacher probability information as well as text. Different vocabularies required learned alignment modules, so KD changed both the objective and training components.')
    p('Each arm completed 12 epochs, continuing a rank-16 query/value LoRA adapter for both tasks while freezing the 582-million-parameter backbone. The learning rate was 3 × 10⁻⁵ with 5% warmup and cosine decay. Original-reference validation loss selected checkpoints; inference retained English-first rewriting and the mT5 decoding settings above.')
    label('Matched judge evaluation')
    p('The unchanged model and four selected variants received the same 50 validation conversations in both styles. Gemma 4 12B judged replies with model identities hidden, using the group’s rubric and ten simulated personas (temperature 0, seed 425). Appropriateness, naturalness and style fit used 1–5 scales; “would send” is the percentage judged sendable with minimal edits. There were 5,000 rating cells; we averaged personas within each conversation, then across conversations.')
    scores = json.loads((ROOT / 'experiments/mt5_distill_judge_v1_results.json').read_text())
    lookup = {(r['model'], r['style']): r for r in scores['summary']}
    rows = []
    for arm, name in [('unchanged', 'Unchanged mT5 A'), ('sealion_ce', 'SEA-LION text'), ('sealion_kd', 'SEA-LION text +KD'), ('qwen_ce', 'Qwen text'), ('qwen_kd', 'Qwen text +KD')]:
        en, sg = lookup[arm, 'STANDARD'], lookup[arm, 'LOCAL']
        rows.append([name, *[f'{en[k]:.2f} / {sg[k]:.2f}' for k in ['appropriateness', 'naturalness', 'style_fit']], f"{en['would_send_pct']:.1f} / {sg['would_send_pct']:.1f}"])
    table('Table 3.3f. Matched judge scores for adapted mT5 (English / Singlish).', ['mT5 variant', 'Appropriateness /5', 'Naturalness /5', 'Style fit /5', 'Would send (%)'], rows)
    note('Text = teacher-text cross-entropy; +KD = text plus aligned soft distillation. Selected epochs: SEA-LION text 3; SEA-LION +KD 1; Qwen text 2; Qwen +KD 1. Ten simulated personas are correlated judgments from one model, not independent human participants.')
    p('SEA-LION text had the highest observed appropriateness and would-send means. Its paired appropriateness gains over unchanged mT5 were 0.25 for English and 0.12 for Singlish, but conversation-level 95% bootstrap intervals included zero (−0.13 to 0.66; −0.37 to 0.59). Singlish style fit fell from 4.16 to 3.90. Soft KD showed no consistent benefit; the text-trained variant remains a candidate for further study, rather than an established replacement.')
    p('Teacher behaviour was not fully transferred. For a printer-booking change, Qwen replied “Sure, go ahead and change your booking”, while its text-trained student replied “sure, i’ll try to use it later”, changing whose action was discussed. The teachers were not judged in this run, so this illustrates a gap without measuring its average size.')
    p('These repeatedly inspected validation cases do not establish untouched test performance. High naturalness can coexist with contextual errors, and English style-fit scores saturated at 5/5. The judge also assessed drafts separately: a KD model could say the quiz was finished in English and unfinished in Singlish without both replies being marked unsuitable. Joint meaning checks and blinded human style assessment remain necessary; the evidence does not yet justify replacing the deployed SEA-LION generator with a distilled student.')
    return blocks


def main():
    OUT.mkdir(exist_ok=True)
    raw = SOURCE.read_bytes()
    previous = json.loads((BASE / 'distillation_update/updated_section33_content.json').read_text())
    record_review(raw, previous)
    image_path = make_figure()
    blocks = revised_blocks()
    helper_spec = importlib.util.spec_from_file_location('previous_editor', BASE / 'update_distillation_report.py')
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    with zipfile.ZipFile(SOURCE) as z:
        infos = z.infolist()
        parts = {i.filename: z.read(i.filename) for i in infos}
    root = E.fromstring(parts['word/document.xml'])
    body = root.find('w:body', NS)
    nodes = list(body)
    start = next(i for i, n in enumerate(nodes) if text(n).strip() == '3.3 Transformer reply generation (V5)')
    end = next(i for i, n in enumerate(nodes) if text(n).strip().startswith('3.4 Reranking,'))
    before, after = [canon(n) for n in nodes[:start+1]], [canon(n) for n in nodes[end:]]
    rels = E.fromstring(parts['word/_rels/document.xml.rels'])
    old_rels = [canon(r) for r in rels]
    rid = 'rIdSection33Pipeline'
    assert all(r.get('Id') != rid for r in rels)
    E.SubElement(rels, '{'+PR+'}Relationship', Id=rid, Type=R+'/image', Target='media/section33_english_first.png')
    drawing_id = max([int(n.get('id')) for n in root.iter() if n.tag.endswith('}docPr')], default=0) + 1
    drawing = Document()
    run = drawing.add_paragraph().add_run()
    run.add_picture(str(image_path), width=Inches(6.5))
    image_paragraph = E.fromstring(drawing.paragraphs[0]._p.xml.encode())
    image_paragraph.xpath('.//a:blip', namespaces={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'})[0].set('{'+R+'}embed', rid)
    docpr = image_paragraph.xpath('.//wp:docPr', namespaces={'wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'})[0]
    docpr.set('id', str(drawing_id))
    docpr.set('name', 'English-first reply generation')
    docpr.set('descr', 'Independent generation of two styles compared with English generation followed by a Singlish rewrite of that draft.')
    pp = image_paragraph.find('w:pPr', NS)
    if pp is None:
        pp = E.Element('{'+W+'}pPr')
        image_paragraph.insert(0, pp)
    E.SubElement(pp, '{'+W+'}keepNext')
    new = []
    for b in blocks:
        kind = b['type']
        if kind == 'table':
            new.extend([helper.paragraph(b['caption'], kind='caption', keep=True, size=20), helper.table(b), helper.paragraph('', size=20)])
        elif kind == 'figure':
            new.extend([image_paragraph, helper.paragraph(b['caption'], kind='caption', size=19)])
        elif kind == 'label':
            new.append(helper.paragraph(b['text'], keep=True, bold=True))
        elif kind == 'note':
            new.append(helper.paragraph(b['text'], kind='caption', size=19))
        else:
            new.append(helper.paragraph(b['text'], kind=kind))
    for n in nodes[start+1:end]:
        body.remove(n)
    for i, n in enumerate(new, start+1):
        body.insert(i, n)
    assert [canon(n) for n in list(body)[:start+1]] == before
    assert [canon(n) for n in list(body)[start+1+len(new):]] == after
    assert [canon(r) for r in list(rels)[:-1]] == old_rels
    assert sum(b['type'] == 'table' for b in blocks) == 6
    assert not any('placeholder' in text(n).lower() for n in new)
    old_words = len(' '.join(text(n) for n in nodes[start+1:end]).split())
    new_words = len(' '.join(text(n) for n in new).split())
    assert 1700 <= new_words < old_words, (old_words, new_words)
    assert 'Extension="png"' in parts['[Content_Types].xml'].decode()
    if BACKUP.exists():
        assert BACKUP.read_bytes() == raw, 'Preserve the earlier backup; avoid duplicate revision.'
    else:
        shutil.copy2(SOURCE, BACKUP)
    assert SOURCE.read_bytes() == raw, 'Source changed during revision.'
    temp = SOURCE.with_name('.Copy of Final Report.editorial-review.tmp.docx')
    replacements = {'word/document.xml': E.tostring(root, encoding='UTF-8', xml_declaration=True, standalone=True),
                    'word/_rels/document.xml.rels': E.tostring(rels, encoding='UTF-8', xml_declaration=True, standalone=True)}
    media = 'word/media/section33_english_first.png'
    assert media not in parts
    with zipfile.ZipFile(temp, 'w') as z:
        for i in infos:
            z.writestr(i, replacements.get(i.filename, parts[i.filename]))
        z.writestr(media, image_path.read_bytes())
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None
        assert all(z.read(name) == value for name, value in parts.items() if name not in replacements)
        assert z.read(media) == image_path.read_bytes()
    os.replace(temp, SOURCE)
    (OUT / 'revised_section33_content.json').write_text(json.dumps(blocks, ensure_ascii=False, indent=2)+'\n')
    receipt = {'document':str(SOURCE), 'backup':str(BACKUP), 'review_document':str(OUT/'Section 3.3 - Editorial Review.docx'),
               'source_sha256':sha(raw), 'revised_sha256':sha(SOURCE.read_bytes()), 'before_words':old_words, 'after_words':new_words,
               'reduction_percent':round(100*(old_words-new_words)/old_words,1), 'native_tables':6, 'embedded_figures':1, 'remaining_placeholders':0,
               'outside_section_xml_unchanged':True, 'existing_relationships_preserved':True,
               'existing_package_parts_changed':['word/document.xml','word/_rels/document.xml.rels'], 'added_package_parts':[media],
               'model_training_or_evaluation_run':False, 'datasets_changed':False,
               'review_is_independent_blind_grade':False}
    (OUT/'revision_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
