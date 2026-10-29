"""Pinned DailyDialog turn preparation, with no remote code execution or AI labels."""
from hashlib import sha256
import json
from pathlib import Path
import random
import re
import unicodedata
import urllib.request
import zipfile

from replycoach.reranker_bundle import pair_payload

DATASET = 'roskoN/dailydialog'
DATASET_REVISION = '5214b2a66405abf87fd229e5c1007985501ffe3e'
ARCHIVE_URL = f'https://huggingface.co/datasets/{DATASET}/resolve/{DATASET_REVISION}/train.zip'
ARCHIVE_SHA256 = 'c5179ab5a9a86a77b9d29114087c6b82cc4cb366abea25a43a0c24be761133f7'


def normalized(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    return ' '.join(re.findall(r'\w+', text))


def download_archive(path, *, fetch=None):
    """Write only a verified archive. The mirror's Python loader is never executed."""
    path = Path(path)
    if path.exists():
        raw = path.read_bytes()
    elif fetch is not None:
        raw = fetch(ARCHIVE_URL)
    else:
        with urllib.request.urlopen(ARCHIVE_URL, timeout=120) as response:
            raw = response.read(10_000_001)
    if sha256(raw).hexdigest() != ARCHIVE_SHA256:
        raise ValueError('DailyDialog archive hash mismatch; do not train on changed data')
    if not path.exists():
        path.write_bytes(raw)
    return path


def read_dialogues(path):
    # Read exactly one text member; no extractall/path traversal or dataset scripts.
    with zipfile.ZipFile(path) as archive:
        raw = archive.read('train/dialogues_train.txt').decode('utf-8-sig')
    dialogues=[]
    for line in raw.splitlines():
        if not line.strip():
            continue
        turns=[turn.strip() for turn in line.split('__eou__')]
        if turns[-1]=='':
            turns.pop()  # the final delimiter is not an utterance
        if any(not turn for turn in turns):
            raise ValueError('empty internal dialogue turn would change speaker parity')
        dialogues.append(turns)
    return dialogues


def query_for_turn(turns, target, *, max_context=6):
    """Assume alternating source utterance turns, relative to the candidate You."""
    if not 1 <= target < len(turns):
        raise ValueError('candidate must follow an incoming dialogue turn')
    start = max(0, target - 1 - max_context)
    memory = [f'{"You" if (target-i)%2 == 0 else "Peer"}: {turns[i]}'
              for i in range(start, target-1)]
    return '\n'.join(['Conversation (oldest first):', *memory,
                      f'Incoming peer message: {turns[target-1]}',
                      'Requested reply style: STANDARD', 'User reply:'])


def project_texts(bundle):
    """Use only text already in the permitted train/validation upload bundle."""
    texts = set()
    prefixes = ['Peer: ', 'You: ', 'Other: ', 'Unknown: ', 'Incoming peer message: ']
    for split in ['train', 'validation']:
        for row in bundle[split]:
            texts.add(row['reply'])
            for line in row['query'].splitlines():
                for prefix in prefixes:
                    if line.startswith(prefix):
                        texts.add(line[len(prefix):])
                        break
    return texts


def prepare_pairs(dialogues, *, excluded_texts=(), validation_fraction=.1,
                  max_train_queries=4000, max_validation_queries=400, negatives=2, seed=425):
    if (not 0 < validation_fraction < .5 or type(max_train_queries) is not int
            or max_train_queries < 1 or type(max_validation_queries) is not int
            or max_validation_queries < 1 or type(negatives) is not int or not 1 <= negatives <= 4):
        raise ValueError('invalid DailyDialog sampling settings')
    excluded = {normalized(t) for t in excluded_texts}
    unique, duplicates, overlaps, unusable = {}, 0, 0, 0
    for turns in dialogues:
        if (not isinstance(turns, list) or len(turns) < 2
                or any(not isinstance(t, str) or not t.strip() for t in turns)):
            raise ValueError('invalid DailyDialog dialogue; need nonempty whole turns')
        if any(not normalized(t) for t in turns):
            unusable += 1
            continue  # drop the whole dialogue; deleting a turn would corrupt role parity
        key = sha256(json.dumps([normalized(t) for t in turns]).encode()).hexdigest()
        if key in unique:
            duplicates += 1
            continue
        # Record deduplication before overlap filtering; never keep partial dialogues.
        unique[key] = list(turns)
    for key, turns in list(unique.items()):
        if any(normalized(t) in excluded for t in turns):
            overlaps += 1
            del unique[key]
    keys = sorted(unique)
    if len(keys) < 4:
        raise ValueError('need at least four distinct nonoverlapping DailyDialog dialogues')
    random.Random(seed).shuffle(keys)
    count = max(1, int(len(keys)*validation_fraction))
    ids = {'validation': sorted(keys[:count]), 'train': sorted(keys[count:])}
    def examples(split, cap):
        candidates = [(key, i) for key in ids[split]
                      for i in range(1, len(unique[key]))
                      if 1 <= len(unique[key][i].split()) <= 80
                      and len(unique[key][i-1].split()) <= 100]
        random.Random(seed + (1 if split == 'train' else 2)).shuffle(candidates)
        return candidates[:cap]
    selected = {'train': examples('train', max_train_queries),
                'validation': examples('validation', max_validation_queries)}
    # Donors can use any retained training dialogue, never validation or test.
    donors = [(key, text) for key in ids['train'] for text in unique[key][1:]
              if 1 <= len(text.split()) <= 80]
    random.Random(seed+3).shuffle(donors)
    result = {}
    for split in ['train', 'validation']:
        rows = []
        for key, target in selected[split]:
            turns = unique[key]
            own = {normalized(t) for t in turns}
            base = {'item_id': 'dd-'+key, 'style': 'STANDARD',
                    'query': query_for_turn(turns, target)}
            candidate = dict(base, donor_id='dd-'+key, reply=turns[target], label=1,
                             provenance='dailydialog_observed_next_turn_weak')
            offset = int(sha256(f'{seed}:{key}:{target}'.encode()).hexdigest()[:8], 16) % len(donors) if donors else 0
            negatives_rows, seen = [], set()
            for j in range(len(donors)):
                donor, text = donors[(offset+j) % len(donors)]
                norm = normalized(text)
                if donor != key and norm not in own and norm not in seen:
                    negatives_rows.append(dict(base, donor_id='dd-'+donor, reply=text, label=0,
                                               provenance='dailydialog_mismatch_weak'))
                    seen.add(norm)
                    if len(negatives_rows) == negatives:
                        break
            if len(negatives_rows) == negatives:
                rows.extend([candidate, *negatives_rows])
        if not rows:
            raise ValueError('DailyDialog split has no usable query/negative groups')
        result[split] = rows
    result['manifest'] = {
        'record_type': 'dailydialog_reranker_adaptation', 'source': DATASET,
        'source_revision': DATASET_REVISION, 'archive_sha256': ARCHIVE_SHA256,
        'source_split': 'train_only', 'licence': 'CC-BY-NC-SA-4.0',
        'licence_url': 'https://creativecommons.org/licenses/by-nc-sa/4.0/',
        'citation': 'Li et al. (2017). DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset. IJCNLP.',
        'source_paper': 'https://aclanthology.org/I17-1099/',
        'changes': 'Normalized dialogue deduplication, project-text exclusion, dialogue split and weak reply-suitability pairs.',
        'speaker_policy': 'assumed_alternating_utterance_turns_not_verified_ids',
        'style_policy': 'STANDARD_only_no_invented_LOCAL_gold',
        'positive_policy': 'Observed next turns are weak positives, not reviewed reply-quality gold.',
        'negative_policy': 'Different-training-dialogue mismatches; possible false negatives.',
        'train_ids': ['dd-'+k for k in ids['train']],
        'validation_ids': ['dd-'+k for k in ids['validation']],
        'train_fingerprint': sha256(json.dumps(ids['train']).encode()).hexdigest(),
        'validation_fingerprint': sha256(json.dumps(ids['validation']).encode()).hexdigest(),
        'pair_sha256': {s: sha256(pair_payload(result[s])).hexdigest() for s in result},
        'duplicate_dialogues_removed': duplicates, 'project_overlap_dialogues_removed': overlaps,
        'unusable_dialogues_removed': unusable,
        'max_train_queries': max_train_queries, 'max_validation_queries': max_validation_queries,
        'negatives_per_query': negatives, 'seed': seed, 'max_context': 6,
        'pair_counts': {s: len(result[s]) for s in result},
    }
    return result


def run_pipeline(bundle_path, output, *, source=None, max_train_queries=4000,
                 max_validation_queries=400, adaptation_epochs=2, refinement_epochs=12,
                 smoke=False, model_factory=None, max_steps=None):
    """Run locally on Kaggle GPU; never upload, submit jobs or change demo defaults."""
    import gc
    import torch
    from replycoach.qwen_fit import fit_pairs, run_training
    from replycoach.qwen_reranker import MODEL
    from replycoach.reranker_bundle import load_training_bundle
    from replycoach.reranker_hard_pairs import apply_reviewed_pairs
    bundle_path, output = Path(bundle_path), Path(output)
    bundle=load_training_bundle(bundle_path)
    if (bundle_path/'reviewed_pairs.json').is_file():
        apply_reviewed_pairs(bundle,json.loads((bundle_path/'reviewed_pairs.json').read_text(encoding='utf-8')))
    if (output.exists() or bundle_path.resolve() in output.resolve().parents
            or output.resolve() in bundle_path.resolve().parents):
        raise ValueError('two-stage output must be fresh and outside the private input bundle')
    if not smoke and not torch.cuda.is_available():
        raise ValueError('two-stage Qwen training requires a GPU')
    if not (type(adaptation_epochs) is int and 1 <= adaptation_epochs <= 12
            and type(refinement_epochs) is int and 1 <= refinement_epochs <= 12):
        raise ValueError('each stage needs an epoch budget between 1 and 12')
    output.mkdir(parents=True)
    archive=download_archive(output/'dailydialog_train.zip')
    prepared=prepare_pairs(read_dialogues(archive),excluded_texts=project_texts(bundle),
                           max_train_queries=max_train_queries,max_validation_queries=max_validation_queries)
    prepared_path=output/'adaptation_pairs'
    prepared_path.mkdir()
    for split in ['train','validation']:
        (prepared_path/f'{split}_pairs.jsonl').write_bytes(pair_payload(prepared[split]))
    (prepared_path/'manifest.json').write_text(json.dumps(prepared['manifest'],indent=2),encoding='utf-8')
    print(json.dumps({'event':'dailydialog_prepared',**prepared['manifest']['pair_counts'],
                      'overlap_dialogues_removed':prepared['manifest']['project_overlap_dialogues_removed']}),flush=True)
    common=dict(source=source or MODEL,smoke=smoke,model_factory=model_factory,max_steps=max_steps)
    first=fit_pairs(prepared['train'],prepared['validation'],prepared['manifest'],
                    output/'stage1_dailydialog',input_path=prepared_path,
                    epochs=adaptation_epochs,early_stopping_patience=1,**common)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    second=run_training(bundle_path,output/'stage2_reviewed',
                        initial_adapter=output/'stage1_dailydialog/best',
                        epochs=refinement_epochs,early_stopping_patience=2,**common)
    result={'record_type':'dailydialog_then_reviewed_qwen','completed':True,
            'production_checkpoint':not smoke,'adaptation':first,'refinement':second,
            'final_adapter':'stage2_reviewed/best','held_out_project_test_loaded':False,
            'demo_promoted':False,'fresh_optimizer_per_stage':True}
    (output/'experiment.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
