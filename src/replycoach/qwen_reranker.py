"""Pinned Qwen causal yes/no scorer; no model-source signals or judge labels."""
from hashlib import sha256, file_digest
import json
from pathlib import Path

MODEL = 'Qwen/Qwen3-Reranker-0.6B'
REVISION = 'e61197ed45024b0ed8a2d74b80b4d909f1255473'
QUERY_FORMAT = 'speaker-conversation-style-v2'
INSTRUCTION = (
    'Judge whether the candidate is an appropriate next reply by You to the latest '
    'incoming peer message in the conversation. Prioritize answering that message '
    'and preserving speaker roles, facts, times and commitments. Respect the requested '
    'reply style: STANDARD is standard English; LOCAL is English with optional light '
    'Singaporean phrasing. Plain English is also acceptable for LOCAL. Do not prefer '
    'slang or Chinese words over a relevant answer.'
)
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based '
          'on the Query and the Instruct provided. Note that the answer can only be '
          '"yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
PROMPT_SHA256 = sha256((PREFIX + INSTRUCTION + SUFFIX).encode()).hexdigest()


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def encode_inputs(tokenizer, pairs, *, max_length=512):
    """Keep instruction and reply intact; remove only oldest query tokens."""
    prefix = tokenizer.encode(PREFIX + '<Instruct>: ' + INSTRUCTION + '\n<Query>: ', add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    rows, truncated = [], []
    for query, reply in pairs:
        if not query.strip() or not reply.strip():
            raise ValueError('query and reply must be nonempty')
        context = tokenizer.encode(query, add_special_tokens=False)
        document = tokenizer.encode('\n<Document>: ' + reply, add_special_tokens=False)
        budget = max_length-len(prefix)-len(document)-len(suffix)
        if budget < 1:
            raise ValueError('reply and instruction exceed the token budget')
        truncated.append(len(context) > budget)
        rows.append(prefix + context[-budget:] + document + suffix)
    return tokenizer.pad({'input_ids': rows}, padding=True, return_tensors='pt'), truncated


def binary_logits(logits, *, yes_id, no_id):
    return logits[:, yes_id].float() - logits[:, no_id].float()


def load_base(source, device='cpu'):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    local = Path(source).is_dir()
    options = {'local_files_only': True} if local else {'revision': REVISION}
    tokenizer = AutoTokenizer.from_pretrained(source, padding_side='left', trust_remote_code=False, **options)
    model = AutoModelForCausalLM.from_pretrained(
        source, trust_remote_code=False, use_safetensors=True,
        dtype=torch.float16 if str(device).startswith('cuda') else torch.float32, **options,
    ).to(device).eval()
    return model, tokenizer


def score_batch(model, tokenizer, pairs, *, max_length=512):
    encoded, flags = encode_inputs(tokenizer, pairs, max_length=max_length)
    encoded = {k:v.to(next(model.parameters()).device) for k,v in encoded.items()}
    logits = model(**encoded, logits_to_keep=1, use_cache=False).logits[:, -1, :]
    yes_id, no_id = tokenizer.convert_tokens_to_ids('yes'), tokenizer.convert_tokens_to_ids('no')
    if yes_id == no_id or yes_id == tokenizer.unk_token_id or no_id == tokenizer.unk_token_id:
        raise ValueError('tokenizer must have distinct yes/no tokens')
    return binary_logits(logits, yes_id=yes_id, no_id=no_id), flags


class QwenReranker:
    def __init__(self, source, *, adapter=None, device='cpu', batch_size=2):
        self.model, self.tokenizer = load_base(source, device)
        self.batch_size = batch_size
        self.identity = {'model': MODEL, 'revision': REVISION, 'query_format': QUERY_FORMAT,
                         'prompt_sha256': PROMPT_SHA256, 'max_length':512, 'device':device,
                         'fine_tuned': False, 'score_kind':'yes_minus_no_logit'}
        if adapter is not None:
            from peft import PeftModel
            path=Path(adapter)
            meta=json.loads((path/'qwen_training.json').read_text(encoding='utf-8'))
            if (meta.get('completed') is not True or meta.get('production_checkpoint') is not True
                    or meta.get('base_model') != MODEL or meta.get('base_revision') != REVISION
                    or meta.get('prompt_sha256') != PROMPT_SHA256
                    or not meta.get('optimizer_steps',0) or meta.get('weights_changed') is not True
                    or meta.get('adapter_sha256') != file_sha256(path/'adapter_model.safetensors')
                    or meta.get('base_weights_sha256') != file_sha256(Path(source)/'model.safetensors')):
                raise ValueError('Qwen adapter provenance or weights do not match')
            self.model = PeftModel.from_pretrained(self.model, path, is_trainable=False).eval()
            self.identity.update(fine_tuned=True, adapter_sha256=meta['adapter_sha256'],
                                 train_fingerprint=meta['train_fingerprint'],
                                 validation_fingerprint=meta['validation_fingerprint'])

    def score_pairs(self, pairs):
        import torch
        from replycoach.shared_ranker import PairScore
        values=[]
        with torch.inference_mode():
            for start in range(0,len(pairs),self.batch_size):
                scores, flags=score_batch(self.model,self.tokenizer,pairs[start:start+self.batch_size])
                values.extend(PairScore(float(value), flag) for value,flag in zip(scores.tolist(),flags))
        return values
