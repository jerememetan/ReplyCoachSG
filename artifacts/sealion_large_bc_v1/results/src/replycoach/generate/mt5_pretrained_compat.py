"""Preserve original mT5's shared input embeddings and independent output head.

Transformers 5.19 forces MT5Config.tie_word_embeddings=True. Original Google
mT5 checkpoints serialize False, share encoder/decoder input embeddings, and
have a distinct output head. Keep that exact mapping without changing layers.
"""
from transformers import MT5Config, MT5ForConditionalGeneration


def original_config(raw):
    if raw.get('tie_word_embeddings') is not False:
        raise ValueError('Expected original untied mT5 output head')
    cfg = MT5Config(**raw)
    cfg.tie_word_embeddings = False
    return cfg


class OriginalMT5(MT5ForConditionalGeneration):
    def get_expanded_tied_weights_keys(self, all_submodels=False):
        if all_submodels:
            return super().get_expanded_tied_weights_keys(all_submodels=True)
        return {'encoder.embed_tokens.weight': 'shared.weight',
                'decoder.embed_tokens.weight': 'shared.weight'}


def verify_original_embeddings(model):
    shared = model.shared.weight
    if model.config.tie_word_embeddings:
        raise ValueError('Original mT5 output-head configuration changed')
    if model.encoder.embed_tokens.weight is not shared or model.decoder.embed_tokens.weight is not shared:
        raise ValueError('Original mT5 input embeddings must be shared')
    if model.lm_head.weight is shared:
        raise ValueError('Original mT5 output head must remain independent')


from transformers.tokenization_utils_sentencepiece import SentencePieceBackend


class OriginalMT5Tokenizer(SentencePieceBackend):
    vocab_files_names = {'vocab_file': 'spiece.model'}
    model_input_names = ['input_ids', 'attention_mask']

    def __init__(self, vocab_file, **kwargs):
        kwargs.setdefault('eos_token', '</s>')
        kwargs.setdefault('pad_token', '<pad>')
        kwargs.setdefault('unk_token', '<unk>')
        kwargs.setdefault('special_tokens_pattern', 'eos')
        kwargs.setdefault('legacy', True)
        kwargs.setdefault('clean_up_tokenization_spaces', False)
        super().__init__(vocab_file=vocab_file, **kwargs)

    def convert_tokens_to_string(self, tokens):
        return self.sp_model.decode(tokens)


def original_tokenizer(path):
    """Use original SentencePiece directly, including its tie breaks/byte fallback."""
    from pathlib import Path
    return OriginalMT5Tokenizer(vocab_file=str(Path(path) / 'spiece.model'))
