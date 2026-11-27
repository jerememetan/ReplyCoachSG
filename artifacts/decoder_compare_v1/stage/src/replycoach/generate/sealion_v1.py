"""Compatibility wrapper for the pinned SEA-LION v1 learned-position MPT.

Keeps upstream architecture, weights and forward/loss code intact. Transformers5
removed the default tie_word_embeddings setting and GenerationMixin inheritance
that the upstream Transformers4.34 implementation relies on.
"""
from transformers.generation import GenerationMixin
import json
from pathlib import Path
from replycoach.vendor.sealion_v1.configuration_mpt import MPTConfig
from replycoach.vendor.sealion_v1.modeling_mpt import MPTForCausalLM
from replycoach.vendor.sealion_v1.tokenization_SEA_BPE import SEABPETokenizer


class SEALionCausalLM(MPTForCausalLM,GenerationMixin):
    def __init__(self,config):
        super().__init__(config)
        # Upstream predates the mandatory Transformers5 loading bookkeeping.
        # This does not add layers or change the upstream forward/loss code.
        self.post_init()


def config_for_checkpoint(path,init_device='meta'):
    # from_pretrained only applies override kwargs for existing attributes in
    # Transformers5. The historical default is absent in this checkpoint JSON,
    # so instantiate from its dictionary with the default explicitly restored.
    values=json.loads((Path(path)/'config.json').read_text())
    config=MPTConfig(**{**values,'tie_word_embeddings':True,'init_device':init_device})
    if not config.learned_pos_emb or config.attn_config['alibi']:
        raise ValueError('Expected upstream learned-position, non-ALiBi SEA-LION v1 architecture')
    return config


def checkpoint_tokenizer(path):
    tok=SEABPETokenizer.from_pretrained(path,local_files_only=True)
    if tok.eos_token_id!=1:raise ValueError('Unexpected official EOS ID')
    tok.pad_token=tok.eos_token
    tok.padding_side='right'
    return tok


def checkpoint_loading_problems(info):
    # Transformers versions return some key collections as sets. Normalize at
    # the boundary so both strict checks and persisted reports use plain JSON.
    return {key:list(info.get(key,[])) for key in
            ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']}
