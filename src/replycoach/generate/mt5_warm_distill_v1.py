"""Continue a saved shared mT5 task adapter with balanced paired supervision."""
from pathlib import Path

import torch
from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file


def load_shared_adapter(base, path, trainable=True, expected_parameters=1769472):
    """Load trained tensors verbatim; no get_peft_model or adapter reinitialization."""
    path = Path(path)
    expected = load_file(str(path / 'adapter_model.safetensors'))
    model = PeftModel.from_pretrained(base, path, is_trainable=trainable)
    actual = get_peft_model_state_dict(model)
    if set(expected) != set(actual) or any(not torch.equal(v, actual[k].detach().cpu()) for k, v in expected.items()):
        raise ValueError('Loaded shared adapter differs from preserved source tensors')
    if list(model.peft_config) != ['default'] or list(model.active_adapters) != ['default']:
        raise ValueError('One shared default adapter is required for both tasks')
    total = sum(v.numel() for v in actual.values())
    if expected_parameters is not None and total != expected_parameters:
        raise ValueError('Unexpected shared LoRA parameter count')
    active = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if trainable:
        if not active or any('lora_' not in name for name, _ in active):
            raise ValueError('Only the saved LoRA may be trainable')
        if sum(p.numel() for _, p in active) != total:
            raise ValueError('All saved shared LoRA tensors must be trainable')
    elif active:
        raise ValueError('Unchanged baseline must be fully frozen')
    return model


def paired_update_groups(conversations, seed, epoch, pairs_per_update=4):
    """Each optimizer group has equal English/rewrite counts for the same IDs."""
    if conversations < 1 or pairs_per_update < 1:
        raise ValueError('Positive conversation and update sizes required')
    order = torch.randperm(conversations, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
    for start in range(0, conversations, pairs_per_update):
        yield [(stage, index) for index in order[start:start + pairs_per_update]
               for stage in ('english', 'singlish')]


def selection_score(scores):
    """Select one shared adapter on equal-weight reference CE across both tasks."""
    return .5 * (scores['english']['token_ce'] + scores['singlish']['token_ce'])
