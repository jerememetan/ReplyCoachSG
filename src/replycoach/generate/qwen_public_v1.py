"""Exact optimizer-boundary resume for the public Qwen LoRA adaptation stage."""
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict


def resume_state(model, optimizer, report):
    return dict(adapter={k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()},
                optimizer=optimizer.state_dict(), report=report,
                cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_state(model, optimizer, state):
    current = get_peft_model_state_dict(model)
    saved = state['adapter']
    if set(current) != set(saved) or any(current[k].shape != saved[k].shape for k in current):
        raise ValueError('Resume adapter shape/key mismatch')
    set_peft_model_state_dict(model, saved)
    if any(not torch.equal(v.detach().cpu(), saved[k]) for k, v in get_peft_model_state_dict(model).items()):
        raise ValueError('Resume adapter tensor mismatch')
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['cpu_rng'])
    if state['cuda_rng']:
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['report']
