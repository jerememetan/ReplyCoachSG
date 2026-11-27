"""DSKD-inspired cross-model attention for an encoder-decoder student.

Align response positions in both native vocabularies; never compare token IDs
across tokenizers. Projectors are training-only. This is an explicit adaptation
of DSKD-CMA, with per-space token means and native output-head scaling.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def causal_response_positions(prompt_ids, response_ids, eos_id, limit=384):
    """Hidden at prompt[-1] predicts response[0]; include response EOS once."""
    if not prompt_ids or not response_ids or eos_id in response_ids:
        raise ValueError('Expected nonempty prompt/text without embedded EOS')
    target = list(response_ids) + [eos_id]
    combined = list(prompt_ids) + target
    if len(combined) > limit:
        raise ValueError('Teacher sequence exceeds budget; no truncation')
    start = len(prompt_ids) - 1
    return combined, target, slice(start, start + len(target))


def forward_kl(student_logits, teacher_logits, mask, temperature):
    """Mean T^2 KL(teacher || student) over supplied response positions."""
    if temperature <= 0 or student_logits.shape != teacher_logits.shape:
        raise ValueError('Positive temperature and matching native-space logits required')
    # Select first so padded NaN/extreme values cannot contaminate reductions.
    student = student_logits[mask].float()
    teacher = teacher_logits[mask].detach().float()
    if not student.numel():
        return student.sum()  # differentiable zero for empty correctness gate
    log_s = F.log_softmax(student / temperature, dim=-1)
    log_t = F.log_softmax(teacher / temperature, dim=-1)
    return (log_t.exp() * (log_t - log_s)).sum(-1).mean() * temperature ** 2


def response_ce(logits, labels):
    mask = labels.ne(-100)
    if not mask.any():
        raise ValueError('No response targets')
    return F.cross_entropy(logits[mask].float(), labels[mask])


def normalized(values, mask):
    # Same global-std normalization idea as DSKD, restricted to valid positions.
    clean = values.detach().float().masked_fill(~mask.unsqueeze(-1), 0)
    scale = clean[mask].std(correction=0).clamp_min(1e-6)
    return clean / scale


class FrozenHead(nn.Module):
    """A native frozen linear head, with pre/post scaling kept distinct."""
    def __init__(self, weight, bias=None, pre_scale=1., post_scale=1.):
        super().__init__()
        self.register_buffer('weight', weight.detach())
        self.register_buffer('bias', None if bias is None else bias.detach())
        self.pre_scale = float(pre_scale)
        self.post_scale = float(post_scale)

    def forward(self, hidden):
        # mT5 pre-scaling occurs in hidden dtype, before linear projection.
        hidden = hidden.to(self.weight.dtype) * self.pre_scale
        logits = F.linear(hidden, self.weight, self.bias)
        return logits * self.post_scale


def mt5_decoder_head(model):
    """Head for the pinned mT5 forward, which projects decoder output directly.

    Legacy tie_word_embeddings metadata does not specify a hidden-state scale
    in this runtime. Actual-checkpoint replay is mandatory in the caller.
    """
    native = model.get_output_embeddings()
    return FrozenHead(native.weight, getattr(native, 'bias', None))


class DualSpaceCMA(nn.Module):
    def __init__(self, student_width, teacher_width, temperature=2., alpha=.5):
        super().__init__()
        if not 0 < alpha < 1 or temperature <= 0:
            raise ValueError('Expected mixed CE/KD and positive temperature')
        self.query = nn.Linear(2 * student_width, 2 * teacher_width, bias=False)
        self.s2t = nn.Linear(student_width, teacher_width, bias=False)
        self.t2s = nn.Linear(teacher_width, student_width, bias=False)
        self.temperature, self.alpha = temperature, alpha

    def forward(self, student_logits, student_hidden, student_index, labels,
                teacher_hidden, teacher_index, teacher_target_embed, teacher_labels,
                student_head, teacher_head):
        smask, tmask = labels.ne(-100), teacher_labels.ne(-100)
        if not smask.any(-1).all() or not tmask.any(-1).all():
            raise ValueError('Every item needs student and teacher response tokens')
        # Padded positions cannot influence attention, normalization or gradients.
        si = student_index.detach().float().masked_fill(~smask.unsqueeze(-1), 0)
        sh = student_hidden.float().masked_fill(~smask.unsqueeze(-1), 0)
        ti = normalized(teacher_index, tmask)
        tv = normalized(teacher_hidden, tmask) + normalized(teacher_target_embed, tmask)
        align = self.query(si) @ ti.transpose(-1, -2) / math.sqrt(ti.shape[-1])
        t2s_weight = align.masked_fill(~tmask.unsqueeze(1), -torch.inf).softmax(-1)
        s2t_weight = align.transpose(-1, -2).masked_fill(~smask.unsqueeze(1), -torch.inf).softmax(-1)
        mapped_teacher = student_head(t2s_weight @ self.t2s(tv))
        mapped_student = teacher_head(s2t_weight @ self.s2t(sh))
        # Teacher distribution is frozen even when its output-head input is mapped.
        with torch.no_grad():
            teacher_logits = teacher_head(teacher_hidden)
        hard_ce = response_ce(student_logits, labels)
        alignment_ce = response_ce(mapped_teacher, labels)
        correct = mapped_teacher.detach().argmax(-1).eq(labels) & smask
        t2s_kl = (forward_kl(student_logits, mapped_teacher, correct, self.temperature)
                  * correct.sum() / smask.sum())
        s2t_kl = forward_kl(mapped_student, teacher_logits, tmask, self.temperature)
        kd = alignment_ce + t2s_kl + s2t_kl
        loss = (1 - self.alpha) * hard_ce + self.alpha * kd
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite cross-tokenizer KD loss')
        metrics = dict(hard_ce=hard_ce.detach(), alignment_ce=alignment_ce.detach(),
                       t2s_kl=t2s_kl.detach(), s2t_kl=s2t_kl.detach(),
                       total=loss.detach(), t2s_gate_fraction=correct.sum() / smask.sum())
        return loss, metrics
