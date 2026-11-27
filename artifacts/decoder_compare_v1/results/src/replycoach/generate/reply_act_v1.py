"""Experimental mT5 encoder reply-act head and decoder memory conditioning.

Inputs contain conversation only. Reference reply acts supervise a multi-label
head; the normal decoder uses its own predicted probabilities during training
and inference. Oracle acts are a separately named diagnostic route.
"""
from __future__ import annotations

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput

REPLY_ACTS = (
    'accept', 'decline', 'counterpropose', 'clarify', 'request_action',
    'answer_or_inform', 'offer_or_commit', 'acknowledge', 'support',
    'apologise', 'express_uncertainty',
)


def act_vector(acts):
    if not acts or len(set(acts)) != len(acts) or any(a not in REPLY_ACTS for a in acts):
        raise ValueError('Expected nonempty unique reply acts from the frozen taxonomy')
    return [float(a in acts) for a in REPLY_ACTS]


class ReplyActMT5(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        width = model.config.d_model
        self.classifier = nn.Linear(width, len(REPLY_ACTS))
        self.act_embeddings = nn.Parameter(torch.empty(len(REPLY_ACTS), width))
        nn.init.normal_(self.act_embeddings, std=0.02)
        self.act_norm = nn.LayerNorm(width)
        self.register_buffer('act_pos_weight', torch.ones(len(REPLY_ACTS)))

    def memory(self, input_ids, attention_mask, route='predicted', oracle_acts=None):
        if route not in {'none', 'predicted', 'oracle'}:
            raise ValueError(route)
        if route != 'oracle' and oracle_acts is not None:
            raise ValueError('Reference acts are only allowed in the explicit oracle route')
        encoded = self.model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = encoded.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        logits = self.classifier(pooled)
        if route == 'none':
            return BaseModelOutput(last_hidden_state=hidden), attention_mask, logits
        if route == 'oracle':
            if oracle_acts is None or oracle_acts.shape != logits.shape:
                raise ValueError('Oracle acts must have batch by taxonomy shape')
            if not torch.all((oracle_acts == 0) | (oracle_acts == 1)) or not torch.all(oracle_acts.sum(-1) > 0):
                raise ValueError('Oracle acts must be nonempty multi-hot vectors')
            weights = oracle_acts.to(logits.dtype)
        else:
            weights = logits.sigmoid()
        act = weights @ self.act_embeddings / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        act = self.act_norm(act).unsqueeze(1)
        memory = torch.cat([act, hidden], dim=1)
        mask = torch.cat([torch.ones_like(attention_mask[:, :1]), attention_mask], dim=1)
        return BaseModelOutput(last_hidden_state=memory), mask, logits

    def forward(self, input_ids, attention_mask, labels, reply_acts, condition=True, auxiliary=True, act_weight=0.3):
        encoded, mask, logits = self.memory(input_ids, attention_mask, 'predicted' if condition else 'none')
        output = self.model(encoder_outputs=encoded, attention_mask=mask, labels=labels, return_dict=True)
        act_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, reply_acts.to(logits.dtype), pos_weight=self.act_pos_weight)
        loss = output.loss + act_weight * act_loss if auxiliary else output.loss
        return {'loss': loss, 'token_loss': output.loss, 'act_loss': act_loss, 'act_logits': logits}

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, route='predicted', oracle_acts=None, **kwargs):
        encoded, mask, logits = self.memory(input_ids, attention_mask, route, oracle_acts)
        tokens = self.model.generate(encoder_outputs=encoded, attention_mask=mask, **kwargs)
        return tokens, logits.sigmoid()


def predicted_acts(probabilities, threshold=0.5):
    """Display hard labels; decoder conditioning uses all soft probabilities."""
    rows = probabilities.detach().cpu().tolist()
    return [[a for a, p in zip(REPLY_ACTS, row) if p >= threshold] or [REPLY_ACTS[max(range(len(row)), key=row.__getitem__)]] for row in rows]
