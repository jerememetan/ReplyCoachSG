"""Reply-act conditioning with an identity initialization and aligned oracle training.

A bounded gated residual modifies existing encoder states rather than inserting
an extra cross-attention position. At initialization the gate is exactly zero,
so every route has exactly the original mT5 decoder memory.
"""
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput
from replycoach.generate.reply_act_v1 import ReplyActMT5


class GatedReplyActMT5(ReplyActMT5):
    residual_bound = 0.1

    def __init__(self, model):
        super().__init__(model)
        self.act_gate = nn.Parameter(torch.zeros(()))

    def memory(self, input_ids, attention_mask, route='predicted', oracle_acts=None):
        if route not in {'none', 'predicted', 'oracle'}:
            raise ValueError(route)
        if route != 'oracle' and oracle_acts is not None:
            raise ValueError('Reference acts are only allowed in the explicit oracle route')
        hidden = self.model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask,
                                         return_dict=True).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        logits = self.classifier(pooled)
        if route != 'none':
            if route == 'oracle':
                if oracle_acts is None or oracle_acts.shape != logits.shape:
                    raise ValueError('Oracle acts must have batch by taxonomy shape')
                if not torch.all((oracle_acts == 0) | (oracle_acts == 1)) or not torch.all(oracle_acts.sum(-1) > 0):
                    raise ValueError('Oracle acts must be nonempty multi-hot vectors')
                weights = oracle_acts.to(logits.dtype)
            else:
                weights = logits.sigmoid()
            act = weights @ self.act_embeddings / weights.sum(-1, keepdim=True).clamp_min(1e-6)
            residual = self.act_norm(act).unsqueeze(1)
            hidden = hidden + self.residual_bound * self.act_gate.tanh() * residual * mask
        return BaseModelOutput(last_hidden_state=hidden), attention_mask, logits

    def forward(self, input_ids, attention_mask, labels, reply_acts, condition=True,
                auxiliary=True, act_weight=0.3, training_route='predicted'):
        if training_route not in {'predicted', 'oracle'}:
            raise ValueError(training_route)
        route = training_route if condition else 'none'
        memory, mask, logits = self.memory(input_ids, attention_mask, route,
                                          reply_acts if route == 'oracle' else None)
        out = self.model(encoder_outputs=memory, attention_mask=mask, labels=labels, return_dict=True)
        act_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, reply_acts.to(logits.dtype), pos_weight=self.act_pos_weight)
        return dict(loss=out.loss + act_weight * act_loss if auxiliary else out.loss,
                    token_loss=out.loss, act_loss=act_loss, act_logits=logits)
