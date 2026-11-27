"""Explicit paired reply-act supervision for the mT5 renderer.

Ambiguous paired responses supervise the decoder, never a unique inferred
intention. Only original-item annotations supervise the conversation-only head.
"""
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput
from replycoach.generate.reply_act_v1 import REPLY_ACTS, act_vector
from replycoach.generate.reply_act_v2 import GatedReplyActMT5
from replycoach.generate.style_control import mt5_input

ACT_DESCRIPTIONS = {
    'accept':'accept the proposal', 'decline':'decline the proposal or request',
    'counterpropose':'suggest an alternative', 'clarify':'ask for clarification',
    'request_action':'ask the peer to take an action',
    'answer_or_inform':'answer or give information',
    'offer_or_commit':'offer help or commit to an action',
    'acknowledge':'acknowledge the message', 'support':'offer emotional support',
    'apologise':'apologise', 'express_uncertainty':'express uncertainty',
}


def conversation_source(row):
    return mt5_input('STANDARD',row['message'],row['context'],True,row['speakers'])


def conditioned_source(row, acts):
    act_vector(acts)
    base=conversation_source(row)
    assert base.endswith(' | You:')
    goals='; '.join(ACT_DESCRIPTIONS[a] for a in acts)
    return base[:-len(' | You:')] + f' | Reply goal: {goals} | You:'


def supported_act_patterns(training_rows):
    """Candidate joint acts come only from training; alternatives need not be unique."""
    vectors={tuple(act_vector(r['reply_acts'])) for r in training_rows}
    return sorted(vectors)


def choose_predicted_acts(logits, patterns):
    """Pick a coherent observed act combination, not independent contradictory labels."""
    if not patterns:raise ValueError('Training-derived act patterns are required')
    vectors=torch.tensor(patterns,device=logits.device,dtype=logits.dtype)
    scores=(nn.functional.logsigmoid(logits).unsqueeze(1)*vectors.unsqueeze(0)
            + nn.functional.logsigmoid(-logits).unsqueeze(1)*(1-vectors).unsqueeze(0)).sum(-1)
    chosen=vectors[scores.argmax(-1)].detach().cpu().tolist()
    return [[a for a,value in zip(REPLY_ACTS,row) if value] for row in chosen]


def per_row_nll(logits, labels):
    loss=nn.functional.cross_entropy(logits.transpose(1,2),labels,ignore_index=-100,reduction='none')
    return loss.sum(-1)/labels.ne(-100).sum(-1).clamp_min(1)


class ActFollowMT5(GatedReplyActMT5):
    def __init__(self, model, mode):
        if mode not in {'direct','gated','text'}:raise ValueError(mode)
        super().__init__(model)
        self.mode=mode

    def render_memory(self, base, mask, acts, conditioned_ids=None, conditioned_mask=None, disabled=False):
        if self.mode=='direct' or disabled:return base,mask
        if acts.shape[-1]!=len(REPLY_ACTS) or not torch.all(acts.sum(-1)>0):
            raise ValueError('Nonempty act vectors required')
        if self.mode=='gated':
            weights=acts.to(base.last_hidden_state.dtype)
            vector=self.act_norm(weights@self.act_embeddings/weights.sum(-1,keepdim=True))
            hidden=base.last_hidden_state+.1*self.act_gate.tanh()*vector.unsqueeze(1)*mask.unsqueeze(-1)
            return BaseModelOutput(last_hidden_state=hidden),mask
        if conditioned_ids is None or conditioned_mask is None:
            raise ValueError('Text-goal route requires explicitly serialized act inputs')
        encoded=self.model.get_encoder()(input_ids=conditioned_ids,attention_mask=conditioned_mask,return_dict=True)
        return encoded,conditioned_mask

    def forward(self, input_ids, attention_mask, labels, acts, original_mask,
                conditioned_ids=None, conditioned_mask=None, wrong_acts=None,
                wrong_ids=None, wrong_mask=None, paired_mask=None,
                rank_weight=.2, margin=.2):
        base,mask,logits=self.memory(input_ids,attention_mask,'none')
        encoded,actual_mask=self.render_memory(base,mask,acts,conditioned_ids,conditioned_mask)
        positive=self.model(encoder_outputs=encoded,attention_mask=actual_mask,labels=labels,return_dict=True)
        act_bce=nn.functional.binary_cross_entropy_with_logits(logits,acts.to(logits.dtype),pos_weight=self.act_pos_weight,reduction='none').mean(-1)
        original_mask=original_mask.to(act_bce.dtype)
        act_loss=(act_bce*original_mask).sum()/original_mask.sum().clamp_min(1)
        rank_loss=positive.loss.new_zeros(())
        if self.mode!='direct' and paired_mask is not None and torch.any(paired_mask):
            if wrong_acts is None:raise ValueError('Counterpart acts required for paired ranking')
            negative_memory,negative_mask=self.render_memory(base,mask,wrong_acts,wrong_ids,wrong_mask)
            negative=self.model(encoder_outputs=negative_memory,attention_mask=negative_mask,labels=labels,return_dict=True)
            rank=nn.functional.relu(margin+per_row_nll(positive.logits,labels)-per_row_nll(negative.logits,labels))
            paired_mask=paired_mask.to(rank.dtype)
            rank_loss=(rank*paired_mask).sum()/paired_mask.sum().clamp_min(1)
        return dict(loss=positive.loss+.3*act_loss+rank_weight*rank_loss,
                    token_loss=positive.loss,act_loss=act_loss,rank_loss=rank_loss,act_logits=logits)
