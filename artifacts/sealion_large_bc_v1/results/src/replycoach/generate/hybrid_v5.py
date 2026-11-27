"""Compose an existing STANDARD generator and a separately trained LOCAL rewriter."""
import math,os
from pathlib import Path
from replycoach.generate.v5_provider import V5Generator,DECODING
from replycoach.generate.rewrite_v5 import rewrite_standard
from replycoach.generate.style_control import REWRITE_STYLE
from replycoach.schema import Candidate,STYLES
from replycoach.io import ROOT

LOCAL_REWRITE_ENV='V5_LOCAL_REWRITE_CHECKPOINT'


def configured_rewriter():
    value=os.environ.get(LOCAL_REWRITE_ENV,'').strip()
    if not value:return None
    path=Path(value)
    return path if path.is_absolute() else ROOT/path


class HybridV5Generator:
    source='V5'

    def __init__(self,standard_checkpoint,rewrite_checkpoint,device=None):
        self.standard=V5Generator(Path(standard_checkpoint),device=device)
        self.local=V5Generator(Path(rewrite_checkpoint),device=self.standard.device)
        self.checkpoint=self.standard.checkpoint
        if self.standard.format.get('joint') or self.standard.format.get('rewrite_local'):
            raise ValueError('Hybrid STANDARD component must generate independent STANDARD replies')
        if not self.local.format.get('rewrite_local') or self.local.format.get('joint'):
            raise ValueError('Hybrid LOCAL component must be trained for LOCAL_REWRITE')

    def generate_rows(self,rows):
        """Batch API preserving supplied explicit roles, returning actual rewrite anchors."""
        from replycoach.generate.train_v5 import generate_replies
        s=self.standard;l=self.local
        standard=generate_replies(s.model,s.tokenizer,rows,'STANDARD',s.max_source,s.max_target,
                                  s.device,DECODING,s.format['speaker_labels'])
        standard=[text.strip() for text in standard]
        if len(standard)!=len(rows):raise ValueError('Incomplete STANDARD generation')
        local=rewrite_standard(l.model,l.tokenizer,rows,standard,l.max_source,l.max_target,
                               l.device,DECODING,l.format['speaker_labels'])
        if len(local)!=len(rows) or any(not text.strip() for text in local):
            raise ValueError('Incomplete or empty LOCAL generation')
        return [dict(item_id=row.get('item_id'),standard=a,local=b.strip(),local_conditioned_on=a)
                for row,a,b in zip(rows,standard,local)]

    def generate(self,message,context):
        from replycoach.generate.rerank_v5 import reply_logprobs
        row=dict(context=list(context),message=message,response='')
        pair=self.generate_rows([row])[0];candidates=[]
        for style,component,task in [('STANDARD',self.standard,'STANDARD'),('LOCAL',self.local,REWRITE_STYLE)]:
            text=pair[style.lower()]
            scored={**row,'response':text}
            if style=='LOCAL':scored['standard_reply']=pair['standard']
            logp=reply_logprobs(component.model,component.tokenizer,[scored],task,
                               component.max_source,component.max_target,component.device,
                               component.format['speaker_labels'])[0]
            meta=dict(checkpoint=str(component.checkpoint),standard_checkpoint=str(self.standard.checkpoint),
                      local_rewrite_checkpoint=str(self.local.checkpoint),rewrite_local=True,
                      local_conditioned_on=pair['standard'],two_checkpoint_pipeline=True,
                      decoding={**DECODING,'max_new_tokens':component.max_target},
                      speakers='guessed: alternating back from message' if context else None,
                      score='exp(mean token log-probability) under generating component; not reply quality')
            candidate=Candidate(context=list(context),message=message,candidate_text=text,style=style,
                                source=self.source,score=min(1.,math.exp(logp)),meta=meta)
            candidate.validate();candidates.append(candidate)
        return candidates
