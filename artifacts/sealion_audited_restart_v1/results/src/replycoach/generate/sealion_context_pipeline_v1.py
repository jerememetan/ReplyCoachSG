"""Optional explicit-context policy around the unchanged exported SEA-LION."""
import re
try:
    from .sealion_context_guard_v1 import guard_context
    from .sealion_style_guard_v2 import guard as rewrite_guard
except ImportError:  # Portable study payload, with sibling modules.
    from sealion_context_guard_v1 import guard_context
    from sealion_style_guard_v2 import guard as rewrite_guard


def check_rewrite(anchor, candidate):
    safety=rewrite_guard(anchor,candidate)
    wh=r'\s*(?:what|which|who|whose|where|when|why|how)\b'
    if re.match(wh,candidate,re.I) and not re.match(wh,anchor,re.I):
        safety['passed']=False
        safety['reasons'].append('introduced_information_question')
    past=r'\b(?:we|I|you) (?:were|was|had been|have been) (?:talking|discussing)\b'
    if re.search(past,candidate,re.I) and not re.search(past,anchor,re.I):
        safety['passed']=False
        safety['reasons'].append('introduced_past_conversation')
    return safety


def repair_pair(row, original, rewrite):
    """rewrite(row, English) returns actual model text/status; preserve provenance."""
    checked=guard_context(row,original['english'])
    result=dict(original,original_english=original['english'],original_singlish=original['singlish'],
        context_policy=checked,english_provenance='neural_original',singlish_provenance='neural_original')
    if checked['action']=='keep_model':return result
    anchor=checked['english']
    generated=rewrite(row,anchor)
    safety=check_rewrite(anchor,generated['reply'])
    passed=safety['passed'] and generated['status']=='complete'
    result.update(english=anchor,standard_anchor=anchor,english_status='complete',
        english_provenance='context_rule:'+checked['rule'],
        raw_repaired_singlish=generated['reply'],rewrite_guard=safety,
        singlish=generated['reply'] if passed else anchor,
        singlish_status='complete' if passed else 'english_fallback',
        singlish_provenance='neural_rewrite_of_rule_anchor' if passed else 'english_fallback_rewrite_guard')
    return result


class ContextGuardedPipeline:
    """Inject an original exported ReplyPipeline; no model training or APIs."""
    def __init__(self, original_pipeline):
        if original_pipeline.kind!='sealion':raise ValueError('Requires the original SEA-LION export')
        self.original=original_pipeline

    def _rewrite(self,row,anchor):
        from replycoach.generate.sealion_rewrite_v1 import rewrite_source,BOUNDARY
        from replycoach.generate.sealion_beam_v1 import beam_decode
        self.original._activate('singlish')
        prompt=self.original.tokenizer.encode(rewrite_source(row,anchor),add_special_tokens=False)
        return beam_decode(self.original.model,self.original.tokenizer,prompt,BOUNDARY,
            num_beams=4,no_repeat_ngram_size=3,max_new_tokens=64,length_penalty=1.,device=self.original.device)

    def repair_pairs(self,rows,originals):
        if len(rows)!=len(originals):raise ValueError('Input/output count mismatch')
        for row,original in zip(rows,originals,strict=True):
            if row['item_id']!=original['item_id']:raise ValueError('Input/output ID mismatch')
        return [repair_pair(row,original,self._rewrite) for row,original in zip(rows,originals,strict=True)]

    def generate_pairs(self,rows):
        return self.repair_pairs(rows,self.original.generate_pairs(rows))
