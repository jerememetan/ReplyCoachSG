"""Contraction-normalized surface guard; frozen style-v1 experiment is unchanged."""
import importlib.util
from pathlib import Path
import re

SPEC=importlib.util.spec_from_file_location('frozen_style_base',Path(__file__).with_name('sealion_style_v1.py'))
BASE=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

CONTRACTIONS={"can't":"can not","cannot":"can not","won't":"will not","don't":"do not",
    "doesn't":"does not","didn't":"did not","haven't":"have not","hasn't":"has not",
    "hadn't":"had not","isn't":"is not","aren't":"are not","wasn't":"was not",
    "weren't":"were not","couldn't":"could not","wouldn't":"would not","shouldn't":"should not",
    "i'm":"I am","i've":"I have","i'll":"I will","you're":"you are","you've":"you have",
    "you'll":"you will","we're":"we are","we've":"we have","we'll":"we will",
    "they're":"they are","they've":"they have","they'll":"they will",
    "it's":"it is","that's":"that is","there's":"there is","what's":"what is",
    "where's":"where is","let's":"let us"}


def normalize(text):
    value=text.replace('’',"'")
    for key,replacement in CONTRACTIONS.items():
        value=re.sub(r'(?i)\b'+re.escape(key)+r'\b',replacement,value)
    # A comma/exclamation marks these as discourse cues, not person names.
    value=re.sub(r'(?i)^(wah|aiyo|alamak|eh|hmm|oh|haha)(?=[,!])',lambda match:match[0].lower(),value)
    return value


def guard(anchor,candidate):
    result=BASE.guard(normalize(anchor),normalize(candidate))
    result['normalization']='contractions_v2'
    return result


def rank(anchor,candidates,lexicon,copy_weight=.5,style_weight=.2):
    scored=[]
    for item in candidates:
        checked=guard(anchor,item['text'])
        if item.get('status','complete')!='complete':
            checked['passed']=False;checked['reasons'].append('incomplete_generation')
        copying=BASE.copy_features(anchor,item['text'])
        local=BASE.style_score(item['text'],lexicon)
        score=-item['mean_nll']+style_weight*local-copy_weight*copying['unnecessary_copy']-.25*copying['particle_only_change']
        scored.append(dict(item,guard=checked,copy=copying,style_score=local,rerank_score=score))
    return sorted([item for item in scored if item['guard']['passed']],key=lambda item:item['rerank_score'],reverse=True),scored
