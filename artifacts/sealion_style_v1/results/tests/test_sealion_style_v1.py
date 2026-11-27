import importlib.util
import itertools
import math
from pathlib import Path
import re
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
PATH = next(p for p in [ROOT/'src/replycoach/generate/sealion_style_v1.py',
                        ROOT/'sealion_style_v1.py'] if p.exists())
SPEC = importlib.util.spec_from_file_location('style_experiment', PATH)
style = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(style)


def test_guards_keep_facts_direction_negation_and_question_force():
    anchor = 'I am waiting outside seminar room 4A.'
    assert style.guard(anchor, 'I waiting outside seminar room 4A.')['passed']
    for changed in ['I waiting inside seminar room 4A.',
                    'I waiting outside seminar room 4B.',
                    'I am waiting outside seminar room.']:
        assert not style.guard(anchor, changed)['passed']
    assert not style.guard('I will return the tripod to you.', 'You will return the tripod to me.')['passed']
    assert not style.guard('I have not booked the room.', 'I booked the room already.')['passed']
    assert not style.guard('My bottle is purple.', 'My bottle blue.')['passed']
    assert not style.guard('The budget is $7.', 'The budget is $70.')['passed']
    assert not style.guard('My budget is forty dollars.', 'My budget is fifty dollars.')['passed']
    assert style.guard('I will send $19 to Mei.', 'I send $19 to mei.')['passed']
    assert not style.guard('You can ask others for help.', 'Can ask others for help?')['passed']


def test_legitimate_short_copies_and_cosmetic_edits_are_distinguished():
    assert not style.copy_features('Okay.', 'okay')['unnecessary_copy']
    assert not style.copy_features('Purple.', 'Purple.')['unnecessary_copy']
    assert style.copy_features('I have booked the room.', 'I have booked the room!')['unnecessary_copy']
    assert style.copy_features('I have booked the room.', 'I have booked the room lah.')['particle_only_change']


def test_grammar_proposals_preserve_original_slots_and_stay_optional():
    for anchor in ['I am waiting outside seminar room 4A.',
                   "Yes, I've already booked the room this morning.",
                   'Could you get me a cup of coffee?', 'My bottle is purple.']:
        candidates = style.grammar_candidates(anchor)
        assert candidates
        assert all(style.guard(anchor, x['text'])['passed'] for x in candidates)
        assert all(not (set(style.words(x['text'])) & style.PARTICLES) for x in candidates)
    assert not style.grammar_candidates('I have not booked room 4A yet.')
    assert not any(x['provenance']=='grammar:apology_lexicon' for x in style.grammar_candidates('Sorry for your loss.'))


def test_copy_penalty_selects_available_faithful_alternative_not_wrong_fact():
    anchor='I am waiting outside seminar room 4A.'
    candidates=[dict(text=anchor,mean_nll=.1),
                dict(text='I waiting outside seminar room 4A.',mean_nll=.2),
                dict(text='I waiting inside seminar room 4A.',mean_nll=.01)]
    lexicon=dict(weights={})
    unpenalized,_=style.rank_candidates(anchor,candidates,lexicon,copy_weight=0.,style_weight=0.)
    penalized,_=style.rank_candidates(anchor,candidates,lexicon,copy_weight=.5,style_weight=0.)
    assert unpenalized[0]['text']==anchor
    assert penalized[0]['text']==candidates[1]['text']
    assert all(x['text']!=candidates[2]['text'] for x in penalized)
    unfinished=[dict(text=candidates[1]['text'],mean_nll=.001,status='length_limit')]
    valid,details=style.rank_candidates(anchor,unfinished,lexicon)
    assert not valid and 'incomplete_generation' in details[0]['guard']['reasons']


def test_lexicon_is_training_only_and_auxiliary_targets_are_selective():
    train=[dict(standard_anchor='I am waiting outside.',response='I waiting outside.')]*5
    learned=style.fit_lexicon(train)
    assert learned['weights']['i waiting']>0
    assert 'validation-only phrase' not in learned['weights']
    assert style.auxiliary_eligible(dict(standard_anchor='My bottle is purple.',response='My bottle purple.'))
    assert not style.auxiliary_eligible(dict(standard_anchor='Okay.',response='Okay lah.'))
    assert not style.auxiliary_eligible(dict(standard_anchor='My bottle is purple.',response='My bottle blue.'))


def test_copy_margin_has_correct_gradient_direction_and_matched_positive_term():
    pos=torch.tensor(1.,requires_grad=True);negative=torch.tensor(.8,requires_grad=True)
    treatment=style.copy_margin_loss(pos,negative,weight=.1);treatment.backward()
    assert pos.grad>0 and negative.grad<0
    p=torch.tensor(1.,requires_grad=True);n=torch.tensor(.8,requires_grad=True)
    control=style.copy_margin_loss(p,n,weight=0.);control.backward()
    assert p.grad>0 and n.grad==0


class Tokenizer:
    eos_token_id=0
    def decode(self,tokens,skip_special_tokens=True):
        return ''.join({0:'',1:'A',2:'B'}[token] for token in tokens)


class Markov:
    def eval(self):return self
    def __call__(self,input_ids,attention_mask,past_key_values=None,use_cache=True):
        prefix=input_ids if past_key_values is None else torch.cat([past_key_values[0][0],input_ids],dim=1)
        table=torch.tensor([[.15,.55,.30],[.65,.10,.25],[.30,.60,.10]])
        return SimpleNamespace(logits=table[prefix[:,-1]].log()[:,None,:],past_key_values=[(prefix,prefix.clone())])


def test_nbest_matches_exhaustive_top_k_with_cache_and_length_bound():
    result=style.beam_nbest(Markov(),Tokenizer(),[2],re.compile('IMPOSSIBLE'),
        num_beams=16,returns=5,max_new_tokens=3,no_repeat_ngram_size=0,device='cpu')
    table=[[.15,.55,.30],[.65,.10,.25],[.30,.60,.10]];options=[]
    def visit(prefix,total,last):
        if prefix and (prefix[-1]==0 or len(prefix)==3):
            options.append((prefix,total/len(prefix)));return
        for token,p in enumerate(table[last]):visit(prefix+[token],total+math.log(p),token)
    visit([],0.,2)
    expected=sorted(options,key=lambda x:x[1],reverse=True)[:5]
    # Equal-probability paths can swap order under torch.topk. Require the exact
    # top-k set, monotone scores and the exhaustive score for every returned path.
    assert {tuple(x['token_ids']) for x in result}=={tuple(x[0]) for x in expected}
    oracle={tuple(tokens):score for tokens,score in options}
    assert all(a['normalized_score']>=b['normalized_score']-1e-6 for a,b in zip(result,result[1:]))
    for actual in result:
        assert math.isclose(actual['normalized_score'],oracle[tuple(actual['token_ids'])],abs_tol=1e-6)
