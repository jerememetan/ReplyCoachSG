import importlib
import importlib.util
import json
import random
import re

import pytest

from replycoach.lexicon import LocalTerm
from replycoach.reranker_bundle import QUERY_FORMAT
from replycoach.schema import Candidate
from replycoach.shared_ranker import PairScore


def api():
    assert importlib.util.find_spec('replycoach.lexicon_refinement'), 'V7 refinement is not implemented'
    return importlib.import_module('replycoach.lexicon_refinement')


def key(text):
    return ' '.join(text.split()).casefold()


@pytest.mark.parametrize('reply', [
    'Sure, see you at 7pm.', 'Can lah!', 'Can lah lor.', 'Already makan leh?',
    'Okay', '可以，谢谢。', 'I cannot go at 7pm, sorry.',
])
def test_exactly_three_unique_variations_even_without_lexical_matches(reply):
    variations = api().generate_local_variants(reply, rng=random.Random(425))
    assert len(variations) == 3
    assert len({key(reply), *(key(text) for text in variations)}) == 4
    assert all(text.strip() for text in variations)
    if '7pm' in reply:
        assert all('7pm' in text for text in variations)
    if 'cannot' in reply:
        assert all('cannot' in text for text in variations)
    if reply[-1] in '.!?。':
        assert all(text[-1] == reply[-1] for text in variations)


def test_randomness_is_reproducible_when_injected_and_varies_across_requests():
    m = api()
    text = 'Sorry, please wait. We can eat at the coffee shop.'
    assert m.generate_local_variants(text, rng=random.Random(1)) == m.generate_local_variants(text, rng=random.Random(1))
    pools = {tuple(sorted(m.generate_local_variants(text, rng=random.Random(seed)))) for seed in range(12)}
    assert len(pools) > 1


def test_whole_phrase_replacement_retains_case_and_does_not_cascade():
    lexicon = [LocalTerm('wait ah', 'please wait', 'request'),
               LocalTerm('makan', 'eat', 'lexicon'),
               LocalTerm('paiseh', 'sorry', 'lexicon')]
    variations = api().generate_local_variants('PLEASE WAIT, sorry. I can eat, not beat.', lexicon=lexicon, rng=random.Random(2))
    assert set(variations) == {
        'WAIT AH, sorry. I can eat, not beat.',
        'PLEASE WAIT, paiseh. I can eat, not beat.',
        'PLEASE WAIT, sorry. I can makan, not beat.',
    }


def test_disabled_ambiguous_location_status_and_non_english_entries_are_excluded():
    variations = api().generate_local_variants(
        'Yes, I am coming by train to eat at the coffee shop, okay?', rng=random.Random(5))
    for text in variations:
        assert 'Yes' in text and 'coming' in text and 'train' in text and 'coffee shop' in text
        assert not re.search(r'[\u4e00-\u9fff]', text)
        assert not any(form in text.lower() for form in ['chi fan', 'ke yi', 'mrt', 'kopitiam'])


def test_urls_email_and_inline_code_are_preserved():
    text = 'Use https://nice.example/eat and sorry@example.com with `please wait`.'
    for changed in api().generate_local_variants(text, rng=random.Random(1)):
        assert 'https://nice.example/eat' in changed
        assert 'sorry@example.com' in changed
        assert '`please wait`' in changed


@pytest.mark.parametrize('text, protected', [
    ('Use https://nice.example/lah.', 'https://nice.example/lah'),
    ('Contact someone@example.lor.', 'someone@example.lor'),
    ('Use `lah`.', '`lah`'),
])
def test_particle_like_endings_inside_protected_text_are_preserved(text, protected):
    for changed in api().generate_local_variants(text, rng=random.Random(1)):
        assert protected in changed


def test_existing_final_particles_are_replaced_instead_of_stacked():
    for changed in api().generate_local_variants('Sure lah lor!', rng=random.Random(4)):
        assert len(re.findall(r'\b(?:lah|leh|lor|sia|meh|hor)\b', changed, re.I)) == 1


def test_lexicon_failure_never_pads_with_duplicates_or_invents_terms():
    m = api()
    with pytest.raises(ValueError, match='three distinct'):
        m.generate_local_variants('We can eat.', lexicon=[LocalTerm('makan', 'eat', 'lexicon')])
    with pytest.raises(ValueError, match='nonempty'):
        m.generate_local_variants(' ')
    lexicon = [LocalTerm('lah', ' ', 'particle', apply=False)]
    with pytest.raises(ValueError, match='three distinct'):
        m.generate_local_variants('Okay.', lexicon=lexicon)


class Scorer:
    identity = {'model': 'test-only', 'revision': 'fixed', 'query_format': QUERY_FORMAT,
                'fine_tuned': True, 'adapter_sha256': 'fixed', 'score_kind': 'yes_minus_no_logit'}

    def __init__(self, original_wins=False):
        self.calls = []
        self.original_wins = original_wins

    def score_pairs(self, pairs):
        self.calls.append(pairs)
        return [PairScore(10 if (reply == 'Sure, 7pm is fine.') == self.original_wins else -1) for _, reply in pairs]


def original():
    return Candidate(['Are you coming?', 'Yes.'], '7pm?', 'Sure, 7pm is fine.', 'LOCAL', 'V6', 3.,
                     meta={'contributing_models': ['V5'], 'fallback_models': [],
                           'original_proposals': [{'model': 'V5', 'reply': 'Sure, 7pm is fine.'}]})


@pytest.mark.parametrize('original_wins', [False, True])
def test_second_pass_scores_four_candidates_with_same_query_and_can_keep_original(original_wins):
    parent = original()
    before = parent.to_dict()
    scorer = Scorer(original_wins)
    result = api().refine_local_reply(parent, scorer, speakers=['peer', 'user'])
    assert result.complete and result.error is None
    assert len(scorer.calls) == 1 and len(scorer.calls[0]) == 4
    assert {reply for _, reply in scorer.calls[0]} == {parent.candidate_text, *result.variations}
    assert len({query for query, _ in scorer.calls[0]}) == 1
    query = scorer.calls[0][0][0]
    assert 'Peer: Are you coming?\nYou: Yes.' in query and 'Requested reply style: LOCAL' in query
    assert (result.selected.candidate_text == parent.candidate_text) == original_wins
    assert result.selected.source == 'V7'
    assert result.selected.meta['ranker'] == scorer.identity
    assert result.selected.meta['contributing_models'] == ['V5']
    assert result.selected.meta['original_proposals'] == before['meta']['original_proposals']
    assert result.selected.meta['v7']['is_original'] == original_wins
    assert parent.to_dict() == before
    assert result.generation_seconds >= 0 and result.ranking_seconds >= 0
    payload = result.to_dict()
    assert len(payload['ranked_candidates']) == 4 and payload['candidate_count'] == 4
    assert payload['selected']['candidate_text'] == result.selected.candidate_text
    assert Candidate.from_dict(payload['selected']).source == 'V7'
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize('failure', ['exception', 'missing_score', 'nan_score'])
def test_second_pass_errors_are_visible_and_retain_original(failure):
    class Broken(Scorer):
        def score_pairs(self, pairs):
            if failure == 'exception':
                raise RuntimeError('ranker stopped')
            return [] if failure == 'missing_score' else [PairScore(float('nan'))] * 4

    parent = original()
    result = api().refine_local_reply(parent, Broken())
    assert result.error and not result.complete and result.ranking is None
    assert result.selected is parent
    json.dumps(result.to_dict(), allow_nan=False)


def test_wrong_style_is_rejected_before_scoring():
    parent = original()
    parent.style = 'STANDARD'
    scorer = Scorer()
    with pytest.raises(ValueError, match='LOCAL'):
        api().refine_local_reply(parent, scorer)
    assert not scorer.calls
