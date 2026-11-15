import importlib,random
import pytest
from replycoach.schema import Candidate
from replycoach.shared_ranker import PairScore


def api():return importlib.import_module('replycoach.lexical_refinement_v2')


class Rolls:
    def __init__(self,values):self.values=iter(values)
    def random(self):return next(self.values)
    def choice(self,values):return values[0]


def test_each_phrase_has_its_own_half_probability_roll():
    m=api()
    assert m.randomize_local_text('Never mind, please meet tomorrow.',rng=Rolls([.49,.51,.1]))=='Nvm, please meet tmr.'
    assert m.randomize_local_text('nevermind please tomorrow',rng=Rolls([.9,.9,.9]))=='nevermind please tomorrow'
    assert m.randomize_local_text('NEVER MIND, PLEASE TOMORROW!',rng=Rolls([0,0,0]))=='NVM, PLS TMR!'
    assert m.randomize_local_text('please please',rng=Rolls([0,1]))=='pls please'


def test_longest_phrase_boundaries_and_protected_spans_never_cascade():
    m=api()
    original='thank you, thanks. tomorrowland https://x.test/tomorrow a@tomorrow.sg `please tomorrow`'
    actual=m.randomize_local_text(original,rng=Rolls([0,0]))
    assert actual=='thanks, thx. tomorrowland https://x.test/tomorrow a@tomorrow.sg `please tomorrow`'


def test_three_distinct_changed_candidates_are_seed_reproducible_without_particles():
    m=api();reply='Never mind, please meet tomorrow.'
    variants=m.generate_lexical_variants(reply,rng=random.Random(425))
    assert len(variants)==len(set(variants))==6 and reply not in variants
    assert variants==m.generate_lexical_variants(reply,rng=random.Random(425))
    assert all(not any(w in v.lower().split() for w in ['lah','leh','lor','sia']) for v in variants)


def test_variant_slots_cover_distinct_phrases_before_combinations():
    variants = api().generate_lexical_variants('Please eat.', rng=random.Random(425))
    assert {v.casefold() for v in variants[:2]} & {'pls eat.', 'plz eat.'}
    assert 'Please makan.' in variants[:2]
    assert variants[2].casefold() in {'pls makan.', 'plz makan.'}


def test_extra_slots_extend_the_same_three_candidate_pool():
    m = api()
    reply = 'Please eat tomorrow.'
    small = m.generate_lexical_variants(reply, rng=random.Random(425), limit=3)
    large = m.generate_lexical_variants(reply, rng=random.Random(425))
    assert len(large) == len(set(large)) == 6
    assert large[:3] == small
    assert {v.casefold() for v in small} <= {
        'pls eat tomorrow.', 'plz eat tomorrow.', 'please makan tomorrow.',
        'please eat tmr.', 'please eat tmrw.',
    }


def test_particle_variants_apply_to_whole_words_in_any_position():
    m = api()
    for prefix, suffix, forms in (
        ('', ' meet at 7?', ['Can ah', 'Can lah', 'Can lor']),
        ('I ', ' meet at 7.', ['can ah', 'can lah', 'can lor']),
        ('', '.', ['Can ah', 'Can lah', 'Can lor']),
    ):
        can = 'can' if prefix else 'Can'
        assert set(m.generate_lexical_variants(prefix + can + suffix)) == {
            prefix + form + suffix for form in forms
        }
    assert m.generate_lexical_variants('`can` candy.') == []


@pytest.mark.parametrize(('word', 'expected'), [
    ('Ok', {'Ok lor, meet at 7.', 'Ok ah, meet at 7.', 'Ok la, meet at 7.'}),
    ('Okay', {'Okay lor, meet at 7.', 'Okay ah, meet at 7.', 'Okay la, meet at 7.'}),
])
def test_ok_and_okay_forms_can_appear_inside_a_reply(word, expected):
    m = api()
    candidates = {variant for seed in range(4) for variant in
                  m.generate_lexical_variants(f'{word}, meet at 7.', rng=random.Random(seed))}
    assert expected <= candidates


@pytest.mark.parametrize(('reply', 'expected'), [
    ('No worries, I will wait.', 'No worries lah, I will wait.'),
    ("Let's go!", "Let's go lah!"),
    ("Let's go!", "Let's go ah!"),
    ('Let’s go!', 'Let’s go lor!'),
    ('Take care!', 'Take care ah!'),
    ('Ya, meet at 7.', 'Ya lor, meet at 7.'),
    ('Ya, meet at 7.', 'Ya sia, meet at 7.'),
    ('Ya, meet at 7.', 'Ya ah, meet at 7.'),
    ('Not sure.', 'Not sure sia.'),
    ('Nothing much.', 'Nothing much lah.'),
])
def test_particle_options_for_common_chat_phrases(reply, expected):
    assert expected in api().generate_lexical_variants(reply, rng=random.Random(425))


def original(text='Never mind, please meet tomorrow.'):
    return Candidate(['previous'], 'incoming',text,'LOCAL','SEALION',.5)


def test_refinement_scores_up_to_seven_local_candidates_and_can_keep_original():
    m=api();calls=[];base=original()
    class Scorer:
        identity={'model':'test-only','revision':'fixed'}
        def score_pairs(self,pairs):
            calls.append(pairs)
            return [PairScore(10 if reply==base.candidate_text else 0) for _,reply in pairs]
    result=m.refine_lexical_local(base,lambda:Scorer(),['user'],rng=random.Random(425))
    assert result.complete and result.selected.candidate_text==base.candidate_text
    assert len(calls)==1 and len(calls[0])==7
    assert result.to_dict()['method']=='lexical_phrase_diversity_v3'


@pytest.mark.parametrize(('reply', 'expected'), [
    ('Eat!', {'Makan!'}),
    ('Please!', {'Pls!', 'Plz!'}),
])
def test_short_reply_ranks_available_variants(reply, expected):
    m=api();calls=[]
    class Scorer:
        identity={'model':'test-only','revision':'fixed'}
        def score_pairs(self,pairs):
            calls.append(pairs)
            return [PairScore(0 if text==reply else 1) for _,text in pairs]
    result=m.refine_lexical_local(original(reply),Scorer,['user'],rng=random.Random(425))
    assert result.complete and result.to_dict()['status']=='completed'
    assert set(result.variations)==expected
    assert result.selected.candidate_text in expected
    assert len(calls)==1 and {text for _,text in calls[0]}==expected | {reply}
    assert result.to_dict()['candidate_count']==len(expected)+1


def test_no_edits_skips_without_loading_ranker():
    m=api()
    def forbidden():raise AssertionError('No ranker needed')
    result=m.refine_lexical_local(original('See everyone at noon.'),forbidden,['user'])
    assert result.complete and result.skipped_reason and not result.error
    assert result.selected.candidate_text=='See everyone at noon.' and result.variations==[]
    assert result.to_dict()['status']=='skipped'


def test_ranker_failure_retains_original_and_records_error():
    m=api();base=original()
    def failed():raise RuntimeError('Qwen unavailable')
    result=m.refine_lexical_local(base,failed,['user'],rng=random.Random(425))
    assert not result.complete and result.error=='Qwen unavailable'
    assert result.selected is base and len(result.variations)==6


def test_missing_dictionary_is_a_serializable_failure(monkeypatch):
    m=api()
    def missing():raise FileNotFoundError('Chat dictionary missing')
    monkeypatch.setattr(m,'_dictionary',missing)
    result=m.refine_lexical_local(original(),lambda:None,['user'])
    assert result.selected.candidate_text==original().candidate_text
    assert result.to_dict()['error']=='Chat dictionary missing'
