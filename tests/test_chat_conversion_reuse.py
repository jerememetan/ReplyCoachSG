import importlib.util,json
from pathlib import Path
from replycoach.io import ROOT


def builder():
    spec=importlib.util.spec_from_file_location('build_chat_variations',ROOT/'scripts/update_chat_variations.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_reuses_actual_augmentation_allowlist_without_changing_source_files():
    from replycoach.reranker_packaging import abbreviation_snapshot
    sources=[ROOT/'scripts/data-split/near_dups.py',ROOT/'data/lexicons/local_terms.json']
    before=[p.read_bytes() for p in sources]
    result=builder().build_dictionary(json.loads((ROOT/'data/lexicons/chat_variations.json').read_text(encoding='utf-8')))
    entries={p:set(e['replacements']) for e in result['conversions'] for p in e['phrases']}
    mapping,source=abbreviation_snapshot()
    assert all(set(forms)<=entries[word] for word,forms in mapping.items())
    assert entries['eat']=={'makan'} and entries['invite']=={'jio'}
    assert entries['sorry']=={'paiseh'}
    assert {'thx','xiexie','xie xie'}<=entries['thanks']
    assert result['sources']['augmentation_abbreviations']==source
    assert before==[p.read_bytes() for p in sources]
    assert 'treat' not in entries and 'train' not in entries and 'nice' not in entries
    assert not {'gg','gna','thr','anot'} & {v for values in entries.values() for v in values}
    assert builder().build_dictionary(result)==result


def test_expanded_dictionary_uses_current_random_matching_without_cascades(monkeypatch):
    from replycoach import lexical_refinement_v2 as v2
    from test_lexical_refinement_v2 import Rolls
    data=builder().build_dictionary(json.loads((ROOT/'data/lexicons/chat_variations.json').read_text(encoding='utf-8')))
    import re
    terms={p:tuple(e['replacements']) for e in data['conversions'] for p in e['phrases']}
    pattern=re.compile(r'(?<!\w)(?:'+'|'.join(r'\s+'.join(map(re.escape,p.split())) for p in sorted(terms,key=lambda p:(-len(p),p)))+r')(?!\w)',re.I)
    monkeypatch.setattr(v2,'_dictionary',lambda:(terms,pattern,'test-only'))
    assert v2.randomize_local_text('You invite me to eat after school.',rng=Rolls([0,0,0,0]))=='U jio me to makan after sch.'
    assert v2.randomize_local_text('you invite me to eat',rng=Rolls([1,1,1]))=='you invite me to eat'
    assert v2.randomize_local_text('Train yourself for tomorrow.',rng=Rolls([1,0]))=='Train yourself for tmr.'


def test_chat_rules_adapted_from_peh_preserve_meaning():
    from replycoach.lexical_refinement_v2 import randomize_local_text
    from test_lexical_refinement_v2 import Rolls

    reply = 'I don’t know. We only need one. You can also do it like that.'
    assert randomize_local_text(reply, rng=Rolls([0, 0, 1, 1, 1, 0, 0])) == (
        'I dunno. We oni need one. You can oso do it liddat.'
    )
    assert randomize_local_text('I like that.', rng=Rolls([0, 0])) == 'I rike tat.'


def test_wiktionary_phrases_match_their_intended_context():
    from replycoach.lexical_refinement_v2 import randomize_local_text
    from test_lexical_refinement_v2 import Rolls

    examples = {
        'Save me a seat?': 'Chope a seat for me?',
        'Can we get the food to go?': 'Can we dabao the food?',
        'Let’s hang out after class.': 'Let’s lepak after class.',
        'Just make a rough guess.': 'Just agak-agak.',
        'Save the file. Can we get it to go?': 'Save the file. Can we geddit to go?',
    }
    for original, expected in examples.items():
        rolls = [1, 0] if 'Can we' in original else [0]
        assert randomize_local_text(original, rng=Rolls(rolls)) == expected


def test_remaining_peh_forms_are_available_in_the_live_dictionary():
    from replycoach.lexical_refinement_v2 import _dictionary

    terms, _, _ = _dictionary()
    expected = {
        'because': 'cos',
        "can't make it": 'cmi',
        'can’t make it': 'cmi',
        'cant make it': 'cmi',
        'china': 'cheena',
        'english': 'engrish',
        'singaporean': 'sinkie',
        'singaporeans': 'sinkies',
        'government': 'gahment',
    }
    for phrase, replacement in expected.items():
        assert replacement in terms[phrase]
    assert terms['you'] == ('u',)
    assert builder().build_dictionary({'conversions': [
        {'phrases': ['you'], 'replacements': ['u', 'chiu']},
    ]})['conversions'][0]['replacements'] == ['u']


def test_peh_cleanup_gives_local_spellings_in_reverse():
    from replycoach.lexical_refinement_v2 import _dictionary, randomize_local_text
    from test_lexical_refinement_v2 import Rolls

    terms, _, _ = _dictionary()
    assert randomize_local_text('People will be here tonight.', rng=Rolls([0])) == (
        'Ppl will be here tonight.'
    )
    assert terms['people'] == ('ppl', 'pple')
    assert terms['please'] == ('pls', 'plz')
    assert 'ur' in terms['your']
    assert randomize_local_text('Cant.', rng=Rolls([])) == 'Cant.'
