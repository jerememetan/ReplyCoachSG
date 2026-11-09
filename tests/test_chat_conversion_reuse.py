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
    assert not {'gg','gna','thr','ur','anot'} & {v for values in entries.values() for v in values}
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
    assert v2.randomize_local_text('Train yourself for tomorrow.',rng=Rolls([0]))=='Train yourself for tmr.'
