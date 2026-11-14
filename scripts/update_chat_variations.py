"""Refresh the inference dictionary from existing, unchanged mapping sources."""
import copy,json
from hashlib import sha256

from replycoach.io import ROOT
from replycoach.reranker_packaging import abbreviation_snapshot

LOCAL_CANONICALS={'eat','invite','sorry','do not have'}
PEH_SOURCE = (
    'https://github.com/justinpeh97/singlish-english-nmt/blob/main/'
    'data_generate/'
)
PEH_CHAT_RULES = (
    (('already',), 'oredi'),
    (('also',), 'oso'),
    (('because',), 'cos'),
    (("can't make it", 'can’t make it', 'cant make it'), 'cmi'),
    (('china',), 'cheena'),
    (("don't know", 'don’t know', 'dont know'), 'dunno'),
    (("don't need", 'don’t need', 'dont need'), 'dunnid'),
    (('do it like that',), 'do it liddat'),
    (('english',), 'engrish'),
    (('government',), 'gahment'),
    (('only',), 'oni'),
    (('relax',), 'relak'),
    (('sabotage',), 'sabo'),
    (('singaporean',), 'sinkie'),
    (('singaporeans',), 'sinkies'),
    (('you',), 'chiu'),
)
# Reverse the cleanup pairs so their casual spellings become V7 variants.
PEH_CLEANUP_RULES = (
    ("already", "alr", "alrd", "ald"),
    ("at the moment", "atm"),
    ("average", "avg"),
    ("back", "bck"),
    ("because", "becoz"),
    ("become", "becum"),
    ("boy", "boi"),
    ("boyfriend", "boyf"),
    ("business", "biz"),
    ("by the way", "btw"),
    ("confirm", "cfm"),
    ("cos", "coz"),
    ("definitely", "defo", "deffo"),
    ("don't", "donch", "dun"),
    ("don’t", "donch", "dun"),
    ("dont", "donch", "dun"),
    ("don't know", "dk"),
    ("don’t know", "dk"),
    ("dont know", "dk"),
    ("don't want", "dunwan", "dw"),
    ("don’t want", "dunwan", "dw"),
    ("dont want", "dunwan", "dw"),
    ("every", "ebery"),
    ("favourite", "fav"),
    ("for real", "fr"),
    ("from", "frm"),
    ("fuck", "fk"),
    ("fucker", "fker"),
    ("get it", "geddit"),
    ("girl", "gurl"),
    ("girlfriend", "girlf"),
    ("got", "gt"),
    ("government", "gov", "govt"),
    ("happy", "heypi"),
    ("have", "hv", "hav", "hab"),
    ("hong kong", "hk"),
    ("i don't want", "idw"),
    ("i don’t want", "idw"),
    ("i dont want", "idw"),
    ("in my honest opinion", "imho"),
    ("is", "ish"),
    ("just", "jus", "juz"),
    ("know", "noe"),
    ("last", "rast"),
    ("like", "rike", "riek"),
    ("look", "rook"),
    ("looks", "rooks"),
    ("many", "mani"),
    ("market", "mkt"),
    ("minutes", "mins"),
    ("money", "marnee"),
    ("month", "mth"),
    ("more", "moar"),
    ("need", "nid"),
    ("never", "nvr", "nv"),
    ("night", "nite"),
    ("nights", "nites"),
    ("now", "nao"),
    ("on the way", "otw"),
    ("people", "ppl", "pple"),
    ("please", "pls", "plz"),
    ("police", "polis"),
    ("remember", "rem", "rmb"),
    ("report", "repork"),
    ("rich", "lich"),
    ("right", "rite"),
    ("school", "sch"),
    ("shit", "shiat", "shiate"),
    ("should", "shld", "shd"),
    ("singapore", "sg", "sinkapore", "sgp"),
    ("sleep", "slp"),
    ("student", "stewdent"),
    ("stupid", "stoopid"),
    ("that", "tat"),
    ("then", "den"),
    ("think", "tink"),
    ("this", "dis", "tis"),
    ("thought", "tot"),
    ("through", "thru"),
    ("uncle", "unker"),
    ("very", "bery", "beri"),
    ("video", "vid"),
    ("videos", "vids"),
    ("week", "wk"),
    ("what", "wat", "wad", "wht"),
    ("with", "wif"),
    ("without", "w/o"),
    ("won't", "wun"),
    ("won’t", "wun"),
    ("wont", "wun"),
    ("would", "wld"),
    ("year old", "yr old"),
    ("years old", "yrs old"),
    ("yesterday", "ytd"),
    ("your", "ur", "yur"),
    ("yourself", "yrself"),
)
WIKTIONARY_CHAT_RULES = (
    ('save me a seat', 'chope a seat for me', 'chope'),
    ('get the food to go', 'dabao the food', 'tapao'),
    ('hang out after class', 'lepak after class', 'lepak'),
    ('just make a rough guess', 'just agak-agak', 'agak-agak'),
)


def build_dictionary(existing):
    result=copy.deepcopy(existing)
    entries=result['conversions']
    def merge(phrase,forms):
        entry=next((e for e in entries if phrase in e['phrases']),None)
        if entry is None:
            entry={'phrases':[phrase],'replacements':[]};entries.append(entry)
        entry['replacements']=list(dict.fromkeys([*entry['replacements'],*forms]))
    abbreviations,source=abbreviation_snapshot()
    for phrase,forms in sorted(abbreviations.items()):merge(phrase,forms)
    path=ROOT/'data/lexicons/local_terms.json';raw=path.read_bytes()
    imported=[]
    for entry in json.loads(raw)['terms']:
        if entry.get('apply',True) and entry['canonical'] in LOCAL_CANONICALS and entry['kind'] in {'lexicon','negation'}:
            merge(entry['canonical'],[entry['surface']]);imported.append(entry['canonical'])
    for phrase in ['thank you','thanks']:merge(phrase,['xiexie','xie xie'])
    for phrases, replacement in PEH_CHAT_RULES:
        for phrase in phrases:
            merge(phrase, [replacement])
    for phrase, *forms in PEH_CLEANUP_RULES:
        merge(phrase, forms)
    for phrase, replacement, _ in WIKTIONARY_CHAT_RULES:
        merge(phrase, [replacement])
    result.update(version='chat-variations-v2-peh-cleanup-forms',
        notes='Optional casual abbreviations and selected local-word substitutions. Independent 50% phrase rolls; longest whole phrases; no cascading or particle padding. Ambiguous normalization mappings are not blindly reversed.',
        sources={'augmentation_abbreviations':source,
            'local_terms':{'path':'data/lexicons/local_terms.json','sha256':sha256(raw).hexdigest(),'reused_canonicals':sorted(imported)},
            'user_additions':['thank you/thanks -> xiexie / xie xie'],
            'peh_chat_forms': {'author': 'Justin Peh', 'url': PEH_SOURCE + 'clean_english_to_singlish.txt',
                               'note': 'All 16 chat forms with adapted phrase matching.'},
            'peh_cleanup_forms': {'author': 'Justin Peh', 'url': PEH_SOURCE + 'clean_english_replace.txt',
                                  'source_blob_sha': '2cfbff4064dd16b466ba77815474494d67ea4d90',
                                  'note': 'Reversed spelling pairs, excluding punctuation-only changes.'},
            'wiktionary_chat_forms': {
                'note': 'Original phrase rules based on Wiktionary word senses.',
                'entries': {phrase: f'https://en.wiktionary.org/wiki/{entry}#English'
                            for phrase, _, entry in WIKTIONARY_CHAT_RULES},
            }})
    return result


def main():
    path=ROOT/'data/lexicons/chat_variations.json'
    result=build_dictionary(json.loads(path.read_text(encoding='utf-8')))
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'groups':len(result['conversions']),
        'input_phrases':sum(len(e['phrases']) for e in result['conversions']),
        'replacement_forms':sum(len(e['replacements']) for e in result['conversions']),
        'version':result['version']}))


if __name__=='__main__':main()
