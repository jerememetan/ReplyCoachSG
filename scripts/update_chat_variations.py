"""Refresh the inference dictionary from existing, unchanged mapping sources."""
import copy,json
from hashlib import sha256

from replycoach.io import ROOT
from replycoach.reranker_packaging import abbreviation_snapshot

LOCAL_CANONICALS={'eat','invite','sorry','do not have'}


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
    result.update(version='chat-variations-v2-augmentation-reuse',
        notes='Optional casual abbreviations and selected local-word substitutions. Independent 50% phrase rolls; longest whole phrases; no cascading or particle padding. Ambiguous normalization mappings are not blindly reversed.',
        sources={'augmentation_abbreviations':source,
            'local_terms':{'path':'data/lexicons/local_terms.json','sha256':sha256(raw).hexdigest(),'reused_canonicals':sorted(imported)},
            'user_additions':['thank you/thanks -> xiexie / xie xie']})
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
