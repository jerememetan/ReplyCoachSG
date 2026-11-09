"""Anonymous six-system packets for the fixed SEA-LION pipeline re-evaluation."""
import random

SYSTEMS = ('BM25', 'MiniLM', 'V4', 'V5', 'V6', 'V7')
VERSION_LABELS = {'BM25':'V0/V1: BM25 with normalization', 'MiniLM':'V2: MiniLM',
                  'V4':'V4: attention RNN', 'V5':'V5: SEA-LION 3B',
                  'V6':'V6: Qwen + DailyDialog selection', 'V7':'V7: Local lexicon refinement'}


def make_current_packet(items, outputs):
    pairs=[]
    if set(outputs) != {i.id for i in items}:
        raise ValueError('Every frozen conversation needs a complete output')
    for index,item in enumerate(items,1):
        output=outputs[item.id]
        if (not output.get('complete') or output.get('source_errors') or output.get('ranking_error')
                or not output.get('v7',{}).get('complete')):
            raise ValueError('Cannot grade a partial current pipeline')
        systems={}
        for source in SYSTEMS[:4]:
            candidates=output['proposals'].get(source,[])
            if len(candidates)!=2 or {c['style'] for c in candidates}!={'STANDARD','LOCAL'}:
                raise ValueError('Source must supply exactly both styles')
            systems[source]={c['style']:c for c in candidates}
        systems['V6']={'STANDARD':output['standard'],'LOCAL':output['v7']['original']}
        systems['V7']={'STANDARD':output['standard'],'LOCAL':output['local']}
        if output['local'] != output['v7']['selected']:
            raise ValueError('Top-level Local must be the saved V7 winner')
        for source in SYSTEMS:
            for style in ['STANDARD','LOCAL']:
                candidate=systems[source][style]
                if candidate['style']!=style or not candidate['candidate_text'].strip():
                    raise ValueError('Missing or mismatched final candidate')
                public=dict(schema_version=1,conversation_id=f'T{index:06d}',
                    conversation=[dict(speaker=t.speaker,text=t.text) for t in item.session_memory],
                    incoming_message=dict(speaker='peer',text=item.message.text),
                    style=style,reply=candidate['candidate_text'])
                private=dict(schema_version=1,conversation_id=public['conversation_id'],item_id=item.id,
                    system=source,style=style,memory_mode='M2',
                    retrieval_method='bm25' if source=='BM25' else 'minilm' if source=='MiniLM' else 'none',
                    requested_mode='retrieval_only' if source in ['BM25','MiniLM'] else
                                   'generator_only' if source in ['V4','V5'] else 'shared_selection',
                    evaluation_split='test',candidate=candidate,version_label=VERSION_LABELS[source])
                pairs.append((public,private))
    random.Random(425).shuffle(pairs)
    for index,(public,private) in enumerate(pairs,1):
        public['candidate_id']=private['candidate_id']=f'C{index:06d}'
    return [a for a,_ in pairs],[b for _,b in pairs]
