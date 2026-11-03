"""Separate final-test guards and anonymous packets; development guards stay intact."""
import random
from replycoach.reply_bank import validate_development_row
from replycoach.schema import DialogueItem, STYLES
from replycoach.evaluation import _input_key


def validate_test_rows(rows, expected_ids, development_rows):
    if len(rows) != len(expected_ids) or {r.get('id') for r in rows} != set(expected_ids):
        raise ValueError('test coverage must exactly match the frozen split manifest')
    items=[]
    for row in rows:
        if row.get('split') != 'test':
            raise ValueError('final queries must retain split=test')
        # Reuse the independent-review/privacy checks, without changing source rows
        # or allowing test records through the development reply-bank loader.
        validate_development_row({**row, 'split':'dev'})
        items.append(DialogueItem.from_dict(row))
    development=[DialogueItem.from_dict(r) for r in development_rows]
    if ({i.id for i in items}&{i.id for i in development}
            or {_input_key(i) for i in items}&{_input_key(i) for i in development}):
        raise ValueError('test/development overlap')
    if len({_input_key(i) for i in items}) != len(items):
        raise ValueError('duplicate normalized test conversations')
    return sorted(items,key=lambda i:i.id)


def make_packet(items, outputs):
    pairs=[]
    for index,item in enumerate(items,1):
        output=outputs[item.id]
        for source in ['BM25','MiniLM','V4','V5']:
            for candidate in output['proposals'][source]:
                public=dict(schema_version=1,conversation_id=f'T{index:06d}',
                    conversation=[{'speaker':t.speaker,'text':t.text} for t in item.session_memory],
                    incoming_message={'speaker':'peer','text':item.message.text},
                    style=candidate['style'],reply=candidate['candidate_text'])
                private=dict(schema_version=1,item_id=item.id,conversation_id=public['conversation_id'],
                    style=candidate['style'],system=source,memory_mode='M2',
                    retrieval_method='bm25' if source=='BM25' else 'minilm',
                    requested_mode='retrieval_only' if source in ['BM25','MiniLM'] else 'generator_only',
                    evaluation_split='test',candidate=candidate)
                pairs.append((public,private))
    random.Random(425).shuffle(pairs)
    for index,(public,private) in enumerate(pairs,1):
        public['candidate_id']=private['candidate_id']=f'C{index:06d}'
    selections=[]
    for item in items:
        for style in STYLES:
            text=' '.join(outputs[item.id][style.lower()]['candidate_text'].split())
            matches=[p for _,p in pairs if p['item_id']==item.id and p['style']==style
                     and ' '.join(p['candidate']['candidate_text'].split())==text]
            selected=min(matches,key=lambda p:p['candidate_id'])
            selections.append(dict(item_id=item.id,style=style,candidate_id=selected['candidate_id'],
                                   system='V6',evaluation_split='test'))
    return [p for p,_ in pairs],[p for _,p in pairs],selections
