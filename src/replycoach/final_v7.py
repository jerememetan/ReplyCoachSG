"""Reuse held-out ratings for exact replies and packet only new Local winners."""


def make_v7_packet(public, private, v6_selections, local_winners):
    by_id = {row['candidate_id']: row for row in public}
    existing = {}
    for cid, row in sorted(by_id.items()):
        existing.setdefault((private[cid]['item_id'], row['style'], row['reply']), cid)
    new_public, new_private, selections = [], [], []
    for selection in v6_selections:
        item_id, style = selection['item_id'], selection['style']
        cid = selection['candidate_id']
        if style == 'LOCAL':
            reply = local_winners[item_id]
            if by_id[cid]['reply'] != reply:
                cid = existing.get((item_id, style, reply))
                if cid is None:
                    cid = f'C{len(public) + len(new_public) + 1:06d}'
                    new_public.append({**by_id[selection['candidate_id']], 'candidate_id': cid, 'reply': reply})
                    new_private.append(dict(schema_version=1, candidate_id=cid, item_id=item_id,
                                            conversation_id=new_public[-1]['conversation_id'], style=style,
                                            system='V7', memory_mode='M2', retrieval_method='shared',
                                            requested_mode='local_refinement', evaluation_split='test'))
        selections.append({**selection, 'candidate_id': cid, 'system': 'V7'})
    return new_public, new_private, selections
