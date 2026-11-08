from replycoach.final_v7 import make_v7_packet


def test_packet_reuses_exact_replies_for_same_item_and_style_only():
    def row(cid, item, style, reply):
        return dict(candidate_id=cid, conversation_id=item, schema_version=1,
                    conversation=[], incoming_message={'speaker': 'peer', 'text': 'Hello'},
                    style=style, reply=reply)

    public = [row('C1', 'a', 'LOCAL', 'Okay'), row('C2', 'a', 'STANDARD', 'Can lah'),
              row('C3', 'b', 'LOCAL', 'Can lah')]
    private = {r['candidate_id']: {'item_id': r['conversation_id']} for r in public}
    selections = [
        dict(item_id='a', style='LOCAL', candidate_id='C1'),
        dict(item_id='a', style='STANDARD', candidate_id='C2'),
        dict(item_id='b', style='LOCAL', candidate_id='C3'),
    ]
    new_public, new_private, v7 = make_v7_packet(
        public, private, selections, {'a': 'Can lah', 'b': 'Can lah'})
    assert len(new_public) == len(new_private) == 1
    assert new_public[0]['style'] == 'LOCAL'
    assert new_public[0]['reply'] == 'Can lah'
    assert new_public[0]['conversation_id'] == 'a'
    assert [s['candidate_id'] for s in v7] == [new_public[0]['candidate_id'], 'C2', 'C3']
