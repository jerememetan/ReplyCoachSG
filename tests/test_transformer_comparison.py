import pytest

from replycoach.transformer_comparison import build_packet, summarise_comparison, task_key


def test_blinding_keeps_model_names_and_failures_out_of_public_packet():
    inputs = [dict(item_id='dev-1', message='Lunch?', conversation=[{'speaker': 'peer', 'text': 'Hi'}])]
    outputs = {model: {'dev-1': dict(standard='Yes', local='Can lah', error=None)}
               for model in ['old', 'new', 'sea']}
    public, private = build_packet(inputs, outputs)
    assert len(public) == len(private) == 6
    assert all('system' not in row and 'error' not in row for row in public)
    assert {row['system'] for row in private} == {'old', 'new', 'sea'}
    assert len({row['candidate_id'] for row in public}) == 6
    assert all(row['evaluation_split'] == 'validation' for row in private)
    assert task_key(public[0], 'p') == task_key({**public[0], 'candidate_id': 'other'}, 'p')


def test_report_uses_paired_conversations_and_refuses_incomplete_ratings():
    inputs = [dict(item_id=f'dev-{i}', message='Lunch?', conversation=[]) for i in range(4)]
    outputs = {m: {r['item_id']: dict(standard='Yes', local='Can', error=None) for r in inputs}
               for m in ['old', 'new', 'sea']}
    public, private = build_packet(inputs, outputs)
    mapping = {r['candidate_id']: r for r in private}
    ratings = []
    for row in public:
        score = {'old': 2, 'new': 5, 'sea': 3}[mapping[row['candidate_id']]['system']]
        for persona in ['p1', 'p2']:
            ratings.append(dict(candidate_id=row['candidate_id'], persona_id=persona, status='ok',
                                rating=dict(appropriateness=score, naturalness=4, style_fit=4, would_send=True)))
    report = summarise_comparison(public, private, ratings, outputs, ['p1', 'p2'], bootstrap_samples=100)
    assert report['recommended_model'] == 'new'
    assert report['conversations'] == 4
    assert report['expected_ratings'] == 48
    assert all(p['appropriateness_delta_ci95'][0] > 0 for p in report['top_model_paired_comparisons'])
    with pytest.raises(ValueError, match='complete'):
        summarise_comparison(public, private, ratings[:-1], outputs, ['p1', 'p2'])


def test_identical_scores_do_not_claim_a_clear_winner():
    inputs = [dict(item_id='dev-1', message='Lunch?', conversation=[])]
    outputs = {m: {'dev-1': dict(standard='Yes', local='Can', error=None)} for m in ['old', 'new', 'sea']}
    public, private = build_packet(inputs, outputs)
    ratings = [dict(candidate_id=r['candidate_id'], persona_id='p', status='ok',
                    rating=dict(appropriateness=4, naturalness=4, style_fit=4, would_send=True)) for r in public]
    report = summarise_comparison(public, private, ratings, outputs, ['p'], bootstrap_samples=20)
    assert report['recommended_model'] is None
