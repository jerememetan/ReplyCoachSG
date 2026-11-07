"""Offline checks for isolated critic diagnostics and baseline provenance."""
import copy
import importlib
import json
from hashlib import sha256
from pathlib import Path

import pytest


@pytest.fixture
def modes(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('compare_critic_modes')


def test_isolated_prompt_and_payload_do_not_include_other_candidate_or_expected(modes, monkeypatch):
    def request(base, route, payload):
        task = json.loads(payload['messages'][1]['content'])
        assert task == {'conversation': [], 'incoming_peer_message': 'Lunch?',
                        'requested_style': 'LOCAL', 'draft': 'Can.'}
        assert 'Compare A and B' not in payload['messages'][0]['content']
        assert 'choose tie' not in payload['messages'][0]['content']
        return {'done': True, 'message': {'content': json.dumps({'reason': 'Fits.', 'verdict': 'acceptable'})}}
    monkeypatch.setattr(modes, 'request_json', request)
    case = {'id': 'hidden', 'case': 'hidden', 'expected': 'first', 'acceptable': [True, False],
            'conversation': [], 'message': 'Lunch?', 'style': 'LOCAL', 'replies': ['Can.', 'SECRET_OTHER']}
    assert modes.judge_single(case, 'Can.', {'base': 'unused'},
        {'judge_model': 'fake', 'think': False, 'options': {}})['verdict'] == 'acceptable'


@pytest.mark.parametrize('response', [
    {'done': False},
    {'done': True, 'done_reason': 'length'},
    {'done': True, 'message': {'content': '{"reason":"x","verdict":"maybe"}'}},
    {'done': True, 'message': {'content': '{"reason":"","verdict":"acceptable"}'}},
])
def test_rejects_invalid_or_incomplete_single_verdict(modes, monkeypatch, response):
    monkeypatch.setattr(modes, 'request_json', lambda *args: response)
    with pytest.raises(ValueError):
        modes.judge_single({'conversation': [], 'message': 'Hi', 'style': 'STANDARD'}, 'Hi',
                           {'base': 'unused'}, {'judge_model': 'fake', 'think': False, 'options': {}})


def test_uncertainty_is_neither_a_correct_label_nor_false_acceptance(modes):
    result = modes.score_single([{'verdict': 'acceptable'}, {'verdict': 'uncertain'}], [True, False])
    assert result == {'correct': False, 'incorrect_acceptance': False, 'incorrect_rejection': False,
                      'uncertain_candidates': 1, 'candidate_correct': 1}
    assert modes.score_single([{'verdict': 'unacceptable'}, {'verdict': 'acceptable'}],
                              [True, False])['incorrect_acceptance']


def test_baseline_reuse_requires_matching_digest_cases_and_policy(modes, tmp_path):
    cases = [{'id': 'x', 'acceptable': [True, False]}]
    path = tmp_path / 'cases.jsonl'; path.write_text(json.dumps(cases[0])+'\n')
    settings = {'model_digest': 'pinned'}
    baseline = {'status': 'complete', 'purpose': 'critic_calibration_only',
                'prompt_version': modes.PROMPT_VERSION, 'rubric': modes.RUBRIC, 'schema': modes.SCHEMA,
                'cases_sha256': sha256(path.read_bytes()).hexdigest(), 'settings': settings,
                'results': [{**cases[0], 'original': {}, 'swapped': {}}]}
    report = tmp_path / 'baseline.json'; report.write_text(json.dumps(baseline))
    assert 'x' in modes.baseline_results(report, path, cases, settings)
    for field, value in [('settings', {'model_digest': 'different'}), ('rubric', 'different'),
                         ('cases_sha256', 'different'), ('status', 'running'), ('results', [])]:
        changed = copy.deepcopy(baseline); changed[field] = value
        report.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            modes.baseline_results(report, path, cases, settings)


def test_reviewed_diagnostics_keep_rejections_and_do_not_create_preferences(modes, monkeypatch, tmp_path):
    import sys
    runner = importlib.import_module('calibrate_reviewed_critic')
    cases = [dict(id='review-1', conversation=[], message='Which room?', style='STANDARD',
                  draft='Room B12.', acceptable=True, reviewer='user', review_source='chat'),
             dict(id='review-2', conversation=[['user','I cannot meet before eight.']],
                  message='Six?', style='STANDARD', draft='Six works.', acceptable=False,
                  reviewer='user', review_source='chat')]
    casefile = tmp_path/'cases.jsonl'; casefile.write_text(''.join(json.dumps(c)+'\n' for c in cases))
    output = tmp_path/'report.json'
    monkeypatch.setattr(sys, 'argv', ['calibrate',str(casefile),'--output',str(output)])
    monkeypatch.setattr(runner,'prepare_judge',lambda _: {'model_digest':'fixed'})
    monkeypatch.setattr(runner,'judge_single',lambda *args: {'reason':'Test.', 'verdict':'acceptable'})
    runner.main()
    report = json.loads(output.read_text())
    assert report['purpose'] == 'reviewed_critic_diagnostics_only'
    assert report['summary'] == {'cases':2,'correct':1,'incorrect_acceptance':1,
                                 'incorrect_rejection':0,'uncertain':0}
    assert 'questions' not in report
    cases[0].pop('reviewer')
    with pytest.raises(ValueError):
        runner.validate(cases)
