"""Offline data-quality boundaries for local synthetic volume generation."""
import importlib
from pathlib import Path
import pytest


@pytest.fixture
def volume(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('generate_synthetic_volume')


def candidate():
    return {'session_memory':[{'speaker':'user','text':'I can collect your folder.'}],
            'message':'Thanks, the grey folder please.',
            'replies':{'standard':"I'll collect the grey folder.",'local':'Can, grey folder I collect.'}}


def test_writer_schema_cannot_self_approve(volume):
    schema=volume.schema_for(4)
    assert schema['properties']['items']['minItems']==4
    assert schema['properties']['items']['items']['additionalProperties'] is False
    c=candidate(); c['review']={'reviewer':'generator','meaning_preserved':True}
    with pytest.raises(ValueError,match='Unexpected'):
        volume.structural_case(c)


@pytest.mark.parametrize('change',[
    lambda c:c.update(session_memory=[{'speaker':'assistant','text':'hello'}]),
    lambda c:c['replies'].update(local=c['replies']['standard'].lower()),
    lambda c:c.update(message='Email student@example.com now'),
    lambda c:c['replies'].update(standard='word '*25),
])
def test_malformed_private_or_degenerate_cases_rejected(volume,change):
    c=candidate();change(c)
    with pytest.raises(ValueError):volume.structural_case(c)


def test_valid_paired_candidate_keeps_role_context_and_drafts(volume):
    c=candidate();assert volume.structural_case(c)==c
