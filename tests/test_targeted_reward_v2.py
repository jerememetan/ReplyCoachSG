from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import targeted_reward_v2 as scorer
from run_rl_readiness import ModelOutputError

def test_v2_excludes_labels_from_scoring_payload(monkeypatch):
    seen=[]
    def call(config,settings,prompt,payload,schema):
        seen.append(payload)
        return dict(verdict='acceptable',reason='Valid refusal',issues=['none']),{}
    monkeypatch.setattr(scorer,'call',call)
    row=dict(context=['I am busy'],speakers=['user'],message='Join?',draft='No, sorry.',expected_verdict='secret',assessment_reason='hidden')
    assert scorer.score(row,{}, {})['rating']['reward']==1
    assert set(seen[0])=={'conversation','message','style','draft'}
    assert 'secret' not in str(seen) and 'hidden' not in str(seen)

def test_inconsistent_verdict_abstains_without_silent_repair(monkeypatch):
    monkeypatch.setattr(scorer,'call',lambda *a:(dict(verdict='acceptable',reason='Valid',issues=['wrong_role']),{}))
    result=scorer.score(dict(context=[],speakers=[],message='Hi',draft='Hi'),{}, {})
    assert result['rating']['reward'] is None and 'raw_failure' in result


def test_original_prompt_is_unchanged_and_revision_separate():
    import hashlib,json
    from targeted_reward_v1 import PROMPT as old
    root=Path(__file__).parents[1]
    recipe=json.loads((root/'experiments/onpolicy_reward_v1_scorer_recipe.json').read_text())
    assert hashlib.sha256(old.encode()).hexdigest()==recipe['scorer_prompt_sha256']
    assert scorer.PROMPT!=old
