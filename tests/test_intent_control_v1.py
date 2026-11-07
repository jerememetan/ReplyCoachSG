import importlib.util,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_intent_control_v1 import validate_plan

def test_incoming_and_reply_intents_are_separate():
 p=dict(incoming_intent='invite_or_propose',reply_acts=['decline','counterpropose'],facts_and_roles=['USER unavailable Wednesday'],constraints=['Wednesday unavailable'])
 assert validate_plan(p)==p
 with pytest.raises(ValueError,match='reply act'):validate_plan({**p,'reply_acts':['invite_or_propose']})
 with pytest.raises(ValueError,match='incoming intent'):validate_plan({**p,'incoming_intent':'support'})
 with pytest.raises(ValueError):validate_plan({**p,'reply_acts':[]})
