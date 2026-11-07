import pytest
from replycoach.generate.early_stopping import EarlyStopping

def test_overfitting_preserves_baseline_and_stops_after_two_bad_epochs():
    rule=EarlyStopping()
    assert rule.observe(0,2.0)
    assert not rule.observe(1,2.2)
    assert not rule.should_stop
    assert not rule.observe(2,2.3)
    assert rule.should_stop and rule.best_epoch==0 and rule.best==2.0

def test_improvement_resets_patience_and_ties_do_not_replace_best():
    rule=EarlyStopping()
    for epoch,loss in enumerate([2.0,2.1,1.8,1.8]):rule.observe(epoch,loss)
    assert rule.best_epoch==2 and rule.stale==1 and not rule.should_stop
    assert not rule.observe(4,1.9) and rule.should_stop
    with pytest.raises(ValueError):rule.observe(5,float('nan'))
