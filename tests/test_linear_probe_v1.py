import numpy as np
import torch

from replycoach.generate.linear_probe_v1 import dataset, fit_linear, metrics, pooled_hidden_states, SPLITS, TASKS
from replycoach.generate.mt5_matched_v1 import source


def test_counterbalanced_data_splits_and_input_label_isolation():
    rows = dataset()
    assert len(rows) == 1536
    for task in TASKS:
        assert [sum(r["task"] == task and r["split"] == s for r in rows) for s in SPLITS] == [384, 64, 64]
    for row in rows:
        changed = {**row, "label": 1 - row["label"], "class_name": "SECRET GOLD ANSWER"}
        assert source(row, "STANDARD") == source(changed, "STANDARD")
        assert "SECRET GOLD ANSWER" not in source(changed, "STANDARD")


def test_linear_readout_is_affine_and_reports_both_members_of_pairs():
    rows = [r for r in dataset() if r["task"] == "ownership" and r["split"] == "train"]
    x = np.array([[r["label"], i % 7] for i, r in enumerate(rows)], dtype=np.float32)
    y = np.array([r["label"] for r in rows])
    model = fit_linear(x, y, 1.0)
    scaler, classifier = model.steps[0][1], model.steps[1][1]
    explicit = (x - scaler.mean_) / scaler.scale_ @ classifier.coef_.T + classifier.intercept_
    np.testing.assert_allclose(model.decision_function(x), explicit[:, 0], atol=1e-7)
    score = metrics(rows, model.predict_proba(x)[:, 1])
    assert score["accuracy"] == 1 and score["counterfactual_group"]["both_correct"] == 1
    constant = metrics(rows, np.full(len(rows), 0.5))
    assert constant["accuracy"] == 0.5 and constant["counterfactual_group"]["both_correct"] == 0


def test_fixed_pooling_ignores_padding_and_uses_last_real_token():
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    states = tuple(torch.randn(2, 3, 4) for _ in range(13))
    a = pooled_hidden_states(states, mask)
    changed = tuple(s.clone() for s in states)
    for s in changed:
        s[0, 2] = 9999
    b = pooled_hidden_states(changed, mask)
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    torch.testing.assert_close(a["layer12_last"][0], states[-1][0, 1])
