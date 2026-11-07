import torch
from replycoach.generate.sealion_contrast_v1 import contrast_loss, eligible, selection_score


def test_ranking_gradient_lowers_positive_and_raises_negative_nll():
    positive = torch.tensor(2.0, requires_grad=True)
    negative = torch.tensor(1.0, requires_grad=True)
    contrast_loss(positive, negative).backward()
    assert positive.grad == 1.25
    assert negative.grad == -0.25


def test_zero_weight_preserves_existing_primary_gradient():
    x = torch.tensor(2.0, requires_grad=True)
    (x*x).backward(retain_graph=True)
    before = x.grad.clone()
    (0.0*contrast_loss(x, x/2)).backward()
    assert torch.equal(x.grad, before)


def test_replay_guard_and_baseline_eligibility():
    assert eligible(2.0, 2.0)
    assert eligible(2.04, 2.0)
    assert not eligible(2.041, 2.0)
    assert selection_score(2.0, 1.0) < selection_score(2.03, 1.0)
