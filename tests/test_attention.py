import torch
import pytest

from replycoach.generate.seq2seq import Seq2Seq


def test_attention_assigns_no_weight_to_source_padding():
    model = Seq2Seq(vocab_size=12, embed_dim=8, hidden_size=16, use_attention=True)
    source = torch.tensor([[4, 5, 6, 3], [4, 3, 0, 0]])
    lengths = torch.tensor([4, 2])
    target = torch.tensor([[2, 6, 3], [2, 7, 3]])
    attention_weights = []
    hook = model.decoder.register_forward_hook(
        lambda _module, _inputs, output: attention_weights.append(output[2].detach())
    )
    try:
        with torch.no_grad():
            model(source, lengths, target)
    finally:
        hook.remove()

    assert attention_weights
    for weights in attention_weights:
        assert torch.all(weights[1, :, 2:] == 0)
        assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))


@pytest.mark.parametrize("attention", [False, True])
@pytest.mark.parametrize("target_length", [2, 5])
def test_batched_teacher_forcing_matches_step_outputs_and_gradients(attention, target_length):
    torch.manual_seed(425)
    model = Seq2Seq(vocab_size=12, embed_dim=8, hidden_size=16, use_attention=attention)
    source = torch.tensor([[4, 5, 6, 3], [4, 3, 0, 0]])
    lengths = torch.tensor([4, 2])
    target = torch.tensor([[2, 6, 5, 7, 3], [2, 7, 6, 3, 0]])[:, :target_length]
    outputs, hidden = model.encoder(source, lengths)
    mask = torch.arange(outputs.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    steps = []
    for t in range(target.size(1) - 1):
        if attention:
            logits, hidden, _ = model.decoder(target[:, t], hidden, outputs, mask)
        else:
            logits, hidden, _ = model.decoder(target[:, t], hidden)
        steps.append(logits)
    expected = torch.stack(steps, dim=1)
    expected.square().sum().backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad()
    actual = model(source, lengths, target)
    actual.square().sum().backward()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, gradients[name], atol=1e-6, rtol=1e-5)
