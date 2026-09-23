import torch

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
