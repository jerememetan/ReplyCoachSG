import importlib
import pytest
import torch


class Tokens:
    def encode(self, text, **kwargs):
        return list(text.encode('utf-8'))

    def pad(self, rows, **kwargs):
        width = max(map(len, rows['input_ids']))
        ids = [[0]*(width-len(row))+row for row in rows['input_ids']]
        return {'input_ids': torch.tensor(ids),
                'attention_mask': torch.tensor([[int(x != 0) for x in row] for row in ids])}


def api():
    return importlib.import_module('replycoach.qwen_reranker')


def test_official_template_and_reply_are_preserved_when_old_context_is_truncated():
    m = api()
    query = 'old message '*1000 + 'Incoming peer message: how are you?'
    encoded, flags = m.encode_inputs(Tokens(), [(query, "I'm fine, thanks")], max_length=900)
    text = bytes(encoded['input_ids'][0].tolist()).decode()
    assert len(encoded['input_ids'][0]) == 900 and flags == [True]
    assert text.startswith(m.PREFIX) and text.endswith(m.SUFFIX)
    assert m.INSTRUCTION in text
    assert 'Incoming peer message: how are you?' in text
    assert '<Document>: I\'m fine, thanks' in text


def test_small_pairs_keep_all_context_and_oversized_replies_are_rejected():
    m=api()
    encoded, flags=m.encode_inputs(Tokens(), [('peer: hello', 'hi')], max_length=900)
    assert flags == [False]
    assert 'peer: hello' in bytes(encoded['input_ids'][0].tolist()).decode()
    with pytest.raises(ValueError, match='reply'):
        m.encode_inputs(Tokens(), [('hello', 'x'*1000)], max_length=900)


def test_loss_and_ranking_depend_only_on_yes_and_no_tokens():
    m=api()
    logits=torch.tensor([[100., 2., -1.], [-100., -3., 1.]], requires_grad=True)
    scores=m.binary_logits(logits, yes_id=1, no_id=2)
    assert scores.tolist() == [3., -4.]
    loss=torch.nn.functional.binary_cross_entropy_with_logits(scores, torch.tensor([1., 0.]))
    loss.backward()
    assert logits.grad[:,0].tolist() == [0., 0.]
    assert logits.grad[0,1] < 0 and logits.grad[1,1] > 0
