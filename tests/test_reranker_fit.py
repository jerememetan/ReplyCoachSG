import importlib
import importlib.util
import json

import pytest
import torch

from test_reranker_training import inputs


def api():
    assert importlib.util.find_spec('replycoach.reranker_fit'), 'Optimizer training is not implemented'
    return importlib.import_module('replycoach.reranker_fit')


class TinyTokenizer:
    def __call__(self, queries, replies, **kwargs):
        return {'input_ids': torch.tensor([[len(r) % 7 + 1] for r in replies], dtype=torch.float32)}

    def save_pretrained(self, path):
        (path/'tokenizer_config.json').write_text('{}')


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, input_ids):
        return type('Output', (), {'logits': self.linear(input_ids)})()

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path/'test_only.pt')


def bundle_path(inputs, tmp_path):
    from replycoach.reranker_training import build_training_bundle, write_training_bundle
    path = tmp_path/'bundle'
    write_training_bundle(path, build_training_bundle(inputs))
    return path


def test_optimizer_updates_weights_and_saves_a_marked_smoke_checkpoint(inputs, tmp_path):
    m = api()
    path = bundle_path(inputs, tmp_path)
    out = tmp_path/'trained'
    result = m.run_training(path, out, smoke=True, max_steps=3, epochs=1, batch_size=4,
                            learning_rate=.01, model_factory=lambda: (TinyModel(), TinyTokenizer()))
    assert result['optimizer_steps'] == 3
    assert result['weights_changed'] is True
    assert result['initial_weights_sha256'] != result['best_weights_sha256']
    assert result['production_checkpoint'] is False and result['device'] == 'cpu'
    assert result['best_epoch'] == 1 and result['history'][0]['validation_loss'] > 0
    saved = json.loads((out/'best/v6_training.json').read_text())
    assert saved['optimizer_steps'] == 3 and saved['base_revision']
    assert (out/'best/tokenizer_config.json').is_file()


def test_invalid_bundle_is_rejected_before_model_loading(inputs, tmp_path):
    m = api()
    path = bundle_path(inputs, tmp_path)
    (path/'train_pairs.jsonl').write_text('{}\n')
    def forbidden():
        pytest.fail('Model loaded before checking data')
    with pytest.raises(ValueError, match='hash'):
        m.run_training(path, tmp_path/'out', smoke=True, model_factory=forbidden)


def test_production_training_refuses_cpu_and_test_factories(inputs, tmp_path, monkeypatch):
    m = api()
    path = bundle_path(inputs, tmp_path)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(ValueError, match='GPU|CUDA'):
        m.run_training(path, tmp_path/'out')
    with pytest.raises(ValueError, match='smoke'):
        m.run_training(path, tmp_path/'out', model_factory=lambda: (TinyModel(), TinyTokenizer()))


def test_existing_output_is_never_replaced(inputs, tmp_path):
    m = api()
    path = bundle_path(inputs, tmp_path)
    out = tmp_path/'existing'
    out.mkdir()
    (out/'keep.txt').write_text('keep')
    with pytest.raises(FileExistsError):
        m.run_training(path, out, smoke=True)
    assert (out/'keep.txt').read_text() == 'keep'


def test_nonfinite_loss_cannot_publish_a_finished_checkpoint(inputs, tmp_path):
    m = api()
    path = bundle_path(inputs, tmp_path)
    class Invalid(TinyModel):
        def forward(self, input_ids):
            return type('Output', (), {'logits': self.linear(input_ids)*float('nan')})()
    with pytest.raises(ValueError, match='finite'):
        m.run_training(path, tmp_path/'out', smoke=True,
                       model_factory=lambda: (Invalid(), TinyTokenizer()))
    assert not (tmp_path/'out/best/v6_training.json').exists()
