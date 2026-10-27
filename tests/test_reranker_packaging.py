import importlib
import importlib.util
import json

import pytest

from test_reranker_training import inputs


def api():
    assert importlib.util.find_spec('replycoach.reranker_packaging'), 'Private Kaggle packaging is not implemented'
    return importlib.import_module('replycoach.reranker_packaging')


def test_package_is_private_and_contains_only_the_minimal_allowlist(inputs, tmp_path):
    m = api()
    from replycoach.reranker_training import build_training_bundle, write_training_bundle
    source = tmp_path/'bundle'
    write_training_bundle(source, build_training_bundle(inputs))
    (source/'.env').write_text('DO_NOT_UPLOAD=secret')
    (source/'ratings.jsonl').write_text('DO_NOT_UPLOAD')
    out = tmp_path/'package'
    result = m.package_kaggle(source, out, owner='testowner', tag='20261027')
    files = {p.name for p in (out/'dataset').iterdir()}
    assert files == {'manifest.json', 'train_pairs.jsonl', 'validation_pairs.jsonl',
                     'reranker_bundle.py', 'reranker_fit.py', 'train_reranker.py',
                     'dataset-metadata.json', 'package_receipt.json'}
    text = '\n'.join(p.read_text(encoding='utf-8') for p in (out/'dataset').iterdir())
    assert 'DO_NOT_UPLOAD' not in text and 'PRIVATE_AUTHOR' not in text
    metadata = json.loads((out/'notebook/kernel-metadata.json').read_text())
    assert metadata['is_private'] is True and metadata['enable_gpu'] is True
    assert metadata['dataset_sources'] == [result['dataset_id']]
    notebook = json.loads((out/'notebook/E_reranker_kaggle.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            assert cell['outputs'] == [] and cell['execution_count'] is None
            compile(''.join(cell['source']), '<notebook>', 'exec')
    assert 'from replycoach.' not in (out/'dataset/reranker_fit.py').read_text()


def test_packaging_refuses_existing_output_and_invalid_owner(tmp_path):
    m = api()
    out = tmp_path/'existing'
    out.mkdir()
    with pytest.raises(FileExistsError):
        m.package_kaggle(tmp_path/'bundle', out, owner='testowner', tag='20261027')
    with pytest.raises(ValueError, match='owner'):
        m.package_kaggle(tmp_path/'bundle', tmp_path/'new', owner='../escape', tag='20261027')
