import json
from pathlib import Path
from test_reranker_fit import bundle_path
from test_reranker_training import inputs


def test_qwen_package_is_private_minimal_and_ports_real_training_code(inputs,tmp_path):
    from replycoach.qwen_packaging import package_qwen
    bundle=bundle_path(inputs,tmp_path)
    out=tmp_path/'package'
    receipt=package_qwen(bundle,out,owner='test-owner',tag='20261028-01')
    dataset=out/'dataset'
    assert set(p.name for p in dataset.iterdir()) == {
        'manifest.json','train_pairs.jsonl','validation_pairs.jsonl','qwen_reranker.py',
        'qwen_fit.py','reranker_bundle.py','reranker_hard_pairs.py','train_qwen_reranker.py',
        'dataset-metadata.json','package_receipt.json'}
    meta=json.loads((out/'notebook/kernel-metadata.json').read_text())
    assert meta['is_private'] and meta['enable_gpu']
    assert meta['dataset_sources']==[receipt['dataset_id']]
    notebook=json.loads((out/'notebook/Qwen_reranker.ipynb').read_text())
    assert all(not cell.get('outputs') for cell in notebook['cells'])
    setup=''.join(notebook['cells'][0]['source'])
    assert '"uninstall","-y","torchao"' in setup
    assert setup.index('"uninstall"') < setup.index('"install"')
    for path in dataset.glob('*.py'):
        compile(path.read_text(),path.name,'exec')
    assert 'from replycoach.' not in (dataset/'qwen_fit.py').read_text()
    assert 'ratings' not in receipt['files_sha256']
