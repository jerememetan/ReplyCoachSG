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


def test_extended_notebook_passes_explicit_epoch_and_patience_settings(inputs,tmp_path):
    from replycoach.qwen_packaging import package_qwen
    out=tmp_path/'extended'
    package_qwen(bundle_path(inputs,tmp_path),out,owner='test-owner',tag='20261028-12',
                 epochs=12,early_stopping_patience=2)
    notebook=json.loads((out/'notebook/Qwen_reranker.ipynb').read_text())
    source=''.join(notebook['cells'][1]['source'])
    assert '"--epochs","12"' in source
    assert '"--early-stopping-patience","2"' in source


def test_dailydialog_package_has_two_stage_notebook_and_updated_receipt(inputs,tmp_path):
    from replycoach import qwen_packaging as m
    assert hasattr(m,'package_dailydialog_qwen'), 'Two-stage Kaggle packaging is not implemented'
    from hashlib import sha256
    out=tmp_path/'two_stage'
    receipt=m.package_dailydialog_qwen(bundle_path(inputs,tmp_path),out,owner='test-owner',tag='20261028-20')
    dataset=out/'dataset'
    # Kaggle consumes this upload-control file instead of mounting it in /kaggle/input.
    assert 'dataset-metadata.json' not in receipt['files_sha256']
    assert receipt['gradient_policy']=='finite_only_amp_retry_v1'
    meta=json.loads((out/'notebook/kernel-metadata.json').read_text())
    assert meta['id']==receipt['notebook_id']=='test-owner/replycoach-dailydialog-qwen-20261028-20'
    assert (dataset/'dailydialog_reranker.py').is_file()
    assert (dataset/'train_dailydialog_qwen.py').is_file()
    for name,digest in receipt['files_sha256'].items():
        assert sha256((dataset/name).read_bytes()).hexdigest()==digest
    notebook=json.loads((out/'notebook/Qwen_reranker.ipynb').read_text())
    source='\n'.join(''.join(c['source']) for c in notebook['cells'])
    assert 'train_dailydialog_qwen.py' in source
    assert '/kaggle/working' in source
    assert 'make_archive' in source
    for p in dataset.glob('*.py'):
        code=p.read_text(encoding='utf-8')
        # Qwen's inference-only score_pairs lazily imports the app's PairScore;
        # standalone training uses score_batch and must need no other repo code.
        assert 'from replycoach.' not in code.replace('from replycoach.shared_ranker import PairScore','')
        compile(code,p.name,'exec')
