"""Allowlisted, private Kaggle input packaging; never upload the repository."""
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil

from replycoach.reranker_bundle import load_training_bundle


def kaggle_notebook():
    def code(text):
        return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                'source': text.splitlines(keepends=True)}
    description = (
        '# ReplyCoach V6 — MiniLM cross-encoder fine-tuning\n\n'
        'Private GPU experiment. Train: 190 reviewed conversations / 1,140 pairs. '
        'Validation: 50 conversations / 300 pairs. Mismatched replies are weak '
        'negatives, not human-negative gold. No test items or AI judging labels '
        'are attached. Fixed seed 425, AdamW 2e-5, batch16, maximum four epochs. '
        'Select the best checkpoint by validation pair loss; compare reply '
        'quality separately after training. Outputs contain model weights.\n')
    return {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python'}},
            'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': [description]},
                      code('import subprocess, sys\nsubprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", '
                           '"transformers==5.19.0", "tokenizers==0.23.2", "safetensors==0.8.0"])\n'),
                      code('from pathlib import Path\nimport json, os, torch\n'
                           'assert torch.cuda.is_available(), "Select a Kaggle GPU before running production training"\n'
                           'print(json.dumps({"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}))\n'
                           'bundles = [p.parent for p in Path("/kaggle/input").rglob("manifest.json") '
                           'if (p.parent/"train_pairs.jsonl").is_file() and (p.parent/"train_reranker.py").is_file()]\n'
                           'assert len(bundles) == 1, "Attach exactly one private V6 training dataset"\n'
                           'bundle = bundles[0]\noutput = Path("/kaggle/working/reranker_v6")\n'
                           'assert not output.exists(), "Use a fresh saved run; do not overwrite an earlier checkpoint"\n'),
                      code('training_env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")\n'
                           'subprocess.check_call([sys.executable, str(bundle/"train_reranker.py"), '
                           '"--bundle", str(bundle), "--output", str(output)], env=training_env)\n'),
                      code('import shutil\n'
                           'result = json.loads((output/"best/v6_training.json").read_text())\n'
                           'assert result["completed"] and result["production_checkpoint"]\n'
                           'assert result["device"] == "cuda" and result["weights_changed"] and result["optimizer_steps"] > 0\n'
                           'archive = shutil.make_archive("/kaggle/working/replycoach_v6_checkpoint", "zip", output)\n'
                           'print(json.dumps({"checkpoint": archive, "best_epoch": result["best_epoch"], '
                           '"validation_pair_loss": result["best_validation_loss"], "optimizer_steps": result["optimizer_steps"]}))\n')]}


def package_kaggle(bundle, output, *, owner, tag):
    output, bundle = Path(output), Path(bundle)
    if output.exists():
        raise FileExistsError(f'output already exists: {output}')
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{2,49}', owner) or not re.fullmatch(r'[0-9]{8}(?:-[a-z0-9]+)?', tag):
        raise ValueError('invalid Kaggle owner or run tag')
    load_training_bundle(bundle)
    root = Path(__file__).resolve().parents[2]
    dataset, notebook = output/'dataset', output/'notebook'
    dataset.mkdir(parents=True)
    notebook.mkdir()
    for name in ['manifest.json', 'train_pairs.jsonl', 'validation_pairs.jsonl']:
        shutil.copyfile(bundle/name, dataset/name)
    for source, name, old, new in [
        (root/'src/replycoach/reranker_bundle.py', 'reranker_bundle.py', None, None),
        (root/'src/replycoach/reranker_fit.py', 'reranker_fit.py',
         'from replycoach.reranker_bundle import', 'from reranker_bundle import'),
        (root/'scripts/train_reranker.py', 'train_reranker.py',
         'from replycoach.reranker_fit import', 'from reranker_fit import'),
    ]:
        text = source.read_text(encoding='utf-8')
        if old:
            if text.count(old) != 1:
                raise ValueError('portable trainer import contract changed')
            text = text.replace(old, new, 1)
        compile(text, name, 'exec')
        (dataset/name).write_text(text, encoding='utf-8', newline='\n')
    dataset_id = f'{owner}/replycoach-v6-training-{tag}'
    notebook_id = f'{owner}/replycoach-v6-minilm-fine-tuning-{tag}'
    metadata = {'id': dataset_id, 'title': f'ReplyCoach V6 private training {tag}',
                'licenses': [{'name': 'other'}]}
    (dataset/'dataset-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    receipt = {'dataset_id': dataset_id, 'notebook_id': notebook_id, 'private_required': True,
               'files_sha256': {p.name: sha256(p.read_bytes()).hexdigest() for p in dataset.iterdir()}}
    (dataset/'package_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    (notebook/'E_reranker_kaggle.ipynb').write_text(json.dumps(kaggle_notebook(), indent=2), encoding='utf-8')
    kernel = {'id': notebook_id, 'title': f'ReplyCoach V6 MiniLM fine-tuning {tag}',
              'code_file': 'E_reranker_kaggle.ipynb', 'language': 'python', 'kernel_type': 'notebook',
              'is_private': True, 'enable_gpu': True, 'enable_internet': True,
              'dataset_sources': [dataset_id], 'competition_sources': [], 'kernel_sources': []}
    (notebook/'kernel-metadata.json').write_text(json.dumps(kernel, indent=2), encoding='utf-8')
    return receipt
