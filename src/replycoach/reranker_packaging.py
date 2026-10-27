"""Allowlisted, private Kaggle input packaging; never upload the repository."""
from hashlib import sha256
import ast
import json
from pathlib import Path
import re
import shutil

from replycoach.reranker_bundle import load_training_bundle
from replycoach.reranker_hard_pairs import apply_reviewed_pairs


def abbreviation_snapshot():
    root = Path(__file__).resolve().parents[2]
    path = root/'scripts/data-split/near_dups.py'
    raw = path.read_bytes()
    tree = ast.parse(raw.decode('utf-8'))
    tables = [ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ABBR'
                                                       for t in node.targets)]
    if len(tables) != 1:
        raise ValueError('expected exactly one literal ABBR mapping')
    safe = {'tmr', 'tmrw', 'alr', 'ltr', 'lib', 'sch', 'cmg', 'cfm', 'rmb',
            'smth', 'u', 'bmt', 'ard', 'tdy', 'tgt', 'qn', 'ans'}
    mapping = {}
    for short, full in sorted(tables[0].items()):
        if short in safe:
            mapping.setdefault(full, []).append(short)
    return mapping, {'path': 'scripts/data-split/near_dups.py', 'sha256': sha256(raw).hexdigest()}


def kaggle_notebook(*, reviewed_pairs=False):
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
        'quality separately after training. Seeded input-only abbreviation augmentation '
        'adds up to three distinct variants per training conversation; no-op variants '
        'are skipped. Validation, labels and replies are unchanged. Expanded rows '
        'inherit labels, including possible false negatives. Outputs contain model '
        'weights and the expanded training-pair audit file.\n')
    if reviewed_pairs:
        description += ('\nThis run adds explicitly human-reviewed alternatives from training items. '
                        'Suitable alternatives are additional positives. Reviewed decisions replace '
                        'any matching weak labels. Abbreviation augmentation is disabled for this experiment.\n')
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
                           '# Set copies to 0 for a control run; use a fresh output directory each time.\n'
                           'AUGMENTATION_COPIES = 3\nREPLACEMENT_PROBABILITY = 0.35\nAUGMENTATION_SEED = 425\n'
                           f'USE_REVIEWED_PAIRS = {reviewed_pairs!r}\n'
                           'if USE_REVIEWED_PAIRS:\n'
                           '    extra_args = ["--reviewed-pairs", str(bundle/"reviewed_pairs.json")]\n'
                           'else:\n'
                           '    extra_args = ["--augmentation-map", str(bundle/"abbreviations.json"), '
                           '"--augmentation-copies", str(AUGMENTATION_COPIES), '
                           '"--augmentation-probability", str(REPLACEMENT_PROBABILITY), '
                           '"--augmentation-seed", str(AUGMENTATION_SEED)]\n'
                           'subprocess.check_call([sys.executable, str(bundle/"train_reranker.py"), '
                           '"--bundle", str(bundle), "--output", str(output), *extra_args], env=training_env)\n'),
                      code('import shutil\n'
                           'result = json.loads((output/"best/v6_training.json").read_text())\n'
                           'assert result["completed"] and result["production_checkpoint"]\n'
                           'assert result["device"] == "cuda" and result["weights_changed"] and result["optimizer_steps"] > 0\n'
                           'print(json.dumps({"augmentation": result["augmentation"], '
                           '"reviewed_pairs": result["reviewed_pairs"], '
                           '"actual_train_pairs": result["train_pair_count"], "validation_pairs": result["validation_pair_count"]}))\n'
                           'archive = shutil.make_archive("/kaggle/working/replycoach_v6_checkpoint", "zip", output)\n'
                           'print(json.dumps({"checkpoint": archive, "best_epoch": result["best_epoch"], '
                           '"validation_pair_loss": result["best_validation_loss"], "optimizer_steps": result["optimizer_steps"]}))\n')]}


def package_kaggle(bundle, output, *, owner, tag, reviewed_pairs=None):
    output, bundle = Path(output), Path(bundle)
    if output.exists():
        raise FileExistsError(f'output already exists: {output}')
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{2,49}', owner) or not re.fullmatch(r'[0-9]{8}(?:-[a-z0-9]+)?', tag):
        raise ValueError('invalid Kaggle owner or run tag')
    loaded = load_training_bundle(bundle)
    if reviewed_pairs is not None:
        apply_reviewed_pairs(loaded, json.loads(Path(reviewed_pairs).read_text(encoding='utf-8')))
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
        (root/'src/replycoach/reranker_augmentation.py', 'reranker_augmentation.py',
         'from replycoach.reranker_bundle import', 'from reranker_bundle import'),
        (root/'src/replycoach/reranker_hard_pairs.py', 'reranker_hard_pairs.py',
         'from replycoach.reranker_bundle import', 'from reranker_bundle import'),
        (root/'scripts/train_reranker.py', 'train_reranker.py',
         'from replycoach.reranker_fit import', 'from reranker_fit import'),
    ]:
        text = source.read_text(encoding='utf-8')
        if old:
            if text.count(old) != 1:
                raise ValueError('portable trainer import contract changed')
            text = text.replace(old, new, 1)
        if name == 'reranker_fit.py':
            text = text.replace('from replycoach.reranker_augmentation import',
                                'from reranker_augmentation import')
            text = text.replace('from replycoach.reranker_hard_pairs import',
                                'from reranker_hard_pairs import')
        compile(text, name, 'exec')
        (dataset/name).write_text(text, encoding='utf-8', newline='\n')
    mapping, source = abbreviation_snapshot()
    (dataset/'abbreviations.json').write_text(json.dumps({'mapping': mapping, 'source': source},
                                                      indent=2, sort_keys=True), encoding='utf-8')
    if reviewed_pairs is not None:
        shutil.copyfile(reviewed_pairs, dataset/'reviewed_pairs.json')
    dataset_id = f'{owner}/replycoach-v6-training-{tag}'
    notebook_id = f'{owner}/replycoach-v6-minilm-fine-tuning-{tag}'
    metadata = {'id': dataset_id, 'title': f'ReplyCoach V6 private training {tag}',
                'licenses': [{'name': 'other'}]}
    (dataset/'dataset-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    receipt = {'dataset_id': dataset_id, 'notebook_id': notebook_id, 'private_required': True,
               'files_sha256': {p.name: sha256(p.read_bytes()).hexdigest() for p in dataset.iterdir()}}
    (dataset/'package_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    (notebook/'E_reranker_kaggle.ipynb').write_text(
        json.dumps(kaggle_notebook(reviewed_pairs=reviewed_pairs is not None), indent=2), encoding='utf-8')
    kernel = {'id': notebook_id, 'title': f'ReplyCoach V6 MiniLM fine-tuning {tag}',
              'code_file': 'E_reranker_kaggle.ipynb', 'language': 'python', 'kernel_type': 'notebook',
              'is_private': True, 'enable_gpu': True, 'enable_internet': True,
              'dataset_sources': [dataset_id], 'competition_sources': [], 'kernel_sources': []}
    (notebook/'kernel-metadata.json').write_text(json.dumps(kernel, indent=2), encoding='utf-8')
    return receipt
