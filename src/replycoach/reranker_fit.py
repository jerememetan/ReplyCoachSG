"""Portable, auditable GPU fine-tuning of the pinned V6 cross-encoder."""
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import random
from time import perf_counter

import torch

from replycoach.reranker_bundle import load_training_bundle, pair_payload
from replycoach.reranker_augmentation import augment_training_pairs
from replycoach.reranker_hard_pairs import apply_reviewed_pairs


def weights_sha256(model):
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _load_base(manifest):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    kwargs = {'revision': manifest['base_revision'], 'trust_remote_code': False}
    return (AutoModelForSequenceClassification.from_pretrained(manifest['base_model'], **kwargs),
            AutoTokenizer.from_pretrained(manifest['base_model'], **kwargs))


def _batch(model, tokenizer, rows, device):
    encoded = tokenizer([p['query'] for p in rows], [p['reply'] for p in rows],
                        padding=True, truncation='only_first', max_length=512, return_tensors='pt')
    encoded = {k: v.to(device) for k, v in encoded.items()}
    logits = model(**encoded).logits
    if logits.ndim != 2 or logits.shape[1] != 1:
        raise ValueError('classifier must return one logit per pair')
    labels = torch.tensor([p['label'] for p in rows], dtype=torch.float32, device=device)
    return logits[:, 0], labels


def run_training(bundle_path, output, *, epochs=4, batch_size=16, learning_rate=2e-5,
                 seed=425, smoke=False, max_steps=None, model_factory=None,
                 augmentation_map=None, augmentation_copies=3,
                 augmentation_probability=.35, augmentation_seed=425, reviewed_pairs=None):
    output, bundle_path = Path(output), Path(bundle_path)
    if output.exists():
        raise FileExistsError(f'output already exists: {output}')
    if output.resolve() == bundle_path.resolve() or bundle_path.resolve() in output.resolve().parents:
        raise ValueError('output must be outside the input bundle')
    bundle = load_training_bundle(bundle_path)
    if model_factory is not None and not smoke:
        raise ValueError('test factories are allowed only for explicit smoke runs')
    if (type(epochs) is not int or not 1 <= epochs <= 4 or type(batch_size) is not int
            or batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0
            or type(seed) is not int):
        raise ValueError('invalid fixed-experiment training settings')
    if max_steps is not None and (not smoke or type(max_steps) is not int or max_steps < 1):
        raise ValueError('step limits are positive integers allowed only in smoke runs')
    if reviewed_pairs is not None and augmentation_map is not None:
        raise ValueError('keep reviewed-pair and abbreviation experiments separate; do not mix these options')
    train_rows, augmentation, reviewed = bundle['train'], {'method': 'none'}, {'method': 'none'}
    if reviewed_pairs is not None:
        supplement = json.loads(Path(reviewed_pairs).read_text(encoding='utf-8'))
        train_rows, reviewed = apply_reviewed_pairs(bundle, supplement)
    if augmentation_map is not None:
        snapshot = json.loads(Path(augmentation_map).read_text(encoding='utf-8'))
        train_rows, augmentation = augment_training_pairs(bundle, snapshot['mapping'],
                copies=augmentation_copies, probability=augmentation_probability, seed=augmentation_seed)
        augmentation['source'] = snapshot['source']
    print(json.dumps({'event': 'v6_training_data', 'train_pairs': len(train_rows),
                      'validation_pairs': len(bundle['validation']), 'augmentation': augmentation,
                      'reviewed_pairs': reviewed}), flush=True)
    if not smoke and not torch.cuda.is_available():
        raise ValueError('production V6 training requires a CUDA GPU; use explicit smoke mode for CPU checks')
    device = torch.device('cpu' if smoke else 'cuda')
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model, tokenizer = model_factory() if model_factory else _load_base(bundle['manifest'])
    tokenizer.truncation_side = 'left'
    initial_hash = weights_sha256(model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    output.mkdir(parents=True, exist_ok=False)
    if augmentation_map is not None or reviewed_pairs is not None:
        (output/'training_pairs.jsonl').write_bytes(pair_payload(train_rows))
    best_path = output/'best'
    best_loss, best_epoch, steps = float('inf'), None, 0
    history, elapsed = [], perf_counter()
    for epoch in range(1, epochs+1):
        order = list(train_rows)
        random.Random(seed+epoch).shuffle(order)
        model.train()
        total_loss, count = 0., 0
        for offset in range(0, len(order), batch_size):
            rows = order[offset:offset+batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits, labels = _batch(model, tokenizer, rows, device)
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError('training loss must be finite')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            total_loss += float(loss.detach())*len(rows)
            count += len(rows)
            if max_steps is not None and steps >= max_steps:
                break
        model.eval()
        validation_loss = 0.
        with torch.inference_mode():
            for offset in range(0, len(bundle['validation']), batch_size):
                rows = bundle['validation'][offset:offset+batch_size]
                logits, labels = _batch(model, tokenizer, rows, device)
                loss = loss_fn(logits, labels)
                if not torch.isfinite(loss):
                    raise ValueError('validation loss must be finite')
                validation_loss += float(loss)*len(rows)
        validation_loss /= len(bundle['validation'])
        row = {'epoch': epoch, 'train_loss': total_loss/count, 'validation_loss': validation_loss,
               'optimizer_steps': steps, 'elapsed_seconds': perf_counter()-elapsed}
        history.append(row)
        print(json.dumps({'event': 'v6_epoch', **row}), flush=True)
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_path.mkdir(exist_ok=True)
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            best_hash = weights_sha256(model)
        if max_steps is not None and steps >= max_steps:
            break
    if steps == 0 or initial_hash == best_hash:
        raise ValueError('training must perform optimizer updates and change the saved weights')
    packages = {}
    for name in ['torch', 'transformers', 'tokenizers']:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = 'unavailable'
    result = {**bundle['manifest'], 'record_type': 'v6_cross_encoder_training', 'completed': True,
              'production_checkpoint': not smoke, 'device': str(device),
              'device_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU smoke only',
              'optimizer': 'AdamW', 'learning_rate': learning_rate, 'weight_decay': .01,
              'batch_size': batch_size, 'epochs_requested': epochs, 'optimizer_steps': steps,
              'best_epoch': best_epoch, 'best_validation_loss': best_loss,
              'initial_weights_sha256': initial_hash, 'best_weights_sha256': best_hash,
              'weights_changed': initial_hash != best_hash, 'history': history, 'packages': packages,
              'elapsed_seconds': perf_counter()-elapsed,
              'train_pair_count': len(train_rows), 'validation_pair_count': len(bundle['validation']),
              'augmentation': augmentation, 'reviewed_pairs': reviewed}
    (output/'history.json').write_text(json.dumps(history, indent=2, allow_nan=False), encoding='utf-8')
    (best_path/'v6_training.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'event': 'v6_complete', 'optimizer_steps': steps,
                      'production_checkpoint': not smoke, 'best_epoch': best_epoch}), flush=True)
    return result
