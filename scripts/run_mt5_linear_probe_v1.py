"""Frozen encoder extraction, train-only linear readouts, reserved diagnostics."""

from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path
import random
import time
import traceback

import numpy as np
import peft
from peft import PeftModel
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import torch
import transformers
from transformers import AutoTokenizer

from replycoach.generate.linear_probe_v1 import CLASS_NAMES, SPLITS, TASKS, fit_linear, metrics, pooled_hidden_states, validate_dataset
from replycoach.generate.mt5_matched_v1 import source
from run_mt5_adapter_switch_v1 import load_base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
R = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save():
    OUT.mkdir(exist_ok=True)
    path = OUT / "results.tmp"
    path.write_text(json.dumps(R, indent=2) + "\n")
    path.replace(OUT / "results.json")


def verify_files(mapping):
    for path, expected in mapping.items():
        if sha(path) != expected:
            raise ValueError("Frozen file changed: " + path)


def indices(rows, task, split):
    return np.array([i for i, row in enumerate(rows) if row["task"] == task and row["split"] == split])


def choose(candidates):
    return min(candidates, key=lambda c: (-c["validation"]["accuracy"], c["validation"]["log_loss"], c["C"], c["feature"]))


def score_splits(rows, task, probability_function):
    result = {}
    for split in SPLITS:
        ix = indices(rows, task, split)
        result[split] = metrics([rows[i] for i in ix], probability_function(ix))
    return result


def predictions(rows, task, probabilities):
    return [dict(item_id=r["item_id"], split=r["split"], label=r["label"], predicted_label=int(p >= 0.5),
                 predicted_class=CLASS_NAMES[task][int(p >= 0.5)], positive_probability=float(p))
            for r, p in zip(rows, probabilities, strict=True) if r["task"] == task]


def lexical_baseline(rows, texts, strengths):
    result = {}
    texts = np.array(texts)
    for task in TASKS:
        tr, va = indices(rows, task, "train"), indices(rows, task, "validation")
        candidates = []
        for strength in strengths:
            model = make_pipeline(TfidfVectorizer(ngram_range=(1, 1)), LogisticRegression(C=strength, max_iter=1500, random_state=425))
            model.fit(texts[tr], [rows[i]["label"] for i in tr])
            candidates.append(dict(feature="word_unigram_tfidf", C=strength,
                validation=metrics([rows[i] for i in va], model.predict_proba(texts[va])[:, 1]), model=model))
        selected = choose(candidates)
        probabilities = selected.pop("model").predict_proba(texts)[:, 1]
        result[task] = dict(selected=selected, metrics=score_splits(rows, task, lambda ix: probabilities[ix]),
                            predictions=predictions(rows, task, probabilities))
        print("lexical", task, result[task]["metrics"]["heldout_development"], flush=True)
    return result


def train_probes(rows, features, strengths, stage):
    result = {}
    for task in TASKS:
        tr, va = indices(rows, task, "train"), indices(rows, task, "validation")
        labels = np.array([rows[i]["label"] for i in tr])
        candidates = []
        for name, values in features.items():
            for strength in strengths:
                model = fit_linear(values[tr], labels, strength)
                candidate = dict(feature=name, C=strength,
                    train=metrics([rows[i] for i in tr], model.predict_proba(values[tr])[:, 1]),
                    validation=metrics([rows[i] for i in va], model.predict_proba(values[va])[:, 1]))
                candidates.append(candidate)
        selected = choose(candidates)
        # Refit only TRAIN with the selected validation configuration.
        values = features[selected["feature"]]
        model = fit_linear(values[tr], labels, selected["C"])
        probability = model.predict_proba(values)[:, 1]
        shuffled = np.random.default_rng(425).permutation(labels)
        control = fit_linear(values[tr], shuffled, selected["C"])
        control_probability = control.predict_proba(values)[:, 1]
        scaler, classifier = model.steps[0][1], model.steps[1][1]
        np.savez_compressed(OUT / f"{stage}_{task}_linear_readout.npz", mean=scaler.mean_, scale=scaler.scale_,
                            coefficient=classifier.coef_, intercept=classifier.intercept_, classes=classifier.classes_)
        result[task] = dict(selected=selected, validation_grid=candidates,
                            metrics=score_splits(rows, task, lambda ix: probability[ix]),
                            shuffled_train_labels=dict(configuration="Same selected feature and C; no retuning the control", metrics=score_splits(rows, task, lambda ix: control_probability[ix])),
                            predictions=predictions(rows, task, probability),
                            trainable_readout_parameters=int(classifier.coef_.size + classifier.intercept_.size))
        print(stage, task, "selected", selected["feature"], selected["C"],
              "heldout", result[task]["metrics"]["heldout_development"]["accuracy"], flush=True)
    return result


@torch.inference_mode()
def extract(model, tokenizer, texts):
    collected = defaultdict(list)
    encoder = model.get_encoder()
    for start in range(0, len(texts), 16):
        batch = tokenizer(texts[start:start + 16], padding=True, return_tensors="pt")
        if batch["input_ids"].shape[1] > 256:
            raise ValueError("Diagnostic source cannot be truncated")
        encoded = encoder(**{k: v.cuda() for k, v in batch.items()}, output_hidden_states=True, return_dict=True)
        for name, values in pooled_hidden_states(encoded.hidden_states, batch["attention_mask"].cuda()).items():
            if not torch.isfinite(values).all():
                raise FloatingPointError("Nonfinite encoder feature")
            collected[name].append(values.float().cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in collected.items()}


@torch.inference_mode()
def generation_examples(model, tokenizer, rows):
    result = []
    for start in range(0, len(rows), 4):
        part = rows[start:start + 4]
        inputs = tokenizer([source(r, "STANDARD") for r in part], padding=True, return_tensors="pt")
        generated = model.generate(**{k: v.cuda() for k, v in inputs.items()}, max_new_tokens=48,
                                   num_beams=4, no_repeat_ngram_size=3, do_sample=False, use_cache=True)
        for row, tokens in zip(part, generated, strict=True):
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            result.append(dict(item_id=row["item_id"], text=text, token_ids=tokens.tolist(),
                               status="empty" if not text.strip() else "complete" if tokenizer.eos_token_id in tokens.tolist() else "length_limit"))
    return result


def main():
    global R
    if (OUT / "results.json").exists():
        raise FileExistsError("Fresh experiment snapshot required")
    recipe = json.loads((ROOT / "recipe.json").read_text())
    verify_files({str(ROOT / p): h for p, h in recipe["files_sha256"].items()})
    verify_files(recipe["checkpoint_files_sha256"])
    rows = json.loads((ROOT / "data/diagnostics/mt5_linear_probe_v1/rows.json").read_text())
    validate_dataset(rows)
    texts = [source(r, "STANDARD") for r in rows]
    torch.set_num_threads(4)
    if "L40S" not in torch.cuda.get_device_name(0):
        raise ValueError("L40S required")
    tokenizer = AutoTokenizer.from_pretrained(recipe["tokenizer_checkpoint"], local_files_only=True)
    R = dict(status="running", recipe=recipe, stages={}, frozen_encoder=True, locked_test_used=False, default_changed=False,
             environment=dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__,
                              sklearn=sklearn.__version__, gpu=torch.cuda.get_device_name(0)))
    save()
    started = time.monotonic()
    try:
        R["lexical_baseline"] = lexical_baseline(rows, texts, recipe["C_grid"])
        save()
        for name, specification in recipe["models"].items():
            torch.manual_seed(425)
            model = load_base(Path(specification["checkpoint"]))
            if specification.get("adapter"):
                model = PeftModel.from_pretrained(model, specification["adapter"], is_trainable=False)
            model.requires_grad_(False)
            model.eval()
            if any(p.requires_grad for p in model.parameters()):
                raise ValueError("A frozen probe must not train the backbone or adapter")
            features = extract(model, tokenizer, texts)
            np.savez_compressed(OUT / f"{name}_frozen_features.npz", **features, item_ids=np.array([r["item_id"] for r in rows]))
            R["stages"][name] = dict(status="features_extracted", feature_shapes={k: list(v.shape) for k, v in features.items()},
                                     trainable_encoder_parameters=0)
            save()
            R["stages"][name]["tasks"] = train_probes(rows, features, recipe["C_grid"], name)
            example_ids = set(recipe["generation_example_ids"])
            examples = sorted([r for r in rows if r["item_id"] in example_ids], key=lambda r: r["item_id"])
            R["stages"][name]["generation_examples"] = generation_examples(model, tokenizer, examples)
            if any(p.grad is not None for p in model.parameters()):
                raise ValueError("Frozen encoder unexpectedly accumulated gradients")
            R["stages"][name]["status"] = "complete"
            save()
            del model, features
            gc.collect()
            torch.cuda.empty_cache()
        verify_files(recipe["checkpoint_files_sha256"])
        R.update(status="complete", source_checkpoints_unchanged=True, elapsed_seconds=time.monotonic() - started)
        save()
    except Exception as error:
        R.update(status="failed", error=str(error), traceback=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
