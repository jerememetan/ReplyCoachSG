"""Actual reply generation with trained vocabulary heads or output MLPs."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import traceback

import torch
import transformers
from transformers import AutoTokenizer

import run_mt5_adapter_switch_v1 as core
from replycoach.generate.mt5_matched_v1 import encode, paired_rows, reply_loss, task_rows
from replycoach.generate.mt5_output_heads_v1 import configure_head, load_head, save_head, train_head_mode

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
R = {}


def read(path):
    return json.loads((ROOT / path).read_text())


def rows(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save():
    OUT.mkdir(exist_ok=True)
    temporary = OUT / "results.tmp"
    temporary.write_text(json.dumps(R, indent=2) + "\n")
    temporary.replace(OUT / "results.json")


def clear():
    gc.collect()
    torch.cuda.empty_cache()


def fresh_model():
    return configure_head(core.load_base(ROOT / "model"), R["recipe"]["variant"], bottleneck=256, dropout=0.1)


def fit(stage, train, valid, rate):
    torch.manual_seed(425)
    torch.cuda.manual_seed_all(425)
    random.seed(425)
    name = f"{stage}_lr{rate:.0e}".replace("-0", "-")
    model = fresh_model()
    params = [p for p in model.parameters() if p.requires_grad]
    expected = R["recipe"]["trainable_parameters"]
    if sum(p.numel() for p in params) != expected:
        raise ValueError("Wrong trainable parameter count")
    frozen = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    mode = "rewrite" if stage == "rewrite" else "direct"
    encoded = [encode(row, core.TOK, mode, 256, 64) for row in train]
    validation = [encode(row, core.TOK, mode, 256, 64) for row in valid]
    phase = dict(status="running", stage=stage, learning_rate=rate, history=[],
                 train_examples=len(train), validation_examples=len(valid),
                 trainable_parameters=expected, frozen_parameters=sum(p.numel() for _, p in frozen),
                 initial_source="Fresh original fully adapted base and output projection; independent task head")
    R["phases"][name] = phase
    save()
    optimizer = torch.optim.AdamW(params, lr=rate, weight_decay=0.01, foreach=False)
    best, stale, steps = float("inf"), 0, 0
    for epoch in range(13):
        if epoch:
            train_head_mode(model)
            optimizer.zero_grad(set_to_none=True)
            order = torch.randperm(len(encoded), generator=torch.Generator().manual_seed(425 + epoch)).tolist()
            micro, accum = (1, 8) if stage == "rewrite" else (2, 2)
            batches = [order[i:i + micro] for i in range(0, len(order), micro)]
            for i, indices in enumerate(batches):
                x, mask, labels = core.tensor([encoded[j] for j in indices])
                output = model(input_ids=x, attention_mask=mask, labels=labels, use_cache=False)
                loss = reply_loss(output.logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite reply loss")
                group = min(accum, len(batches) - (i // accum) * accum)
                (loss / group).backward()
                if (i + 1) % accum == 0 or i + 1 == len(batches):
                    norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("Nonfinite gradients")
                    if any(p.grad is not None for _, p in frozen):
                        raise ValueError("Frozen weights received gradients")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    steps += 1
                    if steps == 1:
                        phase["first_update_finite"] = True
                        phase["frozen_parameters_received_no_gradients"] = True
                        save()
                        print(name, "first update finite", flush=True)
        scores = core.validation(model, validation)
        phase["history"].append(dict(epoch=epoch, optimizer_steps=steps, validation=scores))
        if scores["token_ce"] < best:
            best, stale = scores["token_ce"], 0
            phase.update(best_epoch=epoch, best_validation=scores)
            save_head(model, OUT / name / "best", R["recipe"]["variant"], dict(
                stage=stage, epoch=epoch, learning_rate=rate, bottleneck=256, dropout=0.1,
                source_model_sha256=R["source_manifest"]["files_sha256"]["model.safetensors"],
                input_style="LOCAL_REWRITE" if stage == "rewrite" else "STANDARD"))
        elif epoch:
            stale += 1
        phase["stale_epochs"] = stale
        save()
        print(name, epoch, scores, flush=True)
        if stale >= 2:
            break
    phase["status"] = "complete"
    save()
    del optimizer, frozen, params, model, output, loss
    clear()
    return name


def select_stage(stage, train, valid):
    candidates = [fit(stage, train, valid, rate) for rate in R["recipe"]["learning_rates"]]
    selected = min(candidates, key=lambda name: (R["phases"][name]["best_validation"]["token_ce"], R["phases"][name]["learning_rate"]))
    target = OUT / stage
    target.mkdir()
    (target / "best").symlink_to(Path("..") / selected / "best", target_is_directory=True)
    R["selected"][stage] = dict(phase=selected, best_epoch=R["phases"][selected]["best_epoch"],
                               best_validation=R["phases"][selected]["best_validation"],
                               learning_rate=R["phases"][selected]["learning_rate"], checkpoint=str(OUT / stage / "best"))
    save()
    return OUT / stage / "best"


def main():
    global R
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["linear", "mlp"], required=True)
    args = parser.parse_args()
    if (OUT / "results.json").exists():
        raise FileExistsError("Fresh snapshot required")
    recipe = read("recipe.json")
    manifest = read("source_manifest.json")
    if args.variant != recipe["variant"]:
        raise ValueError("Variant differs from frozen recipe")
    for name, expected in recipe["files_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Snapshot changed: " + name)
    for name, expected in manifest["files_sha256"].items():
        if sha(ROOT / "model" / name) != expected:
            raise ValueError("Original adapted model changed: " + name)
    if not (ROOT / recipe["training_file"]).read_bytes().startswith((ROOT / "data/dev/train.jsonl").read_bytes()):
        raise ValueError("Team data changed")
    torch.set_num_threads(4)
    if "L40S" not in torch.cuda.get_device_name(0):
        raise ValueError("L40S required")
    core.TOK = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True)
    R = dict(status="running", recipe=recipe, source_manifest=manifest, phases={}, selected={}, outputs={}, default_changed=False,
             environment=dict(torch=torch.__version__, transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0)))
    save()
    try:
        train, valid = rows(recipe["training_file"]), rows("data/dev/validation.jsonl")
        audit = read("data/training_expansion/sealion_rewrite_v1/pair_audit.json")
        if sha(ROOT / recipe["training_file"]) != audit["source_sha256"]:
            raise ValueError("Rewrite pair audit mismatch")
        standard = [r for r in task_rows(train) if r["style"] == "STANDARD"]
        vstandard = [r for r in task_rows(valid) if r["style"] == "STANDARD"]
        if (len(standard), len(vstandard)) != (790, 50):
            raise ValueError("English counts changed")
        english = select_stage("standard", standard, vstandard)
        english_hashes = {p.name: sha(p) for p in english.iterdir() if p.is_file()}
        reused = read("data/evaluation/sealion_full_rewrite_v1/inputs.json")
        fresh = read("data/evaluation/restored_pipeline_v1/inputs.json")
        cases = reused + fresh
        model = load_head(fresh_model(), english, args.variant)
        for method in core.DECODE:
            R["outputs"][method] = dict(english=core.generate(model, cases, "STANDARD", method))
            save()
        del model
        clear()
        train_ids = {r["id"] for r in audit["items"] if r["split"] == "train" and r["eligible"]}
        valid_ids = {r["id"] for r in audit["items"] if r["split"] == "validation" and r["eligible"]}
        rt, rv = paired_rows(train, train_ids), paired_rows(valid, valid_ids)
        if (len(rt), len(rv)) != (745, 42):
            raise ValueError("Rewrite counts changed")
        rewrite = select_stage("rewrite", rt, rv)
        model = fresh_model()
        challenge = read("data/training_expansion/sealion_fidelity_v1/challenge.json")
        baseline = read("baseline_results.json")
        lora = read("lora_results.json")
        R["head_integrity"] = dict(english_hashes=english_hashes, independent_task_heads=True, roundtrip={})
        for method in core.DECODE:
            load_head(model, english, args.variant)
            replay = core.generate(model, cases, "STANDARD", method)
            if replay != R["outputs"][method]["english"]:
                raise ValueError("Saved English head did not replay exactly")
            R["head_integrity"]["roundtrip"][method] = len(replay)
            load_head(model, rewrite, args.variant)
            own = [r["text"] for r in R["outputs"][method]["english"]]
            for reference in [baseline, lora]:
                if [r["item_id"] for r in reference["outputs"][method]["english"]] != [r["item_id"] for r in cases]:
                    raise ValueError("Baseline case mismatch")
            groups = [
                ("actual_english", cases, own),
                ("shared_mt5_english", cases, [r["text"] for r in baseline["outputs"][method]["english"]]),
                ("shared_lora_english", cases, [r["text"] for r in lora["outputs"][method]["english"]]),
                ("shared_sea_english", reused, [r["actual_anchor"] for r in reused]),
                ("diagnostic", [r for r in reused if "diagnostic_anchor" in r], [r["diagnostic_anchor"] for r in reused if "diagnostic_anchor" in r]),
                ("fresh_diagnostic", fresh, [r["diagnostic_anchor"] for r in fresh]),
                ("challenge", challenge, [r["standard_anchor"] for r in challenge]),
            ]
            for name, part, anchors in groups:
                generated = core.generate(model, part, "LOCAL_REWRITE", method, anchors)
                R["outputs"][method][name] = [dict(**row, standard_anchor=anchor) for row, anchor in zip(generated, anchors, strict=True)]
                save()
        for name, expected in manifest["files_sha256"].items():
            if sha(ROOT / "model" / name) != expected:
                raise ValueError("Source checkpoint changed")
        if any(sha(english / name) != expected for name, expected in english_hashes.items()):
            raise ValueError("English head changed during rewrite training/evaluation")
        flags = [dict(method=m, route=route, item_id=r["item_id"], status=r["status"])
                 for m, routes in R["outputs"].items() for route, part in routes.items() for r in part if r["status"] != "complete"]
        R.update(status="complete_with_flags" if flags else "complete", original_checkpoint_unchanged=True,
                 english_head_unchanged=True, flagged_outputs=flags)
        save()
    except Exception as error:
        R.update(status="failed", error=str(error), traceback=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
