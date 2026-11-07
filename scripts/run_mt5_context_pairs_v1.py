"""Paired fixed-budget context-pair ablation using one shared mT5 LoRA."""

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import traceback

import torch
import transformers
import peft
from peft import PeftModel
from transformers import AutoTokenizer

import run_mt5_adapter_switch_v1 as core
from replycoach.generate.context_pair_ablation_v1 import backward_auxiliary, encode_row, schedule_sha256
from replycoach.generate.mt5_adapter_switch_v1 import new_task_adapter
from replycoach.generate.mt5_matched_v1 import reply_loss, source

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
R = {}


def read(path):
    return json.loads((ROOT / path).read_text())


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


def clear():
    gc.collect()
    torch.cuda.empty_cache()


def adapter_digest(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if "lora_" in name:
            h.update(name.encode())
            h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def validation(model):
    report = {name: core.validation(model, values) for name, values in VALID.items()}
    report["macro_ce"] = (report["english"]["token_ce"] + report["rewrite"]["token_ce"]) / 2
    return report


@torch.no_grad()
def generate(model, cases, style, method, anchors=None, batch_size=4):
    model.eval()
    result = []
    for start in range(0, len(cases), batch_size):
        part = cases[start:start + batch_size]
        ready, texts, answers = [], [], {}
        for j, row in enumerate(part):
            anchor = None if anchors is None else anchors[start + j]
            if anchors is not None and not anchor.strip():
                answers[j] = dict(item_id=row["item_id"], text="", status="failed_empty_anchor", token_ids=[])
            else:
                ready.append(j)
                texts.append(source(row, style, anchor))
        if ready:
            inputs = core.TOK(texts, return_tensors="pt", padding=True)
            if inputs["input_ids"].shape[1] > 192:
                raise ValueError("No generation input truncation permitted")
            output = model.generate(**{k: v.cuda() for k, v in inputs.items()}, max_new_tokens=48, use_cache=True, **core.DECODE[method])
            for j, tokens in zip(ready, output, strict=True):
                text = core.TOK.decode(tokens, skip_special_tokens=True)
                status = "failed_empty" if not text.strip() else "complete" if core.TOK.eos_token_id in tokens.tolist() else "length_limit"
                answers[j] = dict(item_id=part[j]["item_id"], text=text, token_ids=tokens.tolist(), status=status)
        result.extend(answers[j] for j in range(len(part)))
    return result


def evaluate(model, secondary=False):
    reused = read("data/evaluation/sealion_full_rewrite_v1/inputs.json")
    old_fresh = read("data/evaluation/restored_pipeline_v1/inputs.json")
    new_fresh = read("data/evaluation/context_pairs_v1/inputs.json")
    cases = reused + old_fresh + new_fresh
    baseline = read("baseline_results.json")
    lora = read("lora_results.json")
    challenge = read("data/training_expansion/sealion_fidelity_v1/challenge.json")
    result = {}
    for method in core.DECODE:
        english = generate(model, cases, "STANDARD", method)
        own = [x["text"] for x in english]
        routes = {"english": english}
        groups = [("actual_english", cases, own)]
        if not secondary:
            diagnostic_cases = [r for r in reused if "diagnostic_anchor" in r] + old_fresh + new_fresh
            groups += [
                ("shared_full_english", reused + old_fresh, [x["text"] for x in baseline["outputs"][method]["english"]]),
                ("shared_lora_english", reused + old_fresh, [x["text"] for x in lora["outputs"][method]["english"]]),
                ("diagnostic", diagnostic_cases, [r["diagnostic_anchor"] for r in diagnostic_cases]),
                ("challenge", challenge, [r["standard_anchor"] for r in challenge]),
            ]
        for name, part, anchors in groups:
            outputs = generate(model, part, "LOCAL_REWRITE", method, anchors)
            routes[name] = [dict(**row, standard_anchor=anchor) for row, anchor in zip(outputs, anchors, strict=True)]
        result[method] = routes
    return result


def load_adapter(path):
    return PeftModel.from_pretrained(core.load_base(ROOT / "model"), path, is_trainable=False).eval()


def replay_old():
    specification = R["recipe"]["older_reference"]
    for path, expected in specification["files_sha256"].items():
        if sha(path) != expected:
            raise ValueError("Preserved older adapter changed")
    model = load_adapter(Path(specification["checkpoint"]))
    historical = read("older_results.json")["final_validation_outputs"]
    records = read("validation_conversations.json")
    rows = [dict(item_id=r["id"], context=[t["text"] for t in r["session_memory"]], speakers=[t["speaker"] for t in r["session_memory"]], message=r["message"]["text"]) for r in records]
    if [r["item_id"] for r in rows] != [r["item_id"] for r in historical]:
        raise ValueError("Historical replay item ordering changed")
    english = generate(model, rows, "STANDARD", "beam4", batch_size=50)
    local = generate(model, rows, "LOCAL_REWRITE", "beam4", [r["text"] for r in english], batch_size=50)
    differences = [dict(item_id=old["item_id"], old_english=old["generated_standard"], replay_english=e["text"], old_local=old["generated_local"], replay_local=l["text"])
                   for old, e, l in zip(historical, english, local, strict=True) if old["generated_standard"] != e["text"] or old["generated_local"] != l["text"]]
    R["older_reference"] = dict(checkpoint=specification["checkpoint"], historical_pairs=50,
                                exact_historical_pairs=50 - len(differences), differences=differences,
                                validation=validation(model), outputs=evaluate(model))
    for path, expected in specification["files_sha256"].items():
        if sha(path) != expected:
            raise ValueError("Older adapter mutated")
    R["older_reference"]["checkpoint_unchanged"] = True
    save()
    print("Older historical pairs reproduced:", 50 - len(differences), "/50", flush=True)
    del model
    clear()


def train(seed, arm):
    name = f"seed{seed}_{arm}"
    plan = read(f"schedules/seed{seed}.json")
    expected = R["recipe"]["schedules"][str(seed)]
    if schedule_sha256(plan) != expected["canonical_sha256"]:
        raise ValueError("Unpaired training schedule")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    model = new_task_adapter(core.load_base(ROOT / "model"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    params = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in params) != 1769472:
        raise ValueError("Shared adapter parameter count changed")
    phase = dict(status="running", seed=seed, arm=arm, initial_adapter_sha256=adapter_digest(model),
                 schedule_sha256=expected["canonical_sha256"], history=[], primary_examples_per_step=15,
                 auxiliary_examples_forwarded_per_step=1, auxiliary_loss_weight=0.0 if arm == "control" else R["recipe"]["auxiliary_loss_weight"],
                 trainable_parameters=1769472, main_rows=1535, auxiliary_rows=180, outputs={})
    R["runs"][name] = phase
    save()
    optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01, foreach=False)
    snapshots = [0, 128, 256, 512, 784]
    best = float("inf")

    def snapshot(step):
        nonlocal best
        scores = validation(model)
        path = OUT / name / f"step_{step:04}"
        model.save_pretrained(path)
        core.TOK.save_pretrained(path)
        (path / "v5_input.json").write_text(json.dumps(dict(speaker_labels=True, joint=False, rewrite_local=True, max_source=192, max_target=48)))
        phase["history"].append(dict(step=step, validation=scores, checkpoint=str(path)))
        if scores["macro_ce"] < best:
            best = scores["macro_ce"]
            phase.update(best_step=step, best_validation=scores)
        save()
        print(name, "step", step, "macroCE", scores["macro_ce"], flush=True)

    snapshot(0)
    for step, entry in enumerate(plan, 1):
        # Both arms forward identical shapes/inputs and consume identical dropout
        # RNG draws, including the zero-weight auxiliary graph in the control.
        torch.manual_seed(seed * 1_000_000 + step)
        torch.cuda.manual_seed_all(seed * 1_000_000 + step)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        examples = [MAIN[i] for i in entry["main"]]
        total_tokens = sum(len(target) for _, target in examples)
        for start in range(0, len(examples), 3):
            x, mask, labels = core.tensor(examples[start:start + 3])
            output = model(input_ids=x, attention_mask=mask, labels=labels, use_cache=False)
            loss = reply_loss(output.logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite primary loss")
            (loss * int(labels.ne(-100).sum()) / total_tokens).backward()
        before = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params] if step == 1 else None
        x, mask, labels = core.tensor([AUXILIARY[entry["auxiliary"]]])
        auxiliary = reply_loss(model(input_ids=x, attention_mask=mask, labels=labels, use_cache=False).logits, labels)
        backward_auxiliary(auxiliary, phase["auxiliary_loss_weight"])
        if step == 1:
            changed = any(not torch.equal(old, p.grad) for old, p in zip(before, params, strict=True))
            if changed != (arm == "context"):
                raise ValueError("Auxiliary intervention gradient audit failed")
            phase["auxiliary_gradient_changed_primary_accumulation"] = changed
            del before
        if any(p.grad is not None for n, p in model.named_parameters() if "lora_" not in n):
            raise ValueError("Frozen base received gradients")
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite gradients")
        optimizer.step()
        if step == 1:
            phase["first_update_finite"] = True
            save()
            print(name, "first update finite; auxiliary gradient audit passed", flush=True)
        if step in snapshots:
            snapshot(step)
    folder = OUT / name
    (folder / "final").symlink_to("step_0784", target_is_directory=True)
    (folder / "best").symlink_to(f"step_{phase['best_step']:04}", target_is_directory=True)
    phase.update(training_status="complete", optimizer_steps=784, primary_exposures=11760, auxiliary_forward_exposures=784,
                 supervised_auxiliary_exposures=0 if arm == "control" else 784, frozen_base_received_no_gradients=True)
    save()
    del optimizer, params, model, output, loss, auxiliary
    clear()
    for selection in ["final", "best"]:
        if selection == "best" and phase["best_step"] == 784:
            phase["outputs"][selection] = {m: {k: copy.deepcopy(v) for k, v in routes.items() if k in ["english", "actual_english"]} for m, routes in phase["outputs"]["final"].items()}
            phase["best_generation_reuses_final"] = True
        else:
            model = load_adapter(folder / selection)
            phase["outputs"][selection] = evaluate(model, secondary=selection == "best")
            del model
            clear()
        save()
    phase["status"] = "complete"
    save()


def main():
    global R, MAIN, AUXILIARY, VALID
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--arms", choices=["control", "context"], nargs="+", required=True)
    parser.add_argument("--replay-old", action="store_true")
    args = parser.parse_args()
    if (OUT / "results.json").exists():
        raise FileExistsError("Fresh snapshot required")
    recipe, manifest = read("recipe.json"), read("source_manifest.json")
    for path, expected in recipe["files_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("Frozen snapshot changed: " + path)
    for path, expected in manifest["files_sha256"].items():
        if sha(ROOT / "model" / path) != expected:
            raise ValueError("Adapted base changed: " + path)
    torch.set_num_threads(4)
    if "L40S" not in torch.cuda.get_device_name(0):
        raise ValueError("L40S required")
    core.TOK = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True)
    MAIN = [encode_row(row, core.TOK) for row in read("main_rows.json")]
    AUXILIARY = [encode_row(row, core.TOK) for row in read("context_rows.json")]
    VALID = {name: [encode_row(row, core.TOK) for row in values] for name, values in read("validation_rows.json").items()}
    if (len(MAIN), len(AUXILIARY), len(VALID["english"]), len(VALID["rewrite"])) != (1535, 180, 50, 42):
        raise ValueError("Frozen dataset counts changed")
    R = dict(status="running", recipe=recipe, source_manifest=manifest, runs={}, default_changed=False,
             environment=dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__, gpu=torch.cuda.get_device_name(0)))
    save()
    try:
        if args.replay_old:
            replay_old()
        for seed in args.seeds:
            for arm in args.arms:
                train(seed, arm)
        for path, expected in manifest["files_sha256"].items():
            if sha(ROOT / "model" / path) != expected:
                raise ValueError("Original base mutated")
        flags = [dict(run=name, selection=selection, method=method, route=route, item_id=row["item_id"], status=row["status"])
                 for name, run in R["runs"].items() for selection, outputs in run["outputs"].items()
                 for method, routes in outputs.items() for route, part in routes.items() for row in part if row["status"] != "complete"]
        R.update(status="complete_with_flags" if flags else "complete", original_checkpoint_unchanged=True, flagged_outputs=flags)
        save()
    except Exception as error:
        R.update(status="failed", error=str(error), traceback=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
