"""Freeze a matched auxiliary-loss experiment and its older reference."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import tarfile

from transformers import AutoTokenizer
from replycoach.generate.context_pair_ablation_v1 import encode_row, extract_context_pairs, main_rows, schedule, schedule_sha256
from replycoach.generate.mt5_matched_v1 import source

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/training_diagnostics/context_pairs_v1"
TRAIN = "data/training_expansion/generated600_repair_v1/combined_train.jsonl"


def read(path):
    return json.loads((ROOT / path).read_text())


def rows(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise FileExistsError("Preserve frozen artifact: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def main():
    team = rows("data/dev/train.jsonl")
    expanded = rows(TRAIN)
    valid = rows("data/dev/validation.jsonl")
    protected = ROOT / "data/dev/train.jsonl"
    if sha(protected) != "03485c7d6b510711fbdee5f3eac0044e334d9d7a25bf3c5c86665b0646e8eee7":
        raise ValueError("Protected team data changed")
    if not (ROOT / TRAIN).read_bytes().startswith(protected.read_bytes()):
        raise ValueError("Expanded source changed original records")
    if len(team) != 190 or len(expanded) != 790 or {r["id"] for r in expanded} & {r["id"] for r in valid}:
        raise ValueError("Counts or training-validation separation changed")
    audit = read("data/training_expansion/sealion_rewrite_v1/pair_audit.json")
    if sha(ROOT / TRAIN) != audit["source_sha256"]:
        raise ValueError("Pair eligibility audit changed")
    allowed = lambda split: {r["id"] for r in audit["items"] if r["split"] == split and r["eligible"]}
    primary = main_rows(expanded, allowed("train"))
    auxiliary = extract_context_pairs(team)
    all_validation = main_rows(valid, allowed("validation"))
    validation = {"english": [r for r in all_validation if r["style"] == "STANDARD"],
                  "rewrite": [r for r in all_validation if r["style"] == "LOCAL_REWRITE"]}
    if (len(primary), len(auxiliary), len(validation["english"]), len(validation["rewrite"])) != (1535, 180, 50, 42):
        raise ValueError("Frozen objective counts changed")
    if not {r["origin_item_id"] for r in auxiliary} <= {r["id"] for r in team}:
        raise ValueError("Context extraction left protected training split")
    tokenizer = AutoTokenizer.from_pretrained("/tmp/replycoach_mt5_adapted_tokenizer", local_files_only=True)
    budgets = {}
    for name, values in [("main", primary), ("auxiliary", auxiliary), *validation.items()]:
        encoded = [encode_row(row, tokenizer) for row in values]
        budgets[name] = dict(count=len(encoded), max_source=max(len(x) for x, y in encoded), max_target=max(len(y) for x, y in encoded), target_tokens=sum(len(y) for x, y in encoded))
    new = read("data/evaluation/context_pairs_v1/inputs.json")
    reused = read("data/evaluation/sealion_full_rewrite_v1/inputs.json") + read("data/evaluation/restored_pipeline_v1/inputs.json")
    norm = lambda text: re.sub(r"\W+", "", text.casefold())
    excluded = {norm(source(r, "STANDARD")) for r in [*primary, *reused]}
    assert len(new) == 20 and len({r["pair_id"] for r in new}) == 10
    assert len({norm(source(r, "STANDARD")) for r in new}) == 20
    assert all(norm(source(r, "STANDARD")) not in excluded for r in new)
    for row in new:
        assert len(tokenizer(source(row, "STANDARD"))["input_ids"]) <= 192
        assert len(tokenizer(source(row, "LOCAL_REWRITE", row["diagnostic_anchor"]))["input_ids"]) <= 192
    older = read("artifacts/shared_intent_v1/seeds12/rewrite_s1.json")
    assert [r["id"] for r in valid] == [r["item_id"] for r in older["final_validation_outputs"]]
    older_reference = json.loads(Path("/tmp/replycoach_mt5_context_pairs_v1_older_manifest.json").read_text())
    mapping = {
        "main_rows.json": frozen(DATA / "main_rows.json", primary),
        "context_rows.json": frozen(DATA / "context_rows.json", auxiliary),
        "validation_rows.json": frozen(DATA / "validation_rows.json", validation),
        "validation_conversations.json": frozen(DATA / "validation_conversations.json", valid),
        "source_manifest.json": ROOT / "experiments/mt5_restored_pipeline_v1_source_manifest.json",
        "older_results.json": ROOT / "artifacts/shared_intent_v1/seeds12/rewrite_s1.json",
        "baseline_results.json": ROOT / "artifacts/mt5_restored_pipeline_v1/results.json",
        "lora_results.json": ROOT / "artifacts/mt5_adapter_switch_v1/results.json",
    }
    schedules = {}
    for seed in [1, 2]:
        plan = schedule(1535, 180, seed)
        mapping[f"schedules/seed{seed}.json"] = frozen(DATA / f"schedule_seed{seed}.json", plan)
        schedules[str(seed)] = dict(canonical_sha256=schedule_sha256(plan), optimizer_steps=784,
                                   main_exposures=11760, auxiliary_forward_exposures=784,
                                   main_style_exposures=dict(Counter(primary[i]["style"] for step in plan for i in step["main"])))
    files = sorted(ROOT.glob("src/**/*.py")) + [ROOT / p for p in [
        "scripts/run_mt5_context_pairs_v1.py", "scripts/prepare_mt5_context_pairs_v1.py", "scripts/run_mt5_adapter_switch_v1.py",
        "scripts/v5_mt5_context_control_v1.sbatch", "scripts/v5_mt5_context_treatment_v1.sbatch", "scripts/v5_mt5_context_seed2_v1.sbatch",
        "tests/test_context_pair_ablation_v1.py", "tests/test_mt5_adapter_switch_v1.py", "tests/test_mt5_restored_pipeline_v1.py",
        "data/dev/train.jsonl", TRAIN, "data/dev/validation.jsonl",
        "data/training_expansion/sealion_rewrite_v1/pair_audit.json", "data/training_expansion/sealion_fidelity_v1/challenge.json",
        "data/evaluation/sealion_full_rewrite_v1/inputs.json", "data/evaluation/restored_pipeline_v1/inputs.json",
        "data/evaluation/context_pairs_v1/inputs.json", "data/evaluation/context_pairs_v1/manifest.json",
    ]]
    mapping.update({str(p.relative_to(ROOT)): p for p in files})
    recipe = dict(
        experiment="mt5_context_pairs_v1", question="Does supervised learning from the original180 context pairs help under a shared-adapter recipe, with other factors and primary exposures matched?",
        base=read("experiments/mt5_restored_pipeline_v1_source_manifest.json")["source_path"],
        topology="One shared rank16q/v LoRA(alpha32/dropout0.1),1,769,472trainable, frozen582,401,280base; mixed STANDARD and LOCAL_REWRITE tasks; same adapter generates English then rewrites it.",
        main_pool=dict(english=790, eligible_rewrite=745, total=1535, source=TRAIN),
        auxiliary_pool=dict(original_team_context_pairs=180, possible_full790_context_pairs=len(extract_context_pairs(expanded)), source="Unchanged original190 training conversations only", tag="REPLY"),
        auxiliary_loss_weight=180 / 1535, objective="Lmain_token_mean + coefficient*Laux_token_mean. Coefficient0 for control,180/1535 for context arm; primary coefficient1 in both.",
        matching="Identical15main examples and1auxiliary example forwarded per optimizer step;5main microbatches of3, then1auxiliary microbatch. Auxiliary graph executes with zero loss weight in control. Same per-step seeded dropout, adapter initialization, primary target-token normalization, LR and optimizer updates.",
        seeds=[1, 2], schedules=schedules, optimizer=dict(name="AdamW", learning_rate=3e-4, weight_decay=0.01, gradient_clip=1.0),
        budget=dict(steps=784, main_per_step=15, auxiliary_per_step=1, primary_examples=11760, no_early_stopping=True, precision="FP32"),
        checkpoint_selection=dict(primary="Final update784 fixed before outcomes", secondary="Minimum0.5English+0.5reference-anchored rewrite token CE at steps0/128/256/512/784; no stopping or output-based selection"),
        retained_steps=[0,128,256,512,784], older_reference=dict(**older_reference, job_id=351130, generation_reference="Older final rewrite_s1, seed1, original190 with different historical paired targets and180contextpairs; reference, not an isolated causal comparison"),
        evaluation=dict(existing=106, frozen_before_training_new=20, fact_change_pairs=10,
                        primary_routes=["english", "actual_english", "shared_full_english", "shared_lora_english", "diagnostic", "challenge"],
                        secondary_routes=["english", "actual_english"], methods=["greedy", "beam4_no_repeat3"], max_new_tokens=48,
                        expected_primary_outputs=1116, expected_secondary_outputs=504,
                        criteria=["Contextual appropriateness, allowing valid alternative replies", "Chosen-English-to-LOCAL meaning preservation", "Natural Singlish; no mandatory particles"], locked_test_used=False),
        budgets=budgets, team_originals_unchanged=True, default_changed=False,
        limitations=["This isolates auxiliary supervision under the shared-adapter/current790-target recipe; it does not isolate changes against the earlier two-adapter runs or recreate the historical190 target set.",
                     "Reused validation and assistant-authored new development, not independent human gold or unbiased project test accuracy.",
                     "A finite784-update budget and one auxiliary coefficient cannot rule out other training schedules or mixing ratios.",
                     "Different example lengths mean unequal work among optimizer steps, but each paired arm forwards the exact same batches/shapes; both retain gradients through the auxiliary graph.",
                     "Old final reference must pass or explicitly report historical50-pair replay differences before being treated as reproduced."],
        files_sha256={name: sha(path) for name, path in mapping.items()},
    )
    target = frozen(ROOT / "experiments/mt5_context_pairs_v1_recipe.json", recipe)
    with tarfile.open("/tmp/replycoach_mt5_context_pairs_v1.tgz", "w:gz") as archive:
        for name, path in mapping.items():
            archive.add(path, arcname=name)
        archive.add(target, arcname="recipe.json")
    print(json.dumps(dict(files=len(mapping), budgets=budgets, schedules=schedules, context_weight=recipe["auxiliary_loss_weight"], evaluation_cases=126)))


if __name__ == "__main__":
    main()
