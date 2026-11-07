"""Freeze actual token-head generation experiments, distinct from probing."""

import hashlib
import json
from pathlib import Path
import tarfile

from transformers import AutoTokenizer
from replycoach.generate.mt5_matched_v1 import encode, paired_rows, task_rows

ROOT = Path(__file__).resolve().parents[1]
TRAIN = "data/training_expansion/generated600_repair_v1/combined_train.jsonl"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def rows(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line.strip()]


def main():
    audit = read("data/training_expansion/sealion_rewrite_v1/pair_audit.json")
    team = ROOT / "data/dev/train.jsonl"
    if sha(team) != "03485c7d6b510711fbdee5f3eac0044e334d9d7a25bf3c5c86665b0646e8eee7":
        raise ValueError("Protected team data changed")
    if not (ROOT / TRAIN).read_bytes().startswith(team.read_bytes()) or sha(ROOT / TRAIN) != audit["source_sha256"]:
        raise ValueError("Audited corrected data changed")
    for path in ["artifacts/mt5_restored_pipeline_v1/results.json", "artifacts/mt5_adapter_switch_v1/results.json"]:
        if read(path)["status"] != "complete":
            raise ValueError("Expected completed baseline")
    tokenizer = AutoTokenizer.from_pretrained("/tmp/replycoach_mt5_adapted_tokenizer", local_files_only=True)
    budgets = {}
    for split, path, standard_count, rewrite_count in [
        ("train", TRAIN, 790, 745), ("validation", "data/dev/validation.jsonl", 50, 42),
    ]:
        records = rows(path)
        allowed = {r["id"] for r in audit["items"] if r["split"] == split and r["eligible"]}
        standard = [r for r in task_rows(records) if r["style"] == "STANDARD"]
        rewrite = paired_rows(records, allowed)
        if (len(standard), len(rewrite)) != (standard_count, rewrite_count):
            raise ValueError("Unexpected pair counts")
        for name, records, mode in [("standard", standard, "direct"), ("rewrite", rewrite, "rewrite")]:
            encoded = [encode(r, tokenizer, mode, 256, 64) for r in records]
            budgets[f"{name}/{split}"] = dict(count=len(encoded), max_source=max(len(x) for x, y in encoded), max_target=max(len(y) for x, y in encoded))
    files = sorted(ROOT.glob("src/**/*.py")) + [ROOT / p for p in [
        "scripts/run_mt5_adapter_switch_v1.py", "scripts/run_mt5_head_generation_v1.py",
        "scripts/prepare_mt5_head_generation_v1.py", "scripts/v5_mt5_linear_generation_v1.sbatch",
        "scripts/v5_mt5_mlp_generation_v1.sbatch", "tests/test_mt5_output_heads_v1.py", "tests/test_mt5_restored_pipeline_v1.py",
        TRAIN, "data/dev/train.jsonl", "data/dev/validation.jsonl",
        "data/training_expansion/sealion_rewrite_v1/pair_audit.json",
        "data/training_expansion/sealion_fidelity_v1/challenge.json",
        "data/evaluation/sealion_full_rewrite_v1/inputs.json",
        "data/evaluation/restored_pipeline_v1/inputs.json", "data/evaluation/restored_pipeline_v1/manifest.json",
    ]]
    mapping = {str(p.relative_to(ROOT)): p for p in files}
    mapping["source_manifest.json"] = ROOT / "experiments/mt5_restored_pipeline_v1_source_manifest.json"
    mapping["baseline_results.json"] = ROOT / "artifacts/mt5_restored_pipeline_v1/results.json"
    mapping["lora_results.json"] = ROOT / "artifacts/mt5_adapter_switch_v1/results.json"
    manifest = read("experiments/mt5_restored_pipeline_v1_source_manifest.json")
    for variant, count in [("linear", 192086016), ("mlp", 394240)]:
        recipe = dict(
            experiment=f"mt5_{variant}_generation_v1", variant=variant, seed=425,
            starting_checkpoint=manifest["source_path"], source_lineage=manifest["lineage"],
            objective="Actual autoregressive next-token reply generation; no diagnostic classification head or LoRA.",
            architecture="Train only the existing bias-free768→250112 vocabulary projection" if variant == "linear" else "Frozen vocabulary projection after identity-initialized residual768→256→768 GELU/dropout0.1 MLP; train only the MLP.",
            trainable_parameters=count, frozen_encoder_decoder=True,
            topology="Independent English and Singlish task heads, initialized separately from the same adapted base. Switch saved heads between generation and rewriting; no stacking.",
            training_file=TRAIN, standard=dict(train=790, validation=50, microbatch=2, accumulation=2),
            rewrite=dict(train=745, validation=42, microbatch=1, accumulation=8, input="Conversation plus reference English draft in training; actual chosen English draft at inference"),
            learning_rates=[3e-5, 3e-4], optimizer=dict(name="AdamW", weight_decay=0.01, clip_norm=1.0),
            selection=dict(max_epochs=12, patience=2, epoch0_eligible=True, per_stage="Minimum validation reply token CE across independently trained LR candidates; tie lower LR"),
            precision="FP32; preserve adapted independent output head and exact tokenizer.",
            training_mode="Encoder and decoder stay in eval mode with frozen parameters; linear head trains alone, or MLP alone with its own dropout. No gradient checkpointing or backbone input-grad hook needed.",
            evaluation=dict(reused_cases=82, previously_frozen_cases=24, methods=["greedy", "beam4_no_repeat3"], max_new_tokens=64,
                            routes=["english", "actual_english", "shared_mt5_english", "shared_lora_english", "shared_sea_english", "diagnostic", "fresh_diagnostic", "challenge"],
                            expected_entries=1160, locked_test_used=False),
            baselines=dict(full_english_job=356868, independent_lora_job=357048), budgets=budgets,
            team_originals_unchanged=190, default_changed=False,
            limitations=["Reused development; no independent human gold or unbiased accuracy claim.",
                         "Two-LR search uses more selection budget than the earlier single-LR LoRA run; matched3e-5 results remain separately recorded.",
                         "Backbone dropout disabled for frozen-feature head training, unlike LoRA training; this is an alternative recipe, not an isolated architecture-only comparison.",
                         "Both tasks start from the original adapted output head, not the task-trained English head; renderer conditions on English through its input.",
                         "Best head per learning rate retained, not separate weights for every epoch."],
            files_sha256={name: sha(path) for name, path in mapping.items()},
        )
        target = ROOT / f"experiments/mt5_{variant}_generation_v1_recipe.json"
        if target.exists() and json.loads(target.read_text()) != recipe:
            raise FileExistsError("Preserve frozen recipe")
        target.write_text(json.dumps(recipe, indent=2) + "\n")
        with tarfile.open(f"/tmp/replycoach_mt5_{variant}_generation_v1.tgz", "w:gz") as archive:
            for name, path in mapping.items():
                archive.add(path, arcname=name)
            archive.add(target, arcname="recipe.json")
        print(json.dumps(dict(variant=variant, trainable_parameters=count, frozen_files=len(mapping), budgets=budgets, expected_entries=1160)))


if __name__ == "__main__":
    main()
