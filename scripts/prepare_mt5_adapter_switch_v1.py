"""Freeze the data, code and full-English baseline for an isolated adapter run."""

import hashlib
import json
import tarfile
from pathlib import Path

from transformers import AutoTokenizer
from replycoach.generate.mt5_matched_v1 import encode, paired_rows, source, task_rows

ROOT = Path(__file__).resolve().parents[1]
TRAIN = "data/training_expansion/generated600_repair_v1/combined_train.jsonl"


def read(path):
    return json.loads((ROOT / path).read_text())


def rows(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit = read("data/training_expansion/sealion_rewrite_v1/pair_audit.json")
    original = ROOT / "data/dev/train.jsonl"
    if sha(original) != "03485c7d6b510711fbdee5f3eac0044e334d9d7a25bf3c5c86665b0646e8eee7":
        raise ValueError("Protected original team data changed")
    if not (ROOT / TRAIN).read_bytes().startswith(original.read_bytes()) or sha(ROOT / TRAIN) != audit["source_sha256"]:
        raise ValueError("Corrected training data no longer matches audited pairs")
    baseline = read("artifacts/mt5_restored_pipeline_v1/results.json")
    if baseline["status"] != "complete" or not baseline["original_checkpoint_unchanged"]:
        raise ValueError("Expected completed, verified full-English baseline")
    tokenizer = AutoTokenizer.from_pretrained("/tmp/replycoach_mt5_adapted_tokenizer", local_files_only=True)
    manifest = read("experiments/mt5_restored_pipeline_v1_source_manifest.json")
    for name in ["tokenizer.json", "tokenizer_config.json"]:
        if sha(Path("/tmp/replycoach_mt5_adapted_tokenizer") / name) != manifest["files_sha256"][name]:
            raise ValueError("Preflight tokenizer differs from adapted source")
    budgets = {}
    for split, path, expected_standard, expected_rewrite in [
        ("train", TRAIN, 790, 745), ("validation", "data/dev/validation.jsonl", 50, 42),
    ]:
        records = rows(path)
        eligible = {r["id"] for r in audit["items"] if r["split"] == split and r["eligible"]}
        standard = [r for r in task_rows(records) if r["style"] == "STANDARD"]
        rewrite = paired_rows(records, eligible)
        if (len(standard), len(rewrite)) != (expected_standard, expected_rewrite):
            raise ValueError("Unexpected training/validation counts")
        for name, examples, mode in [("standard", standard, "direct"), ("rewrite", rewrite, "rewrite")]:
            encoded = [encode(r, tokenizer, mode, 256, 64) for r in examples]
            budgets[f"{name}/{split}"] = dict(count=len(encoded), max_source=max(len(x) for x, y in encoded), max_target=max(len(y) for x, y in encoded))
    reused = read("data/evaluation/sealion_full_rewrite_v1/inputs.json")
    fresh = read("data/evaluation/restored_pipeline_v1/inputs.json")
    challenge = read("data/training_expansion/sealion_fidelity_v1/challenge.json")
    for method in ["greedy", "beam4"]:
        previous = baseline["outputs"][method]["english"]
        if [r["item_id"] for r in previous] != [r["item_id"] for r in reused + fresh]:
            raise ValueError("Baseline/evaluation case mismatch")
        for row, draft in zip(reused + fresh, previous, strict=True):
            if len(tokenizer(source(row, "LOCAL_REWRITE", draft["text"]))["input_ids"]) > 256:
                raise ValueError("Frozen baseline draft exceeds source budget")

    files = sorted(ROOT.glob("src/**/*.py")) + [ROOT / p for p in [
        "scripts/run_mt5_adapter_switch_v1.py", "scripts/prepare_mt5_adapter_switch_v1.py",
        "scripts/v5_mt5_adapter_switch_v1.sbatch", "tests/test_mt5_adapter_switch_v1.py",
        "tests/test_mt5_restored_pipeline_v1.py", "tests/test_mt5_matched_v1.py",
        TRAIN, "data/dev/train.jsonl", "data/dev/validation.jsonl",
        "data/training_expansion/sealion_rewrite_v1/pair_audit.json",
        "data/training_expansion/sealion_fidelity_v1/challenge.json",
        "data/evaluation/sealion_full_rewrite_v1/inputs.json",
        "data/evaluation/restored_pipeline_v1/inputs.json",
        "data/evaluation/restored_pipeline_v1/manifest.json",
    ]]
    mapping = {str(p.relative_to(ROOT)): p for p in files}
    mapping["source_manifest.json"] = ROOT / "experiments/mt5_restored_pipeline_v1_source_manifest.json"
    mapping["baseline_results.json"] = ROOT / "artifacts/mt5_restored_pipeline_v1/results.json"
    recipe = dict(
        experiment="mt5_adapter_switch_v1", seed=425,
        starting_checkpoint=manifest["source_path"], starting_lineage=manifest["lineage"],
        topology="One frozen fully adapted base; independently initialized English and Singlish LoRAs. Switch exactly one active adapter per inference stage.",
        public_training="Reuse completed full-model span and style-labelled dialogue adaptation; no additional public training.",
        lora=dict(rank=16, alpha=32, dropout=0.1, target_modules=["q", "v"], trainable_parameters_per_stage=1769472),
        standard=dict(train=790, validation=50, microbatch=2, accumulation=2),
        rewrite=dict(train=745, validation=42, microbatch=1, accumulation=8, input="Conversation plus reference English draft during training; actual generated English at ordinary inference."),
        optimizer=dict(name="AdamW", learning_rate=3e-5, weight_decay=0.01, gradient_clip=1.0),
        selection=dict(max_epochs=12, patience=2, epoch_zero_eligible=True, metric="Token-averaged development reply cross entropy"),
        precision="FP32; preserve adapted tokenizer and separately stored output head",
        evaluation=dict(reused=82, fresh_previously_evaluated=24, diagnostic=32, fresh_diagnostic=24, challenge=len(challenge),
                        routes=["english", "actual_english", "shared_mt5_english", "shared_sea_english", "diagnostic", "fresh_diagnostic", "challenge"],
                        methods=["greedy", "beam4_no_repeat_ngram3"], max_new_tokens=64,
                        baseline_job=356868, expected_recorded_entries=948, locked_test_used=False),
        controls="Same source checkpoint, tokenizer, corrected data, selection policy, LR, batches and decoders as baseline356868. Both task stages now use independent adapters on the common base.",
        limitations=["Development has been repeatedly reused; no independent human gold or unbiased accuracy claim.",
                     "Equal LR is a controlled starting point, not proof of optimal LoRA learning rate.",
                     "Topology changes the renderer base as well as English trainable parameters; renderer comparison on fixed drafts is included.",
                     "45 incompatible paired intents excluded only from rewrite supervision; protected 190 original rows remain unchanged."],
        budgets=budgets, team_originals_unchanged=190, default_changed=False,
        files_sha256={name: sha(p) for name, p in mapping.items()},
    )
    destination = ROOT / "experiments/mt5_adapter_switch_v1_recipe.json"
    if destination.exists() and json.loads(destination.read_text()) != recipe:
        raise FileExistsError("Preserve previously frozen recipe")
    destination.write_text(json.dumps(recipe, indent=2) + "\n")
    with tarfile.open("/tmp/replycoach_mt5_adapter_switch_v1.tgz", "w:gz") as archive:
        for name, path in mapping.items():
            archive.add(path, arcname=name)
        archive.add(destination, arcname="recipe.json")
    print(json.dumps(dict(files=len(mapping), budgets=budgets, evaluation_entries=948, team_originals_unchanged=190)))


if __name__ == "__main__":
    main()
