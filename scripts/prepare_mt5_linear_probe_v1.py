"""Freeze counterbalanced synthetic diagnostics and checkpoint fingerprints."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import tarfile

from transformers import AutoTokenizer
from replycoach.generate.linear_probe_v1 import CLASS_NAMES, TASKS, dataset
from replycoach.generate.mt5_matched_v1 import source

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_frozen(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise FileExistsError("Preserve frozen artifact: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    rows = dataset()
    original = ROOT / "data/dev/train.jsonl"
    if sha(original) != "03485c7d6b510711fbdee5f3eac0044e334d9d7a25bf3c5c86665b0646e8eee7":
        raise ValueError("Original team data changed")
    data = ROOT / "data/diagnostics/mt5_linear_probe_v1/rows.json"
    write_frozen(data, rows)
    source_info = json.loads(Path("/tmp/replycoach_mt5_linear_probe_v1_sources.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained("/tmp/replycoach_mt5_adapted_tokenizer", local_files_only=True)
    texts = [source(r, "STANDARD") for r in rows]
    lengths = [len(tokenizer(text)["input_ids"]) for text in texts]
    if max(lengths) > 256 or len(set(texts)) != len(texts):
        raise ValueError("Truncation or duplicate encoder inputs")
    families = defaultdict(list)
    for row in rows:
        if row["split"] == "heldout_development":
            families[row["family"]].append(row)
    examples = []
    rng = random.Random(425)
    for family, part in sorted(families.items()):
        groups = sorted({r["contrast_group"] for r in part})
        chosen = set(rng.sample(groups, 2))
        examples.extend(r["item_id"] for r in part if r["contrast_group"] in chosen)
    if len(examples) != 24:
        raise ValueError("Expected 24 paired generation examples")
    manifest = dict(
        provenance="Assistant-authored controlled synthetic diagnostic conversations; never project generator training or independent human gold.",
        purpose="Measure linear recoverability of queried conversation facts from frozen mT5 encoder representations.",
        counts={task: dict(Counter(r["split"] for r in rows if r["task"] == task)) for task in TASKS},
        classes=CLASS_NAMES,
        controls=["Balanced labels separately in each task and split.",
                  "Disjoint template families and concrete entity strings across train, validation and reserved development.",
                  "Both positive and negative facts in every conversation; entity-label binding, fact order and query counterbalanced.",
                  "Every opposite-label counterfactual pair has identical word counts; unigram TF-IDF cannot distinguish that pair.",
                  "Labels and diagnostic metadata never enter encoder inputs.",
                  "No reads or edits to the locked project test; original190 team records untouched."],
        selection="Validation chooses feature pooling/layer and regularization; reserved development used once after selection.",
        limitations=["One synthetic pilot with 64 reserved examples (32 paired groups) per task; correlated templates and no population accuracy claim.",
                     "High accuracy shows linearly accessible information under this design, not that the generator uses it or general reasoning.",
                     "Low accuracy does not prove information is absent; pooling, layers, labels and limited data constrain the readout.",
                     "All original models are pretrained; split disjointness applies to this synthetic diagnostic, not guaranteed unseen pretraining strings."],
    )
    manifest_path = data.parent / "manifest.json"
    write_frozen(manifest_path, manifest)
    files = sorted(ROOT.glob("src/**/*.py")) + [ROOT / p for p in [
        "scripts/run_mt5_linear_probe_v1.py", "scripts/prepare_mt5_linear_probe_v1.py",
        "scripts/run_mt5_adapter_switch_v1.py", "scripts/v5_mt5_linear_probe_v1.sbatch",
        "tests/test_linear_probe_v1.py", "tests/test_mt5_adapter_switch_v1.py", "tests/test_mt5_restored_pipeline_v1.py",
    ]] + [data, manifest_path]
    mapping = {str(p.relative_to(ROOT)): p for p in files}
    recipe = dict(experiment="mt5_linear_probe_v1", seed=425, **source_info,
        backbone_training="None. All encoder, decoder and adapter parameters frozen; sklearn learns linear heads on cached features only.",
        readout="Training-only StandardScaler followed by binary logistic regression, with 768 coefficients plus one intercept per selected head.",
        feature_candidates=["layer0_mean", "layer6_mean", "layer12_mean", "layer12_last"],
        C_grid=[0.01, 0.1, 1.0, 10.0],
        selection="Highest validation accuracy, then lowest validation log loss, then lower C, then feature name. No reserved-development tuning.",
        controls=["Word-unigram TF-IDF linear classifier", "Uncontextualized layer0 mean among feature candidates", "Shuffled training labels using the selected true-label feature and C without retuning"],
        metrics=["accuracy", "macro_f1", "log_loss", "both members correct and prediction-flip rates for query and fact-binding pairs"],
        rows=len(rows), per_task=manifest["counts"], source_max=max(lengths), extraction_microbatch=16,
        generation_example_ids=sorted(examples), generation="24 preselected balanced paired reserved examples per frozen checkpoint; beam4/no-repeat3/max48, no gold inputs; verbatim qualitative diagnostics only.",
        locked_test_used=False, team_originals_unchanged=190, default_changed=False,
        limitations=manifest["limitations"],
        files_sha256={name: sha(path) for name, path in mapping.items()},
    )
    target = ROOT / "experiments/mt5_linear_probe_v1_recipe.json"
    write_frozen(target, recipe)
    with tarfile.open("/tmp/replycoach_mt5_linear_probe_v1.tgz", "w:gz") as archive:
        for name, path in mapping.items():
            archive.add(path, arcname=name)
        archive.add(target, arcname="recipe.json")
    print(json.dumps(dict(files=len(mapping), rows=len(rows), counts=manifest["counts"], max_source=max(lengths), models=list(recipe["models"]), generation_examples=len(examples))))


if __name__ == "__main__":
    main()
