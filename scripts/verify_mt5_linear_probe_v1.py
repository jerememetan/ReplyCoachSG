"""Audit saved predictions against cached frozen features and linear heads."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from replycoach.generate.linear_probe_v1 import SPLITS, TASKS, metrics, validate_dataset


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def close(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name in expected:
            close(actual[name], expected[name])
    else:
        # sklearn1.9 may return float32 probabilities; JSON restores them as
        # float64, which changes log-loss rounding on recomputation.
        assert np.isclose(actual, expected, atol=1e-6, rtol=1e-6), (actual, expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    report = json.loads((root / "artifacts/results.json").read_text())
    rows = json.loads((root / "data/diagnostics/mt5_linear_probe_v1/rows.json").read_text())
    validate_dataset(rows)
    by_id = {r["item_id"]: r for r in rows}
    row_index = {r["item_id"]: i for i, r in enumerate(rows)}
    assert report["status"] == "complete" and report["source_checkpoints_unchanged"]
    assert report["frozen_encoder"] and not report["locked_test_used"] and not report["default_changed"]
    assert set(report["stages"]) == set(report["recipe"]["models"])
    summary = dict(status="verified_complete", source_checkpoints_unchanged=True,
                   frozen_encoder_trainable_parameters=0, locked_test_used=False, default_changed=False,
                   raw_report_sha256=sha(root / "artifacts/results.json"), source_root=str(root),
                   elapsed_seconds=report["elapsed_seconds"], tasks={}, artifacts_sha256={},
                   limitations=report["recipe"]["limitations"], numeric_verification_tolerance=1e-6)
    verified = 0
    for task in TASKS:
        summary["tasks"][task] = {}
        lexical = report["lexical_baseline"][task]
        assert lexical["metrics"]["heldout_development"]["accuracy"] == 0.5
        summary["tasks"][task]["lexical_control"] = lexical["metrics"]["heldout_development"]
    for stage, content in report["stages"].items():
        assert content["status"] == "complete" and content["trainable_encoder_parameters"] == 0
        cache_path = root / f"artifacts/{stage}_frozen_features.npz"
        features = np.load(cache_path, allow_pickle=False)
        assert features["item_ids"].tolist() == [r["item_id"] for r in rows]
        summary["artifacts_sha256"][cache_path.name] = sha(cache_path)
        for task, result in content["tasks"].items():
            assert len(result["validation_grid"]) == 16
            chosen = min(result["validation_grid"], key=lambda c: (-c["validation"]["accuracy"], c["validation"]["log_loss"], c["C"], c["feature"]))
            assert chosen == result["selected"]
            assert result["trainable_readout_parameters"] == 769
            predicted = result["predictions"]
            assert all(p["predicted_label"] == int(p["positive_probability"] >= 0.5) for p in predicted)
            assert {p["item_id"] for p in predicted} == {r["item_id"] for r in rows if r["task"] == task}
            path = root / f"artifacts/{stage}_{task}_linear_readout.npz"
            weights = np.load(path, allow_pickle=False)
            probability_dtype = weights["coefficient"].dtype
            for split in SPLITS:
                part = [p for p in predicted if p["split"] == split]
                assert all(p["label"] == by_id[p["item_id"]]["label"] for p in part)
                # log_loss clips with the input dtype's epsilon. Restore the
                # saved classifier's probability dtype: float32 probabilities
                # of exactly0/1 have different log-loss clipping than float64.
                probabilities = np.asarray([p["positive_probability"] for p in part], dtype=probability_dtype)
                recalculated = metrics([by_id[p["item_id"]] for p in part], probabilities)
                close(recalculated, result["metrics"][split])
            assert weights["coefficient"].shape == (1, 768) and weights["classes"].tolist() == [0, 1]
            # Match sklearn's in-place float32 scaling before its float64 linear
            # classifier. No fit or parameter changes occur in this audit.
            z = features[chosen["feature"]][[row_index[p["item_id"]] for p in predicted]].copy()
            z -= weights["mean"]
            z /= weights["scale"]
            logits = (z @ weights["coefficient"].T + weights["intercept"])[:, 0]
            probability = 1 / (1 + np.exp(-logits))
            np.testing.assert_allclose(probability, [p["positive_probability"] for p in predicted], atol=1e-6, rtol=1e-6)
            verified += len(predicted)
            summary["artifacts_sha256"][path.name] = sha(path)
            summary["tasks"][task][stage] = dict(
                selected_feature=chosen["feature"], selected_C=chosen["C"],
                saved_readout_dtype=str(probability_dtype),
                train=result["metrics"]["train"], validation=result["metrics"]["validation"],
                heldout_development=result["metrics"]["heldout_development"],
                shuffled_label_control=result["shuffled_train_labels"]["metrics"]["heldout_development"],
            )
        assert len(content["generation_examples"]) == 24
        assert {r["item_id"] for r in content["generation_examples"]} == set(report["recipe"]["generation_example_ids"])
    summary.update(verified_predictions=verified, verified_readout_files=9, cached_feature_files=3,
                   recorded_generation_examples=72, generation_flags=[dict(stage=stage, **r) for stage, part in report["stages"].items() for r in part["generation_examples"] if r["status"] != "complete"])
    target = root / "artifacts/verification.json"
    target.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(status=summary["status"], verified_predictions=verified,
                          heldout_accuracy={task: {stage: value.get("heldout_development", value).get("accuracy") for stage, value in content.items()} for task, content in summary["tasks"].items()})))


if __name__ == "__main__":
    main()
