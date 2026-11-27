import json
from pathlib import Path
import subprocess
import sys

from replycoach.io import ROOT


def test_preparation_excludes_normalized_complete_validation_dialogues(tmp_path):
    # Synthetic parser fixture only, never used for model training/gold data.
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    output = tmp_path / "pairs.jsonl"
    train.write_text(json.dumps({"utterances": ["ALPHA!", "beta."]}) + "\n" +
                     json.dumps({"utterances": ["gamma", "delta"]}) + "\n", encoding="utf-8")
    validation.write_text(json.dumps({"utterances": ["alpha", "BETA"]}) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts/prepare_dailydialog_pairs.py"),
                    "--input", str(train), "--exclude-dialogues", str(validation),
                    "--output", str(output)], check=True, capture_output=True)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["message"] == "gamma" and rows[0]["response"] == "delta"
