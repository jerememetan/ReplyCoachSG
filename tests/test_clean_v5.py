import json
from pathlib import Path

import pytest

from replycoach.generate.clean_v5 import filter_pairs, keep_threshold, relatedness


def write_lines(path: Path, count: int) -> Path:
    rows = [{"context": [], "message": f"message {index}", "response": f"reply number {index}", "source": "cosem"}
            for index in range(count)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_relatedness_is_how_much_the_message_helps():
    assert relatedness({"fit": [-1.0, -2.0], "without": [-3.0, -2.5]}) == [2.0, 0.5]


def test_filter_keeps_the_most_related_pairs_in_their_original_order(tmp_path: Path):
    pairs = write_lines(tmp_path / "pairs.jsonl", 4)
    # Relatedness 0.1, 2.0, 1.0, -0.5: the reply in pair 1 depends most on its message.
    scores = {"fit": [-2.0, -1.0, -1.5, -3.0], "without": [-2.1, -3.0, -2.5, -2.5]}

    result = filter_pairs(pairs, scores, 0.5, tmp_path / "clean.jsonl")

    kept = [json.loads(line)["message"] for line in (tmp_path / "clean.jsonl").read_text().splitlines()]
    assert kept == ["message 1", "message 2"]
    assert result == {"pairs": 4, "kept": 2, "threshold": 1.0}


def test_filter_refuses_scores_from_another_file_and_overwriting_its_input(tmp_path: Path):
    pairs = write_lines(tmp_path / "pairs.jsonl", 3)
    with pytest.raises(ValueError, match="2 scores for 3 pairs"):
        filter_pairs(pairs, {"fit": [0.0, 0.0], "without": [0.0, 0.0]}, 0.5, tmp_path / "clean.jsonl")
    with pytest.raises(ValueError, match="new file"):
        filter_pairs(pairs, {"fit": [0.0] * 3, "without": [0.0] * 3}, 0.5, pairs)
    with pytest.raises(ValueError, match="between 0 and 1"):
        keep_threshold([1.0], 0)


def test_score_gives_each_pair_a_fit_with_and_without_its_message(tmp_path: Path):
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from test_train_v5 import COSEM, tiny_model, write_pairs

    from replycoach.generate.clean_v5 import score, scores_path

    checkpoint = Path(tiny_model(tmp_path / "tiny"))
    pairs = write_pairs(tmp_path / "cosem_train.jsonl", "cosem", COSEM)

    report = score(checkpoint, pairs, batch_size=2, max_source=32, max_target=8, output_dir=tmp_path)

    assert len(report["fit"]) == len(report["without"]) == len(COSEM)
    assert all(value < 0 for value in report["fit"] + report["without"])  # mean log-probabilities
    assert json.loads(scores_path(pairs, tmp_path).read_text())["fit"] == report["fit"]
    with pytest.raises(ValueError, match="already exists"):
        score(checkpoint, pairs, output_dir=tmp_path)
