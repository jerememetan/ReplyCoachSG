import importlib.util
import json
import random
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from test_train_v5 import tiny_model

from replycoach.generate.spans_v5 import corrupt, train_spans

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_singlish_text.py"
spec = importlib.util.spec_from_file_location("prepare_singlish_text", SCRIPT)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)

TEXTS = [
    "eh later go makan at canteen you free anot",
    "wat time e post office close i wan to renew lor",
    "den shall i join u oso can gif me a lift",
    "library got seat anot come level three",
    "i reach already you where sia",
    "project meeting tmr can but after lunch ok",
]
SENTINELS = [f"<extra_id_{index}>" for index in range(100)]


def refill(source: list[int], target: list[int], sentinels: list[int]) -> list[int]:
    """Put each hidden span back where its sentinel sits in the input."""
    spans: dict[int, list[int]] = {}
    current = None
    for token in target:
        if token in sentinels:
            current = spans.setdefault(token, [])
        elif current is not None:
            current.append(token)
    return [piece for token in source for piece in (spans[token] if token in sentinels else [token])]


def test_corrupt_hides_spans_that_the_target_restores():
    ids = list(range(10, 50))
    sentinels = list(range(1000, 900, -1))
    source, target = corrupt(ids, random.Random(7), sentinels, eos_id=1)

    assert source[-1] == 1 and target[-1] == 1
    assert target[0] == sentinels[0]
    hidden = len(target) - 1 - sum(token in sentinels for token in target)
    assert 0 < hidden <= 0.3 * len(ids)
    assert refill(source[:-1], target[:-1], sentinels) == ids


def test_prepare_packs_messages_and_drops_private_ones(tmp_path: Path):
    assert prepare.pack(["a b", "c d e", "f"], max_words=4) == ["a b", "c d e | f"]
    assert not prepare.usable("call me at 91234567")
    assert not prepare.usable("in <#> mins")
    assert prepare.usable("ok can lah")

    xml = "".join(
        f'<message id="{index}"><text>{text}</text><source><userProfile><userID>{user}</userID>'
        f"<country>{country}</country></userProfile></source></message>"
        for index, (text, user, country) in enumerate([
            ("Bugis oso near wat", "51", "SG"), ("call 91234567", "51", "SG"),
            ("Den only weekdays got", "52", "Singapore"), ("see you at 5", "60", "India"),
        ])
    )
    archive = tmp_path / "sms.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("smsCorpus_en.xml", f"<smsCorpus>{xml}</smsCorpus>")

    assert prepare.nus_messages(archive) == {"nus-51": ["Bugis oso near wat"], "nus-52": ["Den only weekdays got"]}


def write_texts(path: Path, texts: list[str]) -> Path:
    rows = [{"source": "nus_sms" if index % 2 else "cosem", "group": f"g{index}", "text": text}
            for index, text in enumerate(texts)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_train_spans_saves_best_model_and_examples(tmp_path: Path):
    model_name = tiny_model(tmp_path / "tiny", [*TEXTS, *SENTINELS])
    train = write_texts(tmp_path / "train.jsonl", TEXTS)
    validation = write_texts(tmp_path / "validation.jsonl", TEXTS[:2])
    settings = dict(model_name=model_name, epochs=2, batch_size=2, warmup_steps=1, max_tokens=16, eval_every=2,
                    output_name="tiny_spans", output_dir=tmp_path / "artifacts")

    report = train_spans(train, validation, **settings)

    assert report["train_chunks"] == 6
    assert report["validation_chunks_by_source"] == {"cosem": 1, "nus_sms": 1}
    assert report["history"][0]["step"] == 0
    assert [entry["step"] for entry in report["history"][1:]] == [2, 3, 4, 6]
    assert {"validation_loss_cosem", "validation_loss_nus_sms"} <= set(report["history"][1])
    assert (tmp_path / "artifacts" / "tiny_spans" / "best" / "config.json").exists()
    assert len(report["examples"]) == 2 and "<extra_id_0>" in report["examples"][0]["input"]
    with pytest.raises(ValueError, match="already exists"):
        train_spans(train, validation, **settings)
