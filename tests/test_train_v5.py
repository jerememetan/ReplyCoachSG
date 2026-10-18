import json
from pathlib import Path

import pytest

pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from transformers import MT5Config, MT5ForConditionalGeneration, PreTrainedTokenizerFast

from replycoach.generate.train_v5 import STATE_FILE, detokenise, encode_rows, load_pairs, train_pretrain

DAILYDIALOG = [
    (["Hello , how are you ?"], "Shall we have lunch ?", "Sure , where shall we go ?"),
    ([], "Is the library open today ?", "Yes , it closes at nine ."),
    (["I am running late ."], "Can you wait for me ?", "No problem , take your time ."),
    ([], "Did you finish the report ?", "Not yet , I need one more day ."),
]
COSEM = [
    (["eh where are you"], "later go makan?", "can lah meet at canteen"),
    ([], "library got seat anot", "got lah come level three"),
    (["i reach already"], "you where sia", "wait ah five more min"),
    ([], "project meeting tmr can", "can but after lunch ok"),
]


def write_pairs(path: Path, source: str, pairs) -> Path:
    rows = [{"context": context, "message": message, "response": response, "source": source}
            for context, message, response in pairs]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def tiny_model(path: Path, extra_texts: list[str] = ()) -> str:
    """A word-level tokenizer and a one-layer mT5, so the test needs no download."""
    texts = [text for pairs in (DAILYDIALOG, COSEM) for context, message, response in pairs
             for text in [*context, message, response]] + list(extra_texts)
    words = {word for text in texts for word in detokenise(text).split()}
    words |= {"STANDARD", "LOCAL", "context:", "message:", "|"}
    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2}
    for word in sorted(words):
        vocab[word] = len(vocab)
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    backend.post_processor = tokenizers.processors.TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
    PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="</s>", unk_token="<unk>"
    ).save_pretrained(path)
    config = MT5Config(
        vocab_size=len(vocab), d_model=16, d_kv=4, d_ff=32, num_layers=1, num_decoder_layers=1, num_heads=2,
        pad_token_id=0, eos_token_id=1, decoder_start_token_id=0,
    )
    MT5ForConditionalGeneration(config).save_pretrained(path)
    return str(path)


def test_detokenise_restores_normal_spacing():
    assert detokenise("Well , that ' s too far . Can you wait ?") == "Well, that's too far. Can you wait?"


def test_load_pairs_rejects_unknown_source(tmp_path: Path):
    path = write_pairs(tmp_path / "pairs.jsonl", "movie", DAILYDIALOG[:1])
    with pytest.raises(ValueError, match="source must be one of"):
        load_pairs(path)


def test_encode_rows_drops_oldest_context_before_the_message(tmp_path: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tiny_model(tmp_path / "tiny"))
    row = {
        "source": "cosem", "context": ["eh where are you", "i reach already"],
        "message": "later go makan?", "response": "can lah meet at canteen",
    }
    full, _ = encode_rows([row], tokenizer, max_source=64, max_target=8)[0]
    source, target = encode_rows([row], tokenizer, max_source=len(full) - 1, max_target=4)[0]
    text = tokenizer.decode(source)
    assert text.startswith("LOCAL context: i reach already message: later go makan?")
    assert "where" not in text
    assert source[-1] == tokenizer.eos_token_id
    assert len(target) == 4 and target[-1] == tokenizer.eos_token_id
    forced, _ = encode_rows([row], tokenizer, max_source=64, max_target=8, style="STANDARD")[0]
    assert tokenizer.decode(forced).startswith("STANDARD context:")


def test_pretrain_writes_best_model_and_resumes_mid_run(tmp_path: Path):
    model_name = tiny_model(tmp_path / "tiny")
    train = [write_pairs(tmp_path / "dd_train.jsonl", "dailydialog", DAILYDIALOG),
             write_pairs(tmp_path / "cosem_train.jsonl", "cosem", COSEM)]
    validation = [write_pairs(tmp_path / "dd_val.jsonl", "dailydialog", DAILYDIALOG[:2]),
                  write_pairs(tmp_path / "cosem_val.jsonl", "cosem", COSEM[:2])]
    settings = dict(model_name=model_name, batch_size=2, eval_every=2, warmup_steps=2, max_source=32,
                    max_target=12, output_name="tiny_run", output_dir=tmp_path / "artifacts")

    report = train_pretrain(train, validation, epochs=1, **settings)

    assert report["train_pairs_by_source"] == {"cosem": 4, "dailydialog": 4}
    assert report["style_by_source"] == {"cosem": "LOCAL", "dailydialog": "STANDARD"}
    assert [entry["step"] for entry in report["history"]] == [2, 4]
    assert {"validation_loss_cosem", "validation_loss_dailydialog"} <= set(report["history"][0])
    assert len(report["examples"]) == 4
    assert (tmp_path / "artifacts" / "tiny_run" / "best" / "config.json").exists()
    saved_report = json.loads((tmp_path / "artifacts" / "tiny_run.json").read_text(encoding="utf-8"))
    assert saved_report["best_step"] == report["best_step"]

    state = tmp_path / "artifacts" / "tiny_run" / STATE_FILE
    resumed = train_pretrain(train, validation, epochs=2, resume_from=state, **settings)

    assert resumed["resume_step"] == 4
    assert [entry["step"] for entry in resumed["history"]] == [2, 4, 6, 8]
    with pytest.raises(ValueError, match="already completed"):
        train_pretrain(train, validation, epochs=2, resume_from=state, **settings)
    with pytest.raises(ValueError, match="do not match"):
        train_pretrain(train, validation, epochs=3, resume_from=state, **{**settings, "batch_size": 4})


def test_pretrain_requires_validation_for_every_training_source(tmp_path: Path):
    train = [write_pairs(tmp_path / "dd_train.jsonl", "dailydialog", DAILYDIALOG),
             write_pairs(tmp_path / "cosem_train.jsonl", "cosem", COSEM)]
    validation = [write_pairs(tmp_path / "dd_val.jsonl", "dailydialog", DAILYDIALOG[:2])]
    with pytest.raises(ValueError, match="no validation pairs for training source"):
        train_pretrain(train, validation, output_dir=tmp_path / "artifacts")
