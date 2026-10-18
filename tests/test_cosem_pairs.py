import importlib.util
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_cosem_pairs.py"
spec = importlib.util.spec_from_file_location("prepare_cosem_pairs", SCRIPT)
cosem = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cosem)


def tagged(chat: str, number: int, speaker: str, text: str) -> str:
    return f"<<COSEM:{chat}-{number}-{speaker}-2019>>\t{text}"


def test_read_lines_orders_chunks_and_parse_joins_continuation_lines(tmp_path: Path):
    archive = tmp_path / "corpus.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("CoSEM_v5/COSEM_v5_chunk_10.txt", tagged("19CF01", 3, "21SGCHF00", "ok see you"))
        handle.writestr(
            "CoSEM_v5/COSEM_v5_chunk_2.txt",
            "﻿" + tagged("19CF01", 1, "21SGCHF00", "Meeting venue") + "\nClementi MRT\n"
            + tagged("19CF01", 2, "22SGCHM‬01", ": can lah"),
        )
        handle.writestr("__MACOSX/CoSEM_v5/._COSEM_v5_chunk_2.txt", "junk")

    chats = cosem.parse_chats(cosem.read_lines(archive))

    assert chats == {"19CF01": [
        (1, "21SGCHF00", "Meeting venue Clementi MRT"),
        (2, "22SGCHM01", "can lah"),
        (3, "21SGCHF00", "ok see you"),
    ]}


def test_pairs_merge_a_speakers_lines_and_stop_at_numbering_holes():
    messages = [
        (1, "A", "eh you all free later"),
        (2, "B", "why"),
        (3, "B", "what time"),
        (4, "A", "want to study together at library"),
        (5, "A", "maybe 3pm"),
        (20, "B", "anyone seen my charger around here"),
        (21, "A", "no leh never see it"),
    ]

    rows = cosem.pairs_from_chat("19CF01", messages)

    assert rows == [
        {"context": [], "message": "eh you all free later",
         "response": "why what time", "source": "cosem", "chat": "19CF01"},
        {"context": ["eh you all free later"], "message": "why what time",
         "response": "want to study together at library maybe 3pm", "source": "cosem", "chat": "19CF01"},
        {"context": [], "message": "anyone seen my charger around here",
         "response": "no leh never see it", "source": "cosem", "chat": "19CF01"},
    ]


def test_pairs_skip_placeholders_short_mandarin_and_vulgar_replies():
    def responses(reply: str, message: str = "later want go eat or not") -> list[str]:
        return [row["response"] for row in cosem.pairs_from_chat("c", [(1, "A", message), (2, "B", reply)])]

    assert responses("can lah see you there") == ["can lah see you there"]
    assert responses("ok can") == []
    assert responses("😂 😂 😂 lol") == []
    assert responses("check this out {{URL}} now") == []
    assert responses("ask J.T/F/CH/21 lah she knows") == []
    assert responses("我 也 想 去 吃饭 leh") == []
    assert responses("wtf why so early one") == []
    assert responses("can lah see you there", message="<Media omitted>") == []
    assert responses("can lah see you there", message="😂😂") == []


def test_clip_keeps_whole_lines_nearest_the_reply():
    lines = ["one two three", "four five", "six"]
    assert cosem.clip(lines, 3, "tail") == "four five six"
    assert cosem.clip(lines, 3, "head") == "one two three"
    assert cosem.clip(lines, 2, "head") is None


def test_split_holds_out_whole_chats_and_caps_stock_replies():
    pairs_by_chat = {
        f"chat{index:02d}": [
            {"context": [], "message": f"message {index} {number}", "response": f"reply number {index} {number}",
             "source": "cosem", "chat": f"chat{index:02d}"}
            for number in range(40)
        ] + [
            {"context": [], "message": f"thanks {index} {number}", "response": "ok no problem lah",
             "source": "cosem", "chat": f"chat{index:02d}"}
            for number in range(10)
        ]
        for index in range(30)
    }

    train, validation, validation_chats = cosem.split_pairs(pairs_by_chat, max_train=600, max_validation=60, seed=425)

    assert validation and len(validation) <= 60 and len(train) <= 600
    assert {row["chat"] for row in validation} <= set(validation_chats)
    assert not {row["chat"] for row in train} & set(validation_chats)
    assert sum(row["response"] == "ok no problem lah" for row in train) == cosem.MAX_RESPONSE_REPEATS
    assert cosem.split_pairs(pairs_by_chat, 600, 60, seed=425)[0] == train
