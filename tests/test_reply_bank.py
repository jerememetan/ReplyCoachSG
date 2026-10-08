import copy
from pathlib import Path

import pytest

from replycoach.io import load_items, write_jsonl


def development_row():
    row = load_items()[0].to_dict()
    row.update(split="dev", status="completed", author="Author", reviewer="Reviewer")
    return row


def load_bank(path: Path | None = None):
    from replycoach.reply_bank import load_reply_bank

    return load_reply_bank(path)


def test_pilot_bank_is_identified_as_draft():
    bank = load_bank()
    assert len(bank.items) == 20
    assert "pilot" in bank.label.lower()
    assert "draft" in bank.label.lower()
    assert len(bank.fingerprint) == 64


def test_completed_development_bank_loads_without_changing_file(tmp_path):
    path = tmp_path / "dev.jsonl"
    write_jsonl(path, [development_row()])
    before = path.read_bytes()
    bank = load_bank(path)
    assert bank.items[0].split == "dev"
    assert "reviewed" in bank.label.lower()
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"split": "test"}, "split.*dev"),
        ({"split": "unassigned"}, "split.*dev"),
        ({"split": "pilot"}, "split.*dev"),
        ({"status": "draft_needs_team_review"}, "completed"),
        ({"reviewer": ""}, "reviewer"),
        ({"reviewer": " author "}, "independent"),
        ({"reviewer": None}, "reviewer"),
        ({"author": ""}, "author"),
        ({"author": "unassigned"}, "author"),
        ({"privacy_ok": False}, "privacy_ok"),
        ({"privacy_ok": "false"}, "privacy_ok"),
        ({"privacy_ok": 1}, "privacy_ok"),
    ],
)
def test_development_bank_rejects_unsafe_or_unreviewed_rows(tmp_path, changes, error):
    row = development_row()
    row.update(changes)
    path = tmp_path / "bank.jsonl"
    write_jsonl(path, [row])
    with pytest.raises(ValueError, match=error):
        load_bank(path)


@pytest.mark.parametrize("missing", ["split", "author", "privacy_ok"])
def test_development_bank_requires_explicit_fields(tmp_path, missing):
    row = development_row()
    del row[missing]
    path = tmp_path / "bank.jsonl"
    write_jsonl(path, [row])
    with pytest.raises(ValueError):
        load_bank(path)


def test_mixed_bank_is_rejected_not_silently_filtered(tmp_path):
    held_out = development_row()
    held_out.update(id="HELD-OUT", split="test")
    path = tmp_path / "mixed.jsonl"
    write_jsonl(path, [development_row(), held_out])
    with pytest.raises(ValueError, match="HELD-OUT.*split"):
        load_bank(path)


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "duplicate.jsonl"
    write_jsonl(path, [development_row(), development_row()])
    with pytest.raises(ValueError, match="duplicate"):
        load_bank(path)


def test_empty_bank_is_rejected(tmp_path):
    path = tmp_path / "empty.jsonl"
    write_jsonl(path, [])
    with pytest.raises(ValueError, match="empty"):
        load_bank(path)


def test_schema_error_identifies_record(tmp_path):
    row = development_row()
    row["replies"]["standard"] = ""
    path = tmp_path / "invalid.jsonl"
    write_jsonl(path, [row])
    with pytest.raises(ValueError, match="PILOT-001.*standard"):
        load_bank(path)


def test_content_identity_changes_when_replies_change(tmp_path):
    row = development_row()
    path = tmp_path / "dev.jsonl"
    write_jsonl(path, [row])
    first = load_bank(path)
    updated = copy.deepcopy(row)
    updated["replies"]["standard"] = "Yes, lunch first sounds good."
    write_jsonl(path, [updated])
    second = load_bank(path)
    assert first.fingerprint != second.fingerprint
    assert first.serialized != second.serialized


def test_non_object_record_reports_validation_error(tmp_path):
    path = tmp_path / "array.jsonl"
    write_jsonl(path, [[]])
    with pytest.raises(ValueError, match="object"):
        load_bank(path)
