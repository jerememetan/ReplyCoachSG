from replycoach.io import load_items, save_items


def test_save_items_writes_complete_jsonl_without_leaving_a_temp_file(tmp_path):
    source_items = load_items()
    source_items[0].reviewer = "Ava"
    source_items[0].reviewer_notes = "Checked local wording."
    destination = tmp_path / "pilot.jsonl"

    save_items(destination, source_items)

    saved_items = load_items(destination)
    assert saved_items[0].reviewer == "Ava"
    assert saved_items[0].reviewer_notes == "Checked local wording."
    assert not list(tmp_path.glob(".*.jsonl"))
