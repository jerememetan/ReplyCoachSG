import importlib
import json
import subprocess
import sys
from dataclasses import replace

import pytest

from replycoach.io import ROOT, load_items, write_jsonl
from replycoach.retrieval.v0 import V0Retriever


def evaluation():
    return importlib.import_module("replycoach.evaluation")


@pytest.fixture
def dev_inputs(tmp_path):
    rows = [item.to_dict() for item in load_items()[:3]]
    for row in rows:
        row.update(split="dev", status="completed", author="Author", reviewer="Reviewer")
    queries = tmp_path / "validation.jsonl"
    bank = tmp_path / "train.jsonl"
    write_jsonl(queries, rows[:1])
    write_jsonl(bank, rows[1:])
    return queries, bank


def test_evaluation_accepts_disjoint_reviewed_dev_files_without_editing_them(dev_inputs):
    queries, bank = dev_inputs
    before = queries.read_bytes(), bank.read_bytes()
    inputs = evaluation().load_evaluation_inputs(queries, bank)
    assert len(inputs.queries.items) == 1
    assert len(inputs.bank.items) == 2
    assert (queries.read_bytes(), bank.read_bytes()) == before


@pytest.mark.parametrize("field,value", [("split", "test"), ("split", "unassigned"),
                                       ("status", "draft_needs_team_review"), ("privacy_ok", False)])
def test_evaluation_rejects_test_unassigned_unreviewed_or_unsafe_queries(dev_inputs, field, value):
    queries, bank = dev_inputs
    row = json.loads(queries.read_text(encoding="utf-8"))
    row[field] = value
    write_jsonl(queries, [row])
    with pytest.raises(ValueError):
        evaluation().load_evaluation_inputs(queries, bank)


def test_evaluation_rejects_query_ids_present_in_the_bank(dev_inputs):
    queries, bank = dev_inputs
    queries.write_bytes(bank.read_bytes())
    with pytest.raises(ValueError, match="overlap"):
        evaluation().load_evaluation_inputs(queries, bank)


def test_evaluation_rejects_copied_inputs_even_if_ids_differ(dev_inputs):
    queries, bank = dev_inputs
    row = json.loads(bank.read_text(encoding="utf-8").splitlines()[0])
    row["id"] = "REIDENTIFIED"
    write_jsonl(queries, [row])
    with pytest.raises(ValueError, match="overlap"):
        evaluation().load_evaluation_inputs(queries, bank)


def test_export_contains_three_honest_modes_and_sources_without_global_logging(dev_inputs, stub_generator, monkeypatch):
    from replycoach import systems

    def forbidden_log(*args, **kwargs):
        pytest.fail("Evaluation must not append to the global runtime log")

    monkeypatch.setattr(systems, "log_event", forbidden_log)
    inputs = evaluation().load_evaluation_inputs(*dev_inputs)
    retriever = V0Retriever.from_items(inputs.bank.items)
    rows = evaluation().collect_dev_outputs(
        inputs, retriever_factory=lambda: retriever, generator_factory=lambda: stub_generator,
    )
    assert len(rows) == 3
    assert {row["requested_mode"] for row in rows} == {"retrieval_only", "generator_only", "hybrid"}
    assert all(row["requested_mode"] == row["effective_mode"] and not row["errors"] for row in rows)
    assert all(row["bank_fingerprint"] == inputs.bank.fingerprint for row in rows)
    assert len(stub_generator.calls) == 1
    assert all(row["item_id"] == inputs.queries.items[0].id for row in rows)
    assert all("replies" not in row for row in rows)  # no gold answer included


def test_export_refuses_partial_comparisons_when_model_is_pending(dev_inputs):
    inputs = evaluation().load_evaluation_inputs(*dev_inputs)
    with pytest.raises(ValueError, match="incomplete"):
        evaluation().collect_dev_outputs(
            inputs, retriever_factory=lambda: V0Retriever.from_items(inputs.bank.items),
            generator_factory=None,
        )


def test_output_is_utf8_and_never_overwrites_existing_file(tmp_path):
    path = tmp_path / "results.jsonl"
    evaluation().write_dev_outputs(path, [{"reply": "可以, lunch sounds good."}])
    assert json.loads(path.read_text(encoding="utf-8"))["reply"].startswith("可以")
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        evaluation().write_dev_outputs(path, [{"reply": "replacement"}])
    assert path.read_bytes() == before


def test_output_cannot_replace_an_input(dev_inputs):
    queries, bank = dev_inputs
    before = queries.read_bytes(), bank.read_bytes()
    with pytest.raises(ValueError, match="input"):
        evaluation().write_dev_outputs(queries, [{"reply": "bad"}], protected_paths=(queries, bank))
    assert (queries.read_bytes(), bank.read_bytes()) == before


def test_serialization_error_does_not_leave_partial_output(tmp_path):
    path = tmp_path / "results.jsonl"
    with pytest.raises((TypeError, ValueError)):
        evaluation().write_dev_outputs(path, [{"reply": "ok"}, {"bad": float("nan")}])
    assert not path.exists()


def test_failed_generation_cannot_publish_a_report(dev_inputs, stub_generator, tmp_path):
    inputs = evaluation().load_evaluation_inputs(*dev_inputs)
    stub_generator.outputs[0] = replace(stub_generator.outputs[0], candidate_text="<unk>")
    output = tmp_path / "results.jsonl"
    with pytest.raises(ValueError, match="incomplete"):
        rows = evaluation().collect_dev_outputs(
            inputs, retriever_factory=lambda: V0Retriever.from_items(inputs.bank.items),
            generator_factory=lambda: stub_generator,
        )
        evaluation().write_dev_outputs(output, rows)
    assert not output.exists()


def run_export_cli(*args):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/export_dev_outputs.py"), *map(str, args)],
        capture_output=True, text=True, encoding="utf-8", cwd=ROOT,
    )


def test_cli_check_only_validates_data_without_loading_a_pending_model(dev_inputs):
    queries, bank = dev_inputs
    result = run_export_cli("--queries", queries, "--bank", bank, "--check-only")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["queries"] == 1


def test_cli_pending_model_does_not_create_output(dev_inputs, tmp_path, monkeypatch):
    monkeypatch.setenv("REPLYCOACH_V4_CHECKPOINT", str(tmp_path / "missing_v4.pt"))
    queries, bank = dev_inputs
    output = tmp_path / "pending.jsonl"
    result = run_export_cli("--queries", queries, "--bank", bank, "--output", output)
    assert result.returncode != 0
    assert "Role C" in result.stderr
    assert not output.exists()


def test_cli_names_role_a_when_default_development_files_are_missing():
    result = run_export_cli("--queries", ROOT / "data/dev/not_yet_delivered.jsonl", "--check-only")
    assert result.returncode != 0
    assert "Role A" in result.stderr


def test_retrieval_only_export_does_not_require_a_generator(dev_inputs):
    inputs = evaluation().load_evaluation_inputs(*dev_inputs)

    def forbidden_generator():
        pytest.fail("Retrieval-only export must not load a generator")

    rows = evaluation().collect_dev_outputs(
        inputs, mode="retrieval_only", generator_factory=forbidden_generator,
        retriever_factory=lambda: V0Retriever.from_items(inputs.bank.items),
    )
    assert len(rows) == 1
    assert rows[0]["effective_mode"] == "retrieval_only"
    assert rows[0]["generation_count"] == 0


def test_cli_retrieval_only_can_export_before_generator_handoff(dev_inputs, tmp_path):
    queries, bank = dev_inputs
    output = tmp_path / "retrieval.jsonl"
    result = run_export_cli("--queries", queries, "--bank", bank, "--mode", "retrieval_only", "--output", output)
    assert result.returncode == 0, result.stderr
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["requested_mode"] == row["effective_mode"] == "retrieval_only"


def test_collector_cannot_bypass_dev_validation_with_manually_constructed_inputs():
    from replycoach.reply_bank import load_reply_bank

    inputs = evaluation().EvaluationInputs(load_reply_bank(), load_reply_bank())
    with pytest.raises(ValueError, match="dev"):
        evaluation().collect_dev_outputs(inputs, mode="retrieval_only", retriever_factory=None, generator_factory=None)


def test_exclusive_publication_preserves_a_file_created_concurrently(tmp_path, monkeypatch):
    module = evaluation()
    target = tmp_path / "results.jsonl"
    actual_link = module.os.link

    def concurrent_writer(source, destination):
        Path(destination).write_text("other run", encoding="utf-8")
        return actual_link(source, destination)

    from pathlib import Path

    monkeypatch.setattr(module.os, "link", concurrent_writer)
    with pytest.raises(FileExistsError):
        module.write_dev_outputs(target, [{"reply": "new"}])
    assert target.read_text(encoding="utf-8") == "other run"
    assert list(tmp_path.glob("*.tmp")) == []
