import copy
import importlib
import json
import subprocess
import sys
from hashlib import sha256

import pytest

from replycoach.evaluation import collect_dev_outputs, load_evaluation_inputs
from replycoach.io import DATA_DIR, ROOT, load_items, read_jsonl, write_jsonl
from replycoach.retrieval.v0 import V0Retriever


def judging():
    return importlib.import_module("replycoach.judging")


@pytest.fixture
def exported_dev(tmp_path):
    rows = [item.to_dict() for item in load_items()[:6]]
    for row in rows:
        row.update(split="dev", status="completed", author="PRIVATE_AUTHOR",
                   reviewer="PRIVATE_REVIEWER", notes="PRIVATE_NOTES")
    queries, bank = tmp_path / "validation.jsonl", tmp_path / "train.jsonl"
    write_jsonl(queries, rows[:2])
    write_jsonl(bank, rows[2:])
    inputs = load_evaluation_inputs(queries, bank)
    retriever = V0Retriever.from_items(inputs.bank.items)
    outputs = collect_dev_outputs(inputs, mode="retrieval_only",
                                  retriever_factory=lambda: retriever, generator_factory=None)
    path = tmp_path / "BM25 export with spaces.jsonl"
    write_jsonl(path, outputs)
    return queries, bank, path, outputs


def packet(exported_dev, *, seed=425, items=None, systems=None):
    queries, bank, path, _ = exported_dev
    module = judging()
    return module.prepare_judging_packet(
        queries, bank, systems or [module.SystemExport("BM25", path, "retrieval_only")],
        seed=seed, items=items,
    )


def test_packet_blinds_metadata_preserves_both_styles_and_original_context(exported_dev):
    public, private = packet(exported_dev)
    queries, bank, path, outputs = exported_dev
    inputs = load_evaluation_inputs(queries, bank)
    by_id = {item.id: item for item in inputs.queries.items}
    mappings = {row["candidate_id"]: row for row in private}
    original = {row["item_id"]: row for row in outputs}
    assert len(public) == len(private) == 4
    assert len({row["conversation_id"] for row in public}) == 2
    assert {row["style"] for row in public} == {"STANDARD", "LOCAL"}
    for row in public:
        assert set(row) == {"schema_version", "candidate_id", "conversation_id",
                            "conversation", "incoming_message", "style", "reply"}
        mapping = mappings[row["candidate_id"]]
        item = by_id[mapping["item_id"]]
        assert mapping["conversation_id"] == row["conversation_id"]
        assert row["conversation"] == [dict(speaker=t.speaker, text=t.text) for t in item.session_memory]
        assert row["incoming_message"] == {"speaker": "peer", "text": item.message.text}
        assert row["reply"] == original[item.id][row["style"].lower()]["candidate_text"]
        assert mapping["system"] == "BM25"
        assert mapping["source_export_sha256"] == sha256(path.read_bytes()).hexdigest()
        assert mapping["bank_fingerprint"] == inputs.bank.fingerprint
        assert mapping["candidate"] == original[item.id][row["style"].lower()]
    assert "PRIVATE_" not in json.dumps(public)


def test_external_export_records_only_filename_not_personal_directories(exported_dev):
    _, private = packet(exported_dev)
    path = exported_dev[2]
    assert all(row["source_export"] == path.name for row in private)
    assert all(row["source_export_sha256"] == sha256(path.read_bytes()).hexdigest()
               for row in private)


def test_internal_export_records_forward_slash_repo_relative_path(exported_dev, tmp_path, monkeypatch):
    module = judging()
    repo = tmp_path / "repo"
    export = repo / "experiments" / "logs" / "bm25 output.jsonl"
    write_jsonl(export, exported_dev[3])
    monkeypatch.setattr(module, "ROOT", repo, raising=False)
    _, private = packet(exported_dev, systems=[module.SystemExport("BM25", export, "retrieval_only")])
    assert all(row["source_export"] == "experiments/logs/bm25 output.jsonl" for row in private)
    assert all(row["source_export_sha256"] == sha256(export.read_bytes()).hexdigest()
               for row in private)


def test_every_system_has_identical_coverage_even_for_a_seeded_pilot(exported_dev):
    path = exported_dev[2]
    module = judging()
    systems = [module.SystemExport("BM25", path, "retrieval_only"),
               module.SystemExport("Baseline replay", path, "retrieval_only")]
    first = packet(exported_dev, systems=systems, items=1)
    assert first == packet(exported_dev, systems=systems, items=1)
    public, private = first
    assert len(public) == 4
    assert len({row["conversation_id"] for row in public}) == 1
    coverage = [{(r["item_id"], r["style"]) for r in private if r["system"] == s.label} for s in systems]
    assert coverage[0] == coverage[1]
    assert packet(exported_dev, systems=systems, seed=426) != packet(exported_dev, systems=systems)


@pytest.mark.parametrize("case", ["missing", "extra", "duplicate", "fingerprint", "bank",
                                    "message", "partial", "error", "empty_reply", "style",
                                    "mixed_memory", "mixed_method", "count", "record_type",
                                    "string_score", "boolean_score"])
def test_invalid_exports_fail_instead_of_silently_dropping_or_relabelling(exported_dev, case):
    queries, bank, path, outputs = exported_dev
    rows = copy.deepcopy(outputs)
    if case == "missing":
        rows.pop()
    elif case == "extra":
        extra = copy.deepcopy(rows[0])
        extra["item_id"] = "UNKNOWN_QUERY"
        rows.append(extra)
    elif case == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif case == "fingerprint":
        rows[0]["queries_fingerprint"] = "wrong"
    elif case == "bank":
        rows[0]["bank_fingerprint"] = "wrong"
    elif case == "message":
        rows[0]["message"] = "not the evaluated input"
    elif case == "partial":
        rows[0]["effective_mode"] = "fallback_only"
    elif case == "error":
        rows[0]["errors"] = {"retrieval": "failed"}
    elif case == "empty_reply":
        rows[0]["local"]["candidate_text"] = " "
    elif case == "style":
        rows[0]["standard"]["style"] = "LOCAL"
    elif case == "mixed_memory":
        rows[0]["memory_mode"] = "M0"
    elif case == "mixed_method":
        rows[0]["retrieval_method"] = "minilm"
    elif case == "count":
        rows[0]["retrieval_count"] = 0
    elif case == "record_type":
        rows[0]["record_type"] = "final_test_comparison"
    elif case == "string_score":
        rows[0]["standard"]["score"] = "NaN"
    elif case == "boolean_score":
        rows[0]["standard"]["score"] = True
    write_jsonl(path, rows)
    with pytest.raises(ValueError):
        packet(exported_dev)


@pytest.mark.parametrize("field,value", [("split", "test"), ("split", "unassigned"),
                                        ("privacy_ok", False), ("status", "draft_needs_team_review")])
def test_packet_keeps_existing_development_only_guards(exported_dev, field, value):
    queries = exported_dev[0]
    rows = read_jsonl(queries)
    rows[0][field] = value
    write_jsonl(queries, rows)
    with pytest.raises(ValueError):
        packet(exported_dev)


@pytest.mark.parametrize("items", [0, -1, 3, True])
def test_invalid_pilot_sizes_are_rejected(exported_dev, items):
    with pytest.raises(ValueError):
        packet(exported_dev, items=items)


def test_duplicate_system_labels_and_unknown_modes_are_rejected(exported_dev):
    module = judging()
    path = exported_dev[2]
    with pytest.raises(ValueError):
        packet(exported_dev, systems=[module.SystemExport("BM25", path, "retrieval_only"),
                                      module.SystemExport("bm25", path, "retrieval_only")])
    with pytest.raises(ValueError):
        packet(exported_dev, systems=[module.SystemExport("BM25", path, "invented")])


def test_multi_mode_files_require_and_respect_explicit_selection(exported_dev):
    path, outputs = exported_dev[2:]
    rows = copy.deepcopy(outputs)
    for row in copy.deepcopy(outputs):
        row.update(requested_mode="hybrid", effective_mode="hybrid", generation_count=2)
        rows.append(row)
    write_jsonl(path, rows)
    public, private = packet(exported_dev)
    assert len(public) == 4
    assert all(row["requested_mode"] == "retrieval_only" for row in private)


def test_invalid_numeric_json_and_non_object_rows_are_rejected(exported_dev):
    path = exported_dev[2]
    for body in ('{"score": NaN}\n', '[]\n', '{broken}\n'):
        path.write_text(body, encoding="utf-8")
        with pytest.raises(ValueError):
            packet(exported_dev)


def test_writer_publishes_two_bound_utf8_files_and_never_overwrites(exported_dev, tmp_path):
    module = judging()
    public, private = packet(exported_dev)
    public[0]["reply"] = "可以, lunch sounds good."
    out_dir = tmp_path / "packet"
    module.write_judging_packet(out_dir, public, private)
    public_path, private_path = out_dir / "judge_inputs.jsonl", out_dir / "PRIVATE_mapping.jsonl"
    assert {p.name for p in out_dir.iterdir()} == {public_path.name, private_path.name}
    assert read_jsonl(public_path) == public
    saved_private = read_jsonl(private_path)
    assert all(row["judge_inputs_sha256"] == sha256(public_path.read_bytes()).hexdigest()
               for row in saved_private)
    assert {r["candidate_id"] for r in saved_private} == {r["candidate_id"] for r in public}
    before = public_path.read_bytes(), private_path.read_bytes()
    with pytest.raises(FileExistsError):
        module.write_judging_packet(out_dir, public, private)
    assert (public_path.read_bytes(), private_path.read_bytes()) == before


def test_invalid_packets_are_not_partially_written(exported_dev, tmp_path):
    public, private = packet(exported_dev)
    private[0]["not_json"] = float("nan")
    output = tmp_path / "bad_packet"
    with pytest.raises(ValueError):
        judging().write_judging_packet(output, public, private)
    assert not output.exists()
    private[0].pop("not_json")
    public[0]["system"] = "must stay private"
    with pytest.raises(ValueError):
        judging().write_judging_packet(output, public, private)
    assert not output.exists()


def test_writer_does_not_write_under_data(exported_dev):
    public, private = packet(exported_dev)
    output = DATA_DIR / "forbidden_judging_packet"
    assert not output.exists()
    with pytest.raises(ValueError, match="data/"):
        judging().write_judging_packet(output, public, private)
    assert not output.exists()


def run_cli(exported_dev, *extra):
    queries, bank, path, _ = exported_dev
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/prepare_judging_inputs.py"),
         "--queries", str(queries), "--bank", str(bank),
         "--system", "BM25", str(path.resolve()), "retrieval_only", *map(str, extra)],
        capture_output=True, text=True, encoding="utf-8", cwd=ROOT,
    )


def test_cli_check_only_uses_absolute_paths_and_writes_nothing(exported_dev, tmp_path):
    output = tmp_path / "dry_run"
    result = run_cli(exported_dev, "--check-only", "--out-dir", output)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["candidate_count"] == 4
    assert not output.exists()


def test_cli_pilot_creates_packet_without_a_generator_or_judging_model(exported_dev, tmp_path):
    output = tmp_path / "pilot"
    result = run_cli(exported_dev, "--items", 1, "--out-dir", output)
    assert result.returncode == 0, result.stderr
    assert len(read_jsonl(output / "judge_inputs.jsonl")) == 2
    assert json.loads(result.stdout)["status"] == "prepared"


def test_cli_validation_failure_creates_no_packet(exported_dev, tmp_path):
    write_jsonl(exported_dev[2], exported_dev[3][:1])
    output = tmp_path / "incomplete"
    result = run_cli(exported_dev, "--out-dir", output)
    assert result.returncode != 0
    assert "coverage" in result.stderr.lower()
    assert not output.exists()
