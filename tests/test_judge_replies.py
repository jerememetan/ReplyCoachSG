"""Small offline checks for the local judging command; no real model calls."""

import contextlib
import csv
import importlib.util
import io
import json
import os
import subprocess
import sys
import threading
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def runner():
    path = ROOT / "scripts/judge_replies.py"
    assert path.exists(), "The local judging runner has not been implemented"
    spec = importlib.util.spec_from_file_location("judge_replies", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rating(score=4, send=True):
    return dict(appropriateness=score, naturalness=score, style_fit=score,
                would_send=send, explanation="Plausible in context.",
                problem="none", uncertain=False)


def packet(folder):
    folder.mkdir()
    public, private = [], []
    for system in ("BM25", "MiniLM"):
        for style in ("STANDARD", "LOCAL"):
            cid = f"C{len(public) + 1}"
            public.append(dict(schema_version=1, candidate_id=cid, conversation_id="Q1",
                               conversation=[dict(speaker="user", text="are we meeting?")],
                               incoming_message=dict(speaker="peer", text="lunch at noon?"),
                               style=style, reply="GOOD" if system == "BM25" else "WEAK"))
            private.append(dict(schema_version=1, candidate_id=cid, conversation_id="Q1",
                                item_id="PRIVATE_ITEM", style=style, system=system,
                                requested_mode="retrieval_only", memory_mode="M2",
                                retrieval_method=system.lower()))
    public_path = folder / "judge_inputs.jsonl"
    public_path.write_text("".join(json.dumps(r) + "\n" for r in public), encoding="utf-8")
    digest = sha256(public_path.read_bytes()).hexdigest()
    for row in private:
        row["judge_inputs_sha256"] = digest
    (folder / "PRIVATE_mapping.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in private), encoding="utf-8")
    return public_path


@contextlib.contextmanager
def ollama_stub():
    state = dict(requests=[], bad=False)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = dict(models=[dict(name="fixture:7b", digest="fixed-test-digest")])
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(body)
            candidate = json.loads(body["messages"][1]["content"])
            good = candidate["reply"] == "GOOD"
            response = dict(done=True, done_reason="stop", eval_count=20,
                            message=dict(content=json.dumps(
                                {} if state["bad"] else rating(5 if good else 2, good))))
            if state["bad"] == "message":
                response["message"] = "malformed Ollama message"
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def setup_run(tmp_path, url):
    source = tmp_path / "packet"
    packet(source)
    personas = tmp_path / "personas.csv"
    personas.write_text('id,description\nP1,"A concise student."\nP2,"A chatty student."\n', encoding="utf-8")
    env = tmp_path / ".env"
    env.write_text(f'OLLAMA_MODEL="fixture:7b"\nOLLAMA_BASE_URL={url}\n', encoding="utf-8")
    return [str(source), "--personas", str(personas), "--env-file", str(env)]


def test_schema_personas_and_environment(tmp_path):
    app = runner()
    app.validate_rating(rating())
    for change in (dict(style_fit=True), dict(naturalness=4.0), dict(appropriateness=0),
                   dict(would_send="yes"), dict(uncertain="no"), dict(problem="invented"),
                   dict(explanation=""), dict(extra="not allowed")):
        with pytest.raises(ValueError):
            app.validate_rating({**rating(), **change})
    assert len(app.load_personas(ROOT / "data/evaluation/personas.csv")) == 10
    env = tmp_path / ".env"
    env.write_text('OLLAMA_MODEL="fixture:7b"\nIGNORED_VALUE=anything\n', encoding="utf-8")
    assert app.read_env(env)["OLLAMA_MODEL"] == "fixture:7b"
    for url in ("https://external.example", "http://127.0.0.1@external.example", "http://localhost/path"):
        with pytest.raises(ValueError):
            app.local_url(url)


def test_http_judging_summary_and_resume_preserve_blinding(tmp_path):
    app = runner()
    with ollama_stub() as (url, state), contextlib.redirect_stdout(io.StringIO()):
        args = setup_run(tmp_path, url)
        assert app.main(args) == 0
        assert len(state["requests"]) == 8
        for request in state["requests"]:
            assert request["stream"] is False
            assert request["format"]["properties"]["style_fit"]["type"] == "integer"
            prompt = json.dumps(request["messages"])
            assert all(secret not in prompt for secret in ("BM25", "MiniLM", "PRIVATE_ITEM"))
        report = list(csv.DictReader((tmp_path / "packet/ratings/summary.csv").open(encoding="utf-8")))
        assert len(report) == 4
        assert all(float(r["appropriateness"]) == (5 if r["system"] == "BM25" else 2) for r in report)
        assert all(float(r["would_send_pct"]) == (100 if r["system"] == "BM25" else 0) for r in report)
        assert app.main(args + ["--resume"]) == 0
        assert len(state["requests"]) == 8
        assert app.main(args + ["--summary-only"]) == 0
        (tmp_path / "personas.csv").write_text('id,description\nP1,"Changed."\n', encoding="utf-8")
        with pytest.raises(ValueError, match="changed"):
            app.main(args + ["--resume"])
        assert len(state["requests"]) == 8


def test_bad_responses_are_recorded_not_scored_and_can_be_retried(tmp_path):
    app = runner()
    with ollama_stub() as (url, state), contextlib.redirect_stdout(io.StringIO()):
        args = setup_run(tmp_path, url)
        state["bad"] = True
        assert app.main(args) == 1
        journal = tmp_path / "packet/ratings/ratings.jsonl"
        failed = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
        assert len(failed) == 8
        assert all(r["status"] == "error" and r["rating"] is None and r["attempts"] == 2 for r in failed)
        state["bad"] = False
        assert app.main(args + ["--resume"]) == 0
        assert len(journal.read_text(encoding="utf-8").splitlines()) == 16


def test_corrupt_packet_is_rejected_before_model_calls(tmp_path):
    app = runner()
    with ollama_stub() as (url, state):
        args = setup_run(tmp_path, url)
        public = tmp_path / "packet/judge_inputs.jsonl"
        public.write_text(public.read_text(encoding="utf-8").replace("GOOD", "CHANGED"), encoding="utf-8")
        with pytest.raises(ValueError, match="hash"):
            app.main(args)
        assert not state["requests"]
        assert not (tmp_path / "packet/ratings").exists()


def test_malformed_ollama_message_is_recorded_as_a_failure(tmp_path):
    app = runner()
    with ollama_stub() as (url, state), contextlib.redirect_stdout(io.StringIO()):
        state["bad"] = "message"
        assert app.main(setup_run(tmp_path, url)) == 1
        rows = [json.loads(line) for line in (tmp_path / "packet/ratings/ratings.jsonl").read_text(encoding="utf-8").splitlines()]
        assert all(row["status"] == "error" and row["rating"] is None for row in rows)


def test_summary_weights_conversations_not_persona_count():
    app = runner()
    mappings = {"C1": dict(system="BM25", memory_mode="M2", retrieval_method="bm25",
                            requested_mode="retrieval_only", style="STANDARD", conversation_id="Q1"),
                "C2": dict(system="BM25", memory_mode="M2", retrieval_method="bm25",
                            requested_mode="retrieval_only", style="STANDARD", conversation_id="Q2")}
    rows = [dict(candidate_id=cid, persona_id=pid, status="ok", rating=rating(score, send))
            for cid, pid, score, send in (("C1", "P1", 1, False), ("C2", "P1", 5, True), ("C2", "P2", 5, True))]
    summary = app.summarise(rows, mappings, 2)
    assert summary[0]["appropriateness"] == 3
    assert summary[0]["would_send_pct"] == 50
    assert summary[0]["rated"] == 3 and summary[0]["expected"] == 4
    with contextlib.redirect_stdout(io.StringIO()) as output:
        app.show_summary(summary)
    lines = [line for line in output.getvalue().splitlines() if " | " in line]
    assert lines[0].index("Style") == lines[1].index("STANDARD")


def run_pipeline(*args):
    script = ROOT / "scripts/run_judging.py"
    assert script.exists(), "The one-command judging pipeline has not been implemented"
    return subprocess.run([sys.executable, "-B", str(script), *map(str, args)], cwd=ROOT,
                          env={**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"},
                          capture_output=True, text=True, encoding="utf-8", timeout=180)


def test_pipeline_dry_run_creates_no_files_or_model_requests(tmp_path):
    env = tmp_path / "offline.env"
    env.write_text("OLLAMA_MODEL=fixture:7b\nOLLAMA_BASE_URL=http://127.0.0.1:1\n", encoding="utf-8")
    out = tmp_path / "not-created"
    result = run_pipeline("--check-only", "--items", 1, "--env-file", env, "--run-dir", out)
    assert result.returncode == 0, result.stderr
    assert "40 judgments" in result.stdout
    assert not out.exists()


def test_pipeline_runs_real_exports_packet_judging_and_resumes_without_reexport(tmp_path):
    with ollama_stub() as (url, state):
        setup_run(tmp_path, url)
        out = tmp_path / "one-command"
        args = ["--items", 1, "--run-dir", out, "--personas", tmp_path / "personas.csv",
                "--env-file", tmp_path / ".env"]
        result = run_pipeline(*args)
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(state["requests"]) == 8
        exports = list((out / "exports").glob("*.jsonl"))
        assert len(exports) == 2
        before = {p.name: p.read_bytes() for p in exports}
        public = [json.loads(line) for line in (out / "packet/judge_inputs.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(public) == 4
        with (out / "packet/ratings/summary.csv").open(encoding="utf-8") as handle:
            report = list(csv.DictReader(handle))
        assert {row["system"] for row in report} == {"BM25_M0", "BM25_M2"}
        assert all(row["rated"] == row["expected"] == "2" for row in report)
        resumed = run_pipeline(*args, "--resume")
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        assert len(state["requests"]) == 8
        assert {p.name: p.read_bytes() for p in exports} == before
        summary = run_pipeline("--run-dir", out, "--summary-only")
        assert summary.returncode == 0, summary.stderr
        assert "BM25_M0" in summary.stdout and "Send %" in summary.stdout
        assert len(state["requests"]) == 8


def test_pipeline_missing_local_model_fails_before_export_or_judging(tmp_path):
    with ollama_stub() as (url, state):
        env = tmp_path / "missing-model.env"
        env.write_text(f"OLLAMA_MODEL=not-installed:7b\nOLLAMA_BASE_URL={url}\n", encoding="utf-8")
        out = tmp_path / "not-created"
        result = run_pipeline("--env-file", env, "--run-dir", out)
        assert result.returncode != 0
        assert "not installed" in result.stderr
        assert not out.exists() and not state["requests"]
