from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from replycoach.reply_bank import load_reply_bank
from replycoach.retrieval.v0 import V0Retriever
from replycoach.retrieval.v2 import V2Retriever
from replycoach.systems import run_system

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = ROOT / "data" / "pilot" / "long_context_diagnostics.jsonl"
BANK_PATH = ROOT / "data" / "dev" / "train.jsonl"
EXPERIMENTS = ROOT / "experiments"

def read_jsonl(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main():
    cases = read_jsonl(DIAGNOSTICS)
    bank = load_reply_bank(BANK_PATH)
    configs = [
        ("bm25", V0Retriever),
        ("minilm", V2Retriever),
    ]
    modes = ["M0", "M1", "M2"]
    all_results = []

    for retriever_name, retriever_class in configs:
        start_build = time.perf_counter()
        if retriever_name == "bm25":
            retriever = retriever_class.from_items(bank.items)
        else:
            retriever = retriever_class.from_items(bank.items, cache_file=None)
        build_seconds = time.perf_counter() - start_build

        for memory_mode in modes:
            rows = []
            timings = []
            errors = 0

            for case in cases:
                raw_message = case.get("message", case.get("incoming_message", ""))
                if isinstance(raw_message, dict):
                    message = str(raw_message.get("text", ""))
                else:
                    message = str(raw_message)

                session = case.get("session_memory", case.get("recent_conversation", case.get("session", [])))
                if isinstance(session, list):
                    session = [
                        turn if isinstance(turn, str)
                        else f"{turn.get('speaker', '')}: {turn.get('text', '')}".strip()
                        for turn in session
                    ]

                started = time.perf_counter()
                try:
                    result = run_system(
                        message=message,
                        session=session,
                        memory_mode=memory_mode,
                        mode="retrieval_only",
                        retrieval_method=retriever_name,
                        retriever_factory=lambda: retriever,
                        top_k=3,
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    timings.append(elapsed_ms)

                    coach_result = getattr(result, "result", result)
                    candidates = getattr(coach_result, "candidates", []) or []
                    selected = []
                    for candidate in candidates:
                        selected.append({
                            "candidate_id": candidate.item_id,
                            "style": candidate.style,
                            "source": candidate.source,
                            "score": float(candidate.score),
                            "text": candidate.candidate_text,
                            "meta": candidate.meta,
                        })

                    prepared = getattr(result, "prepared_input", None)
                    selected_memory = getattr(prepared, "memory", None)
                    retrieval_memory = getattr(prepared, "retrieval_memory", None)

                    record = {
                        "id": case.get("id"),
                        "retriever": retriever_name,
                        "memory_mode": memory_mode,
                        "incoming_message": message,
                        "session_turn_count": len(session),
                        "selected_memory": selected_memory,
                        "retrieval_memory": retrieval_memory,
                        "candidates": selected,
                        "errors": {},
                        "elapsed_ms": round(elapsed_ms, 3),
                        "expected_context": case.get("notes", ""),
                        "diagnostic_only": True,
                    }
                except Exception as exc:
                    errors += 1
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    record = {
                        "id": case.get("id"),
                        "retriever": retriever_name,
                        "memory_mode": memory_mode,
                        "incoming_message": message,
                        "session_turn_count": len(session),
                        "candidates": [],
                        "errors": {"message": repr(exc)},
                        "elapsed_ms": round(elapsed_ms, 3),
                        "expected_context": case.get("notes", ""),
                        "diagnostic_only": True,
                    }

                rows.append(record)
                all_results.append(record)

            out = EXPERIMENTS / f"long_context_{retriever_name}_{memory_mode.lower()}.jsonl"
            write_jsonl(out, rows)
            if timings:
                print(
                    f"{retriever_name.upper():6} {memory_mode} | "
                    f"n={len(timings)} | mean={statistics.mean(timings):.2f} ms | "
                    f"median={statistics.median(timings):.2f} ms | "
                    f"index_build={build_seconds:.2f}s | errors={errors}"
                )
            else:
                print(f"{retriever_name.upper():6} {memory_mode} | no successful cases | errors={errors}")

    combined = EXPERIMENTS / "long_context_all_results.jsonl"
    write_jsonl(combined, all_results)
    print(f"Saved {len(all_results)} configuration-case results to {combined}")
    print("These are preliminary diagnostic results, not an approved benchmark.")

if __name__ == "__main__":
    main()
