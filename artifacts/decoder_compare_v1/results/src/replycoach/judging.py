"""Prepare blinded development replies for an external judging runner; no LLM calls."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random

from replycoach.evaluation import load_evaluation_inputs, validate_output_path, write_dev_outputs
from replycoach.io import ROOT
from replycoach.schema import Candidate, STYLES
from replycoach.systems import SYSTEM_MODES

PUBLIC_FIELDS = {"schema_version", "candidate_id", "conversation_id", "conversation",
                 "incoming_message", "style", "reply"}


@dataclass(frozen=True)
class SystemExport:
    label: str
    path: Path
    mode: str


def _reject_nonfinite(value: str):
    raise ValueError(f"non-finite JSON number: {value}")


def _load_export(spec: SystemExport, inputs) -> tuple[dict[str, dict], str]:
    raw = Path(spec.path).read_bytes()
    query_items = {item.id: item for item in inputs.queries.items}
    selected, configurations = {}, set()
    for line_no, line in enumerate(raw.decode("utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=_reject_nonfinite)
            json.dumps(row, allow_nan=False)  # Also rejects numeric overflow such as 1e400.
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{spec.label}: invalid JSON on line {line_no}: {exc}") from exc
        if not isinstance(row, dict) or row.get("record_type") != "development_comparison":
            raise ValueError(f"{spec.label}: only development_comparison exports are accepted")
        if row.get("requested_mode") not in SYSTEM_MODES:
            raise ValueError(f"{spec.label}: invalid requested_mode")
        if row["requested_mode"] != spec.mode:
            continue
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or item_id not in query_items or item_id in selected:
            raise ValueError(f"{spec.label}: query coverage contains an unknown or duplicate item")
        if (row.get("bank_fingerprint") != inputs.bank.fingerprint
                or row.get("queries_fingerprint") != inputs.queries.fingerprint):
            raise ValueError(f"{spec.label}: dataset fingerprints do not match the current inputs")
        if row.get("message") != query_items[item_id].message.text:
            raise ValueError(f"{spec.label}: {item_id} message differs from the evaluated query")
        if row.get("effective_mode") != spec.mode or row.get("errors") != {}:
            raise ValueError(f"{spec.label}: {item_id} is a failed or degraded comparison")
        memory, method = row.get("memory_mode"), row.get("retrieval_method")
        if memory not in ("M0", "M1", "M2") or method not in ("bm25", "minilm"):
            raise ValueError(f"{spec.label}: invalid memory/retrieval configuration")
        configurations.add((memory, method))
        for key, required in (("retrieval_count", spec.mode != "generator_only"),
                              ("generation_count", spec.mode != "retrieval_only")):
            count = row.get(key)
            if type(count) is not int or count < 0 or (count > 0) != required:
                raise ValueError(f"{spec.label}: {item_id} has inconsistent {key}")
        for style in STYLES:
            candidate = row.get(style.lower())
            if (not isinstance(candidate, dict) or candidate.get("style") != style
                    or not isinstance(candidate.get("candidate_text"), str)
                    or not candidate["candidate_text"].strip()):
                raise ValueError(f"{spec.label}: {item_id} needs a nonempty {style} candidate")
            score = candidate.get("score")
            if type(score) not in (int, float) or not math.isfinite(score):
                raise ValueError(f"{spec.label}: {item_id} candidate score must be a finite number")
            try:
                Candidate.from_dict(candidate)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{spec.label}: invalid {style} candidate for {item_id}: {exc}") from exc
        selected[item_id] = row
    if set(selected) != set(query_items):
        raise ValueError(f"{spec.label}: query coverage mismatch; expected every validation item exactly once")
    if len(configurations) != 1:
        raise ValueError(f"{spec.label}: mixed memory/retrieval settings; use one export run per system")
    return selected, sha256(raw).hexdigest()


def prepare_judging_packet(
    queries_path: Path, bank_path: Path, systems: list[SystemExport], *,
    seed: int = 425, items: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Match full development exports before selecting any common pilot subset."""
    inputs = load_evaluation_inputs(queries_path, bank_path)
    if not systems:
        raise ValueError("at least one system export is required")
    labels = [spec.label.strip().casefold() for spec in systems]
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("system labels must be nonempty and unique (case-insensitive)")
    if any(spec.mode not in SYSTEM_MODES for spec in systems):
        raise ValueError(f"system mode must be one of {SYSTEM_MODES}")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if items is not None and (type(items) is not int or not 1 <= items <= len(inputs.queries.items)):
        raise ValueError("items must be between 1 and the full validation count")
    exports = [(spec, *_load_export(spec, inputs)) for spec in systems]
    rng = random.Random(seed)
    queries = sorted(inputs.queries.items, key=lambda item: item.id)
    rng.shuffle(queries)
    if items is not None:
        queries = queries[:items]
    pairs = []
    for conversation_index, item in enumerate(queries, start=1):
        conversation_id = f"Q{conversation_index:06d}"
        for spec, outputs, export_hash in exports:
            output = outputs[item.id]
            export_path = Path(spec.path).resolve()
            export_name = (export_path.relative_to(ROOT).as_posix()
                           if export_path.is_relative_to(ROOT) else export_path.name)
            for style in STYLES:
                candidate = output[style.lower()]
                public = {
                    "schema_version": 1, "conversation_id": conversation_id,
                    "conversation": [{"speaker": turn.speaker, "text": turn.text}
                                     for turn in item.session_memory],
                    "incoming_message": {"speaker": "peer", "text": item.message.text},
                    "style": style, "reply": candidate["candidate_text"],
                }
                private = {
                    "schema_version": 1, "conversation_id": conversation_id,
                    "item_id": item.id, "style": style, "system": spec.label.strip(),
                    "source_export": export_name,
                    "source_export_sha256": export_hash,
                    "requested_mode": spec.mode, "memory_mode": output["memory_mode"],
                    "retrieval_method": output["retrieval_method"],
                    "bank_fingerprint": inputs.bank.fingerprint,
                    "queries_fingerprint": inputs.queries.fingerprint,
                    "shuffle_seed": seed, "conversation_count": len(queries),
                    "candidate": candidate,
                }
                pairs.append((public, private))
    # Assign candidate IDs only after shuffling: IDs do not encode source/style.
    rng.shuffle(pairs)
    for index, (public, private) in enumerate(pairs, start=1):
        public["candidate_id"] = private["candidate_id"] = f"C{index:06d}"
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def write_judging_packet(
    out_dir: Path, public: list[dict], private: list[dict], *,
    protected_paths: tuple[Path, ...] = (),
) -> None:
    """Publish into a new directory, preserving prior runs and source datasets."""
    target = validate_output_path(out_dir, protected_paths)
    if not public or any(set(row) != PUBLIC_FIELDS for row in public):
        raise ValueError("public judging rows must contain only the documented anonymous fields")
    public_ids = [row["candidate_id"] for row in public]
    private_ids = [row.get("candidate_id") for row in private]
    if len(set(public_ids)) != len(public_ids) or sorted(public_ids) != sorted(private_ids):
        raise ValueError("private mapping must match every public candidate exactly once")
    # Validate both payloads before creating a directory or publishing either file.
    for rows in (public, private):
        json.dumps(rows, ensure_ascii=False, allow_nan=False)
    target.mkdir(parents=True, exist_ok=False)
    public_path = target / "judge_inputs.jsonl"
    write_dev_outputs(public_path, public)
    public_hash = sha256(public_path.read_bytes()).hexdigest()
    bound_private = [{**row, "judge_inputs_sha256": public_hash} for row in private]
    write_dev_outputs(target / "PRIVATE_mapping.jsonl", bound_private)
