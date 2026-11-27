"""Development comparisons with split, leakage and output-publication guards."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from replycoach.generate.providers import ReplyGenerator
from replycoach.io import DATA_DIR
from replycoach.pipeline import Retriever
from replycoach.reply_bank import ReplyBank, load_reply_bank, validate_development_row
from replycoach.retrieval.v1 import detect_and_normalise
from replycoach.schema import DialogueItem
from replycoach.systems import SYSTEM_MODES, compare_system_modes, run_system

# ##TODO: FOR ROLE A — publish reviewed dev train/validation files here.
# Keep every row split="dev"; their union is the 240-item development set.
# Publish the 60 locked test items separately. See docs/role_handoffs.md.
DEFAULT_BANK = DATA_DIR / "dev" / "train.jsonl"
DEFAULT_QUERIES = DATA_DIR / "dev" / "validation.jsonl"


@dataclass
class EvaluationInputs:
    queries: ReplyBank
    bank: ReplyBank


def _input_key(item: DialogueItem) -> tuple[str, ...]:
    return tuple(detect_and_normalise(text).normalised.casefold()
                 for text in [*item.memory_texts(), item.message.text])


def _validate_inputs(inputs: EvaluationInputs) -> None:
    queries, bank = inputs.queries, inputs.bank
    for dataset in (queries, bank):
        if not dataset.items:
            raise ValueError("development datasets must be nonempty")
        seen = set()
        for item in dataset.items:
            item.validate()
            validate_development_row(item.to_dict())
            if not isinstance(item.id, str) or not item.id.strip() or item.id in seen:
                raise ValueError("development IDs must be nonempty and unique")
            seen.add(item.id)
            texts = [*item.memory_texts(), item.message.text, item.replies.standard, item.replies.local]
            if any(not isinstance(text, str) or not text.strip() for text in texts):
                raise ValueError("development conversation/reply text must be nonempty strings")
    shared_ids = {item.id for item in queries.items} & {item.id for item in bank.items}
    shared_inputs = {_input_key(item) for item in queries.items} & {_input_key(item) for item in bank.items}
    if shared_ids or shared_inputs:
        raise ValueError("validation/reply-bank overlap: use disjoint IDs and normalized conversation inputs")


def load_evaluation_inputs(queries_path: Path, bank_path: Path) -> EvaluationInputs:
    inputs = EvaluationInputs(load_reply_bank(queries_path), load_reply_bank(bank_path))
    _validate_inputs(inputs)
    return inputs


def collect_dev_outputs(
    inputs: EvaluationInputs, *, retriever_factory: Callable[[], Retriever] | None,
    generator_factory: Callable[[], ReplyGenerator] | None,
    retrieval_method: str = "bm25", memory_mode: str = "M2", mode: str = "all",
) -> list[dict]:
    """Only export modes that ran with their required sources; never partial hybrids."""
    _validate_inputs(inputs)
    if mode not in (*SYSTEM_MODES, "all"):
        raise ValueError("export mode must be all or a valid system mode")
    rows = []
    for item in inputs.queries.items:
        kwargs = {"retrieval_method": retrieval_method, "retriever_factory": retriever_factory,
                  "generator_factory": generator_factory}
        if mode == "all":
            results = compare_system_modes(item.message.text, item.memory_texts(), memory_mode, **kwargs).results
        else:
            results = {mode: run_system(item.message.text, item.memory_texts(), memory_mode, mode=mode, **kwargs)}
        for requested, output in results.items():
            if output.errors or output.effective_mode != requested:
                details = "; ".join(output.errors.values()) or output.effective_mode
                raise ValueError(f"{item.id}: incomplete {requested} comparison: {details}")
            rows.append({
                "record_type": "development_comparison", "item_id": item.id,
                "memory_mode": memory_mode, "bank_fingerprint": inputs.bank.fingerprint,
                "queries_fingerprint": inputs.queries.fingerprint, **output.to_dict(),
            })
    return rows


def validate_output_path(path: Path, protected_paths: tuple[Path, ...] = ()) -> Path:
    target = Path(path).resolve()
    if any(target == Path(protected).resolve() for protected in protected_paths):
        raise ValueError("output cannot replace an input dataset")
    if target == DATA_DIR.resolve() or DATA_DIR.resolve() in target.parents:
        raise ValueError("write results outside data/; input datasets must stay unchanged")
    if target.exists():
        raise FileExistsError(f"output already exists: {target}; choose a new run filename")
    return target


def write_dev_outputs(path: Path, rows: list[dict], *, protected_paths: tuple[Path, ...] = ()) -> None:
    """Publish a complete UTF-8 JSONL file atomically, without replacing files."""
    target = validate_output_path(path, protected_paths)
    payload = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=f".{target.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        # Hard-link publication is exclusive: unlike replace(), an existing
        # target (including one created concurrently) is never overwritten.
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
