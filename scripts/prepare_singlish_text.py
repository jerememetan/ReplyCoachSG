"""Collect plain Singapore English chat text for V5 span-corruption training (stage 0).

Span corruption needs text, not message/reply pairs, so this also uses the NUS SMS Corpus, which has no
reply threading. Sources, both optional:

- NUS SMS Corpus (Chen & Kan, 2013), ``smsCorpus_en_xml_2015.03.09_all.zip`` from
  https://github.com/kite1988/nus-sms-corpus. Only messages from Singapore senders are kept.
- CoSEM (Gonzales et al., 2023), ``corpus_COSEM_v5.zip``. Chats that stage 1 validates on are left out,
  so stage 1's CoSEM validation loss is not measured on text the model has already read.

Short messages are packed into chunks of consecutive messages from one sender (NUS) or one chat (CoSEM).
Whole senders/chats are held out for validation. Messages with long numbers, emails, URLs or placeholders,
vulgarities, or Chinese characters are dropped.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("prepare_cosem_pairs", Path(__file__).with_name("prepare_cosem_pairs.py"))
cosem = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cosem)

SINGAPORE = {"SG", "Singapore"}
# Phone numbers, emails, links, and the corpus's "<#>" number placeholder.
PRIVATE = re.compile(r"\d{7,}|\S@\S|https?://|www\.|<#>", re.IGNORECASE)
CHUNK_WORDS = 60
SEPARATOR = " | "  # the same turn separator mt5_input uses


def usable(text: str) -> bool:
    return bool(text) and any(char.isalpha() for char in text) and not (
        PRIVATE.search(text) or cosem.UNUSABLE.search(text) or cosem.BLOCKED.search(text) or cosem.CJK.search(text)
    )


def nus_messages(source: Path) -> dict[str, list[str]]:
    """Messages from Singapore senders, grouped by sender in corpus order."""
    with zipfile.ZipFile(source) as archive:
        name = next(name for name in archive.namelist() if name.endswith(".xml"))
        root = ET.fromstring(archive.read(name))
    groups: dict[str, list[str]] = {}
    for message in root.iter("message"):
        if message.findtext("source/userProfile/country") not in SINGAPORE:
            continue
        text = " ".join((message.findtext("text") or "").split())
        if usable(text):
            groups.setdefault(f"nus-{message.findtext('source/userProfile/userID')}", []).append(text)
    return groups


def cosem_messages(source: Path, exclude_chats: set[str]) -> dict[str, list[str]]:
    chats = cosem.parse_chats(cosem.read_lines(source))
    return {
        f"cosem-{chat}": [text for _, _, text in messages if usable(text)]
        for chat, messages in chats.items() if chat not in exclude_chats
    }


def pack(messages: list[str], max_words: int = CHUNK_WORDS) -> list[str]:
    """Join consecutive messages until the next one would pass ``max_words``; a longer message stands alone."""
    chunks: list[str] = []
    current: list[str] = []
    words = 0
    for text in messages:
        length = len(text.split())
        if current and words + length > max_words:
            chunks.append(SEPARATOR.join(current))
            current, words = [], 0
        current.append(text)
        words += length
    if current:
        chunks.append(SEPARATOR.join(current))
    return chunks


def split_groups(chunks_by_group: dict[str, list[str]], validation_share: float, seed: int):
    """Hold out whole groups until about ``validation_share`` of the chunks are in validation."""
    groups = sorted(chunks_by_group)
    random.Random(seed).shuffle(groups)
    target = validation_share * sum(len(chunks) for chunks in chunks_by_group.values())
    validation: list[str] = []
    held_out = 0
    for group in groups:
        if held_out >= target:
            break
        if len(chunks_by_group[group]) <= target - held_out:  # one huge group would overshoot
            validation.append(group)
            held_out += len(chunks_by_group[group])
    return [group for group in groups if group not in set(validation)], validation


def rows_for(groups: list[str], chunks_by_group: dict[str, list[str]]) -> list[dict]:
    return [
        {"source": group.split("-", 1)[0].replace("nus", "nus_sms"), "group": group, "text": chunk}
        for group in groups for chunk in chunks_by_group[group]
    ]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nus-sms", type=Path, help="smsCorpus_en_xml_2015.03.09_all.zip")
    parser.add_argument("--cosem", type=Path, help="corpus_COSEM_v5.zip or its extracted folder")
    parser.add_argument("--cosem-validation-pairs", type=Path, default=Path("data/public/cosem_validation_pairs.jsonl"),
                        help="Stage 1 CoSEM validation pairs; their chats are left out.")
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--validation-share", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=425)
    args = parser.parse_args()
    if args.nus_sms is None and args.cosem is None:
        parser.error("give --nus-sms, --cosem, or both")
    if not 0 < args.validation_share < 0.5:
        parser.error("--validation-share must be between 0 and 0.5")
    if args.train_output.resolve() == args.validation_output.resolve():
        parser.error("training and validation outputs must differ")

    messages: dict[str, list[str]] = {}
    if args.nus_sms:
        messages.update(nus_messages(args.nus_sms))
    if args.cosem:
        if not args.cosem_validation_pairs.exists():
            parser.error(f"{args.cosem_validation_pairs} not found; run prepare_cosem_pairs.py first")
        exclude = {json.loads(line)["chat"] for line in args.cosem_validation_pairs.open(encoding="utf-8") if line.strip()}
        messages.update(cosem_messages(args.cosem, exclude))
    chunks_by_group = {group: pack(texts) for group, texts in messages.items() if texts}
    if not chunks_by_group:
        parser.error("no usable messages found")

    train_rows, validation_rows = [], []
    for prefix in sorted({group.split("-", 1)[0] for group in chunks_by_group}):
        # Split each source separately so both appear in validation.
        source_chunks = {group: chunks for group, chunks in chunks_by_group.items() if group.startswith(f"{prefix}-")}
        train_groups, validation_groups = split_groups(source_chunks, args.validation_share, args.seed)
        train_rows += rows_for(train_groups, source_chunks)
        validation_rows += rows_for(validation_groups, source_chunks)
        words = sum(len(text.split()) for chunks in source_chunks.values() for text in chunks)
        print(f"{prefix}: {len(source_chunks)} groups, {sum(map(len, source_chunks.values()))} chunks, {words} words, "
              f"{len(validation_groups)} groups held out")
    random.Random(args.seed).shuffle(train_rows)
    write_rows(args.train_output, train_rows)
    write_rows(args.validation_output, validation_rows)
    print(f"wrote {len(train_rows)} chunks to {args.train_output}")
    print(f"wrote {len(validation_rows)} chunks to {args.validation_output}")


if __name__ == "__main__":
    main()
