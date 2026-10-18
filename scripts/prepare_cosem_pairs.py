"""Convert the Corpus of Singapore English Messages (CoSEM) into (context, response) pairs.

This script does not download data. Point it at the local ``corpus_COSEM_v5.zip``
(or the extracted folder) from https://github.com/wdwgonzales/CoSEM after Role A
ticks the licence checklist. CoSEM is CC BY-NC-SA 4.0; cite Gonzales et al. (2023).

Output rows match scripts/prepare_dailydialog_pairs.py, with source "cosem".
Whole chats are held out for validation, so no chat feeds both files.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import zipfile
from collections import Counter
from pathlib import Path

# <<COSEM:17CF01-12-21SGCHF00-2016>>\ttext  ->  chat id, line number, speaker code, year
TAG = re.compile(r"^<<COSEM:([^-\s>]+)-(\d+)-([^-\s>]+)-[^>\s]*>>\t?(.*)$")
INVISIBLE = re.compile("[\ufeff\u200e\u200f\u202a-\u202e]")
# Scrubber placeholders, export placeholders, anonymised speaker codes, mis-encoded emoji.
UNUSABLE = re.compile(
    r"\{\{|\{[A-Z]{1,4}\}|<[^>]*omitted>|(?<!\S)(?=\S*[A-Z])\S+/\S+/\S+"
    r"|deleted this message|missed (?:voice|video) call"
    # UTF-8 read as a legacy code page: a lead byte shown as a Latin letter, then a continuation byte.
    "|[\u00c2-\u00f4\u201a][\u0080-\u00bf\u02c6\u02dc\u2018-\u2022\u2030\u2039\u203a\u20ac\u2122\u0152\u0153\u0160\u0161\u0178\u017d\u017e\u00c0-\u00ff]",
    re.IGNORECASE,
)
CJK = re.compile(r"[\u3400-\u9fff]")
# A reply coach must not suggest vulgarities, so they are kept out of the responses.
BLOCKED = re.compile(
    r"\b(?:fuck\w*|fk|fking|wtf|stfu|shit\w*|bitch\w*|cunt\w*|dick\w*|pussy|asshole\w*|bastard\w*|slut\w*|whore\w*"
    r"|knn|knnbccb|ccb|cb|cibai|chibai|chee ?bye|lanjiao|lan ?jiao|kanina|puki|pukimak|nabei|nbcb)\b",
    re.IGNORECASE,
)

MAX_LINE_GAP = 3  # deduplication left holes in the line numbers; larger holes break the thread
EMOJI_LOSS_RATE = 0.2  # chats whose emoji were exported as "?" end far more lines with "?"
MAX_CONTEXT = 2
MAX_MESSAGE_WORDS = 40
MIN_RESPONSE_WORDS = 3
MAX_RESPONSE_WORDS = 30
MAX_RESPONSE_REPEATS = 5
MAX_CHAT_SHARE = 0.05
MIN_VALIDATION_CHATS = 5


def read_lines(source: Path) -> list[str]:
    """Return every corpus line in chunk order from the release zip or its extracted folder."""

    def chunk_number(name: str) -> int:
        match = re.search(r"(\d+)\.txt$", name)
        return int(match.group(1)) if match else 0

    lines: list[str] = []
    if source.is_dir():
        for path in sorted(source.rglob("*.txt"), key=lambda path: chunk_number(path.name)):
            lines.extend(path.read_text(encoding="utf-8", errors="replace").splitlines())
        return lines
    with zipfile.ZipFile(source) as archive:
        names = [
            name for name in archive.namelist()
            if name.endswith(".txt") and "__MACOSX" not in name and not Path(name).name.startswith(".")
        ]
        for name in sorted(names, key=chunk_number):
            lines.extend(archive.read(name).decode("utf-8", errors="replace").splitlines())
    return lines


def parse_chats(lines: list[str]) -> dict[str, list[tuple[int, str, str]]]:
    """Group lines into chats of (line number, speaker, text). Untagged lines continue the previous message."""
    chats: dict[str, list[tuple[int, str, str]]] = {}
    current: list[tuple[int, str, str]] | None = None
    for raw in lines:
        line = INVISIBLE.sub("", raw)
        match = TAG.match(line)
        if match:
            chat, number, speaker, text = match.groups()
            current = chats.setdefault(chat, [])
            current.append((int(number), speaker, " ".join(text.split()).removeprefix(": ")))
        elif current and line.strip():
            number, speaker, text = current[-1]
            current[-1] = (number, speaker, f"{text} {' '.join(line.split())}".strip())
    return chats


def lost_emoji(messages: list[tuple[int, str, str]]) -> bool:
    return sum(text.endswith("?") for _, _, text in messages) > EMOJI_LOSS_RATE * len(messages)


def build_turns(messages: list[tuple[int, str, str]]) -> list[dict]:
    """Merge consecutive lines from one speaker into a turn; a hole in the numbering starts a new thread."""
    turns: list[dict] = []
    previous_number = None
    for number, speaker, text in messages:
        gap = previous_number is not None and number - previous_number > MAX_LINE_GAP
        previous_number = number
        if turns and not gap and turns[-1]["speaker"] == speaker:
            turns[-1]["lines"].append(text)
        else:
            turns.append({"speaker": speaker, "lines": [text], "new_thread": gap})
    return turns


def clip(lines: list[str], max_words: int, keep: str) -> str | None:
    """Keep whole lines from the end ("tail") or the start ("head") of a turn within the word budget."""
    if any(not line or UNUSABLE.search(line) for line in lines):
        return None
    ordered = list(reversed(lines)) if keep == "tail" else lines
    kept: list[str] = []
    words = 0
    for line in ordered:
        words += len(line.split())
        if words > max_words:
            break
        kept.append(line)
    if not kept:
        return None
    return " ".join(reversed(kept) if keep == "tail" else kept)


def usable_response(text: str | None) -> bool:
    if text is None or len(text.split()) < MIN_RESPONSE_WORDS:
        return False
    if CJK.search(text) or BLOCKED.search(text):  # English-led replies only; do not teach Mandarin output
        return False
    return sum(any(char.isalpha() for char in word) for word in text.split()) >= 2


def pairs_from_chat(chat: str, messages: list[tuple[int, str, str]]) -> list[dict]:
    turns = build_turns(messages)
    rows = []
    for index in range(1, len(turns)):
        if turns[index]["new_thread"]:
            continue
        message = clip(turns[index - 1]["lines"], MAX_MESSAGE_WORDS, "tail")
        response = clip(turns[index]["lines"], MAX_RESPONSE_WORDS, "head")
        if message is None or not any(char.isalpha() for char in message) or not usable_response(response):
            continue
        context: list[str] = []
        for earlier in range(index - 2, max(-1, index - 2 - MAX_CONTEXT), -1):
            # Stop at a thread start: the turn before it belongs to another exchange.
            if turns[earlier + 1]["new_thread"]:
                break
            text = clip(turns[earlier]["lines"], MAX_MESSAGE_WORDS, "tail")
            if text is None:
                break
            context.insert(0, text)
        rows.append({"context": context, "message": message, "response": response, "source": "cosem", "chat": chat})
    return rows


def limit_repeats(rows: list[dict]) -> list[dict]:
    """Drop duplicate pairs and stop stock replies ("ok can", "haha yes lor") from dominating."""
    seen_pairs: set[tuple[str, str]] = set()
    response_counts: Counter[str] = Counter()
    kept = []
    for row in rows:
        response = row["response"].lower()
        pair = (row["message"].lower(), response)
        if pair in seen_pairs or response_counts[response] >= MAX_RESPONSE_REPEATS:
            continue
        seen_pairs.add(pair)
        response_counts[response] += 1
        kept.append(row)
    return kept


def split_pairs(
    pairs_by_chat: dict[str, list[dict]], max_train: int, max_validation: int, seed: int
) -> tuple[list[dict], list[dict], list[str]]:
    """Hold out whole chats for validation, then sample both sets with a per-chat cap."""
    rng = random.Random(seed)
    total = sum(len(rows) for rows in pairs_by_chat.values())
    chats = sorted(pairs_by_chat)
    rng.shuffle(chats)
    validation_cap = max(1, max_validation // MIN_VALIDATION_CHATS)
    validation_chats: list[str] = []
    held_out = 0
    for chat in chats:
        # Aim past the limit because limit_repeats removes some rows afterwards.
        if held_out >= 1.5 * max_validation:
            break
        if len(pairs_by_chat[chat]) > MAX_CHAT_SHARE * total:
            continue
        validation_chats.append(chat)
        held_out += min(len(pairs_by_chat[chat]), validation_cap)

    def sample(names: list[str], limit: int, cap: int) -> list[dict]:
        rows: list[dict] = []
        for chat in names:
            chat_rows = list(pairs_by_chat[chat])
            rng.shuffle(chat_rows)
            rows.extend(chat_rows[:cap])
        rng.shuffle(rows)
        return limit_repeats(rows)[:limit]

    train_chats = [chat for chat in chats if chat not in set(validation_chats)]
    train = sample(train_chats, max_train, max(1, int(MAX_CHAT_SHARE * max_train)))
    return train, sample(validation_chats, max_validation, validation_cap), sorted(validation_chats)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="corpus_COSEM_v5.zip or its extracted folder")
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--max-train-pairs", type=int, default=80000, help="About the size of DailyDialog train.")
    parser.add_argument("--max-validation-pairs", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=425)
    args = parser.parse_args()
    if args.max_train_pairs < 1 or args.max_validation_pairs < 1:
        parser.error("pair limits must be positive")
    if args.train_output.resolve() == args.validation_output.resolve():
        parser.error("training and validation outputs must differ")

    chats = parse_chats(read_lines(args.input))
    if not chats:
        parser.error(f"no <<COSEM:...>> lines found in {args.input}")
    kept = {chat: messages for chat, messages in chats.items() if not lost_emoji(messages)}
    pairs_by_chat = {chat: pairs_from_chat(chat, messages) for chat, messages in kept.items()}
    pairs_by_chat = {chat: rows for chat, rows in pairs_by_chat.items() if rows}
    train, validation, validation_chats = split_pairs(
        pairs_by_chat, args.max_train_pairs, args.max_validation_pairs, args.seed
    )
    write_rows(args.train_output, train)
    write_rows(args.validation_output, validation)
    print(f"chats: {len(chats)} parsed, {len(chats) - len(kept)} dropped for lost emoji, {len(pairs_by_chat)} with pairs")
    print(f"candidate pairs: {sum(len(rows) for rows in pairs_by_chat.values())}")
    print(f"validation chats: {', '.join(validation_chats)}")
    print(f"wrote {len(train)} pairs to {args.train_output}")
    print(f"wrote {len(validation)} pairs to {args.validation_output}")


if __name__ == "__main__":
    main()
