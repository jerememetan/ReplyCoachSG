"""Blind A/B comparison of two V5 runs on the validation items, rated by a person in the terminal.

Each validation item and style is one question: the conversation, the friend's message, and the two runs'
replies as 1 and 2 in a random order, so the rater cannot tell which run wrote which. The rater answers

    1 or 2   that reply is better
    =        both are about as good
    x        both are bad
    q        stop for now; answers are saved after each question and ``rate`` carries on where it stopped

Questions where both runs wrote the same reply are not asked; they count as ties. ``score`` then reveals
which run won how often and a sign test: if the runs were really equally good, how likely is a split at least
this uneven? Development items only, rated by a person; no LLM ratings.

    python -m replycoach.generate.compare_v5 make artifacts/spk1_lora_cp.json artifacts/spk1_ul025_s425.json --name ul025
    python -m replycoach.generate.compare_v5 rate ul025
    python -m replycoach.generate.compare_v5 score ul025

``make-prefs`` asks the same kind of question on the training items, about one model's own replies: its usual
reply against one of its sampled replies (a ``rerank_v5 candidates`` file made with ``--validation-items
data/dev/train.jsonl``). The picks are the (better, worse) pairs for ``finetune_v5 --preferences``.

    python -m replycoach.generate.compare_v5 make-prefs artifacts/candidates_train_spk1.json --name prefs1
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from replycoach.generate.style_control import SPEAKER_LABELS
from replycoach.generate.train_v5 import ARTIFACTS
from replycoach.io import DATA_DIR, load_items

STYLES = ("STANDARD", "LOCAL")
ANSWERS = {"1": "first", "2": "second", "=": "tie", "x": "both_bad"}


def sheet_path(name: str, output_dir: Path = ARTIFACTS) -> Path:
    return output_dir / f"ab_{name}.json"


def make_questions(run_a: dict, run_b: dict, items_path: Path, seed: int = 425) -> tuple[list[dict], dict]:
    """Questions in a shuffled order with the two replies in a shuffled order, and the same-reply count by style."""
    items = {item.id: item for item in load_items(items_path)}
    if any(item.split != "dev" for item in items.values()):
        raise ValueError(f"{items_path}: only development items may be rated here")
    outputs = [{row["item_id"]: row for row in run["validation_outputs"]} for run in (run_a, run_b)]
    if set(outputs[0]) != set(outputs[1]) or not set(outputs[0]) <= set(items):
        raise ValueError("both runs must have outputs for the same development items")
    rng = random.Random(seed)
    questions, same = [], {"ALL": 0, **{style: 0 for style in STYLES}}
    for item_id in sorted(outputs[0]):
        item = items[item_id]
        for style in STYLES:
            key = f"generated_{style.lower()}"
            replies = [output[item_id][key].strip() for output in outputs]
            if replies[0] == replies[1]:
                same["ALL"] += 1
                same[style] += 1
                continue
            questions.append(question(item, style, replies, rng))
    rng.shuffle(questions)
    return questions, same


def make_preference_questions(candidates: dict, items_path: Path, seed: int = 425) -> list[dict]:
    """For each training item and style, the model's usual (beam) reply as A against one sampled reply as B."""
    items = {item.id: item for item in load_items(items_path)}
    if any(item.split != "dev" for item in items.values()):
        raise ValueError(f"{items_path}: only development items may be rated here")
    outside = [entry["item_id"] for entry in candidates["items"] if entry["item_id"] not in items]
    if outside:
        raise ValueError(f"candidates for items outside {items_path}: {', '.join(outside[:3])}")
    rng = random.Random(seed)
    questions = []
    for entry in candidates["items"]:
        for style in STYLES:
            beam = [candidate["text"] for candidate in entry["candidates"][style] if candidate["beam"]]
            samples = [candidate["text"] for candidate in entry["candidates"][style] if not candidate["beam"]]
            if beam and samples:
                questions.append(question(items[entry["item_id"]], style, [beam[0], rng.choice(samples)], rng))
    rng.shuffle(questions)
    return questions


def question(item, style: str, replies: list[str], rng: random.Random) -> dict:
    """One question with reply A (``replies[0]``) shown first or second at random."""
    a_first = rng.random() < 0.5
    return {
        "item_id": item.id, "style": style,
        "conversation": [[turn.speaker, turn.text] for turn in item.session_memory],
        "message": item.message.text,
        "replies": replies if a_first else replies[::-1],
        "a_first": a_first,  # the key: whether reply 1 is A
        "answer": None,
    }


def show(question: dict, number: int, total: int) -> str:
    lines = [f"\n----- {number}/{total}   style: {question['style']} -----"]
    lines += [f"  {SPEAKER_LABELS[speaker]}: {text}" for speaker, text in question["conversation"]]
    lines += [f"  Friend: {question['message']}", "", f"  1) {question['replies'][0]}",
              f"  2) {question['replies'][1]}"]
    return "\n".join(lines)


def rate(sheet: dict, path: Path, ask=input, say=print) -> None:
    """Ask every unanswered question, saving the sheet after each answer."""
    questions = sheet["questions"]
    for number, question in enumerate(questions, 1):
        if question["answer"] is not None:
            continue
        say(show(question, number, len(questions)))
        while True:
            reply = ask("  better? 1 / 2 / = same / x both bad / q quit: ").strip().lower()
            if reply == "q":
                return
            if reply in ANSWERS:
                question["answer"] = ANSWERS[reply]
                path.write_text(json.dumps(sheet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                break


def sign_test(wins: int, losses: int) -> float:
    """Two-sided chance of a split at least this uneven if each non-tie question were a fair coin toss."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def score(sheet: dict) -> dict:
    """Wins for each run, overall and per style, from the answered questions."""
    runs = sheet["runs"]
    result = {}
    for style in (None, *STYLES):
        counts = {runs[0]: 0, runs[1]: 0, "tie": 0, "both_bad": 0, "same_reply": 0}
        for question in sheet["questions"]:
            if question["answer"] is None or style not in (None, question["style"]):
                continue
            if question["answer"] in ("tie", "both_bad"):
                counts[question["answer"]] += 1
            else:
                from_a = (question["answer"] == "first") == question["a_first"]
                counts[runs[0] if from_a else runs[1]] += 1
        counts["same_reply"] = sheet["same_reply"][style or "ALL"]
        counts["sign_test_p"] = round(sign_test(counts[runs[0]], counts[runs[1]]), 4)
        result[style or "ALL"] = counts
    result["answered"] = sum(question["answer"] is not None for question in sheet["questions"])
    result["questions"] = len(sheet["questions"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Blind A/B comparison of two V5 runs, rated in the terminal.")
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make", help="Build a blind sheet from two run reports.")
    make.add_argument("run_a", type=Path)
    make.add_argument("run_b", type=Path)
    make.add_argument("--name", required=True, help="Written to artifacts/ab_<name>.json")
    make.add_argument("--validation-items", type=Path, default=DATA_DIR / "dev" / "validation.jsonl")
    make.add_argument("--seed", type=int, default=425)
    prefs = commands.add_parser("make-prefs", help="Build a sheet of the model's own replies on training items.")
    prefs.add_argument("candidates", type=Path)
    prefs.add_argument("--name", required=True, help="Written to artifacts/ab_<name>.json")
    prefs.add_argument("--train-items", type=Path, default=DATA_DIR / "dev" / "train.jsonl")
    prefs.add_argument("--seed", type=int, default=425)
    for command, text in (("rate", "Answer the questions (resumes where you stopped)."),
                          ("score", "Reveal which run won.")):
        commands.add_parser(command, help=text).add_argument("name")
    args = parser.parse_args()

    path = sheet_path(args.name)
    if args.command in ("make", "make-prefs") and path.exists():
        raise SystemExit(f"{path} already exists; choose a new --name")
    if args.command == "make-prefs":
        candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
        questions = make_preference_questions(candidates, args.train_items, args.seed)
        sheet = {"runs": ["usual_reply", "sampled_reply"], "candidates": str(args.candidates), "seed": args.seed,
                 "same_reply": {"ALL": 0, **{style: 0 for style in STYLES}}, "questions": questions}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sheet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}: {len(questions)} questions on training items")
        return
    if args.command == "make":
        if args.run_a.stem == args.run_b.stem:
            raise SystemExit("the two runs need different file names")
        reports = [json.loads(run.read_text(encoding="utf-8")) for run in (args.run_a, args.run_b)]
        questions, same = make_questions(*reports, args.validation_items, args.seed)
        sheet = {"runs": [args.run_a.stem, args.run_b.stem], "seed": args.seed, "same_reply": same,
                 "questions": questions}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sheet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}: {len(questions)} questions, {same['ALL']} skipped because both runs wrote the same reply")
        return
    sheet = json.loads(path.read_text(encoding="utf-8"))
    if args.command == "rate":
        rate(sheet, path)
        done = sum(question["answer"] is not None for question in sheet["questions"])
        print(f"\n{done}/{len(sheet['questions'])} answered; saved to {path}")
        return
    result = score(sheet)
    if result["answered"] < result["questions"]:
        print(f"only {result['answered']}/{result['questions']} answered so far")
    runs = sheet["runs"]
    print(f"{'':10}{runs[0]:>20}{runs[1]:>20}{'same':>8}{'both bad':>10}{'same reply':>12}{'p':>8}")
    for style in ("ALL", *STYLES):
        counts = result[style]
        print(f"{style:10}{counts[runs[0]]:>20}{counts[runs[1]]:>20}{counts['tie']:>8}{counts['both_bad']:>10}"
              f"{counts['same_reply']:>12}{counts['sign_test_p']:>8}")


if __name__ == "__main__":
    main()
