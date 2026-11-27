"""Measure V5 validation replies, and re-generate them from a saved checkpoint with other decoding settings.

Decoding needs no training, so settings such as blocking n-grams copied from the input are compared on one
checkpoint and the same validation items. The numbers are the problems seen in stage 2:
LOCAL replies without a local term, replies that copy the input, and both styles giving the same reply.

    python -m replycoach.generate.decode_v5 decode --checkpoint artifacts/mt5_base_lora/best --name base_enc3 \
        --encoder-no-repeat-ngram 3
    python -m replycoach.generate.decode_v5 summary artifacts/*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from replycoach.generate.train_v5 import ARTIFACTS
from replycoach.lexicon import LocalTerm, load_lexicon
from replycoach.tokenize import LATIN_OR_CJK, normalize_unicode, tokenize

# Texting markers added to the team lexicon for measuring style only. The lexicon was built for retrieval and
# leaves out ambiguous words such as "ah", so it finds a local term in just 12% of the LOCAL references.
# These words appear in at least three LOCAL references and at most one STANDARD reference.
LOCAL_MARKERS = (
    "ya", "yah", "yea", "ah", "la", "lah", "lor", "leh", "liao", "meh", "sia", "hor", "bah", "eh", "u", "alr",
    "abit", "okok", "aiyo", "walao", "bro", "gg", "anot", "chope", "paiseh", "steady", "makan", "dabao", "jio",
)


def marker_terms(lexicon: list[LocalTerm] | None = None) -> list[list[str]]:
    """Each style marker as a word list: the applied lexicon terms plus ``LOCAL_MARKERS``."""
    surfaces = {term.surface for term in (lexicon or load_lexicon()) if term.apply} | set(LOCAL_MARKERS)
    return [words for words in map(tokenize, sorted(surfaces)) if words]


def has_local_term(text: str, terms: list[list[str]]) -> bool:
    """True if any marker appears as whole words (or CJK characters) in ``text``."""
    words = tokenize(text)
    return any(words[start : start + len(term)] == term for term in terms for start in range(len(words)))


def ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[start : start + n]) for start in range(len(words) - n + 1)}


def copies_input(reply: str, context: list[str], message: str, n: int = 3) -> bool:
    """True if the reply repeats any run of ``n`` words from the context or the incoming message."""
    seen = set().union(*(ngrams(tokenize(text), n) for text in [*context, message]))
    return bool(ngrams(tokenize(reply), n) & seen)


# An echo: the friend says what *they* did or will do ("sorry, I took the wrong bus") and the reply says it as the
# responder's own ("Okay, I'll take the wrong bus"). Found by comparing the few words after each "I" in both texts.
SUBJECTS = {"i", "i'm", "im", "i'll", "i'd", "i've"}
# Words skipped when looking for the action after "I", so "i might be ten mins late" has the action "ten".
NOT_ACTIONS = {
    "am", "was", "be", "been", "will", "would", "can", "could", "might", "may", "must", "should", "shall", "going",
    "gonna", "have", "has", "had", "do", "did", "just", "only", "still", "already", "alr", "really", "think", "then",
    "too", "very", "so",
}
ECHO_WINDOW = 4


def stem(word: str) -> str:
    """Crude suffix stripping so "ordered", "waiting" and "missed" match "order", "wait" and "miss"."""
    for suffix, min_length in (("ing", 5), ("ed", 4), ("s", 3)):
        if word.endswith(suffix) and len(word) > min_length and not word.endswith("ss"):
            return word[: -len(suffix)]
    return word


def first_person_clauses(text: str) -> list[tuple[int, str | None, set[tuple[str, str]]]]:
    """For each "I" in ``text``: its character offset, the first action word after it, and the word pairs after it."""
    words = [(match.group(), match.start()) for match in LATIN_OR_CJK.finditer(normalize_unicode(text).lower())]
    clauses = []
    for index, (word, start) in enumerate(words):
        if word in SUBJECTS:
            after = [stem(following) for following, _ in words[index + 1 : index + 1 + ECHO_WINDOW]]
            action = next((following for following in after if following not in NOT_ACTIONS), None)
            clauses.append((start, action, set(zip(after, after[1:]))))
    return clauses


def echo_start(reply: str, message: str) -> int | None:
    """Character offset of the "I" where ``reply`` echoes the friend's own "I ..." from ``message``, else None.

    An "I" in the reply echoes one in the message when the first action after each is the same, or the next few
    words share a pair. It catches most echoes but not all ("I took" against "I'll take" only matches through
    "wrong bus"), and it flags 2 of the 380 human training replies, such as "i'll order when i arrive" after
    "i ordered first".
    """
    theirs = first_person_clauses(message)
    for start, action, pairs in first_person_clauses(reply):
        if any((action is not None and action == other_action) or pairs & other_pairs
               for _, other_action, other_pairs in theirs):
            return start
    return None


def reply_metrics(outputs: list[dict], lexicon: list[LocalTerm] | None = None) -> dict:
    """Rates over validation items; the reference rates show what the team's own replies do."""
    terms = marker_terms(lexicon)
    count = len(outputs)

    def rate(test) -> float:
        return round(sum(1 for row in outputs if test(row)) / count, 3)

    return {
        "items": count,
        "local_term_generated_local": rate(lambda row: has_local_term(row["generated_local"], terms)),
        "local_term_reference_local": rate(lambda row: has_local_term(row["reference_local"], terms)),
        "local_term_reference_standard": rate(lambda row: has_local_term(row["reference_standard"], terms)),
        "local_term_generated_standard": rate(lambda row: has_local_term(row["generated_standard"], terms)),
        "copies_input_generated_standard": rate(
            lambda row: copies_input(row["generated_standard"], row["context"], row["message"])),
        "copies_input_generated_local": rate(
            lambda row: copies_input(row["generated_local"], row["context"], row["message"])),
        "copies_input_reference_local": rate(
            lambda row: copies_input(row["reference_local"], row["context"], row["message"])),
        # The same rule picks the unlikelihood negatives (finetune_v5), so after that training it is no longer
        # independent evidence; read the replies too.
        "echoes_message_generated_standard": rate(
            lambda row: echo_start(row["generated_standard"], row["message"]) is not None),
        "echoes_message_generated_local": rate(
            lambda row: echo_start(row["generated_local"], row["message"]) is not None),
        "same_reply_both_styles": rate(
            lambda row: row["generated_standard"].strip().lower() == row["generated_local"].strip().lower()),
    }


def decode(checkpoint: Path, name: str, decoding: dict, validation_path: Path | None = None,
           max_source: int = 128, max_target: int = 48, output_dir: Path = ARTIFACTS) -> dict:
    # Imported here so ``summary`` runs on a laptop without loading torch and peft.
    import torch
    from transformers import AutoTokenizer

    from replycoach.generate.finetune_v5 import (
        DEV_VALIDATION, input_format, item_rows, load_finetuned, validation_outputs,
    )

    report_path = output_dir / f"decode_{name}.json"
    if report_path.exists():
        raise ValueError(f"{report_path} already exists; choose a new --name")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_finetuned(checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    checkpoint_format = input_format(checkpoint)
    speaker_labels = checkpoint_format["speaker_labels"]
    outputs = validation_outputs(
        model, tokenizer, item_rows(validation_path or DEV_VALIDATION), max_source, max_target, device, decoding,
        speaker_labels, checkpoint_format["joint"],
    )
    report = {"version": "V5", "stage": "decode", "checkpoint": str(checkpoint), "decoding": decoding,
              "speaker_labels": speaker_labels, "joint": checkpoint_format["joint"],
              "metrics": reply_metrics(outputs), "validation_outputs": outputs}
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def summary_rows(paths: list[Path]) -> list[dict]:
    """One row per report that holds validation outputs (stage 2 runs and decode runs)."""
    lexicon = load_lexicon()
    rows = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report.get("validation_outputs") or "reference_local" not in report["validation_outputs"][0]:
            continue  # stage 1 reports hold public-pair outputs, not development items
        rows.append({"run": path.stem, "best_validation_loss": report.get("best_validation_loss"),
                     "decoding": report.get("decoding"), **reply_metrics(report["validation_outputs"], lexicon)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="V5: re-decode validation replies, or summarise saved runs.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("decode", help="Generate validation replies from a checkpoint with chosen settings.")
    run.add_argument("--checkpoint", type=Path, required=True, help="e.g. artifacts/mt5_base_lora/best")
    run.add_argument("--name", required=True, help="Written to artifacts/decode_<name>.json")
    run.add_argument("--validation-items", type=Path)
    run.add_argument("--num-beams", type=int, default=4)
    run.add_argument("--no-repeat-ngram", type=int, default=3, help="Block n-grams repeated within the reply.")
    run.add_argument("--encoder-no-repeat-ngram", type=int, default=0,
                     help="Block n-grams copied from the input (0 = off, the stage 2 default).")
    run.add_argument("--max-source-tokens", type=int, default=128)
    run.add_argument("--max-target-tokens", type=int, default=48)
    summary = commands.add_parser("summary", help="Print reply metrics for saved run reports.")
    summary.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()

    if args.command == "decode":
        decoding = {"num_beams": args.num_beams, "no_repeat_ngram_size": args.no_repeat_ngram,
                    "encoder_no_repeat_ngram_size": args.encoder_no_repeat_ngram}
        report = decode(args.checkpoint, args.name, decoding, args.validation_items,
                        args.max_source_tokens, args.max_target_tokens)
        print(json.dumps(report["metrics"], indent=2))
        return
    rows = summary_rows(args.reports)
    columns = ["best_validation_loss", "local_term_generated_local", "local_term_reference_local",
               "copies_input_generated_standard", "copies_input_generated_local",
               "echoes_message_generated_standard", "echoes_message_generated_local", "same_reply_both_styles"]
    headers = ["loss", "LOCAL term", "ref term", "copy STD", "copy LOC", "echo STD", "echo LOC", "same"]
    print(f"{'run':<28}" + "".join(f"{header:>11}" for header in headers))
    for row in rows:
        print(f"{row['run']:<28}" + "".join(f"{'-' if row[column] is None else row[column]:>11}" for column in columns))


if __name__ == "__main__":
    main()
