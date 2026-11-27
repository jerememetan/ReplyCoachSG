"""V5 clean CoSEM: drop stage 1 pairs whose reply does not answer its message.

In a hand-check of 60 random full-CoSEM pairs, 20 replies did not answer the message they were paired with: they
answered an older turn, changed topic, or followed a filler message ("Haha", "HMMM"). Stage 1 learns from those
that a reply need not answer the message. Following Akama et al. (2020), "Filtering Noisy Dialogue Corpora by
Connectivity and Content Relatedness", each pair is scored by how much the message helps a trained model predict
the reply:

    relatedness = fit - without

fit      mean log-probability of the reply's tokens given the context and the message (``reply_logprobs``)
without  the same with the message left empty; high when the reply follows from the context alone, or from nothing

A reply that answers the message becomes much more likely once the message is shown, so it scores high; a reply
that ignores the message scores near 0. The filter keeps the highest-scoring share of pairs. Nothing is generated
or labelled: the model's scores only decide which real pairs to leave out.

Step 1, on the GPU (``score``), writes artifacts/scores_<pairs file name>.json:

    python -m replycoach.generate.clean_v5 score --checkpoint artifacts/mt5_base_cosem_full/best \\
        --pairs data/public/cosem_full_train_pairs.jsonl

Step 2, anywhere (``filter``):

    python -m replycoach.generate.clean_v5 filter --pairs data/public/cosem_full_train_pairs.jsonl \\
        --scores artifacts/scores_cosem_full_train_pairs.json --keep 0.7 --output data/public/cosem_clean_train_pairs.jsonl

The CoSEM validation file is not filtered, so stage 1 validation stays on the same pairs as earlier runs. Expect
its CoSEM loss to rise a little, as those pairs keep their noise; judge the clean base by stage 2 and by rating.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from replycoach.generate.train_v5 import ARTIFACTS, load_pairs

CHUNK = 20000  # pairs encoded at a time, so 275,000 pairs never sit in memory as token ids at once


def scores_path(pairs_path: Path, output_dir: Path = ARTIFACTS) -> Path:
    return output_dir / f"scores_{Path(pairs_path).stem}.json"


def score(checkpoint: Path, pairs_path: Path, batch_size: int = 64, max_source: int = 128, max_target: int = 48,
          output_dir: Path = ARTIFACTS) -> dict:
    """Score every pair in ``pairs_path`` with and without its message; write ``scores_path(pairs_path)``."""
    import torch
    from transformers import AutoTokenizer

    from replycoach.generate.finetune_v5 import input_format, load_finetuned
    from replycoach.generate.rerank_v5 import reply_logprobs

    path = scores_path(pairs_path, output_dir)
    if path.exists():
        raise ValueError(f"{path} already exists; delete it to score again")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_finetuned(checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    speaker_labels = input_format(checkpoint)["speaker_labels"]
    rows = load_pairs(pairs_path)
    fit, without = [], []
    for start in range(0, len(rows), CHUNK):
        chunk = rows[start : start + CHUNK]
        # No style given, so each pair gets the tag stage 1 trained its source with (LOCAL for CoSEM).
        settings = (None, max_source, max_target, device, speaker_labels, batch_size)
        fit += reply_logprobs(model, tokenizer, chunk, *settings)
        without += reply_logprobs(model, tokenizer, [{**row, "message": ""} for row in chunk], *settings)
        print(json.dumps({"scored": len(fit), "of": len(rows)}), flush=True)

    report = {"version": "V5", "stage": "clean_scores", "checkpoint": str(checkpoint), "pairs": str(pairs_path),
              "speaker_labels": speaker_labels, "fit": [round(value, 4) for value in fit],
              "without": [round(value, 4) for value in without]}
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    return report


def relatedness(scores: dict) -> list[float]:
    return [fit - without for fit, without in zip(scores["fit"], scores["without"])]


def read_rows(pairs_path: Path) -> list[str]:
    """The pair file's lines as written, in the order ``load_pairs`` reads them (blank lines skipped)."""
    return [line for line in Path(pairs_path).read_text(encoding="utf-8").splitlines() if line.strip()]


def keep_threshold(values: list[float], keep: float) -> float:
    """The relatedness at or above which the top ``keep`` share of pairs lies."""
    if not 0 < keep <= 1:
        raise ValueError("--keep must be a share between 0 and 1")
    ordered = sorted(values, reverse=True)
    return ordered[max(0, round(keep * len(ordered)) - 1)]


def filter_pairs(pairs_path: Path, scores: dict, keep: float, output_path: Path) -> dict:
    """Write the pairs of ``pairs_path`` whose relatedness is in the top ``keep`` share, in their original order."""
    lines = read_rows(pairs_path)
    values = relatedness(scores)
    if len(values) != len(lines):
        raise ValueError(f"{len(values)} scores for {len(lines)} pairs; score this exact file again")
    if Path(output_path).resolve() == Path(pairs_path).resolve():
        raise ValueError("write the clean pairs to a new file")
    threshold = keep_threshold(values, keep)
    kept = [line for line, value in zip(lines, values) if value >= threshold]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
    return {"pairs": len(lines), "kept": len(kept), "threshold": round(threshold, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Drop stage 1 pairs whose reply does not answer its message.")
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("score", help="GPU: score each pair with and without its message.")
    make.add_argument("--checkpoint", type=Path, required=True, help="A stage 1 checkpoint trained on these pairs.")
    make.add_argument("--pairs", type=Path, required=True)
    make.add_argument("--batch-size", type=int, default=64)
    keep = commands.add_parser("filter", help="Keep the pairs whose reply depends most on the message.")
    keep.add_argument("--pairs", type=Path, required=True)
    keep.add_argument("--scores", type=Path, required=True)
    keep.add_argument("--keep", type=float, required=True, help="Share of pairs to keep, e.g. 0.7.")
    keep.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "score":
        report = score(args.checkpoint, args.pairs, args.batch_size)
        print(f"scored {len(report['fit'])} pairs; wrote {scores_path(args.pairs)}")
        return
    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    result = filter_pairs(args.pairs, scores, args.keep, args.output)
    print(f"kept {result['kept']} of {result['pairs']} pairs (relatedness >= {result['threshold']}); "
          f"wrote {args.output}")


if __name__ == "__main__":
    main()
