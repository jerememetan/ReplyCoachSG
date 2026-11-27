"""V5 sample-and-rerank: generate many replies per validation item, then pick one with a weighted score.

Step 1, on the GPU (``candidates``): for each item and style, the usual beam-search reply plus ``samples``
nucleus-sampled replies, each scored by the model twice. Step 2, on a laptop (``pick``): choose one reply by

    score = fit - mmi * generic - copy * copied + style * marker

fit      mean log-probability of the reply's tokens given the conversation; higher means the model likes it more
generic  the same given only the style tag; high for replies that fit any conversation ("ok, see you later")
copied   share of the reply's 3-word runs that also appear in the context or message, from 0 to 1
marker   1 if the reply has a local marker; it counts for a LOCAL reply and against a STANDARD one

All weights at 0 pick the reply the model itself likes most. Subtracting ``generic`` is the anti-language-model
form of Li et al. (2016), "A Diversity-Promoting Objective Function for Neural Conversation Models".
The weights are tried on the same 50 validation items they are measured on, so read a gain as optimistic.

    python -m replycoach.generate.rerank_v5 candidates --checkpoint artifacts/base_spans_lora/best --name spans16
    python -m replycoach.generate.rerank_v5 pick artifacts/candidates_spans16.json --grid
    python -m replycoach.generate.rerank_v5 pick artifacts/candidates_spans16.json --name spans16_c2 --copy 2

With ``--validation-items data/dev/train.jsonl`` the candidates are for the training items instead; their echo
replies are the negatives for unlikelihood training (``finetune_v5 --negatives``).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from replycoach.generate.decode_v5 import has_local_term, marker_terms, ngrams, reply_metrics
from replycoach.generate.train_v5 import ARTIFACTS
from replycoach.tokenize import tokenize

STYLES = ("STANDARD", "LOCAL")


def copied_share(reply: str, context: list[str], message: str, n: int = 3) -> float:
    """Share of the reply's ``n``-word runs found in the input; 0 for replies shorter than ``n`` words."""
    reply_ngrams = ngrams(tokenize(reply), n)
    if not reply_ngrams:
        return 0.0
    seen = set().union(*(ngrams(tokenize(text), n) for text in [*context, message]))
    return len(reply_ngrams & seen) / len(reply_ngrams)


def reply_logprobs(model, tokenizer, rows: list[dict], style: str, max_source: int, max_target: int, device,
                   speaker_labels: bool = False, batch_size: int = 64) -> list[float]:
    """Mean log p(token) of each row's ``response`` given its input, under teacher forcing."""
    import torch

    from replycoach.generate.train_v5 import collate, encode_rows

    model.eval()
    encoded = encode_rows(rows, tokenizer, max_source, max_target, style=style, speaker_labels=speaker_labels)
    scores = []
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            input_ids, attention_mask, labels = collate(encoded[start : start + batch_size], tokenizer.pad_token_id)
            labels = labels.to(device)
            logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels).logits
            # The log-probability the model gave each actual reply token; padding (-100) is masked out.
            mask = labels.ne(-100)
            token_logp = logits.log_softmax(-1).gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            scores += ((token_logp * mask).sum(-1) / mask.sum(-1)).tolist()
    return scores


def sample_replies(model, tokenizer, rows: list[dict], style: str, samples: int, top_p: float, max_source: int,
                   max_target: int, device, speaker_labels: bool = False, chunk: int = 8) -> list[list[str]]:
    """``samples`` replies per row by nucleus sampling: each next token is drawn from the smallest set of tokens
    whose probabilities add up to ``top_p`` (Holtzman et al., 2020), so replies vary but stay plausible."""
    from replycoach.generate.train_v5 import collate, encode_rows

    model.eval()
    replies = []
    for start in range(0, len(rows), chunk):
        encoded = encode_rows(rows[start : start + chunk], tokenizer, max_source, max_target, style=style,
                              speaker_labels=speaker_labels)
        input_ids, attention_mask, _ = collate(encoded, tokenizer.pad_token_id)
        output = model.generate(
            input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), max_new_tokens=max_target,
            do_sample=True, top_p=top_p, num_return_sequences=samples,
        )
        texts = tokenizer.batch_decode(output, skip_special_tokens=True)
        replies += [texts[index : index + samples] for index in range(0, len(texts), samples)]
    return replies


def candidates(checkpoint: Path, name: str, samples: int = 16, top_p: float = 0.9, seed: int = 425,
               validation_path: Path | None = None, max_source: int = 128, max_target: int = 48,
               output_dir: Path = ARTIFACTS) -> dict:
    """Generate and score candidate replies for every validation item and style; write candidates_<name>.json."""
    import torch
    from transformers import AutoTokenizer

    from replycoach.generate.finetune_v5 import DEV_VALIDATION, input_format, item_rows, load_finetuned
    from replycoach.generate.train_v5 import generate_replies

    report_path = output_dir / f"candidates_{name}.json"
    if report_path.exists():
        raise ValueError(f"{report_path} already exists; choose a new --name")
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_finetuned(checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if input_format(checkpoint)["joint"]:
        raise ValueError(f"{checkpoint} writes both styles in one reply; candidates are made one style at a time")
    speaker_labels = input_format(checkpoint)["speaker_labels"]
    validation_rows = item_rows(validation_path or DEV_VALIDATION)
    rows = [row for row in validation_rows if row["style"] == "STANDARD"]  # one row per item
    local_references = {row["item_id"]: row["response"] for row in validation_rows if row["style"] == "LOCAL"}
    settings = (max_source, max_target, device)

    items = [{"item_id": row["item_id"], "context": row["context"], "message": row["message"],
              "reference_standard": row["response"], "reference_local": local_references[row["item_id"]],
              "candidates": {}} for row in rows]
    for style in STYLES:
        beam = generate_replies(model, tokenizer, rows, style, *settings, speaker_labels=speaker_labels)
        sampled = sample_replies(model, tokenizer, rows, style, samples, top_p, *settings, speaker_labels)
        # The beam reply first, then distinct non-empty samples, so the old output can always be picked.
        texts = [list(dict.fromkeys(text.strip() for text in [beam[index], *sampled[index]] if text.strip()))
                 for index in range(len(rows))]
        flat = [(row, text) for row, item_texts in zip(rows, texts) for text in item_texts]
        fit = reply_logprobs(model, tokenizer, [{**row, "response": text} for row, text in flat], style,
                             *settings, speaker_labels)
        # The same replies after a bare input: no context, empty message, so only the style tag is left.
        generic = reply_logprobs(
            model, tokenizer,
            [{**row, "context": [], "speakers": [], "message": "", "response": text} for row, text in flat],
            style, *settings, speaker_labels,
        )
        scores = iter(zip(fit, generic))
        for item, item_texts, beam_text in zip(items, texts, beam):
            item["candidates"][style] = []
            for text in item_texts:
                fit_score, generic_score = next(scores)
                item["candidates"][style].append({"text": text, "beam": text == beam_text.strip(),
                                                  "fit": round(fit_score, 4), "generic": round(generic_score, 4)})

    report = {"version": "V5", "stage": "candidates", "checkpoint": str(checkpoint), "speaker_labels": speaker_labels,
              "samples": samples, "top_p": top_p, "seed": seed, "items": items}
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def pick(report: dict, mmi: float = 0.0, copy: float = 0.0, style: float = 0.0) -> list[dict]:
    """One reply per item and style by the weighted score, in the ``validation_outputs`` format."""
    terms = marker_terms()

    def score(item: dict, tag: str, candidate: dict) -> float:
        marker = has_local_term(candidate["text"], terms)
        return (candidate["fit"] - mmi * candidate["generic"]
                - copy * copied_share(candidate["text"], item["context"], item["message"])
                + style * (marker if tag == "LOCAL" else -marker))

    outputs = []
    for item in report["items"]:
        chosen = {tag: max(item["candidates"][tag], key=lambda candidate: score(item, tag, candidate))
                  for tag in STYLES}
        outputs.append({
            "item_id": item["item_id"], "context": item["context"], "message": item["message"],
            "reference_standard": item["reference_standard"], "generated_standard": chosen["STANDARD"]["text"],
            "reference_local": item["reference_local"], "generated_local": chosen["LOCAL"]["text"],
            "beam_kept": {tag: chosen[tag]["beam"] for tag in STYLES},
        })
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="V5 sample-and-rerank on the validation items.")
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("candidates", help="GPU: generate and score candidate replies.")
    make.add_argument("--checkpoint", type=Path, required=True)
    make.add_argument("--name", required=True, help="Written to artifacts/candidates_<name>.json")
    make.add_argument("--samples", type=int, default=16)
    make.add_argument("--top-p", type=float, default=0.9)
    make.add_argument("--seed", type=int, default=425)
    make.add_argument("--validation-items", type=Path)
    choose = commands.add_parser("pick", help="Laptop: choose replies from a candidates file.")
    choose.add_argument("candidates", type=Path)
    choose.add_argument("--mmi", type=float, default=0.0, help="Weight against generic replies.")
    choose.add_argument("--copy", type=float, default=0.0, help="Weight against copying the input.")
    choose.add_argument("--style", type=float, default=0.0, help="Weight for local markers in LOCAL only.")
    choose.add_argument("--name", help="Also write artifacts/rerank_<name>.json for the summary table.")
    choose.add_argument("--grid", action="store_true", help="Print metrics for a small set of weights instead.")
    args = parser.parse_args()

    if args.command == "candidates":
        report = candidates(args.checkpoint, args.name, args.samples, args.top_p, args.seed, args.validation_items)
        counts = [len(item["candidates"][tag]) for item in report["items"] for tag in STYLES]
        print(f"{len(report['items'])} items, {sum(counts) / len(counts):.1f} distinct candidates per item and style")
        return
    report = json.loads(args.candidates.read_text(encoding="utf-8"))
    settings = (list(itertools.product((0.0, 0.5), (0.0, 2.0), (0.0, 1.0))) if args.grid
                else [(args.mmi, args.copy, args.style)])
    headers = ["LOCAL term", "copy STD", "copy LOC", "same", "beam kept"]
    print(f"{'mmi':>5}{'copy':>6}{'style':>7}" + "".join(f"{header:>11}" for header in headers))
    for mmi, copy, style in settings:
        outputs = pick(report, mmi, copy, style)
        metrics = reply_metrics(outputs)
        kept = sum(sum(row["beam_kept"].values()) for row in outputs) / (2 * len(outputs))
        values = [metrics["local_term_generated_local"], metrics["copies_input_generated_standard"],
                  metrics["copies_input_generated_local"], metrics["same_reply_both_styles"], round(kept, 3)]
        print(f"{mmi:>5}{copy:>6}{style:>7}" + "".join(f"{value:>11}" for value in values))
    if args.name and not args.grid:
        path = ARTIFACTS / f"rerank_{args.name}.json"
        path.write_text(json.dumps({"version": "V5", "stage": "rerank", "candidates": str(args.candidates),
                                    "weights": {"mmi": args.mmi, "copy": args.copy, "style": args.style},
                                    "metrics": reply_metrics(outputs), "validation_outputs": outputs},
                                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
