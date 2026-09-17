"""Train V3 RNN encoder-decoder. Week 4: one successful training batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from replycoach.generate import Vocab
from replycoach.generate.seq2seq import Seq2Seq
from replycoach.io import ROOT, load_items
from replycoach.schema import DialogueItem

ARTIFACTS = ROOT / "artifacts"


def style_tag(style: str) -> str:
    return "<standard>" if style == "STANDARD" else "<local>"


def pair_texts(items: list[DialogueItem]) -> list[tuple[str, str]]:
    pairs = []
    for item in items:
        context = " ".join(item.memory_texts())
        src = f"{context} {item.message.text}".strip()
        pairs.append((f"{style_tag('STANDARD')} {src}", item.replies.standard))
        pairs.append((f"{style_tag('LOCAL')} {src}", item.replies.local))
    return pairs


def collate(batch: list[tuple[list[int], list[int]]]):
    src = pad_sequence([torch.tensor(src_ids) for src_ids, _ in batch], batch_first=True, padding_value=0)
    tgt = pad_sequence([torch.tensor(tgt_ids) for _, tgt_ids in batch], batch_first=True, padding_value=0)
    src_lengths = torch.tensor([len(src_ids) for src_ids, _ in batch])
    return src, src_lengths, tgt


def run_one_batch(
    data_path: Path | None = None,
    batch_size: int = 4,
    embed_dim: int = 64,
    hidden_size: int = 128,
    use_attention: bool = False,
    seed: int = 425,
) -> dict:
    torch.manual_seed(seed)
    items = load_items(data_path)
    pairs = pair_texts(items)
    vocab = Vocab.build([text for pair in pairs for text in pair], max_size=5000)
    encoded = [(vocab.encode(src), vocab.encode(tgt)) for src, tgt in pairs]
    batch = encoded[:batch_size]
    src, src_lengths, tgt = collate(batch)
    model = Seq2Seq(len(vocab), embed_dim=embed_dim, hidden_size=hidden_size, use_attention=use_attention)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model.train()
    optimizer.zero_grad()
    logits = model(src, src_lengths, tgt, teacher_forcing=True)
    loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ckpt_path = ARTIFACTS / ("rnn_v4_smoke.pt" if use_attention else "rnn_v3_smoke.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab": vocab.to_dict(),
            "use_attention": use_attention,
            "embed_dim": embed_dim,
            "hidden_size": hidden_size,
            "loss": float(loss.item()),
        },
        ckpt_path,
    )
    report = {
        "version": "V4" if use_attention else "V3",
        "items": len(items),
        "pairs": len(pairs),
        "batch_size": batch_size,
        "vocab_size": len(vocab),
        "loss": round(float(loss.item()), 4),
        "checkpoint": str(ckpt_path),
        "device": "cpu",
    }
    report_path = ARTIFACTS / "rnn_one_batch.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V3/V4 RNN for one batch (Week 4 smoke test).")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--attention", action="store_true", help="Train V4 attention-RNN instead of V3.")
    args = parser.parse_args()
    report = run_one_batch(data_path=args.data, batch_size=args.batch_size, use_attention=args.attention)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
