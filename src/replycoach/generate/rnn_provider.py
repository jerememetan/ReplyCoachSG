"""Inference for task-trained V3/V4 checkpoints, with explicit style tokens."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch

from replycoach.generate import BOS, EOS, PAD, UNK, Vocab
from replycoach.generate.seq2seq import Seq2Seq
from replycoach.schema import Candidate, STYLES

FORMAT = "replycoach-rnn-task-v1"
STYLE_TOKENS = {"STANDARD": "<standard>", "LOCAL": "<local>"}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_source(vocab: Vocab, style: str, message: str, context: list[str], limit: int) -> list[int]:
    """BOS, style, latest context/message tokens, EOS; never truncate the style."""
    if style not in STYLES or limit < 4:
        raise ValueError("invalid style or source token limit")
    words = vocab.encode(" ".join([*context, message]), add_bos=False, add_eos=False)
    return [vocab.token_to_id[BOS], vocab.token_to_id[STYLE_TOKENS[style]],
            *words[-(limit - 3):], vocab.token_to_id[EOS]]


@torch.inference_mode()
def decode(model, vocab, source_ids, max_target, device):
    src = torch.tensor([source_ids], device=device)
    lengths = torch.tensor([len(source_ids)])
    outputs, hidden = model.encoder(src, lengths)
    token = torch.tensor([vocab.token_to_id[BOS]], device=device)
    suppressed = [vocab.token_to_id[t] for t in (PAD, UNK, BOS, *STYLE_TOKENS.values())]
    ids, log_probabilities = [], []
    ended = False
    for step in range(max_target - 1):
        if model.use_attention:
            logits, hidden, _ = model.decoder(token, hidden, outputs)
        else:
            logits, hidden, _ = model.decoder(token, hidden)
        # Choose only resolvable words; score with the ORIGINAL distribution.
        raw_log_probs = logits.log_softmax(-1)
        choices = logits.clone()
        choices[:, suppressed] = -torch.inf
        if step == 0:
            choices[:, vocab.token_to_id[EOS]] = -torch.inf
        token = choices.argmax(-1)
        idx = int(token.item())
        log_probabilities.append(float(raw_log_probs[0, idx]))
        if idx == vocab.token_to_id[EOS]:
            ended = True
            break
        ids.append(idx)
    mean_log_prob = sum(log_probabilities) / len(log_probabilities)
    return vocab.decode(ids), math.exp(mean_log_prob), {
        "generated_tokens": len(ids), "terminated_with_eos": ended,
        "mean_token_log_probability": mean_log_prob,
    }


class RNNGenerator:
    def __init__(self, checkpoint: Path, *, device: str = "cpu"):
        self.path = Path(checkpoint).resolve()
        saved = torch.load(self.path, map_location="cpu", weights_only=True)
        if saved.get("format") != FORMAT or not saved.get("task_trained"):
            raise ValueError("expected a task-trained checkpoint; smoke/public-only weights are not providers")
        self.source = "V4" if saved["use_attention"] else "V3"
        self.vocab = Vocab.from_dict(saved["vocab"])
        if any(token not in self.vocab.token_to_id for token in STYLE_TOKENS.values()):
            raise ValueError("task checkpoint is missing style tokens")
        self.device = torch.device(device)
        self.model = Seq2Seq(len(self.vocab), saved["embed_dim"], saved["hidden_size"],
                             saved["use_attention"]).to(self.device)
        self.model.load_state_dict(saved["model_state"])
        self.model.eval()
        self.max_source = saved["max_source_tokens"]
        self.max_target = saved["max_target_tokens"]
        if self.max_source < 4 or self.max_target < 3:
            raise ValueError("invalid checkpoint token limits")
        self.checkpoint_hash = file_hash(self.path)
        self.provenance = saved["provenance"]
        self.best_epoch = saved["epoch"]

    def generate(self, message: str, context: list[str]) -> list[Candidate]:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("incoming message must be nonempty")
        if not isinstance(context, list) or not all(isinstance(t, str) for t in context):
            raise ValueError("context must be an ordered list of strings")
        candidates = []
        for style in STYLES:
            source = task_source(self.vocab, style, message, context, self.max_source)
            text, score, details = decode(self.model, self.vocab, source, self.max_target, self.device)
            if not text.strip() or not math.isfinite(score):
                raise RuntimeError("decoder did not produce a valid candidate")
            candidates.append(Candidate(
                context=list(context), message=message, candidate_text=text, style=style,
                source=self.source, score=score, item_id=None,
                meta={"checkpoint_id": self.path.name, "checkpoint_sha256": self.checkpoint_hash,
                      "format": FORMAT, "epoch": self.best_epoch, "decoding": "constrained_greedy",
                      "max_source_tokens": self.max_source, "max_target_tokens": self.max_target,
                      "score_method": "exp(mean raw token log probability), including EOS when emitted",
                      "score_limitations": "Uncalibrated model likelihood; not correctness or style accuracy.",
                      "suppressed_tokens": [PAD, UNK, BOS, *STYLE_TOKENS.values()],
                      "minimum_output_tokens": 1, "device": str(self.device),
                      "train_sha256": self.provenance["train_sha256"],
                      "validation_sha256": self.provenance["validation_sha256"], **details},
            ))
        return candidates
