"""V5 for the app: a trained stage 2 checkpoint answering one message in both styles (``providers.load_v5``).

The checkpoint is ``$V5_CHECKPOINT`` if set, else ``DEFAULT_CHECKPOINT``. Its ``v5_input.json`` says how it was
trained, so the input is built the same way: the style tag, ``Friend:``/``You:`` speaker labels if it used them,
and one joint ``BOTH`` reply split into the two styles if it was trained with ``--joint``.

Each Candidate's score is exp(mean token log-probability) of the reply under the model, so a value in (0, 1]:
the geometric mean of the probabilities of the reply's tokens. It says how sure the model is of its own wording,
not how good the reply is; short, common replies ("okay, see you") score high, and a joint reply's two halves
share the score of the whole reply.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from replycoach.generate.style_control import JOINT_STYLE, split_joint
from replycoach.generate.train_v5 import ARTIFACTS
from replycoach.io import ROOT
from replycoach.schema import STYLES, Candidate

CHECKPOINT_ENV = "V5_CHECKPOINT"
# The best V5 run as of 2026-10-25: speaker labels, context pairs, unlikelihood weight 0.25, seed 425.
DEFAULT_CHECKPOINT = ARTIFACTS / "spk1_ul025_s425" / "best"
DECODING = {"num_beams": 4, "no_repeat_ngram_size": 3}


def configured_checkpoint() -> Path:
    """``$V5_CHECKPOINT`` (relative paths are from the repo root), else ``DEFAULT_CHECKPOINT``."""
    path = Path(os.environ.get(CHECKPOINT_ENV) or DEFAULT_CHECKPOINT)
    return path if path.is_absolute() else ROOT / path


def load_checkpoint(checkpoint: Path, device):
    """A full model, or a LoRA adapter on its base; a relative base path is read from the repo root, so loading
    does not depend on the folder the app or export was started from."""
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM

    adapter_config = checkpoint / "adapter_config.json"
    if not adapter_config.exists():
        return AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
    base = Path(json.loads(adapter_config.read_text(encoding="utf-8"))["base_model_name_or_path"])
    if not base.is_absolute():
        base = ROOT / base
    model = AutoModelForSeq2SeqLM.from_pretrained(base)
    return PeftModel.from_pretrained(model, checkpoint).to(device)


class V5Generator:
    source = "V5"

    def __init__(self, checkpoint: Path, max_source: int = 128, max_target: int = 48, device=None):
        import torch
        from transformers import AutoTokenizer

        from replycoach.generate.finetune_v5 import input_format

        self.checkpoint = Path(checkpoint)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_checkpoint(self.checkpoint, self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.format = input_format(self.checkpoint)
        self.max_source, self.max_target = max_source, max_target

    def generate(self, message: str, context: list[str]) -> list[Candidate]:
        """One STANDARD and one LOCAL Candidate for ``message`` after ``context`` (oldest turn first).

        The app passes context as plain strings, so with speaker labels ``mt5_input`` assumes the two people
        took turns back from the message, which holds for 225 of the 240 development items.
        """
        from replycoach.generate.finetune_v5 import joint_decoding
        from replycoach.generate.rerank_v5 import reply_logprobs
        from replycoach.generate.train_v5 import generate_replies

        row = {"context": list(context), "message": message, "response": ""}
        speaker_labels, joint = self.format["speaker_labels"], self.format["joint"]
        if joint:
            limit = 2 * self.max_target
            [text] = generate_replies(self.model, self.tokenizer, [row], JOINT_STYLE, self.max_source, limit,
                                      self.device, joint_decoding(self.tokenizer, DECODING), speaker_labels)
            halves = split_joint(text)
            replies = dict(zip(STYLES, halves or (text.strip(), text.strip())))
            scored = [(JOINT_STYLE, text)]
        else:
            limit = self.max_target
            replies = {style: generate_replies(self.model, self.tokenizer, [row], style, self.max_source, limit,
                                               self.device, DECODING, speaker_labels)[0].strip()
                       for style in STYLES}
            scored = list(replies.items())
        logprobs = {style: reply_logprobs(self.model, self.tokenizer, [{**row, "response": text}], style,
                                          self.max_source, limit, self.device, speaker_labels)[0]
                    for style, text in scored}
        meta = {
            "checkpoint": str(self.checkpoint), "speaker_labels": speaker_labels, "joint": joint,
            "decoding": {**DECODING, "max_new_tokens": limit, "joint_repeats_blocked_per_half": joint},
            "speakers": "guessed: alternating back from the message" if speaker_labels and context else None,
            "score": "exp(mean token log-probability) of the reply under the model",
            **({"joint_split": halves is not None} if joint else {}),
        }
        candidates = []
        for style in STYLES:
            if not replies[style]:
                raise ValueError(f"V5 generated an empty {style} reply")
            logprob = logprobs[JOINT_STYLE if joint else style]
            candidate = Candidate(context=list(context), message=message, candidate_text=replies[style], style=style,
                                  source=self.source, score=min(1.0, math.exp(logprob)), meta=meta)
            candidate.validate()
            candidates.append(candidate)
        return candidates
