"""V5 style-control wrappers. Development data only; never touch the locked test set."""

from __future__ import annotations

import re

# Item speakers as input labels; each is one mT5 token ("Them" would split into two).
SPEAKER_LABELS = {"peer": "Friend", "user": "You", "other": "Other"}


# BOTH asks for the two styles in one reply, STANDARD first: "<standard> | LOCAL: <local>" (finetune_v5 --joint).
JOINT_STYLE = "BOTH"
REWRITE_STYLE = "LOCAL_REWRITE"
JOINT_SEPARATOR = " | LOCAL: "
JOINT_SPLIT = re.compile(r"\s*\|\s*LOCAL\s*:\s*")


def joint_reply(standard: str, local: str) -> str:
    return f"{standard.strip()}{JOINT_SEPARATOR}{local.strip()}"


def split_joint(text: str) -> tuple[str, str] | None:
    """The STANDARD and LOCAL parts of a joint reply, or None if the model left out the separator."""
    parts = JOINT_SPLIT.split(text, maxsplit=1)
    return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else None


def style_prefix(style: str) -> str:
    # REPLY tags training-only pairs whose reply was not written as either style (see finetune_v5.context_pairs).
    if style not in {"STANDARD", "LOCAL", "REPLY", JOINT_STYLE, REWRITE_STYLE}:
        raise ValueError(style)
    return style


def mt5_input(style: str, message: str, context: list[str] | None = None, speaker_labels: bool = False,
              speakers: list[str] | None = None, standard_reply: str | None = None) -> str:
    """The V5 source text. With ``speaker_labels`` each turn says who wrote it and the input ends at ``You:``.

    ``speakers`` gives each context turn's item speaker (``peer``, ``user`` or ``other``); the incoming message
    is always from the peer. Without them, as when the app passes plain strings, speakers are guessed by
    assuming the two people take turns back from the message, which 225 of the 240 development items do.
    """
    if style == REWRITE_STYLE:
        if not isinstance(standard_reply, str) or not standard_reply.strip():
            raise ValueError("LOCAL_REWRITE requires a nonempty chosen standard reply")
        source = mt5_input("STANDARD", message, context, speaker_labels, speakers)
        return f"{REWRITE_STYLE} {source[len('STANDARD '):]} | Standard draft: {standard_reply.strip()} | Singlish rewrite:"
    if standard_reply is not None:
        raise ValueError("standard_reply is only valid for LOCAL_REWRITE")
    if not speaker_labels:
        memory = " | ".join(context or [])
        return f"{style_prefix(style)} context: {memory} message: {message}"
    context = list(context or [])
    if speakers is None:
        speakers = ["user" if (len(context) - index) % 2 else "peer" for index in range(len(context))]
    if len(speakers) != len(context):
        raise ValueError(f"{len(speakers)} speakers for {len(context)} context turns")
    turns = [f"{SPEAKER_LABELS[speaker]}: {text}" for speaker, text in zip([*speakers, "peer"], [*context, message])]
    return f"{style_prefix(style)} {' | '.join(turns)} | You:"
