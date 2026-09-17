"""V5 style-control wrappers. Development data only; never touch the locked test set."""

from __future__ import annotations


def style_prefix(style: str) -> str:
    if style not in {"STANDARD", "LOCAL"}:
        raise ValueError(style)
    return style


def mt5_input(style: str, message: str, context: list[str] | None = None) -> str:
    memory = " | ".join(context or [])
    return f"{style_prefix(style)} context: {memory} message: {message}"
