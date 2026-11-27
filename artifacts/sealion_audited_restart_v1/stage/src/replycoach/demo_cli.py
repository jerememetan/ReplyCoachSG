"""CLI demo: retrieve a standard English reply and a light local reply."""

from __future__ import annotations

import argparse
import json
import sys

from replycoach.pipeline import ReplyCoach


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def format_result(result, show_candidates: bool = True) -> str:
    lines = [
        f"message:     {result.message}",
        f"normalised:  {result.normalised_message}",
        f"detected:    {', '.join(term['surface'] for term in result.detected_terms) or '(none)'}",
        f"memory:      {result.memory or '(none)'}",
        "",
        f"STANDARD [{result.standard.source} {result.standard.score:.3f}]",
        f"  {result.standard.candidate_text}",
        f"LOCAL    [{result.local.source} {result.local.score:.3f}]",
        f"  {result.local.candidate_text}",
    ]
    if show_candidates:
        lines.append("\nAll candidates:")
        for cand in result.candidates:
            lines.append(
                f"  [{cand.style:9} {cand.source:8} {cand.score:6.3f} {cand.item_id}] {cand.candidate_text}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    _configure_stdio()
    parser = argparse.ArgumentParser(description="Reply Coach V0 demo")
    parser.add_argument("message", nargs="?", default="eh later go makan? can 吃饭 then library")
    parser.add_argument("--session", nargs="*", default=["lecture ending soon sia", "yeah I'm starving"])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--memory", default="M2", choices=["M0", "M1", "M2"])
    args = parser.parse_args(argv)
    coach = ReplyCoach.from_pilot(memory_mode=args.memory)
    result = coach.suggest(args.message, session=args.session)
    if args.json:
        print(
            json.dumps(
                {
                    "message": result.message,
                    "normalised_message": result.normalised_message,
                    "detected_terms": result.detected_terms,
                    "memory": result.memory,
                    "standard": result.standard.to_dict(),
                    "local": result.local.to_dict(),
                    "candidates": [c.to_dict() for c in result.candidates],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(format_result(result))


if __name__ == "__main__":
    main()
