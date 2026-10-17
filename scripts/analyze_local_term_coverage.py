"""Read-only local-term coverage analysis for ReplyCoachSG development exports."""

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEXICON_PATH = ROOT / "data" / "lexicons" / "local_terms.json"
EXPORTS = sorted((ROOT / "experiments").glob("retrieval_*.jsonl"))


def load_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def contains_surface(text, surface):
    text = text.casefold()
    surface = surface.casefold()

    # Chinese characters are matched directly; Latin terms require boundaries.
    if any("\u3400" <= char <= "\u9fff" for char in surface):
        return surface in text

    pattern = (
        r"(?<![A-Za-z0-9_])"
        + re.escape(surface)
        + r"(?![A-Za-z0-9_])"
    )
    return re.search(pattern, text) is not None


def matched_terms(text, terms):
    return {
        term["surface"]
        for term in terms
        if term.get("apply", True)
        and contains_surface(text, term["surface"])
    }


def main():
    lexicon = json.loads(LEXICON_PATH.read_text(encoding="utf-8"))
    terms = lexicon["terms"]

    # Do not count terms explicitly marked apply:false.
    eligible = [t for t in terms if t.get("apply", True)]
    core = [t for t in eligible if t.get("kind") not in
            {"particle", "zh", "pinyin"}]
    particles = [t for t in eligible if t.get("kind") == "particle"]
    mandarin = [t for t in eligible if t.get("kind") in {"zh", "pinyin"}]

    if not EXPORTS:
        raise SystemExit("No experiments/retrieval_*.jsonl files found.")

    print("Read-only development analysis")
    print("Exports:", len(EXPORTS))
    print("Eligible core terms:", len(core))
    print("Eligible particles:", len(particles))
    print("Mandarin/pinyin terms tracked separately:", len(mandarin))
    print()

    for path in EXPORTS:
        rows = load_jsonl(path)
        preservation_total = 0
        preservation_kept = 0
        unprompted_core = Counter()
        local_core_total = 0
        standard_core_total = 0
        local_particle_total = 0
        standard_particle_total = 0
        mandarin_additions = 0
        fallbacks = Counter()

        for row in rows:
            # Preservation is measured against the current message only.
            message = row["message"]
            memory = " ".join(row.get("memory", []))
            local_text = row["local"]["candidate_text"]
            standard_text = row["standard"]["candidate_text"]

            source_core = matched_terms(message, core)
            local_core = matched_terms(local_text, core)
            standard_core = matched_terms(standard_text, core)
            source_particles = matched_terms(message, particles)
            local_particles = matched_terms(local_text, particles)
            standard_particles = matched_terms(standard_text, particles)

            preservation_total += len(source_core)
            preservation_kept += len(source_core & local_core)

            grounded_text = f"{message} {memory}"
            grounded_core = matched_terms(grounded_text, core)
            for term in local_core - grounded_core:
                unprompted_core[term] += 1

            local_core_total += len(local_core)
            standard_core_total += len(standard_core)
            local_particle_total += len(local_particles)
            standard_particle_total += len(standard_particles)

            # New Mandarin/pinyin terms are diagnostic, not rewarded.
            source_mandarin = matched_terms(message, mandarin)
            reply_mandarin = matched_terms(local_text, mandarin)
            mandarin_additions += len(reply_mandarin - source_mandarin)

            for style in ("standard", "local"):
                if row[style]["source"] == "fallback":
                    fallbacks[style] += 1

        coverage = (
            100 * preservation_kept / preservation_total
            if preservation_total else None
        )

        print(f"=== {path.name} ===")
        print(f"Rows: {len(rows)}")
        print(
            "Core source-term preservation: "
            f"{preservation_kept}/{preservation_total}"
            + (f" ({coverage:.1f}%)" if coverage is not None
               else " (N/A: no source terms)")
        )
        print(f"Core local-term matches in LOCAL replies: {local_core_total}")
        print(f"Unprompted core term additions: {dict(unprompted_core)}")
        print(f"Core local-term matches in STANDARD replies: {standard_core_total}")
        print(
            "Particle matches (LOCAL / STANDARD): "
            f"{local_particle_total} / {standard_particle_total}"
        )
        print(f"New Mandarin/pinyin surfaces in LOCAL replies: {mandarin_additions}")
        print(f"Fallback replies (STANDARD / LOCAL): "
              f"{fallbacks['standard']} / {fallbacks['local']}")
        print()


if __name__ == "__main__":
    main()
