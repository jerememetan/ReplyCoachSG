from replycoach.io import load_items
from replycoach.logging_utils import log_event
from replycoach.memory import select_memory
from replycoach.retrieval.v1 import detect_and_normalise, retrieval_query
from replycoach.retrieval.v0 import V0Retriever


PILOT_IDS = [
    "PILOT-001",
    "PILOT-002",
    "PILOT-007",
    "PILOT-012",
    "PILOT-019",
]


def main() -> None:
    items = load_items()
    items_by_id = {item.id: item for item in items}

    retriever = V0Retriever.from_items(items)

    for pilot_id in PILOT_IDS:
        item = items_by_id[pilot_id]

        message = item.message.text
        session = item.memory_texts()

        memory = select_memory(session, message, mode="M2")

        detection = detect_and_normalise(message)

        query = retrieval_query(" ".join(memory + [message]))

        retrieved = retriever.retrieve(
            query,
            context=memory,
            top_k=3,
            method="bm25",
            source="V0",
        )

        pairs = {}

        for candidate in retrieved:
            rank = candidate.meta["rank"]
            pair = pairs.setdefault(
                rank,
                {
                    "rank": rank,
                    "item_id": candidate.item_id,
                    "score": candidate.score,
                },
            )

            pair[candidate.style.lower()] = candidate.candidate_text

        top3_pairs = [pairs[rank] for rank in sorted(pairs)]

        log_event(
            "b_week4_retrieval",
            {
                "pilot_id": pilot_id,
                "message": message,
                "normalised": detection.normalised,
                "memory": memory,
                "method": "bm25",
                "top_k": 3,
                "retrieved_pairs": top3_pairs,
            },
        )

        print(f"\n{pilot_id}")
        print(f"Message: {message}")
        print(f"Memory: {memory}")

        for pair in top3_pairs:
            print(
                f"  #{pair['rank']} {pair['item_id']} "
                f"(score={pair['score']:.3f})"
            )
            print(f"     STANDARD: {pair['standard']}")
            print(f"     LOCAL:    {pair['local']}")


if __name__ == "__main__":
    main()