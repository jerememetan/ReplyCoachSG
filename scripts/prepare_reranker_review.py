"""Mine a training-only review queue, or export its completed human decisions."""
import argparse
import json
from pathlib import Path

from replycoach.evaluation import DEFAULT_BANK, DEFAULT_QUERIES, load_evaluation_inputs, validate_output_path
from replycoach.reranker_training import build_training_bundle
from replycoach.reranker_review import create_review_file, export_reviewed_pairs, load_review_file, mine_review_batch
from replycoach.retrieval.v0 import V0Retriever
from replycoach.shared_ranker import CrossEncoderScorer
from replycoach.io import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train', type=Path, default=DEFAULT_BANK)
    parser.add_argument('--validation', type=Path, default=DEFAULT_QUERIES)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'artifacts/reranker_v6/best')
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--export-review', type=Path, help='Export this completed queue instead of mining')
    args = parser.parse_args()
    validate_output_path(args.output, (args.train, args.validation))
    inputs = load_evaluation_inputs(args.validation, args.train)
    bundle = build_training_bundle(inputs)
    if args.export_review:
        payload = export_reviewed_pairs(load_review_file(args.export_review, bundle), bundle)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps({'status': 'reviewed pairs exported', 'pairs': len(payload['pairs']),
                          'skipped': payload['skipped_count']}))
    else:
        batch = mine_review_batch(bundle, V0Retriever.from_items(inputs.bank.items),
                                  CrossEncoderScorer(checkpoint=args.checkpoint), count=args.count)
        create_review_file(args.output, batch, bundle)
        print(json.dumps({'status': 'pending human review', 'comparisons': len(batch['comparisons']),
                          'output': str(args.output), 'training_labels_assigned': 0}))


if __name__ == '__main__':
    main()
