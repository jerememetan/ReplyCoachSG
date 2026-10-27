"""Export a minimal private V6 training bundle from the reviewed development split."""
import argparse
from pathlib import Path

from replycoach.evaluation import DEFAULT_BANK, DEFAULT_QUERIES, load_evaluation_inputs, validate_output_path
from replycoach.reranker_training import build_training_bundle, write_training_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=DEFAULT_BANK)
    parser.add_argument('--validation', type=Path, default=DEFAULT_QUERIES)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    validate_output_path(args.output, (args.train, args.validation))
    bundle = build_training_bundle(load_evaluation_inputs(args.validation, args.train))
    write_training_bundle(args.output, bundle)
    print({split: {'pairs': len(bundle[split]), 'positives': sum(p['label'] for p in bundle[split])}
           for split in ['train', 'validation']})


if __name__ == '__main__':
    main()
