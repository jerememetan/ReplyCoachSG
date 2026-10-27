"""Run the fixed V6 cross-encoder experiment; production training requires CUDA."""
import argparse
from pathlib import Path

from replycoach.reranker_fit import run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--batch-size', type=int, default=16)
    additions = parser.add_mutually_exclusive_group()
    additions.add_argument('--reviewed-pairs', type=Path, help='Exported completed human-review sidecar')
    additions.add_argument('--augmentation-map', type=Path,
                        help='Optional packaged abbreviations.json; omitted preserves baseline training')
    parser.add_argument('--augmentation-copies', type=int, default=3)
    parser.add_argument('--augmentation-probability', type=float, default=.35)
    parser.add_argument('--augmentation-seed', type=int, default=425)
    args = parser.parse_args()
    run_training(args.bundle, args.output, smoke=args.smoke, max_steps=args.max_steps,
                 batch_size=args.batch_size, augmentation_map=args.augmentation_map,
                 augmentation_copies=args.augmentation_copies,
                 augmentation_probability=args.augmentation_probability,
                 augmentation_seed=args.augmentation_seed, reviewed_pairs=args.reviewed_pairs)


if __name__ == '__main__':
    main()
