"""Prepare allowlisted private Kaggle uploads after the optimizer smoke check."""
import argparse
import json
from pathlib import Path

from replycoach.reranker_packaging import package_kaggle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--reviewed-pairs', type=Path)
    args = parser.parse_args()
    print(json.dumps(package_kaggle(args.bundle, args.output, owner=args.owner, tag=args.tag,
                                   reviewed_pairs=args.reviewed_pairs), indent=2))


if __name__ == '__main__':
    main()
