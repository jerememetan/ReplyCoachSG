"""Two-stage Qwen reply-suitability training. Intended to be run by the user on Kaggle."""
import argparse
import json
from pathlib import Path

from replycoach.dailydialog_reranker import run_pipeline
from replycoach.qwen_reranker import MODEL


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',default=MODEL)
    parser.add_argument('--max-train-queries',type=int,default=4000)
    parser.add_argument('--max-validation-queries',type=int,default=400)
    parser.add_argument('--adaptation-epochs',type=int,default=2)
    parser.add_argument('--refinement-epochs',type=int,default=12)
    args=parser.parse_args()
    result=run_pipeline(args.bundle,args.output,source=args.source,
                        max_train_queries=args.max_train_queries,
                        max_validation_queries=args.max_validation_queries,
                        adaptation_epochs=args.adaptation_epochs,
                        refinement_epochs=args.refinement_epochs)
    print(json.dumps({'completed':result['completed'],'final_adapter':str(args.output/result['final_adapter'])}))


if __name__=='__main__':
    main()
