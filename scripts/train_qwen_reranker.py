"""Train an independently saved Qwen LoRA reranker; no judge labels or test data."""
import argparse
import json
from pathlib import Path
from replycoach.qwen_fit import run_training
from replycoach.qwen_reranker import MODEL

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',default=MODEL)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--max-steps',type=int)
    args=parser.parse_args()
    result=run_training(args.bundle,args.output,source=args.source,smoke=args.smoke,max_steps=args.max_steps)
    print(json.dumps({k:result[k] for k in ['completed','production_checkpoint','optimizer_steps','best_epoch','best_validation_loss']}))
