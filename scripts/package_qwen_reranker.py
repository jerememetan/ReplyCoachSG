"""Create an allowlisted private Qwen training bundle and notebook."""
import argparse
import json
from pathlib import Path
from replycoach.qwen_packaging import package_qwen

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--owner',required=True)
    p.add_argument('--tag',required=True)
    a=p.parse_args()
    result=package_qwen(a.bundle,a.output,owner=a.owner,tag=a.tag)
    print(json.dumps({k:result[k] for k in ['dataset_id','notebook_id','private_required']}))
