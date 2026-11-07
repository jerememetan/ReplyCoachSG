"""Replay an exported bundle while denying access to original model folders/network."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    bundle=args.bundle.resolve()
    recipe=json.loads((bundle/'export_recipe.json').read_text())
    blocked=[str(Path(r['source']).resolve()) for r in recipe['source_directories']]
    def audit(event,values):
        if event=='open' and isinstance(values[0],(str,bytes,os.PathLike)):
            path=os.path.abspath(os.fsdecode(values[0]))
            if any(path==prefix or path.startswith(prefix+os.sep) for prefix in blocked):
                raise PermissionError('Portable replay may not read original checkpoint: '+path)
        if event=='socket.connect':
            raise PermissionError('Portable replay must work without network connections')
    sys.addaudithook(audit)
    sys.argv=[str(bundle/'infer.py'),'--device','cuda','--verify-examples','--output',str(args.output.resolve())]
    runpy.run_path(str(bundle/'infer.py'),run_name='__main__')


if __name__=='__main__':main()
