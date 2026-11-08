"""Isolated, persistent JSON-lines worker for the exact supplied bundle loader."""
import contextlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    # Windows redirected pipes otherwise inherit the local code page.
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    root = Path(sys.argv[1]).resolve()
    device = sys.argv[2]
    # Import the bundle's own replycoach package, never the demo's loaded package.
    sys.path.insert(0, str(root))
    pipeline = None
    for line in sys.stdin:
        try:
            rows = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if pipeline is None:
                    spec = importlib.util.spec_from_file_location('bundle_infer', root / 'infer.py')
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    import torch
                    torch.set_num_threads(4)
                    pipeline = module.ReplyPipeline(root, device=device)
                result = {'outputs': pipeline.generate_pairs(rows),
                          'device': str(getattr(pipeline, 'device', device))}
                if result['device'].startswith('cuda'):
                    torch.cuda.synchronize()
                    result['gpu'] = torch.cuda.get_device_name(0)
                    result['peak_gpu_bytes'] = torch.cuda.max_memory_allocated()
        except Exception as exc:
            result = {'error': f'{type(exc).__name__}: {exc}'}
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
