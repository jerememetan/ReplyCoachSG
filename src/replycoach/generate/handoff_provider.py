"""Expose exact portable model bundles without changing frozen V5 checkpoints."""
import atexit
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading

from replycoach.generate.providers import GeneratorUnavailable
from replycoach.schema import Candidate

SOURCES = {'mt5': 'MT5_A', 'sealion': 'SEALION'}


def pair_candidates(pair, source, bundle, message, context):
    candidates = []
    for field, style in [('english', 'STANDARD'), ('singlish', 'LOCAL')]:
        status = pair.get(field + '_status')
        text = pair.get(field, '').strip()
        if status != 'complete' or not text:
            raise GeneratorUnavailable(f'{source} {style} generation failed: {status}')
        candidates.append(Candidate(list(context), message, text, style, source, 0.5,
            meta={'checkpoint': str(bundle), 'model': source, 'generation_status': status,
                  'standard_anchor': pair.get('standard_anchor'),
                  'score_kind': 'unscored_bundle_output',
                  'speakers': 'guessed: alternating back from the incoming peer message'}))
    return candidates


class HandoffGenerator:
    def __init__(self, bundle, kind, device='auto'):
        self.bundle = Path(bundle).resolve()
        self.source = SOURCES[kind]
        self.device = device
        for name in ['manifest.json', 'infer.py']:
            if not (self.bundle / name).is_file():
                raise GeneratorUnavailable(f'{self.source} bundle missing {name}: {self.bundle}')
        if json.loads((self.bundle / 'manifest.json').read_text(encoding='utf-8'))['kind'] != kind:
            raise GeneratorUnavailable(f'Wrong model kind in {self.source} bundle')
        self.process = None
        self.lock = threading.Lock()
        self.responses = queue.Queue()
        self.log = None
        atexit.register(self.close)

    def _start(self):
        from replycoach.io import ROOT
        log_dir = ROOT / 'output' / 'handoff_logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log = (log_dir / f'{self.source}.log').open('a', encoding='utf-8')
        self.process = subprocess.Popen(
            [sys.executable, '-I', '-u', str(Path(__file__).with_name('handoff_worker.py')),
             str(self.bundle), self.device], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, text=True, encoding='utf-8',
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.responses = queue.Queue()
        def read_responses(process, responses):
            for line in process.stdout:
                responses.put(line)
            responses.put(None)
        threading.Thread(target=read_responses, args=(self.process, self.responses), daemon=True).start()

    def generate(self, message, context):
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                self.close()
                self._start()
            speakers = ['user' if (len(context) - i) % 2 else 'peer' for i in range(len(context))]
            rows = [dict(item_id='demo', message=message, context=list(context), speakers=speakers)]
            try:
                self.process.stdin.write(json.dumps(rows, ensure_ascii=False) + '\n')
                self.process.stdin.flush()
                line = self.responses.get(timeout=600)
                if line is None:
                    raise GeneratorUnavailable(f'{self.source} worker exited; see output/handoff_logs')
                response = json.loads(line)
                if 'error' in response:
                    raise GeneratorUnavailable(response['error'])
                candidates = pair_candidates(response['outputs'][0], self.source, self.bundle, message, context)
                for candidate in candidates:
                    candidate.meta.update(runtime_device=response.get('device'), gpu=response.get('gpu'),
                                          peak_gpu_bytes=response.get('peak_gpu_bytes'))
                return candidates
            except (queue.Empty, BrokenPipeError) as exc:
                self.close()
                raise GeneratorUnavailable(f'{self.source} worker timed out or disconnected') from exc

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
            self.process.wait(timeout=10)
            for stream in [self.process.stdin, self.process.stdout]:
                if stream: stream.close()
            self.process = None
        if self.log:
            self.log.close()
            self.log = None
