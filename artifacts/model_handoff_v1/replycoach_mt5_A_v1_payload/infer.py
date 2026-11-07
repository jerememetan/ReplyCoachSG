"""Portable loader for the exact unmerged ReplyCoach mT5/SEA-LION handoff bundles."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

BUNDLE = Path(__file__).resolve().parent
sys.path.insert(0, str(BUNDLE/'src'))
if (BUNDLE/'deps').exists():
    sys.path.insert(0, str(BUNDLE/'deps'))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def verify_files(root):
    manifest = json.loads((root/'manifest.json').read_text())
    for name, info in manifest['files'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('Missing or invalid bundle file: '+name)
        if path.stat().st_size != info['bytes'] or sha(path) != info['sha256']:
            raise ValueError('Bundle checksum mismatch: '+name)
    return manifest


def validate_inputs(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError('Inputs must be a nonempty JSON list')
    seen = set()
    for row in rows:
        if not isinstance(row.get('item_id'), str) or not row['item_id'] or row['item_id'] in seen:
            raise ValueError('Every input needs a unique nonempty item_id')
        seen.add(row['item_id'])
        if not isinstance(row.get('message'), str) or not row['message'].strip():
            raise ValueError('Every input needs a nonempty message')
        if not isinstance(row.get('context'), list) or not all(isinstance(t, str) for t in row['context']):
            raise ValueError('context must be a list of text turns')
        speakers = row.get('speakers')
        if not isinstance(speakers, list) or len(speakers) != len(row['context']) or not set(speakers) <= {'user','peer','other'}:
            raise ValueError('speakers must match context and contain user/peer/other')


class ReplyPipeline:
    """Call generate_pairs with item_id/context/speakers/message dictionaries."""
    def __init__(self, root=BUNDLE, device='auto', verify=True):
        import torch
        from peft import PeftModel
        self.root = Path(root).resolve()
        self.manifest = verify_files(self.root) if verify else json.loads((self.root/'manifest.json').read_text())
        self.kind = self.manifest['kind']
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
        self.loading_problems = {}
        if self.kind == 'mt5':
            from transformers import AutoTokenizer, MT5Config
            from replycoach.generate.mt5_pretrained_compat import OriginalMT5
            raw = json.loads((self.root/'base/config.json').read_text())
            cfg = MT5Config(**raw)
            cfg.tie_word_embeddings = raw['tie_word_embeddings']
            base, info = OriginalMT5.from_pretrained(self.root/'base', config=cfg, dtype=torch.float32,
                    local_files_only=True, output_loading_info=True)
            self._strict(info)
            if not (base.shared.weight is base.encoder.embed_tokens.weight is base.decoder.embed_tokens.weight):
                raise ValueError('Shared mT5 input embeddings changed')
            if base.lm_head.weight is base.shared.weight or sum(p.numel() for p in base.parameters()) != 582401280:
                raise ValueError('Independent mT5 output head or parameter count changed')
            self.tokenizer = AutoTokenizer.from_pretrained(self.root/'base', local_files_only=True)
            self.model = PeftModel.from_pretrained(base.to(self.device), self.root/'adapters/shared', is_trainable=False)
        elif self.kind == 'sealion':
            from replycoach.generate.sealion_v1 import SEALionCausalLM, config_for_checkpoint, checkpoint_tokenizer
            use_cuda = self.device.startswith('cuda')
            if use_cuda and not torch.cuda.is_bf16_supported():
                raise ValueError('SEA export reproduces BF16 inference; a BF16-capable CUDA GPU is required')
            dtype = torch.bfloat16 if use_cuda else torch.float32
            base, info = SEALionCausalLM.from_pretrained(self.root/'base', config=config_for_checkpoint(self.root/'base'),
                    dtype=dtype, device_map={'':self.device}, local_files_only=True, output_loading_info=True)
            self._strict(info)
            if sum(p.numel() for p in base.parameters()) != 3178255360:
                raise ValueError('SEA-LION base parameter count changed')
            self.tokenizer = checkpoint_tokenizer(self.root/'base')
            self.model = PeftModel.from_pretrained(base, self.root/'adapters/english', adapter_name='english', is_trainable=False)
            self.model.load_adapter(self.root/'adapters/singlish', adapter_name='singlish', is_trainable=False)
        else:
            raise ValueError('Unknown bundle model kind')
        self.model.requires_grad_(False)
        self.model.eval()

    def _strict(self, info):
        self.loading_problems = {k:list(info.get(k) or []) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']}
        if any(self.loading_problems.values()):
            raise ValueError('Strict checkpoint loading failed: '+str(self.loading_problems))

    def _activate(self, adapter):
        self.model.set_adapter(adapter)
        self.model.requires_grad_(False)
        self.model.eval()

    def _mt5_generate(self, rows, style, anchors=None):
        import torch
        from replycoach.generate.style_control import mt5_input
        result = []
        for start in range(0, len(rows), 50):
            part = rows[start:start+50]
            prompts, ready, answers = [], [], {}
            for i, row in enumerate(part):
                anchor = None if anchors is None else anchors[start+i]
                if anchors is not None and not anchor.strip():
                    answers[i] = dict(text='',status='failed_empty_anchor',token_ids=[])
                    continue
                ready.append(i)
                prompts.append(mt5_input(style,row['message'],row['context'],True,row['speakers'],standard_reply=anchor))
            if ready:
                inputs = self.tokenizer(prompts,return_tensors='pt',padding=True)
                if inputs['input_ids'].shape[1] > 192:
                    raise ValueError('mT5 source exceeds192 tokens; no silent truncation')
                with torch.no_grad():
                    output = self.model.generate(**{k:v.to(self.device) for k,v in inputs.items()},num_beams=4,
                        no_repeat_ngram_size=3,do_sample=False,max_new_tokens=48,use_cache=True)
                for i,tokens in zip(ready,output,strict=True):
                    text = self.tokenizer.decode(tokens,skip_special_tokens=True)
                    status = 'failed_empty' if not text.strip() else 'complete' if self.tokenizer.eos_token_id in tokens.tolist() else 'length_limit'
                    answers[i] = dict(text=text,status=status,token_ids=tokens.tolist())
            result.extend(answers[i] for i in range(len(part)))
        return result

    def generate_pairs(self, rows):
        validate_inputs(rows)
        if self.kind == 'mt5':
            english = self._mt5_generate(rows,'STANDARD')
            local = self._mt5_generate(rows,'LOCAL_REWRITE',[r['text'] for r in english])
        else:
            from replycoach.generate.sealion_data import source
            from replycoach.generate.sealion_dual_v1 import BOUNDARY
            from replycoach.generate.sealion_rewrite_v1 import rewrite_source, BOUNDARY as REWRITE_BOUNDARY
            from replycoach.generate.sealion_beam_v1 import beam_decode
            self._activate('english')
            english = []
            for row in rows:
                out = beam_decode(self.model,self.tokenizer,self.tokenizer.encode(source(row,'STANDARD'),add_special_tokens=False),
                    BOUNDARY,num_beams=4,no_repeat_ngram_size=3,max_new_tokens=64,length_penalty=1.,device=self.device)
                english.append(dict(text=out['reply'],status=out['status'],token_ids=out['token_ids']))
            self._activate('singlish')
            local = []
            for row,anchor in zip(rows,english,strict=True):
                if not anchor['text'].strip():
                    local.append(dict(text='',status='failed_empty_anchor',token_ids=[]))
                    continue
                out = beam_decode(self.model,self.tokenizer,self.tokenizer.encode(rewrite_source(row,anchor['text']),add_special_tokens=False),
                    REWRITE_BOUNDARY,num_beams=4,no_repeat_ngram_size=3,max_new_tokens=64,length_penalty=1.,device=self.device)
                local.append(dict(text=out['reply'],status=out['status'],token_ids=out['token_ids']))
        return [dict(item_id=row['item_id'],english=e['text'],singlish=l['text'],standard_anchor=e['text'],
                     english_status=e['status'],singlish_status=l['status'])
                for row,e,l in zip(rows,english,local,strict=True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=BUNDLE)
    parser.add_argument('--inputs',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',default='auto',help='auto, cuda, cuda:0, or cpu')
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--verify-examples',action='store_true')
    args = parser.parse_args()
    root = args.bundle.resolve()
    if args.verify_only:
        manifest = verify_files(root)
        print(json.dumps(dict(status='verified',files=len(manifest['files']),kind=manifest['kind'])))
        return
    if args.output is None:
        parser.error('--output is required for inference')
    if args.output.exists():
        raise FileExistsError('Preserve previous output: '+str(args.output))
    import torch
    torch.set_num_threads(4)
    inputs = args.inputs or root/'examples/inputs.json'
    rows = json.loads(inputs.read_text())
    validate_inputs(rows)
    pipeline = ReplyPipeline(root,args.device)
    result = dict(status='complete',kind=pipeline.kind,device=pipeline.device,strict_loading=pipeline.loading_problems,
                  gpu=torch.cuda.get_device_name(0) if pipeline.device.startswith('cuda') else None,
                  input_sha256=sha(inputs),outputs=pipeline.generate_pairs(rows))
    result['flags'] = [r['item_id'] for r in result['outputs'] if r['english_status']!='complete' or r['singlish_status']!='complete']
    if args.verify_examples:
        if rows != json.loads((root/'examples/inputs.json').read_text()):
            raise ValueError('Example verification requires the frozen example inputs')
        expected = json.loads((root/'examples/expected_outputs.json').read_text())
        result['replay_differences'] = [dict(item_id=a['item_id'],expected=e,actual=a)
            for a,e in zip(result['outputs'],expected,strict=True)
            if any(a[k]!=e[k] for k in ['item_id','english','singlish'])]
        result['exact_example_pairs'] = len(rows)-len(result['replay_differences'])
        if result['replay_differences']:result['status']='complete_with_replay_differences'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='outputs'},indent=2))
    if result.get('replay_differences') or result['flags']:
        raise SystemExit(2)


if __name__=='__main__':main()
