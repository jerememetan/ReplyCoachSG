"""Strictly export a cached original mT5-base checkpoint for matched runs."""
import argparse,hashlib,json
from pathlib import Path
import torch
from replycoach.generate.mt5_pretrained_compat import OriginalMT5,original_config,verify_original_embeddings,original_tokenizer

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve existing base')
    raw=json.loads((a.source/'config.json').read_text());cfg=original_config(raw)
    model,info=OriginalMT5.from_pretrained(a.source,config=cfg,dtype=torch.bfloat16,local_files_only=True,output_loading_info=True)
    problems={k:list(info.get(k) or []) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']}
    if any(problems.values()):raise ValueError(str(problems))
    verify_original_embeddings(model);model.save_pretrained(a.output);original_tokenizer(a.source).save_pretrained(a.output)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=dict(repo_id='google/mt5-base',revision=a.source.name,source_weights_sha256=sha(a.source/'pytorch_model.bin'),strict_loading=problems,parameters=sum(p.numel() for p in model.parameters()),base_precision='bfloat16',original_tie_word_embeddings=False,saved_files_sha256={p.name:sha(p) for p in a.output.iterdir() if p.is_file()},tokenizer_compatibility='Original SentencePiece backend preserves segmentation tie breaks and byte fallback.',scope='Original cached pretrained checkpoint; no additional public/task training.')
    (a.output.parent/'base_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
