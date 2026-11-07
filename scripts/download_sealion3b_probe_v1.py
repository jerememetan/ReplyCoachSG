"""Download only pinned official weights/tokenizer/config; no remote code execution."""
import hashlib,json,time
from pathlib import Path
from huggingface_hub import hf_hub_download
ROOT=Path(__file__).resolve().parents[1]


def main():
    recipe=json.loads((ROOT/'recipe.json').read_text());folder=ROOT/'model';folder.mkdir(exist_ok=True)
    begin=time.monotonic()
    for name in ['config.json','generation_config.json','tokenizer_config.json','tokenizer.model','special_tokens_map.json','LICENSE','model.safetensors']:
        hf_hub_download(recipe['repo_id'],name,revision=recipe['revision'],local_dir=folder)
        print('Downloaded',name,flush=True)
    h=hashlib.sha256()
    with (folder/'model.safetensors').open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    if h.hexdigest()!=recipe['weights_sha256']:raise ValueError('Official weight checksum mismatch')
    (ROOT/'download_manifest.json').write_text(json.dumps(dict(status='complete',revision=recipe['revision'],weights_sha256=h.hexdigest(),weights_bytes=(folder/'model.safetensors').stat().st_size,seconds=time.monotonic()-begin),indent=2)+'\n')
    print('Pinned checkpoint checksum verified',flush=True)

if __name__=='__main__':main()
