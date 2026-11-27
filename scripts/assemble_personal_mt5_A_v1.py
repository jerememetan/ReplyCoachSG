"""Install exact exported mT5 A files using its identical local base tensor file."""
import hashlib
import json
import os
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    assert Path.home().name=='hm.teoh.2024'
    target=Path.home()/'ReplyCoachSG_checkpoint_handoff_v1/replycoach_mt5_A_v1'
    base=Path.home()/'ReplyCoachSG_mt5_mlp_generation_v1/model/model.safetensors'
    assert sha(base)=='20b20e1ac9382c2ca83b6e0b2c665ec4fb1104f853c0e2489d9b0ad205f871ac'
    created=not target.exists()
    if created:
        target.mkdir()
        with tarfile.open(ROOT/'mt5_runtime_payload.tgz') as tar:
            assert all(m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts for m in tar.getmembers())
            tar.extractall(target,filter='data')
        os.link(base,target/'base/model.safetensors')
    assert sha(target/'manifest.json')=='4d31e79604d8818e430c7cf288ae9c0b738ca18db18e309fbcef6e929c7c6165'
    manifest=json.loads((target/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        p=target/name;assert p.stat().st_size==expected['bytes'] and sha(p)==expected['sha256'],name
    result=dict(status='verified',bundle=str(target),files=len(manifest['files']),created=created,
        base_weights_sha256=sha(base),manifest_sha256=sha(target/'manifest.json'),
        base_tensor_storage_shared_by_hardlink=True,source_base_modified=False,
        note='Exact exported model/config/tokenizer/loader bytes. The already present identical base tensor file is hard-linked to avoid a second2.3GB storage allocation; this is an inference installation, not an independent backup.')
    (ROOT/'mt5_bundle_assembly.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
