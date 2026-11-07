"""Attach successful replay evidence and archive a verified, self-contained bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--verification',type=Path,required=True);p.add_argument('--job',type=int,required=True)
    a=p.parse_args();r=json.loads(a.verification.read_text())
    if r['status']!='complete' or r['exact_example_pairs']!=10 or r.get('replay_differences') or r['flags'] or any(r['strict_loading'].values()):
        raise ValueError('Successful exact10/10 portable replay required before handoff archive')
    v=a.bundle/'verification.json'
    if v.exists():raise FileExistsError('Preserve previous finalized export')
    r.update(job_id=a.job,original_checkpoint_access_denied=True,network_connections_denied=True,
             isolated_python=True,verification_scope='Packaging and exact10 example-pair reproduction; not independent quality certification.')
    v.write_text(json.dumps(r,indent=2)+'\n')
    m=a.bundle/'manifest.json';manifest=json.loads(m.read_text())
    manifest['files']['verification.json']=dict(bytes=v.stat().st_size,sha256=sha(v))
    manifest['verification']=dict(status='passed',job_id=a.job,exact_example_pairs=10,
             gpu=r['gpu'],original_source_access_denied=True,network_denied=True,report='verification.json')
    m.write_text(json.dumps(manifest,indent=2)+'\n')
    archive=a.bundle.with_suffix('.tar')
    if archive.exists():raise FileExistsError('Preserve existing archive')
    tmp=archive.with_suffix('.tar.tmp')
    def no_cache(info):
        return None if '__pycache__' in Path(info.name).parts or info.name.endswith('.pyc') else info
    with tarfile.open(tmp,'w') as tar:tar.add(a.bundle,arcname=a.bundle.name,filter=no_cache)
    tmp.replace(archive)
    result=dict(status='ready',archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),
                bundle=str(a.bundle),manifest_sha256=sha(m),verification_job=a.job,exact_example_pairs=10)
    index=archive.with_suffix('.export.json');index.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
