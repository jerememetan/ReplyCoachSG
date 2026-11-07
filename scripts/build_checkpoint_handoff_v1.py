"""Copy immutable model files into a portable bundle; retain byte-identical weights."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--payload',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--archive',action='store_true')
    args=parser.parse_args();recipe=json.loads((args.payload/'export_recipe.json').read_text())
    if args.output.exists():raise FileExistsError('Preserve previous export')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(args.payload,args.output)
    for mapping in recipe['source_directories']:
        source=Path(mapping['source']);target=args.output/mapping['destination']
        if mapping.get('all_files'):
            shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        else:
            target.mkdir(parents=True)
            for name in mapping['files']:
                shutil.copy2(source/name,target/name)
        for name,h in mapping.get('expected_sha256',{}).items():
            if sha(target/name)!=h:raise ValueError('Source identity mismatch: '+str(source/name))
    files={}
    for path in sorted(args.output.rglob('*')):
        if path.is_symlink():raise ValueError('Export must not depend on symlinks')
        if path.is_file():files[str(path.relative_to(args.output))]=dict(bytes=path.stat().st_size,sha256=sha(path))
    manifest=dict(kind=recipe['kind'],name=recipe['name'],created_at_utc=datetime.now(timezone.utc).isoformat(),
        model=recipe['model'],lineage=recipe['lineage'],decoding=recipe['decoding'],files=files,
        weights_merged=False,weights_quantized=False,base_included=True,adapters_included=recipe['adapters'],
        expected_example_pairs=10,default_promoted=False,verification='Pending relocated-bundle GPU replay.')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),files=len(files),bytes=sum(v['bytes'] for v in files.values()))),flush=True)
    if args.archive:
        archive=args.output.with_suffix('.tar')
        if archive.exists():raise FileExistsError('Preserve previous archive')
        with tarfile.open(archive,'w') as tar:tar.add(args.output,arcname=args.output.name)
        print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive))),flush=True)


if __name__=='__main__':main()
