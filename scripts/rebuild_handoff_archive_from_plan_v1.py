"""Rebuild a derived archive using verified local headers and existing bundle files."""
import argparse
import base64
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--archive',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text());bundle=args.bundle.resolve()
    old=(bundle/'manifest.json').read_bytes()
    if hashlib.sha256(old).hexdigest()!=plan['expected_old_manifest_sha256']:
        raise ValueError('Unexpected bundle documentation revision')
    # Only an incomplete, derived temporary archive from this card update is removed.
    temporary=args.archive.with_suffix('.tar.model_cards_tmp')
    if temporary.exists():temporary.unlink()
    for relative,data in plan['documentation'].items():
        target=bundle/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(data)
    manifest=plan['manifest'];(bundle/'manifest.json').write_text(plan['manifest_text'])
    archive_hash=hashlib.sha256();written=0
    # The verified complete local archive is retained. Truncating this derived
    # cluster copy avoids a second weight-sized temporary file under its quota.
    with args.archive.open('wb') as output:
        def write(data):
            nonlocal written
            output.write(data);archive_hash.update(data);written+=len(data)
        for entry in plan['entries']:
            write(base64.b64decode(entry['header']))
            if entry['file']:
                relative=entry['relative'];path=(bundle/relative).resolve()
                if not path.is_relative_to(bundle):raise ValueError('Invalid export path')
                if path.stat().st_size!=entry['size']:raise ValueError('File size changed: '+relative)
                h=hashlib.sha256()
                with path.open('rb') as source:
                    for block in iter(lambda:source.read(8*1024*1024),b''):
                        h.update(block);write(block)
                if relative=='manifest.json':expected=hashlib.sha256(plan['manifest_text'].encode()).hexdigest()
                else:expected=manifest['files'][relative]['sha256']
                if h.hexdigest()!=expected:raise ValueError('Source checksum mismatch: '+relative)
                write(bytes((-entry['size'])%512))
        write(bytes(plan['archive_bytes']-written))
    if written!=plan['archive_bytes'] or archive_hash.hexdigest()!=plan['archive_sha256']:
        raise ValueError('Rebuilt archive differs from verified local archive')
    indexfile=args.archive.with_suffix('.export.json');index=json.loads(indexfile.read_text())
    index.setdefault('previous_documentation_revisions',[]).append(dict(sha256=index['sha256'],manifest_sha256=index['manifest_sha256'],bytes=index['bytes']))
    index.update(bytes=written,sha256=archive_hash.hexdigest(),manifest_sha256=hashlib.sha256(plan['manifest_text'].encode()).hexdigest(),
        model_cards_revision='model-cards-v1',documentation_only_update=True,weight_files_unchanged=3)
    indexfile.write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps(dict(status='ready',bytes=written,sha256=archive_hash.hexdigest(),matches_verified_local_archive=True)))


if __name__=='__main__':main()
