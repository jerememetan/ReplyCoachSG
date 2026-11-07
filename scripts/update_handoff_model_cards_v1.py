"""Add the team's model-card formats to existing exports without changing model files."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]
NAMES={'mt5':'replycoach_mt5_A_v1','sealion':'replycoach_sealion_repair25_v1'}
STEMS={'mt5':'mt5_reference_A','sealion':'sealion_repair25'}


def digest(data):return hashlib.sha256(data).hexdigest()
def encode(value):return (json.dumps(value,indent=2)+'\n').encode()
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def component(kind,adapter):
    mt5=kind=='mt5';base='google/mt5-base' if mt5 else 'aisingapore/SEA-LION-v1-3B'
    if mt5:
        title='mT5 A shared adapter, seed1 final update784'
        detail='1,769,472 FP32 adapter parameters; rank16, alpha32, dropout0.1 on q/v. Training job351130 used the separate reviewed190-conversation target derivative:190 English,190 LOCAL_REWRITE and180 original history replies. The original team records are unchanged. Targets include14 assistant English corrections, Gemma4 12B LOCAL drafts and117 assistant revisions.'
        selection='Final update784 is shipped, rather than epoch4/update140 with minimum reference loss. Exploratory assistant review:28/50 all-three paired passes on reused validation; not independent human accuracy.'
    elif adapter=='english':
        title='SEA-LION English adapter, full-data task epoch2'
        detail='10,485,760 FP32 adapter parameters; rank8, alpha16, dropout0.05 on Wqkv/out_proj/up_proj/down_proj. Job356177 continued public LoRA adaptation on117,634 pairs with1,580 English/direct-LOCAL targets from790 project conversations. The source is190 unchanged team records plus600 corrected generated examples.'
        selection='Selected task epoch2 by equal-style reference CE2.553996; English CE1.969433 on50 reused validation conversations. Later repair selected English step0, so these original English weights are retained.'
    else:
        title='SEA-LION Singlish adapter, selected repair update25'
        detail='10,485,760 FP32 adapter parameters; rank8, alpha16, dropout0.05 on Wqkv/out_proj/up_proj/down_proj. Independent copy of the selected full-data English adapter →745-pair anchored renderer, job356662 epoch1 → contrastive repair, job357651 update25. The repair pool contains745 replay examples and128 assistant-authored positive/negative pairs; selected update25 has200 replay exposures and25 auxiliary-pair exposures.'
        selection='Selected rewrite CE2.337944 on42 reused validation pairs; synthetic ranking30/32, not generated-reply accuracy. Later repair updates up to200 were run but are not this checkpoint. Full126-case quality review remains pending.'
    return f'''---
base_model: {base}
library_name: peft
language:
- en
license: other
pipeline_tag: {'text2text-generation' if mt5 else 'text-generation'}
tags:
- lora
- singlish
---

# {title}

This is one component of the self-contained ReplyCoach export. Use the export-root [README.md](../../README.md), [MODEL_CARD.md](../../MODEL_CARD.md) and [INFERENCE_GUIDE.md](../../INFERENCE_GUIDE.md) for the complete model card and loading instructions.

## Model and training

{detail}

## Selection and evaluation

{selection}

## Loading

Load from the export root with infer.py/ReplyPipeline and the included base/. {'The shared adapter generates English and then rewrites that exact English draft to Singlish.' if mt5 else 'Only one of the English/Singlish adapters is active at a time. Singlish generation receives the exact generated English draft as its anchor.'} Original cluster paths in adapter_config.json are provenance and are not required at runtime. An upstream base substituted for the included adapted mT5 base, or a generic replacement SEA architecture, will not reproduce the export.

## Limitations and licence

This is an experimental reply-suggestion component, not a certified independent benchmark winner. Role/context errors, altered meaning and weak Singlish remain possible. Generated/assistant-edited labels are part of this lineage; no runtime teacher is used. The included upstream base/code notices do not settle all fine-tuned-checkpoint/data redistribution terms; the project release licence is unresolved. No public release or course-compliance approval is asserted.

## Authors and runtime

CS425 Group9, Role D. Verified export runtime: Python3.13.1, Torch2.14.1+cu130, Transformers5.19.0, PEFT0.21.2. Exact10/10 pipeline replay evidence is in ../../verification.json; that measures packaging reproducibility, not conversational accuracy.
'''


def prepare():
    folder=ROOT/'artifacts/model_handoff_v1';cards=folder/'model_cards';cards.mkdir(exist_ok=True)
    for kind,name in NAMES.items():
        stem=STEMS[kind];archive=folder/(name+'.tar')
        with tarfile.open(archive) as tar:
            raw=tar.extractfile(name+'/manifest.json').read();manifest=json.loads(raw)
            index=json.loads((folder/(name+'.export.json')).read_text())
            if digest(raw)!=index['manifest_sha256']:raise ValueError('Unexpected old manifest')
            spec=json.load(tar.extractfile(name+'/training_specs.json'))
            guide=tar.extractfile(name+'/INFERENCE_GUIDE.md' if name+'/INFERENCE_GUIDE.md' in tar.getnames() else name+'/README.md').read().decode()
        summary=(ROOT/'roles/D_transformer'/(stem+'_MODEL_CARD.md')).read_text()
        full=(ROOT/'roles/D_transformer'/(stem+'_README.md')).read_text()
        shutil.copy2(ROOT/'roles/D_transformer'/(stem+'_MODEL_CARD.md'),cards/(stem+'_MODEL_CARD.md'))
        shutil.copy2(ROOT/'roles/D_transformer'/(stem+'_README.md'),cards/(stem+'_README.md'))
        files={'README.md':full,'MODEL_CARD.md':summary,'INFERENCE_GUIDE.md':guide}
        for adapter in manifest['adapters_included']:files[f'adapters/{adapter}/README.md']=component(kind,adapter)
        if kind=='mt5':
            spec['scope']='Exported mT5 A only: historical rewrite_s1 seed1 final update784.'
            spec['historical_five_sample_provenance']=spec.pop('sample_source',None)
            spec['sample_source']=dict(file='examples/inputs.json',conversations=10,seed=1632249497,
                selection='Uniform draw from45 project validation conversations after excluding previous displayed5; selected before outputs, unchanged.',source='data/dev/validation.jsonl')
            spec['shared_task_settings']['task_types']=['STANDARD','LOCAL_REWRITE','REPLY']
            spec['comparison_limits']=['Reference-English-conditioned LOCAL CE is not generated-English pair accuracy.','These are reused development conversations and assistant review, not an independent project-test score.']
            evidence=json.loads((ROOT/'experiments/shared_intent_v1_replication.json').read_text())['per_seed_reviews']['1']['counts']['rewrite']
            evaluation=dict(assistant_review=evidence,review_source='experiments/shared_intent_v1_replication.json',
                review_limit='Reused50 development conversations; author/reviewer overlap, not independent human gold.',
                exported_checkpoint_ce=dict(english=2.1749,reference_anchored_local=3.8045,macro=2.9897),
                minimum_loss_alternative=dict(epoch=4,step=140,macro_ce=2.5333,exported=False),matched_v4_comparison=None,
                packaging_replay=manifest['verification'],locked_project_test_used=False)
        else:
            spec['selected_repair_exposures']=dict(english_new_updates=0,singlish_updates=25,ordinary_examples=200,auxiliary_positive_negative_pairs=25,available_auxiliary_training_pairs=128)
            spec['project_training']['task_torch_rng_reset']=10425
            source=json.loads((ROOT/'experiments/sealion_contrast_v1_results.json').read_text())
            evaluation={k:source[k] for k in ['stages','changes_vs_baseline','flags','random10_review']}
            evaluation.update(source_report='experiments/sealion_contrast_v1_results.json',
                ranking_limit='32 same-template held-entity synthetic NLL comparisons, not generated-reply accuracy.',
                matched_v4_comparison=None,full_126_quality_review='pending',packaging_replay=manifest['verification'],locked_project_test_used=False)
        files['training_specs.json']=encode(spec).decode();files['evaluation_summary.json']=encode(evaluation).decode()
        patch=dict(name=name,kind=kind,revision='model-cards-v1',timestamp=int(datetime.now(timezone.utc).timestamp()),
            expected_old_manifest_sha256=digest(raw),previous_archive_sha256=index['sha256'],files=files,
            source_cards=[f'roles/D_transformer/{stem}_MODEL_CARD.md',f'roles/D_transformer/{stem}_README.md'])
        path=cards/(name+'_patch.json');path.write_bytes(encode(patch));print(str(path))


class CheckedReader:
    def __init__(self,stream):self.stream=stream;self.hash=hashlib.sha256()
    def read(self,size=-1):
        block=self.stream.read(size);self.hash.update(block);return block


def apply(archive,patchfile,bundle=None):
    patch=json.loads(patchfile.read_text());name=patch['name'];prefix=name+'/'
    allowed={'README.md','MODEL_CARD.md','INFERENCE_GUIDE.md','training_specs.json','evaluation_summary.json'}
    replacements={k:v.encode() for k,v in patch['files'].items()}
    for path in replacements:
        if path not in allowed and not (path.startswith('adapters/') and path.count('/')==2 and path.endswith('/README.md')):
            raise ValueError('Model-card revision may not change runtime/model file: '+path)
    tmp=archive.with_suffix('.tar.model_cards_tmp')
    if tmp.exists():raise FileExistsError('Preserve incomplete archive for inspection')
    unchanged=0;weight_files=0
    with tarfile.open(archive,'r') as old:
        raw=old.extractfile(prefix+'manifest.json').read()
        if digest(raw)!=patch['expected_old_manifest_sha256']:raise ValueError('Unexpected source archive revision')
        manifest=json.loads(raw)
        for relative,data in replacements.items():manifest['files'][relative]=dict(bytes=len(data),sha256=digest(data))
        manifest['model_cards']=dict(revision=patch['revision'],summary='MODEL_CARD.md',hugging_face='README.md',inference_guide='INFERENCE_GUIDE.md',evaluation='evaluation_summary.json')
        manifest['verification'].update(documentation_updated_after_replay=True,inference_files_unchanged=True)
        new_manifest=encode(manifest)
        with tarfile.open(tmp,'w',format=tarfile.PAX_FORMAT) as new:
            for member in old.getmembers():
                if not (member.name==name or member.name.startswith(prefix)):raise ValueError('Unexpected archive root')
                relative=member.name[len(prefix):]
                if relative in replacements or relative=='manifest.json':continue
                if member.issym() or member.islnk():raise ValueError('Export must remain self-contained')
                if member.isfile():
                    expected=manifest['files'][relative]
                    reader=CheckedReader(old.extractfile(member));new.addfile(member,reader)
                    if reader.hash.hexdigest()!=expected['sha256']:raise ValueError('Unchanged file checksum failed: '+relative)
                    unchanged+=1;weight_files+=int(relative.endswith('.safetensors'))
                else:new.addfile(member)
            for relative,data in {**replacements,'manifest.json':new_manifest}.items():
                info=tarfile.TarInfo(prefix+relative);info.size=len(data);info.mode=0o644;info.mtime=patch['timestamp']
                new.addfile(info,io.BytesIO(data))
    with tarfile.open(tmp) as check:
        if len(check.getnames())!=len(set(check.getnames())):raise ValueError('Duplicate archive entries')
        if check.extractfile(prefix+'manifest.json').read()!=new_manifest:raise ValueError('Updated manifest mismatch')
        for relative,data in replacements.items():
            if check.extractfile(prefix+relative).read()!=data:raise ValueError('Documentation mismatch')
    new_sha=sha(tmp)
    if bundle:
        old_bundle=(bundle/'manifest.json').read_bytes()
        if digest(old_bundle)!=patch['expected_old_manifest_sha256']:raise ValueError('Unexpected extracted bundle revision')
        for relative,data in replacements.items():
            target=bundle/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        (bundle/'manifest.json').write_bytes(new_manifest)
    tmp.replace(archive)
    for suffix in ['.export.json','.local_verified.json']:
        indexfile=archive.with_suffix(suffix)
        if indexfile.exists():
            index=json.loads(indexfile.read_text());index.setdefault('previous_documentation_revisions',[]).append(
                dict(sha256=index['sha256'],manifest_sha256=index['manifest_sha256'],bytes=index['bytes']))
            index.update(bytes=archive.stat().st_size,sha256=new_sha,manifest_sha256=digest(new_manifest),
                model_cards_revision=patch['revision'],documentation_only_update=True,unchanged_files_checked=unchanged,weight_files_unchanged=weight_files)
            if suffix=='.local_verified.json':index['verification']=manifest['verification']
            indexfile.write_bytes(encode(index))
    result=dict(archive=str(archive),sha256=new_sha,bytes=archive.stat().st_size,manifest_sha256=digest(new_manifest),
        documentation_only=True,unchanged_files_checked=unchanged,weight_files_unchanged=weight_files,updated_files=list(replacements),model_cards_revision=patch['revision'])
    archive.with_suffix('.model_cards_update.json').write_bytes(encode(result));print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--archive',type=Path);parser.add_argument('--patch',type=Path);parser.add_argument('--bundle',type=Path)
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.archive and args.patch:apply(args.archive,args.patch,args.bundle)
    else:parser.error('Use --prepare or --archive and --patch')


if __name__=='__main__':main()
