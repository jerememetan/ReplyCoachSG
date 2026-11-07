"""Freeze the two selected handoff payloads, without copying large local weights."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/model_handoff_v1'
VERSIONS={'torch':'2.14.1','transformers':'5.19.0','peft':'0.21.2','safetensors':'0.8.0',
          'tokenizers':'0.23.2','accelerate':'1.15.0','sentencepiece':'0.2.2','einops':'0.8.1','numpy':'2.5.3','packaging':'26.3'}


def read(path):return json.loads((ROOT/path).read_text())
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    models=read('experiments/mt5_reference_pack_v1/manifest.json')
    examples=read('experiments/mt5_reference_random10_1632249497/paired_outputs_with_sealion.json')
    original=read('experiments/mt5_reference_pack_v1/training_specs.json')
    inputs=[{k:r[k] for k in ['item_id','context','speakers','message']} for r in examples['cases']]
    common=['src/replycoach/__init__.py','src/replycoach/schema.py','src/replycoach/tokenize.py',
            'src/replycoach/generate/__init__.py']
    archive_info={}
    for kind,name in [('mt5','replycoach_mt5_A_v1'),('sealion','replycoach_sealion_repair25_v1')]:
        payload=OUT/(name+'_payload')
        if payload.exists():raise FileExistsError('Preserve frozen payload')
        payload.mkdir()
        sources=common+(['src/replycoach/generate/style_control.py','src/replycoach/generate/mt5_pretrained_compat.py'] if kind=='mt5' else
            ['src/replycoach/generate/sealion_v1.py','src/replycoach/generate/sealion_data.py',
             'src/replycoach/generate/sealion_dual_v1.py','src/replycoach/generate/sealion_rewrite_v1.py',
             'src/replycoach/generate/sealion_beam_v1.py','src/replycoach/vendor/__init__.py',
             *[str(p.relative_to(ROOT)) for p in (ROOT/'src/replycoach/vendor/sealion_v1').iterdir()
               if p.is_file() and p.suffix!='.pyc']])
        for source in sources:
            target=payload/source;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/source,target)
        shutil.copy2(ROOT/'scripts/run_checkpoint_handoff_v1.py',payload/'infer.py')
        (payload/'examples').mkdir();write(payload/'examples/inputs.json',inputs)
        key='historical_final' if kind=='mt5' else 'sealion_latest'
        expected=[dict(item_id=r['item_id'],english=r['models'][key]['english'],singlish=r['models'][key].get('singlish',r['models'][key].get('local'))) for r in examples['cases']]
        write(payload/'examples/expected_outputs.json',expected)
        (payload/'requirements.txt').write_text('\n'.join(f'{p}=={v}' for p,v in VERSIONS.items())+'\n')
        write(payload/'runtime.json',dict(python='3.13.1',packages=VERSIONS,tested_torch_build='2.14.1+cu130',
            tested_hardware='NVIDIA L40S and/or RTX3090; per-bundle actual replay hardware recorded in verification.json',
            cpu='Loader supports CPU FP32; real exported-model replay is tested on GPU unless verification.json states otherwise.'))
        (payload/'licenses').mkdir()
        if kind=='mt5':
            spec=models['models']['historical_final'];base=models['common_base']
            license_url='https://raw.githubusercontent.com/google-research/multilingual-t5/master/LICENSE'
            license_text=urlopen(license_url,timeout=30).read()
            if not license_text.lstrip().startswith(b'Apache License'):raise ValueError('Unexpected upstream license')
            (payload/'licenses/mt5_APACHE_2.txt').write_bytes(license_text)
            model_card=urlopen('https://huggingface.co/google/mt5-base/raw/main/README.md',timeout=30).read()
            (payload/'licenses/mt5_model_card.md').write_bytes(model_card)
            selected={k:v for k,v in original.items() if k!='checkpoints'};selected['checkpoint']=original['checkpoints']['A']
            write(payload/'training_specs.json',selected)
            recipe=dict(kind=kind,name=name,model='google/mt5-base,582401280base +1769472LoRA parameters',
                lineage='Historical shared_intent rewrite_s1 seed1 final784,trainingjob351130. Same fully adapted base and exact shared LoRA used in reviewed examples.',
                adapters=['shared'],decoding=dict(num_beams=4,no_repeat_ngram_size=3,max_new_tokens=48,max_source_tokens=192,do_sample=False),
                source_directories=[dict(source=base['source_path'],destination='base',files=list(base['files_sha256']),expected_sha256=base['files_sha256']),
                    dict(source=spec['checkpoint'],destination='adapters/shared',files=['adapter_model.safetensors','adapter_config.json','README.md','v5_input.json'],
                        expected_sha256={'adapter_model.safetensors':spec['adapter_sha256'],'adapter_config.json':spec['config_sha256']})])
            description='mT5 A: older shared-intent seed1 final step784. One shared adapter performs English generation and Singlish rewriting over the included fully adapted base.'
            caution='Use the included adapted base. The loader preserves its independent output head and exact tokenizer. The original Google base has different weights.'
        else:
            shutil.copy2(ROOT/'src/replycoach/vendor/sealion_v1/LICENSE',payload/'licenses/sealion_MIT.txt')
            recipe=dict(kind=kind,name=name,model='aisingapore/SEA-LION-v1-3B,3178255360base +2x10485760adapter parameters',
                lineage='English adapter from full117634-public-pair trainingjob356177; rewrite from356662 followed by contrastive repair357651,selectedstep25. English repair selectedstep0. Latest displayed candidate, comprehensive quality review pending.',
                adapters=['english','singlish'],decoding=dict(num_beams=4,no_repeat_ngram_size=3,max_new_tokens=64,max_context_tokens=384,length_penalty=1.,do_sample=False),
                source_directories=[dict(source='/common/home/users/h/hm.teoh.2024/ReplyCoachSG_sealion3b_probe_v1/model',destination='base',
                    files=['model.safetensors','config.json','generation_config.json','tokenizer.model','tokenizer_config.json','special_tokens_map.json','LICENSE'],
                    expected_sha256={'model.safetensors':'acba6d6560f61ac72cc95feca5c89f6aabbbf1e9e89c18440237331fcb5ae107'}),
                    dict(source='/common/home/users/h/hm.teoh.2024/ReplyCoachSG_sealion_full_v1/artifacts/best_adapter',destination='adapters/english',files=['adapter_model.safetensors','adapter_config.json','README.md'],
                        expected_sha256={'adapter_model.safetensors':'55a7f02626c08894bfdc4b44f68dacf86e8f4edafb7aa6bd78aead9162698913','adapter_config.json':'f5bb26bc63c376e64ed060f54cc87a44ece14a7328f064d02abc487aa75578bc'}),
                    dict(source='/common/home/users/h/hm.teoh.2024/ReplyCoachSG_sealion_contrast_repair_v1_retry1/artifacts/rewrite/selected',destination='adapters/singlish',files=['adapter_model.safetensors','adapter_config.json','README.md'],
                        expected_sha256={'adapter_model.safetensors':'6ea7066deda4f4e9e5f4a9e070d46427536a0dac05d5a416864e97c2dc3f5db2','adapter_config.json':'d05758cc0b34f769e22b48a3816493697528813351bc4048c400d0ee4d5a97dc'}),
                    dict(source='/common/home/users/h/hm.teoh.2024/ReplyCoachSG_sealion3b_probe_v1/deps/einops',destination='deps/einops',all_files=True),
                    dict(source='/common/home/users/h/hm.teoh.2024/ReplyCoachSG_sealion3b_probe_v1/deps/einops-0.8.1.dist-info',destination='deps/einops-0.8.1.dist-info',all_files=True)])
            write(payload/'training_specs.json',dict(base_model='aisingapore/SEA-LION-v1-3B',revision='836890428766e94322a4b7de27fb081a604703a9',
                architecture='Original learned-position, non-ALiBi MPT decoder; included compatibility code required.',
                public_pairs=dict(cosem=53691,dailydialog=63943,total=117634,method='LoRA adaptation'),
                project_training=dict(conversations=790,original_team=190,corrected_generated=600,english_and_direct_local_targets=1580),
                renderer_training=dict(eligible_pairs=745,validation_pairs=42),repair=read('experiments/sealion_contrast_v1_recipe.json')['training'],
                selected_repair_steps=dict(english=0,singlish=25),lora=dict(rank=8,alpha=16,dropout=.05,target_modules=['Wqkv','out_proj','up_proj','down_proj']),
                quality='Latest candidate shown to user; promising selected renderer, not a certified deployment or project default.'))
            description='SEA-LION English→Singlish pipeline: original full-data English adapter plus the latest selected step25 Singlish repair adapter. This is one shared base with two switchable adapters.'
            caution='Use both adapters with the included original learned-position SEA-LION implementation. Keep the English draft as the rewrite anchor. The selected candidate still has context/meaning errors.'
        write(payload/'export_recipe.json',recipe)
        text=f'''# {name}

{description}

This folder includes the base weights, selected adapter weights, tokenizer, compatible model code, exact decoding settings, training provenance, upstream notices and10 example input/output pairs. No model download is required. It contains no full training dataset, project test set, credentials or optimizer states. Weights retain their original stored precision; no merging or quantization was applied.

## Run

Tested runtime: Python3.13.1, Torch2.14.1+cu130, Transformers5.19.0, PEFT0.21.2. Install a PyTorch build appropriate for your machine and the pinned dependencies:

```bash
python -m pip install -r requirements.txt
python infer.py --verify-only
python infer.py --device cuda --verify-examples --output /tmp/{kind}_example_replay.json
python infer.py --device cuda --inputs your_inputs.json --output your_replies.json
```

Output files must not already exist. The default inputs are the included10 reused validation examples, not an independent accuracy test. A completed exact replay checks packaging, not reply correctness. The package verifier checks every shipped file against manifest.json before model loading. Source checkpoint paths in export_recipe.json are provenance only; the loader reads bundle-relative files.

Inputs are a JSON list. Each object has item_id, context (oldest-first text list), speakers (matching user/peer/other list), and message (incoming peer text). See examples/inputs.json. Output objects contain english, singlish, standard_anchor and generation status. Input budgets are checked without silent truncation. Beam4/no-repeat-ngram3 decoding is fixed to the demonstrated settings.

## Python API

```python
import json
from infer import ReplyPipeline
model = ReplyPipeline(device="cuda")
rows = json.load(open("examples/inputs.json"))
replies = model.generate_pairs(rows)
```

{caution}

CUDA inference is tested by the exporter. CPU FP32 loading is implemented; complete model CPU replay/performance is not verified unless verification.json explicitly records it. SEA-LION CPU loading uses approximately12.7GB of raw FP32 base weights plus runtime memory. Original GPU dtype is BF16 for SEA and FP32 for mT5. The packaging test records the actual GPU and runtime in verification.json. Hardware/library changes may affect numerical replay.

The full training specifications are in training_specs.json. Original training checkpoints are unchanged. The new location uses ordinary files, with no symlinks back to the cluster.
'''
        (payload/'README.md').write_text(text)
        archive=OUT/(name+'_payload.tgz')
        with tarfile.open(archive,'w:gz') as tar:tar.add(payload,arcname='payload');tar.add(ROOT/'scripts/build_checkpoint_handoff_v1.py',arcname='build_checkpoint_handoff_v1.py')
        archive_info[kind]=dict(name=name,payload=str(payload.relative_to(ROOT)),archive=str(archive.relative_to(ROOT)),
            payload_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),state='prepared')
    write(OUT/'preparation.json',archive_info)
    print(json.dumps(archive_info,indent=2))


if __name__=='__main__':main()
