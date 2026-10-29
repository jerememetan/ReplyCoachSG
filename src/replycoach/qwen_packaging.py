"""Minimal private Kaggle package for the fixed Qwen LoRA experiment."""
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil

from replycoach.reranker_bundle import load_training_bundle
from replycoach.reranker_hard_pairs import apply_reviewed_pairs


def package_qwen(bundle,output,*,owner,tag,epochs=4,early_stopping_patience=0):
    bundle,output=Path(bundle),Path(output)
    if output.exists():
        raise FileExistsError('Qwen package output must be new')
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{2,49}',owner) or not re.fullmatch(r'\d{8}-\d{2}',tag):
        raise ValueError('invalid owner or YYYYMMDD-NN run tag')
    if (type(epochs) is not int or not 1 <= epochs <= 12
            or type(early_stopping_patience) is not int or not 0 <= early_stopping_patience <= 12):
        raise ValueError('invalid Qwen epoch or early-stopping settings')
    loaded=load_training_bundle(bundle)
    if (bundle/'reviewed_pairs.json').is_file():
        apply_reviewed_pairs(loaded,json.loads((bundle/'reviewed_pairs.json').read_text()))
    dataset,notebook=output/'dataset',output/'notebook'
    dataset.mkdir(parents=True)
    notebook.mkdir()
    for name in ['manifest.json','train_pairs.jsonl','validation_pairs.jsonl','reviewed_pairs.json']:
        if (bundle/name).is_file():
            shutil.copyfile(bundle/name,dataset/name)
    root=Path(__file__).resolve().parents[2]
    sources=[root/'src/replycoach'/f'{name}.py' for name in
             ['qwen_reranker','qwen_fit','reranker_bundle','reranker_hard_pairs']]
    sources.append(root/'scripts/train_qwen_reranker.py')
    for source in sources:
        text=source.read_text(encoding='utf-8')
        for module in ['qwen_reranker','qwen_fit','reranker_bundle','reranker_hard_pairs']:
            text=text.replace(f'from replycoach.{module} import',f'from {module} import')
        compile(text,source.name,'exec')
        (dataset/source.name).write_text(text,encoding='utf-8',newline='\n')
    dataset_id=f'{owner}/replycoach-qwen-pairs-{tag}'
    notebook_id=f'{owner}/replycoach-qwen-lora-{tag}'
    (dataset/'dataset-metadata.json').write_text(json.dumps({
        'id':dataset_id,'title':f'ReplyCoach Qwen pairs {tag}','licenses':[{'name':'other'}]},indent=2))
    receipt={'dataset_id':dataset_id,'notebook_id':notebook_id,'private_required':True,
             'files_sha256':{p.name:sha256(p.read_bytes()).hexdigest() for p in dataset.iterdir()}}
    (dataset/'package_receipt.json').write_text(json.dumps(receipt,indent=2))
    def cell(text):
        return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
                'source':text.splitlines(keepends=True)}
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
        'cells':[cell('import subprocess,sys\n'
                      '# Kaggle includes an old optional torchao; unquantized LoRA does not use it.\n'
                      'subprocess.check_call([sys.executable,"-m","pip","uninstall","-y","torchao"])\n'
                      'subprocess.check_call([sys.executable,"-m","pip","install","--quiet",'
                      '"transformers==5.19.0","tokenizers==0.23.2","safetensors==0.8.0","peft==0.21.2"])\n'),
                 cell('from pathlib import Path\nimport os,json,torch\n'
                      'assert torch.cuda.is_available(),"GPU required"\n'
                      'matches=[p.parent for p in Path("/kaggle/input").rglob("manifest.json") '
                      'if (p.parent/"train_qwen_reranker.py").is_file()]\n'
                      'assert len(matches)==1,"Attach exactly one private Qwen training dataset"\n'
                      'bundle=matches[0]\n'
                      'env=dict(os.environ,OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",PYTHONUNBUFFERED="1")\n'
                      'subprocess.check_call([sys.executable,str(bundle/"train_qwen_reranker.py"),'
                      '"--bundle",str(bundle),"--output","/kaggle/working/qwen_lora",'
                      f'"--epochs","{epochs}","--early-stopping-patience","{early_stopping_patience}"],env=env)\n'),
                 cell('result=json.loads(Path("/kaggle/working/qwen_lora/best/qwen_training.json").read_text())\n'
                      'assert result["completed"] and result["production_checkpoint"] and result["base_unchanged"]\n'
                      'print(json.dumps({k:result[k] for k in ["best_epoch","best_validation_loss","optimizer_steps","adapter_sha256"]}))\n')]}
    (notebook/'Qwen_reranker.ipynb').write_text(json.dumps(nb,indent=2))
    (notebook/'kernel-metadata.json').write_text(json.dumps({
        'id':notebook_id,'title':f'ReplyCoach Qwen LoRA {tag}','code_file':'Qwen_reranker.ipynb',
        'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,
        'enable_internet':True,'dataset_sources':[dataset_id],'competition_sources':[],'kernel_sources':[]},indent=2))
    return receipt


def package_dailydialog_qwen(bundle,output,*,owner,tag):
    """Extend the proven minimal upload package with a standalone two-stage notebook."""
    receipt=package_qwen(bundle,output,owner=owner,tag=tag)
    output=Path(output)
    dataset=output/'dataset'
    root=Path(__file__).resolve().parents[2]
    modules=['qwen_reranker','qwen_fit','reranker_bundle','reranker_hard_pairs','dailydialog_reranker']
    for source in [root/'src/replycoach/dailydialog_reranker.py',root/'scripts/train_dailydialog_qwen.py']:
        text=source.read_text(encoding='utf-8')
        for module in modules:
            text=text.replace(f'from replycoach.{module} import',f'from {module} import')
        compile(text,source.name,'exec')
        (dataset/source.name).write_text(text,encoding='utf-8',newline='\n')
    notebook_path=output/'notebook/Qwen_reranker.ipynb'
    shutil.copyfile(root/'notebooks/E_qwen_dailydialog_kaggle.ipynb',notebook_path)
    receipt['experiment']='dailydialog_then_reviewed_qwen'
    receipt['gradient_policy']='finite_only_amp_retry_v1'
    receipt['notebook_id']=f'{owner}/replycoach-dailydialog-qwen-{tag}'
    receipt['files_sha256']={p.name:sha256(p.read_bytes()).hexdigest() for p in dataset.iterdir()
                             if p.is_file() and p.name not in {'package_receipt.json','dataset-metadata.json'}}
    (dataset/'package_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    meta_path=output/'notebook/kernel-metadata.json'
    meta=json.loads(meta_path.read_text(encoding='utf-8'))
    meta['id']=receipt['notebook_id']
    meta['title']=f'ReplyCoach DailyDialog Qwen {tag}'
    meta_path.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return receipt
