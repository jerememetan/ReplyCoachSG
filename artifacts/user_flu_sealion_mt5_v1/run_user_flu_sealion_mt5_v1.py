"""Two original portable inference pipelines on one user-supplied flu conversation."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    assert Path.home().name=='hm.teoh.2024'
    assert not (ROOT/'comparison.json').exists()
    inputs=json.loads((ROOT/'inputs.json').read_text());recipe=json.loads((ROOT/'recipe.json').read_text())
    assert sha(ROOT/'inputs.json')==recipe['inputs_sha256'] and len(inputs)==1
    handoff=Path.home()/'ReplyCoachSG_checkpoint_handoff_v1'
    models={'original_sealion':(handoff/'replycoach_sealion_repair25_v1','1cb24b55984b73c20166331d4d872895d00d9d4854b2a2b7c8f5245453db7f1a'),
        'mt5_A':(handoff/'replycoach_mt5_A_v1','4d31e79604d8818e430c7cf288ae9c0b738ca18db18e309fbcef6e929c7c6165')}
    results={}
    for name,(bundle,expected) in models.items():
        assert sha(bundle/'manifest.json')==expected
        target=ROOT/(name+'.json')
        with (ROOT/(name+'_inference.log')).open('w') as log:
            job=subprocess.run([sys.executable,'-I',str(bundle/'infer.py'),'--device','cuda','--inputs',str(ROOT/'inputs.json'),'--output',str(target)],stdout=log,stderr=subprocess.STDOUT)
        if not target.exists():raise RuntimeError(name+' failed; inspect '+name+'_inference.log')
        report=json.loads(target.read_text())
        assert report['input_sha256']==recipe['inputs_sha256']
        assert [r['item_id'] for r in report['outputs']]==[r['item_id'] for r in inputs]
        assert all(r['standard_anchor']==r['english'] for r in report['outputs'])
        # Separate processes avoid mixing the two packaged source namespaces.
        results[name]=dict(bundle=str(bundle),manifest_sha256=expected,exit_code=job.returncode,**report)
        print(name,'status',report['status'],'flags',report['flags'],flush=True)
    rows=[]
    for i,row in enumerate(inputs):
        rows.append(dict(row,outputs={name:r['outputs'][i] for name,r in results.items()}))
    status='complete' if all(r['status']=='complete' and not r['flags'] and r['exit_code']==0 for r in results.values()) else 'complete_with_flags'
    report=dict(status=status,recipe=recipe,inputs=inputs,models=results,comparison=rows,
        training_performed=False,default_changed=False,no_target_reply_supplied=True,
        note='The exact user-supplied conversation and both outputs are shown unchanged. Models use their original stored beam pipelines. No output-based selection or reranking; own-English Singlish anchors.')
    (ROOT/'comparison.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    fields=['item_id','topic','context','speakers','message','sealion_english','sealion_singlish','mt5_english','mt5_singlish']
    with (ROOT/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({**{k:json.dumps(row[k]) if k in {'context','speakers'} else row[k] for k in fields[:5]},
            'sealion_english':row['outputs']['original_sealion']['english'],'sealion_singlish':row['outputs']['original_sealion']['singlish'],
            'mt5_english':row['outputs']['mt5_A']['english'],'mt5_singlish':row['outputs']['mt5_A']['singlish']})
    print('USER FLU COMPARISON COMPLETE',status,flush=True)


if __name__=='__main__':main()
