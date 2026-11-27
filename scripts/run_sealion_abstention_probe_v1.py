"""Read-only presentation probes of original, selected and final snapshots."""
import gc
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run_sealion_abstention_v1 as experiment
core=experiment.core


def main():
    assert Path.home().name=='hm.teoh.2024'
    core.torch.set_num_threads(4)
    assert experiment.read('results.json')['status']=='complete'
    recipe=experiment.read('probe/recipe.json')
    assert core.sha(ROOT/'probe/inputs.json')==recipe['sha256']
    rows=experiment.read('probe/inputs.json')
    routes={'baseline':None}
    for arm in ['restored_context_control','selective_abstention']:
        for label in ['selected','step_0280']:routes[arm+'_'+label]=ROOT/'outputs'/arm/label
    reports={}
    for name,path in routes.items():
        model=core.load(path)
        model.load_adapter(core.BUNDLE/'adapters/singlish',adapter_name='singlish',is_trainable=False)
        result=experiment.paired_evaluation(model,core.checkpoint_tokenizer(core.BUNDLE/'base'),rows,f'probe/outputs/{name}.json')
        reports[name]=dict(status=result['status'],metrics=result['metrics'],flags=result['flags'],
            weights_sha256=core.sha((path or core.BUNDLE/'adapters/english')/'adapter_model.safetensors'))
        del model;gc.collect();core.torch.cuda.empty_cache()
        print('Probe',name,json.dumps(result['metrics']),flush=True)
    experiment.save('probe/results.json',dict(status='complete',recipe=recipe,reports=reports,
        training_performed=False,checkpoint_selection_allowed=False))


if __name__=='__main__':main()
