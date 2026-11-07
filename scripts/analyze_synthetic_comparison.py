"""Analyze matched fixed-update control versus paired-synthetic training runs."""
import argparse,json
from pathlib import Path
from statistics import mean,stdev
from analyze_dpo_comparison import diagnostics


def analyze(directory,seeds,count):
    reports={}; shared=('base_model','train_items','validation_items','lora','epochs_requested',
        'patience','batch_size','learning_rate','speaker_labels','max_source_tokens',
        'max_target_tokens','joint','unlikelihood','preferences','max_steps')
    for seed in seeds:
        pair={arm:json.loads((directory/f'{arm}_s{seed}.json').read_text()) for arm in ('control','synthetic')}
        for key in shared:
            if pair['control'].get(key)!=pair['synthetic'].get(key):
                raise ValueError(f'Unmatched {key}, seed {seed}')
        if pair['control']['synthetic'] is not None or pair['synthetic']['synthetic']['items']!=count:
            raise ValueError('Expected control and the reviewed synthetic dataset')
        if pair['control']['max_steps']!=280: raise ValueError('Wrong update budget')
        outputs=[{r['item_id']:r for r in pair[a]['validation_outputs']} for a in ('control','synthetic')]
        if len(outputs[0])!=50 or outputs[0].keys()!=outputs[1].keys():
            raise ValueError('Expected same 50 validation conversations')
        for item in outputs[0]:
            for key in ('context','message','reference_standard','reference_local'):
                if outputs[0][item][key]!=outputs[1][item][key]: raise ValueError('Different validation inputs')
        for arm,r in pair.items():
            if r['seed']!=seed or not 0<r['optimizer_steps']<=280: raise ValueError('Wrong seed or step count')
            best=next(h for h in r['history'] if h['epoch']==r['best_epoch'])
            reports[f'{arm}_s{seed}']={
                'seed':seed,'arm':arm,'best_epoch':r['best_epoch'],'best_validation_loss':r['best_validation_loss'],
                'standard_loss':best['validation_loss_standard'],'local_loss':best['validation_loss_local'],
                'optimizer_steps':r['optimizer_steps'],'best_optimizer_steps':best['optimizer_steps'],
                'epochs_completed':r['history'][-1]['epoch'],'stopped_early':r.get('stopped_early',False),
                'stopped_at_step_budget':r.get('stopped_at_step_budget',False),
                'peak_cuda_gb':r.get('peak_cuda_gb'),'training_seconds':r['history'][-1]['seconds'],
                'train_pairs':r['train_pairs'],'history':r['history'],'diagnostics':diagnostics(r['validation_outputs'])}
    result={'synthetic_items':count,'runs':reports,'arms':{},'paired_loss_deltas_synthetic_minus_control':{},
        'limits':['same update cap; early stopping, per-example exposure, token/FLOP counts and checkpoint opportunities can differ',
                  'validation is reused development data, not final test accuracy',
                  'synthetic examples have model/assistant review, not independent human review']}
    for key in ('best_validation_loss','standard_loss','local_loss'):
        delta=[reports[f'synthetic_s{s}'][key]-reports[f'control_s{s}'][key] for s in seeds]
        result['paired_loss_deltas_synthetic_minus_control'][key]={'by_seed':dict(zip(map(str,seeds),delta)),'mean':mean(delta)}
    for arm in ('control','synthetic'):
        vals=[reports[f'{arm}_s{s}']['best_validation_loss'] for s in seeds]
        result['arms'][arm]={'mean_best_validation_loss':mean(vals),'seed_sd':stdev(vals) if len(vals)>1 else None}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('directory',type=Path)
    p.add_argument('--seeds',nargs='+',type=int,default=[425,1,2]); p.add_argument('--count',type=int,required=True)
    p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    if args.output.exists(): raise FileExistsError('Choose fresh summary')
    result=analyze(args.directory,args.seeds,args.count)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))


if __name__=='__main__': main()
