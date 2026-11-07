"""Compare fixed-scenario chat-register augmentation against formal inputs/control."""
import argparse,json
from pathlib import Path
from statistics import mean
from analyze_dpo_comparison import diagnostics


def analyze(previous,current):
    out={}; shared=('base_model','train_items','train_pairs','train_pairs_by_kind','validation_items',
        'lora','epochs_requested','patience','batch_size','learning_rate','speaker_labels',
        'max_source_tokens','max_target_tokens','joint','unlikelihood','preferences','max_steps')
    for seed in (425,1,2):
        source={'control':previous/f'control_s{seed}.json','formal':previous/f'synthetic_s{seed}.json',
                'register':current/f'register_s{seed}.json'}
        reports={arm:json.loads(p.read_text()) for arm,p in source.items()}
        for key in shared:
            if reports['formal'].get(key)!=reports['register'].get(key):
                raise ValueError(f'Unmatched formal/register {key}, seed {seed}')
        if reports['formal']['synthetic']['selected_ids']!=reports['register']['synthetic']['selected_ids']:
            raise ValueError('Scenario selection/order changed')
        views=[{r['item_id']:r for r in report['validation_outputs']} for report in reports.values()]
        if any(len(rows)!=50 or rows.keys()!=views[0].keys() for rows in views):
            raise ValueError('Incomplete or different validation IDs')
        for item in views[0]:
            for field in ('context','message','reference_standard','reference_local'):
                if any(v[item][field]!=views[0][item][field] for v in views):
                    raise ValueError('Validation inputs changed')
        for arm,r in reports.items():
            if r['seed']!=seed or r['optimizer_steps']!=280: raise ValueError('Wrong seed/update count')
            h=next(v for v in r['history'] if v['epoch']==r['best_epoch'])
            out[f'{arm}_s{seed}']={'arm':arm,'seed':seed,'best_validation_loss':r['best_validation_loss'],
                'standard_loss':h['validation_loss_standard'],'local_loss':h['validation_loss_local'],
                'best_epoch':r['best_epoch'],'epochs_completed':r['history'][-1]['epoch'],
                'optimizer_steps':r['optimizer_steps'],'best_optimizer_steps':h['optimizer_steps'],
                'history':r['history'],'peak_cuda_gb':r.get('peak_cuda_gb'),
                'training_seconds':r['history'][-1]['seconds'],'diagnostics':diagnostics(r['validation_outputs'])}
    summary={'runs':out,'arms':{},'register_minus_formal':{},'register_minus_control':{},
             'limits':['same 59 scenarios/targets; only inputs reworded, but tokenization/length differ',
                       'reused validation and assistant-authored/reviewed data, not independent gold',
                       'qualitative reviewer has seen previous comparator outputs; anonymization is not full blinding']}
    for arm in ('control','formal','register'):
        summary['arms'][arm]={'mean_best_validation_loss':mean(out[f'{arm}_s{s}']['best_validation_loss'] for s in (425,1,2))}
    for baseline in ('formal','control'):
        for field in ('best_validation_loss','standard_loss','local_loss'):
            vals={str(s):out[f'register_s{s}'][field]-out[f'{baseline}_s{s}'][field] for s in (425,1,2)}
            summary['register_minus_'+baseline][field]={'by_seed':vals,'mean':mean(vals.values())}
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('previous',type=Path);p.add_argument('current',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('Choose fresh summary')
    result=analyze(args.previous,args.current);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))


if __name__=='__main__':main()
