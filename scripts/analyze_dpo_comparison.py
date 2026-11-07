"""Summarize matched validation runs without loading models or reading test data."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from statistics import mean, stdev
import unicodedata


def normal(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC',text).casefold()))


def diagnostics(rows):
    result={}
    for style in ('standard','local'):
        texts=[r['generated_'+style] for r in rows]
        counts=Counter(normal(t) for t in texts)
        result[style]={
            'outputs':len(texts), 'blank':sum(not t.strip() for t in texts),
            'unique_normalized':len(counts),
            'exact_message_echo':sum(normal(r['generated_'+style])==normal(r['message']) for r in rows),
            'exact_context_turn_copy':sum(normal(r['generated_'+style]) in [normal(t) for t in r['context']] for r in rows),
            'mean_words':round(mean(len(normal(t).split()) for t in texts),3),
            'most_common':counts.most_common(5)}
    result['same_text_both_styles']=sum(normal(r['generated_standard'])==normal(r['generated_local']) for r in rows)
    return result


def analyze(directory, seeds):
    reports={}
    common=('base_model','train_items','train_pairs','train_pairs_by_kind','validation_items',
            'lora','epochs_requested','patience','batch_size','learning_rate','speaker_labels',
            'max_source_tokens','max_target_tokens','joint','unlikelihood','synthetic')
    for seed in seeds:
        pair={arm:json.loads((directory/f'{arm}_s{seed}.json').read_text()) for arm in ('sft','dpo')}
        for key in common:
            if pair['sft'].get(key)!=pair['dpo'].get(key):
                raise ValueError(f'Unmatched {key} for seed {seed}')
        if pair['sft']['preferences'] is not None or pair['dpo']['preferences']['pairs']!=73:
            raise ValueError('Expected supervised control and 73-pair DPO arm')
        outputs=[{r['item_id']:r for r in pair[arm]['validation_outputs']} for arm in ('sft','dpo')]
        if len(outputs[0])!=50 or outputs[0].keys()!=outputs[1].keys():
            raise ValueError('Expected same 50 validation conversations')
        for item in outputs[0]:
            for key in ('context','message','reference_standard','reference_local'):
                if outputs[0][item][key]!=outputs[1][item][key]:
                    raise ValueError('Validation inputs or references differ')
        for arm,r in pair.items():
            if r['seed'] != seed:
                raise ValueError('Run seed differs from expected paired seed')
            reports[f'{arm}_s{seed}']={
                'seed':seed,'arm':arm,'best_epoch':r['best_epoch'],
                'best_validation_loss':r['best_validation_loss'],
                'best_validation_loss_standard':next(h['validation_loss_standard'] for h in r['history'] if h['epoch']==r['best_epoch']),
                'best_validation_loss_local':next(h['validation_loss_local'] for h in r['history'] if h['epoch']==r['best_epoch']),
                'epochs_completed':r['history'][-1]['epoch'], 'stopped_early':r.get('stopped_early',False),
                'peak_cuda_gb':r.get('peak_cuda_gb'),'training_seconds':r['history'][-1]['seconds'],
                'history':r['history'],'diagnostics':diagnostics(r['validation_outputs'])}
    summary={'runs':reports,'paired_loss_deltas_dpo_minus_sft':{},'arms':{},
             'limits':['validation loss selected checkpoints, so it is not independent final quality evidence',
                       'three seeds and 50 reused validation conversations do not establish general reliability',
                       'copy/repetition metrics are diagnostics, not semantic accuracy',
                       'DPO training agreement measures relative movement on training pairs, not held-out reply quality']}
    for style,key in [('all','best_validation_loss'),('standard','best_validation_loss_standard'),('local','best_validation_loss_local')]:
        values=[reports[f'dpo_s{s}'][key]-reports[f'sft_s{s}'][key] for s in seeds]
        summary['paired_loss_deltas_dpo_minus_sft'][style]={'by_seed':dict(zip(map(str,seeds),values)), 'mean':mean(values)}
    for arm in ('sft','dpo'):
        values=[reports[f'{arm}_s{s}']['best_validation_loss'] for s in seeds]
        summary['arms'][arm]={'mean_best_validation_loss':mean(values),'seed_sd':stdev(values) if len(values)>1 else None}
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--seeds',nargs='+',type=int,default=[425,1,2])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Choose a fresh summary')
    result=analyze(args.directory,args.seeds)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))


if __name__=='__main__': main()
