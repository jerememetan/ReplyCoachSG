"""Complete LOCAL postprocessing only; preserve failed original report/checkpoints."""
import gc,json
from pathlib import Path
import torch
from replycoach.generate.v5_provider import V5Generator,DECODING
from replycoach.generate.rewrite_v5 import rewrite_standard
from replycoach.generate.style_control import mt5_input
from run_mt5_target_repair_v1 import ROOT,OUT,read,dump,original_validation


def main():
    report=read('artifacts/results.json')
    report['status']='recovering_postprocessing'
    data_map=dict(fresh=read('data/evaluation/mt5_target_repair_v1/inputs.json'),reused_diagnostics=read('data/evaluation/mt5_grounding_audit_v1/inputs.json'),original_validation=original_validation())
    torch.set_num_threads(4)
    local=V5Generator(Path(report['recipe']['rewrite_checkpoint']),device='cuda')
    for key,run in report['runs'].items():
        for name,items in run['outputs'].items():
            eligible=[]
            for row,item in zip(data_map[name],items,strict=True):
                if row['item_id']!=item['item_id']:raise ValueError('ID mismatch')
                if 'local' in item:continue
                anchor=item['standard']
                if not anchor.strip():item['rewrite_status']='blocked_empty_standard';continue
                text=mt5_input('LOCAL_REWRITE',row['message'],row['context'],True,row['speakers'],standard_reply=anchor)
                if len(local.tokenizer(text)['input_ids'])>local.max_source:item['rewrite_status']='blocked_source_budget';continue
                eligible.append(({**row,'response':''},item))
            for start in range(0,len(eligible),4):
                part=eligible[start:start+4];anchors=[item['standard'] for _,item in part]
                replies=rewrite_standard(local.model,local.tokenizer,[row for row,_ in part],anchors,local.max_source,local.max_target,'cuda',DECODING,True)
                for (_,item),anchor,text in zip(part,anchors,replies,strict=True):
                    item.update(local_standard_anchor=anchor,local=text,rewrite_status='complete' if text.strip() else 'failed_empty_local')
                dump(OUT/'results_recovered.json',report)
            print('Recovered',key,name,flush=True)
    failed=[dict(run=k,route=route,item_id=x['item_id'],status=x['rewrite_status']) for k,run in report['runs'].items() for route,items in run['outputs'].items() for x in items if x['rewrite_status']!='complete']
    report.update(status='complete_with_failures' if failed else 'complete',failed_stages=failed,recovery_note='Original training job failed with missing response field in LOCAL wrapper. English outputs/checkpoints unchanged. Original results.json preserved; no retraining.')
    dump(OUT/'results_recovered.json',report)
    print('Recovery complete',report['status'],flush=True)

if __name__=='__main__':main()
