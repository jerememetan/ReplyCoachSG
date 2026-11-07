"""Audit matched runs and prepare unchanged generated replies for qualitative review."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--repair',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    reports={k:json.loads(p.read_text()) for k,p in [('control',args.control),('repair',args.repair)]}
    summary=dict(status='running', stages={}, quality_promotion=False,
                 note='Synthetic ranking and validation CE do not certify reply quality. Reused development, not independent human ratings.')
    for stage in ['english','rewrite']:
        if not all(stage in r['stages'] for r in reports.values()):continue
        phases={k:r['stages'][stage] for k,r in reports.items()}
        a,b=phases.values()
        assert a['initial_adapter_sha256']==b['initial_adapter_sha256']
        assert a['schedule_sha256']==b['schedule_sha256']
        if a['history'] and b['history']:assert a['history'][0]['scores']==b['history'][0]['scores']
        for arm,phase in phases.items():
            if 'auxiliary_changed_primary_gradient' in phase:
                assert phase['auxiliary_changed_primary_gradient']==(arm=='repair')
                assert phase['frozen_base_received_no_gradients']
            if phase.get('status')=='complete':
                assert phase['optimizer_steps']==200 and phase['ordinary_exposures']==1600
                assert phase['auxiliary_pair_exposures']==200
                selected=next(x for x in phase['history'] if x['step']==phase['selected_step'])
                assert selected['replay_guard_passed']
        summary['stages'][stage]=dict(matched_initial_adapter=True,matched_schedule=True,
            arms={k:{field:p.get(field) for field in ['status','optimizer_steps','selected_step','selected_validation',
                 'auxiliary_changed_primary_gradient','frozen_base_received_no_gradients']} for k,p in phases.items()})
    done=all(r['status'] in ['complete','complete_with_flags'] for r in reports.values())
    if done:
        assert reports['control']['baseline_english']==reports['repair']['baseline_english']
        assert reports['control']['baseline_rewrite']==reports['repair']['baseline_rewrite']
        cases=sum((json.loads((ROOT/p).read_text()) for p in ['data/evaluation/sealion_full_rewrite_v1/inputs.json',
                  'data/evaluation/restored_pipeline_v1/inputs.json','data/evaluation/context_pairs_v1/inputs.json']),[])
        rows=[];counts={}
        for method in ['greedy','beam4_ng3']:
            expected=[r['item_id'] for r in cases]
            for arm,r in reports.items():
                assert r['original_adapters_unchanged']
                assert [x['item_id'] for x in r['outputs']['english'][method]]==expected
                for route in ['own_english','fixed_baseline_english']:
                    assert [x['item_id'] for x in r['outputs']['rewrite'][method][route]]==expected
                for route,rr in r['outputs']['rewrite'][method].items():
                    for x in rr:
                        if x['output']['status']=='complete':assert x['output']['standard_anchor']==x['standard_anchor']
                for anchor_group,english_key in [('own_english','outputs'),('fixed_baseline_english','baseline_english')]:
                    english=r['outputs']['english'][method] if english_key=='outputs' else r['baseline_english'][method]
                    assert [x['standard_anchor'] for x in r['outputs']['rewrite'][method][anchor_group]]==[x['output']['reply'] for x in english]
            for index,case in enumerate(cases):
                row=dict(method=method,**case,models={})
                for arm,r in reports.items():
                    row['models'][arm]=dict(english=r['outputs']['english'][method][index]['output'],
                                            local=r['outputs']['rewrite'][method]['own_english'][index]['output'])
                row['models']['baseline']=dict(english=reports['control']['baseline_english'][method][index]['output'],
                                                local=reports['control']['baseline_rewrite'][method][index]['output'])
                rows.append(row)
            counts[method]={arm:dict(
                english_changed=sum(x['output']['reply']!=y['output']['reply'] for x,y in zip(r['outputs']['english'][method],r['baseline_english'][method],strict=True)),
                paired_local_changed=sum(x['output']['reply']!=y['output']['reply'] for x,y in zip(r['outputs']['rewrite'][method]['own_english'],r['baseline_rewrite'][method],strict=True)))
                for arm,r in reports.items()}
        summary.update(status='training_and_generation_complete',changes_vs_baseline=counts,
                       flags={k:r.get('flags',[]) for k,r in reports.items()},all_anchor_routes_verified=True,
                       baseline_replay_identical_between_arms=True,review_pending=True)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        packet=args.output.with_name(args.output.stem+'_paired_outputs.json')
        packet.write_text(json.dumps(dict(note='All126 conversations under two decoding methods, unedited outputs. No automatic correctness ratings.',cases=rows),indent=2)+'\n')
        summary['paired_output_packet']=str(packet.resolve().relative_to(ROOT))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
