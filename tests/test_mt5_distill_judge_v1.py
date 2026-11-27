"""Evaluation-integrity checks: matching, deduplication, missingness and aggregation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
prepare=module('prepare_distill_judge',ROOT/'scripts/prepare_mt5_distill_judge_v1.py')
run=module('run_distill_judge',ROOT/'scripts/run_mt5_distill_judge_v1.py')

class EvaluationIntegrity(unittest.TestCase):
    def setUp(self):
        stage=ROOT/'artifacts/mt5_distill_judge_v1/stage'
        self.config=json.loads((stage/'config.json').read_text())
        self.public=json.loads((stage/'judge_inputs.json').read_text())
        self.mapping=json.loads((stage/'PRIVATE_mapping.json').read_text())

    def test_identical_text_never_shared_across_conversations_or_styles(self):
        queries=[];rows=[]
        for i in range(2):
            queries.append({'id':f'ITEM-{i}','message':{'text':'hello'},'session_memory':[]})
            rows.append({'item_id':f'ITEM-{i}','message':'hello','context':[],'speakers':[],**{m+'_'+s:'same' for m in prepare.ARMS for s in ['english','singlish']}})
        packet=prepare.build_packet(rows,queries)
        self.assertEqual(len(packet),4)
        self.assertEqual(sum(len(g['private']['models']) for g in packet),20)
        self.assertEqual(len({(g['public']['conversation_id'],g['public']['style']) for g in packet}),4)
        self.assertTrue(all('models' not in g['public'] and 'item_id' not in g['public'] for g in packet))

    def test_every_model_has_identical_complete_coverage(self):
        coverage=run.verify_packet(self.config,self.public,self.mapping)
        self.assertEqual({len(v) for v in coverage.values()},{100})
        self.assertEqual(len(run.tasks_for(self.public,self.config['personas'])),3390)
        self.assertEqual(sum(len(v['models']) for v in self.mapping.values())*10,5000)

    def test_model_metadata_in_public_packet_is_rejected(self):
        public=[dict(r) for r in self.public];public[0]['system']='unchanged'
        with self.assertRaises(AssertionError):run.verify_packet(self.config,public,self.mapping)

    def test_incomplete_run_cannot_publish_rankings(self):
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            run.expand_and_summarise(self.config,self.public,self.mapping,{})

    def test_conversations_weighted_equally_despite_deduplication(self):
        one_case=next(iter(self.mapping.values()))['item_id'];latest={}
        for r,p in run.tasks_for(self.public,self.config['personas']):
            score=5 if self.mapping[r['candidate_id']]['item_id']==one_case else 1
            latest[(r['candidate_id'],p['id'])]={'status':'ok','rating':{'appropriateness':score,'naturalness':score,'style_fit':score,'would_send':score==5,'uncertain':False}}
        expanded,summaries,differences=run.expand_and_summarise(self.config,self.public,self.mapping,latest)
        self.assertEqual(len(expanded),5000)
        self.assertEqual(len(summaries),10)
        for row in summaries:
            self.assertAlmostEqual(row['appropriateness'],1.08)
            self.assertEqual(row['would_send_pct'],2)
        self.assertTrue(all(row['paired_mean_difference']==row['ci95_low']==row['ci95_high']==0 for row in differences))

    def test_resume_rejects_different_run_identity(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'ratings.jsonl';path.write_text(json.dumps({'candidate_id':'C1','persona_id':'P01','run_id':'wrong','status':'error','rating':None})+'\n')
            with self.assertRaisesRegex(ValueError,'Journal'):
                run.load_journal(path,'correct',{('C1','P01')},None)

if __name__=='__main__':unittest.main()
