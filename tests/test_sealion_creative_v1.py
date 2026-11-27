import copy
import unittest
from replycoach.generate.sealion_creative_v1 import dataset,audit,schedule,example


class CreativeDataTests(unittest.TestCase):
    def test_split_counts_and_integrity(self):
        data=dataset();self.assertEqual({k:len(v) for k,v in data.items()},{'train':100,'validation':20,'challenge':20})
        for rows in data.values():
            for row in rows:audit(row)

    def test_each_context_gets_equal_multi_reference_exposure(self):
        plan=schedule(790);seen={i:[0,0,0] for i in range(100)}
        for step in plan:
            self.assertEqual(len(step['ordinary']),4);self.assertEqual(len(step['targeted']),4)
            for i,v in zip(step['targeted'],step['variants'],strict=True):seen[i][v]+=1
        self.assertTrue(all(counts==[4,4,4] for counts in seen.values()))
        self.assertEqual(len(plan),300)

    def test_schedule_replays_all_original_examples(self):
        plan=schedule(790);self.assertEqual({i for s in plan for i in s['ordinary']},set(range(790)))
        self.assertEqual(plan,schedule(790))

    def test_missing_call_topic_has_useful_choices(self):
        row=example('unknown_call',0,'train')
        self.assertIn('opinion',row['alternatives'][1]['english'])
        self.assertTrue(all('not available' not in r['english'] for r in row['alternatives']))
        self.assertEqual(row['negative_reason'],'speaker_reversal')

    def test_known_call_topic_is_preserved_in_every_style(self):
        row=example('known_call',0,'train')
        for r in row['alternatives']:
            self.assertIn('our presentation outline',r['english']);self.assertIn('our presentation outline',r['singlish'])

    def test_borrowing_directions_are_distinct(self):
        lend=example('lend_request',0,'train');borrow=example('borrow_offer',0,'train')
        self.assertIn('my bike pump',lend['alternatives'][0]['english'])
        self.assertIn('your bike pump',borrow['alternatives'][0]['english'])

    def test_bad_target_cannot_be_an_accepted_reference(self):
        row=copy.deepcopy(example('completed',0,'train'));row['negative']=row['alternatives'][0]['english']
        with self.assertRaises(AssertionError):audit(row)


class SamplingTests(unittest.TestCase):
    def test_nucleus_keeps_top_token_at_small_threshold(self):
        import torch
        from replycoach.generate.sealion_creative_sampling_v1 import nucleus_probabilities
        ids,probs=nucleus_probabilities(torch.tensor([0.,5.,1.]),1.,.01)
        self.assertEqual(ids[0].item(),1);self.assertEqual(probs[0].item(),1.)

    def test_sampler_stops_at_eos_with_reproducible_seed(self):
        import torch,re
        from types import SimpleNamespace
        from replycoach.generate.sealion_creative_sampling_v1 import sample_decode
        class Model:
            def eval(self):pass
            def __call__(self,**kw):
                token=2 if kw['past_key_values'] is None else 1
                logits=torch.full((1,1,4),-100.);logits[0,0,token]=100.
                return SimpleNamespace(logits=logits,past_key_values=('cached',))
        class Tok:
            eos_token_id=1
            def decode(self,tokens,**kw):return 'hello' if 2 in tokens else ''
        args=(Model(),Tok(),[3],re.compile('UNMATCHED'))
        a=sample_decode(*args,seed=42,device='cpu');b=sample_decode(*args,seed=42,device='cpu')
        self.assertEqual(a,b);self.assertEqual(a['token_ids'],[2,1]);self.assertEqual(a['status'],'complete')


if __name__=='__main__':unittest.main()
