import copy
import unittest
from replycoach.generate.mt5_explicit_plan_v1 import planner_source,planned_reply_source,creative_notes,ledger_notes,generator_schedule
from replycoach.generate.sealion_creative_v1 import dataset
from replycoach.generate.mt5_cocktail_v1 import make_cases


class ExplicitPlanTests(unittest.TestCase):
    def test_gold_notes_do_not_read_reference_replies(self):
        for row in dataset()['train']:
            changed=copy.deepcopy(row);changed['alternatives']=[{'english':'FORBIDDEN','singlish':'FORBIDDEN'}];changed['negative']='FORBIDDEN'
            self.assertEqual(creative_notes(row),creative_notes(changed))

    def test_planner_input_excludes_gold_metadata(self):
        row=dataset()['train'][0];changed=copy.deepcopy(row)
        changed['family']='FORBIDDEN';changed['alternatives']=[];changed['gold_notes']='FORBIDDEN';changed['criterion']='FORBIDDEN'
        self.assertEqual(planner_source(row),planner_source(changed))

    def test_generated_note_string_is_really_in_reply_input(self):
        row=dataset()['train'][0]
        self.assertIn('UNIQUE_PREDICTED_NOTE',planned_reply_source(row,'UNIQUE_PREDICTED_NOTE'))
        self.assertNotEqual(planned_reply_source(row,'first'),planned_reply_source(row,'second'))

    def test_no_notes_ablation_has_no_hidden_oracle(self):
        row=dataset()['train'][0];text=planned_reply_source(row,'')
        self.assertIn('Context notes: (none)',text)
        self.assertNotIn(creative_notes(row),text)

    def test_caller_direction_and_missing_purpose_allow_creativity(self):
        row=next(r for r in dataset()['train'] if r['family']=='unknown_call')
        notes=creative_notes(row)
        self.assertIn('User initiated',notes);self.assertIn('new intention',notes)
        self.assertNotIn('refuse',notes.lower())

    def test_ownership_notes_distinguish_lending_and_borrowing(self):
        rows=dataset()['train']
        lend=creative_notes(next(r for r in rows if r['family']=='lend_request'))
        borrow=creative_notes(next(r for r in rows if r['family']=='borrow_offer'))
        self.assertIn('User is lender',lend);self.assertIn('User would borrow',borrow)

    def test_missing_fact_label_does_not_forbid_reply(self):
        row=next(r for r in make_cases('train',1) if r['state']=='other_owner')
        notes=ledger_notes(row)
        self.assertIn('unspecified',notes);self.assertIn('new proposal',notes)

    def test_reply_schedule_balances_contexts_and_variants(self):
        plan=generator_schedule(790);seen=[[0,0,0] for _ in range(100)]
        for s in plan:
            self.assertEqual(len(s['ordinary']),6)
            for i,v in zip(s['new'],s['refs'],strict=True):seen[i][v]+=1
        self.assertTrue(all(v==[2,2,2] for v in seen));self.assertEqual(len(plan),300)


if __name__=='__main__':unittest.main()
