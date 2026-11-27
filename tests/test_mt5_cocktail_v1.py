import copy
import unittest
from replycoach.generate.mt5_cocktail_v1 import make_cases, audit_case, required_fact, answer_label, auxiliary_source, training_schedule, question


class CocktailTests(unittest.TestCase):
    def test_split_integrity_and_ledger(self):
        for split,n in [('train',32),('validation',6),('challenge',4)]:
            for row in make_cases(split,n):
                audit_case(row)

    def test_other_persons_fact_does_not_answer_question(self):
        rows=make_cases('train',2)
        for row in rows:
            if row['state']=='other_owner':
                self.assertEqual(answer_label(row),'MISSING')
                self.assertTrue(required_fact(row).endswith('|unknown'))
                self.assertTrue(row['ledger'])

    def test_speaker_swap_changes_answerability(self):
        row=next(r for r in make_cases('train',1) if r['state']=='other_owner')
        changed=copy.deepcopy(row);changed['owner']=changed['ledger'][0]['owner']
        self.assertEqual(answer_label(row),'MISSING')
        self.assertEqual(answer_label(changed),'ANSWERABLE')

    def test_empty_history_with_incoming_evidence_answers(self):
        row=next(r for r in make_cases('train',1) if r['state']=='latest_fact')
        self.assertFalse(row['context']);self.assertEqual(answer_label(row),'ANSWERABLE')
        self.assertIn(row['ledger'][0]['value'],required_fact(row))

    def test_self_contained_chat_and_math_are_answerable(self):
        for row in make_cases('train',1):
            if row.get('self_contained'):
                self.assertEqual(answer_label(row),'ANSWERABLE')

    def test_debt_direction_matches_incoming_speaker(self):
        self.assertEqual(question('amount_owed','user'),'How much did you say you owe me?')
        self.assertEqual(question('amount_owed','peer'),'How much did I say I owe you?')

    def test_runtime_sources_exclude_gold_ledger(self):
        row=make_cases('train',1)[0];before=auxiliary_source(row,'FACTS')
        row=copy.deepcopy(row);row['ledger']=[];row['fact_target']='FORBIDDEN';row['answer_target']='FORBIDDEN'
        self.assertEqual(auxiliary_source(row,'FACTS'),before)

    def test_matched_schedule_is_reproducible(self):
        a=training_schedule(1535,800,steps=3)
        self.assertEqual(a,training_schedule(1535,800,steps=3))
        self.assertTrue(all(len(s['main'])==12 and len(s['facts'])==2 and len(s['answerability'])==2 for s in a))


if __name__=='__main__':unittest.main()
