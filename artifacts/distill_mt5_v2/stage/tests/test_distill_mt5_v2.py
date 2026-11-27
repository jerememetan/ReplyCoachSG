import unittest
from replycoach.generate.distill_mt5_v2 import digest, reviewed, common_subset, student_rows


class DistillationReviewTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(item_id='a', context=['Hi'], speakers=['peer'], message='Can you come?')
        self.raw = dict(**self.row, english=dict(reply='Yes.', status='complete'))
        self.decision = dict(item_id='a', raw_sha256=digest(self.raw), decision='accept', reason='Correct acceptance.')

    def test_missing_reviews_do_not_pass_and_stale_reviews_fail(self):
        self.assertEqual(reviewed([self.raw], [], 'english'), {})
        changed = dict(self.raw, english=dict(reply='No.', status='complete'))
        with self.assertRaises(ValueError):
            reviewed([changed], [self.decision], 'english')

    def test_exact_accepted_english_becomes_anchor_label(self):
        english = reviewed([self.raw], [dict(self.decision, edited_target='Yes, I can come.')], 'english')
        rows = student_rows([self.row], english, {'a': dict(target='Can, I come.')})
        self.assertEqual(rows[0]['standard'], 'Yes, I can come.')
        self.assertEqual(rows[0]['local'], 'Can, I come.')
        original = [dict(self.row, standard='Sure!', local='Can!'), dict(self.row, item_id='b', standard='No.', local='Cannot.')]
        arms = common_subset(original, {'sealion': rows, 'qwen': rows})
        self.assertTrue(all([r['item_id'] for r in data] == ['a'] for data in arms.values()))

    def test_input_changes_and_incomplete_generation_cannot_silently_pass(self):
        raw = dict(self.raw, english=dict(reply='Yes', status='length_limit'))
        with self.assertRaises(ValueError):
            reviewed([raw], [dict(self.decision, raw_sha256=digest(raw))], 'english')
        with self.assertRaises(ValueError):
            common_subset([dict(self.row, standard='Yes', local='Can')], {'qwen': [dict(self.row, message='Changed', standard='Yes', local='Can')]})


if __name__ == '__main__':
    unittest.main()
