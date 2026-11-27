import unittest
from replycoach.generate.distill_mt5_v2 import digest, reviewed, common_subset, student_rows
from replycoach.generate.distill_mt5_v2_matching import match_reviewed_arms


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

    def test_original_ineligible_inputs_are_excluded_from_all_matched_arms(self):
        extra = dict(self.row, item_id='original-pair-ineligible')
        original = [dict(self.row, standard='Original yes', local='Original can')]
        teachers = {
            arm: [dict(extra, standard='Teacher extra', local='Extra'),
                  dict(self.row, standard=arm + ' yes', local=arm + ' can')]
            for arm in ('sealion', 'qwen')
        }
        arms = match_reviewed_arms(original, teachers, [extra, self.row])
        self.assertEqual(list(arms), ['original', 'sealion', 'qwen'])
        self.assertTrue(all([r['item_id'] for r in rows] == ['a'] for rows in arms.values()))
        self.assertEqual(arms['qwen'][0]['standard'], 'qwen yes')
        self.assertEqual(arms['original'][0]['standard'], 'Original yes')

    def test_excluded_teacher_inputs_are_validated_before_filtering(self):
        extra = dict(self.row, item_id='original-pair-ineligible')
        original = [dict(self.row, standard='Yes', local='Can')]
        for invalid in (dict(extra, message='Changed'), dict(extra, item_id='unknown')):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                match_reviewed_arms(original, {'qwen': [dict(invalid, standard='Yes', local='Can')]}, [self.row, extra])

    def test_duplicate_full_pool_and_excluded_teacher_rows_fail(self):
        original = [dict(self.row, standard='Yes', local='Can')]
        extra = dict(self.row, item_id='original-pair-ineligible')
        with self.assertRaises(ValueError):
            match_reviewed_arms(original, {}, [self.row, self.row])
        with self.assertRaises(ValueError):
            match_reviewed_arms(original, {'qwen': [extra, extra]}, [self.row, extra])
        with self.assertRaises(ValueError):
            match_reviewed_arms([dict(original[0], message='Changed')], {}, [self.row])


if __name__ == '__main__':
    unittest.main()
