import unittest
from replycoach.generate.sealion_abstention_v1 import canonical_refusal, uncertainty_cue, value_match, metrics, eligible, audit_question_perspective


class AbstentionTests(unittest.TestCase):
    def test_canonical_refusal_ignores_only_surface_format(self):
        self.assertTrue(canonical_refusal(' INFORMATION NOT AVAILABLE! '))
        self.assertFalse(canonical_refusal('Information not available. It starts at 7 pm.'))

    def test_uncertainty_does_not_imply_standard_refusal(self):
        self.assertTrue(uncertainty_cue('I do not know; let me check.'))
        self.assertFalse(canonical_refusal('I do not know; let me check.'))

    def test_embedded_claim_is_not_counted_as_abstention(self):
        self.assertFalse(canonical_refusal('Information not available, but the room is A312.'))
        self.assertTrue(uncertainty_cue('Information not available, but the room is A312.'))

    def test_value_match_requires_token_boundaries(self):
        self.assertFalse(value_match('The price is $19.',[['9']]))
        self.assertTrue(value_match('The price is $9.',[['9']]))

    def test_value_aliases(self):
        self.assertTrue(value_match('On floor eight.',[['8','eight']]))
        self.assertFalse(value_match('On floor nine.',[['8','eight']]))

    def test_balanced_metric_does_not_reward_always_refusing(self):
        rows=[dict(item_id=str(i),requires_refusal=i<3,reference_values=[],english='Information not available.',english_status='complete') for i in range(7)]
        score=metrics(rows)
        self.assertEqual(score['answerability_balanced_score'],.5)
        self.assertEqual(score['exact_over_refusals'],4)
        self.assertFalse(eligible(score,score))

    def test_correct_selective_behavior(self):
        rows=[dict(item_id='a',requires_refusal=True,reference_values=[],english='Information not available.',english_status='complete'),
            dict(item_id='b',requires_refusal=False,reference_values=[['G214']],english='It is in G214.',english_status='complete')]
        score=metrics(rows)
        self.assertEqual(score['answerability_balanced_score'],1.)
        self.assertEqual(score['supported_value_matches'],1)
        self.assertTrue(eligible(score,score))

    def test_generation_failure_rejects_snapshot(self):
        rows=[dict(item_id='a',requires_refusal=False,reference_values=[],english='Hello.',english_status='length_limit')]
        score=metrics(rows)
        self.assertFalse(eligible(score,score))

    def test_ordinary_rows_do_not_become_false_ground_truth(self):
        rows=[dict(item_id='old',requires_refusal=None,reference_values=[],english='Hello.',english_status='complete')]
        score=metrics(rows)
        self.assertEqual(score['missing_cases'],0)
        self.assertEqual(score['answerable_cases'],0)

    def test_authored_user_fact_question_must_not_refer_to_friend_self(self):
        self.assertFalse(audit_question_perspective(dict(family='item_colour',message='What colour did I say my water bottle was?')))

    def test_authored_debt_question_keeps_direction(self):
        self.assertFalse(audit_question_perspective(dict(family='amount_owed',message='How much did I say I owe you?')))
        self.assertTrue(audit_question_perspective(dict(family='amount_owed',message='How much did you say you owe me?')))

    def test_authored_user_fact_question_accepts_correct_perspective(self):
        self.assertTrue(audit_question_perspective(dict(family='meeting_room',message='Which room did you say you booked?')))

    def test_perspective_audit_does_not_relabel_original_team_inputs(self):
        self.assertTrue(audit_question_perspective(dict(message='What did I say about my timetable?')))


if __name__=='__main__':unittest.main()
