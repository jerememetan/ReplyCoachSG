import unittest
from replycoach.generate.sealion_context_guard_v1 import guard_context, NEUTRAL_CALL
from replycoach.generate.sealion_context_pipeline_v1 import repair_pair


def case(context,message,speakers=None):
    return dict(context=context,speakers=speakers or ['user']*len(context),message=message)


class ContextGuardTests(unittest.TestCase):
    def test_call_purpose_unknown_does_not_invent_topic(self):
        r=case(['Can you ring me?','Busy now.','Later is okay.'],
            'Not sure when. What did you want to call me about?',['user','peer','user'])
        out=guard_context(r,'Can you call me now?')
        self.assertEqual(out['english'],NEUTRAL_CALL)
        self.assertEqual(out['rule'],'unstated_call_purpose_text_first')

    def test_known_purpose_is_grounded(self):
        r=case(['I need your advice on my timetable. Could we talk?'],'What is the call about?')
        self.assertEqual(guard_context(r,'What do you want?')['english'],'I need your advice on my timetable.')

    def test_generic_chat_word_does_not_ground_an_invented_topic(self):
        r=case(['Could we chat on the phone for a minute?','Another time is okay.'],
            'What is the call about?')
        self.assertEqual(guard_context(r,"Let's chat about work for a few minutes.")['english'],NEUTRAL_CALL)

    def test_purpose_rule_does_not_trigger_for_time_question(self):
        r=case(['Can you call me?'],'When are you free to call?')
        self.assertEqual(guard_context(r,'After lunch.')['action'],'keep_model')

    def test_noncall_about_question_is_unchanged(self):
        r=case(['I bought a book.'],'What is it about?')
        self.assertEqual(guard_context(r,'It is about cooking.')['action'],'keep_model')

    def test_explicit_budget_blocks_over_budget_offer(self):
        r=case(['I can spend no more than $22.','The ticket is $39.'],
            'Should I book the ticket?',['user','peer'])
        out=guard_context(r,'Yes please.')
        self.assertEqual(out['rule'],'explicit_budget_limit')
        self.assertIn('$22',out['english'])

    def test_affordable_choice_uses_prices_not_cap_as_price(self):
        r=case(['I have $17 left to spend.','The choices cost $12 and $30.'],
            'Which option can you afford?',['user','peer'])
        out=guard_context(r,'The $30 option.')
        self.assertIn('$12 option',out['english']);self.assertIn('$17 budget',out['english'])

    def test_explicit_gift_does_not_apply_my_budget(self):
        r=case(['My budget is $10.'],"I'll pay for the $30 ticket. Want me to book it?")
        self.assertEqual(guard_context(r,'Thanks, yes please.')['action'],'keep_model')

    def test_ambiguous_multiple_prices_not_forced(self):
        r=case(['My budget is $10.','They have tickets for $8 and $30.'],
            'Should I buy a ticket?',['user','peer'])
        self.assertEqual(guard_context(r,'Which ticket?')['action'],'keep_model')

    def test_time_bound_respects_am_pm(self):
        r=case(['I cannot leave work until 7:20 pm.'],'Can we meet at 6 pm?')
        out=guard_context(r,'Yes, 6 pm works.')
        self.assertEqual(out['rule'],'explicit_earliest_time');self.assertIn('7:20 pm',out['english'])

    def test_before_then_resolves_same_sentence_time(self):
        r=case(['My train arrives at 8 pm so I cannot reach town before then.'],'Can you be there at 6 pm?')
        self.assertEqual(guard_context(r,'Sure.')['rule'],'explicit_earliest_time')

    def test_after_bound_is_not_automatically_accepted_or_rejected(self):
        r=case(['I cannot leave until 3 pm.'],'Can we meet at 5 pm?')
        self.assertEqual(guard_context(r,'I will check.')['action'],'keep_model')

    def test_unknown_availability_does_not_become_time_constraint(self):
        r=case(['I cannot say whether I will reach before 8 pm.'],'Can we meet at 6 pm?')
        self.assertEqual(guard_context(r,'I am not sure.')['action'],'keep_model')

    def test_busy_morning_does_not_block_afternoon(self):
        r=case(['I have an exam on Thursday morning.'],'Would Thursday afternoon work?')
        self.assertEqual(guard_context(r,'I will check.')['action'],'keep_model')

    def test_busy_matching_period(self):
        r=case(['I have an exam on Thursday morning.'],'Would Thursday morning work?')
        self.assertEqual(guard_context(r,'Yes.')['rule'],'explicit_busy_day')

    def test_uncertain_obligation_does_not_become_unavailable(self):
        r=case(['I might have a shift on Tuesday.'],'Can we meet on Tuesday?')
        self.assertEqual(guard_context(r,'Let me check.')['action'],'keep_model')

    def test_no_exam_is_not_an_obligation(self):
        r=case(['I do not have an exam on Tuesday.'],'Can we meet on Tuesday?')
        self.assertEqual(guard_context(r,'I will check.')['action'],'keep_model')

    def test_peer_borrowing_keeps_return_direction(self):
        r=case(['I borrowed your torch last week.'],'Who should return the torch?',['peer'])
        self.assertEqual(guard_context(r,'I will return it.')['english'],'You need to return my torch to me.')

    def test_user_borrowing_keeps_return_direction(self):
        r=case(['I borrowed your toolbox.'],'Who should return the toolbox?')
        self.assertEqual(guard_context(r,'You return it.')['english'],'I need to return your toolbox to you.')

    def test_completed_return_does_not_require_return_again(self):
        r=case(['I borrowed your torch.','I already returned it.'],'Who should return the torch?')
        self.assertEqual(guard_context(r,'It is returned already.')['action'],'keep_model')

    def test_multiple_objects_with_ambiguous_pronoun_abstains(self):
        r=case(['I borrowed your torch. You borrowed my headphones.'],'Who should return it?')
        self.assertEqual(guard_context(r,'Which item?')['action'],'keep_model')

    def test_location_comes_from_me_not_friend(self):
        r=case(['I am waiting by the east lift.','I am at the bookshop.'],
            'Where can I find you?',['user','peer'])
        self.assertEqual(guard_context(r,'At the bookshop.')['english'],'I am waiting by the east lift.')

    def test_event_location_is_not_personal_location(self):
        r=case(['I am at the bookshop.'],'Where is the workshop held?')
        self.assertEqual(guard_context(r,'In the hall.')['action'],'keep_model')

    def test_clarification_retrieves_requested_object(self):
        r=case(['Could you proofread the introduction? The charts are checked.'],
            'Do you need the introduction or the charts proofread?')
        self.assertEqual(guard_context(r,'Both.')['english'],'Could you proofread the introduction?')

    def test_when_question_does_not_retrieve_object_instead(self):
        r=case(['Could you review the notes?'],'What time should I review them?')
        self.assertEqual(guard_context(r,'After class.')['action'],'keep_model')

    def test_bad_rewrite_cannot_reverse_grounded_return_direction(self):
        r=case(['I borrowed your torch last week.'],'Who should return the torch?',['peer'])
        original=dict(item_id='x',english='I will return it.',singlish='I return it.')
        out=repair_pair(r,original,lambda row,anchor:dict(reply='I need to return your torch to you.',status='complete'))
        self.assertEqual(out['singlish_status'],'english_fallback')
        self.assertEqual(out['singlish'],out['english'])
        self.assertEqual(out['original_english'],'I will return it.')

    def test_supported_rewrite_records_its_rule_anchor(self):
        r=case(['Can you ring me?'],'What do you want to call me about?')
        original=dict(item_id='x',english='Call me now.',singlish='Call me now lah.')
        out=repair_pair(r,original,lambda row,anchor:dict(reply=anchor,status='complete'))
        self.assertEqual(out['english'],NEUTRAL_CALL)
        self.assertEqual(out['singlish_provenance'],'neural_rewrite_of_rule_anchor')
        self.assertTrue(out['english_provenance'].startswith('context_rule:'))

    def test_unmatched_pair_is_preserved_without_rewriting(self):
        r=case(['We are having lunch.'],'How was your day?')
        original=dict(item_id='x',english='Good, thanks.',singlish='Good lah, thanks.')
        def forbidden(*args):raise AssertionError('Unmatched input must not rewrite')
        out=repair_pair(r,original,forbidden)
        self.assertEqual(out['english'],original['english'])
        self.assertEqual(out['singlish'],original['singlish'])

    def test_already_correct_object_answer_is_preserved(self):
        r=case(['Could you send me the route map?'],'Which one did you ask for again?')
        self.assertEqual(guard_context(r,'The route map please.')['action'],'keep_model')

    def test_rewrite_cannot_turn_request_into_which_question(self):
        r=case(['Could you proofread the introduction?'],'Do you need the introduction or the charts proofread?')
        original=dict(item_id='x',english='Both.',singlish='Both ah.')
        out=repair_pair(r,original,lambda row,anchor:dict(reply='Which introduction you want ah?',status='complete'))
        self.assertEqual(out['singlish_status'],'english_fallback')
        self.assertIn('introduced_information_question',out['rewrite_guard']['reasons'])


if __name__=='__main__':unittest.main()
