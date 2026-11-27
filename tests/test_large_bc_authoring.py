"""Protect the speaker and review boundaries that failed during initial authoring."""
import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('authoring',Path(__file__).resolve().parents[1]/'scripts/sealion_large_bc_data.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class AuthoringBoundaries(unittest.TestCase):
 def test_resume_counts_failed_attempts_and_reserved_interrupted_call(self):
  self.assertEqual(m.latest_attempt([{'attempt':1},{'attempt':5}],[{'attempt':6}],{'attempts':6}),6)
  self.assertEqual(m.latest_attempt([{'attempt':1},{'attempt':5}],[{'attempt':6}],{'attempts':6},7),7)
 def test_incoming_friend_and_candidate_you_have_distinct_roles(self):
  row=m.structural(dict(history=[dict(speaker='You',text='I borrowed your umbrella.'),dict(speaker='Friend',text='Return it tomorrow please.')],incoming_friend='Can you pass it to me after class?',you_drafts=[dict(standard='Yes, after class.',local='can, after class'),dict(standard='Could I return it before class instead?',local='before class instead can?')]))
  self.assertEqual(row['speakers'],['user','peer'])
  self.assertTrue(m.render(row).endswith('Friend: Can you pass it to me after class?'))
  self.assertTrue(m.render(row).startswith('You: I borrowed your umbrella.'))
 def test_reference_drafts_do_not_enter_labeled_chat(self):
  row=dict(context=['Hi.'],speakers=['user'],message='Hello.',drafts=[dict(standard='SECRET_EN',local='SECRET_LOCAL')])
  self.assertNotIn('SECRET',m.render(row))
 def test_natural_identical_short_styles_are_allowed(self):
  row=m.structural(dict(history=[],incoming_friend='Want dinner?',you_drafts=[dict(standard='Yes',local='Yes'),dict(standard='No',local='No')]))
  self.assertEqual(len(row['drafts']),2)
 def test_accept_verdict_cannot_override_specific_errors(self):
  review=dict(a_chat_facts='You borrowed it.',b_english_errors='Reverses owner.',c_rewrite_errors='none',d_style_errors='none',z_verdict='accept')
  self.assertFalse(m.passed(review))
  review['b_english_errors']='none';self.assertTrue(m.passed(review))
if __name__=='__main__':unittest.main()
