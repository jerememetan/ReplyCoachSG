import copy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('audited_helper',Path(__file__).resolve().parents[1]/'src/replycoach/generate/sealion_audited_restart_v1.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


class Boundaries(unittest.TestCase):
    def setUp(self):
        self.row=dict(item_id='a',context=['I have your book.'],speakers=['user'],message='Return it tomorrow?',standard='Yes, tomorrow.',local='okok')

    def test_reference_reply_cannot_enter_direct_input(self):
        other=dict(self.row,standard='SECRET_EN',local='SECRET_SG')
        self.assertEqual(h.prompt(self.row,'STANDARD'),h.prompt(other,'STANDARD'))

    def test_only_supplied_english_enters_rewrite(self):
        other=dict(self.row,standard='SECRET_EN',local='SECRET_SG')
        self.assertEqual(h.prompt(self.row,'LOCAL_REWRITE','Chosen text'),h.prompt(other,'LOCAL_REWRITE','Chosen text'))
        self.assertIn('Chosen text',h.prompt(other,'LOCAL_REWRITE','Chosen text'))
        self.assertNotIn('SECRET',h.prompt(other,'LOCAL_REWRITE','Chosen text'))

    def test_empty_anchor_fails(self):
        with self.assertRaises(ValueError):h.prompt(self.row,'LOCAL_REWRITE',' ')

    def test_roles_are_not_alternation_guesses(self):
        self.assertIn('You: I have your book.\nFriend: Return it tomorrow?',h.prompt(self.row,'STANDARD'))
        with self.assertRaises(ValueError):h.prompt(dict(self.row,speakers=[]),'LOCAL')

    def test_duplicate_content_across_different_ids_fails(self):
        with self.assertRaises(ValueError):h.check_splits({'train':[self.row],'val':[dict(self.row,item_id='b')]})

    def test_split_overlap_fails(self):
        with self.assertRaises(ValueError):h.check_splits({'train':[self.row],'val':[self.row]})

    def test_quarantine_is_not_a_source_edit(self):
        records=[dict(id=n,session_memory=[],message={'text':n},replies={'standard':'yes','local':'okok'}) for n in ['a','b']]
        before=copy.deepcopy(records);selected=h.filter_training(records,{'b'})
        self.assertEqual([r['item_id'] for r in selected],['a']);self.assertEqual(records,before)

    def test_short_contextual_reply_is_allowed(self):
        h.check_splits({'train':[self.row],'val':[dict(self.row,item_id='b',message='Hello')]})
        self.assertEqual(self.row['local'],'okok')

    def test_nonfinite_selection_is_rejected(self):
        with self.assertRaises(ValueError):h.selection(float('nan'),1.)
        self.assertEqual(h.selection(2.,4.),3.)


if __name__=='__main__':unittest.main()
