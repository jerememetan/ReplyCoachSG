import json
from pathlib import Path
import unittest
from replycoach.generate.sealion_target_overlay_v1 import authored_pairs,schedule

ROOT=Path(__file__).resolve().parents[1]


class OverlayTests(unittest.TestCase):
    def test_overlay_only_changes_synthetic_rewrite_targets(self):
        source={r['item_id']:r for r in json.loads((ROOT/'artifacts/sealion_style_v1/train.json').read_text())}
        overlay=json.loads((ROOT/'artifacts/sealion_rewrite_targets_v1/overlay.json').read_text())
        self.assertEqual(len(overlay),120);self.assertEqual(sum(r['changed'] for r in overlay),100)
        for row in overlay:
            self.assertTrue(row['item_id'].startswith('SYN-'))
            for key in ['context','speakers','message','standard_anchor','response']:
                self.assertEqual(row[key],source[row['item_id']][key])

    def test_replay_and_overlay_are_disjoint_and_cover_original_pool(self):
        source=json.loads((ROOT/'artifacts/sealion_style_v1/train.json').read_text())
        ids={r['item_id'] for r in json.loads((ROOT/'artifacts/sealion_rewrite_targets_v1/overlay.json').read_text())}
        replay=[r for r in source if r['item_id'] not in ids]
        self.assertEqual(len(replay),625);self.assertFalse(ids&{r['item_id'] for r in replay})
        self.assertEqual(len(ids|{r['item_id'] for r in replay}),745)

    def test_takeover_and_receipt_do_not_add_or_reverse_actions(self):
        overlay=json.loads((ROOT/'artifacts/sealion_rewrite_targets_v1/overlay.json').read_text())
        takeover=next(r for r in overlay if r['item_id']=='SYN-volume500-v1-full-124-02')
        self.assertIn('I can take over for you',takeover['new_response'])
        receipt=next(r for r in overlay if r['item_id']=='SYN-HUMAN-EXP1-013')
        self.assertNotIn('take a photo',receipt['new_response'])

    def test_already_natural_controls_remain_identical(self):
        overlay=json.loads((ROOT/'artifacts/sealion_rewrite_targets_v1/overlay.json').read_text())
        controls=[r for r in overlay if not r['changed']]
        self.assertEqual(len(controls),20)
        self.assertTrue(all(r['new_response']==r['response'] for r in controls))

    def test_shared_and_heldout_example_counts_and_separation(self):
        groups=authored_pairs();self.assertEqual({k:len(v) for k,v in groups.items()},
            {'common_train':20,'validation':20,'test':20,'fidelity_controls':10})
        key=lambda r:(tuple(r['context']),tuple(r['speakers']),r['message'],r['standard_anchor'])
        seen=set()
        for values in groups.values():
            keys={key(r) for r in values};self.assertEqual(len(keys),len(values));self.assertFalse(seen&keys);seen|=keys

    def test_common_casual_examples_are_natural_style_not_all_particles(self):
        rows=authored_pairs()['common_train']
        self.assertTrue(any('makan' in r['response'] for r in rows))
        self.assertTrue(any(not any(p in r['response'].lower().split() for p in ['lah','leh','ah','sia']) for r in rows))

    def test_neutral_english_copy_can_be_acceptable(self):
        row=next(r for r in authored_pairs()['validation'] if r['topic']=='thanks')
        self.assertEqual(row['standard_anchor'],row['response'])

    def test_schedule_matches_exposures_and_coverage(self):
        a=schedule(625);self.assertEqual(a,schedule(625));self.assertEqual(len(a),240)
        visits=[0]*120;common=[0]*20
        for row in a:
            self.assertEqual(len(row['replay']),5);self.assertEqual(len(row['overlay']),2)
            for i in row['overlay']:visits[i]+=1
            common[row['common']]+=1
        self.assertEqual(set(visits),{4});self.assertEqual(set(common),{12})
        self.assertEqual({i for row in a for i in row['replay']},set(range(625)))

    def test_fidelity_control_names_amounts_and_modality(self):
        rows={r['topic']:r for r in authored_pairs()['fidelity_controls']}
        for span in ['N416','15:20']:self.assertIn(span,rows['booking']['response'])
        for span in ['$18','Hana','already']:self.assertIn(span,rows['paid']['response'])
        self.assertIn('might',rows['maybe_late']['response'])
        self.assertIn('my charger',rows['lend']['response'].lower())
        self.assertIn('to you',rows['return']['response'])


if __name__=='__main__':unittest.main()
