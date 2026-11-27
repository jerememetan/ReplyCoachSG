import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('schedule',Path(__file__).resolve().parents[1]/'src/replycoach/generate/sealion_large_bc_v1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ScheduleTests(unittest.TestCase):
 def test_warmup_peak_decay_and_final_update(self):
  values=[m.lr_factor(i,100,5) for i in range(1,101)]
  self.assertAlmostEqual(values[0],.2);self.assertEqual(values[4],1.);self.assertEqual(values[-1],0.)
  self.assertTrue(all(a<b for a,b in zip(values[:4],values[1:5])))
  self.assertTrue(all(a>b for a,b in zip(values[4:-1],values[5:])))
 def test_invalid_schedule_positions_cannot_silently_restart(self):
  for args in [(0,100,5),(101,100,5),(1,5,5),(1,100,0)]:
   with self.assertRaises(ValueError):m.lr_factor(*args)
 def test_schedule_is_per_optimizer_update_with_partial_batch(self):
  import math
  examples=17;epochs=4;accumulation=8
  total=math.ceil(examples/accumulation)*epochs
  warmup=max(1,math.ceil(total*.05))
  updates=[i for i in range(1,total+1)]
  self.assertEqual(len(updates),12)
  self.assertEqual(m.lr_factor(updates[-1],total,warmup),0.)
if __name__=='__main__':unittest.main()
