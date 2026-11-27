"""Continuation preserves trained LoRA tensors and updates one shared adapter."""
import tempfile
import unittest
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import MT5Config
from replycoach.generate.mt5_pretrained_compat import OriginalMT5
from replycoach.generate.mt5_warm_distill_v1 import load_shared_adapter, paired_update_groups, selection_score


class WarmDistillTests(unittest.TestCase):
    def test_balanced_complete_deterministic_pair_groups(self):
        groups = list(paired_update_groups(7, 425, 1))
        self.assertEqual([len(g) for g in groups], [8, 6])
        self.assertEqual(groups, list(paired_update_groups(7, 425, 1)))
        self.assertNotEqual(groups, list(paired_update_groups(7, 425, 2)))
        self.assertEqual(sorted(i for g in groups for s, i in g if s == 'english'), list(range(7)))
        self.assertEqual(sorted(i for g in groups for s, i in g if s == 'singlish'), list(range(7)))
        for group in groups:
            for j in range(0, len(group), 2):
                self.assertEqual(group[j][0], 'english')
                self.assertEqual(group[j + 1], ('singlish', group[j][1]))

    def test_joint_selection_can_retain_unchanged_epoch_zero(self):
        scores = [dict(english=dict(token_ce=2.), singlish=dict(token_ce=3.)),
                  dict(english=dict(token_ce=1.9), singlish=dict(token_ce=3.4))]
        self.assertEqual(min(range(len(scores)), key=lambda i: selection_score(scores[i])), 0)
        self.assertEqual(selection_score(scores[0]), 2.5)

    def test_resume_trained_tensors_and_shared_updates_leave_base_frozen(self):
        torch.manual_seed(425)
        def base():
            return OriginalMT5(MT5Config(vocab_size=23, d_model=8, d_kv=4, d_ff=16,
                num_layers=1, num_decoder_layers=1, num_heads=2, dropout_rate=0.,
                pad_token_id=0, eos_token_id=1, decoder_start_token_id=0, tie_word_embeddings=True))
        original = get_peft_model(base(), LoraConfig(task_type='SEQ_2_SEQ_LM',r=2,lora_alpha=4,
                                                   lora_dropout=0.,target_modules=['q','v']))
        # Both matrices nonzero, as in a trained adapter; detect accidental reset.
        with torch.no_grad():
            for n, p in original.named_parameters():
                if 'lora_' in n: p.normal_(0., .05)
        expected = {k:v.detach().clone() for k,v in get_peft_model_state_dict(original).items()}
        with tempfile.TemporaryDirectory() as directory:
            original.save_pretrained(directory)
            model = load_shared_adapter(base(), Path(directory), expected_parameters=None)
            for k, v in get_peft_model_state_dict(model).items():torch.testing.assert_close(v, expected[k], rtol=0, atol=0)
            frozen = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=.01)
            for y in (torch.tensor([[7, 8, 1]]), torch.tensor([[9, 10, 1]])):
                opt.zero_grad(set_to_none=True)
                model(input_ids=torch.tensor([[4,5,1]]), labels=y).loss.backward()
                self.assertTrue(any(p.grad is not None and p.grad.norm() > 0 for p in params))
                opt.step()
            self.assertTrue(any(not torch.equal(v, expected[k]) for k,v in get_peft_model_state_dict(model).items()))
            self.assertEqual(list(model.active_adapters), ['default'])
            for n,p in model.named_parameters():
                if n in frozen:
                    self.assertIsNone(p.grad)
                    torch.testing.assert_close(p, frozen[n], rtol=0, atol=0)
            inference = load_shared_adapter(base(), Path(directory), trainable=False, expected_parameters=None)
            self.assertFalse(any(p.requires_grad for p in inference.parameters()))


if __name__ == '__main__':unittest.main()
