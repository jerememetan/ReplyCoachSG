"""Small tensor checks; actual checkpoint replay is a separate GPU gate."""
import unittest
import torch
from torch.nn import functional as F
from replycoach.generate.mt5_soft_kd_v1 import (
    causal_response_positions, forward_kl, FrozenHead, DualSpaceCMA, mt5_decoder_head,
)


class SoftKDTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(425)

    def test_causal_shift_and_eos(self):
        combined, targets, positions = causal_response_positions([8, 9], [3, 4, 5], 1)
        self.assertEqual(combined[positions], [9, 3, 4, 5])
        self.assertEqual(targets, [3, 4, 5, 1])
        self.assertEqual(combined[positions.stop], 1)
        with self.assertRaises(ValueError):
            causal_response_positions([8], [3, 1], 1)
        with self.assertRaises(ValueError):
            causal_response_positions([8], [3, 4], 1, limit=3)

    def test_forward_kl_direction_temperature_and_detachment(self):
        student = torch.tensor([[[2., -1., .5]]], requires_grad=True)
        teacher = torch.tensor([[[-2., 3., 1.]]], requires_grad=True)
        mask = torch.tensor([[True]])
        for temp in (1., 2., 4.):
            expected = F.kl_div(F.log_softmax(student / temp, -1),
                                F.softmax(teacher / temp, -1), reduction='sum') * temp**2
            actual = forward_kl(student, teacher, mask, temp)
            torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertGreater(student.grad.norm().item(), 0)
        self.assertIsNone(teacher.grad)
        self.assertLess(abs(forward_kl(student, student, mask, 2.).item()), 1e-6)
        empty = forward_kl(student, teacher, ~mask, 2.)
        self.assertEqual(empty.item(), 0)
        self.assertTrue(empty.requires_grad)

    def test_native_head_scaling_and_frozen_weights(self):
        weight = torch.randn(11, 4, requires_grad=True)
        bias = torch.randn(11, requires_grad=True)
        hidden = torch.randn(2, 3, 4, requires_grad=True)
        head = FrozenHead(weight, bias, pre_scale=.125, post_scale=2.)
        expected = F.linear(hidden * .125, weight.detach(), bias.detach()) * 2
        torch.testing.assert_close(head(hidden), expected)
        head(hidden).square().mean().backward()
        self.assertIsNone(weight.grad)
        self.assertIsNone(bias.grad)
        self.assertGreater(hidden.grad.norm().item(), 0)

    def test_actual_mt5_forward_with_legacy_tie_metadata(self):
        from transformers import MT5Config
        from replycoach.generate.mt5_pretrained_compat import OriginalMT5
        cfg = MT5Config(vocab_size=23, d_model=8, d_kv=4, d_ff=16,
                        num_layers=1, num_decoder_layers=1, num_heads=2,
                        dropout_rate=0., pad_token_id=0, eos_token_id=1,
                        decoder_start_token_id=0, tie_word_embeddings=True)
        model = OriginalMT5(cfg).eval().requires_grad_(False)
        x, y = torch.tensor([[5, 6, 1]]), torch.tensor([[7, 8, 1]])
        with torch.no_grad():
            out = model(input_ids=x, labels=y, output_hidden_states=True, use_cache=False)
        replay = mt5_decoder_head(model)(out.decoder_hidden_states[-1])
        torch.testing.assert_close(replay, out.logits)
        # This is the erroneous legacy-config scaling caught by GPU gate373424.
        self.assertFalse(torch.allclose(replay * cfg.d_model ** -.5, out.logits))

    def test_different_vocab_lengths_padding_and_gradients(self):
        kd = DualSpaceCMA(4, 6)
        student_head = FrozenHead(torch.randn(9, 4))
        teacher_head = FrozenHead(torch.randn(13, 6))
        hidden = torch.randn(2, 4, 4, requires_grad=True)
        index = torch.randn(2, 4, 8, requires_grad=True)
        th = torch.randn(2, 6, 6, requires_grad=True)
        ti = torch.randn(2, 6, 12, requires_grad=True)
        te = torch.randn(2, 6, 6, requires_grad=True)
        labels = torch.tensor([[2, 3, 1, -100], [4, 1, -100, -100]])
        tlabels = torch.tensor([[3, 4, 5, 6, 1, -100], [5, 6, 1, -100, -100, -100]])
        def run(h, i, t, ix, e):
            return kd(student_head(h), h, i, labels, t, ix, e, tlabels, student_head, teacher_head)
        loss, metrics = run(hidden, index, th, ti, te)
        changed = []
        for value, mask in ((hidden, labels.eq(-100)), (index, labels.eq(-100)),
                            (th, tlabels.eq(-100)), (ti, tlabels.eq(-100)), (te, tlabels.eq(-100))):
            clone = value.detach().clone()
            clone[mask] = float('nan')
            changed.append(clone)
        loss2, metrics2 = run(*changed)
        torch.testing.assert_close(loss, loss2)
        for name in metrics:
            torch.testing.assert_close(metrics[name], metrics2[name])
        loss.backward()
        self.assertGreater(hidden.grad.norm().item(), 0)
        self.assertEqual(hidden.grad[labels.eq(-100)].abs().sum().item(), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
                            for p in kd.parameters()))
        for frozen in (index, th, ti, te):
            self.assertIsNone(frozen.grad)
        # Removing the real teacher distribution changes the objective: this is
        # a distributional KL term, not only teacher-text CE.
        modified = th.detach().clone() * -2
        self.assertGreater(abs(run(hidden.detach(), index, modified, ti, te)[1]['s2t_kl'].item()
                               - metrics['s2t_kl'].item()), 1e-4)


if __name__ == '__main__':
    unittest.main()
