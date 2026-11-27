import unittest

from replycoach.generate.decoder_compare_v1 import encode, lr_multiplier, validate_groups


class Tokenizer:
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 2 for c in text]


def row(ident, message='done?', local='done already'):
    return dict(item_id=ident, context=[], speakers=[], message=message,
                standard='It is done.', local=local)


class DecoderComparisonTests(unittest.TestCase):
    def test_only_reply_and_eos_are_supervised(self):
        r = row('train')
        for stage, style in [('task', 'STANDARD'), ('task', 'LOCAL'), ('rewrite', None)]:
            e = encode(r, Tokenizer(), stage, style, 1000)
            n = e['prompt_tokens']
            self.assertEqual(e['labels'][:n], [-100] * n)
            self.assertEqual(e['labels'][n:], e['input_ids'][n:])
            self.assertEqual(e['labels'][-1], 1)
        with self.assertRaises(ValueError):
            encode(r, Tokenizer(), 'task', 'LOCAL', 3)

    def test_schedule_uses_optimizer_updates(self):
        values = [lr_multiplier(i, 20, 2) for i in range(1, 21)]
        self.assertEqual(values[:2], [.5, 1.])
        self.assertEqual(values[-1], 0.)
        self.assertTrue(all(a >= b for a, b in zip(values[1:], values[2:])))
        with self.assertRaises(ValueError):
            lr_multiplier(0, 20, 2)

    def test_detects_leakage_even_with_different_ids(self):
        train = [row('t')]
        old = [row('v', 'coming?')]
        development = [row('d', 'where?')]
        evaluation = [row('v', 'coming?'), row('e', 'when?')]
        validate_groups(train, train, old, development, evaluation, evaluation[1:])
        with self.assertRaises(ValueError):
            validate_groups(train, train, old, development, [row('e')], [])
        with self.assertRaises(ValueError):
            validate_groups(train, [row('t', local='different')], old,
                            development, evaluation, [])


if __name__ == '__main__':
    unittest.main()
