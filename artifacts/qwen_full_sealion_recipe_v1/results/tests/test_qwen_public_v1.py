import io
import unittest

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import Qwen2Config, Qwen2ForCausalLM

from replycoach.generate.qwen_public_v1 import resume_state, restore_state


class PublicResumeTests(unittest.TestCase):
    def test_resume_reproduces_next_dropout_and_adam_update(self):
        def make():
            torch.manual_seed(425)
            model = Qwen2ForCausalLM(Qwen2Config(vocab_size=48, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                head_dim=8, max_position_embeddings=32, attn_implementation='eager'))
            model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=2, lora_alpha=4,
                lora_dropout=.2, target_modules=['q_proj', 'v_proj']))
            return model, torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        tokens = torch.tensor([[3, 4, 5, 6]])
        def update(model, optimizer):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=tokens, labels=tokens).loss
            loss.backward()
            optimizer.step()
            return float(loss.detach())
        first, optimizer = make()
        frozen = {k: v.detach().clone() for k, v in first.named_parameters() if 'lora_' not in k}
        update(first, optimizer)
        buffer = io.BytesIO()
        torch.save(resume_state(first, optimizer, dict(examples_seen=16)), buffer)
        expected_loss = update(first, optimizer)
        expected = {k: v.detach().clone() for k, v in get_peft_model_state_dict(first).items()}
        resumed, resumed_optimizer = make()
        buffer.seek(0)
        report = restore_state(resumed, resumed_optimizer, torch.load(buffer, weights_only=False))
        self.assertEqual(report['examples_seen'], 16)
        self.assertEqual(update(resumed, resumed_optimizer), expected_loss)
        for k, v in get_peft_model_state_dict(resumed).items():
            self.assertTrue(torch.equal(v, expected[k]), k)
        for k, v in resumed.named_parameters():
            if 'lora_' not in k:
                self.assertTrue(torch.equal(v, frozen[k]), k)


if __name__ == '__main__':
    unittest.main()
