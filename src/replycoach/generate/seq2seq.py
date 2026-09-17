"""V3 encoder-decoder LSTM and V4 attention decoder.

V3 is the reliable CPU baseline. V4 adds Luong dot-product attention on the
same encoder. V5 (mT5) lives in notebooks and is compared, not stacked.
"""

from __future__ import annotations

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_size: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, hidden_size, batch_first=True)

    def forward(self, src: torch.Tensor, lengths: torch.Tensor):
        embedded = self.embed(src)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden


class Decoder(nn.Module):
    """V3 decoder: no attention."""

    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_size: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, token: torch.Tensor, hidden):
        embedded = self.embed(token).unsqueeze(1)
        output, hidden = self.rnn(embedded, hidden)
        logits = self.out(output.squeeze(1))
        return logits, hidden, None


class AttentionDecoder(nn.Module):
    """V4 decoder: Luong dot-product attention over encoder states."""

    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_size: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, token: torch.Tensor, hidden, encoder_outputs: torch.Tensor):
        embedded = self.embed(token).unsqueeze(1)
        output, hidden = self.rnn(embedded, hidden)
        query = output
        scores = torch.bmm(query, encoder_outputs.transpose(1, 2))
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights, encoder_outputs)
        combined = torch.tanh(self.attn_combine(torch.cat([output, context], dim=-1)))
        logits = self.out(combined.squeeze(1))
        return logits, hidden, weights


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_size: int = 128, use_attention: bool = False):
        super().__init__()
        self.use_attention = use_attention
        self.encoder = Encoder(vocab_size, embed_dim, hidden_size)
        if use_attention:
            self.decoder = AttentionDecoder(vocab_size, embed_dim, hidden_size)
        else:
            self.decoder = Decoder(vocab_size, embed_dim, hidden_size)

    def forward(self, src, src_lengths, tgt, teacher_forcing: bool = True):
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        bos = tgt[:, 0]
        inputs = bos
        logits_steps = []
        for t in range(1, tgt.size(1)):
            if self.use_attention:
                step_logits, hidden, _ = self.decoder(inputs, hidden, encoder_outputs)
            else:
                step_logits, hidden, _ = self.decoder(inputs, hidden)
            logits_steps.append(step_logits)
            inputs = tgt[:, t] if teacher_forcing else step_logits.argmax(dim=-1)
        return torch.stack(logits_steps, dim=1)
