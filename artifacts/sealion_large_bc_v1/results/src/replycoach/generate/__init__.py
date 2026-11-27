"""Tiny word-level vocabulary for the V3/V4 RNN."""

from __future__ import annotations

from collections import Counter

from replycoach.tokenize import tokenize

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIAL = [PAD, UNK, BOS, EOS]


class Vocab:
    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {idx: token for token, idx in token_to_id.items()}

    @classmethod
    def build(cls, texts: list[str], min_freq: int = 1, max_size: int = 4000) -> "Vocab":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tokenize(text))
        token_to_id = {token: idx for idx, token in enumerate(SPECIAL)}
        for token, freq in counts.most_common():
            if freq < min_freq:
                continue
            if token in token_to_id:
                continue
            token_to_id[token] = len(token_to_id)
            if len(token_to_id) >= max_size:
                break
        return cls(token_to_id)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = [self.token_to_id[BOS]] if add_bos else []
        unk = self.token_to_id[UNK]
        ids.extend(self.token_to_id.get(token, unk) for token in tokenize(text))
        if add_eos:
            ids.append(self.token_to_id[EOS])
        return ids

    def decode(self, ids: list[int]) -> str:
        skip = {self.token_to_id[PAD], self.token_to_id[BOS], self.token_to_id[EOS]}
        tokens = []
        for idx in ids:
            if idx in skip:
                continue
            token = self.id_to_token.get(idx, UNK)
            if token == UNK:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def __len__(self) -> int:
        return len(self.token_to_id)

    def to_dict(self) -> dict[str, int]:
        return dict(self.token_to_id)

    @classmethod
    def from_dict(cls, token_to_id: dict[str, int]) -> "Vocab":
        return cls(token_to_id)
