"""Small factual repair objective and baseline-inclusive checkpoint selection."""
import torch
from replycoach.generate.sealion_dual_v1 import reply_loss


def contrast_loss(positive, negative, margin=0.2):
    """Reply-token-normalized NLL ranking; lower positive NLL is better."""
    return positive + 0.25 * torch.relu(margin + positive - negative)


def eligible(replay_ce, initial_replay_ce):
    return replay_ce <= 1.02 * initial_replay_ce


def selection_score(replay_ce, positive_ce):
    return 0.8 * replay_ce + 0.2 * positive_ce


def encoded_pair(row, tokenizer, stage):
    from replycoach.generate.sealion_data import encode_reply
    from replycoach.generate.sealion_rewrite_v1 import encode_rewrite
    item = dict(row, style="STANDARD")
    if stage == "rewrite":
        item["message"] = row["rewrite_message"]
        encoder = encode_rewrite
    elif stage == "english":
        encoder = encode_reply
    else:
        raise ValueError(stage)
    return tuple(encoder(dict(item, response=row[stage][key]), tokenizer, 384)
                 for key in ("positive", "negative"))


def example_loss(model, example, device="cuda"):
    data = {k: torch.tensor([example[k]], device=device)
            for k in ("input_ids", "attention_mask", "labels")}
    labels = data.pop("labels")
    return reply_loss(model(**data, use_cache=False).logits, labels)
