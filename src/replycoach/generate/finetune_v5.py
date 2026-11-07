"""Train V5 stage 2: fine-tune the stage 1 mT5 on team-authored development items with LoRA.

Each item gives two pairs in the ``mt5_input`` format stage 1 used:
``STANDARD context: ... message: ...`` -> ``replies.standard`` and ``LOCAL ...`` -> ``replies.local``.
One adapter learns both styles and the tag chooses between them. Only development items are read:
data/dev/train.jsonl trains, data/dev/validation.jsonl picks the epoch. The test set never enters.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from peft import AutoPeftModelForSeq2SeqLM, LoraConfig, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from replycoach.generate.train_v5 import (
    ARTIFACTS, INPUT_FORMAT, collate, encode_rows, epoch_batches, generate_replies, load_pairs, validation_loss,
)
from replycoach.generate.decode_v5 import echo_start
from replycoach.generate.style_control import JOINT_SEPARATOR, JOINT_STYLE, joint_reply, split_joint
from replycoach.io import DATA_DIR, load_items

STYLES = ("STANDARD", "LOCAL")
DEV_TRAIN = DATA_DIR / "dev" / "train.jsonl"
DEV_VALIDATION = DATA_DIR / "dev" / "validation.jsonl"


def item_rows(path: Path) -> list[dict]:
    """Two rows per development item, one per style, sharing the item's context, its speakers and the message."""
    items = load_items(path)
    not_dev = [item.id for item in items if item.split != "dev"]
    if not_dev:
        raise ValueError(f"{path}: only development items may be used, found {', '.join(not_dev[:3])}")
    if not items:
        raise ValueError(f"no development items in {path}")
    return [
        {"item_id": item.id, "style": style, "context": item.memory_texts(),
         "speakers": [turn.speaker for turn in item.session_memory], "message": item.message.text,
         "response": response}
        for item in items
        for style, response in (("STANDARD", item.replies.standard), ("LOCAL", item.replies.local))
    ]


def joint_rows(rows: list[dict]) -> list[dict]:
    """One row per item whose reply is both styles at once, STANDARD first (``joint_reply``).

    Written in one go, the LOCAL half is generated while the model sees the STANDARD half it just wrote, so the
    two say the same thing; generated separately, each style is its own guess and they can contradict.
    """
    local = {row["item_id"]: row["response"] for row in rows if row["style"] == "LOCAL"}
    return [{**row, "style": JOINT_STYLE, "response": joint_reply(row["response"], local[row["item_id"]])}
            for row in rows if row["style"] == "STANDARD"]


def context_pairs(path: Path) -> list[dict]:
    """Replies already inside the items' contexts: each user turn that answers a peer turn becomes a pair.

    The team wrote these turns as part of the conversation, not as STANDARD or LOCAL versions, so they get the
    ``REPLY`` tag: they teach the model to answer the other person without changing what the style tags mean.
    A user turn right after another user turn (a double text) is skipped, since it answers no one.
    """
    rows = []
    for item in load_items(path):
        turns = item.session_memory
        for index in range(1, len(turns)):
            if turns[index].speaker == "user" and turns[index - 1].speaker != "user":
                rows.append({"item_id": item.id, "style": "REPLY", "message": turns[index - 1].text,
                             "context": [turn.text for turn in turns[: index - 1]],
                             "speakers": [turn.speaker for turn in turns[: index - 1]],
                             "response": turns[index].text})
    return rows


def shorter_contexts(row: dict) -> list[dict]:
    """Copies of a row with less context: the last turn only, then none.

    The app's M0/M1/M2 memory modes pass different amounts of context, and a model that also sees the bare
    message has to answer the message instead of echoing the context. The reply stays the team's own text.
    """
    cuts = [len(row["context"]) - keep for keep in (1, 0) if keep < len(row["context"])]
    return [{**row, "context": row["context"][cut:], "speakers": row["speakers"][cut:]} for cut in cuts]


def echo_negatives(candidates_path: Path, train_rows: list[dict], per_row: int = 3) -> tuple[list[dict], int]:
    """Replies the model should not give: its own sampled replies to training items that echo the friend.

    ``candidates_path`` is a ``rerank_v5 candidates`` file made on the training items. Each candidate that
    ``echo_start`` flags becomes a row with the item's input, the bad reply and ``penalise_from``, the character
    where the echo starts. Up to ``per_row`` per item and style. When the rule also flags the team's own reply
    for that item and style, the rule is not trusted there and the row is skipped; the count is returned.
    """
    report = json.loads(Path(candidates_path).read_text(encoding="utf-8"))
    rows = {(row["item_id"], row["style"]): row for row in train_rows}
    # Negatives from validation items would leak them into training.
    outside = [item["item_id"] for item in report["items"] if (item["item_id"], "STANDARD") not in rows]
    if outside:
        raise ValueError(f"{candidates_path} has items outside the training file: {', '.join(outside[:3])}")
    negatives, skipped = [], 0
    for item in report["items"]:
        for style in STYLES:
            row = rows[(item["item_id"], style)]
            if echo_start(row["response"], row["message"]) is not None:
                skipped += 1
                continue
            found = []
            for candidate in item["candidates"][style]:
                start = echo_start(candidate["text"], row["message"])
                if start is not None and candidate["text"].strip().lower() != row["response"].strip().lower():
                    found.append({**row, "response": candidate["text"], "penalise_from": start})
            negatives += found[:per_row]
    return negatives, skipped


def encode_negatives(rows: list[dict], tokenizer, max_source: int, max_target: int,
                     speaker_labels: bool = False) -> list[tuple[list[int], list[int], int]]:
    """(source ids, reply ids, first reply token to penalise) for each negative.

    Words before the echoing "I", such as "Okay,", are fine in a good reply too, so only the echo itself and
    what follows are pushed down. Negatives whose echo was cut off by ``max_target`` are dropped.
    """
    if not rows:
        return []
    encoded = []
    for (source, target), row in zip(encode_rows(rows, tokenizer, max_source, max_target,
                                                 speaker_labels=speaker_labels), rows):
        prefix = row["response"][: row["penalise_from"]].rstrip()
        skip = len(tokenizer(prefix, add_special_tokens=False)["input_ids"]) if prefix else 0
        if skip < len(target) - 1:  # the last token is the end-of-reply marker, which is never penalised
            encoded.append((source, target, skip))
    return encoded


def unlikelihood_loss(model, batch: list[tuple[list[int], list[int], int]], pad_id: int, device):
    """Mean of -log(1 - p) over the penalised tokens of the bad replies (Li et al., 2020, "Don't Say That!").

    p is the probability the model gives each token of the bad reply, with the reply so far fed in as usual.
    The penalty is near 0 when p is small and grows without limit as p nears 1, so minimising it pushes those
    tokens down, while the ordinary loss on the team's replies keeps pushing the good tokens up.
    """
    input_ids, attention_mask, labels = collate([(source, target) for source, target, _ in batch], pad_id)
    penalise = torch.zeros_like(labels, dtype=torch.bool)
    for index, (_, target, skip) in enumerate(batch):
        penalise[index, skip : len(target) - 1] = True
    labels = labels.to(device)
    logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels).logits
    probability = logits.softmax(-1).gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return -torch.log((1 - probability).clamp(min=1e-5))[penalise.to(device)].mean()


def preference_pairs(sheet_path: Path, train_rows: list[dict]) -> tuple[list[tuple[dict, dict]], int, int]:
    """(better, worse) rows from a person's picks in a ``compare_v5 make-prefs`` sheet.

    Each question answered 1 or 2 gives the item's input twice: with the picked reply, and with the other one.
    A question answered "both bad" gives the team's own reply as better than each of the two, so the replies
    the person rejected outright are pushed down too. Questions answered "same" give no pair.
    Returns the pairs, the count of "same" answers, and how many pairs came from "both bad".
    """
    sheet = json.loads(Path(sheet_path).read_text(encoding="utf-8"))
    critic = sheet.get("training_critic")
    if critic is not None and (critic.get("status") != "complete" or
                               critic.get("purpose") != "training_preferences"):
        raise ValueError("critic preference run must be complete and designated for training")
    if critic is not None and critic.get("prompt_version") != "selectable-v5-user-reviewed":
        raise ValueError("unsupported critic prompt version; regenerate labels under the selectable-draft rubric")
    rows = {(row["item_id"], row["style"]): row for row in train_rows}
    # Pairs from validation items would leak them into training.
    outside = [question["item_id"] for question in sheet["questions"]
               if (question["item_id"], question["style"]) not in rows]
    if outside:
        raise ValueError(f"{sheet_path} has items outside the training file: {', '.join(outside[:3])}")
    pairs, unused, from_both_bad = [], 0, 0
    for question in sheet["questions"]:
        row = rows[(question["item_id"], question["style"])]
        if question["answer"] == "both_bad":
            bad = [reply for reply in question["replies"] if reply.strip() != row["response"].strip()]
            pairs += [(row, {**row, "response": reply}) for reply in bad]
            from_both_bad += len(bad)
            continue
        if question["answer"] not in ("first", "second"):
            unused += question["answer"] is not None
            continue
        better, worse = question["replies"] if question["answer"] == "first" else question["replies"][::-1]
        pairs.append(({**row, "response": better}, {**row, "response": worse}))
    return pairs, unused, from_both_bad


def sequence_logprobs(model, encoded: list[tuple[list[int], list[int]]], pad_id: int, device):
    """Total log-probability of each reply given its input: how likely the model is to write exactly that reply."""
    input_ids, attention_mask, labels = collate(encoded, pad_id)
    labels = labels.to(device)
    logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels).logits
    mask = labels.ne(-100)
    token_logp = logits.log_softmax(-1).gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(-1)


def dpo_loss(model, batch: list[tuple], reference, pad_id: int, device, beta: float):
    """Direct Preference Optimization (Rafailov et al., 2023) on a batch of (better ids, worse ids) pairs.

    ``reference`` holds each pair's two log-probabilities under the stage 1 model before training. Training
    pushes the better reply up and the worse one down, measured against where each started:

        loss = -log sigmoid(beta * ((better - better_at_start) - (worse - worse_at_start)))

    So a pair stops pulling once the model clearly prefers the better reply, and nothing is pushed without limit.
    Also returns the share of pairs where the model has moved towards the better reply.
    """
    logprobs = sequence_logprobs(model, [better for better, _ in batch] + [worse for _, worse in batch], pad_id, device)
    better, worse = logprobs[: len(batch)], logprobs[len(batch) :]
    reference = reference.to(device)
    margin = beta * ((better - reference[:, 0]) - (worse - reference[:, 1]))
    return -torch.nn.functional.logsigmoid(margin).mean(), float((margin > 0).float().mean())


def input_format(checkpoint: Path) -> dict:
    """How ``checkpoint`` expects its input; checkpoints saved before speaker labels have no file."""
    path = Path(checkpoint) / INPUT_FORMAT
    return {"speaker_labels": False, "joint": False,
            **(json.loads(path.read_text(encoding="utf-8")) if path.exists() else {})}


def load_finetuned(checkpoint: Path, device="cpu"):
    """A LoRA checkpoint holds only the adapter and names its base model; a full fine-tune holds everything."""
    if (checkpoint / "adapter_config.json").exists():
        return AutoPeftModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
    return AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)


def finetune(
    base_model: str,
    train_path: Path = DEV_TRAIN,
    validation_path: Path = DEV_VALIDATION,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    lora_targets: tuple[str, ...] = ("q", "v"),
    epochs: int = 30,
    patience: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    max_source: int = 128,
    max_target: int = 48,
    output_name: str = "mt5_v5_lora",
    output_dir: Path = ARTIFACTS,
    seed: int = 425,
    context_variants: bool = False,
    use_context_pairs: bool = False,
    rehearsal_paths: list[Path] = (),
    rehearsal_count: int = 400,
    speaker_labels: bool = False,
    negatives_path: Path | None = None,
    unlikelihood_weight: float = 1.0,
    negatives_per_row: int = 3,
    preferences_path: Path | None = None,
    dpo_weight: float = 1.0,
    dpo_beta: float = 0.1,
    joint: bool = False,
    synthetic_path: Path | None = None,
    synthetic_count: int = 190,
    max_steps: int | None = None,
    teacher_path: Path | None = None,
    teacher_mode: str = "replace",
    rewrite_local: bool = False,
    save_final: bool = False,
    standard_repeats: int = 1,
    save_steps: tuple[int, ...] = (),
) -> dict:
    """Fine-tune ``base_model`` (a stage 1 checkpoint). ``lora_rank=0`` trains every weight instead of an adapter.

    Augmentation touches training only, so every run is scored on the same unchanged validation pairs:
    ``context_variants`` adds shorter-context copies of each item, and ``rehearsal_paths`` mixes in
    ``rehearsal_count`` stage 1 public pairs so the model keeps its general reply ability.
    ``use_context_pairs`` adds the replies already inside the training items' contexts (``context_pairs``).
    ``speaker_labels`` marks each turn with the item's real speaker, ``Friend:`` or ``You:`` (see ``mt5_input``).
    ``negatives_path`` adds unlikelihood training: each step also pushes down a batch of echo replies from
    ``echo_negatives``, weighted by ``unlikelihood_weight``. Validation loss stays the ordinary loss on the
    team's replies, so it can rise a little; count echoes in the replies to judge it.
    ``preferences_path`` adds DPO: each step also trains on a batch of a person's (better, worse) picks from
    ``preference_pairs``, weighted by ``dpo_weight``; the ordinary loss keeps the team's replies as the anchor.
    ``joint`` trains each item as one reply holding both styles (``joint_rows``), with ``max_target`` tokens for
    each half. Its validation loss also covers both halves, so compare it with separate-style runs by reading or
    rating the replies, not by the loss.
    """
    if epochs < 1 or patience < 1 or batch_size < 1 or lora_rank < 0 or learning_rate <= 0 or rehearsal_count < 0:
        raise ValueError("epochs, patience and batch size must be positive, rank and rehearsal count "
                         "non-negative, learning rate positive")
    if not output_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("output name may contain only letters, digits, underscores, and hyphens")
    run_dir = output_dir / output_name
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("max_steps must be a positive integer")
    if any(type(step) is not int or step < 1 for step in save_steps) or len(set(save_steps)) != len(save_steps):
        raise ValueError('save_steps must contain unique positive integers')
    if save_steps and (max_steps is None or max(save_steps) > max_steps):
        raise ValueError('save_steps requires a max_steps budget covering every requested step')
    if any((run_dir / f'step_{step}').exists() for step in save_steps):
        raise ValueError('Requested step checkpoint already exists; choose a fresh output name')
    best_dir = run_dir / "best"
    # Runs take minutes, so there is no resume: a reused name would mix two runs' files.
    if best_dir.exists():
        raise ValueError(f"{best_dir} already exists; choose a new --output-name")
    if negatives_path is not None and (unlikelihood_weight <= 0 or negatives_per_row < 1):
        raise ValueError("unlikelihood weight and negatives per row must be positive")
    if preferences_path is not None and (dpo_weight <= 0 or dpo_beta <= 0):
        raise ValueError("DPO weight and beta must be positive")
    if joint and (negatives_path or preferences_path):
        # Both are made of single-style replies, which a joint model is never asked for.
        raise ValueError("joint training does not take --negatives or --preferences yet")
    if rewrite_local and (joint or negatives_path or preferences_path or synthetic_path or context_variants
                          or (teacher_path is not None and teacher_mode == 'augment')):
        raise ValueError('Rewrite experiment requires one paired target per input and no other auxiliary objectives')
    if input_format(base_model)["speaker_labels"] and not speaker_labels:
        raise ValueError(f"{base_model} was trained with speaker labels; add --speaker-labels")
    torch.manual_seed(seed)

    if standard_repeats != 1 and not rewrite_local:
        raise ValueError('STANDARD repeats requires sequential rewrite training')
    from replycoach.generate.rewrite_v5 import emphasize_standard
    emphasize_standard([], standard_repeats)
    train_rows = item_rows(train_path)
    validation_rows = item_rows(validation_path)
    overlap = {row["item_id"] for row in train_rows} & {row["item_id"] for row in validation_rows}
    if overlap:
        raise ValueError(f"items in both training and validation: {', '.join(sorted(overlap)[:3])}")
    teacher_report = None
    if teacher_mode not in ('replace', 'augment'):
        raise ValueError('teacher_mode must be replace or augment')
    if teacher_path is not None:
        if synthetic_path or preferences_path or negatives_path or context_variants or joint:
            raise ValueError('Teacher experiment must isolate target changes; other auxiliary objectives are disabled')
        from replycoach.generate.teacher_v5 import load_teacher_targets
        teacher_rows, teacher_report = load_teacher_targets(teacher_path, train_rows, train_path)
        teacher_report['mode'] = teacher_mode
        train_rows = teacher_rows if teacher_mode == 'replace' else [*train_rows, *teacher_rows]
    synthetic = []
    synthetic_report = None
    if synthetic_path is not None:
        from replycoach.generate.synthetic_v5 import load_synthetic, manifest
        if synthetic_count < 1:
            raise ValueError("synthetic count must be positive")
        available = load_synthetic(synthetic_path, [*train_rows, *validation_rows])
        # Sample items, never individual styles, to preserve paired supervision.
        pairs = [available[i:i + 2] for i in range(0, len(available), 2)]
        selected = random.Random(seed).sample(pairs, min(synthetic_count, len(pairs)))
        synthetic = [row for pair in selected for row in pair]
        synthetic_report = {**manifest(synthetic_path, len(pairs)), "selected_items": len(selected),
                            "selected_ids": [pair[0]["item_id"] for pair in selected]}
        if joint:
            synthetic = joint_rows(synthetic)
    if joint:
        train_rows = joint_rows(train_rows)
    if rewrite_local:
        from replycoach.generate.rewrite_v5 import rewrite_rows
        train_rows = rewrite_rows(train_rows)
    item_pairs = len(train_rows)
    standard_extra = emphasize_standard([r for r in train_rows if r['style'] == 'STANDARD'], standard_repeats - 1) if standard_repeats > 1 else []
    target_limit = 2 * max_target if joint else max_target
    variants = [variant for row in train_rows for variant in shorter_contexts(row)] if context_variants else []
    # Read from the training file only, so no validation conversation is ever trained on.
    inner = context_pairs(train_path) if use_context_pairs else []
    public = [row for path in rehearsal_paths for row in load_pairs(path)]
    rehearsal = random.Random(seed).sample(public, min(rehearsal_count, len(public)))
    negatives, skipped = echo_negatives(negatives_path, train_rows, negatives_per_row) if negatives_path else ([], 0)
    preferences, unused_answers, from_both_bad = (preference_pairs(preferences_path, train_rows) if preferences_path
                                                  else ([], 0, 0))
    if preferences_path and not preferences:
        raise ValueError(f"no questions answered 1, 2 or both bad in {preferences_path}")
    train_rows = [*train_rows, *variants, *inner, *rehearsal, *synthetic, *standard_extra]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    pad_id = tokenizer.pad_token_id
    train_encoded = encode_rows(train_rows, tokenizer, max_source, target_limit, speaker_labels=speaker_labels)
    negative_encoded = encode_negatives(negatives, tokenizer, max_source, max_target, speaker_labels)
    if negatives_path and not negative_encoded:
        raise ValueError(f"no echo replies found in {negatives_path}")
    preference_encoded = list(zip(
        *(encode_rows([pair[side] for pair in preferences], tokenizer, max_source, max_target,
                      speaker_labels=speaker_labels) for side in (0, 1))
    )) if preferences else []
    # Validated per style, so the report shows if one style lags behind the other.
    loss_rows = validation_rows
    if rewrite_local:
        loss_rows = rewrite_rows(validation_rows)
    validation_encoded = {
        style: encode_rows([row for row in loss_rows if row["style"] ==
                           ('LOCAL_REWRITE' if rewrite_local and style == 'LOCAL' else style)], tokenizer, max_source, max_target,
                           speaker_labels=speaker_labels)
        for style in STYLES
    } if not joint else {JOINT_STYLE: encode_rows(joint_rows(validation_rows), tokenizer, max_source, target_limit,
                                                  speaker_labels=speaker_labels)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Full precision, as in stage 1: mT5 overflows under fp16.
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def evaluate() -> dict[str, float]:
        return {style: validation_loss(model, encoded, 2 * batch_size, pad_id, device)
                for style, encoded in validation_encoded.items()}

    # The stage 1 model alone on the same items: the number fine-tuning has to beat.
    before = evaluate()
    # DPO measures every change against the stage 1 model, so its view of each preference pair is taken now.
    reference = torch.zeros(len(preference_encoded), 2)
    with torch.no_grad():
        for start in range(0, len(preference_encoded), batch_size):
            chunk = preference_encoded[start : start + batch_size]
            for side in (0, 1):
                reference[start : start + len(chunk), side] = sequence_logprobs(
                    model, [pair[side] for pair in chunk], pad_id, device).cpu()

    if lora_rank > 0:
        # Freeze every stage 1 weight and add trainable rank-r matrices beside the chosen attention projections.
        # B starts at zero, so the wrapped model starts out identical to the stage 1 model.
        config = LoraConfig(
            task_type="SEQ_2_SEQ_LM", r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=list(lora_targets),
        )
        model = get_peft_model(model, config).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)

    report_path = output_dir / f"{output_name}.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "version": "V5", "stage": "finetune", "method": "lora" if lora_rank > 0 else "full",
        "base_model": str(base_model), "train_path": str(train_path), "validation_path": str(validation_path),
        "train_items": len({row["item_id"] for row in train_rows[:item_pairs]}), "train_pairs": len(train_encoded),
        "train_pairs_by_kind": {"items": item_pairs, "context_variants": len(variants), "context_pairs": len(inner),
                                "rehearsal": len(rehearsal)},
        "rehearsal_paths": [str(path) for path in rehearsal_paths],
        "synthetic": synthetic_report,
        "teacher": teacher_report,
        "rewrite_local": rewrite_local,
        "standard_repeats": standard_repeats,
        "standard_extra_rows": len(standard_extra),
        "local_validation_conditioning": "authored reference standard (oracle; not independent LOCAL loss)" if rewrite_local else "conversation only",
        "validation_items": len(validation_rows) // 2, "seed": seed, "speaker_labels": speaker_labels,
        "joint": joint,
        "unlikelihood": {
            "candidates": str(negatives_path), "weight": unlikelihood_weight, "per_row": negatives_per_row,
            "negatives": len(negative_encoded), "items": len({row["item_id"] for row in negatives}),
            "rows_skipped_reference_echoes": skipped,
        } if negatives_path else None,
        "preferences": {
            "sheet": str(preferences_path), "pairs": len(preference_encoded), "from_both_bad": from_both_bad,
            "unused_answers": unused_answers, "weight": dpo_weight, "beta": dpo_beta,
        } if preferences_path else None,
        "lora": {"rank": lora_rank, "alpha": lora_alpha, "dropout": lora_dropout, "targets": list(lora_targets)}
                if lora_rank > 0 else None,
        "epochs_requested": epochs, "patience": patience, "batch_size": batch_size, "learning_rate": learning_rate,
        "max_source_tokens": max_source, "max_target_tokens": max_target,
        "device": str(device), "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "checkpoint": str(best_dir),
        "history": [{
            "epoch": 0, "validation_loss": round(sum(before.values()) / len(before), 4),
            **{f"validation_loss_{style.lower()}": round(value, 4) for style, value in before.items()},
        }],
    }
    if synthetic_path is not None:
        report["train_pairs_by_kind"]["synthetic"] = len(synthetic)
    print(json.dumps(report["history"][0]), flush=True)

    best_loss, stale = float("inf"), 0
    negative_order = torch.Generator().manual_seed(seed)
    preference_order = torch.Generator().manual_seed(seed)
    started = time.perf_counter()
    optimizer_steps = 0
    report["max_steps"] = max_steps
    report['requested_save_steps'] = sorted(save_steps)
    report['step_checkpoints'] = {}
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, epoch_tokens, epoch_penalty, penalty_steps = 0.0, 0, 0.0, 0
        epoch_dpo, epoch_agreement, dpo_steps = 0.0, 0.0, 0
        for batch in epoch_batches(len(train_encoded), batch_size, seed, epoch):
            input_ids, attention_mask, labels = collate([train_encoded[index] for index in batch], pad_id)
            optimizer.zero_grad()
            likelihood = model(
                input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels.to(device)
            ).loss
            loss = likelihood
            if negative_encoded:
                # A random batch of echo replies alongside every batch of team replies.
                picks = torch.randint(len(negative_encoded), (min(batch_size, len(negative_encoded)),),
                                      generator=negative_order).tolist()
                penalty = unlikelihood_loss(model, [negative_encoded[index] for index in picks], pad_id, device)
                loss = loss + unlikelihood_weight * penalty
                epoch_penalty += float(penalty.item())
                penalty_steps += 1
            if preference_encoded:
                # A random batch of preference pairs alongside every batch of team replies.
                picks = torch.randint(len(preference_encoded), (min(batch_size, len(preference_encoded)),),
                                      generator=preference_order).tolist()
                preference, agreement = dpo_loss(model, [preference_encoded[index] for index in picks],
                                                 reference[picks], pad_id, device, dpo_beta)
                loss = loss + dpo_weight * preference
                epoch_dpo += float(preference.item())
                epoch_agreement += agreement
                dpo_steps += 1
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss is not finite in epoch {epoch}; lower --learning-rate")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer_steps += 1
            if optimizer_steps in save_steps:
                checkpoint = run_dir / f'step_{optimizer_steps}'
                model.save_pretrained(checkpoint)
                tokenizer.save_pretrained(checkpoint)
                (checkpoint / INPUT_FORMAT).write_text(json.dumps({
                    'speaker_labels': speaker_labels, 'joint': joint,
                    **({'rewrite_local': True, 'max_source': max_source, 'max_target': max_target}
                       if rewrite_local else {})}) + '\n', encoding='utf-8')
                report['step_checkpoints'][str(optimizer_steps)] = str(checkpoint)
            token_count = int(labels.ne(-100).sum().item())
            epoch_loss += float(likelihood.item()) * token_count
            epoch_tokens += token_count
            if max_steps is not None and optimizer_steps >= max_steps:
                break

        losses = evaluate()
        # Unweighted mean, so neither style decides the checkpoint alone.
        mean_loss = sum(losses.values()) / len(losses)
        entry = {
            "epoch": epoch, "train_loss": round(epoch_loss / epoch_tokens, 4), "validation_loss": round(mean_loss, 4),
            **{f"validation_loss_{style.lower()}": round(value, 4) for style, value in losses.items()},
            **({"unlikelihood_loss": round(epoch_penalty / penalty_steps, 4)} if penalty_steps else {}),
            **({"dpo_loss": round(epoch_dpo / dpo_steps, 4), "dpo_agreement": round(epoch_agreement / dpo_steps, 3)}
               if dpo_steps else {}),
            "seconds": round(time.perf_counter() - started, 1),
            "optimizer_steps": optimizer_steps,
        }
        report["optimizer_steps"] = optimizer_steps
        if max_steps is not None and optimizer_steps >= max_steps:
            report["stopped_at_step_budget"] = True
        report["history"].append(entry)
        print(json.dumps(entry), flush=True)
        if mean_loss < best_loss:
            best_loss, stale = mean_loss, 0
            report["best_epoch"], report["best_validation_loss"] = epoch, round(mean_loss, 4)
            model.save_pretrained(best_dir)  # a LoRA model saves only its adapter
            tokenizer.save_pretrained(best_dir)
            (best_dir / INPUT_FORMAT).write_text(json.dumps({"speaker_labels": speaker_labels, "joint": joint,
                **({"rewrite_local": True, "max_source": max_source, "max_target": max_target} if rewrite_local else {})}) + "\n",
                                                 encoding="utf-8")
        else:
            stale += 1
        if device.type == "cuda":
            report["peak_cuda_gb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if report.get("stopped_at_step_budget"):
            break
        if stale >= patience:
            # Validation has stopped improving; more epochs would only memorise the training items.
            report["stopped_early"] = True
            break

    if save_final:
        final_dir = run_dir / "final"
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        (final_dir / INPUT_FORMAT).write_text(json.dumps({"speaker_labels": speaker_labels, "joint": joint,
            **({"rewrite_local": True, "max_source": max_source, "max_target": max_target} if rewrite_local else {})}) + "\n", encoding="utf-8")
        report['final_checkpoint'] = str(final_dir)
        report['final_validation_outputs'] = validation_outputs(
            model, tokenizer, validation_rows, max_source, max_target, device,
            speaker_labels=speaker_labels, joint=joint, rewrite_local=rewrite_local)
    # Replies from the best epoch for every validation item, to read and to compare with V4.
    model = load_finetuned(best_dir, device)
    report["validation_outputs"] = validation_outputs(
        model, tokenizer, validation_rows, max_source, max_target, device, speaker_labels=speaker_labels, joint=joint,
        rewrite_local=rewrite_local
    )
    if joint:
        report["joint_unsplit"] = sum(not row["joint_split"] for row in report["validation_outputs"])
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


class NoRepeatWithinHalf(LogitsProcessor):
    """``no_repeat_ngram_size`` for a joint reply, applied to each half on its own.

    No run of ``n`` tokens may repeat inside the STANDARD half or inside the LOCAL half, so neither half can loop
    ("i wait for you. i wait for you. ..."), but the LOCAL half may reuse the STANDARD half's words. The LOCAL
    half starts after the last ``marker``, the token ids of the separator's ``LOCAL:``.
    """

    def __init__(self, n: int, marker: list[int]):
        self.n, self.marker = n, marker

    def __call__(self, input_ids, scores):
        size = len(self.marker)
        for row, ids in enumerate(input_ids.tolist()):
            starts = [index + size for index in range(len(ids) - size + 1) if ids[index : index + size] == self.marker]
            half = ids[max(starts, default=0):]
            # The last n - 1 tokens, and every token that followed the same n - 1 tokens earlier in this half.
            prefix = half[len(half) - self.n + 1:] if self.n > 1 else []
            banned = {half[index + self.n - 1] for index in range(len(half) - self.n + 1)
                      if half[index : index + self.n - 1] == prefix}
            scores[row, list(banned)] = -float("inf")
        return scores


def joint_decoding(tokenizer, decoding: dict | None) -> dict:
    """``decoding`` with its ``no_repeat_ngram_size`` (3 unless set) applied within each half of a joint reply."""
    n = (decoding or {}).get("no_repeat_ngram_size", 3)
    # "| LOCAL:" without its first token, which is "▁|" after a space and "|" straight after a word.
    marker = tokenizer(JOINT_SEPARATOR.strip(), add_special_tokens=False).input_ids[1:]
    processors = LogitsProcessorList([NoRepeatWithinHalf(n, marker)] if n else [])
    return {**(decoding or {}), "no_repeat_ngram_size": 0, "logits_processor": processors}


def validation_outputs(model, tokenizer, validation_rows: list[dict], max_source: int, max_target: int, device,
                       decoding: dict | None = None, speaker_labels: bool = False, joint: bool = False,
                       rewrite_local: bool = False) -> list[dict]:
    """Both styles generated for each validation item, next to the team's two references.

    A ``joint`` model writes both in one reply of up to ``2 * max_target`` tokens, which is then split. Its
    repeated n-grams are blocked within each half (``joint_decoding``), not across the whole reply, which would
    stop the LOCAL half reusing the STANDARD half's words.
    A reply without the separator is used for both styles and marked ``joint_split: False``.
    """
    references = {row["item_id"]: row for row in validation_rows if row["style"] == "STANDARD"}
    local_references = {row["item_id"]: row["response"] for row in validation_rows if row["style"] == "LOCAL"}
    rows = list(references.values())
    if joint and rewrite_local:
        raise ValueError('Joint and sequential rewrite are different generation modes')
    if rewrite_local:
        from replycoach.generate.rewrite_v5 import generate_pair
        generated = generate_pair(model, tokenizer, rows, max_source, max_target, device, decoding, speaker_labels)
    elif joint:
        texts = generate_replies(model, tokenizer, rows, JOINT_STYLE, max_source, 2 * max_target, device,
                                 joint_decoding(tokenizer, decoding), speaker_labels)
        halves = [split_joint(text) or (text.strip(), text.strip()) for text in texts]
        generated = {"STANDARD": [standard for standard, _ in halves], "LOCAL": [local for _, local in halves]}
    else:
        generated = {
            style: generate_replies(model, tokenizer, rows, style, max_source, max_target, device, decoding,
                                    speaker_labels)
            for style in STYLES
        }
    return [
        {
            "item_id": row["item_id"], "context": row["context"], "message": row["message"],
            "reference_standard": row["response"], "generated_standard": generated["STANDARD"][index],
            "reference_local": local_references[row["item_id"]], "generated_local": generated["LOCAL"][index],
            **({"local_conditioned_on": generated['STANDARD'][index],
                "rewrite_skipped_empty_standard": not bool(generated['STANDARD'][index].strip())} if rewrite_local else {}),
            **({"joint_reply": texts[index], "joint_split": split_joint(texts[index]) is not None} if joint else {}),
        }
        for index, row in enumerate(rows)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="V5 stage 2: LoRA fine-tune of a stage 1 mT5 on development items.")
    parser.add_argument("--base-model", required=True, help="Stage 1 checkpoint, e.g. artifacts/mt5_v5_pretrain/best")
    parser.add_argument("--train-items", type=Path, default=DEV_TRAIN)
    parser.add_argument("--synthetic-items", type=Path, help="Reviewed paired synthetic JSONL; see docs/synthetic_v5.md")
    parser.add_argument("--synthetic-count", type=int, default=190, help="Maximum synthetic items per epoch")
    parser.add_argument("--teacher-targets", type=Path, help="Audited teacher targets for exactly the original training inputs")
    parser.add_argument("--teacher-mode", choices=['replace', 'augment'], default='replace')
    parser.add_argument("--rewrite-local", action='store_true', help='Generate STANDARD then condition LOCAL on that exact draft')
    parser.add_argument("--save-final", action='store_true', help='Save final-step weights and outputs as well as best-by-loss')
    parser.add_argument("--standard-repeats", type=int, default=1, help='STANDARD exposure per paired rewrite row')
    parser.add_argument('--save-steps', type=int, nargs='+', default=[], help='Save exact optimizer-step checkpoints without interrupting training')
    parser.add_argument("--validation-items", type=Path, default=DEV_VALIDATION)
    parser.add_argument("--lora-rank", type=int, default=16, help="0 trains every weight instead of an adapter.")
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--lora-targets", nargs="+", default=["q", "v"], help="mT5 layers: q k v o wi_0 wi_1 wo")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5, help="Stop after this many epochs without improvement.")
    parser.add_argument("--max-steps", type=int, help="Cap optimizer updates, validating the final partial epoch too.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-source-tokens", type=int, default=128)
    parser.add_argument("--max-target-tokens", type=int, default=48)
    parser.add_argument("--output-name", default="mt5_v5_lora")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--context-variants", action="store_true", help="Also train each item with less context.")
    parser.add_argument("--context-pairs", action="store_true",
                        help="Also train on the replies inside the items' contexts, tagged REPLY.")
    parser.add_argument("--rehearsal-pairs", type=Path, nargs="+", default=[],
                        help="Stage 1 pair files to mix into training, e.g. data/public/cosem_train_pairs.jsonl")
    parser.add_argument("--rehearsal-count", type=int, default=400, help="Public pairs to mix in from those files.")
    parser.add_argument("--speaker-labels", action="store_true", help="Mark each turn Friend: or You:.")
    parser.add_argument("--joint", action="store_true",
                        help="Write both styles in one reply, STANDARD then LOCAL, so they say the same thing.")
    parser.add_argument("--seed", type=int, default=425, help="Change it to check a result is not luck.")
    parser.add_argument("--negatives", type=Path,
                        help="A rerank_v5 candidates file made on the training items; its echo replies are pushed down.")
    parser.add_argument("--unlikelihood-weight", type=float, default=1.0)
    parser.add_argument("--negatives-per-row", type=int, default=3, help="Echo replies kept per item and style.")
    parser.add_argument("--preferences", type=Path,
                        help="A rated compare_v5 make-prefs sheet; its picks and \"both bad\" answers train the model by DPO.")
    parser.add_argument("--dpo-weight", type=float, default=1.0)
    parser.add_argument("--dpo-beta", type=float, default=0.1,
                        help="How far the model may move from stage 1 on the picked pairs; higher moves less.")
    args = parser.parse_args()
    report = finetune(
        args.base_model, args.train_items, args.validation_items, args.lora_rank, args.lora_alpha,
        args.lora_dropout, tuple(args.lora_targets), args.epochs, args.patience, args.batch_size,
        args.learning_rate, args.max_source_tokens, args.max_target_tokens, args.output_name, args.output_dir,
        context_variants=args.context_variants, use_context_pairs=args.context_pairs,
        rehearsal_paths=args.rehearsal_pairs,
        rehearsal_count=args.rehearsal_count, speaker_labels=args.speaker_labels, seed=args.seed,
        negatives_path=args.negatives, unlikelihood_weight=args.unlikelihood_weight,
        negatives_per_row=args.negatives_per_row, preferences_path=args.preferences, dpo_weight=args.dpo_weight,
        dpo_beta=args.dpo_beta, joint=args.joint, synthetic_path=args.synthetic_items,
        synthetic_count=args.synthetic_count, max_steps=args.max_steps,
        teacher_path=args.teacher_targets, teacher_mode=args.teacher_mode,
        rewrite_local=args.rewrite_local, save_final=args.save_final, standard_repeats=args.standard_repeats,
        save_steps=tuple(args.save_steps),
    )
    print(f"stage 1 validation loss: {report['history'][0]['validation_loss']}; "
          f"best: {report['best_validation_loss']} (epoch {report['best_epoch']})")


if __name__ == "__main__":
    main()
