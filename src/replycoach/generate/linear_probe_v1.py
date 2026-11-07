"""Counterbalanced conversation facts and strictly linear diagnostic readouts."""

from collections import Counter, defaultdict
import random
import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from replycoach.generate.mt5_matched_v1 import source

TASKS = ("availability", "completion", "ownership")
SPLITS = ("train", "validation", "heldout_development")
CLASS_NAMES = {
    "availability": ["unavailable", "available"],
    "completion": ["incomplete", "completed"],
    "ownership": ["peer_owns", "user_owns"],
}

# Each family states BOTH labels. Query target, fact order and the association
# between entity and label are independently counterbalanced.
TEMPLATES = {
    "availability": [
        ("I can meet {positive}.", "I cannot meet {negative}.", "Are you available {query}?"),
        ("I am free {positive}.", "I am busy {negative}.", "Can you meet {query}?"),
        ("{positive} works for me.", "{negative} does not work for me.", "Does {query} work for you?"),
        ("I am available {positive}.", "I am unavailable {negative}.", "Will you be free {query}?"),
        ("I have time to meet {positive}.", "I have no time to meet {negative}.", "Do you have time {query}?"),
        ("Meeting {positive} is possible for me.", "Meeting {negative} is impossible for me.", "Is meeting {query} possible for you?"),
        ("I can fit you in {positive}.", "I cannot fit you in {negative}.", "Would you be able to see me {query}?"),
        ("I kept that slot open for our meeting {positive}.", "I have another commitment {negative} and cannot meet then.", "Could we get together {query}?"),
    ],
    "completion": [
        ("I have finished {positive}.", "I have not finished {negative}.", "Have you finished {query}?"),
        ("I have completed {positive}.", "I have not completed {negative}.", "Have you completed {query}?"),
        ("I am done with {positive}.", "I am not done with {negative}.", "Are you done with {query}?"),
        ("My work on {positive} is complete.", "My work on {negative} is incomplete.", "Is your work on {query} complete?"),
        ("I finished working on {positive}.", "I still need to finish working on {negative}.", "Is your work on {query} finished?"),
        ("I have got {positive} done.", "I have yet to get {negative} done.", "Have you got {query} done?"),
        ("I have wrapped up {positive}.", "I have not even started {negative}.", "Is {query} finished already?"),
        ("There is nothing left for me to do on {positive}.", "I still have work remaining on {negative}.", "Have you done everything needed for {query}?"),
    ],
    "ownership": [
        ("The {positive} belongs to me.", "The {negative} belongs to you.", "Who owns the {query}?"),
        ("I lent you my {positive}.", "You lent me your {negative}.", "Whose {query} is it originally?"),
        ("You are borrowing my {positive}.", "I am borrowing your {negative}.", "Who is the owner of the {query}?"),
        ("You have my {positive} on loan.", "I have your {negative} on loan.", "Who does the {query} belong to?"),
        ("I own the {positive} you are using.", "You own the {negative} I am using.", "Remind me who owns the {query}?"),
        ("You borrowed the {positive} from me.", "I borrowed the {negative} from you.", "Who originally owned the {query}?"),
        ("You need to give my {positive} back to me.", "I need to give your {negative} back to you.", "Who is supposed to get the {query} back?"),
        ("I let you use the {positive} that is mine.", "You let me use the {negative} that is yours.", "Which one of us actually owns the {query}?"),
    ],
}

ENTITIES = {
    "availability": {
        "train": [(f"on {day} at {a}", f"on {day} at {b}") for day in ["Monday", "Tuesday", "Wednesday"] for a, b in [("08:10", "09:20"), ("10:30", "11:40"), ("13:15", "14:25"), ("16:35", "17:45")]],
        "validation": [(f"on {day} at {a}", f"on {day} at {b}") for day in ["Thursday", "Friday"] for a, b in [("08:35", "10:05"), ("15:50", "18:20")]],
        "heldout_development": [(f"on {day} at {a}", f"on {day} at {b}") for day in ["Saturday", "Sunday"] for a, b in [("07:55", "09:45"), ("12:40", "19:10")]],
    },
    "completion": {
        "train": [("the lab worksheet", "the poster draft"), ("the slides", "the spreadsheet"), ("the essay", "the survey"), ("the budget report", "the attendance list"), ("the invitation cards", "the meeting notes"), ("the packing checklist", "the practice questions"), ("the video edit", "the email draft"), ("the registration form", "the project outline"), ("the reading summary", "the timetable"), ("the event plan", "the room booking"), ("the presentation script", "the expense claim"), ("the signboard", "the assignment cover")],
        "validation": [("the route map", "the menu design"), ("the quiz answers", "the feedback sheet"), ("the brochure", "the equipment list"), ("the seating chart", "the progress log")],
        "heldout_development": [("the welcome kits", "the route sketch"), ("the rehearsal recording", "the sponsorship letter"), ("the safety briefing", "the photo album"), ("the certificate layout", "the inventory count")],
    },
    "ownership": {
        "train": [("calculator", "umbrella"), ("charger", "ruler"), ("textbook", "notebook"), ("headphones", "camera"), ("pen", "pencil"), ("bottle", "lunchbox"), ("jacket", "scarf"), ("laptop", "tablet"), ("football", "basketball"), ("folder", "stapler"), ("keyboard", "mouse"), ("helmet", "pump")],
        "validation": [("compass", "protractor"), ("torch", "tripod"), ("microphone", "speaker"), ("raincoat", "cap")],
        "heldout_development": [("goggles", "whistle"), ("sketchbook", "easel"), ("thermos", "mug"), ("powerbank", "adapter")],
    },
}
FAMILIES = {"train": range(4), "validation": range(4, 6), "heldout_development": range(6, 8)}


def dataset():
    result = []
    for task in TASKS:
        for split in SPLITS:
            for family in FAMILIES[split]:
                positive, negative, question = TEMPLATES[task][family]
                for entity_index, entities in enumerate(ENTITIES[task][split]):
                    for order in range(2):
                        for binding in range(2):
                            values = dict(positive=entities[binding], negative=entities[1 - binding])
                            facts = [positive.format(**values), negative.format(**values)]
                            if order:
                                facts.reverse()
                            group = f"{task}-{split}-f{family}-e{entity_index}-o{order}"
                            for query in range(2):
                                label = int(query == binding)
                                result.append(dict(
                                    item_id=f"LP1-{group}-b{binding}-q{query}",
                                    split=split, task=task, label=label, class_name=CLASS_NAMES[task][label],
                                    family=f"{task}-{family}", entities=list(entities),
                                    contrast_group=f"{group}-b{binding}", counterfactual_group=f"{group}-q{query}",
                                    context=[" ".join(facts)], speakers=["user"],
                                    message=question.format(query=entities[query]),
                                ))
    random.Random(425).shuffle(result)
    validate_dataset(result)
    return result


def validate_dataset(rows):
    if len({r["item_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate diagnostic IDs")
    for task in TASKS:
        families, entities = {}, {}
        for split in SPLITS:
            part = [r for r in rows if r["task"] == task and r["split"] == split]
            counts = Counter(r["label"] for r in part)
            if not part or counts[0] != counts[1]:
                raise ValueError("Each task/split must be balanced")
            families[split] = {r["family"] for r in part}
            entities[split] = {e for r in part for e in r["entities"]}
        for i, a in enumerate(SPLITS):
            for b in SPLITS[i + 1:]:
                if families[a] & families[b] or entities[a] & entities[b]:
                    raise ValueError("Family/entity overlap across diagnostic splits")
    groups = defaultdict(list)
    for r in rows:
        groups[r["counterfactual_group"]].append(r)
    for part in groups.values():
        if len(part) != 2 or {r["label"] for r in part} != {0, 1}:
            raise ValueError("Counterfactual groups require opposite labels")
        bags = [Counter(re.findall(r"\w+", source(r, "STANDARD").lower())) for r in part]
        if bags[0] != bags[1]:
            raise ValueError("Counterfactual pair has a bag-of-words label shortcut")


def fit_linear(features, labels, strength):
    # Scaling is fitted on TRAIN only. Affine scaling + logistic regression is
    # still a linear classifier in the original frozen feature coordinates.
    model = make_pipeline(StandardScaler(), LogisticRegression(C=strength, solver="lbfgs", max_iter=1500, random_state=425))
    return model.fit(features, labels)


def pooled_hidden_states(states, mask):
    """No learned pooling: fixed masked means and the last real encoder token."""
    import torch
    weights = mask.unsqueeze(-1).to(states[0].dtype)
    last_positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask).masked_fill(~mask.bool(), -1).max(-1).values
    if torch.any(last_positions < 0):
        raise ValueError("Empty encoder sequence")
    layers = sorted({0, (len(states) - 1) // 2, len(states) - 1})
    result = {f"layer{layer}_mean": (states[layer] * weights).sum(1) / weights.sum(1) for layer in layers}
    result[f"layer{len(states)-1}_last"] = states[-1][torch.arange(mask.shape[0], device=mask.device), last_positions]
    return result


def metrics(rows, probabilities):
    labels = np.array([r["label"] for r in rows])
    probabilities = np.asarray(probabilities)
    predicted = (probabilities >= 0.5).astype(int)
    result = dict(examples=len(rows), accuracy=float(accuracy_score(labels, predicted)),
                  macro_f1=float(f1_score(labels, predicted, average="macro", zero_division=0)),
                  log_loss=float(log_loss(labels, np.column_stack([1 - probabilities, probabilities]), labels=[0, 1])))
    for field in ["contrast_group", "counterfactual_group"]:
        grouped = defaultdict(list)
        for i, row in enumerate(rows):
            grouped[row[field]].append(i)
        if any(len(g) != 2 for g in grouped.values()):
            raise ValueError("Incomplete paired diagnostic evaluation")
        result[field] = dict(groups=len(grouped),
            both_correct=float(np.mean([all(predicted[i] == labels[i] for i in g) for g in grouped.values()])),
            prediction_flip_rate=float(np.mean([predicted[g[0]] != predicted[g[1]] for g in grouped.values()])))
    return result
