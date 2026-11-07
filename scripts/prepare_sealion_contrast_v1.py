"""Author balanced fact bindings; freeze training, held-template validation and reused eval."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/training_expansion/sealion_contrast_v1"


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(variant, binding, query, category):
    # The fifth entity pair is reserved for validation. Each binding/query has
    # an equally frequent counterfactual, so keywords alone cannot choose a fact.
    names = ["pottery", "robotics", "photography", "gardening", "ceramics"]
    a, b = [("Tuesday", "Sunday"), ("Monday", "Saturday"), ("Wednesday", "Sunday"),
            ("Tuesday", "Saturday"), ("Monday", "Sunday")][variant]
    noun1, noun2 = [("consent form", "claim form"), ("housing form", "travel form"),
                    ("entry form", "waiver form"), ("booking form", "refund form"),
                    ("visitor form", "permit form")][variant]
    obj1, obj2 = [("tripod", "stapler"), ("umbrella", "toolbox"), ("marker", "headset"),
                  ("thermos", "torch"), ("pencil case", "measuring tape")][variant]
    slot1, slot2 = [("09:17", "16:42"), ("10:26", "18:53"), ("11:38", "17:49"),
                    ("12:43", "19:16"), ("07:24", "15:57")][variant]
    amount1, amount2 = [("13", "29"), ("17", "46"), ("23", "58"), ("31", "67"), ("19", "43")][variant]
    room1, room2 = [("K428", "L529"), ("M637", "N748"), ("P852", "Q963"),
                    ("R576", "S687"), ("T794", "U895")][variant]
    # Keep one current user statement containing both bindings. All labels are
    # entailed facts; a different freely chosen conversational intention is not a negative.
    if category == "availability":
        free, busy = ((a, b) if binding == 0 else (b, a))
        day = [a, b][query]; yes = day == free
        context = f"I am free on {free}, but I am unavailable on {busy}."
        message = f"Are you free on {day}?"
        pos = f"Yes, I am free on {day}." if yes else f"I am unavailable on {day}."
        neg = f"I am unavailable on {day}." if yes else f"Yes, I am free on {day}."
        local = f"{day} I free." if yes else f"{day} I cannot make it."
        wrong = f"{day} I cannot make it." if yes else f"{day} I free."
        neutral = "Remind me which days work for you?"
    elif category == "completion":
        done, pending = ((noun1, noun2) if binding == 0 else (noun2, noun1))
        noun = [noun1, noun2][query]; yes = noun == done
        context = f"I have submitted the {done}. I have not submitted the {pending}."
        message = f"Have you submitted the {noun}?"
        pos = f"I have submitted the {noun}." if yes else f"I have not submitted the {noun} yet."
        neg = f"I have not submitted the {noun} yet." if yes else f"I have submitted the {noun}."
        local = f"The {noun} I submit already." if yes else f"The {noun} I haven't submit yet."
        wrong = f"The {noun} I haven't submit yet." if yes else f"The {noun} I submit already."
        neutral = "What is the status of your forms?"
    elif category == "ownership":
        mine, yours = ((obj1, obj2) if binding == 0 else (obj2, obj1))
        obj = [obj1, obj2][query]; yes = obj == mine
        context = f"The {mine} belongs to me. The {yours} belongs to you."
        message = f"Who owns the {obj}?"
        pos = f"The {obj} belongs to me." if yes else f"The {obj} belongs to you."
        neg = f"The {obj} belongs to you." if yes else f"The {obj} belongs to me."
        local = f"The {obj} is mine." if yes else f"The {obj} is yours."
        wrong = f"The {obj} is yours." if yes else f"The {obj} is mine."
        neutral = "Whose things are these again?"
    elif category == "return_direction":
        borrowed, lent = ((obj1, obj2) if binding == 0 else (obj2, obj1))
        obj = [obj1, obj2][query]; yes = obj == borrowed
        context = f"I borrowed your {borrowed}. You borrowed my {lent}. We are returning both today."
        message = f"Who should return the {obj} to whom?"
        pos = f"I will return the {obj} to you." if yes else f"You will return the {obj} to me."
        neg = f"You will return the {obj} to me." if yes else f"I will return the {obj} to you."
        local = f"I return the {obj} to you later." if yes else f"Later you return the {obj} to me."
        wrong = f"Later you return the {obj} to me." if yes else f"I return the {obj} to you later."
        neutral = "How are we returning these things?"
    elif category == "exact_time":
        first, second = ((slot1, slot2) if binding == 0 else (slot2, slot1))
        event = ["practice", "briefing"][query]; value, other = [(first, second), (second, first)][query]
        context = f"My {names[variant]} practice starts at {first}. My briefing starts at {second}."
        message = f"What time does your {event} start?"
        pos = f"My {event} starts at {value}."; neg = f"My {event} starts at {other}."
        local = f"My {event} starts at {value}."; wrong = f"My {event} starts at {other}."
        neutral = "What are the times for your activities?"
    elif category == "amount":
        budget, fee = ((amount1, amount2) if binding == 0 else (amount2, amount1))
        label = ["budget", "entry fee"][query]; value, other = [(budget, fee), (fee, budget)][query]
        context = f"My budget is ${budget}. The {names[variant]} event entry fee is ${fee}."
        message = f"How much is the {label}?"
        pos = f"The {label} is ${value}."; neg = f"The {label} is ${other}."
        local = f"The {label} is ${value} lah."; wrong = f"The {label} is ${other} lah."
        neutral = "Remind me of the two amounts?"
    elif category == "room":
        first, second = ((room1, room2) if binding == 0 else (room2, room1))
        event = ["workshop", "seminar"][query]; value, other = [(first, second), (second, first)][query]
        context = f"My workshop is in {first}. My seminar is in {second}."
        message = f"Which room is your {event} in?"
        pos = f"My {event} is in {value}."; neg = f"My {event} is in {other}."
        local = f"My {event} at {value}."; wrong = f"My {event} at {other}."
        neutral = "Which rooms are your events in?"
    elif category == "before_after":
        before, after = (("collection", "delivery") if binding == 0 else ("delivery", "collection"))
        event = ["collection", "delivery"][query]; yes = event == before
        context = f"The {before} must happen before {slot1}. The {after} must happen after {slot1}."
        message = f"Does the {event} need to be before or after {slot1}?"
        direction, opposite = ("before", "after") if yes else ("after", "before")
        pos = f"The {event} must be {direction} {slot1}."; neg = f"The {event} must be {opposite} {slot1}."
        local = f"The {event} must be {direction} {slot1} ah."; wrong = f"The {event} must be {opposite} {slot1} ah."
        neutral = "What are the timing constraints again?"
    else:
        raise ValueError(category)
    return dict(item_id=f"CONTRAST-{category}-{variant}-{binding}-{query}", category=category,
                context=[context], speakers=["user"], message=message, rewrite_message=neutral,
                standard_anchor=pos, english=dict(positive=pos, negative=neg),
                rewrite=dict(positive=local, negative=wrong),
                negative_reason="Contradicts an explicitly stated actor, binding, polarity or exact value.")


def main():
    categories = ["availability", "completion", "ownership", "return_direction", "exact_time", "amount", "room", "before_after"]
    DEST.mkdir(parents=True, exist_ok=True)
    groups = {}
    for split, variants in [("train", range(4)), ("validation", [4])]:
        rows = [build(v, b, q, c) for c in categories for v in variants for b in range(2) for q in range(2)]
        groups[split] = rows
        path = DEST / f"{split}.json"
        if path.exists() and json.loads(path.read_text()) != rows:
            raise FileExistsError("Do not overwrite a different frozen corpus")
        path.write_text(json.dumps(rows, indent=2) + "\n")
    evaluation = sum((read(p) for p in ["data/evaluation/sealion_full_rewrite_v1/inputs.json", "data/evaluation/restored_pipeline_v1/inputs.json", "data/evaluation/context_pairs_v1/inputs.json"]), [])
    def key(row):
        return json.dumps([row["context"], row["speakers"], row["message"]], sort_keys=True)
    keys = {key(r) for r in evaluation}
    allrows = groups["train"] + groups["validation"]
    assert len(evaluation) == len({r['item_id'] for r in evaluation}) == 126
    assert len(groups['train']) == 128 and len(groups['validation']) == 32
    assert not keys.intersection(key(r) for r in allrows)
    assert not {key(r) for r in groups['train']}.intersection(key(r) for r in groups['validation'])
    assert sha(ROOT/'data/dev/train.jsonl') == '03485c7d6b510711fbdee5f3eac0044e334d9d7a25bf3c5c86665b0646e8eee7'
    manifest = dict(train=128, validation=32, categories=categories, provenance="Assistant-authored synthetic fact bindings; not human gold.",
                    validation_scope="Same templates, held-out entity/value combinations; not an independent generalization test.",
                    development_evaluation_cases=126, training_evaluation_prompt_overlap=0,
                    original_team_sha256=sha(ROOT/'data/dev/train.jsonl'),
                    files_sha256={f'{s}.json':sha(DEST/f'{s}.json') for s in groups})
    (DEST/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    vendor = read('src/replycoach/vendor/sealion_v1/provenance.json')
    initial = dict(
        english=dict(checkpoint='/common/home/users/h/hm.teoh.2024/ReplyCoachSG_sealion_full_v1/artifacts/best_adapter',
                     files_sha256={'adapter_model.safetensors':'55a7f02626c08894bfdc4b44f68dacf86e8f4edafb7aa6bd78aead9162698913',
                                   'adapter_config.json':'f5bb26bc63c376e64ed060f54cc87a44ece14a7328f064d02abc487aa75578bc'}),
        rewrite=dict(checkpoint='/common/home/users/h/hm.teoh.2024/ReplyCoachSG_sealion_full_rewrite_v1/artifacts/best_adapter',
                     files_sha256={'adapter_model.safetensors':'2db1043556b5678683ea33e0f91ddd924ac98b14168ed043dd57b2ce64cf9426',
                                   'adapter_config.json':'238ae221adf866990a14e822621dae070a067eb00aeb3b27f61bce7e5a33627b'}))
    files = [*ROOT.glob('src/**/*.py'),
             *[p for p in (ROOT/'src/replycoach/vendor/sealion_v1').iterdir() if p.is_file() and p.suffix!='.pyc'],
             *[ROOT/'scripts'/p for p in ['run_sealion_contrast_v1.py','prepare_sealion_contrast_v1.py',
                 'run_sealion_dual_v1.py','run_sealion_dialogue_v1.py','run_sealion_expanded_v1.py','run_sealion_full_rewrite_v1.py','v5_sealion_contrast_v1.sbatch']],
             *DEST.glob('*.json'),
             *[ROOT/p for p in ['data/dev/train.jsonl','data/dev/validation.jsonl',
                 'data/training_expansion/generated600_repair_v1/combined_train.jsonl',
                 'data/training_expansion/sealion_rewrite_v1/pair_audit.json',
                 'data/evaluation/sealion_full_rewrite_v1/inputs.json',
                 'data/evaluation/restored_pipeline_v1/inputs.json','data/evaluation/context_pairs_v1/inputs.json',
                 'tests/test_sealion_contrast_v1.py','tests/test_sealion_beam_v1.py','tests/test_sealion_rewrite_v1.py']]]
    files = list(dict.fromkeys(files))
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in files}
    hashes['baseline_beam_results.json'] = sha(ROOT/'artifacts/sealion_beam_v1/results.json')
    recipe = dict(experiment='sealion_contrast_v1', revision=vendor['revision'], weights_sha256=vendor['weights']['lfs']['sha256'],
        initial_adapters=initial, arms=['control','repair'], seed=425, gpu='NVIDIA L40S',
        training=dict(stages=['english','rewrite'], ordinary_train=[790,745], ordinary_validation=[50,42],
            new_synthetic_train=128, new_synthetic_validation=32, optimizer_steps=200, replay_examples_per_step=8,
            auxiliary_pairs_per_step=1, auxiliary_weight=dict(control=0.,repair=.1), learning_rate=1e-5,
            weight_decay=.01, objective='mean of 8 ordinary causal reply CEs + weight*(positive reply CE + .25*relu(.2+positiveCE-negativeCE))',
            matched='Same initial adapters, replay/auxiliary order and per-step dropout seed; both arms forward/backward auxiliary graphs.',
            snapshots=[0,25,50,100,150,200], selection='Step0 eligible; ordinary validation token CE <=1.02*initial; minimize .8*ordinaryCE+.2*synthetic positive example-mean CE.',
            checkpoint_selection_limit='Guard is a loss constraint, not a guarantee of contextual quality. No automatic default promotion.'),
        evaluation=dict(cases=126, methods=['greedy','beam4_ng3'], max_new_tokens=64,
            routes=['own_english','fixed_baseline_english','diagnostic'],
            note='Reused development cases; none included in new training. Synthetic validation uses held entity combinations within templates. Human review pending.'),
        default_changed=False, files_sha256=hashes)
    recipe_path=ROOT/'experiments/sealion_contrast_v1_recipe.json'
    recipe_path.write_text(json.dumps(recipe,indent=2)+'\n')
    with tarfile.open('/tmp/replycoach_sealion_contrast_v1.tgz','w:gz') as tar:
        for p in files: tar.add(p, arcname=str(p.relative_to(ROOT)))
        tar.add(ROOT/'artifacts/sealion_beam_v1/results.json',arcname='baseline_beam_results.json')
        tar.add(recipe_path,arcname='recipe.json')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
