"""Score all item pairs for semantic near-duplication. Writes near_dup_pairs.csv (for human review).

Two items are 'near-duplicates' if a model trained/tuned on one would effectively have seen the other:
same incoming message template and same situation, regardless of author. Exact-overlap checks miss these.
"""
import csv
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
MIN_SCORE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.45

ABBR = {"tmr": "tomorrow", "tmrw": "tomorrow", "alr": "already", "ltr": "later", "lib": "library",
        "sch": "school", "thr": "there", "gna": "going", "gg": "going", "cmg": "coming", "cfm": "confirm",
        "rmb": "remember", "smth": "something", "u": "you", "ur": "your", "anot": "or not", "bmt": "badminton",
        "ard": "around", "tdy": "today", "tgt": "together", "qn": "question", "ans": "answer"}


def norm(t: str) -> str:
    t = re.sub(r"^\s*(peer|user)\s*:\s*", "", t.lower())
    t = re.sub(r"\d+(:\d+)?\s*(am|pm)?", " NUM ", t)
    t = re.sub(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\b", " NUM ", t)
    t = re.sub(r"[^\w\s一-鿿]", " ", t)
    return " ".join(ABBR.get(w, w) for w in t.split())


items = []
for r in "ABCDE":
    for line in (IN / f"items_{r}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))

msg = [norm(i["message"]["text"]) for i in items]
mem = [norm(" ".join(m["text"] for m in i["session_memory"][-2:])) for i in items]
rep = [norm(i["replies"]["standard"]) for i in items]


def cos(texts, analyzer, ngram):
    v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, sublinear_tf=True, min_df=1)
    X = v.fit_transform(texts)
    return (X @ X.T).toarray()


s_msg = 0.5 * cos(msg, "word", (1, 2)) + 0.5 * cos(msg, "char_wb", (3, 5))
s_mem = 0.5 * cos(mem, "word", (1, 2)) + 0.5 * cos(mem, "char_wb", (3, 5))
s_rep = 0.5 * cos(rep, "word", (1, 2)) + 0.5 * cos(rep, "char_wb", (3, 5))
# the incoming message matters most (it is the model input); context and reply break ties
score = 0.55 * s_msg + 0.25 * s_mem + 0.20 * s_rep

rows = []
for a, b in combinations(range(len(items)), 2):
    ia, ib = items[a], items[b]
    # same-scenario/intent pairs are more suspicious, but templates cross scenarios too
    same_intent = ia["intent"] == ib["intent"]
    sc = score[a, b] + (0.03 if same_intent else 0)
    if sc >= MIN_SCORE or s_msg[a, b] >= 0.75:
        rows.append({
            "score": round(float(sc), 3), "msg_sim": round(float(s_msg[a, b]), 3),
            "ctx_sim": round(float(s_mem[a, b]), 3), "reply_sim": round(float(s_rep[a, b]), 3),
            "id_a": ia["id"], "id_b": ib["id"],
            "msg_a": ia["message"]["text"], "msg_b": ib["message"]["text"],
            "std_a": ia["replies"]["standard"], "std_b": ib["replies"]["standard"],
            "same_author": ia["author"] == ib["author"],
        })
rows.sort(key=lambda r: -r["score"])
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(len(rows), "candidate pairs >=", MIN_SCORE)
for r in rows[:70]:
    print(f"{r['score']:.2f} m={r['msg_sim']:.2f} c={r['ctx_sim']:.2f} r={r['reply_sim']:.2f} "
          f"{r['id_a'][5:]} | {r['id_b'][5:]} :: {r['msg_a'][:48]!r} ~ {r['msg_b'][:48]!r}")
