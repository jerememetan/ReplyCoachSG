"""Independent post-hoc checks of the published split files. Usage: verify_split.py <root> <pairs.csv> <manual.json>"""
import csv, json, re, sys
from pathlib import Path
root = Path(sys.argv[1]); pairs = Path(sys.argv[2]); manual = json.loads(Path(sys.argv[3]).read_text())
rd = lambda p: [json.loads(l) for l in (root / p).read_text(encoding="utf-8").splitlines() if l.strip()]
tr, va, te = rd("data/dev/train.jsonl"), rd("data/dev/validation.jsonl"), rd("data/test/test.jsonl")
part = {**{i["id"]: "train" for i in tr}, **{i["id"]: "validation" for i in va}, **{i["id"]: "test" for i in te}}
ok = True
def check(name, cond, extra=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, extra)
check("sizes 190/50/60", (len(tr), len(va), len(te)) == (190, 50, 60))
check("all 300 ids unique across files", len(part) == 300 and len(tr)+len(va)+len(te) == 300)
check("dev files all split=dev", all(i["split"] == "dev" for i in tr + va))
check("test file all split=test", all(i["split"] == "test" for i in te))
check("all completed + privacy_ok", all(i["status"] == "completed" and i["privacy_ok"] is True for i in tr+va+te))
norm = lambda t: re.sub(r"\W+", " ", re.sub(r"^\s*(peer|user)\s*:\s*", "", t.lower())).strip()
conv = lambda i: norm(" ".join(m["text"] for m in i["session_memory"]) + " || " + i["message"]["text"])
for a, an, b, bn in ((tr, "train", va, "validation"), (tr+va, "dev", te, "test")):
    check(f"no exact message overlap {an} vs {bn}", not ({norm(i['message']['text']) for i in a} & {norm(i['message']['text']) for i in b}))
    check(f"no exact conversation overlap {an} vs {bn}", not ({conv(i) for i in a} & {conv(i) for i in b}))
# groups never straddle partitions
bad = []
for row in csv.DictReader(pairs.open(encoding="utf-8")):
    if float(row["score"]) >= 0.42 and part[row["id_a"]] != part[row["id_b"]]:
        bad.append((row["id_a"], row["id_b"], row["score"]))
for g in manual["groups"]:
    if len({part[i] for i in g["ids"]}) > 1: bad.append(g["ids"])
check("auto and manual near-dup groups kept on one side", not bad, str(bad))
# remaining cross-partition high-ish pairs (informational)
lo = [(r["id_a"], r["id_b"], r["score"], part[r["id_a"]], part[r["id_b"]]) for r in csv.DictReader(pairs.open(encoding="utf-8"))
      if part[r["id_a"]] != part[r["id_b"]] and "test" in (part[r["id_a"]], part[r["id_b"]]) and float(r["score"]) >= 0.34]
print("INFO dev<->test pairs with score>=0.34 (review):", len(lo))
for r in lo: print("   ", r)
# records identical except split
src = {}
for r in "ABCDE":
    for l in Path(f"data/authoring/items_{r}.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip(): d = json.loads(l); src[d["id"]] = d
check("records untouched except split", all({**src[i["id"]], "split": i["split"]} == i for i in tr+va+te))
for fid in ("ITEM-D-002", "ITEM-D-004", "ITEM-D-029", "ITEM-E-055", "ITEM-E-024", "ITEM-D-009"):
    print("INFO flagged", fid, "->", part[fid])
sys.exit(0 if ok else 1)
