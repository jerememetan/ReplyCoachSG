"""
Fixed, group-aware, stratified 240 dev / 60 test split plus a validation subset inside dev.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROLES = "ABCDE"
REVIEWER_OF = {"A": "E", "B": "A", "C": "B", "D": "C", "E": "D"}  # author role -> reviewer role


# ----------------------------------------------------------------------------- loading and gate
def load_items(items_dir: Path):
    items = []
    for r in ROLES:
        for line in (items_dir / f"items_{r}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                it = json.loads(line)
                it["__role"] = r
                items.append(it)
    items.sort(key=lambda i: i["id"])
    return items


def gate(items):
    errs = []
    if len(items) != 300:
        errs.append(f"expected 300 items, found {len(items)}")
    ids = Counter(i["id"] for i in items)
    errs += [f"duplicate id {k}" for k, v in ids.items() if v > 1]
    author = {}
    for i in items:
        author.setdefault(i["__role"], set()).add(i["author"])
    for r in ROLES:
        if len(author[r]) != 1:
            errs.append(f"role {r} has several authors {author[r]}")
        n = sum(1 for i in items if i["__role"] == r)
        if n != 60:
            errs.append(f"role {r} has {n} items, expected 60")
    name = {r: next(iter(author[r])) for r in ROLES}
    for i in items:
        if i.get("status") != "completed":
            errs.append(f"{i['id']} status={i.get('status')}")
        if i.get("privacy_ok") is not True:
            errs.append(f"{i['id']} privacy_ok={i.get('privacy_ok')}")
        if i.get("reviewer") != name[REVIEWER_OF[i["__role"]]]:
            errs.append(f"{i['id']} reviewer {i.get('reviewer')} breaks the review chain")
        if i.get("reviewer") == i.get("author"):
            errs.append(f"{i['id']} reviewed by its own author")
    return errs


# ----------------------------------------------------------------------------- near-duplicate groups
def build_groups(items, pairs_csv: Path, threshold: float, manual_json: Path):
    idx = {it["id"]: n for n, it in enumerate(items)}
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    n_auto = 0
    if pairs_csv and pairs_csv.exists():
        for row in csv.DictReader(pairs_csv.open(encoding="utf-8")):
            if float(row["score"]) >= threshold:
                union(idx[row["id_a"]], idx[row["id_b"]])
                n_auto += 1
    n_manual = 0
    if manual_json and manual_json.exists():
        for g in json.loads(manual_json.read_text(encoding="utf-8"))["groups"]:
            for other in g["ids"][1:]:
                union(idx[g["ids"][0]], idx[other])
            n_manual += 1
    # exact duplicate normalised messages always share a group
    seen = {}
    for n, it in enumerate(items):
        key = re.sub(r"\W+", " ", it["message"]["text"].lower()).strip()
        if key in seen:
            union(n, seen[key])
        else:
            seen[key] = n
    comps = defaultdict(list)
    for n in range(len(items)):
        comps[find(n)].append(n)
    groups = sorted(comps.values(), key=lambda g: g[0])
    print(f"near-dup groups: {n_auto} auto pairs >= {threshold}, {n_manual} manual groups -> "
          f"{sum(1 for g in groups if len(g) > 1)} multi-item groups, {len(groups)} units")
    return groups


# ----------------------------------------------------------------------------- stratified assignment
def features(items, subset):
    """One-hot matrix over the strata we want balanced; weights favour author balance."""
    cols, weights = [], []

    def add(name, vals, w):
        for v in sorted(set(vals)):
            cols.append((name, v, np.array([1.0 if x == v else 0.0 for x in vals])))
            weights.append(w)

    its = [items[i] for i in subset]
    add("author", [i["author"] for i in its], 4.0)
    add("scenario", [i["scenario"] for i in its], 2.0)
    add("intent", [i["intent"] for i in its], 2.0)
    add("local_terms", [min(len(i["message"].get("local_terms", [])), 2) for i in its], 1.0)
    add("code_mix", [bool(set(i["message"].get("language_mix", [])) & {"zh", "pinyin"}) for i in its], 1.0)
    F = np.stack([c[2] for c in cols], axis=1)
    return F, np.array(weights), [(c[0], c[1]) for c in cols]


def anneal(items, subset, units, k, rng, iters=80000):
    """Choose units (index lists into `items`, all inside `subset`) holding exactly k items, to minimise
    deviation from proportional representation. Units are never split."""
    pos = {g: p for p, g in enumerate(subset)}
    F, W, _ = features(items, subset)
    frac = k / len(subset)
    target = frac * F.sum(axis=0)
    scale = np.maximum(target, 1.0)

    def cost(x):
        d = F.T @ x - target
        return float((W * d * d / scale).sum())

    sizes = [len(u) for u in units]
    order = list(range(len(units)))
    rng.shuffle(order)
    chosen, total = [], 0
    for u in order:  # random start that hits exactly k
        if total + sizes[u] <= k:
            chosen.append(u)
            total += sizes[u]
    if total != k:
        raise RuntimeError("cannot reach exact size with unit sizes")
    sel = set(chosen)
    x = np.zeros(len(subset))
    for u in sel:
        for g in units[u]:
            x[pos[g]] = 1.0
    cur = cost(x)
    by_size_in, by_size_out = defaultdict(list), defaultdict(list)
    for u in range(len(units)):
        (by_size_in if u in sel else by_size_out)[sizes[u]].append(u)
    best, best_sel = cur, set(sel)
    for t in range(iters):
        T = 1.5 * (0.002 / 1.5) ** (t / iters)
        s = rng.choice([s for s in by_size_in if by_size_in[s] and by_size_out[s]])
        a = rng.randrange(len(by_size_in[s]))
        b = rng.randrange(len(by_size_out[s]))
        ua, ub = by_size_in[s][a], by_size_out[s][b]
        for g in units[ua]:
            x[pos[g]] = 0.0
        for g in units[ub]:
            x[pos[g]] = 1.0
        new = cost(x)
        if new <= cur or rng.random() < math.exp((cur - new) / T):
            by_size_in[s][a], by_size_out[s][b] = ub, ua
            sel.discard(ua)
            sel.add(ub)
            cur = new
            if cur < best:
                best, best_sel = cur, set(sel)
        else:
            for g in units[ub]:
                x[pos[g]] = 0.0
            for g in units[ua]:
                x[pos[g]] = 1.0
    return sorted(best_sel), best


# ----------------------------------------------------------------------------- output helpers
def sha256(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_jsonl(path: Path, items, split_value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for it in items:
            rec = {k: v for k, v in it.items() if not k.startswith("__")}
            rec["split"] = split_value
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def table(items, key):
    c = Counter()
    for it, part in items:
        c[(key(it), part)] += 1
    keys = sorted({k for k, _ in c})
    rows = [f"| {k} | {c[(k,'train')]} | {c[(k,'validation')]} | {c[(k,'test')]} | "
            f"{c[(k,'train')]+c[(k,'validation')]+c[(k,'test')]} |" for k in keys]
    return "| value | train | validation | test | total |\n|---|---|---|---|---|\n" + "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pairs", type=Path)
    ap.add_argument("--manual", type=Path)
    ap.add_argument("--threshold", type=float, default=0.42)
    ap.add_argument("--seed", type=int, default=425)
    ap.add_argument("--val-size", type=int, default=50)
    ap.add_argument("--force", action="store_true", help="re-randomise even if a manifest exists (pre-freeze only)")
    args = ap.parse_args()

    items = load_items(args.items_dir)
    errs = gate(items)
    if errs:
        raise SystemExit("GATE FAILED:\n  " + "\n  ".join(errs))
    out = args.out_dir
    manifest_path = out / "data/split/manifest.json"
    ids = [i["id"] for i in items]

    if manifest_path.exists() and not args.force:
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        if set(man["assignment"]) != set(ids):
            raise SystemExit("manifest ids do not match the item files; refusing to continue")
        part = man["assignment"]
        groups_by_id = man["group_of"]
        print("applying frozen manifest (seed", man["seed"], ")")
    else:
        groups = build_groups(items, args.pairs, args.threshold, args.manual)
        rng = random.Random(args.seed)
        all_idx = list(range(len(items)))
        test_units, c1 = anneal(items, all_idx, groups, 60, rng)
        test_idx = {g for u in test_units for g in groups[u]}
        dev_idx = [g for g in all_idx if g not in test_idx]
        dev_units = [g for g in groups if g[0] not in test_idx]
        val_units, c2 = anneal(items, dev_idx, dev_units, args.val_size, rng)
        val_idx = {g for u in val_units for g in dev_units[u]}
        part = {}
        for n, it in enumerate(items):
            part[it["id"]] = "test" if n in test_idx else ("validation" if n in val_idx else "train")
        groups_by_id = {}
        for gi, g in enumerate(groups):
            for n in g:
                groups_by_id[items[n]["id"]] = f"G{gi:03d}" if len(g) > 1 else None
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({
            "seed": args.seed, "val_size": args.val_size, "near_dup_threshold": args.threshold,
            "stratified_by": ["author", "scenario", "intent", "local_terms_count(0/1/2+)", "code_mix(zh/pinyin)"],
            "method": "group-aware simulated annealing, groups never split, exact sizes 60 test / val_size validation",
            "cost_test": round(c1, 4), "cost_validation": round(c2, 4),
            "assignment": part, "group_of": groups_by_id}, ensure_ascii=False, indent=1), encoding="utf-8")

    by = {"train": [], "validation": [], "test": []}
    for it in items:
        by[part[it["id"]]].append(it)
    write_jsonl(out / "data/dev/train.jsonl", by["train"], "dev")
    write_jsonl(out / "data/dev/validation.jsonl", by["validation"], "dev")
    write_jsonl(out / "data/test/test.jsonl", by["test"], "test")
    (out / "data/split/val_ids.json").write_text(
        json.dumps([i["id"] for i in by["validation"]], indent=1), encoding="utf-8")

    # ---- report
    paired = [(it, part[it["id"]]) for it in items]
    files = {p: sha256(out / p) for p in ("data/dev/train.jsonl", "data/dev/validation.jsonl", "data/test/test.jsonl")}
    rep = {"counts": {k: len(v) for k, v in by.items()}, "sha256": files,
           "manifest_sha256": sha256(manifest_path)}
    (out / "data/split/split_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    md = [f"# Split report\n\nseed={man['seed']}  method: {man['method']}\n",
          f"stratified by: {', '.join(man['stratified_by'])}\n",
          f"near-duplicate threshold: {man['near_dup_threshold']} (+ manual groups)\n",
          f"counts: train {len(by['train'])}, validation {len(by['validation'])}, test {len(by['test'])}\n",
          "## By author\n" + table(paired, lambda i: i["author"]),
          "## By scenario\n" + table(paired, lambda i: i["scenario"]),
          "## By intent\n" + table(paired, lambda i: i["intent"]),
          "## By number of local_terms (0/1/2+)\n" + table(paired, lambda i: min(len(i['message'].get('local_terms', [])), 2)),
          "## SHA-256\n" + "\n".join(f"- `{p}`  {h}" for p, h in {**files, 'data/split/manifest.json': rep['manifest_sha256']}.items())]
    (out / "data/split/split_report.md").write_text("\n\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(rep["counts"]))


if __name__ == "__main__":
    main()
