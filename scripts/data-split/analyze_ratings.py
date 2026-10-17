"""
Reports per system and style: mean score per dimension with a 95% bootstrap CI (resampling items), problem rates,
pairwise paired differences between systems, and inter-rater agreement (Krippendorff's alpha, ordinal).
Rows with a missing score are dropped for that dimension. Ratings are never auto-generated.
"""
import argparse
import csv
import glob
import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from openpyxl import load_workbook

DIMS = ["appropriate", "natural", "style_fit"]


def load_key(p):
    key = {}
    for r in csv.DictReader(Path(p).open(encoding="utf-8-sig")):
        key[r["row_id"]] = {"item_id": r["item_id"], "style": r["style"], "systems": r["systems"].split("|")}
    return key


def load_sheet(p):
    wb = load_workbook(p, data_only=True)
    rater = wb["Rater"]["B1"].value
    ws = wb["Rate"]
    hdr = [c.value for c in ws[1]]
    ix = {h: n for n, h in enumerate(hdr)}
    out = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row[ix["row_id"]]:
            continue
        rec = {"rater": rater, "row_id": row[ix["row_id"]], "problem": row[ix["problem"]] or "none"}
        for d, h in zip(DIMS, ("appropriate (1-5)", "natural (1-5)", "style fit (1-5)")):
            v = row[ix[h]]
            rec[d] = int(v) if isinstance(v, (int, float)) and 1 <= v <= 5 else None
        out.append(rec)
    return out


def krippendorff_alpha_ordinal(units):
    """units: list of lists of ratings (ints 1..5), only units with >=2 ratings."""
    units = [u for u in units if len(u) >= 2]
    vals = [1, 2, 3, 4, 5]
    o = {(c, k): 0.0 for c in vals for k in vals}
    for u in units:
        m = len(u)
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[(u[i], u[j])] += 1.0 / (m - 1)
    nc = {c: sum(o[(c, k)] for k in vals) for c in vals}
    n = sum(nc.values())
    if n <= 1:
        return None

    def d2(c, k):
        lo, hi = min(c, k), max(c, k)
        s = sum(nc[g] for g in range(lo, hi + 1)) - (nc[c] + nc[k]) / 2.0
        return s * s

    do = sum(o[(c, k)] * d2(c, k) for c in vals for k in vals) / n
    de = sum(nc[c] * nc[k] * d2(c, k) for c in vals for k in vals) / (n * (n - 1))
    return None if de == 0 else 1.0 - do / de


def boot_ci(item_means, rng, reps=4000):
    """item_means: {item: value}; percentile CI of the mean across items."""
    v = list(item_means.values())
    if len(v) < 2:
        return (None, None)
    ms = sorted(sum(rng.choice(v) for _ in v) / len(v) for _ in range(reps))
    return ms[int(0.025 * reps)], ms[int(0.975 * reps) - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", required=True)
    ap.add_argument("--sheets", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=425)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    key = load_key(args.key)
    ratings = []
    for pat in args.sheets:
        for p in sorted(glob.glob(pat)):
            ratings += load_sheet(p)
    unknown = {r["row_id"] for r in ratings} - set(key)
    if unknown:
        raise SystemExit(f"row ids not in the key: {sorted(unknown)[:5]}")
    if not ratings:
        raise SystemExit("no ratings found")

    # agreement per dimension
    agree = {}
    for d in DIMS:
        per_row = defaultdict(list)
        for r in ratings:
            if r[d] is not None:
                per_row[r["row_id"]].append(r[d])
        multi = [v for v in per_row.values() if len(v) >= 2]
        pairs = [(a, b) for v in multi for a, b in combinations(v, 2)]
        agree[d] = {
            "rows_with_2plus_ratings": len(multi),
            "alpha_ordinal": krippendorff_alpha_ordinal(multi),
            "exact_agreement": (sum(a == b for a, b in pairs) / len(pairs)) if pairs else None,
            "within_1": (sum(abs(a - b) <= 1 for a, b in pairs) / len(pairs)) if pairs else None,
        }

    # per system/style: average raters within a row first, then mean over (item) values
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # (system,style)->dim->item->[row means]
    probs = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    by_row = defaultdict(list)
    for r in ratings:
        by_row[r["row_id"]].append(r)
    for rid, rs in by_row.items():
        k = key[rid]
        for s in k["systems"]:
            for style in (k["style"], "both"):
                totals[(s, style)] += len(rs)
                for r in rs:
                    probs[(s, style)][r["problem"]] += 1
                for d in DIMS:
                    vals = [r[d] for r in rs if r[d] is not None]
                    if vals:
                        cells[(s, style)][d][k["item_id"]].append(sum(vals) / len(vals))
    results = []
    for (s, style), dd in sorted(cells.items()):
        rec = {"system": s, "style": style, "ratings": totals[(s, style)]}
        allm = defaultdict(list)
        for d in DIMS:
            im = {i: sum(v) / len(v) for i, v in dd[d].items()}
            rec[d] = sum(im.values()) / len(im) if im else None
            lo, hi = boot_ci(im, rng)
            rec[d + "_ci"] = f"[{lo:.2f}, {hi:.2f}]" if lo is not None else ""
            for i, v in im.items():
                allm[i].append(v)
        im_all = {i: sum(v) / len(v) for i, v in allm.items()}
        rec["overall"] = sum(im_all.values()) / len(im_all) if im_all else None
        lo, hi = boot_ci(im_all, rng)
        rec["overall_ci"] = f"[{lo:.2f}, {hi:.2f}]" if lo is not None else ""
        bad = sum(v for p, v in probs[(s, style)].items() if p != "none")
        rec["problem_rate"] = bad / totals[(s, style)] if totals[(s, style)] else None
        results.append(rec)

    # pairwise paired difference on overall (both styles), per item
    sysitems = {}
    for (s, style), dd in cells.items():
        if style != "both":
            continue
        allm = defaultdict(list)
        for d in DIMS:
            for i, v in dd[d].items():
                allm[i].append(sum(v) / len(v))
        sysitems[s] = {i: sum(v) / len(v) for i, v in allm.items()}
    pair_rows = []
    for a, b in combinations(sorted(sysitems), 2):
        common = sorted(set(sysitems[a]) & set(sysitems[b]))
        diff = {i: sysitems[a][i] - sysitems[b][i] for i in common}
        if len(diff) < 2:
            continue
        lo, hi = boot_ci(diff, rng)
        pair_rows.append({"a": a, "b": b, "items": len(diff), "mean_diff_a_minus_b": sum(diff.values()) / len(diff),
                          "ci": f"[{lo:.2f}, {hi:.2f}]",
                          "a_wins": sum(v > 0.25 for v in diff.values()), "b_wins": sum(v < -0.25 for v in diff.values())})

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "system_scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    (args.out / "agreement.json").write_text(json.dumps(agree, indent=1), encoding="utf-8")
    f3 = lambda x: "" if x is None else f"{x:.2f}"
    md = ["# Blind rating results", "", "Scores are means over items (raters averaged within a row); CI = 95% bootstrap over items.", "",
          "## Per system", "", "| system | style | n ratings | appropriate | natural | style fit | overall | problem rate |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['system']} | {r['style']} | {r['ratings']} | {f3(r['appropriate'])} {r['appropriate_ci']} | "
                  f"{f3(r['natural'])} {r['natural_ci']} | {f3(r['style_fit'])} {r['style_fit_ci']} | "
                  f"{f3(r['overall'])} {r['overall_ci']} | {f3(r['problem_rate'])} |")
    md += ["", "## Paired differences on overall score (both styles)", "", "| A | B | items | mean A-B | 95% CI | A better (>0.25) | B better |", "|---|---|---|---|---|---|---|"]
    for r in pair_rows:
        md.append(f"| {r['a']} | {r['b']} | {r['items']} | {r['mean_diff_a_minus_b']:+.2f} | {r['ci']} | {r['a_wins']} | {r['b_wins']} |")
    md += ["", "## Inter-rater agreement", "", "| dimension | rows with 2+ raters | Krippendorff alpha (ordinal) | exact agreement | within 1 point |", "|---|---|---|---|---|"]
    for d, a in agree.items():
        md.append(f"| {d} | {a['rows_with_2plus_ratings']} | {f3(a['alpha_ordinal'])} | {f3(a['exact_agreement'])} | {f3(a['within_1'])} |")
    md += ["", "Rule of thumb: alpha < 0.4 means the rubric or the raters need calibration before the scores mean much."]
    (args.out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
