"""
Build a blind human-rating kit from system outputs on the SAME items.
"""
import argparse
import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

PROBLEMS = ["none", "garbled_or_unreadable", "offensive_or_unsafe", "personal_info", "generic_or_unrelated_fallback"]
SPEAKER = {"user": "You", "peer": "Friend"}


def dotted(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def same_person(a, b):
    a, b = a.strip().lower(), b.strip().lower()
    return a == b or a.split()[0] == b.split()[0]


def load_system(spec, id_field, std_field, loc_field):
    name, rest = spec.split("=", 1)
    path, _, filt = rest.partition(":")
    flt = dict(kv.split("=", 1) for kv in filt.split(",") if kv)
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if any(str(dotted(row, k)) != v for k, v in flt.items()):
            continue
        iid, st, lo = dotted(row, id_field), dotted(row, std_field), dotted(row, loc_field)
        if iid is None or not isinstance(st, str) or not isinstance(lo, str) or not st.strip() or not lo.strip():
            raise SystemExit(f"{name}: row without usable id/standard/local ({id_field},{std_field},{loc_field}): "
                             f"{json.dumps(row, ensure_ascii=False)[:300]}")
        if iid in out:
            raise SystemExit(f"{name}: several rows for {iid}; add a filter such as :mode=hybrid")
        out[iid] = {"standard": st.strip(), "local": lo.strip()}
    return name, out


def conversation(item):
    lines = [f"{SPEAKER.get(m['speaker'], m['speaker'])}: {m['text']}" for m in item["session_memory"]]
    return "\n".join(lines)


def write_sheet(path, rater, rows, rubric_text):
    wb = Workbook()
    ws = wb.active
    ws.title = "Rate"
    head = ["#", "row_id", "conversation so far", "friend's new message", "style label", "suggested reply",
            "appropriate (1-5)", "natural (1-5)", "style fit (1-5)", "problem", "comment"]
    ws.append(head)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="305496")
        c.alignment = Alignment(wrap_text=True, vertical="center")
    for n, r in enumerate(rows, 1):
        ws.append([n, r["row_id"], r["conversation"], r["message"], r["style"].upper(), r["reply"],
                   None, None, None, None, None])
    for col, w in zip("ABCDEFGHIJK", (5, 11, 42, 34, 11, 40, 12, 12, 12, 24, 30)):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
        for c in row[6:9]:
            c.fill = PatternFill("solid", fgColor="FFF2CC")
        row[9].fill = PatternFill("solid", fgColor="FFF2CC")
    ws.freeze_panes = "C2"
    n_last = max(len(rows) + 1, 2)
    dv = DataValidation(type="whole", operator="between", formula1="1", formula2="5", allow_blank=True,
                        showErrorMessage=True, errorTitle="1-5 only", error="Enter a whole number from 1 to 5.")
    dv.add(f"G2:I{n_last}")
    dp = DataValidation(type="list", formula1='"' + ",".join(PROBLEMS) + '"', allow_blank=True)
    dp.add(f"J2:J{n_last}")
    ws.add_data_validation(dv)
    ws.add_data_validation(dp)
    rb = wb.create_sheet("Rubric")
    for line in rubric_text.splitlines():
        rb.append([line])
    rb.column_dimensions["A"].width = 130
    for row in rb.iter_rows():
        row[0].alignment = Alignment(wrap_text=True, vertical="top")
    info = wb.create_sheet("Rater")
    info.append(["rater", rater])
    info.append(["Do not edit the row_id column. Save as .xlsx and send it back unchanged in layout."])
    wb.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--system", action="append", default=[], help="NAME=FILE[:k=v,...]")
    ap.add_argument("--raters", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--rubric", type=Path, default=Path(__file__).with_name("rating_rubric.md"))
    ap.add_argument("--no-reference", action="store_true", help="omit the team-authored reply as a hidden system 'REF'")
    ap.add_argument("--per-item-raters", type=int, default=2)
    ap.add_argument("--items", type=int, help="random subset of this many items (default all)")
    ap.add_argument("--calibration", action="store_true",
                    help="everyone rates the same rows; author/reviewer exclusion is waived. Discard these scores.")
    ap.add_argument("--seed", type=int, default=425)
    ap.add_argument("--id-field", default="item_id")
    ap.add_argument("--standard-field", default="standard")
    ap.add_argument("--local-field", default="local")
    ap.add_argument("--final-test-ack", action="store_true",
                    help="required when --queries holds split=test items: final, once-only evaluation")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    items = [json.loads(l) for l in args.queries.read_text(encoding="utf-8").splitlines() if l.strip()]
    if any(i.get("split") == "test" for i in items) and not args.final_test_ack:
        raise SystemExit("These queries contain held-out test items. Rate them only once, after systems are frozen; "
                         "pass --final-test-ack to confirm.")
    if args.items:
        items = sorted(rng.sample(items, args.items), key=lambda i: i["id"])
    byid = {i["id"]: i for i in items}
    raters = [r.strip() for r in args.raters.split(",") if r.strip()]

    systems = {}
    for spec in args.system:
        name, out = load_system(spec, args.id_field, args.standard_field, args.local_field)
        missing = sorted(set(byid) - set(out))
        if missing:
            raise SystemExit(f"{name} lacks outputs for {len(missing)} items, e.g. {missing[:5]}. "
                             "Systems must be compared on identical items.")
        systems[name] = out
    if not args.no_reference:
        systems["REF"] = {i["id"]: {"standard": i["replies"]["standard"].strip(), "local": i["replies"]["local"].strip()}
                          for i in items}
    if len(systems) < 2:
        raise SystemExit("need at least two systems (or one system plus the reference)")

    # one blind row per distinct (item, style, text); identical text from several systems is rated once
    used_ids, rows = set(), []
    for iid in sorted(byid):
        for style in ("standard", "local"):
            by_text = defaultdict(list)
            for sname, out in systems.items():
                by_text[re.sub(r"\s+", " ", out[iid][style]).strip()].append(sname)
            for text, snames in by_text.items():
                while True:
                    rid = "R" + "".join(rng.choice("0123456789ABCDEFGHJKMNPQRSTVWXYZ") for _ in range(6))
                    if rid not in used_ids:
                        used_ids.add(rid)
                        break
                rows.append({"row_id": rid, "item_id": iid, "style": style, "reply": text, "systems": sorted(snames),
                             "conversation": conversation(byid[iid]), "message": byid[iid]["message"]["text"]})

    # assign raters to items
    load = {r: 0 for r in raters}
    rows_per_item = defaultdict(int)
    for r in rows:
        rows_per_item[r["item_id"]] += 1
    assign = {}
    if args.calibration:
        for iid in byid:
            assign[iid] = list(raters)
    else:
        order = sorted(byid)
        rng.shuffle(order)
        for iid in order:
            it = byid[iid]
            elig = [r for r in raters if not same_person(r, it["author"]) and not same_person(r, it.get("reviewer", ""))
                    and not same_person(r, it.get("second_reviewer") or "-")]
            if len(elig) < args.per_item_raters:
                raise SystemExit(f"{iid}: only {len(elig)} eligible raters (author/reviewer excluded)")
            rng.shuffle(elig)
            elig.sort(key=lambda r: load[r])
            pick = elig[:args.per_item_raters]
            for r in pick:
                load[r] += rows_per_item[iid]
            assign[iid] = pick

    rubric_text = args.rubric.read_text(encoding="utf-8") if args.rubric.exists() else "See rating_rubric.md"
    sheets = args.out_dir / "rater_sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    counts = {}
    for rater in raters:
        mine = [i for i in assign if rater in assign[i]]
        rng.shuffle(mine)  # separate shuffle per rater
        rws = []
        for iid in mine:
            block = [r for r in rows if r["item_id"] == iid]
            rng.shuffle(block)
            rws += block
        counts[rater] = len(rws)
        write_sheet(sheets / (re.sub(r"[^A-Za-z0-9]+", "_", rater).strip("_") + ".xlsx"), rater, rws, rubric_text)

    with (args.out_dir / "PRIVATE_rating_key.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "item_id", "style", "systems", "reply"])
        for r in rows:
            w.writerow([r["row_id"], r["item_id"], r["style"], "|".join(r["systems"]), r["reply"]])
    with (args.out_dir / "PRIVATE_assignments.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "author", "reviewer", "raters"])
        for iid in sorted(assign):
            w.writerow([iid, byid[iid]["author"], byid[iid].get("reviewer", ""), "|".join(assign[iid])])
    man = {"seed": args.seed, "queries": str(args.queries), "n_items": len(byid), "systems": sorted(systems),
           "calibration": args.calibration, "per_item_raters": args.per_item_raters, "rows_per_rater": counts,
           "distinct_rows": len(rows),
           "queries_sha256": hashlib.sha256(args.queries.read_bytes()).hexdigest()}
    (args.out_dir / "kit_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
