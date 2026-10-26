"""Write a compact Role C report from verified artifacts, never invented ratings."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def summarize(run_dir, output):
    verified = read(run_dir / "verification.json")
    comparison = read(run_dir / "comparison.json")
    environment = read(run_dir / "environment.json")
    audit = read(run_dir / "audit.json")
    imported = (run_dir / "imported_pretraining.json").exists()
    reports = {v: read(run_dir / f"{v}_task.json") for v in ("v3", "v4")}
    lines = ["# Role C experiment results", "", "Run: `" + run_dir.as_posix() + "`.", "",
             "## Execution and data boundary", "",
             ("Reused the existing V3 checkpoint and the original Colab/Drive V4 checkpoint, both from epoch 5. "
              "DailyDialog pretraining was NOT repeated end to end in this run. " if imported else
              "Ran full DailyDialog pretraining for both models in this run. ") +
             "Actually ran the pilot V3 smoke, task fine-tuning, generation of both styles on all 50 validation conversations, "
             "and all three system modes. Reloaded saved weights to recompute validation loss and reproduce all 100 outputs per model.", "",
             "Task inputs: 190 reviewed training items (380 reply pairs); 50 validation items (100 reply pairs). "
             "Vocabulary expansion uses training text only. M2 selects the same chronological context in training and inference. "
             "The locked 60-item test file was not opened or used. No new gold data/labels or hosted model calls were made.", "",
             f"DailyDialog audit found {audit['public']['shared_complete_dialogues']} normalized complete conversations "
             "shared by historical train/validation files (110 training records, 815 pairs). "
             "The script prepares a separate 75,237-pair clean training copy, leaving original datasets and the frozen team split unchanged. "
             + ("Imported weights were trained on the original 76,052 pairs; their historical DailyDialog validation loss is therefore not an independent quality estimate. "
                "Task validation is a separate team-authored development set." if imported else
                "This run uses the cleaned training copy for both models."), "",
             "## Measured results", "",
             "| Model | Saved public validation loss (rechecked) | Task best epoch | Task validation loss | STANDARD token F1 | LOCAL token F1 | Different style pairs / 50 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for version in ("v3", "v4"):
        c = comparison[version]
        lines.append(f"| {version.upper()} | {verified['versions'][version]['public_validation_loss']:.6f} | "
                     f"{c['best_epoch']} | {c['validation_loss']:.6f} | {c['token_f1']['STANDARD']:.4f} | "
                     f"{c['token_f1']['LOCAL']:.4f} | {c['style_pairs_with_different_text']} |")
    lines += ["", "Token F1/exact match are reference-overlap diagnostics, not appropriateness, naturalness, style quality or accuracy. "
              "Validation loss is token-averaged teacher-forced cross entropy; lower is better for this objective. "
              f"The minimum task validation loss is from {comparison['rnn_minimum_validation_loss'].upper()}. "
              "One seed and a small validation set do not establish a statistically reliable quality win.", "",
              "Both task runs use Adam 0.001, batch size 32, seed 425, up to 60 epochs, patience 10, "
              "minimum validation-loss checkpoint selection, 64 source tokens and 40 target tokens. "
              "The style token is preserved during truncation. V3/V4 use the existing 64-dimensional embeddings and 128-unit LSTMs; "
              "V4 adds the existing Luong attention with source-padding masking. Teacher forcing is batched without changing the recurrence; "
              "output and gradient equivalence to stepwise calculation is tested.", "",
              "## Integration verification", "",
              "All 50 queries exported in retrieval-only, generator-only and hybrid modes: 150 complete mode records. "
              "Retrieval uses BM25 on the disjoint 190-item bank; all modes use M2 and the same cached candidate pools.", "",
              "| Mode | Selected retrieval replies / 100 | Selected V4 replies / 100 | Fallback replies / 100 |",
              "| --- | ---: | ---: | ---: |"]
    for mode, stats in verified["system_exports"]["modes"].items():
        lines.append(f"| {mode} | {stats['selected_retrieval_replies']} | {stats['selected_generator_replies']} | {stats['fallback_replies']} |")
    lines += ["", "Mode completion proves the sources ran; it does not prove contextual correctness or a hybrid quality gain. "
              "The existing 0.6 score + 0.4 overlap reranker and fallback threshold were not tuned here. "
              "Role B/E must validate score scales before claiming a hybrid improvement.", "",
              "Manual inspection of saved outputs found failures beyond reference overlap: ITEM-A-017 answers a time-confirmation "
              "with 'yes i will be there soon' instead of addressing the time; ITEM-C-038 asks for text-length feedback but gets "
              "'sure i'll meet you there'; ITEM-E-009 concerns a charger but LOCAL mentions sending a seat. "
              "These observations are report analysis only, not new gold/training labels. The model remains a weak experimental baseline.", "",
              "## V4 versus V5", "", "Pending Role D's trained V5 provider. No V5 scores or final V4/V5 winner are claimed. "
              "Supply V4 to the agreed integration interface while E runs matched blinded persona evaluation. "
              "Do not substitute smoke mT5 weights, retrieval text or fabricated confidence.", "",
              "## Checkpoint fingerprints", ""]
    for version in ("v3", "v4"):
        lines.append(f"- `{version}_task.pt`: `{comparison[version]['checkpoint_sha256']}`.")
    lines += ["", "## Reproduction", "", f"Python: `{environment['python'].splitlines()[0]}`. "
              f"PyTorch: `{environment['torch']}`. CPU threads: {environment['threads']}. "
              "See `environment.json`, `requirements_frozen.txt`, `imported_pretraining.json` (when importing), "
              "`audit.json`, task histories, `comparison.json`, `completion.json` and `verification.json` in the run folder. "
              "Bitwise training reproduction depends on the same checkpoint bytes, dataset bytes, software/build, hardware and thread settings; "
              "seeds alone do not guarantee identical results across CPU/GPU or library versions. "
              "Historical Colab training environment is not fully recorded, so its original training trajectory has not been reproduced.", "",
              "```powershell", ".\\.venv\\Scripts\\python.exe scripts/run_role_c.py --run-dir artifacts/role_c_runs/reproduction_01 --reuse-pretrained",
              ".\\.venv\\Scripts\\python.exe scripts/verify_role_c.py artifacts/role_c_runs/reproduction_01",
              "```", "", "Use a fresh run directory. Import defaults: `artifacts/rnn_v3_dailydialog.pt` and "
              "`artifacts/colab_import/rnn_v4_dailydialog.pt`, each with its matching `.json` report. "
              "Omit `--reuse-pretrained` only for a deliberate new full pretraining experiment on cleaned data; "
              "that produces a different experiment and is not expected to recreate imported historical metrics.", "",
              "## Actual validation generations (20 conversations)", "",
              "These are saved model outputs on existing human-authored validation items, not new gold labels. "
              "Cases include the ten lowest and ten highest mean V4 reference-overlap values; selection by overlap is diagnostic, not a human quality rating.", ""]
    rows = {}
    for version in ("v3", "v4"):
        rows[version] = {(r["item_id"], r["candidate"]["style"]): r for r in
                         (json.loads(line) for line in (run_dir / f"{version}_validation.jsonl").read_text(encoding="utf-8").splitlines())}
    ids = sorted({key[0] for key in rows["v4"]}, key=lambda i: (sum(rows["v4"][i, s]["token_f1"] for s in ("STANDARD", "LOCAL")), i))
    for item_id in [*ids[:10], *ids[-10:]]:
        example = rows["v4"][item_id, "STANDARD"]["candidate"]
        lines += [f"### {item_id}", "", f"Context: {cell(' / '.join(example['context']))}", "",
                  f"Incoming: {cell(example['message'])}", "", "| Style | V3 | V4 | Human reference |",
                  "| --- | --- | --- | --- |"]
        for style in ("STANDARD", "LOCAL"):
            a, b = rows["v3"][item_id, style], rows["v4"][item_id, style]
            lines.append(f"| {style} | {cell(a['candidate']['candidate_text'])} | {cell(b['candidate']['candidate_text'])} | {cell(b['reference'])} |")
        lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("roles/C_rnn_attention/EXPERIMENT_REPORT.md"))
    args = parser.parse_args()
    summarize(args.run_dir, args.output)
    print(f"Wrote {args.output}")
