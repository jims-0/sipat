"""Measure Sipat on FIELD photos - phone pictures labelled by a farmer or the MAO.

    python scripts/evaluate_field.py sipat_field_set_20261019.zip
    python scripts/evaluate_field.py datasets/field          (folders named by class)
    python scripts/evaluate_field.py datasets/field_plantdoc --benchmark [--compare old.json]

The zip is what the app's Estadistika page downloads ("field test set"): one
folder per true class (PlantVillage folder names, as in app/ml/classes.py).
Unlike evaluate.py this runs the APP's own pipeline - the .tflite model through
LiteRT with the app's confidence threshold - so the numbers are what a farmer
actually gets. No TensorFlow needed.

Writes app/ml/model/field_report.json (shown on the Estadistika page) and
docs/FIELD_EVALUATION.md.

--benchmark is for the public PlantDoc field photos (make_plantdoc_field_set.py)
instead of the team's own: it writes app/ml/model/field_benchmark.json (shown
on the Training Evidence page) and docs/FIELD_BENCHMARK.md, so it never
overwrites the farmers' field report. --compare old.json records an earlier
model's benchmark next to the new one.
"""

import json
import sys
import tempfile
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ml import inference  # noqa: E402
from app.ml.classes import CLASSES  # noqa: E402

MIN_PER_CLASS = 30  # below this a per-class number is reported, but flagged
NON_CROP = ("Other_plant", "Background_without_leaves")
BENCHMARK_NOTE = ("PlantDoc test split (Singh et al. 2020, CODS-COMAD; CC BY 4.0): real-world leaf "
                  "photos - fields, gardens, hands, cluttered backgrounds - that the model never "
                  "trained on. Labels are PlantDoc's. Off-list crops (apple, grape, ...) count as "
                  "correct only when the app answers 'other plant'.")


def collect(folder: Path) -> list[tuple[Path, str]]:
    items, unknown = [], set()
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        if sub.name not in CLASSES:
            unknown.add(sub.name)
            continue
        items += [(f, sub.name) for f in sorted(sub.iterdir())
                  if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if unknown:
        print(f"Skipped folders that are not Sipat classes: {', '.join(sorted(unknown))}")
    return items


def evaluate(items: list[tuple[Path, str]], predict=inference.predict_disease) -> dict:
    """(photo, true class) pairs -> the report. `predict` is swappable for tests."""
    rows = []
    for path, truth in items:
        out = predict(path)
        if out["status"] == "error":
            print(f"  could not read {path.name}: {out['error']}")
            continue
        rows.append({"file": path.name, "true": truth, "predicted": out["class_key"],
                     "status": out["status"], "confidence": out["confidence"],
                     "top3": [g["class"] for g in out["top3"]]})
    n = len(rows)
    if not n:
        raise SystemExit("No readable labelled photos.")

    # "answered" = the app gave a firm answer (ok / not_leaf / other_plant)
    # rather than "unsure - retake the photo".
    answered = [r for r in rows if r["status"] != "unsure"]
    # The crops Sipat diagnoses, without the off-list/no-leaf photos that are
    # easy to get right and would lift the headline number.
    crop = [r for r in rows if r["true"] not in NON_CROP]
    per_class = []
    for key in sorted({r["true"] for r in rows}):
        mine = [r for r in rows if r["true"] == key]
        predicted_as = [r for r in rows if r["predicted"] == key]
        tp = sum(r["predicted"] == key for r in mine)
        per_class.append({
            "class": key, "name": CLASSES[key]["name"], "photos": len(mine),
            "recall": round(tp / len(mine), 4),
            "precision": round(tp / len(predicted_as), 4) if predicted_as else None,
            "unsure": sum(r["status"] == "unsure" for r in mine),
            "enough_data": len(mine) >= MIN_PER_CLASS,
        })
    mistakes = {}
    for r in rows:
        if r["predicted"] != r["true"]:
            mistakes[(r["true"], r["predicted"])] = mistakes.get((r["true"], r["predicted"]), 0) + 1
    return {
        "generated": date.today().isoformat(),
        "photos": n,
        "accuracy": round(sum(r["predicted"] == r["true"] for r in rows) / n, 4),
        "top3_accuracy": round(sum(r["true"] in r["top3"] for r in rows) / n, 4),
        "answered_share": round(len(answered) / n, 4),
        "crop_photos": len(crop),
        "crop_accuracy": (round(sum(r["predicted"] == r["true"] for r in crop) / len(crop), 4)
                          if crop else None),
        "accuracy_when_answered": (round(sum(r["predicted"] == r["true"] for r in answered) / len(answered), 4)
                                   if answered else 0.0),
        "per_class": per_class,
        "top_mistakes": [{"count": c, "true": t, "predicted": p}
                         for (t, p), c in sorted(mistakes.items(), key=lambda kv: -kv[1])[:10]],
        "rows": rows,
        "note": ("Labels come from the person who took the photo (farmer, MAO or agronomist), "
                 "entered in the app. Classes with fewer than "
                 f"{MIN_PER_CLASS} photos are reported but not yet reliable."),
    }


def write_markdown(r: dict, path: Path, title: str = "field evaluation (phone photos)") -> None:
    lab_path = inference.MODEL_DIR / "evaluation_report.json"
    lab = json.loads(lab_path.read_text()) if lab_path.exists() else None

    def vs(key):
        return f"{lab[key]:.1%}" if lab else "-"

    lines = [
        f"# Sipat - {title}",
        "",
        "Generated by `scripts/evaluate_field.py`. Do not edit by hand; re-run the script.",
        "",
        f"- {r['photos']} labelled field photos, measured {r['generated']}",
        "- Pipeline: the app itself (LiteRT .tflite, the app's confidence threshold)",
        f"- {r['note']}",
        "",
        "| | Field photos | Lab test set (evaluate.py) |",
        "|---|---|---|",
        f"| Accuracy | {r['accuracy']:.1%} | {vs('accuracy')} |",
        *([f"| Accuracy on the diagnosed crops only ({r['crop_photos']} photos) "
           f"| {r['crop_accuracy']:.1%} | - |"] if r.get("crop_accuracy") is not None else []),
        f"| Top-3 accuracy | {r['top3_accuracy']:.1%} | {vs('top3_accuracy')} |",
        f"| Answered (not \"unsure\") | {r['answered_share']:.1%} | - |",
        f"| Accuracy when answered | {r['accuracy_when_answered']:.1%} | - |",
        "",
    ]
    if r.get("before"):
        b = r["before"]
        lines += [
            "## Before / after",
            "",
            "| Model | Accuracy | Diagnosed crops only | Top-3 |",
            "|---|---|---|---|",
            f"| Before: {b['model']} | {b['accuracy']:.1%} | {b['crop_accuracy']:.1%} | {b['top3_accuracy']:.1%} |",
            f"| Now: {r['model']} | {r['accuracy']:.1%} | {r['crop_accuracy']:.1%} | {r['top3_accuracy']:.1%} |",
            "",
        ]
    lines += [
        "## Per class",
        "",
        "| Class | Photos | Recall | Precision | Unsure | Enough data? |",
        "|---|---|---|---|---|---|",
    ]
    for c in r["per_class"]:
        prec = f"{c['precision']:.2f}" if c["precision"] is not None else "-"
        enough = "yes" if c["enough_data"] else f"no (<{MIN_PER_CLASS})"
        lines.append(f"| {c['name']} | {c['photos']} | {c['recall']:.2f} | {prec} | {c['unsure']} | {enough} |")
    if r["top_mistakes"]:
        lines += ["", "## Most common mistakes", "", "| Count | True | Predicted as |", "|---|---|---|"]
        for m in r["top_mistakes"]:
            lines.append(f"| {m['count']} | {CLASSES[m['true']]['name']} | {CLASSES[m['predicted']]['name']} |")
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_items(source: Path, tmp: str) -> list[tuple[Path, str]]:
    """A class-per-folder directory, or the app's zip (extracted into tmp)."""
    if source.suffix.lower() != ".zip":
        return collect(source)
    root = Path(tmp).resolve()
    with zipfile.ZipFile(source) as zf:
        for member in zf.namelist():
            if not (root / member).resolve().is_relative_to(root):
                raise SystemExit(f"Refusing unsafe path in zip: {member}")
        zf.extractall(root)
    return collect(root)


def model_name() -> str:
    info_path = inference.MODEL_DIR / "training_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    return f"run {info.get('run', '?')} ({info.get('dataset', '?')})"


def main(argv: list[str]) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default=str(ROOT / "datasets" / "field"))
    ap.add_argument("--benchmark", action="store_true", help="public PlantDoc photos, not the team's")
    ap.add_argument("--compare", help="an earlier field_benchmark.json / field_report.json to compare with")
    args = ap.parse_args(argv[1:])
    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"{source} not found. Download the field test set from the app's Estadistika page.")
    with tempfile.TemporaryDirectory() as tmp:
        report = evaluate(load_items(source, tmp))
    report["model"] = model_name()
    if args.benchmark:
        report["note"] = BENCHMARK_NOTE
    if args.compare:
        old = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        report["before"] = {"model": old.get("model", "earlier model"), "photos": old["photos"],
                            "accuracy": old["accuracy"], "top3_accuracy": old["top3_accuracy"],
                            "crop_accuracy": old.get("crop_accuracy")}
    name, md, title = (("field_benchmark.json", "FIELD_BENCHMARK.md", "field benchmark (PlantDoc photos)")
                       if args.benchmark else
                       ("field_report.json", "FIELD_EVALUATION.md", "field evaluation (phone photos)"))
    (inference.MODEL_DIR / name).write_text(json.dumps(report, indent=1), encoding="utf-8")
    write_markdown(report, ROOT / "docs" / md, title)
    print(json.dumps({k: report[k] for k in ("photos", "accuracy", "crop_photos", "crop_accuracy",
                                              "top3_accuracy", "answered_share",
                                              "accuracy_when_answered")}, indent=1))


if __name__ == "__main__":
    main(sys.argv)
