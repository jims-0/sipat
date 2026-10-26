"""PlantDoc (real-world leaf photos) -> a Sipat field test set + extra training photos.

    git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git datasets/raw/plantdoc
    python scripts/make_plantdoc_field_set.py
    python scripts/evaluate_field.py datasets/field_plantdoc

PlantDoc (Singh et al., 2020, "PlantDoc: A Dataset for Visual Plant Disease
Detection", CODS-COMAD; CC BY 4.0) is leaf photos taken outside the lab -
fields, gardens, hands, cluttered backgrounds - for the same crops Sipat
knows. Sipat never trained on any of it, so it measures what PlantVillage's
plain-background test set cannot.

PlantDoc's test/ split -> datasets/field_plantdoc/ (field evaluation only).
PlantDoc's train/ split -> datasets/plantdoc_train/ (train_model.py mixes it
in); a train photo with the same bytes as a test photo is dropped.

Some file names are web URLs with '?' in them, which Windows cannot check out,
so the photos are read from git objects and saved under numbered names.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ml.classes import CLASSES  # noqa: E402

REPO = ROOT / "datasets" / "raw" / "plantdoc"
OUT = ROOT / "datasets" / "field_plantdoc"
TRAIN_OUT = ROOT / "datasets" / "plantdoc_train"

# PlantDoc folder -> Sipat class. PlantDoc's test split has no Potato healthy,
# Corn healthy or Target Spot; those Sipat classes are simply absent.
MAP = {
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Corn Gray leaf spot": "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn___Common_rust",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
}
# Crops Sipat does not diagnose: the right answer is "other plant".
OTHER = ("Apple", "Blueberry", "Cherry", "Peach", "Raspberry", "Soyabean",
         "Squash", "Strawberry", "grape")


def target(folder: str) -> str | None:
    if folder in MAP:
        return MAP[folder]
    if folder.startswith(OTHER):
        return "Other_plant"
    return None


def listing(split: str) -> list[tuple[str, str, str]]:
    """[(blob sha, PlantDoc folder, file name)] for one split."""
    # -z: raw paths, no quoting of the odd URL-style names
    out = subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "-z", "HEAD", split],
                         capture_output=True, text=True, check=True, encoding="utf-8").stdout
    rows = []
    for line in filter(None, out.split("\0")):
        meta, path = line.split("\t", 1)
        parts = path.split("/")
        if len(parts) == 3:
            rows.append((meta.split()[2], parts[1], parts[2]))
    return rows


def export(rows, out: Path, skip_shas=frozenset()) -> None:
    counts, skipped, dupes = {}, set(), 0
    for sha, folder, name in rows:
        cls = target(folder)
        if cls is None:
            skipped.add(folder)
            continue
        if sha in skip_shas:  # same bytes as a test photo: keep it out of training
            dupes += 1
            continue
        ext = Path(name).suffix.lower()
        ext = ext if ext in (".jpg", ".jpeg", ".png") else ".jpg"
        n = counts.get(cls, 0) + 1
        counts[cls] = n
        dest = out / cls / f"plantdoc_{n:04d}{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(subprocess.run(["git", "-C", str(REPO), "cat-file", "blob", sha],
                                        capture_output=True, check=True).stdout)
    for cls in sorted(counts):
        print(f"{counts[cls]:4d}  {cls}")
    print(f"{sum(counts.values())} photos -> {out}" + (f" ({dupes} duplicates of test photos dropped)" if dupes else ""))
    if skipped:
        print(f"Not mapped: {', '.join(sorted(skipped))}")


def main() -> None:
    if not (REPO / ".git").exists():
        raise SystemExit(f"{REPO} not found - clone PlantDoc first (see the top of this file).")
    assert all(v in CLASSES for v in MAP.values())
    test = listing("test")
    print("== test split (field evaluation, never trained on)")
    export(test, OUT)
    # git blob sha = hash of the bytes, so equal sha means an identical photo
    print("== train split (extra training photos, see tg_data.PLANTDOC_DIR)")
    export(listing("train"), TRAIN_OUT, skip_shas=frozenset(sha for sha, _, _ in test))


if __name__ == "__main__":
    main()
