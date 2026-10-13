"""Unpack the PlantVillage photos Sipat trains on.

    python scripts/prepare_dataset.py datasets/raw/plantvillage.zip

Source: "Plant leaf diseases dataset without augmentation" (PlantVillage,
Mendeley Data, doi:10.17632/tywbtsjrjv.1). Takes the tomato, pepper, potato and
corn folders plus the background (no leaf) folder as they are, and pools a
fixed sample of the other species into one "Other_plant" class.

Only file names are taken from the zip, never its paths, so a bad entry
cannot write outside datasets/plantvillage.
"""

import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tg_data import DATA_DIR, LABELS  # noqa: E402

from app.ml.classes import OTHER_PLANT  # noqa: E402

OTHER_PER_SPECIES_FOLDER = 90  # x ~20 folders -> ~1,800 "other plant" photos
SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def main(zip_path: str) -> None:
    keep = set(LABELS) - {OTHER_PLANT}
    by_folder = defaultdict(list)
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            parts = name.split("/")
            if len(parts) < 3 or not parts[-1]:
                continue
            if not parts[-1].lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            by_folder[parts[-2]].append(name)

        missing = keep - set(by_folder)
        if missing:
            raise SystemExit(f"Zip lacks expected folders: {sorted(missing)}")

        written = defaultdict(int)
        for folder, members in sorted(by_folder.items()):
            members.sort()
            if folder in keep:
                target, chosen = folder, members
            else:
                # Evenly spaced, deterministic sample - same photos every run.
                step = max(1, len(members) // OTHER_PER_SPECIES_FOLDER)
                target, chosen = OTHER_PLANT, members[::step][:OTHER_PER_SPECIES_FOLDER]
            out_dir = DATA_DIR / target
            out_dir.mkdir(parents=True, exist_ok=True)
            for member in chosen:
                base = SAFE.sub("_", member.rsplit("/", 1)[-1])
                if target == OTHER_PLANT:
                    base = SAFE.sub("_", folder) + "__" + base
                dest = out_dir / base
                if not dest.exists():
                    dest.write_bytes(z.read(member))
                written[target] += 1

    for folder in LABELS:
        print(f"{written.get(folder, 0):6d}  {folder}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
