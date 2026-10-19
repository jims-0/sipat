"""One sample photo per disease for the Aklatan (library) page.

    python scripts/make_library_thumbs.py

Takes the first TEST-split photo (tg_data.bucket) of each crop class from
datasets/plantvillage and writes a small JPEG to app/static/library/<slug>.jpg.
PlantVillage (Mendeley Data doi:10.17632/tywbtsjrjv.1) is CC BY 4.0; the
library page carries the credit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tg_data  # noqa: E402

from PIL import Image  # noqa: E402

from app.ml.classes import CLASSES, is_diagnosis, slug  # noqa: E402

OUT = tg_data.ROOT / "app" / "static" / "library"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for key in CLASSES:
        if not is_diagnosis(key):
            continue
        photos = sorted(p for p in (tg_data.DATA_DIR / key).glob("*")
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and tg_data.bucket(p.name) == "test")
        if not photos:
            print(f"no photos for {key}")
            continue
        with Image.open(photos[0]) as img:
            img = img.convert("RGB")
            img.thumbnail((320, 320), Image.Resampling.LANCZOS)
            img.save(OUT / f"{slug(key)}.jpg", "JPEG", quality=80, optimize=True)
        print(f"{slug(key)}.jpg  <- {photos[0].name}")


if __name__ == "__main__":
    main()
