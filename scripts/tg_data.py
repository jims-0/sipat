"""Shared by prepare / train / evaluate: where the photos are and how they are split.

The split is decided by a hash of each file's name, not by a random shuffle, so
train_model.py and evaluate.py always agree on which photos are the TEST set -
and those are never seen during training. Re-running gives the same split.
"""

import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ml.classes import CLASSES  # noqa: E402

DATA_DIR = ROOT / "datasets" / "plantvillage"
# Real-world photos (PlantDoc train split, scripts/make_plantdoc_field_set.py).
# Training only: they go to train/val, never to the lab test set, and the
# PlantDoc TEST split is kept apart for evaluate_field.py.
PLANTDOC_DIR = ROOT / "datasets" / "plantdoc_train"
PLANTDOC_REPEAT = 3        # ~1,300 field photos next to ~13,000 lab ones
PLANTDOC_MAX_OTHER = 300   # PlantDoc has ~1,000 off-list crops; keep them from dominating
MODEL_DIR = ROOT / "app" / "ml" / "model"
RUNS_DIR = ROOT / "training_runs"
IMG_SIZE = 224
LABELS = list(CLASSES)  # model output order

TRAIN_PCT, VAL_PCT = 70, 15  # the remaining 15% is the test set


def bucket(filename: str) -> str:
    h = int(hashlib.md5(filename.encode("utf-8")).hexdigest(), 16) % 100
    if h < TRAIN_PCT:
        return "train"
    if h < TRAIN_PCT + VAL_PCT:
        return "val"
    return "test"


def split_files(max_train_per_class: int | None = None,
                field_photos: bool = False,
                plantdoc_repeat: int = PLANTDOC_REPEAT) -> dict[str, list[tuple[str, int]]]:
    """{"train"|"val"|"test": [(path, label_index), ...]}.

    field_photos=True (training) also mixes in PlantDoc: 85% to train (each
    repeated plantdoc_repeat times), 15% to val. The test list is the same
    PlantVillage test set either way."""
    out = {"train": [], "val": [], "test": []}
    for idx, folder in enumerate(LABELS):
        files = sorted(p for p in (DATA_DIR / folder).glob("*")
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not files:
            raise SystemExit(f"No photos for {folder} in {DATA_DIR} - run prepare_dataset.py first")
        per = {"train": [], "val": [], "test": []}
        for p in files:
            per[bucket(p.name)].append((str(p), idx))
        if max_train_per_class:
            per["train"] = per["train"][:max_train_per_class]
        for k in out:
            out[k].extend(per[k])
        if field_photos:
            extra = sorted(p for p in (PLANTDOC_DIR / folder).glob("*")
                           if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            if folder == "Other_plant":
                extra = extra[:PLANTDOC_MAX_OTHER]
            for p in extra:
                h = int(hashlib.md5(p.name.encode("utf-8")).hexdigest(), 16) % 100
                if h < 85:
                    out["train"].extend([(str(p), idx)] * plantdoc_repeat)
                else:
                    out["val"].append((str(p), idx))
    if field_photos and not PLANTDOC_DIR.exists():
        raise SystemExit(f"No {PLANTDOC_DIR} - run make_plantdoc_field_set.py first")
    return out


def load_dataset(items, *, batch_size=32, training=False, seed=7):
    import tensorflow as tf

    paths = [p for p, _ in items]
    labels = [l for _, l in items]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def _load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        return tf.cast(img, tf.float32), label  # 0-255; the model rescales itself

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _augment(img, label):
    """PlantVillage photos are tidy (one leaf, plain backdrop, even light).
    Phone photos in a field are not, so training sees flips, turns, zoom and
    lighting changes - the cheapest step towards field robustness."""
    import tensorflow as tf

    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    scale = tf.random.uniform([], 0.8, 1.0)
    crop = tf.cast(scale * IMG_SIZE, tf.int32)
    img = tf.image.random_crop(img, (crop, crop, 3))
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.image.random_brightness(img, 40.0)
    img = tf.image.random_contrast(img, 0.75, 1.25)
    img = tf.image.random_saturation(img, 0.8, 1.2)
    return tf.clip_by_value(img, 0.0, 255.0), label


def ensure_dirs():
    for d in (MODEL_DIR, RUNS_DIR):
        os.makedirs(d, exist_ok=True)
