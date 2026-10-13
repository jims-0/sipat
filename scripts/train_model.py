"""Train the Tanim Guard leaf-disease model (transfer learning, CPU-friendly).

    python scripts/train_model.py                      # full run (~1 h on a laptop CPU)
    python scripts/train_model.py --max-per-class 60 --head-epochs 1 --fine-epochs 0   # smoke test

Model: MobileNetV3-Small pretrained on ImageNet, with a new 21-way softmax head.
  1. Head only (base frozen), lr 1e-3 - learns the classes from ImageNet features.
  2. Fine-tune the top of the base, lr 1e-4 - adapts the features to leaves.
Large classes are capped (--max-per-class) and the rest weighted, so a class with
152 photos (Potato healthy) is not drowned by one with 5,357.

Writes training_runs/<stamp>/ (history, best checkpoint) and, when done, the
model the app loads: app/ml/model/tanim_guard.keras + labels.json.
Then run scripts/evaluate.py for the test-set numbers.
"""

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tg_data  # noqa: E402


def build_model(n_classes: int):
    import keras
    from keras import layers

    base = keras.applications.MobileNetV3Small(
        input_shape=(tg_data.IMG_SIZE, tg_data.IMG_SIZE, 3), include_top=False,
        weights="imagenet", include_preprocessing=True)  # takes 0-255 pixels
    base.trainable = False
    # Built on base.input/base.output (not as a nested layer) so Grad-CAM can
    # reach the last convolution by name.
    x = layers.GlobalAveragePooling2D(name="pool")(base.output)
    x = layers.Dropout(0.25, name="dropout")(x)
    out = layers.Dense(n_classes, activation="softmax", name="predictions")(x)
    return keras.Model(base.input, out, name="tanim_guard"), base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-class", type=int, default=700)
    ap.add_argument("--head-epochs", type=int, default=6)
    ap.add_argument("--fine-epochs", type=int, default=5)
    ap.add_argument("--fine-tune-layers", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(7)
    tg_data.ensure_dirs()
    run_dir = tg_data.RUNS_DIR / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)

    split = tg_data.split_files(args.max_per_class)
    counts = Counter(label for _, label in split["train"])
    total = sum(counts.values())
    class_weight = {i: total / (len(tg_data.LABELS) * counts[i]) for i in counts}
    print({k: len(v) for k, v in split.items()}, "train photos per class:",
          min(counts.values()), "-", max(counts.values()))

    train = tg_data.load_dataset(split["train"], batch_size=args.batch_size, training=True)
    val = tg_data.load_dataset(split["val"], batch_size=args.batch_size)

    model, base = build_model(len(tg_data.LABELS))
    best = str(run_dir / "best.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(best, monitor="val_accuracy", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv"), append=True),
    ]

    started = time.time()
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train, validation_data=val, epochs=args.head_epochs,
              class_weight=class_weight, callbacks=callbacks, verbose=2)

    if args.fine_epochs:
        base.trainable = True
        for layer in base.layers[:-args.fine_tune_layers]:
            layer.trainable = False
        for layer in base.layers:  # BatchNorm stays frozen while fine-tuning small data
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False
        model.compile(optimizer=keras.optimizers.Adam(1e-4),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(train, validation_data=val, epochs=args.head_epochs + args.fine_epochs,
                  initial_epoch=args.head_epochs, class_weight=class_weight,
                  callbacks=callbacks, verbose=2)

    final = keras.models.load_model(best) if Path(best).exists() else model
    final.save(tg_data.MODEL_DIR / "tanim_guard.keras")
    (tg_data.MODEL_DIR / "labels.json").write_text(json.dumps(tg_data.LABELS, indent=1))
    meta = {
        "architecture": "MobileNetV3Small (ImageNet) + GAP + Dropout 0.25 + Dense softmax",
        "image_size": tg_data.IMG_SIZE,
        "classes": len(tg_data.LABELS),
        "train_photos": len(split["train"]), "val_photos": len(split["val"]),
        "test_photos": len(split["test"]),
        "max_train_per_class": args.max_per_class,
        "head_epochs": args.head_epochs, "fine_epochs": args.fine_epochs,
        "fine_tune_layers": args.fine_tune_layers,
        "training_minutes": round((time.time() - started) / 60, 1),
        "tensorflow": tf.__version__, "keras": keras.__version__,
        "run": run_dir.name,
        "dataset": "PlantVillage, Mendeley Data doi:10.17632/tywbtsjrjv.1 (without augmentation)",
    }
    (tg_data.MODEL_DIR / "training_info.json").write_text(json.dumps(meta, indent=1))
    shutil.copy(run_dir / "history.csv", tg_data.MODEL_DIR / "training_history.csv")
    print("saved", tg_data.MODEL_DIR / "tanim_guard.keras", meta)


if __name__ == "__main__":
    main()
