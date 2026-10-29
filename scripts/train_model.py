"""Train the Sipat leaf-disease model (transfer learning, CPU-friendly).

    python scripts/train_model.py                      # full run (~1 h on a laptop CPU)
    python scripts/train_model.py --max-per-class 60 --head-epochs 1 --fine-epochs 0   # smoke test

Model: an ImageNet-pretrained backbone (--backbone, default MobileNetV3-Small;
the shipped model is --backbone effv2b0, see TRAINING.md)
with a new 21-way softmax head.
  1. Head only (base frozen), lr 1e-3 - learns the classes from ImageNet features.
  2. Fine-tune the top of the base, lr 1e-4 - adapts the features to leaves.
Large classes are capped (--max-per-class) and the rest weighted, so a class with
152 photos (Potato healthy) is not drowned by one with 5,357.

With field photos, the kept epoch is the one most accurate on the PlantDoc
validation photos (val_field_accuracy), not on the ~4,400 lab ones - the lab
number saturates near 95% while the field number keeps moving. The PlantDoc
TEST split (evaluate_field.py) is never used to choose anything.

Writes training_runs/<stamp>/ (history, best checkpoint, training_info.json)
and, unless --candidate, the model the app loads: app/ml/model/tanim_guard.keras
+ labels.json. Then run scripts/evaluate.py for the test-set numbers.
"""

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tg_data  # noqa: E402


BACKBONES = {  # all take 0-255 pixels (include_preprocessing) and end in a 7x7 map
    "small": ("MobileNetV3Small", "MobileNetV3Small"),
    "large": ("MobileNetV3Large", "MobileNetV3Large"),
    "effv2b0": ("EfficientNetV2B0", "EfficientNetV2-B0"),
}


def build_model(n_classes: int, backbone: str = "small"):
    import keras
    from keras import layers

    base = getattr(keras.applications, BACKBONES[backbone][0])(
        input_shape=(tg_data.IMG_SIZE, tg_data.IMG_SIZE, 3), include_top=False,
        weights="imagenet", include_preprocessing=True)
    base.trainable = False
    # Built on base.input/base.output (not as a nested layer) so Grad-CAM can
    # reach the last convolution by name.
    x = layers.GlobalAveragePooling2D(name="pool")(base.output)
    x = layers.Dropout(0.25, name="dropout")(x)
    out = layers.Dense(n_classes, activation="softmax", name="predictions")(x)
    return keras.Model(base.input, out, name="tanim_guard"), base


def field_val_callback(ds):
    """Adds val_field_accuracy / val_field_loss (PlantDoc validation photos) to
    each epoch's logs. Listed before the checkpoint so it can monitor them."""
    import keras

    class FieldVal(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            loss, acc = self.model.evaluate(ds, verbose=0)
            logs["val_field_loss"], logs["val_field_accuracy"] = loss, acc
            print(f"  field val: {acc:.3f}")

    return FieldVal()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-class", type=int, default=700)
    ap.add_argument("--head-epochs", type=int, default=6)
    ap.add_argument("--fine-epochs", type=int, default=5)
    ap.add_argument("--fine-tune-layers", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-field-photos", action="store_true",
                    help="PlantVillage only (how the 2026-10-14 model was trained)")
    ap.add_argument("--backbone", choices=sorted(BACKBONES), default="small")
    ap.add_argument("--plantdoc-repeat", type=int, default=tg_data.PLANTDOC_REPEAT)
    ap.add_argument("--candidate", action="store_true",
                    help="write only training_runs/<stamp>/, leave the app's model alone")
    ap.add_argument("--resume", type=Path,
                    help="continue fine-tuning a killed run from its best.keras (same flags)")
    args = ap.parse_args()

    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(7)
    tg_data.ensure_dirs()
    if args.resume:
        run_dir = args.resume
        done = list(csv.DictReader(open(run_dir / "history.csv", newline="")))
    else:
        run_dir = tg_data.RUNS_DIR / time.strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True)

    field = not args.no_field_photos
    split = tg_data.split_files(args.max_per_class, field_photos=field,
                                plantdoc_repeat=args.plantdoc_repeat)
    plantdoc = {k: sum(tg_data.PLANTDOC_DIR.name in p for p, _ in v) for k, v in split.items()}
    counts = Counter(label for _, label in split["train"])
    total = sum(counts.values())
    class_weight = {i: total / (len(tg_data.LABELS) * counts[i]) for i in counts}
    print({k: len(v) for k, v in split.items()}, "train photos per class:",
          min(counts.values()), "-", max(counts.values()))

    train = tg_data.load_dataset(split["train"], batch_size=args.batch_size, training=True)
    val = tg_data.load_dataset(split["val"], batch_size=args.batch_size)

    model, base = build_model(len(tg_data.LABELS), args.backbone)
    best = str(run_dir / "best.keras")
    monitor, stop_on = "val_accuracy", "val_loss"
    callbacks = []
    if field:
        field_val = [it for it in split["val"] if tg_data.PLANTDOC_DIR.name in it[0]]
        callbacks.append(field_val_callback(
            tg_data.load_dataset(field_val, batch_size=args.batch_size)))
        monitor = stop_on = "val_field_accuracy"
    # On resume, only a better epoch than the saved one may overwrite best.keras
    beat = max(float(r[monitor]) for r in done) if args.resume else None
    callbacks += [
        keras.callbacks.ModelCheckpoint(best, monitor=monitor, mode="max", save_best_only=True,
                                        initial_value_threshold=beat),
        keras.callbacks.EarlyStopping(monitor=stop_on, patience=3, restore_best_weights=True,
                                      mode="max" if stop_on.endswith("accuracy") else "min"),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv"), append=True),
    ]

    started = time.time()
    start_fine = args.head_epochs
    if args.resume:
        model.load_weights(best)
        start_fine = len(done)
        print("resuming", run_dir.name, "at epoch", start_fine + 1, "best", monitor, beat)
    else:
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
                  initial_epoch=start_fine, class_weight=class_weight,
                  callbacks=callbacks, verbose=2)

    final = keras.models.load_model(best) if Path(best).exists() else model
    meta = {
        "architecture": f"{BACKBONES[args.backbone][1]} (ImageNet) + GAP + Dropout 0.25 + Dense softmax",
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
        "dataset": "PlantVillage, Mendeley Data doi:10.17632/tywbtsjrjv.1 (without augmentation)"
                   + (" + PlantDoc train split (Singh et al. 2020, CC BY 4.0)" if field else ""),
        "plantdoc_train_photos": plantdoc["train"] // args.plantdoc_repeat if field else 0,
        "plantdoc_repeat": args.plantdoc_repeat if field else 0,
        "plantdoc_val_photos": plantdoc["val"],
        "selected_on": monitor,
    }
    if field:
        rows = list(csv.DictReader(open(run_dir / "history.csv", newline="")))
        meta["best_val_field_accuracy"] = round(max(float(r[monitor]) for r in rows), 4)
    (run_dir / "training_info.json").write_text(json.dumps(meta, indent=1))
    if args.candidate:
        print("candidate only, app model untouched:", run_dir, meta)
        return
    install(run_dir, final)


def install(run_dir: Path, final=None):
    """Make a run's best checkpoint the model the app loads."""
    import keras

    final = final or keras.models.load_model(run_dir / "best.keras")
    final.save(tg_data.MODEL_DIR / "tanim_guard.keras")
    (tg_data.MODEL_DIR / "labels.json").write_text(json.dumps(tg_data.LABELS, indent=1))
    shutil.copy(run_dir / "training_info.json", tg_data.MODEL_DIR / "training_info.json")
    shutil.copy(run_dir / "history.csv", tg_data.MODEL_DIR / "training_history.csv")
    print("saved", tg_data.MODEL_DIR / "tanim_guard.keras", "from", run_dir.name)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--install"]:  # train_model.py --install training_runs/<stamp>
        install(Path(sys.argv[2]))
    else:
        main()
