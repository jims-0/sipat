# Training the Tanim Guard model

Everything here is reproducible: the same commands on the same dataset give the
same train/validation/test split (it is decided by a hash of each file name).

## 1. Get the data

PlantVillage, "Plant leaf diseases dataset without augmentation"
(Mendeley Data, doi:10.17632/tywbtsjrjv.1, ~870 MB):

    mkdir datasets\raw
    curl -L -o datasets\raw\plantvillage.zip "https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded"
    python scripts\prepare_dataset.py datasets\raw\plantvillage.zip

`prepare_dataset.py` keeps the tomato, pepper, potato and corn folders plus
PlantVillage's background (no leaf) folder, and pools an even sample of the
other species (apple, grape, soybean...) into one **Other_plant** class, so a
leaf the model does not know is not forced into a tomato disease.

`datasets/` is git-ignored - it is large and re-downloadable.

## 2. Install the training packages

    pip install -r requirements-train.txt          # Python 3.10-3.12
    pip install --pre -r requirements-train.txt    # Python 3.14 (only a TF pre-release exists)

## 3. Train, evaluate, export

    python scripts\train_model.py        # ~30 min on a laptop CPU (no GPU needed)
    python scripts\evaluate.py           # test-set numbers -> docs\EVALUATION.md
    python scripts\export_tflite.py      # what the app runs -> app\ml\model\tanim_guard.tflite

| Step | Writes |
|---|---|
| train_model.py | `app/ml/model/tanim_guard.keras`, `labels.json`, `training_info.json`, `training_history.csv`; full run log in `training_runs/` |
| evaluate.py | `app/ml/model/evaluation_report.json` (the app reads its confidence threshold here), `docs/EVALUATION.md`, `docs/confusion_matrix.png` |
| export_tflite.py | `app/ml/model/tanim_guard.tflite`, `cam_weights.npy`, `tflite_info.json` |

## How the model is built

- **MobileNetV3-Small**, pretrained on ImageNet, 224x224 input. Chosen for
  speed on ordinary laptops and phones.
- New head: global average pooling, dropout 0.25, dense softmax over 21 classes
  (19 crop classes + "not a leaf" + "other plant").
- **Stage 1** (6 epochs): base frozen, Adam lr 1e-3 - learn the classes from
  ImageNet features.
- **Stage 2** (5 epochs): top 40 layers of the base unfrozen (BatchNorm kept
  frozen), Adam lr 1e-4 - adapt the features to leaves.
- Large classes are capped at 700 training photos and small ones weighted, so
  Potato healthy (152 photos in total) is not drowned by Yellow Leaf Curl (5,357).
- Training photos are randomly flipped, rotated, zoomed and changed in
  brightness, contrast and saturation - a cheap step towards phone photos.

## The split

70 % train / 15 % validation / 15 % test, decided by an MD5 hash of each file
name. Validation picks the best epoch; **the test photos are used only by
evaluate.py**, after training is finished.

## Limits to state honestly

- PlantVillage photos are single leaves on plain backgrounds under even light.
  Accuracy on real field photos is expected to be lower and has **not been
  measured** yet. The next step is a small field test set (photos taken by
  the team, labelled by an agriculturist) evaluated the same way.
- No rice: PlantVillage has none. To add rice, put photos in
  `datasets/plantvillage/<Rice_class_name>/`, add the classes to
  `app/ml/classes.py` and `app/ml/treatment_data.py`, and retrain.
