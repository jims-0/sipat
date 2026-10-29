# Training the Sipat model

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

PlantDoc (Singh et al. 2020, "PlantDoc: A Dataset for Visual Plant Disease
Detection", CODS-COMAD; CC BY 4.0, ~930 MB) - real-world leaf photos for the
same crops, taken in fields and gardens:

    git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git datasets\raw\plantdoc
    python scripts\make_plantdoc_field_set.py

This maps PlantDoc's classes to Sipat's (other crops -> Other_plant) and writes
its **train** split to `datasets/plantdoc_train/` (mixed into training) and its
**test** split to `datasets/field_plantdoc/` (the field benchmark, never
trained on). A train photo with the same bytes as a test photo is dropped (11
were). Some PlantDoc file names contain `?`, which Windows cannot check out -
the script reads them straight from git, so none are lost.

`datasets/` is git-ignored - it is large and re-downloadable.

## 2. Install the training packages

    pip install -r requirements-train.txt          # Python 3.10-3.12
    pip install --pre -r requirements-train.txt    # Python 3.14 (only a TF pre-release exists)

## 3. Train, evaluate, export

    python scripts\train_model.py        # ~20-30 min on a laptop CPU (no GPU needed)
    python scripts\evaluate.py           # lab test-set numbers -> docs\EVALUATION.md
    python scripts\export_tflite.py      # what the app runs -> app\ml\model\tanim_guard.tflite
    python scripts\evaluate_field.py datasets\field_plantdoc --benchmark --compare docs\training_evidence\field_benchmark_before_plantdoc_20261014.json
                                         # real-world photos -> docs\FIELD_BENCHMARK.md

`train_model.py --no-field-photos` trains on PlantVillage only, the way the
first model (run 20261014_010227) was trained.

| Step | Writes |
|---|---|
| train_model.py | `app/ml/model/tanim_guard.keras`, `labels.json`, `training_info.json`, `training_history.csv`; full run log in `training_runs/` |
| evaluate.py | `app/ml/model/evaluation_report.json` (the app reads its confidence threshold here), `docs/EVALUATION.md`, `docs/confusion_matrix.png` |
| export_tflite.py | `app/ml/model/tanim_guard.tflite`, `cam_weights.npy`, `tflite_info.json` |

## How the model is built

- **Transfer learning**: we did not design the network. We start from
  **EfficientNetV2-B0** (Tan & Le 2021), already pretrained on ImageNet
  (1.2 million general photos), and re-train it on leaf photos. This is the
  standard method when there are thousands, not millions, of labelled photos.
  224x224 input. Chosen on 2026-10-29 from three candidates (MobileNetV3-Small,
  MobileNetV3-Large, EfficientNetV2-B0) by accuracy on the PlantDoc validation
  photos: 63.4 %, 66.8 %, 68.3 %. Still fast enough on ordinary laptops and phones.
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
- **Real-world photos in training**: the PlantDoc train split (1,436 photos
  used, the "other crop" ones capped at 300) is mixed in, each shown 3 times
  per epoch so it is not drowned by ~13,000 lab photos. 85 % go to training,
  15 % to validation. This roughly doubled accuracy on real-world photos (see
  "Measuring on field photos") without lowering the lab number.

## The split

70 % train / 15 % validation / 15 % test, decided by an MD5 hash of each file
name. Validation picks the best epoch; **the test photos are used only by
evaluate.py**, after training is finished. PlantDoc photos never enter the lab
test set, so the lab number stays comparable with the first model.

## Measuring on field photos

The lab test set says nothing about phone photos in a real field.

### Public benchmark: PlantDoc test split (done)

236 real-world photos the model never trained on (128 of the 4 crops Sipat
diagnoses + 108 of other crops, where the right answer is "other plant"), run
through the app's own pipeline (LiteRT, the app's threshold):

| | PlantVillage only (run 20261014_010227) | + PlantDoc, MobileNetV3-Small (run 20261027_001851) | + PlantDoc, EfficientNetV2-B0 (run 20261029_220930, current) |
|---|---|---|---|
| Lab test set, 4,454 photos | 95.0 % | 95.1 % | 94.1 % |
| Real-world photos of the 4 crops, 128 | 20.3 % | 43.8 % | **50.8 %** |
| All 236 real-world photos | 47.0 % | 58.5 % | 62.7 % |
| Real-world, right answer in top 3 | 68.6 % | 85.6 % | 90.3 % |
| Real-world, accuracy when it answers (not "unsure") | 57.8 % | 79.1 % | 76.3 % |

Full report: `docs/FIELD_BENCHMARK.md`. The before numbers are kept in
`docs/training_evidence/field_benchmark_before_plantdoc_20261014.json`.
PlantDoc's per-disease counts are small (4-12 photos), so read the totals,
not single diseases.

### Your own field photos

To measure on photos the team takes:

1. Use the app on real plants. On each result, tap 👍 if it was right, or 👎 and
   pick what it really was (best confirmed by the MAO or an agronomist). An
   "unsure" result can be labelled too.
2. On **Estadistika**, download the field test set (`sipat_field_set_<date>.zip`:
   one folder per true class + `labels.csv`).
3. Run the app's own pipeline (LiteRT, the app threshold - no TensorFlow) on it:

       python scripts\evaluate_field.py sipat_field_set_20261019.zip

   It writes `docs/FIELD_EVALUATION.md` (field vs lab, per class, mistakes) and
   `app/ml/model/field_report.json`, which the Estadistika page then shows.

Aim for at least 30 photos per class; smaller classes are reported but flagged.
Do not use PlantVillage photos here - that would only repeat the lab number.

## Limits to state honestly

- **Lab vs field**: 94 % is on PlantVillage photos (single leaf, plain
  background, even light). On real-world photos (PlantDoc) it is **50.8 %**
  for the 4 crops, 90.3 % with the top 3. Say both numbers. The app's
  "unsure, retake the photo" answer exists because of this: when it does
  answer firmly on a field photo it is right 76 % of the time.
- PlantDoc photos are web images, not photos from Philippine farms. Photos from
  Philippine farms, labelled by the MAO or an agriculturist, are the next
  step (see "Your own field photos").
- No rice: PlantVillage has none. To add rice, put photos in
  `datasets/plantvillage/<Rice_class_name>/`, add the classes to
  `app/ml/classes.py` and `app/ml/treatment_data.py`, and retrain.
