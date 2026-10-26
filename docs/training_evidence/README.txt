Sipat - raw evidence from the training run that produced the shipped model
==========================================================================

Run 20261014_010227, 14 Oct 2026, 01:02-01:30, laptop CPU, 28.1 minutes.

training_log_20261014_010227.txt
    The console output of scripts/train_model.py, unedited. It shows the
    photo counts (13,240 train / 4,383 validation / 4,454 test), all 11
    epochs with training and validation accuracy and loss (epochs 1-6 with
    the base frozen, 7-11 fine-tuning), and the saved model's settings at
    the end. The TensorFlow warnings at the top are normal.

The same per-epoch numbers are in app/ml/model/training_history.csv (the app
draws its chart from it), and the settings in app/ml/model/training_info.json.

Test-set results (scripts/evaluate.py, run after training, on photos the
model never saw): docs/EVALUATION.md, docs/confusion_matrix.png and
app/ml/model/evaluation_report.json.

The trained model itself is app/ml/model/tanim_guard.tflite (what the app
runs). The photos are not included - PlantVillage is public and
re-downloadable, see TRAINING.md step 1.
