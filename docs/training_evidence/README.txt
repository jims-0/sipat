Sipat - raw evidence from the training runs
===========================================

CURRENT MODEL (the one the app runs): run 20261027_001851, 27 Oct 2026,
laptop CPU, 17.8 minutes. PlantVillage + PlantDoc train split.

training_log_20261027_001851.txt
    Console output of scripts/train_model.py, unedited: photo counts, all 11
    epochs (1-6 base frozen, 7-11 fine-tuning) with training and validation
    accuracy and loss, and the saved model's settings at the end.

FIRST MODEL (replaced): run 20261014_010227, 14 Oct 2026, 28.1 minutes,
PlantVillage only.

training_log_20261014_010227.txt
    Its console output, same format.
field_benchmark_before_plantdoc_20261014.json
    That model's score on the PlantDoc real-world test photos (20.3 % on the
    4 crops) - the "before" number in docs/FIELD_BENCHMARK.md.

Where the other numbers are:
    app/ml/model/training_info.json        current run's settings
    app/ml/model/training_history.csv      current run, per epoch (the app's chart)
    app/ml/model/evaluation_report.json    lab test set (docs/EVALUATION.md)
    app/ml/model/field_benchmark.json      real-world photos (docs/FIELD_BENCHMARK.md)
    docs/confusion_matrix.png              lab test set

The trained model is app/ml/model/tanim_guard.tflite. The photos are not
included: PlantVillage and PlantDoc are public, see TRAINING.md step 1.
