"""Export the trained Keras model to TensorFlow Lite - what the app actually runs.

    python scripts/export_tflite.py

Why: TensorFlow itself is ~500 MB and, on Python 3.14, only a pre-release
exists, so `pip install tensorflow` fails on a fresh laptop. The app runs the
.tflite file with LiteRT (pip install ai-edge-litert, a few MB) instead. The
same .tflite is what an on-device phone app would use.

The exported model has TWO outputs: the last feature map (7x7xC) and the class
probabilities. With the dense head's weights (cam_weights.npy) that is enough
for the class activation map: CAM_c = sum_k w[k, c] * A_k (Zhou et al. 2016),
which for a GAP + single dense layer head is Grad-CAM on the class score.

Then checks the .tflite against the Keras model on test photos: same top class.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tg_data  # noqa: E402


def main():
    import keras
    import numpy as np
    import tensorflow as tf

    model = keras.models.load_model(tg_data.MODEL_DIR / "tanim_guard.keras")
    features = model.get_layer("pool").input
    two_out = keras.Model(model.inputs, [features, model.output])

    converter = tf.lite.TFLiteConverter.from_keras_model(two_out)
    # No quantisation: it shrank the file 4x but changed the top answer on ~1% of
    # test photos, and the accuracy we report must be the accuracy the app runs.
    if "--quantize" in sys.argv:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite = converter.convert()
    out_path = tg_data.MODEL_DIR / "tanim_guard.tflite"
    out_path.write_bytes(tflite)
    kernel = model.get_layer("predictions").get_weights()[0].astype(np.float32)  # [C, classes]
    np.save(tg_data.MODEL_DIR / "cam_weights.npy", kernel)

    # Parity check on a slice of the test set.
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=str(out_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]["index"]
    prob_out = next(d["index"] for d in interp.get_output_details() if len(d["shape"]) == 2)
    test = tg_data.split_files()["test"][::25]
    agree = 0
    for path, _label in test:
        img = tf.image.resize(tf.io.decode_image(tf.io.read_file(path), channels=3,
                                                 expand_animations=False), (224, 224))
        x = tf.cast(img, tf.float32)[None, ...].numpy()
        keras_top = int(model.predict(x, verbose=0)[0].argmax())
        interp.set_tensor(inp, x)
        interp.invoke()
        agree += int(interp.get_tensor(prob_out)[0].argmax() == keras_top)
    info = {"tflite_mb": round(len(tflite) / 1e6, 2), "parity_checked": len(test),
            "same_top_class": agree}
    (tg_data.MODEL_DIR / "tflite_info.json").write_text(json.dumps(info, indent=1))
    print(info)


if __name__ == "__main__":
    main()
