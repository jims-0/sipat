"""Heat map: which parts of the photo drove the model's answer.

Class Activation Mapping (Zhou et al. 2016). The model ends in global average
pooling and one dense layer, so the score for class c is a weighted sum of the
pooled feature channels; un-pooling that sum gives where on the photo the
evidence for c sits:  CAM_c(x, y) = sum_k w[k, c] * A_k(x, y).
For this architecture that is exactly Grad-CAM on the class score
(Selvaraju et al. 2017), computed without TensorFlow.

The previous version drew a fixed blob left of centre on every photo. This one
is computed from the model for the photo in hand - if the heat sits on the
background instead of the lesions, that is real information about the model.
"""

import numpy as np
from PIL import Image, ImageOps

from app.ml import inference


def heatmap(features: np.ndarray, class_idx: int) -> np.ndarray:
    """[h, w] map in 0..1 from the feature map [h, w, C]."""
    weights = inference.load()["cam_weights"][:, class_idx]
    cam = np.maximum(features @ weights, 0)
    peak = cam.max()
    return cam / peak if peak > 0 else cam


def _colorize(cam: np.ndarray) -> np.ndarray:
    """0..1 -> RGB, blue (low) through yellow to red (high)."""
    r = np.clip(1.5 - np.abs(4 * cam - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * cam - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * cam - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def generate_gradcam(input_image_path, output_image_path, class_idx) -> bool:
    """Save the overlay. False (and no file) if it cannot be computed - the page
    then shows no heat map rather than a misleading copy of the photo."""
    try:
        _probs, features = inference.run(input_image_path)
        cam = heatmap(features, class_idx)
        with Image.open(input_image_path) as src:
            photo = ImageOps.exif_transpose(src).convert("RGB")
        overlay = Image.fromarray(_colorize(cam)).resize(photo.size, Image.Resampling.BICUBIC)
        # Weight the blend by the heat itself so cool areas keep the real photo.
        alpha = np.asarray(Image.fromarray((cam * 255).astype(np.uint8))
                           .resize(photo.size, Image.Resampling.BICUBIC), dtype=np.float32)[..., None] / 255
        blended = (np.asarray(photo, np.float32) * (1 - 0.55 * alpha)
                   + np.asarray(overlay, np.float32) * 0.55 * alpha)
        Image.fromarray(blended.clip(0, 255).astype(np.uint8)).save(output_image_path, quality=85)
        return True
    except Exception:
        return False
