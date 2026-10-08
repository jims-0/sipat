"""
TensorFlow Lite Model Inference
"""
import os
import numpy as np
from PIL import Image

# Disease mapping (PlantVillage dataset classes relevant to Philippine crops)
DISEASE_CLASSES = {
    0: {'name': 'Tomato Late Blight', 'name_tl': 'Late Blight sa Kamatis', 'crop': 'tomato'},
    1: {'name': 'Tomato Early Blight', 'name_tl': 'Early Blight sa Kamatis', 'crop': 'tomato'},
    2: {'name': 'Tomato Leaf Mold', 'name_tl': 'Amag sa Dahon ng Kamatis', 'crop': 'tomato'},
    3: {'name': 'Tomato Septoria Leaf Spot', 'name_tl': 'Septoria Leaf Spot sa Kamatis', 'crop': 'tomato'},
    4: {'name': 'Tomato Spider Mites', 'name_tl': 'Spider Mites sa Kamatis', 'crop': 'tomato'},
    5: {'name': 'Tomato Target Spot', 'name_tl': 'Target Spot sa Kamatis', 'crop': 'tomato'},
    6: {'name': 'Tomato Yellow Leaf Curl Virus', 'name_tl': 'Yellow Leaf Curl Virus sa Kamatis', 'crop': 'tomato'},
    7: {'name': 'Tomato Mosaic Virus', 'name_tl': 'Mosaic Virus sa Kamatis', 'crop': 'tomato'},
    8: {'name': 'Tomato Healthy', 'name_tl': 'Malusog na Kamatis', 'crop': 'tomato'},
    9: {'name': 'Pepper Bacterial Spot', 'name_tl': 'Bacterial Spot sa Sili', 'crop': 'pepper'},
    10: {'name': 'Pepper Healthy', 'name_tl': 'Malusog na Sili', 'crop': 'pepper'},
    11: {'name': 'Potato Early Blight', 'name_tl': 'Early Blight sa Patatas', 'crop': 'potato'},
    12: {'name': 'Potato Late Blight', 'name_tl': 'Late Blight sa Patatas', 'crop': 'potato'},
    13: {'name': 'Potato Healthy', 'name_tl': 'Malusog na Patatas', 'crop': 'potato'},
    14: {'name': 'Rice Blast', 'name_tl': 'Blast sa Palay', 'crop': 'rice'},
    15: {'name': 'Rice Brown Spot', 'name_tl': 'Brown Spot sa Palay', 'crop': 'rice'},
    16: {'name': 'Rice Healthy', 'name_tl': 'Malusog na Palay', 'crop': 'rice'},
    17: {'name': 'Corn Common Rust', 'name_tl': 'Common Rust sa Mais', 'crop': 'corn'},
    18: {'name': 'Corn Gray Leaf Spot', 'name_tl': 'Gray Leaf Spot sa Mais', 'crop': 'corn'},
    19: {'name': 'Corn Healthy', 'name_tl': 'Malusog na Mais', 'crop': 'corn'},
}

def load_model():
    """Load TensorFlow Lite model (mock for now - replace with real model)"""
    # In production, this would load the actual .tflite file:
    # import tensorflow as tf
    # model_path = os.path.join(os.path.dirname(__file__), '../static/models/crop_disease_model.tflite')
    # interpreter = tf.lite.Interpreter(model_path=model_path)
    # interpreter.allocate_tensors()
    # return interpreter

    return None  # Mock model for now

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for model input"""
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize(target_size, Image.Resampling.LANCZOS)

    # Convert to numpy array and normalize
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    return img_array

def predict_disease(image_path):
    """
    Run inference on uploaded image

    Args:
        image_path: Path to uploaded leaf image

    Returns:
        dict with disease_name, confidence, crop_type, class_idx
    """
    try:
        # Load and preprocess image
        img_array = preprocess_image(image_path)

        # MOCK PREDICTION (replace with real TFLite inference)
        # In production:
        # interpreter = load_model()
        # input_details = interpreter.get_input_details()
        # output_details = interpreter.get_output_details()
        # interpreter.set_tensor(input_details[0]['index'], img_array)
        # interpreter.invoke()
        # predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        # For now, return a mock prediction
        # Simulate a tomato late blight detection with high confidence
        mock_class_idx = 0  # Tomato Late Blight
        mock_confidence = 0.92

        disease_info = DISEASE_CLASSES[mock_class_idx]

        return {
            'disease_name': disease_info['name'],
            'disease_name_tl': disease_info['name_tl'],
            'crop_type': disease_info['crop'],
            'confidence': mock_confidence,
            'class_idx': mock_class_idx,
            'error': None
        }

    except Exception as e:
        return {
            'disease_name': None,
            'disease_name_tl': None,
            'crop_type': None,
            'confidence': 0.0,
            'class_idx': None,
            'error': str(e)
        }

def get_disease_info(class_idx):
    """Get disease information by class index"""
    return DISEASE_CLASSES.get(class_idx, {
        'name': 'Unknown Disease',
        'name_tl': 'Hindi Kilalang Sakit',
        'crop': 'unknown'
    })
