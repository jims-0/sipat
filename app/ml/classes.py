"""The classes the model knows - one list, used by training, inference and treatment.

Keys are the PlantVillage folder names (the dataset's own labels), so the
training script, the saved labels.json and this table can never drift apart.

Rice is deliberately absent: PlantVillage has no rice photos, and a class the
model was never shown cannot be predicted. It returns when a rice dataset is
added to datasets/ and the model is retrained (see TRAINING.md).
"""

NOT_A_LEAF = "Background_without_leaves"
# Leaves of plants Sipat does not cover (apple, grape, soybean...), pooled
# from the rest of PlantVillage. Without it, a mango leaf would still be forced
# into one of the tomato/corn/potato/pepper diseases.
OTHER_PLANT = "Other_plant"

CLASSES = {
    "Tomato___Bacterial_spot": {"name": "Tomato Bacterial Spot", "name_tl": "Bacterial Spot sa Kamatis", "crop": "tomato"},
    "Tomato___Early_blight": {"name": "Tomato Early Blight", "name_tl": "Early Blight sa Kamatis", "crop": "tomato"},
    "Tomato___Late_blight": {"name": "Tomato Late Blight", "name_tl": "Late Blight sa Kamatis", "crop": "tomato"},
    "Tomato___Leaf_Mold": {"name": "Tomato Leaf Mold", "name_tl": "Amag sa Dahon ng Kamatis", "crop": "tomato"},
    "Tomato___Septoria_leaf_spot": {"name": "Tomato Septoria Leaf Spot", "name_tl": "Septoria Leaf Spot sa Kamatis", "crop": "tomato"},
    "Tomato___Spider_mites Two-spotted_spider_mite": {"name": "Tomato Spider Mites", "name_tl": "Spider Mites sa Kamatis", "crop": "tomato"},
    "Tomato___Target_Spot": {"name": "Tomato Target Spot", "name_tl": "Target Spot sa Kamatis", "crop": "tomato"},
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {"name": "Tomato Yellow Leaf Curl Virus", "name_tl": "Yellow Leaf Curl Virus sa Kamatis", "crop": "tomato"},
    "Tomato___Tomato_mosaic_virus": {"name": "Tomato Mosaic Virus", "name_tl": "Mosaic Virus sa Kamatis", "crop": "tomato"},
    "Tomato___healthy": {"name": "Tomato Healthy", "name_tl": "Malusog na Kamatis", "crop": "tomato", "healthy": True},
    "Pepper,_bell___Bacterial_spot": {"name": "Pepper Bacterial Spot", "name_tl": "Bacterial Spot sa Sili", "crop": "pepper"},
    "Pepper,_bell___healthy": {"name": "Pepper Healthy", "name_tl": "Malusog na Sili", "crop": "pepper", "healthy": True},
    "Potato___Early_blight": {"name": "Potato Early Blight", "name_tl": "Early Blight sa Patatas", "crop": "potato"},
    "Potato___Late_blight": {"name": "Potato Late Blight", "name_tl": "Late Blight sa Patatas", "crop": "potato"},
    "Potato___healthy": {"name": "Potato Healthy", "name_tl": "Malusog na Patatas", "crop": "potato", "healthy": True},
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": {"name": "Corn Gray Leaf Spot", "name_tl": "Gray Leaf Spot sa Mais", "crop": "corn"},
    "Corn___Common_rust": {"name": "Corn Common Rust", "name_tl": "Common Rust sa Mais", "crop": "corn"},
    "Corn___Northern_Leaf_Blight": {"name": "Corn Northern Leaf Blight", "name_tl": "Northern Leaf Blight sa Mais", "crop": "corn"},
    "Corn___healthy": {"name": "Corn Healthy", "name_tl": "Malusog na Mais", "crop": "corn", "healthy": True},
    NOT_A_LEAF: {"name": "Not a leaf", "name_tl": "Hindi dahon", "crop": "none"},
    OTHER_PLANT: {"name": "Unsupported plant", "name_tl": "Hindi suportadong halaman", "crop": "other"},
}


def info(folder: str) -> dict:
    return CLASSES.get(folder, {"name": "Unknown", "name_tl": "Hindi kilala", "crop": "unknown"})


def is_diagnosis(folder: str) -> bool:
    """A crop result, as opposed to 'not a leaf' or 'a plant we do not cover'."""
    return folder in CLASSES and folder not in (NOT_A_LEAF, OTHER_PLANT)
