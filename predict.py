import tensorflow as tf
import numpy as np
import sys
import json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load trained models
models = {
    "mobilenet": tf.keras.models.load_model("food101_mobilenet.keras"),
    "NASNetMobile": tf.keras.models.load_model("food101_NASNetMobile.keras")
}

# Load label mapping
label_mapping_path = "label_mapping.json"
with open(label_mapping_path, "r") as f:
    label_data = json.load(f)
    class_names = label_data["classes"]
    label_names = label_data["labels"]

# Function to preprocess input image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to classify food image using an ensemble of models
def predict_food(img_path):
    img_array = preprocess_image(img_path)
    
    # Get predictions from all models
    predictions = [model.predict(img_array) for model in models.values()]
    
    # Average predictions across models
    avg_prediction = np.mean(predictions, axis=0)
    
    predicted_class_index = np.argmax(avg_prediction)
    predicted_class = label_names[predicted_class_index]
    confidence = np.max(avg_prediction)

    return predicted_class, confidence


# Predict food category
predicted_food, confidence = predict_food("capre.png")
print(f"Predicted Food: {predicted_food} ({confidence:.2f} confidence)")
