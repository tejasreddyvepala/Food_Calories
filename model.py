import tensorflow as tf
import os
import numpy as np
import random
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0, NASNetMobile
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import json

# Define dataset paths
base_path = "/Users/tejasreddyvepala/Desktop/github/food-101/"
images_path = os.path.join(base_path, "images")
train_txt_path = os.path.join(base_path, "meta/train.txt")
test_txt_path = os.path.join(base_path, "meta/test.txt")
classes_txt_path = os.path.join(base_path, "meta/classes.txt")
labels_txt_path = os.path.join(base_path, "meta/labels.txt")

# Load class names from classes.txt
with open(classes_txt_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load human-readable labels from labels.txt
with open(labels_txt_path, "r") as f:
    label_names = [line.strip() for line in f.readlines()]

# Ensure class names and labels match
assert len(class_names) == len(label_names), "Mismatch between classes.txt and labels.txt!"

# Load train and test file paths
def load_image_paths(txt_path):
    with open(txt_path, "r") as f:
        lines = f.read().splitlines()
    return [os.path.join(images_path, line + ".jpg") for line in lines]

train_images = load_image_paths(train_txt_path)
test_images = load_image_paths(test_txt_path)

# Convert food category names into numerical labels using classes.txt
train_labels = [class_names.index(path.split("/")[-2]) for path in train_images]
test_labels = [class_names.index(path.split("/")[-2]) for path in test_images]

# Function to load and preprocess images
def preprocess_image(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0  # Normalize
    return img, label

# Create TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

# Apply preprocessing
train_dataset = train_dataset.map(preprocess_image).batch(16).shuffle(200).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.map(preprocess_image).batch(16).prefetch(tf.data.AUTOTUNE)

# List of models to train
models = {
    # i replaced model names here and trained one by one to save the models and use later.
    "NASNetMobile": NASNetMobile
}

# Train and save each model
for model_name, model_base in models.items():
    print(f"\n🔹 Training {model_name.upper()}...\n")

    base_model = model_base(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    predictions = Dense(len(class_names), activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train model
    history = model.fit(train_dataset, validation_data=test_dataset, epochs=5)

    # Save model
    model_save_path = f"/Users/tejasreddyvepala/Desktop/github/Food_Calories/food101_{model_name}.keras"
    model.save(model_save_path)
    print(f"✅ Saved {model_name.upper()} model at: {model_save_path}")

# Save label mapping for prediction
label_mapping_path = "/Users/tejasreddyvepala/Desktop/github/Food_Calories/label_mapping.json"
with open(label_mapping_path, "w") as f:
    json.dump({"classes": class_names, "labels": label_names}, f)

print(f"\n✅ Label mapping saved at: {label_mapping_path}")
