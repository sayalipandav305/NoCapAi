# predict_image.py

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

IMG_SIZE = (128, 128)

# Load model
model = tf.keras.models.load_model('deepfake_model.h5')

# Function to predict
def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    
    if prediction < 0.5:
        print(f"🔎 Prediction: REAL (Confidence: {1 - prediction:.2f})")
    else:
        print(f"⚠️ Prediction: FAKE (Confidence: {prediction:.2f})")

# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py <image_path>")
    else:
        predict_image(sys.argv[1])
