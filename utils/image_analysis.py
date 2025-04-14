import cv2
import numpy as np
from scipy.fftpack import dct
from PIL import Image
import io
import os

def analyze_image(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            return "Could not load image."

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (256, 256))

        # Discrete Cosine Transform
        dct_transformed = dct(dct(resized.T, norm='ortho').T, norm='ortho')
        mean_dct = np.mean(np.abs(dct_transformed[20:100, 20:100]))  # Crop to high-freq block

        # Error Level Analysis (ELA)
        ela_path = image_path.replace(".", "_ela.")
        ela_image = perform_ela(image_path, ela_path)
        ela_score = np.mean(ela_image)

        # Combined analysis logic
        if mean_dct < 20 and ela_score > 15:
            return "Manipulated (High ELA + Low DCT mean)"
        elif ela_score > 25:
            return "Manipulated (Very high ELA noise)"
        else:
            return "Image appears to be real"

    except Exception as e:
        return f"Error: {str(e)}"

def perform_ela(original_path, ela_output_path, quality=95):
    original = Image.open(original_path).convert("RGB")

    # Save as a high-quality JPEG
    buffer = io.BytesIO()
    original.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer)

    # Compute difference
    ela_image = Image.new("RGB", original.size)
    diff = np.abs(np.array(original).astype(int) - np.array(recompressed).astype(int))
    ela_image = Image.fromarray(np.clip(diff * 10, 0, 255).astype('uint8'))

    # Save and return intensity for analysis
    ela_image.save(ela_output_path)
    gray_ela = cv2.cvtColor(np.array(ela_image), cv2.COLOR_RGB2GRAY)
    return gray_ela
