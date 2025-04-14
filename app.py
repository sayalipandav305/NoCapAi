from flask import Flask, request, render_template
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import torch
import torch.nn.functional as F
import os
import io
import mysql.connector
import cv2
import tempfile
import numpy as np

app = Flask(__name__)

# Load model and feature extractor
model = ViTForImageClassification.from_pretrained("prithivMLmods/Deep-Fake-Detector-Model")
feature_extractor = ViTFeatureExtractor.from_pretrained("prithivMLmods/Deep-Fake-Detector-Model")

# MySQL config
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'pandav91',
    'database': 'deepfake_db'
}

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CONFIDENCE_THRESHOLD = 0.8
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def save_to_mysql(filename, media_type, prediction, confidence, reason, file_bytes):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    query = '''
        INSERT INTO results (filename, media_type, prediction, confidence, reason)
        VALUES (%s, %s, %s, %s, %s)
    '''
    cursor.execute(query, (filename, media_type, prediction, confidence, reason))
    conn.commit()
    cursor.close()
    conn.close()

def crop_face_from_image(pil_image):
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    faces = FACE_CASCADE.detectMultiScale(cv_image, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return pil_image  # fallback: no face detected
    x, y, w, h = faces[0]
    face_img = cv_image[y:y+h, x:x+w]
    return Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

def is_blurry(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < 50

def predict_image(image):
    image = crop_face_from_image(image)
    if is_blurry(image):
        return "Uncertain", 0.0

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=1)
    confidence = torch.max(probs).item()
    predicted_class_idx = logits.argmax(-1).item()
    label = model.config.id2label[predicted_class_idx]

    if confidence < CONFIDENCE_THRESHOLD:
        label = "Uncertain"

    return label, confidence

def predict_video(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(file_bytes)
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(frame_count // 15, 1)

    predictions = []
    confidences = []

    for i in range(15):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        success, frame = cap.read()
        if not success:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        label, confidence = predict_image(pil_image)
        if label != "Uncertain":
            predictions.append(label)
            confidences.append(confidence)

    cap.release()
    os.remove(video_path)

    if not predictions:
        return "Uncertain", 0.0

    real_votes = sum(c for l, c in zip(predictions, confidences) if l == "Real")
    fake_votes = sum(c for l, c in zip(predictions, confidences) if l == "Fake")

    if abs(real_votes - fake_votes) < 0.2:
        return "Uncertain", sum(confidences) / len(confidences)
    return ("Fake" if fake_votes > real_votes else "Real", sum(confidences) / len(confidences))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['media']
        if file.filename == '':
            return render_template('index.html', error='No file selected.')

        ext = os.path.splitext(file.filename)[1].lower()
        file_bytes = file.read()
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if ext in ['.jpg', '.jpeg', '.png']:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            label, confidence = predict_image(image)

            with open(file_path, 'wb') as f:
                f.write(file_bytes)

            reason = {
                "Real": "Face appears natural with no evidence of manipulation.",
                "Fake": "Detected artifacts or features consistent with deepfake generation.",
                "Uncertain": "Unable to confidently assess authenticity."
            }[label]

            save_to_mysql(filename, 'image', label, confidence, reason, file_bytes)

            return render_template('result.html', prediction=label, confidence=confidence, reason=reason, file_url=f"/static/uploads/{filename}")

        elif ext in ['.mp4', '.avi', '.mov']:
            label, confidence = predict_video(file_bytes)

            with open(file_path, 'wb') as f:
                f.write(file_bytes)

            reason = {
                "Real": "Video shows consistent and natural facial motion across frames.",
                "Fake": "Inconsistencies across frames suggest deepfake manipulation.",
                "Uncertain": "Could not confidently determine authenticity from frames."
            }[label]

            save_to_mysql(filename, 'video', label, confidence, reason, file_bytes)

            return render_template('result.html', prediction=label, confidence=confidence, reason=reason, file_url=f"/static/uploads/{filename}")
        else:
            return render_template('index.html', error='Unsupported file format.')

    return render_template('index.html')

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)
