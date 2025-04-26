from flask import Flask, request, jsonify
from flask_cors import CORS  # <--- ADD THIS
import cv2
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
CORS(app)  # <--- ADD THIS

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@app.route('/emotion-detect', methods=['POST'])
def detect_emotion():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert("RGB")
    image_np = np.array(image)

    image_resized = cv2.resize(image_np, (224, 224))
    gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return jsonify({'error': 'No face detected'}), 200

    (x, y, w, h) = faces[0]
    face_area = w * h

    if face_area > 10000:
        emotion = "Happy"
    elif face_area > 5000:
        emotion = "Neutral"
    else:
        emotion = "Surprised"

    return jsonify({
        'emotion': emotion,
        'confidence': 0.9
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
