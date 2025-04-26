from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import os
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
emotion_model = load_model('emotion_model.hdf5')  # <- load your pre-trained emotion detection model

# Emotion labels based on training
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

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

    (x, y, w, h) = faces[0]  # only first face for now
    face_roi = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face_roi, (48, 48))  # most emotion models are trained on 48x48
    face_normalized = face_resized / 255.0
    face_reshaped = np.reshape(face_normalized, (1, 48, 48, 1))

    # Predict emotion
    prediction = emotion_model.predict(face_reshaped)
    emotion_index = np.argmax(prediction)
    emotion = emotion_labels[emotion_index]
    confidence = float(np.max(prediction))

    return jsonify({
        'emotion': emotion,
        'confidence': confidence
    })
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # this line is very important
    app.run(host="0.0.0.0", port=port)

