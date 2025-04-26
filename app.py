from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import get_custom_objects
import os

# === FIX for old 'lr' in Adam optimizer ===
def legacy_adam_optimizer(*args, **kwargs):
    if 'lr' in kwargs:
        kwargs['learning_rate'] = kwargs.pop('lr')
    return Adam(*args, **kwargs)

get_custom_objects()['Adam'] = legacy_adam_optimizer

# === Load the pre-trained model ===
emotion_model = load_model('emotion_model.hdf5')

# === Create Flask app ===
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# === Define emotion labels ===
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/emotion-detect', methods=['POST'])
def detect_emotion():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    image_np = np.array(image)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Resize for the model input
    resized = cv2.resize(gray, (48, 48))
    resized = resized.astype('float32') / 255.0
    resized = np.expand_dims(resized, axis=-1)
    resized = np.expand_dims(resized, axis=0)  # Add batch dimension

    # Predict emotion
    predictions = emotion_model.predict(resized)
    emotion_idx = np.argmax(predictions)
    emotion = emotion_labels[emotion_idx]
    confidence = float(np.max(predictions))

    return jsonify({
        'emotion': emotion,
        'confidence': round(confidence, 2)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
