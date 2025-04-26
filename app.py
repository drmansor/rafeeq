from flask import Flask, request, jsonify
import cv2
from fer import FER
import numpy as np
from PIL import Image

app = Flask(__name__)
detector = FER(mtcnn=True)

@app.route('/emotion-detect', methods=['POST'])
def emotion_detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    image_np = np.array(image)

    emotions = detector.detect_emotions(image_np)
    if not emotions:
        return jsonify({'error': 'No face detected'}), 200

    top_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
    return jsonify({
        'emotion': top_emotion,
        'confidence': emotions[0]['emotions'][top_emotion]
    })

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 locally
    app.run(host='0.0.0.0', port=port)
