from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from io import BytesIO

app = FastAPI()

# Load the FER detector
emotion_detector = FER(mtcnn=True)

@app.post("/emotion-detect")
async def detect_emotion(image: UploadFile = File(...)):
    # Read the uploaded image file
    image_bytes = await image.read()
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Detect emotions
    result = emotion_detector.detect_emotions(img)

    if not result:
        return JSONResponse(content={"error": "No face detected"}, status_code=400)

    # Assume the first face detected
    emotions = result[0]["emotions"]
    top_emotion = max(emotions, key=emotions.get)

    return {
        "top_emotion": top_emotion,
        "all_emotions": emotions
    }
