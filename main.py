from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os
import shutil
import uuid
import tensorflow as tf
from keras.applications import efficientnet_v2
from keras.applications.efficientnet_v2 import preprocess_input
from mtcnn import MTCNN

app = FastAPI(
    title="DeepFake Detection API",
    description="Keras EfficientNetV2-based DeepFake video detector",
    version="1.0"
)

IMG_SIZE = 380
model = None
detector = None


@app.on_event("startup")
def load_model():
    """Lazy-load model and MTCNN detector once at startup"""
    global model, detector
    print("⏳ Loading EfficientNetV2S and MTCNN...")

    base_model = efficientnet_v2.EfficientNetV2S(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg",
    )
    x = tf.keras.layers.Dense(1, activation="sigmoid")(base_model.output)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    model.trainable = False
    detector = MTCNN()

    print("✅ Model and MTCNN loaded successfully.")


def confident_strategy(preds, t=0.8):
    preds = np.array(preds)
    sz = len(preds)
    fakes = np.count_nonzero(preds > t)
    if fakes > sz // 2.5 and fakes > 11:
        return np.mean(preds[preds > t])
    elif np.count_nonzero(preds < 0.2) > 0.9 * sz:
        return np.mean(preds[preds < 0.2])
    else:
        return np.mean(preds)


def process_video(video_path: str) -> float:
    """Extract frames, detect faces, and predict fake probability"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, 32).astype(int)
    preds = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)

        if faces:
            x, y, w, h = faces[0]["box"]
            # Clamp coordinates within image boundaries
            x, y = max(0, x), max(0, y)
            face = rgb[y:y + h, x:x + w]
            if face.size == 0:
                continue

            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            img_array = preprocess_input(np.expand_dims(face.astype(np.float32), 0))
            pred = float(model.predict(img_array, verbose=0)[0][0])
            preds.append(pred)

    cap.release()
    if not preds:
        return 0.5  # uncertain
    return confident_strategy(preds)


@app.get("/")
def root():
    return {"status": "ok", "message": "DeepFake Detection API is running"}


@app.post("/predict/")
async def predict(video: UploadFile = File(...)):
    """Accepts an MP4 file and returns DeepFake classification"""
    temp_name = f"temp_{uuid.uuid4()}.mp4"
    with open(temp_name, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    try:
        confidence = process_video(temp_name)
        label = "fake" if confidence > 0.5 else "real"
        return JSONResponse({
            "prediction": label,
            "confidence": round(confidence, 4)
        })
    finally:
        os.remove(temp_name)
